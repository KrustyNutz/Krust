from __future__ import annotations

# ====================================================================
# NEXUS ALPHA | Overnight Drift Bot (production)
#
# Single-strategy first implementation. Strategy:
#   - Submit BUY MOC (market-on-close) order on SPY ~3:55 PM ET
#   - Submit SELL MOO (market-on-open) order at next session ~9:28 AM ET
#   - Captures the SPX overnight risk premium (Bessembinder 2018,
#     Lou et al 2019 — SPX has earned ~7%/yr overnight vs ~1%/yr intraday).
#
# Why MOC/MOO instead of regular market orders:
#   - Auction prints have no bid/ask spread cost
#   - Better fills than crossing the spread at 3:59 PM
#   - Standard practice for overnight-edge strategies
#
# Risk controls:
#   - Skip Fridays (3-day weekend gap has weaker / negative drift)
#   - Skip day before market holidays (multi-day gap risk)
#   - Position sized by leverage tier (1x conservative, 2x moderate, 3x aggressive)
#   - Hard portfolio drawdown halt from high-water mark
#   - State persisted to JSON so restarts don't double-enter or miss exits
#
# The other three NEXUS ALPHA strategies (VRP, RSI(2), Sector Momentum)
# remain in this file as scaffolding for future builds; they are NOT
# wired into the production runner yet.
#
# Usage:
#   python nexus_alpha.py                       # paper mode, 1x leverage
#   python nexus_alpha.py --leverage 2.0        # paper mode, 2x leverage
#   python nexus_alpha.py --live                # live (paper-account) trading
#   python nexus_alpha.py --reset-state         # clear persisted state
# ====================================================================

import os
import sys
import time
import json
import math
import logging
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date, time as dtime
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from alpaca_trade_api.rest import REST

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    from datetime import timezone
    # Naive EST fallback if zoneinfo unavailable
    ET = timezone(timedelta(hours=-5))


# ====================================================================
# CONFIG
# ====================================================================
API_KEY = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

LOG_DIR = Path("logs/alpha")
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = LOG_DIR / "state.json"
TRADES_CSV = LOG_DIR / "overnight_trades.csv"

# Strategy
TICKER = "SPY"
ASSET_ANNUAL_VOL = 0.18              # SPY ~18% annualized historically
PORTFOLIO_VOL_TARGET = 0.12          # baseline 12% target (scaled by leverage)
MAX_PORTFOLIO_DRAWDOWN = 0.15        # halt at -15% from HWM
MOC_SUBMIT_TIME = dtime(15, 55)      # submit BUY MOC at/after 3:55 PM ET
MOC_CUTOFF_TIME = dtime(15, 58)      # stop trying after 3:58 (Alpaca cutoff ~3:59)
MOO_SUBMIT_TIME = dtime(9, 25)       # submit SELL MOO at/after 9:25 AM ET
MOO_CUTOFF_TIME = dtime(9, 28)       # before 9:28 Alpaca cutoff
SAFETY_FRIDAY_SKIP = True            # 3-day weekend gap is weaker edge
SAFETY_PREHOLIDAY_SKIP = True        # avoid multi-day risk gaps

# Polling
TICK_INTERVAL_SECONDS = 30           # plenty for daily-window strategy

# ====================================================================
# LOGGING
# ====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "nexus_alpha.log", mode='a'),
    ],
)
log = logging.getLogger("nexus.alpha")


# ====================================================================
# STATE — persisted across restarts
# ====================================================================
@dataclass
class PersistedState:
    """Everything we need to know on restart to not double-trade or
    miss an exit. Written to disk after every state-changing event."""
    position_open: bool = False
    shares: int = 0
    entry_order_id: str = ""
    entry_submitted_date: str = ""   # ISO date string
    entry_price: float = 0.0
    exit_order_id: str = ""
    exit_submitted_date: str = ""
    high_water_mark: float = 0.0
    halted: bool = False
    realised_pnl_total: float = 0.0

    @classmethod
    def load(cls) -> "PersistedState":
        if not STATE_PATH.exists():
            return cls()
        try:
            return cls(**json.loads(STATE_PATH.read_text()))
        except Exception as e:
            log.warning(f"State file corrupt ({e}); starting fresh.")
            return cls()

    def save(self):
        STATE_PATH.write_text(json.dumps(asdict(self), indent=2))


# ====================================================================
# HELPERS
# ====================================================================
def now_et() -> datetime:
    return datetime.now(tz=ET)


def is_friday(d: datetime) -> bool:
    return d.weekday() == 4


def shares_for_leverage(equity: float, price: float, leverage: float) -> int:
    """
    Size = equity × (vol_target / asset_vol) × leverage / price.

    On $50k, SPY $600, vol_target 12%, asset_vol 18%, leverage 2x:
       size = 50000 × 0.667 × 2 / 600 = ~111 shares = ~$66.7k notional.
    """
    if price <= 0 or equity <= 0:
        return 0
    notional = equity * (PORTFOLIO_VOL_TARGET / ASSET_ANNUAL_VOL) * leverage
    return max(int(notional / price), 0)


def append_trade_csv(row: dict):
    write_header = not TRADES_CSV.exists()
    with open(TRADES_CSV, "a") as f:
        if write_header:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")


# ====================================================================
# OVERNIGHT DRIFT BOT
# ====================================================================
class OvernightDriftBot:
    def __init__(self, leverage: float, paper_sim: bool):
        if not API_KEY or not SECRET_KEY:
            log.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
            sys.exit(1)
        self.leverage = leverage
        self.paper_sim = paper_sim  # True = no real orders, simulate fills
        self.api = REST(API_KEY, SECRET_KEY, BASE_URL)
        self.state = PersistedState.load()
        self.starting_equity = self._fetch_equity()
        if self.state.high_water_mark == 0:
            self.state.high_water_mark = self.starting_equity
            self.state.save()
        log.info("=" * 64)
        log.info(f"  NEXUS ALPHA — Overnight Drift Bot")
        log.info(f"  Mode: {'PAPER-SIM' if paper_sim else 'LIVE (paper account)'}  "
                 f"Leverage: {leverage}x")
        log.info(f"  Starting equity: ${self.starting_equity:,.2f}  "
                 f"HWM: ${self.state.high_water_mark:,.2f}")
        if self.state.position_open:
            log.info(f"  RESUMING with open position: {self.state.shares} {TICKER} "
                     f"@ ${self.state.entry_price:.2f}")
        log.info("=" * 64)
        self._holiday_cache: List[date] = []
        self._holiday_cache_loaded_at: Optional[datetime] = None

    # ----- Account helpers -----
    def _fetch_equity(self) -> float:
        try:
            return float(self.api.get_account().equity)
        except Exception as e:
            log.warning(f"Equity fetch failed ({e}); using last HWM as proxy.")
            return self.state.high_water_mark or 50_000.0

    def _fetch_price(self) -> float:
        try:
            quote = self.api.get_latest_quote(TICKER)
            bid = float(getattr(quote, 'bp', 0) or 0)
            ask = float(getattr(quote, 'ap', 0) or 0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
        except Exception as e:
            log.debug(f"Quote fetch failed: {e}")
        return 0.0

    def _is_trading_day(self, d: datetime) -> bool:
        if d.weekday() >= 5:
            return False
        return d.date() not in self._upcoming_holidays()

    def _upcoming_holidays(self) -> List[date]:
        """Cache Alpaca's market calendar for the next ~60 days."""
        if (self._holiday_cache_loaded_at and
                (now_et() - self._holiday_cache_loaded_at).days < 7):
            return self._holiday_cache
        try:
            today = now_et().date()
            end = today + timedelta(days=60)
            cal = self.api.get_calendar(
                start=today.isoformat(), end=end.isoformat()
            )
            trading_days = {entry.date.date() if hasattr(entry.date, 'date')
                            else entry.date for entry in cal}
            self._holiday_cache = [
                today + timedelta(days=i)
                for i in range(60)
                if (today + timedelta(days=i)).weekday() < 5
                and (today + timedelta(days=i)) not in trading_days
            ]
            self._holiday_cache_loaded_at = now_et()
            log.info(f"Cached {len(self._holiday_cache)} upcoming market closures.")
        except Exception as e:
            log.warning(f"Calendar fetch failed ({e}); proceeding without "
                        f"holiday awareness.")
            self._holiday_cache = []
        return self._holiday_cache

    def _is_pre_holiday(self, d: datetime) -> bool:
        """True if next trading day is more than 1 calendar day away."""
        tomorrow = d.date() + timedelta(days=1)
        # If tomorrow is weekend or holiday, this is a multi-day gap entry
        check = tomorrow
        days_until_open = 1
        while days_until_open <= 5:
            if check.weekday() < 5 and check not in self._upcoming_holidays():
                break
            check += timedelta(days=1)
            days_until_open += 1
        return days_until_open > 1

    # ----- Entry / exit guards -----
    def _can_enter_today(self, d: datetime) -> tuple[bool, str]:
        if self.state.halted:
            return False, "halted (drawdown)"
        if self.state.position_open:
            return False, "position already open"
        if not self._is_trading_day(d):
            return False, "not a trading day"
        if SAFETY_FRIDAY_SKIP and is_friday(d):
            return False, "friday skip (3-day weekend gap)"
        if SAFETY_PREHOLIDAY_SKIP and self._is_pre_holiday(d):
            return False, "pre-holiday skip (multi-day gap)"
        return True, ""

    # ----- Reconciliation -----
    def _reconcile(self):
        """Compare persisted state to actual Alpaca positions; fix drift."""
        if self.paper_sim:
            return
        try:
            positions = {p.symbol: float(p.qty) for p in self.api.list_positions()}
        except Exception as e:
            log.warning(f"Position reconcile failed: {e}")
            return
        actual_qty = int(positions.get(TICKER, 0))
        if self.state.position_open and actual_qty == 0:
            log.warning(f"State said position open ({self.state.shares} sh) but "
                        f"broker shows none. Clearing state.")
            self._clear_position_state()
        elif not self.state.position_open and actual_qty > 0:
            log.warning(f"Broker shows {actual_qty} sh but state says flat. "
                        f"Adopting broker truth.")
            self.state.position_open = True
            self.state.shares = actual_qty
            self.state.save()

    # ----- Order submission -----
    def _submit_moc_buy(self, qty: int) -> Optional[str]:
        """Market-on-close BUY. Returns order id, or None on failure."""
        if self.paper_sim:
            sim_id = f"paper-moc-{int(time.time())}"
            log.info(f"[PAPER-SIM] BUY MOC {qty} {TICKER} (order {sim_id})")
            return sim_id
        try:
            order = self.api.submit_order(
                symbol=TICKER, qty=qty, side='buy',
                type='market', time_in_force='cls',
            )
            log.info(f"BUY MOC submitted: {qty} {TICKER} (order {order.id})")
            return order.id
        except Exception as e:
            log.error(f"BUY MOC failed: {e}")
            return None

    def _submit_moo_sell(self, qty: int) -> Optional[str]:
        """Market-on-open SELL. Returns order id, or None on failure."""
        if self.paper_sim:
            sim_id = f"paper-moo-{int(time.time())}"
            log.info(f"[PAPER-SIM] SELL MOO {qty} {TICKER} (order {sim_id})")
            return sim_id
        try:
            order = self.api.submit_order(
                symbol=TICKER, qty=qty, side='sell',
                type='market', time_in_force='opg',
            )
            log.info(f"SELL MOO submitted: {qty} {TICKER} (order {order.id})")
            return order.id
        except Exception as e:
            log.error(f"SELL MOO failed: {e}")
            return None

    def _check_fill(self, order_id: str) -> Optional[float]:
        """Return filled_avg_price if filled, else None."""
        if self.paper_sim or not order_id:
            return None
        try:
            o = self.api.get_order(order_id)
            if o.status == 'filled' and o.filled_avg_price:
                return float(o.filled_avg_price)
        except Exception as e:
            log.debug(f"Order status check {order_id}: {e}")
        return None

    # ----- State management -----
    def _clear_position_state(self):
        self.state.position_open = False
        self.state.shares = 0
        self.state.entry_order_id = ""
        self.state.entry_submitted_date = ""
        self.state.entry_price = 0.0
        self.state.exit_order_id = ""
        self.state.exit_submitted_date = ""
        self.state.save()

    # ----- Drawdown halt -----
    def _check_drawdown(self, equity: float):
        if equity > self.state.high_water_mark:
            self.state.high_water_mark = equity
            self.state.save()
        dd = (self.state.high_water_mark - equity) / self.state.high_water_mark \
            if self.state.high_water_mark > 0 else 0
        if dd >= MAX_PORTFOLIO_DRAWDOWN and not self.state.halted:
            log.critical(f"DRAWDOWN HALT: equity ${equity:,.2f} is "
                         f"{dd*100:.1f}% below HWM ${self.state.high_water_mark:,.2f}. "
                         f"No new entries until --reset-state.")
            self.state.halted = True
            self.state.save()

    # ----- Main tick -----
    def tick(self):
        equity = self._fetch_equity()
        self._check_drawdown(equity)
        self._reconcile()

        d = now_et()
        t = d.time()

        # ENTRY WINDOW
        if MOC_SUBMIT_TIME <= t < MOC_CUTOFF_TIME:
            today_iso = d.date().isoformat()
            if self.state.entry_submitted_date == today_iso:
                return  # already submitted today
            allowed, reason = self._can_enter_today(d)
            if not allowed:
                if self.state.entry_submitted_date != today_iso:
                    log.info(f"Entry skipped: {reason}")
                    self.state.entry_submitted_date = today_iso  # cool-off
                    self.state.save()
                return
            price = self._fetch_price()
            if price <= 0:
                log.warning("No price quote; cannot size entry.")
                return
            qty = shares_for_leverage(equity, price, self.leverage)
            if qty <= 0:
                log.warning(f"Size = 0 shares (equity ${equity:.0f}, "
                            f"price ${price:.2f}). Skipping.")
                return
            order_id = self._submit_moc_buy(qty)
            if order_id:
                self.state.entry_order_id = order_id
                self.state.entry_submitted_date = today_iso
                self.state.shares = qty
                self.state.entry_price = price  # approximate; reconciled at fill
                self.state.position_open = True
                self.state.save()

        # EXIT WINDOW
        elif MOO_SUBMIT_TIME <= t < MOO_CUTOFF_TIME:
            if not self.state.position_open:
                return
            today_iso = d.date().isoformat()
            if self.state.exit_submitted_date == today_iso:
                return  # already submitted today
            order_id = self._submit_moo_sell(self.state.shares)
            if order_id:
                self.state.exit_order_id = order_id
                self.state.exit_submitted_date = today_iso
                self.state.save()

        # MISSED-EXIT RECOVERY: if we have an open position from a prior
        # session and the MOO window passed without an order, flatten at
        # market on the next trading-day tick. Holding extra nights drifts
        # from the documented edge (overnight ≠ multi-night).
        elif (self.state.position_open and
              not self.state.exit_order_id and
              self._is_trading_day(d) and
              MOO_CUTOFF_TIME <= t < MOC_SUBMIT_TIME):
            log.warning(f"MISSED EXIT WINDOW — flattening {self.state.shares} "
                        f"{TICKER} at market to stay true to strategy intent.")
            if self.paper_sim:
                fallback_id = f"paper-rescue-{int(time.time())}"
            else:
                try:
                    o = self.api.submit_order(
                        symbol=TICKER, qty=self.state.shares, side='sell',
                        type='market', time_in_force='day',
                    )
                    fallback_id = o.id
                except Exception as e:
                    log.error(f"Rescue sell failed: {e}")
                    return
            self.state.exit_order_id = fallback_id
            self.state.exit_submitted_date = d.date().isoformat()
            self.state.save()

        # POST-EXIT: record PnL once the exit fill is known. order_id is
        # cleared in _clear_position_state, preventing repeat recordings.
        elif self.state.exit_order_id and self.state.position_open:
            exit_price = self._check_fill(self.state.exit_order_id)
            if self.paper_sim:
                # Simulate the exit fill at the current bid/ask midpoint
                exit_price = self._fetch_price() or self.state.entry_price
            if exit_price:
                pnl = (exit_price - self.state.entry_price) * self.state.shares
                self.state.realised_pnl_total += pnl
                row = {
                    "entry_date": (self.state.entry_submitted_date or "?"),
                    "exit_date": self.state.exit_submitted_date,
                    "shares": self.state.shares,
                    "entry_price": f"{self.state.entry_price:.4f}",
                    "exit_price": f"{exit_price:.4f}",
                    "pnl": f"{pnl:.2f}",
                    "leverage": self.leverage,
                    "mode": "paper-sim" if self.paper_sim else "live",
                }
                append_trade_csv(row)
                log.info(f"ROUND-TRIP COMPLETE: {self.state.shares} {TICKER}  "
                         f"entry ${self.state.entry_price:.2f} → "
                         f"exit ${exit_price:.2f}  PnL ${pnl:+,.2f}  "
                         f"(total realised ${self.state.realised_pnl_total:+,.2f})")
                self._clear_position_state()

    def run(self):
        try:
            while True:
                self.tick()
                time.sleep(TICK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            log.info("Shutdown requested. Persisting state and exiting.")
            self.state.save()
            self._print_summary()
            sys.exit(0)

    def _print_summary(self):
        print()
        print("=" * 64)
        print(f"  NEXUS ALPHA SESSION SUMMARY")
        print("=" * 64)
        equity = self._fetch_equity()
        print(f"  Starting equity:    ${self.starting_equity:,.2f}")
        print(f"  Current equity:     ${equity:,.2f}")
        print(f"  Net change:         ${equity - self.starting_equity:+,.2f}")
        print(f"  High-water mark:    ${self.state.high_water_mark:,.2f}")
        print(f"  Realised PnL (cum): ${self.state.realised_pnl_total:+,.2f}")
        print(f"  Halted:             {self.state.halted}")
        if self.state.position_open:
            print(f"  Open position:      {self.state.shares} {TICKER} @ "
                  f"${self.state.entry_price:.2f}")
        print(f"  Trade log:          {TRADES_CSV}")
        print("=" * 64)


# ====================================================================
# OTHER STRATEGY SCAFFOLDS (kept for future expansion, not wired in)
# ====================================================================
# VRPStrategy, RSI2Strategy, SectorMomentumStrategy from the earlier
# draft are deliberately omitted from this production build. They will
# return as their own focused implementations once Overnight Drift has
# logged 30+ days of paper trades with PnL matching expectations.


# ====================================================================
# CLI
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="NEXUS ALPHA — Overnight Drift Bot",
    )
    parser.add_argument("--live", action="store_true",
                        help="Submit real orders to the Alpaca paper account "
                             "(default: simulate fills locally, no orders sent)")
    parser.add_argument("--leverage", type=float, default=1.0,
                        choices=[1.0, 2.0, 3.0],
                        help="Position size leverage tier (default: 1.0)")
    parser.add_argument("--reset-state", action="store_true",
                        help="Clear persisted state (including drawdown halt) "
                             "and exit")
    args = parser.parse_args()

    if args.reset_state:
        if STATE_PATH.exists():
            STATE_PATH.unlink()
            print(f"Cleared state file: {STATE_PATH}")
        else:
            print("No state file to clear.")
        sys.exit(0)

    bot = OvernightDriftBot(leverage=args.leverage, paper_sim=not args.live)
    bot.run()


if __name__ == "__main__":
    main()
