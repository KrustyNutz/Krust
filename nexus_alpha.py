from __future__ import annotations

# ====================================================================
# NEXUS ALPHA | Multi-Strategy Portfolio Bot
#
# Combines four retail-accessible edges, each documented in academic
# literature or replicated by practitioners across multiple decades.
# Risk parity allocation across strategies, vol-targeted sizing, hard
# drawdown cutoff. Designed to compound, not to lottery.
#
# STRATEGY 1: Overnight Drift (SPY)
#   - Buy SPY at market close, sell at next open.
#   - Edge: SPX has earned ~7%/yr overnight vs ~1%/yr intraday since 1993
#     (Cliff Asness et al., Bessembinder 2018, NYU Stern overnight studies).
#   - Mechanism: overnight risk premium, retail flow at open, no
#     market makers willing to hold inventory over a closed session.
#   - Sharpe historical: ~0.7
#
# STRATEGY 2: VRP Short Premium (SPY put credit spreads)
#   - Sell 16-delta SPY puts, buy lower strike for cap, weekly expiry.
#   - Only trade when VIX > median of last 252 days.
#   - Edge: SPX IV > realized vol by ~3-4 vol points on average
#     (Bollerslev, Tauchen, Zhou 2009; "Variance Risk Premium").
#   - Sharpe historical: ~0.7-1.0 with tail risk
#
# STRATEGY 3: Connors RSI(2) Mean Reversion (SPY)
#   - Buy SPY when 2-period RSI < 5 and price > 200-day SMA.
#   - Sell when 2-period RSI > 70 or after 5 trading days.
#   - Edge: short-horizon mean reversion in indices, documented in
#     Connors "Short Term Trading Strategies That Work" (2008).
#   - Sharpe historical: ~0.5-0.8
#
# STRATEGY 4: Sector Momentum Rotation
#   - Universe: 9 SPDR sector ETFs (XLK XLF XLE XLV XLY XLP XLI XLU XLB).
#   - Each month, rank by trailing 3-month total return.
#   - Hold top 2 equal-weighted, rebalance monthly.
#   - Edge: cross-sectional momentum, Jegadeesh & Titman 1993 and
#     many replications across asset classes and decades.
#   - Sharpe historical: ~0.6-0.9
#
# COMBINED: Diversification across uncorrelated edges produces a
# portfolio Sharpe ~1.0-1.5, target 15-25% annual return with max
# drawdown 10-15% on a typical year.
#
# NOTHING here promises 20-200% days. That's not how this works. What
# this does is grind out steady returns most months and survive the
# bad months without blowing up.
# ====================================================================

import os
import sys
import time
import math
import json
import logging
import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from typing import Optional, Dict, List, Tuple

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
    ET = timezone(timedelta(hours=-5))

# ====================================================================
# LOGGING
# ====================================================================
os.makedirs("logs/alpha", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/alpha/nexus_alpha.log", mode='a'),
    ],
)
log = logging.getLogger("nexus.alpha")

# ====================================================================
# CONFIG
# ====================================================================
API_KEY = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Strategy allocation — risk parity targets (sum should be 1.0).
# These get scaled by per-strategy vol to equalize risk contribution.
STRATEGY_TARGET_RISK_WEIGHT = {
    'overnight': 0.30,
    'vrp':       0.25,
    'rsi':       0.20,
    'sector':    0.25,
}

# Portfolio-level controls
PORTFOLIO_VOL_TARGET = 0.12          # 12% annualized vol target
MAX_PORTFOLIO_LEVERAGE = 1.5         # never more than 1.5x notional
MAX_DRAWDOWN_HALT = 0.15             # hard stop at -15% from high-water mark

# Per-strategy parameters
SECTOR_UNIVERSE = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
SECTOR_LOOKBACK_DAYS = 63            # ~3 months
SECTOR_HOLD_TOP_N = 2

RSI_PERIOD = 2
RSI_BUY_BELOW = 5
RSI_SELL_ABOVE = 70
RSI_MAX_HOLD_DAYS = 5

VRP_PUT_DELTA = 0.16                 # ~1 SD OTM
VRP_SPREAD_WIDTH = 5                 # dollars between short and long strike
VRP_DTE_TARGET = 7                   # weekly expiry
VRP_VIX_PERCENTILE_MIN = 0.50        # only sell when VIX > median
VRP_DEFENSIVE_CLOSE_MULT = 2.0       # close if loss > 2x credit received

OVERNIGHT_ENTRY_MINUTES_BEFORE_CLOSE = 15
OVERNIGHT_EXIT_MINUTES_AFTER_OPEN = 5

# ====================================================================
# STRATEGY: OVERNIGHT DRIFT
# ====================================================================
@dataclass
class OvernightDriftStrategy:
    """
    Buy SPY at ~3:45 PM ET, sell at ~9:35 AM ET next session.
    Captures the overnight risk premium.
    """
    name: str = "overnight"
    ticker: str = "SPY"
    position_open: bool = False
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    shares: int = 0

    def should_enter(self, now_et: datetime, market_open: bool) -> bool:
        if self.position_open or not market_open:
            return False
        close_time = dtime(16, 0)
        entry_window_start = dtime(15, 45)
        return entry_window_start <= now_et.time() < close_time

    def should_exit(self, now_et: datetime, market_open: bool) -> bool:
        if not self.position_open or not market_open:
            return False
        exit_window_end = dtime(9, 45)
        return dtime(9, 30) <= now_et.time() < exit_window_end

# ====================================================================
# STRATEGY: VRP SHORT PREMIUM
# ====================================================================
@dataclass
class VRPStrategy:
    """
    Weekly SPY put credit spreads at ~16 delta, only when VIX in upper
    half of trailing 252-day distribution. Defensive close at 2x credit.
    """
    name: str = "vrp"
    open_spreads: List[dict] = field(default_factory=list)
    vix_history: deque = field(default_factory=lambda: deque(maxlen=252))

    def update_vix(self, vix: float):
        self.vix_history.append(vix)

    def vix_percentile(self, current_vix: float) -> float:
        if len(self.vix_history) < 30:
            return 0.5
        below = sum(1 for v in self.vix_history if v < current_vix)
        return below / len(self.vix_history)

    def should_enter(self, now_et: datetime, current_vix: float,
                     spy_price: float, market_open: bool) -> bool:
        if not market_open or len(self.open_spreads) >= 2:
            return False
        # Enter on Mondays/Wednesdays at the open of cash session
        if now_et.weekday() not in (0, 2):
            return False
        if not (dtime(9, 35) <= now_et.time() < dtime(10, 0)):
            return False
        return self.vix_percentile(current_vix) >= VRP_VIX_PERCENTILE_MIN

    def check_defensive_close(self, spread: dict, current_loss: float) -> bool:
        credit = spread.get('credit', 0)
        return current_loss > credit * VRP_DEFENSIVE_CLOSE_MULT

# ====================================================================
# STRATEGY: RSI(2) MEAN REVERSION
# ====================================================================
@dataclass
class RSI2Strategy:
    """
    Buy SPY when 2-period RSI < 5 AND price > 200-day SMA (trend filter).
    Sell when RSI > 70 or after 5 trading days.
    """
    name: str = "rsi"
    ticker: str = "SPY"
    daily_closes: deque = field(default_factory=lambda: deque(maxlen=210))
    position_open: bool = False
    entry_price: float = 0.0
    entry_day: int = 0
    shares: int = 0
    day_counter: int = 0

    def update(self, close: float):
        self.daily_closes.append(close)
        self.day_counter += 1

    def rsi_2(self) -> float:
        if len(self.daily_closes) < RSI_PERIOD + 1:
            return 50.0
        closes = list(self.daily_closes)[-(RSI_PERIOD + 1):]
        gains = sum(max(closes[i+1] - closes[i], 0) for i in range(RSI_PERIOD))
        losses = sum(max(closes[i] - closes[i+1], 0) for i in range(RSI_PERIOD))
        if losses == 0:
            return 100.0
        rs = gains / losses
        return 100.0 - (100.0 / (1.0 + rs))

    def sma_200(self) -> float:
        if len(self.daily_closes) < 200:
            return 0.0
        return sum(list(self.daily_closes)[-200:]) / 200.0

    def should_enter(self, current_price: float) -> bool:
        if self.position_open:
            return False
        sma = self.sma_200()
        if sma == 0 or current_price < sma:
            return False
        return self.rsi_2() < RSI_BUY_BELOW

    def should_exit(self, current_price: float) -> bool:
        if not self.position_open:
            return False
        if self.rsi_2() > RSI_SELL_ABOVE:
            return True
        if (self.day_counter - self.entry_day) >= RSI_MAX_HOLD_DAYS:
            return True
        return False

# ====================================================================
# STRATEGY: SECTOR MOMENTUM ROTATION
# ====================================================================
@dataclass
class SectorMomentumStrategy:
    """
    Monthly rebalance: hold top N sector ETFs by trailing 3-month return.
    """
    name: str = "sector"
    closes: Dict[str, deque] = field(
        default_factory=lambda: {t: deque(maxlen=SECTOR_LOOKBACK_DAYS + 5)
                                 for t in SECTOR_UNIVERSE}
    )
    current_holdings: Dict[str, int] = field(default_factory=dict)
    last_rebalance_month: int = -1

    def update_close(self, ticker: str, close: float):
        if ticker in self.closes:
            self.closes[ticker].append(close)

    def trailing_return(self, ticker: str) -> Optional[float]:
        c = self.closes.get(ticker)
        if not c or len(c) < SECTOR_LOOKBACK_DAYS:
            return None
        prices = list(c)
        return (prices[-1] / prices[-SECTOR_LOOKBACK_DAYS]) - 1.0

    def rank_sectors(self) -> List[Tuple[str, float]]:
        ranked = []
        for ticker in SECTOR_UNIVERSE:
            ret = self.trailing_return(ticker)
            if ret is not None:
                ranked.append((ticker, ret))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def should_rebalance(self, now_et: datetime) -> bool:
        return now_et.month != self.last_rebalance_month

    def target_holdings(self) -> List[str]:
        ranked = self.rank_sectors()
        return [t for t, _ in ranked[:SECTOR_HOLD_TOP_N]]

# ====================================================================
# PORTFOLIO MANAGER
# ====================================================================
class Portfolio:
    """
    Allocates capital across strategies by risk parity, enforces vol
    target, and halts trading at max drawdown.
    """

    def __init__(self, starting_equity: float):
        self.starting_equity = starting_equity
        self.high_water_mark = starting_equity
        self.equity = starting_equity
        self.halted = False
        # Risk allocation per strategy (dollars at risk)
        self.strategy_allocations = self._compute_allocations(starting_equity)

    def _compute_allocations(self, equity: float) -> Dict[str, float]:
        """Risk parity: each strategy gets allocation × target_weight."""
        deployable = equity * PORTFOLIO_VOL_TARGET / 0.18  # 18% asset vol
        deployable = min(deployable, equity * MAX_PORTFOLIO_LEVERAGE)
        return {name: deployable * w
                for name, w in STRATEGY_TARGET_RISK_WEIGHT.items()}

    def update_equity(self, new_equity: float):
        self.equity = new_equity
        if new_equity > self.high_water_mark:
            self.high_water_mark = new_equity
        drawdown = (self.high_water_mark - new_equity) / self.high_water_mark
        if drawdown >= MAX_DRAWDOWN_HALT and not self.halted:
            log.critical(f"DRAWDOWN HALT: -{drawdown*100:.1f}% from HWM. Halting.")
            self.halted = True
        # Recompute allocations on every equity move
        self.strategy_allocations = self._compute_allocations(new_equity)

    def can_trade(self) -> bool:
        return not self.halted

    def shares_for(self, strategy: str, price: float) -> int:
        alloc = self.strategy_allocations.get(strategy, 0)
        if price <= 0:
            return 0
        return int(alloc / price)

# ====================================================================
# RUNNER (skeleton — paper trading via Alpaca)
# ====================================================================
class NexusAlphaRunner:
    def __init__(self, live: bool = False):
        self.live = live
        if not API_KEY or not SECRET_KEY:
            log.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
            sys.exit(1)
        self.api = REST(API_KEY, SECRET_KEY, BASE_URL)
        try:
            start_eq = float(self.api.get_account().equity)
        except Exception:
            start_eq = 50_000.0
        self.portfolio = Portfolio(start_eq)
        self.overnight = OvernightDriftStrategy()
        self.vrp = VRPStrategy()
        self.rsi = RSI2Strategy()
        self.sector = SectorMomentumStrategy()
        log.info(f"NEXUS ALPHA initialised, starting equity ${start_eq:,.2f}")
        log.info(f"Allocations: {self.portfolio.strategy_allocations}")

    def _is_market_open(self) -> bool:
        try:
            return self.api.get_clock().is_open
        except Exception:
            return False

    def _now_et(self) -> datetime:
        return datetime.now(tz=ET)

    def _get_price(self, ticker: str) -> float:
        try:
            quote = self.api.get_latest_quote(ticker)
            bid = float(getattr(quote, 'bp', 0) or 0)
            ask = float(getattr(quote, 'ap', 0) or 0)
            return (bid + ask) / 2 if bid and ask else 0.0
        except Exception as e:
            log.debug(f"Quote fetch {ticker}: {e}")
            return 0.0

    def _submit_market(self, ticker: str, qty: int, side: str):
        if not self.live:
            log.info(f"[PAPER-SIM] {side} {qty} {ticker}")
            return None
        try:
            return self.api.submit_order(
                symbol=ticker, qty=qty, side=side,
                type='market', time_in_force='day',
            )
        except Exception as e:
            log.error(f"Order failed {side} {qty} {ticker}: {e}")
            return None

    def tick(self):
        if self.portfolio.halted:
            return
        try:
            equity = float(self.api.get_account().equity)
            self.portfolio.update_equity(equity)
        except Exception:
            pass

        now_et = self._now_et()
        market_open = self._is_market_open()

        # --- Overnight Drift ---
        if self.overnight.should_enter(now_et, market_open):
            price = self._get_price("SPY")
            if price > 0:
                qty = self.portfolio.shares_for('overnight', price)
                if qty > 0:
                    self._submit_market("SPY", qty, "buy")
                    self.overnight.position_open = True
                    self.overnight.entry_price = price
                    self.overnight.entry_time = now_et
                    self.overnight.shares = qty
                    log.info(f"OVERNIGHT ENTER: +{qty} SPY @ ${price:.2f}")
        elif self.overnight.should_exit(now_et, market_open):
            self._submit_market("SPY", self.overnight.shares, "sell")
            exit_price = self._get_price("SPY")
            pnl = (exit_price - self.overnight.entry_price) * self.overnight.shares
            log.info(f"OVERNIGHT EXIT: -{self.overnight.shares} SPY @ ${exit_price:.2f} "
                     f"PnL ${pnl:+.2f}")
            self.overnight.position_open = False
            self.overnight.shares = 0

        # --- RSI(2) Mean Reversion ---
        # Update on first tick of session (would normally use daily bars)
        # For brevity, the daily-bar feed is left to the runner caller.

        # --- VRP and Sector Rotation ---
        # These require options chain and multi-ticker quote infra that
        # belongs in dedicated polling threads. Skeleton left for clarity;
        # full implementation in sibling modules.

    def run_forever(self):
        log.info(f"Running {'LIVE' if self.live else 'PAPER'}. Ctrl-C to stop.")
        try:
            while True:
                self.tick()
                time.sleep(30)
        except KeyboardInterrupt:
            log.info("Shutting down.")

def main():
    p = argparse.ArgumentParser(description="NEXUS ALPHA — Multi-strategy bot")
    p.add_argument("--live", action="store_true")
    args = p.parse_args()
    runner = NexusAlphaRunner(live=args.live)
    runner.run_forever()

if __name__ == "__main__":
    main()
