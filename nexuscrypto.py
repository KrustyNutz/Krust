from __future__ import annotations

# ====================================================================
# NEXUS CRYPTO V2.1 | Regime-Aware Crypto Scalper
#
# Forked from NEXUS V2 (equities). Key differences:
#   - 24/7 market: no session open/close, uses volume-window awareness
#   - Wider stops: crypto moves 5-10x more than SPY per tick
#   - Fractional sizing: trades 0.001+ BTC, not 100 shares
#   - Rolling VWAP: anchored to sliding window, not session open
#   - Crypto-specific news keywords (SEC, ETF, halving, hack)
#   - Higher vol thresholds: SPY's "volatile" is crypto's "Tuesday"
#   - Symbols: BTCUSD, ETHUSD via Alpaca crypto API
#
# Usage:
#   python nexus_crypto.py                # Observe mode
#   python nexus_crypto.py --live         # Paper trading
#   python nexus_crypto.py --headless     # No GUI
# ====================================================================

import threading
import time
import asyncio
import sys
import os
import re
import math
import logging
import argparse
import json
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict, List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# alpaca-trade-api is in maintenance mode; crypto support is brittle.
# If orders/news fail, migrate to alpaca-py.
from alpaca_trade_api.rest import REST

try:
    import websocket as ws_lib
except ImportError:
    # `log` is not initialised yet — use stderr so this surfaces.
    print("ERROR: Missing websocket-client: pip install websocket-client", file=sys.stderr)
    sys.exit(1)

try:
    import tkinter as tk
    HAS_TK = True
except ImportError:
    HAS_TK = False

from filters import OneEuroFilter
from signal_logger import SignalEvent, SignalLogger

# ====================================================================
# LOGGING
# ====================================================================
os.makedirs("logs/crypto", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/crypto/nexus_crypto.log", mode='a'),
    ]
)
log = logging.getLogger("nexus.crypto")

# ====================================================================
# CONFIG
# ====================================================================
API_KEY = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not SECRET_KEY:
    log.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
    sys.exit(1)

# --- Crypto Watchlist ---
# Alpaca crypto uses BTCUSD/ETHUSD for old SDK
WATCHLIST = ["BTCUSD", "ETHUSD"]
DISPLAY_NAMES = {"BTCUSD": "BTC", "ETHUSD": "ETH"}

# --- Volume Windows (UTC) ---
# Crypto has no open/close but volume peaks during US + Asia sessions
# US session:  13:30-20:00 UTC (9:30-4:00 ET)
# Asia session: 00:00-08:00 UTC
# These are the best windows for scalping — liquidity is highest
HIGH_VOL_WINDOWS = [
    (13, 30, 20, 0),   # US session
    (0, 0, 8, 0),      # Asia session
]
REQUIRE_HIGH_VOL = False  # Set True to only trade during peak windows


def in_high_vol_window(now_utc: datetime) -> bool:
    minutes = now_utc.hour * 60 + now_utc.minute
    for (sh, sm, eh, em) in HIGH_VOL_WINDOWS:
        start = sh * 60 + sm
        end = eh * 60 + em
        if start <= minutes < end:
            return True
    return False

# --- Risk Parameters ---
MAX_SESSION_LOSS = 500.00       # Tighter for crypto volatility
SIGNAL_COOLDOWN_SEC = 180       # 3 min between signals (crypto trends longer)
# Spread is stored as a fraction (e.g. 0.001 = 10 bps). 10 bps is wide for BTC
# in normal markets — anything past this means dislocated/illiquid book.
MAX_SPREAD_FRACTION = 0.001
MAX_OPEN_POSITIONS = 2

# --- Position Sizing (fractional) ---
BTC_BASE_QTY = 0.002            # ~$200 at $100k BTC
ETH_BASE_QTY = 0.05             # ~$200 at $4k ETH
BTC_MIN_QTY = 0.0001            # Alpaca crypto minimums
ETH_MIN_QTY = 0.001
MAX_QTY_MULTIPLIER = 3.0        # High-confidence = up to 3x base
QTY_CONFIDENCE_MULT = 3.0       # Multiplier applied to confidence before capping

# --- Stop Parameters (% based, MUCH wider than equities) ---
TRAILING_STOP_PCT = 0.25        # 0.25% trailing (vs 0.10% for SPY)
HARD_STOP_PCT = 0.50            # 0.50% max loss per trade
MAX_HOLD_SECONDS = 600          # 10 min max hold (crypto trends longer)
# Cooldown to avoid immediate reentry after a stop-out (crypto != securities,
# so this is not an IRS wash-sale concern — just trade-flow hygiene).
REENTRY_COOLDOWN = 60

# --- Signal Thresholds ---
# Crypto needs different thresholds because volatility is structurally higher
EMA_PERIOD = 100
WARMUP_SECONDS = 180            # 3 min warmup (crypto needs more data)

SIGNAL_THRESHOLD = {
    'TRENDING': 5.0,
    'MEAN_REVERT': 5.5,
    'VOLATILE': 99.0,
}

REGIME_WEIGHTS = {
    'TRENDING': {
        'momentum': 3.0,
        'trade_flow': 2.0,
        'vwap_dev': 0.5,
        'rsi': 1.0,
        'obi': 1.0,           # Crypto OBI is noisier — lower weight
        'mtf': 2.5,
    },
    'MEAN_REVERT': {
        'momentum': 0.5,
        'trade_flow': 1.0,
        'vwap_dev': 3.0,
        'rsi': 2.5,
        'obi': 0.5,
        'mtf': 1.0,
    },
    'VOLATILE': {
        'momentum': 0.0,
        'trade_flow': 0.0,
        'vwap_dev': 0.0,
        'rsi': 0.0,
        'obi': 0.0,
        'mtf': 0.0,
    },
}

# --- Crypto News Keywords ---
CRYPTO_NEWS_KEYWORDS = {
    'sec': (90, 3), 'securities and exchange': (90, 3),
    'etf approval': (60, 3), 'etf rejected': (60, 3), 'etf denied': (60, 3),
    'bitcoin etf': (60, 3), 'ethereum etf': (60, 3),
    'halving': (120, 2), 'halvening': (120, 2),
    'hack': (60, 3), 'hacked': (60, 3), 'exploit': (60, 3),
    'exchange hack': (90, 3), 'stolen': (60, 2),
    'regulation': (60, 2), 'ban': (90, 3), 'banned': (90, 3),
    'tether': (45, 2), 'usdt depeg': (90, 3), 'stablecoin': (45, 2),
    'whale': (30, 1), 'whale alert': (30, 1),
    'fed': (60, 2), 'fomc': (90, 3), 'interest rate': (60, 2),
    'tariff': (60, 2), 'trade war': (60, 2),
    'mt gox': (60, 2), 'genesis': (45, 2),
    'binance': (45, 2), 'coinbase': (45, 2), 'kraken': (45, 2),
}

BULLISH_WORDS = {
    'surge', 'rally', 'bullish', 'soar', 'jump', 'gain', 'climb',
    'breakout', 'approval', 'approved', 'adoption', 'institutional',
    'inflow', 'inflows', 'accumulation', 'buy', 'upgrade', 'ath',
    'record high', 'moon', 'pump', 'recovery', 'rebound',
}

BEARISH_WORDS = {
    'crash', 'plunge', 'bearish', 'tumble', 'dump', 'sell', 'selloff',
    'hack', 'hacked', 'exploit', 'stolen', 'ban', 'banned', 'rejected',
    'denied', 'fear', 'panic', 'liquidation', 'liquidated', 'outflow',
    'outflows', 'depeg', 'depegged', 'rug', 'scam', 'fraud',
}


# ====================================================================
# SHARED GUI STATE
# ====================================================================
def _empty_ticker_state() -> dict:
    return {'price': 0, 'smooth': 0, 'vwap': 0, 'z_score': 0,
            'velocity': 0, 'accel': 0, 'rsi': 50, 'obi': 0,
            'spread': 0, 'regime': '---', 'vol_ratio': 0,
            'net_flow': 0, 'confidence': 0, 'votes': {},
            'bar5s': 0, 'bar30s': 0, 'bar5m': 0,
            'last_signal': 'NONE', 'last_active_signal': 'NONE'}


gui_state = {
    'equity': 0.0, 'session_pnl': 0.0, 'mode': 'OBSERVE',
    'status': 'INITIALIZING...',
    'signal_counts': {'CALL': 0, 'PUT': 0, 'NONE': 0, 'GATED': 0},
    'tickers': {t: _empty_ticker_state() for t in WATCHLIST},
    'trade_log': deque(maxlen=15),
    'positions': {}, 'outcomes': {}, 'news': {},
}

# Thread-safety: gui_log() runs on the worker thread, update_gui() on the
# Tk main thread. Mutating a list while another thread iterates raises
# RuntimeError; the lock + deque + snapshot in update_gui closes that race.
_gui_lock = threading.Lock()


def gui_log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    with _gui_lock:
        gui_state['trade_log'].appendleft(entry)


# ====================================================================
# REUSABLE COMPONENTS (same logic as equity NEXUS, tuned for crypto)
# ====================================================================

class BarAggregator:
    def __init__(self, period_seconds: int):
        self.period = period_seconds
        self.bars: deque = deque(maxlen=200)
        self._current_open = 0.0
        self._current_high = -math.inf
        self._current_low = math.inf
        self._current_close = 0.0
        self._current_ticks = 0
        self._current_cum_pv = 0.0   # Σ(price · size)
        self._current_cum_v = 0.0    # Σ(size)
        self._bar_start = 0.0

    def tick(self, price: float, now: float, size: float = 1.0):
        if self._bar_start == 0:
            self._bar_start = now
        if self._current_open == 0:
            self._current_open = price
        self._current_high = max(self._current_high, price)
        self._current_low = min(self._current_low, price)
        self._current_close = price
        self._current_ticks += 1
        # Real VWAP needs trade size. If callers pass size=1 it degrades to
        # the previous mean-price behaviour without breaking anything.
        self._current_cum_pv += price * size
        self._current_cum_v += size
        if (now - self._bar_start) >= self.period and self._current_ticks > 0:
            vwap = (self._current_cum_pv / self._current_cum_v
                    if self._current_cum_v > 0 else self._current_close)
            bar = {
                'open': self._current_open, 'high': self._current_high,
                'low': self._current_low, 'close': self._current_close,
                'ticks': self._current_ticks,
                'vwap': vwap,
                'range_pct': ((self._current_high - self._current_low) /
                              self._current_low * 100) if self._current_low > 0 else 0,
                'time': now,
            }
            self.bars.append(bar)
            self._current_open = 0.0
            self._current_high = -math.inf
            self._current_low = math.inf
            self._current_ticks = 0
            self._current_cum_pv = 0.0
            self._current_cum_v = 0.0
            self._bar_start = now
            return bar
        return None

    def direction(self, lookback: int = 3) -> float:
        if len(self.bars) < lookback:
            return 0.0
        recent = list(self.bars)[-lookback:]
        dirs = []
        for b in recent:
            if b['close'] > b['open']:
                dirs.append(1.0)
            elif b['close'] < b['open']:
                dirs.append(-1.0)
            else:
                dirs.append(0.0)
        return sum(dirs) / len(dirs)

    def avg_range(self, lookback: int = 10) -> float:
        if len(self.bars) < 2:
            return 0.0
        recent = list(self.bars)[-lookback:]
        return sum(b['range_pct'] for b in recent) / len(recent) if recent else 0.0


class RollingVWAP:
    """Rolling VWAP over a fixed window (not session-anchored)."""
    def __init__(self, window: int = 600):
        self.window = window
        self.prices: deque = deque(maxlen=window)
        self.vwap = 0.0
        self.std = 0.0

    def update(self, price: float):
        self.prices.append(price)
        n = len(self.prices)
        if n > 10:
            self.vwap = sum(self.prices) / n
            # Sample variance (n-1) to match RegimeDetector._stdev.
            variance = sum((p - self.vwap) ** 2 for p in self.prices) / (n - 1)
            self.std = math.sqrt(max(variance, 0.0))

    def z_score(self, price: float) -> float:
        return (price - self.vwap) / self.std if self.std > 0 else 0.0


class TradeFlowAnalyzer:
    def __init__(self, window: int = 60):
        self.last_price = 0.0
        self.last_direction = 0
        self.deltas: deque = deque(maxlen=window)

    def classify_tick(self, price: float) -> int:
        if self.last_price == 0:
            self.last_price = price
            return 0
        if price > self.last_price:
            direction = 1
        elif price < self.last_price:
            direction = -1
        else:
            direction = self.last_direction
        self.last_price = price
        self.last_direction = direction
        self.deltas.append(direction)
        return direction

    def net_flow(self) -> float:
        return sum(self.deltas) / len(self.deltas) if self.deltas else 0.0

    def flow_acceleration(self) -> float:
        if len(self.deltas) < 10:
            return 0.0
        d = list(self.deltas)
        mid = len(d) // 2
        first = sum(d[:mid]) / mid
        second = sum(d[mid:]) / (len(d) - mid)
        return second - first


class RSICalculator:
    def __init__(self, period: int = 14):
        self.period = period
        self.gains: deque = deque(maxlen=period)
        self.losses: deque = deque(maxlen=period)
        self.last_price = 0.0
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.count = 0
        self.value = 50.0

    def update(self, price: float) -> float:
        if self.last_price > 0:
            delta = price - self.last_price
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            self.count += 1
            if self.count <= self.period:
                self.gains.append(gain)
                self.losses.append(loss)
                if self.count == self.period:
                    self.avg_gain = sum(self.gains) / self.period
                    self.avg_loss = sum(self.losses) / self.period
            else:
                self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
                self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period
            if self.avg_loss == 0:
                self.value = 100.0
            else:
                rs = self.avg_gain / self.avg_loss
                self.value = 100.0 - (100.0 / (1.0 + rs))
        self.last_price = price
        return self.value


class RegimeDetector:
    def __init__(self, fast_window: int = 30, slow_window: int = 300):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.returns: deque = deque(maxlen=slow_window)
        self.last_price = 0.0
        self.regime = "MEAN_REVERT"
        self.vol_ratio = 1.0
        self.fast_vol = 0.0
        self.slow_vol = 0.0

    def update(self, price: float, bar5s_dir: float, bar30s_dir: float,
               z_score: float) -> str:
        if self.last_price > 0:
            ret = ((price - self.last_price) / self.last_price) * 100.0
            self.returns.append(ret)
        self.last_price = price
        returns_list = list(self.returns)
        self.fast_vol = self._stdev(returns_list, self.fast_window)
        self.slow_vol = self._stdev(returns_list, self.slow_window)
        self.vol_ratio = (self.fast_vol / self.slow_vol) if self.slow_vol > 0 else 1.0

        dir_agreement = bar5s_dir * bar30s_dir

        # Crypto-tuned thresholds: higher vol baseline
        if self.vol_ratio > 2.5 or self.fast_vol > 0.08:
            self.regime = "VOLATILE"
        elif abs(dir_agreement) > 0.3 and abs(bar30s_dir) > 0.5:
            self.regime = "TRENDING"
        else:
            self.regime = "MEAN_REVERT"
        return self.regime

    @staticmethod
    def _stdev(data: list, window: int, min_sample: int = 30) -> float:
        # Return a partial estimate once min_sample is hit so vol_ratio
        # doesn't snap from 1.0 → real-value the instant the full window fills.
        if len(data) < min_sample:
            return 0.0
        subset = data[-window:]
        if len(subset) < 2:
            return 0.0
        mean = sum(subset) / len(subset)
        variance = sum((x - mean) ** 2 for x in subset) / (len(subset) - 1)
        return math.sqrt(variance)


# ====================================================================
# SIGNAL COMPONENTS (same as equity, tuned thresholds)
# ====================================================================

def vote_momentum(velocity: float, accel: float) -> int:
    # Crypto velocity thresholds: ~5x wider than equity
    if velocity > 0.001 and accel > 0:
        return 1
    elif velocity < -0.001 and accel < 0:
        return -1
    return 0

def vote_trade_flow(net_flow: float, flow_accel: float) -> int:
    if net_flow > 0.15 and flow_accel > 0:
        return 1
    elif net_flow < -0.15 and flow_accel < 0:
        return -1
    return 0

def vote_vwap_deviation(z_score: float, regime: str) -> int:
    if regime == "MEAN_REVERT":
        if z_score < -2.0:
            return 1
        elif z_score > 2.0:
            return -1
    elif regime == "TRENDING":
        if z_score > 0.5:
            return 1
        elif z_score < -0.5:
            return -1
    return 0

def vote_rsi(rsi_value: float, regime: str) -> int:
    if regime == "MEAN_REVERT":
        if rsi_value < 30:
            return 1
        elif rsi_value > 70:
            return -1
    elif regime == "TRENDING":
        if rsi_value < 25:
            return 1
        elif rsi_value > 75:
            return -1
    return 0

def vote_obi(obi: float) -> int:
    # Wider threshold for crypto — OBI is noisier
    if obi > 0.12:
        return 1
    elif obi < -0.12:
        return -1
    return 0

def vote_mtf(bar5s_dir: float, bar30s_dir: float, bar5m_dir: float) -> int:
    if bar5s_dir > 0.3 and bar30s_dir > 0.3 and bar5m_dir > 0:
        return 1
    elif bar5s_dir < -0.3 and bar30s_dir < -0.3 and bar5m_dir < 0:
        return -1
    return 0

def compute_composite_signal(votes, regime, spread, vol_ratio):
    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS['MEAN_REVERT'])
    threshold = SIGNAL_THRESHOLD.get(regime, 99.0)
    raw_score = 0.0
    max_possible = 0.0
    components = []
    for name, vote in votes.items():
        w = weights.get(name, 0.0)
        raw_score += vote * w
        max_possible += abs(w)
        if vote != 0:
            components.append(f"{name}={vote:+d}")
    spread_penalty = min((spread / MAX_SPREAD_FRACTION) * 0.3, 0.3)
    adjusted_score = raw_score * (1.0 - spread_penalty)
    if vol_ratio > 1.5:
        adjusted_score *= (1.0 / vol_ratio)
    confidence = abs(adjusted_score) / max_possible if max_possible > 0 else 0
    reason_parts = [f"regime={regime}", f"score={adjusted_score:.2f}/{threshold:.1f}"]
    reason_parts.extend(components)
    reason = " | ".join(reason_parts)
    if adjusted_score > threshold:
        return "CALL", confidence, reason, adjusted_score
    elif adjusted_score < -threshold:
        return "PUT", confidence, reason, adjusted_score
    return "NONE", confidence, reason, adjusted_score


# ====================================================================
# DATA STREAM (Raw websocket to v1beta3 — old SDK endpoint is dead)
# ====================================================================
# Map between order symbols (BTCUSD) and stream symbols (BTC/USD)
STREAM_SYMBOLS = {"BTCUSD": "BTC/USD", "ETHUSD": "ETH/USD"}
REVERSE_SYMBOLS = {"BTC/USD": "BTCUSD", "ETH/USD": "ETHUSD"}
CRYPTO_WS_URL = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"

class CryptoDataStream:
    def __init__(self):
        self.market_data = {
            t: {'p': 0.0, 's': 0.0, 'spread': 0.0, 'obi': 0.0}
            for t in WATCHLIST
        }
        self._connected = False
        self._ws = None

    def _on_open(self, ws):
        log.info("Crypto WS connected. Authenticating...")
        auth_msg = json.dumps({
            "action": "auth",
            "key": API_KEY,
            "secret": SECRET_KEY
        })
        ws.send(auth_msg)

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except Exception:
            return

        if not isinstance(data, list):
            data = [data]

        for msg in data:
            msg_type = msg.get("T", "")

            if msg_type == "success":
                if msg.get("msg") == "authenticated":
                    log.info("Crypto WS authenticated. Subscribing...")
                    stream_syms = [STREAM_SYMBOLS[t] for t in WATCHLIST]
                    sub_msg = json.dumps({
                        "action": "subscribe",
                        "trades": stream_syms,
                        "quotes": stream_syms,
                    })
                    ws.send(sub_msg)
                    self._connected = True
                elif msg.get("msg") == "connected":
                    log.info("Crypto WS handshake OK.")

            elif msg_type == "subscription":
                log.info(f"Subscribed: trades={msg.get('trades',[])} "
                         f"quotes={msg.get('quotes',[])}")

            elif msg_type == "t":  # Trade
                stream_sym = msg.get("S", "")
                ticker = REVERSE_SYMBOLS.get(stream_sym, "")
                if ticker and ticker in self.market_data:
                    self.market_data[ticker]['p'] = float(msg.get("p", 0))
                    self.market_data[ticker]['s'] = float(msg.get("s", 0))

            elif msg_type == "q":  # Quote
                stream_sym = msg.get("S", "")
                ticker = REVERSE_SYMBOLS.get(stream_sym, "")
                if ticker and ticker in self.market_data:
                    bp = float(msg.get("bp", 0))
                    ap = float(msg.get("ap", 0))
                    bs = float(msg.get("bs", 0))
                    ask_s = float(msg.get("as", 0))
                    if bp > 0 and ap > 0:
                        self.market_data[ticker]['spread'] = (ap - bp) / bp
                    total = bs + ask_s
                    if total > 0:
                        self.market_data[ticker]['obi'] = (bs - ask_s) / total

            elif msg_type == "error":
                log.error(f"Crypto WS error: {msg}")

    def _on_error(self, ws, error):
        log.error(f"Crypto WS error: {error}")

    def _on_close(self, ws, close_status, close_msg):
        log.warning(f"Crypto WS closed: {close_status} {close_msg}")
        self._connected = False

    def start_stream(self):
        log.info(f"Connecting to {CRYPTO_WS_URL}...")
        while True:
            try:
                self._ws = ws_lib.WebSocketApp(
                    CRYPTO_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                log.error(f"Crypto stream crashed: {e}")
            log.info("Reconnecting in 5s...")
            time.sleep(5)


# ====================================================================
# EXECUTION ENGINE (Crypto — fractional sizing)
# ====================================================================
class CryptoExecutionEngine:
    def __init__(self, live: bool = False):
        self.live = live
        self.api = REST(API_KEY, SECRET_KEY, BASE_URL)
        try:
            self.start_equity = float(self.api.get_account().equity)
        except Exception:
            self.start_equity = 50000.0
        self.session_pnl = 0.0
        self.realized_pnl = 0.0           # local tally, refreshed by sync()
        self.last_signal_time: Dict[str, float] = {}
        self.open_trades: Dict[str, dict] = {}
        self.last_close_time: Dict[str, float] = {}

    def _close_managed_positions(self):
        """Close only positions this bot opened — never the whole account."""
        for ticker in list(self.open_trades.keys()):
            trade = self.open_trades[ticker]
            close_side = 'sell' if trade['order_side'] == 'buy' else 'buy'
            order_symbol = STREAM_SYMBOLS.get(ticker, ticker)
            for sym in (order_symbol, ticker):
                try:
                    self.api.submit_order(
                        symbol=sym, qty=trade['qty'], side=close_side,
                        type='market', time_in_force='gtc',
                    )
                    break
                except Exception as e:
                    log.error(f"Kill-switch close failed for {sym}: {e}")

    def sync(self):
        try:
            acc = self.api.get_account()
            equity = float(acc.equity)
            self.session_pnl = equity - self.start_equity
            if self.session_pnl < -MAX_SESSION_LOSS:
                log.critical(f"MAX LOSS: ${self.session_pnl:.2f} — flattening managed positions")
                if self.live:
                    self._close_managed_positions()
                # sys.exit() from a daemon thread is a no-op; os._exit kills
                # the whole process so the GUI doesn't keep running blind.
                os._exit(1)
        except Exception as e:
            log.debug(f"Account sync: {e}")
            return

        # Position sync — filter to our watchlist so unrelated account
        # holdings (equities, other coins) don't leak into the GUI.
        try:
            positions = self.api.list_positions()
            pos_dict = {}
            for p in positions:
                try:
                    sym = p.symbol
                    # Alpaca may report either 'BTCUSD' or 'BTC/USD'
                    canonical = REVERSE_SYMBOLS.get(sym, sym)
                    if canonical not in WATCHLIST:
                        continue
                    pos_dict[canonical] = {
                        'gain': float(p.unrealized_plpc) * 100,
                        'pnl': float(p.unrealized_pl),
                        'qty': float(p.qty),
                    }
                except Exception:
                    pass
            gui_state['positions'] = pos_dict
        except Exception as e:
            log.debug(f"Position sync: {e}")

    def can_signal(self, ticker: str, now: float) -> tuple[bool, str]:
        """Returns (allowed, reason_if_blocked)."""
        last = self.last_signal_time.get(ticker, 0)
        if (now - last) < SIGNAL_COOLDOWN_SEC:
            return False, "cooldown"
        if len(self.open_trades) >= MAX_OPEN_POSITIONS:
            return False, "max_positions"
        if ticker in self.open_trades:
            return False, "already_open"
        last_close = self.last_close_time.get(ticker, 0)
        if (now - last_close) < REENTRY_COOLDOWN:
            return False, "reentry_cooldown"
        if REQUIRE_HIGH_VOL and not in_high_vol_window(datetime.utcnow()):
            return False, "outside_vol_window"
        return True, ""

    def manage_positions(self, ticker: str, current_price: float, now: float):
        if ticker not in self.open_trades:
            return
        trade = self.open_trades[ticker]
        entry = trade['entry']
        side = trade['side']
        elapsed = now - trade['time']
        conf = trade.get('confidence', 0.5)

        # Confidence-scaled stops
        if conf >= 0.7:
            stop_scale = 1.5
        elif conf >= 0.4:
            stop_scale = 1.0
        else:
            stop_scale = 0.75

        trail = TRAILING_STOP_PCT * stop_scale
        hard = HARD_STOP_PCT * stop_scale

        if side == "CALL":
            pnl_pct = ((current_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - current_price) / entry) * 100
        trade['peak'] = max(trade.get('peak', pnl_pct), pnl_pct)
        peak = trade['peak']
        exit_reason = None
        if pnl_pct < -hard:
            exit_reason = f"HARD_STOP {pnl_pct:+.3f}%"
        elif peak > 0.05 and pnl_pct < (peak - trail):
            exit_reason = f"TRAIL_STOP {pnl_pct:+.3f}% (pk={peak:.3f}%)"
        elif elapsed > MAX_HOLD_SECONDS:
            exit_reason = f"TIME_STOP {elapsed:.0f}s"
        if exit_reason:
            self._close_position(ticker, exit_reason, pnl_pct)

    def execute(self, ticker: str, side: str, price: float,
                spread: float, confidence: float, now: float):
        self.last_signal_time[ticker] = now
        if not self.live:
            return
        if spread > MAX_SPREAD_FRACTION:
            return
        if ticker in self.open_trades:
            return

        # Fractional sizing — capped by MAX_QTY_MULTIPLIER, floored at
        # exchange minimum so low-confidence signals don't generate
        # silently-rejected sub-minimum orders.
        base = BTC_BASE_QTY if "BTC" in ticker else ETH_BASE_QTY
        minimum = BTC_MIN_QTY if "BTC" in ticker else ETH_MIN_QTY
        raw_mult = min(confidence * QTY_CONFIDENCE_MULT, MAX_QTY_MULTIPLIER)
        qty = max(round(base * raw_mult, 6), minimum)

        order_side = 'buy' if side == "CALL" else 'sell'
        order_symbol = STREAM_SYMBOLS.get(ticker, ticker)

        def _submit(sym):
            self.api.submit_order(
                symbol=sym, qty=qty, side=order_side,
                type='market', time_in_force='gtc',
            )

        for attempt_sym in (order_symbol, ticker):
            try:
                _submit(attempt_sym)
                self.open_trades[ticker] = {
                    'side': side, 'entry': price, 'qty': qty,
                    'time': now, 'peak': 0.0, 'order_side': order_side,
                    'confidence': confidence,
                }
                name = DISPLAY_NAMES.get(ticker, ticker)
                log.info(f"CRYPTO: {order_side.upper()} {qty} {name} @ ${price:,.2f}")
                gui_log(f"ENTRY: {qty} {name} {side} @ ${price:,.2f}")
                return
            except Exception as e:
                log.error(f"Crypto order failed for {attempt_sym}: {e}")
                if attempt_sym == ticker:  # last attempt
                    return

    def _close_position(self, ticker: str, reason: str, pnl_pct: float):
        if ticker not in self.open_trades:
            return
        trade = self.open_trades[ticker]
        self.last_close_time[ticker] = time.time()

        # Local realized-PnL tally so the GUI doesn't show stale equity for
        # the second between close and next account sync.
        notional = trade['entry'] * trade['qty']
        realized = notional * (pnl_pct / 100.0)
        self.realized_pnl += realized
        self.session_pnl += realized

        if not self.live:
            del self.open_trades[ticker]
            return

        close_side = 'sell' if trade['order_side'] == 'buy' else 'buy'
        order_symbol = STREAM_SYMBOLS.get(ticker, ticker)
        closed = False
        for attempt_sym in (order_symbol, ticker):
            try:
                self.api.submit_order(
                    symbol=attempt_sym, qty=trade['qty'], side=close_side,
                    type='market', time_in_force='gtc',
                )
                closed = True
                break
            except Exception as e:
                log.error(f"Close failed for {attempt_sym}: {e}")

        result = "WIN" if pnl_pct > 0 else "LOSS"
        name = DISPLAY_NAMES.get(ticker, ticker)
        if closed:
            log.info(f"CLOSED: {name} {reason} | {result} {pnl_pct:+.3f}%")
            gui_log(f"EXIT: {name} {result} {pnl_pct:+.3f}% ({reason})")
        else:
            log.critical(f"COULD NOT CLOSE {name} — position still open on exchange!")
            gui_log(f"!! CLOSE FAILED {name} — manual intervention")
        del self.open_trades[ticker]


# ====================================================================
# TICKER STATE
# ====================================================================
class CryptoTickerState:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.price_filter = OneEuroFilter(min_cutoff=0.005, beta=0.4)
        self.bars_5s = BarAggregator(5)
        self.bars_30s = BarAggregator(30)
        self.bars_5m = BarAggregator(300)
        self.vwap = RollingVWAP(window=600)  # 10-min rolling
        self.trade_flow = TradeFlowAnalyzer(window=60)
        self.rsi = RSICalculator(period=14)
        self.regime = RegimeDetector(fast_window=30, slow_window=300)
        self.last_smooth_price = 0.0
        self.last_velocity = 0.0
        self.last_time = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0


# ====================================================================
# NEWS (simplified inline — crypto-specific)
# ====================================================================
class CryptoNewsEngine:
    def __init__(self, api, poll_interval: int = 120):
        self.api = api
        self.poll_interval = poll_interval
        self.last_poll = 0.0
        self.seen: set = set()
        self.sentiment: float = 0.0
        self.halt_until: float = 0
        self.halt_reason: str = ""
        self.latest: str = ""
        self._api_unavailable = False  # so we only warn once

    def poll(self, now: float):
        if (now - self.last_poll) < self.poll_interval:
            return
        self.last_poll = now
        if self._api_unavailable:
            return
        try:
            try:
                news = self.api.get_news("BTC/USD,ETH/USD", limit=5)
            except Exception:
                try:
                    news = self.api.get_news("BTCUSD,ETHUSD", limit=5)
                except Exception as e:
                    log.warning(
                        f"News API unavailable for crypto symbols ({e}); "
                        f"sentiment filter disabled for this session."
                    )
                    self._api_unavailable = True
                    return
            for item in news:
                headline = getattr(item, 'headline', '') or ''
                if not headline or headline in self.seen:
                    continue
                self.seen.add(headline)
                self.latest = headline[:80]
                sent = self._score(headline)
                self.sentiment = sent
                # Check impact
                text = headline.lower()
                for kw, (settle, impact) in CRYPTO_NEWS_KEYWORDS.items():
                    if kw in text and impact >= 3:
                        self.halt_until = now + settle
                        self.halt_reason = kw
                        direction = "BULL" if sent > 0 else ("BEAR" if sent < 0 else "???")
                        log.warning(f"CRYPTO NEWS [{kw}]: {headline[:50]} | {direction}")
                        gui_log(f"NEWS: {headline[:40]} [{direction}]")
                        break
        except Exception as e:
            log.debug(f"News poll error: {e}")

    def _score(self, text: str) -> float:
        words = set(re.findall(r'\b\w+\b', text.lower()))
        bull = len(words & BULLISH_WORDS)
        bear = len(words & BEARISH_WORDS)
        total = bull + bear
        return (bull - bear) / total if total > 0 else 0.0

    def is_halted(self, now: float) -> Optional[str]:
        if now < self.halt_until:
            remaining = int(self.halt_until - now)
            return f"{self.halt_reason} ({remaining}s)"
        return None

    def sentiment_vote(self) -> int:
        if self.sentiment > 0.2:
            return 1
        elif self.sentiment < -0.2:
            return -1
        return 0

    def summary_for_gui(self) -> dict:
        return {
            'status': f"Latest: {self.latest[:40]}" if self.latest else "Monitoring...",
            'latest': self.latest[:60] if self.latest else "---",
            'halted': time.time() < self.halt_until,
            'halt_reason': self.halt_reason if time.time() < self.halt_until else "",
        }


# ====================================================================
# MAIN LOOP
# ====================================================================
def crypto_loop(ds: CryptoDataStream, live: bool = False):
    engine = CryptoExecutionEngine(live=live)
    sig_logger = SignalLogger(csv_dir="logs/crypto")
    states = {t: CryptoTickerState(t) for t in WATCHLIST}
    news = CryptoNewsEngine(engine.api)

    # Wall-clock warmup so a slow/empty stream on one ticker can't stretch
    # warmup arbitrarily (or skip it entirely).
    warmup_start = time.time()
    last_flush = time.time()
    # Pre-gating tally: CALL/PUT/NONE = what the composite voted; GATED =
    # would-have-been signals that gating suppressed.
    signal_counts = {'CALL': 0, 'PUT': 0, 'NONE': 0, 'GATED': 0}
    all_outcomes = []

    mode_str = "LIVE" if live else "OBSERVE"
    log.info("=" * 60)
    log.info(f"  NEXUS CRYPTO V2.1 — {mode_str}")
    log.info(f"  Watching: {', '.join(DISPLAY_NAMES.get(t, t) for t in WATCHLIST)}")
    log.info(f"  Stops: trail={TRAILING_STOP_PCT}% hard={HARD_STOP_PCT}%")
    log.info("=" * 60)

    try:
        while True:
            now = time.time()
            engine.sync()
            news.poll(now)
            gui_state['news'] = news.summary_for_gui()

            news_halt = news.is_halted(now)
            if news_halt:
                gui_state['status'] = f"NEWS HALT: {news_halt}"

            warmup_elapsed = now - warmup_start
            warming_up = warmup_elapsed < WARMUP_SECONDS
            if warming_up:
                gui_state['status'] = f"WARMING UP ({int(WARMUP_SECONDS - warmup_elapsed)}s)"
            elif not news_halt:
                gui_state['status'] = f"CRYPTO {'LIVE' if live else 'OBSERVING'}"

            for ticker in WATCHLIST:
                data = ds.market_data[ticker]
                raw_price = data['p']
                trade_size = data.get('s', 1.0) or 1.0
                spread = data['spread']
                obi = data['obi']
                if raw_price == 0:
                    continue

                engine.manage_positions(ticker, raw_price, now)
                st = states[ticker]

                smooth = st.price_filter.filter(raw_price, now)
                dt = max(now - st.last_time, 0.001) if st.last_time > 0 else 1.0
                if st.last_smooth_price > 0:
                    st.velocity = ((smooth - st.last_smooth_price) /
                                   st.last_smooth_price * 100.0) / dt
                    st.acceleration = (st.velocity - st.last_velocity) / dt
                st.last_smooth_price = smooth
                st.last_velocity = st.velocity
                st.last_time = now

                st.vwap.update(raw_price)
                st.trade_flow.classify_tick(raw_price)
                st.rsi.update(raw_price)
                st.bars_5s.tick(raw_price, now, size=trade_size)
                st.bars_30s.tick(raw_price, now, size=trade_size)
                st.bars_5m.tick(raw_price, now, size=trade_size)

                sig_logger.update_outcomes(ticker, raw_price, now)

                if warming_up:
                    continue

                bar5s_dir = st.bars_5s.direction(3)
                bar30s_dir = st.bars_30s.direction(3)
                bar5m_dir = st.bars_5m.direction(3)
                z_score = st.vwap.z_score(smooth)
                current_regime = st.regime.update(raw_price, bar5s_dir, bar30s_dir, z_score)

                votes = {
                    'momentum': vote_momentum(st.velocity, st.acceleration),
                    'trade_flow': vote_trade_flow(st.trade_flow.net_flow(),
                                                  st.trade_flow.flow_acceleration()),
                    'vwap_dev': vote_vwap_deviation(z_score, current_regime),
                    'rsi': vote_rsi(st.rsi.value, current_regime),
                    'obi': vote_obi(obi),
                    'mtf': vote_mtf(bar5s_dir, bar30s_dir, bar5m_dir),
                }

                signal, confidence, reason, raw_score = compute_composite_signal(
                    votes, current_regime, spread, st.regime.vol_ratio
                )

                # --- News sentiment filter ---
                NEWS_DISAGREE_MIN_SCORE = 7.0
                if signal in ("CALL", "PUT"):
                    news_vote = news.sentiment_vote()
                    if news_vote != 0:
                        if (signal == "CALL" and news_vote > 0) or \
                           (signal == "PUT" and news_vote < 0):
                            confidence = min(confidence * 1.1, 1.0)
                            reason += " | news_agrees"
                        elif (signal == "CALL" and news_vote < 0) or \
                             (signal == "PUT" and news_vote > 0):
                            if abs(raw_score) >= NEWS_DISAGREE_MIN_SCORE:
                                confidence *= 0.7
                                reason += " | news_disagrees(strong, allowed)"
                            else:
                                signal = "NONE"
                                reason += f" | GATED:news_disagrees(score={abs(raw_score):.1f}<{NEWS_DISAGREE_MIN_SCORE})"

                # Gating — track pre-gating intent separately so NONE in
                # signal_counts means "composite voted NONE", not
                # "voted CALL but got blocked".
                gated_reason = None
                if signal != "NONE":
                    if news_halt:
                        gated_reason = f"news({news_halt[:30]})"
                    elif spread > MAX_SPREAD_FRACTION:
                        gated_reason = f"spread({spread*100:.3f}%)"
                    else:
                        allowed, block_reason = engine.can_signal(ticker, now)
                        if not allowed:
                            gated_reason = block_reason

                    if gated_reason:
                        reason += f" | GATED:{gated_reason}"
                        signal_counts['GATED'] += 1
                        signal = "NONE"
                    else:
                        signal_counts[signal] += 1
                else:
                    signal_counts['NONE'] += 1

                # Update GUI
                gui_state['signal_counts'] = dict(signal_counts)
                gui_state['equity'] = engine.start_equity + engine.session_pnl
                gui_state['session_pnl'] = engine.session_pnl
                td = gui_state['tickers'][ticker]
                td['price'] = raw_price
                td['smooth'] = round(smooth, 2)
                td['vwap'] = round(st.vwap.vwap, 2)
                td['z_score'] = round(z_score, 2)
                td['velocity'] = round(st.velocity, 6)
                td['accel'] = round(st.acceleration, 6)
                td['rsi'] = round(st.rsi.value, 1)
                td['obi'] = round(obi, 3)
                td['spread'] = round(spread * 100, 4)
                td['regime'] = current_regime
                td['vol_ratio'] = round(st.regime.vol_ratio, 2)
                td['net_flow'] = round(st.trade_flow.net_flow(), 3)
                td['confidence'] = round(confidence, 2)
                td['votes'] = dict(votes)
                td['bar5s'] = round(bar5s_dir, 2)
                td['bar30s'] = round(bar30s_dir, 2)
                td['bar5m'] = round(bar5m_dir, 2)
                td['last_signal'] = signal
                if signal in ("CALL", "PUT"):
                    td['last_active_signal'] = signal
                    name = DISPLAY_NAMES.get(ticker, ticker)
                    gui_log(f"{signal} {name} conf={confidence:.2f} [{current_regime}]")

                # Log signal
                event = SignalEvent(
                    timestamp=now,
                    time_str=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    ticker=ticker, raw_price=raw_price,
                    smooth_price=round(smooth, 4),
                    velocity=round(st.velocity, 8),
                    acceleration=round(st.acceleration, 8),
                    jerk=round(z_score, 4),
                    obi=round(obi, 4),
                    atr_pct=round(st.regime.fast_vol * 100, 4),
                    ema=round(st.vwap.vwap, 4),
                    spread=round(spread, 6),
                    signal=signal, reason=reason,
                )
                sig_logger.log_signal(event)

                if signal in ("CALL", "PUT"):
                    engine.execute(ticker, signal, raw_price, spread, confidence, now)

            # Periodic flush
            if now - last_flush > 30:
                # Prefer a public iterator on SignalLogger; fall back to the
                # private buffer if the module hasn't been updated to expose one.
                if hasattr(sig_logger, 'iter_pending_outcomes'):
                    pending = sig_logger.iter_pending_outcomes()
                else:
                    pending = getattr(sig_logger, '_fired_buffer', [])
                for event in pending:
                    if event.pnl_120s_pct is not None:
                        if not any(abs(o[0] - event.timestamp) < 0.01 for o in all_outcomes):
                            all_outcomes.append((event.timestamp, event.pnl_120s_pct, event.signal))
                            result = "WIN" if event.pnl_120s_pct > 0 else "LOSS"
                            gui_log(f"{result} {event.ticker} {event.pnl_120s_pct:+.4f}% @ 120s")

                sig_logger.flush_completed_outcomes()
                last_flush = now

                all_pnls = [o[1] for o in all_outcomes]
                if all_pnls:
                    wins = sum(1 for v in all_pnls if v > 0)
                    gui_state['outcomes'] = {
                        'count': len(all_pnls),
                        'win_rate': (wins / len(all_pnls)) * 100,
                        'avg_pnl': sum(all_pnls) / len(all_pnls),
                    }

                sc = signal_counts
                # Status line iterates WATCHLIST so adding a symbol doesn't
                # break the f-string with a KeyError.
                px_str = " ".join(
                    f"{DISPLAY_NAMES.get(t, t)}=${ds.market_data[t]['p']:,.2f}"
                    for t in WATCHLIST
                )
                rg_str = " ".join(
                    f"{DISPLAY_NAMES.get(t, t)}={states[t].regime.regime}"
                    for t in WATCHLIST
                )
                log.info(
                    f"[STATUS] {sc['CALL']}C/{sc['PUT']}P/{sc['NONE']}none/"
                    f"{sc['GATED']}gated | PnL=${engine.session_pnl:+.2f} | "
                    f"{px_str} | Regimes: {rg_str}"
                )

            time.sleep(1)

    except KeyboardInterrupt:
        log.info("Shutting down crypto NEXUS...")
        sig_logger.close()
        if live:
            # Close only positions this bot opened, not the entire account.
            engine._close_managed_positions()
        sys.exit(0)


# ====================================================================
# GUI (adapted for crypto)
# ====================================================================
class CryptoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"NEXUS CRYPTO | {gui_state['mode']}")
        self.root.geometry("640x920")
        self.root.configure(bg="#050505")
        self.root.attributes("-topmost", True)

        self.bg = "#050505"
        self.card = "#111111"
        self.fg = "#e0e0e0"
        self.green = "#00ff66"
        self.red = "#ff003c"
        self.gold = "#ffb700"
        self.cyan = "#00f0ff"
        self.dim = "#666666"
        self.orange = "#ff8800"

        self.regime_colors = {
            'TRENDING': self.cyan, 'MEAN_REVERT': self.gold,
            'VOLATILE': self.red, '---': self.dim,
        }

        self.build_ui()
        self.update_gui()

    def build_ui(self):
        hdr = tk.Frame(self.root, bg=self.orange, pady=4)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="N E X U S  C R Y P T O", font=("Consolas", 18, "bold"),
                 bg=self.orange, fg=self.bg).pack()

        eq = tk.Frame(self.root, bg=self.card)
        eq.pack(fill=tk.X, padx=10, pady=(8, 4))
        self.lbl_equity = tk.Label(eq, text="$0", font=("Consolas", 24, "bold"),
                                   bg=self.card, fg=self.fg)
        self.lbl_equity.pack(pady=(6, 0))
        self.lbl_pnl = tk.Label(eq, text="PnL: $0", font=("Consolas", 12),
                                bg=self.card, fg=self.dim)
        self.lbl_pnl.pack()
        self.lbl_status = tk.Label(eq, text="...", font=("Consolas", 10, "bold"),
                                   bg=self.card, fg=self.gold)
        self.lbl_status.pack(pady=(2, 6))

        stats = tk.Frame(self.root, bg=self.bg)
        stats.pack(fill=tk.X, padx=10)
        self.lbl_signals = tk.Label(stats, text="...", font=("Consolas", 9),
                                    bg=self.bg, fg=self.dim)
        self.lbl_signals.pack(anchor="w")
        self.lbl_outcomes = tk.Label(stats, text="", font=("Consolas", 9),
                                     bg=self.bg, fg=self.dim)
        self.lbl_outcomes.pack(anchor="w")
        self.lbl_news = tk.Label(stats, text="News: ---", font=("Consolas", 9),
                                 bg=self.bg, fg=self.dim)
        self.lbl_news.pack(anchor="w")

        self.ticker_panels = {}
        for ticker in WATCHLIST:
            self.ticker_panels[ticker] = self._build_panel(ticker)

        tk.Label(self.root, text="POSITIONS", font=("Consolas", 10, "bold"),
                 bg=self.bg, fg=self.dim).pack(anchor="w", padx=10, pady=(6, 2))
        self.pos_frame = tk.Frame(self.root, bg=self.card)
        self.pos_frame.pack(fill=tk.X, padx=10)

        tk.Label(self.root, text="LOG", font=("Consolas", 10, "bold"),
                 bg=self.bg, fg=self.dim).pack(anchor="w", padx=10, pady=(6, 2))
        self.log_frame = tk.Frame(self.root, bg=self.card)
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _build_panel(self, ticker):
        name = DISPLAY_NAMES.get(ticker, ticker)
        fr = tk.Frame(self.root, bg=self.card)
        fr.pack(fill=tk.X, padx=10, pady=3)
        r1 = tk.Frame(fr, bg=self.card)
        r1.pack(fill=tk.X, padx=8, pady=(5, 0))
        tk.Label(r1, text=name, font=("Consolas", 14, "bold"),
                 bg=self.card, fg=self.fg).pack(side=tk.LEFT)
        lbl_regime = tk.Label(r1, text="---", font=("Consolas", 10, "bold"),
                              bg=self.card, fg=self.dim)
        lbl_regime.pack(side=tk.RIGHT)
        lbl_price = tk.Label(r1, text="$0", font=("Consolas", 14),
                             bg=self.card, fg=self.fg)
        lbl_price.pack(side=tk.RIGHT, padx=8)
        lbl_stats = tk.Label(fr, text="...", font=("Consolas", 9),
                             bg=self.card, fg=self.dim)
        lbl_stats.pack(anchor="w", padx=8, pady=1)
        lbl_deriv = tk.Label(fr, text="...", font=("Consolas", 9),
                             bg=self.card, fg=self.dim)
        lbl_deriv.pack(anchor="w", padx=8, pady=1)
        lbl_votes = tk.Label(fr, text="...", font=("Consolas", 9),
                             bg=self.card, fg=self.dim)
        lbl_votes.pack(anchor="w", padx=8, pady=(1, 5))
        return {'lbl_price': lbl_price, 'lbl_regime': lbl_regime,
                'lbl_stats': lbl_stats, 'lbl_deriv': lbl_deriv, 'lbl_votes': lbl_votes}

    def update_gui(self):
        self.lbl_equity.config(text=f"${gui_state['equity']:,.2f}")
        pnl = gui_state['session_pnl']
        self.lbl_pnl.config(text=f"PnL: ${pnl:+.2f}",
                            fg=self.green if pnl >= 0 else self.red)
        status = gui_state['status']
        scol = self.green if "OBSERVING" in status or "LIVE" in status else self.gold
        self.lbl_status.config(text=status, fg=scol)

        sc = gui_state['signal_counts']
        self.lbl_signals.config(
            text=f"Signals: {sc.get('CALL',0)}C / {sc.get('PUT',0)}P / "
                 f"{sc.get('NONE',0)} none / {sc.get('GATED',0)} gated")

        oc = gui_state.get('outcomes', {})
        if oc.get('count', 0) > 0:
            avg = oc.get('avg_pnl', 0)
            self.lbl_outcomes.config(
                text=f"Outcomes: {oc['count']} | Win: {oc.get('win_rate',0):.0f}% | "
                     f"Avg: {avg:+.4f}%",
                fg=self.green if avg > 0 else self.red)

        news_data = gui_state.get('news', {})
        if news_data.get('halted'):
            self.lbl_news.config(text=f"NEWS HALT: {news_data.get('halt_reason','')}",
                                fg=self.red)
        else:
            self.lbl_news.config(text=f"News: {news_data.get('latest','---')[:50]}",
                                fg=self.dim)

        for ticker in WATCHLIST:
            td = gui_state['tickers'].get(ticker, {})
            p = self.ticker_panels[ticker]
            p['lbl_price'].config(text=f"${td.get('price', 0):,.2f}")
            regime = td.get('regime', '---')
            p['lbl_regime'].config(text=regime, fg=self.regime_colors.get(regime, self.dim))
            z = td.get('z_score', 0)
            p['lbl_stats'].config(
                text=f"VWAP:${td.get('vwap',0):,.2f}  Z:{z:+.2f}  "
                     f"RSI:{td.get('rsi',50):.0f}  OBI:{td.get('obi',0):+.3f}  "
                     f"Spd:{td.get('spread',0):.3f}%",
                fg=self.green if abs(z) < 1.5 else self.gold)
            v = td.get('velocity', 0)
            p['lbl_deriv'].config(
                text=f"Vel:{v:+.5f}  Acc:{td.get('accel',0):+.5f}  "
                     f"Flow:{td.get('net_flow',0):+.3f}  VR:{td.get('vol_ratio',0):.2f}",
                fg=self.green if v > 0 else self.red)
            votes = td.get('votes', {})
            if votes:
                vote_text = "  ".join(f"{n[:4]}:{v:+d}" for n, v in votes.items())
                # Show the most recent non-NONE signal so it doesn't flash
                # off on the next tick.
                sig = td.get('last_active_signal', 'NONE')
                conf = td.get('confidence', 0)
                sig_col = self.green if sig == "CALL" else (self.red if sig == "PUT" else self.dim)
                p['lbl_votes'].config(text=f"[{vote_text}] -> {sig} ({conf:.0%})", fg=sig_col)

        for w in self.pos_frame.winfo_children():
            w.destroy()
        positions = gui_state.get('positions', {})
        if not positions:
            tk.Label(self.pos_frame, text="  No positions", font=("Consolas", 9),
                     bg=self.card, fg=self.dim).pack(anchor="w", padx=6, pady=4)
        else:
            for sym, d in positions.items():
                col = self.green if d['gain'] > 0 else self.red
                tk.Label(self.pos_frame,
                         text=f"  {sym}  {d['gain']:+.2f}%  (${d['pnl']:+.2f})  {d['qty']}",
                         font=("Consolas", 9, "bold"), bg=self.card, fg=col
                         ).pack(anchor="w", padx=4, pady=1)

        for w in self.log_frame.winfo_children():
            w.destroy()
        # Snapshot under the lock so we don't iterate a deque the worker
        # thread is concurrently mutating.
        with _gui_lock:
            log_snapshot = list(gui_state.get('trade_log', []))
        for entry in log_snapshot:
            col = self.cyan
            if "CALL" in entry:
                col = self.green
            elif "PUT" in entry:
                col = self.red
            elif "WIN" in entry:
                col = self.green
            elif "LOSS" in entry:
                col = self.red
            tk.Label(self.log_frame, text=entry, font=("Consolas", 9),
                     bg=self.card, fg=col, anchor="w").pack(fill=tk.X, padx=6, pady=1)

        self.root.after(500, self.update_gui)


# ====================================================================
# ENTRY
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="NEXUS Crypto V2.1")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    gui_state['mode'] = 'LIVE' if args.live else 'OBSERVE'

    ds = CryptoDataStream()
    threading.Thread(target=ds.start_stream, daemon=True).start()
    threading.Thread(target=crypto_loop, args=(ds, args.live), daemon=True).start()

    if HAS_TK and not args.headless:
        root = tk.Tk()
        CryptoGUI(root)
        root.mainloop()
    else:
        log.info("Running headless.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()