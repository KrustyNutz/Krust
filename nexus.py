from __future__ import annotations

# ====================================================================
# NEXUS V1.0 | REGIME-AWARE MULTI-SIGNAL FUSION ENGINE
#
# Architecture based on quantitative microstructure research:
#   - Cont et al. (2014): Order flow imbalance predicts short-horizon
#     price changes, impact scales inversely with liquidity depth
#   - Intraday SPY mean-reverts at <5min, trends at >15min
#   - Regime detection determines which sub-strategy to deploy
#   - Multi-timeframe confluence filters noise from signal
#   - Tick-rule trade classification estimates aggressor flow
#
# Design principle: Don't predict direction from one signal.
# Detect WHEN conditions are favorable, then use the strongest
# available confluence. Gate → Signal → Size → Manage.
#
# Usage:
#   python nexus.py                # Observe mode (default, no trades)
#   python nexus.py --live         # Paper trading with live orders
#   python nexus.py --headless     # No GUI output
# ====================================================================

import threading
import time
import asyncio
import sys
import os
import math
import logging
import argparse
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Optional, Dict, List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from alpaca_trade_api.stream import Stream
from alpaca_trade_api.rest import REST

try:
    import tkinter as tk
    HAS_TK = True
except ImportError:
    HAS_TK = False

# ====================================================================
# TIMEZONE: Always use Eastern Time for market logic.
# This auto-detects your local timezone and converts, so NEXUS works
# correctly whether you're in LA, NY, London, or Tokyo.
# ====================================================================
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    ET = ZoneInfo("America/New_York")
except ImportError:
    # Python 3.8 fallback chain
    try:
        from backports.zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
    except ImportError:
        try:
            import pytz
            ET = pytz.timezone("America/New_York")
        except ImportError:
            # Last resort: detect DST manually
            import calendar
            _now = datetime.utcnow()
            # US DST: 2nd Sunday March - 1st Sunday November
            _mar = datetime(_now.year, 3, 1)
            _nov = datetime(_now.year, 11, 1)
            _dst_start = _mar + timedelta(days=(6 - _mar.weekday()) % 7 + 7)
            _dst_end = _nov + timedelta(days=(6 - _nov.weekday()) % 7)
            _is_dst = _dst_start <= _now.replace(tzinfo=None) < _dst_end
            _offset = -4 if _is_dst else -5
            ET = timezone(timedelta(hours=_offset))
            logging.getLogger("nexus").info(
                f"Using manual UTC{_offset} ({'EDT' if _is_dst else 'EST'}). "
                f"For auto DST: pip install pytz"
            )

def now_et() -> datetime:
    """Current time in US Eastern, regardless of local timezone."""
    return datetime.now(tz=ET)

from filters import OneEuroFilter
from signal_logger import SignalEvent, SignalLogger
from expiry import build_occ_symbol
from levels import ChartContext, fetch_prior_day
from news import NewsEngine

# ====================================================================
# LOGGING
# ====================================================================
os.makedirs("logs/nexus", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/nexus/nexus.log", mode='a'),
    ]
)
log = logging.getLogger("nexus")

# ====================================================================
# CONFIG
# ====================================================================
API_KEY = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not SECRET_KEY:
    log.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
    sys.exit(1)

WATCHLIST = ["SPY", "QQQ"]
DATA_FEED = "iex"

# --- Session Timing (Eastern) ---
SESSION_START_BUFFER_MIN = 5
SESSION_END_BUFFER_MIN = 5
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 16
TRADING_CUTOFF_HOUR = 12       # Stop opening new positions at noon
TRADING_CUTOFF_MIN = 0
FLATTEN_HOUR = 12              # Close ALL positions by this time
FLATTEN_MIN = 15

# --- Risk Parameters ---
MAX_SESSION_LOSS = 2000.00
TRADE_MODE = "shares"          # "shares" or "options"
SHARE_QTY = 100                # Base shares per trade (confidence-scaled)
MAX_SHARES = 300               # Max shares per position
POSITION_SIZE = 20             # Options qty (if TRADE_MODE="options")
SIGNAL_COOLDOWN_SEC = 120
MAX_SPREAD_PCT = 0.0012
DTE = 0
MAX_OPEN_POSITIONS = 2         # Max simultaneous positions
TRAILING_STOP_PCT = 0.10       # 0.10% trailing stop (was 0.05% — too tight, noise triggers)
HARD_STOP_PCT = 0.15           # 0.15% max loss per trade
MAX_HOLD_SECONDS = 300         # 5 min max hold per trade
WASH_TRADE_COOLDOWN = 30       # Seconds to wait after closing before re-entering same ticker

# --- Signal Weights by Regime ---
# Each component votes [-1, 0, +1]. These weights determine influence.
# Tuned so the right signals matter in the right regime.
REGIME_WEIGHTS = {
    'TRENDING': {
        'momentum': 3.0,      # Primary driver in trends
        'trade_flow': 2.0,    # Confirms institutional participation
        'vwap_dev': 0.5,      # Slight penalty for being far from VWAP (overextended)
        'rsi': 0.0,           # Disabled: -47pp lift on May 7; 10% WR when agreeing
        'obi': 1.5,           # Order book confirmation
        'mtf': 2.5,           # Multi-timeframe agreement is critical
    },
    'MEAN_REVERT': {
        'momentum': 0.5,      # Fade momentum in range
        'trade_flow': 1.0,    # Still useful
        'vwap_dev': 3.0,      # Primary driver — fade deviations
        'rsi': 0.0,           # Disabled: same inversion holds in MEAN_REVERT data
        'obi': 1.0,           # Confirmation
        'mtf': 1.0,           # Less important in range
    },
    'VOLATILE': {
        'momentum': 0.0,      # Don't trust momentum in chaos
        'trade_flow': 0.5,    # Weak signal
        'vwap_dev': 0.0,      # VWAP bands are too wide to be useful
        'rsi': 0.5,           # Extremes happen and persist
        'obi': 0.0,           # Order book is unreliable
        'mtf': 0.0,           # Timeframes disagree = noise
    },
}

# Composite score thresholds to fire a signal
# Higher = more selective = fewer but better signals
SIGNAL_THRESHOLD = {
    'TRENDING': 5.0,          # Raised from 4.0 — fewer but better signals
    'MEAN_REVERT': 99.0,      # Disabled: 15% WR / -0.023% avg across 13 May 7 signals
    'VOLATILE': 99.0,         # Effectively disabled — don't trade chaos
}

# ====================================================================
# SHARED GUI STATE
# ====================================================================
gui_state = {
    'equity': 0.0,
    'session_pnl': 0.0,
    'mode': 'OBSERVE',
    'status': 'INITIALIZING...',
    'signal_counts': {'CALL': 0, 'PUT': 0, 'NONE': 0},
    'tickers': {
        'SPY': {
            'price': 0.0, 'smooth': 0.0, 'vwap': 0.0, 'z_score': 0.0,
            'velocity': 0.0, 'accel': 0.0, 'rsi': 50.0, 'obi': 0.0,
            'spread': 0.0, 'regime': '---', 'vol_ratio': 0.0,
            'net_flow': 0.0, 'confidence': 0.0,
            'votes': {},
            'bar5s': 0.0, 'bar30s': 0.0, 'bar5m': 0.0,
            'last_signal': 'NONE',
        },
        'QQQ': {
            'price': 0.0, 'smooth': 0.0, 'vwap': 0.0, 'z_score': 0.0,
            'velocity': 0.0, 'accel': 0.0, 'rsi': 50.0, 'obi': 0.0,
            'spread': 0.0, 'regime': '---', 'vol_ratio': 0.0,
            'net_flow': 0.0, 'confidence': 0.0,
            'votes': {},
            'bar5s': 0.0, 'bar30s': 0.0, 'bar5m': 0.0,
            'last_signal': 'NONE',
        },
    },
    'trade_log': [],
    'positions': {},
    'outcomes': {},
}


def gui_log(msg: str):
    ts = now_et().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    gui_state['trade_log'].insert(0, entry)
    if len(gui_state['trade_log']) > 15:
        gui_state['trade_log'].pop()


# ====================================================================
# MULTI-TIMEFRAME BAR AGGREGATOR
# ====================================================================
class BarAggregator:
    """
    Aggregates 1-second ticks into bars of different durations.
    Tracks open, high, low, close, volume (tick count), and VWAP.
    """

    def __init__(self, period_seconds: int):
        self.period = period_seconds
        self.bars: deque = deque(maxlen=200)  # rolling history
        self._current_open = 0.0
        self._current_high = -math.inf
        self._current_low = math.inf
        self._current_close = 0.0
        self._current_ticks = 0
        self._current_cum_pv = 0.0
        self._bar_start = 0.0

    def tick(self, price: float, now: float):
        """Feed a new tick. Returns a completed bar dict if period elapsed, else None."""
        if self._bar_start == 0:
            self._bar_start = now

        # Always set open on the first tick of a new bar
        if self._current_open == 0:
            self._current_open = price

        self._current_high = max(self._current_high, price)
        self._current_low = min(self._current_low, price)
        self._current_close = price
        self._current_ticks += 1
        self._current_cum_pv += price

        if (now - self._bar_start) >= self.period and self._current_ticks > 0:
            bar = {
                'open': self._current_open,
                'high': self._current_high,
                'low': self._current_low,
                'close': self._current_close,
                'ticks': self._current_ticks,
                'vwap': self._current_cum_pv / self._current_ticks,
                'range_pct': ((self._current_high - self._current_low) /
                              self._current_low * 100) if self._current_low > 0 else 0,
                'time': now,
            }
            self.bars.append(bar)
            # Reset for next bar
            self._current_open = 0.0  # Will be set by next tick
            self._current_high = -math.inf
            self._current_low = math.inf
            self._current_ticks = 0
            self._current_cum_pv = 0.0
            self._bar_start = now
            return bar
        return None

    def direction(self, lookback: int = 3) -> float:
        """
        Returns average bar direction over last N bars.
        +1 = all up bars, -1 = all down bars, 0 = mixed.
        """
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
        """Average bar range in % over last N bars."""
        if len(self.bars) < 2:
            return 0.0
        recent = list(self.bars)[-lookback:]
        ranges = [b['range_pct'] for b in recent]
        return sum(ranges) / len(ranges) if ranges else 0.0


# ====================================================================
# VWAP ENGINE (Session-Anchored, Institutional Grade)
# ====================================================================
class VWAPEngine:
    """
    Running session VWAP with standard deviation bands.
    Resets at market open each day.
    """

    def __init__(self):
        self.cum_volume = 0.0
        self.cum_pv = 0.0
        self.cum_pv2 = 0.0
        self.vwap = 0.0
        self.std = 0.0
        self.tick_count = 0

    def update(self, price: float, volume: float = 1.0):
        self.cum_volume += volume
        self.cum_pv += price * volume
        self.cum_pv2 += (price ** 2) * volume
        self.tick_count += 1

        if self.cum_volume > 0:
            self.vwap = self.cum_pv / self.cum_volume
            variance = max((self.cum_pv2 / self.cum_volume) - self.vwap ** 2, 0.0)
            self.std = math.sqrt(variance)

    def z_score(self, price: float) -> float:
        """How many standard deviations from VWAP."""
        if self.std > 0:
            return (price - self.vwap) / self.std
        return 0.0

    def reset(self):
        self.__init__()


# ====================================================================
# TRADE FLOW ANALYZER (Tick Rule Classification)
# ====================================================================
class TradeFlowAnalyzer:
    """
    Classifies trades as buyer- or seller-initiated using the tick rule.
    Computes cumulative delta (net buying pressure) and flow rate.
    """

    def __init__(self, window: int = 60):
        self.last_price = 0.0
        self.last_direction = 0  # +1 buy, -1 sell
        self.deltas: deque = deque(maxlen=window)
        self.cum_delta = 0.0
        self.buy_count = 0
        self.sell_count = 0

    def classify_tick(self, price: float) -> int:
        """
        Tick rule: uptick = buy, downtick = sell, zero-tick = carry forward.
        Returns +1 (buy), -1 (sell), or 0 (neutral).
        """
        if self.last_price == 0:
            self.last_price = price
            return 0

        if price > self.last_price:
            direction = 1
        elif price < self.last_price:
            direction = -1
        else:
            direction = self.last_direction  # carry forward

        self.last_price = price
        self.last_direction = direction
        self.deltas.append(direction)

        if direction > 0:
            self.buy_count += 1
        elif direction < 0:
            self.sell_count += 1

        return direction

    def net_flow(self) -> float:
        """
        Net flow over recent window: +1.0 = all buys, -1.0 = all sells.
        """
        if not self.deltas:
            return 0.0
        return sum(self.deltas) / len(self.deltas)

    def flow_acceleration(self) -> float:
        """
        Is buying/selling accelerating or decelerating?
        Compare first half vs second half of window.
        """
        if len(self.deltas) < 10:
            return 0.0
        d = list(self.deltas)
        mid = len(d) // 2
        first_half = sum(d[:mid]) / mid
        second_half = sum(d[mid:]) / (len(d) - mid)
        return second_half - first_half


# ====================================================================
# RSI CALCULATOR
# ====================================================================
class RSICalculator:
    """Wilder's RSI with exponential smoothing."""

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
                # Wilder smoothing
                self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
                self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

            if self.avg_loss == 0:
                self.value = 100.0
            else:
                rs = self.avg_gain / self.avg_loss
                self.value = 100.0 - (100.0 / (1.0 + rs))

        self.last_price = price
        return self.value


# ====================================================================
# REGIME DETECTOR
# ====================================================================
class RegimeDetector:
    """
    Classifies current market into one of three regimes:
      TRENDING:     directional, moderate vol, persistent momentum
      MEAN_REVERT:  range-bound, low vol, oscillating around VWAP
      VOLATILE:     high vol, erratic, unpredictable

    Uses:
      - Ratio of fast vol to slow vol (vol regime)
      - Directional consistency across timeframes
      - VWAP z-score magnitude and persistence
    """

    def __init__(self, fast_window: int = 30, slow_window: int = 300):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.returns: deque = deque(maxlen=slow_window)
        self.last_price = 0.0
        self.regime = "MEAN_REVERT"  # default assumption
        self.vol_ratio = 1.0
        self.fast_vol = 0.0
        self.slow_vol = 0.0
        self.regime_confidence = 0.0

    def update(self, price: float, bar5s_dir: float, bar30s_dir: float,
               z_score: float) -> str:
        if self.last_price > 0 and self.last_price != 0:
            ret = ((price - self.last_price) / self.last_price) * 100.0
            self.returns.append(ret)
        self.last_price = price

        returns_list = list(self.returns)

        # Compute volatilities
        self.fast_vol = self._stdev(returns_list, self.fast_window)
        self.slow_vol = self._stdev(returns_list, self.slow_window)
        self.vol_ratio = (self.fast_vol / self.slow_vol) if self.slow_vol > 0 else 1.0

        # Direction consistency: do 5s and 30s bars agree?
        dir_agreement = bar5s_dir * bar30s_dir  # positive = agree

        # Regime classification
        if self.vol_ratio > 2.0 or self.fast_vol > 0.02:
            self.regime = "VOLATILE"
            self.regime_confidence = min(self.vol_ratio / 3.0, 1.0)

        elif abs(dir_agreement) > 0.3 and abs(bar30s_dir) > 0.5:
            # Persistent direction across timeframes
            self.regime = "TRENDING"
            self.regime_confidence = abs(dir_agreement)

        else:
            self.regime = "MEAN_REVERT"
            self.regime_confidence = 1.0 - abs(bar30s_dir)

        return self.regime

    @staticmethod
    def _stdev(data: list, window: int) -> float:
        if len(data) < max(window, 5):
            return 0.0
        subset = data[-window:]
        if len(subset) < 2:
            return 0.0
        mean = sum(subset) / len(subset)
        variance = sum((x - mean) ** 2 for x in subset) / (len(subset) - 1)
        return math.sqrt(variance)


# ====================================================================
# SIGNAL COMPONENTS — Each votes [-1, 0, +1]
# ====================================================================

def vote_momentum(velocity: float, accel: float, threshold: float = 0.0002) -> int:
    """Smoothed velocity + acceleration agreement."""
    if velocity > threshold and accel > 0:
        return 1
    elif velocity < -threshold and accel < 0:
        return -1
    return 0


def vote_trade_flow(net_flow: float, flow_accel: float,
                    threshold: float = 0.15) -> int:
    """Net buying/selling pressure from tick-rule classification."""
    if net_flow > threshold and flow_accel > 0:
        return 1
    elif net_flow < -threshold and flow_accel < 0:
        return -1
    return 0


def vote_vwap_deviation(z_score: float, regime: str) -> int:
    """
    In MEAN_REVERT: fade extremes (z < -2 = buy, z > 2 = sell)
    In TRENDING: go with the deviation (above VWAP = bullish)
    """
    if regime == "MEAN_REVERT":
        if z_score < -2.0:
            return 1   # Buy the dip
        elif z_score > 2.0:
            return -1  # Sell the rip
    elif regime == "TRENDING":
        if z_score > 0.5:
            return 1   # Above VWAP in trend = bullish
        elif z_score < -0.5:
            return -1  # Below VWAP in trend = bearish
    return 0


def vote_rsi(rsi_value: float, regime: str) -> int:
    """
    In MEAN_REVERT: standard oversold/overbought.
    In TRENDING: looser thresholds (trends can stay overbought).
    """
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


def vote_obi(obi: float, threshold: float = 0.08) -> int:
    """Order book imbalance from L1 quotes."""
    if obi > threshold:
        return 1
    elif obi < -threshold:
        return -1
    return 0


def vote_mtf(bar5s_dir: float, bar30s_dir: float, bar5m_dir: float) -> int:
    """
    Multi-timeframe agreement. All three must point the same way.
    This is the most powerful noise filter.
    """
    if bar5s_dir > 0.3 and bar30s_dir > 0.3 and bar5m_dir > 0:
        return 1
    elif bar5s_dir < -0.3 and bar30s_dir < -0.3 and bar5m_dir < 0:
        return -1
    return 0


# ====================================================================
# SIGNAL FUSION ENGINE
# ====================================================================

def compute_composite_signal(
    votes: Dict[str, int],
    regime: str,
    spread: float,
    vol_ratio: float,
) -> Tuple[str, float, str]:
    """
    Fuse individual component votes into a composite signal.

    Returns: (signal, confidence, reason)
      signal: "CALL", "PUT", or "NONE"
      confidence: 0.0 to 1.0 (used for position sizing)
      reason: human-readable breakdown
    """
    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS['MEAN_REVERT'])
    threshold = SIGNAL_THRESHOLD.get(regime, 99.0)

    # Weighted sum
    raw_score = 0.0
    max_possible = 0.0
    components = []

    for name, vote in votes.items():
        w = weights.get(name, 0.0)
        raw_score += vote * w
        max_possible += abs(w)
        if vote != 0:
            components.append(f"{name}={vote:+d}")

    # Spread penalty: wider spread = need stronger signal
    spread_penalty = (spread / MAX_SPREAD_PCT) if spread > 0 else 0
    adjusted_score = raw_score * (1.0 - spread_penalty * 0.3)

    # Vol penalty: higher vol ratio = less trust in signals
    if vol_ratio > 1.5:
        adjusted_score *= (1.0 / vol_ratio)

    # Normalize confidence
    confidence = abs(adjusted_score) / max_possible if max_possible > 0 else 0

    reason_parts = [f"regime={regime}", f"score={adjusted_score:.2f}/{threshold:.1f}"]
    reason_parts.extend(components)
    reason = " | ".join(reason_parts)

    if adjusted_score > threshold:
        return "CALL", confidence, reason, adjusted_score
    elif adjusted_score < -threshold:
        return "PUT", confidence, reason, adjusted_score
    else:
        return "NONE", confidence, reason, adjusted_score


# ====================================================================
# TIME-OF-DAY GATE
# ====================================================================

def is_tradeable_time() -> bool:
    """
    Only trade during the safe window: after first 5 min, before last 5 min.
    The open and close are dominated by noise, trapped traders, and
    institutional rebalancing — not a retail edge.
    """
    now = now_et()
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=0, second=0)
    safe_start = market_open + timedelta(minutes=SESSION_START_BUFFER_MIN)
    safe_end = market_close - timedelta(minutes=SESSION_END_BUFFER_MIN)
    return safe_start <= now <= safe_end


# ====================================================================
# DATA STREAM
# ====================================================================
class DataStream:
    def __init__(self):
        self.market_data = {t: {'p': 0.0, 'spread': 0.0, 'obi': 0.0} for t in WATCHLIST}
        self.stream = Stream(API_KEY, SECRET_KEY, base_url=BASE_URL, data_feed=DATA_FEED)

    async def on_trade(self, t):
        self.market_data[t.symbol]['p'] = float(t.price)

    async def on_quote(self, q):
        if q.bid_price > 0 and q.ask_price > 0:
            self.market_data[q.symbol]['spread'] = (q.ask_price - q.bid_price) / q.bid_price
        total = q.bid_size + q.ask_size
        if total > 0:
            self.market_data[q.symbol]['obi'] = (q.bid_size - q.ask_size) / total

    def start_stream(self):
        log.info("Starting market data stream...")
        for t in WATCHLIST:
            self.stream.subscribe_trades(self.on_trade, t)
            self.stream.subscribe_quotes(self.on_quote, t)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            self.stream.run()
        except Exception as e:
            log.error(f"Stream died: {e}")


# ====================================================================
# EXECUTION ENGINE (shares + options, with position management)
# ====================================================================
class ExecutionEngine:
    def __init__(self, live: bool = False):
        self.live = live
        self.api = REST(API_KEY, SECRET_KEY, BASE_URL)
        try:
            self.start_equity = float(self.api.get_account().equity)
        except Exception:
            self.start_equity = 50000.0
        self.active_symbols: List[str] = []
        self.last_signal_time: Dict[str, float] = {}
        self.session_pnl = 0.0
        self.session_ended = False

        # Position tracking for share trades
        self.open_trades: Dict[str, dict] = {}  # ticker -> {side, entry, qty, time, peak}
        self.last_close_time: Dict[str, float] = {}  # ticker -> timestamp of last close

    def sync(self):
        try:
            acc = self.api.get_account()
            equity = float(acc.equity)
            self.session_pnl = equity - self.start_equity
            if self.session_pnl < -MAX_SESSION_LOSS:
                log.critical(f"MAX SESSION LOSS: ${self.session_pnl:.2f}. Killing.")
                if self.live:
                    self.api.close_all_positions()
                self.session_ended = True
                sys.exit(1)
            positions = self.api.list_positions()
            self.active_symbols = [p.symbol for p in positions]

            # Build GUI position dict while we have the data
            pos_dict = {}
            for p in positions:
                try:
                    pos_dict[p.symbol] = {
                        'gain': float(p.unrealized_plpc) * 100,
                        'pnl': float(p.unrealized_pl),
                        'qty': abs(int(p.qty)),
                    }
                except Exception:
                    pass
            gui_state['positions'] = pos_dict

        except Exception as e:
            log.error(f"Sync error: {e}")

    def check_session_cutoff(self, now_dt: datetime):
        """Stop new trades and flatten positions at cutoff time."""
        cutoff = now_dt.replace(hour=TRADING_CUTOFF_HOUR, minute=TRADING_CUTOFF_MIN, second=0)
        flatten = now_dt.replace(hour=FLATTEN_HOUR, minute=FLATTEN_MIN, second=0)

        if now_dt >= flatten and not self.session_ended:
            if self.live and self.active_symbols:
                log.warning("SESSION CUTOFF: Flattening all positions.")
                try:
                    self.api.close_all_positions()
                except Exception as e:
                    log.error(f"Flatten error: {e}")
                gui_log("SESSION END: All positions closed.")
            self.open_trades.clear()
            self.session_ended = True

        return now_dt >= cutoff  # True = no new trades

    def can_signal(self, ticker: str, now: float) -> bool:
        """Cooldown, capacity, and wash trade check."""
        if self.session_ended:
            return False
        last = self.last_signal_time.get(ticker, 0)
        if (now - last) < SIGNAL_COOLDOWN_SEC:
            return False
        if len(self.open_trades) >= MAX_OPEN_POSITIONS:
            return False
        if ticker in self.open_trades:
            return False
        # Wash trade prevention: don't re-enter within 30s of closing
        last_close = self.last_close_time.get(ticker, 0)
        if (now - last_close) < WASH_TRADE_COOLDOWN:
            return False
        return True

    def manage_positions(self, ticker: str, current_price: float, now: float):
        """Trailing stop + hard stop + time stop. Stops scale with confidence."""
        if ticker not in self.open_trades:
            return

        trade = self.open_trades[ticker]
        entry = trade['entry']
        side = trade['side']
        elapsed = now - trade['time']
        conf = trade.get('confidence', 0.5)

        # Confidence-scaled stops: high conf = more room to breathe
        # conf 0.7+ -> 1.5x stops, conf 0.4-0.7 -> 1.0x, conf <0.4 -> 0.75x
        if conf >= 0.7:
            stop_scale = 1.5
        elif conf >= 0.4:
            stop_scale = 1.0
        else:
            stop_scale = 0.75

        trail = TRAILING_STOP_PCT * stop_scale
        hard = HARD_STOP_PCT * stop_scale

        # Compute current PnL %
        if side == "CALL":
            pnl_pct = ((current_price - entry) / entry) * 100
            trade['peak'] = max(trade.get('peak', pnl_pct), pnl_pct)
        else:  # PUT (short)
            pnl_pct = ((entry - current_price) / entry) * 100
            trade['peak'] = max(trade.get('peak', pnl_pct), pnl_pct)

        peak = trade['peak']
        exit_reason = None

        # Hard stop loss
        if pnl_pct < -hard:
            exit_reason = f"HARD_STOP pnl={pnl_pct:+.3f}%"

        # Trailing stop (only after going positive)
        elif peak > 0.02 and pnl_pct < (peak - trail):
            exit_reason = f"TRAIL_STOP pnl={pnl_pct:+.3f}% peak={peak:.3f}%"

        # Time stop
        elif elapsed > MAX_HOLD_SECONDS:
            exit_reason = f"TIME_STOP {elapsed:.0f}s"

        if exit_reason:
            self._close_position(ticker, exit_reason, pnl_pct)

    def execute(self, ticker: str, side: str, price: float,
                spread: float, confidence: float, now: float):
        """Place order — shares or options based on TRADE_MODE."""
        self.last_signal_time[ticker] = now

        if not self.live:
            return

        if spread > MAX_SPREAD_PCT:
            return

        if TRADE_MODE == "shares":
            self._execute_shares(ticker, side, price, confidence, now)
        else:
            self._execute_options(ticker, side, price, confidence, now)

    def _execute_shares(self, ticker: str, side: str, price: float,
                        confidence: float, now: float):
        """Buy/sell shares directly."""
        if ticker in self.open_trades:
            return

        qty = min(max(int(SHARE_QTY * confidence), 50), MAX_SHARES)
        order_side = 'buy' if side == "CALL" else 'sell'

        try:
            self.api.submit_order(
                symbol=ticker, qty=qty, side=order_side,
                type='market', time_in_force='day'
            )
            self.open_trades[ticker] = {
                'side': side, 'entry': price, 'qty': qty,
                'time': now, 'peak': 0.0, 'order_side': order_side,
                'confidence': confidence,
            }
            log.info(f"SHARES: {order_side.upper()} {qty}x {ticker} @ ${price:.2f} (conf={confidence:.2f})")
            gui_log(f"ENTRY: {qty}sh {side} {ticker} @ ${price:.2f}")
        except Exception as e:
            log.error(f"Share order failed: {e}")

    def _execute_options(self, ticker: str, side: str, price: float,
                         confidence: float, now: float):
        """Original options execution."""
        contract = build_occ_symbol(ticker, price, side, dte=DTE)
        if not contract or contract in self.active_symbols:
            return
        qty = max(10, int(POSITION_SIZE * confidence))
        try:
            self.api.submit_order(
                symbol=contract, qty=qty, side='buy',
                type='market', time_in_force='day'
            )
            log.info(f"OPTIONS: {qty}x {side} {ticker} @ ${price:.2f}")
        except Exception as e:
            log.error(f"Options order failed: {e}")

    def _close_position(self, ticker: str, reason: str, pnl_pct: float):
        """Close a share position."""
        if ticker not in self.open_trades:
            return

        trade = self.open_trades[ticker]
        self.last_close_time[ticker] = time.time()  # Wash trade prevention

        if not self.live:
            del self.open_trades[ticker]
            return

        try:
            # Close by submitting opposite order
            close_side = 'sell' if trade['order_side'] == 'buy' else 'buy'
            self.api.submit_order(
                symbol=ticker, qty=trade['qty'], side=close_side,
                type='market', time_in_force='day'
            )
            result = "WIN" if pnl_pct > 0 else "LOSS"
            log.info(f"CLOSED: {ticker} {reason} | {result} {pnl_pct:+.3f}%")
            gui_log(f"EXIT: {ticker} {result} {pnl_pct:+.3f}% ({reason})")
            del self.open_trades[ticker]
        except Exception as e:
            log.error(f"Close position failed for {ticker}: {e}")


# ====================================================================
# TICKER STATE — all analysis components for one symbol
# ====================================================================
class TickerState:
    def __init__(self, symbol: str):
        self.symbol = symbol

        # Price smoothing
        self.price_filter = OneEuroFilter(min_cutoff=0.01, beta=0.5)

        # Multi-timeframe bars
        self.bars_5s = BarAggregator(5)
        self.bars_30s = BarAggregator(30)
        self.bars_5m = BarAggregator(300)

        # Analysis components
        self.vwap = VWAPEngine()
        self.trade_flow = TradeFlowAnalyzer(window=60)
        self.rsi = RSICalculator(period=14)
        self.regime = RegimeDetector(fast_window=30, slow_window=300)

        # Derivative tracking
        self.last_smooth_price = 0.0
        self.last_velocity = 0.0
        self.last_time = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0

        # Chart context (levels + patterns)
        self.chart = ChartContext(symbol, round_increment=5.0)


# ====================================================================
# MAIN LOOP
# ====================================================================
def nexus_loop(ds: DataStream, live: bool = False):
    engine = ExecutionEngine(live=live)
    sig_logger = SignalLogger(csv_dir="logs/nexus")
    states = {t: TickerState(t) for t in WATCHLIST}

    # Fetch prior day levels for chart context
    for ticker in WATCHLIST:
        try:
            pd_data = fetch_prior_day(engine.api, ticker)
            states[ticker].chart.levels.set_prior_day(
                high=pd_data['high'], low=pd_data['low'],
                close=pd_data['close'], vwap=pd_data['vwap']
            )
            gui_log(f"Levels loaded: {ticker} PDH={pd_data['high']:.2f} PDL={pd_data['low']:.2f}")
        except Exception as e:
            log.warning(f"Could not fetch prior day for {ticker}: {e}")

    warmup_ticks = 0
    WARMUP_REQUIRED = 120
    last_flush = time.time()

    # News engine
    news = NewsEngine(engine.api, WATCHLIST, poll_interval=180)

    mode_str = "LIVE" if live else "OBSERVE"
    trade_str = f"SHARES ({SHARE_QTY})" if TRADE_MODE == "shares" else f"OPTIONS ({POSITION_SIZE}x)"
    et_now = now_et()
    log.info("=" * 60)
    log.info(f"  NEXUS V2.1 — {mode_str} MODE — {trade_str}")
    log.info(f"  Regime-Aware Multi-Signal Fusion Engine")
    log.info(f"  Watching: {', '.join(WATCHLIST)}")
    log.info(f"  Eastern Time: {et_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    log.info(f"  Trading: {MARKET_OPEN_HOUR}:{MARKET_OPEN_MIN:02d}+{SESSION_START_BUFFER_MIN}min "
             f"to {TRADING_CUTOFF_HOUR}:{TRADING_CUTOFF_MIN:02d} ET")
    log.info(f"  Flatten: {FLATTEN_HOUR}:{FLATTEN_MIN:02d} ET | "
             f"Stops: trail={TRAILING_STOP_PCT}% hard={HARD_STOP_PCT}%")
    log.info("=" * 60)

    signal_counts = {'CALL': 0, 'PUT': 0, 'NONE': 0}
    all_outcomes = []  # list of (timestamp, pnl_pct, signal) tuples

    try:
        while True:
            now = time.time()
            engine.sync()

            # --- News polling ---
            news.poll(now)
            gui_state['news'] = news.summary_for_gui()

            # --- Session cutoff ---
            now_dt = now_et()
            past_cutoff = engine.check_session_cutoff(now_dt)

            # --- After-hours auto-shutdown (ET-aware) ---
            if now_dt.hour >= 17:  # 5 PM ET — market closed an hour ago
                log.info("After-hours detected. Shutting down.")
                sig_logger.close()
                sys.exit(0)

            # --- News halt check ---
            news_halt = news.is_halted(now)
            if news_halt:
                gui_state['status'] = f"NEWS HALT: {news_halt}"

            for ticker in WATCHLIST:
                data = ds.market_data[ticker]
                raw_price = data['p']
                spread = data['spread']
                obi = data['obi']
                if raw_price == 0:
                    continue

                # --- Position management (every tick) ---
                engine.manage_positions(ticker, raw_price, now)

                st = states[ticker]

                # --- Smooth price ---
                smooth = st.price_filter.filter(raw_price, now)

                # --- Derivatives ---
                dt = now - st.last_time if st.last_time > 0 else 1.0
                dt = max(dt, 0.001)
                if st.last_smooth_price > 0:
                    st.velocity = ((smooth - st.last_smooth_price) /
                                   st.last_smooth_price * 100.0) / dt
                    st.acceleration = (st.velocity - st.last_velocity) / dt
                st.last_smooth_price = smooth
                st.last_velocity = st.velocity
                st.last_time = now

                # --- Update all components ---
                st.vwap.update(raw_price)
                st.trade_flow.classify_tick(raw_price)
                st.rsi.update(raw_price)
                st.bars_5s.tick(raw_price, now)
                bar30 = st.bars_30s.tick(raw_price, now)
                st.bars_5m.tick(raw_price, now)

                # --- Chart context: tick + completed bars ---
                st.chart.update_tick(raw_price)
                if bar30:
                    st.chart.update_bar(bar30['high'], bar30['low'], bar30['close'])

                # --- Update outcome tracker ---
                sig_logger.update_outcomes(ticker, raw_price, now)

                # --- Warmup gate ---
                warmup_ticks += 1
                if warmup_ticks < WARMUP_REQUIRED * len(WATCHLIST):
                    remaining = WARMUP_REQUIRED - (warmup_ticks // len(WATCHLIST))
                    gui_state['status'] = f"WARMING UP ({remaining}s)"
                    continue

                gui_state['status'] = f"NEXUS {'LIVE' if live else 'OBSERVING'}"

                # --- Time-of-day gate ---
                tradeable = is_tradeable_time()

                # --- Regime detection ---
                bar5s_dir = st.bars_5s.direction(lookback=3)
                bar30s_dir = st.bars_30s.direction(lookback=3)
                bar5m_dir = st.bars_5m.direction(lookback=3)
                z_score = st.vwap.z_score(smooth)

                current_regime = st.regime.update(
                    raw_price, bar5s_dir, bar30s_dir, z_score
                )

                # --- Collect votes ---
                votes = {
                    'momentum': vote_momentum(st.velocity, st.acceleration),
                    'trade_flow': vote_trade_flow(
                        st.trade_flow.net_flow(),
                        st.trade_flow.flow_acceleration()
                    ),
                    'vwap_dev': vote_vwap_deviation(z_score, current_regime),
                    'rsi': vote_rsi(st.rsi.value, current_regime),
                    'obi': vote_obi(obi),
                    'mtf': vote_mtf(bar5s_dir, bar30s_dir, bar5m_dir),
                }

                # --- Fuse signals ---
                signal, confidence, reason, raw_score = compute_composite_signal(
                    votes, current_regime, spread, st.regime.vol_ratio
                )

                # --- Gating ---
                if signal != "NONE":
                    if not tradeable:
                        signal = "NONE"
                        reason += " | GATED:time"
                    elif past_cutoff:
                        signal = "NONE"
                        reason += " | GATED:session_cutoff"
                    elif news_halt:
                        signal = "NONE"
                        reason += f" | GATED:news({news_halt[:30]})"
                    elif not engine.can_signal(ticker, now):
                        signal = "NONE"
                        reason += " | GATED:cooldown"
                    elif spread > MAX_SPREAD_PCT:
                        signal = "NONE"
                        reason += f" | GATED:spread({spread*100:.3f}%)"

                # --- Chart-based gating ---
                if signal == "CALL":
                    gate = st.chart.should_gate_call(raw_price)
                    if gate:
                        signal = "NONE"
                        reason += f" | GATED:chart({gate})"
                elif signal == "PUT":
                    gate = st.chart.should_gate_put(raw_price)
                    if gate:
                        signal = "NONE"
                        reason += f" | GATED:chart({gate})"

                # --- Breakout boost ---
                if signal in ("CALL", "PUT"):
                    boost = st.chart.breakout_boost()
                    if boost > 1.0:
                        confidence = min(confidence * boost, 1.0)
                        reason += f" | BOOST:breakout({boost:.2f}x)"

                # --- News trade riding ---
                # Even if normal signal is NONE, a confirmed news event can fire
                news_trade = news.check_news_trade(ticker, raw_price, now)
                if news_trade and not past_cutoff and not engine.session_ended:
                    if len(engine.open_trades) < MAX_OPEN_POSITIONS and \
                       ticker not in engine.open_trades:
                        if signal == "NONE" or signal == news_trade['signal']:
                            signal = news_trade['signal']
                            confidence = min(confidence * news_trade['confidence_boost'], 1.0)
                            confidence = max(confidence, 0.7)
                            reason += f" | {news_trade['reason']}"

                # --- News sentiment filter ---
                # news_agrees: boost confidence (trend confirmation)
                # news_disagrees + weak score: GATE (counter-trend garbage)
                # news_disagrees + strong score (7+): allow (legitimate reversal)
                NEWS_DISAGREE_MIN_SCORE = 8.0

                if signal in ("CALL", "PUT") and not news_trade:
                    news_vote = news.sentiment_vote(ticker)
                    if news_vote != 0:
                        if (signal == "CALL" and news_vote > 0) or \
                           (signal == "PUT" and news_vote < 0):
                            confidence = min(confidence * 1.1, 1.0)
                            reason += " | news_agrees"
                        elif (signal == "CALL" and news_vote < 0) or \
                             (signal == "PUT" and news_vote > 0):
                            if abs(raw_score) >= NEWS_DISAGREE_MIN_SCORE:
                                # Strong counter-trend signal — allow but reduce confidence
                                confidence *= 0.7
                                reason += " | news_disagrees(strong, allowed)"
                            else:
                                # Weak counter-trend signal — gate it
                                signal = "NONE"
                                reason += f" | GATED:news_disagrees(score={abs(raw_score):.1f}<{NEWS_DISAGREE_MIN_SCORE})"

                signal_counts[signal] += 1

                # --- Update GUI state ---
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
                td['chart'] = st.chart.summary()

                if signal in ("CALL", "PUT"):
                    gui_log(f"{signal} {ticker} conf={confidence:.2f} [{current_regime}]")

                # --- Log ---
                event = SignalEvent(
                    timestamp=now,
                    time_str=now_et().strftime("%H:%M:%S.%f")[:-3],
                    ticker=ticker,
                    raw_price=raw_price,
                    smooth_price=round(smooth, 4),
                    velocity=round(st.velocity, 8),
                    acceleration=round(st.acceleration, 8),
                    jerk=round(z_score, 4),        # repurposed: VWAP z-score
                    obi=round(obi, 4),
                    atr_pct=round(st.regime.fast_vol * 100, 4),
                    ema=round(st.vwap.vwap, 4),    # VWAP instead of EMA
                    spread=round(spread, 6),
                    signal=signal,
                    reason=reason,
                )
                sig_logger.log_signal(event)

                # --- Execute ---
                if signal in ("CALL", "PUT"):
                    engine.execute(ticker, signal, raw_price, spread, confidence, now)
                    log.info(
                        f"[{ticker}] {signal} | conf={confidence:.2f} | "
                        f"regime={current_regime} | {reason}"
                    )

            # --- Periodic housekeeping ---
            if now - last_flush > 30:
                # Collect completed outcomes BEFORE flush removes them
                for event in sig_logger._fired_buffer:
                    if event.pnl_120s_pct is not None:
                        if not any(abs(o[0] - event.timestamp) < 0.01
                                   for o in all_outcomes):
                            all_outcomes.append(
                                (event.timestamp, event.pnl_120s_pct, event.signal)
                            )
                            result = "WIN" if event.pnl_120s_pct > 0 else "LOSS"
                            gui_log(
                                f"{result} {event.ticker} {event.signal} "
                                f"{event.pnl_120s_pct:+.4f}% @ 120s"
                            )

                # Now flush
                sig_logger.flush_completed_outcomes()
                last_flush = now
                sc = signal_counts

                # Update GUI outcome stats
                all_pnls = [o[1] for o in all_outcomes]
                if all_pnls:
                    wins = sum(1 for v in all_pnls if v > 0)
                    gui_state['outcomes'] = {
                        'count': len(all_pnls),
                        'win_rate': (wins / len(all_pnls)) * 100,
                        'avg_pnl': sum(all_pnls) / len(all_pnls),
                        'total_pnl': sum(all_pnls),
                        'wins': wins,
                        'losses': len(all_pnls) - wins,
                    }

                log.info(
                    f"[STATUS] {sc['CALL']}C/{sc['PUT']}P/{sc['NONE']}skip | "
                    f"PnL=${engine.session_pnl:+.2f} | "
                    f"SPY={ds.market_data['SPY']['p']:.2f} "
                    f"QQQ={ds.market_data['QQQ']['p']:.2f} | "
                    f"Regimes: SPY={states['SPY'].regime.regime} "
                    f"QQQ={states['QQQ'].regime.regime}"
                )

            time.sleep(1)

    except KeyboardInterrupt:
        log.info("Shutting down NEXUS...")
        sig_logger.close()

        print("\n" + "=" * 60)
        print("  NEXUS SESSION SUMMARY")
        print("=" * 60)
        print(f"  Signals: {signal_counts['CALL']}C / {signal_counts['PUT']}P / {signal_counts['NONE']} skip")
        print(f"  Session PnL: ${engine.session_pnl:+.2f}")
        print(f"\n  Analyze outcomes:")
        print(f"    python signal_logger.py logs/nexus/outcomes_"
              f"{now_et().strftime('%Y-%m-%d')}.csv")
        print("=" * 60)

        if live:
            try:
                engine.api.close_all_positions()
                log.info("All positions closed.")
            except Exception as e:
                log.error(f"Flatten failed: {e}")
        sys.exit(0)


# ====================================================================
# NEXUS GUI
# ====================================================================
class NexusGUI:
    def __init__(self, root):
        self.root = root
        mode = gui_state.get('mode', 'OBSERVE')
        self.root.title(f"NEXUS V2.0 | {mode} | {TRADE_MODE.upper()}")
        self.root.geometry("680x1100")
        self.root.configure(bg="#06080c")
        self.root.attributes("-topmost", True)
        self.root.resizable(True, True)

        # === COLOR PALETTE (cyberpunk terminal) ===
        self.bg = "#06080c"
        self.card = "#0d1117"
        self.card_border = "#1a1f2e"
        self.fg = "#c9d1d9"
        self.fg_bright = "#f0f6fc"
        self.green = "#00ff87"
        self.green_dim = "#0a4a2a"
        self.red = "#ff3860"
        self.red_dim = "#4a0a1a"
        self.gold = "#ffc857"
        self.cyan = "#00d4ff"
        self.cyan_dim = "#0a2a3a"
        self.purple = "#bd93f9"
        self.dim = "#484f58"
        self.accent = "#58a6ff"
        self.orange = "#f0883e"

        self.regime_colors = {
            'TRENDING': self.cyan,
            'MEAN_REVERT': self.gold,
            'VOLATILE': self.red,
            '---': self.dim,
        }
        self.regime_icons = {
            'TRENDING': "\u2197",    # ↗
            'MEAN_REVERT': "\u21c4", # ⇄
            'VOLATILE': "\u26a1",    # ⚡
            '---': "\u2022",         # •
        }

        # Track last PnL for flash effect
        self._last_pnl = 0.0
        self._flash_count = 0

        self.build_ui()
        self.update_gui()

    def _card(self, parent, highlight=None, **kwargs):
        """Create a card frame with optional colored left border."""
        outer = tk.Frame(parent, bg=highlight or self.card_border, padx=1, pady=1, **kwargs)
        inner = tk.Frame(outer, bg=self.card)
        inner.pack(fill=tk.BOTH, expand=True)
        return outer, inner

    def _sep(self, parent, color=None):
        """Thin horizontal separator line."""
        tk.Frame(parent, bg=color or self.card_border, height=1).pack(fill=tk.X, padx=0, pady=0)

    def _vote_bar(self, parent, vote, name_short, width=50, height=14):
        """Draw a mini colored bar for a vote: green(+1), red(-1), dim(0)."""
        c = tk.Canvas(parent, width=width, height=height, bg=self.card,
                      highlightthickness=0, bd=0)
        mid = width // 2
        if vote > 0:
            c.create_rectangle(mid, 1, mid + (mid - 4), height - 1,
                               fill=self.green, outline="")
            c.create_text(4, height // 2, text=name_short, anchor="w",
                          fill=self.green, font=("Consolas", 7, "bold"))
        elif vote < 0:
            c.create_rectangle(4, 1, mid, height - 1,
                               fill=self.red, outline="")
            c.create_text(4, height // 2, text=name_short, anchor="w",
                          fill=self.red, font=("Consolas", 7, "bold"))
        else:
            c.create_rectangle(mid - 2, 4, mid + 2, height - 4,
                               fill=self.dim, outline="")
            c.create_text(4, height // 2, text=name_short, anchor="w",
                          fill=self.dim, font=("Consolas", 7))
        return c

    def build_ui(self):
        # Scrollable container
        self.canvas = tk.Canvas(self.root, bg=self.bg, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas, bg=self.bg)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Mouse scroll binding
        self.canvas.bind_all("<MouseWheel>",
                             lambda e: self.canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        container = self.scroll_frame

        # === HEADER BAR ===
        hdr = tk.Frame(container, bg=self.cyan, pady=1)
        hdr.pack(fill=tk.X)
        hdr_inner = tk.Frame(hdr, bg="#080c12", pady=6)
        hdr_inner.pack(fill=tk.X, padx=2, pady=(2, 0))

        # Title row
        title_row = tk.Frame(hdr_inner, bg="#080c12")
        title_row.pack()
        tk.Label(title_row, text="\u25c8", font=("Consolas", 22),
                 bg="#080c12", fg=self.cyan).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(title_row, text="N E X U S", font=("Consolas", 22, "bold"),
                 bg="#080c12", fg=self.fg_bright).pack(side=tk.LEFT)
        tk.Label(title_row, text=" V2.1", font=("Consolas", 22),
                 bg="#080c12", fg=self.cyan).pack(side=tk.LEFT)

        mode_color = self.green if gui_state.get('mode') == 'LIVE' else self.gold
        mode_text = f"\u25cf  {TRADE_MODE.upper()} MODE  \u2502  {gui_state.get('mode', 'OBSERVE')}"
        tk.Label(hdr_inner, text=mode_text, font=("Consolas", 9, "bold"),
                 bg="#080c12", fg=mode_color).pack(pady=(0, 2))

        # Cyan accent line
        tk.Frame(container, bg=self.cyan, height=2).pack(fill=tk.X)

        # === EQUITY CARD ===
        eq_outer, eq = self._card(container, highlight=self.cyan_dim)
        eq_outer.pack(fill=tk.X, padx=10, pady=(8, 4))

        eq_top = tk.Frame(eq, bg=self.card)
        eq_top.pack(fill=tk.X, padx=12, pady=(8, 0))
        tk.Label(eq_top, text="PORTFOLIO", font=("Consolas", 8, "bold"),
                 bg=self.card, fg=self.dim).pack(anchor="w")

        self.lbl_equity = tk.Label(eq, text="$0.00", font=("Consolas", 28, "bold"),
                                   bg=self.card, fg=self.fg_bright)
        self.lbl_equity.pack(padx=12, anchor="w")

        pnl_row = tk.Frame(eq, bg=self.card)
        pnl_row.pack(fill=tk.X, padx=12, pady=(0, 4))
        self.lbl_pnl = tk.Label(pnl_row, text="\u25b2 $0.00", font=("Consolas", 13, "bold"),
                                bg=self.card, fg=self.dim)
        self.lbl_pnl.pack(side=tk.LEFT)
        self.lbl_status = tk.Label(pnl_row, text="INIT", font=("Consolas", 9, "bold"),
                                   bg=self.card, fg=self.gold)
        self.lbl_status.pack(side=tk.RIGHT, padx=(0, 4))

        # PnL bar canvas
        self.pnl_canvas = tk.Canvas(eq, width=620, height=6, bg="#1a1f2e",
                                     highlightthickness=0)
        self.pnl_canvas.pack(padx=12, pady=(0, 8))

        # === STATS ROW (signals + outcomes + news) ===
        stats_outer, stats = self._card(container)
        stats_outer.pack(fill=tk.X, padx=10, pady=2)

        self.lbl_sig_stats = tk.Label(stats, text="\u25cf  Signals: ---",
                                      font=("Consolas", 9), bg=self.card, fg=self.dim)
        self.lbl_sig_stats.pack(anchor="w", padx=8, pady=(4, 0))
        self.lbl_outcome_stats = tk.Label(stats, text="\u25cf  Outcomes: waiting...",
                                          font=("Consolas", 9), bg=self.card, fg=self.dim)
        self.lbl_outcome_stats.pack(anchor="w", padx=8)
        self.lbl_news = tk.Label(stats, text="\u25cf  News: monitoring...",
                                 font=("Consolas", 9), bg=self.card, fg=self.dim)
        self.lbl_news.pack(anchor="w", padx=8, pady=(0, 4))

        # === TICKER PANELS ===
        self.ticker_panels = {}
        for ticker in WATCHLIST:
            self.ticker_panels[ticker] = self._build_ticker_panel(container, ticker)

        # === POSITIONS ===
        pos_hdr = tk.Frame(container, bg=self.bg)
        pos_hdr.pack(fill=tk.X, padx=12, pady=(8, 2))
        tk.Label(pos_hdr, text="\u2588\u2588 POSITIONS", font=("Consolas", 9, "bold"),
                 bg=self.bg, fg=self.accent).pack(anchor="w")
        pos_outer, pos_inner = self._card(container)
        pos_outer.pack(fill=tk.X, padx=10)
        self.pos_frame = pos_inner

        # === SIGNAL LOG ===
        log_hdr = tk.Frame(container, bg=self.bg)
        log_hdr.pack(fill=tk.X, padx=12, pady=(8, 2))
        tk.Label(log_hdr, text="\u2588\u2588 SIGNAL LOG", font=("Consolas", 9, "bold"),
                 bg=self.bg, fg=self.accent).pack(anchor="w")
        log_outer, log_inner = self._card(container)
        log_outer.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.log_frame = log_inner

    def _build_ticker_panel(self, parent, ticker):
        # Colored left-accent border based on ticker
        accent = self.cyan if ticker == "SPY" else self.purple
        outer = tk.Frame(parent, bg=self.bg)
        outer.pack(fill=tk.X, padx=10, pady=3)

        # Left accent stripe + card
        inner_frame = tk.Frame(outer, bg=self.card_border)
        inner_frame.pack(fill=tk.X)

        accent_stripe = tk.Frame(inner_frame, bg=accent, width=3)
        accent_stripe.pack(side=tk.LEFT, fill=tk.Y)

        fr = tk.Frame(inner_frame, bg=self.card)
        fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Row 1: Ticker + Price + Regime badge ---
        r1 = tk.Frame(fr, bg=self.card)
        r1.pack(fill=tk.X, padx=10, pady=(6, 0))

        tk.Label(r1, text=ticker, font=("Consolas", 16, "bold"),
                 bg=self.card, fg=accent).pack(side=tk.LEFT)

        # Regime badge (right side)
        regime_frame = tk.Frame(r1, bg=self.card)
        regime_frame.pack(side=tk.RIGHT)
        lbl_regime_icon = tk.Label(regime_frame, text="\u2022", font=("Consolas", 12),
                                   bg=self.card, fg=self.dim)
        lbl_regime_icon.pack(side=tk.LEFT)
        lbl_regime = tk.Label(regime_frame, text="---", font=("Consolas", 10, "bold"),
                              bg=self.card, fg=self.dim)
        lbl_regime.pack(side=tk.LEFT, padx=(2, 0))

        lbl_price = tk.Label(r1, text="$0.00", font=("Consolas", 16, "bold"),
                             bg=self.card, fg=self.fg_bright)
        lbl_price.pack(side=tk.RIGHT, padx=(0, 12))

        self._sep(fr, self.card_border)

        # --- Row 2: Key indicators in a grid layout ---
        r2 = tk.Frame(fr, bg=self.card)
        r2.pack(fill=tk.X, padx=10, pady=3)

        # Mini indicator cells
        indicators = ['VWAP', 'Z-Score', 'RSI', 'OBI', 'Spread']
        self_labels = {}
        for i, name in enumerate(indicators):
            cell = tk.Frame(r2, bg=self.card)
            cell.pack(side=tk.LEFT, expand=True, fill=tk.X)
            tk.Label(cell, text=name.upper(), font=("Consolas", 7),
                     bg=self.card, fg=self.dim).pack()
            lbl = tk.Label(cell, text="---", font=("Consolas", 9, "bold"),
                           bg=self.card, fg=self.fg)
            lbl.pack()
            self_labels[name] = lbl

        self._sep(fr, self.card_border)

        # --- Row 3: Velocity / Flow / VolRatio ---
        r3 = tk.Frame(fr, bg=self.card)
        r3.pack(fill=tk.X, padx=10, pady=2)
        lbl_deriv = tk.Label(r3, text="", font=("Consolas", 9),
                             bg=self.card, fg=self.dim)
        lbl_deriv.pack(anchor="w")

        # --- Row 4: MTF direction arrows ---
        r4 = tk.Frame(fr, bg=self.card)
        r4.pack(fill=tk.X, padx=10, pady=2)
        lbl_mtf = tk.Label(r4, text="", font=("Consolas", 9),
                           bg=self.card, fg=self.dim)
        lbl_mtf.pack(anchor="w")

        self._sep(fr, self.card_border)

        # --- Row 5: Vote bars + Signal ---
        r5 = tk.Frame(fr, bg=self.card)
        r5.pack(fill=tk.X, padx=10, pady=(3, 0))
        vote_bar_frame = tk.Frame(r5, bg=self.card)
        vote_bar_frame.pack(anchor="w")

        # Signal output
        sig_row = tk.Frame(fr, bg=self.card)
        sig_row.pack(fill=tk.X, padx=10, pady=(2, 0))
        lbl_signal = tk.Label(sig_row, text="\u2500\u2500  NONE", font=("Consolas", 11, "bold"),
                              bg=self.card, fg=self.dim)
        lbl_signal.pack(side=tk.LEFT)
        lbl_conf = tk.Label(sig_row, text="0%", font=("Consolas", 10),
                            bg=self.card, fg=self.dim)
        lbl_conf.pack(side=tk.RIGHT, padx=(0, 4))

        # --- Row 6: Chart context ---
        r6 = tk.Frame(fr, bg=self.card)
        r6.pack(fill=tk.X, padx=10, pady=(2, 6))
        lbl_chart = tk.Label(r6, text="", font=("Consolas", 8),
                             bg=self.card, fg=self.dim)
        lbl_chart.pack(anchor="w")

        return {
            'lbl_price': lbl_price,
            'lbl_regime': lbl_regime,
            'lbl_regime_icon': lbl_regime_icon,
            'ind_labels': self_labels,
            'lbl_deriv': lbl_deriv,
            'lbl_mtf': lbl_mtf,
            'vote_bar_frame': vote_bar_frame,
            'lbl_signal': lbl_signal,
            'lbl_conf': lbl_conf,
            'lbl_chart': lbl_chart,
            'accent': accent,
        }

    def update_gui(self):
        # --- Equity ---
        equity = gui_state['equity']
        self.lbl_equity.config(text=f"${equity:,.2f}")

        pnl = gui_state['session_pnl']
        if pnl >= 0:
            arrow = "\u25b2"
            pcol = self.green
        else:
            arrow = "\u25bc"
            pcol = self.red
        self.lbl_pnl.config(text=f"{arrow} ${abs(pnl):,.2f} today", fg=pcol)

        # PnL bar visualization
        self.pnl_canvas.delete("all")
        max_loss = 2000
        bar_width = 610
        mid = bar_width // 2
        if pnl >= 0:
            fill_w = min(int((pnl / max_loss) * mid), mid)
            self.pnl_canvas.create_rectangle(mid, 0, mid + fill_w, 6,
                                              fill=self.green, outline="")
        else:
            fill_w = min(int((abs(pnl) / max_loss) * mid), mid)
            self.pnl_canvas.create_rectangle(mid - fill_w, 0, mid, 6,
                                              fill=self.red, outline="")
        self.pnl_canvas.create_line(mid, 0, mid, 6, fill=self.dim)

        # --- Status ---
        status = gui_state['status']
        if "LIVE" in status:
            scol = self.green
            status = "\u25cf LIVE  " + status.replace("LIVE", "").strip()
        elif "OBSERVING" in status:
            scol = self.cyan
            status = "\u25cb OBS  " + status.replace("OBSERVING", "").strip()
        elif "WARM" in status:
            scol = self.gold
            status = "\u25d4 " + status
        else:
            scol = self.gold
        self.lbl_status.config(text=status, fg=scol)

        # --- Signal counts ---
        sc = gui_state['signal_counts']
        calls = sc.get('CALL', 0)
        puts = sc.get('PUT', 0)
        total = calls + puts
        self.lbl_sig_stats.config(
            text=f"\u25cf  Signals: {total} fired  "
                 f"({calls}\u25b2 / {puts}\u25bc)  \u2502  "
                 f"{sc.get('NONE', 0)} filtered"
        )

        # --- Outcomes ---
        oc = gui_state.get('outcomes', {})
        if oc.get('count', 0) > 0:
            wr = oc.get('win_rate', 0)
            avg = oc.get('avg_pnl', 0)
            avg_col = self.green if avg > 0 else self.red
            spread_cost = 0.0014 if TRADE_MODE == "shares" else 0.10
            edge = avg - spread_cost
            self.lbl_outcome_stats.config(
                text=f"\u25cf  {oc['count']} outcomes  \u2502  "
                     f"Win: {wr:.0f}%  \u2502  "
                     f"Avg: {avg:+.4f}%  \u2502  "
                     f"Edge: {edge:+.4f}%",
                fg=avg_col
            )
        else:
            self.lbl_outcome_stats.config(
                text="\u25cb  Outcomes: collecting data...", fg=self.dim)

        # --- News ---
        news_data = gui_state.get('news', {})
        if news_data.get('halted'):
            self.lbl_news.config(
                text=f"\u26a0  NEWS HALT: {news_data.get('halt_reason', '')[:45]}",
                fg=self.red)
        elif news_data.get('riding'):
            self.lbl_news.config(
                text=f"\u26a1  RIDING: {news_data.get('latest', '')[:45]}",
                fg=self.green)
        elif news_data.get('latest', '---') != '---':
            self.lbl_news.config(
                text=f"\u25cf  {news_data['latest'][:50]}", fg=self.dim)
        else:
            self.lbl_news.config(text="\u25cb  News: monitoring...", fg=self.dim)

        # --- Ticker Panels ---
        for ticker in WATCHLIST:
            td = gui_state['tickers'].get(ticker, {})
            p = self.ticker_panels[ticker]

            # Price (flash effect on change)
            price = td.get('price', 0)
            p['lbl_price'].config(text=f"${price:.2f}")

            # Regime badge
            regime = td.get('regime', '---')
            rcol = self.regime_colors.get(regime, self.dim)
            icon = self.regime_icons.get(regime, "\u2022")
            p['lbl_regime'].config(text=regime, fg=rcol)
            p['lbl_regime_icon'].config(text=icon, fg=rcol)

            # Indicator grid
            z = td.get('z_score', 0)
            rsi = td.get('rsi', 50)
            obi = td.get('obi', 0)
            spread = td.get('spread', 0)

            p['ind_labels']['VWAP'].config(text=f"${td.get('vwap', 0):.2f}")
            p['ind_labels']['Z-Score'].config(
                text=f"{z:+.2f}",
                fg=self.green if abs(z) < 1.5 else (self.gold if abs(z) < 2.5 else self.red))
            p['ind_labels']['RSI'].config(
                text=f"{rsi:.0f}",
                fg=self.red if rsi > 70 else (self.green if rsi < 30 else self.fg))
            p['ind_labels']['OBI'].config(
                text=f"{obi:+.3f}",
                fg=self.green if obi > 0.08 else (self.red if obi < -0.08 else self.fg))
            p['ind_labels']['Spread'].config(
                text=f"{spread:.3f}%",
                fg=self.green if spread < 0.001 else (self.red if spread > 0.003 else self.fg))

            # Derivatives
            v = td.get('velocity', 0)
            vr = td.get('vol_ratio', 0)
            flow = td.get('net_flow', 0)
            v_arrow = "\u25b2" if v > 0 else "\u25bc"
            p['lbl_deriv'].config(
                text=f"Vel:{v_arrow}{abs(v):.5f}  "
                     f"Flow:{'>' if flow > 0 else '<'}{abs(flow):.3f}  "
                     f"VR:{vr:.2f}",
                fg=self.green if v > 0 else self.red)

            # MTF arrows with Unicode
            b5 = td.get('bar5s', 0)
            b30 = td.get('bar30s', 0)
            b5m = td.get('bar5m', 0)

            def tf_arrow(val):
                if val > 0.3: return ("\u25b2", self.green)   # ▲
                elif val < -0.3: return ("\u25bc", self.red)   # ▼
                return ("\u2500", self.dim)                     # ─

            a5, c5 = tf_arrow(b5)
            a30, c30 = tf_arrow(b30)
            a5m, c5m = tf_arrow(b5m)

            # Check agreement
            all_up = b5 > 0.3 and b30 > 0.3 and b5m > 0
            all_dn = b5 < -0.3 and b30 < -0.3 and b5m < 0
            agree = all_up or all_dn
            agree_text = "\u2713 ALIGNED" if agree else "\u2717 mixed"
            agree_col = self.green if agree else self.dim

            p['lbl_mtf'].config(
                text=f"5s:{a5}  30s:{a30}  5m:{a5m}   {agree_text}",
                fg=agree_col)

            # Vote bars (rebuild each update)
            for w in p['vote_bar_frame'].winfo_children():
                w.destroy()
            votes = td.get('votes', {})
            if votes:
                name_map = {
                    'momentum': 'MOM', 'trade_flow': 'FLW',
                    'vwap_dev': 'VWP', 'rsi': 'RSI',
                    'obi': 'OBI', 'mtf': 'MTF',
                }
                for vname, val in votes.items():
                    short = name_map.get(vname, vname[:3].upper())
                    bar = self._vote_bar(p['vote_bar_frame'], val, short,
                                         width=58, height=16)
                    bar.pack(side=tk.LEFT, padx=1)

            # Signal output
            sig = td.get('last_signal', 'NONE')
            conf = td.get('confidence', 0)
            if sig == "CALL":
                sig_text = "\u25b2\u25b2 CALL"
                sig_col = self.green
                p['lbl_signal'].config(bg=self.green_dim)
            elif sig == "PUT":
                sig_text = "\u25bc\u25bc PUT"
                sig_col = self.red
                p['lbl_signal'].config(bg=self.red_dim)
            else:
                sig_text = "\u2500\u2500 NONE"
                sig_col = self.dim
                p['lbl_signal'].config(bg=self.card)
            p['lbl_signal'].config(text=f"  {sig_text}  ", fg=sig_col)
            p['lbl_conf'].config(text=f"{conf:.0%}", fg=sig_col)

            # Chart context
            chart = td.get('chart', {})
            if chart:
                struct = chart.get('structure', '---')
                struct_icon = "\u25b2" if struct == "BULLISH" else (
                    "\u25bc" if struct == "BEARISH" else "\u2500")
                struct_col = self.green if struct == "BULLISH" else (
                    self.red if struct == "BEARISH" else self.dim)
                squeeze = "\u26a1SQZ" if chart.get('consolidating') else ""
                brk = chart.get('breakout', '---')
                res = chart.get('at_res', '---')
                sup = chart.get('at_sup', '---')
                room_u = chart.get('room_up', 0)
                room_d = chart.get('room_down', 0)
                p['lbl_chart'].config(
                    text=f"{struct_icon}{struct}  {squeeze}  "
                         f"R:{res} S:{sup}  "
                         f"\u2191{room_u:.1f}%/\u2193{room_d:.1f}%  "
                         f"Brk:{brk}",
                    fg=struct_col)

        # --- Positions ---
        for w in self.pos_frame.winfo_children():
            w.destroy()
        positions = gui_state.get('positions', {})
        if not positions:
            msg = "\u25cb  No positions" if gui_state['mode'] == 'OBSERVE' else "\u25cb  Flat"
            tk.Label(self.pos_frame, text=f"  {msg}",
                     font=("Consolas", 9), bg=self.card, fg=self.dim
                     ).pack(anchor="w", padx=8, pady=4)
        else:
            for sym, d in positions.items():
                gain = d['gain']
                pnl_d = d['pnl']
                col = self.green if gain > 0 else self.red
                arrow = "\u25b2" if gain > 0 else "\u25bc"
                row = tk.Frame(self.pos_frame, bg=self.card)
                row.pack(fill=tk.X, padx=8, pady=2)
                tk.Label(row, text=f"  {sym[:12]}",
                         font=("Consolas", 10, "bold"), bg=self.card, fg=self.fg
                         ).pack(side=tk.LEFT)
                tk.Label(row, text=f"{arrow} {gain:+.2f}%  (${pnl_d:+.2f})",
                         font=("Consolas", 10, "bold"), bg=self.card, fg=col
                         ).pack(side=tk.RIGHT, padx=(0, 8))
                tk.Label(row, text=f"{d['qty']}x",
                         font=("Consolas", 9), bg=self.card, fg=self.dim
                         ).pack(side=tk.RIGHT, padx=(0, 8))

        # --- Signal Log ---
        for w in self.log_frame.winfo_children():
            w.destroy()
        for entry in gui_state.get('trade_log', []):
            if "CALL" in entry:
                col, icon = self.green, "\u25b2"
            elif "PUT" in entry:
                col, icon = self.red, "\u25bc"
            elif "WIN" in entry:
                col, icon = self.green, "\u2713"
            elif "LOSS" in entry:
                col, icon = self.red, "\u2717"
            elif "NEWS" in entry:
                col, icon = self.gold, "\u26a1"
            elif "EXIT" in entry:
                col, icon = self.orange, "\u25c9"
            elif "SESSION" in entry:
                col, icon = self.purple, "\u25a0"
            else:
                col, icon = self.cyan, "\u25cf"
            tk.Label(self.log_frame, text=f" {icon} {entry}",
                     font=("Consolas", 9), bg=self.card, fg=col, anchor="w"
                     ).pack(fill=tk.X, padx=6, pady=1)

        self.root.after(500, self.update_gui)


# ====================================================================
# ENTRY
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="NEXUS V1.0 — Regime-Aware Fusion Engine")
    parser.add_argument("--live", action="store_true",
                        help="Enable live paper trading (default: observe only)")
    parser.add_argument("--headless", action="store_true",
                        help="No GUI, log to console + file only")
    args = parser.parse_args()

    gui_state['mode'] = 'LIVE' if args.live else 'OBSERVE'

    ds = DataStream()
    threading.Thread(target=ds.start_stream, daemon=True).start()
    threading.Thread(target=nexus_loop, args=(ds, args.live), daemon=True).start()

    if HAS_TK and not args.headless:
        root = tk.Tk()
        app = NexusGUI(root)
        root.mainloop()
    else:
        log.info("Running headless (no GUI).")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()