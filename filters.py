# ====================================================================
# FILTERS.PY | One-Euro Filter for Price Derivative Smoothing
# Ported from the Symbiote pipeline — adaptive low-pass that tracks
# fast moves tightly but smooths noise when price is quiet.
# ====================================================================

from __future__ import annotations
import math
import time
from typing import Optional


class LowPassFilter:
    """Simple first-order exponential low-pass filter."""

    __slots__ = ('_y', '_initialized')

    def __init__(self):
        self._y = 0.0
        self._initialized = False

    def filter(self, x: float, alpha: float) -> float:
        if not self._initialized:
            self._y = x
            self._initialized = True
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    @property
    def value(self) -> float:
        return self._y

    def reset(self):
        self._initialized = False
        self._y = 0.0


class OneEuroFilter:
    """
    One-Euro Filter: adaptive low-pass that increases cutoff frequency
    when the signal changes rapidly (large derivative), giving tight
    tracking of fast moves while heavily smoothing slow/still signals.

    Parameters:
        min_cutoff : base cutoff when signal is still (lower = smoother)
        beta       : derivative sensitivity (higher = faster response to change)
        d_cutoff   : cutoff for the derivative filter itself
    """

    __slots__ = ('_min_cutoff', '_beta', '_d_cutoff', '_x_filt', '_dx_filt',
                 '_last_time', '_freq')

    def __init__(self, min_cutoff: float = 0.004, beta: float = 0.7, d_cutoff: float = 1.0):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filt = LowPassFilter()
        self._dx_filt = LowPassFilter()
        self._last_time = 0.0
        self._freq = 1.0  # initial assumed frequency

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """
        Filter a new sample.

        Args:
            x: raw value (price, velocity, etc.)
            timestamp: epoch seconds. If None, uses time.time().

        Returns:
            Filtered value.
        """
        if timestamp is None:
            timestamp = time.time()

        if self._last_time > 0 and timestamp > self._last_time:
            self._freq = 1.0 / (timestamp - self._last_time)
        self._last_time = timestamp

        # Estimate derivative
        prev = self._x_filt.value if self._x_filt._initialized else x
        dx = (x - prev) * self._freq if self._freq > 0 else 0.0

        # Filter the derivative
        alpha_d = self._alpha(self._d_cutoff, self._freq)
        edx = self._dx_filt.filter(dx, alpha_d)

        # Adaptive cutoff: higher when signal is changing fast
        cutoff = self._min_cutoff + self._beta * abs(edx)
        alpha = self._alpha(cutoff, self._freq)

        return self._x_filt.filter(x, alpha)

    def reset(self):
        self._x_filt.reset()
        self._dx_filt.reset()
        self._last_time = 0.0

    @property
    def value(self) -> float:
        return self._x_filt.value


class SmoothedDerivatives:
    """
    Wraps three One-Euro filters to produce smoothed velocity,
    acceleration, and jerk from raw price ticks.

    All derivatives are percentage-based (relative to price).
    """

    __slots__ = ('_price_filt', '_vel_filt', '_acc_filt',
                 '_last_price', '_last_vel', '_last_acc', '_last_time')

    def __init__(self,
                 price_min_cutoff: float = 0.01,
                 price_beta: float = 0.5,
                 deriv_min_cutoff: float = 0.005,
                 deriv_beta: float = 0.7):
        self._price_filt = OneEuroFilter(min_cutoff=price_min_cutoff, beta=price_beta)
        self._vel_filt = OneEuroFilter(min_cutoff=deriv_min_cutoff, beta=deriv_beta)
        self._acc_filt = OneEuroFilter(min_cutoff=deriv_min_cutoff, beta=deriv_beta)
        self._last_price = 0.0
        self._last_vel = 0.0
        self._last_acc = 0.0
        self._last_time = 0.0

    def update(self, raw_price: float, timestamp: Optional[float] = None) -> dict:
        """
        Feed a new raw price tick, get back smoothed derivatives.

        Returns dict with keys: price, velocity, acceleration, jerk
        All derivative values are % per second.
        """
        if timestamp is None:
            timestamp = time.time()

        # Smooth the price itself
        price = self._price_filt.filter(raw_price, timestamp)

        dt = timestamp - self._last_time if self._last_time > 0 else 1.0
        dt = max(dt, 0.001)  # prevent div/0

        if self._last_price > 0:
            # Raw % velocity
            raw_vel = ((price - self._last_price) / self._last_price * 100.0) / dt
            vel = self._vel_filt.filter(raw_vel, timestamp)

            raw_acc = (vel - self._last_vel) / dt
            acc = self._acc_filt.filter(raw_acc, timestamp)

            jerk = (acc - self._last_acc) / dt
        else:
            vel, acc, jerk = 0.0, 0.0, 0.0

        self._last_price = price
        self._last_vel = vel
        self._last_acc = acc
        self._last_time = timestamp

        return {
            'price': price,
            'velocity': vel,
            'acceleration': acc,
            'jerk': jerk,
        }

    def reset(self):
        self._price_filt.reset()
        self._vel_filt.reset()
        self._acc_filt.reset()
        self._last_price = 0.0
        self._last_vel = 0.0
        self._last_acc = 0.0
        self._last_time = 0.0


# ====================================================================
# Self-test: sine wave + noise
# ====================================================================
if __name__ == "__main__":
    import random

    filt = SmoothedDerivatives()
    t = 0.0

    print("time,raw,smoothed_price,velocity,acceleration")
    for i in range(200):
        t += 1.0
        # Simulated price: ~550 + slow sine + noise
        raw = 550.0 + 2.0 * math.sin(t * 0.05) + random.gauss(0, 0.15)
        out = filt.update(raw, timestamp=t)
        if i % 10 == 0:
            print(f"{t:.0f},{raw:.4f},{out['price']:.4f},"
                  f"{out['velocity']:.6f},{out['acceleration']:.6f}")
