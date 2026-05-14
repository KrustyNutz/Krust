from __future__ import annotations

# ====================================================================
# NEWS.PY | Smart News Riding for NEXUS V2
#
# You can't beat HFTs to the initial spike. But you CAN ride the
# follow-through wave that lasts 5-15 minutes after high-impact news.
#
# Flow:
#   1. Detect high-impact headline (FOMC, CPI, earnings, etc.)
#   2. Score sentiment: bullish or bearish
#   3. WAIT 60-90s for initial whipsaw to settle
#   4. Check if price action CONFIRMS the headline direction
#   5. If confirmed -> signal aggressive entry with confidence boost
#   6. If not confirmed -> news was priced in, resume normal
# ====================================================================

import time
import logging
import re
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict

log = logging.getLogger("nexus.news")

# ====================================================================
# IMPACT KEYWORDS: (settle_seconds, impact_level)
# ====================================================================
HIGH_IMPACT_KEYWORDS = {
    'fomc': (90, 3), 'federal reserve': (90, 3), 'rate decision': (90, 3),
    'rate cut': (60, 3), 'rate hike': (60, 3), 'interest rate': (60, 2),
    'powell': (90, 3), 'fed chair': (90, 3),
    'cpi': (60, 3), 'inflation data': (60, 3), 'pce': (60, 3),
    'jobs report': (60, 3), 'nonfarm': (60, 3), 'non-farm': (60, 3),
    'unemployment': (60, 2), 'gdp': (60, 2), 'ppi': (45, 2),
    'retail sales': (45, 2), 'consumer confidence': (45, 2),
    'tariff': (60, 3), 'trade war': (60, 3), 'sanctions': (45, 2),
    'circuit breaker': (120, 3), 'trading halt': (90, 3),
}

BULLISH_WORDS = {
    'surge', 'surges', 'soar', 'soars', 'rally', 'rallies', 'jump', 'jumps',
    'gain', 'gains', 'climb', 'climbs', 'rise', 'rises', 'bullish', 'upbeat',
    'optimistic', 'strong', 'beat', 'beats', 'exceed', 'exceeds', 'record high',
    'breakout', 'upgrade', 'outperform', 'buy', 'cut', 'easing', 'stimulus',
    'dovish', 'rebound', 'recovery', 'boom', 'above expectations',
}

BEARISH_WORDS = {
    'plunge', 'plunges', 'crash', 'crashes', 'tumble', 'tumbles', 'sink',
    'sinks', 'drop', 'drops', 'fall', 'falls', 'decline', 'declines',
    'bearish', 'fear', 'fears', 'weak', 'miss', 'misses', 'disappoint',
    'warning', 'downgrade', 'underperform', 'sell', 'selloff', 'sell-off',
    'recession', 'layoff', 'loss', 'losses', 'hike', 'hawkish',
    'tightening', 'contraction', 'below expectations', 'tariff', 'trade war',
}


class NewsEvent:
    """Tracks a high-impact news event through settle -> confirm -> ride."""

    def __init__(self, headline: str, sentiment: float, impact_level: int,
                 settle_seconds: int, detected_at: float, trigger: str):
        self.headline = headline
        self.sentiment = sentiment
        self.impact_level = impact_level
        self.settle_seconds = settle_seconds
        self.detected_at = detected_at
        self.trigger = trigger
        self.settle_until = detected_at + settle_seconds
        self.phase = "SETTLING"  # SETTLING -> CONFIRMING -> RIDING -> DONE
        self.price_at_detection = 0.0
        self.signal_fired = False
        self.expires_at = detected_at + 600  # 10 min max

    def is_active(self, now: float) -> bool:
        return now < self.expires_at and self.phase != "DONE"

    def is_settling(self, now: float) -> bool:
        return now < self.settle_until

    def expected_direction(self) -> Optional[str]:
        if self.sentiment > 0.15:
            return "CALL"
        elif self.sentiment < -0.15:
            return "PUT"
        return None


class NewsEngine:
    def __init__(self, api, tickers: List[str], poll_interval: int = 120):
        self.api = api
        self.tickers = tickers
        self.poll_interval = poll_interval
        self.last_poll_time: float = 0
        self.seen_headlines: set = set()
        self.active_events: List[NewsEvent] = []
        self.sentiment: Dict[str, float] = {t: 0.0 for t in tickers}
        self.status: str = "Monitoring..."
        self.latest_headline: str = ""
        # Track prices at detection for confirmation
        self.prices_at_detection: Dict[str, float] = {}

    def poll(self, now: float):
        if (now - self.last_poll_time) < self.poll_interval:
            return
        self.last_poll_time = now
        try:
            self._fetch_and_process(now)
        except Exception as e:
            log.warning(f"News poll failed: {e}")
        self.active_events = [e for e in self.active_events if e.is_active(now)]

    def _fetch_and_process(self, now: float):
        symbols_str = ",".join(self.tickers)
        try:
            raw_news = self.api.get_news(symbols_str, limit=10)
        except AttributeError:
            try:
                resp = self.api._request('GET', '/v1beta1/news',
                                         data={'symbols': symbols_str, 'limit': 10, 'sort': 'desc'})
                raw_news = resp if isinstance(resp, list) else []
            except Exception:
                return
        except Exception:
            return

        if not raw_news:
            return

        for item in raw_news:
            try:
                headline = getattr(item, 'headline', '') or ''
                summary = getattr(item, 'summary', '') or ''
                if isinstance(item, dict):
                    headline = item.get('headline', '')
                    summary = item.get('summary', '')

                if not headline or headline in self.seen_headlines:
                    continue
                self.seen_headlines.add(headline)

                sentiment = self._score(headline + " " + summary)
                text_lower = headline.lower()

                # Check impact
                max_settle, max_impact, trigger = 0, 0, ""
                for kw, (settle, impact) in HIGH_IMPACT_KEYWORDS.items():
                    if kw in text_lower and impact > max_impact:
                        max_settle, max_impact, trigger = settle, impact, kw

                if max_impact >= 2:
                    event = NewsEvent(headline, sentiment, max_impact,
                                     max_settle, now, trigger)
                    self.active_events.append(event)
                    direction = event.expected_direction() or "UNCLEAR"
                    log.warning(f"NEWS [{trigger}]: {headline[:50]} | "
                                f"sent={sentiment:+.2f} dir={direction} | "
                                f"settling {max_settle}s")

                self.latest_headline = headline[:80]
                # Update sentiment
                for ticker in self.tickers:
                    if sentiment != 0:
                        self.sentiment[ticker] = sentiment

            except Exception:
                continue

        self.status = f"Latest: {self.latest_headline[:40]}..."

    def _score(self, text: str) -> float:
        words = set(re.findall(r'\b\w+\b', text.lower()))
        bull = len(words & BULLISH_WORDS)
        bear = len(words & BEARISH_WORDS)
        total = bull + bear
        return (bull - bear) / total if total > 0 else 0.0

    # ================================================================
    # PUBLIC INTERFACE
    # ================================================================

    def is_halted(self, now: float) -> Optional[str]:
        """Only halt during settle phase of high-impact events."""
        for event in self.active_events:
            if event.impact_level >= 3 and event.is_settling(now):
                remaining = int(event.settle_until - now)
                return f"{event.trigger}: settling ({remaining}s)"
        return None

    def check_news_trade(self, ticker: str, current_price: float,
                         now: float) -> Optional[dict]:
        """
        After settle, check if price confirms headline direction.
        Returns trade signal dict or None.
        """
        detection_price = self.prices_at_detection.get(ticker, 0)

        for event in self.active_events:
            if event.signal_fired or event.phase == "DONE":
                continue
            if event.is_settling(now):
                # Store price at detection for later comparison
                if event.price_at_detection == 0:
                    event.price_at_detection = current_price
                    self.prices_at_detection[ticker] = current_price
                continue
            if event.impact_level < 2:
                continue

            # Settle just ended
            if event.phase == "SETTLING":
                event.phase = "CONFIRMING"
                if event.price_at_detection == 0:
                    event.price_at_detection = current_price
                log.info(f"NEWS settle done [{event.trigger}]. Checking confirmation...")

            if event.phase == "CONFIRMING":
                expected = event.expected_direction()
                if not expected:
                    event.phase = "DONE"
                    continue

                ref_price = event.price_at_detection
                if ref_price <= 0:
                    ref_price = current_price

                move_pct = ((current_price - ref_price) / ref_price) * 100

                # Confirmation: price moved 0.03%+ in expected direction
                confirmed = False
                if expected == "CALL" and move_pct > 0.03:
                    confirmed = True
                elif expected == "PUT" and move_pct < -0.03:
                    confirmed = True

                if confirmed:
                    event.phase = "RIDING"
                    event.signal_fired = True
                    boost = 1.5 if event.impact_level >= 3 else 1.2

                    log.warning(f"NEWS RIDE: {expected} [{event.trigger}] "
                                f"move={move_pct:+.3f}% boost={boost}x")

                    return {
                        'signal': expected,
                        'confidence_boost': boost,
                        'reason': f"news_ride({event.trigger} "
                                  f"sent={event.sentiment:+.2f} "
                                  f"move={move_pct:+.3f}%)",
                        'wider_stops': True,
                    }

                # Timeout: 2 min past settle without confirmation
                if now > event.settle_until + 120:
                    event.phase = "DONE"
                    log.info(f"NEWS [{event.trigger}] not confirmed. Skipping.")

        return None

    def sentiment_vote(self, ticker: str) -> int:
        sent = self.sentiment.get(ticker, 0.0)
        if sent > 0.2:
            return 1
        elif sent < -0.2:
            return -1
        return 0

    def has_active_event(self) -> bool:
        return any(e.impact_level >= 2 for e in self.active_events)

    def summary_for_gui(self) -> dict:
        now = time.time()
        return {
            'status': self.status,
            'latest': self.latest_headline[:60] if self.latest_headline else "---",
            'halted': any(e.is_settling(now) and e.impact_level >= 3
                         for e in self.active_events),
            'halt_reason': next(
                (f"{e.trigger} settling" for e in self.active_events
                 if e.is_settling(now) and e.impact_level >= 3), ""),
            'riding': any(e.phase == "RIDING" for e in self.active_events),
            'active_events': len(self.active_events),
            'sentiment': dict(self.sentiment),
        }


# Standalone helper
def gui_log(msg):
    log.info(msg)


if __name__ == "__main__":
    tests = [
        "Fed cuts rates by 50bps, markets surge",
        "CPI comes in hotter than expected, inflation fears mount",
        "FOMC holds rates steady, hawkish tone surprises",
        "Jobs report beats expectations, unemployment falls",
        "New tariffs announced on Chinese imports",
        "GDP growth disappoints, recession fears grow",
    ]
    for h in tests:
        words = set(re.findall(r'\b\w+\b', h.lower()))
        bull = len(words & BULLISH_WORDS)
        bear = len(words & BEARISH_WORDS)
        total = bull + bear
        sent = (bull - bear) / total if total > 0 else 0
        direction = "CALL" if sent > 0.15 else ("PUT" if sent < -0.15 else "???")
        impact = max((i for kw, (s, i) in HIGH_IMPACT_KEYWORDS.items()
                      if kw in h.lower()), default=0)
        print(f"  {sent:+.2f} {direction:4} impact={impact} | {h[:55]}")
