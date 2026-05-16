# ====================================================================
# SIGNAL_LOGGER.PY | Record every signal + hypothetical outcome
# Writes to CSV always, InfluxDB optionally.
# After a week of data you can answer: "do these signals have edge?"
# ====================================================================

from __future__ import annotations
import csv
import os
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque

logger = logging.getLogger("nexus.signals")


@dataclass
class SignalEvent:
    """A single moment where entry conditions were evaluated."""
    timestamp: float
    time_str: str
    ticker: str
    raw_price: float
    smooth_price: float
    velocity: float
    acceleration: float
    jerk: float
    obi: float
    atr_pct: float
    ema: float
    spread: float
    signal: str           # "CALL", "PUT", or "NONE"
    reason: str           # why signal fired or didn't
    # Hypothetical outcome tracking (filled in later)
    price_30s: Optional[float] = None
    price_60s: Optional[float] = None
    price_120s: Optional[float] = None
    pnl_30s_pct: Optional[float] = None
    pnl_60s_pct: Optional[float] = None
    pnl_120s_pct: Optional[float] = None


class OutcomeTracker:
    """
    Tracks hypothetical outcomes for fired signals.
    When a CALL/PUT signal fires, we record the entry price and check
    what happened 30/60/120 seconds later to measure signal quality.
    """

    def __init__(self):
        # Pending signals awaiting price lookups: deque of (signal_event, target_times)
        self._pending = deque(maxlen=500)  # deque of (SignalEvent, {offset: price})

    def register(self, event: SignalEvent):
        """Register a signal that fired (CALL or PUT) for outcome tracking."""
        if event.signal == "NONE":
            return
        targets = {
            30: event.timestamp + 30,
            60: event.timestamp + 60,
            120: event.timestamp + 120,
        }
        self._pending.append((event, targets))

    def check_outcomes(self, ticker: str, current_price: float, now: float):
        """
        Call every tick. For any pending signals whose target times have
        passed, fill in the outcome prices and compute hypothetical PnL.
        """
        still_pending = deque()
        for event, targets in self._pending:
            if event.ticker != ticker:
                still_pending.append((event, targets))
                continue

            all_filled = True
            for secs, target_time in targets.items():
                if now >= target_time:
                    attr_price = f"price_{secs}s"
                    attr_pnl = f"pnl_{secs}s_pct"
                    if getattr(event, attr_price) is None:
                        setattr(event, attr_price, current_price)
                        # PnL: positive = signal was correct
                        if event.signal == "CALL":
                            pnl = ((current_price - event.smooth_price) / event.smooth_price) * 100.0
                        else:  # PUT
                            pnl = ((event.smooth_price - current_price) / event.smooth_price) * 100.0
                        setattr(event, attr_pnl, round(pnl, 6))
                else:
                    all_filled = False

            if not all_filled:
                still_pending.append((event, targets))

        self._pending = still_pending


class SignalLogger:
    """
    Logs signal events to CSV and optionally to InfluxDB.
    CSV is the source of truth. InfluxDB is for Grafana dashboards.
    """

    CSV_HEADERS = [
        'timestamp', 'time_str', 'ticker', 'raw_price', 'smooth_price',
        'velocity', 'acceleration', 'jerk', 'obi', 'atr_pct', 'ema',
        'spread', 'signal', 'reason',
        'price_30s', 'price_60s', 'price_120s',
        'pnl_30s_pct', 'pnl_60s_pct', 'pnl_120s_pct',
    ]

    def __init__(self, csv_dir: str = "logs", influx_config: Optional[dict] = None):
        self._csv_dir = csv_dir
        os.makedirs(csv_dir, exist_ok=True)

        # One CSV per trading day
        today = datetime.now().strftime("%Y-%m-%d")
        self._csv_path = os.path.join(csv_dir, f"signals_{today}.csv")
        self._csv_file = None
        self._csv_writer = None
        self._init_csv()

        # Buffer for signals that fired (need outcome backfill)
        self._fired_buffer = []  # list of SignalEvent
        self.outcome_tracker = OutcomeTracker()

        # Optional InfluxDB
        self._influx_write = None
        if influx_config:
            self._init_influx(influx_config)

    def _init_csv(self):
        file_exists = os.path.exists(self._csv_path) and os.path.getsize(self._csv_path) > 0
        self._csv_file = open(self._csv_path, 'a', newline='', buffering=1)
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_HEADERS)
        if not file_exists:
            self._csv_writer.writeheader()
        logger.info(f"Signal CSV: {self._csv_path}")

    def _init_influx(self, config: dict):
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS
            client = InfluxDBClient(
                url=config['url'],
                token=config['token'],
                org=config['org'],
            )
            # SYNCHRONOUS is the right default for low-volume signal writes;
            # async batching would risk losing the tail on crash.
            self._influx_write = client.write_api(write_options=SYNCHRONOUS)
            self._influx_bucket = config['bucket']
            self._influx_org = config['org']
            logger.info(f"InfluxDB connected: {config['url']}")
        except ImportError:
            logger.warning("influxdb_client not installed. pip install influxdb-client")
        except Exception as e:
            logger.warning(f"InfluxDB init failed: {e}")

    def log_signal(self, event: SignalEvent):
        """Log a signal evaluation (CALL, PUT, or NONE)."""
        # Write NONE signals immediately (no outcome tracking needed)
        if event.signal == "NONE":
            self._write_csv(event)
            return

        # For fired signals, register for outcome tracking and buffer
        self.outcome_tracker.register(event)
        self._fired_buffer.append(event)
        self._write_csv(event)  # Write immediately, will append outcome later
        self._write_influx(event)

    def update_outcomes(self, ticker: str, current_price: float, now: float):
        """Call every tick to check if any pending outcomes are ready."""
        self.outcome_tracker.check_outcomes(ticker, current_price, now)

    def iter_pending_outcomes(self):
        """
        Iterate fired signals awaiting (or now holding) outcome backfill.
        Public API so callers don't reach into _fired_buffer.
        """
        return tuple(self._fired_buffer)

    def flush_completed_outcomes(self):
        """
        Periodically rewrite fired signals that now have full outcomes.
        Call this less frequently (e.g., every 60s) to avoid I/O churn.
        """
        completed = []
        still_pending = []

        for event in self._fired_buffer:
            if event.pnl_120s_pct is not None:
                completed.append(event)
            else:
                still_pending.append(event)

        self._fired_buffer = still_pending

        if completed:
            # Append completed outcomes to a separate outcome file
            today = datetime.now().strftime("%Y-%m-%d")
            outcome_path = os.path.join(self._csv_dir, f"outcomes_{today}.csv")
            file_exists = os.path.exists(outcome_path) and os.path.getsize(outcome_path) > 0

            with open(outcome_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                if not file_exists:
                    writer.writeheader()
                for event in completed:
                    writer.writerow(asdict(event))

            logger.info(f"Flushed {len(completed)} completed outcomes")

            # Also push to InfluxDB
            for event in completed:
                self._write_influx_outcome(event)

    def _write_csv(self, event: SignalEvent):
        try:
            self._csv_writer.writerow(asdict(event))
        except Exception as e:
            logger.error(f"CSV write error: {e}")

    def _write_influx(self, event: SignalEvent):
        if not self._influx_write:
            return
        try:
            from influxdb_client import Point
            p = (Point("signal")
                 .tag("ticker", event.ticker)
                 .tag("signal", event.signal)
                 .tag("reason", event.reason)
                 .field("raw_price", event.raw_price)
                 .field("smooth_price", event.smooth_price)
                 .field("velocity", event.velocity)
                 .field("acceleration", event.acceleration)
                 .field("jerk", event.jerk)
                 .field("obi", event.obi)
                 .field("atr_pct", event.atr_pct)
                 .field("spread", event.spread)
                 .time(int(event.timestamp * 1e9)))
            self._influx_write.write(bucket=self._influx_bucket, record=p)
        except Exception as e:
            logger.error(f"InfluxDB write error: {e}")

    def _write_influx_outcome(self, event: SignalEvent):
        if not self._influx_write:
            return
        try:
            from influxdb_client import Point
            p = (Point("outcome")
                 .tag("ticker", event.ticker)
                 .tag("signal", event.signal)
                 .field("entry_price", event.smooth_price)
                 .field("pnl_30s", event.pnl_30s_pct or 0.0)
                 .field("pnl_60s", event.pnl_60s_pct or 0.0)
                 .field("pnl_120s", event.pnl_120s_pct or 0.0)
                 .time(int(event.timestamp * 1e9)))
            self._influx_write.write(bucket=self._influx_bucket, record=p)
        except Exception as e:
            logger.error(f"InfluxDB outcome write error: {e}")

    def close(self):
        """Flush and close all resources."""
        self.flush_completed_outcomes()
        if self._csv_file:
            self._csv_file.close()


# ====================================================================
# Quick analysis helper — run on the outcomes CSV after collecting data
# ====================================================================
def analyze_outcomes(csv_path: str):
    """Print signal quality stats from an outcomes CSV."""
    import statistics

    calls = {'30': [], '60': [], '120': []}
    puts = {'30': [], '60': [], '120': []}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bucket = calls if row['signal'] == 'CALL' else puts
            for s in ['30', '60', '120']:
                val = row.get(f'pnl_{s}s_pct')
                if val and val != '':
                    bucket[s].append(float(val))

    print(f"\n{'='*60}")
    print(f"SIGNAL QUALITY REPORT: {csv_path}")
    print(f"{'='*60}")

    for label, data in [("CALL", calls), ("PUT", puts)]:
        print(f"\n--- {label} signals ---")
        for horizon in ['30', '60', '120']:
            vals = data[horizon]
            if not vals:
                print(f"  {horizon}s: no data")
                continue
            wins = sum(1 for v in vals if v > 0)
            avg = statistics.mean(vals)
            med = statistics.median(vals)
            total = len(vals)
            print(f"  {horizon}s: n={total}, win_rate={wins/total*100:.1f}%, "
                  f"avg={avg:+.4f}%, median={med:+.4f}%")

    total_signals = len(calls['120']) + len(puts['120'])
    if total_signals > 0:
        all_pnl = calls['120'] + puts['120']
        overall_avg = statistics.mean(all_pnl)
        overall_wr = sum(1 for v in all_pnl if v > 0) / len(all_pnl) * 100
        print(f"\nOVERALL (120s): n={total_signals}, win_rate={overall_wr:.1f}%, avg={overall_avg:+.4f}%")
        # Rough edge estimate: if avg > spread cost, you have edge
        print(f"Estimated edge after ~0.10% round-trip spread: {overall_avg - 0.10:+.4f}%")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_outcomes(sys.argv[1])
    else:
        print("Usage: python signal_logger.py <outcomes_YYYY-MM-DD.csv>")