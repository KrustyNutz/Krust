from __future__ import annotations

# ====================================================================
# BACKTEST.PY | NEXUS Crypto signal expectancy analyzer
#
# Answers the only question that matters for profit:
#   "Does this strategy have edge AFTER spread and fees?"
#
# Reads outcomes_YYYY-MM-DD.csv produced by signal_logger.py and reports:
#   - Net expectancy per trade (overall, by signal, ticker, regime,
#     and |raw_score| confidence bucket)
#   - Threshold sensitivity sweep (what happens if you raise the bar?)
#   - Compounded equity curve under a realistic cost model
#   - A blunt verdict: tradable, marginal, or do-not-trade
#
# Usage:
#   python backtest.py logs/crypto/outcomes_2026-05-16.csv
#   python backtest.py logs/crypto/outcomes_*.csv --fee 0.25 --size 0.10
# ====================================================================

import csv
import sys
import argparse
import re
import statistics
from collections import defaultdict
from typing import Callable, Iterable

# Reason format from compute_composite_signal:
#   "regime=TRENDING | score=5.20/5.0 | momentum=+1 | trade_flow=+1 | ..."
_REGIME_RX = re.compile(r"regime=(\w+)")
_SCORE_RX = re.compile(r"score=([+-]?\d+\.?\d*)")


# ====================================================================
# LOADING
# ====================================================================
def load_outcomes(paths: list[str]) -> list[dict]:
    """Load fired signals with completed 120s outcomes from one or more CSVs."""
    rows = []
    for path in paths:
        try:
            f = open(path)
        except OSError as e:
            print(f"  skip {path}: {e}", file=sys.stderr)
            continue
        with f:
            for row in csv.DictReader(f):
                if row.get('signal') not in ('CALL', 'PUT'):
                    continue
                if not row.get('pnl_120s_pct'):
                    continue
                for k in ('pnl_30s_pct', 'pnl_60s_pct', 'pnl_120s_pct',
                          'spread', 'raw_price', 'timestamp'):
                    v = row.get(k)
                    try:
                        row[k] = float(v) if v not in (None, '') else None
                    except (ValueError, TypeError):
                        row[k] = None
                reason = row.get('reason', '') or ''
                m = _REGIME_RX.search(reason)
                row['regime'] = m.group(1) if m else 'UNKNOWN'
                m = _SCORE_RX.search(reason)
                row['raw_score'] = float(m.group(1)) if m else 0.0
                rows.append(row)
    return rows


# ====================================================================
# COST MODEL
# ====================================================================
def apply_cost(gross_pct: float, spread_frac: float, fee_pct: float) -> float:
    """
    Net PnL after realistic round-trip cost.

    Spread is crossed on both entry and exit (assume taker fills both
    sides), so the full spread is paid round-trip. Fee is per-side, so
    round-trip = 2 * fee_pct.
    """
    spread_cost_pct = (spread_frac or 0) * 100.0
    fee_cost_pct = 2.0 * fee_pct
    return gross_pct - spread_cost_pct - fee_cost_pct


# ====================================================================
# DESCRIPTIVE STATS
# ====================================================================
def describe(values: list[float], label: str) -> str:
    if not values:
        return f"  {label:14}  no data"
    n = len(values)
    win_rate = sum(1 for v in values if v > 0) / n * 100
    avg = statistics.mean(values)
    med = statistics.median(values)
    total = sum(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    sharpe = avg / sd if sd > 0 else 0.0
    return (f"  {label:14}  n={n:4d}  WR={win_rate:5.1f}%  "
            f"avg={avg:+7.4f}%  med={med:+7.4f}%  "
            f"Σ={total:+8.3f}%  edge/σ={sharpe:+.3f}")


def by_bucket(rows: list[dict], key_fn: Callable[[dict], str],
              horizon: str, fee: float) -> dict[str, list[float]]:
    pnl_key = f'pnl_{horizon}s_pct'
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        gross = row.get(pnl_key)
        if gross is None:
            continue
        net = apply_cost(gross, row.get('spread') or 0, fee)
        buckets[key_fn(row)].append(net)
    return buckets


# ====================================================================
# PARAMETER SWEEPS
# ====================================================================
def threshold_sweep(rows: list[dict], horizon: str, fee: float,
                    thresholds: Iterable[float]) -> None:
    pnl_key = f'pnl_{horizon}s_pct'
    print(f"\n--- Threshold sensitivity (horizon={horizon}s) ---")
    print(f"  {'thresh':>7}  {'n':>5}  {'WR':>6}  {'avg':>10}  {'Σ_net':>10}")
    for t in thresholds:
        nets = []
        for row in rows:
            if abs(row.get('raw_score', 0)) < t:
                continue
            gross = row.get(pnl_key)
            if gross is None:
                continue
            nets.append(apply_cost(gross, row.get('spread') or 0, fee))
        if not nets:
            print(f"  {t:7.2f}  no signals at or above this threshold")
            continue
        wr = sum(1 for v in nets if v > 0) / len(nets) * 100
        avg = statistics.mean(nets)
        print(f"  {t:7.2f}  {len(nets):5d}  {wr:5.1f}%  "
              f"{avg:+10.4f}%  {sum(nets):+10.3f}%")


# ====================================================================
# EQUITY CURVE
# ====================================================================
def equity_curve(rows: list[dict], horizon: str, fee: float,
                 size_frac: float, cooldown_sec: int) -> dict:
    """
    Walk through trades chronologically, compounding.

    cooldown_sec mirrors SIGNAL_COOLDOWN_SEC in the live bot — without it,
    we'd count trades the bot would never have actually taken (overlapping
    signals on the same ticker).
    """
    pnl_key = f'pnl_{horizon}s_pct'
    by_ts = sorted((r for r in rows if r.get('timestamp') is not None),
                   key=lambda r: r['timestamp'])
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    returns = []
    last_trade_ts: dict[str, float] = {}

    for row in by_ts:
        ts = row['timestamp']
        ticker = row.get('ticker', '')
        last = last_trade_ts.get(ticker, 0)
        if ts - last < cooldown_sec:
            continue
        gross = row.get(pnl_key)
        if gross is None:
            continue
        net_pct = apply_cost(gross, row.get('spread') or 0, fee)
        ret = (net_pct / 100.0) * size_frac
        equity *= (1.0 + ret)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0)
        returns.append(ret)
        last_trade_ts[ticker] = ts

    final_return = (equity - 1.0) * 100.0
    sharpe = (statistics.mean(returns) / statistics.stdev(returns)
              if len(returns) > 1 and statistics.stdev(returns) > 0 else 0.0)
    return {
        'n_trades': len(returns),
        'final_equity': equity,
        'final_return_pct': final_return,
        'max_drawdown_pct': max_dd * 100,
        'per_trade_sharpe': sharpe,
    }


# ====================================================================
# REPORT
# ====================================================================
def report(rows: list[dict], horizon: str, fee: float,
           size_frac: float, cooldown_sec: int) -> None:
    print(f"\n{'='*72}")
    print(f"  NEXUS CRYPTO BACKTEST")
    print(f"  horizon={horizon}s  fee={fee:.3f}%/side  size={size_frac*100:.1f}%"
          f"  cooldown={cooldown_sec}s")
    print(f"{'='*72}")

    pnl_key = f'pnl_{horizon}s_pct'
    all_nets = [apply_cost(r[pnl_key], r.get('spread') or 0, fee)
                for r in rows if r.get(pnl_key) is not None]

    print("\n--- Overall (net of cost) ---")
    print(describe(all_nets, "ALL"))

    print("\n--- By direction ---")
    by_sig = by_bucket(rows, lambda r: r['signal'], horizon, fee)
    for sig in ('CALL', 'PUT'):
        print(describe(by_sig.get(sig, []), sig))

    print("\n--- By ticker ---")
    by_tkr = by_bucket(rows, lambda r: r.get('ticker', '?'), horizon, fee)
    for tkr in sorted(by_tkr):
        print(describe(by_tkr[tkr], tkr))

    print("\n--- By regime ---")
    by_reg = by_bucket(rows, lambda r: r.get('regime', '?'), horizon, fee)
    for reg in sorted(by_reg):
        print(describe(by_reg[reg], reg))

    print("\n--- By |raw_score| bucket ---")
    def score_bucket(r: dict) -> str:
        s = abs(r.get('raw_score', 0))
        if s < 4:  return "<4"
        if s < 6:  return "4-6"
        if s < 8:  return "6-8"
        return "8+"
    by_score = by_bucket(rows, score_bucket, horizon, fee)
    for b in ("<4", "4-6", "6-8", "8+"):
        print(describe(by_score.get(b, []), b))

    threshold_sweep(rows, horizon, fee,
                    thresholds=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    ec = equity_curve(rows, horizon, fee, size_frac, cooldown_sec)
    print(f"\n--- Equity curve (compounded, size={size_frac*100:.1f}% per trade) ---")
    print(f"  Trades taken:      {ec['n_trades']}  "
          f"({len(all_nets) - ec['n_trades']} skipped by cooldown)")
    print(f"  Final equity:      ${ec['final_equity']:.5f} "
          f"(starting at $1.000)")
    print(f"  Total return:      {ec['final_return_pct']:+.3f}%")
    print(f"  Max drawdown:      {ec['max_drawdown_pct']:.3f}%")
    print(f"  Per-trade Sharpe:  {ec['per_trade_sharpe']:+.3f}")

    # ----- Verdict -----
    print(f"\n{'='*72}")
    if not all_nets:
        print("  VERDICT: No completed outcomes to evaluate.")
        print(f"{'='*72}\n")
        return

    avg_net = statistics.mean(all_nets)
    n = len(all_nets)
    if n < 30:
        verdict = "INSUFFICIENT_DATA"
        msg = (f"Only {n} signals — too few to draw conclusions. "
               f"Need ≥30 per regime for any confidence, ≥100+ for stability.")
    elif avg_net <= 0:
        verdict = "NO_EDGE"
        msg = (f"Net expectancy is {avg_net:+.4f}%/trade after costs. "
               f"Do NOT trade live. Raise threshold, prune losing regimes, "
               f"or rethink the signal.")
    elif avg_net < 0.02:
        verdict = "MARGINAL"
        msg = (f"Edge is {avg_net:+.4f}%/trade — positive but thin. "
               f"Single slippage event eats it. Tighten before risking capital.")
    else:
        verdict = "TRADABLE"
        msg = (f"Edge is {avg_net:+.4f}%/trade (n={n}). "
               f"Sharpe {ec['per_trade_sharpe']:+.3f}, max DD "
               f"{ec['max_drawdown_pct']:.2f}%. Worth paper-trading at scale.")
    print(f"  VERDICT: {verdict}")
    print(f"  {msg}")
    print(f"{'='*72}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest NEXUS Crypto signals for net expectancy."
    )
    parser.add_argument("paths", nargs="+",
                        help="outcomes_YYYY-MM-DD.csv path(s)")
    parser.add_argument("--horizon", default="120",
                        choices=["30", "60", "120"],
                        help="Outcome horizon in seconds (default: 120)")
    parser.add_argument("--fee", type=float, default=0.15,
                        help="Per-side fee in percent (default: 0.15)")
    parser.add_argument("--size", type=float, default=0.10,
                        help="Position size as fraction of equity "
                             "(default: 0.10 = 10%%)")
    parser.add_argument("--cooldown", type=int, default=180,
                        help="Per-ticker cooldown in seconds, mirrors "
                             "SIGNAL_COOLDOWN_SEC (default: 180)")
    args = parser.parse_args()

    rows = load_outcomes(args.paths)
    print(f"Loaded {len(rows)} fired-signal rows with completed outcomes")
    if not rows:
        print("\nNo data. Run the bot in observe mode first:")
        print("  python nexuscrypto.py --headless")
        print("Wait at least 10 minutes after the first signal so the 120s")
        print("outcomes get backfilled, then re-run this script against")
        print("logs/crypto/outcomes_<today>.csv")
        sys.exit(0)
    report(rows, args.horizon, args.fee, args.size, args.cooldown)


if __name__ == "__main__":
    main()
