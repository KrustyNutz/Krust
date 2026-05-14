from __future__ import annotations

# ====================================================================
# ANALYZE_COMPONENTS.PY | Per-voter contribution analysis
#
# Reads outcomes_<date>.csv files written by signal_logger and scores
# each of the 6 voters (momentum, trade_flow, vwap_dev, rsi, obi, mtf)
# by:
#   - how often it agreed / abstained / disagreed with the fired signal
#   - win rate in each state
#   - avg PnL in each state
#   - LIFT = WR(agree) - WR(disagree) -- higher = more discriminating
#
# Also reports overall stats, per-regime stats, and the news_agrees/
# news_disagrees gate's contribution.
#
# Usage:
#   python analyze_components.py
#       (auto-discovers logs/outcomes_*.csv and logs/crypto/outcomes_*.csv)
#
#   python analyze_components.py logs/outcomes_2026-05-12.csv [...]
#       (explicit file list)
#
#   python analyze_components.py --horizon 60
#       (use 60s PnL as win/loss criterion; default 120s)
#
#   python analyze_components.py --regime TRENDING
#       (filter to one regime)
#
#   python analyze_components.py --ticker SPY
#       (filter to one ticker)
# ====================================================================

import argparse
import csv
import glob
import os
import re
import sys
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

VOTERS = ['momentum', 'trade_flow', 'vwap_dev', 'rsi', 'obi', 'mtf']
HORIZONS = [30, 60, 120]

REASON_REGIME_RE = re.compile(r'regime=([A-Z_]+)')
REASON_SCORE_RE = re.compile(r'score=(-?\d+\.\d+)/')
REASON_VOTE_RE = re.compile(r'\b(' + '|'.join(VOTERS) + r')=([+-]\d+)')


def parse_reason(reason: str) -> Tuple[str, float, Dict[str, int], Dict[str, bool]]:
    """
    Extract structured fields from the reason string.

    Returns:
      regime: e.g. "TRENDING" / "MEAN_REVERT" / "VOLATILE" / ""
      score:  the adjusted_score float (raw composite score)
      votes:  {voter_name: -1|+1} for non-zero voters only (absent = 0)
      flags:  {'news_agrees': bool, 'news_disagrees': bool, 'breakout_boost': bool}
    """
    regime = ''
    score = 0.0
    votes: Dict[str, int] = {}
    flags = {'news_agrees': False, 'news_disagrees': False, 'breakout_boost': False}

    if not reason:
        return regime, score, votes, flags

    m = REASON_REGIME_RE.search(reason)
    if m:
        regime = m.group(1)

    m = REASON_SCORE_RE.search(reason)
    if m:
        try:
            score = float(m.group(1))
        except ValueError:
            pass

    for vname, vstr in REASON_VOTE_RE.findall(reason):
        try:
            votes[vname] = int(vstr)
        except ValueError:
            pass

    if 'news_agrees' in reason:
        flags['news_agrees'] = True
    if 'news_disagrees' in reason:
        flags['news_disagrees'] = True
    if 'BOOST:breakout' in reason:
        flags['breakout_boost'] = True

    return regime, score, votes, flags


def signal_dir(signal: str) -> int:
    if signal == 'CALL':
        return 1
    if signal == 'PUT':
        return -1
    return 0


def agreement(vote: int, side: int) -> str:
    """Return 'agree' / 'disagree' / 'abstain' for a voter vs signal side."""
    if vote == 0:
        return 'abstain'
    if vote * side > 0:
        return 'agree'
    return 'disagree'


def safe_float(s: Optional[str]) -> Optional[float]:
    if s is None or s == '':
        return None
    try:
        return float(s)
    except ValueError:
        return None


def find_similar(name: str) -> List[str]:
    """If `name` doesn't exist, search the tree for a file with that basename."""
    base = os.path.basename(name)
    hits: List[str] = []
    for root, _dirs, files in os.walk('.'):
        if base in files:
            hits.append(os.path.join(root, base))
    return hits


def resolve_paths(args_paths: List[str]) -> List[str]:
    """Resolve user-supplied paths; if a path is missing, search for it."""
    resolved: List[str] = []
    for p in args_paths:
        if os.path.exists(p):
            resolved.append(p)
            continue
        candidates = find_similar(p)
        if candidates:
            print(f"note: '{p}' not found at that path; using:", file=sys.stderr)
            for c in candidates:
                print(f"        {c}", file=sys.stderr)
            resolved.extend(candidates)
        else:
            print(f"warn: '{p}' not found and no file with that name exists "
                  f"under {os.getcwd()}", file=sys.stderr)
    return resolved


def load_rows(paths: List[str], debug: bool = False) -> List[dict]:
    """Read all outcomes rows from the given CSV paths, with parsed reasons."""
    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"warn: missing {path}", file=sys.stderr)
            continue
        per_file_total = 0
        per_file_fired = 0
        per_file_with_pnl = 0
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for raw in reader:
                per_file_total += 1
                signal = (raw.get('signal') or '').strip()
                if signal not in ('CALL', 'PUT'):
                    continue
                per_file_fired += 1
                side = signal_dir(signal)
                regime, score, votes, flags = parse_reason(raw.get('reason', ''))
                pnl_30 = safe_float(raw.get('pnl_30s_pct'))
                pnl_60 = safe_float(raw.get('pnl_60s_pct'))
                pnl_120 = safe_float(raw.get('pnl_120s_pct'))
                if pnl_120 is not None:
                    per_file_with_pnl += 1
                row = {
                    'source': os.path.basename(path),
                    'ticker': raw.get('ticker', ''),
                    'signal': signal,
                    'side': side,
                    'regime': regime,
                    'score': score,
                    'reason': raw.get('reason', ''),
                    'votes': votes,
                    'flags': flags,
                    'pnl_30s': pnl_30,
                    'pnl_60s': pnl_60,
                    'pnl_120s': pnl_120,
                }
                rows.append(row)
        if debug:
            print(f"  {path}: total_rows={per_file_total}  "
                  f"fired={per_file_fired}  with_120s_pnl={per_file_with_pnl}",
                  file=sys.stderr)
        elif per_file_fired > 0 and per_file_with_pnl == 0:
            print(f"warn: {path} has {per_file_fired} fired signals but NONE have "
                  f"completed PnL columns. The bot may have been killed before "
                  f"flush_completed_outcomes() ran. Try a different date's file.",
                  file=sys.stderr)
    return rows


def summarize(pnls: List[float]) -> Tuple[int, float, float, float]:
    """Return (n, win_rate_pct, avg_pnl_pct, median_pnl_pct)."""
    if not pnls:
        return 0, 0.0, 0.0, 0.0
    wins = sum(1 for v in pnls if v > 0)
    wr = wins / len(pnls) * 100.0
    avg = statistics.mean(pnls)
    med = statistics.median(pnls)
    return len(pnls), wr, avg, med


def print_section(title: str):
    print()
    print('=' * 68)
    print(title)
    print('=' * 68)


def print_overall(rows: List[dict], horizon: int):
    key = f'pnl_{horizon}s'
    pnls = [r[key] for r in rows if r[key] is not None]
    n, wr, avg, med = summarize(pnls)
    print(f"\nFired signals with completed {horizon}s outcome: n={n}")
    if not n:
        return
    print(f"  win rate : {wr:5.1f}%")
    print(f"  avg PnL  : {avg:+.4f}%")
    print(f"  median   : {med:+.4f}%")

    # Also show 30 / 60 / 120 side-by-side
    print(f"\n  horizon │     n  │ win % │  avg %  │ median %")
    print(f"  ────────┼────────┼───────┼─────────┼─────────")
    for h in HORIZONS:
        k = f'pnl_{h}s'
        hp = [r[k] for r in rows if r[k] is not None]
        hn, hw, ha, hm = summarize(hp)
        print(f"  {h:>3}s    │ {hn:>5}  │ {hw:5.1f} │ {ha:+7.4f} │ {hm:+7.4f}")


def print_signal_split(rows: List[dict], horizon: int):
    key = f'pnl_{horizon}s'
    for side_label in ('CALL', 'PUT'):
        sub = [r for r in rows if r['signal'] == side_label and r[key] is not None]
        pnls = [r[key] for r in sub]
        n, wr, avg, med = summarize(pnls)
        if n == 0:
            print(f"\n  {side_label}: no data")
            continue
        print(f"\n  {side_label}: n={n}  win={wr:5.1f}%  avg={avg:+.4f}%  median={med:+.4f}%")


def print_per_voter(rows: List[dict], horizon: int):
    """For each voter, show agree/abstain/disagree counts, win rates, lift."""
    key = f'pnl_{horizon}s'
    print(f"\nPer-voter contribution ({horizon}s horizon, baseline = abstain):")
    print()
    print(f"  voter        │ state    │     n │  freq │  win % │  avg %  │ lift vs disagree")
    print(f"  ─────────────┼──────────┼───────┼───────┼────────┼─────────┼──────────────────")

    total = sum(1 for r in rows if r[key] is not None)

    summary_table: List[Tuple[str, float]] = []  # (voter, lift_pp)

    for voter in VOTERS:
        buckets = {'agree': [], 'abstain': [], 'disagree': []}
        for r in rows:
            if r[key] is None:
                continue
            v = r['votes'].get(voter, 0)
            buckets[agreement(v, r['side'])].append(r[key])

        wr_by_state: Dict[str, float] = {}
        first = True
        for state in ('agree', 'abstain', 'disagree'):
            vals = buckets[state]
            n, wr, avg, _ = summarize(vals)
            freq = (n / total * 100.0) if total else 0.0
            wr_by_state[state] = wr if n else float('nan')
            label = voter if first else ''
            first = False
            n_disp = f"{n:>5}"
            if n == 0:
                print(f"  {label:<12} │ {state:<8} │ {n_disp} │   -   │    -   │    -    │       -")
            else:
                print(f"  {label:<12} │ {state:<8} │ {n_disp} │ {freq:5.1f} │ {wr:6.1f} │ {avg:+7.4f} │")

        # Lift = WR(agree) - WR(disagree)  (pp).  Higher = more discriminating.
        wa = wr_by_state.get('agree', float('nan'))
        wd = wr_by_state.get('disagree', float('nan'))
        if wa == wa and wd == wd:  # both not nan
            lift = wa - wd
            summary_table.append((voter, lift))
            print(f"  {'':<12} │ {'':<8} │       │       │        │         │   {lift:+6.1f} pp")
        else:
            summary_table.append((voter, float('nan')))
        print(f"  ─────────────┼──────────┼───────┼───────┼────────┼─────────┼──────────────────")

    # Ranked summary
    print()
    print("Voter ranking by lift (agree win% minus disagree win%):")
    ranked = sorted(
        summary_table,
        key=lambda x: (float('-inf') if x[1] != x[1] else x[1]),
        reverse=True,
    )
    for i, (v, lift) in enumerate(ranked, 1):
        if lift != lift:
            print(f"  {i}. {v:<12}   insufficient data")
        else:
            tag = ''
            if lift > 15:
                tag = '   <-- strong contributor'
            elif lift > 5:
                tag = '   <-- positive'
            elif lift < -5:
                tag = '   <-- INVERTED (votes against winners)'
            elif abs(lift) <= 5:
                tag = '   <-- noise / weak'
            print(f"  {i}. {v:<12}   {lift:+6.1f} pp{tag}")


def print_per_regime(rows: List[dict], horizon: int):
    key = f'pnl_{horizon}s'
    by_regime: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r[key] is None or not r['regime']:
            continue
        by_regime[r['regime']].append(r[key])

    if not by_regime:
        return
    print(f"\nPer-regime ({horizon}s):")
    print(f"  regime       │     n │ win % │  avg %  │ median %")
    print(f"  ─────────────┼───────┼───────┼─────────┼─────────")
    for regime in sorted(by_regime.keys()):
        n, wr, avg, med = summarize(by_regime[regime])
        print(f"  {regime:<12} │ {n:>5} │ {wr:5.1f} │ {avg:+7.4f} │ {med:+7.4f}")


def print_per_ticker(rows: List[dict], horizon: int):
    key = f'pnl_{horizon}s'
    by_ticker: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r[key] is None:
            continue
        by_ticker[r['ticker']].append(r[key])

    if len(by_ticker) <= 1:
        return
    print(f"\nPer-ticker ({horizon}s):")
    print(f"  ticker  │     n │ win % │  avg %")
    print(f"  ────────┼───────┼───────┼────────")
    for tkr in sorted(by_ticker.keys()):
        n, wr, avg, _ = summarize(by_ticker[tkr])
        print(f"  {tkr:<7} │ {n:>5} │ {wr:5.1f} │ {avg:+7.4f}")


def print_news_filter(rows: List[dict], horizon: int):
    """Audit the news_agrees / news_disagrees gate."""
    key = f'pnl_{horizon}s'
    agrees = [r[key] for r in rows if r[key] is not None and r['flags']['news_agrees']]
    disagrees_kept = [r[key] for r in rows
                      if r[key] is not None and r['flags']['news_disagrees']]
    neutral = [r[key] for r in rows
               if r[key] is not None
               and not r['flags']['news_agrees']
               and not r['flags']['news_disagrees']]

    if not (agrees or disagrees_kept or neutral):
        return

    print(f"\nNews filter audit ({horizon}s; disagrees-kept = score>=7.0 override):")
    print(f"  bucket             │     n │ win % │  avg %")
    print(f"  ───────────────────┼───────┼───────┼────────")
    for label, vals in (
        ('news_agrees',         agrees),
        ('neutral (no news)',   neutral),
        ('news_disagrees kept', disagrees_kept),
    ):
        n, wr, avg, _ = summarize(vals)
        if n == 0:
            print(f"  {label:<18} │     0 │   -   │    -")
        else:
            print(f"  {label:<18} │ {n:>5} │ {wr:5.1f} │ {avg:+7.4f}")


def print_score_buckets(rows: List[dict], horizon: int):
    """How does win-rate evolve as the raw composite score grows?"""
    key = f'pnl_{horizon}s'
    buckets: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r[key] is None:
            continue
        s = abs(r['score'])
        if s < 5:
            b = '<5.0   (sub-threshold)'
        elif s < 6:
            b = '5.0-6.0'
        elif s < 7:
            b = '6.0-7.0'
        elif s < 8:
            b = '7.0-8.0'
        else:
            b = '>=8.0'
        buckets[b].append(r[key])

    if not buckets:
        return
    print(f"\nScore-tier win rate ({horizon}s):")
    print(f"  |score|              │     n │ win % │  avg %")
    print(f"  ─────────────────────┼───────┼───────┼────────")
    order = ['<5.0   (sub-threshold)', '5.0-6.0', '6.0-7.0', '7.0-8.0', '>=8.0']
    for b in order:
        if b not in buckets:
            continue
        n, wr, avg, _ = summarize(buckets[b])
        print(f"  {b:<20} │ {n:>5} │ {wr:5.1f} │ {avg:+7.4f}")


def discover_paths(args_paths: List[str]) -> List[str]:
    if args_paths:
        return resolve_paths(args_paths)
    # Recursive: covers logs/, logs/nexus/, logs/crypto/, or any nested layout.
    paths = sorted(glob.glob('logs/**/outcomes_*.csv', recursive=True))
    paths += sorted(glob.glob('logs/outcomes_*.csv'))
    # de-dupe while preserving order
    seen = set()
    unique = []
    for p in paths:
        np = os.path.normpath(p)
        if np not in seen:
            seen.add(np)
            unique.append(np)
    return unique


def main():
    ap = argparse.ArgumentParser(
        description='NEXUS per-component contribution analysis')
    ap.add_argument('paths', nargs='*',
                    help='outcomes CSV files (default: auto-discover logs/)')
    ap.add_argument('--horizon', type=int, default=120, choices=HORIZONS,
                    help='PnL horizon for win/loss criterion (default 120)')
    ap.add_argument('--regime', default=None,
                    help='Filter to one regime (TRENDING / MEAN_REVERT / VOLATILE)')
    ap.add_argument('--ticker', default=None,
                    help='Filter to one ticker symbol')
    ap.add_argument('--debug', action='store_true',
                    help='Print per-file row counts and parse stats')
    args = ap.parse_args()

    paths = discover_paths(args.paths)
    if not paths:
        print("No outcomes CSVs found. Searched:")
        print("  ./logs/**/outcomes_*.csv  (recursive)")
        print(f"Current dir: {os.getcwd()}")
        print("Pass an explicit path, or run from the directory containing logs/.")
        sys.exit(1)

    if args.debug:
        print("Discovered files:", file=sys.stderr)
    rows = load_rows(paths, debug=args.debug)
    if args.regime:
        rows = [r for r in rows if r['regime'] == args.regime]
    if args.ticker:
        rows = [r for r in rows if r['ticker'] == args.ticker]

    print_section(
        f"NEXUS COMPONENT CONTRIBUTION  |  horizon={args.horizon}s")
    print(f"Sources ({len(paths)}):")
    for p in paths:
        print(f"  {p}")
    if args.regime:
        print(f"Filter: regime={args.regime}")
    if args.ticker:
        print(f"Filter: ticker={args.ticker}")

    if not rows:
        print("\nNo fired signals (CALL/PUT) with completed outcomes in the inputs.")
        sys.exit(0)

    print_overall(rows, args.horizon)
    print_signal_split(rows, args.horizon)
    print_section("PER-VOTER CONTRIBUTION")
    print_per_voter(rows, args.horizon)
    print_section("CONTEXT BREAKDOWNS")
    print_per_regime(rows, args.horizon)
    print_per_ticker(rows, args.horizon)
    print_news_filter(rows, args.horizon)
    print_score_buckets(rows, args.horizon)
    print()


if __name__ == '__main__':
    main()
