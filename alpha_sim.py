from __future__ import annotations

# ====================================================================
# ALPHA_SIM.PY | Monte Carlo simulator for NEXUS ALPHA portfolio
#
# Models each of the four strategies with parameters calibrated to
# academic/industry literature, then simulates 1000+ years of returns
# under realistic cost, correlation, and tail-event assumptions.
#
# Outputs distribution of outcomes so the user knows what to expect
# BEFORE committing capital.
# ====================================================================

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================
# CALIBRATION CONSTANTS (from cited academic/industry research)
# ============================================================
# All Sharpe and return numbers are GROSS of strategy-level costs.
# Net numbers fall out of the simulation after applying per-trade cost.

STRATEGY_PARAMS = {
    # Bessembinder 2018, Lou et al 2019 ("A Tug of War: Overnight vs Intraday")
    'overnight': {
        'annual_return_mean': 0.07,
        'annual_vol': 0.10,
        'trades_per_year': 252,
        'cost_per_trade_bps': 0.4,   # SPY MOC/MOO ~0.4bp realistic; Alpaca $0 comm
        'fat_tail_prob': 0.01,
        'fat_tail_loss_mult': 4.0,
    },
    # Bollerslev/Tauchen/Zhou 2009, Cboe Vol Premia Index, multiple replications
    'vrp': {
        'annual_return_mean': 0.09,
        'annual_vol': 0.13,
        'trades_per_year': 52,       # weekly
        'cost_per_trade_bps': 3.0,   # combo spread + slip on SPY weekly options
        'fat_tail_prob': 0.05,       # 5% of weeks have vol spike
        'fat_tail_loss_mult': 6.0,
    },
    # Connors "Short Term Trading Strategies That Work" 2008, replicated to 2024
    'rsi': {
        'annual_return_mean': 0.06,
        'annual_vol': 0.08,
        'trades_per_year': 40,
        'cost_per_trade_bps': 0.8,   # SPY share trade, very liquid
        'fat_tail_prob': 0.02,
        'fat_tail_loss_mult': 3.0,
    },
    # Jegadeesh & Titman 1993, Asness et al "Value and Momentum Everywhere" 2013
    'sector': {
        'annual_return_mean': 0.08,
        'annual_vol': 0.14,
        'trades_per_year': 24,
        'cost_per_trade_bps': 1.5,   # SPDR sector ETFs, liquid
        'fat_tail_prob': 0.03,
        'fat_tail_loss_mult': 4.0,
    },
}

# Cross-strategy correlation matrix.
# Overnight and VRP both load on vol regime (high correlation in shocks).
# RSI and sector are more independent.
CORRELATION = {
    ('overnight', 'overnight'): 1.00,
    ('overnight', 'vrp'):       0.30,
    ('overnight', 'rsi'):       0.40,
    ('overnight', 'sector'):    0.50,
    ('vrp',       'vrp'):       1.00,
    ('vrp',       'rsi'):       0.20,
    ('vrp',       'sector'):    0.25,
    ('rsi',       'rsi'):       1.00,
    ('rsi',       'sector'):    0.35,
    ('sector',    'sector'):    1.00,
}

WEIGHTS = {
    'overnight': 0.30,
    'vrp':       0.25,
    'rsi':       0.20,
    'sector':    0.25,
}


# ============================================================
# SINGLE STRATEGY SIMULATION
# ============================================================
def simulate_strategy_year(name: str, rng: random.Random) -> Dict[str, float]:
    p = STRATEGY_PARAMS[name]
    daily_return_mean = p['annual_return_mean'] / 252
    daily_vol = p['annual_vol'] / math.sqrt(252)
    cost_per_trade = p['cost_per_trade_bps'] / 10_000

    n_trades = p['trades_per_year']
    cost_per_day = (n_trades * cost_per_trade) / 252

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    daily_pnl = []

    for _ in range(252):
        # Base normal return
        ret = rng.gauss(daily_return_mean, daily_vol)
        # Fat-tail event check
        if rng.random() < p['fat_tail_prob'] / 252:
            shock = -abs(rng.gauss(0, daily_vol)) * p['fat_tail_loss_mult']
            ret += shock
        # Apply cost drag
        ret -= cost_per_day
        equity *= (1 + ret)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        daily_pnl.append(ret)

    return {
        'final_equity': equity,
        'annual_return': equity - 1,
        'max_drawdown': max_dd,
        'realized_vol': statistics.stdev(daily_pnl) * math.sqrt(252)
            if len(daily_pnl) > 1 else 0,
        'sharpe': (statistics.mean(daily_pnl) / statistics.stdev(daily_pnl)
                   * math.sqrt(252)) if statistics.stdev(daily_pnl) > 0 else 0,
    }


# ============================================================
# CORRELATED PORTFOLIO SIMULATION
# ============================================================
def _corr(a: str, b: str) -> float:
    return CORRELATION.get((a, b)) or CORRELATION.get((b, a)) or 0.0

def simulate_portfolio_year(rng: random.Random, leverage: float = 1.0,
                            halt_dd: float = 0.15) -> Dict[str, float]:
    """
    Generate correlated returns for all four strategies, combine by
    static weights, optionally lever, halt at drawdown threshold.
    """
    strats = list(STRATEGY_PARAMS.keys())
    n = len(strats)

    avg_corr = sum(_corr(strats[i], strats[j])
                   for i in range(n) for j in range(n)
                   if i != j) / (n * (n - 1))
    sqrt_c = math.sqrt(max(avg_corr, 0))
    sqrt_1mc = math.sqrt(max(1 - avg_corr, 0))

    def correlated_draws():
        common = rng.gauss(0, 1)
        return [sqrt_c * common + sqrt_1mc * rng.gauss(0, 1) for _ in range(n)]

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    halted = False
    daily_pnl = []

    for _ in range(252):
        if halted:
            daily_pnl.append(0)
            continue
        draws = correlated_draws()
        day_return = 0.0
        for i, s in enumerate(strats):
            p = STRATEGY_PARAMS[s]
            daily_mean = p['annual_return_mean'] / 252
            daily_vol = p['annual_vol'] / math.sqrt(252)
            cost = (p['trades_per_year'] * p['cost_per_trade_bps'] /
                    10_000) / 252
            r = daily_mean + daily_vol * draws[i] - cost
            if rng.random() < p['fat_tail_prob'] / 252:
                r -= abs(rng.gauss(0, daily_vol)) * p['fat_tail_loss_mult']
            day_return += WEIGHTS[s] * r
        day_return *= leverage
        equity *= (1 + day_return)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        if dd >= halt_dd:
            halted = True
        daily_pnl.append(day_return)

    return {
        'final_equity': equity,
        'annual_return': equity - 1,
        'max_drawdown': max_dd,
        'sharpe': (statistics.mean(daily_pnl) / statistics.stdev(daily_pnl)
                   * math.sqrt(252)) if len(daily_pnl) > 1 and
                  statistics.stdev(daily_pnl) > 0 else 0,
        'halted': halted,
    }


# ============================================================
# REPORTING
# ============================================================
def describe_distribution(label: str, values: List[float], unit: str = "%"):
    if not values:
        print(f"  {label:24}  no data")
        return
    sorted_v = sorted(values)
    n = len(values)
    mean = statistics.mean(values)
    median = statistics.median(values)
    p5 = sorted_v[max(0, int(n * 0.05))]
    p95 = sorted_v[min(n - 1, int(n * 0.95))]
    if unit == "%":
        scale = 100
        suffix = "%"
    else:
        scale = 1
        suffix = ""
    def f(v):
        return f"{v*scale:+7.2f}{suffix}"
    print(f"  {label:24}  "
          f"mean={f(mean)}  median={f(median)}  "
          f"5%ile={f(p5)}  95%ile={f(p95)}")


def run_simulation(n_years: int = 5000, seed: int = 42):
    rng = random.Random(seed)
    print(f"\n{'='*82}")
    print(f"  NEXUS ALPHA — MONTE CARLO SIMULATION   ({n_years:,} simulated years)")
    print(f"{'='*82}")

    # Per-strategy stats
    print(f"\n--- Individual strategy statistics (standalone, no portfolio overlay) ---")
    for sname in STRATEGY_PARAMS.keys():
        runs = [simulate_strategy_year(sname, random.Random(seed + i))
                for i in range(n_years)]
        returns = [r['annual_return'] for r in runs]
        drawdowns = [r['max_drawdown'] for r in runs]
        sharpes = [r['sharpe'] for r in runs]
        prob_profit = sum(1 for r in returns if r > 0) / len(returns) * 100
        prob_loss10 = sum(1 for r in returns if r < -0.10) / len(returns) * 100
        print(f"\n  {sname.upper():24}")
        describe_distribution("  annual return", returns)
        describe_distribution("  max drawdown ", drawdowns)
        describe_distribution("  realised sharpe", sharpes, unit="raw")
        print(f"    P(profitable year):      {prob_profit:5.1f}%")
        print(f"    P(loss > 10%):           {prob_loss10:5.1f}%")

    # Three leverage scenarios so the user can see the risk/return curve.
    for lev, label, halt in [
        (1.0,  "CONSERVATIVE (1.0× leverage, -15% drawdown halt)", 0.15),
        (2.0,  "MODERATE     (2.0× leverage, -20% drawdown halt)", 0.20),
        (3.0,  "AGGRESSIVE   (3.0× leverage, -25% drawdown halt)", 0.25),
    ]:
        print(f"\n--- {label} ---")
        runs = [simulate_portfolio_year(random.Random(seed + i * 7),
                                        leverage=lev, halt_dd=halt)
                for i in range(n_years)]
        returns = [r['annual_return'] for r in runs]
        drawdowns = [r['max_drawdown'] for r in runs]
        sharpes = [r['sharpe'] for r in runs]
        halted = sum(1 for r in runs if r['halted'])
        prob_profit = sum(1 for r in returns if r > 0) / len(returns) * 100
        prob_loss10 = sum(1 for r in returns if r < -0.10) / len(returns) * 100
        prob_gain20 = sum(1 for r in returns if r > 0.20) / len(returns) * 100
        prob_gain50 = sum(1 for r in returns if r > 0.50) / len(returns) * 100

        describe_distribution("PORTFOLIO return", returns)
        describe_distribution("PORTFOLIO drawdown", drawdowns)
        describe_distribution("PORTFOLIO sharpe", sharpes, unit="raw")
        print(f"    P(profitable year):       {prob_profit:5.1f}%")
        print(f"    P(year > +20%):           {prob_gain20:5.1f}%")
        print(f"    P(year > +50%):           {prob_gain50:5.1f}%")
        print(f"    P(loss > 10%):            {prob_loss10:5.1f}%")
        print(f"    P(triggered halt):        {halted/len(runs)*100:5.1f}%")

        # Dollar terms on $50k
        pnls = sorted(r * 50_000 for r in returns)
        print(f"    ON $50K — best 10%:  ${pnls[int(0.90*len(pnls))]:+10,.0f}   "
              f"median: ${pnls[len(pnls)//2]:+10,.0f}   "
              f"worst 10%: ${pnls[int(0.10*len(pnls))]:+10,.0f}")

        # Compounding
        for years in [3, 10]:
            compounded = []
            for _ in range(2000):
                equity = 50_000
                for _ in range(years):
                    equity *= (1 + random.choice(returns))
                compounded.append(equity)
            compounded.sort()
            med = compounded[len(compounded) // 2]
            p10 = compounded[int(len(compounded) * 0.10)]
            p90 = compounded[int(len(compounded) * 0.90)]
            print(f"    $50K after {years:2d}yr:  "
                  f"10%ile ${p10:>10,.0f}   median ${med:>10,.0f}   "
                  f"90%ile ${p90:>10,.0f}")

    print(f"\n{'='*82}\n")

    return {}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_simulation(n_years=args.years, seed=args.seed)
