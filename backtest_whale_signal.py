#!/usr/bin/env python3
"""
Whale Signal A/B Backtest
Validates that the profiled whale estimator outperforms the naive one.

The key question: does knowing a whale's strategy type and category
specialization actually produce better probability estimates?

Approach:
    1. Generate a synthetic population of whales with KNOWN strategies
       (conviction traders, market makers, specialists, etc.)
    2. Simulate markets with KNOWN true probabilities
    3. Place whale positions based on their strategy type
       (conviction traders lean toward truth, MMs are random)
    4. Run both estimators on the same data
    5. Compare Brier scores

This gives a clean signal because we know ground truth — something
we can't get from live markets until months of data accumulate.

Usage:
    python backtest_whale_signal.py
    python run_pipeline.py --whale-backtest
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


# ─── Synthetic Whale Generation ──────────────────────────

@dataclass
class SyntheticWhale:
    """A whale with a known strategy type for backtesting."""
    wallet: str
    name: str
    strategy: str           # CONVICTION, MARKET_MAKER, SPECIALIST, DIVERSIFIED
    pnl: float
    specialty_category: str # Category they're best at (for SPECIALIST)
    skill: float            # 0-1, how good they are at predicting (for non-MMs)
    position_size_range: tuple[float, float]  # (min, max) position size


def generate_whale_population(n: int = 30, seed: int = 42) -> list[SyntheticWhale]:
    """
    Generate a realistic distribution of whale types.
    
    Based on what you'd actually see on Polymarket's leaderboard:
    - ~20% market makers (high PnL from spread capture, no directional skill)
    - ~30% conviction traders (fewer, larger bets, genuinely skilled)
    - ~25% specialists (skilled in one category, mediocre elsewhere)
    - ~25% diversified (moderate skill across categories)
    """
    rng = np.random.RandomState(seed)
    whales = []
    
    categories = ["POLITICS", "CRYPTO", "ECONOMICS", "SPORTS", "POP_CULTURE"]
    
    strategies = (
        ["MARKET_MAKER"] * max(1, int(n * 0.20)) +
        ["CONVICTION"] * max(1, int(n * 0.30)) +
        ["SPECIALIST"] * max(1, int(n * 0.25)) +
        ["DIVERSIFIED"] * max(1, int(n * 0.25))
    )
    # Pad or trim to exactly n
    while len(strategies) < n:
        strategies.append(rng.choice(["CONVICTION", "SPECIALIST", "DIVERSIFIED"]))
    strategies = strategies[:n]
    rng.shuffle(strategies)
    
    for i, strategy in enumerate(strategies):
        if strategy == "MARKET_MAKER":
            pnl = rng.uniform(50000, 500000)  # MMs have high PnL from volume
            skill = 0.50  # No directional edge — they're random
            size_range = (2000, 15000)
        elif strategy == "CONVICTION":
            pnl = rng.uniform(10000, 150000)
            skill = rng.uniform(0.58, 0.72)  # Genuinely skilled
            size_range = (3000, 20000)
        elif strategy == "SPECIALIST":
            pnl = rng.uniform(8000, 80000)
            skill = rng.uniform(0.55, 0.68)  # Skilled in specialty
            size_range = (1000, 10000)
        else:  # DIVERSIFIED
            pnl = rng.uniform(5000, 60000)
            skill = rng.uniform(0.52, 0.60)  # Moderate edge
            size_range = (500, 5000)
        
        whales.append(SyntheticWhale(
            wallet=f"0x{i:040x}",
            name=f"{strategy.lower()}_{i:02d}",
            strategy=strategy,
            pnl=pnl,
            specialty_category=rng.choice(categories),
            skill=skill,
            position_size_range=size_range,
        ))
    
    return whales


# ─── Market Simulation ───────────────────────────────────

@dataclass
class SimulatedMarket:
    """A market with a known true probability."""
    market_id: str
    question: str
    category: str
    true_prob: float       # Ground truth (unknown to estimators)
    market_price: float    # Current market price (noisy observation of true_prob)
    whale_positions: list  # Populated by simulate_whale_behavior


def generate_markets(
    n: int = 200,
    categories: list[str] = None,
    seed: int = 123,
) -> list[SimulatedMarket]:
    """Generate markets with known true probabilities."""
    rng = np.random.RandomState(seed)
    categories = categories or ["POLITICS", "CRYPTO", "ECONOMICS", "SPORTS", "POP_CULTURE"]
    
    markets = []
    for i in range(n):
        true_prob = rng.beta(2, 2)  # Centered distribution, some at extremes
        true_prob = np.clip(true_prob, 0.05, 0.95)
        
        # Market price = noisy observation of true prob
        noise = rng.normal(0, 0.08)
        market_price = np.clip(true_prob + noise, 0.05, 0.95)
        
        markets.append(SimulatedMarket(
            market_id=f"sim-{i:04d}",
            question=f"Simulated market #{i} ({categories[i % len(categories)]})",
            category=categories[i % len(categories)],
            true_prob=true_prob,
            market_price=market_price,
            whale_positions=[],
        ))
    
    return markets


def simulate_whale_behavior(
    markets: list[SimulatedMarket],
    whales: list[SyntheticWhale],
    participation_rate: float = 0.15,
    seed: int = 456,
):
    """
    Simulate which whales take positions in which markets, and on which side.
    
    Key behavioral differences by strategy:
    - CONVICTION: Only enters markets where they see edge. Leans toward truth.
    - SPECIALIST: Strong lean in their category, weak/random elsewhere.
    - MARKET_MAKER: Positions on both sides, essentially random direction.
    - DIVERSIFIED: Moderate lean toward truth across categories.
    """
    rng = np.random.RandomState(seed)
    
    for market in markets:
        market.whale_positions = []
        
        for whale in whales:
            # Decide whether this whale participates
            if rng.random() > participation_rate:
                continue
            
            # Determine the whale's "view" of the true probability
            if whale.strategy == "MARKET_MAKER":
                # MMs don't have a directional view — random side
                whale_view = 0.5
            elif whale.strategy == "SPECIALIST":
                if market.category == whale.specialty_category:
                    # In their specialty: genuinely sees toward truth
                    whale_view = whale.skill * market.true_prob + (1 - whale.skill) * 0.5
                else:
                    # Outside specialty: barely better than random
                    reduced_skill = 0.50 + (whale.skill - 0.50) * 0.2
                    whale_view = reduced_skill * market.true_prob + (1 - reduced_skill) * 0.5
            else:
                # CONVICTION and DIVERSIFIED: lean toward truth
                whale_view = whale.skill * market.true_prob + (1 - whale.skill) * 0.5
            
            # Add noise to the whale's view
            whale_view += rng.normal(0, 0.08)
            whale_view = np.clip(whale_view, 0.05, 0.95)
            
            # Decide side: if whale_view > market_price → buy YES, else NO
            if whale_view > market.market_price + 0.02:
                side = "YES"
            elif whale_view < market.market_price - 0.02:
                side = "NO"
            else:
                # Close to market price — MMs might still trade, others skip
                if whale.strategy == "MARKET_MAKER":
                    side = "YES" if rng.random() > 0.5 else "NO"
                else:
                    continue  # No edge, skip
            
            size = rng.uniform(*whale.position_size_range)
            
            market.whale_positions.append({
                "side": side,
                "size": size,
                "trader_pnl": whale.pnl,
                "whale_name": whale.name,
                "whale_rank": str(whales.index(whale) + 1),
                "whale_volume": whale.pnl * 20,  # Rough volume estimate
                # Profile data (only used by profiled estimator)
                "_true_strategy": whale.strategy,
                "_true_skill": whale.skill,
                "_specialty": whale.specialty_category,
            })


def build_profiled_positions(markets: list[SimulatedMarket]):
    """
    Add profile metadata to whale positions.
    
    In production, this comes from WhaleProfiler. In the backtest,
    we synthesize it from the known strategy types (with some noise
    to simulate imperfect classification).
    """
    rng = np.random.RandomState(789)
    
    for market in markets:
        for wp in market.whale_positions:
            true_strategy = wp["_true_strategy"]
            true_skill = wp["_true_skill"]
            specialty = wp["_specialty"]
            
            # Simulate imperfect strategy classification
            # 85% correct, 15% misclassified
            if rng.random() < 0.85:
                classified_strategy = true_strategy
            else:
                classified_strategy = rng.choice(
                    ["CONVICTION", "DIVERSIFIED", "SPECIALIST", "MARKET_MAKER"]
                )
            
            # Compute signal weight based on classified strategy
            strategy_mult = {
                "CONVICTION": 1.3,
                "SPECIALIST": 1.2,
                "DIVERSIFIED": 1.0,
                "MARKET_MAKER": 0.15,
            }.get(classified_strategy, 0.6)
            
            # Category credibility
            if classified_strategy == "SPECIALIST" and market.category == specialty:
                cat_cred = min(1.0, true_skill * 1.2)
            else:
                cat_cred = min(1.0, true_skill * 0.7)
            
            signal_weight = min(1.0, cat_cred * strategy_mult)
            
            wp["profile_strategy"] = classified_strategy
            wp["profile_signal_weight"] = signal_weight
            wp["profile_category_credibility"] = cat_cred
            wp["profile_win_rate"] = true_skill if true_strategy != "MARKET_MAKER" else None
            wp["profile_primary_category"] = specialty


# ─── Estimator Comparison ────────────────────────────────

def run_naive_estimator(market_price: float, whale_positions: list) -> float:
    """
    The OLD whale_tracker_estimator logic (with corrected blend weight).
    Weights only by PnL and position size. No strategy awareness.
    """
    if not whale_positions:
        return market_price
    
    yes_signal = 0.0
    no_signal = 0.0
    
    for wp in whale_positions:
        credibility = min(1.0, max(0.0, wp.get("trader_pnl", 0) / 10000))
        size_weight = min(1.0, wp.get("size", 0) / 5000)
        signal = credibility * size_weight
        
        if wp.get("side", "").upper() in ("YES", "BUY"):
            yes_signal += signal
        else:
            no_signal += signal
    
    total = yes_signal + no_signal
    if total == 0:
        return market_price
    
    whale_lean = yes_signal / total
    # Corrected: 7% blend (was 30% — backtesting proved that was way too aggressive)
    return 0.93 * market_price + 0.07 * whale_lean


def run_profiled_estimator(market_price: float, whale_positions: list) -> float:
    """
    The NEW profiled_whale_estimator logic (with corrected blend weight).
    Uses strategy type, category credibility, and signal weight.
    Discounts market makers. Conservative blend (3-10%).
    """
    if not whale_positions:
        return market_price
    
    yes_signal = 0.0
    no_signal = 0.0
    credible_count = 0
    
    for wp in whale_positions:
        signal_weight = wp.get("profile_signal_weight")
        
        if signal_weight is not None:
            size_weight = min(1.0, wp.get("size", 0) / 5000)
            signal = signal_weight * size_weight
            
            strategy = wp.get("profile_strategy", "UNKNOWN")
            if strategy == "MARKET_MAKER" and signal_weight < 0.2:
                continue
            
            credible_count += 1
        else:
            credibility = min(1.0, max(0.0, wp.get("trader_pnl", 0) / 10000))
            size_weight = min(1.0, wp.get("size", 0) / 5000)
            signal = credibility * size_weight
            credible_count += 1
        
        side = wp.get("side", "").upper()
        if side in ("YES", "BUY"):
            yes_signal += signal
        else:
            no_signal += signal
    
    total = yes_signal + no_signal
    if total == 0:
        return market_price
    
    whale_lean = yes_signal / total
    # Corrected: conservative 3-10% blend, scaling with credible whale count
    whale_blend = min(0.10, 0.03 + credible_count * 0.025)
    
    return (1 - whale_blend) * market_price + whale_blend * whale_lean


# ─── Scoring ─────────────────────────────────────────────

def brier_score(predicted: float, actual: bool) -> float:
    """Brier score for a single prediction. Lower is better."""
    outcome = 1.0 if actual else 0.0
    return (predicted - outcome) ** 2


def run_backtest(
    n_markets: int = 500,
    n_whales: int = 30,
    participation_rate: float = 0.15,
    n_trials: int = 5,
):
    """
    Run the full A/B backtest across multiple random seeds.
    
    For each trial:
    1. Generate whales and markets
    2. Simulate whale behavior
    3. Run both estimators
    4. Compare Brier scores
    """
    print("\n" + "=" * 75)
    print("  WHALE SIGNAL A/B BACKTEST")
    print("  Comparing naive (PnL-only) vs profiled (strategy-aware) estimator")
    print("=" * 75)
    print(f"\n  Config: {n_markets} markets × {n_trials} trials, "
          f"{n_whales} whales, {participation_rate:.0%} participation rate\n")
    
    all_naive_brier = []
    all_profiled_brier = []
    all_market_brier = []  # Baseline: just using market price
    
    category_results = {}  # Track per-category improvement
    
    for trial in range(n_trials):
        seed_base = trial * 1000
        
        whales = generate_whale_population(n_whales, seed=seed_base)
        markets = generate_markets(n_markets, seed=seed_base + 100)
        simulate_whale_behavior(markets, whales, participation_rate, seed=seed_base + 200)
        build_profiled_positions(markets)
        
        # Single RNG for outcome resolution — deterministic per trial
        outcome_rng = np.random.RandomState(seed_base + 300)
        
        trial_naive = []
        trial_profiled = []
        trial_market = []
        
        for market in markets:
            # Always advance RNG to maintain determinism
            outcome_draw = outcome_rng.random()
            
            if not market.whale_positions:
                continue  # Skip markets with no whale signals
            
            # Resolve: did YES happen?
            outcome = outcome_draw < market.true_prob
            
            # Market price baseline
            market_est = market.market_price
            
            # Naive estimator
            naive_est = run_naive_estimator(market.market_price, market.whale_positions)
            
            # Profiled estimator
            profiled_est = run_profiled_estimator(market.market_price, market.whale_positions)
            
            trial_market.append(brier_score(market_est, outcome))
            trial_naive.append(brier_score(naive_est, outcome))
            trial_profiled.append(brier_score(profiled_est, outcome))
            
            # Per-category tracking
            cat = market.category
            if cat not in category_results:
                category_results[cat] = {"naive": [], "profiled": [], "market": []}
            category_results[cat]["naive"].append(brier_score(naive_est, outcome))
            category_results[cat]["profiled"].append(brier_score(profiled_est, outcome))
            category_results[cat]["market"].append(brier_score(market_est, outcome))
        
        avg_market = np.mean(trial_market) if trial_market else 0
        avg_naive = np.mean(trial_naive) if trial_naive else 0
        avg_profiled = np.mean(trial_profiled) if trial_profiled else 0
        
        all_market_brier.append(avg_market)
        all_naive_brier.append(avg_naive)
        all_profiled_brier.append(avg_profiled)
        
        n_with_whales = len(trial_naive)
        improvement = (avg_naive - avg_profiled) / avg_naive * 100 if avg_naive > 0 else 0
        
        print(f"  Trial {trial+1}: {n_with_whales} markets with whale signals")
        print(f"    Market price only : {avg_market:.4f}")
        print(f"    Naive whale       : {avg_naive:.4f}")
        print(f"    Profiled whale    : {avg_profiled:.4f}  ({improvement:+.1f}% vs naive)")
    
    # ─── Aggregate Results ────────────────────────────────
    print("\n" + "-" * 75)
    print("  AGGREGATE RESULTS (lower Brier score = better)")
    print("-" * 75)
    
    mean_market = np.mean(all_market_brier)
    mean_naive = np.mean(all_naive_brier)
    mean_profiled = np.mean(all_profiled_brier)
    
    std_naive = np.std(all_naive_brier)
    std_profiled = np.std(all_profiled_brier)
    
    improvement = (mean_naive - mean_profiled) / mean_naive * 100
    
    print(f"\n  Market price (baseline)  : {mean_market:.4f}")
    print(f"  Naive whale estimator    : {mean_naive:.4f} (±{std_naive:.4f})")
    print(f"  Profiled whale estimator : {mean_profiled:.4f} (±{std_profiled:.4f})")
    print(f"\n  Profiled improvement vs naive : {improvement:+.1f}%")
    print(f"  Profiled improvement vs market: {(mean_market - mean_profiled) / mean_market * 100:+.1f}%")
    
    # Does the whale signal help at all vs market price?
    naive_vs_market = (mean_market - mean_naive) / mean_market * 100
    profiled_vs_market = (mean_market - mean_profiled) / mean_market * 100
    
    print(f"\n  Naive whale vs market price   : {naive_vs_market:+.1f}%")
    print(f"  Profiled whale vs market price : {profiled_vs_market:+.1f}%")
    
    # ─── Per-Category Breakdown ──────────────────────────
    print("\n" + "-" * 75)
    print("  PER-CATEGORY BREAKDOWN")
    print("-" * 75)
    print(f"  {'Category':<15} {'Market':>8} {'Naive':>8} {'Profiled':>8} {'Improvement':>12}")
    
    for cat in sorted(category_results.keys()):
        data = category_results[cat]
        m = np.mean(data["market"])
        n = np.mean(data["naive"])
        p = np.mean(data["profiled"])
        imp = (n - p) / n * 100 if n > 0 else 0
        print(f"  {cat:<15} {m:>8.4f} {n:>8.4f} {p:>8.4f} {imp:>+11.1f}%")
    
    # ─── Whale Strategy Impact ───────────────────────────
    print("\n" + "-" * 75)
    print("  WHY PROFILING HELPS — Strategy Impact Analysis")
    print("-" * 75)
    
    # Rerun one trial to collect per-whale-type statistics
    whales = generate_whale_population(n_whales, seed=0)
    markets = generate_markets(n_markets, seed=100)
    simulate_whale_behavior(markets, whales, participation_rate, seed=200)
    build_profiled_positions(markets)
    
    # Track whether whale TRADES would be profitable (not just directionally correct)
    # A whale who buys YES at market_price=0.30 when true_prob=0.40 is making a +EV
    # trade even though YES only resolves 40% of the time.
    strat_counts = {}
    strat_ev = {}  # Expected value of following each strategy's trades
    for market in markets:
        for wp in market.whale_positions:
            strat = wp["_true_strategy"]
            strat_counts[strat] = strat_counts.get(strat, 0) + 1
            
            # EV of the trade: buy YES at market_price, worth true_prob
            if wp["side"] == "YES":
                trade_ev = market.true_prob - market.market_price
            else:
                trade_ev = (1 - market.true_prob) - (1 - market.market_price)
            
            if strat not in strat_ev:
                strat_ev[strat] = []
            strat_ev[strat].append(trade_ev)
    
    print(f"\n  {'Strategy':<15} {'Trades':>8} {'Avg EV':>10} {'% Positive EV':>14}")
    for strat in ["CONVICTION", "SPECIALIST", "DIVERSIFIED", "MARKET_MAKER"]:
        count = strat_counts.get(strat, 0)
        evs = strat_ev.get(strat, [])
        avg_ev = np.mean(evs) if evs else 0
        pct_pos = np.mean([1 for e in evs if e > 0]) / len(evs) * 100 if evs else 0
        marker = " ← noise" if strat == "MARKET_MAKER" else (" ← signal" if avg_ev > 0.005 else "")
        print(f"  {strat:<15} {count:>8} {avg_ev:>+10.4f} {pct_pos:>13.1f}%{marker}")
    
    print(f"""
  Key insight: conviction and specialist traders make +EV trades (they correctly
  identify mispriced markets). Market makers are near-zero EV — their positions
  are about liquidity provision, not directional conviction.

  The naive estimator gives market makers MORE weight (because they have higher
  PnL from volume, not from directional skill). The profiled estimator discounts
  them to ~15% signal weight and lets the conviction traders' signal dominate.
""")
    
    print("=" * 75)
    
    return {
        "mean_market": mean_market,
        "mean_naive": mean_naive,
        "mean_profiled": mean_profiled,
        "improvement_pct": improvement,
    }


if __name__ == "__main__":
    results = run_backtest(
        n_markets=500,
        n_whales=30,
        participation_rate=0.15,
        n_trials=5,
    )
