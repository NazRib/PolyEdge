#!/usr/bin/env python3
"""
Backtest: compare the basic ensemble vs the LLM-enhanced ensemble
over a simulated market run.

This gives you a feel for how much the LLM estimator improves things
before you connect it to live data and spend API credits.

Usage:
    python backtest_llm.py
"""

import logging
import numpy as np
from core.probability import (
    EnsembleEstimator,
    ProbabilityEstimate,
    CalibrationTracker,
    base_rate_estimator,
    momentum_estimator,
    book_imbalance_estimator,
    whale_tracker_estimator,
)
from core.llm_estimator import (
    SimulatedLLMEstimator,
    CalibrationModel,
)
from core.kelly import kelly_criterion
from core.paper_trader import PaperTrader

logging.basicConfig(level=logging.WARNING)


def generate_synthetic_markets(n: int = 200, seed: int = 42) -> list[dict]:
    """
    Generate synthetic markets with known true probabilities.
    
    Each market has:
    - A "true" probability (hidden from the estimator)
    - A market price (true prob + market noise)
    - Order book, price history, whale data (all correlated with true prob)
    - An actual outcome (sampled from true prob)
    """
    rng = np.random.RandomState(seed)
    
    categories = ["economics", "politics", "sports", "crypto", "corporate", "culture"]
    questions = [
        "Will {} happen by {}?",
        "Will {} exceed {}?", 
        "Will {} beat {}?",
        "Will {} win {}?",
    ]
    
    markets = []
    for i in range(n):
        # True probability (unknown to our system)
        true_prob = rng.beta(2, 2)
        
        # Market price: true prob + noise (market is good but not perfect)
        market_noise = rng.normal(0, 0.08)
        market_price = np.clip(true_prob + market_noise, 0.05, 0.95)
        
        # Category
        category = rng.choice(categories)
        
        # Price history: noisy path from an earlier price toward current
        n_points = 20
        start_price = market_price + rng.normal(0, 0.06)
        price_path = np.linspace(start_price, market_price, n_points)
        price_path += rng.normal(0, 0.015, n_points)
        price_path = np.clip(price_path, 0.01, 0.99)
        price_history = [(j, float(price_path[j])) for j in range(n_points)]
        
        # Order book: slight lean toward the "correct" side
        base_depth = rng.uniform(5000, 50000)
        if true_prob > 0.5:
            bid_depth = base_depth * (1 + (true_prob - 0.5) * 0.5 + rng.normal(0, 0.1))
            ask_depth = base_depth * (1 - (true_prob - 0.5) * 0.3 + rng.normal(0, 0.1))
        else:
            bid_depth = base_depth * (1 - (0.5 - true_prob) * 0.3 + rng.normal(0, 0.1))
            ask_depth = base_depth * (1 + (0.5 - true_prob) * 0.5 + rng.normal(0, 0.1))
        bid_depth = max(1000, bid_depth)
        ask_depth = max(1000, ask_depth)
        
        # Whale positions (occasionally informative)
        whale_positions = []
        if rng.random() < 0.3:  # 30% chance of whale activity
            whale_side = "YES" if true_prob > 0.55 else "NO" if true_prob < 0.45 else rng.choice(["YES", "NO"])
            whale_positions.append({
                "side": whale_side,
                "size": float(rng.uniform(3000, 20000)),
                "trader_pnl": float(rng.uniform(5000, 100000)),
            })
        
        # Actual outcome
        outcome = bool(rng.random() < true_prob)
        
        volume_24h = float(rng.uniform(1000, 200000))
        
        markets.append({
            "market_id": f"sim-{i:04d}",
            "question": f"Simulated {category} market #{i+1}",
            "description": "",
            "category": category,
            "market_price": float(market_price),
            "volume_24h": volume_24h,
            "volume_total": volume_24h * rng.uniform(5, 50),
            "liquidity": float(rng.uniform(5000, 200000)),
            "days_to_resolution": float(rng.uniform(1, 60)),
            "price_history": price_history,
            "bid_depth": float(bid_depth),
            "ask_depth": float(ask_depth),
            "spread": float(rng.uniform(0.01, 0.08)),
            "whale_positions": whale_positions,
            "tags": [category],
            # Hidden ground truth:
            "_true_prob": float(true_prob),
            "_outcome": outcome,
        })
    
    return markets


def run_strategy(
    name: str,
    ensemble: EnsembleEstimator,
    markets: list[dict],
    bankroll: float = 1000.0,
    kelly_frac: float = 0.25,
    min_edge: float = 0.05,
) -> dict:
    """Run a strategy over synthetic markets and return performance metrics."""
    
    tracker = CalibrationTracker()
    trader = PaperTrader(bankroll=bankroll, data_dir=f"data/backtest_{name}")
    
    trades_entered = 0
    
    for m in markets:
        # Estimate
        estimate = ensemble.estimate(m)
        
        # Track calibration for ALL estimates (not just traded ones)
        # This tells us if our estimator is well-calibrated in general
        tracker.record(estimate.probability, m["_outcome"], m["market_id"])
        
        # Size position
        position = kelly_criterion(
            estimated_prob=estimate.probability,
            market_price=estimate.market_price,
            bankroll=trader.bankroll,
            kelly_fraction=kelly_frac,
            min_edge=min_edge,
            confidence=estimate.confidence,
        )
        
        # Paper trade
        trade = trader.enter_trade(estimate, position)
        if trade:
            trades_entered += 1
            trader.resolve_trade(trade.trade_id, m["_outcome"])
    
    # Compute metrics
    resolved = [t for t in trader.trades if t.status.startswith("RESOLVED")]
    wins = [t for t in resolved if t.status == "RESOLVED_WIN"]
    
    total_pnl = sum(t.pnl for t in resolved)
    
    return {
        "name": name,
        "brier_score": tracker.brier_score,
        "trades_entered": trades_entered,
        "trades_total": len(markets),
        "selectivity": trades_entered / len(markets),
        "wins": len(wins),
        "losses": len(resolved) - len(wins),
        "win_rate": len(wins) / len(resolved) if resolved else 0,
        "total_pnl": total_pnl,
        "final_bankroll": trader.bankroll,
        "return_pct": (trader.bankroll / bankroll - 1),
        "avg_pnl_per_trade": total_pnl / len(resolved) if resolved else 0,
        "calibration_curve": tracker.calibration_curve(5),
    }


def main():
    print("\n" + "=" * 70)
    print("  BACKTEST: Basic Ensemble vs LLM-Enhanced Ensemble")
    print("=" * 70)
    
    # Generate markets
    print("\nGenerating 200 synthetic markets...")
    markets = generate_synthetic_markets(n=200, seed=42)
    
    true_probs = [m["_true_prob"] for m in markets]
    mkt_prices = [m["market_price"] for m in markets]
    market_brier = np.mean([(p - (1 if m["_outcome"] else 0)) ** 2 
                            for p, m in zip(mkt_prices, markets)])
    
    print(f"  Market's own Brier score: {market_brier:.4f} (this is the benchmark to beat)")
    
    # ── Strategy 1: Basic ensemble (no LLM) ──────────────
    basic = EnsembleEstimator()
    basic.add_estimator("base_rate", base_rate_estimator, weight=0.30)
    basic.add_estimator("momentum", momentum_estimator, weight=0.30)
    basic.add_estimator("book_imbalance", book_imbalance_estimator, weight=0.20)
    basic.add_estimator("whale_tracker", whale_tracker_estimator, weight=0.20)
    
    print("\nRunning basic ensemble...")
    basic_results = run_strategy("basic", basic, markets)
    
    # ── Strategy 2: LLM-enhanced ensemble ─────────────────
    # Simulated LLM with moderate skill (skill=0.35 means it captures 35% of true edge)
    llm_sim = SimulatedLLMEstimator(skill_level=0.35)
    
    enhanced = EnsembleEstimator()
    enhanced.add_estimator("base_rate", base_rate_estimator, weight=0.15)
    enhanced.add_estimator("llm", llm_sim.estimate_for_ensemble, weight=0.40)
    enhanced.add_estimator("momentum", momentum_estimator, weight=0.20)
    enhanced.add_estimator("book_imbalance", book_imbalance_estimator, weight=0.15)
    enhanced.add_estimator("whale_tracker", whale_tracker_estimator, weight=0.10)
    
    print("Running LLM-enhanced ensemble...")
    llm_results = run_strategy("llm_enhanced", enhanced, markets)
    
    # ── Strategy 3: LLM with higher skill (what good calibration gets you) ──
    llm_skilled = SimulatedLLMEstimator(skill_level=0.50)
    
    skilled = EnsembleEstimator()
    skilled.add_estimator("base_rate", base_rate_estimator, weight=0.10)
    skilled.add_estimator("llm", llm_skilled.estimate_for_ensemble, weight=0.50)
    skilled.add_estimator("momentum", momentum_estimator, weight=0.20)
    skilled.add_estimator("book_imbalance", book_imbalance_estimator, weight=0.10)
    skilled.add_estimator("whale_tracker", whale_tracker_estimator, weight=0.10)
    
    print("Running skilled LLM ensemble...")
    skilled_results = run_strategy("llm_skilled", skilled, markets)
    
    # ── Results Comparison ────────────────────────────────
    all_results = [basic_results, llm_results, skilled_results]
    
    print("\n\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)
    
    # Header
    print(f"\n  {'Metric':<28} ", end="")
    for r in all_results:
        print(f"{'│ ' + r['name']:>22}", end="")
    print(f"{'│ market':>22}")
    print("  " + "─" * 94)
    
    # Brier Score (lower is better)
    print(f"  {'Brier score':<28} ", end="")
    for r in all_results:
        bs = r["brier_score"]
        indicator = "🟢" if bs < 0.20 else "🟡" if bs < 0.25 else "🔴"
        print(f"│ {bs:>16.4f} {indicator}  ", end="")
    print(f"│ {market_brier:>16.4f} 📊  ")
    
    # vs Market
    print(f"  {'vs market Brier':<28} ", end="")
    for r in all_results:
        diff = (market_brier - r["brier_score"]) / market_brier
        print(f"│ {diff:>+16.1%}     ", end="")
    print(f"│ {'baseline':>16}     ")
    
    # Trades
    print(f"  {'Trades entered':<28} ", end="")
    for r in all_results:
        print(f"│ {r['trades_entered']:>11} / {r['trades_total']:<3}  ", end="")
    print(f"│ {'—':>16}     ")
    
    # Selectivity
    print(f"  {'Selectivity':<28} ", end="")
    for r in all_results:
        print(f"│ {r['selectivity']:>16.0%}     ", end="")
    print(f"│ {'—':>16}     ")
    
    # Win rate
    print(f"  {'Win rate':<28} ", end="")
    for r in all_results:
        print(f"│ {r['win_rate']:>16.0%}     ", end="")
    print(f"│ {'—':>16}     ")
    
    # PnL
    print(f"  {'Total PnL':<28} ", end="")
    for r in all_results:
        pnl = r["total_pnl"]
        print(f"│ ${pnl:>+14,.2f}     ", end="")
    print(f"│ {'—':>16}     ")
    
    # Return
    print(f"  {'Return':<28} ", end="")
    for r in all_results:
        print(f"│ {r['return_pct']:>+16.1%}     ", end="")
    print(f"│ {'—':>16}     ")
    
    # Avg PnL per trade
    print(f"  {'Avg PnL / trade':<28} ", end="")
    for r in all_results:
        print(f"│ ${r['avg_pnl_per_trade']:>+14,.2f}     ", end="")
    print(f"│ {'—':>16}     ")
    
    print("  " + "─" * 94)
    
    # Calibration curves
    print("\n\n  CALIBRATION CURVES (predicted vs actual frequency):")
    for r in all_results:
        print(f"\n  {r['name']}:")
        curve = r["calibration_curve"]
        for center, freq, count in curve:
            bar_predicted = "█" * int(center * 30)
            bar_actual = "▓" * int(freq * 30)
            print(f"    {center:>4.0%}: predicted {bar_predicted:<16} actual {bar_actual:<16} ({count} mkts)")
    
    # Interpretation
    print("\n\n  INTERPRETATION:")
    best = min(all_results, key=lambda r: r["brier_score"])
    worst = max(all_results, key=lambda r: r["brier_score"])
    
    print(f"  Best Brier score: {best['name']} ({best['brier_score']:.4f})")
    print(f"  Best return:      {max(all_results, key=lambda r: r['return_pct'])['name']} "
          f"({max(r['return_pct'] for r in all_results):+.1%})")
    
    improvement = (basic_results["brier_score"] - llm_results["brier_score"]) / basic_results["brier_score"]
    print(f"\n  LLM estimator improved Brier score by {improvement:.1%} over basic ensemble")
    
    if llm_results["brier_score"] < market_brier:
        print("  ✅ LLM ensemble beats the market's own pricing!")
    else:
        print("  ⚠️  LLM ensemble does not yet beat market pricing")
    
    if skilled_results["brier_score"] < market_brier:
        print("  ✅ Skilled LLM shows what's achievable with better calibration")
    
    print(f"""
  KEY TAKEAWAY:
  The simulated LLM at skill=0.35 captures about a third of the true 
  edge. Even this modest skill level can produce positive returns when 
  combined with disciplined Kelly sizing and selectivity (only trading 
  when edge exceeds threshold).
  
  To improve from here:
  1. Connect real Claude API calls (replace SimulatedLLMEstimator with LLMEstimator)
  2. Add domain-specific context (news, data feeds) to the prompts
  3. Track resolved markets and let the calibration layer learn
  4. Expand the base rate database with real historical data
  5. Paper trade for 50+ resolved markets before risking real capital
""")


if __name__ == "__main__":
    main()
