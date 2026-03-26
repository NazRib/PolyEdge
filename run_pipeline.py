#!/usr/bin/env python3
"""
Polymarket Edge — Main Pipeline Runner

Usage:
    python run_pipeline.py                    # Full pipeline (scan → estimate → paper trade)
    python run_pipeline.py --scan-only        # Just scan and show opportunities
    python run_pipeline.py --enriched         # Enriched pipeline (with context enrichment)
    python run_pipeline.py --enriched --live  # Enriched + real Claude API calls
    python run_pipeline.py --enrich-demo      # Demo the context enrichment sources
    python run_pipeline.py --report           # Show paper trading report
    python run_pipeline.py --demo             # Run with simulated data (no API calls)
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    args = sys.argv[1:]
    
    if "--demo" in args:
        run_demo()
    elif "--report" in args:
        from core.paper_trader import PaperTrader
        trader = PaperTrader()
        print(trader.report())
    elif "--scan-only" in args:
        from core.market_scanner import demo_scan
        demo_scan()
    elif "--enriched" in args:
        from strategies.enriched_edge_detector import run_enriched_pipeline
        use_live = "--live" in args
        run_enriched_pipeline(bankroll=1000, use_live_llm=use_live)
    elif "--enrich-demo" in args:
        from core.context_enricher import demo as enrich_demo
        enrich_demo()
    else:
        from strategies.edge_detector import run_pipeline
        run_pipeline()


def run_demo():
    """
    Run a full demo with simulated data to show how the system works.
    No API calls needed — uses synthetic markets.
    """
    from core.probability import (
        ProbabilityEstimate, EnsembleEstimator, BayesianUpdater,
        CalibrationTracker, create_default_ensemble,
    )
    from core.kelly import kelly_criterion, size_multiple_positions
    from core.paper_trader import PaperTrader
    
    import numpy as np
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("  POLYMARKET EDGE — Demo Mode (Simulated Data)")
    print("=" * 70)
    
    # ─── Part 1: Probability Estimation Demo ─────────────
    print("\n\n📊 PART 1: Probability Estimation")
    print("-" * 50)
    
    ensemble = create_default_ensemble()
    
    demo_markets = [
        {
            "market_id": "fed-rate-cut-june",
            "question": "Will the Fed cut rates in June 2026?",
            "market_price": 0.42,
            "volume_24h": 85000,
            "days_to_resolution": 45,
            "price_history": [(i, 0.38 + i * 0.004 + np.random.normal(0, 0.01)) for i in range(20)],
            "bid_depth": 18000,
            "ask_depth": 9000,
            "whale_positions": [
                {"side": "YES", "size": 8000, "trader_pnl": 45000},
                {"side": "YES", "size": 3000, "trader_pnl": 12000},
                {"side": "NO", "size": 2000, "trader_pnl": 3000},
            ],
        },
        {
            "market_id": "btc-150k-2026",
            "question": "Will Bitcoin exceed $150K in 2026?",
            "market_price": 0.28,
            "volume_24h": 120000,
            "days_to_resolution": 280,
            "price_history": [(i, 0.30 - i * 0.001 + np.random.normal(0, 0.015)) for i in range(20)],
            "bid_depth": 12000,
            "ask_depth": 14000,
            "whale_positions": [
                {"side": "NO", "size": 6000, "trader_pnl": 30000},
            ],
        },
        {
            "market_id": "uk-election-2026",
            "question": "Will Labour win the next UK election?",
            "market_price": 0.65,
            "volume_24h": 45000,
            "days_to_resolution": 30,
            "price_history": [(i, 0.60 + i * 0.005 + np.random.normal(0, 0.008)) for i in range(20)],
            "bid_depth": 7000,
            "ask_depth": 7500,
            "whale_positions": [],
        },
        {
            "market_id": "nvidia-earnings-beat",
            "question": "Will NVIDIA beat Q2 earnings estimates?",
            "market_price": 0.73,
            "volume_24h": 65000,
            "days_to_resolution": 12,
            "price_history": [(i, 0.70 + i * 0.003 + np.random.normal(0, 0.012)) for i in range(20)],
            "bid_depth": 20000,
            "ask_depth": 8000,
            "whale_positions": [
                {"side": "YES", "size": 15000, "trader_pnl": 80000},
            ],
        },
        {
            "market_id": "recession-2026",
            "question": "US recession declared in 2026?",
            "market_price": 0.18,
            "volume_24h": 95000,
            "days_to_resolution": 200,
            "price_history": [(i, 0.15 + i * 0.003 + np.random.normal(0, 0.01)) for i in range(20)],
            "bid_depth": 5000,
            "ask_depth": 15000,
            "whale_positions": [
                {"side": "NO", "size": 10000, "trader_pnl": 50000},
            ],
        },
    ]
    
    print(f"\n{'Market':<45} {'Mkt Price':>10} {'Our Est':>10} {'Edge':>10} {'Direction':>12}")
    print("-" * 90)
    
    estimates = []
    for ctx in demo_markets:
        est = ensemble.estimate(ctx)
        estimates.append((ctx, est))
        print(
            f"{est.question:<45} "
            f"{est.market_price:>9.0%} "
            f"{est.probability:>9.1%} "
            f"{est.edge:>+9.1%} "
            f"{est.edge_direction:>12}"
        )
    
    # ─── Part 2: Position Sizing Demo ────────────────────
    print("\n\n💰 PART 2: Kelly Criterion Position Sizing")
    print("-" * 50)
    
    bankroll = 1000.0
    
    print(f"\nBankroll: ${bankroll:,.2f} | Kelly Fraction: 0.25x\n")
    print(f"{'Market':<40} {'Side':>5} {'Amount':>10} {'Shares':>8} {'Edge':>7} {'EV':>8}")
    print("-" * 85)
    
    positions = []
    for ctx, est in estimates:
        pos = kelly_criterion(
            estimated_prob=est.probability,
            market_price=est.market_price,
            bankroll=bankroll,
            kelly_fraction=0.25,
            min_edge=0.03,
            confidence=est.confidence,
        )
        positions.append((ctx, est, pos))
        
        if pos.should_trade:
            print(
                f"{est.question:<40} "
                f"{pos.side:>5} "
                f"${pos.dollar_amount:>8.2f} "
                f"{pos.shares:>7.1f} "
                f"{pos.edge:>+6.1%} "
                f"${pos.expected_profit:>+7.2f}"
            )
        else:
            print(
                f"{est.question:<40} "
                f"{'SKIP':>5} "
                f"{'—':>10} "
                f"{'—':>8} "
                f"{pos.edge:>+6.1%} "
                f"{'—':>8}  ({pos.rejection_reason})"
            )
    
    # ─── Part 3: Bayesian Updating Demo ──────────────────
    print("\n\n🔄 PART 3: Bayesian Updating")
    print("-" * 50)
    
    updater = BayesianUpdater(prior=0.42, strength=5)
    print(f"\nTracking: 'Fed rate cut in June 2026?'")
    print(f"Starting estimate: {updater.mean:.1%} (±{updater.std:.1%})")
    
    events = [
        ("Inflation report comes in below expectations", 2.5),
        ("Fed chair signals patience on cuts", 0.4),
        ("Jobs report strong — economy resilient", 0.6),
        ("Bond market prices in rate cut", 1.8),
        ("European central bank cuts rates", 1.3),
    ]
    
    for event, lr in events:
        updater.update_with_evidence(lr, label=event)
        ci = updater.credible_interval_90
        print(f"  📰 {event}")
        print(f"     → {updater.mean:.1%} (90% CI: {ci[0]:.1%}–{ci[1]:.1%})")
    
    # ─── Part 4: Paper Trading Simulation ────────────────
    print("\n\n🧪 PART 4: Paper Trading Simulation (50 markets)")
    print("-" * 50)
    
    trader = PaperTrader(bankroll=1000, data_dir="data/demo")
    
    # Simulate 50 markets with realistic-ish outcomes
    # Key: our estimator is slightly better than random but not perfect
    n_markets = 50
    for i in range(n_markets):
        # Generate a random market
        true_prob = np.random.beta(2, 2)  # True probability (unknown to us)
        market_price = true_prob + np.random.normal(0, 0.10)
        market_price = np.clip(market_price, 0.05, 0.95)
        
        # Our estimate: market price + some signal + noise
        # Signal: we see ~30% of the true edge
        signal = (true_prob - market_price) * 0.30
        noise = np.random.normal(0, 0.05)
        our_estimate = market_price + signal + noise
        our_estimate = np.clip(our_estimate, 0.05, 0.95)
        
        confidence = 0.4 + np.random.uniform(0, 0.4)
        
        est = ProbabilityEstimate(
            market_id=f"sim-{i:03d}",
            question=f"Simulated market #{i+1}",
            probability=our_estimate,
            market_price=market_price,
            edge=our_estimate - market_price,
            confidence=confidence,
        )
        
        pos = kelly_criterion(
            estimated_prob=our_estimate,
            market_price=market_price,
            bankroll=trader.bankroll,
            kelly_fraction=0.25,
            min_edge=0.05,
            confidence=confidence,
        )
        
        trade = trader.enter_trade(est, pos)
        
        if trade:
            # Resolve: outcome determined by true probability
            outcome = np.random.random() < true_prob
            trader.resolve_trade(trade.trade_id, outcome)
    
    # Print full report
    print(trader.report())
    
    # ─── Summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SYSTEM READY")
    print("=" * 70)
    print("""
  Next steps to go live with real Polymarket data:
  
  1. Run: python run_pipeline.py --scan-only
     → Scans live markets and shows opportunities
     
  2. Run: python run_pipeline.py
     → Full pipeline with paper trading
     
  3. Add your own estimators in core/probability.py:
     - LLM-based probability estimation
     - News/sentiment analysis  
     - Historical base rate lookups
     - Cross-platform arbitrage signals
     
  4. Track your Brier score until you have 50+ resolved 
     predictions and a score below 0.20
     
  5. Only then: set up wallet auth and move to real trades
""")


if __name__ == "__main__":
    main()
