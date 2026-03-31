"""
Polymarket Edge — Core modules.

Modules:
    api_client          Unified Polymarket API client (Gamma + CLOB + Data)
    market_scanner      Scans and scores tradeable opportunities
    context_enricher    Multi-source enrichment pipeline
    llm_estimator       Claude-powered probability estimation + calibration
    probability         Ensemble framework, Bayesian updating, calibration tracking
    kelly               Kelly Criterion position sizing (fractional Kelly)
    paper_trader        Paper trading engine with P&L and Brier score tracking
    whale_profiler      Behavioral profiling of top traders (strategy classification,
                        category-specific credibility, win rates, persistent storage)
"""
