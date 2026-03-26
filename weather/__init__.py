"""
Weather Module — Specialized pipeline for daily temperature prediction markets.

Polymarket runs daily temperature markets for ~10 cities that resolve on
Weather Underground station data. This module exploits the structural edge
available from numerical weather models: multi-model ensemble forecasts 
provide calibrated probability distributions over temperature buckets that
consistently outperform the crowd pricing.

Usage:
    # Run historical backtest (Phase 1A)
    python -m weather.backtest

    # Run forward data collector (Phase 1B)
    python -m weather.data_collector
"""
