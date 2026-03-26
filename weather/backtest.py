"""
Weather Backtest — Historical Edge Validation
Reconstructs what would have happened over past months of resolved
temperature markets using archived forecast data and historical prices.

This is the go/no-go gate: if the backtest shows no systematic edge,
the weather module is not worth building further.

Usage:
    python -m weather.backtest
    python -m weather.backtest --days 60 --city NYC
    python -m weather.backtest --report-only
"""

import json
import logging
import os
import time
import argparse
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests

from weather.config import (
    STATIONS, StationInfo,
    WEATHER_MIN_EDGE, WEATHER_KELLY_FRACTION,
    BACKTEST_LOOKBACK_DAYS, BACKTEST_LEAD_HOURS, BACKTEST_MIN_MODELS,
    WEATHER_DATA_DIR, BACKTEST_FILE,
)
from weather.models import OpenMeteoClient
from weather.utils import (
    parse_question, parse_slug, build_event_slug,
    parse_buckets_from_outcomes, Bucket,
    compute_bucket_probabilities, compute_bucket_probs_from_point_forecasts,
    model_agreement, classify_confidence, brier_score,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. DATA CLASSES
# ═══════════════════════════════════════════════════════════

@dataclass
class ResolvedMarket:
    """A resolved temperature market with all outcomes and metadata."""
    city: str
    target_date: str            # ISO format date string
    event_slug: str
    unit: str
    bucket_size: int
    
    # Per-bucket data: {bucket_label: {token_id, final_price, outcome}}
    buckets: list[dict] = field(default_factory=list)
    
    # Resolution
    actual_bucket: str = ""     # Which bucket won (final_price ≈ 1.0)
    total_volume: float = 0.0


@dataclass
class BacktestResult:
    """Result for a single resolved market at a specific lead time."""
    city: str
    target_date: str
    lead_hours: int
    
    # Model assessment
    model_bucket_probs: dict[str, float] = field(default_factory=dict)
    model_brier_score: float = 1.0
    model_top_bucket: str = ""
    model_top_prob: float = 0.0
    
    # Market assessment
    market_bucket_probs: dict[str, float] = field(default_factory=dict)
    market_brier_score: float = 1.0
    market_top_bucket: str = ""
    market_top_prob: float = 0.0
    
    # Ground truth
    actual_bucket: str = ""
    actual_temperature: Optional[float] = None
    
    # Edge analysis
    brier_delta: float = 0.0        # market_brier - model_brier (positive = model wins)
    max_edge: float = 0.0           # Largest model_prob - market_price for any bucket
    tradeable_buckets: int = 0      # Buckets where edge > MIN_EDGE
    simulated_pnl: float = 0.0     # What Kelly-sized trades would have returned
    
    # Model diagnostics
    model_agreement_score: float = 0.0
    ensemble_std: float = 0.0
    confidence_tier: str = "LOW"
    per_model_forecasts: dict[str, float] = field(default_factory=dict)
    station_bias: dict[str, float] = field(default_factory=dict)
    
    # Data quality
    n_models: int = 0
    n_ensemble_members: int = 0
    data_quality: str = "ok"        # "ok", "partial", "sparse"


# ═══════════════════════════════════════════════════════════
# 2. MARKET ENUMERATION
# ═══════════════════════════════════════════════════════════

class MarketEnumerator:
    """
    Finds resolved temperature markets on Polymarket.
    
    Two strategies, tried in order:
    1. Tag-based bulk fetch: GET /events?tag_id=X&closed=true (fast, gets all at once)
    2. Slug-based lookup: GET /events/slug/{slug} (fallback, one at a time)
    
    The tag-based approach is preferred because it finds markets in a handful
    of paginated calls rather than one per city×date.
    """
    
    GAMMA_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"
    
    def __init__(self, rate_limit_delay: float = 0.5):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketEdge-Weather/1.0",
        })
        self.rate_limit_delay = rate_limit_delay
        self._last_request = 0.0
        self._temperature_tag_id: Optional[int] = None
    
    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request = time.time()
    
    def _get(self, url: str, params: dict = None) -> Optional[dict | list]:
        self._throttle()
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data if data else None
        except requests.RequestException as e:
            logger.debug(f"Request failed: {url} — {e}")
            return None
    
    # ─── Tag Discovery ───────────────────────────────────
    
    def discover_temperature_tag(self) -> Optional[int]:
        """
        Get the tag_id for temperature markets.
        
        Uses hardcoded IDs from config (discovered from actual market metadata).
        The /tags endpoint doesn't reliably list these tags, so we skip the
        API call and use known values directly.
        
        Falls back to POLYMARKET_TAG_WEATHER if the temperature tag doesn't work.
        """
        from weather.config import POLYMARKET_TAG_TEMPERATURE, POLYMARKET_TAG_WEATHER
        
        if self._temperature_tag_id:
            return self._temperature_tag_id
        
        # Verify the temperature tag works with a quick probe
        probe = self._get(f"{self.GAMMA_URL}/events", params={
            "tag_id": POLYMARKET_TAG_TEMPERATURE,
            "closed": "true",
            "limit": 1,
        })
        
        if probe and isinstance(probe, list) and len(probe) > 0:
            self._temperature_tag_id = POLYMARKET_TAG_TEMPERATURE
            logger.info(f"Using tag_id={POLYMARKET_TAG_TEMPERATURE} (Daily Temperature)")
            return self._temperature_tag_id
        
        # Fallback to broader Weather tag
        probe2 = self._get(f"{self.GAMMA_URL}/events", params={
            "tag_id": POLYMARKET_TAG_WEATHER,
            "closed": "true",
            "limit": 1,
        })
        
        if probe2 and isinstance(probe2, list) and len(probe2) > 0:
            self._temperature_tag_id = POLYMARKET_TAG_WEATHER
            logger.info(f"Using tag_id={POLYMARKET_TAG_WEATHER} (Weather)")
            return self._temperature_tag_id
        
        logger.warning("Neither temperature nor weather tag returned results")
        return None
    
    # ─── Primary: Tag-Based Bulk Enumeration ─────────────
    
    def enumerate_resolved_markets(
        self,
        lookback_days: int = BACKTEST_LOOKBACK_DAYS,
        cities: list[str] = None,
        tag_id: int = None,
    ) -> list[ResolvedMarket]:
        """
        Find all resolved temperature markets.
        
        Primary strategy: tag-based bulk fetch with closed=true.
        Fallback: slug-by-slug lookup if tag discovery fails.
        """
        if cities is None:
            cities = list(STATIONS.keys())
        
        city_set = set(c.lower() for c in cities)
        
        # Try tag-based approach first
        if tag_id is None:
            tag_id = self.discover_temperature_tag()
        
        if tag_id:
            markets = self._enumerate_by_tag(
                tag_id=tag_id,
                lookback_days=lookback_days,
                city_filter=city_set,
            )
            if markets:
                return markets
            logger.info("Tag-based enumeration returned no results, falling back to slug-based")
        
        # Fallback to slug-based
        return self._enumerate_by_slug(
            lookback_days=lookback_days,
            cities=cities,
        )
    
    def _enumerate_by_tag(
        self,
        tag_id: int,
        lookback_days: int,
        city_filter: set[str],
    ) -> list[ResolvedMarket]:
        """
        Fetch resolved temperature events using tag_id + closed=true.
        Paginates through results and filters by question pattern + date range.
        """
        from weather.utils import parse_question
        
        today = date.today()
        cutoff = today - timedelta(days=lookback_days)
        
        logger.info(f"Fetching resolved temperature events (tag_id={tag_id}, closed=true)...")
        
        all_events = []
        offset = 0
        page_size = 50
        max_pages = 40  # Safety limit: 2000 events max
        
        for page in range(max_pages):
            data = self._get(f"{self.GAMMA_URL}/events", params={
                "tag_id": tag_id,
                "closed": "true",
                "limit": page_size,
                "offset": offset,
                "order": "id",
                "ascending": "false",
            })
            
            if not data or not isinstance(data, list) or len(data) == 0:
                break
            
            all_events.extend(data)
            offset += page_size
            logger.info(f"  Page {page + 1}: fetched {len(data)} events (total: {len(all_events)})")
            
            if len(data) < page_size:
                break  # Last page
        
        logger.info(f"  Total events fetched: {len(all_events)}")
        
        # Parse and filter
        markets = []
        for event in all_events:
            raw_markets = event.get("markets", [])
            if not raw_markets:
                continue
            
            # Use the first market's question to identify city/date
            first_q = raw_markets[0].get("question", "") if raw_markets else ""
            parsed = parse_question(first_q)
            if not parsed or not parsed.station:
                continue
            
            # Date filter
            if parsed.target_date < cutoff or parsed.target_date >= today:
                continue
            
            # City filter
            if city_filter and parsed.city.lower() not in city_filter:
                continue
            
            station = parsed.station
            market = self._parse_event_to_resolved(
                event, raw_markets, station, parsed.target_date,
            )
            if market:
                markets.append(market)
        
        logger.info(f"Found {len(markets)} resolved temperature markets after filtering")
        return markets
    
    def _enumerate_by_slug(
        self,
        lookback_days: int,
        cities: list[str],
    ) -> list[ResolvedMarket]:
        """Fallback: construct slugs and fetch one by one."""
        today = date.today()
        markets = []
        total_slugs = lookback_days * len(cities)
        checked = 0
        
        logger.info(f"Slug-based enumeration: {len(cities)} cities × {lookback_days} days")
        
        for day_offset in range(1, lookback_days + 1):
            target = today - timedelta(days=day_offset)
            
            for city_key in cities:
                station = STATIONS[city_key]
                slug = build_event_slug(station.city, target)
                checked += 1
                
                if checked % 50 == 0:
                    logger.info(f"  Progress: {checked}/{total_slugs} slugs checked, {len(markets)} found")
                
                market = self._fetch_resolved_event_by_slug(slug, station, target)
                if market:
                    markets.append(market)
        
        logger.info(f"Found {len(markets)} resolved temperature markets from {total_slugs} checked")
        return markets
    
    def _fetch_resolved_event_by_slug(
        self, slug: str, station: StationInfo, target_date: date,
    ) -> Optional[ResolvedMarket]:
        """Fetch a specific event by slug and extract resolved market data."""
        # Correct endpoint: /events/slug/{slug}
        data = self._get(f"{self.GAMMA_URL}/events/slug/{slug}")
        if not data:
            return None
        
        # The /events/slug/ endpoint returns the event directly (or a list)
        event = data
        if isinstance(data, list):
            event = data[0] if data else None
        if not event:
            return None
        
        raw_markets = event.get("markets", [])
        if not raw_markets and "question" in event:
            raw_markets = [event]
        
        return self._parse_event_to_resolved(event, raw_markets, station, target_date)
    
    # ─── Event Parsing ───────────────────────────────────
    
    def _parse_event_to_resolved(
        self,
        event: dict,
        raw_markets: list[dict],
        station: StationInfo,
        target_date: date,
    ) -> Optional[ResolvedMarket]:
        """Parse a Gamma event response into a ResolvedMarket."""
        if not raw_markets:
            return None
        
        buckets = []
        actual_bucket = ""
        total_volume = 0.0
        event_slug = event.get("slug", "")
        
        for m in raw_markets:
            outcomes = m.get("outcomes", "[]")
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except (json.JSONDecodeError, TypeError):
                    outcomes = []
            
            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except (json.JSONDecodeError, TypeError):
                    prices = []
            prices = [float(p) for p in prices]
            
            token_ids = m.get("clobTokenIds", "[]")
            if isinstance(token_ids, str):
                try:
                    token_ids = json.loads(token_ids)
                except (json.JSONDecodeError, TypeError):
                    token_ids = []
            
            question = m.get("question", "")
            closed = m.get("closed", False)
            vol = float(m.get("volume", 0) or 0)
            total_volume += vol
            
            # In multi-outcome events, each "market" is one bucket
            # The question is like "Highest temperature in NYC on March 24?"
            # and the outcomes are ["Yes", "No"] for each bucket sub-market,
            # or the outcomes list contains the actual bucket labels
            
            if len(outcomes) == 2 and len(prices) == 2:
                # Binary sub-market: the question or groupItemTitle IS the bucket
                bucket_label = m.get("groupItemTitle", "")
                if not bucket_label:
                    # Try to extract from the question
                    bucket_label = question
                
                yes_price = prices[0]
                yes_token = token_ids[0] if token_ids else ""
                
                bucket_info = {
                    "label": bucket_label,
                    "token_id": yes_token,
                    "final_price": yes_price,
                    "volume": vol,
                }
                buckets.append(bucket_info)
                
                # The winning bucket has price ≈ 1.0
                if closed and yes_price > 0.95:
                    actual_bucket = bucket_label
            
            elif len(outcomes) > 2:
                # Multi-outcome market: outcomes list has bucket labels
                for i, outcome_label in enumerate(outcomes):
                    price = prices[i] if i < len(prices) else 0.0
                    token = token_ids[i] if i < len(token_ids) else ""
                    
                    bucket_info = {
                        "label": outcome_label,
                        "token_id": token,
                        "final_price": price,
                        "volume": vol,
                    }
                    buckets.append(bucket_info)
                    
                    if closed and price > 0.95:
                        actual_bucket = outcome_label
        
        if not buckets or not actual_bucket:
            return None
        
        return ResolvedMarket(
            city=station.city,
            target_date=target_date.isoformat(),
            event_slug=event_slug,
            unit=station.unit,
            bucket_size=station.bucket_size,
            buckets=buckets,
            actual_bucket=actual_bucket,
            total_volume=total_volume,
        )
    
    def get_price_history(
        self, token_id: str, start_ts: int, end_ts: int, fidelity: int = 60,
    ) -> list[dict]:
        """
        Fetch CLOB price history for a token in a time range.
        
        Args:
            token_id: CLOB token ID
            start_ts: Unix timestamp for start
            end_ts: Unix timestamp for end
            fidelity: Resolution in minutes (60 = hourly)
        
        Returns:
            List of {"t": unix_timestamp, "p": price} dicts
        """
        data = self._get(f"{self.CLOB_URL}/prices-history", {
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": fidelity,
        })
        if isinstance(data, dict):
            return data.get("history", [])
        return data if isinstance(data, list) else []


# ═══════════════════════════════════════════════════════════
# 3. BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════

class WeatherBacktest:
    """
    Runs the full historical backtest.
    
    For each resolved temperature market:
    1. Get what the weather models predicted at the time (historical forecasts)
    2. Get what the market was pricing at the time (historical CLOB prices)
    3. Compute model-implied bucket probabilities
    4. Compare model vs. market Brier scores
    5. Simulate P&L from trading the edge
    
    Usage:
        bt = WeatherBacktest()
        results = bt.run(lookback_days=90)
        bt.print_report(results)
        
        # With bias correction:
        from weather.bias import BiasTable
        table = BiasTable.load()
        bt = WeatherBacktest(bias_table=table)
        results = bt.run(lookback_days=90)
    """
    
    def __init__(
        self,
        min_edge: float = WEATHER_MIN_EDGE,
        kelly_fraction: float = WEATHER_KELLY_FRACTION,
        bankroll: float = 1000.0,
        bias_table: "Optional[BiasTable]" = None,
    ):
        self.enumerator = MarketEnumerator()
        self.weather_client = OpenMeteoClient()
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.bankroll = bankroll
        self.bias_table = bias_table
    
    def run(
        self,
        lookback_days: int = BACKTEST_LOOKBACK_DAYS,
        cities: list[str] = None,
        lead_hours: list[int] = None,
        save: bool = True,
    ) -> list[BacktestResult]:
        """
        Run the full backtest pipeline.
        
        Args:
            lookback_days: How many days back to search for resolved markets
            cities: List of city keys to include (None = all)
            lead_hours: Lead times to evaluate (None = config defaults)
            save: Whether to save results to disk
        
        Returns:
            List of BacktestResult objects
        """
        if lead_hours is None:
            lead_hours = BACKTEST_LEAD_HOURS
        
        # Step 1: Find resolved markets
        logger.info("=" * 60)
        logger.info("WEATHER BACKTEST — Phase 1A: Historical Edge Validation")
        if self.bias_table and not self.bias_table.is_empty:
            logger.info("  ✅ Bias correction ENABLED")
        else:
            logger.info("  ⬜ Bias correction OFF (run with --bias-correct)")
        logger.info("=" * 60)
        
        resolved = self.enumerator.enumerate_resolved_markets(
            lookback_days=lookback_days,
            cities=cities,
        )
        
        if not resolved:
            logger.warning("No resolved temperature markets found!")
            return []
        
        logger.info(f"\nAnalysing {len(resolved)} resolved markets at {len(lead_hours)} lead times...")
        
        # Step 2: For each market × lead time, run the analysis
        results = []
        for i, market in enumerate(resolved):
            logger.info(
                f"\n[{i+1}/{len(resolved)}] {market.city} — {market.target_date} "
                f"(actual: {market.actual_bucket})"
            )
            
            station = STATIONS.get(market.city)
            if not station:
                logger.warning(f"  Unknown city: {market.city}, skipping")
                continue
            
            target = date.fromisoformat(market.target_date)
            
            # Parse buckets from the market outcomes
            bucket_labels = [b["label"] for b in market.buckets]
            parsed_buckets = parse_buckets_from_outcomes(bucket_labels, market.unit)
            
            if not parsed_buckets:
                logger.warning(f"  Could not parse buckets: {bucket_labels[:3]}...")
                continue
            
            for lh in lead_hours:
                lead_days = max(1, lh // 24)
                
                result = self._analyse_single(
                    market=market,
                    station=station,
                    target=target,
                    parsed_buckets=parsed_buckets,
                    lead_hours=lh,
                    lead_days=lead_days,
                )
                if result:
                    results.append(result)
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Backtest complete: {len(results)} data points from {len(resolved)} markets")
        
        # Save results
        if save:
            self._save_results(results)
        
        return results
    
    def _analyse_single(
        self,
        market: ResolvedMarket,
        station: StationInfo,
        target: date,
        parsed_buckets: list[Bucket],
        lead_hours: int,
        lead_days: int,
    ) -> Optional[BacktestResult]:
        """Analyse a single market at a single lead time."""
        
        result = BacktestResult(
            city=market.city,
            target_date=market.target_date,
            lead_hours=lead_hours,
            actual_bucket=market.actual_bucket,
        )
        
        # 1. Get historical weather forecast
        forecast = self.weather_client.get_historical_forecast(
            station=station,
            target_date=target,
            lead_days=lead_days,
        )
        
        if not forecast.is_valid:
            result.data_quality = "sparse"
            logger.debug(f"  {lead_hours}h: Only {forecast.n_models_available} models, skipping")
            return None
        
        result.n_models = forecast.n_models_available
        result.n_ensemble_members = forecast.n_ensemble_members
        result.per_model_forecasts = forecast.point_forecasts
        result.ensemble_std = forecast.ensemble_std
        
        # 2. Apply bias correction if available
        corrected_forecasts = forecast.point_forecasts
        if self.bias_table and not self.bias_table.is_empty:
            corrected_forecasts = self.bias_table.correct_forecasts(
                city=market.city,
                point_forecasts=forecast.point_forecasts,
            )
        
        # 3. Compute model-implied bucket probabilities
        if forecast.ensemble_members:
            model_probs = compute_bucket_probabilities(
                forecast.ensemble_members, parsed_buckets, unit=market.unit,
            )
        else:
            # Use bias-corrected point forecasts
            model_probs = compute_bucket_probs_from_point_forecasts(
                corrected_forecasts, parsed_buckets, unit=market.unit,
            )
            result.data_quality = "partial"
        
        if not model_probs:
            return None
        
        result.model_bucket_probs = model_probs
        result.model_top_bucket = max(model_probs, key=model_probs.get)
        result.model_top_prob = model_probs[result.model_top_bucket]
        
        # 3. Get historical market prices
        market_probs = self._get_market_prices_at_lead(market, lead_hours)
        
        if not market_probs:
            # No market price data at this lead time — can't compute a meaningful
            # market Brier score. Skip rather than using final/resolved prices
            # (which would give BS ≈ 0.0 and make the comparison meaningless).
            logger.debug(f"  {lead_hours}h: No CLOB price history at this lead time, skipping")
            result.data_quality = "no_market_prices"
            return None
        
        result.market_bucket_probs = market_probs
        result.market_top_bucket = max(market_probs, key=market_probs.get) if market_probs else ""
        result.market_top_prob = market_probs.get(result.market_top_bucket, 0)
        
        # 4. Compute Brier scores
        result.model_brier_score = brier_score(model_probs, market.actual_bucket)
        result.market_brier_score = brier_score(market_probs, market.actual_bucket)
        result.brier_delta = result.market_brier_score - result.model_brier_score
        
        # 5. Model agreement and confidence
        agreement, consensus = model_agreement(
            forecast.point_forecasts, parsed_buckets,
        )
        result.model_agreement_score = agreement
        result.confidence_tier = classify_confidence(
            agreement, forecast.n_models_available,
        )
        
        # 6. Edge analysis
        edges = {}
        for label in model_probs:
            if label in market_probs:
                edges[label] = model_probs[label] - market_probs[label]
        
        if edges:
            result.max_edge = max(edges.values())
            result.tradeable_buckets = sum(1 for e in edges.values() if e > self.min_edge)
        
        # 7. Simulated P&L
        result.simulated_pnl = self._simulate_pnl(
            model_probs, market_probs, market.actual_bucket,
        )
        
        # 8. Station bias (forecast - actual, for each model)
        actual_temp = self.weather_client.get_observed_temperature(station, target)
        if actual_temp is not None:
            result.actual_temperature = actual_temp
            for model_name, forecast_temp in forecast.point_forecasts.items():
                result.station_bias[model_name] = round(forecast_temp - actual_temp, 1)
        
        logger.info(
            f"  {lead_hours:3d}h | Model BS: {result.model_brier_score:.3f} | "
            f"Market BS: {result.market_brier_score:.3f} | "
            f"Delta: {result.brier_delta:+.3f} | "
            f"Edge: {result.max_edge:+.1%} | "
            f"Sim P&L: ${result.simulated_pnl:+.2f} | "
            f"{result.confidence_tier}"
        )
        
        return result
    
    def _get_market_prices_at_lead(
        self, market: ResolvedMarket, lead_hours: int,
    ) -> dict[str, float]:
        """
        Fetch market prices from CLOB history at the specified lead time.
        
        Uses a progressively wider search window — markets with less
        trading activity may need a wider window to find any price data.
        
        Returns dict of {bucket_label: price_at_lead_time}, or empty dict.
        """
        target = date.fromisoformat(market.target_date)
        
        # Calculate the timestamp for "lead_hours before end of target date"
        target_end = datetime(
            target.year, target.month, target.day, 23, 59, 59,
            tzinfo=timezone.utc,
        )
        snapshot_time = target_end - timedelta(hours=lead_hours)
        snapshot_ts = int(snapshot_time.timestamp())
        
        # Try progressively wider windows: 6h, 12h, 24h
        for window_hours in [6, 12, 24]:
            window = window_hours * 3600
            
            prices = {}
            for bucket in market.buckets:
                token_id = bucket.get("token_id", "")
                if not token_id:
                    continue
                
                history = self.enumerator.get_price_history(
                    token_id=token_id,
                    start_ts=snapshot_ts - window,
                    end_ts=snapshot_ts + window,
                    fidelity=60,  # 1-hour resolution
                )
                
                if history:
                    # Find the price closest to our snapshot time
                    closest = min(history, key=lambda h: abs(h["t"] - snapshot_ts))
                    prices[bucket["label"]] = float(closest["p"])
            
            # Need prices for at least half the buckets to be meaningful
            if len(prices) >= len(market.buckets) // 2:
                break
        
        if not prices:
            return {}
        
        # Normalize
        total = sum(prices.values())
        if total > 0 and len(prices) > 1:
            prices = {k: v / total for k, v in prices.items()}
        
        return prices
    
    def _simulate_pnl(
        self,
        model_probs: dict[str, float],
        market_probs: dict[str, float],
        actual_bucket: str,
    ) -> float:
        """
        Simulate P&L from trading all buckets with positive edge.
        
        Uses fractional Kelly sizing.
        """
        total_pnl = 0.0
        
        for label in model_probs:
            if label not in market_probs:
                continue
            
            model_p = model_probs[label]
            market_p = market_probs[label]
            edge = model_p - market_p
            
            if edge <= self.min_edge or market_p <= 0.01 or market_p >= 0.99:
                continue
            
            # Kelly fraction for this bet
            # f* = (bp - q) / b where b = (1/market_p - 1), p = model_p, q = 1 - model_p
            b = (1.0 / market_p) - 1.0
            f_star = (b * model_p - (1.0 - model_p)) / b
            f_star = max(0, f_star) * self.kelly_fraction
            
            # Cap position size
            dollar_amount = min(f_star * self.bankroll, 50.0)
            shares = dollar_amount / market_p
            
            # P&L: if this bucket won, profit = shares * (1 - market_p)
            #       if this bucket lost, loss = -dollar_amount
            if label == actual_bucket:
                total_pnl += shares * (1.0 - market_p)
            else:
                total_pnl -= dollar_amount
        
        return round(total_pnl, 2)
    
    def _save_results(self, results: list[BacktestResult]):
        """Save backtest results to JSON."""
        os.makedirs(WEATHER_DATA_DIR, exist_ok=True)
        
        serializable = []
        for r in results:
            d = asdict(r)
            serializable.append(d)
        
        with open(BACKTEST_FILE, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        
        logger.info(f"Saved {len(results)} results to {BACKTEST_FILE}")
    
    # ─── Reporting ───────────────────────────────────────
    
    def print_report(self, results: list[BacktestResult]):
        """Print a comprehensive backtest summary."""
        if not results:
            print("No results to report.")
            return
        
        print("\n" + "=" * 70)
        print("  WEATHER BACKTEST REPORT — Historical Edge Analysis")
        print("=" * 70)
        
        # Overall metrics
        model_briers = [r.model_brier_score for r in results]
        market_briers = [r.market_brier_score for r in results]
        deltas = [r.brier_delta for r in results]
        pnls = [r.simulated_pnl for r in results]
        
        print(f"\n  Total data points:    {len(results)}")
        print(f"  Unique markets:       {len(set((r.city, r.target_date) for r in results))}")
        print(f"  Cities covered:       {len(set(r.city for r in results))}")
        
        print(f"\n{'─' * 70}")
        print(f"  {'BRIER SCORES':^66}")
        print(f"{'─' * 70}")
        print(f"  Model (lower=better):   {np.mean(model_briers):.4f} ± {np.std(model_briers):.4f}")
        print(f"  Market (baseline):      {np.mean(market_briers):.4f} ± {np.std(market_briers):.4f}")
        print(f"  Delta (pos=model wins): {np.mean(deltas):+.4f} ± {np.std(deltas):.4f}")
        print(f"  Model wins:             {sum(1 for d in deltas if d > 0)}/{len(deltas)} "
              f"({sum(1 for d in deltas if d > 0)/len(deltas):.0%})")
        
        print(f"\n{'─' * 70}")
        print(f"  {'SIMULATED P&L':^66}")
        print(f"{'─' * 70}")
        print(f"  Total P&L:              ${sum(pnls):+,.2f}")
        print(f"  Mean per trade:         ${np.mean(pnls):+.2f}")
        print(f"  Win rate:               {sum(1 for p in pnls if p > 0)/len(pnls):.0%}")
        print(f"  Avg win:                ${np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0:+.2f}")
        print(f"  Avg loss:               ${np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0:+.2f}")
        
        # By lead time
        lead_times = sorted(set(r.lead_hours for r in results))
        if len(lead_times) > 1:
            print(f"\n{'─' * 70}")
            print(f"  {'BY LEAD TIME':^66}")
            print(f"{'─' * 70}")
            print(f"  {'Lead':>6} | {'N':>4} | {'Model BS':>9} | {'Mkt BS':>9} | {'Delta':>8} | {'Sim P&L':>10}")
            print(f"  {'─' * 62}")
            
            for lh in lead_times:
                lr = [r for r in results if r.lead_hours == lh]
                ld = [r.brier_delta for r in lr]
                lp = [r.simulated_pnl for r in lr]
                print(
                    f"  {lh:5d}h | {len(lr):4d} | "
                    f"{np.mean([r.model_brier_score for r in lr]):9.4f} | "
                    f"{np.mean([r.market_brier_score for r in lr]):9.4f} | "
                    f"{np.mean(ld):+8.4f} | "
                    f"${sum(lp):+10.2f}"
                )
        
        # By city
        cities = sorted(set(r.city for r in results))
        if len(cities) > 1:
            print(f"\n{'─' * 70}")
            print(f"  {'BY CITY':^66}")
            print(f"{'─' * 70}")
            print(f"  {'City':<12} | {'N':>4} | {'Delta':>8} | {'Sim P&L':>10} | {'Agree':>6}")
            print(f"  {'─' * 52}")
            
            for city in cities:
                cr = [r for r in results if r.city == city]
                cd = [r.brier_delta for r in cr]
                cp = [r.simulated_pnl for r in cr]
                ca = [r.model_agreement_score for r in cr]
                print(
                    f"  {city:<12} | {len(cr):4d} | "
                    f"{np.mean(cd):+8.4f} | "
                    f"${sum(cp):+10.2f} | "
                    f"{np.mean(ca):5.0%}"
                )
        
        # By confidence tier
        tiers = sorted(set(r.confidence_tier for r in results))
        if len(tiers) > 1:
            print(f"\n{'─' * 70}")
            print(f"  {'BY CONFIDENCE TIER':^66}")
            print(f"{'─' * 70}")
            print(f"  {'Tier':<10} | {'N':>4} | {'Delta':>8} | {'Win%':>6} | {'Sim P&L':>10}")
            print(f"  {'─' * 48}")
            
            for tier in ["LOCK", "STRONG", "SAFE", "NEAR_SAFE", "LOW"]:
                tr = [r for r in results if r.confidence_tier == tier]
                if not tr:
                    continue
                td = [r.brier_delta for r in tr]
                tp = [r.simulated_pnl for r in tr]
                tw = sum(1 for p in tp if p > 0) / len(tp)
                print(
                    f"  {tier:<10} | {len(tr):4d} | "
                    f"{np.mean(td):+8.4f} | "
                    f"{tw:5.0%} | "
                    f"${sum(tp):+10.2f}"
                )
        
        # Station bias summary
        bias_data = {}
        for r in results:
            for model, bias in r.station_bias.items():
                key = (r.city, model)
                if key not in bias_data:
                    bias_data[key] = []
                bias_data[key].append(bias)
        
        if bias_data:
            print(f"\n{'─' * 70}")
            print(f"  {'STATION BIAS (forecast - actual)':^66}")
            print(f"{'─' * 70}")
            
            # Aggregate by city
            city_biases = {}
            for (city, model), biases in bias_data.items():
                if city not in city_biases:
                    city_biases[city] = {}
                city_biases[city][model] = {
                    "mean": np.mean(biases),
                    "std": np.std(biases),
                    "n": len(biases),
                }
            
            models_seen = sorted(set(m for (c, m) in bias_data.keys()))[:5]
            header = f"  {'City':<12} |" + "|".join(f" {m[:8]:>8} " for m in models_seen)
            print(header)
            print(f"  {'─' * (len(header) - 2)}")
            
            for city in sorted(city_biases.keys()):
                row = f"  {city:<12} |"
                for model in models_seen:
                    if model in city_biases[city]:
                        b = city_biases[city][model]
                        row += f" {b['mean']:+7.1f}° "
                    else:
                        row += "     N/A  "
                    row += "|"
                print(row)
        
        # Go/no-go verdict
        mean_delta = np.mean(deltas)
        total_pnl = sum(pnls)
        
        print(f"\n{'=' * 70}")
        if mean_delta > 0.03 and total_pnl > 0:
            print(f"  ✅ GO — Systematic edge detected (Δ={mean_delta:+.4f}, P&L=${total_pnl:+,.2f})")
            print(f"  Proceed to Phase 2: Weather Enricher Integration")
        elif mean_delta > 0.01:
            print(f"  ⚠️  MARGINAL — Some edge detected (Δ={mean_delta:+.4f}, P&L=${total_pnl:+,.2f})")
            print(f"  Consider: extend backtest period, tune parameters, check by city")
        else:
            print(f"  ❌ NO-GO — Insufficient edge (Δ={mean_delta:+.4f}, P&L=${total_pnl:+,.2f})")
            print(f"  The market is already efficient for temperature predictions.")
        print("=" * 70)


# ═══════════════════════════════════════════════════════════
# 4. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Weather Backtest — Historical Edge Validation"
    )
    parser.add_argument(
        "--days", type=int, default=BACKTEST_LOOKBACK_DAYS,
        help=f"Lookback period in days (default: {BACKTEST_LOOKBACK_DAYS})"
    )
    parser.add_argument(
        "--city", type=str, default=None,
        help="Single city to backtest (e.g. NYC, London). Default: all cities"
    )
    parser.add_argument(
        "--min-edge", type=float, default=WEATHER_MIN_EDGE,
        help=f"Minimum edge threshold for simulated trades (default: {WEATHER_MIN_EDGE})"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Only print report from previously saved results (no API calls)"
    )
    parser.add_argument(
        "--bias-correct", action="store_true",
        help="Apply station bias correction (requires prior backtest run to build table)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Load bias table if requested
    bias_table = None
    if args.bias_correct:
        from weather.bias import BiasTable
        
        # Try loading a saved table first
        bias_table = BiasTable.load()
        
        if bias_table.is_empty:
            # Build from backtest results if no saved table exists
            if os.path.exists(BACKTEST_FILE):
                logger.info("No saved bias table — building from backtest results...")
                bias_table = BiasTable.from_backtest_results()
                bias_table.save()
                bias_table.print_table()
            else:
                logger.warning(
                    "Cannot bias-correct: no bias table and no backtest results. "
                    "Run without --bias-correct first, then re-run with it."
                )
                bias_table = None
    
    bt = WeatherBacktest(min_edge=args.min_edge, bias_table=bias_table)
    
    if args.report_only:
        # Load saved results
        if not os.path.exists(BACKTEST_FILE):
            print(f"No saved results at {BACKTEST_FILE}. Run backtest first.")
            return
        
        with open(BACKTEST_FILE) as f:
            raw = json.load(f)
        
        results = []
        for d in raw:
            r = BacktestResult(**{k: v for k, v in d.items() if k in BacktestResult.__dataclass_fields__})
            results.append(r)
        
        bt.print_report(results)
        return
    
    cities = [args.city] if args.city else None
    results = bt.run(lookback_days=args.days, cities=cities)
    bt.print_report(results)


if __name__ == "__main__":
    main()