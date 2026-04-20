"""
Weather Market Scanner & Paper Trader
Standalone pipeline for temperature prediction markets.

Operates independently from the main Polymarket Edge pipeline because
weather markets have fundamentally different characteristics:
    - Multi-outcome (8-10 temperature buckets per event)
    - Probability computed from numerical weather models, not LLM
    - Positions sized across correlated buckets within the same event
    - Triggered by model update cycles, not general market scanning

Usage:
    python -m weather.scanner                    # Scan + show opportunities
    python -m weather.scanner --paper-trade      # Scan + enter paper trades
    python -m weather.scanner --check            # Check open trades for resolution
    python -m weather.scanner --report           # Paper trading performance report
"""

import json
import logging
import os
import time
import argparse
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests

from weather.config import (
    STATIONS, StationInfo,
    WEATHER_MIN_EDGE, WEATHER_KELLY_FRACTION,
    WEATHER_MAX_BUCKET_POSITION, WEATHER_MAX_CITY_EXPOSURE,
    WEATHER_MAX_PORTFOLIO_PCT, WEATHER_DATA_DIR,
    POLYMARKET_TAG_TEMPERATURE, BIAS_TABLE_FILE,
)
from weather.models import OpenMeteoClient
from weather.bias import BiasTable
from weather.utils import (
    parse_question, parse_buckets_from_outcomes, Bucket,
    compute_bucket_probabilities, compute_bucket_probs_from_point_forecasts,
    model_agreement, classify_confidence,
)
from weather.trade_logger import WeatherEventLogger

logger = logging.getLogger(__name__)

# All cities we scan — keeps diagnostic data flowing for model accuracy tracking
SCAN_CITIES = {"NYC", "London", "Hong Kong", "Atlanta", "Beijing", "Denver", "Houston"}

# Cities validated as profitable in paper trading (diagnostics 2026-04-22, n=220 resolved)
# Removed: Hong Kong (low MAE = efficient market, -$356), Houston (-$386), London (-$518)
TRADEABLE_CITIES = {"NYC", "Atlanta", "Beijing", "Denver"}


# ═══════════════════════════════════════════════════════════
# 1. DATA STRUCTURES
# ═══════════════════════════════════════════════════════════

@dataclass
class WeatherEvent:
    """A temperature market event with all its bucket sub-markets."""
    city: str
    station: StationInfo
    target_date: date
    event_slug: str
    
    # Per-bucket data from Polymarket
    buckets: list[dict] = field(default_factory=list)
    # Each bucket: {label, token_id, market_id, market_price, outcomes}
    
    # Weather model forecast
    point_forecasts: dict[str, float] = field(default_factory=dict)
    point_forecasts_corrected: dict[str, float] = field(default_factory=dict)
    ensemble_members: list[float] = field(default_factory=list)
    ensemble_mean: float = 0.0
    ensemble_std: float = 0.0
    n_models: int = 0
    n_ensemble: int = 0
    bias_corrected: bool = False
    
    # Model-implied probability distribution
    model_probs: dict[str, float] = field(default_factory=dict)
    
    # Consensus
    agreement: float = 0.0
    consensus_bucket: str = ""
    confidence_tier: str = "LOW"
    
    # Lead time
    lead_hours: int = 0
    
    @property
    def tradeable_edges(self) -> list[dict]:
        """Buckets where model probability exceeds market price by MIN_EDGE."""
        edges = []
        for bucket in self.buckets:
            label = bucket["label"]
            market_price = bucket["market_price"]
            model_prob = self.model_probs.get(label, 0.0)
            edge = model_prob - market_price
            
            if edge > WEATHER_MIN_EDGE and 0.01 < market_price < 0.99:
                edges.append({
                    "label": label,
                    "market_price": market_price,
                    "model_prob": model_prob,
                    "edge": edge,
                    "token_id": bucket.get("token_id", ""),
                    "market_id": bucket.get("market_id", ""),
                })
        
        edges.sort(key=lambda e: e["edge"], reverse=True)
        return edges


@dataclass
class WeatherTrade:
    """A sized trade opportunity for a specific bucket."""
    city: str
    target_date: str
    bucket_label: str
    side: str                   # Always "YES" for weather (buying the bucket)
    market_price: float
    model_prob: float
    edge: float
    dollar_amount: float
    shares: float
    expected_pnl: float
    confidence: float
    token_id: str
    market_id: str
    event_slug: str
    lead_hours: int
    confidence_tier: str


# ═══════════════════════════════════════════════════════════
# 2. MARKET DISCOVERY
# ═══════════════════════════════════════════════════════════

class WeatherMarketFinder:
    """
    Finds active temperature markets on Polymarket.
    
    Uses the Gamma API with tag-based filtering to find temperature
    events, then parses each event into a WeatherEvent with all
    its bucket sub-markets.
    """
    
    GAMMA_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self, rate_limit_delay: float = 0.5):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketEdge-Weather/1.0",
        })
        self.rate_limit_delay = rate_limit_delay
        self._last_request = 0.0
    
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
    
    def find_active_events(
        self,
        cities: set[str] = None,
        max_lead_hours: int = 96,
    ) -> list[WeatherEvent]:
        """
        Find all active (open) temperature market events.
        
        Fetches open events with the Daily Temperature tag,
        parses each into a WeatherEvent, filters to tradeable cities
        and the target lead time window.
        """
        if cities is None:
            cities = SCAN_CITIES
        
        city_set = {c.lower() for c in cities}
        now = datetime.now(timezone.utc)
        events = []
        
        logger.info("Scanning for active temperature markets...")
        
        # Fetch open temperature events
        offset = 0
        page_size = 50
        
        for page in range(10):  # Safety limit
            data = self._get(f"{self.GAMMA_URL}/events", params={
                "tag_id": POLYMARKET_TAG_TEMPERATURE,
                "closed": "false",
                "limit": page_size,
                "offset": offset,
            })
            
            if not data or not isinstance(data, list) or len(data) == 0:
                break
            
            for event_data in data:
                parsed_event = self._parse_event(event_data, city_set, now, max_lead_hours)
                if parsed_event:
                    events.append(parsed_event)
            
            offset += page_size
            if len(data) < page_size:
                break
        
        logger.info(f"Found {len(events)} tradeable temperature events")
        return events
    
    def _parse_event(
        self,
        event_data: dict,
        city_filter: set[str],
        now: datetime,
        max_lead_hours: int,
    ) -> Optional[WeatherEvent]:
        """Parse a Gamma API event response into a WeatherEvent."""
        raw_markets = event_data.get("markets", [])
        if not raw_markets:
            return None
        
        # Use first market's question to identify city/date
        first_q = raw_markets[0].get("question", "")
        parsed = parse_question(first_q)
        if not parsed or not parsed.station:
            return None
        
        # City filter
        if city_filter and parsed.city.lower() not in city_filter:
            return None
        
        # Lead time filter
        target_end = datetime(
            parsed.target_date.year, parsed.target_date.month, parsed.target_date.day,
            23, 59, 59, tzinfo=timezone.utc,
        )
        lead_hours = max(0, int((target_end - now).total_seconds() / 3600))
        
        if lead_hours <= 0 or lead_hours > max_lead_hours:
            return None
        
        # Parse bucket sub-markets
        buckets = []
        for m in raw_markets:
            label = m.get("groupItemTitle", "")
            if not label:
                continue
            
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
            
            if prices:
                buckets.append({
                    "label": label,
                    "market_price": prices[0],  # Yes price
                    "token_id": token_ids[0] if token_ids else "",
                    "market_id": str(m.get("id", "")),
                    "volume": float(m.get("volume", 0) or 0),
                })
        
        if not buckets:
            return None
        
        return WeatherEvent(
            city=parsed.city,
            station=parsed.station,
            target_date=parsed.target_date,
            event_slug=event_data.get("slug", ""),
            buckets=buckets,
            lead_hours=lead_hours,
        )


# ═══════════════════════════════════════════════════════════
# 3. WEATHER SCANNER PIPELINE
# ═══════════════════════════════════════════════════════════

class WeatherScanner:
    """
    Standalone weather trading pipeline.
    
    Pipeline: discover events → fetch forecasts → bias correct →
    compute probabilities → find edges → size positions → paper trade.
    
    Usage:
        scanner = WeatherScanner()
        
        # Scan and show opportunities
        events = scanner.scan()
        
        # Scan and paper trade
        trades = scanner.scan_and_trade()
        
        # Check resolutions
        scanner.check_resolutions()
    """
    
    def __init__(
        self,
        bankroll: float = 1000.0,
        min_edge: float = WEATHER_MIN_EDGE,
        kelly_fraction: float = WEATHER_KELLY_FRACTION,
        max_bucket_position: float = WEATHER_MAX_BUCKET_POSITION,
        max_city_exposure: float = WEATHER_MAX_CITY_EXPOSURE,
    ):
        self.finder = WeatherMarketFinder()
        self.weather_client = OpenMeteoClient()
        self.bias_table = BiasTable.load(BIAS_TABLE_FILE)
        
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.max_bucket_position = max_bucket_position
        self.max_city_exposure = max_city_exposure
        
        # Paper trader with weather-specific data directory
        from core.paper_trader import PaperTrader
        self.trader = PaperTrader(bankroll=bankroll, data_dir=WEATHER_DATA_DIR)
        self.event_logger = WeatherEventLogger()
    
    def scan(self) -> list[WeatherEvent]:
        """
        Full scan: discover events, fetch forecasts, compute probabilities.
        
        Returns enriched WeatherEvent objects with model probabilities
        and tradeable edges identified.
        """
        print("\n" + "=" * 70)
        print("  WEATHER SCANNER — Temperature Market Pipeline")
        print(f"  Scanning: {', '.join(sorted(SCAN_CITIES))}")
        print(f"  Trading:  {', '.join(sorted(TRADEABLE_CITIES))}")
        print(f"  Min edge: {self.min_edge:.0%} | Kelly: {self.kelly_fraction:.0%}")
        if not self.bias_table.is_empty:
            print(f"  Bias correction: ON")
        else:
            print(f"  Bias correction: OFF (run: python -m weather.bias --build)")
        print("=" * 70)
        
        # Step 1: Find active temperature events
        t0 = time.time()
        events = self.finder.find_active_events()
        
        if not events:
            print("\n  No tradeable temperature markets found.")
            return []
        
        # Step 2: Enrich each event with forecasts
        print(f"\n  Fetching forecasts for {len(events)} events...")
        
        for i, event in enumerate(events):
            self._enrich_event(event)
            
            # Log progress
            n_edges = len(event.tradeable_edges)
            edge_str = f"{n_edges} edges" if n_edges > 0 else "no edge"
            print(
                f"    {event.city:<12} {event.target_date} | "
                f"{event.lead_hours:3d}h | "
                f"{event.n_models} models + {event.n_ensemble} ensemble | "
                f"{event.confidence_tier:<10} | "
                f"{edge_str}"
            )
        
        # Summary
        total_edges = sum(len(e.tradeable_edges) for e in events)
        elapsed = time.time() - t0
        print(f"\n  Scan complete: {len(events)} events, {total_edges} tradeable buckets ({elapsed:.1f}s)")
        
        return events
    
    def scan_and_trade(self) -> list[WeatherTrade]:
        """
        Scan for opportunities and enter paper trades.
        
        Returns list of WeatherTrade objects for trades entered.
        """
        from core.probability import ProbabilityEstimate
        from core.kelly import kelly_criterion, PositionSize
        
        # First check resolutions
        self.check_resolutions()
        
        # Scan
        events = self.scan()
        
        if not events:
            return []
        
        # Step 3: Size and enter trades
        print(f"\n{'─' * 70}")
        print(f"  POSITION SIZING & PAPER TRADING")
        print(f"{'─' * 70}")
        
        all_trades = []
        trades_entered = 0
        trades_skipped = 0
        trades_deduped = 0
        
        # Build set of market_ids we already have open trades in
        open_market_ids = {
            t.market_id for t in self.trader.trades if t.status == "OPEN"
        }
        
        for event in events:
            edges = event.tradeable_edges
            confidence = self._compute_confidence(event)
            event_trades_log = []   # trade dicts for the event logger
            event_skip_reason = ""
            
            if not edges:
                # Log even events with no edge (valuable for model accuracy tracking)
                self.event_logger.log_event(event, confidence, skip_reason="no_edge")
                continue
            
            # City gate: scan-only cities get logged but not traded
            if event.city not in TRADEABLE_CITIES:
                self.event_logger.log_event(event, confidence, skip_reason="scan_only_city")
                continue
            
            # Quality gate: require deterministic models for bias correction
            # and model agreement. Ensemble-only data (0 models) means the
            # deterministic API failed — probabilities are less reliable.
            if event.n_models < 3:
                logger.info(
                    f"    ⚠ {event.city} {event.target_date}: skipping — "
                    f"only {event.n_models} deterministic models (need 3+)"
                )
                trades_skipped += 1
                continue
            
            # Track exposure for this city/date
            city_exposure = self._get_city_exposure(event.city, event.target_date)
            
            for edge_info in edges:
                # Skip if we already have an open trade on this exact bucket
                if edge_info["market_id"] in open_market_ids:
                    trades_deduped += 1
                    event_skip_reason = "deduped"
                    continue
                
                # Check city exposure cap
                if city_exposure >= self.max_city_exposure:
                    trades_skipped += 1
                    event_skip_reason = "city_cap"
                    continue
                
                # Check portfolio-level weather cap
                weather_exposure = self._get_total_weather_exposure()
                if weather_exposure >= self.trader.initial_bankroll * WEATHER_MAX_PORTFOLIO_PCT:
                    trades_skipped += 1
                    event_skip_reason = "portfolio_cap"
                    continue
                
                # Kelly sizing
                position = kelly_criterion(
                    estimated_prob=edge_info["model_prob"],
                    market_price=edge_info["market_price"],
                    bankroll=self.trader.bankroll,
                    kelly_fraction=self.kelly_fraction,
                    min_edge=self.min_edge,
                    max_position_pct=0.05,  # Tighter per-position cap for weather
                    confidence=confidence,
                )
                
                if not position.should_trade:
                    continue
                
                # Cap at max bucket position
                if position.dollar_amount > self.max_bucket_position:
                    ratio = self.max_bucket_position / position.dollar_amount
                    position = PositionSize(
                        should_trade=True,
                        side=position.side,
                        kelly_fraction=position.kelly_fraction,
                        adjusted_fraction=position.adjusted_fraction * ratio,
                        dollar_amount=self.max_bucket_position,
                        shares=position.shares * ratio,
                        entry_price=position.entry_price,
                        max_loss=self.max_bucket_position,
                        expected_profit=position.expected_profit * ratio,
                        expected_return_pct=position.expected_return_pct,
                        risk_reward_ratio=position.risk_reward_ratio,
                        edge=position.edge,
                    )
                
                # Build ProbabilityEstimate for the PaperTrader
                estimate = ProbabilityEstimate(
                    market_id=edge_info["market_id"],
                    question=f"{event.city} {event.target_date} — {edge_info['label']}",
                    probability=edge_info["model_prob"],
                    market_price=edge_info["market_price"],
                    edge=edge_info["edge"],
                    confidence=confidence,
                    components={
                        "city": event.city,
                        "target_date": event.target_date.isoformat(),
                        "bucket": edge_info["label"],
                        "lead_hours": event.lead_hours,
                        "n_models": event.n_models,
                        "n_ensemble": event.n_ensemble,
                        "confidence_tier": event.confidence_tier,
                        "bias_corrected": event.bias_corrected,
                    },
                )
                
                # Enter paper trade
                trade = self.trader.enter_trade(estimate, position)
                if trade:
                    city_exposure += position.dollar_amount
                    trades_entered += 1
                    open_market_ids.add(edge_info["market_id"])
                    
                    # Record for event logger
                    event_trades_log.append({
                        "bucket": edge_info["label"],
                        "side": "YES",
                        "dollars": round(position.dollar_amount, 2),
                        "shares": round(position.shares, 2),
                        "model_prob": round(edge_info["model_prob"], 4),
                        "market_price": round(edge_info["market_price"], 4),
                        "edge": round(edge_info["edge"], 4),
                    })
                    
                    wt = WeatherTrade(
                        city=event.city,
                        target_date=event.target_date.isoformat(),
                        bucket_label=edge_info["label"],
                        side="YES",
                        market_price=edge_info["market_price"],
                        model_prob=edge_info["model_prob"],
                        edge=edge_info["edge"],
                        dollar_amount=position.dollar_amount,
                        shares=position.shares,
                        expected_pnl=position.expected_profit,
                        confidence=confidence,
                        token_id=edge_info["token_id"],
                        market_id=edge_info["market_id"],
                        event_slug=event.event_slug,
                        lead_hours=event.lead_hours,
                        confidence_tier=event.confidence_tier,
                    )
                    all_trades.append(wt)
                    
                    print(
                        f"    📝 {event.city:<10} {event.target_date} | "
                        f"{edge_info['label']:>15} | "
                        f"Model: {edge_info['model_prob']:.0%} vs Mkt: {edge_info['market_price']:.0%} | "
                        f"Edge: {edge_info['edge']:+.0%} | "
                        f"${position.dollar_amount:.2f}"
                    )
            
            # Log the full event snapshot (traded or not)
            self.event_logger.log_event(
                event, confidence,
                trades_entered=event_trades_log if event_trades_log else None,
                skip_reason=event_skip_reason if not event_trades_log else "",
            )
        
        # Summary
        print(f"\n{'─' * 70}")
        skip_msg = f", {trades_skipped} capped" if trades_skipped else ""
        dedup_msg = f", {trades_deduped} already open" if trades_deduped else ""
        print(f"  Entered {trades_entered} trade(s){skip_msg}{dedup_msg}")
        
        # Portfolio status
        snap = self.trader.snapshot()
        open_trades = [t for t in self.trader.trades if t.status == "OPEN"]
        print(f"\n  📊 Weather Portfolio:")
        print(f"     Bankroll: ${self.trader.bankroll:,.2f}")
        print(f"     Open trades: {len(open_trades)}")
        print(f"     Resolved: {snap.resolved_trades} | "
              f"Win rate: {snap.win_rate:.0%}")
        if snap.brier_score is not None:
            print(f"     Brier: {snap.brier_score:.4f}")
        print()
        
        return all_trades
    
    def check_resolutions(self):
        """Check open weather trades for market resolution."""
        open_trades = [t for t in self.trader.trades if t.status == "OPEN"]
        if not open_trades:
            return
        
        print(f"\n  🔍 Checking {len(open_trades)} open weather trade(s)...")
        
        finder = self.finder
        resolved_count = 0
        
        # Group resolved P&L by event_key so we can log once per event
        event_pnl: dict[str, float] = {}       # event_key → cumulative pnl
        event_buckets: dict[str, str] = {}      # event_key → winning bucket label
        event_meta: dict[str, dict] = {}        # event_key → {city, date, station}
        
        for trade in open_trades:
            # Fetch the market's current state
            market_id = trade.market_id
            data = finder._get(f"{finder.GAMMA_URL}/markets/{market_id}")
            
            if not data or not isinstance(data, dict):
                continue
            
            closed = data.get("closed", False)
            if not closed:
                continue
            
            # Check resolution
            prices = data.get("outcomePrices", "[]")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except (json.JSONDecodeError, TypeError):
                    prices = []
            prices = [float(p) for p in prices]
            
            if not prices:
                continue
            
            yes_price = prices[0]
            if yes_price > 0.95:
                outcome = True   # This bucket won
            elif yes_price < 0.05:
                outcome = False  # This bucket lost
            else:
                continue  # Not cleanly resolved
            
            pnl = self.trader.resolve_trade(trade.trade_id, outcome)
            if pnl is not None:
                resolved_count += 1
                status = "✅ WIN" if pnl > 0 else "❌ LOSS"
                print(f"    {status} {trade.question[:50]} → ${pnl:+.2f}")
                
                # Extract city/date from trade components or question
                components = getattr(trade, '_components', None)
                # Parse from question format: "City YYYY-MM-DD — BucketLabel"
                parts = trade.question.split(" — ")
                if len(parts) == 2:
                    city_date = parts[0].rsplit(" ", 1)
                    if len(city_date) == 2:
                        city, date_str = city_date
                        event_key = f"{city}_{date_str}"
                        event_pnl[event_key] = event_pnl.get(event_key, 0.0) + pnl
                        if outcome:  # This was the winning bucket
                            event_buckets[event_key] = parts[1].strip()
                        if event_key not in event_meta:
                            station = STATIONS.get(city)
                            event_meta[event_key] = {
                                "city": city, "date_str": date_str, "station": station,
                            }
        
        if resolved_count > 0:
            print(f"    Resolved {resolved_count}. Bankroll: ${self.trader.bankroll:,.2f}")
        
        # Fetch actual temperatures and update event log for resolved events
        for event_key, meta in event_meta.items():
            station = meta.get("station")
            date_str = meta["date_str"]
            actual_temp = None
            actual_bucket = event_buckets.get(event_key, "")
            
            if station:
                try:
                    from datetime import date as date_cls
                    target = date_cls.fromisoformat(date_str)
                    actual_temp = self.weather_client.get_observed_temperature(station, target)
                except Exception as e:
                    logger.debug(f"Could not fetch actual temp for {event_key}: {e}")
            
            self.event_logger.log_resolution(
                event_key=event_key,
                actual_temperature=actual_temp or 0.0,
                actual_bucket=actual_bucket,
                trade_pnl=event_pnl.get(event_key, 0.0),
            )
    
    # ─── Internal: Forecast & Probability ────────────────
    
    def _enrich_event(self, event: WeatherEvent):
        """Fetch forecasts and compute bucket probabilities for an event."""
        
        # Fetch multi-model forecast
        forecast = self.weather_client.get_multi_model_forecast(
            station=event.station,
            target_date=event.target_date,
        )
        
        event.point_forecasts = forecast.point_forecasts
        event.n_models = forecast.n_models_available
        event.ensemble_members = forecast.ensemble_members
        event.ensemble_mean = forecast.ensemble_mean
        event.ensemble_std = forecast.ensemble_std
        event.n_ensemble = forecast.n_ensemble_members
        
        # Bias correction
        corrected = forecast.point_forecasts
        if not self.bias_table.is_empty:
            corrected = self.bias_table.correct_forecasts(
                event.city, forecast.point_forecasts,
            )
            event.bias_corrected = (corrected != forecast.point_forecasts)
        event.point_forecasts_corrected = corrected
        
        # Parse buckets
        bucket_labels = [b["label"] for b in event.buckets]
        parsed_buckets = parse_buckets_from_outcomes(bucket_labels, event.station.unit)
        
        if not parsed_buckets:
            return
        
        # Compute probabilities
        if forecast.ensemble_members:
            # Bias-correct ensemble members using average city bias
            members = list(forecast.ensemble_members)
            if event.bias_corrected:
                city_biases = self.bias_table.get_city_biases(event.city)
                reliable = [b.mean for b in city_biases.values() if b.is_reliable]
                if reliable:
                    avg_bias = sum(reliable) / len(reliable)
                    members = [m - avg_bias for m in members]
            
            event.model_probs = compute_bucket_probabilities(
                members, parsed_buckets, unit=event.station.unit,
            )
        else:
            event.model_probs = compute_bucket_probs_from_point_forecasts(
                corrected, parsed_buckets, unit=event.station.unit,
            )
        
        # Model agreement
        agreement, consensus = model_agreement(corrected, parsed_buckets)
        event.agreement = agreement
        event.consensus_bucket = consensus
        event.confidence_tier = classify_confidence(agreement, event.n_models)
    
    def _compute_confidence(self, event: WeatherEvent) -> float:
        """Compute confidence score from event characteristics."""
        confidence = 0.5
        
        # Lead time
        if event.lead_hours <= 24:
            confidence = 0.85
        elif event.lead_hours <= 48:
            confidence = 0.75
        elif event.lead_hours <= 72:
            confidence = 0.65
        else:
            confidence = 0.50
        
        # Model agreement boost
        if event.confidence_tier in ("LOCK", "STRONG"):
            confidence = min(0.95, confidence + 0.10)
        elif event.confidence_tier == "NEAR_SAFE":
            confidence = min(0.90, confidence + 0.05)
        
        # Ensemble boost
        if event.n_ensemble > 0:
            confidence = min(0.95, confidence + 0.05)
        
        return confidence
    
    # ─── Internal: Exposure Tracking ─────────────────────
    
    def _get_city_exposure(self, city: str, target_date: date) -> float:
        """Get total $ in open trades for a city/date."""
        total = 0.0
        date_str = target_date.isoformat()
        for trade in self.trader.trades:
            if trade.status != "OPEN":
                continue
            if city in trade.question and date_str in trade.question:
                total += trade.dollar_amount
        return total
    
    def _get_total_weather_exposure(self) -> float:
        """Get total $ in all open weather trades."""
        return sum(
            t.dollar_amount for t in self.trader.trades if t.status == "OPEN"
        )


# ═══════════════════════════════════════════════════════════
# 4. CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Weather Scanner — Temperature Market Pipeline"
    )
    parser.add_argument(
        "--paper-trade", action="store_true",
        help="Scan and enter paper trades for tradeable edges"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check open trades for resolution"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print paper trading performance report"
    )
    parser.add_argument(
        "--bankroll", type=float, default=1000.0,
        help="Starting bankroll for paper trading (default: 1000)"
    )
    parser.add_argument(
        "--min-edge", type=float, default=WEATHER_MIN_EDGE,
        help=f"Minimum edge to trade (default: {WEATHER_MIN_EDGE})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Run diagnostics report on event log"
    )
    args = parser.parse_args()
    
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    scanner = WeatherScanner(
        bankroll=args.bankroll,
        min_edge=args.min_edge,
    )
    
    if args.report:
        snap = scanner.trader.snapshot()
        open_trades = [t for t in scanner.trader.trades if t.status == "OPEN"]
        resolved = [t for t in scanner.trader.trades if t.status.startswith("RESOLVED")]
        wins = [t for t in resolved if t.status == "RESOLVED_WIN"]
        losses = [t for t in resolved if t.status == "RESOLVED_LOSS"]
        
        print(f"\n{'=' * 70}")
        print(f"  WEATHER PAPER TRADING REPORT")
        print(f"{'=' * 70}")
        print(f"  Bankroll: ${scanner.trader.bankroll:,.2f} "
              f"(started: ${scanner.trader.initial_bankroll:,.2f})")
        print(f"  Open: {len(open_trades)} | "
              f"Wins: {len(wins)} | Losses: {len(losses)} | "
              f"Win rate: {snap.win_rate:.0%}")
        
        if resolved:
            total_pnl = sum(t.pnl for t in resolved)
            avg_win = np.mean([t.pnl for t in wins]) if wins else 0
            avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
            print(f"  Total P&L: ${total_pnl:+,.2f}")
            print(f"  Avg win: ${avg_win:+.2f} | Avg loss: ${avg_loss:+.2f}")
        
        if snap.brier_score is not None:
            print(f"  Brier score: {snap.brier_score:.4f}")
        
        if open_trades:
            print(f"\n  Open positions:")
            for t in open_trades:
                print(f"    {t.question[:55]} | ${t.dollar_amount:.2f} | Edge: {t.edge_at_entry:+.0%}")
        
        print()
        return
    
    if args.diagnostics:
        from weather.diagnostics import main as diag_main
        diag_main([])
        return
    
    if args.check:
        scanner.check_resolutions()
        return
    
    if args.paper_trade:
        scanner.scan_and_trade()
    else:
        events = scanner.scan()
        
        # Log all scanned events (even without trades) for diagnostics
        for event in events:
            confidence = scanner._compute_confidence(event)
            skip = "no_edge" if not event.tradeable_edges else "scan_only"
            scanner.event_logger.log_event(event, confidence, skip_reason=skip)
        
        # Show detailed opportunity table
        if events:
            print(f"\n{'─' * 70}")
            print(f"  TRADEABLE EDGES (run with --paper-trade to enter)")
            print(f"{'─' * 70}")
            
            for event in events:
                edges = event.tradeable_edges
                if not edges:
                    continue
                
                print(f"\n  {event.city} — {event.target_date} ({event.lead_hours}h) "
                      f"[{event.confidence_tier}]")
                print(f"  {'Bucket':>18} | {'Market':>7} | {'Model':>7} | {'Edge':>7}")
                print(f"  {'─' * 48}")
                
                for e in edges:
                    print(
                        f"  {e['label']:>18} | "
                        f"{e['market_price']:>6.0%} | "
                        f"{e['model_prob']:>6.0%} | "
                        f"{e['edge']:>+6.0%}"
                    )


if __name__ == "__main__":
    main()