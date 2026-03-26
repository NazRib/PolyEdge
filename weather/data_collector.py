"""
Weather Data Collector — Forward Live Snapshots
Captures daily snapshots of weather model forecasts + Polymarket prices
for active temperature markets. Validates backtest findings in real-time.

Usage:
    python -m weather.data_collector
    python -m weather.data_collector --once       # Single snapshot, no loop
    python -m weather.data_collector --report     # Print status of collected data
"""

import json
import logging
import os
import argparse
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests

from weather.config import (
    STATIONS, StationInfo,
    WEATHER_DATA_DIR, SNAPSHOTS_FILE,
)
from weather.models import OpenMeteoClient
from weather.utils import (
    parse_question, parse_buckets_from_outcomes,
    compute_bucket_probabilities, compute_bucket_probs_from_point_forecasts,
    model_agreement, classify_confidence, brier_score,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════

@dataclass
class LiveSnapshot:
    """A single snapshot of forecast + market data for one city/date."""
    city: str
    target_date: str                # ISO format
    snapshot_time: str              # ISO format — when this was captured
    lead_hours: int                 # Hours until market resolves
    
    # Model forecasts (in station's native unit)
    point_forecasts: dict[str, float] = field(default_factory=dict)
    ensemble_mean: float = 0.0
    ensemble_std: float = 0.0
    n_ensemble_members: int = 0
    n_models: int = 0
    
    # Model-implied probabilities
    model_bucket_probs: dict[str, float] = field(default_factory=dict)
    model_top_bucket: str = ""
    model_agreement_score: float = 0.0
    confidence_tier: str = "LOW"
    
    # Market prices
    market_bucket_probs: dict[str, float] = field(default_factory=dict)
    market_top_bucket: str = ""
    
    # Edge
    max_edge: float = 0.0
    tradeable_buckets: int = 0
    
    # Resolution (filled in later when market resolves)
    actual_bucket: str = ""
    model_brier_score: Optional[float] = None
    market_brier_score: Optional[float] = None
    brier_delta: Optional[float] = None


# ═══════════════════════════════════════════════════════════
# DATA COLLECTOR
# ═══════════════════════════════════════════════════════════

class WeatherDataCollector:
    """
    Captures live snapshots of weather forecasts + market prices.
    
    Designed to run once daily (or multiple times per day to capture
    different model update cycles). Each run scans for active temperature
    markets, pulls current forecasts, and logs the snapshot.
    
    Usage:
        collector = WeatherDataCollector()
        snapshots = collector.collect_snapshot()
        collector.save_snapshots(snapshots)
    """
    
    GAMMA_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self):
        self.weather_client = OpenMeteoClient()
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketEdge-Weather/1.0",
        })
        self._last_request = 0.0
    
    def _throttle(self):
        import time
        elapsed = time.time() - self._last_request
        if elapsed < 0.5:
            import time as t
            t.sleep(0.5 - elapsed)
        self._last_request = __import__("time").time()
    
    def _get(self, url: str, params: dict = None) -> Optional[dict | list]:
        self._throttle()
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.debug(f"Request failed: {url} — {e}")
            return None
    
    def collect_snapshot(self) -> list[LiveSnapshot]:
        """
        Scan for active temperature markets and capture current state.
        
        Returns a list of LiveSnapshot objects, one per city/date.
        """
        logger.info("Collecting weather data snapshot...")
        now = datetime.now(timezone.utc)
        today = date.today()
        snapshots = []
        
        # Check each city for the next 5 days
        for city_key, station in STATIONS.items():
            for day_offset in range(0, 5):
                target = today + timedelta(days=day_offset)
                
                snapshot = self._collect_single(
                    station=station,
                    target_date=target,
                    now=now,
                )
                if snapshot:
                    snapshots.append(snapshot)
        
        logger.info(f"Collected {len(snapshots)} snapshots")
        return snapshots
    
    def _collect_single(
        self,
        station: StationInfo,
        target_date: date,
        now: datetime,
    ) -> Optional[LiveSnapshot]:
        """Collect a snapshot for a single city/date."""
        
        # Calculate lead time
        target_end = datetime(
            target_date.year, target_date.month, target_date.day,
            23, 59, 59, tzinfo=timezone.utc,
        )
        lead_hours = max(0, int((target_end - now).total_seconds() / 3600))
        
        if lead_hours <= 0:
            return None  # Market already resolved or resolving now
        
        # 1. Fetch weather model forecasts
        forecast = self.weather_client.get_multi_model_forecast(
            station=station, target_date=target_date,
        )
        
        if not forecast.is_valid:
            logger.debug(f"  {station.city} {target_date}: insufficient models")
            return None
        
        # 2. Try to find this market on Polymarket
        from weather.utils import build_event_slug
        slug = build_event_slug(station.city, target_date)
        
        market_data = self._get(f"{self.GAMMA_URL}/events/{slug}")
        
        market_probs = {}
        bucket_labels = []
        
        if market_data:
            raw_markets = []
            if isinstance(market_data, dict):
                raw_markets = market_data.get("markets", [])
            elif isinstance(market_data, list):
                raw_markets = market_data
            
            for m in raw_markets:
                label = m.get("groupItemTitle", "")
                prices = m.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    try:
                        prices = json.loads(prices)
                    except (json.JSONDecodeError, TypeError):
                        prices = []
                prices = [float(p) for p in prices]
                
                if label and prices:
                    market_probs[label] = prices[0]  # Yes price
                    bucket_labels.append(label)
        
        if not market_probs:
            logger.debug(f"  {station.city} {target_date}: no market data found")
            return None
        
        # Normalize market prices
        total = sum(market_probs.values())
        if total > 0:
            market_probs = {k: v / total for k, v in market_probs.items()}
        
        # 3. Compute model-implied probabilities
        parsed_buckets = parse_buckets_from_outcomes(bucket_labels, station.unit)
        
        if forecast.ensemble_members and parsed_buckets:
            model_probs = compute_bucket_probabilities(
                forecast.ensemble_members, parsed_buckets, unit=station.unit,
            )
        elif parsed_buckets:
            model_probs = compute_bucket_probs_from_point_forecasts(
                forecast.point_forecasts, parsed_buckets, unit=station.unit,
            )
        else:
            model_probs = {}
        
        # 4. Model agreement
        agreement, consensus = model_agreement(
            forecast.point_forecasts, parsed_buckets,
        ) if parsed_buckets else (0.0, "")
        
        # 5. Edge analysis
        edges = {}
        for label in model_probs:
            if label in market_probs:
                edges[label] = model_probs[label] - market_probs[label]
        
        max_edge = max(edges.values()) if edges else 0.0
        tradeable = sum(1 for e in edges.values() if e > 0.08)
        
        snapshot = LiveSnapshot(
            city=station.city,
            target_date=target_date.isoformat(),
            snapshot_time=now.isoformat(),
            lead_hours=lead_hours,
            point_forecasts=forecast.point_forecasts,
            ensemble_mean=forecast.ensemble_mean,
            ensemble_std=forecast.ensemble_std,
            n_ensemble_members=forecast.n_ensemble_members,
            n_models=forecast.n_models_available,
            model_bucket_probs=model_probs,
            model_top_bucket=max(model_probs, key=model_probs.get) if model_probs else "",
            model_agreement_score=agreement,
            confidence_tier=classify_confidence(agreement, forecast.n_models_available),
            market_bucket_probs=market_probs,
            market_top_bucket=max(market_probs, key=market_probs.get) if market_probs else "",
            max_edge=max_edge,
            tradeable_buckets=tradeable,
        )
        
        logger.info(
            f"  {station.city:>12} | {target_date} | {lead_hours:3d}h | "
            f"Models: {forecast.n_models_available} | "
            f"Edge: {max_edge:+.1%} | "
            f"{snapshot.confidence_tier}"
        )
        
        return snapshot
    
    def save_snapshots(self, snapshots: list[LiveSnapshot]):
        """Append snapshots to the snapshots file."""
        os.makedirs(WEATHER_DATA_DIR, exist_ok=True)
        
        # Load existing snapshots
        existing = []
        if os.path.exists(SNAPSHOTS_FILE):
            with open(SNAPSHOTS_FILE) as f:
                existing = json.load(f)
        
        # Append new
        for s in snapshots:
            existing.append(asdict(s))
        
        with open(SNAPSHOTS_FILE, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        
        logger.info(f"Saved {len(snapshots)} new snapshots (total: {len(existing)})")
    
    def print_status(self):
        """Print a summary of collected data so far."""
        if not os.path.exists(SNAPSHOTS_FILE):
            print("No snapshots collected yet.")
            return
        
        with open(SNAPSHOTS_FILE) as f:
            data = json.load(f)
        
        print(f"\n{'=' * 60}")
        print(f"  WEATHER DATA COLLECTOR — Status")
        print(f"{'=' * 60}")
        print(f"  Total snapshots:  {len(data)}")
        
        if data:
            cities = set(s["city"] for s in data)
            dates = set(s["target_date"] for s in data)
            times = sorted(s["snapshot_time"] for s in data)
            
            print(f"  Cities:           {', '.join(sorted(cities))}")
            print(f"  Date range:       {min(dates)} to {max(dates)}")
            print(f"  Collection range: {times[0][:10]} to {times[-1][:10]}")
            
            # Count resolved vs pending
            resolved = sum(1 for s in data if s.get("actual_bucket"))
            print(f"  Resolved:         {resolved}/{len(data)}")
            
            # Average edge
            edges = [s.get("max_edge", 0) for s in data if s.get("max_edge")]
            if edges:
                print(f"  Mean max edge:    {np.mean(edges):+.1%}")


# ═══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Weather Data Collector — Forward Live Snapshots"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Collect a single snapshot and exit"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print status of collected data"
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
    
    collector = WeatherDataCollector()
    
    if args.report:
        collector.print_status()
        return
    
    snapshots = collector.collect_snapshot()
    if snapshots:
        collector.save_snapshots(snapshots)
    
    if not args.once:
        print("\nRun with --once for a single snapshot, or set up a cron job:")
        print("  # Run every 6 hours to capture different model update cycles")
        print("  0 */6 * * * cd /path/to/project && python -m weather.data_collector --once")


if __name__ == "__main__":
    main()
