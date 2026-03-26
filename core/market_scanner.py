"""
Market Scanner
Scans Polymarket for tradeable opportunities by applying filters
and scoring markets on dimensions like liquidity, edge potential,
and information asymmetry signals.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from core.api_client import PolymarketClient, Market, OrderBook

logger = logging.getLogger(__name__)


@dataclass
class ScoredMarket:
    """A market enriched with trading opportunity scores."""
    market: Market
    order_book: Optional[OrderBook]
    
    # Scores (0-1 scale, higher = better opportunity)
    liquidity_score: float = 0.0
    edge_potential_score: float = 0.0
    timing_score: float = 0.0
    whale_activity_score: float = 0.0
    overall_score: float = 0.0
    
    # Derived metrics
    effective_spread: float = 0.0
    book_imbalance: float = 0.0       # >0 means more bids, <0 more asks
    price_range_score: float = 0.0    # How far from 50/50 (extremes less interesting)
    
    def __repr__(self):
        return (
            f"ScoredMarket(score={self.overall_score:.3f}, "
            f"price={self.market.yes_price:.0%}, "
            f"spread={self.effective_spread:.4f}, "
            f"q='{self.market.question[:50]}...')"
        )


class MarketScanner:
    """
    Scans and scores Polymarket markets for trading opportunities.
    
    The scanner applies a multi-stage pipeline:
    1. Fetch active markets from the API
    2. Apply hard filters (liquidity, volume, time to resolution)
    3. Enrich with order book data
    4. Score on multiple dimensions
    5. Rank and return top opportunities
    
    Usage:
        scanner = MarketScanner()
        opportunities = scanner.scan(top_n=20)
        for opp in opportunities:
            print(f"{opp.overall_score:.3f} | {opp.market.question}")
    """
    
    def __init__(
        self,
        client: PolymarketClient = None,
        min_volume_24h: float = 1000,
        min_liquidity: float = 5000,
        price_range: tuple = (0.05, 0.95),
        max_days_to_resolution: float = 90,
        min_days_to_resolution: float = 1,
    ):
        self.client = client or PolymarketClient()
        self.min_volume_24h = min_volume_24h
        self.min_liquidity = min_liquidity
        self.price_range = price_range
        self.max_days_to_resolution = max_days_to_resolution
        self.min_days_to_resolution = min_days_to_resolution
    
    def scan(
        self,
        top_n: int = 20,
        fetch_order_books: bool = True,
        max_markets: int = 200,
    ) -> list[ScoredMarket]:
        """
        Run full scan pipeline and return top opportunities.
        
        Args:
            top_n: Number of top markets to return
            fetch_order_books: Whether to fetch order book data (slower but richer)
            max_markets: Max markets to fetch from API before filtering
            
        Returns:
            List of ScoredMarket objects, sorted by overall_score descending
        """
        # Stage 1: Fetch markets
        logger.info("Fetching active markets...")
        raw_markets = self.client.get_active_markets(limit=min(max_markets, 100))
        if max_markets > 100:
            raw_markets += self.client.get_active_markets(
                limit=min(max_markets - 100, 100), offset=100
            )
        logger.info(f"Fetched {len(raw_markets)} markets")
        
        # Stage 2: Apply hard filters
        filtered = self._apply_filters(raw_markets)
        logger.info(f"After filtering: {len(filtered)} markets")
        
        # Stage 3: Enrich with order book data
        scored_markets = []
        for market in filtered:
            order_book = None
            if fetch_order_books and market.token_ids:
                try:
                    order_book = self.client.get_order_book(market.token_ids[0])
                except Exception as e:
                    logger.debug(f"Failed to fetch order book: {e}")
            
            scored = ScoredMarket(market=market, order_book=order_book)
            scored_markets.append(scored)
        
        # Stage 4: Score all markets
        for scored in scored_markets:
            self._score_market(scored)
        
        # Stage 5: Rank and return
        scored_markets.sort(key=lambda s: s.overall_score, reverse=True)
        return scored_markets[:top_n]
    
    def _apply_filters(self, markets: list[Market]) -> list[Market]:
        """Apply hard filters to eliminate unsuitable markets."""
        filtered = []
        
        for m in markets:
            # Must be active and not closed
            if m.closed or not m.active:
                continue
            
            # Volume filter
            if m.volume_24h < self.min_volume_24h:
                continue
            
            # Liquidity filter
            if m.liquidity < self.min_liquidity:
                continue
            
            # Price range filter (skip extreme probabilities)
            if not (self.price_range[0] <= m.yes_price <= self.price_range[1]):
                continue
            
            # Time to resolution filter
            days = m.days_to_resolution
            if days is not None:
                if days < self.min_days_to_resolution:
                    continue
                if days > self.max_days_to_resolution:
                    continue
            
            # Must have token IDs for CLOB access
            if not m.token_ids:
                continue
            
            filtered.append(m)
        
        return filtered
    
    def _score_market(self, scored: ScoredMarket):
        """Compute all sub-scores and the overall score for a market."""
        m = scored.market
        ob = scored.order_book
        
        # 1. Liquidity score — higher volume and liquidity is better
        vol_score = min(1.0, m.volume_24h / 100_000)  # Saturates at $100k
        liq_score = min(1.0, m.liquidity / 200_000)    # Saturates at $200k
        scored.liquidity_score = 0.6 * vol_score + 0.4 * liq_score
        
        # 2. Edge potential score — markets near 50/50 have more room for edge,
        #    and wider spreads indicate less efficient pricing
        distance_from_center = abs(m.yes_price - 0.5)
        scored.price_range_score = 1.0 - (distance_from_center * 2)  # 1.0 at 50%, 0 at extremes
        
        spread_bonus = 0.0
        if ob and ob.spread > 0:
            scored.effective_spread = ob.spread
            # Wider spread = more potential edge but also more cost
            # Sweet spot is moderate spread (0.02-0.08)
            if 0.02 <= ob.spread <= 0.08:
                spread_bonus = 0.5
            elif ob.spread < 0.02:
                spread_bonus = 0.2  # Very tight, hard to get edge
            else:
                spread_bonus = 0.3  # Wide spread, higher cost
        
        scored.edge_potential_score = 0.6 * scored.price_range_score + 0.4 * spread_bonus
        
        # 3. Timing score — markets resolving in 3-30 days are ideal
        #    (enough time to react, not too long to lock capital)
        days = m.days_to_resolution
        if days is not None:
            if 3 <= days <= 30:
                scored.timing_score = 1.0
            elif 1 <= days < 3:
                scored.timing_score = 0.5
            elif 30 < days <= 60:
                scored.timing_score = 0.7
            else:
                scored.timing_score = 0.3
        else:
            scored.timing_score = 0.4  # Unknown resolution — moderate score
        
        # 4. Order book imbalance — significant imbalance can signal informed trading
        if ob and ob.bid_depth > 0 and ob.ask_depth > 0:
            total_depth = ob.total_depth
            scored.book_imbalance = (ob.bid_depth - ob.ask_depth) / total_depth
            # Strong imbalance in either direction is interesting
            scored.whale_activity_score = min(1.0, abs(scored.book_imbalance) * 3)
        else:
            scored.whale_activity_score = 0.0
        
        # Overall score — weighted combination
        scored.overall_score = (
            0.25 * scored.liquidity_score
            + 0.35 * scored.edge_potential_score
            + 0.20 * scored.timing_score
            + 0.20 * scored.whale_activity_score
        )
    
    def find_cross_market_anomalies(
        self, markets: list[Market]
    ) -> list[tuple[Market, Market, float]]:
        """
        Find markets that seem inconsistent with each other.
        
        For example, if "Will X happen?" is at 80% but a related
        "Will X happen by date Y?" is at 90%, that's suspicious.
        
        Returns list of (market_a, market_b, inconsistency_score) tuples.
        """
        anomalies = []
        
        # Group markets by event
        by_event = {}
        for m in markets:
            if m.event_slug:
                by_event.setdefault(m.event_slug, []).append(m)
        
        # Within each event, check for pricing inconsistencies
        for event_slug, event_markets in by_event.items():
            if len(event_markets) < 2:
                continue
            
            # In a multi-outcome event, prices should sum to ~1.0
            total_yes = sum(m.yes_price for m in event_markets)
            if len(event_markets) > 2:
                # Multi-outcome: should sum close to 1.0
                deviation = abs(total_yes - 1.0)
                if deviation > 0.05:
                    # Significant mispricing across outcomes
                    for m in event_markets:
                        for m2 in event_markets:
                            if m.id != m2.id:
                                anomalies.append((m, m2, deviation))
        
        anomalies.sort(key=lambda x: x[2], reverse=True)
        return anomalies


def demo_scan():
    """Run a demo scan and print results."""
    print("\n" + "=" * 70)
    print("  POLYMARKET EDGE — Market Scanner")
    print("=" * 70 + "\n")
    
    scanner = MarketScanner()
    opportunities = scanner.scan(top_n=15, fetch_order_books=True)
    
    print(f"{'Score':>6} | {'Price':>6} | {'Vol 24h':>10} | {'Spread':>7} | {'Market'}")
    print("-" * 90)
    
    for opp in opportunities:
        m = opp.market
        spread_str = f"{opp.effective_spread:.4f}" if opp.effective_spread else "  N/A "
        print(
            f"{opp.overall_score:6.3f} | "
            f"{m.yes_price:5.0%} | "
            f"${m.volume_24h:>9,.0f} | "
            f"{spread_str} | "
            f"{m.question[:50]}"
        )
    
    return opportunities


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_scan()
