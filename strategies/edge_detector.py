"""
Edge Detector Strategy
The main strategy that ties together market scanning, probability estimation,
position sizing, and trade execution into a single pipeline.

This is the "brain" of the system — it decides what to trade and how much.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from core.api_client import PolymarketClient, Market
from core.market_scanner import MarketScanner, ScoredMarket
from core.probability import (
    EnsembleEstimator, ProbabilityEstimate,
    create_default_ensemble,
    base_rate_estimator, momentum_estimator,
    book_imbalance_estimator, whale_tracker_estimator,
)
from core.kelly import kelly_criterion, PositionSize
from core.paper_trader import PaperTrader

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """A fully evaluated trade opportunity ready for execution."""
    market: Market
    scored: ScoredMarket
    estimate: ProbabilityEstimate
    position: PositionSize
    
    @property
    def should_trade(self) -> bool:
        return self.position.should_trade
    
    @property
    def edge(self) -> float:
        return self.estimate.edge
    
    def summary(self) -> str:
        if not self.should_trade:
            return (
                f"SKIP | Edge: {self.estimate.edge:+.1%} | "
                f"Score: {self.scored.overall_score:.3f} | "
                f"{self.market.question[:50]}"
            )
        return (
            f"{'🟢' if self.position.expected_profit > 0 else '🔴'} "
            f"{self.position.side} | "
            f"${self.position.dollar_amount:.2f} | "
            f"Edge: {self.estimate.edge:+.1%} | "
            f"EV: ${self.position.expected_profit:+.2f} | "
            f"Score: {self.scored.overall_score:.3f} | "
            f"{self.market.question[:45]}"
        )


class EdgeDetector:
    """
    Full pipeline: scan markets → estimate probabilities → size positions.
    
    Usage:
        detector = EdgeDetector(bankroll=1000)
        signals = detector.run()
        for s in signals:
            if s.should_trade:
                print(s.summary())
    """
    
    def __init__(
        self,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.05,
        min_scanner_score: float = 0.3,
        max_signals: int = 10,
        client: PolymarketClient = None,
        ensemble: EnsembleEstimator = None,
    ):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.min_scanner_score = min_scanner_score
        self.max_signals = max_signals
        
        self.client = client or PolymarketClient()
        self.scanner = MarketScanner(client=self.client)
        self.ensemble = ensemble or create_default_ensemble()
    
    def run(self, fetch_order_books: bool = True) -> list[TradeSignal]:
        """
        Run the full edge detection pipeline.
        
        Returns:
            List of TradeSignal objects, sorted by expected value descending.
            Includes both tradeable and skipped signals for analysis.
        """
        # Step 1: Scan for candidate markets
        logger.info("Step 1: Scanning markets...")
        candidates = self.scanner.scan(
            top_n=50,
            fetch_order_books=fetch_order_books,
        )
        
        # Filter by minimum scanner score
        candidates = [c for c in candidates if c.overall_score >= self.min_scanner_score]
        logger.info(f"  Found {len(candidates)} candidates above score threshold")
        
        # Step 2: Estimate probabilities for each candidate
        logger.info("Step 2: Estimating probabilities...")
        signals = []
        
        for scored in candidates:
            market = scored.market
            
            # Build context for the estimator
            context = self._build_context(market, scored)
            
            # Run ensemble estimation
            estimate = self.ensemble.estimate(context)
            
            # Step 3: Size position if there's edge
            position = kelly_criterion(
                estimated_prob=estimate.probability,
                market_price=estimate.market_price,
                bankroll=self.bankroll,
                kelly_fraction=self.kelly_fraction,
                min_edge=self.min_edge,
                confidence=estimate.confidence,
            )
            
            signal = TradeSignal(
                market=market,
                scored=scored,
                estimate=estimate,
                position=position,
            )
            signals.append(signal)
        
        # Sort by expected value
        signals.sort(
            key=lambda s: s.position.expected_profit if s.should_trade else -1,
            reverse=True,
        )
        
        tradeable = [s for s in signals if s.should_trade]
        logger.info(
            f"  Generated {len(signals)} signals, {len(tradeable)} tradeable"
        )
        
        return signals[:self.max_signals]
    
    def _build_context(self, market: Market, scored: ScoredMarket) -> dict:
        """Build the context dict that estimators need."""
        context = {
            "market_id": market.id,
            "question": market.question,
            "description": market.description,
            "market_price": market.yes_price,
            "volume_24h": market.volume_24h,
            "liquidity": market.liquidity,
            "days_to_resolution": market.days_to_resolution,
            "category": market.category,
            "tags": market.tags,
        }
        
        # Add order book data if available
        if scored.order_book:
            context["bid_depth"] = scored.order_book.bid_depth
            context["ask_depth"] = scored.order_book.ask_depth
            context["spread"] = scored.order_book.spread
            context["midpoint"] = scored.order_book.midpoint
        
        # Fetch price history for momentum signal
        if market.token_ids:
            try:
                history = self.client.get_price_history(
                    market.token_ids[0], interval="1d"
                )
                if history:
                    context["price_history"] = [
                        (h.get("t", 0), float(h.get("p", market.yes_price)))
                        for h in history
                    ]
            except Exception:
                pass  # Price history is optional
        
        return context
    
    def run_and_trade(self, trader: PaperTrader) -> list[TradeSignal]:
        """
        Run pipeline and automatically enter paper trades for all signals.
        
        Args:
            trader: PaperTrader instance to record trades in
        
        Returns:
            List of all signals (tradeable ones will have been entered)
        """
        signals = self.run()
        
        entered = 0
        for signal in signals:
            if signal.should_trade:
                trade = trader.enter_trade(signal.estimate, signal.position)
                if trade:
                    entered += 1
        
        logger.info(f"Entered {entered} paper trades")
        return signals


def run_pipeline(
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.05,
    paper_trade: bool = True,
):
    """
    Run the full edge detection and (optionally) paper trading pipeline.
    
    This is the main entry point for the system.
    """
    print("\n" + "=" * 70)
    print("  POLYMARKET EDGE — Full Pipeline Run")
    print("=" * 70)
    
    # Initialize
    detector = EdgeDetector(
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        min_edge=min_edge,
    )
    
    trader = PaperTrader(bankroll=bankroll) if paper_trade else None
    
    # Run
    if trader:
        signals = detector.run_and_trade(trader)
    else:
        signals = detector.run()
    
    # Report
    tradeable = [s for s in signals if s.should_trade]
    skipped = [s for s in signals if not s.should_trade]
    
    print(f"\n📊 Results: {len(tradeable)} trades / {len(signals)} signals\n")
    
    if tradeable:
        print("  TRADEABLE SIGNALS:")
        print("  " + "-" * 65)
        for s in tradeable:
            print(f"  {s.summary()}")
    
    if skipped[:5]:
        print(f"\n  TOP SKIPPED (showing 5 of {len(skipped)}):")
        print("  " + "-" * 65)
        for s in skipped[:5]:
            print(f"  {s.summary()}")
    
    if trader:
        print(trader.report())
    
    return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_pipeline()
