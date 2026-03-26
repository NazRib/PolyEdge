"""
Enriched Edge Detector
Full pipeline with context enrichment: scan → enrich → estimate → size → trade.

This is the production version of the edge detector that uses all available
information sources before making probability estimates.
"""

import logging
import time
from typing import Optional

from core.api_client import PolymarketClient, Market
from core.market_scanner import MarketScanner, ScoredMarket
from core.probability import (
    EnsembleEstimator, ProbabilityEstimate,
    base_rate_estimator, momentum_estimator,
    book_imbalance_estimator, whale_tracker_estimator,
)
from core.kelly import kelly_criterion, PositionSize
from core.paper_trader import PaperTrader
from core.context_enricher import (
    ContextEnricher, EnrichedContext, build_enriched_forecast_prompt,
)
from core.llm_estimator import (
    LLMEstimator, SimulatedLLMEstimator, CalibrationModel,
    build_market_context, call_claude, parse_llm_response,
    FORECASTER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class EnrichedLLMEstimator:
    """
    LLM estimator that uses enriched context for better probability estimates.
    
    This is the "full power" version that:
    1. Gathers context from news, Kalshi, FRED, and related markets
    2. Builds a rich prompt with all available information
    3. Calls Claude for a probability estimate
    4. Applies calibration correction
    
    For ensemble integration:
        estimator = EnrichedLLMEstimator(enricher=enricher)
        ensemble.add_estimator("llm", estimator.estimate_for_ensemble, weight=0.45)
    """
    
    def __init__(
        self,
        enricher: ContextEnricher = None,
        calibration: CalibrationModel = None,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.enricher = enricher or ContextEnricher()
        self.calibration = calibration or CalibrationModel()
        self.api_key = api_key
        self.model = model
        self._all_markets_cache: list[dict] = []
        self._enrichment_cache: dict[str, EnrichedContext] = {}
    
    def set_market_universe(self, markets: list[dict]):
        """Cache the list of all active markets for related-market enrichment."""
        self._all_markets_cache = markets
    
    def estimate_for_ensemble(self, context_dict: dict) -> tuple[float, float]:
        """
        Ensemble-compatible interface.
        
        1. Enrich the context
        2. Build full prompt
        3. Call Claude
        4. Calibrate
        5. Return (probability, confidence)
        """
        market_id = context_dict.get("market_id", "")
        
        # Check cache
        if market_id in self._enrichment_cache:
            enriched = self._enrichment_cache[market_id]
        else:
            enriched = self.enricher.enrich(
                context_dict, all_markets=self._all_markets_cache
            )
            if market_id:
                self._enrichment_cache[market_id] = enriched
        
        # Build enriched market context
        market_ctx = build_market_context(
            market_data=context_dict,
            order_book_data={
                "bid_depth": context_dict.get("bid_depth", 0),
                "ask_depth": context_dict.get("ask_depth", 0),
                "spread": context_dict.get("spread", 0),
            },
            price_history=context_dict.get("price_history"),
        )
        
        # Build the enriched prompt
        prompt = build_enriched_forecast_prompt(
            market_ctx.to_prompt_context(),
            enriched,
        )
        
        # Call Claude
        raw_response = call_claude(
            user_prompt=prompt,
            system_prompt=FORECASTER_SYSTEM_PROMPT,
            model=self.model,
            temperature=0.2,
            api_key=self.api_key,
        )
        
        parsed = parse_llm_response(raw_response) if raw_response else None
        
        if parsed is None:
            # Fallback: use market price with low confidence
            return context_dict.get("market_price", 0.5), 0.1
        
        raw_prob = float(parsed.get("probability", 0.5))
        category = context_dict.get("category", "")
        
        # Calibrate
        calibrated = self.calibration.calibrate(raw_prob, category)
        
        # Confidence scoring
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.85}
        confidence = confidence_map.get(parsed.get("confidence", "medium"), 0.5)
        
        # Boost confidence if we have enrichment data
        if enriched.sources_used:
            confidence = min(0.95, confidence + 0.05 * len(enriched.sources_used))
        
        # Reduce confidence for large deviations from market
        deviation = abs(calibrated - context_dict.get("market_price", 0.5))
        if deviation > 0.15:
            confidence *= 0.8
        
        return float(calibrated), float(confidence)


class EnrichedEdgeDetector:
    """
    Full enriched pipeline: scan → enrich → estimate → size.
    
    Usage:
        detector = EnrichedEdgeDetector(bankroll=1000)
        signals = detector.run()
    """
    
    def __init__(
        self,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.05,
        min_scanner_score: float = 0.3,
        max_signals: int = 10,
        use_live_llm: bool = False,
        llm_skill_level: float = 0.35,
        data_dir: str = "data",
    ):
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.min_scanner_score = min_scanner_score
        self.max_signals = max_signals
        
        self.client = PolymarketClient()
        self.scanner = MarketScanner(client=self.client)
        
        # Paper trader — loads persisted state (bankroll, open trades, calibration)
        self.trader = PaperTrader(bankroll=bankroll, data_dir=data_dir)
        
        # Context enricher
        self.enricher = ContextEnricher(
            enable_news=use_live_llm,
            enable_kalshi=True,
            enable_fred=True,
            enable_related=True,
            enable_whales=True,
        )
        
        # Build ensemble
        self.ensemble = EnsembleEstimator()
        
        if use_live_llm:
            self.llm_estimator = EnrichedLLMEstimator(enricher=self.enricher)
            self.ensemble.add_estimator(
                "enriched_llm", self.llm_estimator.estimate_for_ensemble, weight=0.45
            )
        else:
            self.llm_estimator = SimulatedLLMEstimator(skill_level=llm_skill_level)
            self.ensemble.add_estimator(
                "simulated_llm", self.llm_estimator.estimate_for_ensemble, weight=0.40
            )
        
        self.ensemble.add_estimator("base_rate", base_rate_estimator, weight=0.15)
        self.ensemble.add_estimator("momentum", momentum_estimator, weight=0.20)
        self.ensemble.add_estimator("book_imbalance", book_imbalance_estimator, weight=0.15)
        self.ensemble.add_estimator("whale_tracker", whale_tracker_estimator, weight=0.10)
    
    def run(self, fetch_order_books: bool = True) -> list[dict]:
        """Run the full enriched pipeline: scan → enrich → estimate → size → trade."""
        
        print("\n" + "=" * 70)
        print("  POLYMARKET EDGE — Enriched Pipeline")
        print("=" * 70)
        
        # Step 0: Check open trades for resolution
        self._check_resolutions()
        
        # Step 1: Scan
        t0 = time.time()
        print("\n📡 Step 1: Scanning markets...")
        candidates = self.scanner.scan(
            top_n=30, fetch_order_books=fetch_order_books
        )
        candidates = [c for c in candidates if c.overall_score >= self.min_scanner_score]
        print(f"   Found {len(candidates)} candidates ({time.time()-t0:.1f}s)")
        
        # Cache all market dicts for related-market lookup
        all_market_dicts = [self._scored_to_dict(c) for c in candidates]
        if hasattr(self.llm_estimator, 'set_market_universe'):
            self.llm_estimator.set_market_universe(all_market_dicts)
        
        # Step 2 & 3: Enrich + Estimate for each candidate
        print(f"\n🔬 Step 2-3: Enriching & estimating {len(candidates)} markets...")
        signals = []
        
        for i, scored in enumerate(candidates):
            market = scored.market
            context = self._build_context(market, scored)
            
            # Log progress
            print(f"\n  [{i+1}/{len(candidates)}] {market.question[:55]}...")
            
            # Throttle between markets to avoid Claude API rate limits
            # Each market triggers ~2 Claude calls (news search + estimation)
            if i > 0:
                time.sleep(1.5)
            
            # Estimate
            estimate = self.ensemble.estimate(context)
            
            # Size — use paper trader's current bankroll
            position = kelly_criterion(
                estimated_prob=estimate.probability,
                market_price=estimate.market_price,
                bankroll=self.trader.bankroll,
                kelly_fraction=self.kelly_fraction,
                min_edge=self.min_edge,
                confidence=estimate.confidence,
            )
            
            signal = {
                "market": market,
                "scored": scored,
                "estimate": estimate,
                "position": position,
                "should_trade": position.should_trade,
            }
            signals.append(signal)
            
            # Brief status
            if position.should_trade:
                print(
                    f"    🟢 {position.side} | Edge: {estimate.edge:+.1%} | "
                    f"${position.dollar_amount:.2f}"
                )
            else:
                print(f"    ⬜ Skip ({position.rejection_reason[:40]})")
        
        # Sort by EV
        signals.sort(
            key=lambda s: s["position"].expected_profit if s["should_trade"] else -999,
            reverse=True,
        )
        
        # Step 4: Enter paper trades
        tradeable = [s for s in signals if s["should_trade"]]
        trades_entered = 0
        
        for s in tradeable[:self.max_signals]:
            # Skip markets we already have an open trade in
            existing_ids = {
                t.market_id for t in self.trader.trades if t.status == "OPEN"
            }
            if s["estimate"].market_id in existing_ids:
                logger.info(
                    f"  Skipping duplicate: already have open trade in "
                    f"{s['market'].question[:40]}..."
                )
                continue
            
            trade = self.trader.enter_trade(s["estimate"], s["position"])
            if trade:
                trades_entered += 1
        
        total_time = time.time() - t0
        
        # Summary
        print(f"\n{'=' * 70}")
        print(f"  RESULTS: {len(tradeable)} signals, {trades_entered} new trades entered ({total_time:.1f}s)")
        print(f"{'=' * 70}\n")
        
        if tradeable:
            print(f"  {'Side':>4} | {'$':>8} | {'Edge':>7} | {'EV':>8} | {'Market'}")
            print("  " + "─" * 65)
            for s in tradeable[:self.max_signals]:
                p = s["position"]
                m = s["market"]
                print(
                    f"  {p.side:>4} | ${p.dollar_amount:>6.2f} | "
                    f"{p.edge:>+6.1%} | ${p.expected_profit:>+6.2f} | "
                    f"{m.question[:40]}"
                )
        
        # Paper trading status
        snap = self.trader.snapshot()
        open_trades = [t for t in self.trader.trades if t.status == "OPEN"]
        
        print(f"\n  📊 Paper Portfolio:")
        print(f"     Bankroll: ${self.trader.bankroll:,.2f} "
              f"(started: ${self.trader.initial_bankroll:,.2f})")
        print(f"     Open trades: {len(open_trades)} | "
              f"Resolved: {snap.resolved_trades} | "
              f"Win rate: {snap.win_rate:.0%}")
        if snap.brier_score is not None:
            print(f"     Brier score: {snap.brier_score:.4f} "
                  f"(vs 0.25 baseline: {(0.25 - snap.brier_score) / 0.25:+.1%})")
            if snap.resolved_trades < 30:
                print(f"     ⏳ {30 - snap.resolved_trades} more resolutions needed "
                      f"for meaningful assessment")
        print()
        
        return signals[:self.max_signals]
    
    def _check_resolutions(self):
        """
        Check open trades for market resolution.
        
        Fetches each open trade's market from the Gamma API.
        If the market is closed, determines the outcome from final prices
        and resolves the trade.
        """
        open_trades = [t for t in self.trader.trades if t.status == "OPEN"]
        if not open_trades:
            return
        
        print(f"\n🔍 Checking {len(open_trades)} open trade(s) for resolution...")
        resolved_count = 0
        
        for trade in open_trades:
            try:
                market = self.client.get_market_by_id(trade.market_id)
                if market is None:
                    continue
                
                if market.closed:
                    # Determine outcome from final prices
                    # When resolved: yes_price → 1.0 means Yes won, → 0.0 means No won
                    yes_price = market.yes_price
                    if yes_price > 0.90:
                        outcome = True   # Yes won
                    elif yes_price < 0.10:
                        outcome = False  # No won
                    else:
                        # Market closed but not clearly resolved (voided?)
                        logger.warning(
                            f"  ⚠ Market closed with ambiguous price {yes_price:.2f}: "
                            f"{trade.question[:40]}..."
                        )
                        continue
                    
                    pnl = self.trader.resolve_trade(trade.trade_id, outcome)
                    if pnl is not None:
                        resolved_count += 1
                        status = "✅ WIN" if pnl > 0 else "❌ LOSS"
                        print(
                            f"  {status} {trade.trade_id}: {trade.side} "
                            f"'{trade.question[:35]}...' → "
                            f"PnL ${pnl:+.2f}"
                        )
            except Exception as e:
                logger.debug(f"Resolution check failed for {trade.trade_id}: {e}")
                continue
        
        if resolved_count > 0:
            print(f"  Resolved {resolved_count} trade(s). "
                  f"Bankroll: ${self.trader.bankroll:,.2f}")
        else:
            print("  No resolutions found.")
    
    def _scored_to_dict(self, scored: ScoredMarket) -> dict:
        """Convert a ScoredMarket to a flat dict for enrichment."""
        m = scored.market
        return {
            "market_id": m.id,
            "question": m.question,
            "description": m.description,
            "market_price": m.yes_price,
            "yes_price": m.yes_price,
            "category": m.category,
            "volume_24h": m.volume_24h,
            "volume_total": m.volume_total,
            "liquidity": m.liquidity,
            "days_to_resolution": m.days_to_resolution,
            "event_slug": m.event_slug,
            "tags": m.tags,
        }
    
    def _build_context(self, market: Market, scored: ScoredMarket) -> dict:
        """Build context dict for the ensemble estimators."""
        ctx = self._scored_to_dict(scored)
        
        # Add order book data
        if scored.order_book:
            ctx["bid_depth"] = scored.order_book.bid_depth
            ctx["ask_depth"] = scored.order_book.ask_depth
            ctx["spread"] = scored.order_book.spread
            ctx["midpoint"] = scored.order_book.midpoint
        
        # Try to get price history
        if market.token_ids:
            try:
                history = self.client.get_price_history(
                    market.token_ids[0], interval="1d"
                )
                if history:
                    ctx["price_history"] = [
                        (h.get("t", 0), float(h.get("p", market.yes_price)))
                        for h in history
                    ]
            except Exception:
                pass
        
        # Pre-fetch whale positions so both the statistical whale_tracker_estimator
        # and the LLM enriched prompt can use them
        if (self.enricher.whale_enricher 
                and self.enricher.whale_enricher.registry_size > 0):
            whale_positions = self.enricher.whale_enricher.get_whale_positions(
                market.id
            )
            if whale_positions:
                ctx["whale_positions"] = whale_positions
        
        return ctx


def run_enriched_pipeline(**kwargs):
    """Convenience function to run the enriched pipeline."""
    detector = EnrichedEdgeDetector(**kwargs)
    return detector.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_enriched_pipeline(bankroll=1000, use_live_llm=False)