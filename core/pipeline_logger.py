"""
Pipeline Logger
Writes a detailed, human-readable log file for each pipeline run,
capturing the full enrichment → estimation → sizing → trade flow
for every market evaluated.

Logs are written to data/logs/run_YYYY-MM-DD_HHMMSS.log

Usage:
    logger = PipelineLogger(data_dir="data")
    logger.log_run_header(strategy_tag, model_tag, n_candidates)

    for each market:
        logger.log_market(...)

    logger.log_run_summary(signals, trades_entered)
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class PipelineLogger:
    """Writes a human-readable log file for one pipeline run."""

    def __init__(self, data_dir: str = "data"):
        self.log_dir = Path(data_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        self.filepath = self.log_dir / f"run_{ts}.log"
        self._market_counter = 0

        # open file handle — closed explicitly or on __del__
        self._fh = open(self.filepath, "w", encoding="utf-8")
        log.info(f"Pipeline log → {self.filepath}")

    # ── lifecycle ────────────────────────────────────────

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __del__(self):
        self.close()

    # ── helpers ──────────────────────────────────────────

    def _w(self, text: str = ""):
        self._fh.write(text + "\n")

    def _sep(self, char: str = "─", width: int = 78, label: str = ""):
        if label:
            pad = max(1, width - len(label) - 5)
            self._w(f"─── {label} {char * pad}")
        else:
            self._w(char * width)

    # ── run-level ────────────────────────────────────────

    def log_run_header(
        self,
        strategy_tag: str,
        model_tag: str,
        n_candidates: int,
        bankroll: float,
        min_edge: float,
        kelly_fraction: float,
    ):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self._w("=" * 78)
        self._w(f"  POLYMARKET EDGE — Pipeline Run Log")
        self._w(f"  {ts}")
        self._w("=" * 78)
        self._w(f"  Strategy:      {strategy_tag}")
        self._w(f"  Model:         {model_tag}")
        self._w(f"  Candidates:    {n_candidates}")
        self._w(f"  Bankroll:      ${bankroll:,.2f}")
        self._w(f"  Min edge:      {min_edge:.0%}")
        self._w(f"  Kelly frac:    {kelly_fraction:.0%}")
        self._w("=" * 78)
        self._w()

    def log_run_summary(
        self,
        signals: list[dict],
        trades_entered: int,
        trades_blocked: int,
        elapsed_seconds: float,
    ):
        tradeable = [s for s in signals if s.get("should_trade")]
        self._w()
        self._w("=" * 78)
        self._w("  RUN SUMMARY")
        self._w("=" * 78)
        self._w(f"  Markets evaluated: {len(signals)}")
        self._w(f"  Tradeable:         {len(tradeable)}")
        self._w(f"  Trades entered:    {trades_entered}")
        if trades_blocked:
            self._w(f"  Blocked (cluster): {trades_blocked}")
        self._w(f"  Elapsed:           {elapsed_seconds:.1f}s")
        self._w("=" * 78)
        self._fh.flush()

    # ── per-market ───────────────────────────────────────

    def log_market(
        self,
        index: int,
        total: int,
        market,                       # Market dataclass
        scored,                       # ScoredMarket
        context: dict,                # raw context dict
        enriched,                     # EnrichedContext | None
        llm_detail: Optional[dict],   # from EnrichedLLMEstimator._last_detail
        estimate,                     # ProbabilityEstimate
        position,                     # PositionSize
        trade_result: Optional[str],  # "entered", "skipped", "blocked", None
    ):
        self._market_counter += 1
        self._w()
        self._w("═" * 78)
        self._w(f"  MARKET {index} of {total}")
        self._w("═" * 78)

        # ── market metadata ──
        self._w(f"  Question:      {market.question}")
        self._w(f"  Market ID:     {market.id}")
        self._w(f"  Market price:  {market.yes_price:.1%}")
        if hasattr(market, "category") and market.category:
            self._w(f"  Category:      {market.category}")
        self._w(f"  Volume 24h:    ${market.volume_24h:,.0f}")
        if hasattr(market, "liquidity"):
            self._w(f"  Liquidity:     ${market.liquidity:,.0f}")
        if hasattr(scored, "overall_score"):
            self._w(f"  Scanner score: {scored.overall_score:.3f}")
        days = getattr(market, "days_to_resolution", None)
        if days is not None:
            self._w(f"  Days to res:   {days:.1f}")
        self._w()

        # ── enrichment ──
        self._log_enrichment(enriched)

        # ── LLM detail ──
        self._log_llm_detail(llm_detail)

        # ── ensemble weighting ──
        self._log_ensemble(estimate)

        # ── trade decision ──
        self._log_trade(estimate, position, trade_result)

        self._fh.flush()

    # ── sub-sections ─────────────────────────────────────

    def _log_enrichment(self, enriched):
        self._sep(label="ENRICHMENT")

        if enriched is None:
            self._w("  (no enrichment — simulated mode)")
            self._w()
            return

        sources = enriched.sources_used or []
        self._w(f"  Sources found: {', '.join(sources) if sources else 'none'} "
                f"({len(sources)} of 5)")
        self._w(f"  Time: {enriched.enrichment_time_seconds:.1f}s")
        self._w()

        # News
        news = enriched.news or {}
        if news.get("headlines") or news.get("key_facts"):
            self._w("  News:")
            for h in news.get("headlines", [])[:5]:
                self._w(f"    • {h}")
            sentiment = news.get("sentiment", "neutral")
            strength = news.get("sentiment_strength", 0)
            self._w(f"    Sentiment: {sentiment} (strength: {strength:.0%})")
            if news.get("key_facts"):
                self._w("    Key facts:")
                for f in news["key_facts"][:5]:
                    self._w(f"      - {f}")
            self._w()

        # Cross-platform
        cp = enriched.cross_platform or {}
        if cp.get("yes_price"):
            diff = cp["yes_price"] - enriched.market_price
            self._w("  Cross-platform (Kalshi):")
            self._w(f"    Kalshi:     {cp['yes_price']:.1%}")
            self._w(f"    Polymarket: {enriched.market_price:.1%}")
            self._w(f"    Diff:       {diff:+.1%}"
                    + ("  ⚠ notable" if abs(diff) > 0.03 else ""))
            self._w()

        # Economic indicators
        econ = enriched.economic_data or {}
        if econ.get("indicators"):
            self._w("  Economic indicators (FRED):")
            for ind in econ["indicators"]:
                val = ind.get("latest_value")
                if val is not None:
                    change = ind.get("change", 0)
                    trend = ind.get("trend", "")
                    self._w(f"    {ind['name']}: {val:.2f} "
                            f"(Δ {change:+.2f}, {trend})")
                else:
                    self._w(f"    {ind['name']}: (relevant — check data)")
            self._w()

        # Related markets
        if enriched.related_markets:
            self._w("  Related markets:")
            for rm in enriched.related_markets[:5]:
                rel = rm.get("relation", "related")
                self._w(f"    [{rel}] {rm.get('question', '?')}: "
                        f"{rm.get('price', 0):.0%}")
            self._w()

        # Whale positions
        if enriched.whale_positions:
            self._w("  Whale positions:")
            for wp in enriched.whale_positions[:6]:
                strat = ""
                if wp.get("profile_strategy"):
                    wr = wp.get("profile_win_rate", 0)
                    strat = f" ({wp['profile_strategy']}, {wr:.0%} win rate)"
                self._w(f"    🐋 {wp.get('wallet', '?')[:12]}… "
                        f"{wp.get('side', '?')} ${wp.get('size', 0):,.0f}{strat}")
            self._w()

    def _log_llm_detail(self, detail: Optional[dict]):
        self._sep(label="LLM ESTIMATION")

        if detail is None:
            self._w("  (no LLM detail — simulated or failed)")
            self._w()
            return

        self._w(f"  Model:    {detail.get('model_tag', '?')}")
        self._w(f"  Provider: {detail.get('provider', '?')}")
        self._w()

        # Prompt length
        prompt_len = detail.get("prompt_length", 0)
        if prompt_len:
            self._w(f"  Prompt length: {prompt_len:,} chars")

        # Raw response
        raw = detail.get("raw_response")
        if raw:
            self._w()
            self._w("  Raw LLM response:")
            # Pretty-print if it looks like JSON
            try:
                obj = json.loads(raw.strip().strip("`").lstrip("json").strip())
                for line in json.dumps(obj, indent=2).splitlines():
                    self._w(f"    {line}")
            except (json.JSONDecodeError, ValueError):
                # Plain text — indent and truncate
                for line in raw.splitlines()[:30]:
                    self._w(f"    {line}")
                if raw.count("\n") > 30:
                    self._w(f"    … ({raw.count(chr(10)) - 30} more lines)")

        # Parsed response
        parsed = detail.get("parsed")
        if parsed:
            self._w()
            self._w("  Parsed response:")
            self._w(f"    Base rate anchor:   {parsed.get('base_rate_anchor', '—')}")
            self._w(f"    Base rate reason:   {parsed.get('base_rate_reasoning', '—')}")

            factors_for = parsed.get("factors_for", [])
            if factors_for:
                self._w(f"    Factors FOR YES:")
                for f in factors_for:
                    self._w(f"      + {f}")

            factors_against = parsed.get("factors_against", [])
            if factors_against:
                self._w(f"    Factors AGAINST:")
                for f in factors_against:
                    self._w(f"      - {f}")

            self._w(f"    Probability:       {parsed.get('probability', '—')}")
            self._w(f"    Confidence:        {parsed.get('confidence', '—')}")
            self._w(f"    Reasoning:         {parsed.get('reasoning', '—')}")
            self._w(f"    Information edge:  {parsed.get('information_edge', '—')}")

        # Calibration
        cal = detail.get("calibration")
        if cal:
            self._w()
            self._w("  Calibration:")
            self._w(f"    Raw LLM probability:    {cal['raw']:.1%}")
            self._w(f"    Calibrated probability: {cal['calibrated']:.1%}")
            self._w(f"    Adjustment:             {cal['calibrated'] - cal['raw']:+.1%}")

        # Confidence breakdown
        conf = detail.get("confidence_breakdown")
        if conf:
            self._w()
            self._w("  Confidence build-up:")
            self._w(f"    Base (from LLM):        {conf['base']:.2f} "
                    f"({conf.get('base_label', '?')})")
            if conf.get("enrichment_bonus", 0):
                self._w(f"    + enrichment bonus:     "
                        f"+{conf['enrichment_bonus']:.2f} "
                        f"({conf.get('n_sources', 0)} sources × 0.05)")
            if conf.get("deviation_penalty") is not None and conf["deviation_penalty"] < 1.0:
                self._w(f"    × deviation penalty:    "
                        f"×{conf['deviation_penalty']:.2f} "
                        f"(deviation {conf.get('deviation', 0):.1%} > 15%)")
            self._w(f"    Final confidence:       {conf['final']:.2f}")

        self._w()

    def _log_ensemble(self, estimate):
        self._sep(label="ENSEMBLE WEIGHTING")

        components = estimate.components or {}
        if not components:
            self._w("  (no component data)")
            self._w()
            return

        # Table header
        self._w(f"  {'Component':<22s} {'Base Wt':>7s}  {'Prob':>6s}  "
                f"{'Conf':>5s}  {'Eff Wt':>7s}  {'Contribution':>12s}")
        self._sep(char=" ", label="")

        total_eff_weight = 0.0
        total_contribution = 0.0

        for name, data in components.items():
            if "error" in data:
                self._w(f"  {name:<22s}    ⚠ ERROR: {data['error'][:40]}")
                continue

            prob = data.get("probability", 0)
            conf = data.get("confidence", 0)
            eff_w = data.get("weight", 0)

            # Recover base weight: eff_w = base_w * conf
            base_w = eff_w / conf if conf > 0 else 0

            contribution = prob * eff_w
            total_eff_weight += eff_w
            total_contribution += contribution

            self._w(f"  {name:<22s} {base_w:>7.3f}  {prob:>5.1%}  "
                    f"{conf:>5.2f}  {eff_w:>7.3f}  "
                    f"{contribution:>11.1%}")

        # Totals
        self._w(f"  {'':<22s} {'':>7s}  {'':>6s}  "
                f"{'':>5s}  {'─' * 7}  {'─' * 12}")
        self._w(f"  {'Totals':<22s} {'':>7s}  {'':>6s}  "
                f"{'':>5s}  {total_eff_weight:>7.3f}  "
                f"{total_contribution:>11.1%}")

        # Final weighted average
        if total_eff_weight > 0:
            final = total_contribution / total_eff_weight
            self._w()
            self._w(f"  Final probability: "
                    f"{total_contribution:.1%} / {total_eff_weight:.3f} = "
                    f"{final:.1%}")
        else:
            self._w()
            self._w(f"  Final probability: {estimate.probability:.1%} (fallback)")

        # Agreement
        self._w(f"  Ensemble confidence (agreement): {estimate.confidence:.2f}")
        self._w()

    def _log_trade(self, estimate, position, trade_result: Optional[str]):
        self._sep(label="TRADE DECISION")

        self._w(f"  Our estimate:    {estimate.probability:.1%}")
        self._w(f"  Market price:    {estimate.market_price:.1%}")
        self._w(f"  Edge:            {estimate.edge:+.1%}")

        if estimate.edge > 0:
            self._w(f"  Direction:       YES (market underpriced)")
        elif estimate.edge < 0:
            self._w(f"  Direction:       NO (market overpriced)")
        else:
            self._w(f"  Direction:       —")
        self._w()

        if not position.should_trade:
            self._w(f"  ❌ NO TRADE: {position.rejection_reason}")
        else:
            self._w(f"  Kelly sizing:")
            self._w(f"    Full Kelly:      {position.kelly_fraction:.1%} of bankroll")
            self._w(f"    Fractional:      {position.adjusted_fraction:.1%} "
                    f"(${position.dollar_amount:.2f})")
            self._w(f"    Shares:          {position.shares:.1f} "
                    f"@ ${position.entry_price:.4f}")
            self._w(f"    Max loss:        ${position.max_loss:.2f}")
            self._w(f"    Expected profit: ${position.expected_profit:.2f}")
            self._w(f"    Risk/reward:     {position.risk_reward_ratio:.2f}")
            self._w()

            if trade_result == "entered":
                self._w(f"  ✅ TRADE ENTERED")
            elif trade_result == "blocked":
                self._w(f"  🚫 BLOCKED (cluster exposure limit)")
            elif trade_result == "duplicate":
                self._w(f"  ⏭  SKIPPED (already have open position)")
            elif trade_result == "skipped":
                self._w(f"  ⬜ SKIPPED")
            else:
                self._w(f"  ⏳ PENDING")

        self._w()
