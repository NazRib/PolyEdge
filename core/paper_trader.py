"""
Paper Trading Engine
Simulates trades without real money to validate your edge before going live.

The cardinal rule: never trade real money until your Brier score demonstrates
a statistically significant edge over a meaningful sample (50+ resolved markets).
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from core.kelly import PositionSize
from core.probability import ProbabilityEstimate, CalibrationTracker

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """A single paper trade."""
    trade_id: str
    market_id: str
    question: str
    side: str                        # "YES" or "NO"
    entry_price: float
    shares: float
    dollar_amount: float
    estimated_probability: float     # Our estimate at time of trade
    market_price_at_entry: float     # Market price when we entered
    edge_at_entry: float
    confidence: float
    
    timestamp: str = ""
    status: str = "OPEN"             # OPEN, RESOLVED_WIN, RESOLVED_LOSS, CLOSED
    exit_price: Optional[float] = None
    resolution: Optional[bool] = None  # True=Yes won, False=No won
    pnl: float = 0.0
    strategy_tag: str = ""           # e.g. "enriched", "enriched+profiles" for A/B comparison
    
    def resolve(self, outcome: bool):
        """Resolve the trade with the actual outcome."""
        self.resolution = outcome
        
        if self.side == "YES":
            if outcome:
                self.pnl = self.shares * 1.0 - self.dollar_amount
                self.status = "RESOLVED_WIN"
            else:
                self.pnl = -self.dollar_amount
                self.status = "RESOLVED_LOSS"
        else:  # NO side
            if not outcome:
                self.pnl = self.shares * 1.0 - self.dollar_amount
                self.status = "RESOLVED_WIN"
            else:
                self.pnl = -self.dollar_amount
                self.status = "RESOLVED_LOSS"
        
        return self.pnl
    
    def mark_to_market(self, current_price: float) -> float:
        """Calculate unrealized PnL based on current market price."""
        if self.side == "YES":
            current_value = self.shares * current_price
        else:
            current_value = self.shares * (1 - current_price)
        return current_value - self.dollar_amount


@dataclass 
class PortfolioSnapshot:
    """Point-in-time state of the paper trading portfolio."""
    timestamp: str
    bankroll: float
    total_invested: float
    unrealized_pnl: float
    realized_pnl: float
    open_positions: int
    resolved_trades: int
    win_rate: float
    brier_score: Optional[float]
    
    @property
    def total_value(self) -> float:
        return self.bankroll + self.unrealized_pnl


class PaperTrader:
    """
    Paper trading engine that tracks positions, PnL, and calibration.
    
    Usage:
        trader = PaperTrader(bankroll=1000)
        trader.enter_trade(estimate, position_size)
        trader.resolve_trade("trade-123", outcome=True)
        print(trader.report())
    """
    
    def __init__(
        self,
        bankroll: float = 1000.0,
        data_dir: str = "data",
    ):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.trades: list[PaperTrade] = []
        self.calibration = CalibrationTracker()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._trade_counter = 0
        
        # Try to load existing state
        self._load_state()
    
    def enter_trade(
        self,
        estimate: ProbabilityEstimate,
        position: PositionSize,
        strategy_tag: str = "",
    ) -> Optional[PaperTrade]:
        """
        Enter a paper trade based on a probability estimate and position sizing.
        
        Args:
            estimate: Probability estimate with market context
            position: Kelly-sized position
            strategy_tag: Label for A/B comparison (e.g. "enriched", "enriched+profiles")
        
        Returns the PaperTrade if entered, None if rejected.
        """
        if not position.should_trade:
            logger.info(f"Trade rejected: {position.rejection_reason}")
            return None
        
        if position.dollar_amount > self.bankroll:
            logger.warning(
                f"Insufficient bankroll: need ${position.dollar_amount:.2f}, "
                f"have ${self.bankroll:.2f}"
            )
            return None
        
        self._trade_counter += 1
        trade = PaperTrade(
            trade_id=f"PT-{self._trade_counter:04d}",
            market_id=estimate.market_id,
            question=estimate.question,
            side=position.side,
            entry_price=position.entry_price,
            shares=position.shares,
            dollar_amount=position.dollar_amount,
            estimated_probability=estimate.probability,
            market_price_at_entry=estimate.market_price,
            edge_at_entry=estimate.edge,
            confidence=estimate.confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy_tag=strategy_tag,
        )
        
        self.bankroll -= position.dollar_amount
        self.trades.append(trade)
        self._save_state()
        
        logger.info(
            f"📝 Entered trade {trade.trade_id}: {trade.side} {trade.question[:50]}... "
            f"@ {trade.entry_price:.2f} for ${trade.dollar_amount:.2f}"
        )
        
        return trade
    
    def resolve_trade(self, trade_id: str, outcome: bool) -> Optional[float]:
        """
        Resolve a trade with the actual outcome.
        
        Args:
            trade_id: The trade to resolve
            outcome: True if Yes won, False if No won
            
        Returns:
            PnL of the trade, or None if trade not found
        """
        trade = self._find_trade(trade_id)
        if trade is None:
            logger.warning(f"Trade {trade_id} not found")
            return None
        
        if trade.status != "OPEN":
            logger.warning(f"Trade {trade_id} already resolved: {trade.status}")
            return None
        
        pnl = trade.resolve(outcome)
        self.bankroll += trade.dollar_amount + pnl  # Return capital + profit/loss
        
        # Record for calibration tracking
        # The prediction we track is: P(the side we bet on wins)
        if trade.side == "YES":
            predicted_prob = trade.estimated_probability
            actual_outcome = outcome
        else:
            predicted_prob = 1 - trade.estimated_probability
            actual_outcome = not outcome
        
        self.calibration.record(predicted_prob, actual_outcome, trade.market_id)
        self._save_state()
        
        status = "✅ WIN" if pnl > 0 else "❌ LOSS"
        logger.info(
            f"{status} Trade {trade_id}: PnL = ${pnl:+.2f} | "
            f"Bankroll: ${self.bankroll:.2f}"
        )
        
        return pnl
    
    def resolve_market(self, market_id: str, outcome: bool) -> list[float]:
        """Resolve all trades in a given market."""
        pnls = []
        for trade in self.trades:
            if trade.market_id == market_id and trade.status == "OPEN":
                pnl = self.resolve_trade(trade.trade_id, outcome)
                if pnl is not None:
                    pnls.append(pnl)
        return pnls
    
    def snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio state."""
        open_trades = [t for t in self.trades if t.status == "OPEN"]
        resolved = [t for t in self.trades if t.status.startswith("RESOLVED")]
        wins = [t for t in resolved if t.status == "RESOLVED_WIN"]
        
        total_invested = sum(t.dollar_amount for t in open_trades)
        realized_pnl = sum(t.pnl for t in resolved)
        
        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            bankroll=self.bankroll,
            total_invested=total_invested,
            unrealized_pnl=0,  # Would need current prices to calculate
            realized_pnl=realized_pnl,
            open_positions=len(open_trades),
            resolved_trades=len(resolved),
            win_rate=len(wins) / len(resolved) if resolved else 0,
            brier_score=self.calibration.brier_score,
        )
    
    def report(self) -> str:
        """Generate a comprehensive performance report."""
        snap = self.snapshot()
        resolved = [t for t in self.trades if t.status.startswith("RESOLVED")]
        
        lines = [
            "",
            "=" * 60,
            "  POLYMARKET EDGE — Paper Trading Report",
            "=" * 60,
            "",
            f"  Starting Bankroll:  ${self.initial_bankroll:>10,.2f}",
            f"  Current Bankroll:   ${self.bankroll:>10,.2f}",
            f"  Realized PnL:       ${snap.realized_pnl:>+10,.2f}",
            f"  Return:             {(self.bankroll / self.initial_bankroll - 1):>+10.1%}",
            "",
            f"  Open Positions:     {snap.open_positions:>10d}",
            f"  Resolved Trades:    {snap.resolved_trades:>10d}",
            f"  Win Rate:           {snap.win_rate:>10.0%}",
            "",
        ]
        
        # Calibration section
        bs = self.calibration.brier_score
        if bs is not None:
            if bs < 0.15:
                emoji = "🟢"
            elif bs < 0.20:
                emoji = "🟡"
            elif bs < 0.25:
                emoji = "🟠"
            else:
                emoji = "🔴"
            
            lines.extend([
                f"  Brier Score:        {bs:>10.4f} {emoji}",
                f"  vs Random (0.25):   {(0.25 - bs) / 0.25:>+10.1%} improvement",
                "",
            ])
            
            # Recommendation
            if len(resolved) < 30:
                lines.append("  ⏳ Need 30+ resolved trades for meaningful assessment")
            elif bs < 0.20:
                lines.append("  ✅ Calibration looks good — consider moving to real money")
            elif bs < 0.25:
                lines.append("  ⚠️  Edge is marginal — keep paper trading and refining")
            else:
                lines.append("  🛑 No demonstrated edge — do NOT trade real money yet")
        else:
            lines.append("  No resolved predictions yet — keep trading!")
        
        # Recent trades
        if resolved:
            lines.extend(["", "  Recent Resolved Trades:"])
            for t in reversed(resolved[-10:]):
                status = "✅" if t.pnl > 0 else "❌"
                lines.append(
                    f"    {status} {t.trade_id} | {t.side} @ {t.entry_price:.2f} | "
                    f"PnL: ${t.pnl:+.2f} | Edge: {t.edge_at_entry:+.1%} | "
                    f"{t.question[:35]}..."
                )
        
        # Calibration curve
        cal_summary = self.calibration.summary()
        if "Calibration curve" in cal_summary:
            lines.extend(["", "  " + cal_summary.replace("\n", "\n  ")])
        
        lines.append("")
        return "\n".join(lines)

    def compare_strategies(self) -> str:
        """
        Compare performance across strategy tags.
        
        Shows side-by-side Brier scores, PnL, and win rates for each
        strategy tag (e.g. "enriched" vs "enriched+profiles").
        """
        # Group trades by strategy_tag
        tags: dict[str, list[PaperTrade]] = {}
        for t in self.trades:
            tag = t.strategy_tag or "untagged"
            if tag not in tags:
                tags[tag] = []
            tags[tag].append(t)

        if len(tags) < 2:
            return (
                "Need trades from at least 2 strategies to compare.\n"
                "Run the pipeline with and without --whale-profiles to generate "
                "trades with different strategy tags."
            )

        lines = [
            "",
            "=" * 80,
            "  STRATEGY COMPARISON — A/B Performance",
            "=" * 80,
            "",
        ]

        for tag in sorted(tags.keys()):
            trades = tags[tag]
            resolved = [t for t in trades if t.status.startswith("RESOLVED")]
            wins = [t for t in resolved if t.pnl > 0]
            total_pnl = sum(t.pnl for t in resolved)
            open_trades = [t for t in trades if t.status == "OPEN"]
            total_risked = sum(t.dollar_amount for t in resolved)

            lines.append(f"  ┌─ {tag} {'─' * max(1, 65 - len(tag))}")
            lines.append(f"  │ Trades:    {len(trades)} total ({len(open_trades)} open, {len(resolved)} resolved)")

            if resolved:
                win_rate = len(wins) / len(resolved)
                lines.append(f"  │ Win rate:  {win_rate:.0%} ({len(wins)}W / {len(resolved) - len(wins)}L)")
                lines.append(f"  │ PnL:      ${total_pnl:+,.2f} (risked ${total_risked:,.2f})")

                if total_risked > 0:
                    roi = total_pnl / total_risked
                    lines.append(f"  │ ROI:      {roi:+.1%}")

                # Compute Brier score for this tag's trades
                brier_sum = 0
                brier_count = 0
                for t in resolved:
                    if t.side == "YES":
                        pred = t.estimated_probability
                        actual = 1.0 if t.resolution else 0.0
                    else:
                        pred = 1 - t.estimated_probability
                        actual = 0.0 if t.resolution else 1.0
                    brier_sum += (pred - actual) ** 2
                    brier_count += 1

                if brier_count >= 5:
                    brier = brier_sum / brier_count
                    improvement = (0.25 - brier) / 0.25 * 100
                    lines.append(f"  │ Brier:    {brier:.4f} ({improvement:+.1f}% vs baseline)")
                else:
                    lines.append(f"  │ Brier:    need {5 - brier_count} more resolutions")

                # Average edge and position size
                avg_edge = np.mean([abs(t.edge_at_entry) for t in resolved])
                avg_size = np.mean([t.dollar_amount for t in trades])
                lines.append(f"  │ Avg edge: {avg_edge:.1%}  |  Avg size: ${avg_size:.2f}")
            else:
                lines.append(f"  │ No resolved trades yet")

            lines.append(f"  └{'─' * 78}")
            lines.append("")

        # Head-to-head on overlapping markets
        if len(tags) == 2:
            tag_names = sorted(tags.keys())
            trades_a = {t.market_id: t for t in tags[tag_names[0]]}
            trades_b = {t.market_id: t for t in tags[tag_names[1]]}
            overlap = set(trades_a.keys()) & set(trades_b.keys())
            only_a = set(trades_a.keys()) - set(trades_b.keys())
            only_b = set(trades_b.keys()) - set(trades_a.keys())

            lines.append(f"  Market overlap:")
            lines.append(f"    Both strategies:  {len(overlap)} markets")
            lines.append(f"    Only {tag_names[0]}: {len(only_a)} markets")
            lines.append(f"    Only {tag_names[1]}: {len(only_b)} markets")

            # Check for directional disagreements
            disagreements = []
            for mid in overlap:
                a, b = trades_a[mid], trades_b[mid]
                if a.side != b.side:
                    disagreements.append((a, b))
            if disagreements:
                lines.append(f"\n  ⚠ Directional disagreements ({len(disagreements)}):")
                for a, b in disagreements[:5]:
                    a_result = f"PnL ${a.pnl:+.2f}" if a.status.startswith("RESOLVED") else "open"
                    b_result = f"PnL ${b.pnl:+.2f}" if b.status.startswith("RESOLVED") else "open"
                    lines.append(
                        f"    {a.question[:40]:40s}"
                    )
                    lines.append(
                        f"      {tag_names[0]:20s}: {a.side} @ {a.entry_price:.2f} ({a_result})"
                    )
                    lines.append(
                        f"      {tag_names[1]:20s}: {b.side} @ {b.entry_price:.2f} ({b_result})"
                    )

        lines.extend(["", "=" * 80])
        return "\n".join(lines)

    def _find_trade(self, trade_id: str) -> Optional[PaperTrade]:
        for t in self.trades:
            if t.trade_id == trade_id:
                return t
        return None
    
    def _save_state(self):
        """Persist state to disk."""
        state = {
            "bankroll": self.bankroll,
            "initial_bankroll": self.initial_bankroll,
            "trade_counter": self._trade_counter,
            "trades": [asdict(t) for t in self.trades],
            "calibration": [
                {"predicted": p, "actual": a, "market_id": mid, "time": ts.isoformat()}
                for p, a, mid, ts in self.calibration.predictions
            ],
        }
        
        filepath = self.data_dir / "paper_trades.json"
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)
    
    def _load_state(self):
        """Load state from disk if available."""
        filepath = self.data_dir / "paper_trades.json"
        if not filepath.exists():
            return
        
        try:
            with open(filepath) as f:
                state = json.load(f)
            
            self.bankroll = state["bankroll"]
            self.initial_bankroll = state["initial_bankroll"]
            self._trade_counter = state["trade_counter"]
            
            self.trades = []
            for t_dict in state.get("trades", []):
                self.trades.append(PaperTrade(**t_dict))
            
            for cal in state.get("calibration", []):
                self.calibration.record(
                    cal["predicted"], 
                    bool(cal["actual"]),
                    cal.get("market_id", ""),
                )
            
            logger.info(
                f"Loaded state: {len(self.trades)} trades, "
                f"bankroll=${self.bankroll:.2f}"
            )
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    print("\n🧪 Paper Trading Demo")
    print("=" * 60)
    
    # Create a fresh paper trader
    trader = PaperTrader(bankroll=1000, data_dir="data/demo")
    
    # Simulate some trades
    from core.probability import ProbabilityEstimate
    from core.kelly import kelly_criterion
    
    demo_markets = [
        ("Will Bitcoin hit $120K?", "btc-120k", 0.35, 0.72, True),
        ("Fed rate cut in June?", "fed-june", 0.45, 0.60, False),
        ("Lakers win NBA title?", "lakers-nba", 0.12, 0.22, False),
        ("GDP growth > 3%?", "gdp-3pct", 0.55, 0.50, True),
        ("Elon tweets about AI?", "elon-ai", 0.85, 0.75, True),
    ]
    
    for question, market_id, mkt_price, our_est, outcome in demo_markets:
        # Create estimate
        estimate = ProbabilityEstimate(
            market_id=market_id,
            question=question,
            probability=our_est,
            market_price=mkt_price,
            edge=our_est - mkt_price,
            confidence=0.6,
        )
        
        # Size the position
        position = kelly_criterion(
            estimated_prob=our_est,
            market_price=mkt_price,
            bankroll=trader.bankroll,
            kelly_fraction=0.25,
        )
        
        # Enter trade
        trade = trader.enter_trade(estimate, position)
        
        # Immediately resolve for demo purposes
        if trade:
            trader.resolve_trade(trade.trade_id, outcome)
    
    # Print report
    print(trader.report())
