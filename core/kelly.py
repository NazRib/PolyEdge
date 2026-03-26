"""
Kelly Criterion Position Sizing
Determines optimal bet size given an estimated edge and odds.

The Kelly formula maximizes long-term geometric growth rate of your bankroll.
In practice, we use fractional Kelly (typically 0.25x) because:
1. Our edge estimates are uncertain — overestimating edge is catastrophic
2. Kelly-optimal bets have high variance — fractional Kelly reduces drawdowns
3. Real markets have transaction costs that Kelly doesn't account for
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Result of a position sizing calculation."""
    should_trade: bool
    side: str                     # "YES" or "NO"
    kelly_fraction: float         # Full Kelly fraction of bankroll
    adjusted_fraction: float      # After applying fractional Kelly
    dollar_amount: float          # Actual $ to risk
    shares: float                 # Number of shares at current price
    entry_price: float            # Price per share
    
    # Risk metrics
    max_loss: float               # Worst case loss
    expected_profit: float        # Expected profit
    expected_return_pct: float    # Expected return as % of position
    risk_reward_ratio: float      # Expected profit / max loss
    edge: float                   # Estimated edge
    
    # Reasoning
    rejection_reason: str = ""
    
    def summary(self) -> str:
        if not self.should_trade:
            return f"NO TRADE: {self.rejection_reason}"
        return (
            f"{self.side} | ${self.dollar_amount:.2f} ({self.adjusted_fraction:.1%} of bankroll) | "
            f"Edge: {self.edge:+.1%} | EV: ${self.expected_profit:.2f} | "
            f"R:R = {self.risk_reward_ratio:.2f}"
        )


def kelly_criterion(
    estimated_prob: float,
    market_price: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.03,
    max_position_pct: float = 0.10,
    min_bet: float = 5.0,
    confidence: float = 1.0,
) -> PositionSize:
    """
    Calculate optimal position size using the Kelly Criterion.
    
    In prediction markets, the payoff structure is:
    - Buy YES at price p: win (1-p) if Yes, lose p if No
    - Buy NO at price (1-p): win p if Yes outcome doesn't happen
    
    The Kelly formula for this structure:
        f* = (bp - q) / b
    where:
        p = estimated probability of winning
        q = 1 - p  
        b = net odds (payout / stake)
    
    Args:
        estimated_prob: Your estimated probability of Yes (0-1)
        market_price: Current market price of Yes share (0-1)
        bankroll: Total available capital
        kelly_fraction: What fraction of full Kelly to use (0.25 = quarter Kelly)
        min_edge: Minimum edge required to trade
        max_position_pct: Maximum position as % of bankroll
        min_bet: Minimum bet size in dollars
        confidence: How confident you are in the estimate (0-1), further scales Kelly
    
    Returns:
        PositionSize with all calculation details
    """
    
    # Determine which side to trade
    edge_yes = estimated_prob - market_price
    edge_no = (1 - estimated_prob) - (1 - market_price)  # = market_price - estimated_prob
    
    if abs(edge_yes) >= abs(edge_no) and edge_yes > 0:
        # BUY YES
        side = "YES"
        prob_win = estimated_prob
        entry_price = market_price
        edge = edge_yes
    elif edge_no > 0:
        # BUY NO
        side = "NO"
        prob_win = 1 - estimated_prob
        entry_price = 1 - market_price
        edge = edge_no
    else:
        # No edge on either side
        return PositionSize(
            should_trade=False, side="", kelly_fraction=0, adjusted_fraction=0,
            dollar_amount=0, shares=0, entry_price=0, max_loss=0,
            expected_profit=0, expected_return_pct=0, risk_reward_ratio=0,
            edge=max(edge_yes, edge_no),
            rejection_reason=f"No edge found (best edge: {max(edge_yes, edge_no):+.1%})"
        )
    
    # Check minimum edge
    if edge < min_edge:
        return PositionSize(
            should_trade=False, side=side, kelly_fraction=0, adjusted_fraction=0,
            dollar_amount=0, shares=0, entry_price=entry_price, max_loss=0,
            expected_profit=0, expected_return_pct=0, risk_reward_ratio=0,
            edge=edge,
            rejection_reason=f"Edge {edge:.1%} below minimum {min_edge:.1%}"
        )
    
    # Kelly calculation
    # b = net odds = (1 - entry_price) / entry_price
    # For a $1 payout market: risk entry_price to win (1 - entry_price)
    if entry_price <= 0 or entry_price >= 1:
        return PositionSize(
            should_trade=False, side=side, kelly_fraction=0, adjusted_fraction=0,
            dollar_amount=0, shares=0, entry_price=entry_price, max_loss=0,
            expected_profit=0, expected_return_pct=0, risk_reward_ratio=0,
            edge=edge,
            rejection_reason=f"Invalid entry price: {entry_price}"
        )
    
    b = (1 - entry_price) / entry_price  # Net odds
    q = 1 - prob_win
    
    # Kelly fraction: f* = (b*p - q) / b
    f_star = (b * prob_win - q) / b
    
    # Safety: if Kelly says don't bet, don't bet
    if f_star <= 0:
        return PositionSize(
            should_trade=False, side=side, kelly_fraction=f_star, adjusted_fraction=0,
            dollar_amount=0, shares=0, entry_price=entry_price, max_loss=0,
            expected_profit=0, expected_return_pct=0, risk_reward_ratio=0,
            edge=edge,
            rejection_reason=f"Kelly fraction negative ({f_star:.4f}) — no edge at these odds"
        )
    
    # Apply fractional Kelly and confidence scaling
    adjusted_f = f_star * kelly_fraction * confidence
    
    # Apply max position cap
    adjusted_f = min(adjusted_f, max_position_pct)
    
    # Calculate dollar amount
    dollar_amount = bankroll * adjusted_f
    
    # Check minimum bet
    if dollar_amount < min_bet:
        return PositionSize(
            should_trade=False, side=side, kelly_fraction=f_star,
            adjusted_fraction=adjusted_f, dollar_amount=dollar_amount,
            shares=0, entry_price=entry_price, max_loss=dollar_amount,
            expected_profit=0, expected_return_pct=0, risk_reward_ratio=0,
            edge=edge,
            rejection_reason=f"Position ${dollar_amount:.2f} below minimum ${min_bet:.2f}"
        )
    
    # Calculate shares and risk metrics
    shares = dollar_amount / entry_price
    max_loss = dollar_amount  # Lose entire position if wrong
    payout_if_win = shares * 1.0  # Each share pays $1 on correct outcome
    profit_if_win = payout_if_win - dollar_amount
    
    expected_profit = prob_win * profit_if_win - q * max_loss
    expected_return_pct = expected_profit / dollar_amount if dollar_amount > 0 else 0
    risk_reward = profit_if_win / max_loss if max_loss > 0 else 0
    
    return PositionSize(
        should_trade=True,
        side=side,
        kelly_fraction=f_star,
        adjusted_fraction=adjusted_f,
        dollar_amount=round(dollar_amount, 2),
        shares=round(shares, 4),
        entry_price=entry_price,
        max_loss=round(max_loss, 2),
        expected_profit=round(expected_profit, 2),
        expected_return_pct=round(expected_return_pct, 4),
        risk_reward_ratio=round(risk_reward, 2),
        edge=round(edge, 4),
    )


def size_multiple_positions(
    opportunities: list[dict],
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_total_exposure: float = 0.50,
    max_correlated_exposure: float = 0.25,
) -> list[PositionSize]:
    """
    Size multiple positions while respecting portfolio-level constraints.
    
    This is important because Kelly assumes independent bets, but prediction
    market positions are often correlated (e.g., multiple Fed-related markets).
    
    Args:
        opportunities: List of dicts with 'estimated_prob', 'market_price', 
                       'confidence', and optionally 'category'
        bankroll: Total capital
        kelly_fraction: Fractional Kelly to use
        max_total_exposure: Max % of bankroll across all positions
        max_correlated_exposure: Max % in correlated positions (same category)
    
    Returns:
        List of PositionSize objects
    """
    # First pass: size each independently
    positions = []
    for opp in opportunities:
        pos = kelly_criterion(
            estimated_prob=opp["estimated_prob"],
            market_price=opp["market_price"],
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            confidence=opp.get("confidence", 1.0),
        )
        pos_with_meta = (pos, opp.get("category", "unknown"))
        positions.append(pos_with_meta)
    
    # Second pass: apply portfolio constraints
    total_exposure = 0.0
    category_exposure = {}
    final_positions = []
    
    # Sort by expected return (best first)
    positions.sort(key=lambda x: x[0].expected_return_pct, reverse=True)
    
    for pos, category in positions:
        if not pos.should_trade:
            final_positions.append(pos)
            continue
        
        # Check total exposure
        new_total = total_exposure + pos.adjusted_fraction
        if new_total > max_total_exposure:
            remaining = max_total_exposure - total_exposure
            if remaining < 0.01:  # Less than 1% left
                pos.should_trade = False
                pos.rejection_reason = "Portfolio exposure limit reached"
                final_positions.append(pos)
                continue
            # Scale down to fit
            scale = remaining / pos.adjusted_fraction
            pos.adjusted_fraction *= scale
            pos.dollar_amount = round(bankroll * pos.adjusted_fraction, 2)
            pos.shares = round(pos.dollar_amount / pos.entry_price, 4)
        
        # Check category exposure
        cat_exp = category_exposure.get(category, 0)
        if cat_exp + pos.adjusted_fraction > max_correlated_exposure:
            remaining = max_correlated_exposure - cat_exp
            if remaining < 0.01:
                pos.should_trade = False
                pos.rejection_reason = f"Correlated exposure limit for '{category}'"
                final_positions.append(pos)
                continue
            scale = remaining / pos.adjusted_fraction
            pos.adjusted_fraction *= scale
            pos.dollar_amount = round(bankroll * pos.adjusted_fraction, 2)
            pos.shares = round(pos.dollar_amount / pos.entry_price, 4)
        
        total_exposure += pos.adjusted_fraction
        category_exposure[category] = category_exposure.get(category, 0) + pos.adjusted_fraction
        final_positions.append(pos)
    
    return final_positions


if __name__ == "__main__":
    print("\n💰 Kelly Criterion Position Sizing Demo")
    print("=" * 60)
    
    bankroll = 1000.0
    
    examples = [
        ("Strong edge, moderate price", 0.70, 0.55),
        ("Small edge, cheap contract", 0.35, 0.25),
        ("Edge on NO side", 0.30, 0.50),
        ("No edge", 0.50, 0.50),
        ("Tiny edge (below threshold)", 0.52, 0.50),
    ]
    
    for label, est_prob, mkt_price in examples:
        pos = kelly_criterion(est_prob, mkt_price, bankroll, kelly_fraction=0.25)
        print(f"\n{label}:")
        print(f"  Est prob: {est_prob:.0%} | Market: {mkt_price:.0%}")
        print(f"  → {pos.summary()}")
