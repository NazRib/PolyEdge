"""
Probability Estimation Framework
Combines multiple signals into a calibrated probability estimate.

The core insight: prediction market edge comes from estimating probabilities
more accurately than the crowd. This module provides a structured way to 
combine different information sources and track calibration over time.
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProbabilityEstimate:
    """A probability estimate with metadata about how it was produced."""
    market_id: str
    question: str
    probability: float              # Our estimated P(Yes)
    market_price: float             # Current market price
    edge: float                     # probability - market_price (positive = underpriced)
    confidence: float               # 0-1, how confident we are in the estimate
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Component estimates that went into the final number
    components: dict = field(default_factory=dict)
    reasoning: str = ""
    
    @property
    def abs_edge(self) -> float:
        return abs(self.edge)
    
    @property
    def edge_direction(self) -> str:
        if self.edge > 0.02:
            return "BUY_YES"   # Market underprices Yes
        elif self.edge < -0.02:
            return "BUY_NO"    # Market underprices No
        return "NO_TRADE"
    
    @property
    def expected_value(self) -> float:
        """Expected value per dollar risked, from the perspective of trading Yes."""
        if self.edge > 0:
            # Buy Yes at market_price, expect to win with prob=self.probability
            return self.probability * (1 - self.market_price) - (1 - self.probability) * self.market_price
        else:
            # Buy No at (1-market_price), expect to win with prob=(1-self.probability)
            no_price = 1 - self.market_price
            no_prob = 1 - self.probability
            return no_prob * (1 - no_price) - (1 - no_prob) * no_price


class BayesianUpdater:
    """
    Maintains and updates a probability estimate using Bayes' theorem.
    
    Represents uncertainty as a Beta distribution, which is conjugate 
    to the Bernoulli likelihood (perfect for binary prediction markets).
    
    Usage:
        updater = BayesianUpdater(prior=0.5, strength=2)
        updater.update_with_evidence(likelihood_ratio=3.0)
        print(updater.current_probability)  # Shifted toward Yes
    """
    
    def __init__(self, prior: float = 0.5, strength: float = 2.0):
        """
        Args:
            prior: Initial probability estimate (0-1)
            strength: How many "pseudo-observations" the prior is worth.
                      Higher = harder to move. 2 = weak prior, 10 = strong prior.
        """
        # Convert prior probability + strength into Beta(alpha, beta) parameters
        self.alpha = prior * strength
        self.beta = (1 - prior) * strength
        self._history = [(datetime.now(timezone.utc), self.mean, "prior")]
    
    @property
    def mean(self) -> float:
        """Current probability estimate (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def variance(self) -> float:
        """Uncertainty in our estimate."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    @property
    def std(self) -> float:
        return math.sqrt(self.variance)
    
    @property
    def confidence(self) -> float:
        """Confidence score: 0 = very uncertain, 1 = very certain."""
        # Based on how concentrated the Beta distribution is
        total = self.alpha + self.beta
        return min(1.0, total / 50.0)  # Saturates at ~50 pseudo-observations
    
    @property
    def credible_interval_90(self) -> tuple[float, float]:
        """90% credible interval for the true probability."""
        from scipy import stats
        dist = stats.beta(self.alpha, self.beta)
        return (dist.ppf(0.05), dist.ppf(0.95))
    
    def update_with_evidence(
        self, likelihood_ratio: float, label: str = ""
    ):
        """
        Update using a likelihood ratio.
        
        A likelihood ratio of 3.0 means the evidence is 3x more likely 
        under "Yes" than under "No". Values > 1 push toward Yes, < 1 toward No.
        
        This is the most flexible update method — you can convert any
        evidence into a likelihood ratio.
        
        Args:
            likelihood_ratio: P(evidence|Yes) / P(evidence|No)
            label: Description of what evidence triggered this update
        """
        # Convert to equivalent binary observations
        # LR = 3 is roughly like seeing 1.5 Yes outcomes
        if likelihood_ratio > 1:
            self.alpha += math.log2(likelihood_ratio)
        else:
            self.beta += math.log2(1 / likelihood_ratio)
        
        self._history.append((datetime.now(timezone.utc), self.mean, label))
    
    def update_with_observation(self, outcome: bool, weight: float = 1.0):
        """
        Update with a direct binary observation (e.g., a poll result).
        
        Args:
            outcome: True for Yes, False for No
            weight: How much weight to give this observation (default 1.0)
        """
        if outcome:
            self.alpha += weight
        else:
            self.beta += weight
        
        label = f"obs:{'Yes' if outcome else 'No'} (w={weight})"
        self._history.append((datetime.now(timezone.utc), self.mean, label))
    
    def update_with_external_estimate(
        self, estimate: float, credibility: float = 0.5, label: str = ""
    ):
        """
        Incorporate an external probability estimate (e.g., from an LLM or model).
        
        Args:
            estimate: The external estimate (0-1)
            credibility: How much weight to give it (0-1). 
                        0.5 = treat as ~5 pseudo-observations
                        1.0 = treat as ~10 pseudo-observations
        """
        pseudo_n = credibility * 10  # Convert credibility to observation count
        self.alpha += estimate * pseudo_n
        self.beta += (1 - estimate) * pseudo_n
        
        self._history.append((datetime.now(timezone.utc), self.mean, label or f"ext:{estimate:.2f}"))


class EnsembleEstimator:
    """
    Combines multiple probability estimation methods into a weighted ensemble.
    
    Each estimator is a function that takes a market context dict and returns 
    a (probability, confidence) tuple. The ensemble weights estimates by both 
    the assigned weight and the reported confidence.
    
    Usage:
        ensemble = EnsembleEstimator()
        ensemble.add_estimator("base_rate", base_rate_fn, weight=0.3)
        ensemble.add_estimator("momentum", momentum_fn, weight=0.2)
        result = ensemble.estimate(market_context)
    """
    
    def __init__(self):
        self.estimators: list[tuple[str, Callable, float]] = []
    
    def add_estimator(
        self, name: str, fn: Callable[[dict], tuple[float, float]], weight: float
    ):
        """
        Register an estimator.
        
        Args:
            name: Human-readable name
            fn: Function(context_dict) -> (probability, confidence)
            weight: Base weight (will be further adjusted by confidence)
        """
        self.estimators.append((name, fn, weight))
    
    def estimate(self, context: dict) -> ProbabilityEstimate:
        """
        Run all estimators and combine into a final estimate.
        
        Args:
            context: Dict with market data, should include at minimum:
                     'market_id', 'question', 'market_price', 'volume_24h',
                     'days_to_resolution', etc.
        
        Returns:
            ProbabilityEstimate with the combined result
        """
        components = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, fn, base_weight in self.estimators:
            try:
                prob, confidence = fn(context)
                # Clamp probability to valid range
                prob = max(0.01, min(0.99, prob))
                confidence = max(0.0, min(1.0, confidence))
                
                # Effective weight = base_weight * confidence
                effective_weight = base_weight * confidence
                weighted_sum += prob * effective_weight
                total_weight += effective_weight
                
                components[name] = {
                    "probability": prob,
                    "confidence": confidence,
                    "weight": effective_weight,
                }
                
            except Exception as e:
                logger.warning(f"Estimator '{name}' failed: {e}")
                components[name] = {"error": str(e)}
        
        # Compute weighted average
        if total_weight > 0:
            final_prob = weighted_sum / total_weight
        else:
            final_prob = context.get("market_price", 0.5)
        
        # Overall confidence is based on agreement between estimators
        probs = [c["probability"] for c in components.values() if "probability" in c]
        if len(probs) > 1:
            agreement = 1.0 - np.std(probs) * 2  # Low std = high agreement
            overall_confidence = max(0.0, min(1.0, agreement))
        else:
            overall_confidence = 0.3
        
        market_price = context.get("market_price", 0.5)
        
        return ProbabilityEstimate(
            market_id=context.get("market_id", ""),
            question=context.get("question", ""),
            probability=final_prob,
            market_price=market_price,
            edge=final_prob - market_price,
            confidence=overall_confidence,
            components=components,
        )


# ─── Built-in Estimator Functions ────────────────────────

def base_rate_estimator(context: dict) -> tuple[float, float]:
    """
    Simple base rate: just returns the market price as a starting point.
    
    This is the "null model" — it trusts the crowd. Useful as an anchor 
    that other estimators adjust away from.
    """
    return context.get("market_price", 0.5), 0.5  # Moderate confidence


def momentum_estimator(context: dict) -> tuple[float, float]:
    """
    Momentum signal: if price has been moving in a direction, 
    estimate it will continue slightly.
    
    Requires 'price_history' in context (list of (timestamp, price) tuples).
    """
    history = context.get("price_history", [])
    if len(history) < 5:
        return context.get("market_price", 0.5), 0.1  # Low confidence
    
    # Calculate recent momentum (last 20% of history vs first 20%)
    n = len(history)
    recent = np.mean([h[1] if isinstance(h, (list, tuple)) else h for h in history[-n//5:]])
    earlier = np.mean([h[1] if isinstance(h, (list, tuple)) else h for h in history[:n//5]])
    
    momentum = recent - earlier
    
    # Project forward slightly (momentum continuation)
    projected = context.get("market_price", 0.5) + momentum * 0.3
    projected = max(0.01, min(0.99, projected))
    
    # Confidence based on strength of trend
    confidence = min(0.7, abs(momentum) * 5)
    
    return projected, confidence


def book_imbalance_estimator(context: dict) -> tuple[float, float]:
    """
    Order book imbalance signal: if there are significantly more bids 
    than asks (or vice versa), informed traders may be accumulating.
    
    Requires 'bid_depth' and 'ask_depth' in context.
    """
    bid_depth = context.get("bid_depth", 0)
    ask_depth = context.get("ask_depth", 0)
    
    if bid_depth == 0 and ask_depth == 0:
        return context.get("market_price", 0.5), 0.0
    
    total = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total  # -1 to +1
    
    # Positive imbalance = more buying pressure = slightly higher true prob
    market_price = context.get("market_price", 0.5)
    adjustment = imbalance * 0.05  # Max 5% adjustment
    projected = max(0.01, min(0.99, market_price + adjustment))
    
    confidence = min(0.5, abs(imbalance) * 0.7)
    
    return projected, confidence


def whale_tracker_estimator(context: dict) -> tuple[float, float]:
    """
    Whale tracking signal: if large, historically profitable wallets 
    are taking positions, follow their direction.
    
    Requires 'whale_positions' in context: list of dicts with 
    'side' ('YES'/'NO'), 'size', and 'trader_pnl'.
    """
    whale_positions = context.get("whale_positions", [])
    
    if not whale_positions:
        return context.get("market_price", 0.5), 0.0
    
    # Weight whale signals by their historical profitability
    yes_signal = 0.0
    no_signal = 0.0
    
    for wp in whale_positions:
        # Positive PnL = more credible whale
        credibility = min(1.0, max(0.0, wp.get("trader_pnl", 0) / 10000))
        size_weight = min(1.0, wp.get("size", 0) / 5000)
        signal = credibility * size_weight
        
        if wp.get("side", "").upper() == "YES" or wp.get("side", "").upper() == "BUY":
            yes_signal += signal
        else:
            no_signal += signal
    
    total = yes_signal + no_signal
    if total == 0:
        return context.get("market_price", 0.5), 0.0
    
    whale_lean = yes_signal / total  # 0 = all No, 1 = all Yes
    
    market_price = context.get("market_price", 0.5)
    # Blend market price with whale lean
    projected = 0.7 * market_price + 0.3 * whale_lean
    confidence = min(0.6, total * 0.2)
    
    return projected, confidence


def profiled_whale_estimator(context: dict) -> tuple[float, float]:
    """
    Profile-aware whale signal: uses behavioral profiles to weight whale
    positions by category-specific credibility and strategy type.
    
    This is a strict upgrade over whale_tracker_estimator. Market makers
    get heavily discounted (their positions are liquidity, not conviction).
    Specialists get boosted in their category. Conviction traders with
    high win rates in the relevant category get the most weight.
    
    Requires 'whale_positions' in context, where each position dict
    may optionally include profile data:
        - 'profile_strategy': str (CONVICTION, MARKET_MAKER, etc.)
        - 'profile_signal_weight': float (0-1, pre-computed by WhaleProfile)
        - 'profile_category_credibility': float (0-1, for this market's category)
        - 'profile_win_rate': float or None
    
    Falls back to the basic PnL-based weighting if profile data is absent,
    so this is backward-compatible with the old WhaleEnricher output.
    """
    whale_positions = context.get("whale_positions", [])
    
    if not whale_positions:
        return context.get("market_price", 0.5), 0.0
    
    yes_signal = 0.0
    no_signal = 0.0
    total_credible_whales = 0
    
    for wp in whale_positions:
        # Use profile-derived signal weight if available
        signal_weight = wp.get("profile_signal_weight")
        
        if signal_weight is not None:
            # ─── Profile-aware path ──────────────────────
            # Signal weight already accounts for strategy type,
            # category credibility, and win rate. Just scale by
            # position size.
            size_weight = min(1.0, wp.get("size", 0) / 5000)
            signal = signal_weight * size_weight
            
            # Skip market makers entirely if their signal weight is tiny
            strategy = wp.get("profile_strategy", "UNKNOWN")
            if strategy == "MARKET_MAKER" and signal_weight < 0.2:
                continue
            
            total_credible_whales += 1
        else:
            # ─── Fallback: basic PnL-based weighting ─────
            # Same as the old whale_tracker_estimator
            credibility = min(1.0, max(0.0, wp.get("trader_pnl", 0) / 10000))
            size_weight = min(1.0, wp.get("size", 0) / 5000)
            signal = credibility * size_weight
            total_credible_whales += 1
        
        side = wp.get("side", "").upper()
        if side in ("YES", "BUY"):
            yes_signal += signal
        else:
            no_signal += signal
    
    total = yes_signal + no_signal
    if total == 0:
        return context.get("market_price", 0.5), 0.0
    
    whale_lean = yes_signal / total  # 0 = all No, 1 = all Yes
    
    market_price = context.get("market_price", 0.5)
    
    # The blend weight depends on how many credible whales we have.
    # 1 whale → conservative blend (80% market / 20% whale)
    # 3+ whales → stronger blend (65% market / 35% whale)
    whale_blend = min(0.35, 0.15 + total_credible_whales * 0.07)
    projected = (1 - whale_blend) * market_price + whale_blend * whale_lean
    projected = max(0.01, min(0.99, projected))
    
    # Confidence scales with signal strength AND number of credible whales
    confidence = min(0.7, total * 0.15 + total_credible_whales * 0.05)
    
    return projected, confidence


def create_default_ensemble() -> EnsembleEstimator:
    """Create an ensemble with all built-in estimators."""
    ensemble = EnsembleEstimator()
    ensemble.add_estimator("base_rate", base_rate_estimator, weight=0.30)
    ensemble.add_estimator("momentum", momentum_estimator, weight=0.25)
    ensemble.add_estimator("book_imbalance", book_imbalance_estimator, weight=0.20)
    ensemble.add_estimator("whale_tracker", whale_tracker_estimator, weight=0.25)
    return ensemble


def create_profiled_ensemble() -> EnsembleEstimator:
    """
    Create an ensemble that uses the profile-aware whale estimator.
    
    Compared to the default ensemble:
    - Uses profiled_whale_estimator instead of whale_tracker_estimator
    - Gives the whale signal slightly more weight (0.30 vs 0.25) because
      profile-aware signals are higher quality (market makers filtered out)
    - Reduces base_rate weight slightly to compensate
    """
    ensemble = EnsembleEstimator()
    ensemble.add_estimator("base_rate", base_rate_estimator, weight=0.25)
    ensemble.add_estimator("momentum", momentum_estimator, weight=0.25)
    ensemble.add_estimator("book_imbalance", book_imbalance_estimator, weight=0.20)
    ensemble.add_estimator("whale_profiled", profiled_whale_estimator, weight=0.30)
    return ensemble


# ─── Calibration Tracking ────────────────────────────────

class CalibrationTracker:
    """
    Tracks prediction calibration over time using Brier scores.
    
    Brier Score = mean( (predicted_probability - actual_outcome)^2 )
    
    Interpretation:
        0.25 = coin flip (predicting 50% for everything)
        0.20 = decent
        0.15 = good
        0.10 = excellent
        0.00 = perfect
    
    Usage:
        tracker = CalibrationTracker()
        tracker.record(predicted=0.7, actual=True)   # You said 70%, it happened
        tracker.record(predicted=0.3, actual=True)    # You said 30%, it happened (bad!)
        print(tracker.brier_score)
    """
    
    def __init__(self):
        self.predictions: list[tuple[float, int, str, datetime]] = []
    
    def record(self, predicted: float, actual: bool, market_id: str = ""):
        """Record a resolved prediction."""
        self.predictions.append((
            predicted,
            1 if actual else 0,
            market_id,
            datetime.now(timezone.utc),
        ))
    
    @property
    def brier_score(self) -> Optional[float]:
        """Current Brier score across all predictions."""
        if not self.predictions:
            return None
        scores = [(p - a) ** 2 for p, a, _, _ in self.predictions]
        return np.mean(scores)
    
    @property
    def n_predictions(self) -> int:
        return len(self.predictions)
    
    def calibration_curve(self, n_bins: int = 10) -> list[tuple[float, float, int]]:
        """
        Compute calibration curve: for each probability bin, 
        what fraction actually occurred?
        
        Returns list of (bin_center, actual_frequency, count) tuples.
        Perfect calibration: bin_center ≈ actual_frequency for all bins.
        """
        if not self.predictions:
            return []
        
        bins = np.linspace(0, 1, n_bins + 1)
        curve = []
        
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            in_bin = [(p, a) for p, a, _, _ in self.predictions if lo <= p < hi]
            if in_bin:
                center = (lo + hi) / 2
                freq = np.mean([a for _, a in in_bin])
                curve.append((center, freq, len(in_bin)))
        
        return curve
    
    def summary(self) -> str:
        """Human-readable calibration summary."""
        if not self.predictions:
            return "No predictions recorded yet."
        
        bs = self.brier_score
        n = self.n_predictions
        
        # Categorize Brier score
        if bs < 0.10:
            quality = "Excellent"
        elif bs < 0.15:
            quality = "Good"
        elif bs < 0.20:
            quality = "Decent"
        elif bs < 0.25:
            quality = "Below average"
        else:
            quality = "Worse than random"
        
        lines = [
            f"Calibration Report ({n} predictions)",
            f"  Brier Score: {bs:.4f} ({quality})",
            f"  vs. coin flip: {bs / 0.25:.0%} of random baseline",
        ]
        
        curve = self.calibration_curve(5)
        if curve:
            lines.append("  Calibration curve:")
            for center, freq, count in curve:
                bar = "█" * int(freq * 20)
                lines.append(f"    {center:.0%}: {freq:.0%} actual ({count} predictions) {bar}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo: create ensemble and estimate for a fake market
    ensemble = create_default_ensemble()
    
    context = {
        "market_id": "demo-123",
        "question": "Will the Fed cut rates in June 2026?",
        "market_price": 0.45,
        "price_history": [(i, 0.40 + i * 0.005) for i in range(20)],
        "bid_depth": 15000,
        "ask_depth": 8000,
        "whale_positions": [
            {"side": "YES", "size": 5000, "trader_pnl": 25000},
            {"side": "NO", "size": 2000, "trader_pnl": 5000},
        ],
    }
    
    estimate = ensemble.estimate(context)
    
    print("\n📊 Probability Estimation Demo")
    print("=" * 50)
    print(f"Question: {estimate.question}")
    print(f"Market price:    {estimate.market_price:.1%}")
    print(f"Our estimate:    {estimate.probability:.1%}")
    print(f"Edge:            {estimate.edge:+.1%}")
    print(f"Direction:       {estimate.edge_direction}")
    print(f"Confidence:      {estimate.confidence:.1%}")
    print(f"Expected Value:  {estimate.expected_value:+.4f}")
    print(f"\nComponents:")
    for name, comp in estimate.components.items():
        if "probability" in comp:
            print(f"  {name}: {comp['probability']:.1%} (conf: {comp['confidence']:.1%}, w: {comp['weight']:.3f})")
