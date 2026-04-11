"""
LLM-Powered Probability Estimator
Uses Claude to analyze prediction market questions with structured context,
then applies a learned calibration layer to correct for systematic biases.

Architecture:
    1. Context Builder — gathers structured info about the market question
       (description, category, historical base rates, recent price action,
        order book signals, time to resolution)
    2. LLM Forecaster — prompts Claude with structured context and asks for
       a probability estimate with reasoning
    3. Calibration Layer — adjusts the raw LLM output based on historical
       performance (LLMs tend to be overconfident, especially near extremes)
    4. Confidence Scorer — estimates how reliable this particular prediction is

Key insight from the research: naive LLM prompting is no better than a coin
flip. The edge comes from (a) structured context injection, (b) chain-of-thought
reasoning with explicit base rate anchoring, and (c) post-hoc calibration.
"""

import json
import math
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from core.llm_providers import (
    call_llm,
    PROVIDER_CLAUDE, PROVIDER_GPT, validate_provider, model_tag_for_provider,
    DEFAULT_CLAUDE_MODEL,
)

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = DEFAULT_CLAUDE_MODEL

# Calibration defaults (learned over time)
DEFAULT_CALIBRATION = {
    "slope": 0.85,       # LLMs are ~15% overconfident on average
    "intercept": 0.075,  # Slight bias toward 50%
    "extreme_pull": 0.05, # Pull extreme estimates toward center
}

# GPT models appear better-calibrated out of the box; start with identity
# calibration (no adjustment) and let the system learn GPT's actual bias
# from resolved trades.  Override per-provider via CalibrationModel(provider=).
PROVIDER_CALIBRATION_DEFAULTS = {
    "claude": DEFAULT_CALIBRATION,
    "gpt": {
        "slope": 1.0,         # No compression — pass through raw estimate
        "intercept": 0.0,     # No intercept shift
        "extreme_pull": 0.0,  # No extreme pulling
    },
}


# ─── Data Structures ────────────────────────────────────

@dataclass
class MarketContext:
    """All the structured context we feed to the LLM."""
    question: str
    description: str
    category: str
    current_price: float
    volume_24h: float
    total_volume: float
    liquidity: float
    days_to_resolution: Optional[float]
    
    # Price dynamics
    price_7d_ago: Optional[float] = None
    price_24h_ago: Optional[float] = None
    price_trend: str = ""              # "rising", "falling", "stable"
    price_volatility: float = 0.0
    
    # Order book signals
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    spread: float = 0.0
    book_imbalance: float = 0.0        # positive = more bids
    
    # Category-specific base rates
    base_rate: Optional[float] = None
    base_rate_source: str = ""
    
    # Related markets
    related_markets: list[dict] = field(default_factory=list)
    
    # Additional context
    tags: list[str] = field(default_factory=list)
    resolution_source: str = ""
    
    def to_prompt_context(self) -> str:
        """Format all context into a structured prompt section."""
        lines = [
            f"QUESTION: {self.question}",
            f"DESCRIPTION: {self.description}" if self.description else "",
            f"CATEGORY: {self.category}" if self.category else "",
            "",
            "MARKET DATA:",
            f"  Current market price (implied probability): {self.current_price:.1%}",
            f"  24h trading volume: ${self.volume_24h:,.0f}",
            f"  Total volume: ${self.total_volume:,.0f}",
            f"  Liquidity: ${self.liquidity:,.0f}",
        ]
        
        if self.days_to_resolution is not None:
            lines.append(f"  Days until resolution: {self.days_to_resolution:.1f}")
        
        # Price dynamics
        if self.price_24h_ago is not None:
            change_24h = self.current_price - self.price_24h_ago
            lines.append(f"  Price 24h ago: {self.price_24h_ago:.1%} (change: {change_24h:+.1%})")
        if self.price_7d_ago is not None:
            change_7d = self.current_price - self.price_7d_ago
            lines.append(f"  Price 7d ago: {self.price_7d_ago:.1%} (change: {change_7d:+.1%})")
        if self.price_trend:
            lines.append(f"  Price trend: {self.price_trend}")
        if self.price_volatility > 0:
            lines.append(f"  Price volatility (7d): {self.price_volatility:.3f}")
        
        # Order book
        if self.bid_depth > 0 or self.ask_depth > 0:
            lines.extend([
                "",
                "ORDER BOOK:",
                f"  Bid depth: ${self.bid_depth:,.0f}",
                f"  Ask depth: ${self.ask_depth:,.0f}",
                f"  Spread: {self.spread:.4f}",
                f"  Imbalance: {self.book_imbalance:+.2f} ({'more buyers' if self.book_imbalance > 0.1 else 'more sellers' if self.book_imbalance < -0.1 else 'balanced'})",
            ])
        
        # Base rate
        if self.base_rate is not None:
            lines.extend([
                "",
                "HISTORICAL BASE RATE:",
                f"  Base rate: {self.base_rate:.1%}",
                f"  Source: {self.base_rate_source}",
            ])
        
        # Related markets
        if self.related_markets:
            lines.extend(["", "RELATED MARKETS:"])
            for rm in self.related_markets[:5]:
                lines.append(
                    f"  - {rm.get('question', 'N/A')}: "
                    f"{rm.get('price', 0):.1%}"
                )
        
        return "\n".join(l for l in lines if l is not None)


@dataclass
class LLMForecast:
    """Raw output from the LLM forecaster."""
    raw_probability: float           # LLM's stated probability
    calibrated_probability: float    # After calibration adjustment
    reasoning: str                   # LLM's chain-of-thought
    key_factors_for: list[str]       # Factors supporting Yes
    key_factors_against: list[str]   # Factors supporting No
    confidence_level: str            # "low", "medium", "high"
    confidence_score: float          # Numeric 0-1
    base_rate_anchor: Optional[float]  # What base rate the LLM used
    information_edge: str            # Where the LLM thinks edge might exist
    model_used: str = DEFAULT_MODEL
    timestamp: str = ""
    
    @property
    def calibration_adjustment(self) -> float:
        return self.calibrated_probability - self.raw_probability


# ─── The Forecasting Prompt ──────────────────────────────

FORECASTER_SYSTEM_PROMPT = """You are a superforecaster — a calibrated probability estimator trained in 
the methods of Philip Tetlock's Good Judgment Project. Your goal is to estimate 
the TRUE probability of an event, which may differ from the current market price.

CRITICAL RULES FOR GOOD FORECASTING:

1. START WITH THE BASE RATE. Before considering any specific evidence, ask: 
   "What is the historical frequency of events like this?" If a base rate is 
   provided, anchor to it. If not, estimate one from your knowledge.

2. UPDATE INCREMENTALLY. From the base rate, adjust up or down based on 
   specific evidence. Each piece of evidence should move you a small amount, 
   not a large amount. Beware of overreacting to vivid or recent information.

3. CONSIDER BOTH SIDES. For every factor pushing toward Yes, identify a factor 
   pushing toward No. Unbalanced reasoning leads to overconfidence.

4. BE PRECISE BUT HONEST ABOUT UNCERTAINTY. Don't round to convenient numbers. 
   0.63 is more useful than "around 60%". But if you're genuinely uncertain, 
   say so — a wide confidence range is more honest than false precision.

5. RESPECT THE MARKET. The current market price represents the wisdom of 
   thousands of traders. You should only deviate from it significantly if you 
   have a clear, articulable reason. The market is wrong sometimes, but it's 
   right more often than any individual.

6. WATCH FOR THESE BIASES:
   - Anchoring too heavily on the market price (you should have your own view)
   - Anchoring too heavily on narratives (stories feel true but base rates win)
   - Neglecting regression to the mean
   - Overweighting recent events
   - Confusing "I want X to happen" with "X will happen"

You must respond ONLY with valid JSON in the exact format specified."""


def build_forecast_prompt(context: MarketContext) -> str:
    """Build the user prompt for the LLM forecaster."""
    return f"""Analyze this prediction market and estimate the true probability of the outcome.

{context.to_prompt_context()}

Respond with ONLY this JSON structure, no other text:
{{
    "base_rate_anchor": <float between 0 and 1 — your starting base rate>,
    "base_rate_reasoning": "<1-2 sentences on where this base rate comes from>",
    "factors_for": ["<factor 1 supporting YES>", "<factor 2>", ...],
    "factors_against": ["<factor 1 supporting NO>", "<factor 2>", ...],
    "probability": <float between 0.01 and 0.99 — your final estimate>,
    "reasoning": "<2-4 sentence explanation of how you moved from base rate to final estimate>",
    "confidence": "<low|medium|high>",
    "information_edge": "<where you think the market might be wrong, or 'none' if market seems efficient>"
}}"""


# ─── LLM Client ─────────────────────────────────────────

def call_claude(
    user_prompt: str,
    system_prompt: str = FORECASTER_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Call the Claude API and return the text response.
    
    Uses low temperature (0.2) for more deterministic probability estimates.
    Retries with exponential backoff on 429 rate limit errors.
    """
    import requests
    
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning("No ANTHROPIC_API_KEY set — returning None")
        return None
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                ANTHROPIC_API_URL, headers=headers, json=payload, timeout=30
            )
            
            if resp.status_code == 429:
                import time
                retry_after = int(resp.headers.get("retry-after", 2 ** (attempt + 1)))
                wait = max(retry_after, 2 ** (attempt + 1))
                logger.info(f"Rate limited (estimator), waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            
            resp.raise_for_status()
            data = resp.json()
            
            # Extract text from response
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block["text"]
            
            return text.strip()
        
        except requests.exceptions.HTTPError:
            # Non-429 HTTP errors — don't retry
            logger.error(f"Claude API call failed: {resp.status_code} {resp.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return None
    
    logger.warning("Claude estimator: max retries exhausted")
    return None


def parse_llm_response(raw_text: str) -> Optional[dict]:
    """Parse the JSON response from the LLM, handling common formatting issues.
    
    GPT reasoning models may wrap JSON in markdown fences, include preamble
    text, or embed it inside longer prose. This parser tries multiple
    extraction strategies before giving up.
    """
    if not raw_text:
        return None
    
    text = raw_text.strip()
    
    # Strategy 1: Strip markdown code fences
    if "```" in text:
        # Find JSON between code fences (most reliable)
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # Remove fences and try the remainder
            text = re.sub(r'```(?:json)?', '', text).strip()
    
    # Strategy 2: Direct parse
    try:
        data = json.loads(text)
        return _validate_forecast(data)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    
    # Strategy 3: Find the first complete JSON object in the text
    # (handles preamble text, trailing explanation, thinking output)
    brace_start = text.find('{')
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start:i + 1]
                    try:
                        data = json.loads(candidate)
                        return _validate_forecast(data)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        break
    
    logger.warning(f"Failed to parse LLM response as JSON.\n  Raw ({len(raw_text)} chars): {raw_text[:300]}")
    return None


def _validate_forecast(data: dict) -> Optional[dict]:
    """Validate and normalise a parsed forecast dict."""
    if not isinstance(data, dict):
        return None
    # Must have a probability field to be a valid forecast
    if "probability" not in data:
        return None
    prob = float(data["probability"])
    data["probability"] = max(0.01, min(0.99, prob))
    return data


# ─── Calibration Layer ───────────────────────────────────

class CalibrationModel:
    """
    Adjusts raw LLM probability estimates to correct for systematic biases.
    
    LLMs tend to be:
    - Overconfident (estimates too close to 0 or 1)
    - Anchored on round numbers (bunching at 0.5, 0.6, 0.7, etc.)
    - Better at some categories than others
    
    The calibration model learns these patterns from resolved predictions
    and applies corrections. It starts with sensible defaults and improves
    as you accumulate data.
    
    The math: We model the true probability as a function of the LLM estimate:
        p_true = sigmoid(slope * logit(p_llm) + intercept)
    
    This is a Platt scaling approach (logistic calibration), which is standard
    in ML for calibrating classifier outputs.
    """
    
    def __init__(self, data_dir: str = "data", provider: str = ""):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        
        # Select defaults based on provider
        defaults = PROVIDER_CALIBRATION_DEFAULTS.get(
            provider, DEFAULT_CALIBRATION
        )
        
        # Platt scaling parameters
        self.slope = defaults["slope"]
        self.intercept = defaults["intercept"]
        self.extreme_pull = defaults["extreme_pull"]
        
        # Category-specific adjustments
        self.category_bias: dict[str, float] = {}
        
        # History of (llm_estimate, actual_outcome, category) for learning
        self.history: list[tuple[float, int, str]] = []
        
        self._load()
    
    def calibrate(self, raw_prob: float, category: str = "") -> float:
        """
        Apply calibration to a raw LLM probability estimate.
        
        Args:
            raw_prob: The LLM's raw estimate (0-1)
            category: Market category for category-specific adjustment
        
        Returns:
            Calibrated probability
        """
        # Clamp input
        raw_prob = max(0.01, min(0.99, raw_prob))
        
        # Step 1: Platt scaling in logit space
        logit = math.log(raw_prob / (1 - raw_prob))
        adjusted_logit = self.slope * logit + self.intercept
        calibrated = 1 / (1 + math.exp(-adjusted_logit))
        
        # Step 2: Pull extremes toward center (LLMs are overconfident at tails)
        if calibrated > 0.85:
            calibrated = calibrated - self.extreme_pull * (calibrated - 0.85)
        elif calibrated < 0.15:
            calibrated = calibrated + self.extreme_pull * (0.15 - calibrated)
        
        # Step 3: Category-specific bias correction
        if category and category in self.category_bias:
            calibrated += self.category_bias[category]
        
        return max(0.01, min(0.99, calibrated))
    
    def record_outcome(self, llm_estimate: float, actual: bool, category: str = ""):
        """Record a resolved prediction for future calibration learning."""
        self.history.append((llm_estimate, 1 if actual else 0, category))
        self._save()
        
        # Re-fit if we have enough data
        if len(self.history) >= 20 and len(self.history) % 10 == 0:
            self._fit()
    
    def _fit(self):
        """
        Re-fit calibration parameters from historical data.
        Uses simple logistic regression in logit space.
        """
        if len(self.history) < 20:
            return
        
        estimates = np.array([h[0] for h in self.history])
        outcomes = np.array([h[1] for h in self.history])
        
        # Compute logits of estimates
        estimates_clipped = np.clip(estimates, 0.01, 0.99)
        logits = np.log(estimates_clipped / (1 - estimates_clipped))
        
        # Simple linear regression: outcome ~ slope * logit + intercept
        # Using least squares in logit space (approximate Platt scaling)
        X = np.column_stack([logits, np.ones_like(logits)])
        
        # Regularized least squares (ridge)
        lambda_reg = 0.1
        XtX = X.T @ X + lambda_reg * np.eye(2)
        Xty = X.T @ outcomes
        
        try:
            params = np.linalg.solve(XtX, Xty)
            
            # Sanity check: slope should be positive and less than 2
            if 0.3 < params[0] < 2.0:
                self.slope = float(params[0])
                self.intercept = float(params[1])
                logger.info(
                    f"Calibration updated: slope={self.slope:.3f}, "
                    f"intercept={self.intercept:.3f} "
                    f"(from {len(self.history)} observations)"
                )
        except np.linalg.LinAlgError:
            pass
        
        # Category-specific bias
        categories = set(h[2] for h in self.history if h[2])
        for cat in categories:
            cat_data = [(e, o) for e, o, c in self.history if c == cat]
            if len(cat_data) >= 10:
                mean_est = np.mean([e for e, _ in cat_data])
                mean_out = np.mean([o for _, o in cat_data])
                self.category_bias[cat] = float(mean_out - mean_est)
        
        self._save()
    
    def diagnostics(self) -> str:
        """Print calibration diagnostics."""
        lines = [
            f"Calibration Model Diagnostics ({len(self.history)} observations)",
            f"  Platt scaling: slope={self.slope:.3f}, intercept={self.intercept:.3f}",
            f"  Extreme pull: {self.extreme_pull:.3f}",
        ]
        
        if self.category_bias:
            lines.append("  Category biases:")
            for cat, bias in sorted(self.category_bias.items()):
                lines.append(f"    {cat}: {bias:+.3f}")
        
        if self.history:
            estimates = [h[0] for h in self.history]
            outcomes = [h[1] for h in self.history]
            
            # Overall Brier score of raw estimates
            brier_raw = np.mean([(e - o) ** 2 for e, o in zip(estimates, outcomes)])
            
            # Brier score of calibrated estimates
            calibrated = [self.calibrate(e, c) for e, _, c in self.history]
            brier_cal = np.mean([(c - o) ** 2 for c, o in zip(calibrated, outcomes)])
            
            lines.extend([
                f"  Brier score (raw):        {brier_raw:.4f}",
                f"  Brier score (calibrated): {brier_cal:.4f}",
                f"  Improvement:              {(brier_raw - brier_cal) / brier_raw:.1%}",
            ])
        
        return "\n".join(lines)
    
    def _save(self):
        suffix = f"_{self.provider}" if self.provider else ""
        filepath = self.data_dir / f"calibration_model{suffix}.json"
        state = {
            "provider": self.provider,
            "slope": self.slope,
            "intercept": self.intercept,
            "extreme_pull": self.extreme_pull,
            "category_bias": self.category_bias,
            "history": self.history,
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
    
    def _load(self):
        suffix = f"_{self.provider}" if self.provider else ""
        filepath = self.data_dir / f"calibration_model{suffix}.json"
        if not filepath.exists():
            return
        try:
            with open(filepath) as f:
                state = json.load(f)
            self.slope = state.get("slope", self.slope)
            self.intercept = state.get("intercept", self.intercept)
            self.extreme_pull = state.get("extreme_pull", self.extreme_pull)
            self.category_bias = state.get("category_bias", {})
            self.history = [tuple(h) for h in state.get("history", [])]
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")


# ─── Context Builders ────────────────────────────────────

# Historical base rates for common market categories
# These are rough starting points — you should refine with actual data
BASE_RATE_DB = {
    # Economic
    "fed_rate_cut": {
        "base_rate": 0.35,
        "source": "Historical avg: Fed cuts ~4x per cycle, ~35% chance per meeting"
    },
    "fed_rate_hike": {
        "base_rate": 0.25,
        "source": "Historical avg: Fed hikes less frequently than holds"
    },
    "recession": {
        "base_rate": 0.15,
        "source": "Historical avg: US recessions occur ~15% of years since 1945"
    },
    "gdp_beat": {
        "base_rate": 0.52,
        "source": "GDP estimates are roughly unbiased, slight beat tendency"
    },
    "inflation_above": {
        "base_rate": 0.50,
        "source": "CPI surprises are roughly symmetric"
    },
    
    # Corporate
    "earnings_beat": {
        "base_rate": 0.68,
        "source": "S&P 500 companies beat estimates ~68% of the time (lowball guidance)"
    },
    "tech_earnings_beat": {
        "base_rate": 0.72,
        "source": "Large-cap tech beats at higher rate due to conservative guidance"
    },
    
    # Politics
    "incumbent_wins": {
        "base_rate": 0.55,
        "source": "Incumbency advantage: ~55% win rate in competitive elections"
    },
    "poll_leader_wins": {
        "base_rate": 0.70,
        "source": "Candidate leading in polls wins ~70% of the time (with caveats)"
    },
    "legislation_passes": {
        "base_rate": 0.30,
        "source": "Most proposed legislation fails; ~30% passage rate for serious bills"
    },
    
    # Sports (very rough)
    "favorite_wins": {
        "base_rate": 0.60,
        "source": "Betting favorites win ~60% of the time across major sports"
    },
    
    # General
    "default": {
        "base_rate": 0.50,
        "source": "No specific base rate available — using uninformative prior"
    },
}


def find_base_rate(question: str, category: str = "") -> tuple[float, str]:
    """
    Look up a historical base rate for a market question.
    
    Uses keyword matching against the base rate database.
    Returns (base_rate, source_description).
    """
    q = question.lower()
    
    # Check for specific patterns
    if any(w in q for w in ["fed", "federal reserve", "fomc"]):
        if any(w in q for w in ["cut", "lower", "reduce"]):
            entry = BASE_RATE_DB["fed_rate_cut"]
            return entry["base_rate"], entry["source"]
        if any(w in q for w in ["hike", "raise", "increase"]):
            entry = BASE_RATE_DB["fed_rate_hike"]
            return entry["base_rate"], entry["source"]
    
    if any(w in q for w in ["recession", "economic downturn"]):
        entry = BASE_RATE_DB["recession"]
        return entry["base_rate"], entry["source"]
    
    if any(w in q for w in ["gdp", "gross domestic"]):
        entry = BASE_RATE_DB["gdp_beat"]
        return entry["base_rate"], entry["source"]
    
    if any(w in q for w in ["inflation", "cpi", "pce"]):
        entry = BASE_RATE_DB["inflation_above"]
        return entry["base_rate"], entry["source"]
    
    if any(w in q for w in ["earnings", "revenue", "quarterly results"]):
        if any(w in q for w in ["nvidia", "apple", "google", "microsoft", "meta", "amazon", "tech"]):
            entry = BASE_RATE_DB["tech_earnings_beat"]
            return entry["base_rate"], entry["source"]
        entry = BASE_RATE_DB["earnings_beat"]
        return entry["base_rate"], entry["source"]
    
    if any(w in q for w in ["election", "vote", "win"]) and any(w in q for w in ["re-elect", "incumbent"]):
        entry = BASE_RATE_DB["incumbent_wins"]
        return entry["base_rate"], entry["source"]
    
    if any(w in q for w in ["bill", "legislation", "act", "pass"]):
        entry = BASE_RATE_DB["legislation_passes"]
        return entry["base_rate"], entry["source"]
    
    entry = BASE_RATE_DB["default"]
    return entry["base_rate"], entry["source"]


def compute_price_dynamics(price_history: list) -> dict:
    """
    Analyze price history to extract trend, volatility, and momentum signals.
    
    Args:
        price_history: List of (timestamp, price) tuples or price dicts
    
    Returns:
        Dict with trend, volatility, and recent prices
    """
    if not price_history or len(price_history) < 3:
        return {}
    
    # Extract prices
    prices = []
    for h in price_history:
        if isinstance(h, (list, tuple)):
            prices.append(float(h[1]))
        elif isinstance(h, dict):
            prices.append(float(h.get("p", h.get("price", 0.5))))
        else:
            prices.append(float(h))
    
    if len(prices) < 3:
        return {}
    
    prices = np.array(prices)
    n = len(prices)
    
    # Trend: compare first third vs last third
    first_third = np.mean(prices[:n//3])
    last_third = np.mean(prices[-n//3:])
    change = last_third - first_third
    
    if change > 0.03:
        trend = "rising"
    elif change < -0.03:
        trend = "falling"
    else:
        trend = "stable"
    
    # Volatility: standard deviation of returns
    returns = np.diff(prices) / prices[:-1]
    volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
    
    return {
        "trend": trend,
        "volatility": volatility,
        "price_24h_ago": float(prices[-min(24, n)]) if n > 1 else None,
        "price_7d_ago": float(prices[0]) if n > 7 else None,
        "current": float(prices[-1]),
    }


def build_market_context(
    market_data: dict,
    order_book_data: Optional[dict] = None,
    price_history: Optional[list] = None,
) -> MarketContext:
    """
    Build a MarketContext from raw market data, order book, and price history.
    
    This is the bridge between the API client's raw data and the LLM prompt.
    """
    question = market_data.get("question", "")
    category = market_data.get("category", "")
    
    # Find base rate
    base_rate, base_source = find_base_rate(question, category)
    
    # Price dynamics
    dynamics = compute_price_dynamics(price_history or [])
    
    # Order book
    bid_depth = 0.0
    ask_depth = 0.0
    spread = 0.0
    book_imbalance = 0.0
    
    if order_book_data:
        bid_depth = order_book_data.get("bid_depth", 0)
        ask_depth = order_book_data.get("ask_depth", 0)
        spread = order_book_data.get("spread", 0)
        total = bid_depth + ask_depth
        if total > 0:
            book_imbalance = (bid_depth - ask_depth) / total
    
    return MarketContext(
        question=question,
        description=market_data.get("description", ""),
        category=category,
        current_price=float(market_data.get("market_price", market_data.get("yes_price", 0.5))),
        volume_24h=float(market_data.get("volume_24h", 0)),
        total_volume=float(market_data.get("volume_total", market_data.get("volume", 0))),
        liquidity=float(market_data.get("liquidity", 0)),
        days_to_resolution=market_data.get("days_to_resolution"),
        price_7d_ago=dynamics.get("price_7d_ago"),
        price_24h_ago=dynamics.get("price_24h_ago"),
        price_trend=dynamics.get("trend", ""),
        price_volatility=dynamics.get("volatility", 0),
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        spread=spread,
        book_imbalance=book_imbalance,
        base_rate=base_rate,
        base_rate_source=base_source,
        tags=market_data.get("tags", []),
        related_markets=market_data.get("related_markets", []),
    )


# ─── Main Estimator ─────────────────────────────────────

class LLMEstimator:
    """
    Full LLM-powered probability estimator with calibration.
    
    Usage:
        estimator = LLMEstimator()
        
        # For ensemble integration:
        prob, confidence = estimator.estimate_for_ensemble(context_dict)
        
        # For standalone use with full details:
        forecast = estimator.forecast(market_context)
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        calibration: CalibrationModel = None,
        api_key: Optional[str] = None,
        n_samples: int = 1,
        temperature: float = 0.2,
        provider: str = PROVIDER_CLAUDE,
    ):
        """
        Args:
            model: Model/deployment name (auto-set per provider if left default)
            calibration: CalibrationModel instance (creates default if None)
            api_key: API key (or set env var per provider)
            n_samples: Number of LLM calls to average (more = better calibration, slower)
            temperature: LLM temperature (lower = more deterministic)
            provider: 'claude' or 'gpt'
        """
        self.provider = validate_provider(provider)
        self.model = model
        self.calibration = calibration or CalibrationModel(provider=self.provider)
        self.api_key = api_key
        self.n_samples = n_samples
        self.temperature = temperature
    
    @property
    def model_tag(self) -> str:
        """Short label for paper-trade tagging (e.g. 'claude-sonnet-4', 'gpt-5.4')."""
        return model_tag_for_provider(self.provider)
    
    def forecast(self, context: MarketContext) -> Optional[LLMForecast]:
        """
        Get a full forecast for a market.
        
        Returns LLMForecast with reasoning, or None if the API call fails.
        """
        prompt = build_forecast_prompt(context)
        
        # Optionally run multiple samples and average
        samples = []
        all_reasoning = []
        
        for i in range(self.n_samples):
            # Vary temperature slightly across samples for diversity
            temp = self.temperature + (i * 0.1) if self.n_samples > 1 else self.temperature
            
            raw = call_llm(
                user_prompt=prompt,
                system_prompt=FORECASTER_SYSTEM_PROMPT,
                provider=self.provider,
                model=self.model,
                temperature=min(temp, 1.0),
                api_key=self.api_key,
            )
            
            parsed = parse_llm_response(raw)
            if parsed:
                samples.append(parsed)
                if parsed.get("reasoning"):
                    all_reasoning.append(parsed["reasoning"])
        
        if not samples:
            logger.warning("All LLM calls failed — no forecast generated")
            return None
        
        # Average across samples
        raw_prob = np.mean([s["probability"] for s in samples])
        
        # Apply calibration
        calibrated = self.calibration.calibrate(raw_prob, context.category)
        
        # Use the first sample's detailed reasoning
        primary = samples[0]
        
        # Confidence scoring
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.85}
        base_confidence = confidence_map.get(primary.get("confidence", "medium"), 0.5)
        
        # Adjust confidence based on:
        # - Sample agreement (if multiple samples)
        if len(samples) > 1:
            sample_std = np.std([s["probability"] for s in samples])
            agreement_bonus = max(0, 0.15 - sample_std)  # Bonus for agreement
            base_confidence = min(0.95, base_confidence + agreement_bonus)
        
        # - Market volume (higher volume = more efficient = lower confidence in our edge)
        if context.volume_24h > 200_000:
            base_confidence *= 0.85  # Discount confidence in very liquid markets
        
        # - How far our estimate is from market (larger deviation = lower confidence)
        deviation = abs(calibrated - context.current_price)
        if deviation > 0.15:
            base_confidence *= 0.8  # Large deviation — we might be wrong
        
        return LLMForecast(
            raw_probability=float(raw_prob),
            calibrated_probability=float(calibrated),
            reasoning=primary.get("reasoning", ""),
            key_factors_for=primary.get("factors_for", []),
            key_factors_against=primary.get("factors_against", []),
            confidence_level=primary.get("confidence", "medium"),
            confidence_score=float(base_confidence),
            base_rate_anchor=primary.get("base_rate_anchor"),
            information_edge=primary.get("information_edge", "none"),
            model_used=self.model_tag,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    def estimate_for_ensemble(self, context_dict: dict) -> tuple[float, float]:
        """
        Ensemble-compatible interface: takes a context dict, returns (prob, confidence).
        
        This is the function you register with EnsembleEstimator:
            ensemble.add_estimator("llm", estimator.estimate_for_ensemble, weight=0.40)
        """
        # Build MarketContext from the flat dict
        market_context = build_market_context(
            market_data=context_dict,
            order_book_data={
                "bid_depth": context_dict.get("bid_depth", 0),
                "ask_depth": context_dict.get("ask_depth", 0),
                "spread": context_dict.get("spread", 0),
            },
            price_history=context_dict.get("price_history"),
        )
        
        forecast = self.forecast(market_context)
        
        if forecast is None:
            # Fallback: return market price with low confidence
            return context_dict.get("market_price", 0.5), 0.1
        
        return forecast.calibrated_probability, forecast.confidence_score
    
    def record_outcome(self, raw_estimate: float, actual: bool, category: str = ""):
        """Record a resolved prediction for calibration learning."""
        self.calibration.record_outcome(raw_estimate, actual, category)


# ─── Offline / Simulated Mode ────────────────────────────

class SimulatedLLMEstimator:
    """
    Simulated LLM estimator for testing without API calls.
    
    Generates realistic-looking estimates by:
    1. Starting from the base rate
    2. Incorporating market price with some noise
    3. Adding signal from order book and momentum
    4. Applying random estimation error
    
    This is useful for backtesting the full pipeline before spending API credits.
    """
    
    def __init__(
        self,
        skill_level: float = 0.3,
        calibration: CalibrationModel = None,
    ):
        """
        Args:
            skill_level: 0-1, how much "true signal" the simulated LLM captures.
                        0.0 = random noise, 1.0 = perfect oracle
        """
        self.skill_level = skill_level
        self.calibration = calibration or CalibrationModel()
    
    def estimate_for_ensemble(self, context_dict: dict) -> tuple[float, float]:
        """Ensemble-compatible simulated estimate."""
        market_price = context_dict.get("market_price", 0.5)
        
        # Find base rate
        base_rate, _ = find_base_rate(
            context_dict.get("question", ""),
            context_dict.get("category", "")
        )
        
        # Simulated LLM: weighted mix of base rate + market price + noise
        signal = 0.0
        
        # Book imbalance signal
        bid = context_dict.get("bid_depth", 0)
        ask = context_dict.get("ask_depth", 0)
        if bid + ask > 0:
            imb = (bid - ask) / (bid + ask)
            signal += imb * 0.03
        
        # Momentum signal
        history = context_dict.get("price_history", [])
        if len(history) >= 5:
            prices = [h[1] if isinstance(h, (list, tuple)) else h for h in history]
            momentum = np.mean(prices[-3:]) - np.mean(prices[:3])
            signal += momentum * 0.2
        
        # Combine
        base = 0.5 * base_rate + 0.5 * market_price
        noise = np.random.normal(0, 0.08 * (1 - self.skill_level))
        estimate = base + signal * self.skill_level + noise
        estimate = np.clip(estimate, 0.02, 0.98)
        
        # Calibrate
        calibrated = self.calibration.calibrate(float(estimate))
        
        # Confidence
        confidence = 0.3 + self.skill_level * 0.4 + np.random.uniform(-0.1, 0.1)
        confidence = np.clip(confidence, 0.1, 0.8)
        
        return float(calibrated), float(confidence)
    
    def record_outcome(self, raw_estimate: float, actual: bool, category: str = ""):
        self.calibration.record_outcome(raw_estimate, actual, category)


# ─── Demo ────────────────────────────────────────────────

def demo():
    """Demonstrate the LLM estimator (simulated mode)."""
    print("\n" + "=" * 70)
    print("  LLM PROBABILITY ESTIMATOR — Demo (Simulated Mode)")
    print("=" * 70)
    
    # 1. Show base rate lookup
    print("\n📚 Base Rate Lookups:")
    test_questions = [
        "Will the Fed cut rates in June 2026?",
        "Will NVIDIA beat Q2 2026 earnings?",
        "Will there be a US recession in 2026?",
        "Will the infrastructure bill pass the Senate?",
        "Who will win the UFC championship?",
    ]
    for q in test_questions:
        rate, source = find_base_rate(q)
        print(f"  {rate:.0%} | {q}")
        print(f"       {source}")
    
    # 2. Show calibration
    print("\n\n🔧 Calibration Layer:")
    cal = CalibrationModel(data_dir="data/demo_cal")
    
    test_probs = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
    print(f"  {'Raw LLM':>10} → {'Calibrated':>10}  (Δ)")
    for p in test_probs:
        c = cal.calibrate(p)
        print(f"  {p:>9.0%} → {c:>9.1%}  ({c - p:+.1%})")
    
    # 3. Show simulated forecasting
    print("\n\n🤖 Simulated Forecasts:")
    sim = SimulatedLLMEstimator(skill_level=0.35)
    
    markets = [
        {
            "question": "Will the Fed cut rates in June 2026?",
            "market_price": 0.42,
            "category": "economics",
            "volume_24h": 85000,
            "bid_depth": 18000,
            "ask_depth": 9000,
            "price_history": [(i, 0.38 + i * 0.004) for i in range(20)],
        },
        {
            "question": "Will NVIDIA beat Q2 2026 earnings?",
            "market_price": 0.73,
            "category": "corporate",
            "volume_24h": 65000,
            "bid_depth": 20000,
            "ask_depth": 8000,
            "price_history": [(i, 0.70 + i * 0.003) for i in range(20)],
        },
        {
            "question": "Will Bitcoin exceed $150K in 2026?",
            "market_price": 0.28,
            "category": "crypto",
            "volume_24h": 120000,
            "bid_depth": 12000,
            "ask_depth": 14000,
            "price_history": [(i, 0.30 - i * 0.001) for i in range(20)],
        },
    ]
    
    np.random.seed(42)
    print(f"\n  {'Market':<45} {'Market':>7} {'Est':>7} {'Edge':>7} {'Conf':>6}")
    print("  " + "-" * 75)
    
    for m in markets:
        prob, conf = sim.estimate_for_ensemble(m)
        edge = prob - m["market_price"]
        print(
            f"  {m['question']:<45} "
            f"{m['market_price']:>6.0%} "
            f"{prob:>6.1%} "
            f"{edge:>+6.1%} "
            f"{conf:>5.0%}"
        )
    
    # 4. Show calibration learning over time
    print("\n\n📈 Calibration Learning (simulating 100 resolved markets):")
    cal2 = CalibrationModel(data_dir="data/demo_cal2")
    
    np.random.seed(123)
    for i in range(100):
        # Simulate: LLM estimate is overconfident by ~15%
        true_prob = np.random.beta(2, 2)
        llm_est = true_prob + (true_prob - 0.5) * 0.30 + np.random.normal(0, 0.05)
        llm_est = np.clip(llm_est, 0.02, 0.98)
        
        actual = np.random.random() < true_prob
        cal2.record_outcome(float(llm_est), bool(actual))
    
    print(f"\n  {cal2.diagnostics()}")
    
    # 5. Integration example
    print("\n\n🔌 Integration with Ensemble:")
    print("""
  from core.probability import EnsembleEstimator, base_rate_estimator
  from core.llm_estimator import LLMEstimator  # or SimulatedLLMEstimator
  
  # With real API:
  llm = LLMEstimator(api_key="sk-ant-...")
  
  # Or simulated for testing:
  llm = SimulatedLLMEstimator(skill_level=0.35)
  
  ensemble = EnsembleEstimator()
  ensemble.add_estimator("base_rate", base_rate_estimator, weight=0.15)
  ensemble.add_estimator("llm", llm.estimate_for_ensemble, weight=0.45)
  ensemble.add_estimator("momentum", momentum_estimator, weight=0.20)
  ensemble.add_estimator("book_imbalance", book_imbalance_estimator, weight=0.20)
  
  # Now use in the edge detector:
  from strategies.edge_detector import EdgeDetector
  detector = EdgeDetector(ensemble=ensemble)
  signals = detector.run()
    """)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    demo()