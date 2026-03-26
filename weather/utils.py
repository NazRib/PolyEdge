"""
Weather Utilities
Question parsing, bucket extraction, temperature conversion,
and probability computation from ensemble forecasts.
"""

import re
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
from scipy.stats import norm

from weather.config import STATIONS, StationInfo, SMOOTHING_SIGMA_F, SMOOTHING_SIGMA_C

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. QUESTION PARSING
# ═══════════════════════════════════════════════════════════

# Matches: "Highest temperature in {City} on {Month} {Day}?"
# Also handles: "Highest temperature in {City} on {Month} {Day}, {Year}?"
_QUESTION_RE = re.compile(
    r"Highest temperature in (.+?) on (\w+ \d{1,2})(?:,?\s*(\d{4}))?",
    re.IGNORECASE,
)

# Slug pattern: "highest-temperature-in-{city}-on-{month}-{day}-{year}"
_SLUG_RE = re.compile(
    r"highest-temperature-in-(.+?)-on-(\w+)-(\d{1,2})-(\d{4})"
)

# Bucket patterns:
#   °F: "82-83°F", "80°F or below", "90°F or above", "50°F or higher", "30°F or lower"
#   °C: "13°C", "7°C or below", "34°C or above", "34°C or higher", "0°C or lower"
_BUCKET_RANGE_F = re.compile(r"(\d+)-(\d+)°?F")
_BUCKET_BELOW_F = re.compile(r"(\d+)°?F or (?:below|lower)")
_BUCKET_ABOVE_F = re.compile(r"(\d+)°?F or (?:above|higher)")
_BUCKET_EXACT_C = re.compile(r"^(-?\d+)°?C$")
_BUCKET_BELOW_C = re.compile(r"(-?\d+)°?C or (?:below|lower)")
_BUCKET_ABOVE_C = re.compile(r"(-?\d+)°?C or (?:above|higher)")

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


@dataclass
class ParsedMarket:
    """Extracted metadata from a temperature market question/slug."""
    city: str
    station: Optional[StationInfo]
    target_date: date
    unit: str               # "F" or "C"
    bucket_size: int         # 2 for °F, 1 for °C


@dataclass
class Bucket:
    """A single temperature bucket with its boundaries."""
    label: str              # Original label, e.g. "82-83°F" or "13°C"
    low: float              # Inclusive lower bound (°F or °C)
    high: float             # Exclusive upper bound
    is_edge: bool           # True for "or below" / "or above" buckets


def parse_question(question: str, year: int = None) -> Optional[ParsedMarket]:
    """
    Extract city, date, and unit info from a Polymarket temperature question.
    
    Args:
        question: The market question, e.g. "Highest temperature in NYC on March 24?"
        year: Override year (defaults to current year)
    
    Returns:
        ParsedMarket or None if the question doesn't match the expected pattern.
    """
    m = _QUESTION_RE.search(question)
    if not m:
        return None
    
    city_raw = m.group(1).strip()
    date_str = m.group(2).strip()
    year_str = m.group(3)
    
    # Parse month and day
    parts = date_str.split()
    if len(parts) != 2:
        return None
    
    month_str, day_str = parts
    month = _MONTH_MAP.get(month_str.lower())
    if month is None:
        return None
    
    day = int(day_str)
    yr = int(year_str) if year_str else (year or datetime.now().year)
    
    try:
        target_date = date(yr, month, day)
    except ValueError:
        return None
    
    # Match city to station registry
    station = _match_city(city_raw)
    unit = station.unit if station else "F"
    bucket_size = station.bucket_size if station else 2
    
    return ParsedMarket(
        city=station.city if station else city_raw,
        station=station,
        target_date=target_date,
        unit=unit,
        bucket_size=bucket_size,
    )


def parse_slug(slug: str) -> Optional[ParsedMarket]:
    """
    Extract city, date from a Polymarket event slug.
    
    Args:
        slug: e.g. "highest-temperature-in-nyc-on-march-24-2026"
    """
    m = _SLUG_RE.search(slug)
    if not m:
        return None
    
    city_raw = m.group(1).replace("-", " ").strip()
    month_str = m.group(2)
    day = int(m.group(3))
    year = int(m.group(4))
    
    month = _MONTH_MAP.get(month_str.lower())
    if month is None:
        return None
    
    try:
        target_date = date(year, month, day)
    except ValueError:
        return None
    
    station = _match_city(city_raw)
    unit = station.unit if station else "F"
    bucket_size = station.bucket_size if station else 2
    
    return ParsedMarket(
        city=station.city if station else city_raw,
        station=station,
        target_date=target_date,
        unit=unit,
        bucket_size=bucket_size,
    )


def build_event_slug(city: str, target_date: date) -> str:
    """Construct the expected Polymarket event slug for a city/date."""
    month_name = target_date.strftime("%B").lower()  # e.g. "march"
    day = target_date.day
    year = target_date.year
    city_slug = city.lower().replace(" ", "-")
    return f"highest-temperature-in-{city_slug}-on-{month_name}-{day}-{year}"


def _match_city(city_raw: str) -> Optional[StationInfo]:
    """Fuzzy-match a city name from a question to our station registry."""
    city_lower = city_raw.lower().strip()
    
    # Direct match
    for key, station in STATIONS.items():
        if city_lower == key.lower() or city_lower == station.city.lower():
            return station
    
    # Substring match (handles "New York City" → "NYC", etc.)
    _ALIASES = {
        "new york": "NYC", "nyc": "NYC",
        "hong kong": "Hong Kong",
        "london": "London", "munich": "Munich",
        "seoul": "Seoul", "beijing": "Beijing",
        "wellington": "Wellington",
        "houston": "Houston", "atlanta": "Atlanta",
        "denver": "Denver",
    }
    for alias, key in _ALIASES.items():
        if alias in city_lower:
            return STATIONS.get(key)
    
    logger.debug(f"Unknown city in temperature market: '{city_raw}'")
    return None


# ═══════════════════════════════════════════════════════════
# 2. BUCKET PARSING
# ═══════════════════════════════════════════════════════════

def parse_bucket(label: str, unit: str = "F") -> Optional[Bucket]:
    """
    Parse a temperature bucket label into numeric boundaries.
    
    Examples:
        "82-83°F"       → Bucket(low=82, high=84, is_edge=False)
        "80°F or below" → Bucket(low=-inf, high=81, is_edge=True)
        "90°F or above" → Bucket(low=90, high=inf, is_edge=True)
        "13°C"          → Bucket(low=13, high=14, is_edge=False)
        "7°C or below"  → Bucket(low=-inf, high=8, is_edge=True)
    """
    label = label.strip()
    
    if unit == "F":
        # "82-83°F" range bucket
        m = _BUCKET_RANGE_F.search(label)
        if m:
            low = float(m.group(1))
            high = float(m.group(2)) + 1  # Exclusive upper bound
            return Bucket(label=label, low=low, high=high, is_edge=False)
        
        # "80°F or below"
        m = _BUCKET_BELOW_F.search(label)
        if m:
            return Bucket(label=label, low=-999, high=float(m.group(1)) + 1, is_edge=True)
        
        # "90°F or above"
        m = _BUCKET_ABOVE_F.search(label)
        if m:
            return Bucket(label=label, low=float(m.group(1)), high=999, is_edge=True)
    
    else:  # °C
        # "13°C" exact bucket
        m = _BUCKET_EXACT_C.search(label)
        if m:
            val = float(m.group(1))
            return Bucket(label=label, low=val, high=val + 1, is_edge=False)
        
        # "7°C or below"
        m = _BUCKET_BELOW_C.search(label)
        if m:
            return Bucket(label=label, low=-999, high=float(m.group(1)) + 1, is_edge=True)
        
        # "34°C or above"
        m = _BUCKET_ABOVE_C.search(label)
        if m:
            return Bucket(label=label, low=float(m.group(1)), high=999, is_edge=True)
    
    logger.debug(f"Could not parse bucket label: '{label}' (unit={unit})")
    return None


def parse_buckets_from_outcomes(outcomes: list[str], unit: str = "F") -> list[Bucket]:
    """Parse all outcome labels from a multi-outcome market into Buckets."""
    buckets = []
    for label in outcomes:
        b = parse_bucket(label, unit)
        if b:
            buckets.append(b)
        else:
            logger.warning(f"Skipping unparseable bucket: '{label}'")
    return buckets


# ═══════════════════════════════════════════════════════════
# 3. PROBABILITY COMPUTATION
# ═══════════════════════════════════════════════════════════

def compute_bucket_probabilities(
    ensemble_forecasts: list[float],
    buckets: list[Bucket],
    smoothing_sigma: float = None,
    unit: str = "F",
) -> dict[str, float]:
    """
    Convert ensemble member forecasts into bucket probabilities.
    
    Uses Gaussian kernel density estimation: for each ensemble member,
    place a Gaussian centered at that member's forecast and compute
    the probability mass falling in each bucket. Average across all members.
    
    This is more principled than raw binning because:
    - It handles the discrete nature of buckets smoothly
    - The σ parameter encodes forecast uncertainty beyond ensemble spread
    - Edge buckets ("or below" / "or above") are handled naturally
    
    Args:
        ensemble_forecasts: List of daily-max temperature forecasts from
                           ensemble members (51 for ECMWF, 31 for GFS, etc.)
        buckets: List of Bucket objects defining the market outcomes
        smoothing_sigma: Gaussian kernel width. Defaults to config value.
        unit: "F" or "C" (used for default sigma selection)
    
    Returns:
        Dict mapping bucket label → probability, summing to ~1.0
    """
    if not ensemble_forecasts or not buckets:
        return {}
    
    if smoothing_sigma is None:
        smoothing_sigma = SMOOTHING_SIGMA_F if unit == "F" else SMOOTHING_SIGMA_C
    
    forecasts = np.array(ensemble_forecasts, dtype=float)
    n_members = len(forecasts)
    
    probs = {}
    for bucket in buckets:
        # For each ensemble member, compute P(temperature falls in this bucket)
        # using a Gaussian centered at the member's forecast.
        #
        # P(low <= T < high) = Phi((high - forecast) / sigma) - Phi((low - forecast) / sigma)
        #
        # For edge buckets, low or high is effectively ±infinity.
        low = bucket.low if bucket.low > -900 else -1e6
        high = bucket.high if bucket.high < 900 else 1e6
        
        member_probs = (
            norm.cdf(high, loc=forecasts, scale=smoothing_sigma)
            - norm.cdf(low, loc=forecasts, scale=smoothing_sigma)
        )
        probs[bucket.label] = float(np.mean(member_probs))
    
    # Normalize to sum to 1.0 (should be very close already)
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    
    return probs


def compute_bucket_probs_from_point_forecasts(
    point_forecasts: dict[str, float],
    buckets: list[Bucket],
    model_sigma: float = None,
    unit: str = "F",
) -> dict[str, float]:
    """
    Fallback: compute bucket probabilities from deterministic point forecasts
    when ensemble data is unavailable.
    
    Treats each model's forecast as a Gaussian and averages the resulting
    distributions. Less principled than the ensemble method but still
    useful when ensemble data has gaps.
    
    Args:
        point_forecasts: Dict of model_name → forecast value
        buckets: List of Bucket objects
        model_sigma: Per-model uncertainty σ. Defaults to 2× the smoothing σ.
        unit: "F" or "C"
    """
    default_sigma = SMOOTHING_SIGMA_F if unit == "F" else SMOOTHING_SIGMA_C
    sigma = model_sigma or (default_sigma * 2.0)
    
    forecasts = list(point_forecasts.values())
    return compute_bucket_probabilities(forecasts, buckets, sigma, unit)


def model_agreement(
    point_forecasts: dict[str, float],
    buckets: list[Bucket],
) -> tuple[float, str]:
    """
    Compute what fraction of deterministic models agree on the same bucket.
    
    Returns:
        (agreement_fraction, consensus_bucket_label)
    """
    if not point_forecasts or not buckets:
        return 0.0, ""
    
    bucket_counts: dict[str, int] = {b.label: 0 for b in buckets}
    
    for model, temp in point_forecasts.items():
        for bucket in buckets:
            low = bucket.low if bucket.low > -900 else -1e6
            high = bucket.high if bucket.high < 900 else 1e6
            if low <= temp < high:
                bucket_counts[bucket.label] += 1
                break
    
    n_models = len(point_forecasts)
    if n_models == 0:
        return 0.0, ""
    
    top_bucket = max(bucket_counts, key=bucket_counts.get)
    agreement = bucket_counts[top_bucket] / n_models
    return agreement, top_bucket


def classify_confidence(
    agreement: float,
    n_models: int,
) -> str:
    """Map agreement level to a confidence tier."""
    from weather.config import CONFIDENCE_TIERS
    
    for tier, reqs in CONFIDENCE_TIERS.items():
        if agreement >= reqs["min_agreement"] and n_models >= reqs["min_models"]:
            return tier
    return "LOW"


# ═══════════════════════════════════════════════════════════
# 4. SCORING
# ═══════════════════════════════════════════════════════════

def brier_score(
    predicted_probs: dict[str, float],
    actual_bucket: str,
) -> float:
    """
    Multi-outcome Brier score for a single market.
    
    BS = (1/K) * Σ (p_k - o_k)²
    
    where o_k = 1 for the winning bucket and 0 for all others.
    Lower is better. 0 = perfect, 1/K = uniform prior baseline.
    
    Args:
        predicted_probs: Dict of bucket_label → predicted probability
        actual_bucket: The bucket that actually won
    
    Returns:
        Brier score (lower = better calibration)
    """
    if not predicted_probs:
        return 1.0
    
    k = len(predicted_probs)
    total = 0.0
    for label, prob in predicted_probs.items():
        outcome = 1.0 if label == actual_bucket else 0.0
        total += (prob - outcome) ** 2
    
    return total / k


def f_to_c(temp_f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (temp_f - 32) * 5 / 9


def c_to_f(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9 / 5 + 32