"""
Weather Module Configuration
Station registry, model selection, and weather-specific trading parameters.
"""

from dataclasses import dataclass


# ─── Station Registry ───────────────────────────────────
# Each city Polymarket runs temperature markets for, mapped to the
# exact weather station used for resolution. Coordinates point to
# the airport station, not the city center.

@dataclass(frozen=True)
class StationInfo:
    """Resolution station metadata for a Polymarket temperature market."""
    city: str
    lat: float
    lon: float
    icao: str
    unit: str               # "F" or "C"
    bucket_size: int         # 2 for °F markets, 1 for °C markets
    wunderground_path: str   # Path for resolution URL
    timezone: str            # IANA timezone for the station


STATIONS: dict[str, StationInfo] = {
    "NYC": StationInfo(
        city="NYC", lat=40.7731, lon=-73.8803,
        icao="KLGA", unit="F", bucket_size=2,
        wunderground_path="us/ny/new-york-city/KLGA",
        timezone="America/New_York",
    ),
    "Houston": StationInfo(
        city="Houston", lat=29.6454, lon=-95.2789,
        icao="KHOU", unit="F", bucket_size=2,
        wunderground_path="us/tx/houston/KHOU",
        timezone="America/Chicago",
    ),
    "Atlanta": StationInfo(
        city="Atlanta", lat=33.6407, lon=-84.4277,
        icao="KATL", unit="F", bucket_size=2,
        wunderground_path="us/ga/atlanta/KATL",
        timezone="America/New_York",
    ),
    "Denver": StationInfo(
        city="Denver", lat=39.8561, lon=-104.6737,
        icao="KDEN", unit="F", bucket_size=2,
        wunderground_path="us/co/denver/KDEN",
        timezone="America/Denver",
    ),
    "London": StationInfo(
        city="London", lat=51.4700, lon=-0.4543,
        icao="EGLL", unit="C", bucket_size=1,
        wunderground_path="gb/london/EGLL",
        timezone="Europe/London",
    ),
    "Munich": StationInfo(
        city="Munich", lat=48.3537, lon=11.7750,
        icao="EDDM", unit="C", bucket_size=1,
        wunderground_path="de/munich/EDDM",
        timezone="Europe/Berlin",
    ),
    "Seoul": StationInfo(
        city="Seoul", lat=37.4602, lon=126.4407,
        icao="RKSI", unit="C", bucket_size=1,
        wunderground_path="kr/seoul/RKSI",
        timezone="Asia/Seoul",
    ),
    "Beijing": StationInfo(
        city="Beijing", lat=40.0799, lon=116.5844,
        icao="ZBAA", unit="C", bucket_size=1,
        wunderground_path="cn/beijing/ZBAA",
        timezone="Asia/Shanghai",
    ),
    "Wellington": StationInfo(
        city="Wellington", lat=-41.3272, lon=174.8050,
        icao="NZWN", unit="C", bucket_size=1,
        wunderground_path="nz/wellington/NZWN",
        timezone="Pacific/Auckland",
    ),
    "Hong Kong": StationInfo(
        city="Hong Kong", lat=22.3080, lon=113.9185,
        icao="VHHH", unit="C", bucket_size=1,
        wunderground_path="cn/hong-kong/VHHH",
        timezone="Asia/Hong_Kong",
    ),
}


# ─── Weather Models ─────────────────────────────────────
# Models to query from Open-Meteo, split by deterministic vs ensemble.

DETERMINISTIC_MODELS = [
    "ecmwf_ifs04",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless",
    "jma_seamless",
]

ENSEMBLE_MODELS = [
    "ecmwf_ifs025",
    "gfs025",
]

# US-only high-resolution model (used when available)
HRRR_MODEL = "ncep_hrrr_conus"


# ─── Probability Computation ────────────────────────────
# Gaussian kernel σ for smoothing ensemble members into bucket probabilities.
# Accounts for sub-grid-scale error and station-specific offset.

SMOOTHING_SIGMA_F = 1.5     # σ in °F for US markets (2°F buckets)
SMOOTHING_SIGMA_C = 0.8     # σ in °C for international markets (1°C buckets)


# ─── Confidence Tiers ────────────────────────────────────

CONFIDENCE_TIERS = {
    "LOCK":      {"min_agreement": 1.00, "min_models": 5},
    "STRONG":    {"min_agreement": 0.90, "min_models": 4},
    "SAFE":      {"min_agreement": 0.80, "min_models": 3},
    "NEAR_SAFE": {"min_agreement": 0.60, "min_models": 3},
}


# ─── Polymarket Tag IDs ──────────────────────────────────
# Discovered from actual temperature market metadata.
# The /tags endpoint doesn't reliably list these, so we hardcode them.

POLYMARKET_TAG_TEMPERATURE = 103040  # "Daily Temperature" (slug: temperature)
POLYMARKET_TAG_WEATHER = 84          # "Weather" (slug: weather)


# ─── Trading Parameters ─────────────────────────────────
# Weather-specific overrides for the main pipeline.

WEATHER_MIN_EDGE = 0.08            # Higher than the 0.05 default for general markets
WEATHER_KELLY_FRACTION = 0.20      # Slightly more conservative than the 0.25 default
WEATHER_MAX_BUCKET_POSITION = 50   # Max $ per individual bucket bet
WEATHER_MAX_CITY_EXPOSURE = 150    # Max $ total across all buckets for one city/date
WEATHER_MAX_PORTFOLIO_PCT = 0.30   # Max % of bankroll in weather positions total


# ─── Backtest Parameters ────────────────────────────────

BACKTEST_LOOKBACK_DAYS = 90        # How far back to scan for resolved markets
BACKTEST_LEAD_HOURS = [72, 48, 24] # Snapshot lead times before resolution
                                    # 6h/12h removed: market has converged by then,
                                    # no edge and wastes API calls (confirmed by
                                    # NYC + London backtests showing negative delta
                                    # at short lead times)
BACKTEST_MIN_MODELS = 3            # Skip market-dates with fewer than this many models


# ─── Data Storage ────────────────────────────────────────

WEATHER_DATA_DIR = "data/weather"
BACKTEST_FILE = "data/weather/backtest_results.json"
SNAPSHOTS_FILE = "data/weather/live_snapshots.json"
BIAS_TABLE_FILE = "data/weather/station_bias.json"