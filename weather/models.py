"""
Open-Meteo Weather Model Client
Fetches multi-model forecasts from Open-Meteo's free APIs.
Supports both live forecasts and historical archived predictions.

Open-Meteo provides access to 15+ weather models from national services
worldwide, all through a single unified API with no authentication required.

Key APIs (each uses a DIFFERENT subdomain):
    - api.open-meteo.com                    — Live forecasts
    - ensemble-api.open-meteo.com           — Live ensemble member forecasts
    - historical-forecast-api.open-meteo.com — Archived forecasts from 2022+
    - previous-runs-api.open-meteo.com      — Archived forecasts at specific lead offsets
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import requests

from weather.config import (
    DETERMINISTIC_MODELS, ENSEMBLE_MODELS, StationInfo,
)

logger = logging.getLogger(__name__)

# Each Open-Meteo product has its own subdomain
URL_FORECAST = "https://api.open-meteo.com"
URL_ENSEMBLE = "https://ensemble-api.open-meteo.com"
URL_HISTORICAL = "https://historical-forecast-api.open-meteo.com"
URL_PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com"


# ─── Data Classes ────────────────────────────────────────

@dataclass
class ModelForecast:
    """Deterministic forecast from a single model for a single date."""
    model: str
    target_date: date
    temperature_max: Optional[float]    # Daily high in the station's native unit
    temperature_max_c: Optional[float]  # Daily high in °C (raw from API)


@dataclass
class EnsembleForecast:
    """Ensemble member forecasts for a single date."""
    model: str              # e.g. "ecmwf_ifs025"
    target_date: date
    members: list[float]    # One temperature_max per ensemble member, in °C
    n_members: int
    mean: float
    std: float


@dataclass
class MultiModelForecast:
    """Combined forecasts from all models for a single station/date."""
    station: StationInfo
    target_date: date
    
    # Deterministic forecasts (model_name → daily high in station's native unit)
    point_forecasts: dict[str, float] = field(default_factory=dict)
    
    # Raw point forecasts in °C (before unit conversion)
    point_forecasts_c: dict[str, float] = field(default_factory=dict)
    
    # Ensemble member values (in station's native unit)
    ensemble_members: list[float] = field(default_factory=list)
    
    # Ensemble metadata
    ensemble_mean: float = 0.0
    ensemble_std: float = 0.0
    n_ensemble_members: int = 0
    
    # Data quality
    n_models_available: int = 0
    models_missing: list[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """At least 3 models returned data."""
        return self.n_models_available >= 3


# ─── API Client ──────────────────────────────────────────

class OpenMeteoClient:
    """
    Client for Open-Meteo weather forecast APIs.
    
    Handles both live forecasts and historical archived data using 
    the same interface. The Historical Forecast API uses the same
    schema as the live API, so callers don't need to distinguish.
    
    Usage:
        client = OpenMeteoClient()
        
        # Live forecast for tomorrow
        forecast = client.get_multi_model_forecast(station, tomorrow)
        
        # Historical: what did models predict on Jan 15 for Jan 20?
        forecast = client.get_historical_forecast(station, date(2026, 1, 20), lead_days=5)
    """
    
    def __init__(self, rate_limit_delay: float = 1.0, max_retries: int = 3):
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,       # 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketEdge-Weather/1.0",
        })
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0
    
    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _get(self, url: str, params: dict) -> Optional[dict]:
        """Make a throttled GET request with retry on SSL/connection errors."""
        for attempt in range(3):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=20)
                resp.raise_for_status()
                return resp.json()
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                if attempt < 2:
                    wait = (attempt + 1) * 2  # 2s, 4s
                    logger.debug(f"SSL/connection error (attempt {attempt+1}/3), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"Open-Meteo request failed after 3 attempts: {e}")
                    return None
            except requests.RequestException as e:
                logger.warning(f"Open-Meteo request failed: {e}")
                return None
    
    # ─── Live Forecasts ──────────────────────────────────
    
    def get_multi_model_forecast(
        self,
        station: StationInfo,
        target_date: date,
    ) -> MultiModelForecast:
        """
        Fetch current deterministic + ensemble forecasts for a station/date.
        
        Makes 2 API calls: one for deterministic models, one for ensembles.
        """
        result = MultiModelForecast(station=station, target_date=target_date)
        
        # 1. Deterministic models
        det = self._fetch_deterministic(
            lat=station.lat, lon=station.lon,
            target_date=target_date,
            models=DETERMINISTIC_MODELS,
            timezone=station.timezone,
        )
        if det:
            result.point_forecasts_c = det
            result.point_forecasts = self._convert_temps(det, station.unit)
            result.n_models_available = len(det)
        
        result.models_missing = [
            m for m in DETERMINISTIC_MODELS if m not in result.point_forecasts_c
        ]
        
        # 2. Ensemble models
        members = self._fetch_ensemble(
            lat=station.lat, lon=station.lon,
            target_date=target_date,
            models=ENSEMBLE_MODELS,
            timezone=station.timezone,
        )
        if members:
            members_native = self._convert_member_list(members, station.unit)
            result.ensemble_members = members_native
            result.n_ensemble_members = len(members_native)
            if members_native:
                import numpy as np
                result.ensemble_mean = float(np.mean(members_native))
                result.ensemble_std = float(np.std(members_native))
        
        return result
    
    # ─── Historical Forecasts ────────────────────────────
    
    def get_historical_forecast(
        self,
        station: StationInfo,
        target_date: date,
        lead_days: int = 1,
    ) -> MultiModelForecast:
        """
        Fetch what weather models predicted for a past date, at a given lead time.
        
        Uses the Previous Runs API for deterministic models. Ensemble member
        data is NOT available historically (those APIs don't archive it), so
        historical forecasts rely on point forecasts only.
        
        Args:
            station: Station to forecast for
            target_date: The date being forecast (e.g. when the market resolved)
            lead_days: How many days before target_date the forecast was issued.
                       1 = "day before" forecast, 2 = "two days before", etc.
        """
        result = MultiModelForecast(station=station, target_date=target_date)
        
        # Deterministic models (previous-runs API → historical-forecast-api fallback)
        det = self._fetch_historical_deterministic(
            lat=station.lat, lon=station.lon,
            target_date=target_date,
            lead_days=lead_days,
            models=DETERMINISTIC_MODELS,
            timezone=station.timezone,
        )
        if det:
            result.point_forecasts_c = det
            result.point_forecasts = self._convert_temps(det, station.unit)
            result.n_models_available = len(det)
        
        result.models_missing = [
            m for m in DETERMINISTIC_MODELS if m not in result.point_forecasts_c
        ]
        
        # NOTE: Ensemble member data is NOT archived by Open-Meteo's historical
        # APIs (previous-runs-api and historical-forecast-api both 404 on /v1/ensemble).
        # For the backtest, bucket probabilities are computed from point forecasts
        # using a wider Gaussian kernel to compensate for the lack of ensemble spread.
        # Live forecasts (get_multi_model_forecast) DO fetch ensemble data.
        
        return result
    
    def get_observed_temperature(
        self,
        station: StationInfo,
        target_date: date,
    ) -> Optional[float]:
        """
        Fetch the actual observed daily high temperature for a past date.
        
        Uses the historical forecast API which archives the best available
        data for past dates.
        
        Returns temperature in the station's native unit (°F or °C), or None.
        """
        params = {
            "latitude": station.lat,
            "longitude": station.lon,
            "daily": "temperature_2m_max",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "timezone": station.timezone,
        }
        
        data = self._get(f"{URL_HISTORICAL}/v1/forecast", params)
        if not data:
            return None
        
        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_max", [])
        if not temps or temps[0] is None:
            return None
        
        temp_c = temps[0]
        if station.unit == "F":
            return round(temp_c * 9 / 5 + 32, 1)
        return round(temp_c, 1)
    
    # ─── Internal: Deterministic Fetches ─────────────────
    
    def _fetch_deterministic(
        self, lat: float, lon: float, target_date: date,
        models: list[str], timezone: str,
    ) -> dict[str, float]:
        """Fetch deterministic daily-max forecasts for multiple models (live)."""
        forecast_days = (target_date - date.today()).days + 1
        if forecast_days < 1:
            forecast_days = 1
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "forecast_days": min(forecast_days, 16),
            "models": ",".join(models),
            "timezone": timezone,
        }
        
        data = self._get(f"{URL_FORECAST}/v1/forecast", params)
        return self._extract_daily_max(data, target_date, models)
    
    def _fetch_historical_deterministic(
        self, lat: float, lon: float, target_date: date,
        lead_days: int, models: list[str], timezone: str,
    ) -> dict[str, float]:
        """
        Fetch archived deterministic forecasts.
        
        Primary: previous-runs API (previous-runs-api.open-meteo.com)
            Uses previous_day=N to get what the model predicted N days before.
        
        Fallback: historical forecast API (historical-forecast-api.open-meteo.com)
            Returns the archived forecast for that date (not lead-time specific,
            but still useful — it's the first-hour splice of each model run).
        """
        # Primary: previous-runs API
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "models": ",".join(models),
            "timezone": timezone,
            "previous_day": lead_days,
        }
        
        data = self._get(f"{URL_PREVIOUS_RUNS}/v1/forecast", params)
        if data and "daily" in data:
            return self._extract_daily_max(data, target_date, models)
        
        # Fallback: historical forecast API (no past_days, just start/end date)
        logger.debug(f"Previous-runs failed, trying historical-forecast-api fallback")
        params_fallback = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "models": ",".join(models),
            "timezone": timezone,
        }
        
        data = self._get(f"{URL_HISTORICAL}/v1/forecast", params_fallback)
        return self._extract_daily_max(data, target_date, models)
    
    # ─── Internal: Ensemble Fetches ──────────────────────
    
    def _fetch_ensemble(
        self, lat: float, lon: float, target_date: date,
        models: list[str], timezone: str,
    ) -> list[float]:
        """Fetch ensemble member forecasts (live)."""
        forecast_days = (target_date - date.today()).days + 1
        if forecast_days < 1:
            forecast_days = 1
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "forecast_days": min(forecast_days, 16),
            "models": ",".join(models),
            "timezone": timezone,
        }
        
        data = self._get(f"{URL_ENSEMBLE}/v1/ensemble", params)
        return self._extract_ensemble_members(data, target_date, models)
    
    def _fetch_historical_ensemble(
        self, lat: float, lon: float, target_date: date,
        lead_days: int, models: list[str], timezone: str,
    ) -> list[float]:
        """
        Fetch archived ensemble member forecasts.
        
        Tries previous-runs API first, then historical-forecast-api.
        Ensemble archives may have sparser coverage than deterministic models.
        """
        # Primary: previous-runs API with ensemble endpoint
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "models": ",".join(models),
            "timezone": timezone,
            "previous_day": lead_days,
        }
        
        data = self._get(f"{URL_PREVIOUS_RUNS}/v1/ensemble", params)
        members = self._extract_ensemble_members(data, target_date, models) if data else []
        if members:
            return members
        
        # Fallback: historical-forecast-api (no lead-time control, but may have data)
        params_fallback = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "models": ",".join(models),
            "timezone": timezone,
        }
        
        data = self._get(f"{URL_HISTORICAL}/v1/ensemble", params_fallback)
        return self._extract_ensemble_members(data, target_date, models) if data else []
    
    # ─── Internal: Response Parsing ──────────────────────
    
    def _extract_daily_max(
        self, data: Optional[dict], target_date: date, models: list[str],
    ) -> dict[str, float]:
        """
        Extract per-model daily max temperature from an Open-Meteo response.
        
        The response structure varies:
        - Single model: {"daily": {"temperature_2m_max": [val1, val2, ...]}}
        - Multi model:  {"daily": {"temperature_2m_max_ecmwf_ifs04": [...], ...}}
        
        All values are in °C.
        """
        if not data or "daily" not in data:
            return {}
        
        daily = data["daily"]
        dates = daily.get("time", [])
        target_str = target_date.isoformat()
        
        # Find index for our target date
        date_idx = None
        for i, d in enumerate(dates):
            if d == target_str:
                date_idx = i
                break
        
        if date_idx is None and dates:
            # If target date not found, try the last available date
            date_idx = len(dates) - 1
        
        if date_idx is None:
            return {}
        
        result = {}
        
        # Try multi-model format first: temperature_2m_max_{model_name}
        for model in models:
            key = f"temperature_2m_max_{model}"
            if key in daily:
                vals = daily[key]
                if date_idx < len(vals) and vals[date_idx] is not None:
                    result[model] = float(vals[date_idx])
        
        # If no model-specific keys found, try single-model format
        if not result and "temperature_2m_max" in daily:
            vals = daily["temperature_2m_max"]
            if date_idx < len(vals) and vals[date_idx] is not None:
                # Attribute to the first model in the list
                result[models[0] if models else "unknown"] = float(vals[date_idx])
        
        return result
    
    def _extract_ensemble_members(
        self, data: Optional[dict], target_date: date, models: list[str],
    ) -> list[float]:
        """
        Extract all ensemble member values from an Open-Meteo ensemble response.
        
        Ensemble responses have keys like:
            temperature_2m_max_member01, temperature_2m_max_member02, ...
        or per-model:
            temperature_2m_max_ecmwf_ifs025_member01, ...
        """
        if not data or "daily" not in data:
            return []
        
        daily = data["daily"]
        dates = daily.get("time", [])
        target_str = target_date.isoformat()
        
        date_idx = None
        for i, d in enumerate(dates):
            if d == target_str:
                date_idx = i
                break
        
        if date_idx is None and dates:
            date_idx = len(dates) - 1
        
        if date_idx is None:
            return []
        
        members = []
        for key, vals in daily.items():
            if "member" in key and "temperature" in key:
                if isinstance(vals, list) and date_idx < len(vals) and vals[date_idx] is not None:
                    members.append(float(vals[date_idx]))
        
        return members
    
    # ─── Internal: Unit Conversion ───────────────────────
    
    def _convert_temps(self, temps_c: dict[str, float], unit: str) -> dict[str, float]:
        """Convert a dict of model→temp from °C to the station's unit."""
        if unit == "C":
            return {k: round(v, 1) for k, v in temps_c.items()}
        else:
            return {k: round(v * 9 / 5 + 32, 1) for k, v in temps_c.items()}
    
    def _convert_member_list(self, members_c: list[float], unit: str) -> list[float]:
        """Convert a list of temperatures from °C to the station's unit."""
        if unit == "C":
            return [round(v, 1) for v in members_c]
        else:
            return [round(v * 9 / 5 + 32, 1) for v in members_c]