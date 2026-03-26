"""
Station Bias Correction
Corrects systematic per-city, per-model forecast errors.

Weather models forecast for a grid cell, not the exact station Polymarket
resolves on. Each model has a consistent directional bias at each station
(e.g., JMA underforecasts Atlanta by 3.5°F). Correcting for this shifts
the probability distribution toward the right bucket.

The bias table is built from backtest results:
    bias = mean(forecast - actual) per city per model

To correct a forecast:
    corrected = raw_forecast - bias

Usage:
    # Build from backtest results
    python -m weather.bias --build

    # Show the current table
    python -m weather.bias --show

    # Build and show
    python -m weather.bias --build --show
"""

import json
import logging
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from weather.config import BACKTEST_FILE, BIAS_TABLE_FILE, WEATHER_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class StationBias:
    """Bias statistics for a single city/model pair."""
    mean: float             # Mean bias (forecast - actual)
    std: float              # Standard deviation of bias
    n_samples: int          # Number of observations
    
    @property
    def is_reliable(self) -> bool:
        """At least 5 samples for a usable correction."""
        return self.n_samples >= 5


class BiasTable:
    """
    Per-city, per-model forecast bias corrections.
    
    Structure: {city: {model: StationBias}}
    
    Built from backtest results where each data point has:
        station_bias = {model: forecast - actual}
    
    Usage:
        table = BiasTable.from_backtest_results("data/weather/backtest_results.json")
        table.save("data/weather/station_bias.json")
        
        # Later:
        table = BiasTable.load("data/weather/station_bias.json")
        corrected = table.correct_forecasts("NYC", {"gfs_seamless": 62.0, "jma_seamless": 58.5})
        # → {"gfs_seamless": 62.0, "jma_seamless": 62.0}  (JMA had -3.5° bias)
    """
    
    def __init__(self):
        self._data: dict[str, dict[str, StationBias]] = {}
    
    # ─── Building ────────────────────────────────────────
    
    @classmethod
    def from_backtest_results(cls, path: str = BACKTEST_FILE) -> "BiasTable":
        """
        Build a bias table from saved backtest results.
        
        Aggregates all station_bias observations per city/model pair,
        computing mean and std.
        """
        table = cls()
        
        if not os.path.exists(path):
            logger.warning(f"Backtest results not found: {path}")
            return table
        
        with open(path) as f:
            results = json.load(f)
        
        # Collect all bias observations: {(city, model): [bias1, bias2, ...]}
        observations: dict[tuple[str, str], list[float]] = {}
        
        for r in results:
            city = r.get("city", "")
            bias_dict = r.get("station_bias", {})
            
            if not city or not bias_dict:
                continue
            
            for model, bias_val in bias_dict.items():
                if bias_val is not None:
                    key = (city, model)
                    if key not in observations:
                        observations[key] = []
                    observations[key].append(float(bias_val))
        
        # Compute statistics
        for (city, model), values in observations.items():
            if city not in table._data:
                table._data[city] = {}
            
            table._data[city][model] = StationBias(
                mean=round(float(np.mean(values)), 2),
                std=round(float(np.std(values)), 2),
                n_samples=len(values),
            )
        
        n_cities = len(table._data)
        n_pairs = sum(len(models) for models in table._data.values())
        logger.info(f"Built bias table: {n_pairs} city/model pairs across {n_cities} cities")
        
        return table
    
    # ─── Persistence ─────────────────────────────────────
    
    def save(self, path: str = BIAS_TABLE_FILE):
        """Save the bias table to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        serializable = {}
        for city, models in self._data.items():
            serializable[city] = {
                model: {
                    "mean": bias.mean,
                    "std": bias.std,
                    "n_samples": bias.n_samples,
                }
                for model, bias in models.items()
            }
        
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Saved bias table to {path}")
    
    @classmethod
    def load(cls, path: str = BIAS_TABLE_FILE) -> "BiasTable":
        """Load a previously saved bias table."""
        table = cls()
        
        if not os.path.exists(path):
            logger.info(f"No bias table at {path} — running without correction")
            return table
        
        with open(path) as f:
            raw = json.load(f)
        
        for city, models in raw.items():
            table._data[city] = {}
            for model, stats in models.items():
                table._data[city][model] = StationBias(
                    mean=stats["mean"],
                    std=stats["std"],
                    n_samples=stats["n_samples"],
                )
        
        n_pairs = sum(len(m) for m in table._data.values())
        logger.info(f"Loaded bias table: {n_pairs} corrections across {len(table._data)} cities")
        
        return table
    
    # ─── Correction ──────────────────────────────────────
    
    def correct_forecasts(
        self,
        city: str,
        point_forecasts: dict[str, float],
        min_samples: int = 5,
    ) -> dict[str, float]:
        """
        Apply bias correction to point forecasts for a given city.
        
        corrected = raw_forecast - mean_bias
        
        Only applies correction when we have enough samples (≥ min_samples).
        Models without reliable bias data are returned uncorrected.
        
        Args:
            city: City name (must match the backtest city names)
            point_forecasts: {model_name: raw_temperature}
            min_samples: Minimum observations required to apply correction
        
        Returns:
            {model_name: corrected_temperature}
        """
        city_biases = self._data.get(city, {})
        
        if not city_biases:
            return dict(point_forecasts)  # No corrections available
        
        corrected = {}
        for model, raw_temp in point_forecasts.items():
            bias = city_biases.get(model)
            
            if bias and bias.n_samples >= min_samples:
                corrected[model] = round(raw_temp - bias.mean, 1)
            else:
                corrected[model] = raw_temp
        
        return corrected
    
    def correct_ensemble_members(
        self,
        city: str,
        members: list[float],
        model: str,
        min_samples: int = 5,
    ) -> list[float]:
        """
        Apply bias correction to ensemble member forecasts.
        
        All members from the same model share the same bias.
        """
        city_biases = self._data.get(city, {})
        bias = city_biases.get(model)
        
        if bias and bias.n_samples >= min_samples:
            return [round(m - bias.mean, 1) for m in members]
        
        return list(members)
    
    # ─── Queries ─────────────────────────────────────────
    
    def get_bias(self, city: str, model: str) -> Optional[StationBias]:
        """Get bias stats for a specific city/model pair."""
        return self._data.get(city, {}).get(model)
    
    def get_city_biases(self, city: str) -> dict[str, StationBias]:
        """Get all model biases for a city."""
        return self._data.get(city, {})
    
    @property
    def cities(self) -> list[str]:
        return sorted(self._data.keys())
    
    @property
    def is_empty(self) -> bool:
        return len(self._data) == 0
    
    # ─── Display ─────────────────────────────────────────
    
    def print_table(self):
        """Print the bias table in a readable format."""
        if self.is_empty:
            print("Bias table is empty.")
            return
        
        # Collect all model names
        all_models = sorted(set(
            model for models in self._data.values() for model in models
        ))
        
        # Truncate model names for display
        short_names = {m: m.replace("_seamless", "").replace("_", " ")[:10] for m in all_models}
        
        print(f"\n{'=' * 70}")
        print(f"  STATION BIAS TABLE (forecast - actual)")
        print(f"{'=' * 70}")
        
        # Header
        header = f"  {'City':<12} |"
        for m in all_models:
            header += f" {short_names[m]:>10} |"
        print(header)
        print(f"  {'─' * (len(header) - 2)}")
        
        # Rows
        for city in sorted(self._data.keys()):
            row = f"  {city:<12} |"
            for model in all_models:
                bias = self._data[city].get(model)
                if bias:
                    reliable = "✓" if bias.is_reliable else "?"
                    row += f" {bias.mean:+6.1f}°({bias.n_samples:>2}){reliable}|"
                else:
                    row += f"       N/A  |"
            print(row)
        
        print(f"\n  ✓ = ≥5 samples (reliable), ? = <5 samples (unreliable)")
        print(f"  Positive = model overforecasts, Negative = model underforecasts")
        
        # Flag problematic models
        print(f"\n  Models with |bias| > 2.0°:")
        for city in sorted(self._data.keys()):
            for model, bias in self._data[city].items():
                if abs(bias.mean) > 2.0 and bias.is_reliable:
                    direction = "over" if bias.mean > 0 else "under"
                    print(f"    {city:<12} {short_names[model]:<10} {bias.mean:+.1f}° ({direction}forecasts)")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Station Bias Correction")
    parser.add_argument("--build", action="store_true", help="Build bias table from backtest results")
    parser.add_argument("--show", action="store_true", help="Show current bias table")
    parser.add_argument("--backtest-file", default=BACKTEST_FILE, help="Path to backtest results")
    parser.add_argument("--output", default=BIAS_TABLE_FILE, help="Path to save bias table")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    if args.build:
        table = BiasTable.from_backtest_results(args.backtest_file)
        table.save(args.output)
        table.print_table()
    elif args.show:
        table = BiasTable.load(args.output)
        table.print_table()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()