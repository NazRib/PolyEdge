"""
Weather Trade Logger
Captures a full decision snapshot for every scanned weather event,
whether or not a trade was entered. This gives us a much larger sample
for evaluating model accuracy, bias correction quality, and edge
attribution than trade-level P&L alone.

Each scan appends one JSONL record per city/date event to:
    data/weather/event_log.jsonl

Records include:
    - Raw and bias-corrected per-model forecasts
    - Full model-implied bucket probability distribution
    - Full market bucket probability distribution
    - Edge per bucket and which buckets were traded
    - Confidence tier, lead hours, model agreement
    - (Post-resolution) actual temperature and actual bucket

Usage:
    logger = WeatherEventLogger()
    logger.log_event(event, trades_entered=[...])
    logger.log_resolution(event_key, actual_temp, actual_bucket)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from weather.config import WEATHER_DATA_DIR

logger = logging.getLogger(__name__)

LOG_FILE = Path(WEATHER_DATA_DIR) / "event_log.jsonl"


@dataclass
class WeatherEventLog:
    """Full decision snapshot for one city/date scan."""

    # ── Identity ──────────────────────────────────────────
    scan_timestamp: str                  # ISO UTC when the scan ran
    event_key: str                       # "{city}_{date}" dedup key
    city: str
    target_date: str                     # ISO date
    event_slug: str
    lead_hours: int

    # ── Raw model forecasts (pre-bias-correction) ─────────
    raw_forecasts: dict[str, float]      # model_name → temp
    n_models: int

    # ── Bias correction ───────────────────────────────────
    bias_corrected: bool
    bias_values: dict[str, float]        # model_name → bias applied
    corrected_forecasts: dict[str, float]  # model_name → corrected temp

    # ── Ensemble data ─────────────────────────────────────
    n_ensemble_members: int
    ensemble_mean: float
    ensemble_std: float

    # ── Model consensus ───────────────────────────────────
    model_agreement: float               # 0-1
    consensus_bucket: str
    confidence_tier: str
    computed_confidence: float            # final confidence score used for Kelly

    # ── Probability distributions ─────────────────────────
    model_bucket_probs: dict[str, float]  # bucket_label → model prob
    market_bucket_probs: dict[str, float] # bucket_label → market price

    # ── Edge analysis ─────────────────────────────────────
    edges: dict[str, float]              # bucket_label → (model_prob - market_price)
    max_edge: float
    n_tradeable_buckets: int             # how many exceeded min_edge

    # ── Trade decisions ───────────────────────────────────
    traded: bool                          # did we enter any trade on this event?
    trades: list[dict] = field(default_factory=list)
    # Each: {bucket, side, dollars, shares, model_prob, market_price, edge}

    skip_reason: str = ""                 # if not traded, why? (no_edge, capped, deduped, etc.)

    # ── Resolution (filled post-hoc) ──────────────────────
    resolved: bool = False
    actual_temperature: Optional[float] = None
    actual_bucket: Optional[str] = None
    model_error: Optional[float] = None   # ensemble_mean - actual
    trade_pnl: Optional[float] = None     # sum of P&L for trades on this event


class WeatherEventLogger:
    """Append-only JSONL logger for weather event snapshots."""

    def __init__(self, log_file: Path = None):
        self.log_file = log_file or LOG_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event,  # WeatherEvent from scanner
        confidence: float,
        trades_entered: list[dict] = None,
        skip_reason: str = "",
    ) -> WeatherEventLog:
        """
        Build and persist a log entry from a scanned WeatherEvent.

        Args:
            event: Enriched WeatherEvent (after _enrich_event)
            confidence: The computed confidence score
            trades_entered: List of trade dicts if any were entered
            skip_reason: Why no trade was entered (if applicable)
        """
        now = datetime.now(timezone.utc).isoformat()
        event_key = f"{event.city}_{event.target_date.isoformat()}"

        # Compute per-bucket edges
        market_probs = {b["label"]: b["market_price"] for b in event.buckets}
        edges = {}
        for label, model_p in event.model_probs.items():
            mkt_p = market_probs.get(label, 0.0)
            edges[label] = round(model_p - mkt_p, 6)

        max_edge = max(edges.values()) if edges else 0.0

        # Compute bias values (raw - corrected) per model
        bias_values = {}
        for model, raw_temp in event.point_forecasts.items():
            corrected_temp = event.point_forecasts_corrected.get(model, raw_temp)
            bias_values[model] = round(raw_temp - corrected_temp, 3)

        traded = bool(trades_entered)
        n_tradeable = len(event.tradeable_edges)

        entry = WeatherEventLog(
            scan_timestamp=now,
            event_key=event_key,
            city=event.city,
            target_date=event.target_date.isoformat(),
            event_slug=event.event_slug,
            lead_hours=event.lead_hours,
            raw_forecasts={k: round(v, 2) for k, v in event.point_forecasts.items()},
            n_models=event.n_models,
            bias_corrected=event.bias_corrected,
            bias_values=bias_values,
            corrected_forecasts={k: round(v, 2) for k, v in event.point_forecasts_corrected.items()},
            n_ensemble_members=event.n_ensemble,
            ensemble_mean=round(event.ensemble_mean, 2),
            ensemble_std=round(event.ensemble_std, 2),
            model_agreement=round(event.agreement, 3),
            consensus_bucket=event.consensus_bucket,
            confidence_tier=event.confidence_tier,
            computed_confidence=round(confidence, 3),
            model_bucket_probs={k: round(v, 6) for k, v in event.model_probs.items()},
            market_bucket_probs={k: round(v, 6) for k, v in market_probs.items()},
            edges=edges,
            max_edge=round(max_edge, 6),
            n_tradeable_buckets=n_tradeable,
            traded=traded,
            trades=trades_entered or [],
            skip_reason=skip_reason if not traded else "",
        )

        self._append(entry)
        return entry

    def log_resolution(
        self,
        event_key: str,
        actual_temperature: float,
        actual_bucket: str,
        trade_pnl: float = 0.0,
    ):
        """
        Update existing log entries for a resolved event.

        Reads the full log, patches matching entries, rewrites.
        This is called from check_resolutions.
        """
        if not self.log_file.exists():
            return

        lines = self.log_file.read_text().strip().split("\n")
        updated = []
        patched = 0

        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                updated.append(line)
                continue

            if record.get("event_key") == event_key and not record.get("resolved"):
                record["resolved"] = True
                record["actual_temperature"] = actual_temperature
                record["actual_bucket"] = actual_bucket
                record["trade_pnl"] = trade_pnl
                # Model error: how far was the ensemble/consensus from actual
                ens_mean = record.get("ensemble_mean", 0)
                if ens_mean:
                    record["model_error"] = round(ens_mean - actual_temperature, 2)
                else:
                    # Use mean of corrected forecasts
                    corrected = record.get("corrected_forecasts", {})
                    if corrected:
                        mean_fc = sum(corrected.values()) / len(corrected)
                        record["model_error"] = round(mean_fc - actual_temperature, 2)

                patched += 1

            updated.append(json.dumps(record, default=str))

        self.log_file.write_text("\n".join(updated) + "\n")
        if patched:
            logger.info(f"Patched {patched} log entries for {event_key}")

    def load_all(self) -> list[dict]:
        """Load all log entries as dicts."""
        if not self.log_file.exists():
            return []
        entries = []
        for line in self.log_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def _append(self, entry: WeatherEventLog):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), default=str) + "\n")
        logger.debug(f"Logged event: {entry.event_key} ({entry.lead_hours}h)")
