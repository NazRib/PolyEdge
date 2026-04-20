"""
Weather Diagnostics
Reads the event log (event_log.jsonl) and produces actionable diagnostics
for fine-tuning the weather prediction strategy.

Reports:
    1. Model accuracy — MAE per model per city, identifies best/worst models
    2. Bias correction impact — did corrections help or hurt?
    3. Edge attribution — where does our edge actually come from?
    4. Calibration by tier — are confidence tiers well-calibrated?
    5. City profitability — which cities are carrying the strategy?
    6. Lead-time analysis — edge quality vs forecast horizon

Usage:
    python -m weather.diagnostics                # Full report
    python -m weather.diagnostics --models       # Model accuracy only
    python -m weather.diagnostics --bias         # Bias correction analysis
    python -m weather.diagnostics --edge         # Edge attribution
    python -m weather.diagnostics --cities       # Per-city breakdown
    python -m weather.diagnostics --json         # Machine-readable output
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from weather.trade_logger import WeatherEventLogger, LOG_FILE

logger = logging.getLogger(__name__)


def load_log(log_file: Path = None) -> list[dict]:
    """Load event log entries."""
    loader = WeatherEventLogger(log_file=log_file)
    entries = loader.load_all()
    if not entries:
        print("  No event log data found. Run the scanner first to generate logs.")
        sys.exit(0)
    return entries


def _dedup_by_event_key(entries: list[dict]) -> list[dict]:
    """
    Deduplicate entries by event_key for P&L metrics.

    The same city/date gets logged multiple times at different lead hours
    (72h, 48h, 24h, 7h, 1h). When log_resolution patches entries, it sets
    trade_pnl on ALL entries with the same event_key — including late scans
    where traded=false because the market had already converged.

    Selection priority:
        1. Prefer entries where traded=True (actual trade decisions)
        2. Among same traded status, prefer shortest lead_hours
    """
    best = {}
    for e in entries:
        key = e.get("event_key", "")
        if not key:
            continue
        existing = best.get(key)
        if existing is None:
            best[key] = e
            continue

        # Prefer traded over non-traded
        e_traded = bool(e.get("traded"))
        ex_traded = bool(existing.get("traded"))
        if e_traded and not ex_traded:
            best[key] = e
        elif e_traded == ex_traded:
            # Same traded status — prefer shorter lead_hours
            if e.get("lead_hours", 999) < existing.get("lead_hours", 999):
                best[key] = e

    return list(best.values())


# ═══════════════════════════════════════════════════════════════
# 1. MODEL ACCURACY
# ═══════════════════════════════════════════════════════════════

def model_accuracy_report(entries: list[dict]) -> dict:
    """
    Per-model MAE and bias, broken down by city.

    Only uses resolved entries where we know the actual temperature.
    """
    resolved = [e for e in entries if e.get("resolved") and e.get("actual_temperature") is not None]
    if not resolved:
        return {"error": "No resolved entries with actual temperature"}

    # Accumulate errors: model → city → list of errors
    raw_errors = defaultdict(lambda: defaultdict(list))      # raw forecast
    corrected_errors = defaultdict(lambda: defaultdict(list))  # after bias correction

    for e in resolved:
        actual = e["actual_temperature"]
        city = e["city"]

        for model, raw_temp in e.get("raw_forecasts", {}).items():
            raw_errors[model][city].append(raw_temp - actual)

        for model, corr_temp in e.get("corrected_forecasts", {}).items():
            corrected_errors[model][city].append(corr_temp - actual)

    # Build report
    report = {"n_resolved": len(resolved), "models": {}}
    all_models = sorted(set(list(raw_errors.keys()) + list(corrected_errors.keys())))

    for model in all_models:
        model_report = {"cities": {}}
        all_raw = []
        all_corr = []

        for city in sorted(set(list(raw_errors[model].keys()) + list(corrected_errors[model].keys()))):
            raw = raw_errors[model].get(city, [])
            corr = corrected_errors[model].get(city, [])
            all_raw.extend(raw)
            all_corr.extend(corr)

            city_data = {}
            if raw:
                city_data["raw_mae"] = round(np.mean(np.abs(raw)), 2)
                city_data["raw_bias"] = round(np.mean(raw), 2)
                city_data["n"] = len(raw)
            if corr:
                city_data["corrected_mae"] = round(np.mean(np.abs(corr)), 2)
                city_data["corrected_bias"] = round(np.mean(corr), 2)
            if raw and corr:
                city_data["correction_helped"] = np.mean(np.abs(corr)) < np.mean(np.abs(raw))
            model_report["cities"][city] = city_data

        if all_raw:
            model_report["overall_raw_mae"] = round(np.mean(np.abs(all_raw)), 2)
            model_report["overall_raw_bias"] = round(np.mean(all_raw), 2)
        if all_corr:
            model_report["overall_corrected_mae"] = round(np.mean(np.abs(all_corr)), 2)
            model_report["overall_corrected_bias"] = round(np.mean(all_corr), 2)

        report["models"][model] = model_report

    return report


def print_model_report(report: dict):
    print(f"\n{'=' * 70}")
    print(f"  MODEL ACCURACY REPORT ({report['n_resolved']} resolved events)")
    print(f"{'=' * 70}")

    for model, data in sorted(report.get("models", {}).items()):
        raw_mae = data.get("overall_raw_mae", "?")
        corr_mae = data.get("overall_corrected_mae", "?")
        raw_bias = data.get("overall_raw_bias", "?")
        arrow = ""
        if isinstance(raw_mae, float) and isinstance(corr_mae, float):
            arrow = " ✅" if corr_mae < raw_mae else " ⚠️"

        print(f"\n  {model}")
        print(f"    Raw MAE: {raw_mae}° | Bias: {raw_bias:+}° | Corrected MAE: {corr_mae}°{arrow}")

        for city, cd in sorted(data.get("cities", {}).items()):
            n = cd.get("n", 0)
            rm = cd.get("raw_mae", "?")
            cm = cd.get("corrected_mae", "?")
            helped = cd.get("correction_helped")
            tag = ""
            if helped is True:
                tag = " ✅"
            elif helped is False:
                tag = " ⚠️ correction hurt"
            print(f"      {city:<12} n={n:>3} | Raw MAE: {rm}° → Corrected: {cm}°{tag}")


# ═══════════════════════════════════════════════════════════════
# 2. BIAS CORRECTION IMPACT
# ═══════════════════════════════════════════════════════════════

def bias_correction_report(entries: list[dict]) -> dict:
    """Measure whether bias correction improves or hurts predictions."""
    resolved = [e for e in entries if e.get("resolved") and e.get("actual_temperature") is not None]
    if not resolved:
        return {"error": "No resolved entries"}

    # Compare: did the corrected forecasts land in the right bucket more often?
    raw_hits = 0
    corrected_hits = 0
    total = 0

    for e in resolved:
        actual_bucket = e.get("actual_bucket", "")
        if not actual_bucket:
            continue

        model_probs = e.get("model_bucket_probs", {})
        model_top = max(model_probs, key=model_probs.get) if model_probs else ""

        # To compare raw vs corrected, we'd need raw bucket probs too.
        # For now, compare forecast mean vs corrected mean distance to actual.
        raw_fc = e.get("raw_forecasts", {})
        corr_fc = e.get("corrected_forecasts", {})
        actual = e["actual_temperature"]

        if raw_fc and corr_fc:
            raw_mean = np.mean(list(raw_fc.values()))
            corr_mean = np.mean(list(corr_fc.values()))
            if abs(corr_mean - actual) < abs(raw_mean - actual):
                corrected_hits += 1
            else:
                raw_hits += 1
            total += 1

    return {
        "n_compared": total,
        "correction_helped_pct": round(corrected_hits / total * 100, 1) if total else 0,
        "correction_hurt_pct": round(raw_hits / total * 100, 1) if total else 0,
        "corrected_wins": corrected_hits,
        "raw_wins": raw_hits,
    }


def print_bias_report(report: dict):
    print(f"\n{'=' * 70}")
    print(f"  BIAS CORRECTION IMPACT ({report.get('n_compared', 0)} resolved events)")
    print(f"{'=' * 70}")
    print(f"  Correction helped: {report.get('corrected_wins', 0)} ({report.get('correction_helped_pct', 0)}%)")
    print(f"  Raw was better:    {report.get('raw_wins', 0)} ({report.get('correction_hurt_pct', 0)}%)")

    helped = report.get("correction_helped_pct", 0)
    if helped > 60:
        print(f"\n  ✅ Bias correction is adding value")
    elif helped > 45:
        print(f"\n  🟡 Bias correction is roughly neutral — may need recalibration")
    else:
        print(f"\n  🔴 Bias correction is hurting — consider rebuilding bias table or disabling")


# ═══════════════════════════════════════════════════════════════
# 3. EDGE ATTRIBUTION
# ═══════════════════════════════════════════════════════════════

def edge_attribution_report(entries: list[dict]) -> dict:
    """
    For traded events, decompose where the edge came from:
    - High model agreement vs dispersed market
    - Bias correction shifting probability mass
    - Tail buckets the market underprices
    """
    traded = [e for e in entries if e.get("traded")]
    if not traded:
        return {"error": "No traded events"}

    # Categorize edges
    agreement_driven = 0    # model agreement > 0.75 and market spread thin
    tail_driven = 0         # best edge is on a non-consensus bucket
    bias_driven = 0         # bias correction was active and shifted consensus
    total = len(traded)

    edge_by_tier = defaultdict(list)
    edge_by_lead = defaultdict(list)

    for e in traded:
        tier = e.get("confidence_tier", "LOW")
        lead = e.get("lead_hours", 0)
        max_edge = e.get("max_edge", 0)

        edge_by_tier[tier].append(max_edge)

        # Bin lead hours
        if lead <= 24:
            edge_by_lead["≤24h"].append(max_edge)
        elif lead <= 48:
            edge_by_lead["24-48h"].append(max_edge)
        elif lead <= 72:
            edge_by_lead["48-72h"].append(max_edge)
        else:
            edge_by_lead[">72h"].append(max_edge)

        # Classification
        agreement = e.get("model_agreement", 0)
        consensus = e.get("consensus_bucket", "")
        edges = e.get("edges", {})
        best_edge_bucket = max(edges, key=edges.get) if edges else ""

        if agreement >= 0.75:
            agreement_driven += 1
        if best_edge_bucket != consensus and best_edge_bucket:
            tail_driven += 1
        if e.get("bias_corrected", False):
            bias_driven += 1

    return {
        "n_traded": total,
        "agreement_driven": agreement_driven,
        "agreement_driven_pct": round(agreement_driven / total * 100, 1),
        "tail_driven": tail_driven,
        "tail_driven_pct": round(tail_driven / total * 100, 1),
        "bias_active": bias_driven,
        "bias_active_pct": round(bias_driven / total * 100, 1),
        "avg_edge_by_tier": {k: round(np.mean(v), 4) for k, v in edge_by_tier.items()},
        "avg_edge_by_lead": {k: round(np.mean(v), 4) for k, v in edge_by_lead.items()},
    }


def print_edge_report(report: dict):
    print(f"\n{'=' * 70}")
    print(f"  EDGE ATTRIBUTION ({report.get('n_traded', 0)} traded events)")
    print(f"{'=' * 70}")

    print(f"\n  Edge sources (categories overlap):")
    print(f"    High agreement (>0.75): {report.get('agreement_driven', 0)} "
          f"({report.get('agreement_driven_pct', 0)}%)")
    print(f"    Tail/non-consensus:     {report.get('tail_driven', 0)} "
          f"({report.get('tail_driven_pct', 0)}%)")
    print(f"    Bias correction active: {report.get('bias_active', 0)} "
          f"({report.get('bias_active_pct', 0)}%)")

    print(f"\n  Avg edge by confidence tier:")
    for tier, avg in sorted(report.get("avg_edge_by_tier", {}).items()):
        print(f"    {tier:<12} {avg:+.1%}")

    print(f"\n  Avg edge by lead time:")
    for lead, avg in sorted(report.get("avg_edge_by_lead", {}).items()):
        print(f"    {lead:<12} {avg:+.1%}")


# ═══════════════════════════════════════════════════════════════
# 4. CITY PROFITABILITY
# ═══════════════════════════════════════════════════════════════

def city_report(entries: list[dict]) -> dict:
    """Per-city scan count, trade count, P&L, and model accuracy."""
    # Per-scan metrics (every entry counts)
    by_city = defaultdict(lambda: {
        "scanned": 0, "traded": 0,
        "avg_edge": [], "model_errors": [],
    })

    for e in entries:
        city = e["city"]
        c = by_city[city]
        c["scanned"] += 1

        if e.get("traded"):
            c["traded"] += 1
            c["avg_edge"].append(e.get("max_edge", 0))

        if e.get("resolved"):
            err = e.get("model_error")
            if err is not None:
                c["model_errors"].append(err)

    # P&L metrics — deduped by event_key to avoid overcounting
    deduped = _dedup_by_event_key(entries)
    pnl_by_city = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0})

    for e in deduped:
        if not e.get("resolved") or not e.get("traded") or e.get("trade_pnl") is None:
            continue
        city = e["city"]
        p = pnl_by_city[city]
        p["total_pnl"] += e["trade_pnl"]
        if e["trade_pnl"] > 0:
            p["wins"] += 1
        elif e["trade_pnl"] < 0:
            p["losses"] += 1

    result = {}
    for city, c in sorted(by_city.items()):
        p = pnl_by_city[city]
        result[city] = {
            "scanned": c["scanned"],
            "traded": c["traded"],
            "wins": p["wins"],
            "losses": p["losses"],
            "total_pnl": round(p["total_pnl"], 2),
            "avg_edge": round(np.mean(c["avg_edge"]), 4) if c["avg_edge"] else 0,
            "model_mae": round(np.mean(np.abs(c["model_errors"])), 2) if c["model_errors"] else None,
            "model_bias": round(np.mean(c["model_errors"]), 2) if c["model_errors"] else None,
        }
    return result


def print_city_report(report: dict):
    print(f"\n{'=' * 70}")
    print(f"  CITY PROFITABILITY")
    print(f"{'=' * 70}")
    print(f"\n  {'City':<12} {'Scanned':>7} {'Traded':>7} {'W':>3} {'L':>3} "
          f"{'P&L':>9} {'AvgEdge':>8} {'MAE':>6} {'Bias':>6}")
    print(f"  {'─' * 66}")

    for city, d in report.items():
        mae = f"{d['model_mae']}°" if d["model_mae"] is not None else "  —"
        bias = f"{d['model_bias']:+.1f}°" if d["model_bias"] is not None else "  —"
        print(f"  {city:<12} {d['scanned']:>7} {d['traded']:>7} {d['wins']:>3} {d['losses']:>3} "
              f"${d['total_pnl']:>+8.2f} {d['avg_edge']:>+7.1%} {mae:>6} {bias:>6}")


# ═══════════════════════════════════════════════════════════════
# 5. CALIBRATION BY TIER
# ═══════════════════════════════════════════════════════════════

def calibration_by_tier(entries: list[dict]) -> dict:
    """
    For resolved traded events, check if confidence tiers predict win rate.
    Deduped by event_key to avoid overcounting from multi-lead-time scans.
    """
    deduped = _dedup_by_event_key(entries)
    resolved_traded = [
        e for e in deduped
        if e.get("resolved") and e.get("traded") and e.get("trade_pnl") is not None
    ]
    if not resolved_traded:
        return {"error": "No resolved traded events"}

    by_tier = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "edges": []})

    for e in resolved_traded:
        tier = e.get("confidence_tier", "LOW")
        pnl = e["trade_pnl"]
        t = by_tier[tier]
        t["pnl"] += pnl
        t["edges"].append(e.get("max_edge", 0))
        if pnl > 0:
            t["wins"] += 1
        else:
            t["losses"] += 1

    result = {}
    for tier, t in by_tier.items():
        total = t["wins"] + t["losses"]
        result[tier] = {
            "n": total,
            "win_rate": round(t["wins"] / total, 3) if total else 0,
            "total_pnl": round(t["pnl"], 2),
            "avg_edge": round(np.mean(t["edges"]), 4) if t["edges"] else 0,
        }
    return result


def print_calibration_report(report: dict):
    print(f"\n{'=' * 70}")
    print(f"  CALIBRATION BY CONFIDENCE TIER")
    print(f"{'=' * 70}")
    print(f"\n  {'Tier':<12} {'N':>5} {'WinRate':>8} {'P&L':>10} {'AvgEdge':>8}")
    print(f"  {'─' * 46}")

    for tier in ["LOCK", "STRONG", "SAFE", "NEAR_SAFE", "LOW"]:
        d = report.get(tier)
        if not d:
            continue
        print(f"  {tier:<12} {d['n']:>5} {d['win_rate']:>7.0%} "
              f"${d['total_pnl']:>+9.2f} {d['avg_edge']:>+7.1%}")


# ═══════════════════════════════════════════════════════════════
# 6. SUMMARY / QUICK HEALTH CHECK
# ═══════════════════════════════════════════════════════════════

def summary_report(entries: list[dict]) -> dict:
    """High-level health metrics."""
    n_total = len(entries)
    n_traded = sum(1 for e in entries if e.get("traded"))
    n_resolved = sum(1 for e in entries if e.get("resolved"))

    # P&L metrics deduped to avoid overcounting from multi-lead-time scans
    deduped = _dedup_by_event_key(entries)
    deduped_traded_resolved = [
        e for e in deduped
        if e.get("resolved") and e.get("traded") and e.get("trade_pnl") is not None
    ]
    n_resolved_traded = len(deduped_traded_resolved)
    total_pnl = sum(e.get("trade_pnl", 0) or 0 for e in deduped_traded_resolved)

    # Trade rate
    trade_rate = n_traded / n_total if n_total else 0

    # Unique cities and dates
    cities = set(e["city"] for e in entries)
    dates = set(e["target_date"] for e in entries)

    return {
        "total_events_logged": n_total,
        "events_traded": n_traded,
        "events_resolved": n_resolved,
        "resolved_with_pnl": n_resolved_traded,
        "trade_rate": round(trade_rate, 3),
        "total_pnl": round(total_pnl, 2),
        "unique_cities": len(cities),
        "unique_dates": len(dates),
        "date_range": [min(dates), max(dates)] if dates else [],
    }


def print_summary(report: dict):
    print(f"\n{'=' * 70}")
    print(f"  WEATHER STRATEGY — DIAGNOSTIC SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Events logged:     {report['total_events_logged']}")
    print(f"  Events traded:     {report['events_traded']} "
          f"({report['trade_rate']:.0%} trade rate)")
    print(f"  Resolved (traded): {report['resolved_with_pnl']}")
    print(f"  Total P&L:         ${report['total_pnl']:+,.2f}")
    print(f"  Cities: {report['unique_cities']} | "
          f"Dates: {report['unique_dates']}")
    if report["date_range"]:
        print(f"  Range: {report['date_range'][0]} → {report['date_range'][1]}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main(argv: list[str] = None):
    parser = argparse.ArgumentParser(
        description="Weather Diagnostics — Analyze event log for strategy tuning"
    )
    parser.add_argument("--models", action="store_true", help="Model accuracy report")
    parser.add_argument("--bias", action="store_true", help="Bias correction impact")
    parser.add_argument("--edge", action="store_true", help="Edge attribution")
    parser.add_argument("--cities", action="store_true", help="Per-city profitability")
    parser.add_argument("--calibration", action="store_true", help="Calibration by tier")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--log-file", type=str, default=None, help="Path to event log")
    args = parser.parse_args(argv)

    log_file = Path(args.log_file) if args.log_file else None
    entries = load_log(log_file)

    # Default: run all reports
    run_all = not any([args.models, args.bias, args.edge, args.cities, args.calibration])

    results = {}

    if run_all or args.models:
        r = model_accuracy_report(entries)
        results["model_accuracy"] = r
        if not args.json:
            print_model_report(r)

    if run_all or args.bias:
        r = bias_correction_report(entries)
        results["bias_correction"] = r
        if not args.json:
            print_bias_report(r)

    if run_all or args.edge:
        r = edge_attribution_report(entries)
        results["edge_attribution"] = r
        if not args.json:
            print_edge_report(r)

    if run_all or args.cities:
        r = city_report(entries)
        results["cities"] = r
        if not args.json:
            print_city_report(r)

    if run_all or args.calibration:
        r = calibration_by_tier(entries)
        results["calibration_by_tier"] = r
        if not args.json:
            print_calibration_report(r)

    if run_all:
        r = summary_report(entries)
        results["summary"] = r
        if not args.json:
            print_summary(r)

    if args.json:
        print(json.dumps(results, indent=2, default=str))

    print()


if __name__ == "__main__":
    main()