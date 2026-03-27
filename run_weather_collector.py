"""
Weather Pipeline Scheduler
Runs data collection + scanner paper trading at 04, 10, 16, 22 UTC
(4h after each major model update cycle).

Each run:
    1. Collects snapshots (forecasts + market prices) for forward validation
    2. Checks open trades for resolution
    3. Scans for tradeable edges and enters paper trades

Leave this running in a terminal.

Usage:
    python run_weather_collector.py
    python run_weather_collector.py --once           # Run one cycle and exit
    python run_weather_collector.py --scan-only      # Skip data collection
    python run_weather_collector.py --collect-only   # Skip scanner
"""

import subprocess
import sys
import time
import argparse
from datetime import datetime, timezone, timedelta

SCHEDULE_HOURS_UTC = [4, 10, 16, 22]


def next_run_time() -> datetime:
    """Calculate the next scheduled run time."""
    now = datetime.now(timezone.utc)

    for day_offset in range(2):
        for hour in SCHEDULE_HOURS_UTC:
            candidate = now.replace(
                hour=hour, minute=0, second=0, microsecond=0
            ) + timedelta(days=day_offset)
            if candidate > now:
                return candidate

    tomorrow = now + timedelta(days=1)
    return tomorrow.replace(
        hour=SCHEDULE_HOURS_UTC[0], minute=0, second=0, microsecond=0
    )


def run_step(label: str, module: str, args: list[str]) -> bool:
    """Run a weather module step."""
    print(f"\n  ▶ {label}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", module] + args,
            capture_output=False,
        )
        ok = result.returncode == 0
        print(f"    {'✅' if ok else '⚠️ '} {label} — {'done' if ok else 'returned errors'}")
        return ok
    except Exception as e:
        print(f"    ❌ {label} — {e}")
        return False


def run_cycle(collect: bool = True, scan: bool = True):
    """Run one full cycle: collect → check resolutions → scan & trade."""
    results = {}

    # Step 1: Data collection (snapshot for validation)
    if collect:
        results["collect"] = run_step(
            "Data collection (snapshots)",
            "weather.data_collector",
            ["--once"],
        )

    # Step 2: Check open trades for resolution
    if scan:
        results["check"] = run_step(
            "Resolution check",
            "weather.scanner",
            ["--check"],
        )

        # Step 3: Scan and paper trade
        results["trade"] = run_step(
            "Scan & paper trade",
            "weather.scanner",
            ["--paper-trade"],
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Weather Pipeline Scheduler"
    )
    parser.add_argument(
        "--collect-only", action="store_true",
        help="Only run data collection, skip scanner"
    )
    parser.add_argument(
        "--scan-only", action="store_true",
        help="Only run scanner, skip data collection"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one cycle immediately and exit"
    )
    args = parser.parse_args()

    collect = not args.scan_only
    scan = not args.collect_only

    mode_parts = []
    if collect:
        mode_parts.append("collect")
    if scan:
        mode_parts.append("scan + trade")
    mode = " + ".join(mode_parts)

    print("=" * 55)
    print("  Weather Pipeline Scheduler")
    print(f"  Mode:     {mode}")
    print(f"  Schedule: {SCHEDULE_HOURS_UTC} UTC")
    print(f"  Started:  {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
    print("=" * 55)

    if args.once:
        print(f"\n{'─' * 55}")
        print(f"  {datetime.now(timezone.utc):%H:%M:%S} UTC — Single run")
        print(f"{'─' * 55}")
        run_cycle(collect=collect, scan=scan)
        return

    while True:
        target = next_run_time()
        now = datetime.now(timezone.utc)
        wait_seconds = (target - now).total_seconds()
        wait_hours = wait_seconds / 3600

        print(f"\n  Next run: {target:%Y-%m-%d %H:%M} UTC ({wait_hours:.1f}h from now)")
        print(f"  Sleeping until then... (Ctrl+C to stop)\n")

        time.sleep(wait_seconds)

        print(f"{'─' * 55}")
        print(f"  {datetime.now(timezone.utc):%H:%M:%S} UTC — Running cycle")
        print(f"{'─' * 55}")

        results = run_cycle(collect=collect, scan=scan)

        passed = sum(1 for v in results.values() if v)
        total = len(results)
        print(f"\n  Cycle complete: {passed}/{total} steps succeeded")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Scheduler stopped.")
        print("  Data and trades are saved. Run these to check status:")
        print("    python -m weather.scanner --report")
        print("    python -m weather.data_collector --report")