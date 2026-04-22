"""
Weather Pipeline Scheduler
Runs data collection + scanner trading at 04, 10, 16, 22 UTC
(4h after each major model update cycle).

Each run:
    1. Collects snapshots (forecasts + market prices) for forward validation
    2. Checks open trades for resolution
    3. Scans for tradeable edges and enters trades (paper or live)

Leave this running in a terminal.

Usage:
    python run_weather_collector.py
    python run_weather_collector.py --once           # Run one cycle and exit
    python run_weather_collector.py --scan-only      # Skip data collection
    python run_weather_collector.py --collect-only   # Skip scanner
    python run_weather_collector.py --live           # Live mode (dry-run by default)
    python run_weather_collector.py --live --no-dry-run  # Real money
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


def run_cycle(collect: bool = True, scan: bool = True, scanner_args: list[str] = None):
    """Run one full cycle: collect → check resolutions → scan & trade."""
    if scanner_args is None:
        scanner_args = ["--paper-trade"]
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
        # Resolution check needs --live (to load live_trades.json) but
        # never --no-dry-run (it doesn't place orders, so no confirmation needed)
        check_args = ["--check"]
        if "--live" in scanner_args:
            check_args.append("--live")
        results["check"] = run_step(
            "Resolution check",
            "weather.scanner",
            check_args,
        )

        # Step 3: Scan and trade
        trade_label = "Scan & trade"
        if "--live" in scanner_args:
            trade_label += " (LIVE" + (" DRY-RUN)" if "--no-dry-run" not in scanner_args else ")")
        results["trade"] = run_step(
            trade_label,
            "weather.scanner",
            scanner_args,
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
    parser.add_argument(
        "--live", action="store_true",
        help="Run scanner in live execution mode (dry-run by default)"
    )
    parser.add_argument(
        "--no-dry-run", action="store_true",
        help="Disable dry-run — place real orders"
    )
    args = parser.parse_args()

    collect = not args.scan_only
    scan = not args.collect_only

    # Build scanner args
    if args.live:
        scanner_args = ["--live"]
        if args.no_dry_run:
            scanner_args.append("--no-dry-run")
            scanner_args.append("--confirmed")  # Skip interactive prompt in subprocess
    else:
        scanner_args = ["--paper-trade"]

    mode_parts = []
    if collect:
        mode_parts.append("collect")
    if scan:
        if args.live:
            mode_parts.append("LIVE trade" + (" (dry-run)" if not args.no_dry_run else ""))
        else:
            mode_parts.append("paper trade")
    mode = " + ".join(mode_parts)

    print("=" * 55)
    print("  Weather Pipeline Scheduler")
    print(f"  Mode:     {mode}")
    print(f"  Schedule: {SCHEDULE_HOURS_UTC} UTC")
    print(f"  Started:  {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
    print("=" * 55)

    # One-time confirmation for real-money mode
    if args.live and args.no_dry_run:
        print("\n" + "!" * 55)
        print("  ⚠️  LIVE TRADING — REAL MONEY")
        print("  The scheduler will place real orders on every cycle.")
        print("!" * 55)
        confirm = input("\n  Type 'CONFIRM' to proceed: ").strip()
        if confirm != "CONFIRM":
            print("  Aborted.")
            return

    if args.once:
        print(f"\n{'─' * 55}")
        print(f"  {datetime.now(timezone.utc):%H:%M:%S} UTC — Single run")
        print(f"{'─' * 55}")
        run_cycle(collect=collect, scan=scan, scanner_args=scanner_args)
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

        results = run_cycle(collect=collect, scan=scan, scanner_args=scanner_args)

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