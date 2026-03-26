"""
Weather Collector Scheduler
Runs the data collector at 04, 10, 16, 22 UTC (4h after each model cycle).
Leave this running in a terminal.

Usage:
    python run_weather_collector.py
"""

import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

SCHEDULE_HOURS_UTC = [4, 10, 16, 22]


def next_run_time() -> datetime:
    """Calculate the next scheduled run time."""
    now = datetime.now(timezone.utc)

    # Check each scheduled hour today and tomorrow
    for day_offset in range(2):
        for hour in SCHEDULE_HOURS_UTC:
            candidate = now.replace(
                hour=hour, minute=0, second=0, microsecond=0
            ) + timedelta(days=day_offset)
            if candidate > now:
                return candidate

    # Shouldn't reach here, but fallback to first slot tomorrow
    tomorrow = now + timedelta(days=1)
    return tomorrow.replace(
        hour=SCHEDULE_HOURS_UTC[0], minute=0, second=0, microsecond=0
    )


def run_collector():
    """Run the data collector once."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "weather.data_collector", "--once"],
            capture_output=False,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  Error running collector: {e}")
        return False


def main():
    print("=" * 50)
    print("  Weather Collector Scheduler")
    print(f"  Schedule: {SCHEDULE_HOURS_UTC} UTC")
    print(f"  Started:  {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
    print("=" * 50)

    while True:
        target = next_run_time()
        now = datetime.now(timezone.utc)
        wait_seconds = (target - now).total_seconds()
        wait_hours = wait_seconds / 3600

        print(f"\n  Next run: {target:%Y-%m-%d %H:%M} UTC ({wait_hours:.1f}h from now)")
        print(f"  Sleeping until then... (Ctrl+C to stop)\n")

        time.sleep(wait_seconds)

        print(f"{'─' * 50}")
        print(f"  {datetime.now(timezone.utc):%H:%M:%S} UTC — Running collector...")
        print(f"{'─' * 50}")

        success = run_collector()

        status = "✅ Done" if success else "⚠️  Collector returned errors"
        print(f"\n  {status}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Scheduler stopped. Data collected so far is saved.")