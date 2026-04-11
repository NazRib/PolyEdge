#!/usr/bin/env python3
"""
Run the Mention Market Strategy.

Usage:
    # Dry run -- scan + estimate, no trades
    python run_mentions.py

    # Live paper trading
    python run_mentions.py --live

    # Use GPT-5.4 instead of Claude
    python run_mentions.py --model gpt

    # Custom parameters
    python run_mentions.py --min-edge 0.08 --kelly 0.15 --max-days 7
"""

import argparse
import logging
import os
import sys

if os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from strategies.mention_strategy import MentionStrategy
from core.llm_providers import provider_ready


def main():
    parser = argparse.ArgumentParser(description="Polymarket Mention Strategy")
    parser.add_argument("--live", action="store_true",
                        help="Enter paper trades (default: dry run only)")
    parser.add_argument("--model", type=str, default="claude",
                        choices=["claude", "gpt"],
                        help="LLM provider: claude or gpt (default: claude)")
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="Minimum edge to trade (default: 0.05)")
    parser.add_argument("--kelly", type=float, default=0.20,
                        help="Kelly fraction (default: 0.20)")
    parser.add_argument("--max-days", type=float, default=14,
                        help="Max days to resolution (default: 14)")
    parser.add_argument("--event-cap", type=float, default=0.25,
                        help="Max bankroll %% per event (default: 0.25)")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Starting bankroll (default: 1000)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Validate LLM provider credentials
    ready, msg = provider_ready(args.model)
    print(f"\n  LLM provider: {args.model} ({msg})")
    if not ready:
        print(f"  Cannot run with --model {args.model} -- credentials missing")
        sys.exit(1)

    strategy = MentionStrategy(
        llm_provider=args.model,
        kelly_fraction=args.kelly,
        min_edge=args.min_edge,
        max_event_exposure_pct=args.event_cap,
        max_days_to_resolution=args.max_days,
    )
    strategy.trader.bankroll = args.bankroll

    if args.live:
        signals = strategy.run()
    else:
        signals = strategy.dry_run()

    # Quick stats
    if signals:
        edges = [s["edge"] for s in signals if "edge" in s]
        actionable = [s for s in signals if abs(s.get("edge", 0)) >= args.min_edge]
        print(f"\nTotal words analyzed: {len(signals)}")
        print(f"Actionable edges (>={args.min_edge:.0%}): {len(actionable)}")
        if edges:
            print(f"Edge range: {min(edges):+.0%} to {max(edges):+.0%}")


if __name__ == "__main__":
    main()