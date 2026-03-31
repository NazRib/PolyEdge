"""
Whale Profiler — Behavioral Analysis of Top Polymarket Traders

Builds rich profiles for each whale by analyzing their positions, trade
patterns, and category-specific performance. Profiles persist across pipeline
runs so knowledge accumulates.

The key insight: not all profitable traders are equally informative signals.
A market maker with $500K PnL provides no directional information. A conviction
trader with $50K PnL who wins 65% of the time on political markets is gold
when you see them enter a political market. This module tells the difference.

Strategy classifications:
    - CONVICTION:    Few large positions, high win rate, concentrated in categories
    - DIVERSIFIED:   Many positions across categories, moderate sizing
    - SPECIALIST:    High concentration in one category, decent win rate there
    - MARKET_MAKER:  Very high position count, balanced YES/NO, high volume/PnL ratio
    - UNKNOWN:       Insufficient data to classify

Usage:
    profiler = WhaleProfiler()
    profiler.build_profiles()                 # Fetch data + compute profiles
    profile = profiler.get_profile(wallet)    # Look up a specific whale
    profiler.save()                           # Persist to disk
"""

import json
import time
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

import requests
import numpy as np

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────

DATA_API_URL = "https://data-api.polymarket.com"

# Categories the Polymarket leaderboard supports
LEADERBOARD_CATEGORIES = [
    "POLITICS", "SPORTS", "CRYPTO", "ECONOMICS",
    "POP_CULTURE", "SCIENCE", "TECHNOLOGY",
]

# Strategy classification thresholds
MARKET_MAKER_POSITION_THRESHOLD = 40   # 40+ open positions → likely MM
MARKET_MAKER_VOL_PNL_RATIO = 50       # Volume / |PnL| > 50 → likely MM
CONVICTION_MAX_POSITIONS = 15          # ≤15 positions → conviction style
CONVICTION_MIN_AVG_SIZE = 500          # $500+ avg position → conviction
SPECIALIST_CONCENTRATION = 0.60        # 60%+ in one category → specialist
MIN_CLOSED_FOR_WIN_RATE = 5            # Need 5+ closed positions for win rate


# ─── Data Classes ────────────────────────────────────────

@dataclass
class CategoryStats:
    """Performance metrics for a whale in a specific market category."""
    category: str
    pnl: float = 0.0
    volume: float = 0.0
    position_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    total_resolved: int = 0

    @property
    def win_rate(self) -> Optional[float]:
        if self.total_resolved < MIN_CLOSED_FOR_WIN_RATE:
            return None
        return self.win_count / self.total_resolved if self.total_resolved > 0 else None

    @property
    def credibility(self) -> float:
        """
        Category-specific credibility score (0-1).

        Combines PnL, win rate, and sample size into a single number.
        Intentionally conservative — requires real evidence before
        giving high credibility.
        """
        # Base: PnL-based credibility (log-scaled, so $10K and $100K don't differ 10x)
        if self.pnl <= 0:
            pnl_score = 0.0
        else:
            pnl_score = min(1.0, np.log1p(self.pnl / 1000) / 5.0)

        # Win rate bonus (only if we have enough data)
        wr = self.win_rate
        if wr is not None and wr > 0.5:
            win_bonus = (wr - 0.5) * 2.0  # 0 at 50%, 1.0 at 100%
            win_bonus = min(0.4, win_bonus)
        else:
            win_bonus = 0.0

        # Sample size discount — fewer resolved markets → less trust
        sample_factor = min(1.0, self.total_resolved / 20)

        return min(1.0, (pnl_score * 0.5 + win_bonus * 0.3 + sample_factor * 0.2))


@dataclass
class WhaleProfile:
    """
    Complete behavioral profile for a single whale.

    Built from leaderboard data + position analysis. Persisted to disk.
    """
    wallet: str
    username: str = ""
    global_pnl: float = 0.0
    global_volume: float = 0.0
    global_rank: str = ""

    # Behavioral metrics (computed from positions)
    open_position_count: int = 0
    closed_position_count: int = 0
    avg_position_size: float = 0.0
    median_position_size: float = 0.0
    max_position_size: float = 0.0
    total_position_value: float = 0.0

    # YES/NO balance (for detecting market makers)
    yes_position_count: int = 0
    no_position_count: int = 0

    # Category breakdown
    category_stats: dict[str, CategoryStats] = field(default_factory=dict)
    primary_category: str = ""
    category_concentration: float = 0.0  # Herfindahl-like: 1.0 = all in one category

    # Strategy classification
    strategy: str = "UNKNOWN"
    strategy_confidence: float = 0.0

    # Overall credibility (aggregated from categories)
    overall_credibility: float = 0.0

    # Win rate (across all categories)
    global_win_count: int = 0
    global_loss_count: int = 0
    global_total_resolved: int = 0

    # Metadata
    profiled_at: str = ""
    data_quality: str = "PARTIAL"  # PARTIAL, FULL, STALE

    @property
    def global_win_rate(self) -> Optional[float]:
        if self.global_total_resolved < MIN_CLOSED_FOR_WIN_RATE:
            return None
        return self.global_win_count / self.global_total_resolved

    @property
    def vol_pnl_ratio(self) -> float:
        """Volume-to-PnL ratio. High values suggest market making."""
        if abs(self.global_pnl) < 100:
            return float('inf')
        return self.global_volume / abs(self.global_pnl)

    def credibility_for_category(self, category: str) -> float:
        """
        Get credibility score for a specific market category.

        Falls back to overall credibility if no category-specific data.
        Applies a discount for categories where the whale has no track record.
        """
        cat_upper = category.upper().replace(" ", "_")

        # Direct match
        if cat_upper in self.category_stats:
            cat_cred = self.category_stats[cat_upper].credibility
            if cat_cred > 0:
                return cat_cred

        # Fuzzy match: try partial matches
        for cat_key, stats in self.category_stats.items():
            if cat_upper in cat_key or cat_key in cat_upper:
                if stats.credibility > 0:
                    return stats.credibility

        # No category data — discount the overall credibility
        return self.overall_credibility * 0.4

    def signal_weight(self, category: str = "") -> float:
        """
        Compute how much weight this whale's signal should get.

        This is the key output — used by the ensemble estimator.
        Market makers get heavily discounted. Conviction traders in
        their specialty get boosted.
        """
        base = self.credibility_for_category(category) if category else self.overall_credibility

        # Strategy multipliers
        strategy_mult = {
            "CONVICTION": 1.3,
            "SPECIALIST": 1.2,
            "DIVERSIFIED": 1.0,
            "MARKET_MAKER": 0.15,  # Heavy discount — MM positions are noise
            "UNKNOWN": 0.6,
        }.get(self.strategy, 0.6)

        return min(1.0, base * strategy_mult)

    def summary(self) -> str:
        wr = self.global_win_rate
        wr_str = f"{wr:.0%}" if wr is not None else "N/A"
        return (
            f"{self.username or self.wallet[:10]}… | "
            f"Strategy: {self.strategy} | "
            f"PnL: ${self.global_pnl:,.0f} | "
            f"Win rate: {wr_str} | "
            f"Positions: {self.open_position_count} open | "
            f"Primary: {self.primary_category or 'N/A'} | "
            f"Signal weight: {self.signal_weight():.2f}"
        )

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        d = {
            "wallet": self.wallet,
            "username": self.username,
            "global_pnl": self.global_pnl,
            "global_volume": self.global_volume,
            "global_rank": self.global_rank,
            "open_position_count": self.open_position_count,
            "closed_position_count": self.closed_position_count,
            "avg_position_size": self.avg_position_size,
            "median_position_size": self.median_position_size,
            "max_position_size": self.max_position_size,
            "total_position_value": self.total_position_value,
            "yes_position_count": self.yes_position_count,
            "no_position_count": self.no_position_count,
            "primary_category": self.primary_category,
            "category_concentration": self.category_concentration,
            "strategy": self.strategy,
            "strategy_confidence": self.strategy_confidence,
            "overall_credibility": self.overall_credibility,
            "global_win_count": self.global_win_count,
            "global_loss_count": self.global_loss_count,
            "global_total_resolved": self.global_total_resolved,
            "profiled_at": self.profiled_at,
            "data_quality": self.data_quality,
            "category_stats": {
                k: {
                    "category": v.category,
                    "pnl": v.pnl,
                    "volume": v.volume,
                    "position_count": v.position_count,
                    "win_count": v.win_count,
                    "loss_count": v.loss_count,
                    "total_resolved": v.total_resolved,
                }
                for k, v in self.category_stats.items()
            },
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WhaleProfile":
        """Deserialize from JSON storage."""
        cat_stats = {}
        for k, v in d.get("category_stats", {}).items():
            cat_stats[k] = CategoryStats(**v)

        profile = cls(
            wallet=d["wallet"],
            username=d.get("username", ""),
            global_pnl=d.get("global_pnl", 0),
            global_volume=d.get("global_volume", 0),
            global_rank=d.get("global_rank", ""),
            open_position_count=d.get("open_position_count", 0),
            closed_position_count=d.get("closed_position_count", 0),
            avg_position_size=d.get("avg_position_size", 0),
            median_position_size=d.get("median_position_size", 0),
            max_position_size=d.get("max_position_size", 0),
            total_position_value=d.get("total_position_value", 0),
            yes_position_count=d.get("yes_position_count", 0),
            no_position_count=d.get("no_position_count", 0),
            primary_category=d.get("primary_category", ""),
            category_concentration=d.get("category_concentration", 0),
            strategy=d.get("strategy", "UNKNOWN"),
            strategy_confidence=d.get("strategy_confidence", 0),
            overall_credibility=d.get("overall_credibility", 0),
            global_win_count=d.get("global_win_count", 0),
            global_loss_count=d.get("global_loss_count", 0),
            global_total_resolved=d.get("global_total_resolved", 0),
            profiled_at=d.get("profiled_at", ""),
            data_quality=d.get("data_quality", "PARTIAL"),
            category_stats=cat_stats,
        )
        return profile


# ─── Whale Profiler ─────────────────────────────────────

class WhaleProfiler:
    """
    Builds and maintains behavioral profiles for top Polymarket traders.

    Workflow:
        1. Fetch leaderboards (overall + per-category) to identify whales
        2. For each whale, fetch open + closed positions
        3. Compute behavioral metrics and classify strategy
        4. Persist profiles to disk

    Profiles are additive — new runs merge with existing data rather
    than replacing it, so knowledge accumulates over time.

    Usage:
        profiler = WhaleProfiler(data_dir="data")
        profiler.build_profiles(max_whales=30)
        profiler.save()

        # Later, in the pipeline:
        profile = profiler.get_profile("0xabc...")
        weight = profile.signal_weight(category="POLITICS")
    """

    def __init__(
        self,
        data_dir: str = "data",
        top_n_whales: int = 50,
        min_pnl: float = 5000.0,
        rate_limit_delay: float = 0.6,
        max_positions_per_whale: int = 200,
    ):
        self.data_dir = data_dir
        self.top_n_whales = top_n_whales
        self.min_pnl = min_pnl
        self.rate_limit_delay = rate_limit_delay
        self.max_positions_per_whale = max_positions_per_whale

        self.profiles_file = os.path.join(data_dir, "whale_profiles.json")
        self._profiles: dict[str, WhaleProfile] = {}
        self._last_request = 0.0

        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketEdge/2.0",
        })

        # Load existing profiles from disk
        self._load_from_disk()

    # ─── HTTP helpers ────────────────────────────────────

    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request = time.time()

    def _get(self, path: str, params: dict = None) -> list | dict:
        self._throttle()
        try:
            resp = self.session.get(
                f"{DATA_API_URL}{path}",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                logger.warning("Rate limited — backing off 5s")
                time.sleep(5)
                return self._get(path, params)
            raise
        except Exception as e:
            logger.debug(f"API call failed: {path} — {e}")
            raise

    # ─── Persistence ─────────────────────────────────────

    def _load_from_disk(self):
        """Load previously saved profiles."""
        if not os.path.exists(self.profiles_file):
            return

        try:
            with open(self.profiles_file, "r") as f:
                data = json.load(f)

            for wallet, profile_dict in data.get("profiles", {}).items():
                self._profiles[wallet] = WhaleProfile.from_dict(profile_dict)

            logger.info(
                f"📂 Loaded {len(self._profiles)} whale profiles from disk "
                f"(saved {data.get('saved_at', 'unknown')})"
            )
        except Exception as e:
            logger.warning(f"Failed to load whale profiles: {e}")

    def save(self):
        """Persist profiles to disk."""
        os.makedirs(self.data_dir, exist_ok=True)

        data = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "profile_count": len(self._profiles),
            "profiles": {
                wallet: profile.to_dict()
                for wallet, profile in self._profiles.items()
            },
        }

        with open(self.profiles_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"💾 Saved {len(self._profiles)} whale profiles to {self.profiles_file}")

    # ─── Profile Access ──────────────────────────────────

    def get_profile(self, wallet: str) -> Optional[WhaleProfile]:
        """Look up a whale's profile by wallet address."""
        return self._profiles.get(wallet.lower())

    @property
    def profiles(self) -> dict[str, WhaleProfile]:
        return self._profiles

    @property
    def profile_count(self) -> int:
        return len(self._profiles)

    # ─── Profile Building ────────────────────────────────

    def build_profiles(self, max_whales: int = None) -> int:
        """
        Build or update profiles for top whales.

        1. Fetches overall leaderboard + per-category leaderboards
        2. For each unique whale, fetches positions (open + closed)
        3. Computes behavioral metrics
        4. Classifies strategy

        Returns the number of profiles built/updated.
        """
        max_whales = max_whales or self.top_n_whales

        logger.info("🐋 Building whale profiles...")
        logger.info(f"   Target: up to {max_whales} whales, min PnL ${self.min_pnl:,.0f}")

        # Step 1: Gather whale candidates from leaderboards
        candidates = self._gather_candidates()
        logger.info(f"   Found {len(candidates)} unique whale candidates across leaderboards")

        # Step 2: Profile each whale
        profiled = 0
        for wallet, base_data in list(candidates.items())[:max_whales]:
            try:
                profile = self._build_single_profile(wallet, base_data)
                if profile:
                    self._profiles[wallet] = profile
                    profiled += 1
                    if profiled % 10 == 0:
                        logger.info(f"   Profiled {profiled}/{min(max_whales, len(candidates))} whales...")
            except Exception as e:
                logger.debug(f"   Failed to profile {wallet[:12]}…: {e}")

        logger.info(f"✅ Profiled {profiled} whales (total in database: {len(self._profiles)})")

        # Auto-save after building
        self.save()

        return profiled

    def _gather_candidates(self) -> dict[str, dict]:
        """
        Fetch leaderboards (overall + per-category) and build a
        deduplicated candidate dict: { wallet -> base_data }.

        base_data includes global PnL/vol from overall leaderboard
        and per-category PnL from category leaderboards.
        """
        candidates: dict[str, dict] = {}

        # Overall leaderboard
        try:
            overall = self._get("/v1/leaderboard", {
                "limit": min(self.top_n_whales, 50),
                "timePeriod": "ALL",
                "orderBy": "PNL",
            })
            if isinstance(overall, list):
                for entry in overall:
                    wallet = entry.get("proxyWallet", "").lower()
                    pnl = float(entry.get("pnl", 0))
                    if wallet and pnl >= self.min_pnl:
                        candidates[wallet] = {
                            "wallet": wallet,
                            "username": entry.get("userName", ""),
                            "global_pnl": pnl,
                            "global_volume": float(entry.get("vol", 0)),
                            "global_rank": str(entry.get("rank", "?")),
                            "category_pnl": {},
                        }
        except Exception as e:
            logger.warning(f"Overall leaderboard fetch failed: {e}")

        # Per-category leaderboards
        for category in LEADERBOARD_CATEGORIES:
            try:
                cat_data = self._get("/v1/leaderboard", {
                    "limit": 25,
                    "timePeriod": "ALL",
                    "category": category,
                    "orderBy": "PNL",
                })
                if not isinstance(cat_data, list):
                    continue

                for entry in cat_data:
                    wallet = entry.get("proxyWallet", "").lower()
                    pnl = float(entry.get("pnl", 0))
                    vol = float(entry.get("vol", 0))

                    if not wallet:
                        continue

                    if wallet in candidates:
                        candidates[wallet]["category_pnl"][category] = {
                            "pnl": pnl, "vol": vol,
                        }
                    elif pnl >= self.min_pnl:
                        candidates[wallet] = {
                            "wallet": wallet,
                            "username": entry.get("userName", ""),
                            "global_pnl": pnl,
                            "global_volume": vol,
                            "global_rank": str(entry.get("rank", "?")),
                            "category_pnl": {category: {"pnl": pnl, "vol": vol}},
                        }

            except Exception as e:
                logger.debug(f"Category leaderboard failed ({category}): {e}")

        # Sort by global PnL descending
        candidates = dict(
            sorted(candidates.items(), key=lambda x: x[1].get("global_pnl", 0), reverse=True)
        )

        return candidates

    def _build_single_profile(self, wallet: str, base_data: dict) -> Optional[WhaleProfile]:
        """
        Build a complete profile for a single whale.

        Fetches their open + closed positions and computes all metrics.
        Merges with any existing profile data.
        """
        profile = WhaleProfile(
            wallet=wallet,
            username=base_data.get("username", ""),
            global_pnl=base_data.get("global_pnl", 0),
            global_volume=base_data.get("global_volume", 0),
            global_rank=base_data.get("global_rank", ""),
            profiled_at=datetime.now(timezone.utc).isoformat(),
        )

        # Seed category stats from leaderboard data
        for cat, cat_data in base_data.get("category_pnl", {}).items():
            profile.category_stats[cat] = CategoryStats(
                category=cat,
                pnl=cat_data.get("pnl", 0),
                volume=cat_data.get("vol", 0),
            )

        # Fetch open positions
        open_positions = self._fetch_positions(wallet, closed=False)
        closed_positions = self._fetch_positions(wallet, closed=True)

        if open_positions is None and closed_positions is None:
            profile.data_quality = "PARTIAL"
            self._classify_strategy(profile)
            return profile

        # Analyze open positions
        if open_positions:
            self._analyze_open_positions(profile, open_positions)

        # Analyze closed positions (for win rates)
        if closed_positions:
            self._analyze_closed_positions(profile, closed_positions)

        # Compute category concentration
        self._compute_category_concentration(profile)

        # Classify strategy
        self._classify_strategy(profile)

        # Compute overall credibility
        self._compute_overall_credibility(profile)

        profile.data_quality = "FULL"

        return profile

    def _fetch_positions(self, wallet: str, closed: bool = False) -> Optional[list]:
        """Fetch open or closed positions for a wallet."""
        endpoint = "/positions" if not closed else "/positions"
        params = {
            "user": wallet,
            "limit": self.max_positions_per_whale,
            "sizeThreshold": 1,
        }
        if closed:
            params["redeemed"] = "true"

        try:
            data = self._get(endpoint, params)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.debug(f"Position fetch failed for {wallet[:12]}…: {e}")
            return None

    def _analyze_open_positions(self, profile: WhaleProfile, positions: list):
        """Extract behavioral metrics from open positions."""
        sizes = []
        for pos in positions:
            size = float(pos.get("currentValue", 0) or pos.get("size", 0) or 0)
            if size <= 0:
                continue

            sizes.append(size)

            # Track YES/NO balance
            outcome = (pos.get("outcome", "") or "").upper()
            if outcome == "YES" or pos.get("outcomeIndex", 0) == 0:
                profile.yes_position_count += 1
            else:
                profile.no_position_count += 1

            # Track category (from market slug or title heuristics)
            cat = self._infer_category(pos)
            if cat and cat in profile.category_stats:
                profile.category_stats[cat].position_count += 1
            elif cat:
                profile.category_stats[cat] = CategoryStats(
                    category=cat, position_count=1,
                )

        if sizes:
            profile.open_position_count = len(sizes)
            profile.avg_position_size = float(np.mean(sizes))
            profile.median_position_size = float(np.median(sizes))
            profile.max_position_size = float(np.max(sizes))
            profile.total_position_value = float(np.sum(sizes))

    def _analyze_closed_positions(self, profile: WhaleProfile, positions: list):
        """Extract win/loss data from resolved positions."""
        profile.closed_position_count = len(positions)

        for pos in positions:
            cash_pnl = float(pos.get("cashPnl", 0) or 0)
            redeemable = pos.get("redeemable", False)

            # A position is "won" if it has positive cashPnl or is redeemable
            is_win = cash_pnl > 0 or redeemable

            if cash_pnl != 0 or redeemable:
                profile.global_total_resolved += 1
                if is_win:
                    profile.global_win_count += 1
                else:
                    profile.global_loss_count += 1

                # Category-level tracking
                cat = self._infer_category(pos)
                if cat:
                    if cat not in profile.category_stats:
                        profile.category_stats[cat] = CategoryStats(category=cat)
                    profile.category_stats[cat].total_resolved += 1
                    if is_win:
                        profile.category_stats[cat].win_count += 1
                    else:
                        profile.category_stats[cat].loss_count += 1

    def _infer_category(self, position: dict) -> str:
        """
        Best-effort category inference from position data.

        The positions endpoint returns titles and slugs but not explicit
        categories. We use keyword heuristics on the title/slug.
        """
        title = (position.get("title", "") or "").lower()
        slug = (position.get("slug", "") or "").lower()
        event_slug = (position.get("eventSlug", "") or "").lower()
        text = f"{title} {slug} {event_slug}"

        # Political keywords
        political = [
            "president", "election", "senate", "congress", "governor",
            "democrat", "republican", "trump", "biden", "vote", "primary",
            "gop", "dnc", "rnc", "political", "cabinet", "impeach",
            "legislation", "bill", "act of congress", "supreme court nomination",
        ]
        if any(kw in text for kw in political):
            return "POLITICS"

        # Sports keywords
        sports = [
            "nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball",
            "baseball", "super bowl", "world cup", "champion", "finals",
            "match", "game", "playoff", "mvp", "scoring", "touchdown",
        ]
        if any(kw in text for kw in sports):
            return "SPORTS"

        # Crypto keywords
        crypto = [
            "bitcoin", "btc", "ethereum", "eth", "crypto", "token",
            "blockchain", "defi", "solana", "sol", "price above",
            "price below", "market cap",
        ]
        if any(kw in text for kw in crypto):
            return "CRYPTO"

        # Economics keywords
        economics = [
            "fed", "rate cut", "rate hike", "inflation", "cpi", "gdp",
            "unemployment", "recession", "interest rate", "tariff",
            "federal reserve", "treasury", "yield",
        ]
        if any(kw in text for kw in economics):
            return "ECONOMICS"

        # Pop culture
        pop_culture = [
            "oscar", "grammy", "emmy", "movie", "film", "album",
            "celebrity", "entertainment", "tv show", "netflix",
            "tiktok", "youtube", "streaming",
        ]
        if any(kw in text for kw in pop_culture):
            return "POP_CULTURE"

        # Science / Technology
        science_tech = [
            "ai", "artificial intelligence", "spacex", "nasa",
            "climate", "weather", "temperature", "vaccine",
            "fda", "drug approval", "research",
        ]
        if any(kw in text for kw in science_tech):
            return "SCIENCE"

        return ""

    def _compute_category_concentration(self, profile: WhaleProfile):
        """
        Compute how concentrated the whale is across categories.

        Uses a Herfindahl-like index on position counts.
        1.0 = all positions in one category, ~0.15 = evenly spread across 7.
        """
        counts = [
            s.position_count for s in profile.category_stats.values()
            if s.position_count > 0
        ]
        if not counts:
            profile.category_concentration = 0.0
            profile.primary_category = ""
            return

        total = sum(counts)
        shares = [c / total for c in counts]
        profile.category_concentration = sum(s ** 2 for s in shares)

        # Primary category = largest share
        max_cat = max(
            profile.category_stats.items(),
            key=lambda x: x[1].position_count,
        )
        profile.primary_category = max_cat[0]

    def _classify_strategy(self, profile: WhaleProfile):
        """
        Classify the whale's trading strategy from behavioral metrics.

        This is the core heuristic that determines how much to trust
        a whale's signal.
        """
        pos_count = profile.open_position_count
        vol_pnl = profile.vol_pnl_ratio
        concentration = profile.category_concentration
        avg_size = profile.avg_position_size

        # YES/NO balance — market makers tend to be more balanced
        total_sided = profile.yes_position_count + profile.no_position_count
        if total_sided > 0:
            yes_pct = profile.yes_position_count / total_sided
            side_balance = 1.0 - abs(yes_pct - 0.5) * 2  # 1.0 = perfectly balanced
        else:
            side_balance = 0.5

        # ─── Market Maker Detection ──────────────────────
        mm_score = 0.0
        if pos_count >= MARKET_MAKER_POSITION_THRESHOLD:
            mm_score += 0.4
        if vol_pnl != float('inf') and vol_pnl >= MARKET_MAKER_VOL_PNL_RATIO:
            mm_score += 0.3
        if side_balance > 0.7:  # Very balanced YES/NO
            mm_score += 0.3

        if mm_score >= 0.6:
            profile.strategy = "MARKET_MAKER"
            profile.strategy_confidence = min(1.0, mm_score)
            return

        # ─── Conviction Trader Detection ─────────────────
        if pos_count <= CONVICTION_MAX_POSITIONS and avg_size >= CONVICTION_MIN_AVG_SIZE:
            profile.strategy = "CONVICTION"
            profile.strategy_confidence = min(1.0, 0.5 + (avg_size / 5000) * 0.3 + (1 - pos_count / 30) * 0.2)
            return

        # ─── Specialist Detection ────────────────────────
        if concentration >= SPECIALIST_CONCENTRATION:
            profile.strategy = "SPECIALIST"
            profile.strategy_confidence = min(1.0, concentration)
            return

        # ─── Diversified ─────────────────────────────────
        if pos_count > CONVICTION_MAX_POSITIONS and concentration < SPECIALIST_CONCENTRATION:
            profile.strategy = "DIVERSIFIED"
            profile.strategy_confidence = 0.6
            return

        profile.strategy = "UNKNOWN"
        profile.strategy_confidence = 0.3

    def _compute_overall_credibility(self, profile: WhaleProfile):
        """Aggregate category credibilities into an overall score."""
        if not profile.category_stats:
            # Fall back to PnL-only
            if profile.global_pnl > 0:
                profile.overall_credibility = min(1.0, np.log1p(profile.global_pnl / 1000) / 5.0)
            return

        # Weighted average of category credibilities (by position count)
        total_weight = 0
        weighted_cred = 0
        for stats in profile.category_stats.values():
            w = max(1, stats.position_count)
            weighted_cred += stats.credibility * w
            total_weight += w

        if total_weight > 0:
            profile.overall_credibility = weighted_cred / total_weight

    # ─── Reporting ───────────────────────────────────────

    def report(self) -> str:
        """Generate a summary report of all whale profiles."""
        if not self._profiles:
            return "No whale profiles available. Run build_profiles() first."

        lines = [
            "=" * 85,
            "  WHALE PROFILER REPORT",
            "=" * 85,
            f"  Total profiles: {len(self._profiles)}",
            "",
        ]

        # Strategy breakdown
        strategies = {}
        for p in self._profiles.values():
            strategies[p.strategy] = strategies.get(p.strategy, 0) + 1

        lines.append("  Strategy Distribution:")
        for strat, count in sorted(strategies.items(), key=lambda x: -x[1]):
            lines.append(f"    {strat:15s} : {count}")
        lines.append("")

        # Top conviction traders (most useful signals)
        conviction = [
            p for p in self._profiles.values()
            if p.strategy in ("CONVICTION", "SPECIALIST") and p.signal_weight() > 0.2
        ]
        conviction.sort(key=lambda p: p.signal_weight(), reverse=True)

        if conviction:
            lines.append("  Top Signal Sources (conviction/specialist traders):")
            for p in conviction[:15]:
                wr = p.global_win_rate
                wr_str = f"{wr:.0%}" if wr is not None else "N/A"
                lines.append(
                    f"    {p.username or p.wallet[:12]:15s} | "
                    f"{p.strategy:12s} | "
                    f"Signal: {p.signal_weight():.2f} | "
                    f"PnL: ${p.global_pnl:>10,.0f} | "
                    f"WR: {wr_str:>4s} | "
                    f"Primary: {p.primary_category or 'N/A'}"
                )
            lines.append("")

        # Market makers (should be discounted)
        mms = [p for p in self._profiles.values() if p.strategy == "MARKET_MAKER"]
        if mms:
            lines.append(f"  Market Makers Identified (signal discounted): {len(mms)}")
            for p in mms[:5]:
                lines.append(
                    f"    {p.username or p.wallet[:12]:15s} | "
                    f"Positions: {p.open_position_count} | "
                    f"Vol/PnL: {p.vol_pnl_ratio:.0f}x | "
                    f"PnL: ${p.global_pnl:,.0f}"
                )
            lines.append("")

        lines.append("=" * 85)
        return "\n".join(lines)
