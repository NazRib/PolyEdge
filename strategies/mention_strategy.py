"""
Mention Market Strategy
Specialized strategy for Polymarket "mention" markets -- will person X
say word Y during a given time window?

Two market types supported:
  - Event-scoped: "Will Trump say X during [specific event]?"
  - Week-scoped:  "Will Trump say X this week?"

Edge thesis: LLMs can estimate word-mention probability better than
retail traders by combining linguistic base rates, agenda context,
and current news. Batch estimation across all words in an event
ensures internal consistency.

Pipeline: discover -> group -> context -> batch estimate -> edge detect -> size -> trade
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from core.api_client import PolymarketClient, Market
from core.kelly import kelly_criterion, PositionSize
from core.llm_providers import (
    call_llm, call_llm_with_search, validate_provider,
    model_tag_for_provider, PROVIDER_CLAUDE,
)
from core.paper_trader import PaperTrader
from core.pipeline_logger import PipelineLogger
from core.probability import ProbabilityEstimate

# Windows console encoding fix
import sys as _sys, os as _os
if _os.name == "nt" and hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)


# ===========================================================
# 1. DATA CLASSES
# ===========================================================

@dataclass
class MentionOutcome:
    """A single word/phrase outcome within a mention event."""
    word: str               # The word or phrase (e.g. "Fantastic", "NATO")
    market: Market          # The Polymarket market object for this outcome
    yes_price: float        # Current YES price (market-implied probability)
    no_price: float

    @property
    def market_prob(self) -> float:
        return self.yes_price


@dataclass
class MentionEvent:
    """
    A group of word-mention markets sharing the same event.

    For example, "What will Trump say during bilateral events with Rutte?"
    contains 30+ word outcomes, each a separate Polymarket market.
    """
    event_slug: str
    event_title: str
    scope: str              # "event" or "weekly"
    subject: str            # Person being tracked (e.g. "Trump")
    end_date: Optional[datetime]
    description: str        # Resolution rules from the event
    outcomes: list[MentionOutcome] = field(default_factory=list)

    @property
    def words(self) -> list[str]:
        return [o.word for o in self.outcomes]

    @property
    def n_outcomes(self) -> int:
        return len(self.outcomes)

    @property
    def total_volume(self) -> float:
        return sum(o.market.volume_total for o in self.outcomes)

    @property
    def days_to_resolution(self) -> Optional[float]:
        if self.end_date is None:
            return None
        delta = self.end_date - datetime.now(timezone.utc)
        return max(0, delta.total_seconds() / 86400)


# ===========================================================
# 2. MENTION SCANNER
# ===========================================================

# Patterns that identify a mention MARKET by its question text
_MENTION_Q_PATTERNS = [
    re.compile(r"what will .+ say", re.IGNORECASE),
    re.compile(r'will .+ say "', re.IGNORECASE),
    re.compile(r"will .+ mention", re.IGNORECASE),
    re.compile(r"what .+ will .+ mention", re.IGNORECASE),
    re.compile(r"what (?:words|terms|phrases) will", re.IGNORECASE),
    re.compile(r"what (?:places|countries|people) will .+ mention", re.IGNORECASE),
    re.compile(r"will .+ post ", re.IGNORECASE),       # "Will Trump post..."
    re.compile(r"what will be said", re.IGNORECASE),    # passive: "What will be said on..."
]

# Patterns that identify a mention EVENT by its title
# (broader than question patterns — also match event-level titles)
_MENTION_EVENT_PATTERNS = [
    re.compile(r"what will .+ say", re.IGNORECASE),
    re.compile(r"what will .+ post", re.IGNORECASE),
    re.compile(r"what will .+ mention", re.IGNORECASE),
    re.compile(r"what will be said", re.IGNORECASE),
    re.compile(r"what .+-named things will", re.IGNORECASE),
    re.compile(r"will .+ say .+ during", re.IGNORECASE),
    re.compile(r"will .+ mention .+ during", re.IGNORECASE),
]

# Generic category values that are NOT word labels
_GENERIC_CATEGORIES = frozenset({
    "", "politics", "mentions", "tweet markets", "sports", "crypto",
    "culture", "science", "world", "business", "tech",
})

# Known slug templates for recurring weekly mention markets.
# {month} and {day} are filled in based on upcoming end-of-week dates.
_WEEKLY_SLUG_TEMPLATES = [
    "what-will-trump-say-this-week-{month}-{day}",
    "what-will-trump-say-in-{month}",
]

# Month name lookup (lowercase, for slugs)
_MONTH_NAMES = [
    "", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


class MentionScanner:
    """
    Discovers mention events via two-phase approach:

    Phase 1 -- Pattern-based slug generation:
      Weekly mention markets follow predictable slugs like
      ``what-will-trump-say-this-week-april-12``.  We generate
      candidate slugs for the next few weeks and fetch them via
      ``GET /events/slug/{slug}`` (the one endpoint confirmed working).

    Phase 2 -- Market-question scanning:
      Fetch high-volume markets from ``/markets``, regex-filter for
      mention-like questions, collect unique ``eventSlug`` values,
      then fetch full events by slug.

    Both phases deduplicate on event slug.
    """

    def __init__(self, client: PolymarketClient = None):
        self.client = client or PolymarketClient()

    def scan(self) -> list[MentionEvent]:
        """
        Discover all active mention events with their outcomes.

        Returns:
            List of MentionEvent, sorted by days_to_resolution ascending.
        """
        seen_slugs: set[str] = set()
        events: list[MentionEvent] = []

        # Phase 1: pattern-based slug lookup
        phase1 = self._phase1_slug_generation()
        for slug, raw_ev in phase1:
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            event = self._parse_event(raw_ev)
            if event and event.n_outcomes >= 2:
                events.append(event)
                logger.info(f"  [slug] Found: {event.event_title[:50]} ({event.n_outcomes} words)")

        # Phase 2: market-question scanning
        phase2_slugs = self._phase2_event_scanning(seen_slugs)
        for slug in phase2_slugs:
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            try:
                raw_ev = self.client.get_event(slug)
            except Exception as e:
                logger.debug(f"  Event fetch failed for slug '{slug}': {e}")
                continue
            if not self._is_mention_event(raw_ev.get("title", "")):
                continue
            event = self._parse_event(raw_ev)
            if event and event.n_outcomes >= 2:
                events.append(event)
                logger.info(f"  [scan] Found: {event.event_title[:50]} ({event.n_outcomes} words)")

        logger.info(
            f"Found {len(events)} mention events total: "
            + ", ".join(f"{e.event_title[:35]}({e.n_outcomes}w)" for e in events)
        )

        events.sort(
            key=lambda e: e.days_to_resolution if e.days_to_resolution is not None else 999
        )
        return events

    # -- Phase 1: slug generation --------------------------

    def _phase1_slug_generation(self) -> list[tuple[str, dict]]:
        """
        Generate candidate slugs and try to fetch them.
        Returns list of (slug, raw_event_dict) for successful fetches.
        """
        from datetime import timedelta

        results: list[tuple[str, dict]] = []
        now = datetime.now(timezone.utc)
        current_month = _MONTH_NAMES[now.month]

        # Generate weekly slug candidates for the next 3 weekends
        candidate_slugs: list[str] = []

        # Monthly slug for the current month
        candidate_slugs.append(f"what-will-trump-say-in-{current_month}")
        candidate_slugs.append(f"what-trump-named-things-will-trump-mention-in-{current_month}")

        # Weekly slugs: try both Saturday and Sunday (Polymarket weeks
        # sometimes end on either day depending on the series)
        for offset in range(-7, 22):
            d = now + timedelta(days=offset)
            if d.weekday() in (5, 6):  # Saturday or Sunday
                month_name = _MONTH_NAMES[d.month]
                candidate_slugs.append(
                    f"what-will-trump-say-this-week-{month_name}-{d.day}")
                candidate_slugs.append(
                    f"what-will-trump-post-this-week-{month_name}-{d.day}-{month_name}-{d.day + 7 if d.day + 7 <= 28 else d.day}")

        # Recurring non-Trump series (best-effort slug guesses)
        candidate_slugs.extend([
            "what-will-karoline-leavitt-say-during-the-next-white-house-press-briefing",
            "what-will-keir-starmer-say-at-the-next-prime-ministers-questions-event",
            "what-will-powell-say-during-april-press-conference",
        ])

        logger.info(f"  Phase 1: trying {len(candidate_slugs)} candidate slugs...")

        for slug in candidate_slugs:
            try:
                raw_ev = self.client.get_event(slug)
                if raw_ev and raw_ev.get("markets"):
                    results.append((slug, raw_ev))
            except Exception:
                pass  # slug doesn't exist, that's fine

        logger.info(f"  Phase 1: {len(results)} events found via slug patterns")
        return results

    # -- Phase 2: event volume scanning ---------------------

    def _phase2_event_scanning(self, already_found: set[str]) -> list[str]:
        """
        Scan events ordered by volume to find mention events.

        Individual mention sub-markets have low per-word volume (~$3K),
        so they don't appear in the top markets-by-volume listing.
        But the parent EVENT has high aggregate volume ($100K+), so
        scanning events by volume surfaces them reliably.

        Also falls back to scanning individual markets for any mention
        questions whose event_slug we haven't seen yet.
        """
        event_slugs: set[str] = set()

        # Strategy A: Scan events ordered by volume
        for page in range(10):  # up to 500 events
            try:
                raw_events = self.client.get_events_list(
                    limit=50, offset=page * 50, closed=False,
                    order="volume24hr", ascending=False,
                )
            except Exception as e:
                logger.warning(f"  Phase 2 event fetch page {page} failed: {e}")
                break

            if not raw_events:
                break

            for ev in raw_events:
                title = ev.get("title", "")
                slug = ev.get("slug", "")
                if not slug or slug in already_found or slug in event_slugs:
                    continue
                if self._is_mention_event(title):
                    n_markets = len(ev.get("markets", []))
                    event_slugs.add(slug)
                    logger.debug(
                        f"  Phase 2 found: '{title[:50]}' ({n_markets} mkts)"
                    )

        # Strategy B: Also scan high-volume individual markets as fallback
        for page in range(3):
            try:
                markets = self.client.get_active_markets(
                    limit=100, offset=page * 100, order="volume24hr",
                )
            except Exception as e:
                logger.warning(f"  Phase 2 market scan page {page} failed: {e}")
                break

            if not markets:
                break

            for m in markets:
                if not self._is_mention_question(m.question):
                    continue
                slug = m.event_slug
                if slug and slug not in already_found and slug not in event_slugs:
                    event_slugs.add(slug)
                    logger.debug(
                        f"  Phase 2 market hit: slug={slug} q={m.question[:50]}"
                    )

        logger.info(f"  Phase 2: {len(event_slugs)} new event slugs discovered")
        return list(event_slugs)

    @staticmethod
    def _is_mention_question(question: str) -> bool:
        """Does this market question look like a mention market?"""
        for pat in _MENTION_Q_PATTERNS:
            if pat.search(question):
                return True
        return False

    @staticmethod
    def _is_mention_event(title: str) -> bool:
        """Does this event title look like a mention event?"""
        for pat in _MENTION_EVENT_PATTERNS:
            if pat.search(title):
                return True
        return False

    # -- Event parsing (shared) ----------------------------

    def _parse_event(self, raw_ev: dict) -> Optional[MentionEvent]:
        """Parse a raw event dict (with nested markets) into a MentionEvent."""
        title = raw_ev.get("title", "")
        slug = raw_ev.get("slug", "")
        description = raw_ev.get("description", "")
        raw_markets = raw_ev.get("markets", [])

        if not raw_markets:
            return None

        scope = self._classify_scope(title)
        subject = self._extract_subject(title)

        # Parse end date
        end_date = None
        end_str = raw_ev.get("endDate") or raw_ev.get("end_date_iso")
        if not end_str and raw_markets:
            end_str = raw_markets[0].get("endDate") or raw_markets[0].get("end_date_iso")
        if end_str:
            try:
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        event = MentionEvent(
            event_slug=slug,
            event_title=title,
            scope=scope,
            subject=subject,
            end_date=end_date,
            description=(description or "")[:1500],
        )

        for raw_m in raw_markets:
            if raw_m.get("closed") or not raw_m.get("active", True):
                continue

            word = self._extract_word_from_raw(raw_m)
            if not word:
                continue

            try:
                market = self.client._parse_market(raw_m)
            except Exception as e:
                logger.debug(f"Failed to parse sub-market: {e}")
                continue

            event.outcomes.append(MentionOutcome(
                word=word,
                market=market,
                yes_price=market.yes_price,
                no_price=market.no_price,
            ))

        return event

    @staticmethod
    def _classify_scope(title: str) -> str:
        """Classify as 'event' (single speech/meeting) or 'weekly'."""
        t = title.lower()
        if any(kw in t for kw in ["this week", "this month", "in march", "in april",
                                   "in may", "in june", "in july", "in august",
                                   "in september", "in october", "in november",
                                   "in december", "in january", "in february"]):
            return "weekly"
        if any(kw in t for kw in ["during", "bilateral", "address", "speech",
                                   "conference", "summit", "rally", "event"]):
            return "event"
        return "weekly"

    @staticmethod
    def _extract_subject(title: str) -> str:
        """Extract who is being tracked from the event title."""
        t = title.lower()
        # Check known subjects (order matters: more specific first)
        known = [
            ("bernie sanders", "Bernie Sanders"),
            ("karoline leavitt", "Karoline Leavitt"),
            ("keir starmer", "Keir Starmer"),
            ("melania", "Melania Trump"),
            ("mrbeast", "MrBeast"),
            ("powell", "Jerome Powell"),
            ("trump", "Trump"),          # before King Charles — Trump is the speaker
            ("biden", "Biden"),
            ("musk", "Elon Musk"),
            ("elon", "Elon Musk"),
            ("rutte", "Mark Rutte"),
            ("vance", "JD Vance"),
            ("king charles", "King Charles"),
        ]
        for keyword, name in known:
            if keyword in t:
                return name
        # Passive voice: "What will be said on the All-In Podcast"
        if "all-in podcast" in t or "all in podcast" in t:
            return "All-In Podcast hosts"
        if "will be said" in t:
            # Try to extract the venue/show name
            m = re.search(r"said (?:on|during|at) (?:the )?(.+?)(?:\?|$)", title, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return "Unknown"

    @staticmethod
    def _extract_word_from_raw(raw_market: dict) -> Optional[str]:
        """
        Extract the target word/phrase from a raw market dict.

        Polymarket uses ``groupItemTitle`` for the word label within
        a multi-market event.  Falls back to ``category`` and then
        regex on question.
        """
        # Primary: groupItemTitle
        git = (raw_market.get("groupItemTitle") or "").strip()
        if git and git.lower() not in _GENERIC_CATEGORIES:
            return git

        # Secondary: category field
        cat = (raw_market.get("category") or "").strip()
        if cat and cat.lower() not in _GENERIC_CATEGORIES:
            return cat

        # Tertiary: quoted word in question
        question = raw_market.get("question", "")
        m = re.search(r'"(.+?)"', question)
        if m:
            return m.group(1).strip()

        return None


# ===========================================================
# 3. CONTEXT BUILDER (web-search grounded)
# ===========================================================

_CONTEXT_SEARCH_PROMPT = """You are a research assistant gathering context for a prediction market question.

EVENT: {event_title}
SCOPE: {scope}
SUBJECT: {subject}
RESOLUTION WINDOW: ends {end_date}

I need you to find:
1. What is the agenda / topic for this event? Who are the participants?
2. What are the top 3-5 current news topics involving {subject} right now?
3. Is there any publicly available schedule of {subject}'s upcoming appearances?
4. Any recent speeches or statements by {subject} in the last 48 hours?
5. CRITICAL -- BROADCAST FORMAT: Will there be live broadcast or streamed
   remarks at this event? Specifically:
   - Is this an open press conference, public speech, or rally? (live broadcast likely)
   - Is this a closed-door meeting with only a brief pool spray? (very limited remarks)
   - Is this a private meeting with no planned public remarks? (no broadcast)
   - Has the White House or organizers announced a press availability?
   - Will there be a joint press conference or just a photo op?

Return a concise factual summary (no speculation). Focus on what will
likely be discussed and what language/topics are top of mind. Be very
specific about the broadcast/media format -- this is critical for the
prediction market since only LIVE BROADCAST remarks count for resolution.
"""


class MentionContextBuilder:
    """
    Gathers event context using web-search-grounded LLM calls.

    This reuses the existing `call_llm_with_search` from llm_providers,
    which gives the LLM access to web search for grounding.
    """

    def __init__(
        self,
        llm_provider: str = PROVIDER_CLAUDE,
        api_key: Optional[str] = None,
    ):
        self.provider = validate_provider(llm_provider)
        self.api_key = api_key

    def build_context(self, event: MentionEvent) -> str:
        """
        Gather web-search-grounded context for a mention event.

        Returns a text summary suitable for injection into the
        forecasting prompt.
        """
        prompt = _CONTEXT_SEARCH_PROMPT.format(
            event_title=event.event_title,
            scope=event.scope,
            subject=event.subject,
            end_date=event.end_date.strftime("%Y-%m-%d %H:%M ET") if event.end_date else "unknown",
        )

        logger.info(f"  [news] Gathering context for: {event.event_title[:50]}...")
        result = call_llm_with_search(
            user_prompt=prompt,
            provider=self.provider,
            api_key=self.api_key,
        )

        if not result:
            logger.warning("Context search returned empty -- proceeding without context")
            return "(No additional context available)"

        return result


# ===========================================================
# 4. BATCH LLM ESTIMATOR
# ===========================================================

_MENTION_SYSTEM_PROMPT = """You are an expert linguistic forecaster specializing in predicting
what words and phrases public figures will use in speeches and public appearances.

You have deep knowledge of:
- Rhetorical patterns and verbal tics of politicians
- How meeting agendas and current events drive word choice
- Base rates: how often common filler words appear in any speech
- The difference between scripted remarks and off-the-cuff comments
- EVENT BROADCAST FORMATS and how they affect resolution probability

CRITICAL FIRST STEP -- BROADCAST FORMAT ASSESSMENT:
Before estimating ANY word probabilities, you MUST assess whether the event
will produce qualifying remarks. Many prediction markets on mention events
resolve based ONLY on live broadcast or streamed remarks. This means:

- Closed-door meetings with NO live coverage -> ALL words resolve NO
  regardless of what is said behind closed doors
- Brief pool sprays (2-3 min photo op) -> very limited remarks, only
  the most obvious words have a chance, and even those are low probability
- Open press conferences / joint pressers -> full remarks likely,
  normal word estimation applies
- Public speeches / rallies -> extended remarks, high base rates
- Written statements (Truth Social, press releases) -> typically do NOT
  count for event-scoped markets (check resolution rules carefully)

If the event is likely a closed-door meeting with no public remarks,
set ALL probabilities to 1-5% regardless of topic relevance.

If the format is uncertain, discount all probabilities by 40-60%.

CALIBRATION GUIDELINES (for events WITH confirmed live broadcast):
- Words directly on-topic for the event/agenda: 70-95%
- Common verbal tics the person uses in almost every appearance
  (e.g. Trump saying "fantastic", "tremendous", "beautiful"): 80-95%
  for weekly markets, 60-85% for single-event markets
- Words tangentially related to current news: 30-60%
- Obscure or unusual words with no clear contextual hook: 3-15%
- Words the person actively avoids or has no reason to use: 2-8%
- "N+ times" markets: discount by ~15-30% vs single-mention probability
  depending on N

CALIBRATION GUIDELINES (for weekly/monthly markets):
- Weekly markets cover ALL public verbal statements in the window, so
  broadcast format is less of a concern -- there will almost certainly
  be multiple public appearances during a week
- Focus more on topic relevance and verbal habits for these

Think carefully about each word. Consider:
1. FIRST: Will there be live broadcast remarks at this event?
2. Is this word on the meeting agenda or directly related to it?
3. Does this person use this word habitually?
4. Is this word in the current news cycle?
5. Could this word come up in a Q&A even if not in prepared remarks?
6. Are there synonyms the person might use instead?
"""

_MENTION_ESTIMATE_PROMPT = """CONTEXT:
{context}

EVENT: {event_title}
SCOPE: {scope} ({scope_description})
SUBJECT: {subject}
WINDOW: ends {end_date}
RESOLUTION RULES SUMMARY: {rules_summary}

CURRENT MARKET PRICES (for reference -- these are what other traders think):
{price_table}

TASK: Estimate the probability that {subject} will say each word below
during this window. You MUST follow this two-step process:

STEP 1 -- BROADCAST FORMAT ASSESSMENT:
Before estimating any word probabilities, classify the event format:
- "open_broadcast": public speech, press conference, rally with live coverage
- "limited_remarks": brief pool spray or photo op with a few minutes of remarks
- "closed_door": private meeting with no planned live broadcast
- "weekly": weekly/monthly market covering all public appearances (always has coverage)

If "closed_door": set ALL probabilities to 0.01-0.05 regardless of topic.
If "limited_remarks": cap most probabilities at 0.15-0.30, only the most
  obvious agenda words might reach 0.40-0.50.
If "open_broadcast" or "weekly": estimate normally using calibration guidelines.

STEP 2 -- WORD-LEVEL ESTIMATION:
For each word, estimate probability given the broadcast format from Step 1.

WORDS TO ESTIMATE:
{word_list}

Respond with ONLY a JSON object in this exact format (no markdown, no backticks):
{{
  "broadcast_format": "open_broadcast|limited_remarks|closed_door|weekly",
  "broadcast_reasoning": "1-2 sentence explanation of format classification",
  "estimates": [
    {{
      "word": "the word",
      "probability": 0.XX,
      "confidence": 0.X,
      "reasoning": "brief 1-sentence explanation"
    }}
  ]
}}
"""


class MentionEstimator:
    """
    Batch LLM estimator for mention markets.

    Sends all words in a single event to the LLM at once for:
    - Cost efficiency (1 call per event, not per word)
    - Internal consistency (probabilities should be coherent across words)
    """

    def __init__(
        self,
        llm_provider: str = PROVIDER_CLAUDE,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.provider = validate_provider(llm_provider)
        self.api_key = api_key
        self.model = model

    def estimate_batch(
        self,
        event: MentionEvent,
        context: str,
    ) -> dict[str, dict]:
        """
        Estimate mention probabilities for all words in an event.

        Args:
            event: The MentionEvent with all its outcomes
            context: Web-search-grounded context string

        Returns:
            Dict mapping word -> {"probability": float, "confidence": float,
                                   "reasoning": str}
        """
        # Build price table for the prompt
        price_lines = []
        for o in event.outcomes:
            price_lines.append(f"  {o.word}: YES={o.yes_price:.0%}, NO={o.no_price:.0%}")
        price_table = "\n".join(price_lines)

        word_list = "\n".join(f"  - {o.word}" for o in event.outcomes)

        # Truncate rules to keep prompt focused
        rules_summary = event.description[:600]
        if len(event.description) > 600:
            rules_summary += "..."

        scope_desc = (
            "single event/appearance" if event.scope == "event"
            else "any public verbal statement during the week"
        )

        user_prompt = _MENTION_ESTIMATE_PROMPT.format(
            context=context,
            event_title=event.event_title,
            scope=event.scope,
            scope_description=scope_desc,
            subject=event.subject,
            end_date=event.end_date.strftime("%Y-%m-%d %H:%M ET") if event.end_date else "unknown",
            rules_summary=rules_summary,
            price_table=price_table,
            word_list=word_list,
        )

        logger.info(
            f"  [LLM] Estimating {event.n_outcomes} words for: {event.event_title[:50]}..."
        )

        raw = call_llm(
            user_prompt=user_prompt,
            system_prompt=_MENTION_SYSTEM_PROMPT,
            provider=self.provider,
            model=self.model,
            max_tokens=2500,
            temperature=0.2,
            api_key=self.api_key,
        )

        if not raw:
            logger.error("LLM returned empty response for batch estimation")
            return {}, None

        return self._parse_batch_response(raw, event)

    def _parse_batch_response(
        self, raw: str, event: MentionEvent
    ) -> tuple[dict[str, dict], Optional[dict]]:
        """Parse the JSON response from the LLM."""
        # Strip markdown fences if present
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM JSON response:\n{raw[:500]}")
            return {}, None

        # Extract broadcast assessment
        broadcast_info = None
        bfmt = data.get("broadcast_format")
        if bfmt:
            broadcast_info = {
                "format": bfmt,
                "reasoning": data.get("broadcast_reasoning", ""),
            }
            logger.info(f"  [broadcast] Format: {bfmt} -- {broadcast_info['reasoning'][:80]}")

        estimates = data.get("estimates", [])
        result: dict[str, dict] = {}

        # Build a lookup by lowercased word for fuzzy matching
        word_set = {o.word.lower(): o.word for o in event.outcomes}

        for est in estimates:
            word_raw = est.get("word", "")
            prob = float(est.get("probability", 0.5))
            conf = float(est.get("confidence", 0.5))
            reasoning = est.get("reasoning", "")

            # Clamp probability
            prob = max(0.01, min(0.99, prob))
            conf = max(0.1, min(1.0, conf))

            # Match back to our outcome word
            matched_word = word_set.get(word_raw.lower())
            if not matched_word:
                # Try partial match
                for key, val in word_set.items():
                    if key in word_raw.lower() or word_raw.lower() in key:
                        matched_word = val
                        break

            if matched_word:
                result[matched_word] = {
                    "probability": prob,
                    "confidence": conf,
                    "reasoning": reasoning,
                }

        # Log coverage
        n_matched = len(result)
        n_total = event.n_outcomes
        if n_matched < n_total:
            missing = set(event.words) - set(result.keys())
            logger.warning(
                f"LLM only estimated {n_matched}/{n_total} words. "
                f"Missing: {list(missing)[:5]}"
            )

        return result, broadcast_info


# ===========================================================
# 5. MENTION STRATEGY (Full Pipeline)
# ===========================================================

class MentionStrategy:
    """
    Full pipeline for mention markets:
    discover -> group -> context -> batch estimate -> edge detect -> size -> trade

    Correlation-aware: caps total dollar exposure per event since
    all words in the same event share cancellation risk.
    """

    def __init__(
        self,
        client: PolymarketClient = None,
        trader: PaperTrader = None,
        llm_provider: str = PROVIDER_CLAUDE,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        # Sizing params
        kelly_fraction: float = 0.20,
        min_edge: float = 0.05,
        max_event_exposure_pct: float = 0.25,
        min_liquidity: float = 500,
        # Filtering
        max_days_to_resolution: float = 14,
    ):
        self.client = client or PolymarketClient()
        self.scanner = MentionScanner(self.client)
        self.context_builder = MentionContextBuilder(
            llm_provider=llm_provider, api_key=api_key,
        )
        self.estimator = MentionEstimator(
            llm_provider=llm_provider, api_key=api_key, model=model,
        )
        self.trader = trader or PaperTrader(
            bankroll=1000.0,
            data_dir="data/paper_mentions",
        )
        self.llm_provider = llm_provider
        self.model_tag = model_tag_for_provider(validate_provider(llm_provider))
        self.strategy_tag = "mentions"
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_event_exposure_pct = max_event_exposure_pct
        self.min_liquidity = min_liquidity
        self.max_days_to_resolution = max_days_to_resolution

    def run(self) -> list[dict]:
        """Run the full mention market pipeline."""

        print("\n" + "=" * 70)
        print("  POLYMARKET EDGE -- Mention Market Strategy")
        print(f"  Model: {self.model_tag}  |  Kelly: {self.kelly_fraction}")
        print("=" * 70)

        # Step 1: Discover
        print("\n>>> Step 1: Scanning for mention markets...")
        t0 = time.time()
        events = self.scanner.scan()
        print(f"   Found {len(events)} mention events ({time.time()-t0:.1f}s)")

        # Filter by resolution window and minimum outcomes
        events = [
            e for e in events
            if (e.days_to_resolution is not None
                and e.days_to_resolution <= self.max_days_to_resolution
                and e.n_outcomes >= 3)
        ]
        print(f"   After filtering: {len(events)} events within {self.max_days_to_resolution}d")

        if not events:
            print("   No actionable mention events found.")
            return []

        all_signals = []
        existing_ids = {t.market_id for t in self.trader.trades if t.status == "OPEN"}

        for event in events:
            signals = self._process_event(event, existing_ids)
            all_signals.extend(signals)

        # Summary
        trades_entered = [s for s in all_signals if s.get("trade_result") == "entered"]
        trades_skipped = [s for s in all_signals if s.get("should_trade") and s.get("trade_result") != "entered"]

        print("\n" + "=" * 70)
        print(f"  SUMMARY: {len(trades_entered)} trades entered, "
              f"{len(trades_skipped)} sized but capped")
        print(f"  Bankroll: ${self.trader.bankroll:.2f}")
        print("=" * 70)

        return all_signals

    def _process_event(
        self, event: MentionEvent, existing_ids: set[str]
    ) -> list[dict]:
        """Process a single mention event: context -> estimate -> size -> trade."""

        print(f"\n{'-'*60}")
        print(f"  -- {event.event_title[:60]}")
        print(f"     Scope: {event.scope} | Words: {event.n_outcomes} | "
              f"Days left: {event.days_to_resolution:.1f}")
        print(f"{'-'*60}")

        # Step 2: Context
        context = self.context_builder.build_context(event)

        # Step 3: Batch estimate
        time.sleep(1.5)  # throttle between LLM calls
        estimates, broadcast_info = self.estimator.estimate_batch(event, context)
        if not estimates:
            print("   [!]  No estimates returned -- skipping event")
            return []

        if broadcast_info:
            fmt = broadcast_info["format"]
            reason = broadcast_info["reasoning"][:60]
            print(f"   [broadcast] {fmt} -- {reason}")

        # Step 4: Edge detection + sizing with correlation cap
        max_event_dollars = self.trader.bankroll * self.max_event_exposure_pct
        event_dollars_used = 0.0
        signals = []

        # Sort outcomes by absolute edge (biggest edge first)
        outcomes_with_est = []
        for outcome in event.outcomes:
            est = estimates.get(outcome.word)
            if not est:
                continue
            edge = est["probability"] - outcome.market_prob
            outcomes_with_est.append((outcome, est, abs(edge)))

        outcomes_with_est.sort(key=lambda x: x[2], reverse=True)

        for outcome, est, abs_edge in outcomes_with_est:
            word = outcome.word
            prob = est["probability"]
            conf = est["confidence"]
            market_price = outcome.market_prob
            edge = prob - market_price

            # Skip if already holding this market
            if outcome.market.id in existing_ids:
                print(f"   [ ] {word}: skip (open trade exists)")
                continue

            # Skip low-liquidity outcomes
            if outcome.market.liquidity < self.min_liquidity:
                continue

            # Size via Kelly
            position = kelly_criterion(
                estimated_prob=prob,
                market_price=market_price,
                bankroll=self.trader.bankroll,
                kelly_fraction=self.kelly_fraction,
                min_edge=self.min_edge,
                confidence=conf,
            )

            signal = {
                "event_slug": event.event_slug,
                "event_title": event.event_title,
                "word": word,
                "scope": event.scope,
                "market": outcome.market,
                "our_prob": prob,
                "market_prob": market_price,
                "edge": edge,
                "confidence": conf,
                "reasoning": est["reasoning"],
                "position": position,
                "should_trade": position.should_trade,
                "trade_result": None,
            }

            if not position.should_trade:
                emoji = "[.]"
                print(f"   {emoji} {word}: P={prob:.0%} vs Mkt={market_price:.0%} "
                      f"edge={edge:+.0%} -- no trade ({position.rejection_reason})")
                signals.append(signal)
                continue

            # Correlation cap: don't exceed max exposure per event
            if event_dollars_used + position.dollar_amount > max_event_dollars:
                remaining = max_event_dollars - event_dollars_used
                if remaining < 5.0:
                    signal["trade_result"] = "event_cap"
                    print(f"   [!] {word}: P={prob:.0%} edge={edge:+.0%} "
                          f"-- event exposure cap reached")
                    signals.append(signal)
                    continue

            # Step 5: Paper trade
            estimate_obj = ProbabilityEstimate(
                market_id=outcome.market.id,
                question=f"[Mention] {word} -- {event.event_title[:40]}",
                probability=prob,
                market_price=market_price,
                edge=edge,
                confidence=conf,
                reasoning=est["reasoning"],
            )

            trade = self.trader.enter_trade(
                estimate=estimate_obj,
                position=position,
                strategy_tag=self.strategy_tag,
                model_tag=self.model_tag,
            )

            if trade:
                event_dollars_used += position.dollar_amount
                signal["trade_result"] = "entered"
                print(f"   [+] {word}: {position.side} ${position.dollar_amount:.2f} "
                      f"| P={prob:.0%} vs Mkt={market_price:.0%} edge={edge:+.0%}")
            else:
                signal["trade_result"] = "rejected"

            signals.append(signal)

        return signals

    def dry_run(self) -> list[dict]:
        """
        Run the full pipeline but don't enter trades.
        Useful for evaluating the strategy before committing capital.

        Returns list of all signals with estimates and sizing.
        """
        print("\n" + "=" * 70)
        print("  MENTION STRATEGY -- DRY RUN (no trades)")
        print("=" * 70)

        events = self.scanner.scan()
        events = [
            e for e in events
            if (e.days_to_resolution is not None
                and e.days_to_resolution <= self.max_days_to_resolution
                and e.n_outcomes >= 3)
        ]

        if not events:
            print("No actionable mention events found.")
            return []

        all_signals = []

        for event in events:
            print(f"\n{'-'*60}")
            print(f"  -- {event.event_title[:60]}")
            print(f"     Scope: {event.scope} | Words: {event.n_outcomes}")
            print(f"{'-'*60}")

            context = self.context_builder.build_context(event)
            time.sleep(1.5)
            estimates, broadcast_info = self.estimator.estimate_batch(event, context)

            if not estimates:
                print("   [!]  No estimates")
                continue

            if broadcast_info:
                fmt = broadcast_info["format"]
                reason = broadcast_info["reasoning"][:80]
                print(f"\n   [BROADCAST] {fmt}")
                print(f"   {reason}")

            # Print all estimates vs market
            print(f"\n   {'Word':<25} {'Our P':>6} {'Mkt P':>6} {'Edge':>7} {'Conf':>5}  Reasoning")
            print(f"   {'-'*85}")

            for outcome in event.outcomes:
                est = estimates.get(outcome.word)
                if not est:
                    print(f"   {outcome.word:<25} {'?':>6} {outcome.market_prob:>5.0%} {'?':>7} {'?':>5}")
                    continue

                prob = est["probability"]
                edge = prob - outcome.market_prob
                conf = est["confidence"]
                reason = est["reasoning"][:50]

                # Highlight actionable edges
                if abs(edge) >= self.min_edge:
                    marker = "[+]" if edge > 0 else "[-]"
                else:
                    marker = "  "

                print(f" {marker} {outcome.word:<25} {prob:>5.0%} {outcome.market_prob:>5.0%} "
                      f"{edge:>+6.0%} {conf:>5.1f}  {reason}")

                all_signals.append({
                    "event_title": event.event_title,
                    "word": outcome.word,
                    "scope": event.scope,
                    "our_prob": prob,
                    "market_prob": outcome.market_prob,
                    "edge": edge,
                    "confidence": conf,
                    "reasoning": est["reasoning"],
                })

        return all_signals


# ===========================================================
# 6. CLI ENTRY POINT
# ===========================================================

def run_mention_pipeline(dry_run: bool = True, **kwargs) -> list[dict]:
    """
    Entry point for running the mention strategy.

    Args:
        dry_run: If True, scan and estimate but don't enter trades.
        **kwargs: Passed to MentionStrategy constructor.

    Usage:
        # Dry run -- just see what the model thinks
        python -c "from strategies.mention_strategy import run_mention_pipeline; run_mention_pipeline()"

        # Live paper trading
        python -c "from strategies.mention_strategy import run_mention_pipeline; run_mention_pipeline(dry_run=False)"
    """
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    strategy = MentionStrategy(api_key=api_key, **kwargs)

    if dry_run:
        return strategy.dry_run()
    return strategy.run()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    dry = "--live" not in sys.argv
    run_mention_pipeline(dry_run=dry)