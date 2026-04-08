"""
Cross-Event Logical Dependency Detector (Phase A)
Finds pricing inconsistencies across markets in *different* events that
are logically related.

Three dependency types detected:
    1. Implication   — A=YES implies B=YES  →  P(A) <= P(B)
    2. Mutual Exclusion — A and B can't both be YES  →  P(A) + P(B) <= 1.0
    3. Subsumption   — A is a stricter version of B (date/scope)  →  P(A) <= P(B)

Phase A uses keyword overlap + heuristic rules (no LLM calls).
Phase B will add LLM-based relationship classification for ambiguous pairs.

Usage:
    python -m strategies.cross_event_arb              # scan live markets
    python -m strategies.cross_event_arb --min-edge 3 # 3% threshold
    python -m strategies.cross_event_arb --top 500    # scan more markets
    python -m strategies.cross_event_arb --verbose     # show all pairs, not just violations
"""

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from core.api_client import PolymarketClient, Market

logger = logging.getLogger(__name__)

# ─── Stop words for keyword extraction ───────────────────

STOP_WORDS = {
    "will", "the", "a", "an", "in", "on", "by", "be", "is", "are", "was",
    "this", "that", "to", "of", "for", "with", "have", "has", "does", "do",
    "did", "before", "after", "during", "than", "more", "less", "most",
    "what", "when", "where", "how", "who", "which", "would", "could",
    "should", "can", "may", "might", "or", "and", "but", "not", "if",
    "at", "it", "its", "they", "their", "from", "above", "below", "over",
    "under", "between", "any", "each", "all", "both", "other", "such",
    "no", "yes", "up", "down", "out", "into", "about", "then", "so",
    "just", "also", "new", "first", "last", "next", "end", "get",
    "been", "being", "were", "had", "having", "going", "go", "come",
}


# ─── Data Classes ────────────────────────────────────────

@dataclass
class MarketEntity:
    """A market with extracted keywords and entities for pairing."""
    market: Market
    keywords: set[str] = field(default_factory=set)
    entities: list[str] = field(default_factory=list)   # proper nouns, orgs
    numbers: list[str] = field(default_factory=list)     # percentages, amounts
    dates: list[str] = field(default_factory=list)       # date references
    scope_tokens: list[str] = field(default_factory=list)  # "by June", "in 2026"


class DependencyType:
    IMPLICATION = "implication"          # A → B
    MUTUAL_EXCLUSION = "mutual_exclusion"  # ¬(A ∧ B)
    SUBSUMPTION = "subsumption"          # A ⊂ B (date/scope)
    CORRELATED = "correlated"            # related but no strict logical link
    INDEPENDENT = "independent"


@dataclass
class DependencyPair:
    """A detected logical relationship between two markets."""
    market_a: Market
    market_b: Market
    dep_type: str                        # DependencyType value
    direction: str                       # "a_implies_b", "b_implies_a", "symmetric"
    confidence: float                    # 0-1, how certain the heuristic is
    reason: str                          # human-readable explanation
    keyword_overlap: int = 0
    entity_overlap: int = 0


@dataclass
class PriceViolation:
    """A pricing inconsistency in a dependency pair."""
    pair: DependencyPair
    violation_magnitude: float           # how far off prices are (0-1 scale)
    expected_constraint: str             # e.g. "P(A) <= P(B)"
    actual_prices: str                   # e.g. "P(A)=0.65, P(B)=0.55"
    implied_edge_pct: float              # % edge available
    suggested_action: str                # e.g. "BUY B or SELL A"


# ─── Entity & Keyword Extraction ─────────────────────────

def extract_entities(question: str) -> MarketEntity:
    """
    Extract keywords, named entities, numbers, and date references
    from a market question. No NLP library needed — pattern-based.
    """
    me = MarketEntity(market=None)  # market set by caller

    # Keywords: lowercase, remove stop words
    words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
    me.keywords = words - STOP_WORDS

    # Named entities: capitalized words/phrases (2+ chars), excluding sentence starts
    # and date-related words (months get extracted separately as dates)
    _DATE_WORDS = {
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct",
        "Nov", "Dec", "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    }
    sentences = re.split(r'[.?!]\s*', question)
    for sent in sentences:
        tokens = sent.strip().split()
        # Check all tokens except the first (sentence start)
        candidates = tokens[1:] if len(tokens) > 1 else []
        for tok in candidates:
            clean = re.sub(r'[^a-zA-Z]', '', tok)
            if (clean and clean[0].isupper() and len(clean) >= 2
                    and clean not in _DATE_WORDS):
                me.entities.append(clean)
    # Also grab multi-word proper nouns: "Federal Reserve", "Donald Trump"
    multi_ents = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question)
    me.entities.extend(multi_ents)
    # Acronyms: FBI, GDP, CPI, NATO, etc.
    acronyms = re.findall(r'\b([A-Z]{2,6})\b', question)
    me.entities.extend(acronyms)
    me.entities = list(set(me.entities))

    # Numbers & percentages
    me.numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', question)

    # Date references: months, years, "by June", "in Q3", "before 2026"
    date_patterns = re.findall(
        r'\b(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
        r'Q[1-4]|H[12]|20\d{2})\b',
        question, re.IGNORECASE,
    )
    me.dates = [d.lower() for d in date_patterns]

    # Scope tokens: "by <date>", "before <date>", "in <year>"
    scope = re.findall(
        r'\b(by|before|after|in|during)\s+'
        r'(January|February|March|April|May|June|July|August|'
        r'September|October|November|December|'
        r'Q[1-4]|H[12]|20\d{2})\b',
        question, re.IGNORECASE,
    )
    me.scope_tokens = [f"{s[0].lower()} {s[1].lower()}" for s in scope]

    return me


# ─── Candidate Pairing ──────────────────────────────────

def compute_pair_score(a: MarketEntity, b: MarketEntity) -> tuple[int, int]:
    """
    Score how related two markets are by keyword and entity overlap.
    Returns (keyword_overlap, entity_overlap).
    """
    kw_overlap = len(a.keywords & b.keywords)

    # Entity overlap: case-insensitive
    a_ents_lower = {e.lower() for e in a.entities}
    b_ents_lower = {e.lower() for e in b.entities}
    ent_overlap = len(a_ents_lower & b_ents_lower)

    return kw_overlap, ent_overlap


def _question_similarity(q_a: str, q_b: str) -> float:
    """
    Word-level Jaccard similarity between two questions.
    Used to detect near-duplicate markets across different event_slugs.

    Returns 0.0 if the only differences between questions are date/time
    words — those are subsumption candidates, not duplicates.
    """
    words_a = set(q_a.lower().split())
    words_b = set(q_b.lower().split())
    if not words_a or not words_b:
        return 0.0

    jaccard = len(words_a & words_b) / len(words_a | words_b)

    # If high similarity, check if the differing words are all date-related.
    # If so, return 0.0 to let subsumption detection handle them.
    if jaccard >= 0.80:
        diff_words = (words_a ^ words_b)  # symmetric difference
        date_words = set()
        for w in diff_words:
            w_clean = re.sub(r'[^a-z0-9]', '', w)
            if (w_clean in MONTH_ORDER
                    or re.match(r'^20\d{2}$', w_clean)):
                date_words.add(w)
        if diff_words and diff_words == date_words:
            return 0.0  # date-only difference, not a duplicate

    return jaccard


def _question_prefix(question: str, n_words: int = 5) -> str:
    """
    Extract the first N content words from a question for prefix dedup.
    Strips stopwords, punctuation, and numbers to get the core subject.
    """
    words = re.findall(r'\b[a-zA-Z]{2,}\b', question.lower())
    content = [w for w in words if w not in STOP_WORDS]
    return " ".join(content[:n_words])


def _question_skeleton(question: str, entities: list[str]) -> str:
    """
    Create a question template by replacing entity names with a placeholder.
    Markets that share a skeleton are same-event outcomes (e.g. different
    countries competing for the same title, different candidates for same office).

    "Will Congo DR win the 2026 FIFA World Cup?" -> "will _ win the 2026 fifa world cup?"
    "Will Eric Trump win the 2028 US Presidential Election?" -> "will _ win the 2028 us presidential election?"
    """
    skeleton = question
    # Sort entities longest-first to avoid partial replacements
    for ent in sorted(entities, key=len, reverse=True):
        skeleton = re.sub(re.escape(ent), '_', skeleton, flags=re.IGNORECASE)
    # Collapse multiple placeholders and whitespace
    skeleton = re.sub(r'_[\s_]*_', '_', skeleton)
    skeleton = re.sub(r'\s+', ' ', skeleton).strip().lower()
    # Normalize stray punctuation around placeholders (e.g. "_." from "Jr.")
    skeleton = re.sub(r'_\.', '_', skeleton)
    return skeleton


# Patterns that indicate opposing / mutually exclusive framing
_OPPOSING_PATTERNS = [
    # (pattern_a, pattern_b) -- if question A matches pattern_a and B matches
    # pattern_b on the same subject, they are likely mutually exclusive
    (r'\bdemocrat', r'\brepublican'),
    (r'\byes\b.*\bno\b', r'\bno\b.*\byes\b'),
    (r'\bover\b', r'\bunder\b'),
    (r'\babove\b', r'\bbelow\b'),
    (r'\bincrease\b', r'\bdecrease\b'),
    (r'\brise\b', r'\bfall\b'),
    (r'\bwin\b', r'\blose\b'),
    (r'\bhigher\b', r'\blower\b'),
    (r'\bcut\b', r'\braise\b'),
    (r'\bcut\b', r'\bhike\b'),
]


def _has_opposing_pattern(q_a: str, q_b: str) -> bool:
    """Check if two questions contain opposing terms."""
    qa_low = q_a.lower()
    qb_low = q_b.lower()
    for pat_a, pat_b in _OPPOSING_PATTERNS:
        if ((re.search(pat_a, qa_low) and re.search(pat_b, qb_low))
                or (re.search(pat_b, qa_low) and re.search(pat_a, qb_low))):
            return True
    return False


def find_candidate_pairs(
    market_entities: list[MarketEntity],
    min_keyword_overlap: int = 4,
    min_entity_overlap: int = 2,
    dedup_similarity: float = 0.80,
) -> list[tuple[MarketEntity, MarketEntity, int, int]]:
    """
    Find market pairs with enough keyword/entity overlap to be
    potentially logically related. Excludes same-event pairs and
    near-duplicate questions (different event_slug but same market).

    Pairing requires BOTH keyword and entity overlap (or very high
    keyword overlap alone) to avoid sports/commodity combinatorial
    explosion from single shared entity names.

    Returns list of (entity_a, entity_b, kw_overlap, ent_overlap).
    """
    pairs = []
    n = len(market_entities)

    # Pre-compute question prefixes and skeletons for fast dedup
    prefixes = [_question_prefix(me.market.question) for me in market_entities]
    skeletons = [
        _question_skeleton(me.market.question, me.entities)
        for me in market_entities
    ]

    for i in range(n):
        a = market_entities[i]
        for j in range(i + 1, n):
            b = market_entities[j]

            # Skip same-event pairs -- already handled elsewhere
            if (a.market.event_slug
                    and a.market.event_slug == b.market.event_slug):
                continue

            # Skip same-template markets (same skeleton = same-event outcomes)
            # e.g. "Will [Congo] win 2026 FIFA World Cup?" and
            #      "Will [Brazil] win 2026 FIFA World Cup?"
            # These are multi-outcome markets that should be handled by
            # the within-event sum-check, not cross-event analysis.
            if skeletons[i] and skeletons[i] == skeletons[j]:
                continue

            # Skip same-family markets (same prefix = same instrument/matchup)
            # BUT exempt pairs with different dates — those are subsumption candidates
            if prefixes[i] and prefixes[i] == prefixes[j]:
                if a.dates != b.dates or a.scope_tokens != b.scope_tokens:
                    pass  # different dates — let subsumption handle it
                else:
                    continue

            # Skip near-duplicate questions (same market, different event_slug)
            if _question_similarity(
                a.market.question, b.market.question
            ) >= dedup_similarity:
                continue

            kw_ov, ent_ov = compute_pair_score(a, b)

            # Require BOTH keyword and entity overlap to pair,
            # OR very high keyword overlap alone (6+),
            # OR opposing-pattern match with at least 1 shared entity
            #    (e.g. Democrat/Republican + shared "Senate").
            if ((kw_ov >= min_keyword_overlap and ent_ov >= min_entity_overlap)
                    or kw_ov >= 6
                    or (ent_ov >= 1 and _has_opposing_pattern(
                        a.market.question, b.market.question))):
                pairs.append((a, b, kw_ov, ent_ov))

    # Sort by combined relevance (entity overlap weighted higher)
    pairs.sort(key=lambda p: (p[3] * 3 + p[2]), reverse=True)
    return pairs


# ─── Heuristic Dependency Classification ─────────────────

# Month ordering for subsumption detection
MONTH_ORDER = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    "q1": 3, "q2": 6, "q3": 9, "q4": 12,
    "h1": 6, "h2": 12,
}


def _extract_deadline(market: MarketEntity) -> Optional[tuple[int, int]]:
    """
    Try to extract a (year, month) deadline from the market's scope tokens
    and dates. Returns None if no clear deadline found.
    """
    year = None
    month = None

    for d in market.dates:
        if re.match(r'^20\d{2}$', d):
            year = int(d)
        elif d.lower() in MONTH_ORDER:
            month = MONTH_ORDER[d.lower()]

    for scope in market.scope_tokens:
        parts = scope.split()
        if len(parts) == 2:
            token = parts[1].lower()
            if re.match(r'^20\d{2}$', token):
                year = int(token)
            elif token in MONTH_ORDER:
                month = MONTH_ORDER[token]

    # Also try end_date from the market itself
    if year is None and market.market.end_date:
        year = market.market.end_date.year
    if month is None and market.market.end_date:
        month = market.market.end_date.month

    if year is not None:
        return (year, month or 12)
    return None


def classify_dependency(
    a: MarketEntity, b: MarketEntity, kw_overlap: int, ent_overlap: int,
) -> DependencyPair:
    """
    Heuristic classification of the logical relationship between two markets.

    Rules applied in priority order:
    1. Subsumption — same subject, different deadline → narrower ≤ broader
    2. Implication — one question is a strict subset/stronger claim
    3. Mutual exclusion — opposing framing on same subject
    4. Fallback → correlated
    """
    q_a = a.market.question.lower()
    q_b = b.market.question.lower()

    # ── Rule 1: Subsumption (date/scope nesting) ──────────
    # "Will X happen by June?" vs "Will X happen by December?"
    # The June market is subsumed by the December market.
    deadline_a = _extract_deadline(a)
    deadline_b = _extract_deadline(b)

    if deadline_a and deadline_b and deadline_a != deadline_b:
        # Check if the non-date parts of the question are similar enough
        # Strip date-like tokens and compare
        core_a = re.sub(
            r'\b(by|before|after|in|during)\s+\w+\b', '', q_a
        ).strip()
        core_b = re.sub(
            r'\b(by|before|after|in|during)\s+\w+\b', '', q_b
        ).strip()

        # Also strip year references
        core_a = re.sub(r'\b20\d{2}\b', '', core_a).strip()
        core_b = re.sub(r'\b20\d{2}\b', '', core_b).strip()

        # Compute similarity of the core question (Jaccard on words)
        words_a = set(core_a.split()) - STOP_WORDS
        words_b = set(core_b.split()) - STOP_WORDS
        if words_a and words_b:
            jaccard = len(words_a & words_b) / len(words_a | words_b)
        else:
            jaccard = 0

        if jaccard > 0.5:
            # Guard: if the core questions contain opposing terms,
            # this is mutual exclusion, not subsumption
            # e.g. "cut rates" vs "raise rates" are NOT the same subject
            has_opposing = False
            for pat_a_re, pat_b_re in _OPPOSING_PATTERNS:
                if ((re.search(pat_a_re, core_a) and re.search(pat_b_re, core_b))
                        or (re.search(pat_b_re, core_a) and re.search(pat_a_re, core_b))):
                    has_opposing = True
                    break

            if not has_opposing:
                # Earlier deadline is stricter → its probability must be ≤ later
                if deadline_a < deadline_b:
                    return DependencyPair(
                        market_a=a.market, market_b=b.market,
                        dep_type=DependencyType.SUBSUMPTION,
                        direction="a_implies_b",
                        confidence=min(0.9, 0.5 + jaccard * 0.4),
                        reason=(
                            f"Same subject with different deadlines: "
                            f"A={deadline_a} <= B={deadline_b}. "
                            f"Earlier deadline must have lower probability."
                        ),
                        keyword_overlap=kw_overlap,
                        entity_overlap=ent_overlap,
                    )
                else:
                    return DependencyPair(
                        market_a=a.market, market_b=b.market,
                        dep_type=DependencyType.SUBSUMPTION,
                        direction="b_implies_a",
                        confidence=min(0.9, 0.5 + jaccard * 0.4),
                        reason=(
                            f"Same subject with different deadlines: "
                            f"B={deadline_b} <= A={deadline_a}. "
                            f"Earlier deadline must have lower probability."
                        ),
                        keyword_overlap=kw_overlap,
                        entity_overlap=ent_overlap,
                    )

    # ── Rule 2: Implication — DEFERRED TO PHASE B ──────────
    # Heuristic implication detection via broadening words produces
    # too many false positives (e.g. "ceasefire with Iran" matched as
    # implying "military action against Iran" because of shared
    # {iran, april} + broadening word {country}).
    # Implication requires semantic understanding of directionality
    # that only an LLM can provide. Candidate pairs with high overlap
    # will flow to Phase B as "correlated" for LLM classification.

    # ── Rule 3: Mutual Exclusion ──────────────────────────
    # Markets with opposing framing on the same subject
    for pat_a, pat_b in _OPPOSING_PATTERNS:
        if ((re.search(pat_a, q_a) and re.search(pat_b, q_b))
                or (re.search(pat_b, q_a) and re.search(pat_a, q_b))):
            # Only if they share enough subject matter
            # (pattern match is signal, but need enough shared context
            #  to avoid matching unrelated markets that happen to use
            #  opposing words like "rise"/"fall" on different subjects)
            if ent_overlap >= 1 or kw_overlap >= 3:
                return DependencyPair(
                    market_a=a.market, market_b=b.market,
                    dep_type=DependencyType.MUTUAL_EXCLUSION,
                    direction="symmetric",
                    confidence=0.4 + min(0.3, ent_overlap * 0.1),
                    reason=(
                        f"Opposing framing detected on shared subject "
                        f"(entities: {set(a.entities) & set(b.entities)})"
                    ),
                    keyword_overlap=kw_overlap,
                    entity_overlap=ent_overlap,
                )

    # ── Fallback: correlated ──────────────────────────────
    return DependencyPair(
        market_a=a.market, market_b=b.market,
        dep_type=DependencyType.CORRELATED,
        direction="symmetric",
        confidence=0.3 + min(0.2, ent_overlap * 0.05),
        reason=f"Keyword overlap={kw_overlap}, entity overlap={ent_overlap}",
        keyword_overlap=kw_overlap,
        entity_overlap=ent_overlap,
    )


# ─── Price Consistency Check ─────────────────────────────

def check_price_consistency(
    pair: DependencyPair,
    spread_tolerance: float = 0.03,
) -> Optional[PriceViolation]:
    """
    Given a classified dependency pair, check if current prices violate
    the logical constraint. Returns a PriceViolation if found, else None.

    Args:
        pair: classified dependency pair
        spread_tolerance: minimum violation magnitude to flag (accounts
                          for bid-ask spread). Default 3%.
    """
    p_a = pair.market_a.yes_price
    p_b = pair.market_b.yes_price

    if pair.dep_type == DependencyType.IMPLICATION:
        if pair.direction == "a_implies_b":
            # A implies B → P(A) <= P(B)
            violation = p_a - p_b
            if violation > spread_tolerance:
                return PriceViolation(
                    pair=pair,
                    violation_magnitude=violation,
                    expected_constraint=f"P(A) <= P(B)  [A implies B]",
                    actual_prices=f"P(A)={p_a:.1%}, P(B)={p_b:.1%}",
                    implied_edge_pct=violation * 100,
                    suggested_action=f"BUY B ({pair.market_b.question[:50]}) "
                                     f"or SELL A ({pair.market_a.question[:50]})",
                )
        elif pair.direction == "b_implies_a":
            violation = p_b - p_a
            if violation > spread_tolerance:
                return PriceViolation(
                    pair=pair,
                    violation_magnitude=violation,
                    expected_constraint=f"P(B) <= P(A)  [B implies A]",
                    actual_prices=f"P(A)={p_a:.1%}, P(B)={p_b:.1%}",
                    implied_edge_pct=violation * 100,
                    suggested_action=f"BUY A ({pair.market_a.question[:50]}) "
                                     f"or SELL B ({pair.market_b.question[:50]})",
                )

    elif pair.dep_type == DependencyType.SUBSUMPTION:
        if pair.direction == "a_implies_b":
            # A is stricter (earlier deadline) → P(A) <= P(B)
            violation = p_a - p_b
            if violation > spread_tolerance:
                return PriceViolation(
                    pair=pair,
                    violation_magnitude=violation,
                    expected_constraint=f"P(A) <= P(B)  [A has earlier deadline]",
                    actual_prices=f"P(A)={p_a:.1%}, P(B)={p_b:.1%}",
                    implied_edge_pct=violation * 100,
                    suggested_action=f"BUY B (later deadline) or SELL A (earlier)",
                )
        elif pair.direction == "b_implies_a":
            violation = p_b - p_a
            if violation > spread_tolerance:
                return PriceViolation(
                    pair=pair,
                    violation_magnitude=violation,
                    expected_constraint=f"P(B) <= P(A)  [B has earlier deadline]",
                    actual_prices=f"P(A)={p_a:.1%}, P(B)={p_b:.1%}",
                    implied_edge_pct=violation * 100,
                    suggested_action=f"BUY A (later deadline) or SELL B (earlier)",
                )

    elif pair.dep_type == DependencyType.MUTUAL_EXCLUSION:
        # A and B can't both be YES → P(A) + P(B) <= 1.0
        total = p_a + p_b
        violation = total - 1.0
        if violation > spread_tolerance:
            return PriceViolation(
                pair=pair,
                violation_magnitude=violation,
                expected_constraint=f"P(A) + P(B) <= 1.0  [mutually exclusive]",
                actual_prices=f"P(A)={p_a:.1%} + P(B)={p_b:.1%} = {total:.1%}",
                implied_edge_pct=violation * 100,
                suggested_action=f"SELL the overpriced side (one must be NO)",
            )

    return None


# ─── Main Scanner ────────────────────────────────────────

class CrossEventScanner:
    """
    Scans the full market universe for cross-event logical dependency
    violations. Runs once per pipeline cycle.

    Usage:
        scanner = CrossEventScanner()
        violations = scanner.scan()
        for v in violations:
            print(v)
    """

    def __init__(
        self,
        client: PolymarketClient = None,
        min_keyword_overlap: int = 4,
        min_entity_overlap: int = 2,
        spread_tolerance: float = 0.03,
        min_volume_24h: float = 500,
        min_liquidity: float = 2000,
    ):
        self.client = client or PolymarketClient()
        self.min_keyword_overlap = min_keyword_overlap
        self.min_entity_overlap = min_entity_overlap
        self.spread_tolerance = spread_tolerance
        self.min_volume_24h = min_volume_24h
        self.min_liquidity = min_liquidity

    def scan(
        self,
        markets: list[Market] = None,
        max_markets: int = 300,
        include_correlated: bool = False,
    ) -> tuple[list[PriceViolation], list[DependencyPair]]:
        """
        Run the full cross-event scan.

        Args:
            markets: pre-fetched markets (if None, fetches from API)
            max_markets: how many markets to fetch if not provided
            include_correlated: if True, also return correlated pairs
                                (not just strict dependency violations)

        Returns:
            (violations, all_dependency_pairs)
        """
        # 1. Fetch markets
        if markets is None:
            logger.info(f"Fetching up to {max_markets} active markets...")
            markets = self._fetch_markets(max_markets)
        logger.info(f"Scanning {len(markets)} markets for cross-event dependencies")

        # 2. Filter to tradeable markets
        tradeable = [
            m for m in markets
            if m.active
            and not m.closed
            and m.volume_24h >= self.min_volume_24h
            and m.liquidity >= self.min_liquidity
        ]
        logger.info(f"After filters: {len(tradeable)} tradeable markets")

        # 3. Extract entities
        entities = []
        for m in tradeable:
            me = extract_entities(m.question)
            me.market = m
            entities.append(me)

        # 4. Find candidate pairs
        candidates = find_candidate_pairs(
            entities,
            min_keyword_overlap=self.min_keyword_overlap,
            min_entity_overlap=self.min_entity_overlap,
        )
        logger.info(f"Found {len(candidates)} candidate pairs from keyword/entity overlap")

        # 5. Classify dependencies
        dependency_pairs = []
        for a, b, kw_ov, ent_ov in candidates:
            dep = classify_dependency(a, b, kw_ov, ent_ov)
            dependency_pairs.append(dep)

        # Filter out independent/correlated unless requested
        logical_pairs = [
            p for p in dependency_pairs
            if p.dep_type in (
                DependencyType.IMPLICATION,
                DependencyType.MUTUAL_EXCLUSION,
                DependencyType.SUBSUMPTION,
            )
        ]
        if include_correlated:
            logical_pairs = [
                p for p in dependency_pairs
                if p.dep_type != DependencyType.INDEPENDENT
            ]

        logger.info(
            f"Classified: {len(logical_pairs)} logical dependencies "
            f"({sum(1 for p in logical_pairs if p.dep_type == DependencyType.IMPLICATION)} impl, "
            f"{sum(1 for p in logical_pairs if p.dep_type == DependencyType.SUBSUMPTION)} subsum, "
            f"{sum(1 for p in logical_pairs if p.dep_type == DependencyType.MUTUAL_EXCLUSION)} mutex, "
            f"{sum(1 for p in logical_pairs if p.dep_type == DependencyType.CORRELATED)} correlated)"
        )

        # 6. Check price consistency
        violations = []
        for pair in logical_pairs:
            v = check_price_consistency(pair, self.spread_tolerance)
            if v is not None:
                violations.append(v)

        violations.sort(key=lambda v: v.violation_magnitude, reverse=True)
        logger.info(f"Found {len(violations)} price violations above {self.spread_tolerance:.0%} threshold")

        return violations, dependency_pairs

    def _fetch_markets(self, max_markets: int) -> list[Market]:
        """Fetch markets with pagination."""
        all_markets = []
        pages = (max_markets + 99) // 100
        for page in range(pages):
            batch = self.client.get_active_markets(
                limit=100, offset=page * 100
            )
            if not batch:
                break
            all_markets.extend(batch)
            if len(all_markets) >= max_markets:
                break
        return all_markets[:max_markets]

    def get_violations_for_market(
        self, market_id: str, all_pairs: list[DependencyPair],
    ) -> list[DependencyPair]:
        """
        Get all dependency pairs involving a specific market.
        Used by the enrichment pipeline to add cross-event context.
        """
        return [
            p for p in all_pairs
            if p.market_a.id == market_id or p.market_b.id == market_id
        ]


# ─── CLI & Report ────────────────────────────────────────

def format_violation_report(
    violations: list[PriceViolation],
    all_pairs: list[DependencyPair],
) -> str:
    """Format a human-readable report of findings."""
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("  CROSS-EVENT DEPENDENCY SCANNER -- Phase A")
    lines.append("=" * 78)

    # Summary
    type_counts = {}
    for p in all_pairs:
        type_counts[p.dep_type] = type_counts.get(p.dep_type, 0) + 1

    lines.append("")
    lines.append(f"  Dependency pairs found: {len(all_pairs)}")
    for dep_type, count in sorted(type_counts.items()):
        symbol = {
            DependencyType.IMPLICATION: "->",
            DependencyType.SUBSUMPTION: "C=",
            DependencyType.MUTUAL_EXCLUSION: "(+)",
            DependencyType.CORRELATED: "~",
            DependencyType.INDEPENDENT: ".",
        }.get(dep_type, "?")
        lines.append(f"    {symbol} {dep_type}: {count}")

    lines.append(f"\n  Price violations: {len(violations)}")
    lines.append("-" * 78)

    if not violations:
        lines.append("\n  [OK] No price violations found above threshold.")
        lines.append("     Markets appear consistently priced across events.")
    else:
        for i, v in enumerate(violations, 1):
            p = v.pair
            lines.append(f"\n  [!] Violation #{i}  |  Edge: {v.implied_edge_pct:.1f}%  "
                         f"|  Type: {p.dep_type}  |  Confidence: {p.confidence:.0%}")
            lines.append(f"  +-- Market A: {p.market_a.question[:70]}")
            lines.append(f"  |   Price: {p.market_a.yes_price:.1%}  "
                         f"Vol24h: ${p.market_a.volume_24h:,.0f}  "
                         f"Liq: ${p.market_a.liquidity:,.0f}")
            lines.append(f"  +-- Market B: {p.market_b.question[:70]}")
            lines.append(f"  |   Price: {p.market_b.yes_price:.1%}  "
                         f"Vol24h: ${p.market_b.volume_24h:,.0f}  "
                         f"Liq: ${p.market_b.liquidity:,.0f}")
            lines.append(f"  +-- Constraint: {v.expected_constraint}")
            lines.append(f"  +-- Actual:     {v.actual_prices}")
            lines.append(f"  +-- Reason:     {p.reason}")
            lines.append(f"  +-- Action:     {v.suggested_action}")
            lines.append("")

    # Also show top correlated pairs that aren't violations (potential Phase B candidates)
    correlated = [
        p for p in all_pairs
        if p.dep_type == DependencyType.CORRELATED
        and (p.keyword_overlap >= 5 or p.entity_overlap >= 3)
    ]
    if correlated:
        show_n = min(15, len(correlated))
        lines.append("-" * 78)
        lines.append(f"\n  [*] Top correlated pairs (Phase B LLM candidates): "
                     f"{len(correlated)} (showing {show_n})")
        for p in correlated[:show_n]:
            lines.append(f"    ~ {p.market_a.question[:35]}  <->  "
                         f"{p.market_b.question[:35]}")
            lines.append(f"      kw={p.keyword_overlap} ent={p.entity_overlap}  "
                         f"prices: {p.market_a.yes_price:.0%} / {p.market_b.yes_price:.0%}")

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)


def run_cross_event_scan(
    max_markets: int = 300,
    min_edge: float = 3.0,
    verbose: bool = False,
):
    """CLI entry point for the cross-event scanner."""
    scanner = CrossEventScanner(
        spread_tolerance=min_edge / 100.0,
    )
    violations, all_pairs = scanner.scan(
        max_markets=max_markets,
        include_correlated=verbose,
    )
    report = format_violation_report(violations, all_pairs)
    print(report)
    return violations, all_pairs


# ─── Module entry point ──────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = sys.argv[1:]
    max_markets = 300
    min_edge = 3.0
    verbose = False

    if "--top" in args:
        idx = args.index("--top")
        if idx + 1 < len(args):
            max_markets = int(args[idx + 1])

    if "--min-edge" in args:
        idx = args.index("--min-edge")
        if idx + 1 < len(args):
            min_edge = float(args[idx + 1])

    if "--verbose" in args:
        verbose = True

    run_cross_event_scan(
        max_markets=max_markets,
        min_edge=min_edge,
        verbose=verbose,
    )