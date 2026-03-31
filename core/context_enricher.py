"""
Context Enrichment Pipeline
Gathers additional information to feed into the LLM estimator.

Sources:
    1. News Search — uses Claude API with web_search tool to find recent 
       news relevant to the market question
    2. Cross-Platform Prices — fetches matching markets on Kalshi to find 
       arbitrage signals and consensus pricing
    3. Economic Indicators — pulls key FRED data for economics-related markets
    4. Related Markets — finds correlated/related markets on Polymarket
    5. Entity & Keyword Extraction — identifies key entities and search terms

The enriched context is formatted as a structured prompt section that gets
injected into the LLM forecaster prompt, giving it far richer information
to reason over than just the market price.
"""

import re
import json
import time
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
import numpy as np

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


# ═══════════════════════════════════════════════════════════
# 1. NEWS SEARCH ENRICHER
# ═══════════════════════════════════════════════════════════

class NewsEnricher:
    """
    Uses Claude API with web_search tool to find recent news relevant 
    to a prediction market question.
    
    This is the single highest-value enrichment source — recent news that
    the market hasn't fully priced in is where alpha lives.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
    
    def search(self, question: str, category: str = "", max_results: int = 5) -> dict:
        """
        Search for recent news related to a market question.
        
        Returns dict with:
            - headlines: list of relevant headline summaries
            - sentiment: overall sentiment ("bullish_yes", "bearish_yes", "neutral")
            - key_facts: list of key factual findings
            - recency: how recent the most relevant info is
            - search_queries: what was searched
        """
        if not self.api_key:
            logger.info("No API key — skipping news search")
            return self._empty_result()
        
        # Build search-optimized queries from the question
        queries = self._extract_search_queries(question, category)
        
        # Use Claude with web_search to gather and synthesize news
        prompt = self._build_news_prompt(question, queries)
        
        try:
            result = self._call_claude_with_search(prompt)
            if result:
                return self._parse_news_result(result)
        except Exception as e:
            logger.warning(f"News search failed: {e}")
        
        return self._empty_result()
    
    def _extract_search_queries(self, question: str, category: str = "") -> list[str]:
        """Extract 2-3 targeted search queries from the market question."""
        # Remove common prediction market phrasing
        q = question.lower()
        q = re.sub(r'\bwill\b|\bby\b|\bbefore\b|\bin 20\d\d\b', '', q)
        q = re.sub(r'[?!]', '', q)
        q = q.strip()
        
        queries = []
        
        # Primary query: the core topic
        queries.append(f"{q} latest news 2026")
        
        # Secondary query: more specific
        # Extract key entities (capitalized words, numbers, known terms)
        entities = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', question)
        if entities:
            queries.append(f"{' '.join(entities[:3])} news today")
        
        # Category-specific query
        if category:
            queries.append(f"{category} {q[:30]} forecast analysis")
        
        return queries[:3]
    
    def _build_news_prompt(self, question: str, queries: list[str]) -> str:
        return f"""I need to assess the probability of this prediction market outcome:
"{question}"

Search for the most recent and relevant news, data, and expert analysis.
Focus on information from the last 7 days that could affect the probability.

Respond with ONLY this JSON:
{{
    "headlines": ["<headline/summary 1>", "<headline/summary 2>", ...],
    "key_facts": ["<concrete fact 1>", "<concrete fact 2>", ...],
    "sentiment": "<bullish_yes|bearish_yes|neutral>",
    "sentiment_strength": <float 0-1, how strongly the news leans>,
    "recency": "<today|this_week|this_month|older>",
    "confidence_in_findings": <float 0-1>
}}"""

    def _call_claude_with_search(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Claude API with web_search tool enabled and retry on rate limits."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "temperature": 0.1,
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                }
            ],
            "messages": [{"role": "user", "content": prompt}],
        }
        
        for attempt in range(max_retries):
            resp = requests.post(
                ANTHROPIC_API_URL, headers=headers, json=payload, timeout=45
            )
            
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("retry-after", 2 ** (attempt + 1)))
                wait = max(retry_after, 2 ** (attempt + 1))
                logger.info(f"Rate limited (news search), waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            
            resp.raise_for_status()
            data = resp.json()
            
            # Extract text from all content blocks
            text_parts = []
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block["text"])
            
            return "\n".join(text_parts).strip() if text_parts else None
        
        logger.warning("News search: max retries exhausted")
        return None
    
    def _parse_news_result(self, raw: str) -> dict:
        """Parse the JSON response from the news search."""
        # Strip markdown fences
        text = raw.strip()
        if "```" in text:
            # Find JSON between code fences
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)
        
        # Try to find JSON object in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        try:
            data = json.loads(text)
            return {
                "headlines": data.get("headlines", [])[:5],
                "key_facts": data.get("key_facts", [])[:5],
                "sentiment": data.get("sentiment", "neutral"),
                "sentiment_strength": float(data.get("sentiment_strength", 0.0)),
                "recency": data.get("recency", "unknown"),
                "confidence": float(data.get("confidence_in_findings", 0.3)),
                "source": "claude_web_search",
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            # Fallback: extract what we can from plain text
            return {
                "headlines": [],
                "key_facts": [raw[:200]] if raw else [],
                "sentiment": "neutral",
                "sentiment_strength": 0.0,
                "recency": "unknown",
                "confidence": 0.1,
                "source": "claude_web_search_unparsed",
                "raw_text": raw[:500],
            }
    
    def _empty_result(self) -> dict:
        return {
            "headlines": [],
            "key_facts": [],
            "sentiment": "neutral",
            "sentiment_strength": 0.0,
            "recency": "none",
            "confidence": 0.0,
            "source": "none",
        }


# ═══════════════════════════════════════════════════════════
# 2. CROSS-PLATFORM PRICE ENRICHER (Kalshi)
# ═══════════════════════════════════════════════════════════

class KalshiPriceEnricher:
    """
    Fetches matching markets on Kalshi (no auth needed for public data).
    Cross-platform price differences are a direct arbitrage signal.
    
    Kalshi API: https://api.elections.kalshi.com/trade-api/v2
    """
    
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def __init__(self, rate_limit_delay: float = 0.5):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.rate_limit_delay = rate_limit_delay
        self._last_request = 0.0
    
    def find_matching_market(self, question: str) -> Optional[dict]:
        """
        Search Kalshi for a market matching the given question.
        
        Returns dict with Kalshi price data if found, None otherwise.
        """
        # Extract search keywords
        keywords = self._extract_keywords(question)
        
        for query_terms in keywords:
            try:
                result = self._search_markets(query_terms)
                if result:
                    return result
            except Exception as e:
                logger.debug(f"Kalshi search failed for '{query_terms}': {e}")
        
        return None
    
    def get_active_markets(self, limit: int = 50, status: str = "open") -> list[dict]:
        """Fetch active Kalshi markets."""
        self._throttle()
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/markets",
                params={"limit": limit, "status": status},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("markets", [])
        except Exception as e:
            logger.warning(f"Kalshi API error: {e}")
            return []
    
    def _search_markets(self, query: str) -> Optional[dict]:
        """Search Kalshi markets by text."""
        self._throttle()
        try:
            # Kalshi doesn't have a text search endpoint, so we fetch and filter
            resp = self.session.get(
                f"{self.BASE_URL}/markets",
                params={"limit": 200, "status": "open"},
                timeout=10,
            )
            resp.raise_for_status()
            markets = resp.json().get("markets", [])
            
            # Simple text matching
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            best_match = None
            best_score = 0
            
            for m in markets:
                title = (m.get("title", "") + " " + m.get("yes_sub_title", "")).lower()
                # Count matching words
                title_words = set(title.split())
                overlap = len(query_words & title_words)
                
                if overlap > best_score and overlap >= 2:
                    best_score = overlap
                    best_match = m
            
            if best_match:
                yes_price = float(best_match.get("yes_bid_dollars", 0) or 0)
                return {
                    "platform": "kalshi",
                    "ticker": best_match.get("ticker", ""),
                    "title": best_match.get("title", ""),
                    "yes_price": yes_price,
                    "volume": float(best_match.get("volume_fp", 0) or 0),
                    "volume_24h": float(best_match.get("volume_24h_fp", 0) or 0),
                    "match_score": best_score,
                    "status": best_match.get("status", ""),
                }
        except Exception as e:
            logger.debug(f"Kalshi search error: {e}")
        
        return None
    
    def _extract_keywords(self, question: str) -> list[str]:
        """Extract search keyword sets from a question."""
        # Remove common words
        stopwords = {
            "will", "the", "a", "an", "in", "on", "by", "be", "is", "are",
            "this", "that", "to", "of", "for", "with", "have", "has", "does",
            "do", "did", "before", "after", "during", "than", "more", "less",
        }
        
        words = re.findall(r'\b[a-zA-Z0-9]+\b', question.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Return different subsets to try
        results = []
        if len(keywords) >= 3:
            results.append(" ".join(keywords[:4]))
        if len(keywords) >= 2:
            results.append(" ".join(keywords[:2]))
        
        return results
    
    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request = time.time()


# ═══════════════════════════════════════════════════════════
# 3. ECONOMIC INDICATORS ENRICHER (FRED)
# ═══════════════════════════════════════════════════════════

# Key FRED series for prediction market categories
FRED_INDICATORS = {
    # Interest rates & Fed
    "fed_funds_rate": {"series": "DFF", "name": "Fed Funds Rate", "freq": "daily"},
    "treasury_10y": {"series": "DGS10", "name": "10-Year Treasury", "freq": "daily"},
    "treasury_2y": {"series": "DGS2", "name": "2-Year Treasury", "freq": "daily"},
    "yield_curve": {"series": "T10Y2Y", "name": "10Y-2Y Spread", "freq": "daily"},
    
    # Inflation
    "cpi": {"series": "CPIAUCSL", "name": "CPI (All Urban)", "freq": "monthly"},
    "core_cpi": {"series": "CPILFESL", "name": "Core CPI", "freq": "monthly"},
    "pce": {"series": "PCEPI", "name": "PCE Price Index", "freq": "monthly"},
    "breakeven_5y": {"series": "T5YIE", "name": "5-Year Breakeven Inflation", "freq": "daily"},
    
    # Employment
    "unemployment": {"series": "UNRATE", "name": "Unemployment Rate", "freq": "monthly"},
    "nonfarm_payrolls": {"series": "PAYEMS", "name": "Nonfarm Payrolls", "freq": "monthly"},
    "initial_claims": {"series": "ICSA", "name": "Initial Jobless Claims", "freq": "weekly"},
    
    # GDP & Growth
    "gdp": {"series": "GDP", "name": "GDP", "freq": "quarterly"},
    "gdp_growth": {"series": "A191RL1Q225SBEA", "name": "Real GDP Growth", "freq": "quarterly"},
    
    # Markets
    "sp500": {"series": "SP500", "name": "S&P 500", "freq": "daily"},
    "vix": {"series": "VIXCLS", "name": "VIX (Volatility)", "freq": "daily"},
    
    # Housing
    "housing_starts": {"series": "HOUST", "name": "Housing Starts", "freq": "monthly"},
    
    # Consumer
    "consumer_sentiment": {"series": "UMCSENT", "name": "Consumer Sentiment", "freq": "monthly"},
    "retail_sales": {"series": "RSXFS", "name": "Retail Sales", "freq": "monthly"},
}

# Map market topics to relevant FRED indicators
TOPIC_INDICATORS = {
    "fed": ["fed_funds_rate", "treasury_2y", "treasury_10y", "yield_curve", "breakeven_5y"],
    "rate": ["fed_funds_rate", "treasury_2y", "treasury_10y", "yield_curve"],
    "inflation": ["cpi", "core_cpi", "pce", "breakeven_5y"],
    "cpi": ["cpi", "core_cpi", "pce"],
    "recession": ["gdp_growth", "unemployment", "yield_curve", "initial_claims", "consumer_sentiment"],
    "gdp": ["gdp", "gdp_growth"],
    "employment": ["unemployment", "nonfarm_payrolls", "initial_claims"],
    "jobs": ["unemployment", "nonfarm_payrolls", "initial_claims"],
    "unemployment": ["unemployment", "initial_claims"],
    "market": ["sp500", "vix"],
    "stock": ["sp500", "vix"],
    "housing": ["housing_starts"],
    "consumer": ["consumer_sentiment", "retail_sales"],
}


class EconomicEnricher:
    """
    Fetches relevant economic indicators from the FRED API.
    
    Requires a free FRED API key (get one at https://fred.stlouisfed.org/docs/api/api_key.html).
    Set as FRED_API_KEY environment variable.
    """
    
    FRED_BASE = "https://api.stlouisfed.org/fred"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
    
    def get_relevant_indicators(self, question: str) -> dict:
        """
        Fetch economic indicators relevant to the market question.
        
        Returns dict with indicator names, recent values, and trends.
        """
        if not self.api_key:
            logger.info("No FRED_API_KEY — returning static indicators only")
            return self._static_indicators(question)
        
        # Determine which indicators are relevant
        indicator_keys = self._match_indicators(question)
        
        if not indicator_keys:
            return {"indicators": [], "source": "fred", "note": "No relevant indicators found"}
        
        indicators = []
        for key in indicator_keys[:5]:  # Max 5 to limit API calls
            info = FRED_INDICATORS[key]
            try:
                data = self._fetch_series(info["series"], limit=10)
                if data:
                    recent = data[-1]
                    prev = data[-2] if len(data) > 1 else data[-1]
                    
                    indicators.append({
                        "name": info["name"],
                        "series_id": info["series"],
                        "latest_value": recent["value"],
                        "latest_date": recent["date"],
                        "previous_value": prev["value"],
                        "change": recent["value"] - prev["value"],
                        "change_pct": ((recent["value"] - prev["value"]) / prev["value"] * 100
                                       if prev["value"] != 0 else 0),
                        "trend": "rising" if recent["value"] > prev["value"] else "falling",
                    })
            except Exception as e:
                logger.debug(f"FRED fetch failed for {key}: {e}")
        
        return {
            "indicators": indicators,
            "source": "fred_api",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def _match_indicators(self, question: str) -> list[str]:
        """Match a question to relevant FRED indicator keys."""
        q_lower = question.lower()
        matched = set()
        
        for topic, indicators in TOPIC_INDICATORS.items():
            if topic in q_lower:
                matched.update(indicators)
        
        # Always include some baseline indicators for economic markets
        if not matched and any(w in q_lower for w in ["economy", "economic", "growth", "financial"]):
            matched.update(["gdp_growth", "sp500", "unemployment"])
        
        return list(matched)
    
    def _fetch_series(self, series_id: str, limit: int = 10) -> list[dict]:
        """Fetch recent observations for a FRED series."""
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        
        resp = requests.get(
            f"{self.FRED_BASE}/series/observations",
            params=params, timeout=10
        )
        resp.raise_for_status()
        
        observations = resp.json().get("observations", [])
        
        # Parse and filter valid observations
        result = []
        for obs in reversed(observations):  # Reverse to chronological order
            val = obs.get("value", ".")
            if val != "." and val is not None:
                try:
                    result.append({
                        "date": obs["date"],
                        "value": float(val),
                    })
                except (ValueError, TypeError):
                    pass
        
        return result
    
    def _static_indicators(self, question: str) -> dict:
        """
        Return static/cached indicator context when no API key is available.
        Still useful as the LLM knows what these indicators mean.
        """
        indicator_keys = self._match_indicators(question)
        
        indicators = []
        for key in indicator_keys[:5]:
            info = FRED_INDICATORS[key]
            indicators.append({
                "name": info["name"],
                "series_id": info["series"],
                "note": "Live data unavailable — set FRED_API_KEY for real-time data",
            })
        
        return {
            "indicators": indicators,
            "source": "static",
            "note": "Set FRED_API_KEY env var for live economic data",
        }


# ═══════════════════════════════════════════════════════════
# 4. RELATED MARKETS ENRICHER
# ═══════════════════════════════════════════════════════════

class RelatedMarketsEnricher:
    """
    Finds related/correlated markets on Polymarket.
    
    Related market prices provide implicit information:
    - If "Fed cuts in June" is at 40% and "Fed cuts in July" is at 60%,
      that tells you something about the expected timing.
    - If "GDP > 3%" is at 30% and "Recession in 2026" is at 20%,
      those should be somewhat inversely correlated.
    """
    
    GAMMA_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self, rate_limit_delay: float = 0.5):
        self.session = requests.Session()
        self.rate_limit_delay = rate_limit_delay
        self._last_request = 0.0
    
    def find_related(self, market_data: dict, all_markets: list[dict] = None) -> list[dict]:
        """
        Find markets related to the given market.
        
        Strategy:
        1. Same event — other markets in the same event (most relevant)
        2. Same tags/category — markets with similar topics
        3. Keyword overlap — markets with similar questions
        """
        results = []
        
        event_slug = market_data.get("event_slug", "")
        question = market_data.get("question", "")
        market_id = market_data.get("market_id", market_data.get("id", ""))
        
        # 1. Same event markets
        if event_slug:
            try:
                event_markets = self._fetch_event_markets(event_slug)
                for em in event_markets:
                    if str(em.get("id", "")) != str(market_id):
                        prices = em.get("outcomePrices", [0.5, 0.5])
                        if isinstance(prices, str):
                            prices = json.loads(prices)
                        results.append({
                            "question": em.get("question", ""),
                            "price": float(prices[0]) if prices else 0.5,
                            "relation": "same_event",
                            "volume_24h": float(em.get("volume24hr", 0) or 0),
                        })
            except Exception as e:
                logger.debug(f"Event markets fetch failed: {e}")
        
        # 2. Keyword-based search from provided market list
        if all_markets:
            keywords = self._extract_keywords(question)
            for m in all_markets:
                m_id = str(m.get("market_id", m.get("id", "")))
                m_question = m.get("question", "")
                
                if m_id == str(market_id) or not m_question:
                    continue
                
                # Score by keyword overlap
                m_words = set(m_question.lower().split())
                overlap = len(keywords & m_words)
                
                if overlap >= 2:
                    m_price = m.get("yes_price", m.get("market_price", 0.5))
                    if isinstance(m_price, str):
                        m_price = float(m_price)
                    results.append({
                        "question": m_question,
                        "price": m_price,
                        "relation": "keyword_overlap",
                        "overlap_score": overlap,
                        "volume_24h": float(m.get("volume_24h", 0) or 0),
                    })
        
        # Sort by relevance
        results.sort(key=lambda r: (
            0 if r["relation"] == "same_event" else 1,
            -r.get("overlap_score", 0),
            -r.get("volume_24h", 0),
        ))
        
        return results[:8]
    
    def _fetch_event_markets(self, event_slug: str) -> list[dict]:
        """Fetch all markets for an event from Gamma API."""
        self._throttle()
        resp = self.session.get(
            f"{self.GAMMA_URL}/events/{event_slug}",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Events can contain markets directly or nested
        markets = data.get("markets", [])
        if not markets and isinstance(data, list):
            markets = data
        
        return markets
    
    def _extract_keywords(self, question: str) -> set[str]:
        stopwords = {
            "will", "the", "a", "an", "in", "on", "by", "be", "is", "are",
            "this", "that", "to", "of", "for", "with", "have", "has", "does",
            "do", "did", "before", "after", "during", "than", "more", "less",
            "what", "when", "where", "how", "who", "which", "would", "could",
        }
        words = set(re.findall(r'\b[a-z]+\b', question.lower()))
        return words - stopwords
    
    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request = time.time()


# ═══════════════════════════════════════════════════════════
# 5. WHALE TRACKER ENRICHER
# ═══════════════════════════════════════════════════════════

class WhaleEnricher:
    """
    Tracks large, historically profitable wallets ("whales") and detects
    when they hold positions in candidate markets.
    
    Strategy:
        1. Fetch the Polymarket leaderboard once at pipeline start to build
           a "whale registry" — a mapping of wallet → PnL.
        2. For each candidate market, call the /holders endpoint to get top
           holders, then cross-reference against the whale registry.
        3. If a WhaleProfiler is attached, enrich each whale position with
           profile data (strategy type, category credibility, signal weight).
        4. Return whale positions in the format the ensemble's
           profiled_whale_estimator expects.
    
    This is the only enricher that requires a PolymarketClient instance,
    since it uses the Data API.
    """
    
    DATA_API_URL = "https://data-api.polymarket.com"
    
    def __init__(
        self,
        top_n_whales: int = 50,
        time_period: str = "ALL",
        min_pnl: float = 5000.0,
        rate_limit_delay: float = 0.5,
        profiler: "WhaleProfiler | None" = None,
    ):
        """
        Args:
            top_n_whales: How many top traders to track from the leaderboard.
            time_period: Leaderboard window — DAY, WEEK, MONTH, or ALL.
            min_pnl: Minimum PnL (USD) to qualify as a whale.
            rate_limit_delay: Seconds between API calls.
            profiler: Optional WhaleProfiler instance for profile-enriched signals.
        """
        self.top_n_whales = top_n_whales
        self.time_period = time_period
        self.min_pnl = min_pnl
        self.rate_limit_delay = rate_limit_delay
        self.profiler = profiler
        
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketEdge/2.0",
        })
        self._last_request = 0.0
        
        # whale_registry: { proxyWallet_lower -> { pnl, vol, rank, userName } }
        self._whale_registry: dict[str, dict] = {}
        self._registry_loaded = False
    
    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request = time.time()
    
    def _get(self, path: str, params: dict = None) -> list | dict:
        self._throttle()
        resp = self.session.get(
            f"{self.DATA_API_URL}{path}",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    
    def load_whale_registry(self) -> int:
        """
        Fetch the leaderboard and build the whale registry.
        Call once at the start of each pipeline run.
        
        Returns the number of whales loaded.
        """
        try:
            data = self._get("/v1/leaderboard", {
                "limit": min(self.top_n_whales, 50),
                "timePeriod": self.time_period,
                "orderBy": "PNL",
            })
            
            if not isinstance(data, list):
                logger.warning("Leaderboard response was not a list")
                return 0
            
            self._whale_registry = {}
            for entry in data:
                wallet = entry.get("proxyWallet", "").lower()
                pnl = float(entry.get("pnl", 0))
                
                if wallet and pnl >= self.min_pnl:
                    self._whale_registry[wallet] = {
                        "pnl": pnl,
                        "vol": float(entry.get("vol", 0)),
                        "rank": entry.get("rank", "?"),
                        "userName": entry.get("userName", ""),
                    }
            
            self._registry_loaded = True
            profiler_status = f", profiler: {self.profiler.profile_count} profiles" if self.profiler else ""
            logger.info(
                f"🐋 Whale registry loaded: {len(self._whale_registry)} whales "
                f"(top {self.top_n_whales} by PnL, min ${self.min_pnl:,.0f}{profiler_status})"
            )
            return len(self._whale_registry)
        
        except Exception as e:
            logger.warning(f"Failed to load whale registry: {e}")
            self._whale_registry = {}
            return 0
    
    def get_whale_positions(
        self, condition_id: str, market_category: str = ""
    ) -> list[dict]:
        """
        For a given market, find which whales hold positions.
        
        If a WhaleProfiler is attached, each position dict is enriched with:
            - profile_strategy: str (CONVICTION, MARKET_MAKER, etc.)
            - profile_signal_weight: float (0-1)
            - profile_category_credibility: float (0-1)
            - profile_win_rate: float or None
            - profile_primary_category: str
        
        These fields are consumed by profiled_whale_estimator in probability.py
        and by the LLM prompt builder in to_prompt_section().
        
        Args:
            condition_id: The market's condition ID (the Market.id field)
            market_category: Category of the market being evaluated (e.g. "POLITICS")
            
        Returns list of dicts compatible with profiled_whale_estimator:
            [{"side": "YES"/"NO", "size": float, "trader_pnl": float, 
              "whale_name": str, "whale_rank": str, 
              "profile_strategy": str, "profile_signal_weight": float, ...}, ...]
        """
        if not self._registry_loaded:
            self.load_whale_registry()
        
        if not self._whale_registry:
            return []
        
        # Ensure condition_id is 0x-prefixed for the holders endpoint
        if not condition_id.startswith("0x"):
            # Skip non-hex IDs (some markets use numeric IDs from Gamma)
            return []
        
        try:
            holder_data = self._get("/holders", {
                "market": condition_id,
                "limit": 20,
                "minBalance": 100,
            })
        except Exception as e:
            logger.debug(f"Holders fetch failed for {condition_id[:16]}…: {e}")
            return []
        
        whale_positions = []
        
        # holder_data is a list of { "token": "<asset_id>", "holders": [...] }
        for token_group in holder_data:
            holders = token_group.get("holders", [])
            
            for holder in holders:
                wallet = holder.get("proxyWallet", "").lower()
                
                if wallet in self._whale_registry:
                    whale_info = self._whale_registry[wallet]
                    outcome_idx = holder.get("outcomeIndex", 0)
                    side = "YES" if outcome_idx == 0 else "NO"
                    
                    position = {
                        "side": side,
                        "size": float(holder.get("amount", 0)),
                        "trader_pnl": whale_info["pnl"],
                        "whale_name": whale_info.get("userName", holder.get("name", "")),
                        "whale_rank": whale_info["rank"],
                        "whale_volume": whale_info["vol"],
                    }
                    
                    # Enrich with profile data if available
                    if self.profiler:
                        profile = self.profiler.get_profile(wallet)
                        if profile:
                            position["profile_strategy"] = profile.strategy
                            position["profile_signal_weight"] = profile.signal_weight(market_category)
                            position["profile_category_credibility"] = profile.credibility_for_category(market_category)
                            position["profile_win_rate"] = profile.global_win_rate
                            position["profile_primary_category"] = profile.primary_category
                            position["profile_open_positions"] = profile.open_position_count
                            position["profile_strategy_confidence"] = profile.strategy_confidence
                    
                    whale_positions.append(position)
        
        if whale_positions:
            # Log with profile info if available
            signals = []
            for wp in whale_positions[:3]:
                strat = wp.get("profile_strategy", "")
                strat_tag = f" [{strat}]" if strat else ""
                signals.append(f"{wp['side']} ${wp['size']:,.0f}{strat_tag}")
            signals_desc = ", ".join(signals)
            logger.info(
                f"  🐋 Found {len(whale_positions)} whale(s) in market "
                f"(top signals: {signals_desc})"
            )
        
        return whale_positions
    
    @property
    def registry_size(self) -> int:
        return len(self._whale_registry)


# ═══════════════════════════════════════════════════════════
# 6. MASTER ENRICHMENT PIPELINE
# ═══════════════════════════════════════════════════════════

@dataclass
class EnrichedContext:
    """All enrichment data combined into a single object."""
    # Original market data
    market_id: str
    question: str
    market_price: float
    
    # News enrichment
    news: dict = field(default_factory=dict)
    
    # Cross-platform prices
    cross_platform: dict = field(default_factory=dict)
    
    # Economic indicators
    economic_data: dict = field(default_factory=dict)
    
    # Related markets
    related_markets: list[dict] = field(default_factory=list)
    
    # Whale positions
    whale_positions: list[dict] = field(default_factory=list)
    
    # Enrichment metadata
    enrichment_time_seconds: float = 0.0
    sources_used: list[str] = field(default_factory=list)
    
    def to_prompt_section(self) -> str:
        """Format all enrichment data into a prompt section for the LLM."""
        sections = []
        
        # News
        if self.news.get("headlines") or self.news.get("key_facts"):
            sections.append("RECENT NEWS & DEVELOPMENTS:")
            for h in self.news.get("headlines", [])[:4]:
                sections.append(f"  • {h}")
            if self.news.get("key_facts"):
                sections.append("  Key facts:")
                for f in self.news["key_facts"][:4]:
                    sections.append(f"    - {f}")
            if self.news.get("sentiment") != "neutral":
                sections.append(
                    f"  News sentiment: {self.news['sentiment']} "
                    f"(strength: {self.news.get('sentiment_strength', 0):.0%})"
                )
            sections.append("")
        
        # Cross-platform
        if self.cross_platform.get("yes_price"):
            cp = self.cross_platform
            price_diff = cp["yes_price"] - self.market_price
            sections.append("CROSS-PLATFORM PRICING:")
            sections.append(f"  Kalshi price: {cp['yes_price']:.1%} ('{cp.get('title', 'N/A')}')")
            sections.append(f"  Polymarket price: {self.market_price:.1%}")
            sections.append(f"  Price difference: {price_diff:+.1%}")
            if abs(price_diff) > 0.03:
                sections.append(f"  ⚠ Notable cross-platform discrepancy ({abs(price_diff):.1%})")
            sections.append("")
        
        # Economic indicators
        if self.economic_data.get("indicators"):
            sections.append("ECONOMIC INDICATORS:")
            for ind in self.economic_data["indicators"]:
                if "latest_value" in ind:
                    sections.append(
                        f"  {ind['name']}: {ind['latest_value']:.2f} "
                        f"(change: {ind.get('change', 0):+.2f}, {ind.get('trend', 'N/A')})"
                    )
                else:
                    sections.append(f"  {ind['name']}: (relevant indicator — check latest data)")
            sections.append("")
        
        # Related markets
        if self.related_markets:
            sections.append("RELATED MARKETS:")
            for rm in self.related_markets[:5]:
                relation = rm.get("relation", "related")
                sections.append(
                    f"  [{relation}] {rm.get('question', 'N/A')}: "
                    f"{rm.get('price', 0):.0%}"
                )
            sections.append("")
        
        # Whale positions (profile-enriched when available)
        if self.whale_positions:
            yes_whales = [w for w in self.whale_positions if w.get("side") == "YES"]
            no_whales = [w for w in self.whale_positions if w.get("side") == "NO"]
            total_yes = sum(w.get("size", 0) for w in yes_whales)
            total_no = sum(w.get("size", 0) for w in no_whales)
            
            has_profiles = any(w.get("profile_strategy") for w in self.whale_positions)
            
            if has_profiles:
                sections.append("WHALE TRACKER (profiled — strategy type affects signal reliability):")
            else:
                sections.append("WHALE TRACKER (top profitable traders with positions in this market):")
            
            for wp in self.whale_positions[:6]:
                name = wp.get("whale_name", "Anonymous")
                rank = wp.get("whale_rank", "?")
                
                if wp.get("profile_strategy"):
                    # Profile-enriched display
                    strategy = wp["profile_strategy"]
                    signal_w = wp.get("profile_signal_weight", 0)
                    wr = wp.get("profile_win_rate")
                    wr_str = f", win rate: {wr:.0%}" if wr is not None else ""
                    primary = wp.get("profile_primary_category", "")
                    primary_str = f", specializes in: {primary}" if primary else ""
                    
                    # Flag market makers explicitly so the LLM knows to discount
                    if strategy == "MARKET_MAKER":
                        sections.append(
                            f"  #{rank} {name}: {wp['side']} ${wp['size']:,.0f} "
                            f"⚠ MARKET MAKER (signal unreliable — likely providing liquidity, "
                            f"not expressing conviction)"
                        )
                    else:
                        sections.append(
                            f"  #{rank} {name}: {wp['side']} ${wp['size']:,.0f} "
                            f"[{strategy}] signal weight: {signal_w:.2f} "
                            f"(PnL: ${wp['trader_pnl']:,.0f}{wr_str}{primary_str})"
                        )
                else:
                    # Basic display (no profile)
                    sections.append(
                        f"  #{rank} {name}: {wp['side']} ${wp['size']:,.0f} "
                        f"(lifetime PnL: ${wp['trader_pnl']:,.0f})"
                    )
            
            sections.append(f"  Net whale lean: YES ${total_yes:,.0f} vs NO ${total_no:,.0f}")
            if total_yes + total_no > 0:
                pct_yes = total_yes / (total_yes + total_no)
                sections.append(f"  Whale consensus: {pct_yes:.0%} YES / {1-pct_yes:.0%} NO")
            
            # Add profile-aware summary for the LLM
            if has_profiles:
                conviction_whales = [
                    w for w in self.whale_positions
                    if w.get("profile_strategy") in ("CONVICTION", "SPECIALIST")
                    and w.get("profile_signal_weight", 0) > 0.2
                ]
                mm_whales = [
                    w for w in self.whale_positions
                    if w.get("profile_strategy") == "MARKET_MAKER"
                ]
                if conviction_whales:
                    conv_yes = sum(1 for w in conviction_whales if w.get("side") == "YES")
                    conv_no = len(conviction_whales) - conv_yes
                    sections.append(
                        f"  → {len(conviction_whales)} conviction/specialist trader(s) "
                        f"({conv_yes} YES, {conv_no} NO) — these are the most reliable signals"
                    )
                if mm_whales:
                    sections.append(
                        f"  → {len(mm_whales)} market maker(s) detected — discount their positions"
                    )
            sections.append("")
        
        if not sections:
            return "ADDITIONAL CONTEXT: No additional data sources available for this market."
        
        return "\n".join(sections)
    
    def summary(self) -> str:
        """Short summary of what was enriched."""
        parts = []
        if self.news.get("headlines"):
            parts.append(f"{len(self.news['headlines'])} news items")
        if self.cross_platform.get("yes_price"):
            parts.append("Kalshi price")
        if self.economic_data.get("indicators"):
            parts.append(f"{len(self.economic_data['indicators'])} econ indicators")
        if self.related_markets:
            parts.append(f"{len(self.related_markets)} related markets")
        if self.whale_positions:
            parts.append(f"{len(self.whale_positions)} whale positions")
        
        return (
            f"Enriched from {len(self.sources_used)} sources "
            f"({', '.join(parts) or 'none'}) "
            f"in {self.enrichment_time_seconds:.1f}s"
        )


class ContextEnricher:
    """
    Master orchestrator that runs all enrichment sources in sequence
    and combines results.
    
    Usage:
        enricher = ContextEnricher()
        enriched = enricher.enrich(market_data)
        print(enriched.to_prompt_section())
    
    With whale profiling:
        from core.whale_profiler import WhaleProfiler
        profiler = WhaleProfiler()
        profiler.build_profiles()
        enricher = ContextEnricher(whale_profiler=profiler)
    """
    
    def __init__(
        self,
        enable_news: bool = True,
        enable_kalshi: bool = True,
        enable_fred: bool = True,
        enable_related: bool = True,
        enable_whales: bool = True,
        anthropic_api_key: Optional[str] = None,
        fred_api_key: Optional[str] = None,
        whale_profiler: "WhaleProfiler | None" = None,
    ):
        self.news_enricher = NewsEnricher(api_key=anthropic_api_key) if enable_news else None
        self.kalshi_enricher = KalshiPriceEnricher() if enable_kalshi else None
        self.fred_enricher = EconomicEnricher(api_key=fred_api_key) if enable_fred else None
        self.related_enricher = RelatedMarketsEnricher() if enable_related else None
        self.whale_enricher = WhaleEnricher(profiler=whale_profiler) if enable_whales else None
        self.whale_profiler = whale_profiler
        
        # Pre-load the whale registry so it's ready for the pipeline loop
        if self.whale_enricher:
            self.whale_enricher.load_whale_registry()
    
    def enrich(
        self,
        market_data: dict,
        all_markets: list[dict] = None,
    ) -> EnrichedContext:
        """
        Run all enrichment sources for a market.
        
        Args:
            market_data: Dict with at minimum 'question', 'market_price', 'market_id'
            all_markets: Optional list of all active markets (for related market search)
        
        Returns:
            EnrichedContext with all gathered data
        """
        start = time.time()
        sources_used = []
        
        question = market_data.get("question", "")
        market_price = float(market_data.get("market_price", market_data.get("yes_price", 0.5)))
        market_id = market_data.get("market_id", market_data.get("id", ""))
        
        ctx = EnrichedContext(
            market_id=str(market_id),
            question=question,
            market_price=market_price,
        )
        
        # 1. News search
        if self.news_enricher:
            logger.info(f"  📰 Searching news for: {question[:50]}...")
            ctx.news = self.news_enricher.search(
                question, market_data.get("category", "")
            )
            if ctx.news.get("headlines"):
                sources_used.append("news")
        
        # 2. Kalshi cross-platform price
        if self.kalshi_enricher:
            logger.info(f"  💱 Checking Kalshi prices...")
            match = self.kalshi_enricher.find_matching_market(question)
            if match:
                ctx.cross_platform = match
                sources_used.append("kalshi")
        
        # 3. Economic indicators
        if self.fred_enricher:
            logger.info(f"  📊 Fetching economic indicators...")
            ctx.economic_data = self.fred_enricher.get_relevant_indicators(question)
            if ctx.economic_data.get("indicators"):
                sources_used.append("fred")
        
        # 4. Related markets
        if self.related_enricher:
            logger.info(f"  🔗 Finding related markets...")
            ctx.related_markets = self.related_enricher.find_related(
                market_data, all_markets
            )
            if ctx.related_markets:
                sources_used.append("related_markets")
        
        # 5. Whale positions (skip if already pre-fetched in context dict)
        pre_fetched_whales = market_data.get("whale_positions", [])
        if pre_fetched_whales:
            ctx.whale_positions = pre_fetched_whales
            sources_used.append("whales")
        elif self.whale_enricher and self.whale_enricher.registry_size > 0:
            logger.info(f"  🐋 Checking whale positions...")
            market_category = market_data.get("category", "")
            ctx.whale_positions = self.whale_enricher.get_whale_positions(
                str(market_id), market_category=market_category
            )
            if ctx.whale_positions:
                sources_used.append("whales")
        
        ctx.sources_used = sources_used
        ctx.enrichment_time_seconds = time.time() - start
        
        logger.info(f"  ✅ {ctx.summary()}")
        
        return ctx


# ═══════════════════════════════════════════════════════════
# 6. UPDATED LLM PROMPT WITH ENRICHED CONTEXT
# ═══════════════════════════════════════════════════════════

def build_enriched_forecast_prompt(
    market_context_str: str,
    enriched_context: EnrichedContext,
) -> str:
    """
    Build the full LLM forecaster prompt with enriched context injected.
    
    This replaces the basic prompt in llm_estimator.py when enrichment
    data is available.
    """
    enrichment_section = enriched_context.to_prompt_section()
    
    return f"""Analyze this prediction market and estimate the true probability of the outcome.
You have access to market data AND additional research context gathered from 
multiple sources. Use ALL available information to form your estimate.

{market_context_str}

{enrichment_section}

IMPORTANT INSTRUCTIONS:
- The news and cross-platform data above is REAL and CURRENT. Weight it heavily.
- If cross-platform prices differ significantly, consider which platform's traders 
  are likely more informed for this specific topic.
- Economic indicators provide context for macro-related markets — interpret their 
  trend and level, not just the number.
- Related market prices should be logically consistent with your estimate.
- WHALE SIGNALS: If whale positions include strategy profiles, weight them accordingly:
  * CONVICTION and SPECIALIST traders with high signal weights are informative — 
    their positions likely reflect genuine analysis or private information.
  * MARKET MAKER positions are noise — they provide liquidity on both sides and 
    their positions do NOT indicate directional conviction. Discount heavily.
  * Pay attention to category specialization — a politics specialist's signal on 
    a politics market is far more valuable than their signal on a crypto market.
  * If multiple conviction traders agree on a direction, that's a strong signal.

Respond with ONLY this JSON structure, no other text:
{{
    "base_rate_anchor": <float between 0 and 1>,
    "base_rate_reasoning": "<1-2 sentences>",
    "news_impact": "<how does the recent news affect the probability? 1-2 sentences>",
    "cross_platform_note": "<any cross-platform signal? 1 sentence, or 'N/A'>",
    "whale_signal_note": "<how did you weight the whale signals? Which traders were most credible for this market? 1-2 sentences, or 'N/A'>",
    "factors_for": ["<factor 1 supporting YES>", "<factor 2>", ...],
    "factors_against": ["<factor 1 supporting NO>", "<factor 2>", ...],
    "probability": <float between 0.01 and 0.99>,
    "reasoning": "<3-5 sentence explanation integrating all sources>",
    "confidence": "<low|medium|high>",
    "information_edge": "<where you think the market might be wrong, or 'none'>"
}}"""


# ═══════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════

def demo():
    """Demo the enrichment pipeline."""
    print("\n" + "=" * 70)
    print("  CONTEXT ENRICHMENT PIPELINE — Demo")
    print("=" * 70)
    
    # Demo markets
    markets = [
        {
            "market_id": "fed-rate-june",
            "question": "Will the Fed cut rates in June 2026?",
            "market_price": 0.42,
            "category": "economics",
            "volume_24h": 85000,
            "event_slug": "",
        },
        {
            "market_id": "nvidia-earnings",
            "question": "Will NVIDIA beat Q2 2026 earnings estimates?",
            "market_price": 0.73,
            "category": "corporate",
            "volume_24h": 65000,
            "event_slug": "",
        },
        {
            "market_id": "recession-2026",
            "question": "Will there be a US recession in 2026?",
            "market_price": 0.18,
            "category": "economics",
            "volume_24h": 95000,
            "event_slug": "",
        },
    ]
    
    # Create enricher (news/Kalshi/FRED will gracefully degrade without keys)
    enricher = ContextEnricher(
        enable_news=bool(os.environ.get("ANTHROPIC_API_KEY")),
        enable_kalshi=True,   # Public API, no key needed
        enable_fred=bool(os.environ.get("FRED_API_KEY")),
        enable_related=True,  # Public API
    )
    
    api_status = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        api_status.append("✅ News search (Claude)")
    else:
        api_status.append("⬜ News search (set ANTHROPIC_API_KEY)")
    
    api_status.append("✅ Kalshi cross-platform (no key needed)")
    
    if os.environ.get("FRED_API_KEY"):
        api_status.append("✅ FRED economic data")
    else:
        api_status.append("⬜ FRED live data (set FRED_API_KEY)")
    
    api_status.append("✅ Related markets (no key needed)")
    
    print("\nData sources:")
    for s in api_status:
        print(f"  {s}")
    
    for m in markets:
        print(f"\n{'─' * 60}")
        print(f"📋 {m['question']}")
        print(f"   Market price: {m['market_price']:.0%}")
        print(f"{'─' * 60}")
        
        enriched = enricher.enrich(m, all_markets=markets)
        
        print(f"\n{enriched.summary()}")
        
        # Show what would be added to the LLM prompt
        prompt_section = enriched.to_prompt_section()
        if "No additional data" not in prompt_section:
            print(f"\nPrompt injection ({len(prompt_section)} chars):")
            # Indent and truncate for display
            for line in prompt_section.split("\n")[:15]:
                print(f"  │ {line}")
            if prompt_section.count("\n") > 15:
                print(f"  │ ... ({prompt_section.count(chr(10)) - 15} more lines)")
        else:
            print(f"\n  {prompt_section}")
    
    # Show integration example
    print(f"\n\n{'=' * 70}")
    print("  INTEGRATION WITH LLM ESTIMATOR")
    print(f"{'=' * 70}")
    print("""
  from core.context_enricher import ContextEnricher, build_enriched_forecast_prompt
  from core.llm_estimator import LLMEstimator, build_market_context
  
  # Set up enricher
  enricher = ContextEnricher()
  
  # For each market in the pipeline:
  enriched = enricher.enrich(market_data, all_markets=all_active_markets)
  
  # Build the market context (from llm_estimator.py)
  market_ctx = build_market_context(market_data, order_book, price_history)
  
  # Build the enriched prompt
  prompt = build_enriched_forecast_prompt(
      market_ctx.to_prompt_context(),
      enriched,
  )
  
  # Feed to Claude for probability estimation
  raw_response = call_claude(prompt)
  
  # The LLM now sees: market data + news + Kalshi prices + FRED data + related markets
  # This is ~3-5x more context than the basic estimator had!
    """)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    demo()