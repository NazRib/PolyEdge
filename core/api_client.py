"""
Polymarket API Client
Unified interface to Gamma (market discovery), CLOB (pricing/orderbook), 
and Data (user analytics) APIs.

All read endpoints are public — no authentication needed for scanning.
Trading endpoints require wallet auth (Phase 2).
"""

import time
import logging
import requests
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Data Classes ────────────────────────────────────────

@dataclass
class Market:
    """A single binary outcome market on Polymarket."""
    id: str
    question: str
    slug: str
    description: str
    outcome_prices: list[float]       # [yes_price, no_price]
    outcomes: list[str]               # ["Yes", "No"]
    token_ids: list[str]              # CLOB token IDs for each outcome
    volume_24h: float
    volume_total: float
    liquidity: float
    end_date: Optional[datetime]
    active: bool
    closed: bool
    category: str
    tags: list[str] = field(default_factory=list)
    event_slug: str = ""
    event_title: str = ""
    clob_rewards: float = 0.0         # Maker rewards if any
    spread: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    
    @property
    def yes_price(self) -> float:
        return self.outcome_prices[0] if self.outcome_prices else 0.5
    
    @property
    def no_price(self) -> float:
        return self.outcome_prices[1] if len(self.outcome_prices) > 1 else 1 - self.yes_price
    
    @property
    def implied_probability(self) -> float:
        """Market-implied probability of 'Yes' outcome."""
        return self.yes_price
    
    @property
    def days_to_resolution(self) -> Optional[float]:
        if self.end_date is None:
            return None
        delta = self.end_date - datetime.now(timezone.utc)
        return max(0, delta.total_seconds() / 86400)


@dataclass
class OrderBookLevel:
    price: float
    size: float

@dataclass
class OrderBook:
    """Full order book for a market outcome."""
    token_id: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    midpoint: float
    spread: float
    
    @property
    def bid_depth(self) -> float:
        return sum(level.size for level in self.bids)
    
    @property
    def ask_depth(self) -> float:
        return sum(level.size for level in self.asks)
    
    @property
    def total_depth(self) -> float:
        return self.bid_depth + self.ask_depth


@dataclass
class TraderProfile:
    """Public profile of a trader on Polymarket."""
    address: str
    name: str
    profit_loss: float
    volume: float
    positions_count: int
    markets_traded: int


# ─── API Client ──────────────────────────────────────────

class PolymarketClient:
    """
    Unified client for Polymarket's public APIs.
    
    Usage:
        client = PolymarketClient()
        markets = client.get_active_markets()
        book = client.get_order_book(token_id)
    """
    
    def __init__(
        self,
        gamma_url: str = "https://gamma-api.polymarket.com",
        clob_url: str = "https://clob.polymarket.com",
        data_url: str = "https://data-api.polymarket.com",
        rate_limit_delay: float = 0.5,
    ):
        self.gamma_url = gamma_url
        self.clob_url = clob_url
        self.data_url = data_url
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketEdge/1.0",
        })
        self._last_request_time = 0.0
    
    def _throttle(self):
        """Simple rate limiter."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _get(self, base_url: str, path: str, params: dict = None) -> dict | list:
        """Make a throttled GET request."""
        self._throttle()
        url = f"{base_url}{path}"
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {url} — {e}")
            raise
    
    # ─── Gamma API: Market Discovery ─────────────────────
    
    def get_active_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        order: str = "volume24hr",
        ascending: bool = False,
    ) -> list[Market]:
        """
        Fetch active markets from Gamma API.
        
        Args:
            limit: Max markets to return (API max ~100)
            offset: Pagination offset
            order: Sort field — 'volume24hr', 'liquidity', 'endDate', 'startDate'
            ascending: Sort direction
        
        Returns:
            List of Market objects
        """
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        
        raw_markets = self._get(self.gamma_url, "/markets", params)
        
        # Handle both list and paginated dict responses
        if isinstance(raw_markets, dict):
            raw_markets = raw_markets.get("data", raw_markets.get("markets", []))
        
        markets = []
        for m in raw_markets:
            try:
                markets.append(self._parse_market(m))
            except Exception as e:
                logger.warning(f"Failed to parse market: {e}")
                continue
        
        return markets
    
    def get_all_active_markets(self, max_pages: int = 20) -> list[Market]:
        """Paginate through all active markets."""
        all_markets = []
        for page in range(max_pages):
            batch = self.get_active_markets(limit=100, offset=page * 100)
            if not batch:
                break
            all_markets.extend(batch)
            logger.info(f"Fetched page {page + 1}: {len(batch)} markets (total: {len(all_markets)})")
        return all_markets
    
    def get_event(self, slug: str) -> dict:
        """Fetch an event with all its markets by slug."""
        return self._get(self.gamma_url, f"/events/slug/{slug}")

    def get_events_list(
        self,
        limit: int = 50,
        offset: int = 0,
        closed: bool = False,
        order: str = "id",
        ascending: bool = False,
    ) -> list[dict]:
        """
        Fetch events from the Gamma /events endpoint.

        Each event dict contains a nested 'markets' list with all
        sub-markets belonging to that event.

        Args:
            limit: Max events per page (API max ~50)
            offset: Pagination offset
            closed: Include closed events
            order: Sort field ('id', 'volume', 'liquidity', 'startDate', 'endDate')
            ascending: Sort direction

        Returns:
            List of raw event dicts (each with nested 'markets' array)
        """
        params = {
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        raw = self._get(self.gamma_url, "/events", params)
        if isinstance(raw, dict):
            raw = raw.get("data", raw.get("events", []))
        if not isinstance(raw, list):
            return []
        return raw
    
    def get_market_by_slug(self, slug: str) -> Optional[Market]:
        """Fetch a single market by its slug."""
        try:
            data = self._get(self.gamma_url, "/markets", {"slug": slug})
            markets = data if isinstance(data, list) else [data]
            if markets:
                return self._parse_market(markets[0])
        except Exception as e:
            logger.debug(f"Market fetch by slug failed: {e}")
        return None
    
    def get_market_by_id(self, market_id: str) -> Optional[Market]:
        """
        Fetch a single market by its condition ID.
        Useful for checking resolution status of tracked trades.
        """
        try:
            data = self._get(self.gamma_url, "/markets", {"id": market_id})
            markets = data if isinstance(data, list) else [data]
            if markets:
                return self._parse_market(markets[0])
        except Exception as e:
            logger.debug(f"Market fetch by ID failed: {e}")
        return None
    
    def search_markets(self, query: str, limit: int = 20) -> list[Market]:
        """Search markets by text query."""
        params = {"active": "true", "closed": "false", "limit": limit}
        # Gamma API supports text search via the query param or tag filtering
        raw = self._get(self.gamma_url, "/markets", {**params, "tag": query})
        if isinstance(raw, dict):
            raw = raw.get("data", [])
        return [self._parse_market(m) for m in raw if m]
    
    def _parse_market(self, raw: dict) -> Market:
        """Parse a raw API market object into our Market dataclass."""
        # Parse outcome prices (comes as JSON string or list)
        outcome_prices = raw.get("outcomePrices", "[]")
        if isinstance(outcome_prices, str):
            import json
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, TypeError):
                outcome_prices = [0.5, 0.5]
        outcome_prices = [float(p) for p in outcome_prices]
        
        # Parse outcomes
        outcomes = raw.get("outcomes", '["Yes","No"]')
        if isinstance(outcomes, str):
            import json
            try:
                outcomes = json.loads(outcomes)
            except (json.JSONDecodeError, TypeError):
                outcomes = ["Yes", "No"]
        
        # Parse token IDs
        token_ids = raw.get("clobTokenIds", "[]")
        if isinstance(token_ids, str):
            import json
            try:
                token_ids = json.loads(token_ids)
            except (json.JSONDecodeError, TypeError):
                token_ids = []
        
        # Parse end date
        end_date = None
        end_str = raw.get("endDate") or raw.get("end_date_iso")
        if end_str:
            try:
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        
        # Parse tags
        tags = []
        raw_tags = raw.get("tags", [])
        if isinstance(raw_tags, list):
            for t in raw_tags:
                if isinstance(t, dict):
                    tags.append(t.get("label", t.get("slug", "")))
                elif isinstance(t, str):
                    tags.append(t)
        
        return Market(
            id=str(raw.get("id", raw.get("conditionId", ""))),
            question=raw.get("question", raw.get("title", "")),
            slug=raw.get("slug", ""),
            description=raw.get("description", ""),
            outcome_prices=outcome_prices,
            outcomes=outcomes,
            token_ids=token_ids,
            volume_24h=float(raw.get("volume24hr", 0) or 0),
            volume_total=float(raw.get("volume", 0) or 0),
            liquidity=float(raw.get("liquidity", 0) or 0),
            end_date=end_date,
            active=bool(raw.get("active", True)),
            closed=bool(raw.get("closed", False)),
            category=raw.get("category", raw.get("groupItemTitle", "")),
            tags=tags,
            event_slug=raw.get("eventSlug", ""),
            event_title=raw.get("eventTitle", raw.get("groupItemTitle", "")),
        )
    
    # ─── CLOB API: Pricing & Order Book ──────────────────
    
    def get_midpoint(self, token_id: str) -> float:
        """Get the midpoint price for a token."""
        data = self._get(self.clob_url, "/midpoint", {"token_id": token_id})
        return float(data.get("mid", 0.5))
    
    def get_price(self, token_id: str, side: str = "BUY") -> float:
        """Get the best available price for a token on a given side."""
        data = self._get(self.clob_url, "/price", {
            "token_id": token_id,
            "side": side
        })
        return float(data.get("price", 0.5))
    
    def get_order_book(self, token_id: str) -> OrderBook:
        """Get full order book for a token."""
        data = self._get(self.clob_url, "/book", {"token_id": token_id})
        
        bids = [
            OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in data.get("asks", [])
        ]
        
        # Calculate midpoint and spread
        best_bid = bids[0].price if bids else 0
        best_ask = asks[0].price if asks else 1
        midpoint = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        return OrderBook(
            token_id=token_id,
            bids=bids,
            asks=asks,
            midpoint=midpoint,
            spread=spread,
        )
    
    def get_price_history(self, token_id: str, interval: str = "1d") -> list[dict]:
        """
        Get recent price history for a token.
        
        Args:
            token_id: The CLOB token ID
            interval: One of '1h', '6h', '1d', '1w', '1m', 'max'
            
        Returns empty list on 400 errors (resolved/delisted markets).
        """
        try:
            data = self._get(self.clob_url, "/prices-history", {
                "market": token_id,
                "interval": interval,
            })
            return data if isinstance(data, list) else data.get("history", [])
        except requests.RequestException as e:
            if "400" in str(e):
                logger.debug(f"No price history for token {token_id[:20]}… (400)")
            else:
                logger.warning(f"Price history fetch failed: {e}")
            return []
    
    # ─── Data API: User & Whale Analytics ────────────────
    
    def get_top_traders(
        self,
        limit: int = 50,
        time_period: str = "ALL",
        category: str = "OVERALL",
        order_by: str = "PNL",
    ) -> list[dict]:
        """
        Fetch top traders from the leaderboard.
        
        Args:
            limit: Max traders to return (1-50)
            time_period: DAY, WEEK, MONTH, or ALL
            category: OVERALL, POLITICS, SPORTS, CRYPTO, ECONOMICS, etc.
            order_by: PNL or VOL
            
        Returns list of dicts with: rank, proxyWallet, userName, vol, pnl, etc.
        """
        try:
            data = self._get(self.data_url, "/v1/leaderboard", {
                "limit": min(limit, 50),
                "timePeriod": time_period,
                "category": category,
                "orderBy": order_by,
            })
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Leaderboard fetch failed: {e}")
            return []
    
    def get_market_holders(
        self,
        condition_id: str,
        limit: int = 20,
        min_balance: int = 100,
    ) -> list[dict]:
        """
        Get top token holders for a specific market.
        
        Args:
            condition_id: The market's condition ID (0x-prefixed)
            limit: Max holders per token (1-20)
            min_balance: Minimum token balance to include
            
        Returns list of dicts per token, each with 'token' and 'holders' list.
        Holders have: proxyWallet, amount, name, pseudonym, outcomeIndex, etc.
        """
        try:
            data = self._get(self.data_url, "/holders", {
                "market": condition_id,
                "limit": min(limit, 20),
                "minBalance": min_balance,
            })
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Market holders fetch failed: {e}")
            return []
    
    def get_trader_positions(self, address: str, limit: int = 100) -> list[dict]:
        """Get all positions for a given wallet address."""
        try:
            data = self._get(self.data_url, "/positions", {
                "user": address,
                "limit": limit,
                "sizeThreshold": 1,
            })
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Trader positions fetch failed: {e}")
            return []
    
    def get_trader_activity(
        self, address: str, market: str = None, limit: int = 50
    ) -> list[dict]:
        """Get trade activity for a wallet, optionally filtered by market."""
        params = {"user": address, "limit": limit}
        if market:
            params["market"] = market
        data = self._get(self.data_url, "/activity", params)
        return data if isinstance(data, list) else data.get("activity", [])
    
    def get_market_trades(self, condition_id: str, limit: int = 100) -> list[dict]:
        """Get recent trades for a specific market."""
        data = self._get(self.data_url, "/activity", {
            "market": condition_id,
            "limit": limit,
        })
        return data if isinstance(data, list) else []


# ─── Convenience Functions ───────────────────────────────

def quick_scan(n: int = 10) -> list[Market]:
    """Quick scan: fetch top N markets by 24h volume."""
    client = PolymarketClient()
    markets = client.get_active_markets(limit=n, order="volume24hr")
    for m in markets:
        print(f"  {m.yes_price:.0%} | Vol: ${m.volume_24h:,.0f} | {m.question[:70]}")
    return markets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n🔍 Top Polymarket markets by 24h volume:\n")
    quick_scan(15)