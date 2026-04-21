"""
Live Trade Executor for Polymarket CLOB
Executes real trades via the Polymarket CLOB API with comprehensive safety controls.

Exposes the same enter_trade() / resolve_trade() interface as PaperTrader so
the weather scanner can use either without knowing which mode it's in.

Safety controls:
    - Pre-flight balance check before every order
    - Order book depth verification (skips thin books)
    - Daily loss limit circuit breaker
    - Aggressive limit orders (maker, 0% fee) with timeout + cancel
    - Full audit trail in data/weather/live_trades.json
    - Dry-run mode for validation without real execution

Required:
    pip install py-clob-client
    Set POLYMARKET_PRIVATE_KEY (and optionally POLYMARKET_FUNDER)
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

from core.kelly import PositionSize
from core.probability import ProbabilityEstimate

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════

@dataclass
class LiveTrade:
    """
    A live trade with full order lifecycle tracking.

    Mirrors PaperTrade's interface so the scanner can treat both identically,
    but adds CLOB-specific fields for order management and audit.
    """
    # ─── PaperTrade-compatible fields ─────────────────────
    trade_id: str
    market_id: str
    question: str
    side: str                        # "YES" or "NO"
    entry_price: float               # Intended entry price (limit price)
    shares: float                    # Intended shares
    dollar_amount: float             # Intended $ amount
    estimated_probability: float
    market_price_at_entry: float
    edge_at_entry: float
    confidence: float

    timestamp: str = ""
    status: str = "PENDING"          # PENDING, OPEN, FILLED, PARTIAL, CANCELLED, RESOLVED_WIN, RESOLVED_LOSS
    exit_price: Optional[float] = None
    resolution: Optional[bool] = None
    pnl: float = 0.0
    strategy_tag: str = "weather_live"
    model_tag: str = ""

    # ─── CLOB-specific fields ────────────────────────────
    token_id: str = ""               # CLOB token ID for the YES outcome
    order_id: str = ""               # CLOB order ID after submission
    order_type: str = "GTC"          # GTC (limit/maker) or FOK (market/taker)
    neg_risk: bool = True            # Temperature markets are multi-outcome = neg-risk
    tick_size: str = "0.01"

    fill_price: Optional[float] = None    # Actual fill price (may differ from limit)
    fill_shares: float = 0.0              # Actually filled shares
    fill_amount: float = 0.0              # Actually filled $ amount
    fees_paid: float = 0.0                # Taker fees if any

    # Audit trail
    order_submitted_at: str = ""
    order_filled_at: str = ""
    order_cancelled_at: str = ""
    cancel_reason: str = ""
    error_message: str = ""

    # Pre-flight snapshot
    book_best_bid: Optional[float] = None
    book_best_ask: Optional[float] = None
    book_bid_depth: float = 0.0
    book_ask_depth: float = 0.0
    book_spread: float = 0.0

    def resolve(self, outcome: bool) -> float:
        """Resolve the trade with the actual outcome. Same logic as PaperTrade."""
        self.resolution = outcome
        # Use actual fill data if available, otherwise intended
        amount = self.fill_amount if self.fill_amount > 0 else self.dollar_amount
        shares = self.fill_shares if self.fill_shares > 0 else self.shares

        if self.side == "YES":
            if outcome:
                self.pnl = shares * 1.0 - amount - self.fees_paid
                self.status = "RESOLVED_WIN"
            else:
                self.pnl = -amount - self.fees_paid
                self.status = "RESOLVED_LOSS"
        else:
            if not outcome:
                self.pnl = shares * 1.0 - amount - self.fees_paid
                self.status = "RESOLVED_WIN"
            else:
                self.pnl = -amount - self.fees_paid
                self.status = "RESOLVED_LOSS"

        return self.pnl


@dataclass
class ExecutionResult:
    """Result of an order execution attempt."""
    success: bool
    trade: Optional[LiveTrade] = None
    order_id: str = ""
    fill_price: float = 0.0
    fill_shares: float = 0.0
    error: str = ""
    skipped_reason: str = ""


# ═══════════════════════════════════════════════════════════
# LIVE EXECUTOR
# ═══════════════════════════════════════════════════════════

class LiveExecutor:
    """
    Executes real trades on Polymarket via the CLOB API.

    Drop-in replacement for PaperTrader in the weather scanner pipeline.
    The scanner calls enter_trade() with (estimate, position) exactly as
    it does for paper trading — this class handles the CLOB order lifecycle.

    Usage:
        from core.live_executor import LiveExecutor
        from core.wallet import create_clob_client

        client = create_clob_client()
        executor = LiveExecutor(client, bankroll=500, dry_run=True)

        # Same interface as PaperTrader
        trade = executor.enter_trade(estimate, position)
        executor.resolve_trade(trade.trade_id, outcome=True)
    """

    def __init__(
        self,
        clob_client,
        bankroll: float = 500.0,
        data_dir: str = "data/weather",
        dry_run: bool = True,
        daily_loss_limit: float = 100.0,
        order_timeout_seconds: int = 120,
        min_book_depth: float = 100.0,
        tick_size: str = "0.01",
    ):
        """
        Args:
            clob_client: Authenticated ClobClient from core.wallet.
            bankroll: Total capital allocation. Not queried on-chain — this is
                      your self-imposed limit for the weather strategy.
            data_dir: Where to persist live trade state.
            dry_run: If True, simulates everything but does not submit orders.
            daily_loss_limit: Hard stop if cumulative daily losses exceed this.
            order_timeout_seconds: Cancel unfilled GTC orders after this many seconds.
            min_book_depth: Minimum total depth (bid+ask in $) to consider the
                           book liquid enough to trade.
            tick_size: Minimum price increment. "0.01" for most markets.
        """
        self.client = clob_client
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        self.dry_run = dry_run
        self.daily_loss_limit = daily_loss_limit
        self.order_timeout_seconds = order_timeout_seconds
        self.min_book_depth = min_book_depth
        self.tick_size = tick_size

        self.trades: list[LiveTrade] = []
        self._trade_counter = 0
        self._daily_pnl: dict[str, float] = {}  # date_str → cumulative pnl

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.data_dir / "live_trades.json"
        self._load_state()

        mode_str = "DRY RUN" if dry_run else "LIVE"
        logger.info(
            f"LiveExecutor initialized ({mode_str}) — "
            f"bankroll=${bankroll:.2f}, daily_loss_limit=${daily_loss_limit:.2f}"
        )

    # ═══════════════════════════════════════════════════════
    # PUBLIC INTERFACE (PaperTrader-compatible)
    # ═══════════════════════════════════════════════════════

    def enter_trade(
        self,
        estimate: ProbabilityEstimate,
        position: PositionSize,
        strategy_tag: str = "weather_live",
        model_tag: str = "",
    ) -> Optional[LiveTrade]:
        """
        Execute a trade. Same signature as PaperTrader.enter_trade().

        Pipeline:
            1. Validate position
            2. Check circuit breakers (daily loss, bankroll)
            3. Fetch order book → verify depth and current price
            4. Re-verify edge hasn't evaporated since scan
            5. Compute limit price (aggressive maker: best_ask - 1 tick)
            6. Place GTC limit order (or simulate in dry-run)
            7. Monitor fill with timeout
            8. Log result
        """
        if not position.should_trade:
            logger.debug(f"Trade rejected by sizer: {position.rejection_reason}")
            return None

        # ─── Circuit breakers ────────────────────────────
        if position.dollar_amount > self.bankroll:
            logger.warning(
                f"Insufficient bankroll: need ${position.dollar_amount:.2f}, "
                f"have ${self.bankroll:.2f}"
            )
            return None

        if self._daily_loss_exceeded():
            logger.warning(
                f"Daily loss limit hit (${self.daily_loss_limit:.2f}). "
                f"No more trades today."
            )
            return None

        # ─── Pre-flight: balance check (live only) ───────
        if not self.dry_run:
            from core.wallet import get_usdc_balance
            balance = get_usdc_balance(self.client)
            if balance is not None and balance < position.dollar_amount:
                logger.warning(
                    f"Insufficient USDC: need ${position.dollar_amount:.2f}, "
                    f"have ${balance:.2f} on-chain"
                )
                return None

        # ─── Extract token_id ────────────────────────────
        token_id = ""
        components = getattr(estimate, "components", {}) or {}
        if isinstance(components, dict):
            token_id = components.get("token_id", "")

        if not token_id:
            # Try to get it from the estimate market_id context
            # The scanner stores token_id in the edge_info dict — it should be
            # passed through ProbabilityEstimate.components
            logger.error(
                f"No token_id found for {estimate.question[:50]}. "
                f"Cannot place CLOB order."
            )
            return None

        # ─── Pre-flight: order book check ────────────────
        book_snapshot = self._check_order_book(token_id)
        if book_snapshot is None:
            logger.warning(f"Could not fetch order book for {token_id[:20]}…")
            return None

        best_bid, best_ask, bid_depth, ask_depth, spread = book_snapshot

        total_depth = bid_depth + ask_depth
        if total_depth < self.min_book_depth:
            logger.info(
                f"Book too thin: ${total_depth:.0f} depth < "
                f"${self.min_book_depth:.0f} minimum. Skipping."
            )
            return None

        # ─── Re-verify edge at current price ─────────────
        current_price = (best_bid + best_ask) / 2 if best_bid and best_ask else estimate.market_price
        current_edge = estimate.probability - current_price
        if current_edge < 0.05:
            logger.info(
                f"Edge evaporated: was {estimate.edge:+.1%}, "
                f"now {current_edge:+.1%} at midpoint {current_price:.2f}. Skipping."
            )
            return None

        # ─── Compute limit price ─────────────────────────
        # Strategy: aggressive maker — price at best_ask - 1 tick.
        # This gives us maker status (0% fee) while being likely to fill
        # because we're at the top of the bid queue.
        # Fallback: if spread ≤ 2 ticks, the maker advantage is minimal
        # and we may not fill, so use FOK market order instead.
        tick = float(self.tick_size)
        tight_spread = spread <= 2 * tick

        if tight_spread:
            # Spread is very tight — use market order for guaranteed fill
            limit_price = best_ask
            order_type = "FOK"
        else:
            # Place aggressive limit order just below the best ask
            limit_price = round(best_ask - tick, 2)
            # Ensure our limit is at least at the midpoint
            limit_price = max(limit_price, round(current_price, 2))
            order_type = "GTC"

        # Compute shares at our limit price
        shares = position.dollar_amount / limit_price if limit_price > 0 else 0
        shares = round(shares, 2)

        if shares < 1.0:
            logger.info(f"Position too small: {shares:.2f} shares. Skipping.")
            return None

        # ─── Build trade record ──────────────────────────
        self._trade_counter += 1
        trade = LiveTrade(
            trade_id=f"LT-{self._trade_counter:04d}",
            market_id=estimate.market_id,
            question=estimate.question,
            side=position.side,
            entry_price=limit_price,
            shares=shares,
            dollar_amount=position.dollar_amount,
            estimated_probability=estimate.probability,
            market_price_at_entry=current_price,
            edge_at_entry=current_edge,
            confidence=estimate.confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            token_id=token_id,
            order_type=order_type,
            neg_risk=True,  # Temperature markets are always neg-risk
            tick_size=self.tick_size,
            book_best_bid=best_bid,
            book_best_ask=best_ask,
            book_bid_depth=bid_depth,
            book_ask_depth=ask_depth,
            book_spread=spread,
            strategy_tag=strategy_tag,
            model_tag=model_tag,
        )

        # ─── Execute ─────────────────────────────────────
        if self.dry_run:
            result = self._execute_dry_run(trade)
        else:
            result = self._execute_live(trade)

        if not result.success:
            trade.status = "CANCELLED"
            trade.cancel_reason = result.error or result.skipped_reason
            trade.order_cancelled_at = datetime.now(timezone.utc).isoformat()
            self.trades.append(trade)
            self._save_state()
            logger.warning(f"Order failed: {result.error}")
            return None

        # ─── Record successful trade ─────────────────────
        trade.order_id = result.order_id
        trade.fill_price = result.fill_price or limit_price
        trade.fill_shares = result.fill_shares or shares
        trade.fill_amount = trade.fill_price * trade.fill_shares
        trade.status = "FILLED" if not self.dry_run else "OPEN"
        trade.order_filled_at = datetime.now(timezone.utc).isoformat()

        self.bankroll -= trade.fill_amount
        self.trades.append(trade)
        self._save_state()

        mode = "🏦 LIVE" if not self.dry_run else "🔍 DRY"
        logger.info(
            f"{mode} {trade.trade_id}: {trade.side} {trade.question[:45]}… "
            f"@ {trade.fill_price:.2f} ({trade.order_type}) "
            f"${trade.fill_amount:.2f} ({trade.fill_shares:.1f} shares)"
        )

        return trade

    def resolve_trade(self, trade_id: str, outcome: bool) -> Optional[float]:
        """
        Resolve a trade with the actual outcome.
        Same interface as PaperTrader.resolve_trade().

        Note: In live mode, Polymarket handles settlement on-chain automatically.
        This method just updates our internal tracking for P&L reporting.
        """
        trade = self._find_trade(trade_id)
        if trade is None:
            logger.warning(f"Trade {trade_id} not found")
            return None

        if trade.status not in ("OPEN", "FILLED"):
            logger.warning(f"Trade {trade_id} not resolvable: {trade.status}")
            return None

        pnl = trade.resolve(outcome)
        amount = trade.fill_amount if trade.fill_amount > 0 else trade.dollar_amount
        self.bankroll += amount + pnl

        # Track daily P&L
        today = date.today().isoformat()
        self._daily_pnl[today] = self._daily_pnl.get(today, 0.0) + pnl

        self._save_state()

        status = "✅ WIN" if pnl > 0 else "❌ LOSS"
        logger.info(
            f"{status} {trade_id}: PnL=${pnl:+.2f} | "
            f"Bankroll: ${self.bankroll:.2f}"
        )

        return pnl

    def snapshot(self):
        """Return a portfolio snapshot compatible with PaperTrader's."""
        from core.paper_trader import PortfolioSnapshot
        import numpy as np

        open_trades = [t for t in self.trades if t.status in ("OPEN", "FILLED")]
        resolved = [t for t in self.trades if t.status.startswith("RESOLVED")]
        wins = [t for t in resolved if t.status == "RESOLVED_WIN"]

        total_invested = sum(
            (t.fill_amount if t.fill_amount > 0 else t.dollar_amount)
            for t in open_trades
        )
        realized_pnl = sum(t.pnl for t in resolved)
        win_rate = len(wins) / len(resolved) if resolved else 0.0

        # Brier score
        brier = None
        if resolved:
            scores = []
            for t in resolved:
                prob = t.estimated_probability if t.side == "YES" else (1 - t.estimated_probability)
                actual = float(t.resolution if t.side == "YES" else not t.resolution)
                scores.append((prob - actual) ** 2)
            brier = float(np.mean(scores))

        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            bankroll=self.bankroll,
            total_invested=total_invested,
            unrealized_pnl=0.0,  # Would need live price feed for mark-to-market
            realized_pnl=realized_pnl,
            open_positions=len(open_trades),
            resolved_trades=len(resolved),
            win_rate=win_rate,
            brier_score=brier,
        )

    # ═══════════════════════════════════════════════════════
    # ORDER EXECUTION
    # ═══════════════════════════════════════════════════════

    def _execute_dry_run(self, trade: LiveTrade) -> ExecutionResult:
        """Simulate order execution without touching the CLOB."""
        logger.info(
            f"    🔍 DRY RUN: Would place {trade.order_type} {trade.side} "
            f"{trade.shares:.1f} shares @ {trade.entry_price:.2f} "
            f"(token: {trade.token_id[:20]}…)"
        )
        return ExecutionResult(
            success=True,
            trade=trade,
            order_id=f"dry-{trade.trade_id}",
            fill_price=trade.entry_price,
            fill_shares=trade.shares,
        )

    def _execute_live(self, trade: LiveTrade) -> ExecutionResult:
        """
        Place a real order on the Polymarket CLOB.

        For GTC orders: submit, poll for fill, cancel if timeout.
        For FOK orders: submit and check immediate fill.
        """
        from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL

        side = BUY if trade.side == "YES" else SELL
        order_type_enum = OrderType.GTC if trade.order_type == "GTC" else OrderType.FOK

        try:
            if trade.order_type == "FOK":
                # Market order — fill immediately or cancel
                order_args = MarketOrderArgs(
                    token_id=trade.token_id,
                    amount=trade.dollar_amount,
                    side=side,
                    price=trade.entry_price,  # Worst-price limit (slippage protection)
                )
                signed_order = self.client.create_market_order(order_args)
            else:
                # Limit order — sit on the book as maker
                order_args = OrderArgs(
                    token_id=trade.token_id,
                    price=trade.entry_price,
                    size=trade.shares,
                    side=side,
                )
                signed_order = self.client.create_order(order_args)

            trade.order_submitted_at = datetime.now(timezone.utc).isoformat()

            # Post the order
            resp = self.client.post_order(signed_order, order_type_enum)

            if not resp:
                return ExecutionResult(success=False, error="Empty response from post_order")

            # Extract order ID from response
            order_id = ""
            if isinstance(resp, dict):
                order_id = resp.get("orderID", resp.get("id", ""))
                status = resp.get("status", "")
                if status in ("REJECTED", "ERROR"):
                    error_msg = resp.get("errorMessage", resp.get("error", status))
                    return ExecutionResult(success=False, error=f"Order rejected: {error_msg}")
            elif isinstance(resp, str):
                order_id = resp

            if not order_id:
                return ExecutionResult(
                    success=False,
                    error=f"No order ID in response: {resp}"
                )

            trade.order_id = order_id

            # For FOK: immediate fill or nothing
            if trade.order_type == "FOK":
                return ExecutionResult(
                    success=True,
                    trade=trade,
                    order_id=order_id,
                    fill_price=trade.entry_price,
                    fill_shares=trade.shares,
                )

            # For GTC: poll for fill with timeout
            return self._wait_for_fill(trade, order_id)

        except Exception as e:
            error_msg = f"Order execution error: {e}"
            logger.error(error_msg)
            trade.error_message = error_msg
            return ExecutionResult(success=False, error=error_msg)

    def _wait_for_fill(self, trade: LiveTrade, order_id: str) -> ExecutionResult:
        """
        Poll a GTC order for fill status. Cancel after timeout.

        We poll every 5 seconds. If the order is fully filled, great.
        If partially filled, we accept what we got and cancel the rest.
        If unfilled after timeout, cancel entirely.
        """
        from py_clob_client.clob_types import OpenOrderParams

        poll_interval = 5.0
        elapsed = 0.0

        while elapsed < self.order_timeout_seconds:
            time.sleep(poll_interval)
            elapsed += poll_interval

            try:
                # Check if our order is still open
                open_orders = self.client.get_orders(OpenOrderParams())

                our_order = None
                if isinstance(open_orders, list):
                    for o in open_orders:
                        if isinstance(o, dict) and o.get("id") == order_id:
                            our_order = o
                            break

                if our_order is None:
                    # Order no longer in open orders → it was filled
                    logger.info(f"    Order {order_id[:12]}… filled after {elapsed:.0f}s")
                    return ExecutionResult(
                        success=True,
                        trade=trade,
                        order_id=order_id,
                        fill_price=trade.entry_price,
                        fill_shares=trade.shares,
                    )

                # Check for partial fill
                original_size = float(our_order.get("original_size", trade.shares))
                remaining = float(our_order.get("size_matched", 0))
                filled = original_size - remaining if remaining else 0

                if filled > 0:
                    logger.info(
                        f"    Partial fill: {filled:.1f}/{original_size:.1f} shares "
                        f"({elapsed:.0f}s elapsed)"
                    )

            except Exception as e:
                logger.debug(f"Fill check error: {e}")

        # Timeout reached — cancel the order
        logger.info(
            f"    Order {order_id[:12]}… timed out after {self.order_timeout_seconds}s. "
            f"Cancelling."
        )
        try:
            self.client.cancel(order_id)
            trade.order_cancelled_at = datetime.now(timezone.utc).isoformat()
            trade.cancel_reason = "timeout"
        except Exception as e:
            logger.warning(f"Cancel failed (may have filled): {e}")
            # If cancel fails, the order may have filled — treat as success
            return ExecutionResult(
                success=True,
                trade=trade,
                order_id=order_id,
                fill_price=trade.entry_price,
                fill_shares=trade.shares,
            )

        return ExecutionResult(
            success=False,
            error=f"Order timed out after {self.order_timeout_seconds}s",
        )

    # ═══════════════════════════════════════════════════════
    # PRE-FLIGHT CHECKS
    # ═══════════════════════════════════════════════════════

    def _check_order_book(
        self, token_id: str
    ) -> Optional[tuple[float, float, float, float, float]]:
        """
        Fetch the order book and return (best_bid, best_ask, bid_depth, ask_depth, spread).
        Returns None on failure.
        """
        try:
            book = self.client.get_order_book(token_id)

            if hasattr(book, "bids") and hasattr(book, "asks"):
                # py-clob-client returns an OrderBookSummary object
                bids = book.bids if book.bids else []
                asks = book.asks if book.asks else []
            elif isinstance(book, dict):
                bids = book.get("bids", [])
                asks = book.get("asks", [])
            else:
                return None

            # Extract best bid/ask and depth
            best_bid = 0.0
            best_ask = 1.0
            bid_depth = 0.0
            ask_depth = 0.0

            for b in bids:
                price = float(b.get("price", b.price) if isinstance(b, dict) else b.price)
                size = float(b.get("size", b.size) if isinstance(b, dict) else b.size)
                if price > best_bid:
                    best_bid = price
                bid_depth += price * size  # Approximate $ depth

            for a in asks:
                price = float(a.get("price", a.price) if isinstance(a, dict) else a.price)
                size = float(a.get("size", a.size) if isinstance(a, dict) else a.size)
                if price < best_ask:
                    best_ask = price
                ask_depth += price * size

            spread = best_ask - best_bid

            return (best_bid, best_ask, bid_depth, ask_depth, spread)

        except Exception as e:
            logger.debug(f"Order book fetch failed for {token_id[:20]}…: {e}")
            return None

    def _daily_loss_exceeded(self) -> bool:
        """Check if we've hit the daily loss limit."""
        today = date.today().isoformat()
        daily_pnl = self._daily_pnl.get(today, 0.0)
        return daily_pnl < -self.daily_loss_limit

    # ═══════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════

    def _find_trade(self, trade_id: str) -> Optional[LiveTrade]:
        for t in self.trades:
            if t.trade_id == trade_id:
                return t
        return None

    def _save_state(self):
        """Persist all trades to disk."""
        state = {
            "bankroll": self.bankroll,
            "initial_bankroll": self.initial_bankroll,
            "trade_counter": self._trade_counter,
            "daily_pnl": self._daily_pnl,
            "trades": [asdict(t) for t in self.trades],
        }
        tmp = self._state_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        tmp.replace(self._state_file)

    def _load_state(self):
        """Load existing trades from disk."""
        if not self._state_file.exists():
            return

        try:
            content = self._state_file.read_text().strip()
            if not content:
                return  # Empty file, nothing to load

            state = json.loads(content)

            self.bankroll = state.get("bankroll", self.bankroll)
            self.initial_bankroll = state.get("initial_bankroll", self.initial_bankroll)
            self._trade_counter = state.get("trade_counter", 0)
            self._daily_pnl = state.get("daily_pnl", {})

            for td in state.get("trades", []):
                trade = LiveTrade(
                    trade_id=td["trade_id"],
                    market_id=td["market_id"],
                    question=td["question"],
                    side=td["side"],
                    entry_price=td["entry_price"],
                    shares=td["shares"],
                    dollar_amount=td["dollar_amount"],
                    estimated_probability=td["estimated_probability"],
                    market_price_at_entry=td["market_price_at_entry"],
                    edge_at_entry=td["edge_at_entry"],
                    confidence=td["confidence"],
                    timestamp=td.get("timestamp", ""),
                    status=td.get("status", "OPEN"),
                    pnl=td.get("pnl", 0.0),
                    token_id=td.get("token_id", ""),
                    order_id=td.get("order_id", ""),
                    order_type=td.get("order_type", "GTC"),
                    neg_risk=td.get("neg_risk", True),
                    fill_price=td.get("fill_price"),
                    fill_shares=td.get("fill_shares", 0.0),
                    fill_amount=td.get("fill_amount", 0.0),
                    fees_paid=td.get("fees_paid", 0.0),
                    resolution=td.get("resolution"),
                    strategy_tag=td.get("strategy_tag", ""),
                    model_tag=td.get("model_tag", ""),
                )
                self.trades.append(trade)

            open_count = sum(1 for t in self.trades if t.status in ("OPEN", "FILLED"))
            resolved_count = sum(1 for t in self.trades if t.status.startswith("RESOLVED"))
            logger.info(
                f"Loaded {len(self.trades)} live trades "
                f"({open_count} open, {resolved_count} resolved)"
            )
        except Exception as e:
            logger.warning(f"Failed to load live trade state: {e}")

    # ═══════════════════════════════════════════════════════
    # REPORTING
    # ═══════════════════════════════════════════════════════

    def report(self) -> str:
        """Generate a summary report of live trading performance."""
        snap = self.snapshot()
        open_trades = [t for t in self.trades if t.status in ("OPEN", "FILLED")]
        resolved = [t for t in self.trades if t.status.startswith("RESOLVED")]
        wins = [t for t in resolved if t.status == "RESOLVED_WIN"]
        losses = [t for t in resolved if t.status == "RESOLVED_LOSS"]
        cancelled = [t for t in self.trades if t.status == "CANCELLED"]

        lines = [
            "",
            "=" * 70,
            "  LIVE TRADING REPORT",
            "=" * 70,
            f"  Mode: {'DRY RUN' if self.dry_run else 'LIVE'}",
            f"  Bankroll: ${self.bankroll:,.2f} "
            f"(started: ${self.initial_bankroll:,.2f})",
            f"  Open: {len(open_trades)} | "
            f"Resolved: {len(resolved)} | "
            f"Cancelled: {len(cancelled)}",
            f"  Wins: {len(wins)} | Losses: {len(losses)} | "
            f"Win rate: {snap.win_rate:.0%}",
        ]

        if resolved:
            total_pnl = sum(t.pnl for t in resolved)
            total_fees = sum(t.fees_paid for t in resolved)
            lines.append(f"  Total P&L: ${total_pnl:+,.2f} (fees: ${total_fees:.2f})")

        if snap.brier_score is not None:
            lines.append(f"  Brier score: {snap.brier_score:.4f}")

        if self._daily_pnl:
            today = date.today().isoformat()
            today_pnl = self._daily_pnl.get(today, 0.0)
            lines.append(f"  Today's P&L: ${today_pnl:+.2f}")

        if open_trades:
            lines.append(f"\n  Open positions:")
            for t in open_trades:
                amount = t.fill_amount if t.fill_amount > 0 else t.dollar_amount
                lines.append(
                    f"    {t.question[:50]} | "
                    f"${amount:.2f} | Edge: {t.edge_at_entry:+.0%}"
                )

        lines.append("")
        return "\n".join(lines)