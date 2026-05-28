"""Core market data models.

All prices and quantities are stored as float for speed; callers that need
exact decimal arithmetic should convert at the point of use.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(slots=True)
class OrderBookLevel:
    price: float
    size: float


@dataclass(slots=True)
class Trade:
    trade_id: str
    price: float
    size: float
    side: TradeSide
    timestamp_ms: int


@dataclass
class MarketState:
    """Live snapshot of a single instrument's market data.

    Updated in-place by the market data service; consumers should read
    under the asyncio event loop (no locking needed for single-threaded async).
    """

    symbol: str
    product_type: str

    # Orderbook — sorted best-first (bids descending, asks ascending)
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)

    # Rolling trade history, newest last
    trades: Deque[Trade] = field(default_factory=deque)

    # 24h ticker
    open_24h: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    close_24h: float = 0.0
    volume_24h: float = 0.0
    base_volume_24h: float = 0.0

    # Exchange-reported message timestamp (ms) — set by the transport layer
    exchange_ts_ms: int = 0

    # Last update timestamps (unix epoch, seconds)
    orderbook_updated_at: float = 0.0
    trades_updated_at: float = 0.0
    ticker_updated_at: float = 0.0

    # --- convenience properties -------------------------------------------------

    @property
    def best_bid(self) -> OrderBookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> OrderBookLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return (b.price + a.price) / 2.0

    @property
    def spread(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return a.price - b.price

    @property
    def last_trade(self) -> Trade | None:
        return self.trades[-1] if self.trades else None

    # --- mutators ---------------------------------------------------------------

    def apply_orderbook_snapshot(
        self, bids: list[tuple[float, float]], asks: list[tuple[float, float]]
    ) -> None:
        self.bids = [OrderBookLevel(p, s) for p, s in bids]
        self.asks = [OrderBookLevel(p, s) for p, s in asks]
        self.orderbook_updated_at = time.time()

    def apply_orderbook_delta(
        self,
        bid_changes: list[tuple[float, float]],
        ask_changes: list[tuple[float, float]],
    ) -> None:
        """Apply incremental orderbook changes. size=0 means remove the level."""
        self.bids = _apply_book_delta(self.bids, bid_changes, descending=True)
        self.asks = _apply_book_delta(self.asks, ask_changes, descending=False)
        self.orderbook_updated_at = time.time()

    def add_trade(self, trade: Trade, max_history: int = 100) -> None:
        self.trades.append(trade)
        while len(self.trades) > max_history:
            self.trades.popleft()
        self.trades_updated_at = time.time()

    def apply_ticker(self, data: dict[str, float]) -> None:
        self.open_24h = data.get("open24h", self.open_24h)
        self.high_24h = data.get("high24h", self.high_24h)
        self.low_24h = data.get("low24h", self.low_24h)
        self.close_24h = data.get("close24h", self.close_24h)
        self.volume_24h = data.get("baseVolume", self.volume_24h)
        self.base_volume_24h = data.get("quoteVolume", self.base_volume_24h)
        self.ticker_updated_at = time.time()

    def __repr__(self) -> str:
        mid = f"{self.mid_price:.4f}" if self.mid_price else "N/A"
        return f"<MarketState {self.symbol} mid={mid}>"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_book_delta(
    levels: list[OrderBookLevel],
    changes: list[tuple[float, float]],
    *,
    descending: bool,
) -> list[OrderBookLevel]:
    book: dict[float, float] = {lvl.price: lvl.size for lvl in levels}
    for price, size in changes:
        if size == 0.0:
            book.pop(price, None)
        else:
            book[price] = size
    return sorted(
        (OrderBookLevel(p, s) for p, s in book.items()),
        key=lambda lvl: lvl.price,
        reverse=descending,
    )
