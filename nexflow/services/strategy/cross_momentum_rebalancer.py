"""Cross-sectional momentum portfolio rebalancer.

Specification matches validated backtest (mechanism #8, pending verdict):
  Rank:     trailing LOOKBACK_DAYS return at each rebalance date
  Position: long top TOP_N, short bottom TOP_N  (rest flat)
  Rebal:    every REBAL_DAYS calendar days from first signal
  Size:     equal dollar weight = (capital × leverage) / (2 × top_n) per position
  Fee:      taker on every new or changed position

This class maintains state across candles and emits RebalanceOrders when it
is time to adjust the portfolio. The caller (router / live runner) executes them.

Usage (live):
    rebalancer = CrossMomentumRebalancer(symbols, capital=50_000)

    # Called every time a daily candle closes for ANY symbol:
    for candle in daily_candles:
        orders = rebalancer.on_daily_close(symbol, close_price, timestamp_ms)
        if orders:
            execute(orders)   # place / close positions

Usage (backtest): see scripts/backtest_cross_momentum.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

_LOOKBACK_DAYS: int  = 20
_REBAL_DAYS: int     = 7
_TOP_N: int          = 3
_LEVERAGE: float     = 1.0
_DAY_MS: int         = 86_400_000


@dataclass
class RebalanceOrder:
    symbol: str
    action: str          # "OPEN_LONG", "OPEN_SHORT", "CLOSE_LONG", "CLOSE_SHORT"
    notional: float      # dollar amount (for sizing at the exchange)
    reason: str = ""


class CrossMomentumRebalancer:
    """Portfolio-level weekly cross-sectional momentum rebalancer.

    State: per-symbol daily close history (ring buffer of LOOKBACK_DAYS + 1).
    Emits RebalanceOrders once per week when enough ranking data is available.
    """

    def __init__(
        self,
        symbols: list[str],
        capital: float,
        lookback: int = _LOOKBACK_DAYS,
        rebal_days: int = _REBAL_DAYS,
        top_n: int = _TOP_N,
        leverage: float = _LEVERAGE,
    ) -> None:
        self._symbols    = symbols
        self._capital    = capital
        self._lookback   = lookback
        self._rebal_days = rebal_days
        self._top_n      = top_n
        self._leverage   = leverage

        # Per-symbol close history: {symbol: [(ts_ms, close), ...]}
        self._history: dict[str, list[tuple[int, float]]] = {s: [] for s in symbols}

        # Current active positions: {symbol: "LONG"|"SHORT"}
        self._positions: dict[str, str] = {}

        # Last rebalance timestamp (ms)
        self._last_rebal_ts: Optional[int] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def on_daily_close(
        self,
        symbol: str,
        close: float,
        timestamp_ms: int,
    ) -> list[RebalanceOrder]:
        """Feed a daily close. Returns rebalance orders if it's rebalance time.

        Call this for every symbol's daily candle close. Orders are only emitted
        when ALL symbols have been updated for the current timestamp AND it is
        time to rebalance.
        """
        if symbol not in self._history:
            return []

        hist = self._history[symbol]
        # Avoid duplicate timestamps
        if hist and hist[-1][0] == timestamp_ms:
            hist[-1] = (timestamp_ms, close)
        else:
            hist.append((timestamp_ms, close))
            # Keep only what we need
            if len(hist) > self._lookback + 2:
                hist.pop(0)

        # Check if all symbols are updated to this timestamp
        for sym in self._symbols:
            if not self._history[sym] or self._history[sym][-1][0] != timestamp_ms:
                return []  # not all symbols updated yet

        # Check rebalance schedule
        if self._last_rebal_ts is not None:
            elapsed = (timestamp_ms - self._last_rebal_ts) / _DAY_MS
            if elapsed < self._rebal_days:
                return []

        # Need lookback days of history in all symbols
        for sym in self._symbols:
            if len(self._history[sym]) <= self._lookback:
                return []

        return self._rebalance(timestamp_ms)

    # ------------------------------------------------------------------
    # Rebalance logic
    # ------------------------------------------------------------------

    def _rebalance(self, timestamp_ms: int) -> list[RebalanceOrder]:
        # Compute trailing LOOKBACK-day returns
        returns: dict[str, float] = {}
        for sym in self._symbols:
            hist = self._history[sym]
            curr_close = hist[-1][1]
            past_close = hist[-(self._lookback + 1)][1]
            if past_close <= 0:
                continue
            returns[sym] = (curr_close - past_close) / past_close

        if len(returns) < 2 * self._top_n:
            return []

        ranked = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        target_longs  = {sym for sym, _ in ranked[: self._top_n]}
        target_shorts = {sym for sym, _ in ranked[-self._top_n:]}

        # Prevent a symbol from being in both (shouldn't happen with distinct rank)
        target_shorts -= target_longs

        notional = (self._capital * self._leverage) / (2 * self._top_n)
        orders: list[RebalanceOrder] = []

        # Close positions that are no longer in target
        for sym, side in list(self._positions.items()):
            if side == "LONG" and sym not in target_longs:
                orders.append(RebalanceOrder(sym, "CLOSE_LONG", notional,
                                             f"dropped from top-{self._top_n}"))
                del self._positions[sym]
            elif side == "SHORT" and sym not in target_shorts:
                orders.append(RebalanceOrder(sym, "CLOSE_SHORT", notional,
                                             f"dropped from bottom-{self._top_n}"))
                del self._positions[sym]

        # Flip sides if needed
        for sym in list(self._positions.keys()):
            side = self._positions[sym]
            if side == "LONG" and sym in target_shorts:
                orders.append(RebalanceOrder(sym, "CLOSE_LONG", notional, "flip to short"))
                del self._positions[sym]
            elif side == "SHORT" and sym in target_longs:
                orders.append(RebalanceOrder(sym, "CLOSE_SHORT", notional, "flip to long"))
                del self._positions[sym]

        # Open new long positions
        for sym in target_longs:
            if sym not in self._positions:
                orders.append(RebalanceOrder(sym, "OPEN_LONG", notional,
                                             f"ret={returns.get(sym,0):.1%}"))
                self._positions[sym] = "LONG"

        # Open new short positions
        for sym in target_shorts:
            if sym not in self._positions:
                orders.append(RebalanceOrder(sym, "OPEN_SHORT", notional,
                                             f"ret={returns.get(sym,0):.1%}"))
                self._positions[sym] = "SHORT"

        self._last_rebal_ts = timestamp_ms

        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if orders:
            top_ret = {sym: f"{returns[sym]:.1%}" for sym, _ in ranked[:self._top_n]}
            bot_ret = {sym: f"{returns[sym]:.1%}" for sym, _ in ranked[-self._top_n:]}
            print(f"  Rebalance {dt}: longs={top_ret} shorts={bot_ret}  → {len(orders)} order(s)")

        return orders

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def positions(self) -> dict[str, str]:
        """Current position map: {symbol: "LONG"|"SHORT"}."""
        return dict(self._positions)

    def reset(self) -> None:
        """Reset all state (use between backtest runs)."""
        for sym in self._symbols:
            self._history[sym].clear()
        self._positions.clear()
        self._last_rebal_ts = None
