"""Real-time equity curve tracker with live Sharpe and drawdown metrics.

Updated on every candle tick and every trade close. Designed for low-latency
queries from the dashboard — all metrics are O(1) or O(window).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class EquitySnapshot:
    ts: float          # epoch seconds
    equity: float
    drawdown: float    # fraction below peak (0.0 = at peak)
    unrealized: float


class EquityCurveTracker:
    """Tracks equity, drawdown, and rolling Sharpe in real-time.

    Args:
        initial_equity: starting capital
        sharpe_window: number of trade returns used for rolling Sharpe
        snapshot_interval_s: minimum seconds between equity snapshots
    """

    def __init__(
        self,
        initial_equity: float,
        sharpe_window: int = 30,
        snapshot_interval_s: float = 60.0,
    ) -> None:
        self._initial_equity = initial_equity
        self._sharpe_window = sharpe_window
        self._snapshot_interval_s = snapshot_interval_s

        self._equity = initial_equity
        self._peak_equity = initial_equity
        self._realized_pnl = 0.0

        self._trade_returns: deque[float] = deque(maxlen=sharpe_window)
        self._snapshots: list[EquitySnapshot] = []
        self._last_snapshot_ts: float = 0.0

    # ------------------------------------------------------------------
    # Update interface
    # ------------------------------------------------------------------

    def on_trade_closed(self, pnl: float, equity: float) -> None:
        """Call after each trade close with net PnL and updated portfolio equity."""
        self._equity = equity
        self._realized_pnl += pnl
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Per-trade return as fraction of equity at time of trade
        if equity > 0:
            self._trade_returns.append(pnl / max(equity, 1.0))

    def on_equity_update(self, equity: float, unrealized_pnl: float = 0.0) -> None:
        """Call on any equity change (including unrealized mark-to-market)."""
        total = equity + unrealized_pnl
        self._equity = equity
        if total > self._peak_equity:
            self._peak_equity = total

        now = time.time()
        if now - self._last_snapshot_ts >= self._snapshot_interval_s:
            self._snapshots.append(EquitySnapshot(
                ts=now,
                equity=equity,
                drawdown=self.current_drawdown,
                unrealized=unrealized_pnl,
            ))
            self._last_snapshot_ts = now

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    @property
    def current_equity(self) -> float:
        return self._equity

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def net_return(self) -> float:
        """Total return as fraction of initial equity."""
        return (self._equity - self._initial_equity) / self._initial_equity

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as fraction below peak."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._equity) / self._peak_equity)

    @property
    def max_drawdown_observed(self) -> float:
        """Maximum drawdown seen since tracker was created."""
        if not self._snapshots:
            return self.current_drawdown
        return max(s.drawdown for s in self._snapshots)

    def rolling_sharpe(self) -> float:
        """Sharpe ratio over the last sharpe_window trade returns.

        Returns 0.0 when insufficient data.
        """
        returns = list(self._trade_returns)
        n = len(returns)
        if n < 2:
            return 0.0
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
        std = math.sqrt(variance)
        if std < 1e-10:
            return 0.0
        return (mean / std) * math.sqrt(min(n, 252))

    def compute_unrealized(
        self,
        positions: dict,   # symbol → Position
        mid_prices: dict[str, float],
    ) -> float:
        """Compute total unrealized PnL for open positions given current mid prices."""
        total = 0.0
        for symbol, pos in positions.items():
            price = mid_prices.get(symbol)
            if price is None:
                continue
            sign = 1.0 if str(pos.direction).endswith("LONG") else -1.0
            upnl = (price - pos.entry_price) * pos.remaining_size * sign
            total += upnl
        return total

    @property
    def snapshots(self) -> list[EquitySnapshot]:
        return list(self._snapshots)

    def summary(self) -> dict:
        return {
            "equity": round(self._equity, 2),
            "realized_pnl": round(self._realized_pnl, 2),
            "drawdown_pct": round(self.current_drawdown * 100, 3),
            "net_return_pct": round(self.net_return * 100, 3),
            "rolling_sharpe": round(self.rolling_sharpe(), 3),
            "peak_equity": round(self._peak_equity, 2),
        }
