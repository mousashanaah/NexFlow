"""Real-time performance statistics over closed trades.

Updated incrementally after each trade close — no full recomputation.
All properties are O(1) unless noted.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from nexflow.services.strategy.signal_models import ClosedTrade, Direction, ExitReason


@dataclass
class PerformanceSnapshot:
    ts: float
    total_trades: int
    win_rate: float
    long_win_rate: float
    short_win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    total_pnl: float
    total_fees: float
    avg_hold_bars: float
    exit_reason_counts: dict[str, int]


class PerformanceTracker:
    """Incremental live performance stats.

    All metrics are updated on each trade_closed call. Safe to query at any time.
    """

    def __init__(self) -> None:
        self._trades: list[ClosedTrade] = []
        self._wins: list[ClosedTrade] = []
        self._losses: list[ClosedTrade] = []
        self._longs: list[ClosedTrade] = []
        self._shorts: list[ClosedTrade] = []

        # Running accumulators (avoid iterating full list each time)
        self._gross_profit: float = 0.0
        self._gross_loss: float = 0.0    # positive number
        self._total_pnl: float = 0.0
        self._total_fees: float = 0.0
        self._total_hold_bars: int = 0
        self._exit_counts: dict[str, int] = {}

        # Maker fill tracking
        self._maker_attempts: int = 0          # total TP limit orders placed
        self._tp_hits: dict[int, int] = {0: 0, 1: 0, 2: 0}   # tp_idx → fill count
        self._stop_hits: int = 0
        self._maker_latencies: list[int] = []  # bars-to-fill for each TP hit

    def on_maker_attempts(self, n: int) -> None:
        self._maker_attempts += n

    def on_tp_hit(self, tp_idx: int, latency_bars: int) -> None:
        self._tp_hits[tp_idx] = self._tp_hits.get(tp_idx, 0) + 1
        self._maker_latencies.append(latency_bars)

    def on_stop_hit(self) -> None:
        self._stop_hits += 1

    def on_trade_closed(self, trade: ClosedTrade) -> None:
        self._trades.append(trade)
        self._total_pnl += trade.pnl
        self._total_fees += trade.fees
        self._total_hold_bars += trade.hold_bars
        key = trade.exit_reason.value if hasattr(trade.exit_reason, "value") else str(trade.exit_reason)
        self._exit_counts[key] = self._exit_counts.get(key, 0) + 1

        if trade.pnl > 0:
            self._wins.append(trade)
            self._gross_profit += trade.pnl
        else:
            self._losses.append(trade)
            self._gross_loss += abs(trade.pnl)

        if trade.direction is Direction.LONG:
            self._longs.append(trade)
        else:
            self._shorts.append(trade)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    @property
    def win_rate(self) -> float:
        if not self._trades:
            return 0.0
        return len(self._wins) / len(self._trades)

    @property
    def long_win_rate(self) -> float:
        longs_wins = [t for t in self._longs if t.pnl > 0]
        return len(longs_wins) / len(self._longs) if self._longs else 0.0

    @property
    def short_win_rate(self) -> float:
        short_wins = [t for t in self._shorts if t.pnl > 0]
        return len(short_wins) / len(self._shorts) if self._shorts else 0.0

    @property
    def avg_win(self) -> float:
        return self._gross_profit / len(self._wins) if self._wins else 0.0

    @property
    def avg_loss(self) -> float:
        return self._gross_loss / len(self._losses) if self._losses else 0.0

    @property
    def profit_factor(self) -> float:
        if self._gross_loss <= 0:
            return float("inf") if self._gross_profit > 0 else 0.0
        return self._gross_profit / self._gross_loss

    @property
    def expectancy(self) -> float:
        """Expectancy in R: (win_rate × avg_win − loss_rate × avg_loss) / avg_loss."""
        if self.avg_loss <= 0:
            return 0.0
        loss_rate = 1.0 - self.win_rate
        return (self.win_rate * self.avg_win - loss_rate * self.avg_loss) / self.avg_loss

    @property
    def total_pnl(self) -> float:
        return self._total_pnl

    @property
    def total_fees(self) -> float:
        return self._total_fees

    @property
    def avg_hold_bars(self) -> float:
        if not self._trades:
            return 0.0
        return self._total_hold_bars / len(self._trades)

    # ------------------------------------------------------------------
    # Maker fill properties
    # ------------------------------------------------------------------

    @property
    def maker_attempts(self) -> int:
        return self._maker_attempts

    @property
    def maker_fills(self) -> int:
        return sum(self._tp_hits.values())

    @property
    def tp1_hit_rate(self) -> float:
        n = len(self._trades)
        return self._tp_hits.get(0, 0) / n if n else 0.0

    @property
    def tp2_hit_rate(self) -> float:
        n = len(self._trades)
        return self._tp_hits.get(1, 0) / n if n else 0.0

    @property
    def tp3_hit_rate(self) -> float:
        n = len(self._trades)
        return self._tp_hits.get(2, 0) / n if n else 0.0

    @property
    def stop_hit_rate(self) -> float:
        n = len(self._trades)
        return self._stop_hits / n if n else 0.0

    @property
    def avg_maker_latency_bars(self) -> float:
        return sum(self._maker_latencies) / len(self._maker_latencies) if self._maker_latencies else 0.0

    @property
    def maker_fill_rate(self) -> float:
        return self.maker_fills / self._maker_attempts if self._maker_attempts else 0.0

    def snapshot(self) -> PerformanceSnapshot:
        return PerformanceSnapshot(
            ts=time.time(),
            total_trades=self.total_trades,
            win_rate=self.win_rate,
            long_win_rate=self.long_win_rate,
            short_win_rate=self.short_win_rate,
            avg_win=self.avg_win,
            avg_loss=self.avg_loss,
            profit_factor=self.profit_factor,
            expectancy=self.expectancy,
            total_pnl=self.total_pnl,
            total_fees=self.total_fees,
            avg_hold_bars=self.avg_hold_bars,
            exit_reason_counts=dict(self._exit_counts),
        )

    def summary_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "win_rate_pct": round(self.win_rate * 100, 1),
            "long_trades": len(self._longs),
            "short_trades": len(self._shorts),
            "long_wr_pct": round(self.long_win_rate * 100, 1),
            "short_wr_pct": round(self.short_win_rate * 100, 1),
            "profit_factor": round(self.profit_factor, 3) if self.profit_factor != float("inf") else "inf",
            "expectancy_R": round(self.expectancy, 4),
            "total_pnl": round(self.total_pnl, 2),
            "total_fees": round(self.total_fees, 2),
            "avg_hold_bars": round(self.avg_hold_bars, 1),
            "exit_reasons": self._exit_counts,
        }
