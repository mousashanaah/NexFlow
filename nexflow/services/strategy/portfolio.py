"""Portfolio — position lifecycle and equity tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

from nexflow.services.strategy.signal_models import ClosedTrade, Direction, ExitReason


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@dataclass
class TpLevel:
    price: float
    size: float      # quantity to close at this level
    hit: bool = False


@dataclass
class Position:
    symbol: str
    direction: Direction
    entry_price: float
    entry_time: int         # epoch seconds
    equity_at_entry: float

    # Size tracking
    total_size: float
    remaining_size: float

    # Risk levels
    stop_price: float
    tp_levels: list[TpLevel]

    # Accumulators (updated as partials close)
    realized_pnl: float = 0.0
    realized_fees: float = 0.0
    hold_bars: int = 0
    breakeven_moved: bool = False  # True after TP1 hit

    # Weighted-average exit price accumulator
    _exit_pv_sum: float = 0.0     # sum(price × size) for closed portions

    def tick(self) -> None:
        """Call once per 1m bar to increment hold time."""
        self.hold_bars += 1

    def apply_partial_close(
        self, price: float, size: float, fee: float, *, move_stop_to_be: bool = False
    ) -> None:
        self.remaining_size -= size
        pnl_raw = (price - self.entry_price) * size * self._direction_sign()
        self.realized_pnl += pnl_raw - fee
        self.realized_fees += fee
        self._exit_pv_sum += price * size
        if move_stop_to_be and not self.breakeven_moved:
            self.stop_price = self.entry_price
            self.breakeven_moved = True

    def weighted_exit_price(self) -> float:
        closed_size = self.total_size - self.remaining_size
        return self._exit_pv_sum / closed_size if closed_size > 0 else self.entry_price

    def _direction_sign(self) -> float:
        return 1.0 if self.direction is Direction.LONG else -1.0

    def is_closed(self) -> bool:
        return self.remaining_size <= 1e-10


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class Portfolio:
    """Tracks equity, open positions, and trade history."""

    def __init__(self, initial_equity: float = 100_000.0) -> None:
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.positions: dict[str, Position] = {}   # symbol → Position
        self.closed_trades: list[ClosedTrade] = []
        self.equity_curve: list[tuple[int, float]] = [(0, initial_equity)]

        # Daily drawdown tracking
        self._day_start_equity = initial_equity
        self._current_day: int = 0   # UTC day number

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def open_position(self, pos: Position) -> None:
        if self.has_position(pos.symbol):
            raise ValueError(f"Position already open for {pos.symbol}")
        self.positions[pos.symbol] = pos

    def close_position(
        self,
        symbol: str,
        exit_time: int,
        exit_reason: ExitReason,
    ) -> ClosedTrade:
        pos = self.positions.pop(symbol)

        exit_price = pos.weighted_exit_price()
        trade = ClosedTrade(
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            total_size=pos.total_size,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            pnl=pos.realized_pnl,
            fees=pos.realized_fees,
            equity_at_entry=pos.equity_at_entry,
            hold_bars=pos.hold_bars,
            exit_reason=exit_reason,
        )
        self.closed_trades.append(trade)
        self.current_equity += pos.realized_pnl
        self.equity_curve.append((exit_time, self.current_equity))
        return trade

    def record_equity(self, timestamp: int) -> None:
        """Snapshot current equity into the curve (e.g., at bar close)."""
        self.equity_curve.append((timestamp, self.current_equity))

    # ------------------------------------------------------------------
    # Daily drawdown tracking
    # ------------------------------------------------------------------

    def update_day(self, timestamp_s: int) -> None:
        """Reset daily high at UTC midnight boundaries."""
        day = timestamp_s // 86_400
        if day > self._current_day:
            self._current_day = day
            self._day_start_equity = self.current_equity

    def daily_drawdown(self) -> float:
        """Fraction of day-start equity lost so far today."""
        if self._day_start_equity <= 0:
            return 0.0
        loss = self._day_start_equity - self.current_equity
        return max(0.0, loss / self._day_start_equity)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def open_position_count(self) -> int:
        return len(self.positions)
