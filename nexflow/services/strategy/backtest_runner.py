"""Deterministic event-driven backtester operating on finalized candles.

Input:  dict[symbol → dict[timeframe → list[Candle]]] (all candles pre-sorted by close_time)
Output: BacktestMetrics + list[ClosedTrade]

Event ordering within a shared close_time:
    15m → 5m → 1m  (higher timeframes update context before the trigger fires)

Stop/TP evaluation within each 1m bar:
    Uses bar.low and bar.high. If both stop and TP are hit in one bar, the stop
    is assumed to have been hit first (conservative/pessimistic assumption).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from nexflow.logging import get_logger
from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.base_strategy import BaseStrategy
from nexflow.services.strategy.paper_execution import ExecutionConfig, PaperExecution
from nexflow.services.strategy.portfolio import Portfolio, Position, TpLevel
from nexflow.services.strategy.risk_engine import RiskConfig, RiskEngine
from nexflow.services.strategy.signal_models import (
    BacktestMetrics,
    ClosedTrade,
    Direction,
    ExitReason,
    Signal,
)


_log = get_logger(__name__)

# TP ladder: fractions of total position to close at each TP level
_TP_FRACTIONS = [0.50, 0.25, 0.25]

# Timeframe sort order: larger TFs processed before smaller TFs at same timestamp
_TF_ORDER = {"15m": 0, "5m": 1, "1m": 2}


@dataclass
class BacktestConfig:
    initial_equity: float = 100_000.0
    risk: RiskConfig = None          # type: ignore[assignment]
    execution: ExecutionConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.risk is None:
            self.risk = RiskConfig()
        if self.execution is None:
            self.execution = ExecutionConfig()


class BacktestRunner:
    """Drives a strategy over historical candles and produces metrics."""

    def __init__(self, strategy: BaseStrategy, cfg: BacktestConfig | None = None) -> None:
        self._strategy = strategy
        self._cfg = cfg or BacktestConfig()

    def run(self, all_candles: dict[str, dict[str, list[Candle]]]) -> BacktestMetrics:
        """Run the backtest.

        all_candles: {symbol: {timeframe: [sorted Candle list]}}
        Returns BacktestMetrics with full equity curve and trade list.
        """
        self._strategy.reset()
        portfolio = Portfolio(self._cfg.initial_equity)
        risk = RiskEngine(self._cfg.risk)
        execution = PaperExecution(self._cfg.execution)

        # Build sorted event stream
        events = _build_event_stream(all_candles)

        for candle in events:
            symbol = candle.symbol
            tf = candle.timeframe
            bar_time = candle.close_time

            # Daily boundary reset
            portfolio.update_day(bar_time)

            # Advance risk engine cooldown counter on every 1m bar
            if tf == "1m":
                risk.tick()

            # Check stop / TP for open position in this symbol on 1m bars
            if tf == "1m" and portfolio.has_position(symbol):
                pos = portfolio.get_position(symbol)
                assert pos is not None
                pos.tick()
                closed_trade = _evaluate_stops_and_tps(pos, candle, portfolio, execution, bar_time)
                if closed_trade is not None and closed_trade.pnl < 0:
                    risk.on_loss()

            # Feed candle to strategy
            signal = self._strategy.on_candle(candle)

            # Process signal on 1m bars only
            if signal is None or tf != "1m":
                continue

            allowed, reason = risk.check_entry(signal, portfolio)
            if not allowed:
                signal.reject_reason = reason
                _log.debug("backtest.entry_rejected", symbol=symbol, reason=reason)
                continue

            size = risk.compute_position_size(signal, portfolio)
            if size <= 0:
                continue

            pos = _open_position(signal, size, portfolio, execution, bar_time)
            _log.debug(
                "backtest.entry",
                symbol=symbol,
                direction=signal.direction,
                price=pos.entry_price,
                size=size,
                stop=pos.stop_price,
            )

        # Force-close all remaining positions at last known price
        _force_close_all(portfolio, execution, events)

        return _compute_metrics(portfolio, self._cfg.initial_equity)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_event_stream(all_candles: dict[str, dict[str, list[Candle]]]) -> list[Candle]:
    """Merge and sort all candles: ascending close_time, larger TFs first at ties."""
    events: list[Candle] = []
    for sym_candles in all_candles.values():
        for tf_candles in sym_candles.values():
            events.extend(tf_candles)

    events.sort(key=lambda c: (c.close_time, _TF_ORDER.get(c.timeframe, 99)))
    return events


def _open_position(
    signal: Signal,
    size: float,
    portfolio: Portfolio,
    execution: PaperExecution,
    bar_time: int,
) -> Position:
    fill = execution.simulate_entry(signal)
    entry_price = fill.fill_price
    entry_fee = execution.compute_fee(entry_price, size, is_maker=False)

    tp_sizes = [size * f for f in _TP_FRACTIONS]
    tp_levels = [
        TpLevel(price=p, size=s) for p, s in zip(signal.tp_prices, tp_sizes)
    ]

    pos = Position(
        symbol=signal.symbol,
        direction=signal.direction,
        entry_price=entry_price,
        entry_time=bar_time,
        equity_at_entry=portfolio.current_equity,
        total_size=size,
        remaining_size=size,
        stop_price=signal.stop_price,
        tp_levels=tp_levels,
        realized_pnl=-entry_fee,    # entry fee is an immediate cost
        realized_fees=entry_fee,
    )
    portfolio.open_position(pos)
    return pos


def _evaluate_stops_and_tps(
    pos: Position,
    candle: Candle,
    portfolio: Portfolio,
    execution: PaperExecution,
    bar_time: int,
) -> ClosedTrade | None:
    """Check bar high/low against stops and TPs. Returns ClosedTrade if closed."""
    is_long = pos.direction is Direction.LONG

    # --- Stop hit? ---
    stop_hit = (
        (is_long and candle.low <= pos.stop_price)
        or (not is_long and candle.high >= pos.stop_price)
    )

    # --- Check TPs (best-to-worst order is already stored in tp_levels) ---
    # Accumulate TP hits before deciding if stop takes priority
    tps_hit_this_bar: list[int] = []
    for idx, tp in enumerate(pos.tp_levels):
        if tp.hit:
            continue
        if is_long and candle.high >= tp.price:
            tps_hit_this_bar.append(idx)
        elif not is_long and candle.low <= tp.price:
            tps_hit_this_bar.append(idx)

    if stop_hit:
        # Conservative: stop takes priority over TP (assume stop hit first in bar)
        stop_distance = abs(pos.entry_price - pos.stop_price)
        fill = execution.simulate_stop(pos.stop_price, pos.direction, stop_distance)
        fee = execution.compute_fee(fill.fill_price, pos.remaining_size, is_maker=False)
        pos.apply_partial_close(fill.fill_price, pos.remaining_size, fee)
        return portfolio.close_position(pos.symbol, bar_time, ExitReason.STOP)

    # --- Apply TP hits ---
    last_tp_idx = -1
    for idx in tps_hit_this_bar:
        tp = pos.tp_levels[idx]
        fill = execution.simulate_tp(tp.price)
        fee = execution.compute_fee(fill.fill_price, tp.size, is_maker=True)
        move_to_be = (idx == 0)  # move stop to breakeven after TP1
        pos.apply_partial_close(fill.fill_price, tp.size, fee, move_stop_to_be=move_to_be)
        tp.hit = True
        last_tp_idx = idx

    if pos.is_closed():
        reason = ExitReason.TP1 if last_tp_idx == 0 else (
            ExitReason.TP2 if last_tp_idx == 1 else ExitReason.TP3
        )
        return portfolio.close_position(pos.symbol, bar_time, reason)

    return None


def _force_close_all(
    portfolio: Portfolio,
    execution: PaperExecution,
    events: list[Candle],
) -> None:
    """Close any still-open positions at the last available 1m close price."""
    last_prices: dict[str, float] = {}
    for c in events:
        if c.timeframe == "1m":
            last_prices[c.symbol] = c.close

    for symbol in list(portfolio.positions.keys()):
        pos = portfolio.positions[symbol]
        price = last_prices.get(symbol, pos.entry_price)
        fee = execution.compute_fee(price, pos.remaining_size, is_maker=False)
        pos.apply_partial_close(price, pos.remaining_size, fee)
        portfolio.close_position(symbol, events[-1].close_time if events else 0, ExitReason.FORCED)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(portfolio: Portfolio, initial_equity: float) -> BacktestMetrics:
    trades = portfolio.closed_trades
    if not trades:
        return BacktestMetrics(
            total_trades=0, win_rate=0.0, expectancy=0.0, sharpe=0.0,
            max_drawdown=0.0, profit_factor=0.0, avg_hold_bars=0.0,
            net_pnl=0.0, total_fees=0.0,
            long_trades=0, long_win_rate=0.0,
            short_trades=0, short_win_rate=0.0,
            pnl_distribution=[],
            equity_curve=portfolio.equity_curve,
        )

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    longs = [t for t in trades if t.direction is Direction.LONG]
    shorts = [t for t in trades if t.direction is Direction.SHORT]

    win_rate = len(wins) / len(trades)
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0.0
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))

    # Expectancy: expected pnl per dollar risked
    # Approximate risk per trade = equity_at_entry × max_risk_pct
    # We don't store that directly, so use raw pnl expectancy instead
    expectancy = (win_rate * avg_win - (1.0 - win_rate) * avg_loss) / avg_loss if avg_loss > 0 else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Per-trade return as fraction of equity at entry
    trade_returns = [t.pnl / t.equity_at_entry for t in trades if t.equity_at_entry > 0]
    sharpe = _compute_sharpe(trade_returns)

    max_dd = _compute_max_drawdown(portfolio.equity_curve)

    return BacktestMetrics(
        total_trades=len(trades),
        win_rate=win_rate,
        expectancy=expectancy,
        sharpe=sharpe,
        max_drawdown=max_dd,
        profit_factor=profit_factor,
        avg_hold_bars=sum(t.hold_bars for t in trades) / len(trades),
        net_pnl=portfolio.current_equity - initial_equity,
        total_fees=sum(t.fees for t in trades),
        long_trades=len(longs),
        long_win_rate=len([t for t in longs if t.pnl > 0]) / len(longs) if longs else 0.0,
        short_trades=len(shorts),
        short_win_rate=len([t for t in shorts if t.pnl > 0]) / len(shorts) if shorts else 0.0,
        pnl_distribution=[t.pnl for t in trades],
        equity_curve=portfolio.equity_curve,
    )


def _compute_sharpe(trade_returns: list[float]) -> float:
    """Annualised Sharpe from per-trade returns (assumes ~252 active-day equivalents)."""
    n = len(trade_returns)
    if n < 2:
        return 0.0
    mean = sum(trade_returns) / n
    variance = sum((r - mean) ** 2 for r in trade_returns) / (n - 1)
    std = math.sqrt(variance)
    if std < 1e-10:
        return 0.0
    # Scale to "per year" — assume ~4 trades/day × 252 days = ~1000 trades/year
    per_trade_sharpe = mean / std
    annualisation = math.sqrt(min(n * 4, 1000))  # conservative annualisation
    return per_trade_sharpe * annualisation


def _compute_max_drawdown(equity_curve: list[tuple[int, float]]) -> float:
    """Peak-to-trough drawdown as fraction of peak equity."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0][1]
    max_dd = 0.0
    for _, equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd
