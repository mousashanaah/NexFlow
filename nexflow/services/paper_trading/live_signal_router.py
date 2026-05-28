"""Live signal router: finalized candle → strategy → risk → execution → portfolio.

This is the central processing unit for both live and replay modes.
It is called by the CandleEngine callback (live) or directly (replay).

Design constraints:
    - Only processes FINALIZED candles (is_final=True enforced at call site)
    - Stop/TP evaluation happens on every 1m bar for open positions
    - Signals are only generated and acted on for 1m bars (same as backtest)
    - Multi-timeframe context (5m, 15m) is maintained for the strategy window
    - All events are journaled before side-effects are applied
    - Kill conditions are checked before every new entry
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from nexflow.exchange.bitget_constraints import get_constraints, round_price, round_size
from nexflow.logging import get_logger
from nexflow.services.candles.candle_engine import Candle
from nexflow.services.paper_trading.equity_curve_tracker import EquityCurveTracker
from nexflow.services.paper_trading.execution_journal import ExecutionJournal
from nexflow.services.paper_trading.live_risk_monitor import LiveRiskMonitor
from nexflow.services.paper_trading.performance_tracker import PerformanceTracker
from nexflow.services.strategy.base_strategy import BaseStrategy
from nexflow.services.strategy.paper_execution import PaperExecution
from nexflow.services.strategy.portfolio import Portfolio, Position, TpLevel
from nexflow.services.strategy.risk_engine import RiskEngine
from nexflow.services.strategy.signal_models import Direction, ExitReason

_log = get_logger(__name__)

_TP_FRACTIONS = [0.50, 0.25, 0.25]


@dataclass
class RouterState:
    """Snapshot of current router state for dashboard queries."""
    equity: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    open_positions: int = 0
    drawdown: float = 0.0
    rolling_sharpe: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    is_killed: bool = False
    kill_reasons: list[str] = field(default_factory=list)
    last_candle_ts: dict[str, int] = field(default_factory=dict)


class LiveSignalRouter:
    """Routes finalized candles through the full strategy stack.

    Args:
        strategy: initialized strategy instance
        risk: RiskEngine
        execution: PaperExecution
        portfolio: Portfolio
        journal: ExecutionJournal
        risk_monitor: LiveRiskMonitor
        equity_tracker: EquityCurveTracker
        perf_tracker: PerformanceTracker
        mid_prices: shared dict updated by BitgetWSClient callbacks
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        risk: RiskEngine,
        execution: PaperExecution,
        portfolio: Portfolio,
        journal: ExecutionJournal,
        risk_monitor: LiveRiskMonitor,
        equity_tracker: EquityCurveTracker,
        perf_tracker: PerformanceTracker,
        mid_prices: dict[str, float] | None = None,
    ) -> None:
        self._strategy = strategy
        self._risk = risk
        self._execution = execution
        self._portfolio = portfolio
        self._journal = journal
        self._risk_monitor = risk_monitor
        self._equity_tracker = equity_tracker
        self._perf_tracker = perf_tracker
        self._mid_prices: dict[str, float] = mid_prices if mid_prices is not None else {}
        self._bar_counts: dict[str, int] = {}  # symbol → 1m bar count for risk tick

    # ------------------------------------------------------------------
    # Primary entry point — called by CandleEngine or ReplayEngine
    # ------------------------------------------------------------------

    async def on_candle(self, symbol: str, timeframe: str, candle: Candle) -> None:
        """Process one finalized candle. Must be called with is_final=True candles only."""
        bar_time = candle.close_time

        # Update daily boundary tracking
        self._portfolio.update_day(bar_time)

        # Update mid price from latest close
        self._mid_prices[symbol] = candle.close

        if timeframe == "1m":
            self._bar_counts[symbol] = self._bar_counts.get(symbol, 0) + 1
            self._risk.tick()

            # Compute ATR proxy for risk monitor (use candle range as crude estimate)
            atr_proxy = candle.high - candle.low
            self._risk_monitor.on_candle_received(symbol, atr=atr_proxy, spread=candle.spread_avg)
            self._risk_monitor.on_drawdown(self._portfolio.current_equity)

            # Evaluate open positions first (stop / TP)
            if self._portfolio.has_position(symbol):
                pos = self._portfolio.get_position(symbol)
                assert pos is not None
                pos.tick()
                self._evaluate_exits(pos, candle, bar_time)

        # Feed candle to strategy (all timeframes for context building)
        signal = self._strategy.on_candle(candle)

        # Process new entries only on 1m bars
        if signal is None or timeframe != "1m":
            return

        # Log signal before risk check
        self._journal.log_signal(
            symbol=signal.symbol,
            direction=signal.direction.value,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            tp_prices=signal.tp_prices,
            atr=signal.features.get("atr", 0.0),
            features=signal.features,
        )

        # Kill-switch check
        if self._risk_monitor.is_killed:
            reasons = ", ".join(r.value for r in self._risk_monitor.kill_reasons)
            self._journal.log_rejected(signal.symbol, signal.direction.value, f"kill_switch:{reasons}")
            _log.warning("router.kill_switch_blocked", symbol=symbol, reasons=reasons)
            return

        # Already have a position in this symbol?
        if self._portfolio.has_position(symbol):
            self._journal.log_rejected(signal.symbol, signal.direction.value, "position_already_open")
            return

        # Risk engine check
        allowed, reason = self._risk.check_entry(signal, self._portfolio)
        if not allowed:
            self._journal.log_rejected(signal.symbol, signal.direction.value, reason)
            signal.reject_reason = reason
            _log.debug("router.entry_rejected", symbol=symbol, reason=reason)
            return

        # Position sizing
        size = self._risk.compute_position_size(signal, self._portfolio)
        if size <= 0:
            self._journal.log_rejected(signal.symbol, signal.direction.value, "size_zero")
            return

        # Apply exchange constraints if symbol is registered
        try:
            constraints = get_constraints(symbol)
            size = round_size(size, constraints)
            entry_rounded = round_price(signal.entry_price, constraints)
        except KeyError:
            entry_rounded = signal.entry_price

        # Simulate entry fill
        fill = self._execution.simulate_entry(signal)
        entry_price = fill.fill_price
        entry_fee = self._execution.compute_fee(entry_price, size, is_maker=False)
        slippage = abs(entry_price - signal.entry_price)

        # Check slippage anomaly
        if self._risk_monitor.on_fill_slippage(signal.entry_price, entry_price):
            self._journal.log_spread_anomaly(symbol, slippage, signal.features.get("atr", 1.0))

        # Build TP levels
        tp_sizes = [size * f for f in _TP_FRACTIONS]
        tp_levels = [TpLevel(price=p, size=s) for p, s in zip(signal.tp_prices, tp_sizes)]

        pos = Position(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=entry_price,
            entry_time=bar_time,
            equity_at_entry=self._portfolio.current_equity,
            total_size=size,
            remaining_size=size,
            stop_price=signal.stop_price,
            tp_levels=tp_levels,
            realized_pnl=-entry_fee,
            realized_fees=entry_fee,
        )
        self._portfolio.open_position(pos)

        self._journal.log_fill(
            symbol=symbol,
            direction=signal.direction.value,
            fill_price=entry_price,
            size=size,
            fee=entry_fee,
            slippage=slippage,
            equity_after=self._portfolio.current_equity,
        )

        _log.info(
            "router.fill",
            symbol=symbol,
            direction=signal.direction.value,
            price=entry_price,
            size=size,
            stop=signal.stop_price,
        )

    # ------------------------------------------------------------------
    # Exit evaluation (called on every 1m bar for open positions)
    # ------------------------------------------------------------------

    def _evaluate_exits(self, pos: Position, candle: Candle, bar_time: int) -> None:
        is_long = pos.direction is Direction.LONG

        stop_hit = (
            (is_long and candle.low <= pos.stop_price)
            or (not is_long and candle.high >= pos.stop_price)
        )

        tps_hit: list[int] = []
        for idx, tp in enumerate(pos.tp_levels):
            if tp.hit:
                continue
            if is_long and candle.high >= tp.price:
                tps_hit.append(idx)
            elif not is_long and candle.low <= tp.price:
                tps_hit.append(idx)

        if stop_hit:
            stop_dist = abs(pos.entry_price - pos.stop_price)
            fill = self._execution.simulate_stop(pos.stop_price, pos.direction, stop_dist)
            fee = self._execution.compute_fee(fill.fill_price, pos.remaining_size, is_maker=False)
            pos.apply_partial_close(fill.fill_price, pos.remaining_size, fee)
            trade = self._portfolio.close_position(pos.symbol, bar_time, ExitReason.STOP)
            if trade:
                self._journal.log_stop_hit(
                    symbol=pos.symbol,
                    fill_price=fill.fill_price,
                    size=trade.total_size,
                    pnl=trade.pnl,
                    fee=fee,
                    equity_after=self._portfolio.current_equity,
                )
                self._finalize_trade(trade)
            return

        for idx in tps_hit:
            tp = pos.tp_levels[idx]
            fill = self._execution.simulate_tp(tp.price)
            fee = self._execution.compute_fee(fill.fill_price, tp.size, is_maker=True)
            pnl_before = pos.realized_pnl
            pos.apply_partial_close(fill.fill_price, tp.size, fee, move_stop_to_be=(idx == 0))
            tp.hit = True
            partial_pnl = pos.realized_pnl - pnl_before

            self._journal.log_partial_tp(
                symbol=pos.symbol,
                tp_idx=idx,
                fill_price=fill.fill_price,
                size=tp.size,
                pnl=partial_pnl,
                fee=fee,
                equity_after=self._portfolio.current_equity,
            )

        if pos.is_closed():
            last_idx = max(tps_hit) if tps_hit else -1
            reason = {0: ExitReason.TP1, 1: ExitReason.TP2, 2: ExitReason.TP3}.get(last_idx, ExitReason.TP3)
            trade = self._portfolio.close_position(pos.symbol, bar_time, reason)
            if trade:
                self._finalize_trade(trade)

        # Update equity tracker on every 1m bar
        unrealized = self._equity_tracker.compute_unrealized(
            self._portfolio.positions, self._mid_prices
        )
        self._equity_tracker.on_equity_update(self._portfolio.current_equity, unrealized)

    def _finalize_trade(self, trade) -> None:
        self._perf_tracker.on_trade_closed(trade)
        self._equity_tracker.on_trade_closed(trade.pnl, self._portfolio.current_equity)
        self._risk_monitor.on_trade_result(trade.pnl)
        if trade.pnl < 0:
            self._risk.on_loss()
        _log.info(
            "router.trade_closed",
            symbol=trade.symbol,
            pnl=round(trade.pnl, 2),
            exit_reason=trade.exit_reason.value if hasattr(trade.exit_reason, "value") else str(trade.exit_reason),
        )

    # ------------------------------------------------------------------
    # Dashboard state
    # ------------------------------------------------------------------

    def get_state(self) -> RouterState:
        unrealized = self._equity_tracker.compute_unrealized(
            self._portfolio.positions, self._mid_prices
        )
        return RouterState(
            equity=round(self._portfolio.current_equity, 2),
            unrealized_pnl=round(unrealized, 2),
            realized_pnl=round(self._perf_tracker.total_pnl, 2),
            open_positions=self._portfolio.open_position_count(),
            drawdown=round(self._equity_tracker.current_drawdown, 4),
            rolling_sharpe=round(self._equity_tracker.rolling_sharpe(), 3),
            total_trades=self._perf_tracker.total_trades,
            win_rate=round(self._perf_tracker.win_rate, 3),
            is_killed=self._risk_monitor.is_killed,
            kill_reasons=[r.value for r in self._risk_monitor.kill_reasons],
            last_candle_ts=dict(self._risk_monitor._last_candle_ts),
        )

    def force_close_all(self, reason: str = "engine_shutdown") -> None:
        """Close all open positions at last known mid price."""
        for symbol in list(self._portfolio.positions.keys()):
            pos = self._portfolio.positions[symbol]
            price = self._mid_prices.get(symbol, pos.entry_price)
            fee = self._execution.compute_fee(price, pos.remaining_size, is_maker=False)
            pos.apply_partial_close(price, pos.remaining_size, fee)
            trade = self._portfolio.close_position(pos.symbol, int(time.time()), ExitReason.FORCED)
            if trade:
                self._journal.log_force_close(symbol, price, trade.pnl, reason)
                self._finalize_trade(trade)
