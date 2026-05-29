"""Paper trading engine — top-level orchestrator.

Two modes:
    live    — connects to Bitget WebSocket, feeds live trades into CandleEngine,
              routes finalized candles through strategy stack
    replay  — reads parquet candle files and emits them through the identical
              stack for deterministic debugging

Both modes use the same LiveSignalRouter, journal, risk monitor, and portfolio.
The only difference is the source of finalized candles.

Dashboard:
    Printed to stdout on a configurable interval using ANSI clear-to-top.
    No external dependencies (no rich/curses). Safe over SSH and in Docker.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from nexflow.config import NexFlowConfig, get_config
from nexflow.logging import get_logger
from nexflow.services.candles.candle_engine import Candle, CandleEngine, TIMEFRAMES
from nexflow.services.market_data.bitget_ws import BitgetWSClient
from nexflow.services.market_data.telemetry_monitor import TelemetryMonitor
from nexflow.services.paper_trading.equity_curve_tracker import EquityCurveTracker
from nexflow.services.paper_trading.execution_journal import ExecutionJournal
from nexflow.services.paper_trading.live_risk_monitor import LiveRiskConfig, LiveRiskMonitor
from nexflow.services.paper_trading.live_signal_router import LiveSignalRouter, RouterState
from nexflow.services.paper_trading.performance_tracker import PerformanceTracker
from nexflow.services.strategy.momentum_strategy import MomentumConfig, MomentumStrategy
from nexflow.services.strategy.paper_execution import ExecutionConfig, PaperExecution
from nexflow.services.strategy.portfolio import Portfolio
from nexflow.services.strategy.risk_engine import RiskConfig, RiskEngine

_log = get_logger(__name__)


@dataclass
class PaperTraderConfig:
    initial_equity: float = 100_000.0
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    live_risk: LiveRiskConfig = field(default_factory=LiveRiskConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    journal_dir: Path = field(default_factory=lambda: Path("logs/paper"))
    candle_dir: Path = field(default_factory=lambda: Path("data/candles"))
    dashboard_interval_s: float = 5.0
    equity_snapshot_interval_s: float = 60.0
    health_check_interval_s: float = 30.0
    enable_dashboard: bool = True


class PaperTrader:
    """Paper trading engine.

    Usage — live mode::

        trader = PaperTrader(cfg, symbols=["BTCUSDT"])
        asyncio.run(trader.run_live(exchange_cfg, market_data_cfg))

    Usage — replay mode::

        trader = PaperTrader(cfg, symbols=["BTCUSDT"])
        trader.run_replay(all_candles)
    """

    def __init__(
        self,
        cfg: PaperTraderConfig | None = None,
        symbols: list[str] | None = None,
        trade_symbols: list[str] | None = None,
    ) -> None:
        self._cfg = cfg or PaperTraderConfig()
        self._symbols = symbols or ["BTCUSDT"]
        # trade_symbols: subset that are allowed to take entries.
        # Others still receive candles (for strategy warmup) but are blocked.
        self._trade_symbols = set(trade_symbols) if trade_symbols else set(self._symbols)
        self._mid_prices: dict[str, float] = {}

        # Build shared subsystems
        self._portfolio = Portfolio(self._cfg.initial_equity)
        self._risk = RiskEngine(self._cfg.risk)
        self._execution = PaperExecution(self._cfg.execution)
        self._strategy = MomentumStrategy(self._cfg.momentum)
        self._journal = ExecutionJournal(log_dir=self._cfg.journal_dir)
        self._risk_monitor = LiveRiskMonitor(
            cfg=self._cfg.live_risk,
            initial_equity=self._cfg.initial_equity,
        )
        self._equity_tracker = EquityCurveTracker(
            initial_equity=self._cfg.initial_equity,
            snapshot_interval_s=self._cfg.equity_snapshot_interval_s,
        )
        self._perf_tracker = PerformanceTracker()

        self._router = LiveSignalRouter(
            strategy=self._strategy,
            risk=self._risk,
            execution=self._execution,
            portfolio=self._portfolio,
            journal=self._journal,
            risk_monitor=self._risk_monitor,
            equity_tracker=self._equity_tracker,
            perf_tracker=self._perf_tracker,
            mid_prices=self._mid_prices,
            trade_symbols=self._trade_symbols,
        )

        self._running = False

    # ------------------------------------------------------------------
    # Live mode
    # ------------------------------------------------------------------

    async def run_live(self, nexflow_cfg: NexFlowConfig | None = None) -> None:
        """Connect to Bitget WebSocket and run indefinitely."""
        cfg = nexflow_cfg or get_config()
        _log.info("paper_trader.live_start", symbols=self._symbols)

        candle_engine = CandleEngine(config=cfg)
        candle_engine.on_candle_close(self._router.on_candle)

        telemetry = TelemetryMonitor(config=cfg)

        ws_client = BitgetWSClient(config=cfg, symbols=self._symbols)

        async def _on_state(state):
            # Track WS latency for risk monitor
            if state.exchange_ts_ms:
                latency_ms = (time.time() * 1000) - state.exchange_ts_ms
                is_spike = self._risk_monitor.on_latency(latency_ms)
                if is_spike:
                    self._journal.log_latency_spike("feed", latency_ms)
            # Update mid price for this specific symbol only
            mp = state.mid_price
            if mp and mp > 0:
                self._mid_prices[state.symbol] = mp

        ws_client.on_update(_on_state)
        candle_engine.attach(ws_client)
        telemetry.attach(ws_client)

        self._running = True
        tasks = [
            asyncio.create_task(ws_client.start()),
            asyncio.create_task(self._health_check_loop(ws_client)),
        ]
        if self._cfg.enable_dashboard:
            tasks.append(asyncio.create_task(self._dashboard_loop()))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Replay mode
    # ------------------------------------------------------------------

    def run_replay(
        self,
        all_candles: dict[str, dict[str, list[Candle]]],
        on_bar_callback: Callable[[RouterState], None] | None = None,
    ) -> RouterState:
        """Feed historical candles through the live stack synchronously.

        Candles are emitted in the same deterministic order as BacktestRunner:
        ascending close_time, 15m → 5m → 1m at ties.

        Returns final RouterState.
        """
        _log.info("paper_trader.replay_start", symbols=list(all_candles.keys()))

        self._strategy.reset()
        _TF_ORDER = {"15m": 0, "5m": 1, "1m": 2}

        events: list[Candle] = []
        for sym_candles in all_candles.values():
            for tf_candles in sym_candles.values():
                events.extend(tf_candles)
        events.sort(key=lambda c: (c.close_time, _TF_ORDER.get(c.timeframe, 99)))

        loop = asyncio.new_event_loop()
        try:
            for candle in events:
                loop.run_until_complete(
                    self._router.on_candle(candle.symbol, candle.timeframe, candle)
                )
                if on_bar_callback and candle.timeframe == "1m":
                    on_bar_callback(self._router.get_state())
        finally:
            loop.close()

        self._router.force_close_all("replay_end")
        return self._router.get_state()

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def _health_check_loop(self, ws_client: BitgetWSClient) -> None:
        while self._running:
            await asyncio.sleep(self._cfg.health_check_interval_s)

            # Check WS reconnect count
            reconnects = ws_client.reconnect_count
            if reconnects > 0:
                # TelemetryMonitor tracks reconnect events; we update risk monitor here
                pass

            # Check stale feeds
            stale = self._risk_monitor.check_stale_feeds(self._symbols)
            for sym in stale:
                gap = time.time() - self._risk_monitor._last_candle_ts.get(sym, time.time())
                self._journal.log_feed_stale(sym, gap)
                _log.warning("paper_trader.stale_feed", symbol=sym, gap_s=gap)

            # Log kill status changes
            if self._risk_monitor.is_killed:
                reasons = ", ".join(r.value for r in self._risk_monitor.kill_reasons)
                _log.error("paper_trader.kill_active", reasons=reasons)
                self._journal.log_kill_switch(reasons)

            # Periodic equity snapshot
            state = self._router.get_state()
            self._journal.log_equity_snapshot(
                equity=state.equity,
                realized_pnl=state.realized_pnl,
                unrealized_pnl=state.unrealized_pnl,
                drawdown=state.drawdown,
                open_positions=state.open_positions,
            )

    async def _dashboard_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._cfg.dashboard_interval_s)
            _print_dashboard(
                state=self._router.get_state(),
                perf=self._perf_tracker.summary_dict(),
                equity=self._equity_tracker.summary(),
                risk=self._risk_monitor.status_summary(),
                initial_equity=self._cfg.initial_equity,
            )

    def _shutdown(self) -> None:
        self._running = False
        _log.info("paper_trader.shutdown")
        self._router.force_close_all("engine_shutdown")
        self._journal.log_session_end(
            final_equity=self._portfolio.current_equity,
            total_trades=self._perf_tracker.total_trades,
        )
        self._journal.close()


# ---------------------------------------------------------------------------
# Terminal dashboard
# ---------------------------------------------------------------------------

_ANSI_CLEAR = "\033[H\033[2J"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _colour(value: float, positive_good: bool = True) -> str:
    if value > 0:
        return (_GREEN if positive_good else _RED) + f"{value:+.2f}" + _RESET
    if value < 0:
        return (_RED if positive_good else _GREEN) + f"{value:+.2f}" + _RESET
    return f"{value:+.2f}"


def _print_dashboard(
    state: RouterState,
    perf: dict,
    equity: dict,
    risk: dict,
    initial_equity: float,
) -> None:
    out = []
    if sys.stdout.isatty():
        out.append(_ANSI_CLEAR)

    bar = "─" * 60
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    kill_label = (
        f"{_RED}{_BOLD}⚠ KILL ACTIVE: {', '.join(state.kill_reasons)}{_RESET}"
        if state.is_killed else
        f"{_GREEN}● LIVE{_RESET}"
    )

    dd_colour = _RED if state.drawdown > 0.02 else (_YELLOW if state.drawdown > 0.005 else _GREEN)
    sharpe_colour = _GREEN if state.rolling_sharpe > 0 else _RED

    out += [
        f"{_BOLD}╔══ NexFlow Paper Trader ══════════════════════════════════╗{_RESET}",
        f"  {now}    {kill_label}",
        bar,
        f"  {'Equity':20s}: {_BOLD}{state.equity:>12,.2f}{_RESET} USDT",
        f"  {'Realized PnL':20s}: {_colour(state.realized_pnl)}",
        f"  {'Unrealized PnL':20s}: {_colour(state.unrealized_pnl)}",
        f"  {'Drawdown':20s}: {dd_colour}{state.drawdown*100:>8.3f}%{_RESET}",
        f"  {'Rolling Sharpe':20s}: {sharpe_colour}{state.rolling_sharpe:>8.3f}{_RESET}",
        bar,
        f"  {'Open Positions':20s}: {state.open_positions}",
        f"  {'Total Trades':20s}: {state.total_trades}",
        f"  {'Win Rate':20s}: {perf.get('win_rate_pct', 0.0):.1f}%",
        f"  {'Profit Factor':20s}: {perf.get('profit_factor', 0.0)}",
        f"  {'Expectancy (R)':20s}: {perf.get('expectancy_R', 0.0):.4f}",
        bar,
        f"  {'Long  trades':20s}: {perf.get('long_trades', 0):>4}  WR {perf.get('long_wr_pct', 0.0):.1f}%",
        f"  {'Short trades':20s}: {perf.get('short_trades', 0):>4}  WR {perf.get('short_wr_pct', 0.0):.1f}%",
        bar,
        f"  {_CYAN}Maker Fill Tracking{_RESET}",
        f"  {'TP limits placed':20s}: {state.maker_attempts}",
        f"  {'TP fills':20s}: {state.maker_fills}",
        f"  {'Maker fill rate':20s}: {(state.maker_fills/state.maker_attempts*100) if state.maker_attempts else 0:.1f}%",
        f"  {'TP1 hit rate':20s}: {state.tp1_hit_rate*100:.1f}%",
        f"  {'TP2 hit rate':20s}: {state.tp2_hit_rate*100:.1f}%",
        f"  {'TP3 hit rate':20s}: {state.tp3_hit_rate*100:.1f}%",
        f"  {'Stop hit rate':20s}: {state.stop_hit_rate*100:.1f}%",
        f"  {'Avg fill latency':20s}: {state.avg_maker_latency_bars:.1f} bars",
        bar,
        f"  {_CYAN}Risk Monitor{_RESET}",
        f"  {'Reconnects/hr':20s}: {risk.get('reconnects_last_hour', 0)}",
        f"  {'Consec losses':20s}: {risk.get('consecutive_losses', 0)}",
        f"  {'Latency spikes':20s}: {risk.get('consecutive_latency_spikes', 0)}",
        f"{_BOLD}╚{'═'*58}╝{_RESET}",
    ]
    print("\n".join(out), flush=True)
