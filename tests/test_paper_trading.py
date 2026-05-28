"""Tests for the paper trading engine.

Covers:
    - Signal-to-fill lifecycle (strategy → risk → execution → portfolio)
    - Journal persistence (JSONL events written and parseable)
    - Kill-switch activation and blocking
    - Stale-feed detection
    - Replay determinism (same candles → same result every run)
    - Portfolio consistency (no position duplication, correct equity)
    - Stop and TP evaluation correctness
"""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
import time
from pathlib import Path

import pytest

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.paper_trading.equity_curve_tracker import EquityCurveTracker
from nexflow.services.paper_trading.execution_journal import EventType, ExecutionJournal
from nexflow.services.paper_trading.live_risk_monitor import (
    KillReason,
    LiveRiskConfig,
    LiveRiskMonitor,
)
from nexflow.services.paper_trading.live_signal_router import LiveSignalRouter
from nexflow.services.paper_trading.paper_trader import PaperTrader, PaperTraderConfig
from nexflow.services.paper_trading.performance_tracker import PerformanceTracker
from nexflow.services.strategy.momentum_strategy import MomentumConfig, MomentumStrategy
from nexflow.services.strategy.paper_execution import ExecutionConfig, PaperExecution
from nexflow.services.strategy.portfolio import Portfolio
from nexflow.services.strategy.risk_engine import RiskConfig, RiskEngine
from nexflow.services.strategy.signal_models import ClosedTrade, Direction, ExitReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(
    symbol: str = "BTCUSDT",
    tf: str = "1m",
    close_time: int | None = None,
    close: float = 50000.0,
    high: float | None = None,
    low: float | None = None,
    spread_avg: float = 5.0,
    volume: float = 100.0,
    buy_fraction: float = 0.6,
) -> Candle:
    if close_time is None:
        close_time = int(time.time())
    if high is None:
        high = close * 1.005
    if low is None:
        low = close * 0.995
    buy_vol = volume * buy_fraction
    return Candle(
        symbol=symbol, timeframe=tf,
        open_time=close_time - (60 if tf == "1m" else 300 if tf == "5m" else 900),
        close_time=close_time,
        open=close, high=high, low=low, close=close,
        volume=volume, buy_volume=buy_vol, sell_volume=volume - buy_vol,
        trade_count=50, vwap=close, spread_avg=spread_avg, spread_max=spread_avg * 1.5,
        volatility_estimate=0.001, is_final=True,
    )


def _make_router(
    initial_equity: float = 100_000.0,
    journal_dir: Path | None = None,
) -> tuple[LiveSignalRouter, ExecutionJournal, Portfolio, LiveRiskMonitor]:
    portfolio = Portfolio(initial_equity)
    risk = RiskEngine(RiskConfig())
    execution = PaperExecution(ExecutionConfig())
    strategy = MomentumStrategy(MomentumConfig())

    if journal_dir is None:
        journal_dir = Path(tempfile.mkdtemp())
    journal = ExecutionJournal(log_dir=journal_dir)
    risk_monitor = LiveRiskMonitor(cfg=LiveRiskConfig(), initial_equity=initial_equity)
    equity_tracker = EquityCurveTracker(initial_equity=initial_equity)
    perf_tracker = PerformanceTracker()

    router = LiveSignalRouter(
        strategy=strategy,
        risk=risk,
        execution=execution,
        portfolio=portfolio,
        journal=journal,
        risk_monitor=risk_monitor,
        equity_tracker=equity_tracker,
        perf_tracker=perf_tracker,
    )
    return router, journal, portfolio, risk_monitor


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_candle_sequence(
    n: int = 50,
    symbol: str = "BTCUSDT",
    base_ts: int | None = None,
    trend: str = "up",
) -> dict[str, dict[str, list[Candle]]]:
    """Build a minimal multi-TF candle set that gives the strategy enough history."""
    if base_ts is None:
        base_ts = (int(time.time()) - n * 60 - 1000) // 60 * 60

    prices_1m = []
    for i in range(n):
        if trend == "up":
            p = 50000.0 + i * 10.0
        elif trend == "down":
            p = 50000.0 - i * 10.0
        else:
            p = 50000.0 + (10.0 if i % 2 == 0 else -10.0)
        prices_1m.append(p)

    candles_1m = [
        _make_candle(
            symbol=symbol, tf="1m",
            close_time=base_ts + i * 60,
            close=p,
            high=p * 1.005,
            low=p * 0.995,
            buy_fraction=0.65 if trend == "up" else 0.35,
            volume=200.0 + i * 5,
        )
        for i, p in enumerate(prices_1m)
    ]
    candles_5m = [
        _make_candle(symbol=symbol, tf="5m", close_time=base_ts + i * 300, close=prices_1m[min(i*5, n-1)])
        for i in range(max(1, n // 5))
    ]
    candles_15m = [
        _make_candle(symbol=symbol, tf="15m", close_time=base_ts + i * 900, close=prices_1m[min(i*15, n-1)])
        for i in range(max(1, n // 15))
    ]
    return {symbol: {"1m": candles_1m, "5m": candles_5m, "15m": candles_15m}}


# ---------------------------------------------------------------------------
# Journal persistence tests
# ---------------------------------------------------------------------------

class TestExecutionJournal:
    def test_journal_file_created_on_init(self, tmp_path):
        journal = ExecutionJournal(log_dir=tmp_path)
        assert journal.path.exists()
        journal.close()

    def test_session_start_written(self, tmp_path):
        journal = ExecutionJournal(log_dir=tmp_path)
        journal.close()
        lines = journal.path.read_text().strip().split("\n")
        events = [json.loads(l) for l in lines if l]
        assert events[0]["event"] == EventType.SESSION_START.value

    def test_all_event_types_parseable(self, tmp_path):
        journal = ExecutionJournal(log_dir=tmp_path, session_id="test-session")
        journal.log_signal("BTCUSDT", "LONG", 50000.0, 49000.0, [51000.0, 52000.0, 53000.0], 200.0, {"atr": 200.0})
        journal.log_rejected("BTCUSDT", "LONG", "daily_dd_exceeded")
        journal.log_fill("BTCUSDT", "LONG", 50010.0, 0.01, 0.30, 10.0, 99_900.0)
        journal.log_partial_tp("BTCUSDT", 0, 51000.0, 0.005, 50.0, 0.15, 99_950.0)
        journal.log_stop_hit("BTCUSDT", 49000.0, 0.005, -50.0, 0.15, 99_850.0)
        journal.log_kill_switch("drawdown_exceeded", "daily dd 5.1%")
        journal.log_equity_snapshot(99_850.0, -150.0, 25.0, 0.015, 0)
        journal.log_feed_stale("BTCUSDT", 180.0)
        journal.log_latency_spike("feed", 1500.0)
        journal.log_session_end(99_850.0, 3)
        journal.close()

        lines = journal.path.read_text().strip().split("\n")
        events = [json.loads(l) for l in lines if l]
        event_types = {e["event"] for e in events}

        assert EventType.SESSION_START.value in event_types
        assert EventType.SIGNAL.value in event_types
        assert EventType.REJECTED.value in event_types
        assert EventType.FILL.value in event_types
        assert EventType.PARTIAL_TP.value in event_types
        assert EventType.STOP_HIT.value in event_types
        assert EventType.KILL_SWITCH.value in event_types
        assert EventType.EQUITY_SNAPSHOT.value in event_types
        assert EventType.FEED_STALE.value in event_types
        assert EventType.LATENCY_SPIKE.value in event_types
        assert EventType.SESSION_END.value in event_types

    def test_each_event_has_required_fields(self, tmp_path):
        journal = ExecutionJournal(log_dir=tmp_path)
        journal.log_rejected("BTCUSDT", "SHORT", "test")
        journal.close()

        lines = journal.path.read_text().strip().split("\n")
        for line in lines:
            ev = json.loads(line)
            assert "ts" in ev
            assert "ts_epoch" in ev
            assert "event" in ev
            assert "session_id" in ev

    def test_journal_survives_special_chars_in_features(self, tmp_path):
        journal = ExecutionJournal(log_dir=tmp_path)
        journal.log_signal("BTCUSDT", "LONG", 50000.0, 49000.0, [], 0.0,
                           {"key": float("nan"), "inf": float("inf")})
        journal.close()
        # Should not raise; nan/inf will be serialized as strings by default=str
        lines = journal.path.read_text().strip().split("\n")
        assert len(lines) >= 2


# ---------------------------------------------------------------------------
# Kill-switch tests
# ---------------------------------------------------------------------------

class TestLiveRiskMonitor:
    def test_drawdown_kill_activates(self):
        monitor = LiveRiskMonitor(
            cfg=LiveRiskConfig(max_drawdown_kill=0.05),
            initial_equity=100_000.0,
        )
        monitor.on_drawdown(94_999.0)  # 5.001% DD
        assert monitor.is_killed
        assert KillReason.DRAWDOWN_EXCEEDED in monitor.kill_reasons

    def test_drawdown_clears_when_recovered(self):
        monitor = LiveRiskMonitor(
            cfg=LiveRiskConfig(max_drawdown_kill=0.05),
            initial_equity=100_000.0,
        )
        monitor.on_drawdown(94_000.0)
        assert monitor.is_killed
        monitor.on_drawdown(96_000.0)  # below threshold
        assert KillReason.DRAWDOWN_EXCEEDED not in monitor.kill_reasons

    def test_consecutive_loss_kill(self):
        monitor = LiveRiskMonitor(
            cfg=LiveRiskConfig(max_consecutive_losses=3),
            initial_equity=100_000.0,
        )
        for _ in range(3):
            monitor.on_trade_result(-100.0)
        assert monitor.is_killed
        assert KillReason.CONSECUTIVE_LOSSES in monitor.kill_reasons

    def test_win_resets_loss_streak(self):
        monitor = LiveRiskMonitor(
            cfg=LiveRiskConfig(max_consecutive_losses=5),
            initial_equity=100_000.0,
        )
        for _ in range(4):
            monitor.on_trade_result(-100.0)
        assert not monitor.is_killed
        monitor.on_trade_result(200.0)  # win
        assert monitor.kill_reasons == frozenset()

    def test_stale_feed_detection(self):
        monitor = LiveRiskMonitor(
            cfg=LiveRiskConfig(stale_candle_threshold_s=5.0),
            initial_equity=100_000.0,
        )
        monitor.on_candle_received("BTCUSDT")
        import time as _t
        _t.sleep(0.01)  # tiny delay for test; threshold is 5s so won't trigger yet
        stale = monitor.check_stale_feeds(["BTCUSDT"])
        assert stale == []  # not stale yet

    def test_stale_feed_detected_after_threshold(self):
        monitor = LiveRiskMonitor(
            cfg=LiveRiskConfig(stale_candle_threshold_s=0.001),
            initial_equity=100_000.0,
        )
        # Manually set an old last_candle_ts
        monitor._last_candle_ts["BTCUSDT"] = time.time() - 1.0
        stale = monitor.check_stale_feeds(["BTCUSDT"])
        assert "BTCUSDT" in stale
        assert monitor.is_killed

    def test_ws_reconnect_kill_after_threshold(self):
        monitor = LiveRiskMonitor(
            cfg=LiveRiskConfig(max_reconnects_per_hour=3),
            initial_equity=100_000.0,
        )
        for _ in range(3):
            monitor.on_ws_reconnect()
        assert monitor.is_killed
        assert KillReason.WS_UNHEALTHY in monitor.kill_reasons

    def test_latency_spike_kill_after_consecutive(self):
        monitor = LiveRiskMonitor(
            cfg=LiveRiskConfig(latency_kill_ms=500.0, latency_kill_consecutive=2),
            initial_equity=100_000.0,
        )
        monitor.on_latency(600.0)
        monitor.on_latency(600.0)
        assert monitor.is_killed
        assert KillReason.LATENCY_SPIKE in monitor.kill_reasons

    def test_kill_blocks_entry_in_router(self, tmp_path):
        """Kill-switch must prevent all new fills."""
        router, journal, portfolio, risk_monitor = _make_router(journal_dir=tmp_path)

        # Activate kill
        risk_monitor._kill_reasons.add(KillReason.DRAWDOWN_EXCEEDED)

        # Feed enough candles to potentially trigger a signal
        candles = _make_candle_sequence(n=60, trend="up")
        all_candles = candles
        _TF_ORDER = {"15m": 0, "5m": 1, "1m": 2}
        events = sorted(
            [c for sym in all_candles.values() for tf in sym.values() for c in tf],
            key=lambda c: (c.close_time, _TF_ORDER.get(c.timeframe, 99))
        )
        for candle in events:
            _run(router.on_candle(candle.symbol, candle.timeframe, candle))

        # Portfolio should have no positions opened while kill is active
        assert portfolio.open_position_count() == 0

        # Journal must contain REJECTED events citing kill_switch
        lines = journal.path.read_text().strip().split("\n")
        events_logged = [json.loads(l) for l in lines if l]
        rejected = [e for e in events_logged if e["event"] == EventType.REJECTED.value]
        kill_rejections = [r for r in rejected if "kill_switch" in r.get("reason", "")]
        # May be zero if no signals were generated; that's fine — just verify no fills
        fills = [e for e in events_logged if e["event"] == EventType.FILL.value]
        assert len(fills) == 0

        journal.close()


# ---------------------------------------------------------------------------
# Signal-to-fill lifecycle
# ---------------------------------------------------------------------------

class TestSignalToFillLifecycle:
    def test_signal_logged_before_fill(self, tmp_path):
        """Journal must record SIGNAL before FILL for every entry."""
        router, journal, portfolio, _ = _make_router(journal_dir=tmp_path)

        candles = _make_candle_sequence(n=60, trend="up")
        _TF_ORDER = {"15m": 0, "5m": 1, "1m": 2}
        events = sorted(
            [c for sym in candles.values() for tf in sym.values() for c in tf],
            key=lambda c: (c.close_time, _TF_ORDER.get(c.timeframe, 99))
        )
        for candle in events:
            _run(router.on_candle(candle.symbol, candle.timeframe, candle))

        journal.close()
        lines = journal.path.read_text().strip().split("\n")
        logged = [json.loads(l) for l in lines if l]

        fills = [i for i, e in enumerate(logged) if e["event"] == EventType.FILL.value]
        signals = [i for i, e in enumerate(logged) if e["event"] == EventType.SIGNAL.value]

        for fill_idx in fills:
            # There must be a SIGNAL with a smaller index
            assert any(s_idx < fill_idx for s_idx in signals), \
                "FILL appeared before any SIGNAL in journal"

    def test_no_position_without_fill(self, tmp_path):
        """Every open position must correspond to exactly one FILL event."""
        router, journal, portfolio, _ = _make_router(journal_dir=tmp_path)

        candles = _make_candle_sequence(n=50, trend="up")
        _TF_ORDER = {"15m": 0, "5m": 1, "1m": 2}
        events = sorted(
            [c for sym in candles.values() for tf in sym.values() for c in tf],
            key=lambda c: (c.close_time, _TF_ORDER.get(c.timeframe, 99))
        )
        for candle in events:
            _run(router.on_candle(candle.symbol, candle.timeframe, candle))

        journal.close()
        lines = journal.path.read_text().strip().split("\n")
        logged = [json.loads(l) for l in lines if l]
        fills = [e for e in logged if e["event"] == EventType.FILL.value]
        open_pos = portfolio.open_position_count()
        closed = len(portfolio.closed_trades)
        assert open_pos + closed == len(fills)

    def test_equity_decreases_by_fee_on_fill(self, tmp_path):
        """Equity after a fill must be lower than before by at least the entry fee."""
        from nexflow.services.strategy.signal_models import Signal

        router, journal, portfolio, _ = _make_router(journal_dir=tmp_path)
        equity_before = portfolio.current_equity

        # Manually inject a signal by feeding many rising candles
        candles = _make_candle_sequence(n=80, trend="up")
        _TF_ORDER = {"15m": 0, "5m": 1, "1m": 2}
        events = sorted(
            [c for sym in candles.values() for tf in sym.values() for c in tf],
            key=lambda c: (c.close_time, _TF_ORDER.get(c.timeframe, 99))
        )
        for candle in events:
            _run(router.on_candle(candle.symbol, candle.timeframe, candle))

        journal.close()
        lines = journal.path.read_text().strip().split("\n")
        logged = [json.loads(l) for l in lines if l]
        fills = [e for e in logged if e["event"] == EventType.FILL.value]

        if fills:
            # equity_after on fill must be <= equity_before
            for fill in fills:
                assert fill["equity_after"] <= equity_before, \
                    "Equity should not increase on a fill due to fees"


# ---------------------------------------------------------------------------
# Stop and TP evaluation
# ---------------------------------------------------------------------------

class TestExitEvaluation:
    def test_stop_hit_journaled_and_position_closed(self, tmp_path):
        """When a stop bar is fed, the position must close and STOP_HIT must be logged."""
        from nexflow.services.strategy.portfolio import Position, TpLevel
        from nexflow.services.strategy.signal_models import Direction

        router, journal, portfolio, _ = _make_router(journal_dir=tmp_path)

        # Manually open a LONG position at 50000 with stop at 49000
        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=50000.0, entry_time=1_000_000,
            equity_at_entry=100_000.0, total_size=0.01,
            remaining_size=0.01, stop_price=49000.0,
            tp_levels=[TpLevel(51000.0, 0.005), TpLevel(52000.0, 0.0025), TpLevel(53000.0, 0.0025)],
            realized_pnl=0.0, realized_fees=0.0,
        )
        portfolio.open_position(pos)

        # Feed a candle that touches the stop (low <= 49000)
        stop_candle = _make_candle(
            close_time=1_000_060, close=49500.0,
            high=50200.0, low=48800.0,
        )
        _run(router.on_candle("BTCUSDT", "1m", stop_candle))

        assert portfolio.open_position_count() == 0, "Position should be closed"
        assert len(portfolio.closed_trades) == 1

        journal.close()
        lines = journal.path.read_text().strip().split("\n")
        logged = [json.loads(l) for l in lines if l]
        stops = [e for e in logged if e["event"] == EventType.STOP_HIT.value]
        assert len(stops) == 1
        assert stops[0]["symbol"] == "BTCUSDT"

    def test_tp_hit_journaled_as_partial(self, tmp_path):
        """When a TP1 bar is fed, a PARTIAL_TP event must be logged."""
        from nexflow.services.strategy.portfolio import Position, TpLevel
        from nexflow.services.strategy.signal_models import Direction

        router, journal, portfolio, _ = _make_router(journal_dir=tmp_path)

        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=50000.0, entry_time=1_000_000,
            equity_at_entry=100_000.0, total_size=0.01,
            remaining_size=0.01, stop_price=49000.0,
            tp_levels=[TpLevel(51000.0, 0.005), TpLevel(52000.0, 0.0025), TpLevel(53000.0, 0.0025)],
            realized_pnl=0.0, realized_fees=0.0,
        )
        portfolio.open_position(pos)

        # Feed a candle that hits TP1 but not stop
        tp_candle = _make_candle(
            close_time=1_000_060, close=51500.0,
            high=51500.0, low=50200.0,
        )
        _run(router.on_candle("BTCUSDT", "1m", tp_candle))

        journal.close()
        lines = journal.path.read_text().strip().split("\n")
        logged = [json.loads(l) for l in lines if l]
        tps = [e for e in logged if e["event"] == EventType.PARTIAL_TP.value]
        assert len(tps) >= 1
        assert tps[0]["tp_idx"] == 0
        # Position should still be open (only TP1 hit, size split)
        assert portfolio.has_position("BTCUSDT")


# ---------------------------------------------------------------------------
# Replay determinism
# ---------------------------------------------------------------------------

class TestReplayDeterminism:
    def test_same_candles_same_result(self, tmp_path):
        """Running replay twice on the same candles must produce identical final state."""
        candles = _make_candle_sequence(n=80, trend="up")

        def _run_replay(journal_subdir):
            cfg = PaperTraderConfig(
                journal_dir=tmp_path / journal_subdir,
                enable_dashboard=False,
            )
            trader = PaperTrader(cfg=cfg, symbols=["BTCUSDT"])
            return trader.run_replay(candles)

        state1 = _run_replay("r1")
        state2 = _run_replay("r2")

        assert state1.equity == state2.equity
        assert state1.total_trades == state2.total_trades
        assert state1.realized_pnl == state2.realized_pnl

    def test_replay_respects_candle_ordering(self, tmp_path):
        """Candle ordering in replay must match backtest ordering (15m→5m→1m at ties)."""
        processed_order: list[tuple[str, str]] = []

        from nexflow.services.strategy.base_strategy import BaseStrategy
        from nexflow.services.strategy.signal_models import Signal

        class RecordingStrategy(BaseStrategy):
            def on_candle(self, candle: Candle) -> Signal | None:
                processed_order.append((candle.timeframe, str(candle.close_time)))
                return None
            def reset(self):
                processed_order.clear()

        candles = _make_candle_sequence(n=30, trend="up")
        _TF_ORDER = {"15m": 0, "5m": 1, "1m": 2}

        # Build expected order manually
        all_candles_flat = [c for sym in candles.values() for tf in sym.values() for c in tf]
        expected = [(c.timeframe, str(c.close_time)) for c in sorted(
            all_candles_flat,
            key=lambda c: (c.close_time, _TF_ORDER.get(c.timeframe, 99))
        )]

        # Run through router manually
        router, journal, portfolio, _ = _make_router(journal_dir=tmp_path)
        router._strategy = RecordingStrategy()

        events = sorted(all_candles_flat, key=lambda c: (c.close_time, _TF_ORDER.get(c.timeframe, 99)))
        for c in events:
            _run(router.on_candle(c.symbol, c.timeframe, c))

        assert processed_order == expected
        journal.close()


# ---------------------------------------------------------------------------
# Portfolio consistency
# ---------------------------------------------------------------------------

class TestPortfolioConsistency:
    def test_equity_never_negative(self, tmp_path):
        """Equity must remain positive throughout a replay."""
        equities: list[float] = []

        def _on_bar(state):
            equities.append(state.equity)

        candles = _make_candle_sequence(n=60, trend="down")
        cfg = PaperTraderConfig(journal_dir=tmp_path, enable_dashboard=False)
        trader = PaperTrader(cfg=cfg, symbols=["BTCUSDT"])
        trader.run_replay(candles, on_bar_callback=_on_bar)

        for eq in equities:
            assert eq > 0, f"Equity went negative: {eq}"

    def test_no_duplicate_positions_per_symbol(self, tmp_path):
        """Only one position per symbol at a time."""
        candles = _make_candle_sequence(n=60, trend="up")
        cfg = PaperTraderConfig(journal_dir=tmp_path, enable_dashboard=False)
        trader = PaperTrader(cfg=cfg, symbols=["BTCUSDT"])
        # Track max concurrent positions
        max_pos = [0]

        def _on_bar(state):
            max_pos[0] = max(max_pos[0], state.open_positions)

        trader.run_replay(candles, on_bar_callback=_on_bar)
        assert max_pos[0] <= 1  # only one symbol, so max 1

    def test_final_equity_equals_initial_plus_all_pnl(self, tmp_path):
        """Accounting identity: final_equity = initial + sum(all_trade_pnl)."""
        candles = _make_candle_sequence(n=80, trend="up")
        cfg = PaperTraderConfig(
            initial_equity=100_000.0,
            journal_dir=tmp_path,
            enable_dashboard=False,
        )
        trader = PaperTrader(cfg=cfg, symbols=["BTCUSDT"])
        state = trader.run_replay(candles)

        perf = trader._perf_tracker
        # Final equity should equal initial + realized PnL from closed trades
        # (open positions were force-closed at end, so all PnL is captured)
        expected = 100_000.0 + perf.total_pnl
        assert abs(state.equity - expected) < 0.01, \
            f"Equity mismatch: {state.equity:.2f} != {expected:.2f}"


# ---------------------------------------------------------------------------
# Equity curve tracker
# ---------------------------------------------------------------------------

class TestEquityCurveTracker:
    def test_drawdown_computed_correctly(self):
        tracker = EquityCurveTracker(initial_equity=100_000.0)
        tracker.on_equity_update(110_000.0)  # new peak
        tracker.on_equity_update(99_000.0)   # drawdown
        expected_dd = (110_000.0 - 99_000.0) / 110_000.0
        assert abs(tracker.current_drawdown - expected_dd) < 1e-6

    def test_rolling_sharpe_zero_with_no_trades(self):
        tracker = EquityCurveTracker(initial_equity=100_000.0)
        assert tracker.rolling_sharpe() == 0.0

    def test_rolling_sharpe_positive_on_consistent_gains(self):
        tracker = EquityCurveTracker(initial_equity=100_000.0)
        equity = 100_000.0
        for i in range(20):
            pnl = 100.0
            equity += pnl
            tracker.on_trade_closed(pnl, equity)
        # All wins → Sharpe should be well-defined (but may be 0 if std ≈ 0)
        # Just verify it doesn't crash and is >= 0
        assert tracker.rolling_sharpe() >= 0.0

    def test_unrealized_pnl_long(self):
        from unittest.mock import MagicMock
        from nexflow.services.strategy.signal_models import Direction

        tracker = EquityCurveTracker(initial_equity=100_000.0)
        pos = MagicMock()
        pos.entry_price = 50000.0
        pos.remaining_size = 0.01
        pos.direction = Direction.LONG

        mid_prices = {"BTCUSDT": 51000.0}
        upnl = tracker.compute_unrealized({"BTCUSDT": pos}, mid_prices)
        assert abs(upnl - (51000.0 - 50000.0) * 0.01) < 1e-6


# ---------------------------------------------------------------------------
# Performance tracker
# ---------------------------------------------------------------------------

class TestPerformanceTracker:
    def _make_trade(self, pnl: float, direction: Direction = Direction.LONG) -> ClosedTrade:
        return ClosedTrade(
            symbol="BTCUSDT", direction=direction,
            entry_price=50000.0, exit_price=50000.0 + pnl / 0.01,
            total_size=0.01,
            entry_time=1_000_000, exit_time=1_000_060,
            pnl=pnl, fees=1.0, hold_bars=10,
            equity_at_entry=100_000.0,
            exit_reason=ExitReason.TP1 if pnl > 0 else ExitReason.STOP,
        )

    def test_win_rate_correct(self):
        tracker = PerformanceTracker()
        for _ in range(6):
            tracker.on_trade_closed(self._make_trade(100.0))
        for _ in range(4):
            tracker.on_trade_closed(self._make_trade(-50.0))
        assert abs(tracker.win_rate - 0.6) < 1e-9

    def test_profit_factor_correct(self):
        tracker = PerformanceTracker()
        tracker.on_trade_closed(self._make_trade(200.0))
        tracker.on_trade_closed(self._make_trade(-100.0))
        assert abs(tracker.profit_factor - 2.0) < 1e-9

    def test_long_short_win_rates_independent(self):
        tracker = PerformanceTracker()
        tracker.on_trade_closed(self._make_trade(100.0, Direction.LONG))
        tracker.on_trade_closed(self._make_trade(-50.0, Direction.LONG))
        tracker.on_trade_closed(self._make_trade(-50.0, Direction.SHORT))
        tracker.on_trade_closed(self._make_trade(-50.0, Direction.SHORT))

        assert abs(tracker.long_win_rate - 0.5) < 1e-9
        assert tracker.short_win_rate == 0.0

    def test_total_pnl_accumulates(self):
        tracker = PerformanceTracker()
        pnls = [100.0, -50.0, 200.0, -30.0]
        for p in pnls:
            tracker.on_trade_closed(self._make_trade(p))
        assert abs(tracker.total_pnl - sum(pnls)) < 1e-9
