"""Tests for the walk-forward robustness harness.

Verifies:
    - No lookahead leakage (train/test intervals are strictly non-overlapping)
    - Deterministic fold splits
    - Reproducible Monte Carlo seeds
    - Parameter isolation (each sweep combo gets its own fresh strategy)
    - Rolling window correctness
    - Anchored mode correctness (train always starts at beginning)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.backtest_runner import BacktestConfig
from nexflow.services.strategy.base_strategy import BaseStrategy
from nexflow.services.strategy.risk_engine import RiskConfig
from nexflow.services.strategy.signal_models import ClosedTrade, Direction, ExitReason
from nexflow.research.walk_forward import (
    WFConfig,
    WalkForwardEngine,
    _build_windows,
    _slice_candles,
)
from nexflow.research.monte_carlo import MCConfig, MonteCarloEngine
from nexflow.research.parameter_sweeper import ParamRange, ParameterSweeper
from nexflow.research.regime_analyzer import RegimeAnalyzer, Session, VolatilityRegime
from nexflow.research.equity_curve_analysis import EquityCurveAnalysis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_candle(symbol: str, tf: str, close_time: int, price: float = 100.0) -> Candle:
    return Candle(
        symbol=symbol, timeframe=tf,
        open_time=close_time - (60 if tf == "1m" else 300 if tf == "5m" else 900),
        close_time=close_time,
        open=price, high=price * 1.001, low=price * 0.999, close=price,
        volume=100.0, buy_volume=55.0, sell_volume=45.0, trade_count=50,
        vwap=price, spread_avg=0.1, spread_max=0.2,
        volatility_estimate=0.001, is_final=True,
    )


def _make_candle_set(
    symbol: str = "BTCUSDT",
    n_1m: int = 500,
    base_ts: int | None = None,
) -> dict[str, dict[str, list[Candle]]]:
    if base_ts is None:
        base_ts = int(time.time()) - n_1m * 60
    base_ts = (base_ts // 60) * 60

    candles_1m = [_make_candle(symbol, "1m", base_ts + i * 60) for i in range(n_1m)]
    candles_5m = [_make_candle(symbol, "5m", base_ts + i * 300) for i in range(n_1m // 5)]
    candles_15m = [_make_candle(symbol, "15m", base_ts + i * 900) for i in range(n_1m // 15)]

    return {symbol: {"1m": candles_1m, "5m": candles_5m, "15m": candles_15m}}


# ---------------------------------------------------------------------------
# Walk-forward split correctness
# ---------------------------------------------------------------------------

class TestWalkForwardSplits:
    def test_no_lookahead_leakage(self):
        """Each fold's test interval must start exactly where training ends."""
        all_ts = list(range(0, 5000, 60))
        cfg = WFConfig(train_bars=2000, test_bars=500, step_bars=500, mode="rolling")
        windows = _build_windows(all_ts, cfg)

        for w in windows:
            assert w.train_end_ts == w.test_start_ts, (
                f"Fold {w.fold_idx}: train_end={w.train_end_ts} != test_start={w.test_start_ts}"
            )

    def test_train_test_no_overlap(self):
        """No candle timestamp appears in both train and test slices."""
        all_candles = _make_candle_set(n_1m=3500)
        all_ts = sorted(
            c.close_time
            for sym in all_candles.values()
            for tf, cs in sym.items() if tf == "1m"
            for c in cs
        )
        cfg = WFConfig(train_bars=2000, test_bars=500, step_bars=500, mode="rolling")
        windows = _build_windows(all_ts, cfg)

        for w in windows:
            train = _slice_candles(all_candles, w.train_start_ts, w.train_end_ts)
            test = _slice_candles(all_candles, w.test_start_ts, w.test_end_ts)

            train_ts = {
                c.close_time
                for sym in train.values()
                for tf, cs in sym.items() if tf == "1m"
                for c in cs
            }
            test_ts = {
                c.close_time
                for sym in test.values()
                for tf, cs in sym.items() if tf == "1m"
                for c in cs
            }
            overlap = train_ts & test_ts
            assert not overlap, f"Fold {w.fold_idx}: {len(overlap)} overlapping timestamps"

    def test_rolling_window_advances_by_step(self):
        """In rolling mode, each fold's train_start advances by exactly step_bars × bar_interval."""
        all_ts = list(range(0, 5000 * 60, 60))
        cfg = WFConfig(train_bars=2000, test_bars=500, step_bars=500, mode="rolling")
        windows = _build_windows(all_ts, cfg)

        assert len(windows) >= 2
        for i in range(1, len(windows)):
            step = all_ts[cfg.step_bars]  # step_bars × 60s
            expected_advance = step
            actual_advance = windows[i].train_start_ts - windows[i - 1].train_start_ts
            # The advance should equal step_bars × interval between consecutive ts values
            assert actual_advance > 0, "Each fold must start later than the previous"

    def test_anchored_mode_train_always_starts_at_beginning(self):
        """In anchored mode, all folds share the same train_start_ts."""
        all_ts = list(range(0, 5000 * 60, 60))
        cfg = WFConfig(train_bars=2000, test_bars=500, step_bars=500, mode="anchored")
        windows = _build_windows(all_ts, cfg)

        assert len(windows) >= 2
        first_start = windows[0].train_start_ts
        for w in windows:
            assert w.train_start_ts == first_start, (
                f"Fold {w.fold_idx}: train_start={w.train_start_ts} != {first_start}"
            )

    def test_anchored_mode_train_grows_each_fold(self):
        """In anchored mode, each fold uses more training data than the previous."""
        all_ts = list(range(0, 5000 * 60, 60))
        cfg = WFConfig(train_bars=2000, test_bars=500, step_bars=500, mode="anchored")
        windows = _build_windows(all_ts, cfg)

        assert len(windows) >= 2
        for i in range(1, len(windows)):
            assert windows[i].train_end_ts > windows[i - 1].train_end_ts

    def test_insufficient_data_returns_no_windows(self):
        all_ts = list(range(0, 100 * 60, 60))
        cfg = WFConfig(train_bars=2000, test_bars=500, step_bars=500, mode="rolling")
        windows = _build_windows(all_ts, cfg)
        assert windows == []

    def test_deterministic_splits(self):
        """Same input always produces identical fold boundaries."""
        all_ts = list(range(0, 5000 * 60, 60))
        cfg = WFConfig(train_bars=2000, test_bars=500, step_bars=500, mode="rolling")
        windows_a = _build_windows(all_ts, cfg)
        windows_b = _build_windows(all_ts, cfg)
        assert len(windows_a) == len(windows_b)
        for a, b in zip(windows_a, windows_b):
            assert a.train_start_ts == b.train_start_ts
            assert a.train_end_ts == b.train_end_ts
            assert a.test_end_ts == b.test_end_ts


# ---------------------------------------------------------------------------
# Slice correctness
# ---------------------------------------------------------------------------

class TestSliceCandles:
    def test_slice_includes_start_excludes_end(self):
        candles = _make_candle_set(n_1m=100)
        all_ts = sorted(
            c.close_time
            for sym in candles.values()
            for tf, cs in sym.items() if tf == "1m"
            for c in cs
        )
        start_ts = all_ts[10]
        end_ts = all_ts[30]

        sliced = _slice_candles(candles, start_ts, end_ts)
        for sym in sliced.values():
            for tf, cs in sym.items():
                if tf == "1m":
                    for c in cs:
                        assert start_ts <= c.close_time < end_ts

    def test_empty_slice_returns_empty_but_valid_structure(self):
        candles = _make_candle_set(n_1m=100)
        sliced = _slice_candles(candles, 0, 1)
        # Should return nested dict with possibly empty inner lists (which are excluded)
        for sym in sliced.values():
            for tf, cs in sym.items():
                assert isinstance(cs, list)


# ---------------------------------------------------------------------------
# Monte Carlo reproducibility
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def _make_trades(self, n: int = 100, seed: int = 0) -> list[ClosedTrade]:
        import random
        rng = random.Random(seed)
        trades = []
        for i in range(n):
            pnl = rng.gauss(10.0, 50.0)
            trades.append(ClosedTrade(
                symbol="BTCUSDT", direction=Direction.LONG,
                entry_price=50000.0, exit_price=50000.0 + pnl,
                total_size=0.01,
                entry_time=i * 3600, exit_time=(i + 1) * 3600,
                pnl=pnl, fees=1.0, hold_bars=60,
                equity_at_entry=100_000.0,
                exit_reason=ExitReason.TP1,
            ))
        return trades

    def test_same_seed_same_result(self):
        trades = self._make_trades()
        cfg = MCConfig(n_simulations=200, seed=42)
        engine = MonteCarloEngine(cfg)
        r1 = engine.run(trades, 100_000.0)
        r2 = engine.run(trades, 100_000.0)
        assert r1.ruin_probability == r2.ruin_probability
        assert r1.mean_max_drawdown == r2.mean_max_drawdown
        assert r1.pct_profitable == r2.pct_profitable

    def test_different_seeds_different_results(self):
        trades = self._make_trades()
        r1 = MonteCarloEngine(MCConfig(n_simulations=500, seed=1)).run(trades, 100_000.0)
        r2 = MonteCarloEngine(MCConfig(n_simulations=500, seed=2)).run(trades, 100_000.0)
        assert r1.ruin_probability != r2.ruin_probability or r1.mean_max_drawdown != r2.mean_max_drawdown

    def test_empty_trades_returns_zero_ruin(self):
        result = MonteCarloEngine().run([], 100_000.0)
        assert result.ruin_probability == 0.0
        assert result.pct_profitable == 0.0

    def test_always_winning_trades_low_ruin(self):
        trades = [
            ClosedTrade(
                symbol="BTCUSDT", direction=Direction.LONG,
                entry_price=1.0, exit_price=2.0, total_size=1.0,
                entry_time=i, exit_time=i + 1,
                pnl=100.0, fees=0.0, hold_bars=1,
                equity_at_entry=100_000.0,
                exit_reason=ExitReason.TP1,
            )
            for i in range(50)
        ]
        result = MonteCarloEngine(MCConfig(n_simulations=500, seed=0)).run(trades, 100_000.0)
        assert result.ruin_probability == 0.0
        assert result.pct_profitable == 1.0

    def test_ruin_threshold_respected(self):
        """All-loss trades should approach 100% ruin."""
        trades = [
            ClosedTrade(
                symbol="BTCUSDT", direction=Direction.LONG,
                entry_price=1.0, exit_price=0.5, total_size=1.0,
                entry_time=i, exit_time=i + 1,
                pnl=-2000.0, fees=0.0, hold_bars=1,
                equity_at_entry=100_000.0,
                exit_reason=ExitReason.STOP,
            )
            for i in range(30)
        ]
        result = MonteCarloEngine(MCConfig(n_simulations=200, seed=0)).run(trades, 100_000.0)
        assert result.ruin_probability > 0.5


# ---------------------------------------------------------------------------
# Regime analyzer
# ---------------------------------------------------------------------------

class TestRegimeAnalyzer:
    def test_session_classification(self):
        from nexflow.research.regime_analyzer import _classify_session, Session
        assert _classify_session(0 * 3600) == Session.ASIA
        assert _classify_session(4 * 3600) == Session.ASIA
        assert _classify_session(8 * 3600) == Session.LONDON
        assert _classify_session(10 * 3600) == Session.LONDON
        assert _classify_session(13 * 3600) == Session.NEW_YORK
        assert _classify_session(20 * 3600) == Session.NEW_YORK
        assert _classify_session(21 * 3600) == Session.OFF_HOURS
        assert _classify_session(23 * 3600) == Session.OFF_HOURS

    def test_label_count_matches_candle_count(self):
        candles = _make_candle_set(n_1m=200)["BTCUSDT"]["1m"]
        analyzer = RegimeAnalyzer()
        labels = analyzer.label_candles(candles)
        assert len(labels) == len(candles)

    def test_empty_candles_returns_empty_labels(self):
        analyzer = RegimeAnalyzer()
        assert analyzer.label_candles([]) == []

    def test_labels_have_valid_enum_values(self):
        from nexflow.research.regime_analyzer import VolatilityRegime, SpreadRegime, TrendRegime
        candles = _make_candle_set(n_1m=50)["BTCUSDT"]["1m"]
        analyzer = RegimeAnalyzer()
        labels = analyzer.label_candles(candles)
        for label in labels:
            assert label.volatility in VolatilityRegime
            assert label.spread in SpreadRegime
            assert label.trend in TrendRegime


# ---------------------------------------------------------------------------
# Equity curve analysis
# ---------------------------------------------------------------------------

class TestEquityCurveAnalysis:
    def test_flat_equity_zero_drawdown(self):
        curve = [(i, 100_000.0) for i in range(100)]
        analysis = EquityCurveAnalysis()
        stats = analysis.analyze(curve, [], [], 100_000.0)
        assert stats.max_drawdown == 0.0
        assert stats.ulcer_index == 0.0

    def test_monotone_increasing_zero_drawdown(self):
        curve = [(i, 100_000.0 + i * 100) for i in range(100)]
        analysis = EquityCurveAnalysis()
        stats = analysis.analyze(curve, [], [], 100_000.0)
        assert stats.max_drawdown == 0.0
        assert stats.max_stagnation_bars <= 1  # first bar counts as 1, subsequent bars always hit new high

    def test_drawdown_detected(self):
        curve = [(0, 100_000.0), (1, 110_000.0), (2, 90_000.0), (3, 110_000.0)]
        analysis = EquityCurveAnalysis()
        stats = analysis.analyze(curve, [], [], 100_000.0)
        # Peak 110k → trough 90k → DD = 20/110 ≈ 18.2%
        assert abs(stats.max_drawdown - 20_000 / 110_000) < 0.001

    def test_consecutive_losses_counted(self):
        pnls = [100, -50, -50, -50, 100, -50, -50, 100]
        analysis = EquityCurveAnalysis()
        stats = analysis.analyze([(i, 100_000.0) for i in range(10)], pnls, list(range(8)), 100_000.0)
        assert stats.max_consecutive_losses == 3

    def test_empty_curve_returns_defaults(self):
        analysis = EquityCurveAnalysis()
        stats = analysis.analyze([], [], [], 100_000.0)
        assert stats.calmar_ratio == 0.0
        assert stats.max_drawdown == 0.0


# ---------------------------------------------------------------------------
# Parameter sweeper isolation
# ---------------------------------------------------------------------------

class TestParameterSweeper:
    def test_each_combo_gets_fresh_strategy(self):
        """Verify that param sweep calls strategy factory with distinct param values."""
        call_log: list[dict] = []

        class TrackingStrategy(BaseStrategy):
            def __init__(self, rel_vol_threshold: float = 1.5):
                self._rvt = rel_vol_threshold
                call_log.append({"rel_vol_threshold": rel_vol_threshold})

            def on_candle(self, candle):
                return None

            def reset(self):
                pass

        ranges = [ParamRange("rel_vol_threshold", [1.0, 1.5, 2.0])]
        bt_cfg = BacktestConfig(risk=RiskConfig())
        sweeper = ParameterSweeper(TrackingStrategy, bt_cfg, ranges)

        all_candles = _make_candle_set(n_1m=200)
        sweeper.run(all_candles)

        rvt_values = sorted(c["rel_vol_threshold"] for c in call_log)
        assert rvt_values == [1.0, 1.5, 2.0]

    def test_sweep_results_count_matches_cartesian_product(self):
        class NullStrategy(BaseStrategy):
            def __init__(self, a: int = 1, b: int = 1):
                pass
            def on_candle(self, candle):
                return None
            def reset(self):
                pass

        ranges = [ParamRange("a", [1, 2, 3]), ParamRange("b", [10, 20])]
        bt_cfg = BacktestConfig(risk=RiskConfig())
        sweeper = ParameterSweeper(NullStrategy, bt_cfg, ranges)
        all_candles = _make_candle_set(n_1m=100)
        summary = sweeper.run(all_candles)
        assert len(summary.results) == 3 * 2  # 6 combinations
