"""Walk-forward validation engine.

Splits historical candle data into successive train/test windows and runs
a backtest on each. The goal is NOT to maximize returns — it is to verify
that the strategy degrades gracefully when tested on unseen data, and that
performance is consistent across time periods rather than concentrated in a
lucky sub-sample.

Two modes:
    rolling   — train window slides forward by step_bars on each fold
    anchored  — train window always starts at the beginning; only the end expands

Interval convention: half-open [start, end).
    Training  uses candles where start <= close_time < train_end
    Testing   uses candles where train_end <= close_time < test_end
    train_end == test_start  (no gap, no overlap)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.backtest_runner import BacktestConfig, BacktestRunner
from nexflow.services.strategy.base_strategy import BaseStrategy
from nexflow.services.strategy.signal_models import BacktestMetrics


@dataclass
class WFConfig:
    train_bars: int = 2_000       # 1m bars in the training window
    test_bars: int = 500          # 1m bars in each out-of-sample window
    step_bars: int = 500          # how far the window advances each fold (rolling only)
    mode: Literal["rolling", "anchored"] = "rolling"
    min_train_trades: int = 10    # skip fold if too few trades in training phase
    min_test_trades: int = 5      # mark fold as inconclusive if too few trades OOS


@dataclass
class WFWindow:
    fold_idx: int
    train_start_ts: int    # inclusive
    train_end_ts: int      # exclusive
    test_start_ts: int     # inclusive (== train_end_ts)
    test_end_ts: int       # exclusive


@dataclass
class WFFoldResult:
    window: WFWindow
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    is_conclusive: bool    # False when too few OOS trades to draw conclusions


@dataclass
class WalkForwardResult:
    folds: list[WFFoldResult] = field(default_factory=list)

    # Aggregate summaries (computed after all folds)
    mean_oos_win_rate: float = 0.0
    mean_oos_pnl: float = 0.0
    mean_oos_drawdown: float = 0.0
    mean_oos_sharpe: float = 0.0
    degradation_ratio: float = 0.0   # mean(OOS metric) / mean(IS metric); 1.0 = no degradation
    consistency_score: float = 0.0   # fraction of conclusive folds with positive OOS PnL
    total_oos_trades: int = 0


class WalkForwardEngine:
    """Run rolling or anchored walk-forward validation.

    Usage::

        engine = WalkForwardEngine(strategy_factory, bt_cfg, wf_cfg)
        result = engine.run(all_candles)
    """

    def __init__(
        self,
        strategy_factory: type[BaseStrategy],
        bt_cfg: BacktestConfig,
        wf_cfg: WFConfig | None = None,
        strategy_kwargs: dict | None = None,
    ) -> None:
        self._strategy_factory = strategy_factory
        self._bt_cfg = bt_cfg
        self._wf_cfg = wf_cfg or WFConfig()
        self._strategy_kwargs = strategy_kwargs or {}

    def run(self, all_candles: dict[str, dict[str, list[Candle]]]) -> WalkForwardResult:
        """Run walk-forward across all folds and return aggregated results."""
        # Collect all 1m timestamps in sorted order to define fold boundaries
        all_1m_ts = sorted(
            c.close_time
            for sym in all_candles.values()
            for tf, candles in sym.items()
            if tf == "1m"
            for c in candles
        )

        if not all_1m_ts:
            return WalkForwardResult()

        cfg = self._wf_cfg
        windows = _build_windows(all_1m_ts, cfg)
        if not windows:
            return WalkForwardResult()

        folds: list[WFFoldResult] = []
        for window in windows:
            train_candles = _slice_candles(all_candles, window.train_start_ts, window.train_end_ts)
            test_candles = _slice_candles(all_candles, window.test_start_ts, window.test_end_ts)

            train_metrics = self._backtest(train_candles)
            if train_metrics.total_trades < cfg.min_train_trades:
                continue

            test_metrics = self._backtest(test_candles)
            is_conclusive = test_metrics.total_trades >= cfg.min_test_trades

            folds.append(WFFoldResult(
                window=window,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                is_conclusive=is_conclusive,
            ))

        return _aggregate(folds)

    def _backtest(self, candles: dict[str, dict[str, list[Candle]]]) -> BacktestMetrics:
        strategy = self._strategy_factory(**self._strategy_kwargs)
        runner = BacktestRunner(strategy, self._bt_cfg)
        return runner.run(candles)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_windows(sorted_1m_ts: list[int], cfg: WFConfig) -> list[WFWindow]:
    """Build fold windows using 1m bar timestamps as boundary markers."""
    n = len(sorted_1m_ts)
    required = cfg.train_bars + cfg.test_bars
    if n < required:
        return []

    windows: list[WFWindow] = []
    fold_idx = 0

    if cfg.mode == "rolling":
        train_start_idx = 0
        while True:
            train_end_idx = train_start_idx + cfg.train_bars
            test_end_idx = train_end_idx + cfg.test_bars
            if test_end_idx > n:
                break

            windows.append(WFWindow(
                fold_idx=fold_idx,
                train_start_ts=sorted_1m_ts[train_start_idx],
                train_end_ts=sorted_1m_ts[train_end_idx],
                test_start_ts=sorted_1m_ts[train_end_idx],
                test_end_ts=sorted_1m_ts[test_end_idx - 1] + 1,
            ))
            train_start_idx += cfg.step_bars
            fold_idx += 1

    else:  # anchored
        train_end_idx = cfg.train_bars
        while True:
            test_end_idx = train_end_idx + cfg.test_bars
            if test_end_idx > n:
                break

            windows.append(WFWindow(
                fold_idx=fold_idx,
                train_start_ts=sorted_1m_ts[0],
                train_end_ts=sorted_1m_ts[train_end_idx],
                test_start_ts=sorted_1m_ts[train_end_idx],
                test_end_ts=sorted_1m_ts[test_end_idx - 1] + 1,
            ))
            train_end_idx += cfg.step_bars
            fold_idx += 1

    return windows


def _slice_candles(
    all_candles: dict[str, dict[str, list[Candle]]],
    start_ts: int,
    end_ts: int,
) -> dict[str, dict[str, list[Candle]]]:
    """Return candles where start_ts <= close_time < end_ts."""
    result: dict[str, dict[str, list[Candle]]] = {}
    for symbol, tf_map in all_candles.items():
        result[symbol] = {}
        for tf, candles in tf_map.items():
            sliced = [c for c in candles if start_ts <= c.close_time < end_ts]
            if sliced:
                result[symbol][tf] = sliced
    return result


def _aggregate(folds: list[WFFoldResult]) -> WalkForwardResult:
    if not folds:
        return WalkForwardResult(folds=[])

    conclusive = [f for f in folds if f.is_conclusive]

    def _safe_mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    oos_pnls = [f.test_metrics.net_pnl for f in conclusive]
    oos_win_rates = [f.test_metrics.win_rate for f in conclusive]
    oos_drawdowns = [f.test_metrics.max_drawdown for f in conclusive]
    oos_sharpes = [f.test_metrics.sharpe for f in conclusive]

    is_win_rates = [f.train_metrics.win_rate for f in conclusive]
    mean_is_wr = _safe_mean(is_win_rates)
    mean_oos_wr = _safe_mean(oos_win_rates)
    degradation_ratio = (mean_oos_wr / mean_is_wr) if mean_is_wr > 0 else 0.0

    consistency_score = (
        len([p for p in oos_pnls if p > 0]) / len(oos_pnls) if oos_pnls else 0.0
    )

    total_oos_trades = sum(f.test_metrics.total_trades for f in folds)

    return WalkForwardResult(
        folds=folds,
        mean_oos_win_rate=mean_oos_wr,
        mean_oos_pnl=_safe_mean(oos_pnls),
        mean_oos_drawdown=_safe_mean(oos_drawdowns),
        mean_oos_sharpe=_safe_mean(oos_sharpes),
        degradation_ratio=degradation_ratio,
        consistency_score=consistency_score,
        total_oos_trades=total_oos_trades,
    )
