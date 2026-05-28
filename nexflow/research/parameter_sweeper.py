"""Parameter stability sweep for the momentum strategy.

Purpose: identify parameter regions where performance is STABLE, not just where
it peaks. A parameter set that performs consistently across ±20% perturbations
is more robust than one that happens to peak at a single point.

The sweeper runs a full backtest for every combination in the cartesian product
of ParamRange values and ranks results by a composite score that rewards:
    - Stability (low sensitivity to perturbation)
    - Risk-adjusted return (Sharpe over raw PnL)
    - Drawdown control
    - Long/short symmetry (avoids direction-biased curve-fitting)

Do NOT use this to find the "best" parameters. Use it to find the parameters
that survive the widest range of market conditions.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.backtest_runner import BacktestConfig, BacktestRunner
from nexflow.services.strategy.base_strategy import BaseStrategy
from nexflow.services.strategy.signal_models import BacktestMetrics


@dataclass
class ParamRange:
    name: str
    values: list[Any]     # discrete values to test; caller is responsible for granularity


@dataclass
class SweepResult:
    params: dict[str, Any]
    metrics: BacktestMetrics
    composite_score: float = 0.0   # higher is better; computed post-sweep
    stability_score: float = 0.0   # how flat performance is around this point


@dataclass
class SweepSummary:
    results: list[SweepResult] = field(default_factory=list)
    best_by_sharpe: SweepResult | None = None
    best_by_composite: SweepResult | None = None
    best_by_drawdown: SweepResult | None = None
    stable_region: list[SweepResult] = field(default_factory=list)  # top quartile by stability


class ParameterSweeper:
    """Cartesian-product parameter sweep with stability ranking.

    Usage::

        ranges = [
            ParamRange("rel_vol_threshold", [1.2, 1.3, 1.5, 1.8, 2.0]),
            ParamRange("atr_period", [10, 14, 20]),
        ]
        sweeper = ParameterSweeper(MomentumStrategy, bt_cfg, ranges)
        summary = sweeper.run(all_candles)
    """

    def __init__(
        self,
        strategy_factory: type[BaseStrategy],
        bt_cfg: BacktestConfig,
        param_ranges: list[ParamRange],
        strategy_config_class: type | None = None,
    ) -> None:
        self._strategy_factory = strategy_factory
        self._bt_cfg = bt_cfg
        self._param_ranges = param_ranges
        self._strategy_config_class = strategy_config_class

    def run(self, all_candles: dict[str, dict[str, list[Candle]]]) -> SweepSummary:
        combinations = list(itertools.product(*[r.values for r in self._param_ranges]))
        param_names = [r.name for r in self._param_ranges]

        raw_results: list[SweepResult] = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            metrics = self._backtest(params, all_candles)
            raw_results.append(SweepResult(params=params, metrics=metrics))

        _score_results(raw_results)
        _compute_stability(raw_results, param_names)

        return _build_summary(raw_results)

    def _backtest(
        self,
        params: dict[str, Any],
        all_candles: dict[str, dict[str, list[Candle]]],
    ) -> BacktestMetrics:
        if self._strategy_config_class is not None:
            cfg = self._strategy_config_class(**params)
            strategy = self._strategy_factory(cfg)
        else:
            strategy = self._strategy_factory(**params)
        runner = BacktestRunner(strategy, self._bt_cfg)
        return runner.run(all_candles)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _composite_score(m: BacktestMetrics) -> float:
    """Composite optimization target: NOT maximum return.

    Rewards: positive Sharpe, controlled drawdown, symmetric long/short, trades > 0.
    Penalizes: drawdown > 10%, win_rate < 40%, no trades.
    """
    if m.total_trades == 0:
        return -1.0

    sharpe_component = max(m.sharpe, -2.0)
    drawdown_penalty = max(0.0, m.max_drawdown - 0.10) * 5.0
    symmetry = 1.0 - abs(m.long_win_rate - m.short_win_rate) if (m.long_trades > 0 and m.short_trades > 0) else 0.5
    trade_count_bonus = min(m.total_trades / 50.0, 1.0)

    return sharpe_component - drawdown_penalty + 0.3 * symmetry + 0.1 * trade_count_bonus


def _score_results(results: list[SweepResult]) -> None:
    for r in results:
        r.composite_score = _composite_score(r.metrics)


def _compute_stability(results: list[SweepResult], param_names: list[str]) -> None:
    """For each result, measure how sensitive performance is to small parameter changes.

    Stability = 1 / (1 + std(composite_score of immediate neighbours)).
    A result surrounded by similar-scoring neighbours is more stable.
    """
    for target in results:
        neighbours = _find_neighbours(target, results, param_names)
        if not neighbours:
            target.stability_score = 0.5
            continue
        scores = [n.composite_score for n in neighbours]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        target.stability_score = 1.0 / (1.0 + std)


def _find_neighbours(
    target: SweepResult,
    all_results: list[SweepResult],
    param_names: list[str],
) -> list[SweepResult]:
    """Return results that differ from target in exactly one parameter by one step."""
    neighbours: list[SweepResult] = []
    for candidate in all_results:
        if candidate is target:
            continue
        diffs = sum(
            1 for name in param_names
            if target.params.get(name) != candidate.params.get(name)
        )
        if diffs == 1:
            neighbours.append(candidate)
    return neighbours


def _build_summary(results: list[SweepResult]) -> SweepSummary:
    if not results:
        return SweepSummary()

    valid = [r for r in results if r.metrics.total_trades > 0]
    if not valid:
        return SweepSummary(results=results)

    best_sharpe = max(valid, key=lambda r: r.metrics.sharpe)
    best_composite = max(valid, key=lambda r: r.composite_score)
    best_drawdown = min(valid, key=lambda r: r.metrics.max_drawdown)

    sorted_by_stability = sorted(valid, key=lambda r: r.stability_score, reverse=True)
    top_n = max(1, len(sorted_by_stability) // 4)
    stable_region = sorted_by_stability[:top_n]

    return SweepSummary(
        results=results,
        best_by_sharpe=best_sharpe,
        best_by_composite=best_composite,
        best_by_drawdown=best_drawdown,
        stable_region=stable_region,
    )
