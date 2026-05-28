"""Monte Carlo simulation for strategy robustness assessment.

Approach: resample the observed trade PnL sequence (bootstrap with replacement)
to generate synthetic equity curves. This answers the question: "given these
trade outcomes, how bad could drawdown get under different orderings?"

This is NOT return prediction. The simulation uses ONLY the observed trade
distribution — no distributional assumptions, no curve fitting.

Key outputs:
    - Ruin probability (equity falls below ruin_threshold × initial)
    - Drawdown confidence intervals (p5, p50, p95)
    - Equity distribution at horizon end
    - Number of simulations that beat zero PnL

Reproducibility: uses Python stdlib random.Random(seed) so results are
deterministic and environment-independent (no numpy dependency for MC).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from nexflow.services.strategy.signal_models import ClosedTrade


@dataclass
class MCConfig:
    n_simulations: int = 1_000
    seed: int = 42
    ruin_threshold: float = 0.50    # equity below 50% of initial = "ruin"
    confidence_levels: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])


@dataclass
class MCResult:
    n_simulations: int
    ruin_probability: float
    pct_profitable: float                      # fraction of paths with final equity > initial
    drawdown_percentiles: dict[float, float]   # confidence → max drawdown value
    final_equity_percentiles: dict[float, float]
    median_final_equity: float
    mean_max_drawdown: float
    worst_drawdown: float
    best_drawdown: float


class MonteCarloEngine:
    """Bootstrap Monte Carlo over a sequence of closed trades.

    Usage::

        engine = MonteCarloEngine(MCConfig(n_simulations=2000, seed=99))
        result = engine.run(trades, initial_equity=100_000)
    """

    def __init__(self, cfg: MCConfig | None = None) -> None:
        self._cfg = cfg or MCConfig()

    def run(self, trades: list[ClosedTrade], initial_equity: float) -> MCResult:
        """Bootstrap-resample trade sequence and compute path statistics."""
        if not trades:
            return _empty_result(self._cfg)

        pnls = [t.pnl for t in trades]
        n_trades = len(pnls)
        cfg = self._cfg

        rng = random.Random(cfg.seed)

        ruin_count = 0
        profitable_count = 0
        all_max_drawdowns: list[float] = []
        all_final_equities: list[float] = []

        for _ in range(cfg.n_simulations):
            sequence = rng.choices(pnls, k=n_trades)
            equity = initial_equity
            peak = equity
            max_dd = 0.0

            for pnl in sequence:
                equity += pnl
                if equity > peak:
                    peak = equity
                if peak > 0:
                    dd = (peak - equity) / peak
                    if dd > max_dd:
                        max_dd = dd
                if equity <= initial_equity * cfg.ruin_threshold:
                    ruin_count += 1
                    break

            if equity > initial_equity:
                profitable_count += 1

            all_max_drawdowns.append(max_dd)
            all_final_equities.append(equity)

        n = cfg.n_simulations
        dd_pcts = _percentiles(all_max_drawdowns, cfg.confidence_levels)
        eq_pcts = _percentiles(all_final_equities, cfg.confidence_levels)

        return MCResult(
            n_simulations=n,
            ruin_probability=ruin_count / n,
            pct_profitable=profitable_count / n,
            drawdown_percentiles=dd_pcts,
            final_equity_percentiles=eq_pcts,
            median_final_equity=eq_pcts.get(0.50, initial_equity),
            mean_max_drawdown=sum(all_max_drawdowns) / n,
            worst_drawdown=max(all_max_drawdowns) if all_max_drawdowns else 0.0,
            best_drawdown=min(all_max_drawdowns) if all_max_drawdowns else 0.0,
        )


def _percentiles(values: list[float], levels: list[float]) -> dict[float, float]:
    if not values:
        return {lvl: 0.0 for lvl in levels}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result: dict[float, float] = {}
    for lvl in levels:
        idx = min(int(lvl * n), n - 1)
        result[lvl] = sorted_vals[idx]
    return result


def _empty_result(cfg: MCConfig) -> MCResult:
    empty_pcts = {lvl: 0.0 for lvl in cfg.confidence_levels}
    return MCResult(
        n_simulations=cfg.n_simulations,
        ruin_probability=0.0,
        pct_profitable=0.0,
        drawdown_percentiles=empty_pcts,
        final_equity_percentiles=empty_pcts,
        median_final_equity=0.0,
        mean_max_drawdown=0.0,
        worst_drawdown=0.0,
        best_drawdown=0.0,
    )
