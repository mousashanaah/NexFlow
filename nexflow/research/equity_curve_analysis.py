"""Equity curve quality metrics for strategy health assessment.

These metrics are computed on the equity curve produced by a backtest.
The goal is NOT to find the highest return — it is to assess the quality,
smoothness, and consistency of the equity growth process.

Metrics:
    Calmar ratio        : annualised return / max drawdown
    Ulcer index         : RMS of drawdown time series (penalises long drawdowns)
    Rolling Sharpe      : Sharpe computed on rolling N-bar windows (consistency)
    Stagnation periods  : longest stretch without a new equity high (in bars)
    Consecutive losses  : max streak of negative-PnL closed trades
    Monthly consistency : fraction of calendar months with positive PnL
    Risk of ruin proxy  : fraction of rolling windows where drawdown > threshold
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CurveStats:
    calmar_ratio: float = 0.0
    ulcer_index: float = 0.0
    max_drawdown: float = 0.0
    max_stagnation_bars: int = 0
    max_consecutive_losses: int = 0
    monthly_consistency: float = 0.0    # fraction of months with positive PnL
    rolling_sharpe_mean: float = 0.0    # mean of rolling-window Sharpes
    rolling_sharpe_std: float = 0.0     # std of rolling Sharpes (lower = more consistent)
    ruin_proxy: float = 0.0             # fraction of rolling windows where DD > 10%
    trade_count: int = 0
    net_pnl: float = 0.0
    total_bars: int = 0


class EquityCurveAnalysis:
    """Compute quality metrics from an equity curve and trade list.

    equity_curve: list of (timestamp_seconds, equity_value) from Portfolio.
    trade_pnls:   list of per-trade PnL values (signed, in order of close time).
    trade_times:  list of close_time_seconds per trade (same length as trade_pnls).
    """

    def __init__(
        self,
        rolling_sharpe_window: int = 20,
        ruin_dd_threshold: float = 0.10,
    ) -> None:
        self._rolling_window = rolling_sharpe_window
        self._ruin_threshold = ruin_dd_threshold

    def analyze(
        self,
        equity_curve: list[tuple[int, float]],
        trade_pnls: list[float],
        trade_times: list[int],
        initial_equity: float,
    ) -> CurveStats:
        if not equity_curve:
            return CurveStats()

        equities = [e for _, e in equity_curve]
        timestamps = [t for t, _ in equity_curve]

        max_dd = _max_drawdown(equities)
        ulcer = _ulcer_index(equities)
        calmar = _calmar_ratio(equities, timestamps, max_dd)
        stagnation = _max_stagnation(equities)
        consec_losses = _max_consecutive_losses(trade_pnls)
        monthly = _monthly_consistency(trade_pnls, trade_times)
        roll_mean, roll_std = _rolling_sharpe_stats(trade_pnls, self._rolling_window)
        ruin_proxy = _ruin_proxy(equities, self._ruin_threshold)

        return CurveStats(
            calmar_ratio=calmar,
            ulcer_index=ulcer,
            max_drawdown=max_dd,
            max_stagnation_bars=stagnation,
            max_consecutive_losses=consec_losses,
            monthly_consistency=monthly,
            rolling_sharpe_mean=roll_mean,
            rolling_sharpe_std=roll_std,
            ruin_proxy=ruin_proxy,
            trade_count=len(trade_pnls),
            net_pnl=equities[-1] - initial_equity if equities else 0.0,
            total_bars=len(equities),
        )


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def _max_drawdown(equities: list[float]) -> float:
    if not equities:
        return 0.0
    peak = equities[0]
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _ulcer_index(equities: list[float]) -> float:
    """RMS of percentage drawdown at each bar. Penalises long, deep drawdowns."""
    if not equities:
        return 0.0
    peak = equities[0]
    sq_sum = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        sq_sum += dd * dd
    return math.sqrt(sq_sum / len(equities))


def _calmar_ratio(equities: list[float], timestamps: list[int], max_dd: float) -> float:
    """Annualised return divided by max drawdown."""
    if not equities or len(equities) < 2 or max_dd <= 0:
        return 0.0
    hold_years = (timestamps[-1] - timestamps[0]) / (365.25 * 86_400)
    if hold_years < 1 / 365.0 or equities[0] <= 0:
        return 0.0
    total_return = (equities[-1] - equities[0]) / equities[0]
    try:
        annual_return = (1.0 + total_return) ** (1.0 / hold_years) - 1.0
    except (OverflowError, ValueError):
        return 0.0
    return annual_return / max_dd


def _max_stagnation(equities: list[float]) -> int:
    """Longest bar streak without setting a new equity high."""
    if not equities:
        return 0
    peak = equities[0]
    current_streak = 0
    max_streak = 0
    for eq in equities:
        if eq > peak:
            peak = eq
            current_streak = 0
        else:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
    return max_streak


def _max_consecutive_losses(pnls: list[float]) -> int:
    max_streak = 0
    streak = 0
    for pnl in pnls:
        if pnl < 0:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
    return max_streak


def _monthly_consistency(pnls: list[float], times: list[int]) -> float:
    """Fraction of calendar months with net positive PnL."""
    if not pnls or not times:
        return 0.0

    monthly: dict[tuple[int, int], float] = {}
    for pnl, ts in zip(pnls, times):
        # Derive year and month from epoch seconds
        import time as _time
        t = _time.gmtime(ts)
        key = (t.tm_year, t.tm_mon)
        monthly[key] = monthly.get(key, 0.0) + pnl

    if not monthly:
        return 0.0
    positive_months = sum(1 for v in monthly.values() if v > 0)
    return positive_months / len(monthly)


def _rolling_sharpe_stats(pnls: list[float], window: int) -> tuple[float, float]:
    """Compute rolling Sharpe ratios over windows of `window` trades."""
    if len(pnls) < window:
        return 0.0, 0.0

    sharpes: list[float] = []
    for i in range(window, len(pnls) + 1):
        chunk = pnls[i - window : i]
        mean = sum(chunk) / window
        var = sum((r - mean) ** 2 for r in chunk) / (window - 1)
        std = math.sqrt(var)
        if std < 1e-10:
            continue
        sharpes.append(mean / std)

    if not sharpes:
        return 0.0, 0.0

    mean_s = sum(sharpes) / len(sharpes)
    var_s = sum((s - mean_s) ** 2 for s in sharpes) / max(len(sharpes) - 1, 1)
    return mean_s, math.sqrt(var_s)


def _ruin_proxy(equities: list[float], threshold: float) -> float:
    """Fraction of all rolling 50-bar windows where max drawdown exceeded threshold."""
    n = len(equities)
    window = min(50, n)
    if n < window:
        return 0.0

    count = 0
    total = 0
    for i in range(window, n + 1):
        chunk = equities[i - window : i]
        dd = _max_drawdown(chunk)
        if dd > threshold:
            count += 1
        total += 1

    return count / total if total > 0 else 0.0
