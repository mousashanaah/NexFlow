"""Market regime classification for walk-forward regime-conditioned analysis.

Regimes are computed from the candle data itself — no external data required.
They are used ONLY to segment backtest results for analysis, never to make
trading decisions or to filter training data (that would be look-ahead).

Classification dimensions:
    Volatility  : LOW / MEDIUM / HIGH  (based on rolling ATR percentile)
    Spread      : TIGHT / NORMAL / WIDE  (based on spread_avg relative to ATR)
    Session     : ASIA / LONDON / NEW_YORK / OFF_HOURS  (based on UTC hour)
    Trend       : TRENDING / CHOPPY  (based on directional efficiency ratio)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from nexflow.services.candles.candle_engine import Candle


class VolatilityRegime(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SpreadRegime(str, Enum):
    TIGHT = "tight"
    NORMAL = "normal"
    WIDE = "wide"


class Session(str, Enum):
    ASIA = "asia"             # 00:00–08:00 UTC
    LONDON = "london"         # 08:00–13:00 UTC
    NEW_YORK = "new_york"     # 13:00–21:00 UTC
    OFF_HOURS = "off_hours"   # 21:00–00:00 UTC


class TrendRegime(str, Enum):
    TRENDING = "trending"
    CHOPPY = "choppy"


@dataclass(slots=True)
class RegimeLabel:
    close_time: int
    volatility: VolatilityRegime
    spread: SpreadRegime
    session: Session
    trend: TrendRegime


@dataclass
class RegimeStats:
    """Performance statistics for a single regime slice."""
    regime_key: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0


class RegimeAnalyzer:
    """Classify 1m candles into regime labels and segment trade results by regime.

    Usage::

        analyzer = RegimeAnalyzer()
        labels = analyzer.label_candles(candles_1m)
        # Then join labels with ClosedTrade entries by close_time to segment performance.
    """

    def __init__(
        self,
        atr_period: int = 14,
        vol_low_pct: float = 0.33,   # ATR percentile below which = LOW
        vol_high_pct: float = 0.67,
        spread_tight_pct: float = 0.33,
        spread_wide_pct: float = 0.67,
        efficiency_period: int = 10,
        trending_threshold: float = 0.35,
    ) -> None:
        self._atr_period = atr_period
        self._vol_low_pct = vol_low_pct
        self._vol_high_pct = vol_high_pct
        self._spread_tight_pct = spread_tight_pct
        self._spread_wide_pct = spread_wide_pct
        self._efficiency_period = efficiency_period
        self._trending_threshold = trending_threshold

    def label_candles(self, candles: list[Candle]) -> list[RegimeLabel]:
        """Assign a RegimeLabel to each candle. Candles must be sorted by close_time."""
        if not candles:
            return []

        atrs = _compute_atr_series(candles, self._atr_period)
        spreads = [c.spread_avg for c in candles]
        efficiencies = _compute_efficiency_series(candles, self._efficiency_period)

        atr_low_cut, atr_high_cut = _percentile_cuts(atrs, self._vol_low_pct, self._vol_high_pct)
        spread_low_cut, spread_high_cut = _percentile_cuts(spreads, self._spread_tight_pct, self._spread_wide_pct)

        labels: list[RegimeLabel] = []
        for i, candle in enumerate(candles):
            atr = atrs[i]
            spread = spreads[i]
            er = efficiencies[i]

            vol_regime = _classify_volatility(atr, atr_low_cut, atr_high_cut)
            spread_regime = _classify_spread(spread, spread_low_cut, spread_high_cut)
            session = _classify_session(candle.close_time)
            trend_regime = TrendRegime.TRENDING if er >= self._trending_threshold else TrendRegime.CHOPPY

            labels.append(RegimeLabel(
                close_time=candle.close_time,
                volatility=vol_regime,
                spread=spread_regime,
                session=session,
                trend=trend_regime,
            ))

        return labels

    def regime_key(self, label: RegimeLabel) -> str:
        return f"{label.volatility.value}_{label.trend.value}_{label.session.value}"


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_atr_series(candles: list[Candle], period: int) -> list[float]:
    """True range for each bar, then simple moving average over period."""
    n = len(candles)
    trs: list[float] = []
    for i, c in enumerate(candles):
        if i == 0:
            trs.append(c.high - c.low)
        else:
            prev_close = candles[i - 1].close
            tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
            trs.append(tr)

    atrs: list[float] = []
    for i in range(n):
        start = max(0, i - period + 1)
        window = trs[start : i + 1]
        atrs.append(sum(window) / len(window))
    return atrs


def _compute_efficiency_series(candles: list[Candle], period: int) -> list[float]:
    """Kaufman efficiency ratio: net displacement / sum of bar-to-bar distances.

    Values near 1.0 = strongly trending. Values near 0 = choppy/ranging.
    """
    n = len(candles)
    ers: list[float] = []
    for i in range(n):
        start = max(0, i - period + 1)
        window = candles[start : i + 1]
        if len(window) < 2:
            ers.append(0.0)
            continue
        net = abs(window[-1].close - window[0].close)
        path = sum(abs(window[j].close - window[j - 1].close) for j in range(1, len(window)))
        ers.append(net / path if path > 0 else 0.0)
    return ers


def _percentile_cuts(values: list[float], low_pct: float, high_pct: float) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    low_cut = sorted_vals[max(0, int(n * low_pct) - 1)]
    high_cut = sorted_vals[min(n - 1, int(n * high_pct))]
    return low_cut, high_cut


def _classify_volatility(atr: float, low_cut: float, high_cut: float) -> VolatilityRegime:
    if atr <= low_cut:
        return VolatilityRegime.LOW
    if atr >= high_cut:
        return VolatilityRegime.HIGH
    return VolatilityRegime.MEDIUM


def _classify_spread(spread: float, low_cut: float, high_cut: float) -> SpreadRegime:
    if spread <= low_cut:
        return SpreadRegime.TIGHT
    if spread >= high_cut:
        return SpreadRegime.WIDE
    return SpreadRegime.NORMAL


def _classify_session(close_time_s: int) -> Session:
    hour_utc = (close_time_s % 86_400) // 3_600
    if 0 <= hour_utc < 8:
        return Session.ASIA
    if 8 <= hour_utc < 13:
        return Session.LONDON
    if 13 <= hour_utc < 21:
        return Session.NEW_YORK
    return Session.OFF_HOURS
