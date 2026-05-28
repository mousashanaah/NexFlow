"""Volatility expansion + trend continuation strategy.

Core concept:
    Enter only when the market is expanding out of a compression range AND the
    higher-timeframe trend confirms direction. Avoid chop. Hold multi-minute trends.

Signal logic (LONG):
    1. 1m close breaks above rolling 20-bar high (breakout)
    2. Relative volume >= 1.5x 20-bar average (participation)
    3. Bar range >= 0.8x ATR (volatility expansion)
    4. 5m momentum slope > 0 (trend alignment)
    5. Spread/ATR ratio <= 0.30 (acceptable execution cost)
    6. Buy-volume fraction >= 0.55 (directional imbalance)

SHORT is the exact inverse of each condition.

All feature functions are pure module-level functions — they take a list of
Candle objects and return a scalar. The strategy class just manages the rolling
windows and wires the functions together.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.base_strategy import BaseStrategy
from nexflow.services.strategy.signal_models import Direction, Signal


# ---------------------------------------------------------------------------
# Strategy hyper-parameters (conservative, not curve-fitted)
# ---------------------------------------------------------------------------

@dataclass
class MomentumConfig:
    atr_period: int = 14
    vol_period: int = 20          # bars for rolling vol, rel-vol, breakout window
    rel_vol_threshold: float = 1.5
    range_expansion_min: float = 0.8   # bar range / ATR
    imbalance_min: float = 0.55        # buy_vol / total_vol for long
    spread_atr_max: float = 0.30       # spread_avg / ATR
    momentum_period_5m: int = 5        # bars for 5m slope
    atr_stop_mult: float = 1.5         # stop = entry ± mult × ATR
    atr_tp1_mult: float = 1.0          # TP1 = entry ± mult × ATR (50% size)
    atr_tp2_mult: float = 2.0          # TP2 (25%)
    atr_tp3_mult: float = 3.0          # TP3 (25%)
    min_bars_1m: int = 22              # minimum 1m window before signals fire
    min_bars_5m: int = 7               # minimum 5m window for trend context


# ---------------------------------------------------------------------------
# Pure feature functions
# ---------------------------------------------------------------------------

def compute_atr(candles: list[Candle], period: int = 14) -> float:
    """Simple ATR: mean of True Range over the last `period` bars.

    Uses the last `period` complete bars (each bar compared to its predecessor),
    so requires at least `period + 1` candles for a full window; falls back to
    available data gracefully.
    """
    n = len(candles)
    if n < 2:
        return candles[0].high - candles[0].low if n == 1 else 0.0

    start = max(1, n - period)
    trs: list[float] = []
    for i in range(start, n):
        c, p = candles[i], candles[i - 1]
        tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
        trs.append(tr)

    return sum(trs) / len(trs) if trs else 0.0


def compute_rolling_vol(candles: list[Candle], period: int = 20) -> float:
    """Annualised close-to-close log-return std over the last `period` bars.

    Returns 0.0 if fewer than 2 bars are available.
    """
    window = candles[-(period + 1):]
    if len(window) < 2:
        return 0.0

    returns: list[float] = []
    for i in range(1, len(window)):
        prev = window[i - 1].close
        curr = window[i].close
        if prev > 0 and curr > 0:
            returns.append(math.log(curr / prev))

    if len(returns) < 2:
        return 0.0

    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return math.sqrt(variance)


def compute_relative_volume(candles: list[Candle], period: int = 20) -> float:
    """Current bar volume / mean volume of the previous `period` bars.

    Returns 1.0 when there is insufficient history.
    """
    if len(candles) < 2:
        return 1.0
    prev = candles[-(period + 1):-1]
    if not prev:
        return 1.0
    avg = sum(c.volume for c in prev) / len(prev)
    return candles[-1].volume / avg if avg > 0 else 1.0


def compute_momentum_slope(candles: list[Candle], period: int = 5) -> float:
    """Percent change of close over the last `period` bars.

    Positive → upward momentum. Returns 0.0 on insufficient data.
    """
    if len(candles) < period + 1:
        return 0.0
    c_now = candles[-1].close
    c_past = candles[-(period + 1)].close
    return (c_now - c_past) / c_past if c_past > 0 else 0.0


def compute_range_expansion(candle: Candle, atr: float) -> float:
    """Current bar range as a multiple of ATR. > 1.0 means expanding volatility."""
    if atr <= 0:
        return 0.0
    return (candle.high - candle.low) / atr


def compute_spread_regime(candle: Candle, atr: float) -> float:
    """Spread-to-ATR ratio. Low values mean cheap to cross; high values = avoid."""
    if atr <= 0:
        return 999.0
    return candle.spread_avg / atr


def compute_buy_sell_imbalance(candle: Candle) -> float:
    """Buy volume fraction [0, 1]. > 0.5 means net buying pressure."""
    if candle.volume <= 0:
        return 0.5
    return candle.buy_volume / candle.volume


def compute_breakout_level(candles: list[Candle], period: int = 20) -> tuple[float, float]:
    """Return (rolling_high, rolling_low) of the last `period` bars excluding current."""
    lookback = candles[-period - 1:-1] if len(candles) > 1 else candles[:-1]
    if not lookback:
        return (candles[-1].high, candles[-1].low)
    return (max(c.high for c in lookback), min(c.low for c in lookback))


def compute_vwap_deviation(candle: Candle) -> float:
    """(close - vwap) / vwap: positive = close above intra-bar VWAP (bullish)."""
    if candle.vwap <= 0:
        return 0.0
    return (candle.close - candle.vwap) / candle.vwap


# ---------------------------------------------------------------------------
# Core signal computation (pure function — easy to test)
# ---------------------------------------------------------------------------

def compute_signal(
    candles_1m: list[Candle],
    candles_5m: list[Candle],
    cfg: MomentumConfig,
) -> Signal | None:
    """Compute a directional signal from finalized candle windows.

    Returns a Signal (LONG or SHORT) when all entry conditions are met,
    or None when the market is not in a favourable state.

    Both candle lists must end with the most recent finalized bar.
    The most recent 1m bar is the trigger bar; no future data is accessed.
    """
    if len(candles_1m) < cfg.min_bars_1m or len(candles_5m) < cfg.min_bars_5m:
        return None

    trigger = candles_1m[-1]
    symbol = trigger.symbol

    # --- compute features ---
    atr = compute_atr(candles_1m, cfg.atr_period)
    if atr <= 0:
        return None

    rel_vol = compute_relative_volume(candles_1m, cfg.vol_period)
    range_exp = compute_range_expansion(trigger, atr)
    spread_regime = compute_spread_regime(trigger, atr)
    imbalance = compute_buy_sell_imbalance(trigger)
    momentum_5m = compute_momentum_slope(candles_5m, cfg.momentum_period_5m)
    rolling_high, rolling_low = compute_breakout_level(candles_1m, cfg.vol_period)
    vwap_dev = compute_vwap_deviation(trigger)

    features: dict[str, float] = {
        "atr": round(atr, 6),
        "rel_vol": round(rel_vol, 4),
        "range_expansion": round(range_exp, 4),
        "spread_regime": round(spread_regime, 4),
        "buy_sell_imbalance": round(imbalance, 4),
        "momentum_5m": round(momentum_5m, 6),
        "rolling_high": round(rolling_high, 6),
        "rolling_low": round(rolling_low, 6),
        "vwap_deviation": round(vwap_dev, 6),
    }

    # --- universal filters (applied to both directions) ---
    if rel_vol < cfg.rel_vol_threshold:
        return None
    if range_exp < cfg.range_expansion_min:
        return None
    if spread_regime > cfg.spread_atr_max:
        return None

    # --- directional logic ---
    direction: Direction | None = None

    long_ok = (
        trigger.close > rolling_high          # breakout above 20-bar high
        and momentum_5m > 0                   # 5m trend aligned up
        and imbalance >= cfg.imbalance_min    # buying pressure dominates
    )
    short_ok = (
        trigger.close < rolling_low           # breakdown below 20-bar low
        and momentum_5m < 0                   # 5m trend aligned down
        and imbalance <= (1.0 - cfg.imbalance_min)  # selling pressure dominates
    )

    if long_ok:
        direction = Direction.LONG
    elif short_ok:
        direction = Direction.SHORT
    else:
        return None

    # --- build signal levels ---
    entry = trigger.close
    if direction is Direction.LONG:
        stop = entry - cfg.atr_stop_mult * atr
        tp_prices = [
            entry + cfg.atr_tp1_mult * atr,
            entry + cfg.atr_tp2_mult * atr,
            entry + cfg.atr_tp3_mult * atr,
        ]
    else:
        stop = entry + cfg.atr_stop_mult * atr
        tp_prices = [
            entry - cfg.atr_tp1_mult * atr,
            entry - cfg.atr_tp2_mult * atr,
            entry - cfg.atr_tp3_mult * atr,
        ]

    return Signal(
        symbol=symbol,
        direction=direction,
        timeframe="1m",
        bar_close_time=trigger.close_time,
        entry_price=entry,
        stop_price=stop,
        tp_prices=tp_prices,
        atr=atr,
        features=features,
    )


# ---------------------------------------------------------------------------
# Stateful strategy class — manages rolling windows
# ---------------------------------------------------------------------------

class MomentumStrategy(BaseStrategy):
    """Volatility expansion + trend continuation strategy.

    Maintains internal rolling windows for 1m and 5m candles.
    Emits signals only on 1m bar closes, when all conditions pass.
    """

    TRIGGER_TF = "1m"
    CONTEXT_TFS = {"5m", "15m"}

    def __init__(self, cfg: MomentumConfig | None = None) -> None:
        self._cfg = cfg or MomentumConfig()
        # {symbol: {timeframe: deque}}
        self._windows: dict[str, dict[str, deque[Candle]]] = {}

    def on_candle(self, candle: Candle) -> Signal | None:
        sym = candle.symbol
        tf = candle.timeframe

        if sym not in self._windows:
            # +1 so we always have one extra bar for TR / slope computation
            self._windows[sym] = {
                "1m": deque(maxlen=self._cfg.min_bars_1m + 10),
                "5m": deque(maxlen=self._cfg.min_bars_5m + 10),
                "15m": deque(maxlen=20),
            }

        if tf in self._windows[sym]:
            self._windows[sym][tf].append(candle)

        if tf != self.TRIGGER_TF:
            return None

        candles_1m = list(self._windows[sym]["1m"])
        candles_5m = list(self._windows[sym]["5m"])
        return compute_signal(candles_1m, candles_5m, self._cfg)

    def reset(self) -> None:
        self._windows.clear()
