"""Higher-timeframe (4H) channel breakout + trailing stop strategy.

Specification matches the validated backtest (mechanism #7d):
  Entry: close beyond prior LOOKBACK-bar 4H Donchian channel
  Stop:  TRAIL_MULT × ATR(ATR_PERIOD) chandelier trailing stop
  Filter: LONG only when close > SMA(REGIME_MA); SHORT only when close < SMA(REGIME_MA)
         (regime_ma=0 disables the filter)
  Exit:  trailing stop only — no take-profit

Validated parameters (do not change without fresh backtest):
  lookback=30, atr_period=14, trail_mult=3.0, regime_ma=200
  Universe: BTCUSDT, ETHUSDT, SOLUSDT, TRXUSDT

This class is STATELESS regarding open positions — position tracking
lives in the router / backtest runner. It only answers: should I enter?

Usage:
    strategy = HTFTrendStrategy()
    signal = strategy.on_candle(candle_4h)
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.base_strategy import BaseStrategy
from nexflow.services.strategy.signal_models import Direction, Signal

# Validated parameters — do not tune without fresh OOS backtest.
_LOOKBACK: int    = 30
_ATR_PERIOD: int  = 14
_TRAIL_MULT: float = 3.0
_REGIME_MA: int   = 200
_TIMEFRAME: str   = "4H"


def _wilder_atr(
    highs: list[float],
    lows: list[float],
    prev_closes: list[float],
    period: int,
) -> Optional[float]:
    """Wilder smoothed ATR over the last `period` bars."""
    if len(highs) < period + 1:
        return None
    trs = []
    for i in range(1, len(highs)):
        pc = prev_closes[i - 1]
        tr = max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc))
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


class HTFTrendStrategy(BaseStrategy):
    """4H channel breakout with chandelier trailing stop and SMA regime filter.

    Processes only candles whose timeframe matches the configured timeframe (default "4H").
    Returns a Signal on breakout bars that pass the regime filter.

    The Signal carries the initial stop price in `stop_price` and the trail
    multiplier in `features["trail_mult"]`. The router applies the trailing stop
    on every subsequent bar by calling `update_trail`.
    """

    def __init__(
        self,
        lookback: int = _LOOKBACK,
        atr_period: int = _ATR_PERIOD,
        trail_mult: float = _TRAIL_MULT,
        regime_ma: int = _REGIME_MA,
        timeframe: str = _TIMEFRAME,
    ) -> None:
        self._lookback    = lookback
        self._atr_period  = atr_period
        self._trail_mult  = trail_mult
        self._regime_ma   = regime_ma
        self._timeframe   = timeframe

        self._closes: deque[float] = deque(maxlen=max(lookback + 1, regime_ma + 1))
        self._highs:  deque[float] = deque(maxlen=atr_period + 2)
        self._lows:   deque[float] = deque(maxlen=atr_period + 2)
        self._prev_c: deque[float] = deque(maxlen=atr_period + 2)

        # Running SMA sum for O(1) update
        self._sma_window: deque[float] = deque(maxlen=regime_ma) if regime_ma > 0 else None
        self._sma_sum: float = 0.0

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def on_candle(self, candle: Candle) -> Signal | None:
        if candle.timeframe != self._timeframe:
            return None

        prev_close = self._prev_c[-1] if self._prev_c else candle.open
        self._highs.append(candle.high)
        self._lows.append(candle.low)
        self._prev_c.append(candle.close)
        self._closes.append(candle.close)

        # Update O(1) SMA
        sma: Optional[float] = None
        if self._sma_window is not None:
            if len(self._sma_window) == self._regime_ma:
                self._sma_sum -= self._sma_window[0]
            self._sma_window.append(candle.close)
            self._sma_sum += candle.close
            if len(self._sma_window) == self._regime_ma:
                sma = self._sma_sum / self._regime_ma

        # Need enough data for channel + ATR
        if len(self._closes) < self._lookback + 1:
            return None
        if len(self._prev_c) < self._atr_period + 1:
            return None
        # Need SMA if regime filter active
        if self._regime_ma > 0 and sma is None:
            return None

        atr = _wilder_atr(
            list(self._highs),
            list(self._lows),
            list(self._prev_c),
            self._atr_period,
        )
        if atr is None or atr <= 0:
            return None

        # Prior LOOKBACK closes, excluding current bar
        closes_list = list(self._closes)
        prior = closes_list[-(self._lookback + 1):-1]
        if len(prior) < self._lookback:
            return None

        prior_high = max(prior)
        prior_low  = min(prior)
        close = candle.close

        # Breakout check
        if close > prior_high:
            # Regime filter: LONG only above SMA
            if self._regime_ma > 0 and close <= sma:
                return None
            direction = Direction.LONG
            stop = close - self._trail_mult * atr
            tp   = close + 999.0 * atr   # sentinel — never hit
        elif close < prior_low:
            # Regime filter: SHORT only below SMA
            if self._regime_ma > 0 and close >= sma:
                return None
            direction = Direction.SHORT
            stop = close + self._trail_mult * atr
            tp   = close - 999.0 * atr
        else:
            return None

        return Signal(
            symbol         = candle.symbol,
            direction      = direction,
            timeframe      = self._timeframe,
            bar_close_time = candle.close_time,
            entry_price    = close,      # actual fill = next bar open (router handles this)
            stop_price     = stop,
            tp_prices      = [tp],
            atr            = atr,
            features       = {
                "prior_high":  prior_high,
                "prior_low":   prior_low,
                "trail_mult":  self._trail_mult,
                "sma_200":     sma,
            },
        )

    def reset(self) -> None:
        self._closes.clear()
        self._highs.clear()
        self._lows.clear()
        self._prev_c.clear()
        if self._sma_window is not None:
            self._sma_window.clear()
        self._sma_sum = 0.0

    # ------------------------------------------------------------------
    # Trail management (called by router on every bar for open positions)
    # ------------------------------------------------------------------

    def update_trail(
        self,
        direction: Direction,
        current_stop: float,
        candle: Candle,
        atr: float,
    ) -> float:
        """Return the new trailing stop, ratcheting only in the favourable direction."""
        new_stop: float
        if direction == Direction.LONG:
            new_stop = candle.close - self._trail_mult * atr
            return max(current_stop, new_stop)
        else:
            new_stop = candle.close + self._trail_mult * atr
            return min(current_stop, new_stop)
