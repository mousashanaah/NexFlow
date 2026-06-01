"""Daily trend-following strategy — Engine #1.

Specification:
  Entry:   Daily close > highest close of prior LOOKBACK bars  → LONG
           Daily close < lowest close of prior LOOKBACK bars   → SHORT
  Stop:    INITIAL_STOP_MULT × ATR(ATR_PERIOD) from entry price
  Trail:   Highest/lowest close since entry minus/plus TRAIL_MULT × ATR
           Trail only moves in the favourable direction; updates each bar close.
  Exit:    Trail stop hit (evaluated on bar high/low). No take-profit.
  Sizing:  See run_trend_backtest.py — risk engine lives outside this class.

The strategy is candle-agnostic: it ignores any timeframe that is not "1D".
It is stateless regarding open positions — position tracking lives in the
backtest runner. This class only answers: "should I enter, and at what price
and stop?"

Signal.tp_prices is set to a single far-away level (entry ± 999×ATR) so that
the existing Signal model is satisfied. The backtest runner ignores it.
"""

from __future__ import annotations

from collections import deque

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.base_strategy import BaseStrategy
from nexflow.services.strategy.signal_models import Direction, Signal

# Default parameters — do not change without a documented reason and fresh backtest.
_LOOKBACK: int   = 20
_ATR_PERIOD: int = 14
_INITIAL_STOP_MULT: float = 2.5
_TRAIL_MULT: float        = 2.0


class TrendFollowingStrategy(BaseStrategy):
    """Minimal daily trend-following engine.

    Processes only "1D" candles. Returns a Signal on breakout bars.
    Direction flips are signalled by returning the opposite direction while
    the backtest runner has an open position — the runner handles the close.
    """

    def __init__(
        self,
        lookback: int   = _LOOKBACK,
        atr_period: int = _ATR_PERIOD,
        initial_stop_mult: float = _INITIAL_STOP_MULT,
        trail_mult: float        = _TRAIL_MULT,
    ) -> None:
        self._lookback          = lookback
        self._atr_period        = atr_period
        self._initial_stop_mult = initial_stop_mult
        self._trail_mult        = trail_mult

        # Rolling windows — sizes include one extra for the current bar
        self._closes: deque[float] = deque(maxlen=lookback + 1)
        self._highs:  deque[float] = deque(maxlen=atr_period + 2)
        self._lows:   deque[float] = deque(maxlen=atr_period + 2)
        self._prev_c: deque[float] = deque(maxlen=atr_period + 2)

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def on_candle(self, candle: Candle) -> Signal | None:
        if candle.timeframe != "1D":
            return None

        # Append previous close before updating (needed for TR calc)
        prev_close = self._prev_c[-1] if self._prev_c else candle.open
        self._highs.append(candle.high)
        self._lows.append(candle.low)
        self._prev_c.append(candle.close)
        self._closes.append(candle.close)

        # Need enough data
        if len(self._closes) < self._lookback + 1 or len(self._prev_c) < self._atr_period + 1:
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
        prior = list(self._closes)[:-1]
        if len(prior) < self._lookback:
            return None
        prior = prior[-self._lookback:]

        highest = max(prior)
        lowest  = min(prior)
        close   = candle.close

        if close > highest:
            direction = Direction.LONG
            stop = close - self._initial_stop_mult * atr
            tp   = close + 999.0 * atr   # sentinel — never hit
        elif close < lowest:
            direction = Direction.SHORT
            stop = close + self._initial_stop_mult * atr
            tp   = close - 999.0 * atr
        else:
            return None

        return Signal(
            symbol        = candle.symbol,
            direction     = direction,
            timeframe     = "1D",
            bar_close_time= candle.close_time,
            entry_price   = close,          # actual fill = next bar open (handled by runner)
            stop_price    = stop,
            tp_prices     = [tp],
            atr           = atr,
            features      = {
                "highest_prior": highest,
                "lowest_prior":  lowest,
                "trail_mult":    self._trail_mult,
            },
        )

    def reset(self) -> None:
        self._closes.clear()
        self._highs.clear()
        self._lows.clear()
        self._prev_c.clear()

    # ------------------------------------------------------------------
    # Trail stop helper — called by the backtest runner each bar
    # ------------------------------------------------------------------

    @staticmethod
    def update_trail(
        current_trail: float,
        direction: Direction,
        bar_close: float,
        best_close: float,
        atr: float,
        trail_mult: float = _TRAIL_MULT,
    ) -> tuple[float, float]:
        """Return (new_trail_stop, new_best_close).

        Trail only moves in the favourable direction (never widens the risk).
        best_close tracks the highest close since entry (long) or lowest (short).
        """
        if direction is Direction.LONG:
            new_best  = max(best_close, bar_close)
            new_trail = max(current_trail, new_best - trail_mult * atr)
        else:
            new_best  = min(best_close, bar_close)
            new_trail = min(current_trail, new_best + trail_mult * atr)
        return new_trail, new_best


# ---------------------------------------------------------------------------
# ATR calculation
# ---------------------------------------------------------------------------

def _wilder_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int,
) -> float | None:
    """Wilder's smoothed ATR.  closes[i] is the close of bar i (i=0 is oldest)."""
    if len(closes) < period + 1:
        return None

    trs: list[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i]  - lows[i],
            abs(highs[i]  - closes[i - 1]),
            abs(lows[i]   - closes[i - 1]),
        )
        trs.append(tr)

    if len(trs) < period:
        return None

    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period

    return atr
