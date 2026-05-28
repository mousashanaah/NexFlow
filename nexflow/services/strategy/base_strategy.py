"""Abstract base class for all NexFlow strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.signal_models import Signal


class BaseStrategy(ABC):
    """Stateful strategy that receives one candle at a time (any timeframe).

    The strategy is called in strict chronological order. It may accumulate
    rolling windows internally and return a Signal only on the trigger timeframe.

    Guarantees the framework provides:
    - Candles arrive with is_final=True, in ascending close_time order.
    - Higher timeframes (15m, 5m) at a shared timestamp are delivered before 1m.
    - No future data is ever present in the window.
    """

    @abstractmethod
    def on_candle(self, candle: Candle) -> Signal | None:
        """Process one finalized candle.

        Returns a Signal when a tradeable condition is detected, None otherwise.
        Signals should only be emitted for the strategy's trigger timeframe.
        """
        ...

    def reset(self) -> None:
        """Reset all internal state (used between backtest runs)."""
