"""EMA 8/21 Long-Only Trend Strategy — mechanism #12 (GO verdict).

Entry : EMA(8) crosses above EMA(21) on daily close → OPEN LONG
Exit  : EMA(8) crosses below EMA(21) on daily close → CLOSE LONG

Long-only: sits flat during downtrends. Never short.
One position per symbol, equal-weight sizing.

Validated: CAGR 24%, DD 11%, PF 1.95, 518 trades, 2021-2026.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class EMASignal:
    symbol: str
    action: str         # "OPEN_LONG" | "CLOSE_LONG"
    price: float
    timestamp_ms: int
    reason: str = ""


class EMABuffer:
    """Maintains EMA(fast) and EMA(slow) for one symbol using daily closes."""

    def __init__(self, fast: int = 8, slow: int = 21) -> None:
        self._fast_period = fast
        self._slow_period = slow
        self._alpha_fast  = 2.0 / (fast + 1)
        self._alpha_slow  = 2.0 / (slow + 1)
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._bars_seen: int = 0

    def update(self, close: float) -> tuple[Optional[float], Optional[float]]:
        """Feed a daily close. Returns (ema_fast, ema_slow) — None until warmed up."""
        self._bars_seen += 1
        if self._ema_fast is None:
            self._ema_fast = close
        else:
            self._ema_fast = self._alpha_fast * close + (1 - self._alpha_fast) * self._ema_fast

        if self._ema_slow is None:
            self._ema_slow = close
        else:
            self._ema_slow = self._alpha_slow * close + (1 - self._alpha_slow) * self._ema_slow

        # Require at least slow_period bars before emitting signals
        if self._bars_seen < self._slow_period:
            return None, None
        return self._ema_fast, self._ema_slow

    @property
    def warmed_up(self) -> bool:
        return self._bars_seen >= self._slow_period

    def reset(self) -> None:
        self._ema_fast = None
        self._ema_slow = None
        self._bars_seen = 0


class EMATrendStrategy:
    """Portfolio-level EMA 8/21 long-only strategy.

    Call on_daily_close() for each symbol's daily candle close.
    Returns a list of EMASignals when there's a crossover for that symbol.
    Emits at most one OPEN_LONG or CLOSE_LONG per symbol per bar.
    """

    def __init__(
        self,
        symbols: list[str],
        fast: int = 8,
        slow: int = 21,
    ) -> None:
        self._symbols = symbols
        self._fast    = fast
        self._slow    = slow
        self._buffers: dict[str, EMABuffer]   = {s: EMABuffer(fast, slow) for s in symbols}
        self._prev_above: dict[str, Optional[bool]] = {s: None for s in symbols}
        self._positions: dict[str, bool] = {s: False for s in symbols}  # True = in long

    def on_daily_close(
        self,
        symbol: str,
        close: float,
        timestamp_ms: int,
    ) -> list[EMASignal]:
        if symbol not in self._buffers:
            return []

        ema_fast, ema_slow = self._buffers[symbol].update(close)
        if ema_fast is None or ema_slow is None:
            return []

        above = ema_fast > ema_slow
        prev  = self._prev_above[symbol]
        signals: list[EMASignal] = []

        if prev is not None and above != prev:
            if above:
                # EMA fast crossed ABOVE slow → buy signal
                if not self._positions[symbol]:
                    signals.append(EMASignal(
                        symbol=symbol, action="OPEN_LONG",
                        price=close, timestamp_ms=timestamp_ms,
                        reason=f"EMA{self._fast}>{self._slow}",
                    ))
                    self._positions[symbol] = True
            else:
                # EMA fast crossed BELOW slow → exit signal
                if self._positions[symbol]:
                    signals.append(EMASignal(
                        symbol=symbol, action="CLOSE_LONG",
                        price=close, timestamp_ms=timestamp_ms,
                        reason=f"EMA{self._fast}<{self._slow}",
                    ))
                    self._positions[symbol] = False

        self._prev_above[symbol] = above
        return signals

    @property
    def positions(self) -> dict[str, bool]:
        return dict(self._positions)

    def current_signals(self) -> dict[str, str]:
        """Current EMA alignment per symbol: 'LONG' or 'FLAT'."""
        result = {}
        for sym, buf in self._buffers.items():
            if buf._ema_fast is not None and buf._ema_slow is not None and buf.warmed_up:
                result[sym] = "LONG" if buf._ema_fast > buf._ema_slow else "FLAT"
            else:
                result[sym] = "WARMUP"
        return result

    def reset(self, symbol: Optional[str] = None) -> None:
        targets = [symbol] if symbol else self._symbols
        for s in targets:
            if s in self._buffers:
                self._buffers[s].reset()
                self._prev_above[s] = None
                self._positions[s] = False
