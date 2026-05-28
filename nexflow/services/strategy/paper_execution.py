"""Paper execution layer — simulates maker/taker fees, slippage, and spread crossing."""

from __future__ import annotations

from dataclasses import dataclass

from nexflow.services.strategy.signal_models import Direction, Signal


@dataclass
class ExecutionConfig:
    taker_fee: float = 0.0006          # Bitget USDT-Futures taker = 0.06%
    maker_fee: float = 0.0002          # maker = 0.02%
    slippage_atr_fraction: float = 0.05   # slippage = 5% of ATR
    spread_cross_fraction: float = 0.5    # pay half spread on market orders


@dataclass
class FillResult:
    fill_price: float
    fee: float      # in quote currency
    is_maker: bool


class PaperExecution:
    """Simulates realistic fills with fees and slippage.

    Assumptions:
    - Entry orders are market/taker (cross the spread, pay taker fee).
    - Stop-loss orders are market (taker fee, full slippage adverse).
    - Take-profit orders are limit (maker fee, no extra slippage).
    - Slippage scales with ATR (illiquidity proxy) + half-spread for entries.
    """

    def __init__(self, cfg: ExecutionConfig | None = None) -> None:
        self._cfg = cfg or ExecutionConfig()

    def simulate_entry(self, signal: Signal) -> FillResult:
        """Market entry: taker order with ATR-scaled slippage + half-spread crossing."""
        cfg = self._cfg
        # Half-spread crossing proxy: spread_cross_fraction × 1% of ATR
        slip = signal.atr * (cfg.slippage_atr_fraction + cfg.spread_cross_fraction * 0.005)

        if signal.direction is Direction.LONG:
            fill_price = signal.entry_price + slip
        else:
            fill_price = signal.entry_price - slip

        return FillResult(fill_price=fill_price, fee=0.0, is_maker=False)

    def simulate_stop(self, stop_price: float, direction: Direction, stop_distance: float) -> FillResult:
        """Stop-loss fill: adverse slippage proportional to stop distance.

        stop_distance: abs(entry_price - stop_price) — used as ATR proxy.
        """
        cfg = self._cfg
        slip = stop_distance * cfg.slippage_atr_fraction * 0.5

        if direction is Direction.LONG:
            fill_price = stop_price - slip   # long stop fills below stop level
        else:
            fill_price = stop_price + slip

        return FillResult(fill_price=fill_price, fee=0.0, is_maker=False)

    def simulate_tp(self, tp_price: float) -> FillResult:
        """Take-profit fill: limit order at exact TP price, maker fee."""
        return FillResult(fill_price=tp_price, fee=0.0, is_maker=True)

    def compute_fee(self, price: float, size: float, *, is_maker: bool) -> float:
        """Return fee in quote currency."""
        rate = self._cfg.maker_fee if is_maker else self._cfg.taker_fee
        return price * size * rate

