"""Signal, trade, and metrics data models — no logic, pure data."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class ExitReason(str, Enum):
    STOP = "stop"
    TP1 = "tp1"
    TP2 = "tp2"
    TP3 = "tp3"
    FORCED = "forced"       # end of backtest / manual close


@dataclass(slots=True)
class Signal:
    symbol: str
    direction: Direction
    timeframe: str              # trigger timeframe (always "1m")
    bar_close_time: int         # epoch seconds — when signal fired
    entry_price: float          # approximate fill price (close + slippage accounted later)
    stop_price: float           # hard stop level
    tp_prices: list[float]      # take-profit ladder, ordered away from entry
    atr: float                  # ATR used to compute levels
    features: dict[str, float]  # all computed features for audit/research
    reject_reason: str = ""     # non-empty when the signal was blocked by risk engine


@dataclass
class ClosedTrade:
    symbol: str
    direction: Direction
    entry_price: float
    exit_price: float           # weighted-average exit across partial closes
    total_size: float
    entry_time: int             # epoch seconds
    exit_time: int
    pnl: float                  # net pnl in quote currency, after fees
    fees: float
    equity_at_entry: float
    hold_bars: int              # number of 1m bars from entry to final exit
    exit_reason: ExitReason
    features_at_entry: dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestMetrics:
    total_trades: int
    win_rate: float
    expectancy: float           # mean(pnl per trade) / mean(abs(loss)) — per unit risked
    sharpe: float               # annualised per-trade Sharpe
    max_drawdown: float         # peak-to-trough as fraction of peak equity
    profit_factor: float        # gross_profit / gross_loss
    avg_hold_bars: float
    net_pnl: float
    total_fees: float
    long_trades: int
    long_win_rate: float
    short_trades: int
    short_win_rate: float
    pnl_distribution: list[float]   # per-trade net pnl values
    equity_curve: list[tuple[int, float]]  # (epoch_s, equity)
