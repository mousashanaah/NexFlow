"""Tests for the strategy framework — fully deterministic, no network, no randomness."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.backtest_runner import (
    BacktestConfig,
    BacktestRunner,
    _compute_max_drawdown,
    _compute_sharpe,
)
from nexflow.services.strategy.momentum_strategy import (
    MomentumConfig,
    MomentumStrategy,
    compute_atr,
    compute_breakout_level,
    compute_buy_sell_imbalance,
    compute_momentum_slope,
    compute_range_expansion,
    compute_relative_volume,
    compute_rolling_vol,
    compute_signal,
    compute_spread_regime,
)
from nexflow.services.strategy.paper_execution import ExecutionConfig, PaperExecution
from nexflow.services.strategy.portfolio import Portfolio, Position, TpLevel
from nexflow.services.strategy.risk_engine import RiskConfig, RiskEngine
from nexflow.services.strategy.signal_models import Direction, ExitReason, Signal


# ---------------------------------------------------------------------------
# Candle factories
# ---------------------------------------------------------------------------

def _c(
    o: float = 100.0, h: float = 101.0, l: float = 99.0, c: float = 100.0,
    vol: float = 1000.0, buy_vol: float = 550.0,
    spread: float = 0.05, vwap: float | None = None,
    t: int = 1_000_000, tf: str = "1m",
) -> Candle:
    return Candle(
        symbol="BTCUSDT", timeframe=tf,
        open_time=t, close_time=t + 60,
        open=o, high=h, low=l, close=c,
        volume=vol, buy_volume=buy_vol, sell_volume=vol - buy_vol,
        trade_count=50, vwap=vwap or c,
        spread_avg=spread, spread_max=spread * 2.0,
        volatility_estimate=(h - l) / o if o > 0 else 0.0,
        is_final=True,
    )


def _uniform_candles(n: int, price: float = 100.0, step: float = 0.0,
                      h_off: float = 1.0, l_off: float = 1.0,
                      vol: float = 1000.0, buy_frac: float = 0.55,
                      spread: float = 0.05, tf: str = "1m") -> list[Candle]:
    """Generate n candles with uniform structure and optional price trend."""
    result = []
    p = price
    for i in range(n):
        nxt = p + step
        result.append(_c(o=p, h=p + h_off, l=p - l_off, c=nxt,
                          vol=vol, buy_vol=vol * buy_frac,
                          spread=spread, t=1_000_000 + i * 60, tf=tf))
        p = nxt
    return result


# ---------------------------------------------------------------------------
# compute_atr
# ---------------------------------------------------------------------------

def test_compute_atr_uniform_range() -> None:
    """Bars with H-L=2, no gaps → ATR=2."""
    candles = [_c(o=100, h=101, l=99, c=100, t=1_000_000 + i * 60) for i in range(16)]
    atr = compute_atr(candles, period=14)
    assert atr == pytest.approx(2.0, rel=1e-6)


def test_compute_atr_single_bar_fallback() -> None:
    atr = compute_atr([_c(o=100, h=105, l=95)], period=14)
    assert atr == pytest.approx(10.0)


def test_compute_atr_uses_true_range_not_just_hl() -> None:
    """Gap up: prev_close=95, current H=102, L=100 → TR = max(2, 7, 5) = 7."""
    prev = _c(o=90, h=96, l=89, c=95, t=1_000_000)
    curr = _c(o=101, h=102, l=100, c=101, t=1_000_060)
    atr = compute_atr([prev, curr], period=1)
    assert atr == pytest.approx(7.0)


def test_compute_atr_grows_with_larger_ranges() -> None:
    small = [_c(o=100, h=101, l=99, c=100, t=1_000_000 + i * 60) for i in range(15)]
    large = [_c(o=100, h=105, l=95, c=100, t=1_000_000 + i * 60) for i in range(15)]
    assert compute_atr(large, 14) > compute_atr(small, 14)


# ---------------------------------------------------------------------------
# compute_rolling_vol
# ---------------------------------------------------------------------------

def test_compute_rolling_vol_flat_returns_zero() -> None:
    candles = [_c(c=100.0, t=1_000_000 + i * 60) for i in range(22)]
    vol = compute_rolling_vol(candles, period=20)
    assert vol == pytest.approx(0.0, abs=1e-10)


def test_compute_rolling_vol_positive_for_varying() -> None:
    prices = [100.0, 101.0, 99.5, 102.0, 100.5] * 4
    candles = [_c(c=p, t=1_000_000 + i * 60) for i, p in enumerate(prices)]
    vol = compute_rolling_vol(candles, period=len(prices) - 1)
    assert vol > 0.0


def test_compute_rolling_vol_insufficient_data() -> None:
    assert compute_rolling_vol([_c()], period=20) == 0.0


# ---------------------------------------------------------------------------
# compute_relative_volume
# ---------------------------------------------------------------------------

def test_compute_relative_volume_equal_history() -> None:
    """20 bars all at 1000 volume, current at 1500 → rel_vol = 1.5."""
    candles = [_c(vol=1000.0, t=1_000_000 + i * 60) for i in range(20)]
    candles.append(_c(vol=1500.0, t=1_002_000))
    rv = compute_relative_volume(candles, period=20)
    assert rv == pytest.approx(1.5)


def test_compute_relative_volume_below_one() -> None:
    candles = [_c(vol=1000.0, t=1_000_000 + i * 60) for i in range(21)]
    candles[-1] = _c(vol=500.0, t=1_002_060)
    rv = compute_relative_volume(candles, period=20)
    assert rv == pytest.approx(0.5)


def test_compute_relative_volume_insufficient_history() -> None:
    assert compute_relative_volume([_c()], period=20) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_momentum_slope
# ---------------------------------------------------------------------------

def test_compute_momentum_slope_uptrend() -> None:
    candles = [_c(c=float(100 + i), t=1_000_000 + i * 60) for i in range(7)]
    slope = compute_momentum_slope(candles, period=5)
    # close[-1]=106, close[-6]=101 (index 0) → (106-101)/101 ≈ 0.0495
    assert slope > 0


def test_compute_momentum_slope_downtrend() -> None:
    candles = [_c(c=float(100 - i), t=1_000_000 + i * 60) for i in range(7)]
    slope = compute_momentum_slope(candles, period=5)
    assert slope < 0


def test_compute_momentum_slope_flat() -> None:
    candles = [_c(c=100.0, t=1_000_000 + i * 60) for i in range(7)]
    slope = compute_momentum_slope(candles, period=5)
    assert slope == pytest.approx(0.0)


def test_compute_momentum_slope_insufficient_data() -> None:
    candles = [_c(c=100.0, t=1_000_000 + i * 60) for i in range(3)]
    assert compute_momentum_slope(candles, period=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_range_expansion, spread_regime, imbalance
# ---------------------------------------------------------------------------

def test_compute_range_expansion_above_one() -> None:
    candle = _c(h=104.0, l=96.0)  # range=8
    assert compute_range_expansion(candle, atr=4.0) == pytest.approx(2.0)


def test_compute_range_expansion_zero_atr() -> None:
    assert compute_range_expansion(_c(), atr=0.0) == pytest.approx(0.0)


def test_compute_spread_regime_acceptable() -> None:
    candle = _c(spread=0.20)
    assert compute_spread_regime(candle, atr=1.0) == pytest.approx(0.20)


def test_compute_buy_sell_imbalance_buy_heavy() -> None:
    candle = _c(vol=1000.0, buy_vol=700.0)
    assert compute_buy_sell_imbalance(candle) == pytest.approx(0.70)


def test_compute_buy_sell_imbalance_zero_volume() -> None:
    candle = _c(vol=0.0, buy_vol=0.0)
    assert compute_buy_sell_imbalance(candle) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_signal — correctness and rejection paths
# ---------------------------------------------------------------------------

def _make_signal_inputs(
    n_1m: int = 25,
    trend_step: float = 0.5,
    breakout_amount: float = 2.0,
    vol_spike: float = 2.0,
    buy_frac: float = 0.65,
    spread: float = 0.05,
    momentum_5m: float = 0.5,     # >0 for long alignment
    cfg: MomentumConfig | None = None,
) -> tuple[list[Candle], list[Candle], MomentumConfig]:
    cfg = cfg or MomentumConfig()

    # Build 1m candles: slow uptrend then a breakout bar
    price = 100.0
    candles_1m: list[Candle] = []
    for i in range(n_1m - 1):
        candles_1m.append(_c(
            o=price, h=price + 0.8, l=price - 0.8, c=price + trend_step,
            vol=1000.0, buy_vol=1000.0 * 0.52, spread=spread,
            t=1_000_000 + i * 60,
        ))
        price += trend_step

    # Breakout trigger bar: closes above previous 20-bar high with volume spike
    rolling_high, _ = compute_breakout_level(candles_1m, cfg.vol_period)
    breakout_close = rolling_high + breakout_amount
    trigger = _c(
        o=price, h=breakout_close + 0.5, l=price - 0.5, c=breakout_close,
        vol=1000.0 * vol_spike, buy_vol=1000.0 * vol_spike * buy_frac,
        spread=spread, t=1_000_000 + (n_1m - 1) * 60,
    )
    candles_1m.append(trigger)

    # 5m candles: simple uptrend to give positive momentum
    p5 = 100.0
    candles_5m: list[Candle] = []
    for i in range(cfg.min_bars_5m + 2):
        candles_5m.append(_c(
            o=p5, h=p5 + 2, l=p5 - 1, c=p5 + momentum_5m,
            vol=5000.0, buy_vol=2750.0, spread=spread,
            t=900_000 + i * 300, tf="5m",
        ))
        p5 += momentum_5m

    return candles_1m, candles_5m, cfg


def test_signal_long_all_conditions_met() -> None:
    candles_1m, candles_5m, cfg = _make_signal_inputs()
    sig = compute_signal(candles_1m, candles_5m, cfg)
    assert sig is not None
    assert sig.direction is Direction.LONG
    assert sig.stop_price < sig.entry_price
    assert len(sig.tp_prices) == 3
    assert sig.tp_prices[0] < sig.tp_prices[1] < sig.tp_prices[2]


def test_signal_rejected_low_relative_volume() -> None:
    candles_1m, candles_5m, cfg = _make_signal_inputs(vol_spike=1.0)  # below threshold 1.5
    sig = compute_signal(candles_1m, candles_5m, cfg)
    assert sig is None


def test_signal_rejected_wide_spread() -> None:
    """spread/ATR > 0.3 → no signal."""
    # ATR ≈ 1.6 (range 0.8+0.8), spread=0.6 → ratio=0.375 > 0.30
    candles_1m, candles_5m, cfg = _make_signal_inputs(spread=0.60)
    sig = compute_signal(candles_1m, candles_5m, cfg)
    assert sig is None


def test_signal_rejected_insufficient_1m_history() -> None:
    candles_1m, candles_5m, cfg = _make_signal_inputs(n_1m=10)  # below min_bars_1m=22
    sig = compute_signal(candles_1m, candles_5m, cfg)
    assert sig is None


def test_signal_rejected_5m_trend_opposing() -> None:
    """5m momentum is negative → no LONG signal."""
    candles_1m, candles_5m, cfg = _make_signal_inputs(momentum_5m=-0.5)
    sig = compute_signal(candles_1m, candles_5m, cfg)
    assert sig is None


def test_signal_rejected_low_buy_imbalance() -> None:
    """buy fraction 0.40 → below imbalance_min 0.55 → no LONG."""
    candles_1m, candles_5m, cfg = _make_signal_inputs(buy_frac=0.40)
    sig = compute_signal(candles_1m, candles_5m, cfg)
    assert sig is None


def test_signal_short_all_conditions_met() -> None:
    """Mirror: downtrend, breakdown below 20-bar low, selling pressure."""
    cfg = MomentumConfig()
    price = 100.0
    candles_1m: list[Candle] = []
    for i in range(cfg.min_bars_1m - 1):
        candles_1m.append(_c(
            o=price, h=price + 0.8, l=price - 0.8, c=price - 0.5,
            vol=1000.0, buy_vol=400.0, spread=0.05,
            t=1_000_000 + i * 60,
        ))
        price -= 0.5

    _, rolling_low = compute_breakout_level(candles_1m, cfg.vol_period)
    breakdown_close = rolling_low - 2.0
    candles_1m.append(_c(
        o=price, h=price + 0.3, l=breakdown_close - 0.5, c=breakdown_close,
        vol=2500.0, buy_vol=2500.0 * 0.30,  # sell-heavy (30% buys = 70% sells)
        spread=0.05, t=1_000_000 + (cfg.min_bars_1m - 1) * 60,
    ))

    # 5m downtrend
    p5 = 100.0
    candles_5m = []
    for i in range(cfg.min_bars_5m + 2):
        candles_5m.append(_c(
            o=p5, h=p5 + 0.5, l=p5 - 2, c=p5 - 1.0,
            vol=5000.0, buy_vol=1800.0, spread=0.05, t=900_000 + i * 300, tf="5m",
        ))
        p5 -= 1.0

    sig = compute_signal(candles_1m, candles_5m, cfg)
    assert sig is not None
    assert sig.direction is Direction.SHORT
    assert sig.stop_price > sig.entry_price
    assert sig.tp_prices[0] > sig.tp_prices[1]  # TPs descend for short


def test_signal_atr_levels_correct() -> None:
    candles_1m, candles_5m, cfg = _make_signal_inputs()
    sig = compute_signal(candles_1m, candles_5m, cfg)
    assert sig is not None
    atr = sig.atr
    expected_stop = sig.entry_price - cfg.atr_stop_mult * atr
    expected_tp1 = sig.entry_price + cfg.atr_tp1_mult * atr
    assert sig.stop_price == pytest.approx(expected_stop, rel=1e-6)
    assert sig.tp_prices[0] == pytest.approx(expected_tp1, rel=1e-6)


# ---------------------------------------------------------------------------
# RiskEngine
# ---------------------------------------------------------------------------

def _simple_portfolio(equity: float = 100_000.0) -> Portfolio:
    return Portfolio(initial_equity=equity)


def _simple_signal(entry: float = 100.0, stop: float = 98.5, atr: float = 1.0) -> Signal:
    return Signal(
        symbol="BTCUSDT", direction=Direction.LONG, timeframe="1m",
        bar_close_time=1_000_060,
        entry_price=entry, stop_price=stop,
        tp_prices=[entry + atr, entry + 2 * atr, entry + 3 * atr],
        atr=atr, features={},
    )


def test_risk_engine_allows_clean_entry() -> None:
    risk = RiskEngine()
    allowed, reason = risk.check_entry(_simple_signal(), _simple_portfolio())
    assert allowed is True
    assert reason == ""


def test_risk_engine_blocks_daily_drawdown() -> None:
    portfolio = _simple_portfolio(100_000.0)
    # Simulate 2% daily loss
    portfolio._day_start_equity = 100_000.0
    portfolio.current_equity = 97_500.0  # 2.5% loss
    risk = RiskEngine(RiskConfig(daily_drawdown_kill=0.02))
    allowed, reason = risk.check_entry(_simple_signal(), portfolio)
    assert allowed is False
    assert "daily_drawdown" in reason


def test_risk_engine_blocks_cooldown() -> None:
    risk = RiskEngine(RiskConfig(cooldown_after_loss_bars=3))
    risk.on_loss()
    allowed, reason = risk.check_entry(_simple_signal(), _simple_portfolio())
    assert allowed is False
    assert "cooldown" in reason


def test_risk_engine_cooldown_expires() -> None:
    risk = RiskEngine(RiskConfig(cooldown_after_loss_bars=2))
    risk.on_loss()
    risk.tick()
    risk.tick()
    # After 2 ticks, cooldown should be 0
    allowed, _ = risk.check_entry(_simple_signal(), _simple_portfolio())
    assert allowed is True


def test_risk_engine_blocks_max_positions() -> None:
    risk = RiskEngine(RiskConfig(max_concurrent_positions=1))
    portfolio = _simple_portfolio()
    # Manually inject a position
    from nexflow.services.strategy.portfolio import Position, TpLevel
    pos = Position(
        symbol="ETHUSDT", direction=Direction.LONG, entry_price=2000.0,
        entry_time=1000, equity_at_entry=100_000.0,
        total_size=1.0, remaining_size=1.0,
        stop_price=1950.0,
        tp_levels=[TpLevel(2050.0, 0.5), TpLevel(2100.0, 0.25), TpLevel(2150.0, 0.25)],
    )
    portfolio.open_position(pos)
    allowed, reason = risk.check_entry(_simple_signal(), portfolio)
    assert allowed is False
    assert "max_positions" in reason


def test_risk_engine_blocks_duplicate_symbol() -> None:
    risk = RiskEngine()
    portfolio = _simple_portfolio()
    from nexflow.services.strategy.portfolio import Position, TpLevel
    pos = Position(
        symbol="BTCUSDT", direction=Direction.LONG, entry_price=100.0,
        entry_time=1000, equity_at_entry=100_000.0,
        total_size=1.0, remaining_size=1.0,
        stop_price=98.5,
        tp_levels=[TpLevel(101.0, 0.5), TpLevel(102.0, 0.25), TpLevel(103.0, 0.25)],
    )
    portfolio.open_position(pos)
    allowed, reason = risk.check_entry(_simple_signal(), portfolio)
    assert allowed is False
    assert "duplicate" in reason


def test_risk_engine_position_sizing_half_percent() -> None:
    """size = equity × 0.005 / stop_distance, capped at max_notional / entry."""
    risk = RiskEngine(RiskConfig(max_risk_per_trade=0.005, max_position_equity_fraction=0.20))
    portfolio = _simple_portfolio(100_000.0)
    # Wide stop (distance=5) → risk-based size=100 < cap (20k/100=200) → not capped
    sig = _simple_signal(entry=100.0, stop=95.0, atr=5.0)
    size = risk.compute_position_size(sig, portfolio)
    expected = 100_000.0 * 0.005 / 5.0  # = 100 units
    assert size == pytest.approx(expected, rel=1e-6)


def test_risk_engine_sizing_capped_by_max_notional() -> None:
    """If risk-based size exceeds 20% of equity / entry_price, it is capped."""
    risk = RiskEngine(RiskConfig(max_risk_per_trade=0.005, max_position_equity_fraction=0.01))
    sig = _simple_signal(entry=100.0, stop=99.99, atr=0.01)  # tiny stop → huge size
    size = risk.compute_position_size(sig, _simple_portfolio(100_000.0))
    max_size = 100_000.0 * 0.01 / 100.0
    assert size == pytest.approx(max_size, rel=1e-6)


# ---------------------------------------------------------------------------
# PaperExecution — fee and fill correctness
# ---------------------------------------------------------------------------

def test_paper_execution_entry_slips_in_direction() -> None:
    exec_cfg = ExecutionConfig(slippage_atr_fraction=0.1, spread_cross_fraction=0.0)
    pe = PaperExecution(exec_cfg)
    sig = _simple_signal(entry=100.0, atr=2.0)
    fill = pe.simulate_entry(sig)
    # Long entry should fill above entry_price
    assert fill.fill_price > sig.entry_price
    assert fill.fill_price == pytest.approx(100.0 + 2.0 * 0.1, rel=1e-6)


def test_paper_execution_short_entry_slips_down() -> None:
    exec_cfg = ExecutionConfig(slippage_atr_fraction=0.1, spread_cross_fraction=0.0)
    pe = PaperExecution(exec_cfg)
    sig = Signal(
        symbol="BTCUSDT", direction=Direction.SHORT, timeframe="1m",
        bar_close_time=1000, entry_price=100.0, stop_price=101.5,
        tp_prices=[99.0, 98.0, 97.0], atr=2.0, features={},
    )
    fill = pe.simulate_entry(sig)
    assert fill.fill_price < sig.entry_price


def test_paper_execution_stop_fill_adverse() -> None:
    pe = PaperExecution(ExecutionConfig(slippage_atr_fraction=0.1))
    fill = pe.simulate_stop(stop_price=98.5, direction=Direction.LONG, stop_distance=1.5)
    # Long stop fills below stop level
    assert fill.fill_price < 98.5


def test_paper_execution_tp_fill_exact() -> None:
    pe = PaperExecution()
    fill = pe.simulate_tp(tp_price=102.0)
    assert fill.fill_price == pytest.approx(102.0)
    assert fill.is_maker is True


def test_paper_execution_fee_taker() -> None:
    pe = PaperExecution(ExecutionConfig(taker_fee=0.0006))
    fee = pe.compute_fee(price=100.0, size=10.0, is_maker=False)
    assert fee == pytest.approx(100.0 * 10.0 * 0.0006)


def test_paper_execution_fee_maker_lower() -> None:
    pe = PaperExecution(ExecutionConfig(taker_fee=0.0006, maker_fee=0.0002))
    taker = pe.compute_fee(100.0, 10.0, is_maker=False)
    maker = pe.compute_fee(100.0, 10.0, is_maker=True)
    assert maker < taker


# ---------------------------------------------------------------------------
# Portfolio — pnl accounting and position lifecycle
# ---------------------------------------------------------------------------

def _make_position(
    entry: float = 100.0, stop: float = 98.5, size: float = 10.0,
    direction: Direction = Direction.LONG,
    equity: float = 100_000.0,
) -> Position:
    tp1 = entry + 1.5 if direction is Direction.LONG else entry - 1.5
    tp2 = entry + 3.0 if direction is Direction.LONG else entry - 3.0
    tp3 = entry + 4.5 if direction is Direction.LONG else entry - 4.5
    return Position(
        symbol="BTCUSDT", direction=direction,
        entry_price=entry, entry_time=1_000_000,
        equity_at_entry=equity,
        total_size=size, remaining_size=size,
        stop_price=stop,
        tp_levels=[
            TpLevel(tp1, size * 0.5),
            TpLevel(tp2, size * 0.25),
            TpLevel(tp3, size * 0.25),
        ],
    )


def test_portfolio_open_close_round_trip() -> None:
    port = _simple_portfolio()
    pos = _make_position()
    port.open_position(pos)
    assert port.has_position("BTCUSDT")

    # Apply a full exit at 102.0
    fee = 0.06
    pos.apply_partial_close(102.0, pos.remaining_size, fee)
    trade = port.close_position("BTCUSDT", exit_time=1_000_060, exit_reason=ExitReason.TP3)

    assert not port.has_position("BTCUSDT")
    # pnl = (102-100)*10 - 0.06 = 20 - 0.06 = 19.94
    assert trade.pnl == pytest.approx(19.94, rel=1e-6)
    assert port.current_equity == pytest.approx(100_000.0 + 19.94, rel=1e-6)


def test_portfolio_stop_loss_reduces_equity() -> None:
    port = _simple_portfolio()
    pos = _make_position(entry=100.0, stop=98.5, size=10.0)
    port.open_position(pos)

    # Stop fill at 98.4 (slippage below stop)
    loss = (98.4 - 100.0) * 10.0  # = -16.0
    fee = 0.05
    pos.apply_partial_close(98.4, pos.remaining_size, fee)
    trade = port.close_position("BTCUSDT", exit_time=1_000_060, exit_reason=ExitReason.STOP)

    assert trade.pnl == pytest.approx(loss - fee, rel=1e-6)
    assert port.current_equity < 100_000.0


def test_portfolio_partial_tp_then_stop() -> None:
    """TP1 (50% size) then stop on remaining 50% → partial win + partial loss."""
    port = _simple_portfolio()
    pos = _make_position(entry=100.0, stop=98.5, size=10.0)
    port.open_position(pos)

    # TP1 hit: close 5 units at 101.5
    pos.apply_partial_close(101.5, 5.0, fee=0.03, move_stop_to_be=True)
    assert pos.stop_price == pytest.approx(100.0)  # moved to breakeven
    assert pos.remaining_size == pytest.approx(5.0)

    # Stop hit on remaining 5 units at breakeven (100.0)
    pos.apply_partial_close(100.0, 5.0, fee=0.03)
    trade = port.close_position("BTCUSDT", 1_000_120, ExitReason.STOP)

    tp1_pnl = (101.5 - 100.0) * 5.0  # +7.5
    be_pnl = (100.0 - 100.0) * 5.0   # 0.0
    net = tp1_pnl + be_pnl - 0.06    # fees
    assert trade.pnl == pytest.approx(net, rel=1e-6)


def test_portfolio_daily_drawdown_tracking() -> None:
    port = Portfolio(100_000.0)
    port.update_day(86_400)           # day 1 start
    assert port.daily_drawdown() == pytest.approx(0.0)

    port.current_equity = 98_000.0
    assert port.daily_drawdown() == pytest.approx(0.02, rel=1e-6)


def test_portfolio_daily_reset() -> None:
    port = Portfolio(100_000.0)
    port.update_day(86_400)            # day 1
    port.current_equity = 98_000.0    # 2% loss on day 1
    port.update_day(2 * 86_400)       # day 2 — resets baseline
    # Day 2 starts at 98k, no further loss
    assert port.daily_drawdown() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Backtest runner — integration / metrics
# ---------------------------------------------------------------------------

def _make_backtest_candles(n_1m: int = 60, trend_step: float = 0.5) -> dict[str, dict[str, list[Candle]]]:
    """Generate a simple uptrend dataset sufficient for the strategy to fire."""
    cfg = MomentumConfig()
    candles_1m = _uniform_candles(n_1m, price=100.0, step=trend_step,
                                   h_off=0.8, l_off=0.8, vol=1000.0, buy_frac=0.60)

    # Add volume spikes on every 25th bar to trigger breakout
    for i in range(cfg.min_bars_1m, n_1m, 25):
        c = candles_1m[i]
        candles_1m[i] = Candle(
            symbol=c.symbol, timeframe=c.timeframe,
            open_time=c.open_time, close_time=c.close_time,
            open=c.open, high=c.high + 2.0, low=c.low,
            close=c.close + 2.0,         # breakout close
            volume=c.volume * 2.5,        # volume spike
            buy_volume=c.volume * 2.5 * 0.70,
            sell_volume=c.volume * 2.5 * 0.30,
            trade_count=c.trade_count, vwap=c.close + 1.0,
            spread_avg=c.spread_avg, spread_max=c.spread_max,
            volatility_estimate=c.volatility_estimate, is_final=True,
        )

    candles_5m = _uniform_candles(n_1m // 5 + 10, price=100.0, step=trend_step * 5,
                                   h_off=2.5, l_off=2.5, vol=5000.0, buy_frac=0.60,
                                   tf="5m")
    # Fix 5m timestamps to 5m intervals
    candles_5m = [
        Candle(symbol=c.symbol, timeframe="5m",
               open_time=1_000_000 + i * 300, close_time=1_000_300 + i * 300,
               open=c.open, high=c.high, low=c.low, close=c.close,
               volume=c.volume, buy_volume=c.buy_volume, sell_volume=c.sell_volume,
               trade_count=c.trade_count, vwap=c.vwap,
               spread_avg=c.spread_avg, spread_max=c.spread_max,
               volatility_estimate=c.volatility_estimate, is_final=True)
        for i, c in enumerate(candles_5m)
    ]

    return {"BTCUSDT": {"1m": candles_1m, "5m": candles_5m}}


def test_backtest_runs_without_error() -> None:
    all_candles = _make_backtest_candles(n_1m=80)
    strategy = MomentumStrategy()
    runner = BacktestRunner(strategy)
    metrics = runner.run(all_candles)
    assert metrics is not None
    assert metrics.total_trades >= 0


def test_backtest_equity_monotone_without_trades() -> None:
    """With no signals possible (too few bars), equity should stay flat."""
    candles_1m = _uniform_candles(10, price=100.0)  # too few to fire
    candles_5m = _uniform_candles(3, price=100.0, tf="5m")
    all_candles = {"BTCUSDT": {"1m": candles_1m, "5m": candles_5m}}
    metrics = BacktestRunner(MomentumStrategy()).run(all_candles)
    assert metrics.total_trades == 0
    assert metrics.net_pnl == pytest.approx(0.0)


def test_backtest_fees_deducted() -> None:
    """Any executed trade must have positive fees."""
    all_candles = _make_backtest_candles(n_1m=80)
    metrics = BacktestRunner(MomentumStrategy()).run(all_candles)
    if metrics.total_trades > 0:
        assert metrics.total_fees > 0.0


def test_backtest_pnl_distribution_matches_trade_count() -> None:
    all_candles = _make_backtest_candles(n_1m=80)
    metrics = BacktestRunner(MomentumStrategy()).run(all_candles)
    assert len(metrics.pnl_distribution) == metrics.total_trades


def test_backtest_win_rate_in_bounds() -> None:
    all_candles = _make_backtest_candles(n_1m=80)
    metrics = BacktestRunner(MomentumStrategy()).run(all_candles)
    assert 0.0 <= metrics.win_rate <= 1.0


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def test_compute_max_drawdown_known_curve() -> None:
    # Peak=110 at t=1, then drops to 90 → drawdown = (110-90)/110 ≈ 0.1818
    curve = [(0, 100.0), (1, 110.0), (2, 90.0), (3, 95.0)]
    dd = _compute_max_drawdown(curve)
    assert dd == pytest.approx((110.0 - 90.0) / 110.0, rel=1e-6)


def test_compute_max_drawdown_monotone_up() -> None:
    curve = [(i, float(100 + i)) for i in range(10)]
    assert _compute_max_drawdown(curve) == pytest.approx(0.0)


def test_compute_sharpe_uniform_returns_guarded() -> None:
    """Uniform returns have near-zero std; the guard must return 0.0, not inf."""
    returns = [0.01] * 100
    sharpe = _compute_sharpe(returns)
    # Floating-point accumulation may give eps variance; result must be finite and guarded
    assert math.isfinite(sharpe)
    assert sharpe == pytest.approx(0.0, abs=1e-6)


def test_compute_sharpe_mixed_returns() -> None:
    returns = [0.02, -0.01, 0.03, -0.005, 0.015] * 20
    sharpe = _compute_sharpe(returns)
    assert math.isfinite(sharpe)
    assert sharpe > 0  # positive mean return


def test_compute_sharpe_empty() -> None:
    assert _compute_sharpe([]) == 0.0
    assert _compute_sharpe([0.01]) == 0.0


# ---------------------------------------------------------------------------
# Cooldown enforcement (via risk engine integration)
# ---------------------------------------------------------------------------

def test_cooldown_prevents_reentry_after_loss() -> None:
    """After a loss, the risk engine must block new entries for N bars."""
    risk = RiskEngine(RiskConfig(cooldown_after_loss_bars=3))
    portfolio = _simple_portfolio()
    sig = _simple_signal()

    risk.on_loss()

    # Bars 1, 2, 3: blocked
    for _ in range(3):
        allowed, reason = risk.check_entry(sig, portfolio)
        assert allowed is False, "Should still be in cooldown"
        risk.tick()

    # Bar 4: allowed
    allowed, reason = risk.check_entry(sig, portfolio)
    assert allowed is True


def test_daily_drawdown_kill_switch_blocks_all_entries() -> None:
    """Once 2% daily drawdown is reached, no entries should be allowed."""
    risk = RiskEngine(RiskConfig(daily_drawdown_kill=0.02))
    portfolio = _simple_portfolio(100_000.0)
    portfolio._day_start_equity = 100_000.0
    portfolio.current_equity = 97_000.0   # 3% loss > 2% threshold

    allowed, reason = risk.check_entry(_simple_signal(), portfolio)
    assert allowed is False
    assert "daily_drawdown" in reason
