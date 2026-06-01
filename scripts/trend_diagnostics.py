#!/usr/bin/env python3
"""Deep diagnostics for Engine #1 daily trend following results.

Answers four questions:
  1. Profit concentration: how much of total profit comes from top N trades?
  2. R-multiple distribution: where do the exits actually land?
  3. Exit truncation: does the trailing stop cut winners early?
     — Runs the same backtest with trail multipliers [2.0, 2.5, 3.0, 4.0]
     — Reports impact on PF, CAGR, max DD, and winner magnitude
  4. Capital utilization: what % of days was capital actually deployed?

This script does NOT modify parameters. It diagnoses the existing spec.

Usage:
  python scripts/trend_diagnostics.py
  python scripts/trend_diagnostics.py --equity 10000
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import argparse

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required")
    sys.exit(1)

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.signal_models import Direction
from nexflow.services.strategy.trend_strategy import TrendFollowingStrategy, _wilder_atr

# Re-import the backtest engine internals
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from run_trend_backtest import (
    _run, _load_daily, _max_drawdown, _metrics, _ClosedTrade,
    _LOOKBACK, _ATR_PERIOD, _INIT_MULT, _TRAIL_MULT,
    _TAKER_FEE, _RISK_PCT, _MAX_RISK_PCT,
)

_DAY_MS = 86_400_000


# ---------------------------------------------------------------------------
# Run with custom trail multiplier
# ---------------------------------------------------------------------------

def _run_with_trail(
    candles_by_symbol: dict[str, list[Candle]],
    initial_equity: float,
    trail_mult: float,
) -> tuple[list[_ClosedTrade], list[tuple[int, float]]]:
    """Same as _run() but with a custom trail multiplier."""
    from dataclasses import dataclass, field as dc_field

    equity    = initial_equity
    positions: dict = {}
    pending:   dict = {}
    from run_trend_backtest import _BarWindow
    windows   = {s: _BarWindow() for s in candles_by_symbol}
    trades: list[_ClosedTrade] = []
    eq_curve: list[tuple[int, float]] = [(0, equity)]

    all_bars = []
    for sym, candles in candles_by_symbol.items():
        for c in candles:
            all_bars.append((c.open_time, sym, c))
    all_bars.sort(key=lambda x: x[0])

    from run_trend_backtest import _Position

    for bar_open_ms, symbol, candle in all_bars:
        w = windows[symbol]

        if symbol in pending:
            sig_dir, sig_atr, _ = pending.pop(symbol)
            entry_price = candle.open or candle.close

            initial_stop = (
                entry_price - _INIT_MULT * sig_atr if sig_dir is Direction.LONG
                else entry_price + _INIT_MULT * sig_atr
            )
            stop_dist = abs(entry_price - initial_stop)

            if symbol in positions:
                old = positions.pop(symbol)
                exit_price = entry_price
                exit_fee   = exit_price * old.size * _TAKER_FEE
                raw_pnl    = (exit_price - old.entry_price) * old.size * (
                    1 if old.direction is Direction.LONG else -1)
                net_pnl = raw_pnl - exit_fee
                equity += raw_pnl - exit_fee
                risk_amount = abs(old.entry_price - old.initial_stop) * old.size
                trades.append(_ClosedTrade(
                    symbol=symbol, direction=old.direction,
                    entry_price=old.entry_price, exit_price=exit_price,
                    entry_time=old.entry_time, exit_time=bar_open_ms,
                    size=old.size, pnl=net_pnl,
                    fees=old.fees_paid + exit_fee,
                    equity_at_entry=old.equity_at_entry,
                    r_multiple=net_pnl / risk_amount if risk_amount > 0 else 0,
                ))

            if stop_dist <= 0:
                continue
            risk_amount = equity * _RISK_PCT
            size = risk_amount / stop_dist
            entry_fee = entry_price * size * _TAKER_FEE
            equity -= entry_fee

            positions[symbol] = _Position(
                symbol=symbol, direction=sig_dir,
                entry_price=entry_price, entry_time=bar_open_ms,
                size=size, initial_stop=initial_stop,
                trail_stop=initial_stop, best_close=entry_price,
                equity_at_entry=equity, fees_paid=entry_fee,
            )

        if symbol in positions:
            pos = positions[symbol]
            stop_hit = (
                (pos.direction is Direction.LONG  and candle.low  <= pos.trail_stop) or
                (pos.direction is Direction.SHORT and candle.high >= pos.trail_stop)
            )
            if stop_hit:
                exit_price = pos.trail_stop
                exit_fee   = exit_price * pos.size * _TAKER_FEE
                raw_pnl    = (exit_price - pos.entry_price) * pos.size * (
                    1 if pos.direction is Direction.LONG else -1)
                net_pnl = raw_pnl - exit_fee
                equity += raw_pnl - exit_fee
                risk_amount = abs(pos.entry_price - pos.initial_stop) * pos.size
                trades.append(_ClosedTrade(
                    symbol=symbol, direction=pos.direction,
                    entry_price=pos.entry_price, exit_price=exit_price,
                    entry_time=pos.entry_time, exit_time=candle.close_time,
                    size=pos.size, pnl=net_pnl,
                    fees=pos.fees_paid + exit_fee,
                    equity_at_entry=pos.equity_at_entry,
                    r_multiple=net_pnl / risk_amount if risk_amount > 0 else 0,
                ))
                del positions[symbol]

        w.append(candle.close, candle.high, candle.low)

        if symbol in positions:
            pos = positions[symbol]
            atr = _wilder_atr(w.highs, w.lows, w.closes, _ATR_PERIOD)
            if atr and atr > 0:
                new_trail, new_best = TrendFollowingStrategy.update_trail(
                    current_trail=pos.trail_stop, direction=pos.direction,
                    bar_close=candle.close, best_close=pos.best_close,
                    atr=atr, trail_mult=trail_mult,
                )
                pos.trail_stop = new_trail
                pos.best_close = new_best

        if len(w.closes) >= _LOOKBACK + 1:
            atr = _wilder_atr(w.highs, w.lows, w.closes, _ATR_PERIOD)
            if atr and atr > 0:
                prior_closes = w.closes[-(_LOOKBACK + 1):-1]
                highest = max(prior_closes)
                lowest  = min(prior_closes)
                close   = candle.close
                current_dir = positions[symbol].direction if symbol in positions else None
                if close > highest and current_dir is not Direction.LONG:
                    pending[symbol] = (Direction.LONG, atr, candle.close_time)
                elif close < lowest and current_dir is not Direction.SHORT:
                    pending[symbol] = (Direction.SHORT, atr, candle.close_time)

        eq_curve.append((candle.close_time, equity))

    for symbol, pos in list(positions.items()):
        last = candles_by_symbol[symbol][-1]
        exit_price = last.close
        exit_fee   = exit_price * pos.size * _TAKER_FEE
        raw_pnl    = (exit_price - pos.entry_price) * pos.size * (
            1 if pos.direction is Direction.LONG else -1)
        net_pnl = raw_pnl - exit_fee
        equity += raw_pnl - exit_fee
        risk_amount = abs(pos.entry_price - pos.initial_stop) * pos.size
        trades.append(_ClosedTrade(
            symbol=symbol, direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_price,
            entry_time=pos.entry_time, exit_time=last.close_time,
            size=pos.size, pnl=net_pnl,
            fees=pos.fees_paid + exit_fee,
            equity_at_entry=pos.equity_at_entry,
            r_multiple=net_pnl / risk_amount if risk_amount > 0 else 0,
        ))

    eq_curve.append((all_bars[-1][0] if all_bars else 0, equity))
    return trades, eq_curve


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def _profit_concentration(trades: list[_ClosedTrade]) -> None:
    winners = sorted([t for t in trades if t.pnl > 0], key=lambda t: t.pnl, reverse=True)
    total_gross = sum(t.pnl for t in winners)
    total_net   = sum(t.pnl for t in trades)

    print("\n── PROFIT CONCENTRATION ─────────────────────────────────────────")
    print(f"  Total gross profit : ${total_gross:>10,.0f}")
    print(f"  Total net PnL      : ${total_net:>10,.0f}")
    print(f"  Winning trades     : {len(winners)} of {len(trades)}")
    print()
    print(f"  {'Top N':>8}  {'Trades':>7}  {'Gross $':>12}  {'% of Gross':>12}  {'% of Net':>10}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*12}  {'─'*12}  {'─'*10}")

    for n in [1, 3, 5, 10, 20]:
        subset = winners[:n]
        if not subset:
            continue
        gross_n = sum(t.pnl for t in subset)
        pct_gross = gross_n / total_gross * 100 if total_gross > 0 else 0
        pct_net   = gross_n / total_net   * 100 if total_net   > 0 else 0
        print(f"  {n:>8}  {len(subset):>7}  ${gross_n:>10,.0f}  {pct_gross:>10.1f}%  {pct_net:>8.1f}%")

    print()
    print("  Top 10 individual winners:")
    print(f"  {'#':>4}  {'Symbol':<10} {'Dir':<6} {'Entry':>12} {'R':>7}  {'PnL':>12}  {'Entry date'}")
    print(f"  {'─'*4}  {'─'*10} {'─'*6} {'─'*12} {'─'*7}  {'─'*12}  {'─'*12}")
    for i, t in enumerate(winners[:10], 1):
        dt = datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  {i:>4}  {t.symbol:<10} {t.direction.value:<6} "
              f"${t.entry_price:>10,.2f} {t.r_multiple:>+6.1f}R  ${t.pnl:>+10,.0f}  {dt}")


def _r_distribution(trades: list[_ClosedTrade]) -> None:
    r_vals = [t.r_multiple for t in trades]
    buckets = [
        ("< -0.8R",   lambda r: r < -0.8),
        ("-0.8 to 0R",lambda r: -0.8 <= r < 0),
        ("0 to 1R",   lambda r: 0 <= r < 1),
        ("1R to 2R",  lambda r: 1 <= r < 2),
        ("2R to 3R",  lambda r: 2 <= r < 3),
        ("3R to 5R",  lambda r: 3 <= r < 5),
        ("5R to 10R", lambda r: 5 <= r < 10),
        ("10R to 15R",lambda r: 10 <= r < 15),
        ("> 15R",     lambda r: r >= 15),
    ]

    print("\n── R-MULTIPLE DISTRIBUTION ──────────────────────────────────────")
    print(f"  {'Bucket':<14}  {'Count':>6}  {'%':>7}  {'Cum %':>7}")
    print(f"  {'─'*14}  {'─'*6}  {'─'*7}  {'─'*7}")
    cumulative = 0
    for label, fn in buckets:
        count = sum(1 for r in r_vals if fn(r))
        pct   = count / len(r_vals) * 100 if r_vals else 0
        cumulative += pct
        print(f"  {label:<14}  {count:>6}  {pct:>6.1f}%  {cumulative:>6.1f}%")

    thresholds = [1, 2, 3, 5, 7.5, 10, 15]
    print()
    print("  Trades exceeding R threshold:")
    for thr in thresholds:
        count = sum(1 for r in r_vals if r > thr)
        pct   = count / len(r_vals) * 100 if r_vals else 0
        gross = sum(t.pnl for t in trades if t.r_multiple > thr)
        total_gross = sum(t.pnl for t in trades if t.pnl > 0)
        pct_profit  = gross / total_gross * 100 if total_gross > 0 else 0
        print(f"    > {thr:>5.1f}R : {count:>4} trades ({pct:>4.1f}%)  "
              f"${gross:>+10,.0f}  ({pct_profit:.1f}% of gross profit)")


def _trail_comparison(
    candles_by_symbol: dict[str, list[Candle]],
    initial_equity: float,
) -> None:
    trail_mults = [2.0, 2.5, 3.0, 4.0]

    print("\n── TRAILING STOP COMPARISON ─────────────────────────────────────")
    print("  (Diagnostic only — not a recommendation to change parameters)")
    print()
    print(f"  {'Trail':>8}  {'Trades':>7}  {'PF':>7}  {'CAGR':>7}  "
          f"{'MaxDD':>7}  {'Best R':>8}  {'Avg R':>8}  {'>5R':>5}  {'>10R':>5}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  "
          f"{'─'*7}  {'─'*8}  {'─'*8}  {'─'*5}  {'─'*5}")

    for tm in trail_mults:
        t_trades, t_eq = _run_with_trail(candles_by_symbol, initial_equity, tm)
        if not t_trades:
            continue
        first_ms = min(t.entry_time for t in t_trades)
        last_ms  = max(t.exit_time  for t in t_trades)
        final_eq = t_eq[-1][1]
        m = _metrics(t_trades, initial_equity, final_eq, first_ms, last_ms)
        dd = _max_drawdown(t_eq)
        best_r = max(t.r_multiple for t in t_trades)
        avg_r  = sum(t.r_multiple for t in t_trades) / len(t_trades)
        gt5  = sum(1 for t in t_trades if t.r_multiple > 5)
        gt10 = sum(1 for t in t_trades if t.r_multiple > 10)
        marker = " ← current" if abs(tm - _TRAIL_MULT) < 0.01 else ""
        print(f"  {tm:>7.1f}×  {m['total_trades']:>7}  {m['profit_factor']:>7.2f}  "
              f"{m['cagr']*100:>6.1f}%  {dd*100:>6.1f}%  "
              f"{best_r:>+8.1f}R  {avg_r:>+8.2f}R  {gt5:>5}  {gt10:>5}{marker}")


def _utilization(
    trades: list[_ClosedTrade],
    candles_by_symbol: dict[str, list[Candle]],
) -> None:
    if not trades:
        return

    symbols = list(candles_by_symbol.keys())
    first_bar = min(c.open_time for cs in candles_by_symbol.values() for c in cs)
    last_bar  = max(c.open_time for cs in candles_by_symbol.values() for c in cs)
    total_days = (last_bar - first_bar) / _DAY_MS

    print("\n── CAPITAL UTILIZATION ──────────────────────────────────────────")
    print(f"  Total calendar days in study: {total_days:.0f}")
    print()

    total_deployed_days = 0.0
    for sym in symbols:
        sym_trades = [t for t in trades if t.symbol == sym]
        deployed_ms = sum(t.exit_time - t.entry_time for t in sym_trades)
        deployed_days = deployed_ms / _DAY_MS
        total_deployed_days += deployed_days
        pct = deployed_days / total_days * 100 if total_days > 0 else 0
        avg_hold = deployed_ms / len(sym_trades) / _DAY_MS if sym_trades else 0
        print(f"  {sym:<12}  deployed {deployed_days:>5.0f} days  "
              f"({pct:>4.1f}% of period)  avg hold {avg_hold:.0f} days/trade")

    # Combined (both symbols deployed simultaneously counts double)
    combined_pct = total_deployed_days / (total_days * len(symbols)) * 100
    print(f"\n  Combined utilization: {combined_pct:.1f}% of available symbol-days")
    print(f"  Idle capital: {100 - combined_pct:.1f}% of available symbol-days")

    # Hold time distribution
    hold_days = [(t.exit_time - t.entry_time) / _DAY_MS for t in trades]
    hold_days.sort()
    buckets = [(1, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 999)]
    print()
    print(f"  Hold time distribution:")
    print(f"  {'Range (days)':<18}  {'Count':>6}  {'%':>7}")
    print(f"  {'─'*18}  {'─'*6}  {'─'*7}")
    for lo, hi in buckets:
        count = sum(1 for h in hold_days if lo <= h < hi)
        pct   = count / len(hold_days) * 100 if hold_days else 0
        label = f"{lo}–{hi} days" if hi < 999 else f">{lo} days"
        print(f"  {label:<18}  {count:>6}  {pct:>6.1f}%")

    # Impact of idle capital on CAGR
    print()
    print("  CAGR if utilization were doubled (theoretical ceiling):")
    print("  — Not achievable by changing parameters, only by adding a second")
    print("    uncorrelated strategy (the Expansion Engine).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",  nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--equity",   type=float, default=100_000.0)
    parser.add_argument("--data-dir", default="data/candles")
    args = parser.parse_args()

    data_dir = _REPO_ROOT / args.data_dir
    print("\nLoading data ...")
    candles_by_symbol: dict[str, list[Candle]] = {}
    for sym in args.symbols:
        candles_by_symbol[sym] = _load_daily(sym, data_dir)
        print(f"  {sym}: {len(candles_by_symbol[sym])} bars")

    print("Running base backtest ...")
    trades, eq_curve = _run(candles_by_symbol, args.equity)

    if not trades:
        print("[ERROR] No trades generated.")
        sys.exit(1)

    first_ms = min(t.entry_time for t in trades)
    last_ms  = max(t.exit_time  for t in trades)
    final_eq = eq_curve[-1][1]
    m = _metrics(trades, args.equity, final_eq, first_ms, last_ms)
    dd = _max_drawdown(eq_curve)

    print(f"\n{'='*65}")
    print("  ENGINE #1 DIAGNOSTICS")
    print(f"{'='*65}")
    print(f"  Trades: {m['total_trades']}  |  PF: {m['profit_factor']:.2f}  |  "
          f"CAGR: {m['cagr']*100:.1f}%  |  Max DD: {dd*100:.1f}%")

    _profit_concentration(trades)
    _r_distribution(trades)
    _trail_comparison(candles_by_symbol, args.equity)
    _utilization(trades, candles_by_symbol)

    # Deployment recommendation
    print(f"\n{'='*65}")
    print("  DEPLOYMENT RECOMMENDATION")
    print(f"{'='*65}")
    pf  = m['profit_factor']
    cagr = m['cagr']
    concentration = sum(t.pnl for t in sorted(trades, key=lambda t: t.pnl, reverse=True)[:10])
    total_net = sum(t.pnl for t in trades)
    top10_pct = concentration / total_net * 100 if total_net > 0 else 0

    if pf >= 1.3 and cagr >= 0.20 and dd <= 0.20:
        print("\n  ✅  Strong case for paper deployment.")
    elif pf >= 1.2 and top10_pct > 80:
        print("\n  ✅  Genuine trend-following behavior confirmed.")
        print(f"      Top 10 trades = {top10_pct:.0f}% of net PnL — expected for this strategy type.")
        print("      Proceed to paper deployment.")
    else:
        print("\n  ⚠️   Review diagnostics above before deploying.")

    print("=" * 65)


if __name__ == "__main__":
    main()
