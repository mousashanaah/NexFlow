#!/usr/bin/env python3
"""Weekly Open Range Breakout backtest — Engine #3 candidate.

Specification:
  Range:  Monday's high/low define the weekly range.
  Entry:  Tue-Fri close ABOVE Monday's high → LONG at next bar open.
          Tue-Fri close BELOW Monday's low  → SHORT at next bar open.
          One position per week per symbol (first signal wins).
  Stop:   Long stop = Monday's low.  Short stop = Monday's high.
  Exit:   Stop hit intra-week OR Friday close (force-close) — whichever first.
  Size:   1% account equity risk per trade, sized on stop distance.
  Fees:   0.06% taker on entry and exit.

Kill criteria (auto-applied):
  PF < 1.10, max DD > 40%, n_trades < 60,
  OOS PF (2023+) < 0.85 × full-period PF.

Output: year-by-year table + summary stats + kill/proceed verdict + wealth score.

Usage:
  python scripts/backtest_weekly_range.py
  python scripts/backtest_weekly_range.py --symbols BTCUSDT --capital 50000
  python scripts/backtest_weekly_range.py --symbols BTCUSDT ETHUSDT --data data/candles
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.signal_models import Direction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TAKER_FEE    = 0.0006   # 0.06% per side
_RISK_PCT     = 0.01     # 1% account equity per trade

# Kill criteria
_KILL_PF          = 1.10
_KILL_MAX_DD      = 0.40
_KILL_MIN_TRADES  = 60
_KILL_OOS_RATIO   = 0.85   # OOS PF must be >= 0.85 × full-period PF
_OOS_START_YEAR   = 2023

# Weekday constants (UTC)
_MONDAY = 0
_FRIDAY = 4


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _WeekState:
    """Tracks Monday range and weekly signal state for one symbol."""
    mon_high:    float = 0.0
    mon_low:     float = 0.0
    has_range:   bool  = False   # True once Monday bar closes
    signal_fired: bool = False   # True once first signal triggered this week
    pending_dir:  Direction | None = None   # waiting to enter next bar open
    pending_stop: float = 0.0


@dataclass
class _Position:
    symbol:       str
    direction:    Direction
    entry_price:  float
    entry_time:   int        # epoch ms
    size:         float
    stop_price:   float      # Monday's opposite range boundary
    equity_at_entry: float
    fees_paid:    float


@dataclass
class _ClosedTrade:
    symbol:      str
    direction:   Direction
    entry_price: float
    exit_price:  float
    entry_time:  int
    exit_time:   int
    size:        float
    pnl:         float       # net of all fees
    fees:        float
    equity_at_entry: float
    r_multiple:  float


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_daily(symbol: str, data_dir: Path) -> list[Candle]:
    path = data_dir / f"{symbol}_1D.parquet"
    if not path.exists():
        print(f"[ERROR] Missing: {path}")
        print("  Run: python scripts/download_daily_candles.py")
        sys.exit(1)

    tbl = pq.read_table(path)
    df  = tbl.to_pydict()
    n   = len(df["open_time"])

    candles = []
    for i in range(n):
        candles.append(Candle(
            symbol      = df["symbol"][i],
            timeframe   = "1D",
            open_time   = df["open_time"][i],
            close_time  = df["close_time"][i],
            open        = df["open"][i],
            high        = df["high"][i],
            low         = df["low"][i],
            close       = df["close"][i],
            volume      = df["volume"][i],
            buy_volume  = 0.0,
            sell_volume = 0.0,
            trade_count = 0,
            vwap        = df["close"][i],
            spread_avg  = 0.0,
            spread_max  = 0.0,
            volatility_estimate = 0.0,
            is_final    = True,
        ))

    candles.sort(key=lambda c: c.open_time)
    return candles


def _weekday_utc(epoch_ms: int) -> int:
    """Return UTC weekday: Monday=0, ..., Friday=4, ..., Sunday=6."""
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).weekday()


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def _close_position(
    pos: _Position,
    exit_price: float,
    exit_time: int,
    equity: float,
) -> tuple[float, _ClosedTrade]:
    exit_fee   = exit_price * pos.size * _TAKER_FEE
    raw_pnl    = (exit_price - pos.entry_price) * pos.size * (
        1 if pos.direction is Direction.LONG else -1
    )
    net_pnl    = raw_pnl - exit_fee
    new_equity = equity + raw_pnl - exit_fee
    risk_amount = abs(pos.entry_price - pos.stop_price) * pos.size
    r_mult     = net_pnl / risk_amount if risk_amount > 0 else 0.0
    trade = _ClosedTrade(
        symbol          = pos.symbol,
        direction       = pos.direction,
        entry_price     = pos.entry_price,
        exit_price      = exit_price,
        entry_time      = pos.entry_time,
        exit_time       = exit_time,
        size            = pos.size,
        pnl             = net_pnl,
        fees            = pos.fees_paid + exit_fee,
        equity_at_entry = pos.equity_at_entry,
        r_multiple      = r_mult,
    )
    return new_equity, trade


def _run(
    candles_by_symbol: dict[str, list[Candle]],
    initial_equity: float,
) -> tuple[list[_ClosedTrade], list[tuple[int, float]]]:
    equity   = initial_equity
    positions: dict[str, _Position]   = {}
    week_states: dict[str, _WeekState] = {s: _WeekState() for s in candles_by_symbol}

    trades:   list[_ClosedTrade]      = []
    eq_curve: list[tuple[int, float]] = [(0, equity)]

    # Merge all symbols into one chronological stream
    all_bars: list[tuple[int, str, Candle]] = []
    for sym, candles in candles_by_symbol.items():
        for c in candles:
            all_bars.append((c.open_time, sym, c))
    all_bars.sort(key=lambda x: x[0])

    for bar_open_ms, symbol, candle in all_bars:
        ws  = week_states[symbol]
        wd  = _weekday_utc(candle.open_time)

        # ----------------------------------------------------------------
        # 1. Monday bar: reset weekly state, record Monday range at close
        # ----------------------------------------------------------------
        if wd == _MONDAY:
            # Force-close any carry-over position from previous week
            # (should not happen under normal flow, but safety net)
            if symbol in positions:
                equity, trade = _close_position(
                    positions.pop(symbol), candle.open, bar_open_ms, equity
                )
                trades.append(trade)
                eq_curve.append((bar_open_ms, equity))

            # Reset week state entirely
            ws.mon_high     = candle.high
            ws.mon_low      = candle.low
            ws.has_range    = True
            ws.signal_fired = False
            ws.pending_dir  = None
            ws.pending_stop = 0.0
            eq_curve.append((candle.close_time, equity))
            continue   # Monday itself is only for range definition; no entries

        # ----------------------------------------------------------------
        # 2. Tue–Fri: enter pending signal at this bar's open
        # ----------------------------------------------------------------
        if ws.pending_dir is not None and symbol not in positions:
            sig_dir   = ws.pending_dir
            stop_px   = ws.pending_stop
            ws.pending_dir  = None
            ws.pending_stop = 0.0

            entry_price = candle.open if candle.open > 0 else candle.close
            stop_dist   = abs(entry_price - stop_px)

            if stop_dist > 0:
                risk_amount = equity * _RISK_PCT
                size        = risk_amount / stop_dist

                entry_fee   = entry_price * size * _TAKER_FEE
                equity     -= entry_fee

                positions[symbol] = _Position(
                    symbol          = symbol,
                    direction       = sig_dir,
                    entry_price     = entry_price,
                    entry_time      = bar_open_ms,
                    size            = size,
                    stop_price      = stop_px,
                    equity_at_entry = equity,
                    fees_paid       = entry_fee,
                )
                eq_curve.append((bar_open_ms, equity))

        # ----------------------------------------------------------------
        # 3. Check stop for open position (intra-bar using high/low)
        # ----------------------------------------------------------------
        stop_hit_this_bar = False
        if symbol in positions:
            pos = positions[symbol]
            stop_hit = (
                (pos.direction is Direction.LONG  and candle.low  <= pos.stop_price)
                or
                (pos.direction is Direction.SHORT and candle.high >= pos.stop_price)
            )

            if stop_hit:
                exit_price = pos.stop_price
                equity, trade = _close_position(pos, exit_price, candle.close_time, equity)
                trades.append(trade)
                del positions[symbol]
                stop_hit_this_bar = True
                eq_curve.append((candle.close_time, equity))

        # ----------------------------------------------------------------
        # 4. Friday: force-close at Friday close if still open
        # ----------------------------------------------------------------
        if wd == _FRIDAY and symbol in positions and not stop_hit_this_bar:
            pos = positions[symbol]
            equity, trade = _close_position(pos, candle.close, candle.close_time, equity)
            trades.append(trade)
            del positions[symbol]
            eq_curve.append((candle.close_time, equity))

        # ----------------------------------------------------------------
        # 5. Check for breakout signal on Tue–Fri close (if no position)
        #    Only fire once per week (first signal wins)
        # ----------------------------------------------------------------
        if (
            ws.has_range
            and not ws.signal_fired
            and symbol not in positions
            and ws.pending_dir is None
            and wd != _MONDAY
            and wd != _FRIDAY   # Friday signal would have no entry day; skip
        ):
            if candle.close > ws.mon_high:
                ws.signal_fired = True
                ws.pending_dir  = Direction.LONG
                ws.pending_stop = ws.mon_low
            elif candle.close < ws.mon_low:
                ws.signal_fired = True
                ws.pending_dir  = Direction.SHORT
                ws.pending_stop = ws.mon_high

        # Also allow Thursday signal → Friday entry
        elif (
            ws.has_range
            and not ws.signal_fired
            and symbol not in positions
            and ws.pending_dir is None
            and wd == _FRIDAY - 1   # Thursday
        ):
            # Already covered above (Thursday is not MONDAY and not FRIDAY)
            pass

        eq_curve.append((candle.close_time, equity))

    # Force-close anything remaining at final bar
    for symbol, pos in list(positions.items()):
        last_candles = candles_by_symbol[symbol]
        if not last_candles:
            continue
        last = last_candles[-1]
        equity, trade = _close_position(pos, last.close, last.close_time, equity)
        trades.append(trade)

    eq_curve.append((all_bars[-1][0] if all_bars else 0, equity))
    return trades, eq_curve


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _year(epoch_ms: int) -> int:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).year


def _max_drawdown(eq_curve: list[tuple[int, float]]) -> float:
    peak   = eq_curve[0][1] if eq_curve else 0.0
    max_dd = 0.0
    for _, eq in eq_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _pf(trade_list: list[_ClosedTrade]) -> float:
    gp = sum(t.pnl for t in trade_list if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trade_list if t.pnl <= 0))
    return gp / gl if gl > 0 else float("inf")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(
    all_trades: list[_ClosedTrade],
    eq_curve:   list[tuple[int, float]],
    initial_equity: float,
    symbols:    list[str],
) -> None:
    if not all_trades:
        print("\n[RESULT] No trades generated. Check data or parameters.")
        return

    final_equity = eq_curve[-1][1] if eq_curve else initial_equity
    first_ms = min(t.entry_time for t in all_trades)
    last_ms  = max(t.exit_time  for t in all_trades)
    max_dd   = _max_drawdown(eq_curve)

    wins        = [t for t in all_trades if t.pnl > 0]
    losses      = [t for t in all_trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss   = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    years = (last_ms - first_ms) / (1000 * 86400 * 365.25)
    cagr  = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0.0

    n_trades    = len(all_trades)
    win_rate    = len(wins) / n_trades if n_trades else 0.0
    net_pnl     = final_equity - initial_equity
    total_fees  = sum(t.fees for t in all_trades)
    long_trades  = sum(1 for t in all_trades if t.direction is Direction.LONG)
    short_trades = sum(1 for t in all_trades if t.direction is Direction.SHORT)

    r_vals  = [t.r_multiple for t in all_trades]
    avg_r   = sum(r_vals) / len(r_vals)
    best_r  = max(r_vals)
    worst_r = min(r_vals)

    # OOS PF
    oos_trades  = [t for t in all_trades if _year(t.entry_time) >= _OOS_START_YEAR]
    oos_pf      = _pf(oos_trades) if oos_trades else 0.0

    # Wealth score
    wealth_score = profit_factor * math.sqrt(n_trades) * (1 - max_dd)

    years_seen = sorted({_year(t.entry_time) for t in all_trades})

    print("\n" + "=" * 70)
    print("  ENGINE #3 — WEEKLY OPEN RANGE BREAKOUT BACKTEST")
    print("=" * 70)
    print(f"  Symbols       : {', '.join(symbols)}")
    print(f"  Period        : {datetime.fromtimestamp(first_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}"
          f" → {datetime.fromtimestamp(last_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"  Initial equity: ${initial_equity:,.0f}")
    print(f"  Final equity  : ${final_equity:,.0f}")
    print(f"  Net PnL       : ${net_pnl:+,.0f}  ({net_pnl/initial_equity*100:+.1f}%)")
    print(f"  CAGR          : {cagr*100:.1f}%")
    print(f"  Max drawdown  : {max_dd*100:.1f}%")
    print(f"  Profit factor : {profit_factor:.2f}")
    print(f"  Total trades  : {n_trades}  (L:{long_trades}  S:{short_trades})")
    print(f"  Win rate      : {win_rate*100:.1f}%")
    print(f"  Avg R         : {avg_r:.2f}R")
    print(f"  Best R        : {best_r:.1f}R")
    print(f"  Worst R       : {worst_r:.1f}R")
    print(f"  Total fees    : ${total_fees:,.0f}")
    print(f"  OOS PF (≥{_OOS_START_YEAR}) : {oos_pf:.2f}  ({len(oos_trades)} trades)")
    print(f"  Wealth score  : {wealth_score:.2f}  [PF × √n × (1-maxDD)]")

    # Per-symbol breakdown
    print(f"\n{'─'*70}")
    print(f"  {'Symbol':<12} {'Trades':>7} {'WR%':>7} {'PF':>7} {'NetPnL':>12}")
    print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*7} {'─'*12}")
    for sym in symbols:
        sym_trades = [t for t in all_trades if t.symbol == sym]
        if not sym_trades:
            print(f"  {sym:<12} {'0':>7}")
            continue
        sym_wins = [t for t in sym_trades if t.pnl > 0]
        sym_gp   = sum(t.pnl for t in sym_wins)
        sym_gl   = abs(sum(t.pnl for t in sym_trades if t.pnl <= 0))
        sym_pf   = sym_gp / sym_gl if sym_gl > 0 else float("inf")
        sym_wr   = len(sym_wins) / len(sym_trades) * 100
        sym_pnl  = sum(t.pnl for t in sym_trades)
        print(f"  {sym:<12} {len(sym_trades):>7} {sym_wr:>6.1f}% {sym_pf:>7.2f} ${sym_pnl:>+10,.0f}")

    # Year-by-year
    print(f"\n{'─'*70}")
    print(f"  {'Year':<8} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>12}  {'Status'}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*12}  {'─'*10}")
    for yr in years_seen:
        yr_trades = [t for t in all_trades if _year(t.entry_time) == yr]
        if not yr_trades:
            continue
        yr_wins = [t for t in yr_trades if t.pnl > 0]
        yr_gp   = sum(t.pnl for t in yr_wins)
        yr_gl   = abs(sum(t.pnl for t in yr_trades if t.pnl <= 0))
        yr_pf   = yr_gp / yr_gl if yr_gl > 0 else float("inf")
        yr_wr   = len(yr_wins) / len(yr_trades) * 100
        yr_pnl  = sum(t.pnl for t in yr_trades)
        oos_tag = " [OOS]" if yr >= _OOS_START_YEAR else ""
        status  = "✓" if yr_pf >= 1.0 else "✗"
        print(f"  {yr:<8} {len(yr_trades):>7} {yr_wr:>6.1f}% {yr_pf:>7.2f} ${yr_pnl:>+10,.0f}  {status}{oos_tag}")

    # Top winners / losers
    sorted_by_r = sorted(all_trades, key=lambda t: t.r_multiple, reverse=True)
    print(f"\n{'─'*70}")
    print("  Top 5 winners by R:")
    for t in sorted_by_r[:5]:
        dt = datetime.fromtimestamp(t.entry_time/1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"    {t.symbol} {t.direction.value:<6} entry {dt}  {t.r_multiple:>+6.1f}R  ${t.pnl:>+,.0f}")
    print("  Top 5 losers by R:")
    for t in sorted_by_r[-5:]:
        dt = datetime.fromtimestamp(t.entry_time/1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"    {t.symbol} {t.direction.value:<6} entry {dt}  {t.r_multiple:>+6.1f}R  ${t.pnl:>+,.0f}")

    # Regime split
    early  = [t for t in all_trades if _year(t.entry_time) <= 2021]
    recent = [t for t in all_trades if _year(t.entry_time) >= 2022]
    print(f"\n{'─'*70}")
    print("  Regime split:")
    for label, subset in [("≤2021 (bull)", early), ("≥2022 (mixed)", recent)]:
        if not subset:
            print(f"    {label}: no trades")
            continue
        gp = sum(t.pnl for t in subset if t.pnl > 0)
        gl = abs(sum(t.pnl for t in subset if t.pnl <= 0))
        pf = gp / gl if gl > 0 else float("inf")
        print(f"    {label}: {len(subset)} trades, PF {pf:.2f}")

    # ----------------------------------------------------------------
    # VERDICT
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")

    kill_reasons = []

    if profit_factor < _KILL_PF:
        kill_reasons.append(
            f"Profit factor {profit_factor:.2f} < kill threshold {_KILL_PF}"
        )
    if max_dd > _KILL_MAX_DD:
        kill_reasons.append(
            f"Max drawdown {max_dd*100:.1f}% > kill threshold {_KILL_MAX_DD*100:.0f}%"
        )
    if n_trades < _KILL_MIN_TRADES:
        kill_reasons.append(
            f"Trade count {n_trades} < minimum {_KILL_MIN_TRADES}"
        )
    if oos_trades:
        oos_threshold = _KILL_OOS_RATIO * profit_factor
        if oos_pf < oos_threshold:
            kill_reasons.append(
                f"OOS PF {oos_pf:.2f} < {_KILL_OOS_RATIO:.0%} × full-period PF "
                f"({oos_threshold:.2f}) — likely overfitted"
            )
    else:
        kill_reasons.append(
            f"No OOS trades found (need data from {_OOS_START_YEAR}+)"
        )

    if kill_reasons:
        print("\n  KILL — Strategy fails one or more kill criteria:\n")
        for r in kill_reasons:
            print(f"     * {r}")
        print("\n  Action: do NOT deploy. Move to next hypothesis.")
    else:
        print("\n  PROCEED — All kill criteria passed.\n")
        print(f"     * PF {profit_factor:.2f} >= {_KILL_PF}")
        print(f"     * Max DD {max_dd*100:.1f}% <= {_KILL_MAX_DD*100:.0f}%")
        print(f"     * Trades {n_trades} >= {_KILL_MIN_TRADES}")
        print(f"     * OOS PF {oos_pf:.2f} >= {_KILL_OOS_RATIO:.0%} × full-period PF")
        print("\n  Action: proceed to paper deployment / Engine #3 review.")

    print(f"\n  Wealth score  : {wealth_score:.2f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly Open Range Breakout backtest")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--data",    default="data/candles",
                        help="Directory containing *_1D.parquet files")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Starting equity in USDT (default 100000)")
    args = parser.parse_args()

    data_dir = _REPO_ROOT / args.data

    print(f"\nLoading data from {data_dir} ...")
    candles_by_symbol: dict[str, list[Candle]] = {}
    for sym in args.symbols:
        candles = _load_daily(sym, data_dir)
        candles_by_symbol[sym] = candles
        span_start = datetime.fromtimestamp(
            candles[0].open_time / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d")
        span_end = datetime.fromtimestamp(
            candles[-1].open_time / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d")
        print(f"  {sym}: {len(candles)} daily bars  ({span_start} → {span_end})")

    print("\nRunning backtest ...")
    trades, eq_curve = _run(candles_by_symbol, args.capital)

    _print_report(trades, eq_curve, args.capital, args.symbols)


if __name__ == "__main__":
    main()
