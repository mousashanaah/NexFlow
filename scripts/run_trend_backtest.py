#!/usr/bin/env python3
"""Daily trend-following backtest — Engine #1.

Specification implemented exactly:
  Entry:   Daily close > highest close of prior 20 bars  → LONG (next bar open)
           Daily close < lowest close of prior 20 bars   → SHORT (next bar open)
  Stop:    2.5 × ATR(14) from ENTRY price (not signal bar close)
  Trail:   max(current_trail, highest_close_since_entry − 2.0×ATR)  [long]
           min(current_trail, lowest_close_since_entry  + 2.0×ATR)  [short]
           Updates at each daily close AFTER stop check.
  Exit:    Trail stop hit (bar low ≤ trail for long; bar high ≥ trail for short)
           Executed at trail stop price.
  Size:    1% account risk per trade; skip if 1 contract > 3% risk.
  Symbols: BTCUSDT and ETHUSDT (independent positions, 1 per symbol)
  Fees:    TAKER 0.06% on entry and exit (Bitget USDT-Futures)

Output: year-by-year table + aggregate metrics + kill/go verdict.

Usage:
  python scripts/run_trend_backtest.py
  python scripts/run_trend_backtest.py --equity 10000
  python scripts/run_trend_backtest.py --symbols BTCUSDT
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
from nexflow.services.strategy.trend_strategy import TrendFollowingStrategy, _wilder_atr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TAKER_FEE   = 0.0006    # 0.06% per side
_RISK_PCT    = 0.01       # 1% account equity per trade
_MAX_RISK_PCT= 0.03       # skip trade if 1 contract > 3% risk
_LOOKBACK    = 20
_ATR_PERIOD  = 14
_INIT_MULT   = 2.5
_TRAIL_MULT  = 2.0

# Kill criteria (applied to full-period combined result)
_KILL_PF_COMBINED = 1.10
_KILL_PF_SYMBOL   = 0.85
_KILL_MAX_DD      = 0.50
_KILL_MIN_TRADES  = 60

# Go criteria
_GO_PF          = 1.30
_GO_CAGR        = 0.20
_GO_MAX_DD      = 0.40


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    symbol:       str
    direction:    Direction
    entry_price:  float
    entry_time:   int        # epoch ms
    size:         float      # contracts (BTC or ETH)
    initial_stop: float
    trail_stop:   float
    best_close:   float      # highest close (long) or lowest close (short) since entry
    equity_at_entry: float
    fees_paid:    float      # entry fee, already deducted from equity


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
    r_multiple:  float       # pnl / (initial_risk_per_unit × size)


@dataclass
class _BarWindow:
    """Per-symbol rolling window for ATR and signal computation."""
    closes: list[float] = field(default_factory=list)
    highs:  list[float] = field(default_factory=list)
    lows:   list[float] = field(default_factory=list)

    def append(self, c: float, h: float, lo: float) -> None:
        self.closes.append(c)
        self.highs.append(h)
        self.lows.append(lo)


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
            symbol    = df["symbol"][i],
            timeframe = "1D",
            open_time = df["open_time"][i],
            close_time= df["close_time"][i],
            open      = df["open"][i],
            high      = df["high"][i],
            low       = df["low"][i],
            close     = df["close"][i],
            volume    = df["volume"][i],
            buy_volume= 0.0,
            sell_volume=0.0,
            trade_count=0,
            vwap      = df["close"][i],
            spread_avg= 0.0,
            spread_max= 0.0,
            volatility_estimate=0.0,
            is_final  = True,
        ))

    candles.sort(key=lambda c: c.open_time)
    return candles


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def _run(
    candles_by_symbol: dict[str, list[Candle]],
    initial_equity: float,
) -> tuple[list[_ClosedTrade], list[tuple[int, float]]]:
    """
    Returns (closed_trades, equity_curve).
    equity_curve: list of (epoch_ms, equity) at each bar boundary.
    """
    equity   = initial_equity
    positions: dict[str, _Position]      = {}
    pending:   dict[str, tuple[Direction, float, int]] = {}
    # pending: symbol → (direction, signal_atr, signal_time)

    windows:  dict[str, _BarWindow]      = {s: _BarWindow() for s in candles_by_symbol}
    trades:   list[_ClosedTrade]         = []
    eq_curve: list[tuple[int, float]]    = [(0, equity)]

    # Merge all symbols into one chronological stream
    all_bars: list[tuple[int, str, Candle]] = []
    for sym, candles in candles_by_symbol.items():
        for c in candles:
            all_bars.append((c.open_time, sym, c))
    all_bars.sort(key=lambda x: x[0])

    for bar_open_ms, symbol, candle in all_bars:
        w = windows[symbol]

        # ----------------------------------------------------------------
        # 1. Enter pending signal at this bar's open
        # ----------------------------------------------------------------
        if symbol in pending:
            sig_dir, sig_atr, _ = pending.pop(symbol)

            entry_price = candle.open
            if entry_price <= 0:
                entry_price = candle.close  # fallback if open missing

            initial_stop = (
                entry_price - _INIT_MULT * sig_atr
                if sig_dir is Direction.LONG
                else entry_price + _INIT_MULT * sig_atr
            )
            stop_dist = abs(entry_price - initial_stop)

            # Close existing position first (direction flip)
            if symbol in positions:
                old = positions.pop(symbol)
                exit_fee  = old.entry_price * 0.0  # already has entry fee; exit handled below
                # actually, close at this bar's open price
                exit_price = entry_price
                exit_fee   = exit_price * old.size * _TAKER_FEE
                raw_pnl    = (exit_price - old.entry_price) * old.size * (
                    1 if old.direction is Direction.LONG else -1
                )
                net_pnl = raw_pnl - exit_fee  # entry fee already charged at open
                equity += raw_pnl - exit_fee
                risk_amount = abs(old.entry_price - old.initial_stop) * old.size
                r_mult = net_pnl / risk_amount if risk_amount > 0 else 0.0
                trades.append(_ClosedTrade(
                    symbol      = symbol,
                    direction   = old.direction,
                    entry_price = old.entry_price,
                    exit_price  = exit_price,
                    entry_time  = old.entry_time,
                    exit_time   = bar_open_ms,
                    size        = old.size,
                    pnl         = net_pnl,
                    fees        = old.fees_paid + exit_fee,
                    equity_at_entry = old.equity_at_entry,
                    r_multiple  = r_mult,
                ))

            if stop_dist <= 0:
                continue

            risk_amount   = equity * _RISK_PCT
            size          = risk_amount / stop_dist  # crypto contracts (fractional ok for backtest)

            # Safety: skip if minimum meaningful position > 3% risk
            min_size = 0.001  # 0.001 BTC / ETH minimum
            actual_risk_frac = (stop_dist * max(size, min_size)) / equity
            if actual_risk_frac > _MAX_RISK_PCT and size < min_size:
                continue

            entry_fee = entry_price * size * _TAKER_FEE
            equity   -= entry_fee

            positions[symbol] = _Position(
                symbol          = symbol,
                direction       = sig_dir,
                entry_price     = entry_price,
                entry_time      = bar_open_ms,
                size            = size,
                initial_stop    = initial_stop,
                trail_stop      = initial_stop,
                best_close      = entry_price,
                equity_at_entry = equity,
                fees_paid       = entry_fee,
            )

        # ----------------------------------------------------------------
        # 2. Evaluate stop for open position (using bar high/low)
        # ----------------------------------------------------------------
        if symbol in positions:
            pos = positions[symbol]
            stop_hit = (
                (pos.direction is Direction.LONG  and candle.low  <= pos.trail_stop)
                or
                (pos.direction is Direction.SHORT and candle.high >= pos.trail_stop)
            )

            if stop_hit:
                exit_price = pos.trail_stop
                exit_fee   = exit_price * pos.size * _TAKER_FEE
                raw_pnl    = (exit_price - pos.entry_price) * pos.size * (
                    1 if pos.direction is Direction.LONG else -1
                )
                net_pnl  = raw_pnl - exit_fee
                equity  += raw_pnl - exit_fee
                risk_amount = abs(pos.entry_price - pos.initial_stop) * pos.size
                r_mult = net_pnl / risk_amount if risk_amount > 0 else 0.0
                trades.append(_ClosedTrade(
                    symbol      = symbol,
                    direction   = pos.direction,
                    entry_price = pos.entry_price,
                    exit_price  = exit_price,
                    entry_time  = pos.entry_time,
                    exit_time   = candle.close_time,
                    size        = pos.size,
                    pnl         = net_pnl,
                    fees        = pos.fees_paid + exit_fee,
                    equity_at_entry = pos.equity_at_entry,
                    r_multiple  = r_mult,
                ))
                del positions[symbol]

        # ----------------------------------------------------------------
        # 3. Update bar window
        # ----------------------------------------------------------------
        w.append(candle.close, candle.high, candle.low)

        # ----------------------------------------------------------------
        # 4. Update trailing stop for still-open position (uses bar close)
        # ----------------------------------------------------------------
        if symbol in positions:
            pos = positions[symbol]
            atr = _wilder_atr(w.highs, w.lows, w.closes, _ATR_PERIOD)
            if atr and atr > 0:
                new_trail, new_best = TrendFollowingStrategy.update_trail(
                    current_trail = pos.trail_stop,
                    direction     = pos.direction,
                    bar_close     = candle.close,
                    best_close    = pos.best_close,
                    atr           = atr,
                    trail_mult    = _TRAIL_MULT,
                )
                pos.trail_stop  = new_trail
                pos.best_close  = new_best

        # ----------------------------------------------------------------
        # 5. Check for new signal (breakout of prior-20-bar close range)
        # ----------------------------------------------------------------
        if len(w.closes) >= _LOOKBACK + 1:
            atr = _wilder_atr(w.highs, w.lows, w.closes, _ATR_PERIOD)
            if atr and atr > 0:
                prior_closes = w.closes[-(  _LOOKBACK + 1):-1]
                highest = max(prior_closes)
                lowest  = min(prior_closes)
                close   = candle.close

                current_dir = positions[symbol].direction if symbol in positions else None

                if close > highest and current_dir is not Direction.LONG:
                    pending[symbol] = (Direction.LONG, atr, candle.close_time)
                elif close < lowest and current_dir is not Direction.SHORT:
                    pending[symbol] = (Direction.SHORT, atr, candle.close_time)

        eq_curve.append((candle.close_time, equity))

    # Force-close any remaining positions at last bar close
    for symbol, pos in list(positions.items()):
        last_candles = candles_by_symbol[symbol]
        if not last_candles:
            continue
        last = last_candles[-1]
        exit_price = last.close
        exit_fee   = exit_price * pos.size * _TAKER_FEE
        raw_pnl    = (exit_price - pos.entry_price) * pos.size * (
            1 if pos.direction is Direction.LONG else -1
        )
        net_pnl  = raw_pnl - exit_fee
        equity  += raw_pnl - exit_fee
        risk_amount = abs(pos.entry_price - pos.initial_stop) * pos.size
        r_mult = net_pnl / risk_amount if risk_amount > 0 else 0.0
        trades.append(_ClosedTrade(
            symbol      = symbol,
            direction   = pos.direction,
            entry_price = pos.entry_price,
            exit_price  = exit_price,
            entry_time  = pos.entry_time,
            exit_time   = last.close_time,
            size        = pos.size,
            pnl         = net_pnl,
            fees        = pos.fees_paid + exit_fee,
            equity_at_entry = pos.equity_at_entry,
            r_multiple  = r_mult,
        ))

    eq_curve.append((all_bars[-1][0] if all_bars else 0, equity))
    return trades, eq_curve


# ---------------------------------------------------------------------------
# Metrics and reporting
# ---------------------------------------------------------------------------

def _year(epoch_ms: int) -> int:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).year


def _metrics(trades: list[_ClosedTrade], initial_equity: float, final_equity: float,
             first_date: int, last_date: int) -> dict:
    if not trades:
        return {}

    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss   = abs(sum(t.pnl for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    years = (last_date - first_date) / (1000 * 86400 * 365.25)
    cagr  = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0.0

    avg_r = sum(t.r_multiple for t in trades) / len(trades)
    best_r  = max(t.r_multiple for t in trades)
    worst_r = min(t.r_multiple for t in trades)

    return {
        "total_trades":  len(trades),
        "win_rate":      len(wins) / len(trades),
        "profit_factor": pf,
        "cagr":          cagr,
        "net_pnl":       final_equity - initial_equity,
        "total_fees":    sum(t.fees for t in trades),
        "avg_r":         avg_r,
        "best_r":        best_r,
        "worst_r":       worst_r,
        "long_trades":   sum(1 for t in trades if t.direction is Direction.LONG),
        "short_trades":  sum(1 for t in trades if t.direction is Direction.SHORT),
    }


def _max_drawdown(eq_curve: list[tuple[int, float]]) -> float:
    peak = eq_curve[0][1] if eq_curve else 0.0
    max_dd = 0.0
    for _, eq in eq_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _print_report(
    all_trades: list[_ClosedTrade],
    eq_curve: list[tuple[int, float]],
    initial_equity: float,
    symbols: list[str],
) -> None:
    if not all_trades:
        print("\n[RESULT] No trades generated. Check data or parameters.")
        return

    final_equity = eq_curve[-1][1] if eq_curve else initial_equity
    first_ms = min(t.entry_time for t in all_trades)
    last_ms  = max(t.exit_time  for t in all_trades)
    max_dd   = _max_drawdown(eq_curve)

    m = _metrics(all_trades, initial_equity, final_equity, first_ms, last_ms)

    # Year-by-year breakdown
    years_seen = sorted({_year(t.entry_time) for t in all_trades})
    print("\n" + "=" * 70)
    print("  ENGINE #1 — DAILY TREND FOLLOWING BACKTEST")
    print("=" * 70)
    print(f"  Symbols       : {', '.join(symbols)}")
    print(f"  Period        : {datetime.fromtimestamp(first_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}"
          f" → {datetime.fromtimestamp(last_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"  Initial equity: ${initial_equity:,.0f}")
    print(f"  Final equity  : ${final_equity:,.0f}")
    print(f"  Net PnL       : ${m['net_pnl']:+,.0f}  ({m['net_pnl']/initial_equity*100:+.1f}%)")
    print(f"  CAGR          : {m['cagr']*100:.1f}%")
    print(f"  Max drawdown  : {max_dd*100:.1f}%")
    print(f"  Profit factor : {m['profit_factor']:.2f}")
    print(f"  Total trades  : {m['total_trades']}  (L:{m['long_trades']}  S:{m['short_trades']})")
    print(f"  Win rate      : {m['win_rate']*100:.1f}%")
    print(f"  Avg R         : {m['avg_r']:.2f}R")
    print(f"  Best R        : {m['best_r']:.1f}R")
    print(f"  Worst R       : {m['worst_r']:.1f}R")
    print(f"  Total fees    : ${m['total_fees']:,.0f}")

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
        status  = "✓" if yr_pf >= 1.0 else "✗"
        print(f"  {yr:<8} {len(yr_trades):>7} {yr_wr:>6.1f}% {yr_pf:>7.2f} ${yr_pnl:>+10,.0f}  {status}")

    # Largest winners and losers
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

    # Regime check: 2022–2025 vs earlier
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
    go_flags     = []

    # Kill checks
    if m["profit_factor"] < _KILL_PF_COMBINED:
        kill_reasons.append(f"Combined PF {m['profit_factor']:.2f} < kill threshold {_KILL_PF_COMBINED}")
    if m["total_trades"] < _KILL_MIN_TRADES:
        kill_reasons.append(f"Trade count {m['total_trades']} < minimum {_KILL_MIN_TRADES}")
    if max_dd > _KILL_MAX_DD:
        kill_reasons.append(f"Max drawdown {max_dd*100:.1f}% > kill threshold {_KILL_MAX_DD*100:.0f}%")
    for sym in symbols:
        sym_trades = [t for t in all_trades if t.symbol == sym]
        if not sym_trades:
            continue
        sym_gp = sum(t.pnl for t in sym_trades if t.pnl > 0)
        sym_gl = abs(sum(t.pnl for t in sym_trades if t.pnl <= 0))
        sym_pf = sym_gp / sym_gl if sym_gl > 0 else float("inf")
        if sym_pf < _KILL_PF_SYMBOL:
            kill_reasons.append(f"{sym} individual PF {sym_pf:.2f} < kill threshold {_KILL_PF_SYMBOL}")

    # Check regime concentration
    if early and recent:
        ep_gp = sum(t.pnl for t in early if t.pnl > 0)
        ep_gl = abs(sum(t.pnl for t in early if t.pnl <= 0))
        rp_gp = sum(t.pnl for t in recent if t.pnl > 0)
        rp_gl = abs(sum(t.pnl for t in recent if t.pnl <= 0))
        ep_pf = ep_gp / ep_gl if ep_gl > 0 else float("inf")
        rp_pf = rp_gp / rp_gl if rp_gl > 0 else float("inf")
        if ep_pf > 2.0 and rp_pf < 1.0:
            kill_reasons.append(
                f"Regime concentration: pre-2022 PF {ep_pf:.2f} vs post-2022 PF {rp_pf:.2f} — edge may be gone"
            )

    # Go checks
    if m["profit_factor"] >= _GO_PF:
        go_flags.append(f"PF {m['profit_factor']:.2f} ≥ {_GO_PF}")
    if m["cagr"] >= _GO_CAGR:
        go_flags.append(f"CAGR {m['cagr']*100:.1f}% ≥ {_GO_CAGR*100:.0f}%")
    if max_dd <= _GO_MAX_DD:
        go_flags.append(f"Max DD {max_dd*100:.1f}% ≤ {_GO_MAX_DD*100:.0f}%")

    if kill_reasons:
        print("\n  ❌ KILL — Strategy fails one or more kill criteria:\n")
        for r in kill_reasons:
            print(f"     • {r}")
        print("\n  Action: do NOT deploy. Move to next hypothesis.")
    elif len(go_flags) == 3:
        print("\n  ✅ GO — All go criteria met:\n")
        for f in go_flags:
            print(f"     • {f}")
        print("\n  Action: proceed to paper deployment.")
    else:
        passed = len(go_flags)
        print(f"\n  ⚠️  MARGINAL — {passed}/3 go criteria met. Deploy on paper with caution.")
        print("  Passed  :", ", ".join(go_flags) if go_flags else "none")
        unmet = []
        if m["profit_factor"] < _GO_PF:
            unmet.append(f"PF {m['profit_factor']:.2f} < {_GO_PF}")
        if m["cagr"] < _GO_CAGR:
            unmet.append(f"CAGR {m['cagr']*100:.1f}% < {_GO_CAGR*100:.0f}%")
        if max_dd > _GO_MAX_DD:
            unmet.append(f"Max DD {max_dd*100:.1f}% > {_GO_MAX_DD*100:.0f}%")
        print("  Unmet   :", ", ".join(unmet))
        print("\n  Action: run paper trading before adding capital.")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily trend following backtest")
    parser.add_argument("--symbols",  nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--equity",   type=float, default=100_000.0)
    parser.add_argument("--data-dir", default="data/candles")
    args = parser.parse_args()

    data_dir = _REPO_ROOT / args.data_dir

    print(f"\nLoading data from {data_dir} ...")
    candles_by_symbol: dict[str, list[Candle]] = {}
    for sym in args.symbols:
        candles = _load_daily(sym, data_dir)
        candles_by_symbol[sym] = candles
        span_start = datetime.fromtimestamp(candles[0].open_time/1000, tz=timezone.utc).strftime("%Y-%m-%d")
        span_end   = datetime.fromtimestamp(candles[-1].open_time/1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  {sym}: {len(candles)} daily bars  ({span_start} → {span_end})")

    print("\nRunning backtest ...")
    trades, eq_curve = _run(candles_by_symbol, args.equity)

    _print_report(trades, eq_curve, args.equity, args.symbols)


if __name__ == "__main__":
    main()
