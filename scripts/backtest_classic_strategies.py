#!/usr/bin/env python3
"""Comprehensive classic strategy comparison on 12-coin crypto universe.

Tests all major "simple strategies that make traders rich":
  A. SMA(200) binary timing    — long above, short below. Zero tuning.
  B. Golden/Death cross        — SMA(50) x SMA(200). Institutional standard.
  C. EMA(20/50) crossover      — faster version of B. Most popular with day traders.
  D. EMA(8/21) crossover       — aggressive fast-cross used by crypto traders.
  E. Weekly Donchian (13w)     — breakout on weekly bars. Fewer false signals than 4H.
  F. MACD(12,26,9) crossover   — momentum oscillator on daily.
  G. Bollinger Band breakout   — daily close outside 2-sigma band = entry.
  H. SMA(50) alone             — pure medium-term trend, no lagging filter.

Each strategy:
  - Runs on the same 12-coin universe
  - Uses equal capital/12 per active position
  - Taker fee 0.06% per side on every change
  - Reports PF, CAGR, DD, trade count, IS/OOS split

Run all and show a ranking table at the end.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

_DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]
_INITIAL_EQUITY = 100_000.0
_TAKER_FEE      = 0.0006
_CANDLE_DIR     = _REPO_ROOT / "data" / "candles"
_DAY_MS         = 86_400_000


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def sma(closes: list[float], period: int) -> list[float]:
    out = [float("nan")] * len(closes)
    s = 0.0
    for i, c in enumerate(closes):
        s += c
        if i >= period:
            s -= closes[i - period]
        if i >= period - 1:
            out[i] = s / period
    return out


def ema(closes: list[float], period: int) -> list[float]:
    out = [float("nan")] * len(closes)
    alpha = 2.0 / (period + 1)
    for i, c in enumerate(closes):
        if i == 0:
            out[i] = c
        elif math.isnan(out[i - 1]):
            out[i] = c
        else:
            out[i] = alpha * c + (1 - alpha) * out[i - 1]
    # Blank out the initial warmup
    for i in range(min(period - 1, len(out))):
        out[i] = float("nan")
    return out


def macd(closes: list[float], fast=12, slow=26, signal=9):
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    macd_line = [
        f - s if not (math.isnan(f) or math.isnan(s)) else float("nan")
        for f, s in zip(fast_ema, slow_ema)
    ]
    # Signal line = EMA(9) of macd_line
    sig_line = [float("nan")] * len(macd_line)
    first_valid = next((i for i, v in enumerate(macd_line) if not math.isnan(v)), None)
    if first_valid is not None:
        alpha = 2.0 / (signal + 1)
        sig_line[first_valid] = macd_line[first_valid]
        for i in range(first_valid + 1, len(macd_line)):
            if math.isnan(macd_line[i]):
                sig_line[i] = float("nan")
            elif math.isnan(sig_line[i - 1]):
                sig_line[i] = macd_line[i]
            else:
                sig_line[i] = alpha * macd_line[i] + (1 - alpha) * sig_line[i - 1]
        for i in range(first_valid, first_valid + signal - 1):
            sig_line[i] = float("nan")
    return macd_line, sig_line


def bollinger(closes: list[float], period=20, std_mult=2.0):
    mid = sma(closes, period)
    upper = [float("nan")] * len(closes)
    lower = [float("nan")] * len(closes)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1: i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        upper[i] = mean + std_mult * std
        lower[i] = mean - std_mult * std
    return upper, lower


def atr(highs: list[float], lows: list[float], closes: list[float], period=14) -> list[float]:
    out = [float("nan")] * len(closes)
    trs = []
    for i in range(len(closes)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            trs.append(max(highs[i] - lows[i],
                           abs(highs[i] - closes[i-1]),
                           abs(lows[i] - closes[i-1])))
    # Wilder smoothing
    if len(trs) < period:
        return out
    out[period - 1] = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        out[i] = (out[i-1] * (period - 1) + trs[i]) / period
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_all(symbols: list[str], from_ts: int, to_ts: int):
    """Returns {symbol: [(ts, open, high, low, close), ...]} sorted."""
    data = {}
    for sym in symbols:
        path = _CANDLE_DIR / f"{sym}_1D.parquet"
        if not path.exists():
            data[sym] = []
            continue
        tbl = pq.read_table(path, columns=["open_time", "open", "high", "low", "close"])
        rows = sorted(zip(
            tbl.column("open_time").to_pylist(),
            tbl.column("open").to_pylist(),
            tbl.column("high").to_pylist(),
            tbl.column("low").to_pylist(),
            tbl.column("close").to_pylist(),
        ))
        data[sym] = [(ts, o, h, l, c) for ts, o, h, l, c in rows if from_ts <= ts <= to_ts]
    return data


# ---------------------------------------------------------------------------
# Signal generators — return list of (ts, side) where side = "LONG"|"SHORT"|None
# ---------------------------------------------------------------------------
def _signals_sma200(bars):
    closes = [c for _, _, _, _, c in bars]
    s200   = sma(closes, 200)
    return [("LONG" if c > s else "SHORT" if c < s else None)
            if not math.isnan(s) else None
            for c, s in zip(closes, s200)]


def _signals_golden_cross(bars):
    closes = [c for _, _, _, _, c in bars]
    s50  = sma(closes, 50)
    s200 = sma(closes, 200)
    state = [None] * len(closes)
    for i in range(1, len(closes)):
        if math.isnan(s50[i]) or math.isnan(s200[i]):
            continue
        if math.isnan(s50[i-1]) or math.isnan(s200[i-1]):
            continue
        if s50[i] > s200[i]:
            state[i] = "LONG"
        else:
            state[i] = "SHORT"
    return state


def _signals_ema2050(bars):
    closes = [c for _, _, _, _, c in bars]
    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    state = [None] * len(closes)
    for i in range(1, len(closes)):
        if math.isnan(e20[i]) or math.isnan(e50[i]):
            continue
        state[i] = "LONG" if e20[i] > e50[i] else "SHORT"
    return state


def _signals_ema821(bars):
    closes = [c for _, _, _, _, c in bars]
    e8  = ema(closes, 8)
    e21 = ema(closes, 21)
    state = [None] * len(closes)
    for i in range(1, len(closes)):
        if math.isnan(e8[i]) or math.isnan(e21[i]):
            continue
        state[i] = "LONG" if e8[i] > e21[i] else "SHORT"
    return state


def _signals_weekly_donchian(bars, period_weeks=13):
    """13-week = 91-day Donchian channel."""
    period = period_weeks * 7
    highs  = [h for _, _, h, _, _ in bars]
    lows   = [l for _, _, _, l, _ in bars]
    closes = [c for _, _, _, _, c in bars]
    state  = [None] * len(bars)
    for i in range(period, len(bars)):
        window_h = highs[i - period: i]
        window_l = lows[i - period: i]
        if closes[i] > max(window_h):
            state[i] = "LONG"
        elif closes[i] < min(window_l):
            state[i] = "SHORT"
        elif state[i-1] is not None:
            state[i] = state[i-1]  # hold previous
    return state


def _signals_macd(bars):
    closes = [c for _, _, _, _, c in bars]
    macd_line, sig_line = macd(closes)
    state = [None] * len(closes)
    cur = None
    for i in range(1, len(closes)):
        if math.isnan(macd_line[i]) or math.isnan(sig_line[i]):
            continue
        if macd_line[i] > sig_line[i]:
            cur = "LONG"
        else:
            cur = "SHORT"
        state[i] = cur
    return state


def _signals_bollinger_breakout(bars):
    """Enter on close outside bands; exit when price returns inside."""
    closes = [c for _, _, _, _, c in bars]
    upper, lower = bollinger(closes, 20, 2.0)
    state = [None] * len(closes)
    cur = None
    for i in range(1, len(closes)):
        if math.isnan(upper[i]):
            continue
        c = closes[i]
        if c > upper[i]:
            cur = "LONG"
        elif c < lower[i]:
            cur = "SHORT"
        elif cur == "LONG" and c < upper[i] * 0.995:
            cur = None
        elif cur == "SHORT" and c > lower[i] * 1.005:
            cur = None
        state[i] = cur
    return state


def _signals_sma50(bars):
    """Simple SMA(50) direction — fastest pure-MA signal."""
    closes = [c for _, _, _, _, c in bars]
    s50 = sma(closes, 50)
    return [("LONG" if c > s else "SHORT")
            if not math.isnan(s) else None
            for c, s in zip(closes, s50)]


# ---------------------------------------------------------------------------
# Portfolio simulation (shared across all strategies)
# ---------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    equity: float = _INITIAL_EQUITY
    peak: float = _INITIAL_EQUITY
    max_dd: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    total_fees: float = 0.0
    yearly: dict = field(default_factory=dict)
    is_gp: float = 0.0
    is_gl: float = 0.0
    oos_gp: float = 0.0
    oos_gl: float = 0.0

    @property
    def pf(self):
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else float("inf")

    @property
    def win_rate(self):
        return self.n_wins / self.n_trades if self.n_trades > 0 else 0.0

    def cagr(self, from_ts, to_ts):
        years = (to_ts - from_ts) / (1000 * 86400 * 365.25)
        if years <= 0 or self.equity <= 0:
            return -1.0
        return (self.equity / _INITIAL_EQUITY) ** (1 / years) - 1

    @property
    def is_pf(self):
        return self.is_gp / self.is_gl if self.is_gl > 0 else float("inf")

    @property
    def oos_pf(self):
        return self.oos_gp / self.oos_gl if self.oos_gl > 0 else float("inf")


def simulate_portfolio(
    name: str,
    all_data: dict,
    signal_fn: Callable,
    symbols: list[str],
    from_ts: int,
    to_ts: int,
) -> Result:
    result = Result(name=name)
    notional = _INITIAL_EQUITY / len(symbols)

    # Per-symbol: run signal, detect transitions, compute trade P&L
    for sym in symbols:
        bars = all_data.get(sym, [])
        if not bars:
            continue

        signals = signal_fn(bars)
        closes  = [c for _, _, _, _, c in bars]
        timestamps = [ts for ts, _, _, _, _ in bars]

        prev_sig = None
        entry_price = None
        entry_ts = None
        entry_side = None

        def _close_trade(exit_price, exit_ts, exit_sig):
            nonlocal prev_sig, entry_price, entry_ts, entry_side
            if entry_side == "LONG":
                pnl = (exit_price - entry_price) / entry_price * notional
            else:
                pnl = (entry_price - exit_price) / entry_price * notional
            fee = _TAKER_FEE * notional  # exit fee (entry fee already subtracted)
            net = pnl - fee
            result.equity += net
            result.peak = max(result.peak, result.equity)
            dd = (result.peak - result.equity) / result.peak
            result.max_dd = max(result.max_dd, dd)
            if net > 0:
                result.gross_profit += net
                result.n_wins += 1
            else:
                result.gross_loss += abs(net)
            result.n_trades += 1
            yr = datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc).year
            result.yearly[yr] = result.yearly.get(yr, 0.0) + net
            is_oos = datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc).year >= 2023
            if is_oos:
                if net > 0: result.oos_gp += net
                else: result.oos_gl += abs(net)
            else:
                if net > 0: result.is_gp += net
                else: result.is_gl += abs(net)
            prev_sig = exit_sig
            entry_price = entry_ts = entry_side = None

        for i, (ts, sig) in enumerate(zip(timestamps, signals)):
            if sig == prev_sig:
                continue
            c = closes[i]
            # Close existing position on signal change
            if entry_side is not None and sig != entry_side:
                _close_trade(c, ts, sig)
            # Open new position if signal is actionable
            if sig in ("LONG", "SHORT"):
                entry_price = c
                entry_ts = ts
                entry_side = sig
                fee = _TAKER_FEE * notional
                result.equity -= fee
                result.total_fees += fee
                result.total_fees += fee  # pre-count exit fee too? No — count on exit
                prev_sig = sig

        # Close at end of period
        if entry_side is not None:
            ts_last, _, _, _, c_last = bars[-1]
            _close_trade(c_last, ts_last, None)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--from", dest="from_date", default="2021-01-01")
    parser.add_argument("--to",   dest="to_date",   default=None)
    args = parser.parse_args()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt   = (
        datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.to_date else datetime.now(timezone.utc)
    )
    from_ts = int(from_dt.timestamp() * 1000)
    to_ts   = int(to_dt.timestamp() * 1000)

    print("=" * 72)
    print("  CLASSIC STRATEGY SHOOTOUT — 12-coin universe, daily bars")
    print(f"  Period: {from_dt.strftime('%Y-%m-%d')} → {to_dt.strftime('%Y-%m-%d')}")
    print(f"  Capital: ${_INITIAL_EQUITY:,.0f}  |  Fee: {_TAKER_FEE*100:.3f}% taker/side")
    print("=" * 72)
    print()

    print("Loading candles ...")
    all_data = _load_all(args.symbols, from_ts, to_ts)
    for sym, bars in all_data.items():
        if bars:
            print(f"  {sym}: {len(bars):,} bars")
    print()

    def long_only(fn):
        """Wrap a signal function to suppress SHORT signals (go flat instead)."""
        def _wrapped(bars):
            return [s if s == "LONG" else None for s in fn(bars)]
        return _wrapped

    strategies = [
        # Long+Short (full)
        ("A. SMA200 L+S",             _signals_sma200),
        ("B. GoldenCross 50/200 L+S", _signals_golden_cross),
        ("C. EMA 20/50 L+S",          _signals_ema2050),
        ("D. EMA 8/21 L+S",           _signals_ema821),
        ("E. Donchian 13w L+S",       _signals_weekly_donchian),
        ("F. MACD L+S",               _signals_macd),
        ("G. Bollinger BB L+S",       _signals_bollinger_breakout),
        ("H. SMA50 L+S",              _signals_sma50),
        # Long-only (flat in bear)
        ("A2. SMA200 LONG-ONLY",      long_only(_signals_sma200)),
        ("B2. GoldenCross LONG-ONLY", long_only(_signals_golden_cross)),
        ("C2. EMA 20/50 LONG-ONLY",   long_only(_signals_ema2050)),
        ("D2. EMA 8/21 LONG-ONLY",    long_only(_signals_ema821)),
        ("E2. Donchian 13w LONG-ONLY",long_only(_signals_weekly_donchian)),
        ("F2. MACD LONG-ONLY",        long_only(_signals_macd)),
    ]

    results = []
    for name, fn in strategies:
        r = simulate_portfolio(name, all_data, fn, args.symbols, from_ts, to_ts)
        results.append(r)
        print(f"  {name}: PF={r.pf:.2f}  CAGR={r.cagr(from_ts,to_ts)*100:.1f}%  DD={r.max_dd*100:.1f}%  n={r.n_trades}")

    print()
    print("=" * 72)
    print("  FULL RESULTS — sorted by CAGR")
    print("=" * 72)
    results.sort(key=lambda r: r.cagr(from_ts, to_ts), reverse=True)

    for r in results:
        c = r.cagr(from_ts, to_ts)
        net = r.equity - _INITIAL_EQUITY
        status = "GO" if r.pf >= 1.30 and c >= 0.20 and r.max_dd <= 0.40 else \
                 "MARG" if r.pf >= 1.10 and r.max_dd <= 0.40 and r.n_trades >= 60 else "KILL"
        print()
        print(f"  ── {r.name}  [{status}]")
        print(f"     Equity  : ${r.equity:,.0f}  (net ${net:,.0f}, {net/_INITIAL_EQUITY*100:.1f}%)")
        print(f"     CAGR    : {c*100:.1f}%   PF: {r.pf:.2f}   DD: {r.max_dd*100:.1f}%   WR: {r.win_rate*100:.0f}%")
        print(f"     Trades  : {r.n_trades}   Fees: ${r.total_fees:,.0f}")
        print(f"     IS PF   : {r.is_pf:.2f}   OOS PF: {r.oos_pf:.2f}")
        print(f"     By year : ", end="")
        for yr in sorted(r.yearly):
            sign = "+" if r.yearly[yr] >= 0 else ""
            print(f"{yr}:{sign}{r.yearly[yr]/1000:.0f}K  ", end="")
        print()

    print()
    print("=" * 72)
    print("  RANKING TABLE")
    print("=" * 72)
    print(f"  {'Strategy':<28} {'CAGR':>7} {'PF':>6} {'DD':>6} {'n':>5} {'IS PF':>7} {'OOS PF':>7}  Verdict")
    print(f"  {'─'*28} {'─'*7} {'─'*6} {'─'*6} {'─'*5} {'─'*7} {'─'*7}  {'─'*7}")
    for r in results:
        c = r.cagr(from_ts, to_ts)
        status = "✓ GO" if r.pf >= 1.30 and c >= 0.20 and r.max_dd <= 0.40 else \
                 "MARG" if r.pf >= 1.10 and r.max_dd <= 0.40 and r.n_trades >= 60 else "KILL"
        print(f"  {r.name:<28} {c*100:>6.1f}% {r.pf:>6.2f} {r.max_dd*100:>5.1f}% {r.n_trades:>5} {r.is_pf:>7.2f} {r.oos_pf:>7.2f}  {status}")

    print()
    best = results[0]
    bc = best.cagr(from_ts, to_ts)
    go_candidates = [r for r in results if r.pf >= 1.30 and r.cagr(from_ts,to_ts) >= 0.20 and r.max_dd <= 0.40]
    if go_candidates:
        print(f"  ★ GO candidates: {', '.join(r.name for r in go_candidates)}")
    else:
        print(f"  Best: {best.name}  —  CAGR {bc*100:.1f}%, PF {best.pf:.2f}, DD {best.max_dd*100:.1f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
