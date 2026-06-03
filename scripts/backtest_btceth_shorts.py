#!/usr/bin/env python3
"""BTC+ETH Short-Enabled vs Long-Only comparison.

Tests whether adding SHORT capability to EMA strategies on BTC and ETH
(only, not altcoins) significantly improves returns without blowing up DD.

Strategies:
  1. Daily EMA 8/21  — long-only vs long+short
  2. 4H   EMA 5/13  — long-only vs long+short

Capital: $100,000 total → $50,000 per coin (2 coins)
Fee:     0.0006 taker each side
Period:  2021-01-01 to now
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

_CANDLE_DIR    = _REPO_ROOT / "data" / "candles"
_INITIAL_EQUITY = 100_000.0
_CAP_PER_COIN  = 50_000.0      # $50k per coin — only 2 coins
_TAKER_FEE     = 0.0006
_START_DT      = datetime(2021, 1, 1, tzinfo=timezone.utc)
_START_TS      = int(_START_DT.timestamp() * 1000)
_IS_SPLIT_DT   = datetime(2023, 1, 1, tzinfo=timezone.utc)
_IS_SPLIT_TS   = int(_IS_SPLIT_DT.timestamp() * 1000)

_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
def _ema(closes: list[float], period: int) -> list[float]:
    out = [float("nan")] * len(closes)
    alpha = 2.0 / (period + 1)
    seed_set = False
    for i, c in enumerate(closes):
        if not seed_set:
            out[i] = c
            seed_set = True
        else:
            out[i] = alpha * c + (1.0 - alpha) * out[i - 1]
    # blank warmup
    for i in range(min(period - 1, len(out))):
        out[i] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_daily(symbol: str) -> list[tuple]:
    """Returns [(ts_ms, open, high, low, close), ...] sorted, filtered."""
    path = _CANDLE_DIR / f"{symbol}_1D.parquet"
    tbl = pq.read_table(path, columns=["open_time", "open", "high", "low", "close"])
    rows = sorted(zip(
        tbl.column("open_time").to_pylist(),
        tbl.column("open").to_pylist(),
        tbl.column("high").to_pylist(),
        tbl.column("low").to_pylist(),
        tbl.column("close").to_pylist(),
    ))
    return [(ts, o, h, l, c) for ts, o, h, l, c in rows if ts >= _START_TS]


def _load_4h(symbol: str) -> list[tuple]:
    """Resample 1H → 4H (buckets 00,04,08,12,16,20 UTC) and return OHLC bars."""
    path = _CANDLE_DIR / f"{symbol}_1H.parquet"
    tbl = pq.read_table(path, columns=["open_time", "open", "high", "low", "close"])
    rows = sorted(zip(
        tbl.column("open_time").to_pylist(),
        tbl.column("open").to_pylist(),
        tbl.column("high").to_pylist(),
        tbl.column("low").to_pylist(),
        tbl.column("close").to_pylist(),
    ))
    # Group into 4H buckets: floor ts to nearest 4H boundary
    _4H_MS = 4 * 3600 * 1000
    buckets: dict[int, list] = {}
    for ts, o, h, l, c in rows:
        if ts < _START_TS:
            continue
        bucket = (ts // _4H_MS) * _4H_MS
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append((ts, o, h, l, c))

    bars = []
    for bucket_ts in sorted(buckets):
        group = sorted(buckets[bucket_ts])
        if len(group) < 4:  # incomplete 4H bar — skip
            continue
        open_  = group[0][1]
        high_  = max(r[2] for r in group)
        low_   = min(r[3] for r in group)
        close_ = group[-1][4]
        bars.append((bucket_ts, open_, high_, low_, close_))
    return bars


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------
def _signals_ema_cross(bars: list[tuple], fast: int, slow: int,
                        allow_short: bool) -> list[str | None]:
    closes = [c for _, _, _, _, c in bars]
    ef = _ema(closes, fast)
    es = _ema(closes, slow)
    out: list[str | None] = [None] * len(bars)
    for i in range(1, len(bars)):
        if math.isnan(ef[i]) or math.isnan(es[i]):
            continue
        if ef[i] > es[i]:
            out[i] = "LONG"
        elif allow_short:
            out[i] = "SHORT"
        # else: flat (long-only mode)
    return out


# ---------------------------------------------------------------------------
# Backtester — runs one symbol with a given signal list
# ---------------------------------------------------------------------------
def _backtest_symbol(bars: list[tuple], signals: list[str | None],
                     notional: float) -> dict:
    """
    Fixed-notional backtester (matches existing scripts convention).
    notional = fixed dollar amount to risk per position.
    PnL is always computed as price_return * notional (no compounding).
    Equity tracks the running total but notional stays fixed.
    Execution: signal on bar i is acted on at open of bar i+1.
    """
    equity = notional
    position = None   # None | "LONG" | "SHORT"
    entry_price = 0.0
    equity_curve: list[tuple[int, float]] = []  # (ts, cumulative_pnl relative to notional)
    trades: list[dict] = []
    running_pnl = 0.0   # cumulative net P&L

    for i in range(len(bars)):
        ts, o, h, l, c = bars[i]

        # Signal on bar i is acted on at open of bar i+1
        # So for i>0, the execution price is this bar's open
        if i > 0:
            exec_price = o
            sig_prev   = signals[i - 1]  # signal that triggers action at this open

            if sig_prev != position:
                # Close existing position
                if position is not None:
                    if position == "LONG":
                        ret = (exec_price - entry_price) / entry_price
                    else:  # SHORT
                        ret = (entry_price - exec_price) / entry_price
                    close_fee = notional * _TAKER_FEE
                    pnl_net   = notional * ret - close_fee
                    running_pnl += pnl_net
                    trades.append({
                        "side":    position,
                        "entry":   entry_price,
                        "exit":    exec_price,
                        "pnl":     pnl_net,
                        "ts_exit": ts,
                    })
                    position = None

                # Open new position
                if sig_prev is not None:
                    open_fee     = notional * _TAKER_FEE
                    running_pnl -= open_fee
                    entry_price  = exec_price
                    position     = sig_prev

        equity_curve.append((ts, notional + running_pnl))

    # Close last open position at final bar's close
    if position is not None and bars:
        ts, _, _, _, c = bars[-1]
        if position == "LONG":
            ret = (c - entry_price) / entry_price
        else:
            ret = (entry_price - c) / entry_price
        close_fee    = notional * _TAKER_FEE
        pnl_net      = notional * ret - close_fee
        running_pnl += pnl_net
        trades.append({
            "side": position, "entry": entry_price, "exit": c,
            "pnl": pnl_net, "ts_exit": bars[-1][0],
        })
        if equity_curve:
            equity_curve[-1] = (bars[-1][0], notional + running_pnl)

    final_equity = notional + running_pnl
    return {"equity_curve": equity_curve, "trades": trades, "final_equity": final_equity}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _calc_metrics(equity_curve: list[tuple[int, float]], trades: list[dict],
                  initial: float) -> dict:
    if not equity_curve:
        return {}

    equities = [e for _, e in equity_curve]
    timestamps = [ts for ts, _ in equity_curve]

    # CAGR
    t_start = timestamps[0] / 1000
    t_end   = timestamps[-1] / 1000
    years   = (t_end - t_start) / (365.25 * 86400)
    final   = equities[-1]
    cagr    = (final / initial) ** (1 / years) - 1 if years > 0 else 0.0

    # Max drawdown
    peak = initial
    max_dd = 0.0
    for e in equities:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_wins  = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # IS / OOS PF
    is_trades  = [t for t in trades if t["ts_exit"] < _IS_SPLIT_TS]
    oos_trades = [t for t in trades if t["ts_exit"] >= _IS_SPLIT_TS]

    def _pf(ts):
        w = sum(t["pnl"] for t in ts if t["pnl"] > 0)
        l = abs(sum(t["pnl"] for t in ts if t["pnl"] < 0))
        return w / l if l > 0 else float("inf")

    return {
        "final_equity": final,
        "cagr": cagr,
        "max_dd": max_dd,
        "pf": pf,
        "n_trades": len(trades),
        "is_pf":  _pf(is_trades),
        "oos_pf": _pf(oos_trades),
        "equity_curve": equity_curve,
        "trades": trades,
    }


def _year_by_year(equity_curve: list[tuple[int, float]],
                  initial: float) -> dict[int, float]:
    """Returns {year: return_pct}."""
    if not equity_curve:
        return {}
    from collections import defaultdict
    year_start: dict[int, float] = {}
    year_end:   dict[int, float] = {}
    for ts, eq in equity_curve:
        yr = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year
        if yr not in year_start:
            year_start[yr] = eq
        year_end[yr] = eq

    result = {}
    prev_eq = initial
    for yr in sorted(year_start):
        start = prev_eq  # equity at start of year (= end of prev year)
        end   = year_end[yr]
        result[yr] = (end - start) / start if start > 0 else 0.0
        prev_eq = end
    return result


# ---------------------------------------------------------------------------
# Run one variant across both coins
# ---------------------------------------------------------------------------
def _run_variant(name: str, timeframe: str, fast: int, slow: int,
                 allow_short: bool) -> dict:
    """Combine BTC + ETH results into a portfolio metric."""
    all_equity: dict[str, list] = {}
    all_trades: list[dict] = []

    for sym in _SYMBOLS:
        if timeframe == "1D":
            bars = _load_daily(sym)
        else:  # 4H
            bars = _load_4h(sym)

        signals = _signals_ema_cross(bars, fast, slow, allow_short)
        result  = _backtest_symbol(bars, signals, _CAP_PER_COIN)
        all_equity[sym] = result["equity_curve"]
        all_trades.extend(result["trades"])

    # Merge equity curves: sum both symbols at each common timestamp
    # Build per-symbol sorted (ts, equity) lists and walk together
    ts_set = sorted(set(ts for sym in _SYMBOLS for ts, _ in all_equity[sym]))

    # Build fast lookup: for each symbol, a sorted list and a running index
    sym_eq: dict[str, dict[int, float]] = {}
    for sym in _SYMBOLS:
        d: dict[int, float] = {}
        last = _CAP_PER_COIN
        for ts, eq in sorted(all_equity[sym]):
            d[ts] = eq
            last  = eq
        sym_eq[sym] = d

    combined: list[tuple[int, float]] = []
    last_eq: dict[str, float] = {sym: _CAP_PER_COIN for sym in _SYMBOLS}
    for ts in ts_set:
        total = 0.0
        for sym in _SYMBOLS:
            if ts in sym_eq[sym]:
                last_eq[sym] = sym_eq[sym][ts]
            total += last_eq[sym]
        combined.append((ts, total))

    metrics = _calc_metrics(combined, all_trades, _INITIAL_EQUITY)
    metrics["name"] = name
    metrics["yby"]  = _year_by_year(combined, _INITIAL_EQUITY)
    return metrics


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def _fmt_pf(v: float) -> str:
    if v == float("inf"):
        return "  ∞"
    return f"{v:5.2f}"


def _print_result(m: dict):
    name  = m["name"]
    cagr  = m["cagr"] * 100
    dd    = m["max_dd"] * 100
    pf    = m["pf"]
    nt    = m["n_trades"]
    is_pf = m["is_pf"]
    oos_pf = m["oos_pf"]
    final = m["final_equity"]
    total_ret = (final / _INITIAL_EQUITY - 1) * 100

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Final equity : ${final:>12,.0f}  (total ret {total_ret:+.1f}%)")
    print(f"  CAGR         : {cagr:+.1f}%")
    print(f"  Max Drawdown : {dd:.1f}%")
    print(f"  Profit Factor: {_fmt_pf(pf)}")
    print(f"  Trades       : {nt}")
    print(f"  IS  PF (<2023): {_fmt_pf(is_pf)}")
    print(f"  OOS PF (2023+): {_fmt_pf(oos_pf)}")
    print(f"\n  Year-by-year:")
    for yr, ret in sorted(m["yby"].items()):
        bar_len = min(40, int(abs(ret) * 20))
        bar = "█" * bar_len
        sign = "+" if ret >= 0 else "-"
        print(f"    {yr}: {ret*100:+7.1f}%  {sign}{bar}")


def _print_comparison(lo: dict, ls: dict):
    """Side-by-side delta table."""
    print(f"\n{'─'*60}")
    print(f"  DELTA: Long+Short vs Long-Only")
    print(f"{'─'*60}")
    d_cagr = (ls["cagr"] - lo["cagr"]) * 100
    d_dd   = (ls["max_dd"] - lo["max_dd"]) * 100
    d_pf   = ls["pf"] - lo["pf"] if lo["pf"] != float("inf") else float("nan")
    d_nt   = ls["n_trades"] - lo["n_trades"]
    arrow  = lambda v: "▲" if v > 0 else ("▼" if v < 0 else "─")

    print(f"  CAGR delta   : {d_cagr:+.1f}%   {arrow(d_cagr)}")
    print(f"  MaxDD delta  : {d_dd:+.1f}%   {arrow(-d_dd)}")   # negative is better
    print(f"  PF delta     : {d_pf:+.2f}   {arrow(d_pf)}" if not math.isnan(d_pf) else "  PF delta: n/a")
    print(f"  Extra trades : {d_nt:+d}    {arrow(d_nt)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  BTC + ETH: Short-Enabled vs Long-Only Backtest")
    print(f"  Period: 2021-01-01 → {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"  Capital: ${_INITIAL_EQUITY:,.0f}  (${_CAP_PER_COIN:,.0f}/coin)")
    print(f"  Fee: {_TAKER_FEE*100:.2f}% each side (taker)")
    print("=" * 60)

    # ----- Daily EMA 8/21 -----
    print("\n\n>>> DAILY EMA 8/21 <<<")
    d_lo = _run_variant("Daily EMA 8/21 — Long-Only",   "1D", 8, 21, allow_short=False)
    d_ls = _run_variant("Daily EMA 8/21 — Long+Short",  "1D", 8, 21, allow_short=True)
    _print_result(d_lo)
    _print_result(d_ls)
    _print_comparison(d_lo, d_ls)

    # ----- 4H EMA 5/13 -----
    print("\n\n>>> 4H EMA 5/13 <<<")
    h_lo = _run_variant("4H EMA 5/13 — Long-Only",   "4H", 5, 13, allow_short=False)
    h_ls = _run_variant("4H EMA 5/13 — Long+Short",  "4H", 5, 13, allow_short=True)
    _print_result(h_lo)
    _print_result(h_ls)
    _print_comparison(h_lo, h_ls)

    # ----- Summary table -----
    print("\n\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    header = f"  {'Strategy':<35} {'CAGR':>7} {'MaxDD':>7} {'PF':>6} {'Trades':>7} {'IS_PF':>6} {'OOS_PF':>7}"
    print(header)
    print("  " + "-" * 58)
    for m in [d_lo, d_ls, h_lo, h_ls]:
        n   = m["name"][:35]
        c   = f"{m['cagr']*100:+.1f}%"
        d   = f"{m['max_dd']*100:.1f}%"
        p   = _fmt_pf(m["pf"])
        nt  = str(m["n_trades"])
        ip  = _fmt_pf(m["is_pf"])
        op  = _fmt_pf(m["oos_pf"])
        print(f"  {n:<35} {c:>7} {d:>7} {p:>6} {nt:>7} {ip:>6} {op:>7}")

    print("\n  KEY QUESTION: Does shorting BTC+ETH significantly improve returns?")
    d_delta  = (d_ls["cagr"] - d_lo["cagr"]) * 100
    h_delta  = (h_ls["cagr"] - h_lo["cagr"]) * 100
    d_dd_delta = (d_ls["max_dd"] - d_lo["max_dd"]) * 100
    h_dd_delta = (h_ls["max_dd"] - h_lo["max_dd"]) * 100

    print(f"  Daily: Short adds {d_delta:+.1f}% CAGR, DD changes {d_dd_delta:+.1f}%")
    print(f"  4H:    Short adds {h_delta:+.1f}% CAGR, DD changes {h_dd_delta:+.1f}%")

    if d_delta > 5 and d_dd_delta < 10:
        verdict_d = "YES — significant CAGR gain without excessive DD increase (Daily)"
    elif d_delta < 0:
        verdict_d = "NO  — shorts HURT returns on Daily timeframe"
    else:
        verdict_d = "MARGINAL — modest gain, check DD carefully (Daily)"

    if h_delta > 5 and h_dd_delta < 10:
        verdict_h = "YES — significant CAGR gain without excessive DD increase (4H)"
    elif h_delta < 0:
        verdict_h = "NO  — shorts HURT returns on 4H timeframe"
    else:
        verdict_h = "MARGINAL — modest gain, check DD carefully (4H)"

    print(f"\n  VERDICT Daily: {verdict_d}")
    print(f"  VERDICT 4H:    {verdict_h}")
    print()


if __name__ == "__main__":
    main()
