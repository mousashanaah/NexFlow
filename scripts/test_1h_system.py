#!/usr/bin/env python3
"""1H standalone trend system — can it beat daily V8.63?

Tests V8.63-equivalent logic running on 1H candles instead of daily.
Same economic principles, same regime gate (daily BTC SMA200 still used as
macro anchor — regime doesn't make sense on 1H), same hard stop, same fee.

Signals adapted for 1H:
  EMA cross: 8H vs 21H (same ratio as daily 8/21)
  MACD:      12H / 26H / 9H signal
  BTC regime: daily SMA200 still controls long permission (macro context)
  Momentum gate: 20-DAY return > 0 (daily, same gate)
  Hard stop: 15% from entry (same)
  ATR sizing: target 1% daily risk, sized from 1H vol

Key question: do fees + whipsaw destroy the edge, or does the faster
signal capture enough extra trend to compensate?

Variants:
  H1. 1H signals, daily regime gate (direct comparison to V8.63)
  H2. 1H signals, 1H momentum gate (20 × 24H lookback)
  H3. 1H signals + confirmation: signal must hold 3H before entry
  H4. 1H entry, daily EXIT (enter fast, exit slow — hybrid)

All compared to V8.63 daily baseline.

Usage:
  python scripts/test_1h_system.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    print("[ERROR] pip install pyarrow"); sys.exit(1)

import scripts.backtest_full_regime_system as B  # noqa: E402

B._CAPITAL = 5_000.0
_DAY_MS  = 86_400_000
_HOUR_MS = 3_600_000
_FROM = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_TO   = int(datetime.now(timezone.utc).timestamp() * 1000)
_H1A  = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_H1B  = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)

_TAKER_FEE = 0.0006   # 0.06% per side
_SYMBOLS   = B._SYMBOLS

# V8.63 baseline params (daily)
_BASE_DAILY = dict(
    hard_stop_pct=0.15, use_atr_sizing=True,
    asymmetric_regime=True, and_entry=True,
    bear_drop_pct=-0.20, confirm_days=10,
    momentum_gate=True, momentum_gate_days=20,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_1h(symbol: str) -> list[dict]:
    path = B._CANDLE_DIR / f"{symbol}_1H.parquet"
    if not path.exists():
        return []
    tbl = pq.read_table(path, columns=["open_time", "close"])
    rows = [{"ts": int(t), "close": float(c)}
            for t, c in zip(tbl.column("open_time").to_pylist(),
                            tbl.column("close").to_pylist())]
    return sorted(rows, key=lambda x: x["ts"])


def _load_daily(symbol: str) -> list[dict]:
    return B._load_daily(symbol)


# ---------------------------------------------------------------------------
# Build 1H signal dict: {sym: {ts_ms: {ema_long, macd_long, close, sma200_above}}}
# ---------------------------------------------------------------------------

def _build_1h_signals() -> dict:
    """Build per-hour signals for all coins. Regime (SMA200) stays daily."""
    print("  Building 1H signals ...")
    daily_sma200: dict[str, dict[int, bool]] = {}
    for sym in _SYMBOLS:
        bars = _load_daily(sym)
        if not bars:
            continue
        closes = [b["close"] for b in bars]
        ts_list = [b["ts"] for b in bars]
        sma200 = B._sma_series(closes, 200)
        daily_sma200[sym] = {
            ts_list[i]: (closes[i] > sma200[i])
            for i in range(len(ts_list))
            if sma200[i] is not None
        }

    # Daily BTC 30d momentum for regime gate
    btc_daily = _load_daily("BTCUSDT")
    btc_daily_closes = [b["close"] for b in btc_daily]
    btc_daily_ts     = [b["ts"]    for b in btc_daily]
    btc_mom30: dict[int, float] = {}
    for i, t in enumerate(btc_daily_ts):
        if i >= 30:
            btc_mom30[t] = (btc_daily_closes[i] - btc_daily_closes[i-30]) / btc_daily_closes[i-30]

    # Daily 20d coin momentum for momentum gate
    daily_mom20: dict[str, dict[int, float]] = {}
    for sym in _SYMBOLS:
        bars = _load_daily(sym)
        if not bars:
            continue
        closes = [b["close"] for b in bars]
        ts_list = [b["ts"] for b in bars]
        daily_mom20[sym] = {}
        for i, t in enumerate(ts_list):
            if i >= 20:
                daily_mom20[sym][t] = (closes[i] - closes[i-20]) / closes[i-20]

    out: dict[str, dict[int, dict]] = {}
    for sym in _SYMBOLS:
        bars = _load_1h(sym)
        if not bars:
            print(f"    [skip] {sym}: no 1H data")
            continue
        closes  = [b["close"] for b in bars]
        ts_list = [b["ts"]    for b in bars]
        n = len(closes)

        ema8  = B._ema_series(closes, 8)
        ema21 = B._ema_series(closes, 21)
        macd  = B._macd_long_series(closes)

        sym_out: dict[int, dict] = {}
        for i, ts in enumerate(ts_list):
            # Map hourly ts → most recent daily ts for regime/momentum
            day_ts = (ts // _DAY_MS) * _DAY_MS

            ema_long  = bool(ema8[i] and ema21[i] and ema8[i] > ema21[i])
            macd_long = macd[i]

            sym_out[ts] = {
                "close":        closes[i],
                "ema_long":     ema_long,
                "macd_long":    macd_long,
                "sma200_above": daily_sma200.get(sym, {}).get(day_ts, True),
                "day_ts":       day_ts,
            }
        out[sym] = sym_out

    return out, btc_mom30, daily_mom20


# ---------------------------------------------------------------------------
# 1H backtest engine
# ---------------------------------------------------------------------------

def _run_1h(
    sig1h: dict,
    btc_mom30: dict,
    daily_mom20: dict,
    from_ts: int,
    to_ts: int,
    # Signal persistence before entry (hours)
    persist_hours: int = 0,
    # Use daily exit signals (hybrid: 1H entry, daily exit check)
    daily_exit_signals: dict = None,
) -> dict:
    """
    Simplified 1H backtest — same logic as V8.63 but on hourly candles.
    Regime gate still uses daily BTC SMA200 + 30d drop (evaluated per hour
    by mapping to the day's value).
    """
    _CAPITAL = B._CAPITAL
    base_notional = _CAPITAL / len(_SYMBOLS)

    # Build daily BTC bear state (same V8.63 AND-entry logic)
    btc_daily = _load_daily("BTCUSDT")
    btc_daily_closes = [b["close"] for b in btc_daily]
    btc_daily_ts     = [b["ts"]    for b in btc_daily]
    btc_sma200_vals  = B._sma_series(btc_daily_closes, 200)
    btc_daily_sma200 = {
        btc_daily_ts[i]: btc_sma200_vals[i]
        for i in range(len(btc_daily_ts))
        if btc_sma200_vals[i] is not None
    }

    # Compute bear state per day (V8.63 confirm_days=10 logic)
    btc_bear_by_day: dict[int, bool] = {}
    streak = 0
    prev_bear = False
    for i, day_ts in enumerate(btc_daily_ts):
        c = btc_daily_closes[i]
        sma200 = btc_daily_sma200.get(day_ts)
        if sma200 is None:
            btc_bear_by_day[day_ts] = False
            continue
        above = c > sma200
        mom30 = btc_mom30.get(day_ts, 0.0)
        drop_triggered = mom30 < -0.20
        enter_bear = (not above) and drop_triggered
        if above:
            streak += 1
        else:
            streak = 0
        confirmed_bull = streak >= 10
        if prev_bear:
            bear = not confirmed_bull
        else:
            bear = enter_bear
        prev_bear = bear
        btc_bear_by_day[day_ts] = bear

    # All 1H timestamps in range
    all_ts = sorted(set(
        ts for sym in _SYMBOLS
        for ts in sig1h.get(sym, {})
        if from_ts <= ts <= to_ts
    ))

    equity = _CAPITAL
    peak   = _CAPITAL
    max_dd = 0.0
    positions: dict[str, dict] = {}
    trades:    list[dict]      = []
    year_pnl:  dict[int, float] = {}
    persist_streak: dict[str, int] = {s: 0 for s in _SYMBOLS}
    daily_equity: list[float] = []
    prev_day = None

    for ts in all_ts:
        day_ts  = (ts // _DAY_MS) * _DAY_MS
        btc_bear = btc_bear_by_day.get(day_ts, False)
        btc_bull = not btc_bear

        # Track daily equity for Sharpe
        if day_ts != prev_day:
            mtm = sum(
                (sig1h.get(sym, {}).get(ts, {}).get("close", pos["entry"]) - pos["entry"])
                / pos["entry"] * pos["notional"]
                for sym, pos in positions.items()
                if pos.get("side") == "LONG"
            )
            daily_equity.append(equity + mtm)
            prev_day = day_ts

        for sym in _SYMBOLS:
            sig = sig1h.get(sym, {}).get(ts, {})
            if not sig:
                continue
            c = sig["close"]
            ema_long  = sig.get("ema_long",  False)
            macd_long = sig.get("macd_long", False)
            any_long  = ema_long or macd_long
            in_pos = sym in positions and positions[sym].get("side") == "LONG"

            # Hard stop
            if in_pos:
                pos = positions[sym]
                loss = (pos["entry"] - c) / pos["entry"]
                if loss >= 0.15:
                    positions.pop(sym)
                    raw = (c - pos["entry"]) / pos["entry"] * pos["notional"]
                    net = raw - _TAKER_FEE * pos["notional"]
                    equity += net
                    yr = datetime.fromtimestamp(ts/1000, tz=timezone.utc).year
                    year_pnl[yr] = year_pnl.get(yr, 0) + net
                    trades.append({"net": net})
                    in_pos = False

            # Update peak / DD
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

            # Signal persistence tracking
            if any_long and btc_bull:
                persist_streak[sym] = persist_streak.get(sym, 0) + 1
            else:
                persist_streak[sym] = 0

            # Regime gate
            can_long = btc_bull

            # Momentum gate (daily, same as V8.63)
            if can_long and not sig.get("sma200_above", True):
                mom20 = daily_mom20.get(sym, {}).get(day_ts, 0.0)
                if mom20 <= 0:
                    can_long = False

            # Signal persistence gate
            if persist_hours > 0 and persist_streak.get(sym, 0) < persist_hours:
                can_long = False

            # Exit if signal gone or regime changed
            if in_pos and (not any_long or not can_long):
                pos = positions.pop(sym)
                raw = (c - pos["entry"]) / pos["entry"] * pos["notional"]
                net = raw - _TAKER_FEE * pos["notional"]
                equity += net
                yr = datetime.fromtimestamp(ts/1000, tz=timezone.utc).year
                year_pnl[yr] = year_pnl.get(yr, 0) + net
                trades.append({"net": net})
                in_pos = False

            # Entry
            if not in_pos and any_long and can_long:
                equity -= _TAKER_FEE * base_notional
                positions[sym] = {"entry": c, "notional": base_notional, "side": "LONG"}

    # Close remaining
    last_ts = all_ts[-1] if all_ts else to_ts
    for sym, pos in list(positions.items()):
        c = sig1h.get(sym, {}).get(last_ts, {}).get("close", pos["entry"])
        raw = (c - pos["entry"]) / pos["entry"] * pos["notional"]
        net = raw - _TAKER_FEE * pos["notional"]
        equity += net
        yr = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).year
        year_pnl[yr] = year_pnl.get(yr, 0) + net

    # Metrics
    years = (to_ts - from_ts) / (_DAY_MS * 365.25)
    cagr  = (equity / _CAPITAL) ** (1 / years) - 1 if years > 0 and equity > 0 else 0.0
    pf    = 1.0
    wins  = [t["net"] for t in trades if t["net"] > 0]
    loss  = [t["net"] for t in trades if t["net"] <= 0]
    if wins and loss:
        pf = sum(wins) / abs(sum(loss))

    # Sharpe from daily equity
    if len(daily_equity) > 2:
        rets = [(daily_equity[i] - daily_equity[i-1]) / daily_equity[i-1]
                for i in range(1, len(daily_equity)) if daily_equity[i-1] > 0]
        if rets:
            mean = sum(rets) / len(rets)
            std  = (sum((r - mean)**2 for r in rets) / len(rets)) ** 0.5
            sharpe = (mean / std) * (252 ** 0.5) if std > 0 else 0.0
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    return {
        "equity":   equity,
        "cagr":     cagr,
        "max_dd":   max_dd,
        "sharpe":   sharpe,
        "pf":       pf,
        "year_pnl": year_pnl,
        "n_trades": len(trades),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_daily_baseline(from_ts: int, to_ts: int) -> dict:
    sig = B._build_signals(B._SYMBOLS)
    return B._run(sig, True, True, False, True, from_ts, to_ts, **_BASE_DAILY)


def _year_table(rows: list[tuple[str, dict]]):
    all_years = sorted({yr for _, r in rows for yr in r["year_pnl"]})
    col = 12
    print(f"\n  {'System':<36}" + "".join(f"{yr:>{col}}" for yr in all_years)
          + f"{'Total Eq':>{col}}  {'CAGR':>7}  {'DD':>6}  {'Sh':>5}  {'Trades':>7}")
    print("  " + "-" * (36 + col * len(all_years) + col + 34))
    for label, r in rows:
        s = f"  {label:<36}"
        for yr in all_years:
            s += f"{r['year_pnl'].get(yr, 0):>+{col},.0f}"
        s += (f"  ${r['equity']:>9,.0f}  {r['cagr']*100:>6.1f}%"
              f"  {r['max_dd']*100:>5.1f}%  {r['sharpe']:>4.2f}  {r.get('n_trades',0):>7,}")
        print(s)


def main():
    print("=" * 110)
    print("  1H SYSTEM TEST — can hourly signals beat daily V8.63?")
    print("=" * 110)

    sig1h, btc_mom30, daily_mom20 = _build_1h_signals()
    print("  Done.\n")

    def r1h(from_ts, to_ts, **kw):
        return _run_1h(sig1h, btc_mom30, daily_mom20, from_ts, to_ts, **kw)

    print("  Building daily baseline signals ...")
    base_full = _run_daily_baseline(_FROM, _TO)
    base_h1   = _run_daily_baseline(_H1A, _H1B)
    print("  Done.\n")

    rows_full = [("V8.63 daily (baseline)", base_full)]
    rows_h1   = [("V8.63 daily (baseline)", base_h1)]

    variants = [
        ("H1. 1H signals, daily regime",      dict()),
        ("H2. 1H + 3H persist before entry",  dict(persist_hours=3)),
        ("H3. 1H + 6H persist before entry",  dict(persist_hours=6)),
        ("H4. 1H + 12H persist before entry", dict(persist_hours=12)),
    ]

    for label, kw in variants:
        rf = r1h(_FROM, _TO, **kw)
        rh = r1h(_H1A, _H1B, **kw)
        rows_full.append((label, rf))
        rows_h1.append((label, rh))
        print(f"  {label}: done  ({rf['n_trades']:,} trades)")

    print("\n" + "=" * 110)
    print("  YEAR-BY-YEAR EARNINGS — $5K starting, 2021 → 2026")
    print("=" * 110)
    _year_table(rows_full)

    print("\n  H1-2025 isolated window:")
    for label, rh in rows_h1:
        print(f"    {label:<36} CAGR={rh['cagr']*100:>+6.1f}%  DD={rh['max_dd']*100:>4.1f}%  "
              f"trades={rh.get('n_trades', 0):,}")

    print("\n" + "=" * 110)
    print("  KEY: if 1H trades 10x more but earns similar → fees are eating the edge.")
    print("  Demand: higher CAGR AND better H1-2025 AND walk-fwd ≥ 5/6 to consider.")
    print("=" * 110)

    # Walk-forward
    print("\n  Walk-forward (2yr train / 6mo test) ...")
    train = int(2 * 365.25 * _DAY_MS)
    test  = int(0.5 * 365.25 * _DAY_MS)
    for label, kw in [("V8.63 daily (baseline)", None)] + list(variants):
        ws = _FROM; npass = ntot = 0; cagrs = []
        while ws + train + test <= _TO:
            te  = ws + train
            tee = min(te + test, _TO)
            if kw is None:
                r = _run_daily_baseline(te, tee)
            else:
                r = r1h(te, tee, **kw)
            ok = r["pf"] >= 1.10 and r["cagr"] > 0
            npass += ok; ntot += 1; cagrs.append(r["cagr"] * 100)
            ws += test
        avg = sum(cagrs) / len(cagrs) if cagrs else 0
        print(f"  {label:<36} {npass}/{ntot}  avg OOS CAGR {avg:>+6.1f}%  "
              f"windows={['%+.0f%%' % c for c in cagrs]}")

    print()


if __name__ == "__main__":
    main()
