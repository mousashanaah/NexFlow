#!/usr/bin/env python3
"""Confluence position sizing backtest.

Tests whether scaling position size by strategy agreement improves returns:
  1 strategy LONG  → 1.0× base notional
  2 strategies LONG → 1.5× base notional
  3 strategies LONG → 2.0× base notional

Compares against flat equal-weight sizing.

Universe: 12 coins, $100K capital, 2021-2026
Strategies: EMA 8/21 daily + MACD 12/26/9 daily + 4H EMA 5/13
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
except ImportError:
    print("[ERROR] pip install pyarrow"); sys.exit(1)

_CANDLE_DIR = _REPO_ROOT / "data" / "candles"
_TAKER_FEE  = 0.0006
_CAPITAL    = 100_000.0
_SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
    "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT",
    "LINKUSDT","LTCUSDT","DOTUSDT","TRXUSDT",
]
_DAY_MS  = 86_400_000
_HOUR_MS = 3_600_000
_IS_TS   = int(datetime(2023,1,1,tzinfo=timezone.utc).timestamp()*1000)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def _ema_series(closes: list[float], period: int) -> list[Optional[float]]:
    alpha = 2.0 / (period + 1)
    result: list[Optional[float]] = [None] * len(closes)
    ema = None
    for i, c in enumerate(closes):
        ema = alpha * c + (1 - alpha) * ema if ema is not None else c
        if i >= period - 1:
            result[i] = ema
    return result


def _macd_histogram(closes: list[float], fast=12, slow=26, signal=9) -> list[Optional[float]]:
    ef = _ema_series(closes, fast)
    es = _ema_series(closes, slow)
    macds = [ef[i] - es[i] if ef[i] is not None and es[i] is not None else None
             for i in range(len(closes))]
    sig_in = [m for m in macds if m is not None]
    sig_out_raw = _ema_series(sig_in, signal)
    sig_out: list[Optional[float]] = []
    idx = 0
    for m in macds:
        if m is None:
            sig_out.append(None)
        else:
            sig_out.append(sig_out_raw[idx])
            idx += 1
    return [macds[i] - sig_out[i] if macds[i] is not None and sig_out[i] is not None else None
            for i in range(len(closes))]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load_daily(symbol: str) -> list[dict]:
    path = _CANDLE_DIR / f"{symbol}_1D.parquet"
    if not path.exists(): return []
    tbl = pq.read_table(path, columns=["open_time","close"])
    rows = [{"ts":int(ts),"close":float(c)}
            for ts,c in zip(tbl.column("open_time").to_pylist(),
                            tbl.column("close").to_pylist())]
    return sorted(rows, key=lambda x: x["ts"])


def _load_4h(symbol: str) -> list[dict]:
    path = _CANDLE_DIR / f"{symbol}_1H.parquet"
    if not path.exists(): return []
    tbl = pq.read_table(path, columns=["open_time","close"])
    rows = [{"ts":int(ts),"close":float(c)}
            for ts,c in zip(tbl.column("open_time").to_pylist(),
                            tbl.column("close").to_pylist())]
    rows.sort(key=lambda x: x["ts"])
    buckets: dict[int,list] = {}
    for b in rows:
        hour = (b["ts"] % _DAY_MS) // _HOUR_MS
        bts = (b["ts"] // _DAY_MS)*_DAY_MS + (hour//4)*4*_HOUR_MS
        buckets.setdefault(bts,[]).append(b)
    result = []
    for bts in sorted(buckets):
        grp = sorted(buckets[bts], key=lambda x: x["ts"])
        if len(grp) >= 4:
            result.append({"ts":bts,"close":grp[-1]["close"]})
    return result


# ---------------------------------------------------------------------------
# Per-symbol signal generation
# ---------------------------------------------------------------------------
def _symbol_signals(
    symbol: str,
    from_ts: int,
    to_ts: int,
) -> dict[int, dict]:
    """Returns {ts_ms: {"ema":bool, "macd":bool, "h4":bool, "close":float}}
    for daily bars, indicating which strategies are LONG at each bar.
    """
    daily = _load_daily(symbol)
    h4    = _load_4h(symbol)

    if not daily:
        return {}

    closes_d = [b["close"] for b in daily]

    # EMA 8/21 on daily
    ef8  = _ema_series(closes_d, 8)
    ef21 = _ema_series(closes_d, 21)
    ema_long: list[bool] = [
        (ef8[i] is not None and ef21[i] is not None and ef8[i] > ef21[i])
        for i in range(len(closes_d))
    ]

    # MACD on daily — LONG when histogram crosses above 0
    hist = _macd_histogram(closes_d)
    macd_long: list[bool] = [False] * len(closes_d)
    in_long = False
    for i in range(1, len(closes_d)):
        if hist[i-1] is not None and hist[i] is not None:
            if hist[i-1] <= 0 < hist[i]:
                in_long = True
            elif hist[i-1] >= 0 > hist[i]:
                in_long = False
        macd_long[i] = in_long

    # 4H EMA 5/13 — LONG when fast > slow AND daily EMA agrees
    closes_4h = [b["close"] for b in h4]
    ef5  = _ema_series(closes_4h, 5)
    ef13 = _ema_series(closes_4h, 13)
    h4_long_by_ts: dict[int,bool] = {}
    h4_in_long = False
    for i in range(1, len(h4)):
        if ef5[i] is None or ef13[i] is None: continue
        if ef5[i-1] is None or ef13[i-1] is None: continue
        prev_above = ef5[i-1] > ef13[i-1]
        curr_above = ef5[i]   > ef13[i]
        if curr_above != prev_above:
            if curr_above: h4_in_long = True
            else:          h4_in_long = False
        h4_long_by_ts[h4[i]["ts"]] = h4_in_long

    # Combine into daily resolution
    result: dict[int, dict] = {}
    for i, b in enumerate(daily):
        if b["ts"] < from_ts or b["ts"] > to_ts:
            continue
        # Latest 4H state at or before this daily close
        day_end = b["ts"] + _DAY_MS - 1
        h4_state = False
        for ts4 in sorted(h4_long_by_ts):
            if ts4 <= day_end:
                h4_state = h4_long_by_ts[ts4]
        # 4H only valid in daily bull trend
        h4_filtered = h4_state and ema_long[i]

        result[b["ts"]] = {
            "close": b["close"],
            "ema":   ema_long[i],
            "macd":  macd_long[i],
            "h4":    h4_filtered,
        }
    return result


# ---------------------------------------------------------------------------
# Portfolio backtest with sizing variants
# ---------------------------------------------------------------------------
def _backtest(
    symbols: list[str],
    capital: float,
    from_ts: int,
    to_ts: int,
    sizing: str,  # "flat" | "confluence"
) -> dict:
    """
    sizing="flat": fixed 1/12 of capital per coin per strategy slot
    sizing="confluence": scales by number of agreeing strategies (1x/1.5x/2x)
    """
    base_notional = capital / len(symbols)

    # Build all signals first
    all_signals: dict[str, dict[int, dict]] = {}
    for sym in symbols:
        all_signals[sym] = _symbol_signals(sym, from_ts, to_ts)

    # Simulate day by day
    all_ts = sorted(set(ts for sym_sigs in all_signals.values() for ts in sym_sigs))

    equity = capital
    peak   = capital
    max_dd = 0.0
    # Track open positions per symbol: {"entry_price": float, "notional": float, "strategies": set}
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    year_pnl: dict[int, float] = {}

    for ts in all_ts:
        for sym in symbols:
            sigs = all_signals[sym].get(ts)
            if sigs is None:
                continue

            close   = sigs["close"]
            n_long  = sum([sigs["ema"], sigs["macd"], sigs["h4"]])
            any_long = n_long > 0

            if sizing == "confluence":
                mult = {0: 0.0, 1: 1.0, 2: 1.5, 3: 2.0}[n_long]
                target_notional = base_notional * mult
            else:
                target_notional = base_notional if any_long else 0.0

            in_pos = sym in positions

            if in_pos and target_notional == 0.0:
                # Close position
                pos = positions.pop(sym)
                pnl = (close - pos["entry"]) / pos["entry"] * pos["notional"]
                fee = _TAKER_FEE * pos["notional"]
                net = pnl - fee
                equity += net
                trades.append({"ts": ts, "sym": sym, "net": net, "notional": pos["notional"]})
                yr = datetime.fromtimestamp(ts/1000, tz=timezone.utc).year
                year_pnl[yr] = year_pnl.get(yr, 0.0) + net

            elif not in_pos and target_notional > 0.0:
                # Open position
                fee = _TAKER_FEE * target_notional
                equity -= fee
                positions[sym] = {"entry": close, "notional": target_notional, "n_strats": n_long}

            elif in_pos and sizing == "confluence":
                # Resize if confluence changed
                pos = positions[sym]
                if abs(target_notional - pos["notional"]) > pos["notional"] * 0.4:
                    # Close and reopen at new size (only on big changes)
                    pnl = (close - pos["entry"]) / pos["entry"] * pos["notional"]
                    fee_close = _TAKER_FEE * pos["notional"]
                    fee_open  = _TAKER_FEE * target_notional
                    net = pnl - fee_close - fee_open
                    equity += net
                    trades.append({"ts": ts, "sym": sym, "net": net, "notional": pos["notional"]})
                    yr = datetime.fromtimestamp(ts/1000, tz=timezone.utc).year
                    year_pnl[yr] = year_pnl.get(yr, 0.0) + net
                    positions[sym] = {"entry": close, "notional": target_notional, "n_strats": n_long}

        # Update equity curve for DD
        if equity > peak: peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd: max_dd = dd

    # Close open positions at last price
    last_prices = {}
    for sym in symbols:
        sigs = all_signals[sym]
        if sigs:
            last_ts = max(sigs.keys())
            last_prices[sym] = sigs[last_ts]["close"]

    unrealised = 0.0
    for sym, pos in positions.items():
        p = last_prices.get(sym, pos["entry"])
        unrealised += (p - pos["entry"]) / pos["entry"] * pos["notional"]

    total_eq = equity + unrealised
    net_total = total_eq - capital
    years = (to_ts - from_ts) / (1_000 * 86_400 * 365.25)
    cagr  = (total_eq/capital)**(1/years) - 1 if years > 0 and total_eq > 0 else -1.0

    is_t  = [t for t in trades if t["ts"] <  _IS_TS]
    oos_t = [t for t in trades if t["ts"] >= _IS_TS]
    def _pf(ts):
        gw = sum(t["net"] for t in ts if t["net"] > 0)
        gl = sum(abs(t["net"]) for t in ts if t["net"] < 0)
        return gw/gl if gl > 0 else float("inf")

    # Confluence distribution
    conf_counts = {1:0, 2:0, 3:0}
    conf_pnl    = {1:0.0, 2:0.0, 3:0.0}
    for t in trades:
        n = t.get("notional", base_notional)
        tier = round(n / base_notional * 1.0)
        if n <= base_notional * 1.25:  tier = 1
        elif n <= base_notional * 1.75: tier = 2
        else:                           tier = 3
        conf_counts[tier] = conf_counts.get(tier, 0) + 1
        conf_pnl[tier]    = conf_pnl.get(tier, 0.0) + t["net"]

    return {
        "equity": total_eq,
        "net": net_total,
        "cagr": cagr,
        "max_dd": max_dd,
        "n_trades": len(trades),
        "pf": _pf(trades),
        "is_pf": _pf(is_t),
        "oos_pf": _pf(oos_t),
        "year_pnl": year_pnl,
        "conf_counts": conf_counts,
        "conf_pnl": conf_pnl,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    from_ts = int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp()*1000)
    to_ts   = int(datetime.now(timezone.utc).timestamp()*1000)

    print("Confluence Sizing Backtest — 2021 to present")
    print(f"Universe: {len(_SYMBOLS)} coins  |  Capital: ${_CAPITAL:,.0f}")
    print()

    for sizing in ["flat", "confluence"]:
        label = "Flat sizing (1× always)" if sizing == "flat" else "Confluence sizing (1×/1.5×/2×)"
        print(f"{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")
        r = _backtest(_SYMBOLS, _CAPITAL, from_ts, to_ts, sizing)
        print(f"  Final equity : ${r['equity']:>12,.0f}")
        print(f"  Net PnL      : ${r['net']:>+12,.0f}  ({r['net']/_CAPITAL*100:.1f}%)")
        print(f"  CAGR         : {r['cagr']*100:.1f}%")
        print(f"  Max DD       : {r['max_dd']*100:.1f}%")
        print(f"  Profit Factor: {r['pf']:.2f}")
        print(f"  Trades       : {r['n_trades']}")
        print(f"  IS  PF       : {r['is_pf']:.2f}")
        print(f"  OOS PF       : {r['oos_pf']:.2f}")
        print()
        print("  Year-by-year:")
        for yr in sorted(r["year_pnl"]):
            print(f"    {yr}: ${r['year_pnl'][yr]:>+10,.0f}")
        if sizing == "confluence":
            print()
            print("  Confluence breakdown:")
            for tier, mult in [(1,"1.0×"),(2,"1.5×"),(3,"2.0×")]:
                n   = r["conf_counts"].get(tier,0)
                pnl = r["conf_pnl"].get(tier,0.0)
                print(f"    {mult} notional ({tier} strat): {n:>4} closes  ${pnl:>+10,.0f}")
        print()


if __name__ == "__main__":
    main()
