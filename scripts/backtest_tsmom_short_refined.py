#!/usr/bin/env python3
"""TSMOM Short-Only refined — iterating on mechanism B.

Base mechanism: 126-day return < threshold → short that coin.
Refinements tested:
  1. BTC SMA200 master switch (only short when BTC < SMA200)
  2. Higher threshold (-10%, -15%, -20%) to filter noise
  3. Monthly vs weekly rebalance
  4. Worst-N coins only (bottom 2-3 vs all 12)

Universe: 12 coins, $100K, 2021-2026
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
_IS_TS   = int(datetime(2023,1,1,tzinfo=timezone.utc).timestamp()*1000)


def _load_daily(symbol: str) -> list[dict]:
    path = _CANDLE_DIR / f"{symbol}_1D.parquet"
    if not path.exists(): return []
    tbl = pq.read_table(path, columns=["open_time","close"])
    rows = [{"ts":int(ts),"close":float(c)}
            for ts,c in zip(tbl.column("open_time").to_pylist(),
                            tbl.column("close").to_pylist())]
    return sorted(rows, key=lambda x: x["ts"])


def _sma(values: list[float], period: int) -> list[Optional[float]]:
    result: list[Optional[float]] = [None] * len(values)
    for i in range(period-1, len(values)):
        result[i] = sum(values[i-period+1:i+1]) / period
    return result


def _backtest(
    lookback: int,
    threshold: float,      # e.g. -0.10 = -10%
    rebal_days: int,       # 7=weekly, 30=monthly
    max_shorts: int,       # max coins to short at once (12=all, 3=worst 3)
    btc_sma_filter: bool,  # only short when BTC < SMA200
    from_ts: int,
    to_ts: int,
) -> dict:
    # Load all data
    all_bars: dict[str, list[dict]] = {}
    for sym in _SYMBOLS:
        all_bars[sym] = _load_daily(sym)

    # BTC SMA200 for master filter
    btc_bars = all_bars["BTCUSDT"]
    btc_closes = [b["close"] for b in btc_bars]
    btc_sma200 = _sma(btc_closes, 200)
    btc_sma_by_ts = {btc_bars[i]["ts"]: btc_sma200[i]
                     for i in range(len(btc_bars)) if btc_sma200[i] is not None}

    def _btc_bear(ts: int) -> bool:
        """True if BTC is in bear (below SMA200) at this timestamp."""
        if not btc_sma_filter:
            return True
        best_ts = -1; best_val = True
        for dts, sma in btc_sma_by_ts.items():
            if dts <= ts and dts > best_ts:
                best_ts = dts
                best_val = btc_bars[[b["ts"] for b in btc_bars].index(dts)]["close"] < sma
        return best_val

    # Get all unique timestamps in range
    all_ts = sorted(set(
        b["ts"] for sym in _SYMBOLS for b in all_bars[sym]
        if from_ts <= b["ts"] <= to_ts
    ))

    # Build close lookup
    close_by_sym_ts: dict[str, dict[int, float]] = {}
    for sym in _SYMBOLS:
        close_by_sym_ts[sym] = {b["ts"]: b["close"] for b in all_bars[sym]}

    notional = _CAPITAL / len(_SYMBOLS)
    equity = _CAPITAL
    peak   = _CAPITAL
    max_dd = 0.0
    positions: dict[str, dict] = {}  # sym → {entry, notional}
    trades: list[dict] = []
    year_pnl: dict[int, float] = {}
    last_rebal = 0

    for ts in all_ts:
        # Rebalance signal
        do_rebal = (ts - last_rebal) >= rebal_days * _DAY_MS

        if do_rebal:
            last_rebal = ts
            bear_regime = _btc_bear(ts)

            # Compute 126-day returns for all coins
            scores: list[tuple[float, str]] = []  # (return, symbol)
            for sym in _SYMBOLS:
                closes = [(t, c) for t, c in close_by_sym_ts[sym].items() if t <= ts]
                closes.sort()
                if len(closes) < lookback + 1:
                    continue
                current_close = closes[-1][1]
                past_close    = closes[-lookback-1][1]
                ret = (current_close - past_close) / past_close
                scores.append((ret, sym))

            scores.sort()  # most negative first = best short candidates

            # Determine desired shorts
            desired_shorts: set[str] = set()
            if bear_regime:
                count = 0
                for ret, sym in scores:
                    if ret < threshold and count < max_shorts:
                        desired_shorts.add(sym)
                        count += 1

            # Close positions no longer desired
            for sym in list(positions.keys()):
                if sym not in desired_shorts:
                    close_price = close_by_sym_ts[sym].get(ts)
                    if close_price is None: continue
                    pos = positions.pop(sym)
                    raw = (pos["entry"] - close_price) / pos["entry"] * pos["notional"]
                    net = raw - _TAKER_FEE * pos["notional"]
                    equity += net
                    trades.append({"ts":ts,"sym":sym,"net":net})
                    yr = datetime.fromtimestamp(ts/1000,tz=timezone.utc).year
                    year_pnl[yr] = year_pnl.get(yr,0.0) + net

            # Open new desired positions
            for sym in desired_shorts:
                if sym not in positions:
                    close_price = close_by_sym_ts[sym].get(ts)
                    if close_price is None: continue
                    equity -= _TAKER_FEE * notional
                    positions[sym] = {"entry": close_price, "notional": notional}

        # Track equity/DD
        if equity > peak: peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd: max_dd = dd

    # Close all remaining positions
    last_ts = all_ts[-1] if all_ts else to_ts
    for sym, pos in list(positions.items()):
        close_price = close_by_sym_ts[sym].get(last_ts, pos["entry"])
        raw = (pos["entry"] - close_price) / pos["entry"] * pos["notional"]
        net = raw - _TAKER_FEE * pos["notional"]
        equity += net
        trades.append({"ts":last_ts,"sym":sym,"net":net})
        yr = datetime.fromtimestamp(last_ts/1000,tz=timezone.utc).year
        year_pnl[yr] = year_pnl.get(yr,0.0) + net

    total_eq = equity
    net_pnl  = total_eq - _CAPITAL
    years    = (to_ts - from_ts) / (1_000*86_400*365.25)
    cagr     = (total_eq/_CAPITAL)**(1/years)-1 if years>0 and total_eq>0 else -1.0

    gw = sum(t["net"] for t in trades if t["net"]>0)
    gl = sum(abs(t["net"]) for t in trades if t["net"]<0)
    pf = gw/gl if gl>0 else float("inf")

    is_t  = [t for t in trades if t["ts"] <  _IS_TS]
    oos_t = [t for t in trades if t["ts"] >= _IS_TS]
    def _pf(ts): gw=sum(t["net"] for t in ts if t["net"]>0); gl=sum(abs(t["net"]) for t in ts if t["net"]<0); return gw/gl if gl>0 else float("inf")

    return {
        "pf": pf, "cagr": cagr, "max_dd": max_dd,
        "n": len(trades), "net": net_pnl,
        "is_pf": _pf(is_t), "oos_pf": _pf(oos_t),
        "year_pnl": year_pnl,
    }


def _label(lookback, threshold, rebal, max_s, sma_filter):
    return (f"lookback={lookback}d  thresh={threshold*100:.0f}%  "
            f"rebal={rebal}d  top{max_s}  btc_sma={'ON' if sma_filter else 'off'}")


def main():
    from_ts = int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp()*1000)
    to_ts   = int(datetime.now(timezone.utc).timestamp()*1000)

    print("TSMOM Short-Only — Refined Parameter Search")
    print(f"Capital: ${_CAPITAL:,.0f}  |  ${_CAPITAL/12:,.0f}/coin  |  2021-2026")
    print()

    variants = [
        # baseline
        dict(lookback=126, threshold=-0.05, rebal_days=7,  max_shorts=12, btc_sma_filter=False),
        # BTC SMA200 filter only
        dict(lookback=126, threshold=-0.05, rebal_days=7,  max_shorts=12, btc_sma_filter=True),
        # Higher threshold
        dict(lookback=126, threshold=-0.10, rebal_days=7,  max_shorts=12, btc_sma_filter=True),
        dict(lookback=126, threshold=-0.15, rebal_days=7,  max_shorts=12, btc_sma_filter=True),
        dict(lookback=126, threshold=-0.20, rebal_days=7,  max_shorts=12, btc_sma_filter=True),
        # Monthly rebalance
        dict(lookback=126, threshold=-0.15, rebal_days=30, max_shorts=12, btc_sma_filter=True),
        # Worst coins only
        dict(lookback=126, threshold=-0.15, rebal_days=30, max_shorts=5,  btc_sma_filter=True),
        dict(lookback=126, threshold=-0.15, rebal_days=30, max_shorts=3,  btc_sma_filter=True),
        # Longer lookback
        dict(lookback=180, threshold=-0.15, rebal_days=30, max_shorts=5,  btc_sma_filter=True),
        dict(lookback=180, threshold=-0.20, rebal_days=30, max_shorts=5,  btc_sma_filter=True),
        # Best combo attempt
        dict(lookback=180, threshold=-0.20, rebal_days=30, max_shorts=3,  btc_sma_filter=True),
    ]

    best_pf = 0.0
    best_label = ""
    best_result = None

    for v in variants:
        lbl = _label(v["lookback"], v["threshold"], v["rebal_days"], v["max_shorts"], v["btc_sma_filter"])
        r = _backtest(from_ts=from_ts, to_ts=to_ts, **v)
        verdict = "✓ GO" if r["pf"]>=1.10 and r["max_dd"]<=0.50 and r["n"]>=20 else "—"
        print(f"{verdict}  PF={r['pf']:.2f}  CAGR={r['cagr']*100:.1f}%  DD={r['max_dd']*100:.1f}%  "
              f"IS={r['is_pf']:.2f}  OOS={r['oos_pf']:.2f}  n={r['n']}")
        print(f"     {lbl}")
        bear_pnl = sum(r["year_pnl"].get(yr,0) for yr in [2022,2025,2026])
        bull_pnl = sum(r["year_pnl"].get(yr,0) for yr in [2021,2023,2024])
        yr_str = "  ".join(f"{yr}:${r['year_pnl'].get(yr,0):>+,.0f}" for yr in sorted(r["year_pnl"]))
        print(f"     {yr_str}")
        print(f"     Bear total: ${bear_pnl:>+,.0f}   Bull total: ${bull_pnl:>+,.0f}")
        print()
        if r["pf"] > best_pf and r["n"] >= 20:
            best_pf = r["pf"]; best_label = lbl; best_result = r

    if best_result:
        print("="*60)
        print(f"BEST: PF={best_pf:.2f}")
        print(f"  {best_label}")
        print(f"  Net PnL: ${best_result['net']:>+,.0f}  CAGR={best_result['cagr']*100:.1f}%")
        print(f"  Max DD : {best_result['max_dd']*100:.1f}%")
        print("  Year-by-year:")
        for yr in sorted(best_result["year_pnl"]):
            tag = "<<BEAR" if yr in [2022,2025,2026] else ("<<BULL" if yr in [2021,2024] else "")
            print(f"    {yr}: ${best_result['year_pnl'][yr]:>+10,.0f}  {tag}")


if __name__ == "__main__":
    main()
