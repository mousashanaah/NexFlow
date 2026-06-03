#!/usr/bin/env python3
"""BTC+ETH Short-Enabled vs Long-Only comparison.

Capital: $100,000 → $50,000/coin (BTC + ETH only), FIXED notional.
Fee:     0.06% taker each side
Period:  2021-01-01 to now
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
_NOTIONAL   = 50_000.0
_SYMBOLS    = ["BTCUSDT", "ETHUSDT"]
_DAY_MS     = 86_400_000
_HOUR_MS    = 3_600_000
_IS_TS      = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _load_daily(symbol):
    tbl = pq.read_table(_CANDLE_DIR / f"{symbol}_1D.parquet", columns=["open_time","close"])
    rows = [{"ts":int(ts),"close":float(c)} for ts,c in zip(tbl.column("open_time").to_pylist(),tbl.column("close").to_pylist())]
    return sorted(rows, key=lambda x: x["ts"])


def _load_4h(symbol):
    tbl = pq.read_table(_CANDLE_DIR / f"{symbol}_1H.parquet", columns=["open_time","close"])
    rows = [{"ts":int(ts),"close":float(c)} for ts,c in zip(tbl.column("open_time").to_pylist(),tbl.column("close").to_pylist())]
    rows.sort(key=lambda x: x["ts"])
    buckets = {}
    for b in rows:
        hour = (b["ts"] % _DAY_MS) // _HOUR_MS
        bts = (b["ts"] // _DAY_MS) * _DAY_MS + (hour // 4) * 4 * _HOUR_MS
        buckets.setdefault(bts, []).append(b)
    result = []
    for bts in sorted(buckets):
        grp = sorted(buckets[bts], key=lambda x: x["ts"])
        if len(grp) >= 4:
            result.append({"ts": bts, "close": grp[-1]["close"]})
    return result


def _ema_series(closes, period):
    alpha = 2.0 / (period + 1)
    result = [None] * len(closes)
    ema = None
    for i, c in enumerate(closes):
        ema = alpha * c + (1 - alpha) * ema if ema is not None else c
        if i >= period - 1:
            result[i] = ema
    return result


def _backtest_symbol(bars, fast, slow, allow_short, from_ts, to_ts):
    closes = [b["close"] for b in bars]
    ef = _ema_series(closes, fast)
    es = _ema_series(closes, slow)
    total_pnl = 0.0
    trades = []
    position = None
    entry_price = 0.0
    entry_ts = 0
    prev_above = None

    for i, b in enumerate(bars):
        if b["ts"] < from_ts or b["ts"] > to_ts:
            if ef[i] is not None and es[i] is not None:
                prev_above = ef[i] > es[i]
            continue
        if ef[i] is None or es[i] is None:
            continue
        above = ef[i] > es[i]
        desired = "LONG" if above else ("SHORT" if allow_short else None)

        if prev_above is not None and above == prev_above:
            prev_above = above
            continue
        prev_above = above

        if position is not None:
            dir_ = 1 if position == "LONG" else -1
            raw  = (b["close"] - entry_price) / entry_price * dir_ * _NOTIONAL
            net  = raw - 2 * _TAKER_FEE * _NOTIONAL
            total_pnl += net
            trades.append({"side": position, "entry": entry_price, "exit": b["close"],
                           "net": net, "ts_in": entry_ts, "ts_out": b["ts"]})
            position = None

        if desired is not None:
            total_pnl -= _TAKER_FEE * _NOTIONAL
            entry_price = b["close"]
            entry_ts    = b["ts"]
            position    = desired

    if position is not None and bars:
        last = bars[-1]
        dir_ = 1 if position == "LONG" else -1
        raw  = (last["close"] - entry_price) / entry_price * dir_ * _NOTIONAL
        net  = raw - _TAKER_FEE * _NOTIONAL
        total_pnl += net
        trades.append({"side": position, "entry": entry_price, "exit": last["close"],
                       "net": net, "ts_in": entry_ts, "ts_out": last["ts"]})
    return total_pnl, trades


def _report(label, allow_short, all_trades, from_ts, to_ts):
    all_trades.sort(key=lambda t: t["ts_in"])
    equity = _CAPITAL; peak = _CAPITAL; max_dd = 0.0
    wins = losses = 0; gross_win = gross_loss = 0.0
    year_pnl = {}
    for t in all_trades:
        equity += t["net"]
        if equity > peak: peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd: max_dd = dd
        if t["net"] > 0: wins += 1; gross_win += t["net"]
        else: losses += 1; gross_loss += abs(t["net"])
        yr = datetime.fromtimestamp(t["ts_in"]/1000, tz=timezone.utc).year
        year_pnl[yr] = year_pnl.get(yr, 0.0) + t["net"]
    n   = len(all_trades)
    pf  = gross_win / gross_loss if gross_loss > 0 else float("inf")
    wr  = wins / n * 100 if n else 0
    net = equity - _CAPITAL
    years = (to_ts - from_ts) / (1_000 * 86_400 * 365.25)
    cagr  = (equity / _CAPITAL) ** (1 / years) - 1 if years > 0 and equity > 0 else -1.0
    def _pf(ts): gw=sum(t["net"] for t in ts if t["net"]>0); gl=sum(abs(t["net"]) for t in ts if t["net"]<0); return gw/gl if gl>0 else float("inf")
    is_pf  = _pf([t for t in all_trades if t["ts_in"] <  _IS_TS])
    oos_pf = _pf([t for t in all_trades if t["ts_in"] >= _IS_TS])
    long_t  = [t for t in all_trades if t["side"] == "LONG"]
    short_t = [t for t in all_trades if t["side"] == "SHORT"]

    mode = "Long+Short" if allow_short else "Long-Only"
    print(f"\n{'='*60}")
    print(f"  {label} — {mode}")
    print(f"{'='*60}")
    print(f"  Trades : {n}  (WR={wr:.1f}%,  wins={wins}, losses={losses})")
    print(f"  PF     : {pf:.2f}")
    print(f"  Net PnL: ${net:+,.0f}  ({net/_CAPITAL*100:.1f}%)")
    print(f"  CAGR   : {cagr*100:.1f}%")
    print(f"  Max DD : {max_dd*100:.1f}%")
    print(f"  IS  PF : {is_pf:.2f}  ({len([t for t in all_trades if t['ts_in']<_IS_TS])} trades)")
    print(f"  OOS PF : {oos_pf:.2f}  ({len([t for t in all_trades if t['ts_in']>=_IS_TS])} trades)")
    if allow_short:
        print(f"  LONG   : ${sum(t['net'] for t in long_t):+,.0f}  PF={_pf(long_t):.2f}  ({len(long_t)} trades)")
        print(f"  SHORT  : ${sum(t['net'] for t in short_t):+,.0f}  PF={_pf(short_t):.2f}  ({len(short_t)} trades)")
    print("  Year-by-year:")
    for yr in sorted(year_pnl):
        print(f"    {yr}: ${year_pnl[yr]:>+10,.0f}")
    verdict = "✓ GO" if pf>=1.30 and max_dd<=0.40 and n>=60 and cagr>=0.15 else ("MARGINAL" if pf>=1.10 and max_dd<=0.50 and n>=60 else "KILL")
    print(f"\n  VERDICT: {verdict}  (PF={pf:.2f}, CAGR={cagr*100:.1f}%, DD={max_dd*100:.1f}%)")


def main():
    from_ts = int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp()*1000)
    to_ts   = int(datetime.now(timezone.utc).timestamp()*1000)

    print("="*60)
    print("  BTC + ETH: Shorts vs Long-Only  (fixed $50K notional/coin)")
    print(f"  2021-01-01 → {datetime.now(timezone.utc).date()}")
    print("="*60)

    daily = {sym: _load_daily(sym) for sym in _SYMBOLS}
    h4    = {sym: _load_4h(sym)    for sym in _SYMBOLS}

    for allow_short in [False, True]:
        trades = []
        for sym in _SYMBOLS:
            _, t = _backtest_symbol(daily[sym], 8, 21, allow_short, from_ts, to_ts)
            trades.extend(t)
        _report("Daily EMA 8/21", allow_short, trades, from_ts, to_ts)

    for allow_short in [False, True]:
        trades = []
        for sym in _SYMBOLS:
            _, t = _backtest_symbol(h4[sym], 5, 13, allow_short, from_ts, to_ts)
            trades.extend(t)
        _report("4H EMA 5/13", allow_short, trades, from_ts, to_ts)


if __name__ == "__main__":
    main()
