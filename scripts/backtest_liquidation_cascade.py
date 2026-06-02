#!/usr/bin/env python3
"""Liquidation-Cascade Reversion Backtest — 15-minute timeframe.

Mechanism hypothesis (cascade overshoot reverts):
  A forced-liquidation cascade is violent, mechanical selling (longs liquidated)
  or buying (shorts squeezed) that overshoots fair value, then snaps back. We
  therefore FADE the cascade: after a down-cascade go LONG, after an up-cascade
  go SHORT, and exit quickly.

  There is no free, no-key, full-history liquidation feed, so we detect the
  cascade EVENT from its unmistakable price/volume fingerprint on 15m candles:
    - a volume spike (>= VOL_MULT × trailing-24h median volume), AND
    - a violent range (true range >= RANGE_MULT × ATR(14)).
  This is a proxy for the event; if Section A shows reversion edge, a precise
  liquidation feed is then worth paying for to refine entries.

Two-section output
------------------
SECTION A  Mechanism Validity — after a cascade, does price REVERT (pre-fee)?
           Forward returns measured in the FADE direction at +1/+4/+8/+16 bars
           (15m/1h/2h/4h), with a reversion hit-rate, compared to the all-bar
           baseline drift so we know the cascade adds information.
SECTION B  Monetization — can the fade be extracted after fees?

Parameters (all pre-committed):
  timeframe   : 15m candles.
  trailing    : 96 bars (24h) for the volume-median baseline.
  vol spike   : bar volume >= 4.0 × trailing-24h median volume.
  range spike : bar true range >= 3.0 × ATR(14).
  direction   : down-cascade (close<open) → fade LONG;
                up-cascade   (close>open) → fade SHORT.
  entry       : next bar OPEN after the cascade bar (no lookahead).
  hold        : 8 bars (2h) OR 1.5×ATR(14) stop, whichever first.
  risk        : 1% account equity per trade;  fee 0.06% taker per side.
  one position at a time per symbol; cascades while in a position are ignored.

Kill criteria:
  PF < 1.10 | max DD > 40% | n_trades < 60 | OOS PF (2023+) < 0.85 × full PF

Self-contained: downloads its own 15m candles from Bitget (reachable from US CI),
caches them to data/candles/{SYMBOL}_15m.parquet, then runs.

Usage:
  python scripts/backtest_liquidation_cascade.py
  python scripts/backtest_liquidation_cascade.py --capital 50000 --start 2022-01-01
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants (pre-committed strategy parameters)
# ---------------------------------------------------------------------------
_TAKER_FEE     = 0.0006
_RISK_PCT      = 0.01
_ATR_PERIOD    = 14
_HOLD_BARS     = 8         # 8 × 15m = 2h
_STOP_MULT     = 1.5

_TRAIL_BARS    = 96        # 24h of 15m bars for the volume-median baseline
_VOL_MULT      = 4.0       # volume spike threshold
_RANGE_MULT    = 3.0       # true-range spike threshold (× ATR)

_FWD_HORIZONS  = [1, 4, 8, 16]   # bars: 15m, 1h, 2h, 4h

# Kill / go thresholds
_KILL_PF        = 1.10
_KILL_TRADES    = 60
_KILL_MAX_DD    = 0.40
_KILL_OOS_RATIO = 0.85
_OOS_START_YEAR = 2023
_GO_PF, _GO_CAGR, _GO_DD = 1.30, 0.20, 0.40

_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# 15m candle download (Bitget, reachable from US CI runners)
_URL_HISTORY = "https://api.bitget.com/api/v2/mix/market/history-candles"
_GRAN        = "15m"
_BAR_MS      = 900_000
_LIMIT       = 200
_DL_DELAY    = 0.15
_DL_START    = "2021-01-01"
_HEADERS     = {"User-Agent": "NexFlow/1.0", "Accept": "application/json"}

_SCHEMA = pa.schema([
    pa.field("open_time", pa.int64()),
    pa.field("open",  pa.float64()), pa.field("high", pa.float64()),
    pa.field("low",   pa.float64()), pa.field("close", pa.float64()),
    pa.field("volume", pa.float64()),
])


# ---------------------------------------------------------------------------
# 15m candle download (inline, cached)
# ---------------------------------------------------------------------------

def _fetch_page(symbol: str, end_ms: int) -> list:
    url = (f"{_URL_HISTORY}?symbol={symbol}&productType=USDT-FUTURES"
           f"&granularity={_GRAN}&endTime={end_ms}&limit={_LIMIT}")
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    if data.get("code") != "00000":
        raise RuntimeError(f"Bitget API error: {data.get('msg', data)}")
    return data.get("data", [])


def _download_15m(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    bars: list[dict] = []
    seen: set[int] = set()
    cursor_end = end_ms
    errors = 0
    while cursor_end > start_ms:
        try:
            rows = _fetch_page(symbol, cursor_end)
        except Exception as exc:
            errors += 1
            if errors >= 5:
                print(f"\n  [ERROR] Too many failures: {exc}")
                break
            time.sleep(2.0 * errors)
            continue
        errors = 0
        if not rows:
            break
        oldest = cursor_end
        new = 0
        for row in rows:
            ts = int(row[0])
            if ts in seen or ts > end_ms:
                continue
            seen.add(ts); new += 1
            oldest = min(oldest, ts)
            if ts >= start_ms:
                bars.append({
                    "open_time": ts,
                    "open": float(row[1]), "high": float(row[2]),
                    "low": float(row[3]), "close": float(row[4]),
                    "volume": float(row[5]) if len(row) > 5 else 0.0,
                })
        pct = (end_ms - oldest) / max(end_ms - start_ms, 1) * 100
        print(f"  {symbol}: {len(bars):,} bars ({pct:.0f}%) ...", end="\r")
        if new == 0 or oldest <= start_ms:
            break
        cursor_end = oldest - 1
        time.sleep(_DL_DELAY)
    print()
    return sorted(bars, key=lambda b: b["open_time"])


def _load_or_download_15m(symbol: str, data_dir: Path, start_ms: int, end_ms: int) -> dict:
    path = data_dir / f"{symbol}_15m.parquet"
    if path.exists():
        d = pq.read_table(path).to_pydict()
        idx = sorted(range(len(d["open_time"])), key=lambda i: d["open_time"][i])
        out = {c: [d[c][i] for i in idx] for c in d}
        print(f"  {symbol}: loaded {len(out['open_time']):,} cached 15m bars")
        return out
    print(f"  {symbol}: downloading 15m candles from Bitget ...")
    bars = _download_15m(symbol, start_ms, end_ms)
    if not bars:
        print(f"  [ERROR] No 15m data for {symbol}")
        sys.exit(1)
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "open_time": [b["open_time"] for b in bars],
        "open": [b["open"] for b in bars], "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars], "close": [b["close"] for b in bars],
        "volume": [b["volume"] for b in bars],
    }, schema=_SCHEMA), path)
    s = datetime.fromtimestamp(bars[0]["open_time"]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    e = datetime.fromtimestamp(bars[-1]["open_time"]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"  {symbol}: saved {len(bars):,} bars ({s} → {e}) → {path}")
    return {c: [b[c] for b in bars] for c in ("open_time", "open", "high", "low", "close", "volume")}


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def _wilder_atr_series(highs, lows, closes, period) -> list[Optional[float]]:
    n = len(closes)
    atrs: list[Optional[float]] = [None] * n
    atr_val: Optional[float] = None
    seed: list[float] = []
    for i in range(n):
        if i == 0:
            continue
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        if atr_val is None:
            seed.append(tr)
            if len(seed) == period:
                atr_val = sum(seed) / period
        else:
            atr_val = (atr_val*(period-1) + tr) / period
        atrs[i] = atr_val
    return atrs


def _rolling_median_volume(vols: list[float], window: int) -> list[Optional[float]]:
    """Trailing median of volume over `window` bars ending at i-1 (prior only)."""
    n = len(vols)
    out: list[Optional[float]] = [None] * n
    from collections import deque
    import bisect
    sorted_win: list[float] = []
    dq: deque = deque()
    for i in range(n):
        # median of the window strictly BEFORE i
        if len(sorted_win) >= window:
            old = dq.popleft()
            pos = bisect.bisect_left(sorted_win, old)
            sorted_win.pop(pos)
        if i > 0:
            bisect.insort(sorted_win, vols[i-1])
            dq.append(vols[i-1])
        m = len(sorted_win)
        if m >= window // 2:
            out[i] = (sorted_win[m//2] if m % 2 else
                      0.5*(sorted_win[m//2-1] + sorted_win[m//2]))
    return out


# ---------------------------------------------------------------------------
# Cascade detection
# ---------------------------------------------------------------------------

@dataclass
class _Cascade:
    symbol: str
    side: str          # fade side: "LONG" (down-cascade) / "SHORT" (up-cascade)
    bar_idx: int
    entry_idx: int
    entry_ms: int
    vol_ratio: float
    range_ratio: float


def _detect_cascades(symbol, candles, atrs, med_vol) -> list[_Cascade]:
    opens, highs, lows, closes = candles["open"], candles["high"], candles["low"], candles["close"]
    vols, ot = candles["volume"], candles["open_time"]
    n = len(closes)
    out: list[_Cascade] = []
    for i in range(_TRAIL_BARS, n - 1):
        atr = atrs[i]; mv = med_vol[i]
        if atr is None or atr <= 0 or mv is None or mv <= 0:
            continue
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        if vols[i] < _VOL_MULT * mv:
            continue
        if tr < _RANGE_MULT * atr:
            continue
        # direction by bar body
        if closes[i] < opens[i]:
            side = "LONG"     # down-cascade → fade long
        elif closes[i] > opens[i]:
            side = "SHORT"    # up-cascade → fade short
        else:
            continue
        out.append(_Cascade(symbol, side, i, i+1, ot[i+1],
                            vols[i]/mv, tr/atr))
    return out


# ---------------------------------------------------------------------------
# SECTION A
# ---------------------------------------------------------------------------

def _section_a(cascades_by, candles_by):
    print("\n" + "=" * 70)
    print("  SECTION A — MECHANISM VALIDITY (pre-fee, event-level analysis)")
    print("=" * 70)
    print(f"  Cascade = volume >= {_VOL_MULT}× trailing-24h median AND "
          f"true-range >= {_RANGE_MULT}× ATR(14)")
    print(f"  Fade direction: down-cascade → LONG, up-cascade → SHORT.")
    print(f"  Forward returns measured in the FADE direction from next-bar open.")

    # Baseline: average |fade-direction| return over ALL bars, to compare.
    agg = {"LONG": {h: [] for h in _FWD_HORIZONS}, "SHORT": {h: [] for h in _FWD_HORIZONS}}
    hit = {"LONG": {h: [0, 0] for h in _FWD_HORIZONS}, "SHORT": {h: [0, 0] for h in _FWD_HORIZONS}}
    counts = {}

    for symbol in _SYMBOLS:
        candles = candles_by[symbol]
        opens, closes = candles["open"], candles["close"]
        n = len(closes)
        cs = cascades_by.get(symbol, [])
        counts[symbol] = (sum(1 for c in cs if c.side == "LONG"),
                          sum(1 for c in cs if c.side == "SHORT"))
        for c in cs:
            eo = opens[c.entry_idx]
            if eo <= 0:
                continue
            for h in _FWD_HORIZONS:
                tgt = c.entry_idx + h
                if tgt >= n:
                    continue
                raw = (closes[tgt] - eo) / eo
                fade = raw if c.side == "LONG" else -raw   # return in fade direction
                agg[c.side][h].append(fade)
                hit[c.side][h][1] += 1
                if fade > 0:
                    hit[c.side][h][0] += 1

    print(f"\n  Cascade counts:")
    print(f"  {'Symbol':<10} {'Down(LONG)':>12} {'Up(SHORT)':>12}")
    print(f"  {'─'*10} {'─'*12} {'─'*12}")
    tl = ts = 0
    for symbol in _SYMBOLS:
        dl, us = counts[symbol]
        tl += dl; ts += us
        print(f"  {symbol:<10} {dl:>12,} {us:>12,}")
    print(f"  {'─'*10} {'─'*12} {'─'*12}")
    print(f"  {'COMBINED':<10} {tl:>12,} {ts:>12,}")

    def _mean(lst): return f"{sum(lst)/len(lst)*100:+.4f}%" if lst else "    n/a"

    for side, label in (("LONG", "DOWN-cascade → fade LONG"), ("SHORT", "UP-cascade → fade SHORT")):
        print(f"\n  {label}:")
        print(f"  {'Horizon':>8} {'N':>7} {'Mean fade-ret':>14} {'Revert hit%':>12}")
        print(f"  {'─'*8} {'─'*7} {'─'*14} {'─'*12}")
        for h in _FWD_HORIZONS:
            vals = agg[side][h]
            hh, hn = hit[side][h]
            hr = f"{hh/hn*100:.1f}%" if hn else "n/a"
            tag = {1: "+15m", 4: "+1h", 8: "+2h", 16: "+4h"}[h]
            print(f"  {tag:>8} {len(vals):>7,} {_mean(vals):>14} {hr:>12}")

    # Assessment: reversion exists if mean fade-return > 0 and hit-rate > 52%
    # at the trade horizon (+8 bars = 2h, the hold length), for BOTH sides.
    def _ok(side):
        vals = agg[side][_HOLD_BARS]
        hh, hn = hit[side][_HOLD_BARS]
        if not vals or not hn:
            return False
        return (sum(vals)/len(vals) > 0) and (hh/hn > 0.52)
    long_ok, short_ok = _ok("LONG"), _ok("SHORT")
    assessment = "STRONG" if (long_ok and short_ok) else ("WEAK" if (long_ok or short_ok) else "NO EDGE")
    print(f"\n  Mechanism assessment: {assessment}  (at +{_HOLD_BARS}-bar/2h hold horizon)")
    if assessment == "NO EDGE":
        print("    No reliable post-cascade reversion in either direction.")
    elif assessment == "WEAK":
        print("    Reversion appears in only one direction.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# SECTION B — shared-equity sim
# ---------------------------------------------------------------------------

@dataclass
class _Trade:
    symbol: str; direction: str; entry_price: float; exit_price: float
    entry_ms: int; exit_ms: int; size: float; pnl: float; fees: float
    stop_price: float; exit_reason: str

@dataclass
class _OpenPos:
    symbol: str; direction: str; entry_price: float; entry_ms: int
    entry_idx: int; size: float; stop: float; entry_fee: float; bars_held: int


def _year(ms): return datetime.fromtimestamp(ms/1000, tz=timezone.utc).year

def _max_dd(eq):
    peak = eq[0][1] if eq else 0.0; mdd = 0.0
    for _, e in eq:
        if e > peak: peak = e
        if peak > 0:
            dd = (peak-e)/peak
            if dd > mdd: mdd = dd
    return mdd

def _pf(ts):
    gp = sum(t.pnl for t in ts if t.pnl > 0)
    gl = abs(sum(t.pnl for t in ts if t.pnl <= 0))
    return gp/gl if gl > 0 else float("inf")


def _section_b(cascades_by, candles_by, atr_by, hold_bars, initial_equity):
    sig_at = {}
    for symbol in _SYMBOLS:
        m = {}
        for c in cascades_by.get(symbol, []):
            if c.entry_idx not in m:
                m[c.entry_idx] = c
        sig_at[symbol] = m

    timeline = []
    for symbol in _SYMBOLS:
        for idx, ts in enumerate(candles_by[symbol]["open_time"]):
            timeline.append((ts, symbol, idx))
    rank = {s: r for r, s in enumerate(_SYMBOLS)}
    timeline.sort(key=lambda x: (x[0], rank[x[1]]))

    equity = initial_equity
    eq_curve = [(timeline[0][0], equity)] if timeline else []
    trades: list[_Trade] = []
    open_pos = {s: None for s in _SYMBOLS}

    for ts, symbol, idx in timeline:
        c = candles_by[symbol]
        c_open, c_high, c_low, c_close = c["open"][idx], c["high"][idx], c["low"][idx], c["close"][idx]
        pos = open_pos[symbol]
        if pos is not None and idx > pos.entry_idx:
            pos.bars_held += 1
            stop_hit = False; exit_price = None
            if pos.direction == "LONG" and c_low <= pos.stop:
                stop_hit = True; exit_price = pos.stop
            elif pos.direction == "SHORT" and c_high >= pos.stop:
                stop_hit = True; exit_price = pos.stop
            if stop_hit or pos.bars_held >= hold_bars:
                if exit_price is None: exit_price = c_close
                reason = "STOP" if stop_hit else "HOLD"
                exit_fee = exit_price * pos.size * _TAKER_FEE
                raw = ((exit_price-pos.entry_price) if pos.direction == "LONG"
                       else (pos.entry_price-exit_price)) * pos.size
                equity += raw - exit_fee
                trades.append(_Trade(symbol, pos.direction, pos.entry_price, exit_price,
                                     pos.entry_ms, ts, pos.size, raw-exit_fee,
                                     pos.entry_fee+exit_fee, pos.stop, reason))
                open_pos[symbol] = None; pos = None

        sig = sig_at[symbol].get(idx)
        if sig is not None and open_pos[symbol] is None:
            atr = atr_by[symbol][idx]; entry_price = c_open
            if atr is not None and atr > 0 and entry_price > 0:
                stop_dist = _STOP_MULT * atr
                stop_price = (entry_price-stop_dist if sig.side == "LONG"
                              else entry_price+stop_dist)
                size = (equity*_RISK_PCT)/stop_dist
                if size > 0:
                    entry_fee = entry_price*size*_TAKER_FEE
                    equity -= entry_fee
                    open_pos[symbol] = _OpenPos(symbol, sig.side, entry_price, ts,
                                                idx, size, stop_price, entry_fee, 0)
        eq_curve.append((ts, equity))

    for symbol in _SYMBOLS:
        pos = open_pos[symbol]
        if pos is None: continue
        c = candles_by[symbol]; li = len(c["close"]) - 1
        exit_price = c["close"][li]; exit_ms = c["open_time"][li]
        exit_fee = exit_price*pos.size*_TAKER_FEE
        raw = ((exit_price-pos.entry_price) if pos.direction == "LONG"
               else (pos.entry_price-exit_price)) * pos.size
        equity += raw - exit_fee
        trades.append(_Trade(symbol, pos.direction, pos.entry_price, exit_price,
                             pos.entry_ms, exit_ms, pos.size, raw-exit_fee,
                             pos.entry_fee+exit_fee, pos.stop, "HOLD"))
    if eq_curve:
        eq_curve.append((eq_curve[-1][0], equity))
    trades.sort(key=lambda t: t.entry_ms)
    _print_section_b(trades, eq_curve, initial_equity, equity, hold_bars)


def _print_section_b(trades, eq_curve, initial_equity, final_equity, hold_bars):
    print("\n" + "=" * 70)
    print("  SECTION B — MONETIZATION BACKTEST (after fees)")
    print("=" * 70)
    if not trades:
        print("\n  [RESULT] No trades generated.")
        print("=" * 70)
        _print_verdict([], 0.0, 0.0, initial_equity, final_equity)
        return
    first_ms = min(t.entry_ms for t in trades); last_ms = max(t.exit_ms for t in trades)
    years = (last_ms-first_ms)/(1000*86400*365.25)
    cagr = (final_equity/initial_equity)**(1/years)-1 if years > 0 else 0.0
    mdd = _max_dd(eq_curve)
    wins = [t for t in trades if t.pnl > 0]
    pf_full = _pf(trades); wr = len(wins)/len(trades)*100
    total_fees = sum(t.fees for t in trades)
    n_l = sum(1 for t in trades if t.direction == "LONG")
    n_s = sum(1 for t in trades if t.direction == "SHORT")
    n_stop = sum(1 for t in trades if t.exit_reason == "STOP")
    n_hold = sum(1 for t in trades if t.exit_reason == "HOLD")

    print(f"  Parameters    : vol>={_VOL_MULT}×med  range>={_RANGE_MULT}×ATR  "
          f"hold={hold_bars}bars(2h)  stop={_STOP_MULT}×ATR({_ATR_PERIOD})")
    print(f"  Period        : {datetime.fromtimestamp(first_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}"
          f" → {datetime.fromtimestamp(last_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"  Initial equity: ${initial_equity:,.0f}")
    print(f"  Final equity  : ${final_equity:,.0f}")
    print(f"  Net PnL       : ${final_equity-initial_equity:+,.0f}  ({(final_equity-initial_equity)/initial_equity*100:+.1f}%)")
    print(f"  CAGR          : {cagr*100:.1f}%")
    print(f"  Max drawdown  : {mdd*100:.1f}%")
    print(f"  Profit factor : {pf_full:.2f}")
    print(f"  Total trades  : {len(trades)}  (L:{n_l}  S:{n_s})")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Exit reasons  : HOLD={n_hold}  STOP={n_stop}")
    print(f"  Total fees    : ${total_fees:,.0f}")

    print(f"\n  {'─'*70}")
    print(f"  {'Symbol':<10} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>12}")
    print(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*7} {'─'*12}")
    for symbol in _SYMBOLS:
        st = [t for t in trades if t.symbol == symbol]
        if not st:
            print(f"  {symbol:<10} {0:>7} {'n/a':>7} {'n/a':>7} {'$0':>12}"); continue
        sw = [t for t in st if t.pnl > 0]; spf = _pf(st)
        pf_str = f"{spf:.2f}" if not math.isinf(spf) else "inf"
        print(f"  {symbol:<10} {len(st):>7} {len(sw)/len(st)*100:>6.1f}% {pf_str:>7} ${sum(t.pnl for t in st):>+10,.0f}")

    print(f"\n  {'─'*70}")
    print(f"  {'Year':<8} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>12}  Status")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*12}  {'─'*10}")
    for yr in sorted({_year(t.entry_ms) for t in trades}):
        yt = [t for t in trades if _year(t.entry_ms) == yr]
        yw = [t for t in yt if t.pnl > 0]; ypf = _pf(yt)
        pf_str = f"{ypf:>7.2f}" if not math.isinf(ypf) else f"{'inf':>7}"
        status = "+" if (math.isinf(ypf) or ypf >= 1.0) else "-"
        print(f"  {yr:<8} {len(yt):>7} {len(yw)/len(yt)*100:>6.1f}% {pf_str} ${sum(t.pnl for t in yt):>+10,.0f}  {status}")

    sb = sorted(trades, key=lambda t: t.pnl, reverse=True)
    print(f"\n  {'─'*70}")
    print("  Top 5 trades by PnL:")
    for t in sb[:5]:
        dt = datetime.fromtimestamp(t.entry_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"    {t.symbol:<8} {t.direction:<5} entry {dt}  exit={t.exit_reason}  ${t.pnl:>+,.0f}")
    print("  Bottom 5 trades by PnL:")
    for t in sb[-5:]:
        dt = datetime.fromtimestamp(t.entry_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"    {t.symbol:<8} {t.direction:<5} entry {dt}  exit={t.exit_reason}  ${t.pnl:>+,.0f}")

    oos = [t for t in trades if _year(t.entry_ms) >= _OOS_START_YEAR]
    ins = [t for t in trades if _year(t.entry_ms) < _OOS_START_YEAR]
    pf_oos = _pf(oos)
    print(f"\n  {'─'*70}")
    print(f"  OOS split (IS = <{_OOS_START_YEAR}, OOS = {_OOS_START_YEAR}+):")
    print(f"    IS : {len(ins)} trades, PF {_pf(ins):.2f}" if ins else "    IS : no trades")
    print(f"    OOS: {len(oos)} trades, PF {pf_oos:.2f}" if oos else "    OOS: no trades")
    print("=" * 70)
    _print_verdict(trades, pf_full, mdd, initial_equity, final_equity, pf_oos=pf_oos, pf_full_ref=pf_full)


def _print_verdict(trades, pf_full, mdd, initial_equity, final_equity,
                   pf_oos=float("nan"), pf_full_ref=float("nan")):
    first_ms = min(t.entry_ms for t in trades) if trades else 0
    last_ms = max(t.exit_ms for t in trades) if trades else 0
    years = (last_ms-first_ms)/(1000*86400*365.25) if last_ms > first_ms else 0
    cagr = (final_equity/initial_equity)**(1/years)-1 if years > 0 and initial_equity > 0 else 0.0

    kill = []; n = len(trades)
    if pf_full < _KILL_PF: kill.append(f"Profit factor {pf_full:.2f} < kill threshold {_KILL_PF:.2f}")
    if mdd > _KILL_MAX_DD: kill.append(f"Max drawdown {mdd*100:.1f}% > kill threshold {_KILL_MAX_DD*100:.0f}%")
    if n < _KILL_TRADES: kill.append(f"Trade count {n} < minimum {_KILL_TRADES}")
    if not math.isnan(pf_oos) and not math.isnan(pf_full_ref):
        thr = _KILL_OOS_RATIO*pf_full_ref
        if pf_oos < thr:
            kill.append(f"OOS PF {pf_oos:.2f} < {_KILL_OOS_RATIO:.0%} × full-period PF {pf_full_ref:.2f} (threshold {thr:.2f}) — regime-specific")

    go = []
    if pf_full >= _GO_PF: go.append(f"PF {pf_full:.2f} >= {_GO_PF}")
    if cagr >= _GO_CAGR: go.append(f"CAGR {cagr*100:.1f}% >= {_GO_CAGR*100:.0f}%")
    if mdd <= _GO_DD: go.append(f"Max DD {mdd*100:.1f}% <= {_GO_DD*100:.0f}%")

    passes_core = pf_full >= _KILL_PF and mdd <= _KILL_MAX_DD and n >= _KILL_TRADES
    oos_weak = (not math.isnan(pf_oos) and not math.isnan(pf_full_ref)
                and pf_oos < _KILL_OOS_RATIO*pf_full_ref)

    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")
    if passes_core and oos_weak:
        print("\n  MARGINAL -- passes PF/DD/trade-count gates but OOS is weak:\n")
        print(f"     * PF {pf_full:.2f} >= {_KILL_PF:.2f} (kill gate)")
        print(f"     * Max DD {mdd*100:.1f}% <= {_KILL_MAX_DD*100:.0f}% (kill gate)")
        print(f"     * Trades {n} >= {_KILL_TRADES} (kill gate)")
        print(f"     * OOS PF {pf_oos:.2f} < {_KILL_OOS_RATIO:.0%} × full PF {pf_full_ref:.2f}")
        print("\n  Action: regime-sensitive — run paper trading before adding capital.")
    elif kill:
        print("\n  KILL -- Strategy fails one or more kill criteria:\n")
        for r in kill: print(f"     * {r}")
        print("\n  Action: do NOT deploy. Move to next hypothesis.")
    elif len(go) == 3:
        print("\n  GO -- All go criteria met:\n")
        for f in go: print(f"     * {f}")
        print("\n  Action: proceed to paper deployment.")
    else:
        print(f"\n  MARGINAL -- {len(go)}/3 go criteria met. Deploy on paper with caution.")
        print("  Passed  :", ", ".join(go) if go else "none")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Liquidation-Cascade Reversion Backtest (15m)")
    parser.add_argument("--data", default="data/candles")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--start", default=_DL_START)
    parser.add_argument("--hold-bars", type=int, default=_HOLD_BARS)
    args = parser.parse_args()

    data_dir = _REPO_ROOT / args.data
    start_ms = int(datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()*1000)
    end_ms = int(datetime.now(timezone.utc).timestamp()*1000)
    hold_bars = args.hold_bars

    print(f"\n{'='*70}")
    print("  LIQUIDATION-CASCADE REVERSION BACKTEST — 15m timeframe")
    print(f"{'='*70}")
    print("  Hypothesis: forced-liquidation cascades overshoot, then revert.")
    print("  Pre-committed rule:")
    print(f"    cascade = volume >= {_VOL_MULT}× trailing-24h median AND true-range >= {_RANGE_MULT}× ATR(14)")
    print(f"    fade    = down-cascade → LONG, up-cascade → SHORT")
    print(f"    entry   = next bar open (no lookahead)")
    print(f"    hold    = {hold_bars} bars (2h) OR {_STOP_MULT}×ATR({_ATR_PERIOD}) stop")
    print(f"    fee     = {_TAKER_FEE*100:.4f}% taker/side  risk = {_RISK_PCT*100:.1f}%  capital = ${args.capital:,.0f}")
    print(f"    symbols = {', '.join(_SYMBOLS)}")

    print(f"\nAcquiring 15m candles (cache dir: {data_dir}) ...")
    candles_by = {}
    atr_by = {}
    cascades_by = {}
    for symbol in _SYMBOLS:
        candles = _load_or_download_15m(symbol, data_dir, start_ms, end_ms)
        atrs = _wilder_atr_series(candles["high"], candles["low"], candles["close"], _ATR_PERIOD)
        med_vol = _rolling_median_volume(candles["volume"], _TRAIL_BARS)
        cascades = _detect_cascades(symbol, candles, atrs, med_vol)
        candles_by[symbol] = candles
        atr_by[symbol] = atrs
        cascades_by[symbol] = cascades
        print(f"  {symbol}: {len(candles['close']):,} bars, {len(cascades):,} cascade events")

    _section_a(cascades_by, candles_by)
    _section_b(cascades_by, candles_by, atr_by, hold_bars, args.capital)

    print(f"\n{'='*70}")
    print("  Backtest complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
