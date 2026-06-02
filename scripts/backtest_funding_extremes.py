#!/usr/bin/env python3
"""Funding-Rate Extremes Backtest (contrarian fade) — 1H timeframe.

Mechanism hypothesis: extreme POSITIVE funding (crowded longs) precedes
mean-reversion DOWN; extreme NEGATIVE funding (crowded shorts) precedes
mean-reversion UP. This is a CONTRARIAN / FADE strategy.

Two-section output
------------------
SECTION A  Mechanism Validity — does the fade exist (pre-fee, event-level)?
SECTION B  Monetization      — can it be extracted after fees?

Parameters (all pre-committed):
  extreme   : funding in top/bottom 10% of a TRAILING 90-day window (per symbol).
              Rolling 10th/90th percentile from prior obs only — no lookahead.
              Need >= 30 obs in the trailing window or the event is skipped.
  entry     : funding >= 90th-pct trailing  → SHORT at aligned candle open.
              funding <= 10th-pct trailing  → LONG  at aligned candle open.
  hold      : 24 hours (24 × 1H bars) from entry, OR stop, whichever first.
  stop_mult : 2.0 × ATR(14) on the 1H candles from entry price.
  one position at a time per symbol; new extremes while in a position are ignored.

Kill criteria:
  PF < 1.10
  max DD > 40%
  n_trades < 60
  OOS PF (2023+) < 0.85 × full-period PF   ("regime-specific")

Self-contained: downloads its own funding-rate history from Binance (free,
no key) and caches it to data/funding/{SYMBOL}_funding_okx.parquet, then
loads existing 1H candles from data/candles/{SYMBOL}_1H.parquet.

Usage:
  python scripts/backtest_funding_extremes.py
  python scripts/backtest_funding_extremes.py --capital 50000 --threshold-pct 10
  python scripts/backtest_funding_extremes.py --data data/candles --funding-dir data/funding
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
_TAKER_FEE    = 0.0006   # 0.06% per side — Bitget USDT-Perp taker
_RISK_PCT     = 0.01     # 1% account equity per trade
_ATR_PERIOD   = 14
_HOLD_HOURS   = 24       # mean-reversion is a timed trade (overridable via CLI)
_STOP_MULT    = 2.0

# Trailing percentile window for "extreme" funding (per symbol)
_TRAIL_DAYS   = 90
_MIN_OBS      = 30       # need >= this many obs in trailing window or skip

# Kill thresholds
_KILL_PF        = 1.10
_KILL_TRADES    = 60
_KILL_MAX_DD    = 0.40
_KILL_OOS_RATIO = 0.85   # OOS PF must be >= 85% of full-period PF
_OOS_START_YEAR = 2023

# Go thresholds
_GO_PF   = 1.30
_GO_CAGR = 0.20
_GO_DD   = 0.40

# Symbols
_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Funding download — OKX public funding-rate-history (free, no key, full history,
# reachable from US GitHub Actions runners; Binance fapi returns HTTP 451 there).
# Funding rates are tightly correlated across venues, so OKX is a valid proxy for
# the mechanism-validity question on a Bitget-traded book.
_FUNDING_URL   = "https://www.okx.com/api/v5/public/funding-rate-history"
_FUNDING_LIMIT = 100
_FUNDING_DELAY = 0.2
_FUNDING_START = "2021-01-01"
_HEADERS       = {"User-Agent": "NexFlow/1.0", "Accept": "application/json"}
_HOUR_MS       = 3_600_000
_DAY_MS        = 86_400_000

# Map Bitget USDT-perp tickers to OKX swap instIds
_OKX_INST = {"BTCUSDT": "BTC-USDT-SWAP", "ETHUSDT": "ETH-USDT-SWAP"}

_FUNDING_SCHEMA = pa.schema([
    pa.field("symbol",       pa.string()),
    pa.field("timestamp_ms", pa.int64()),
    pa.field("funding_rate", pa.float64()),
])


# ---------------------------------------------------------------------------
# Funding-rate download (inline, cached)
# ---------------------------------------------------------------------------

def _fetch_funding_page(inst_id: str, before_ms: int | None) -> list:
    """OKX funding-rate-history page. `before_ms` (if set) returns rows OLDER than it."""
    url = f"{_FUNDING_URL}?instId={inst_id}&limit={_FUNDING_LIMIT}"
    if before_ms is not None:
        url += f"&after={before_ms}"  # OKX 'after' = records with ts < after (older)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read())
    if payload.get("code") not in ("0", 0):
        raise RuntimeError(f"OKX error: {payload.get('msg', payload)}")
    return payload.get("data", [])


def _download_funding(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """Backward-paginate OKX funding history (newest→oldest).  Returns sorted dicts."""
    inst_id = _OKX_INST.get(symbol)
    if inst_id is None:
        print(f"\n  [ERROR] No OKX instId mapping for {symbol}")
        return []

    rows: list[dict] = []
    seen: set[int]   = set()
    cursor: int | None = None  # None = most recent page
    errors = 0

    while True:
        try:
            page = _fetch_funding_page(inst_id, cursor)
        except Exception as exc:
            errors += 1
            if errors >= 5:
                print(f"\n  [ERROR] Too many failures for {symbol}: {exc}")
                break
            print(f"\n  [WARN] Retrying {symbol}: {exc}")
            time.sleep(2.0 * errors)
            continue
        errors = 0

        if not page:
            break

        oldest = cursor if cursor is not None else end_ms
        new = 0
        for item in page:
            ts = int(item["fundingTime"])
            if ts in seen or ts > end_ms:
                continue
            seen.add(ts)
            oldest = min(oldest, ts)
            if ts >= start_ms:
                new += 1
                rows.append({
                    "symbol":       symbol,
                    "timestamp_ms": ts,
                    "funding_rate": float(item["fundingRate"]),
                })

        total = len(rows)
        pct   = (end_ms - oldest) / max(end_ms - start_ms, 1) * 100
        print(f"  {symbol}: {total:,} funding rows  ({pct:.0f}% complete) ...", end="\r")

        if len(page) < _FUNDING_LIMIT or oldest <= start_ms:
            break
        cursor = oldest  # next page: older than the oldest we've seen
        time.sleep(_FUNDING_DELAY)

    print()
    return sorted(rows, key=lambda r: r["timestamp_ms"])


def _save_funding(symbol: str, rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_funding_okx.parquet"
    table = pa.table({
        "symbol":       [r["symbol"]       for r in rows],
        "timestamp_ms": [r["timestamp_ms"] for r in rows],
        "funding_rate": [r["funding_rate"] for r in rows],
    }, schema=_FUNDING_SCHEMA)
    pq.write_table(table, path)
    return path


def _load_funding(symbol: str, funding_dir: Path) -> list[dict]:
    """Load funding from cache, else download and cache it.

    Returns a list of {timestamp_ms, funding_rate} sorted ascending.
    """
    path = funding_dir / f"{symbol}_funding_okx.parquet"
    if path.exists():
        tbl = pq.read_table(path).to_pydict()
        rows = [
            {"timestamp_ms": tbl["timestamp_ms"][i], "funding_rate": tbl["funding_rate"][i]}
            for i in range(len(tbl["timestamp_ms"]))
        ]
        rows.sort(key=lambda r: r["timestamp_ms"])
        print(f"  {symbol}: loaded {len(rows):,} cached funding rows from {path}")
        return rows

    start_dt = datetime.strptime(_FUNDING_START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    print(f"  {symbol}: no cache — downloading funding from Binance "
          f"({_FUNDING_START} → now) ...")
    dicts = _download_funding(symbol, start_ms, end_ms)
    if not dicts:
        print(f"  [ERROR] No funding data downloaded for {symbol}.")
        return []
    saved = _save_funding(symbol, dicts, funding_dir)
    print(f"  {symbol}: saved {len(dicts):,} funding rows → {saved}")
    return [{"timestamp_ms": d["timestamp_ms"], "funding_rate": d["funding_rate"]} for d in dicts]


# ---------------------------------------------------------------------------
# Candle loading
# ---------------------------------------------------------------------------

def _load_1h(symbol: str, data_dir: Path) -> dict:
    """Return dict of lists keyed by column name, sorted ascending by open_time."""
    path = data_dir / f"{symbol}_1H.parquet"
    if not path.exists():
        print(f"[ERROR] Missing: {path}")
        sys.exit(1)
    tbl = pq.read_table(path)
    d   = tbl.to_pydict()
    idx = sorted(range(len(d["open_time"])), key=lambda i: d["open_time"][i])
    return {col: [d[col][i] for i in idx] for col in d}


# ---------------------------------------------------------------------------
# ATR — Wilder smoothing (same helper as backtest_lead_lag.py)
# ---------------------------------------------------------------------------

def _wilder_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> Optional[float]:
    n = len(closes)
    if n < period + 1:
        return None
    trs = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


# ---------------------------------------------------------------------------
# Percentile helper (linear interpolation, like numpy default)
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolated percentile. `sorted_vals` must be ascending. pct in [0,100]."""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


# ---------------------------------------------------------------------------
# Event construction: align funding extremes to candle bars (no lookahead)
# ---------------------------------------------------------------------------

@dataclass
class _Event:
    symbol:       str
    funding_ms:   int      # funding settlement timestamp
    funding_rate: float
    side:         str      # "SHORT" (positive extreme) or "LONG" (negative extreme)
    entry_idx:    int      # candle index whose open is the entry (bar at/after funding)
    entry_ms:     int      # that candle's open_time
    lo_pct:       float    # trailing 10th-pct at decision time
    hi_pct:       float    # trailing 90th-pct at decision time


def _build_events(
    symbol:    str,
    funding:   list[dict],
    candles:   dict,
    low_pct:   float,
    high_pct:  float,
) -> list[_Event]:
    """Identify extreme funding events via trailing percentile and align to candles.

    Trailing window uses ONLY prior funding observations within _TRAIL_DAYS, so
    the percentile thresholds at each event contain no lookahead. The entry is the
    candle whose open_time is at or immediately after the funding settlement.
    """
    open_times = candles["open_time"]
    n_candles  = len(open_times)
    events: list[_Event] = []

    # funding is sorted ascending; for each obs build trailing-window thresholds.
    times = [f["timestamp_ms"] for f in funding]
    rates = [f["funding_rate"] for f in funding]
    n = len(funding)

    win_start = 0  # left edge of trailing window (sliding)
    for i in range(n):
        t_now = times[i]
        # Advance the window's left edge to keep only obs within _TRAIL_DAYS BEFORE t_now.
        lo_bound = t_now - _TRAIL_DAYS * _DAY_MS
        while win_start < i and times[win_start] < lo_bound:
            win_start += 1
        # Trailing window is prior obs only: [win_start, i)  (exclude current obs).
        window = rates[win_start:i]
        if len(window) < _MIN_OBS:
            continue

        sorted_win = sorted(window)
        lo_thr = _percentile(sorted_win, low_pct)
        hi_thr = _percentile(sorted_win, high_pct)

        rate = rates[i]
        if rate >= hi_thr:
            side = "SHORT"
        elif rate <= lo_thr:
            side = "LONG"
        else:
            continue

        # Align to the candle at or immediately AFTER the funding settlement.
        entry_idx = _first_candle_at_or_after(open_times, t_now)
        if entry_idx is None or entry_idx >= n_candles:
            continue

        events.append(_Event(
            symbol       = symbol,
            funding_ms   = t_now,
            funding_rate = rate,
            side         = side,
            entry_idx    = entry_idx,
            entry_ms     = open_times[entry_idx],
            lo_pct       = lo_thr,
            hi_pct       = hi_thr,
        ))

    return events


def _first_candle_at_or_after(open_times: list[int], ts: int) -> Optional[int]:
    """Binary search for the first candle index whose open_time >= ts."""
    lo, hi = 0, len(open_times)
    while lo < hi:
        mid = (lo + hi) // 2
        if open_times[mid] < ts:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(open_times) else None


# ---------------------------------------------------------------------------
# SECTION A — Mechanism Validity (pre-fee, event-level)
# ---------------------------------------------------------------------------

def _section_a(
    events_by_symbol: dict[str, list[_Event]],
    candles_by_symbol: dict[str, dict],
    funding_by_symbol: dict[str, list[dict]],
    low_pct:  float,
    high_pct: float,
) -> None:
    """Print SECTION A: forward-return analysis of funding extremes."""

    print("\n" + "=" * 70)
    print("  SECTION A — MECHANISM VALIDITY (pre-fee, event-level analysis)")
    print("=" * 70)
    print(f"  Extreme rule    : funding in top/bottom {100-high_pct:.0f}%/{low_pct:.0f}% "
          f"of TRAILING {_TRAIL_DAYS}d window (>= {_MIN_OBS} obs)")
    print(f"  Positive extreme→ SHORT (crowded longs);  Negative extreme→ LONG (crowded shorts)")

    # Forward returns are measured from the aligned entry candle's OPEN to the
    # CLOSE of the candle +Nh later (using the next-bar-after-funding entry).
    def _fwd_ret(candles: dict, entry_idx: int, hours: int) -> Optional[float]:
        opens  = candles["open"]
        closes = candles["close"]
        entry_open = opens[entry_idx]
        tgt = entry_idx + hours          # +Nh = +N 1H bars
        if entry_open <= 0 or tgt >= len(closes):
            return None
        return (closes[tgt] - entry_open) / entry_open

    # Per-symbol then combined aggregation.
    # pos = positive extreme (we short), neg = negative extreme (we long).
    agg = {
        "pos": {"n": 0, "r8": [], "r24": [], "r48": [], "fade24": 0, "fade24_n": 0},
        "neg": {"n": 0, "r8": [], "r24": [], "r48": [], "fade24": 0, "fade24_n": 0},
    }
    per_symbol_counts: dict[str, tuple[int, int]] = {}

    for symbol in _SYMBOLS:
        events  = events_by_symbol.get(symbol, [])
        candles = candles_by_symbol.get(symbol, {})
        n_pos = sum(1 for e in events if e.side == "SHORT")
        n_neg = sum(1 for e in events if e.side == "LONG")
        per_symbol_counts[symbol] = (n_pos, n_neg)

        for e in events:
            bucket = "pos" if e.side == "SHORT" else "neg"
            agg[bucket]["n"] += 1
            r8  = _fwd_ret(candles, e.entry_idx, 8)
            r24 = _fwd_ret(candles, e.entry_idx, 24)
            r48 = _fwd_ret(candles, e.entry_idx, 48)
            if r8  is not None: agg[bucket]["r8"].append(r8)
            if r24 is not None: agg[bucket]["r24"].append(r24)
            if r48 is not None: agg[bucket]["r48"].append(r48)
            # Fade hit-rate uses the +24h forward return.
            if r24 is not None:
                agg[bucket]["fade24_n"] += 1
                if bucket == "pos" and r24 < 0:      # price LOWER after positive extreme
                    agg[bucket]["fade24"] += 1
                elif bucket == "neg" and r24 > 0:    # price HIGHER after negative extreme
                    agg[bucket]["fade24"] += 1

    # Frequency / counts
    print(f"\n  Event counts (extreme funding settlements):")
    print(f"  {'Symbol':<10} {'Pos-extreme(short)':>19} {'Neg-extreme(long)':>19} {'FundingObs':>12} {'%Extreme':>10} {'Ev/mo':>8}")
    print(f"  {'─'*10} {'─'*19} {'─'*19} {'─'*12} {'─'*10} {'─'*8}")
    total_pos = total_neg = total_obs = 0
    total_span_months = 0.0
    for symbol in _SYMBOLS:
        n_pos, n_neg = per_symbol_counts[symbol]
        funding = funding_by_symbol.get(symbol, [])
        n_obs   = len(funding)
        total_pos += n_pos
        total_neg += n_neg
        total_obs += n_obs
        if funding:
            span_months = (funding[-1]["timestamp_ms"] - funding[0]["timestamp_ms"]) / (_DAY_MS * 30.4375)
        else:
            span_months = 0.0
        total_span_months = max(total_span_months, span_months)
        n_ext = n_pos + n_neg
        pct_ext = n_ext / n_obs * 100 if n_obs else 0.0
        ev_mo   = n_ext / span_months if span_months > 0 else 0.0
        print(f"  {symbol:<10} {n_pos:>19} {n_neg:>19} {n_obs:>12,} {pct_ext:>9.1f}% {ev_mo:>8.1f}")
    tot_ext = total_pos + total_neg
    tot_pct = tot_ext / total_obs * 100 if total_obs else 0.0
    tot_evmo = tot_ext / total_span_months if total_span_months > 0 else 0.0
    print(f"  {'─'*10} {'─'*19} {'─'*19} {'─'*12} {'─'*10} {'─'*8}")
    print(f"  {'COMBINED':<10} {total_pos:>19} {total_neg:>19} {total_obs:>12,} {tot_pct:>9.1f}% {tot_evmo:>8.1f}")

    # Forward returns
    def _mean(lst: list[float]) -> str:
        if not lst:
            return "    n/a"
        return f"{sum(lst)/len(lst)*100:+.4f}%"

    print(f"\n  Forward price return from aligned entry (entry = bar open after funding):")
    print(f"  {'Extreme type':<26} {'N':>6} {'+8h':>12} {'+24h':>12} {'+48h':>12}")
    print(f"  {'─'*26} {'─'*6} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'POS-extreme (fade=short)':<26} {agg['pos']['n']:>6} "
          f"{_mean(agg['pos']['r8']):>12} {_mean(agg['pos']['r24']):>12} {_mean(agg['pos']['r48']):>12}")
    print(f"  {'NEG-extreme (fade=long)':<26} {agg['neg']['n']:>6} "
          f"{_mean(agg['neg']['r8']):>12} {_mean(agg['neg']['r24']):>12} {_mean(agg['neg']['r48']):>12}")
    print(f"\n  (POS-extreme: NEGATIVE forward return supports the fade.")
    print(f"   NEG-extreme: POSITIVE forward return supports the fade.)")

    # Fade hit-rate (+24h)
    pos_n   = agg["pos"]["fade24_n"]
    neg_n   = agg["neg"]["fade24_n"]
    pos_hit = agg["pos"]["fade24"]
    neg_hit = agg["neg"]["fade24"]
    comb_n   = pos_n + neg_n
    comb_hit = pos_hit + neg_hit

    def _hr(hit: int, tot: int) -> str:
        return f"{hit/tot*100:.1f}%" if tot else "n/a"

    print(f"\n  Fade hit-rate (+24h):")
    print(f"    POS-extreme (price LOWER after 24h) : {_hr(pos_hit, pos_n)}  ({pos_hit}/{pos_n})")
    print(f"    NEG-extreme (price HIGHER after 24h): {_hr(neg_hit, neg_n)}  ({neg_hit}/{neg_n})")
    print(f"    COMBINED fade hit-rate              : {_hr(comb_hit, comb_n)}  ({comb_hit}/{comb_n})")

    # Mechanism assessment
    comb_hr = comb_hit / comb_n if comb_n else 0.0
    pos_r24 = sum(agg["pos"]["r24"]) / len(agg["pos"]["r24"]) if agg["pos"]["r24"] else float("nan")
    neg_r24 = sum(agg["neg"]["r24"]) / len(agg["neg"]["r24"]) if agg["neg"]["r24"] else float("nan")
    # Predicted signs: pos-extreme forward ret < 0, neg-extreme forward ret > 0.
    signs_ok = (
        not math.isnan(pos_r24) and not math.isnan(neg_r24)
        and pos_r24 < 0 and neg_r24 > 0
    )
    hr_ok = comb_hr > 0.52

    if hr_ok and signs_ok:
        assessment = "STRONG"
    elif hr_ok or signs_ok:
        assessment = "WEAK"
    else:
        assessment = "NO EDGE"

    print(f"\n  Mechanism assessment: {assessment}  "
          f"(combined fade hit-rate {comb_hr*100:.1f}%, "
          f"pos +24h {('%+.4f%%' % (pos_r24*100)) if not math.isnan(pos_r24) else 'n/a'}, "
          f"neg +24h {('%+.4f%%' % (neg_r24*100)) if not math.isnan(neg_r24) else 'n/a'})")
    if assessment == "NO EDGE":
        print(f"    Neither hit-rate (>52%) nor forward-return signs support the fade.")
    elif assessment == "WEAK":
        print(f"    Only one of {{hit-rate>52%, predicted forward-return signs}} holds.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# SECTION B — Monetization backtest (shared equity, both symbols)
# ---------------------------------------------------------------------------

@dataclass
class _Trade:
    symbol:       str
    direction:    str           # "LONG" or "SHORT"
    entry_price:  float
    exit_price:   float
    entry_ms:     int
    exit_ms:      int
    size:         float
    pnl:          float         # net of all fees
    fees:         float
    stop_price:   float
    exit_reason:  str           # "HOLD" or "STOP"
    equity_at_entry: float


def _year(ms: int) -> int:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).year


@dataclass
class _OpenPos:
    symbol:       str
    direction:    str
    entry_price:  float
    entry_ms:     int
    entry_idx:    int
    size:         float
    stop:         float
    entry_fee:    float
    bars_held:    int
    equity_at_entry: float


def _section_b(
    events_by_symbol:  dict[str, list[_Event]],
    candles_by_symbol: dict[str, dict],
    hold_bars:         int,
    initial_equity:    float,
    low_pct:           float,
    high_pct:          float,
) -> None:
    """Bar-by-bar sim across both symbols on a shared equity curve."""

    # Pre-compute, per symbol: ATR(14) at each bar (using bars up to & incl. that
    # bar) and an entry-event map (entry_idx -> Event) for fast lookup.
    atr_by_symbol: dict[str, list[Optional[float]]] = {}
    event_at_idx:  dict[str, dict[int, _Event]]     = {}
    for symbol in _SYMBOLS:
        candles = candles_by_symbol[symbol]
        highs   = candles["high"]
        lows    = candles["low"]
        closes  = candles["close"]
        n       = len(closes)
        atrs: list[Optional[float]] = [None] * n
        # Rolling Wilder ATR computed incrementally.
        atr_val: Optional[float] = None
        tr_seed: list[float] = []
        for i in range(n):
            if i == 0:
                atrs[i] = None
                continue
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]),
            )
            if atr_val is None:
                tr_seed.append(tr)
                if len(tr_seed) == _ATR_PERIOD:
                    atr_val = sum(tr_seed) / _ATR_PERIOD
            else:
                atr_val = (atr_val * (_ATR_PERIOD - 1) + tr) / _ATR_PERIOD
            atrs[i] = atr_val
        atr_by_symbol[symbol] = atrs

        ev_map: dict[int, _Event] = {}
        for e in events_by_symbol.get(symbol, []):
            # First extreme to fire at a given entry bar wins; later ones ignored.
            if e.entry_idx not in ev_map:
                ev_map[e.entry_idx] = e
        event_at_idx[symbol] = ev_map

    # Build a unified, ordered timeline of (timestamp, symbol, candle_idx).
    timeline: list[tuple[int, str, int]] = []
    for symbol in _SYMBOLS:
        for idx, ts in enumerate(candles_by_symbol[symbol]["open_time"]):
            timeline.append((ts, symbol, idx))
    # Deterministic order: by timestamp, then symbol order in _SYMBOLS.
    sym_rank = {s: r for r, s in enumerate(_SYMBOLS)}
    timeline.sort(key=lambda x: (x[0], sym_rank[x[1]]))

    equity = initial_equity
    eq_curve: list[tuple[int, float]] = [(timeline[0][0], equity)] if timeline else []
    trades: list[_Trade] = []
    open_pos: dict[str, Optional[_OpenPos]] = {s: None for s in _SYMBOLS}

    for ts, symbol, idx in timeline:
        candles = candles_by_symbol[symbol]
        c_open  = candles["open"][idx]
        c_high  = candles["high"][idx]
        c_low   = candles["low"][idx]
        c_close = candles["close"][idx]

        # ----------------------------------------------------------------
        # 1. Process an open position on THIS symbol's bar.
        #    The entry bar itself counts as held bar 0 (no same-bar exit on
        #    the open price); exits begin from the bar after entry.
        # ----------------------------------------------------------------
        pos = open_pos[symbol]
        if pos is not None and idx > pos.entry_idx:
            pos.bars_held += 1
            stop_hit = False
            exit_price: Optional[float] = None

            if pos.direction == "LONG" and c_low <= pos.stop:
                stop_hit = True
                exit_price = pos.stop
            elif pos.direction == "SHORT" and c_high >= pos.stop:
                stop_hit = True
                exit_price = pos.stop

            should_exit = stop_hit or (pos.bars_held >= hold_bars)
            if should_exit:
                if exit_price is None:
                    exit_price = c_close  # exit at close on hold expiry
                exit_reason = "STOP" if stop_hit else "HOLD"
                exit_fee = exit_price * pos.size * _TAKER_FEE
                if pos.direction == "LONG":
                    raw_pnl = (exit_price - pos.entry_price) * pos.size
                else:
                    raw_pnl = (pos.entry_price - exit_price) * pos.size
                net_pnl = raw_pnl - exit_fee
                equity += raw_pnl - exit_fee
                trades.append(_Trade(
                    symbol          = symbol,
                    direction       = pos.direction,
                    entry_price     = pos.entry_price,
                    exit_price      = exit_price,
                    entry_ms        = pos.entry_ms,
                    exit_ms         = ts,
                    size            = pos.size,
                    pnl             = net_pnl,
                    fees            = pos.entry_fee + exit_fee,
                    stop_price      = pos.stop,
                    exit_reason     = exit_reason,
                    equity_at_entry = pos.equity_at_entry,
                ))
                open_pos[symbol] = None
                pos = None

        # ----------------------------------------------------------------
        # 2. Check for an extreme event entering at THIS bar's open.
        # ----------------------------------------------------------------
        ev = event_at_idx[symbol].get(idx)
        if ev is not None and open_pos[symbol] is None:
            atr = atr_by_symbol[symbol][idx]
            entry_price = c_open
            if atr is not None and atr > 0 and entry_price > 0:
                stop_dist = _STOP_MULT * atr
                if ev.side == "LONG":
                    stop_price = entry_price - stop_dist
                else:
                    stop_price = entry_price + stop_dist
                risk_dollars = equity * _RISK_PCT
                size = risk_dollars / stop_dist
                if size > 0:
                    entry_fee = entry_price * size * _TAKER_FEE
                    equity   -= entry_fee
                    open_pos[symbol] = _OpenPos(
                        symbol          = symbol,
                        direction       = ev.side,
                        entry_price     = entry_price,
                        entry_ms        = ts,
                        entry_idx       = idx,
                        size            = size,
                        stop            = stop_price,
                        entry_fee       = entry_fee,
                        bars_held       = 0,
                        equity_at_entry = equity,
                    )

        eq_curve.append((ts, equity))

    # Force-close any still-open positions at their last available bar close.
    for symbol in _SYMBOLS:
        pos = open_pos[symbol]
        if pos is None:
            continue
        candles = candles_by_symbol[symbol]
        last_idx = len(candles["close"]) - 1
        exit_price = candles["close"][last_idx]
        exit_ms    = candles["open_time"][last_idx]
        exit_fee   = exit_price * pos.size * _TAKER_FEE
        if pos.direction == "LONG":
            raw_pnl = (exit_price - pos.entry_price) * pos.size
        else:
            raw_pnl = (pos.entry_price - exit_price) * pos.size
        net_pnl = raw_pnl - exit_fee
        equity += raw_pnl - exit_fee
        trades.append(_Trade(
            symbol          = symbol,
            direction       = pos.direction,
            entry_price     = pos.entry_price,
            exit_price      = exit_price,
            entry_ms        = pos.entry_ms,
            exit_ms         = exit_ms,
            size            = pos.size,
            pnl             = net_pnl,
            fees            = pos.entry_fee + exit_fee,
            stop_price      = pos.stop,
            exit_reason     = "HOLD",
            equity_at_entry = pos.equity_at_entry,
        ))
        open_pos[symbol] = None

    if eq_curve:
        eq_curve.append((eq_curve[-1][0], equity))

    trades.sort(key=lambda t: t.entry_ms)
    _print_section_b(trades, eq_curve, initial_equity, equity, hold_bars, low_pct, high_pct)


def _max_dd(eq_curve: list[tuple[int, float]]) -> float:
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


def _pf(trade_list: list[_Trade]) -> float:
    gp = sum(t.pnl for t in trade_list if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trade_list if t.pnl <= 0))
    return gp / gl if gl > 0 else float("inf")


def _print_section_b(
    trades:         list[_Trade],
    eq_curve:       list[tuple[int, float]],
    initial_equity: float,
    final_equity:   float,
    hold_bars:      int,
    low_pct:        float,
    high_pct:       float,
) -> None:
    print("\n" + "=" * 70)
    print("  SECTION B — MONETIZATION BACKTEST (after fees)")
    print("=" * 70)

    if not trades:
        print("\n  [RESULT] No trades generated.")
        print("=" * 70)
        _print_verdict([], 0.0, 0.0, initial_equity, final_equity)
        return

    first_ms = min(t.entry_ms for t in trades)
    last_ms  = max(t.exit_ms  for t in trades)
    years    = (last_ms - first_ms) / (1000 * 86400 * 365.25)
    cagr     = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0.0
    max_drawdown = _max_dd(eq_curve)

    wins      = [t for t in trades if t.pnl > 0]
    pf_full   = _pf(trades)
    wr        = len(wins) / len(trades) * 100
    total_fees= sum(t.fees for t in trades)
    n_longs   = sum(1 for t in trades if t.direction == "LONG")
    n_shorts  = sum(1 for t in trades if t.direction == "SHORT")
    n_stops   = sum(1 for t in trades if t.exit_reason == "STOP")
    n_holds   = sum(1 for t in trades if t.exit_reason == "HOLD")

    print(f"  Parameters    : extreme={low_pct:.0f}/{high_pct:.0f}pct trailing-{_TRAIL_DAYS}d  "
          f"hold={hold_bars}bars  stop={_STOP_MULT}×ATR({_ATR_PERIOD})")
    print(f"  Period        : {datetime.fromtimestamp(first_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}"
          f" → {datetime.fromtimestamp(last_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"  Initial equity: ${initial_equity:,.0f}")
    print(f"  Final equity  : ${final_equity:,.0f}")
    print(f"  Net PnL       : ${final_equity - initial_equity:+,.0f}  ({(final_equity-initial_equity)/initial_equity*100:+.1f}%)")
    print(f"  CAGR          : {cagr*100:.1f}%")
    print(f"  Max drawdown  : {max_drawdown*100:.1f}%")
    print(f"  Profit factor : {pf_full:.2f}")
    print(f"  Total trades  : {len(trades)}  (L:{n_longs}  S:{n_shorts})")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Exit reasons  : HOLD={n_holds}  STOP={n_stops}")
    print(f"  Total fees    : ${total_fees:,.0f}")

    # Per-symbol breakdown
    print(f"\n  {'─'*70}")
    print(f"  {'Symbol':<10} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>12}")
    print(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*7} {'─'*12}")
    for symbol in _SYMBOLS:
        sym_trades = [t for t in trades if t.symbol == symbol]
        if not sym_trades:
            print(f"  {symbol:<10} {0:>7} {'n/a':>7} {'n/a':>7} {'$0':>12}")
            continue
        sym_wins = [t for t in sym_trades if t.pnl > 0]
        sym_pf   = _pf(sym_trades)
        sym_wr   = len(sym_wins) / len(sym_trades) * 100
        sym_pnl  = sum(t.pnl for t in sym_trades)
        pf_str   = f"{sym_pf:.2f}" if not math.isinf(sym_pf) else "inf"
        print(f"  {symbol:<10} {len(sym_trades):>7} {sym_wr:>6.1f}% {pf_str:>7} ${sym_pnl:>+10,.0f}")

    # Year-by-year breakdown
    years_seen = sorted({_year(t.entry_ms) for t in trades})
    print(f"\n  {'─'*70}")
    print(f"  {'Year':<8} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>12}  {'Status'}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*12}  {'─'*10}")
    for yr in years_seen:
        yr_trades = [t for t in trades if _year(t.entry_ms) == yr]
        if not yr_trades:
            continue
        yr_wins  = [t for t in yr_trades if t.pnl > 0]
        yr_gp    = sum(t.pnl for t in yr_wins)
        yr_gl    = abs(sum(t.pnl for t in yr_trades if t.pnl <= 0))
        yr_pf    = yr_gp / yr_gl if yr_gl > 0 else float("inf")
        yr_wr    = len(yr_wins) / len(yr_trades) * 100
        yr_pnl   = sum(t.pnl for t in yr_trades)
        status   = "+" if yr_pf >= 1.0 else "-"
        pf_str   = f"{yr_pf:>7.2f}" if not math.isinf(yr_pf) else f"{'inf':>7}"
        print(f"  {yr:<8} {len(yr_trades):>7} {yr_wr:>6.1f}% {pf_str} ${yr_pnl:>+10,.0f}  {status}")

    # Top 5 / bottom 5 by PnL
    sorted_by_pnl = sorted(trades, key=lambda t: t.pnl, reverse=True)
    print(f"\n  {'─'*70}")
    print("  Top 5 trades by PnL:")
    for t in sorted_by_pnl[:5]:
        dt = datetime.fromtimestamp(t.entry_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"    {t.symbol:<8} {t.direction:<5} entry {dt}  exit={t.exit_reason}  ${t.pnl:>+,.0f}")
    print("  Bottom 5 trades by PnL:")
    for t in sorted_by_pnl[-5:]:
        dt = datetime.fromtimestamp(t.entry_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"    {t.symbol:<8} {t.direction:<5} entry {dt}  exit={t.exit_reason}  ${t.pnl:>+,.0f}")

    # OOS split
    oos_trades = [t for t in trades if _year(t.entry_ms) >= _OOS_START_YEAR]
    is_trades  = [t for t in trades if _year(t.entry_ms) <  _OOS_START_YEAR]
    pf_oos     = _pf(oos_trades)
    pf_is      = _pf(is_trades)

    print(f"\n  {'─'*70}")
    print(f"  OOS split (IS = <{_OOS_START_YEAR}, OOS = {_OOS_START_YEAR}+):")
    if is_trades:
        print(f"    IS  ({min(_year(t.entry_ms) for t in is_trades)}"
              f"–{max(_year(t.entry_ms) for t in is_trades)}): "
              f"{len(is_trades)} trades, PF {pf_is:.2f}")
    else:
        print(f"    IS : no trades")
    if oos_trades:
        print(f"    OOS ({min(_year(t.entry_ms) for t in oos_trades)}"
              f"–{max(_year(t.entry_ms) for t in oos_trades)}): "
              f"{len(oos_trades)} trades, PF {pf_oos:.2f}")
    else:
        print(f"    OOS: no trades")

    print("=" * 70)
    _print_verdict(trades, pf_full, max_drawdown, initial_equity, final_equity,
                   pf_oos=pf_oos, pf_full_ref=pf_full)


def _print_verdict(
    trades:         list[_Trade],
    pf_full:        float,
    max_drawdown:   float,
    initial_equity: float,
    final_equity:   float,
    pf_oos:         float = float("nan"),
    pf_full_ref:    float = float("nan"),
) -> None:
    first_ms = min(t.entry_ms for t in trades) if trades else 0
    last_ms  = max(t.exit_ms  for t in trades) if trades else 0
    years    = (last_ms - first_ms) / (1000 * 86400 * 365.25) if last_ms > first_ms else 0
    cagr     = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 and initial_equity > 0 else 0.0

    kill_reasons: list[str] = []
    go_flags:     list[str] = []

    n_trades = len(trades)

    if pf_full < _KILL_PF:
        kill_reasons.append(f"Profit factor {pf_full:.2f} < kill threshold {_KILL_PF:.2f}")
    if max_drawdown > _KILL_MAX_DD:
        kill_reasons.append(f"Max drawdown {max_drawdown*100:.1f}% > kill threshold {_KILL_MAX_DD*100:.0f}%")
    if n_trades < _KILL_TRADES:
        kill_reasons.append(f"Trade count {n_trades} < minimum {_KILL_TRADES}")
    if not math.isnan(pf_oos) and not math.isnan(pf_full_ref):
        oos_threshold = _KILL_OOS_RATIO * pf_full_ref
        if pf_oos < oos_threshold:
            kill_reasons.append(
                f"OOS PF {pf_oos:.2f} < {_KILL_OOS_RATIO:.0%} × full-period PF {pf_full_ref:.2f} "
                f"(threshold {oos_threshold:.2f}) — regime-specific"
            )

    if pf_full >= _GO_PF:
        go_flags.append(f"PF {pf_full:.2f} >= {_GO_PF}")
    if cagr >= _GO_CAGR:
        go_flags.append(f"CAGR {cagr*100:.1f}% >= {_GO_CAGR*100:.0f}%")
    if max_drawdown <= _GO_DD:
        go_flags.append(f"Max DD {max_drawdown*100:.1f}% <= {_GO_DD*100:.0f}%")

    # MARGINAL if it passes PF, DD and trade-count kill gates but OOS is weak.
    passes_core = (
        pf_full >= _KILL_PF
        and max_drawdown <= _KILL_MAX_DD
        and n_trades >= _KILL_TRADES
    )
    oos_weak = False
    if not math.isnan(pf_oos) and not math.isnan(pf_full_ref):
        oos_weak = pf_oos < _KILL_OOS_RATIO * pf_full_ref

    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")

    if passes_core and oos_weak:
        # Survives the core gates but fails OOS robustness → MARGINAL, not KILL.
        print(f"\n  MARGINAL -- passes PF/DD/trade-count gates but OOS is weak:\n")
        print(f"     * PF {pf_full:.2f} >= {_KILL_PF:.2f} (kill gate)")
        print(f"     * Max DD {max_drawdown*100:.1f}% <= {_KILL_MAX_DD*100:.0f}% (kill gate)")
        print(f"     * Trades {n_trades} >= {_KILL_TRADES} (kill gate)")
        print(f"     * OOS PF {pf_oos:.2f} < {_KILL_OOS_RATIO:.0%} × full PF {pf_full_ref:.2f} "
              f"(threshold {_KILL_OOS_RATIO*pf_full_ref:.2f})")
        print("\n  Action: regime-sensitive — run paper trading before adding capital.")
    elif kill_reasons:
        print("\n  KILL -- Strategy fails one or more kill criteria:\n")
        for r in kill_reasons:
            print(f"     * {r}")
        print("\n  Action: do NOT deploy. Move to next hypothesis.")
    elif len(go_flags) == 3:
        print("\n  GO -- All go criteria met:\n")
        for f in go_flags:
            print(f"     * {f}")
        print("\n  Action: proceed to paper deployment.")
    else:
        passed = len(go_flags)
        print(f"\n  MARGINAL -- {passed}/3 go criteria met. Deploy on paper with caution.")
        print("  Passed  :", ", ".join(go_flags) if go_flags else "none")
        unmet = []
        if pf_full < _GO_PF:
            unmet.append(f"PF {pf_full:.2f} < {_GO_PF}")
        if cagr < _GO_CAGR:
            unmet.append(f"CAGR {cagr*100:.1f}% < {_GO_CAGR*100:.0f}%")
        if max_drawdown > _GO_DD:
            unmet.append(f"Max DD {max_drawdown*100:.1f}% > {_GO_DD*100:.0f}%")
        print("  Unmet   :", ", ".join(unmet))
        print("\n  Action: run paper trading before adding capital.")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Funding-Rate Extremes Backtest (contrarian fade, 1H)")
    parser.add_argument("--data",          default="data/candles",
                        help="Directory containing BTCUSDT_1H.parquet and ETHUSDT_1H.parquet")
    parser.add_argument("--funding-dir",   default="data/funding",
                        help="Directory for cached Binance funding parquets")
    parser.add_argument("--capital",       type=float, default=100_000.0,
                        help="Initial account equity in USDT (default 100000)")
    parser.add_argument("--threshold-pct", type=float, default=10.0,
                        help="Percentile for the extreme split (default 10 = 10/90)")
    parser.add_argument("--hold-hours",    type=int, default=_HOLD_HOURS,
                        help="Hold duration in hours / 1H bars (default 24)")
    args = parser.parse_args()

    data_dir    = _REPO_ROOT / args.data
    funding_dir = _REPO_ROOT / args.funding_dir
    low_pct     = args.threshold_pct
    high_pct    = 100.0 - args.threshold_pct
    hold_bars   = args.hold_hours

    print(f"\n{'='*70}")
    print("  FUNDING-RATE EXTREMES BACKTEST (contrarian fade) — 1H timeframe")
    print(f"{'='*70}")
    print(f"  Hypothesis: extreme POSITIVE funding (crowded longs) → fade SHORT;")
    print(f"              extreme NEGATIVE funding (crowded shorts) → fade LONG.")
    print(f"  Parameters used (pre-committed):")
    print(f"    extreme    = top/bottom {high_pct:.0f}/{low_pct:.0f} pct of TRAILING {_TRAIL_DAYS}d window")
    print(f"    min obs    = {_MIN_OBS} in trailing window (else skip — no lookahead)")
    print(f"    entry      = bar open at/after funding settlement (no lookahead)")
    print(f"    hold       = {hold_bars} bars ({hold_bars}h)  OR  stop, whichever first")
    print(f"    stop_mult  = {_STOP_MULT}x ATR({_ATR_PERIOD})")
    print(f"    fee        = {_TAKER_FEE*100:.4f}% taker (entry + exit)")
    print(f"    risk_pct   = {_RISK_PCT*100:.1f}% per trade")
    print(f"    capital    = ${args.capital:,.0f}")
    print(f"    symbols    = {', '.join(_SYMBOLS)}")

    # --- Funding (download/cache inline) ---
    print(f"\nAcquiring funding-rate history (cache dir: {funding_dir}) ...")
    funding_by_symbol: dict[str, list[dict]] = {}
    for symbol in _SYMBOLS:
        funding_by_symbol[symbol] = _load_funding(symbol, funding_dir)
    if any(not funding_by_symbol[s] for s in _SYMBOLS):
        print("[ERROR] Funding data missing for one or more symbols. Cannot continue.")
        sys.exit(1)

    # --- Candles ---
    print(f"\nLoading 1H candles from {data_dir} ...")
    candles_by_symbol: dict[str, dict] = {}
    for symbol in _SYMBOLS:
        candles_by_symbol[symbol] = _load_1h(symbol, data_dir)
        print(f"  {symbol}: {len(candles_by_symbol[symbol]['open_time']):,} hourly bars")

    # --- Build events (trailing-percentile rule, aligned to candles) ---
    print(f"\nBuilding extreme-funding events (trailing-{_TRAIL_DAYS}d percentile rule) ...")
    events_by_symbol: dict[str, list[_Event]] = {}
    for symbol in _SYMBOLS:
        evs = _build_events(
            symbol, funding_by_symbol[symbol], candles_by_symbol[symbol],
            low_pct, high_pct,
        )
        events_by_symbol[symbol] = evs
        first = datetime.fromtimestamp(evs[0].entry_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if evs else "n/a"
        last  = datetime.fromtimestamp(evs[-1].entry_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if evs else "n/a"
        print(f"  {symbol}: {len(evs):,} events  ({first} → {last})")

    total_events = sum(len(v) for v in events_by_symbol.values())
    if total_events == 0:
        print("[ERROR] No extreme-funding events found. Check data coverage.")
        sys.exit(1)

    # SECTION A — mechanism validity
    _section_a(events_by_symbol, candles_by_symbol, funding_by_symbol, low_pct, high_pct)

    # SECTION B — monetization backtest
    _section_b(events_by_symbol, candles_by_symbol, hold_bars, args.capital, low_pct, high_pct)


if __name__ == "__main__":
    main()
