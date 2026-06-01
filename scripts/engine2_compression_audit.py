#!/usr/bin/env python3
"""Engine #2 pre-audit diagnostics for post-compression volatility expansion.

Two sections:
  1. Compression Frequency Audit
     How often does the compression condition occur?
     Compression: range of prior 6 bars < 0.5 × 20-period median ATR(14)

  2. Expansion Quality Audit
     For each compression breakout event, measure max favorable excursion
     at +4H, +8H, +24H and report % reaching 0.5/1/2/3 ATR before
     suffering 1 ATR adverse move.

Usage:
  python scripts/engine2_compression_audit.py
  python scripts/engine2_compression_audit.py --symbols BTCUSDT
  python scripts/engine2_compression_audit.py --data data/candles
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

_DEFAULT_SYMBOLS   = ["BTCUSDT", "ETHUSDT"]
_ATR_PERIOD        = 14
_COMPRESS_BARS     = 6      # prior N bars must be compressed
_COMPRESS_THRESH   = 0.5    # fraction of median ATR
_MEDIAN_WINDOW     = 20     # bars for median ATR
_MFE_HORIZONS      = [4, 8, 24]
_ATR_TARGETS       = [0.5, 1.0, 2.0, 3.0]
_ADVERSE_LIMIT_ATR = 1.0


# ── ATR (Wilder) ─────────────────────────────────────────────────────────────

def _build_atr(bars: list[dict]) -> list[float]:
    """Return per-bar ATR list (nan for first ATR_PERIOD bars)."""
    atrs: list[float] = []
    prev_close = None
    atr = float("nan")
    for b in bars:
        h, l, c = b["high"], b["low"], b["close"]
        if prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        if len(atrs) < _ATR_PERIOD:
            atrs.append(float("nan"))
        elif len(atrs) == _ATR_PERIOD:
            # seed with simple mean of first ATR_PERIOD TRs
            seed_bars = bars[:_ATR_PERIOD + 1]
            trs = []
            for i in range(1, len(seed_bars)):
                hh = seed_bars[i]["high"]
                ll = seed_bars[i]["low"]
                pc = seed_bars[i - 1]["close"]
                trs.append(max(hh - ll, abs(hh - pc), abs(ll - pc)))
            atr = statistics.mean(trs) if trs else tr
            atrs.append(atr)
        else:
            atr = (atr * (_ATR_PERIOD - 1) + tr) / _ATR_PERIOD
            atrs.append(atr)
        prev_close = c
    return atrs


# ── Load data ────────────────────────────────────────────────────────────────

def _load(symbol: str, data_dir: Path) -> list[dict]:
    path = data_dir / f"{symbol}_1H.parquet"
    if not path.exists():
        print(f"[ERROR] {path} not found. Run: python scripts/download_1h_candles.py")
        sys.exit(1)
    tbl = pq.read_table(path)
    rows = tbl.to_pydict()
    n = len(rows["open_time"])
    bars = [
        {
            "open_time": rows["open_time"][i],
            "open":      rows["open"][i],
            "high":      rows["high"][i],
            "low":       rows["low"][i],
            "close":     rows["close"][i],
        }
        for i in range(n)
    ]
    bars.sort(key=lambda b: b["open_time"])
    return bars


# ── Section 1: Compression Frequency ────────────────────────────────────────

def _is_compressed(bars: list[dict], idx: int, atrs: list[float]) -> bool:
    """True if the _COMPRESS_BARS bars ending at idx-1 are in compression."""
    start = idx - _COMPRESS_BARS
    if start < 0:
        return False
    if any(float("nan") == atrs[j] or atrs[j] != atrs[j] for j in range(max(0, idx - _MEDIAN_WINDOW), idx)):
        return False

    # median ATR over the last MEDIAN_WINDOW bars (ending at idx-1)
    med_slice = [atrs[j] for j in range(max(0, idx - _MEDIAN_WINDOW), idx)
                 if atrs[j] == atrs[j]]  # nan check
    if len(med_slice) < _MEDIAN_WINDOW // 2:
        return False
    med_atr = statistics.median(med_slice)
    if med_atr <= 0:
        return False

    # range of last COMPRESS_BARS bars (bars[start..idx-1])
    window = bars[start:idx]
    bar_range = max(b["high"] for b in window) - min(b["low"] for b in window)
    return bar_range < _COMPRESS_THRESH * med_atr


def _frequency_audit(symbol: str, bars: list[dict], atrs: list[float]) -> None:
    total = 0
    compressed = 0
    by_month: dict[str, int] = defaultdict(int)
    by_month_total: dict[str, int] = defaultdict(int)

    for i in range(_COMPRESS_BARS + _MEDIAN_WINDOW, len(bars)):
        total += 1
        month = datetime.fromtimestamp(
            bars[i]["open_time"] / 1000, tz=timezone.utc
        ).strftime("%Y-%m")
        by_month_total[month] += 1
        if _is_compressed(bars, i, atrs):
            compressed += 1
            by_month[month] += 1

    pct = compressed / total * 100 if total else 0
    print(f"\n{'─'*60}")
    print(f"  {symbol}  Compression Frequency")
    print(f"{'─'*60}")
    print(f"  Eligible bars : {total:,}")
    print(f"  Compressed    : {compressed:,}  ({pct:.1f}% of bars)")

    months = sorted(by_month_total)
    if months:
        print(f"\n  Monthly breakdown  (compressed / total  %)")
        print(f"  {'Month':<10}  {'Compressed':>10}  {'Total':>7}  {'%':>6}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*6}")
        for m in months:
            c  = by_month.get(m, 0)
            t  = by_month_total[m]
            pp = c / t * 100 if t else 0
            print(f"  {m:<10}  {c:>10,}  {t:>7,}  {pp:>5.1f}%")


# ── Section 2: Expansion Quality ────────────────────────────────────────────

def _expansion_audit(symbol: str, bars: list[dict], atrs: list[float]) -> None:
    """
    For every bar i where compression ends AND bar i is a breakout:
      - Long breakout:  close[i] > max(high of compress window)
      - Short breakout: close[i] < min(low  of compress window)

    Walk forward up to +24H. Track MFE (favorable) and MAE (adverse).
    """
    horizon_max = max(_MFE_HORIZONS)
    results_long:  list[dict] = []
    results_short: list[dict] = []

    for i in range(_COMPRESS_BARS + _MEDIAN_WINDOW, len(bars) - horizon_max):
        if not _is_compressed(bars, i, atrs):
            continue

        atr = atrs[i]
        if atr != atr or atr <= 0:
            continue

        # compression window = bars[i - COMPRESS_BARS : i]
        comp_window = bars[i - _COMPRESS_BARS: i]
        comp_high = max(b["high"] for b in comp_window)
        comp_low  = min(b["low"]  for b in comp_window)
        current_close = bars[i]["close"]

        if current_close > comp_high:
            direction = "long"
            entry = current_close
        elif current_close < comp_low:
            direction = "short"
            entry = current_close
        else:
            continue  # not a breakout bar

        mfe_at: dict[int, float] = {}
        mae_at: dict[int, float] = {}
        running_mfe = 0.0
        running_mae = 0.0
        adverse_hit = False
        adverse_hit_before: dict[float, bool] = {t: False for t in _ATR_TARGETS}

        for fwd in range(1, horizon_max + 1):
            fb = bars[i + fwd]
            if direction == "long":
                fav = (fb["high"]  - entry) / atr
                adv = (entry - fb["low"])   / atr
            else:
                fav = (entry - fb["low"])   / atr
                adv = (fb["high"] - entry)  / atr

            running_mfe = max(running_mfe, fav)
            running_mae = max(running_mae, adv)

            if running_mae >= _ADVERSE_LIMIT_ATR:
                adverse_hit = True

            if fwd in _MFE_HORIZONS:
                mfe_at[fwd] = running_mfe
                mae_at[fwd] = running_mae

        record = {
            "entry": entry,
            "atr":   atr,
            "mfe":   mfe_at,
            "mae":   mae_at,
            "adverse_before_target": {
                t: (running_mae >= _ADVERSE_LIMIT_ATR and running_mfe < t)
                for t in _ATR_TARGETS
            },
        }
        if direction == "long":
            results_long.append(record)
        else:
            results_short.append(record)

    def _report(label: str, results: list[dict]) -> None:
        n = len(results)
        if n == 0:
            print(f"\n  {label}: 0 events")
            return
        print(f"\n  {label}  ({n:,} breakout events)")
        # MFE at each horizon
        print(f"\n  Median MFE (in ATR units) at each horizon:")
        print(f"  {'Horizon':>8}  {'Median MFE':>12}  {'Mean MFE':>10}  {'P75 MFE':>9}")
        print(f"  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*9}")
        for h in _MFE_HORIZONS:
            vals = [r["mfe"][h] for r in results if h in r["mfe"]]
            if not vals:
                continue
            print(f"  {f'+{h}H':>8}  {statistics.median(vals):>12.3f}  "
                  f"{statistics.mean(vals):>10.3f}  "
                  f"{sorted(vals)[int(len(vals)*0.75)]:>9.3f}")

        # % reaching ATR target before 1 ATR adverse
        print(f"\n  % reaching target before {_ADVERSE_LIMIT_ATR}×ATR adverse:")
        print(f"  {'Target':>8}  {'Reached (%)':>12}  {'n':>6}")
        print(f"  {'─'*8}  {'─'*12}  {'─'*6}")
        final_mfe = [r["mfe"].get(max(_MFE_HORIZONS), 0) for r in results]
        final_mae = [r["mae"].get(max(_MFE_HORIZONS), 0) for r in results]
        for t in _ATR_TARGETS:
            reached = sum(
                1 for mfe, mae in zip(final_mfe, final_mae)
                if mfe >= t and mae < _ADVERSE_LIMIT_ATR
            )
            pct = reached / n * 100
            print(f"  {f'{t}×ATR':>8}  {pct:>11.1f}%  {reached:>6,}")

    print(f"\n{'─'*60}")
    print(f"  {symbol}  Expansion Quality")
    print(f"{'─'*60}")
    _report("LONG breakouts", results_long)
    _report("SHORT breakouts", results_short)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--data",    default="data/candles")
    args = parser.parse_args()

    data_dir = _REPO_ROOT / args.data

    print("=" * 60)
    print("  Engine #2 Pre-Audit: Compression Diagnostics")
    print(f"  Compression: prior {_COMPRESS_BARS} bars range < "
          f"{_COMPRESS_THRESH}× median ATR({_ATR_PERIOD})")
    print("=" * 60)

    for symbol in args.symbols:
        bars = _load(symbol, data_dir)
        atrs = _build_atr(bars)
        print(f"\n[{symbol}]  {len(bars):,} hourly bars loaded")
        _frequency_audit(symbol, bars, atrs)
        _expansion_audit(symbol, bars, atrs)

    print(f"\n{'═'*60}")
    print("  Audit complete.")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
