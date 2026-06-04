#!/usr/bin/env python3
"""Funding-Rate Crowding Filter for V8.63 — research harness.

Hypothesis
----------
On perpetual futures, the funding rate is a direct read on positioning. When
funding is extremely HIGH, the book is crowded long and longs are paying carry
every 8h just to hold — historically a sign of an overheated, late-stage rally.
If we SKIP new long entries while BTC funding is in an extreme-high state, we
should avoid the worst tops and reduce drawdown without giving up much return.

This uses BTC funding as a MARKET-WIDE gauge (same philosophy as the V8.63
regime gate, which uses BTC's SMA200 to govern the whole 12-coin book). We only
have full funding history for BTC and ETH committed to the repo, and BTC is the
market driver, so BTC funding is the right single-instrument proxy.

Method (no lookahead)
---------------------
  1. Load BTC 8h funding settlements (00/08/16 UTC).
  2. Aggregate to a DAILY funding figure = sum of that UTC day's settlements
     (≈ the daily carry a long pays). Align to the daily candle timestamp.
  3. At each day, compute the trailing-`window`-day percentile of daily funding
     using ONLY prior days. If today's funding >= the `pct`-th percentile, flag
     the day as "extreme high" → block new longs.
  4. Run V8.63 with and without the gate; compare year-by-year on a $5K account.

We sweep a few (window, pct) settings so we can see whether the edge is robust
or just one lucky knob.

Usage:
  python scripts/test_funding_filter.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pyarrow.parquet as pq  # noqa: E402

import scripts.backtest_full_regime_system as B  # noqa: E402

_DAY_MS = 86_400_000
_FUNDING_PATH = _REPO_ROOT / "data" / "funding" / "BTCUSDT_funding.parquet"

# Run on a $5K account to match how the live bot will actually start.
B._CAPITAL = 5_000.0


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolated percentile; `sorted_vals` ascending, pct in [0,100]."""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _load_daily_btc_funding() -> list[tuple[int, float]]:
    """Return [(day_ts, daily_funding_sum)] sorted ascending.

    day_ts is the UTC midnight ms for that day (matches daily candle open_time).
    """
    tbl = pq.read_table(_FUNDING_PATH).to_pydict()
    by_day: dict[int, float] = {}
    for ms, rate in zip(tbl["timestamp_ms"], tbl["funding_rate"]):
        day = (int(ms) // _DAY_MS) * _DAY_MS
        by_day[day] = by_day.get(day, 0.0) + float(rate)
    return sorted(by_day.items())


def _build_funding_high(window: int, pct: float) -> tuple[dict[int, bool], float]:
    """Map day_ts -> True if that day's funding is an extreme-high (trailing).

    Returns (flag_map, blocked_fraction). No lookahead: the percentile at day i
    is computed from days [i-window, i) only (prior days, current excluded).
    """
    daily = _load_daily_btc_funding()
    flag: dict[int, bool] = {}
    n_blocked = 0
    n_eval = 0
    for i, (day, f) in enumerate(daily):
        lo = max(0, i - window)
        window_vals = [v for _, v in daily[lo:i]]  # prior only
        if len(window_vals) < max(20, window // 4):
            flag[day] = False
            continue
        thr = _percentile(sorted(window_vals), pct)
        n_eval += 1
        is_high = f >= thr
        flag[day] = is_high
        if is_high:
            n_blocked += 1
    frac = (n_blocked / n_eval) if n_eval else 0.0
    return flag, frac


def _run_v863(funding_high: dict | None) -> dict:
    return B._run(
        B._build_signals(B._SYMBOLS) if not hasattr(_run_v863, "_sig") else _run_v863._sig,
        True, True, False, True, _FROM_TS, _TO_TS,
        hard_stop_pct=0.15, use_atr_sizing=True,
        asymmetric_regime=True, and_entry=True,
        bear_drop_pct=-0.20, confirm_days=10,
        momentum_gate=True, momentum_gate_days=20,
        funding_high=funding_high,
    )


_FROM_TS = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_TO_TS = int(datetime.now(timezone.utc).timestamp() * 1000)


def _year_table(label: str, r: dict) -> None:
    yrs = sorted(r["year_pnl"])
    cells = "  ".join(f"{yr}:{r['year_pnl'][yr]:>+7,.0f}" for yr in yrs)
    losing = [yr for yr in yrs if r["year_pnl"][yr] < 0]
    print(f"  {label:<22} eq=${r['equity']:>8,.0f}  CAGR={r['cagr']*100:>5.1f}%  "
          f"DD={r['max_dd']*100:>4.1f}%  PF={r['pf']:.2f}  Sharpe={r['sharpe']:.2f}  "
          f"n={r['n']}  losing={losing}")
    print(f"      {cells}")


def main() -> None:
    print("=" * 78)
    print("  FUNDING-RATE CROWDING FILTER — V8.63 with/without, $5K account")
    print("=" * 78)
    print("  Gate: block NEW longs when BTC daily funding >= trailing percentile.")
    print("  (Existing longs are never force-closed; shorts unaffected.)\n")

    print("Building signals for all 12 coins ...")
    signals = B._build_signals(B._SYMBOLS)
    _run_v863._sig = signals  # type: ignore[attr-defined]
    print("Done.\n")

    daily = _load_daily_btc_funding()
    print(f"BTC daily funding series: {len(daily)} days "
          f"({datetime.fromtimestamp(daily[0][0]/1000, tz=timezone.utc).date()} → "
          f"{datetime.fromtimestamp(daily[-1][0]/1000, tz=timezone.utc).date()})\n")

    base = _run_v863(None)
    _year_table("V8.63 (baseline)", base)
    print()

    # Sweep trailing window and extreme percentile.
    for window in (90, 180):
        for pct in (80, 90, 95):
            flag, frac = _build_funding_high(window, pct)
            r = _run_v863(flag)
            _year_table(f"+gate w={window} p={pct} ({frac*100:.0f}% blk)", r)
    print()
    print("=" * 78)
    print("  Read: a useful gate keeps CAGR near baseline while cutting DD and")
    print("  improving the worst years (2022/2025). If it just shaves return with")
    print("  no DD benefit, the funding signal isn't adding edge over the regime gate.")
    print("=" * 78)


if __name__ == "__main__":
    main()
