#!/usr/bin/env python3
"""Universe expansion research — V8.63 on 12 majors vs expanded mid-cap universes.

Hypothesis: trend-following edges are stronger/more numerous in mid-caps
(slower information diffusion, more retail). Same V8.63 rules, more coins —
diversification raises risk-adjusted return without touching the logic.

Honest-test notes:
  - Survivorship bias is real: mid-caps that died (LUNA, FTT) aren't in our
    list because we picked coins that still exist. Results are optimistic.
    We flag this rather than pretend otherwise.
  - Capital per coin shrinks as the universe grows (same $5K, more slots).
    The engine sizes per-coin off base_notional = capital/N automatically.
  - BTC remains the regime anchor in all variants.

Variants:
  A. V8.63 baseline — 12 majors (current live system)
  B. 12 majors + 8 mid-caps  (20 coins)
  C. 12 majors + 20 mid-caps (32 coins)
  D. Mid-caps only (20 coins, BTC kept for regime signal but not traded)

Usage:
  python scripts/test_universe_expansion.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import scripts.backtest_full_regime_system as B  # noqa: E402

B._CAPITAL = 5_000.0
_DAY_MS = 86_400_000
_FROM = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_TO   = int(datetime.now(timezone.utc).timestamp() * 1000)
_H1A  = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_H1B  = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)

_MAJORS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]

_MIDCAPS_8 = [
    "ATOMUSDT", "NEARUSDT", "FILUSDT", "UNIUSDT",
    "AAVEUSDT", "ETCUSDT", "XLMUSDT", "ALGOUSDT",
]

_MIDCAPS_20 = _MIDCAPS_8 + [
    "SANDUSDT", "MANAUSDT", "VETUSDT", "THETAUSDT",
    "AXSUSDT", "CRVUSDT", "GRTUSDT", "CHZUSDT",
    "GALAUSDT", "ICPUSDT", "EGLDUSDT", "KSMUSDT",
]

_BASE = dict(
    hard_stop_pct=0.15, use_atr_sizing=True,
    asymmetric_regime=True, and_entry=True,
    bear_drop_pct=-0.20, confirm_days=10,
    momentum_gate=True, momentum_gate_days=20,
)


def _run_universe(symbols: list[str], from_ts: int, to_ts: int, **override) -> dict:
    """Run V8.63 on an arbitrary coin universe. BTC must be in the signals dict
    for the regime gate even if not traded."""
    p = dict(_BASE)
    p.update(override)
    need = sorted(set(symbols) | {"BTCUSDT"})
    sig = B._build_signals(need)
    old_syms = B._SYMBOLS
    B._SYMBOLS = symbols
    try:
        return B._run(sig, True, True, False, True, from_ts, to_ts, **p)
    finally:
        B._SYMBOLS = old_syms


def _available(symbols: list[str]) -> list[str]:
    """Filter to coins whose daily parquet actually exists."""
    out = []
    for s in symbols:
        if (B._CANDLE_DIR / f"{s}_1D.parquet").exists():
            out.append(s)
        else:
            print(f"  [skip] {s}: no daily parquet")
    return out


def _year_table(rows: list[tuple[str, dict]]):
    all_years = sorted({yr for _, r in rows for yr in r["year_pnl"]})
    col = 12
    print(f"  {'System':<34}" + "".join(f"{yr:>{col}}" for yr in all_years)
          + f"{'Total Eq':>{col}}  {'CAGR':>7}  {'DD':>6}  {'Sharpe':>6}")
    print("  " + "-" * (34 + col * len(all_years) + col + 28))
    for label, r in rows:
        s = f"  {label:<34}"
        for yr in all_years:
            s += f"{r['year_pnl'].get(yr, 0):>+{col},.0f}"
        s += f"  ${r['equity']:>9,.0f}  {r['cagr']*100:>6.1f}%  {r['max_dd']*100:>5.1f}%  {r['sharpe']:>6.2f}"
        print(s)


def _walk(label: str, symbols: list[str]) -> None:
    train = int(2 * 365.25 * _DAY_MS)
    test  = int(0.5 * 365.25 * _DAY_MS)
    ws = _FROM; npass = ntot = 0; cagrs = []
    while ws + train + test <= _TO:
        r = _run_universe(symbols, ws + train, min(ws + train + test, _TO))
        ok = r["pf"] >= 1.10 and r["cagr"] > 0
        npass += ok; ntot += 1; cagrs.append(r["cagr"] * 100)
        ws += test
    avg = sum(cagrs) / len(cagrs) if cagrs else 0.0
    print(f"  {label:<34} walk-fwd {npass}/{ntot}  avg OOS CAGR {avg:>+6.1f}%  "
          f"windows={['%+.0f%%' % c for c in cagrs]}")


def main():
    print("=" * 112)
    print("  UNIVERSE EXPANSION — V8.63 rules, more coins ($5K)")
    print("=" * 112)

    mid8  = _available(_MIDCAPS_8)
    mid20 = _available(_MIDCAPS_20)
    print(f"\n  Mid-caps available: {len(mid20)}/20  (first tier {len(mid8)}/8)")

    universes = {
        "A. 12 majors (V8.63 live)": _MAJORS,
        "B. majors + 8 mid (20)":    _MAJORS + mid8,
        "C. majors + 20 mid (32)":   _MAJORS + mid20,
        "D. 20 mid-caps only":       mid20,
    }

    print("\n  Running full-period backtests ...")
    rows = []
    h1_rows = []
    for label, syms in universes.items():
        if not syms:
            continue
        rf = _run_universe(syms, _FROM, _TO)
        rh = _run_universe(syms, _H1A, _H1B)
        rows.append((label, rf))
        h1_rows.append((label, rh))
        print(f"    {label}: done ({len(syms)} coins)")

    print("\n" + "=" * 112)
    print("  YEAR-BY-YEAR EARNINGS — $5K starting, 2021 → 2026")
    print("=" * 112)
    _year_table(rows)

    print("\n  H1-2025 isolated window (the known weak spot):")
    for label, rh in h1_rows:
        print(f"    {label:<34} CAGR={rh['cagr']*100:>+6.1f}%  DD={rh['max_dd']*100:>4.1f}%")

    print("\n" + "=" * 112)
    print("  WALK-FORWARD (2yr train / 6mo test)")
    print("=" * 112)
    for label, syms in universes.items():
        if syms:
            _walk(label, syms)

    print("\n" + "=" * 112)
    print("  CAVEAT: mid-cap list has survivorship bias (dead coins excluded).")
    print("  Treat improvements as an upper bound; demand a LARGE margin before adding.")
    print("=" * 112)


if __name__ == "__main__":
    main()
