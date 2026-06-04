#!/usr/bin/env python3
"""Deep dive on the efficiency-ratio lever for V8.63 — $5K account.

The hard ER gate fixed the H1-2025 chop window (-31% -> +40%) but cost ~7%
full-period CAGR. This script tests whether ER-as-a-SIZING-DIAL (scale down in
chop instead of fully blocking) keeps the protection at lower cost, then runs a
proper walk-forward + IS/OOS split on the best candidates so we don't fool
ourselves with one lucky window.

Sections:
  1. ER sizing-dial sweep (full run + isolated H1-2025)
  2. Gate + sizing combos
  3. Walk-forward on the strongest candidates vs baseline
  4. IS (2021-2023) / OOS (2024-2026) split

Usage:
  python scripts/test_er_deep.py
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
_TO = int(datetime.now(timezone.utc).timestamp() * 1000)
_H1A = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_H1B = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)

_BASE = dict(
    hard_stop_pct=0.15, use_atr_sizing=True,
    asymmetric_regime=True, and_entry=True,
    bear_drop_pct=-0.20, confirm_days=10,
    momentum_gate=True, momentum_gate_days=20,
)


def _run(a, b, **o):
    p = dict(_BASE); p.update(o)
    return B._run(_SIG, True, True, False, True, a, b, **p)


def _row(label, rf, rh):
    losing = [yr for yr in rf["year_pnl"] if rf["year_pnl"][yr] < 0]
    print(f"  {label:<28} FULL eq=${rf['equity']:>8,.0f} CAGR={rf['cagr']*100:>5.1f}% "
          f"DD={rf['max_dd']*100:>4.1f}% Sh={rf['sharpe']:.2f} lose={str(losing):<14} | "
          f"H1-25 {rh['cagr']*100:>+6.1f}% DD={rh['max_dd']*100:>4.1f}%")


def _walk(label, **o):
    train = int(2 * 365.25 * _DAY_MS); test = int(0.5 * 365.25 * _DAY_MS)
    ws = _FROM; npass = ntot = 0; cagrs = []
    while ws + train + test <= _TO:
        r = _run(ws + train, min(ws + train + test, _TO), **o)
        ok = r["pf"] >= 1.10 and r["cagr"] > 0
        npass += ok; ntot += 1; cagrs.append(r["cagr"] * 100)
        ws += test
    avg = sum(cagrs) / len(cagrs) if cagrs else 0
    print(f"  {label:<28} walk-fwd {npass}/{ntot} pass   avg OOS CAGR {avg:>+6.1f}%   "
          f"windows={['%.0f'%c for c in cagrs]}")


def main():
    global _SIG
    print("=" * 104)
    print("  EFFICIENCY-RATIO DEEP DIVE — V8.63, $5K")
    print("=" * 104)
    print("Building signals ...")
    _SIG = B._build_signals(B._SYMBOLS)
    print("Done.\n")

    print("SECTION 1 — ER sizing dial (scale down in chop, never fully block)")
    _row("baseline", _run(_FROM, _TO), _run(_H1A, _H1B))
    for lo, hi in ((0.15, 0.40), (0.20, 0.45), (0.25, 0.50)):
        for floor in (0.5, 0.33, 0.25):
            kw = dict(er_sizing=True, er_days=30, er_size_lo=lo, er_size_hi=hi, er_size_floor=floor)
            _row(f"size lo={lo} hi={hi} fl={floor}", _run(_FROM, _TO, **kw), _run(_H1A, _H1B, **kw))

    print("\nSECTION 2 — gate + sizing combos")
    combos = {
        "gate thr=0.25 only": dict(er_gate=True, er_days=30, er_threshold=0.25),
        "size 0.2-0.45 fl0.33": dict(er_sizing=True, er_days=30, er_size_lo=0.20, er_size_hi=0.45, er_size_floor=0.33),
        "gate0.22 + size fl0.4": dict(er_gate=True, er_threshold=0.22, er_sizing=True,
                                       er_days=30, er_size_lo=0.20, er_size_hi=0.45, er_size_floor=0.4),
    }
    for label, kw in combos.items():
        _row(label, _run(_FROM, _TO, **kw), _run(_H1A, _H1B, **kw))

    print("\nSECTION 3 — walk-forward (full V8.63 on sliding OOS windows)")
    _walk("baseline")
    _walk("gate thr=0.25", er_gate=True, er_days=30, er_threshold=0.25)
    _walk("gate thr=0.30", er_gate=True, er_days=30, er_threshold=0.30)
    _walk("size 0.2-0.45 fl0.33", er_sizing=True, er_days=30, er_size_lo=0.20, er_size_hi=0.45, er_size_floor=0.33)

    print("\nSECTION 4 — IS (2021-23) / OOS (2024-26) split")
    is_b = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    for label, kw in {"baseline": {}, "gate thr=0.30": dict(er_gate=True, er_days=30, er_threshold=0.30),
                      "size fl0.33": dict(er_sizing=True, er_days=30, er_size_lo=0.20, er_size_hi=0.45, er_size_floor=0.33)}.items():
        ris = _run(_FROM, is_b, **kw); roos = _run(is_b, _TO, **kw)
        print(f"  {label:<28} IS CAGR={ris['cagr']*100:>5.1f}% DD={ris['max_dd']*100:>4.1f}% "
              f"Sh={ris['sharpe']:.2f}  | OOS CAGR={roos['cagr']*100:>5.1f}% "
              f"DD={roos['max_dd']*100:>4.1f}% Sh={roos['sharpe']:.2f}")

    print("\n" + "=" * 104)


if __name__ == "__main__":
    main()
