#!/usr/bin/env python3
"""Chop-gate and correlation-sizing research for V8.63 — $5K account.

Targets the ONE documented weakness of V8.63: choppy-bull whipsaw (the H1-2025
type stretch the walk-forward flagged at -30%). Two principled, non-curve-fit
ideas, each with clear economic logic:

  E. Efficiency-Ratio (choppiness) gate — Kaufman ER measures how directional
     price is (1=clean trend, 0=pure noise). Block NEW longs when the market is
     choppy. Tested both market-wide (BTC ER) and per-coin.

  F. Correlation-aware sizing — when all 12 coins move as one, diversification
     is illusory and effective leverage is high. Scale the whole book down as
     average pairwise correlation rises.

Judged the right way (after the robustness lesson): an idea only counts if it
(a) helps the FULL run AND (b) specifically improves the weak H1-2025 window,
without gutting bull-year capture. A wash = reject.

Usage:
  python scripts/test_chop_and_corr.py
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
# The weak window the walk-forward flagged.
_H1_25_A = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_H1_25_B = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)

_BASE = dict(
    hard_stop_pct=0.15, use_atr_sizing=True,
    asymmetric_regime=True, and_entry=True,
    bear_drop_pct=-0.20, confirm_days=10,
    momentum_gate=True, momentum_gate_days=20,
)


def _run(from_ts, to_ts, **override):
    p = dict(_BASE); p.update(override)
    return B._run(_SIG, True, True, False, True, from_ts, to_ts, **p)


def _row(label, rf, rh):
    losing = [yr for yr in rf["year_pnl"] if rf["year_pnl"][yr] < 0]
    print(f"  {label:<30} FULL: eq=${rf['equity']:>8,.0f} CAGR={rf['cagr']*100:>5.1f}% "
          f"DD={rf['max_dd']*100:>4.1f}% Sharpe={rf['sharpe']:.2f} lose={losing}  | "
          f"H1-25: {rh['cagr']*100:>+6.1f}% DD={rh['max_dd']*100:>4.1f}%")


def main():
    global _SIG
    print("=" * 100)
    print("  CHOP-GATE & CORRELATION-SIZING — V8.63, $5K  (FULL run + isolated H1-2025 window)")
    print("=" * 100)
    print("Building signals ...")
    _SIG = B._build_signals(B._SYMBOLS)
    print("Done.\n")

    bf, bh = _run(_FROM, _TO), _run(_H1_25_A, _H1_25_B)
    _row("V8.63 baseline", bf, bh)

    print("\n  --- E. Efficiency-ratio chop gate (market-wide BTC) ---")
    for days in (20, 30):
        for thr in (0.20, 0.30, 0.40):
            kw = dict(er_gate=True, er_days=days, er_threshold=thr, er_use_btc=True)
            _row(f"BTC-ER d={days} thr={thr}", _run(_FROM, _TO, **kw), _run(_H1_25_A, _H1_25_B, **kw))

    print("\n  --- E2. Efficiency-ratio chop gate (per-coin) ---")
    for thr in (0.20, 0.30, 0.40):
        kw = dict(er_gate=True, er_days=30, er_threshold=thr, er_use_btc=False)
        _row(f"coin-ER d=30 thr={thr}", _run(_FROM, _TO, **kw), _run(_H1_25_A, _H1_25_B, **kw))

    print("\n  --- F. Correlation-aware sizing ---")
    for ref in (0.4, 0.5, 0.6):
        for floor in (0.5, 0.33):
            kw = dict(corr_sizing=True, corr_days=30, corr_ref=ref, corr_floor=floor)
            _row(f"corr ref={ref} floor={floor}", _run(_FROM, _TO, **kw), _run(_H1_25_A, _H1_25_B, **kw))

    print("\n" + "=" * 100)
    print("  KEEP only if FULL stays ~baseline AND H1-25 visibly improves. Wash/worse = reject.")
    print("=" * 100)


if __name__ == "__main__":
    main()
