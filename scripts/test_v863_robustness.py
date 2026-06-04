#!/usr/bin/env python3
"""V8.63 Robustness Check — is it robust or overfit? ($5K account)

Two tests, both designed to TRY TO BREAK V8.63 (not to improve it):

  1. Parameter sensitivity — perturb each key knob one at a time around its
     chosen value. A robust system degrades gracefully; an overfit one falls off
     a cliff the moment you move away from the exact tuned value. We want to see
     a flat plateau around the chosen params, not a sharp spike.

  2. Walk-forward — run the FULL V8.63 (not a stripped base) on sliding
     out-of-sample windows it was never "designed" on. Consistent positive OOS
     windows = the edge generalizes across regimes.

Chosen V8.63 params: confirm_days=10, momentum_gate_days=20, bear_drop_pct=-0.20,
hard_stop_pct=0.15, target_risk=0.01.

Usage:
  python scripts/test_v863_robustness.py
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

_FROM_TS = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_TO_TS = int(datetime.now(timezone.utc).timestamp() * 1000)

# Chosen V8.63 parameters
_BASE = dict(
    hard_stop_pct=0.15, use_atr_sizing=True,
    asymmetric_regime=True, and_entry=True,
    bear_drop_pct=-0.20, confirm_days=10,
    momentum_gate=True, momentum_gate_days=20,
)


def _run(from_ts, to_ts, **override) -> dict:
    params = dict(_BASE)
    params.update(override)
    return B._run(_SIG, True, True, False, True, from_ts, to_ts, **params)


def _line(label: str, r: dict, star: bool = False) -> None:
    losing = [yr for yr in r["year_pnl"] if r["year_pnl"][yr] < 0]
    mark = " <- chosen" if star else ""
    print(f"  {label:<22} eq=${r['equity']:>8,.0f}  CAGR={r['cagr']*100:>5.1f}%  "
          f"DD={r['max_dd']*100:>4.1f}%  PF={r['pf']:.2f}  Sharpe={r['sharpe']:.2f}  "
          f"losing={losing}{mark}")


def _sweep(name: str, key: str, values: list, chosen) -> None:
    print(f"\n  --- {name} (perturbing {key}) ---")
    for v in values:
        r = _run(_FROM_TS, _TO_TS, **{key: v})
        _line(f"{key}={v}", r, star=(v == chosen))


def main() -> None:
    global _SIG
    print("=" * 84)
    print("  V8.63 ROBUSTNESS CHECK — trying to break it, not improve it ($5K)")
    print("=" * 84)
    print("Building signals ...")
    _SIG = B._build_signals(B._SYMBOLS)
    print("Done.")

    print("\n" + "=" * 84)
    print("  TEST 1: PARAMETER SENSITIVITY (one knob at a time)")
    print("=" * 84)
    base = _run(_FROM_TS, _TO_TS)
    _line("baseline V8.63", base, star=True)

    _sweep("Bear-exit confirm days", "confirm_days", [5, 8, 10, 12, 15, 20], 10)
    _sweep("Momentum gate lookback", "momentum_gate_days", [10, 15, 20, 25, 30], 20)
    _sweep("Bear-entry 30d drop",   "bear_drop_pct", [-0.15, -0.18, -0.20, -0.22, -0.25], -0.20)
    _sweep("Hard stop %",           "hard_stop_pct", [0.10, 0.12, 0.15, 0.18, 0.20], 0.15)
    _sweep("Target daily risk",     "target_risk", [0.005, 0.0075, 0.01, 0.0125, 0.015], 0.01)

    print("\n" + "=" * 84)
    print("  TEST 2: WALK-FORWARD (full V8.63 on sliding OOS windows)")
    print("=" * 84)
    train_ms = int(2 * 365.25 * _DAY_MS)
    test_ms = int(0.5 * 365.25 * _DAY_MS)
    step_ms = test_ms
    print(f"  {'OOS window':<26} {'CAGR':>7} {'DD':>6} {'PF':>6} {'Sharpe':>7}  Verdict")
    print(f"  {'-'*26} {'-'*7} {'-'*6} {'-'*6} {'-'*7}  {'-'*7}")
    ws = _FROM_TS
    npass = ntot = 0
    while ws + train_ms + test_ms <= _TO_TS:
        te = ws + train_ms
        tee = min(te + test_ms, _TO_TS)
        r = _run(te, tee)
        d0 = datetime.fromtimestamp(te / 1000, tz=timezone.utc).strftime("%Y-%m")
        d1 = datetime.fromtimestamp(tee / 1000, tz=timezone.utc).strftime("%Y-%m")
        ok = r["pf"] >= 1.10 and r["cagr"] > 0
        npass += ok; ntot += 1
        print(f"  {d0+' -> '+d1:<26} {r['cagr']*100:>6.1f}% {r['max_dd']*100:>5.1f}% "
              f"{r['pf']:>6.2f} {r['sharpe']:>7.2f}  {'PASS' if ok else 'FAIL'}")
        ws += step_ms
    print(f"\n  Walk-forward: {npass}/{ntot} OOS windows passed (PF>=1.10 and CAGR>0)")

    print("\n" + "=" * 84)
    print("  READ: flat plateaus in TEST 1 = not knife-edge / not overfit.")
    print("  High pass-rate in TEST 2 = edge generalizes to unseen periods.")
    print("=" * 84)


if __name__ == "__main__":
    main()
