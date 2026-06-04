#!/usr/bin/env python3
"""Trailing-Stop research for V8.63 longs — $5K account.

The 2023/2025 losing years bled mostly from LONGS giving back open profit in
choppy markets: a coin runs up, the EMA/MACD signal lags, and by the time it
flips the position has round-tripped. A trailing stop locks in a portion of the
move and exits faster than the lagging trend signal.

Two flavours tested here, both on LONGS only (shorts use the existing 15% hard
stop and TSMOM rebalance logic, untouched):

  A. Fixed-% trailing  — exit when price falls `x%` from its peak since entry.
  B. ATR trailing       — exit when price falls `k × ATR(14)` from its peak.
     ATR self-scales to each coin's volatility, so a wide-range coin gets a
     wider leash than a quiet one (a flat % can't do that).

We sweep each and compare against the V8.63 baseline year-by-year. A good
trailing stop improves the worst years and/or DD without gutting bull-year CAGR.

Usage:
  python scripts/test_trailing_stops.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import scripts.backtest_full_regime_system as B  # noqa: E402

B._CAPITAL = 5_000.0

_FROM_TS = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_TO_TS = int(datetime.now(timezone.utc).timestamp() * 1000)


def _run(**kw) -> dict:
    return B._run(
        _SIG, True, True, False, True, _FROM_TS, _TO_TS,
        hard_stop_pct=0.15, use_atr_sizing=True,
        asymmetric_regime=True, and_entry=True,
        bear_drop_pct=-0.20, confirm_days=10,
        momentum_gate=True, momentum_gate_days=20,
        **kw,
    )


def _row(label: str, r: dict) -> None:
    yrs = sorted(r["year_pnl"])
    cells = "  ".join(f"{yr}:{r['year_pnl'][yr]:>+7,.0f}" for yr in yrs)
    losing = [yr for yr in yrs if r["year_pnl"][yr] < 0]
    print(f"  {label:<26} eq=${r['equity']:>8,.0f}  CAGR={r['cagr']*100:>5.1f}%  "
          f"DD={r['max_dd']*100:>4.1f}%  PF={r['pf']:.2f}  Sharpe={r['sharpe']:.2f}  "
          f"n={r['n']}  losing={losing}")
    print(f"      {cells}")


def main() -> None:
    global _SIG
    print("=" * 80)
    print("  TRAILING-STOP RESEARCH — V8.63 longs, $5K account")
    print("=" * 80)
    print("Building signals ...")
    _SIG = B._build_signals(B._SYMBOLS)
    print("Done.\n")

    base = _run()
    _row("V8.63 baseline", base)
    print()

    print("  --- A. Fixed-% trailing stop on longs ---")
    for pct in (0.10, 0.15, 0.20, 0.25, 0.30):
        _row(f"trail {int(pct*100)}% from peak", _run(trailing_stop_pct=pct))
    print()
    print("  --- B. ATR-multiple trailing stop on longs ---")
    for k in (2.0, 3.0, 4.0, 5.0):
        _row(f"trail {k:g}×ATR(14)", _run(atr_trail_mult=k))
    print()
    print("=" * 80)
    print("  Good trailing stop: improves 2023/2025 and/or DD without gutting")
    print("  the 2021/2024 bull-year CAGR. If every setting cuts CAGR with no DD")
    print("  or worst-year benefit, the lagging signal exit is already fine.")
    print("=" * 80)


if __name__ == "__main__":
    main()
