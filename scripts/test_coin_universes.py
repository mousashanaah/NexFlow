#!/usr/bin/env python3
"""Test different coin universe compositions vs V8.63 baseline.

Universes tested:
  Base12  : All 12 coins (V8.63 baseline)
  Top8    : Drop DOT/DOGE/ADA/LTC → BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT LINKUSDT AVAXUSDT TRXUSDT
  Top6    : Only majors → BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT LINKUSDT

All tests use V8.63 params.
Capital scaled to $5K in output (internally runs at $100K, then scales).
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

import scripts.backtest_full_regime_system as bfr

_CANDLE_DIR  = _REPO_ROOT / "data" / "candles"
_TAKER_FEE   = 0.0006
_CAPITAL_SIM = 100_000.0   # internal sim capital
_CAPITAL_OUT = 5_000.0     # display scale
_SCALE       = _CAPITAL_OUT / _CAPITAL_SIM
_DAY_MS      = 86_400_000

# V8.63 params
_V863_PARAMS = dict(
    use_sma200_long_filter=True,
    use_tsmom_short=True,
    use_per_coin_sma200=False,
    confluence=True,
    hard_stop_pct=0.15,
    use_atr_sizing=True,
    asymmetric_regime=True,
    and_entry=True,
    bear_drop_pct=-0.20,
    confirm_days=10,
    momentum_gate=True,
    momentum_gate_days=20,
)

UNIVERSES = {
    "Base12 (V8.63)": [
        "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
        "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT",
        "LINKUSDT","LTCUSDT","DOTUSDT","TRXUSDT",
    ],
    "Top8 (drop DOT/DOGE/ADA/LTC)": [
        "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
        "XRPUSDT","LINKUSDT","AVAXUSDT","TRXUSDT",
    ],
    "Top6 (majors only)": [
        "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
        "XRPUSDT","LINKUSDT",
    ],
}

# Check for any extra coins beyond the 12 base
_AVAILABLE = {p.stem.replace("_1D","") for p in _CANDLE_DIR.glob("*_1D.parquet")}
_BASE12 = set(UNIVERSES["Base12 (V8.63)"])
_EXTRAS = _AVAILABLE - _BASE12
if _EXTRAS:
    print(f"[INFO] Extra coins in data/candles/ beyond base-12: {sorted(_EXTRAS)}")
    # Could add an extended universe here if found
else:
    print("[INFO] No extra coins found beyond the base-12 in data/candles/")


def _run_universe(name: str, symbols: list[str]) -> dict:
    """Run V8.63 for a given symbol list. Patches the module global."""
    # Patch module-level _SYMBOLS so _run() uses them
    original = bfr._SYMBOLS
    bfr._SYMBOLS = symbols

    from_ts = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    to_ts   = int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f"  Building signals for {name} ({len(symbols)} coins)...")
    signals = bfr._build_signals(symbols)

    result = bfr._run(
        signals,
        from_ts=from_ts,
        to_ts=to_ts,
        **_V863_PARAMS,
    )
    bfr._SYMBOLS = original
    return result


def _print_universe(name: str, symbols: list[str], r: dict) -> None:
    scale = _SCALE
    equity_scaled = r["equity"] * scale
    net_scaled    = r["net"] * scale
    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"  Coins: {', '.join(symbols)}")
    print(f"{'='*62}")
    print(f"  Equity (@$5K) : ${equity_scaled:>10,.0f}  (net ${net_scaled:>+,.0f})")
    print(f"  CAGR          : {r['cagr']*100:.1f}%")
    print(f"  Max DD        : {r['max_dd']*100:.1f}%")
    print(f"  Sharpe        : {r['sharpe']:.2f}   Sortino: {r['sortino']:.2f}")
    print(f"  Profit Factor : {r['pf']:.2f}  (IS:{r['is_pf']:.2f}  OOS:{r['oos_pf']:.2f})")
    print(f"  Trades        : {r['n']}")
    print()
    losing_years = []
    for yr in sorted(r["year_pnl"]):
        p = r["year_pnl"][yr] * scale
        tag = " <<BEAR" if yr in [2022, 2025, 2026] else (" <<BULL" if yr in [2021, 2024] else "")
        flag = " OK" if p >= 0 else " LOSS"
        print(f"    {yr}: ${p:>+9,.0f}{tag}{flag}")
        if p < 0:
            losing_years.append(yr)
    print(f"\n  Losing years: {losing_years if losing_years else 'NONE'}")
    verdict = "GO" if r["pf"] >= 1.20 and r["max_dd"] <= 0.45 and r["cagr"] >= 0.15 else (
              "MARGINAL" if r["pf"] >= 1.10 else "KILL")
    print(f"  VERDICT: {verdict}")


def main():
    print("=" * 62)
    print("  Coin Universe Comparison vs V8.63 Baseline")
    print(f"  Display capital: ${_CAPITAL_OUT:,.0f}  |  Sim capital: ${_CAPITAL_SIM:,.0f}")
    print("=" * 62)

    results = {}
    for name, symbols in UNIVERSES.items():
        r = _run_universe(name, symbols)
        results[name] = r
        _print_universe(name, symbols, r)

    # Summary comparison table
    print(f"\n\n{'='*62}")
    print("  SUMMARY — Universe Comparison")
    print(f"{'='*62}")
    print(f"  {'Universe':<35}  {'CAGR':>7}  {'MaxDD':>6}  {'Sharpe':>7}  {'PF':>5}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*5}")
    for name, r in results.items():
        print(f"  {name:<35}  {r['cagr']*100:>6.1f}%  {r['max_dd']*100:>5.1f}%  "
              f"{r['sharpe']:>7.2f}  {r['pf']:>5.2f}")

    # Determine winner
    best = max(results.items(), key=lambda x: x[1]["cagr"])
    best_sharpe = max(results.items(), key=lambda x: x[1]["sharpe"])
    print(f"\n  Best CAGR   : {best[0]}  ({best[1]['cagr']*100:.1f}%)")
    print(f"  Best Sharpe : {best_sharpe[0]}  ({best_sharpe[1]['sharpe']:.2f})")


if __name__ == "__main__":
    main()
