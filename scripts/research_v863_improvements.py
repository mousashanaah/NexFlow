#!/usr/bin/env python3
"""
Research: V8.63 improvements — trailing stop and weekly trend (SMA50) filter.

Tests:
  1. V8.63 base
  2. V8.63 + trailing_stop_pct=0.10
  3. V8.63 + trailing_stop_pct=0.08
  4. V8.63 + use_coin_sma50=True
  5. V8.63 + trailing_stop_pct=0.10 + use_coin_sma50=True

Year-by-year PnL scaled to $5K, CAGR, DD, Sharpe.
"""

from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_src = (_REPO_ROOT / "scripts" / "backtest_full_regime_system.py").read_text()
_ns: dict = {"__name__": "__backtest__",
             "__file__": str(_REPO_ROOT / "scripts" / "backtest_full_regime_system.py")}
exec(compile(_src, "backtest_full_regime_system.py", "exec"), _ns)

_run     = _ns["_run"]
_build_signals = _ns["_build_signals"]
_SYMBOLS = _ns["_SYMBOLS"]
_CAPITAL = _ns["_CAPITAL"]

SCALE = 5_000 / _CAPITAL  # scale $100k → $5k

# V8.63 base parameters (per spec)
V863_BASE = dict(
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

YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


def run_variant(signals, from_ts, to_ts, **overrides):
    kwargs = {**V863_BASE, **overrides, "from_ts": from_ts, "to_ts": to_ts}
    return _run(signals, **kwargs)


def print_variant(label: str, r: dict, base_r: dict | None = None) -> None:
    yp = r["year_pnl"]
    scaled = {yr: yp.get(yr, 0.0) * SCALE for yr in YEARS}

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  CAGR: {r['cagr']*100:.1f}%   Max DD: {r['max_dd']*100:.1f}%   Sharpe: {r['sharpe']:.2f}   Sortino: {r['sortino']:.2f}")
    print(f"  PF: {r['pf']:.2f}  IS PF: {r['is_pf']:.2f}  OOS PF: {r['oos_pf']:.2f}  Trades: {r['n']}")
    print()

    if base_r is not None:
        base_scaled = {yr: base_r["year_pnl"].get(yr, 0.0) * SCALE for yr in YEARS}
        print(f"  {'Year':<6}  {'PnL ($5K)':>12}  {'vs Base':>10}  {'Note'}")
        print(f"  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*20}")
        for yr in YEARS:
            p = scaled.get(yr, 0.0)
            b = base_scaled.get(yr, 0.0)
            delta = p - b
            note = ""
            if yr in (2022, 2023, 2025):
                note = "bear/sideways"
            elif yr in (2021, 2024):
                note = "bull"
            delta_str = f"{delta:>+,.0f}" if b != 0.0 or p != 0.0 else "  —"
            arrow = " ↑" if delta > 200 else (" ↓" if delta < -200 else "  ~")
            print(f"  {yr:<6}  ${p:>+10,.0f}  {delta_str:>10}{arrow}  {note}")
    else:
        print(f"  {'Year':<6}  {'PnL ($5K)':>12}  {'Note'}")
        print(f"  {'─'*6}  {'─'*12}  {'─'*20}")
        for yr in YEARS:
            p = scaled.get(yr, 0.0)
            note = ""
            if yr in (2022, 2023, 2025):
                note = "bear/sideways"
            elif yr in (2021, 2024):
                note = "bull"
            print(f"  {yr:<6}  ${p:>+10,.0f}  {note}")


def main():
    from_ts = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    to_ts   = int(datetime.now(timezone.utc).timestamp() * 1000)

    print("Building signals for all 12 coins...")
    signals = _build_signals(_SYMBOLS)
    print("Done.\n")

    print("=" * 72)
    print("  V8.63 IMPROVEMENT RESEARCH — Trailing Stop & Weekly Trend Filter")
    print("  Capital scaled to $5,000 | 12 coins | 2021-2026")
    print("=" * 72)

    configs = [
        ("1. V8.63 BASE",
         {}),
        ("2. V8.63 + trailing_stop=0.10",
         dict(trailing_stop_pct=0.10)),
        ("3. V8.63 + trailing_stop=0.08",
         dict(trailing_stop_pct=0.08)),
        ("4. V8.63 + use_coin_sma50=True (weekly trend proxy)",
         dict(use_coin_sma50=True)),
        ("5. V8.63 + trailing_stop=0.10 + use_coin_sma50=True",
         dict(trailing_stop_pct=0.10, use_coin_sma50=True)),
    ]

    results = []
    for label, overrides in configs:
        print(f"\nRunning {label[:40]}...", flush=True)
        r = run_variant(signals, from_ts, to_ts, **overrides)
        results.append((label, r, overrides))

    base_r = results[0][1]

    for i, (label, r, overrides) in enumerate(results):
        if i == 0:
            print_variant(label, r, base_r=None)
        else:
            print_variant(label, r, base_r=base_r)

    # ── Summary table ──
    print(f"\n\n{'='*90}")
    print("  SUMMARY TABLE — All Variants vs V8.63 Base (scaled to $5K)")
    print(f"{'='*90}")
    print(f"  {'Variant':<52}  {'CAGR':>6}  {'MaxDD':>6}  {'Shrp':>5}  "
          f"{'2023':>8}  {'2025':>8}  {'Verdict'}")
    print(f"  {'─'*52}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*12}")

    base_yp = base_r["year_pnl"]
    base_23 = base_yp.get(2023, 0.0) * SCALE
    base_25 = base_yp.get(2025, 0.0) * SCALE

    for label, r, overrides in results:
        yp = r["year_pnl"]
        p23 = yp.get(2023, 0.0) * SCALE
        p25 = yp.get(2025, 0.0) * SCALE
        p21 = yp.get(2021, 0.0) * SCALE
        p24 = yp.get(2024, 0.0) * SCALE
        base_21 = base_yp.get(2021, 0.0) * SCALE
        base_24 = base_yp.get(2024, 0.0) * SCALE

        better_23 = p23 > base_23
        better_25 = p25 > base_25
        bull_ok   = (p21 >= base_21 - 1000) and (p24 >= base_24 - 2000)

        if overrides == {}:
            verdict = "BASE"
        elif better_23 and better_25 and bull_ok:
            verdict = "WINNER ✓"
        elif better_23 and better_25:
            verdict = "IMPROVED (bull hurt)"
        elif better_25:
            verdict = "2025 only ↑"
        elif better_23:
            verdict = "2023 only ↑"
        else:
            verdict = "worse"

        short = label[:52]
        print(f"  {short:<52}  {r['cagr']*100:>5.1f}%  {r['max_dd']*100:>5.1f}%  "
              f"{r['sharpe']:>5.2f}  {p23:>+8,.0f}  {p25:>+8,.0f}  {verdict}")

    # ── Analysis ──
    print(f"\n\n{'='*72}")
    print("  ANALYSIS")
    print(f"{'='*72}")

    winners = [(label, r, overrides) for label, r, overrides in results[1:]
               if (r["year_pnl"].get(2023, 0.0) * SCALE > base_23) and
                  (r["year_pnl"].get(2025, 0.0) * SCALE > base_25)]

    if not winners:
        print("  No variant improved BOTH 2023 and 2025 vs V8.63 base.")
        print()
        # Best bear-year improvement
        by_25 = sorted(results[1:], key=lambda x: x[1]["year_pnl"].get(2025, 0.0), reverse=True)
        by_23 = sorted(results[1:], key=lambda x: x[1]["year_pnl"].get(2023, 0.0), reverse=True)
        print("  Best 2025 improvement:")
        for label, r, _ in by_25[:2]:
            p25 = r["year_pnl"].get(2025, 0.0) * SCALE
            d25 = p25 - base_25
            print(f"    {label[:55]}: 2025=${p25:>+,.0f} (delta {d25:>+,.0f})")
        print("  Best 2023 improvement:")
        for label, r, _ in by_23[:2]:
            p23 = r["year_pnl"].get(2023, 0.0) * SCALE
            d23 = p23 - base_23
            print(f"    {label[:55]}: 2023=${p23:>+,.0f} (delta {d23:>+,.0f})")
    else:
        print(f"  {len(winners)} variant(s) improved BOTH 2023 AND 2025:")
        for label, r, overrides in winners:
            p23 = r["year_pnl"].get(2023, 0.0) * SCALE
            p25 = r["year_pnl"].get(2025, 0.0) * SCALE
            p21 = r["year_pnl"].get(2021, 0.0) * SCALE
            p24 = r["year_pnl"].get(2024, 0.0) * SCALE
            print(f"\n  >> {label}")
            print(f"     CAGR: {r['cagr']*100:.1f}%  MaxDD: {r['max_dd']*100:.1f}%  Sharpe: {r['sharpe']:.2f}")
            print(f"     2021: ${p21:>+,.0f} (base {base_yp.get(2021,0)*SCALE:>+,.0f})")
            print(f"     2023: ${p23:>+,.0f} (base {base_23:>+,.0f})  delta={p23-base_23:>+,.0f}")
            print(f"     2024: ${p24:>+,.0f} (base {base_yp.get(2024,0)*SCALE:>+,.0f})")
            print(f"     2025: ${p25:>+,.0f} (base {base_25:>+,.0f})  delta={p25-base_25:>+,.0f}")

    print()
    print("  Risk-adjusted comparison (Sharpe):")
    for label, r, _ in results:
        flag = " <<< BASE" if _ == {} else ""
        print(f"    {label[:52]}: Sharpe {r['sharpe']:.2f}  Sortino {r['sortino']:.2f}  MaxDD {r['max_dd']*100:.1f}%{flag}")

    print()


if __name__ == "__main__":
    main()
