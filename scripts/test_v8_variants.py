#!/usr/bin/env python3
"""Test V8 variants targeting 2023/2025 improvement."""

from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Load the backtest module internals via exec
_src = (_REPO_ROOT / "scripts" / "backtest_full_regime_system.py").read_text()
_ns: dict = {"__name__": "__backtest__",
             "__file__": str(_REPO_ROOT / "scripts" / "backtest_full_regime_system.py")}
exec(compile(_src, "backtest_full_regime_system.py", "exec"), _ns)

_run            = _ns["_run"]
_build_signals  = _ns["_build_signals"]
_SYMBOLS        = _ns["_SYMBOLS"]
_CAPITAL        = _ns["_CAPITAL"]

SCALE = 5_000 / _CAPITAL  # 0.05 to scale $100k results to $5k

# Base V8 kwargs (fixed for all variants)
V8_BASE = dict(
    use_sma200_long_filter=True,
    use_tsmom_short=True,
    use_per_coin_sma200=False,
    confluence=True,
    hard_stop_pct=0.15,
    use_atr_sizing=True,
    asymmetric_regime=True,
    and_entry=True,
    bear_drop_pct=-0.20,
    confirm_days=5,
    momentum_gate=True,
    momentum_gate_days=30,
)

# Base V8 PnL for comparison (scaled to $5K)
BASE_2023 = -710
BASE_2025 = -2_724
BASE_2021 = 28_341
BASE_2024 = 20_726


def run_variant(signals, from_ts, to_ts, label, **overrides):
    kwargs = {**V8_BASE, **overrides, "from_ts": from_ts, "to_ts": to_ts}
    return _run(signals, **kwargs)


def print_variant(label, r, base_2023=BASE_2023, base_2025=BASE_2025):
    yp = r["year_pnl"]
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    scaled = {yr: yp.get(yr, 0.0) * SCALE for yr in years}

    d23 = scaled.get(2023, 0) - base_2023
    d25 = scaled.get(2025, 0) - base_2025
    d21 = scaled.get(2021, 0) - BASE_2021
    d24 = scaled.get(2024, 0) - BASE_2024

    improved_2023 = scaled.get(2023, 0) > base_2023
    improved_2025 = scaled.get(2025, 0) > base_2025

    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"{'='*68}")
    print(f"  CAGR: {r['cagr']*100:.1f}%   Max DD: {r['max_dd']*100:.1f}%   Sharpe: {r['sharpe']:.2f}")
    print()
    print(f"  {'Year':<6} {'PnL ($5K)':>12}  {'vs V8':>10}")
    for yr in years:
        p = scaled.get(yr, 0.0)
        if yr == 2023:
            delta = d23
            tag = f"  {'IMPROVED ✓' if improved_2023 else 'worse ✗'}"
        elif yr == 2025:
            delta = d25
            tag = f"  {'IMPROVED ✓' if improved_2025 else 'worse ✗'}"
        elif yr == 2021:
            delta = d21
            tag = f"  {'hurt ✗' if d21 < -1000 else 'ok'}"
        elif yr == 2024:
            delta = d24
            tag = f"  {'hurt ✗' if d24 < -2000 else 'ok'}"
        else:
            delta = p - {2022: 2318, 2026: 2994}.get(yr, 0)
            tag = ""
        print(f"  {yr:<6} ${p:>+10,.0f}  ({delta:>+8,.0f}){tag}")
    both_improved = improved_2023 and improved_2025
    print(f"\n  2023 improved: {'YES ✓' if improved_2023 else 'NO ✗'}  |  2025 improved: {'YES ✓' if improved_2025 else 'NO ✗'}  |  BOTH: {'YES ✓' if both_improved else 'NO ✗'}")


def main():
    from_ts = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    to_ts   = int(datetime.now(timezone.utc).timestamp() * 1000)

    print("Building signals for all 12 coins...")
    signals = _build_signals(_SYMBOLS)
    print("Done.\n")

    print("=" * 68)
    print("  V8 BASE (reference — scaled to $5K capital)")
    print("=" * 68)
    print(f"  2021: ${BASE_2021:>+,.0f}  2022: ${2318:>+,.0f}  2023: ${BASE_2023:>+,.0f}")
    print(f"  2024: ${BASE_2024:>+,.0f}  2025: ${BASE_2025:>+,.0f}  2026: ${2994:>+,.0f}")
    print(f"  CAGR: 56%   Max DD: 28%   Sharpe: 1.18")

    variants = [
        ("V8.1  bear_drop_pct=-0.15 (stricter bear entry)",
         dict(bear_drop_pct=-0.15)),
        ("V8.2  momentum_gate_days=45 (longer momentum gate)",
         dict(momentum_gate_days=45)),
        ("V8.3  momentum_gate_days=20 (shorter momentum gate)",
         dict(momentum_gate_days=20)),
        ("V8.4  trailing_stop_pct=0.12 (trailing stop on longs)",
         dict(trailing_stop_pct=0.12)),
        ("V8.5  use_coin_sma50=True (per-coin SMA50 filter)",
         dict(use_coin_sma50=True)),
        ("V8.6  confirm_days=10 (longer confirm days)",
         dict(confirm_days=10)),
        ("V8.7  target_risk=0.005 (half position size)",
         dict(target_risk=0.005)),
    ]

    results = []
    for label, overrides in variants:
        print(f"\nRunning {label[:30]}...")
        r = run_variant(signals, from_ts, to_ts, label, **overrides)
        print_variant(label, r)
        results.append((label, r, overrides))

    # Summary table
    print(f"\n\n{'='*90}")
    print("  SUMMARY — All Variants vs V8 Base")
    print(f"{'='*90}")
    print(f"  {'Variant':<48}  {'CAGR':>6}  {'MaxDD':>6}  {'Shrp':>5}  {'2023':>8}  {'2025':>8}  Both?")
    print(f"  {'-'*48}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*8}  {'-----'}")

    # Base row
    print(f"  {'V8 BASE':<48}  {'56.0%':>6}  {'28.0%':>6}  {'1.18':>5}  {BASE_2023:>+8,.0f}  {BASE_2025:>+8,.0f}  ---")

    for label, r, _ in results:
        yp = r["year_pnl"]
        p23 = yp.get(2023, 0) * SCALE
        p25 = yp.get(2025, 0) * SCALE
        both = "YES ✓" if p23 > BASE_2023 and p25 > BASE_2025 else "no"
        short = label[:48]
        print(f"  {short:<48}  {r['cagr']*100:>5.1f}%  {r['max_dd']*100:>5.1f}%  {r['sharpe']:>5.2f}  {p23:>+8,.0f}  {p25:>+8,.0f}  {both}")

    # Top candidates
    print(f"\n\n{'='*68}")
    print("  TOP CANDIDATES (improved both 2023 and 2025 without wrecking 2021/2024)")
    print(f"{'='*68}")
    candidates = []
    for label, r, overrides in results:
        yp = r["year_pnl"]
        p23 = yp.get(2023, 0) * SCALE
        p25 = yp.get(2025, 0) * SCALE
        p21 = yp.get(2021, 0) * SCALE
        p24 = yp.get(2024, 0) * SCALE
        if p23 > BASE_2023 and p25 > BASE_2025:
            # score: improvement in bad years minus penalty for hurting good years
            improvement = (p23 - BASE_2023) + (p25 - BASE_2025)
            penalty = max(0, BASE_2021 - p21) + max(0, BASE_2024 - p24)
            score = improvement - penalty * 0.5
            candidates.append((score, label, r, p23, p25, p21, p24, overrides))

    candidates.sort(reverse=True)
    if not candidates:
        print("  No variant improved BOTH 2023 and 2025 vs V8 base.")
        print("  Showing best single-year improvements instead:")
        # show best 2023 improvers
        by_23 = sorted(results, key=lambda x: x[1]["year_pnl"].get(2023, 0), reverse=True)
        by_25 = sorted(results, key=lambda x: x[1]["year_pnl"].get(2025, 0), reverse=True)
        print(f"\n  Best 2023 improvers:")
        for label, r, _ in by_23[:3]:
            p23 = r["year_pnl"].get(2023, 0) * SCALE
            p25 = r["year_pnl"].get(2025, 0) * SCALE
            print(f"    {label[:50]}: 2023={p23:>+,.0f}  2025={p25:>+,.0f}")
        print(f"\n  Best 2025 improvers:")
        for label, r, _ in by_25[:3]:
            p23 = r["year_pnl"].get(2023, 0) * SCALE
            p25 = r["year_pnl"].get(2025, 0) * SCALE
            print(f"    {label[:50]}: 2025={p25:>+,.0f}  2023={p23:>+,.0f}")
    else:
        for rank, (score, label, r, p23, p25, p21, p24, overrides) in enumerate(candidates[:3], 1):
            print(f"\n  #{rank}: {label}")
            print(f"       Score: {score:,.0f}  |  CAGR: {r['cagr']*100:.1f}%  MaxDD: {r['max_dd']*100:.1f}%  Sharpe: {r['sharpe']:.2f}")
            print(f"       2021: ${p21:>+,.0f} (base ${BASE_2021:>+,.0f})")
            print(f"       2023: ${p23:>+,.0f} (base ${BASE_2023:>+,.0f})  delta={p23-BASE_2023:>+,.0f}")
            print(f"       2024: ${p24:>+,.0f} (base ${BASE_2024:>+,.0f})")
            print(f"       2025: ${p25:>+,.0f} (base ${BASE_2025:>+,.0f})  delta={p25-BASE_2025:>+,.0f}")
            print(f"       Settings: {overrides}")


if __name__ == "__main__":
    main()
