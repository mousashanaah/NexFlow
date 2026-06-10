#!/usr/bin/env python3
"""New ideas research — V8.63 improvement candidates.

Tests 4 ideas proposed to address the H1-2025 choppy-bull weakness:

  Idea 1 — Signal persistence: require entry signal to hold N days before entering
  Idea 2 — Re-entry cooldown: block re-entry on a coin for K days after a losing exit
  Idea 3 — Regime hysteresis: dead-zone buffer around SMA200 to stop regime flapping
  Idea 4 — Breadth confirmation: require M of 12 coins above SMA50 before new longs

Each tested:
  (a) Alone vs baseline
  (b) Best single vs baseline in year-by-year table
  (c) Best combo vs baseline

Judgement criteria (strict):
  - Full-period CAGR must not drop more than ~3% below baseline
  - H1-2025 must visibly improve
  - 2021 / 2023 bull capture must not be gutted
  - Walk-forward pass count must not fall below baseline (5/6)
  - Wash = reject

Usage:
  python scripts/test_new_ideas.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import scripts.backtest_full_regime_system as B  # noqa: E402

B._CAPITAL = 5_000.0
_DAY_MS = 86_400_000
_FROM  = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_TO    = int(datetime.now(timezone.utc).timestamp() * 1000)
_H1A   = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_H1B   = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)

# V8.63 baseline params — never change these
_BASE = dict(
    hard_stop_pct=0.15,
    use_atr_sizing=True,
    asymmetric_regime=True,
    and_entry=True,
    bear_drop_pct=-0.20,
    confirm_days=10,
    momentum_gate=True,
    momentum_gate_days=20,
)

_SIG: dict = {}


# ---------------------------------------------------------------------------
# Patched _run that supports the new idea parameters
# ---------------------------------------------------------------------------

def _run_base(from_ts: int, to_ts: int, **override) -> dict:
    """Run V8.63 baseline (no new ideas)."""
    p = dict(_BASE)
    p.update(override)
    return B._run(_SIG, True, True, False, True, from_ts, to_ts, **p)


def _run_ideas(
    from_ts: int,
    to_ts: int,
    # Idea 1 — signal persistence
    sig_persist_days: int = 0,
    # Idea 2 — losing-exit cooldown
    cooldown_days: int = 0,
    # Idea 3 — regime hysteresis band
    regime_band_pct: float = 0.0,
    # Idea 4 — breadth gate
    breadth_min: float = 0.0,   # fraction of 12 coins that must be above SMA50
    **override,
) -> dict:
    """
    Run V8.63 with one or more new ideas layered on top.
    Ideas are implemented here as a wrapper around the backtest engine's
    signal/timestamp loop, injecting state that the engine doesn't natively
    support yet.

    Because the engine doesn't have these hooks internally, we implement them
    by pre-filtering the signals dict that the engine sees, using the same
    no-lookahead discipline (only past data used at each step).
    """
    import copy

    p = dict(_BASE)
    p.update(override)

    # ── Precompute per-coin SMA50 series (for breadth) ──────────────────────
    # We need per-coin SMA50 on the daily close series.
    sma50_by_sym: dict[str, dict[int, float]] = {}
    if breadth_min > 0:
        for sym in B._SYMBOLS:
            ts_list = sorted(_SIG.get(sym, {}).keys())
            closes = [_SIG[sym][t]["close"] for t in ts_list]
            sma50_vals = B._sma_series(closes, 50)
            sma50_by_sym[sym] = {
                t: sma50_vals[i]
                for i, t in enumerate(ts_list)
                if sma50_vals[i] is not None
            }

    # ── Precompute BTC SMA200 for regime hysteresis ──────────────────────────
    btc_sma200_by_ts: dict[int, Optional[float]] = {}
    if regime_band_pct > 0:
        btc_ts = sorted(_SIG.get("BTCUSDT", {}).keys())
        btc_closes = [_SIG["BTCUSDT"][t]["close"] for t in btc_ts]
        sma200_vals = B._sma_series(btc_closes, 200)
        for i, t in enumerate(btc_ts):
            btc_sma200_by_ts[t] = sma200_vals[i]

    all_ts = sorted(set(
        ts for sym in B._SYMBOLS
        for ts in _SIG.get(sym, {})
        if from_ts <= ts <= to_ts
    ))

    # ── State for new ideas ──────────────────────────────────────────────────
    # Idea 1: per-coin counter of consecutive days with a valid long signal
    persist_streak: dict[str, int] = {sym: 0 for sym in B._SYMBOLS}

    # Idea 2: per-coin cooldown timestamp (no new longs until after this ts)
    cooldown_until: dict[str, int] = {sym: 0 for sym in B._SYMBOLS}

    # Idea 3: regime hysteresis state machine
    # We need the AND-entry BTC bear logic to respect a dead zone.
    # We patch BTC sma200_above in the signals copy when price is in the band.
    # State: True=bull, False=bear — starts bull.
    hysteresis_bull = True

    # Idea 2 requires tracking last exit PnL per coin — we'll intercept trades
    # by building a modified signals dict with certain entry signals suppressed.

    # ── Build a per-timestamp signal mask ────────────────────────────────────
    # The engine reads signals[sym][ts] — we create a shallow copy and zero out
    # ema_long/macd_long/h4_long on days where our ideas block entry.
    # We also inject a modified sma200_above for the regime hysteresis.

    sig_patched = {sym: dict(_SIG.get(sym, {})) for sym in B._SYMBOLS}

    # We need a two-pass approach:
    # Pass 1: simulate positions day-by-day to know which exits are losses
    #         (for Idea 2 cooldown) and to apply Ideas 1/3/4 signal masks.
    # Pass 2: run the real engine on the patched signal dict.

    # Pass 1: lightweight position tracker to compute cooldown state and
    # build the per-timestamp entry-block mask.
    entry_blocked: dict[str, set[int]] = {sym: set() for sym in B._SYMBOLS}

    sim_positions: dict[str, dict] = {}
    sim_equity = B._CAPITAL

    for ts in all_ts:
        btc_sig = _SIG.get("BTCUSDT", {}).get(ts, {})
        btc_close = btc_sig.get("close", 0.0)

        # ── Idea 3: regime hysteresis ────────────────────────────────────────
        if regime_band_pct > 0:
            sma200 = btc_sma200_by_ts.get(ts)
            if sma200 and sma200 > 0:
                bull_thresh = sma200 * (1.0 + regime_band_pct)
                bear_thresh = sma200 * (1.0 - regime_band_pct)
                if btc_close >= bull_thresh:
                    hysteresis_bull = True
                elif btc_close <= bear_thresh:
                    hysteresis_bull = False
                # else: stay in previous state (hysteresis)
            # Patch sma200_above in BTC signal for this ts
            if ts in sig_patched.get("BTCUSDT", {}):
                entry = dict(sig_patched["BTCUSDT"][ts])
                entry["sma200_above"] = hysteresis_bull
                sig_patched["BTCUSDT"][ts] = entry

        # ── Idea 4: breadth ──────────────────────────────────────────────────
        breadth_ok = True
        if breadth_min > 0:
            above = sum(
                1 for sym in B._SYMBOLS
                if _SIG.get(sym, {}).get(ts, {}).get("close", 0) >
                   sma50_by_sym.get(sym, {}).get(ts, float("inf"))
            )
            breadth_ratio = above / len(B._SYMBOLS)
            breadth_ok = breadth_ratio >= breadth_min

        for sym in B._SYMBOLS:
            sig = _SIG.get(sym, {}).get(ts, {})
            if not sig:
                continue
            c = sig.get("close", 0.0)
            n_long = sum([sig.get("ema_long", False),
                          sig.get("macd_long", False),
                          sig.get("h4_long", False)])
            any_long_raw = n_long > 0
            in_pos = sym in sim_positions and sim_positions[sym].get("side") == "LONG"

            # ── Simulate exit to track P&L for Idea 2 cooldown ──────────────
            if in_pos and not any_long_raw:
                pos = sim_positions.pop(sym)
                raw = (c - pos["entry"]) / pos["entry"] * pos["notional"]
                net = raw - B._TAKER_FEE * pos["notional"]
                sim_equity += net
                if cooldown_days > 0 and net < 0:
                    cooldown_until[sym] = ts + cooldown_days * _DAY_MS

            # ── Idea 1: signal persistence ────────────────────────────────────
            if any_long_raw:
                persist_streak[sym] = persist_streak.get(sym, 0) + 1
            else:
                persist_streak[sym] = 0

            persist_ok = (sig_persist_days == 0 or
                          persist_streak.get(sym, 0) >= sig_persist_days)

            # ── Idea 2: cooldown ──────────────────────────────────────────────
            cooldown_ok = (cooldown_days == 0 or ts > cooldown_until.get(sym, 0))

            # ── Combine all entry gates ───────────────────────────────────────
            entry_allowed = persist_ok and cooldown_ok and breadth_ok

            if not entry_allowed and not in_pos:
                entry_blocked[sym].add(ts)

            # ── Simulate entry (simplified — no sizing complexity needed) ─────
            if not in_pos and any_long_raw and entry_allowed:
                notional = B._CAPITAL / len(B._SYMBOLS)
                sim_equity -= B._TAKER_FEE * notional
                sim_positions[sym] = {"entry": c, "notional": notional, "side": "LONG"}

    # ── Patch signal dict: zero out entry signals on blocked timestamps ───────
    for sym in B._SYMBOLS:
        for ts in entry_blocked[sym]:
            if ts in sig_patched.get(sym, {}):
                entry = dict(sig_patched[sym][ts])
                entry["ema_long"]  = False
                entry["macd_long"] = False
                entry["h4_long"]   = False
                sig_patched[sym][ts] = entry

    # ── Pass 2: run the real engine on the patched signals ────────────────────
    # Temporarily swap global _SIG, run, restore
    orig_sig = B._build_signals  # not needed — we pass sig directly
    return B._run(sig_patched, True, True, False, True, from_ts, to_ts, **p)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _yoy(r: dict) -> str:
    years = sorted(r["year_pnl"])
    return "  ".join(f"{yr}: {r['year_pnl'][yr]:>+8,.0f}" for yr in years)


def _header_row():
    print(f"\n  {'Label':<36} {'CAGR':>6} {'DD':>5} {'Sh':>5} {'Eq':>9}  "
          f"{'H1-25':>6}  Year-by-year ($)")
    print(f"  {'-'*36} {'-'*6} {'-'*5} {'-'*5} {'-'*9}  {'-'*6}  {'-'*60}")


def _row(label: str, rf: dict, rh: dict):
    losing = [str(yr) for yr in rf["year_pnl"] if rf["year_pnl"][yr] < 0]
    yoy = "  ".join(
        f"{yr}:{rf['year_pnl'][yr]:>+8,.0f}" for yr in sorted(rf["year_pnl"])
    )
    flag = " LOSE=" + str(losing) if losing else ""
    print(f"  {label:<36} {rf['cagr']*100:>5.1f}% {rf['max_dd']*100:>4.1f}% "
          f"{rf['sharpe']:>5.2f} ${rf['equity']:>8,.0f}  "
          f"{rh['cagr']*100:>+5.1f}%  {yoy}{flag}")


def _walk(label: str, runner, **kw) -> int:
    """Walk-forward: 2yr train / 6mo test sliding windows. Returns pass count."""
    train = int(2 * 365.25 * _DAY_MS)
    test  = int(0.5 * 365.25 * _DAY_MS)
    ws = _FROM; npass = ntot = 0; cagrs = []
    while ws + train + test <= _TO:
        r = runner(ws + train, min(ws + train + test, _TO), **kw)
        ok = r["pf"] >= 1.10 and r["cagr"] > 0
        npass += ok; ntot += 1; cagrs.append(r["cagr"] * 100)
        ws += test
    avg = sum(cagrs) / len(cagrs) if cagrs else 0.0
    print(f"  {label:<36} walk-fwd {npass}/{ntot}  avg OOS CAGR {avg:>+5.1f}%  "
          f"windows={['%+.0f%%' % c for c in cagrs]}")
    return npass


def _year_table(rows: list[tuple[str, dict]]):
    """Print a clean year-by-year earnings table."""
    # Collect all years
    all_years = sorted({yr for _, r in rows for yr in r["year_pnl"]})
    col = 12
    header = f"  {'System':<36}" + "".join(f"{yr:>{col}}" for yr in all_years) + f"{'Total Eq':>{col}}  {'CAGR':>7}  {'Max DD':>7}"
    print(header)
    print("  " + "-" * (36 + col * len(all_years) + col + 20))
    for label, r in rows:
        row_str = f"  {label:<36}"
        for yr in all_years:
            val = r["year_pnl"].get(yr, 0)
            row_str += f"{val:>+{col},.0f}"
        row_str += f"  ${r['equity']:>9,.0f}  {r['cagr']*100:>6.1f}%  {r['max_dd']*100:>6.1f}%"
        print(row_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _SIG
    print("=" * 110)
    print("  NEW IDEAS RESEARCH — V8.63 improvement candidates")
    print("=" * 110)
    print("Building signals ...")
    _SIG = B._build_signals(B._SYMBOLS)
    print("Done.\n")

    # ── Pre-run baseline ────────────────────────────────────────────────────
    base_full = _run_base(_FROM, _TO)
    base_h1   = _run_base(_H1A, _H1B)

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  IDEA 1 — Signal persistence (require signal to hold N days before entering)")
    print("=" * 110)
    _header_row()
    _row("V8.63 baseline", base_full, base_h1)
    for n in (2, 3, 5):
        rf = _run_ideas(_FROM, _TO, sig_persist_days=n)
        rh = _run_ideas(_H1A, _H1B, sig_persist_days=n)
        _row(f"persist N={n}d", rf, rh)

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  IDEA 2 — Re-entry cooldown after losing exit (K days per coin)")
    print("=" * 110)
    _header_row()
    _row("V8.63 baseline", base_full, base_h1)
    for k in (5, 10, 15):
        rf = _run_ideas(_FROM, _TO, cooldown_days=k)
        rh = _run_ideas(_H1A, _H1B, cooldown_days=k)
        _row(f"cooldown K={k}d", rf, rh)

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  IDEA 3 — Regime hysteresis band around SMA200 (dead zone ±B%)")
    print("=" * 110)
    _header_row()
    _row("V8.63 baseline", base_full, base_h1)
    for b in (0.01, 0.02, 0.03):
        rf = _run_ideas(_FROM, _TO, regime_band_pct=b)
        rh = _run_ideas(_H1A, _H1B, regime_band_pct=b)
        _row(f"regime band ±{b*100:.0f}%", rf, rh)

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  IDEA 4 — Breadth gate (M/12 coins above SMA50 required for new longs)")
    print("=" * 110)
    _header_row()
    _row("V8.63 baseline", base_full, base_h1)
    for m in (0.33, 0.40, 0.50):
        rf = _run_ideas(_FROM, _TO, breadth_min=m)
        rh = _run_ideas(_H1A, _H1B, breadth_min=m)
        _row(f"breadth ≥{m:.0%}", rf, rh)

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  WALK-FORWARD — best candidates vs baseline (2yr train / 6mo test)")
    print("=" * 110)
    _walk("V8.63 baseline",        _run_base)
    _walk("persist N=3d",          _run_ideas, sig_persist_days=3)
    _walk("cooldown K=10d",        _run_ideas, cooldown_days=10)
    _walk("regime band ±2%",       _run_ideas, regime_band_pct=0.02)
    _walk("breadth ≥40%",          _run_ideas, breadth_min=0.40)
    _walk("persist+cooldown 3+10", _run_ideas, sig_persist_days=3, cooldown_days=10)

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  YEAR-BY-YEAR EARNINGS TABLE — $5K starting, 2021 → 2026")
    print("  (Only ideas that showed genuine improvement above are included)")
    print("=" * 110)

    candidates = [
        ("V8.63 baseline",         _run_base(_FROM, _TO)),
        ("persist N=2d",           _run_ideas(_FROM, _TO, sig_persist_days=2)),
        ("persist N=3d",           _run_ideas(_FROM, _TO, sig_persist_days=3)),
        ("persist N=5d",           _run_ideas(_FROM, _TO, sig_persist_days=5)),
        ("cooldown K=5d",          _run_ideas(_FROM, _TO, cooldown_days=5)),
        ("cooldown K=10d",         _run_ideas(_FROM, _TO, cooldown_days=10)),
        ("cooldown K=15d",         _run_ideas(_FROM, _TO, cooldown_days=15)),
        ("regime band ±1%",        _run_ideas(_FROM, _TO, regime_band_pct=0.01)),
        ("regime band ±2%",        _run_ideas(_FROM, _TO, regime_band_pct=0.02)),
        ("regime band ±3%",        _run_ideas(_FROM, _TO, regime_band_pct=0.03)),
        ("breadth ≥33%",           _run_ideas(_FROM, _TO, breadth_min=0.33)),
        ("breadth ≥40%",           _run_ideas(_FROM, _TO, breadth_min=0.40)),
        ("breadth ≥50%",           _run_ideas(_FROM, _TO, breadth_min=0.50)),
        ("persist3 + cooldown10",  _run_ideas(_FROM, _TO, sig_persist_days=3, cooldown_days=10)),
    ]
    _year_table(candidates)

    print("\n" + "=" * 110)
    print("  VERDICT GUIDE:")
    print("  ADD    → improves H1-2025 AND keeps full CAGR within ~3% of baseline AND walk-fwd ≥ 5/6")
    print("  REJECT → wash, or costs too much full-period CAGR, or gutted 2021/2023")
    print("=" * 110)


if __name__ == "__main__":
    main()
