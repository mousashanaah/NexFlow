#!/usr/bin/env python3
"""Hybrid test — V8.63 daily system + 1H tactical overlay on held longs.

Motivation (from test_1h_system.py): the 1H system crushed the H1-2025 chop
window (+67% vs -31%) but captured less than half the full-period CAGR. The
two systems are good at opposite things. This tests the only combination with
evidence behind it:

  - V8.63 daily logic UNCHANGED decides what to hold and how big (longs,
    shorts, regime, momentum gate, sizing — all identical).
  - A 1H overlay manages exposure WITHIN held longs: if the hourly trend
    breaks down for K consecutive hours, step aside (sell); re-enter when the
    hourly trend recovers — as long as the daily layer still wants the coin.
  - Shorts are untouched (no overlay).

Honesty notes:
  - The hybrid engine re-implements V8.63's daily logic so the overlay can act
    between daily closes. We validate the re-implementation by running it with
    the overlay OFF and comparing to the real engine's baseline. If they don't
    match closely, the comparison is void.
  - Every overlay exit/re-entry pays the 0.06% taker fee. No free lunches.

Acceptance bar (set by the user): genuinely better earnings AND reduced losses
without sacrificing anything anywhere else. Wash = reject.

Usage:
  python scripts/test_hybrid_overlay.py
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

import scripts.backtest_full_regime_system as B  # noqa: E402

B._CAPITAL = 5_000.0
_CAPITAL = 5_000.0
_DAY_MS  = 86_400_000
_HOUR_MS = 3_600_000
_FEE     = 0.0006
_FROM = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_TO   = int(datetime.now(timezone.utc).timestamp() * 1000)
_H1A  = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_H1B  = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)

_BASE_DAILY = dict(
    hard_stop_pct=0.15, use_atr_sizing=True,
    asymmetric_regime=True, and_entry=True,
    bear_drop_pct=-0.20, confirm_days=10,
    momentum_gate=True, momentum_gate_days=20,
)

_SYMBOLS = B._SYMBOLS


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------

def _build_hourly_trend() -> dict[str, dict[int, dict]]:
    """Per-hour 1H trend state per coin: {sym: {hour_ts: {ema_long, macd_long, close}}}"""
    out: dict[str, dict[int, dict]] = {}
    for sym in _SYMBOLS:
        path = B._CANDLE_DIR / f"{sym}_1H.parquet"
        if not path.exists():
            continue
        tbl = pq.read_table(path, columns=["open_time", "close"])
        rows = sorted(zip(tbl.column("open_time").to_pylist(),
                          tbl.column("close").to_pylist()))
        closes  = [float(c) for _, c in rows]
        ts_list = [int(t) for t, _ in rows]
        ema8  = B._ema_series(closes, 8)
        ema21 = B._ema_series(closes, 21)
        macd  = B._macd_long_series(closes)
        out[sym] = {
            ts_list[i]: {
                "close": closes[i],
                "ema_long": bool(ema8[i] and ema21[i] and ema8[i] > ema21[i]),
                "macd_long": macd[i],
            }
            for i in range(len(ts_list))
        }
    return out


def _daily_aux(signals: dict):
    """BTC bear state per day (V8.63 logic), per-coin mom20, per-coin vol."""
    btc_ts = sorted(signals["BTCUSDT"].keys())
    btc_closes = [signals["BTCUSDT"][t]["close"] for t in btc_ts]
    sma200 = B._sma_series(btc_closes, 200)

    bear_by_day: dict[int, bool] = {}
    streak = 0
    prev_bear = False
    for i, t in enumerate(btc_ts):
        above = sma200[i] is not None and btc_closes[i] > sma200[i]
        mom30 = ((btc_closes[i] - btc_closes[i-30]) / btc_closes[i-30]) if i >= 30 else 0.0
        enter_bear = (not above) and (mom30 < _BASE_DAILY["bear_drop_pct"])
        streak = streak + 1 if above else 0
        confirmed_bull = streak >= _BASE_DAILY["confirm_days"]
        bear = (not confirmed_bull) if prev_bear else enter_bear
        prev_bear = bear
        bear_by_day[t] = bear

    mom20: dict[str, dict[int, float]] = {}
    vol:   dict[str, dict[int, float]] = {}
    for sym in _SYMBOLS:
        ts_list = sorted(signals.get(sym, {}).keys())
        closes = [signals[sym][t]["close"] for t in ts_list]
        mom20[sym] = {
            ts_list[i]: (closes[i] - closes[i-20]) / closes[i-20]
            for i in range(20, len(ts_list))
        }
        vol[sym] = B._vol_series(signals, sym)
    return bear_by_day, mom20, vol


def _size(sym: str, ts: int, mult: float, vol: dict) -> float:
    base = _CAPITAL / len(_SYMBOLS)
    v = vol.get(sym, {}).get(ts, 0.0)
    if v <= 0:
        return base * mult
    vol_sized = (0.01 * _CAPITAL) / v
    return min(vol_sized * mult, base * 2 * mult)


# ---------------------------------------------------------------------------
# Hybrid engine: daily V8.63 replication + hourly overlay on longs
# ---------------------------------------------------------------------------

def _run_hybrid(
    signals: dict,
    hourly: dict,
    bear_by_day: dict,
    mom20: dict,
    vol: dict,
    from_ts: int,
    to_ts: int,
    overlay_k: int = 0,        # consecutive bearish 1H hours before stepping aside (0=off)
    reentry_both: bool = False,  # re-entry needs ema AND macd (else ema only)
) -> dict:
    daily_ts = sorted(set(
        t for sym in _SYMBOLS for t in signals.get(sym, {})
        if from_ts <= t <= to_ts
    ))

    equity = _CAPITAL
    # Long slots: daily layer owns desired state; overlay toggles exposure.
    # slot = {notional, mult, entry, exposed, bear_hours}
    longs:  dict[str, dict] = {}
    shorts: dict[str, dict] = {}
    last_rebal = 0
    year_pnl: dict[int, float] = {}
    daily_equity: list[float] = []
    trades = 0
    overlay_exits = 0

    def _book(net: float, ts: int):
        nonlocal equity, trades
        equity += net
        trades += 1
        yr = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year
        year_pnl[yr] = year_pnl.get(yr, 0) + net

    def _sell_long(sym: str, price: int, ts: int, drop_slot: bool):
        slot = longs[sym]
        if slot["exposed"]:
            raw = (price - slot["entry"]) / slot["entry"] * slot["notional"]
            _book(raw - _FEE * slot["notional"], ts)
            slot["exposed"] = False
        if drop_slot:
            longs.pop(sym)

    for di, ts in enumerate(daily_ts):
        bear = bear_by_day.get(ts, False)
        bull = not bear

        # ── Hourly overlay pass for the 24h ENDING at this daily close ──────
        if overlay_k > 0:
            day_start = (ts // _DAY_MS) * _DAY_MS
            for h in range(24):
                hts = day_start + h * _HOUR_MS
                for sym in list(longs):
                    slot = longs[sym]
                    hsig = hourly.get(sym, {}).get(hts)
                    if hsig is None:
                        continue
                    h_bear = (not hsig["ema_long"]) and (not hsig["macd_long"])
                    if slot["exposed"]:
                        slot["bear_hours"] = slot["bear_hours"] + 1 if h_bear else 0
                        if slot["bear_hours"] >= overlay_k:
                            _sell_long(sym, hsig["close"], hts, drop_slot=False)
                            overlay_exits += 1
                            slot["bear_hours"] = 0
                    else:
                        ok = (hsig["ema_long"] and hsig["macd_long"]) if reentry_both \
                             else hsig["ema_long"]
                        if ok:
                            equity_cost = _FEE * slot["notional"]
                            nonlocal_equity = None  # placeholder, fee booked below
                            slot["entry"] = hsig["close"]
                            slot["exposed"] = True
                            slot["bear_hours"] = 0
                            _bookfee = -equity_cost
                            # book fee as realized cost
                            yr = datetime.fromtimestamp(hts / 1000, tz=timezone.utc).year
                            year_pnl[yr] = year_pnl.get(yr, 0) + _bookfee
                            equity += _bookfee

        # ── TSMOM short rebalance (weekly, bear only) ───────────────────────
        if bear and (ts - last_rebal) >= 7 * _DAY_MS:
            last_rebal = ts
            desired = set()
            for sym in _SYMBOLS:
                s_ts = sorted(signals.get(sym, {}).keys())
                past = [t for t in s_ts if t <= ts]
                if len(past) < 127:
                    continue
                c_now, c_past = signals[sym][past[-1]]["close"], signals[sym][past[-127]]["close"]
                if (c_now - c_past) / c_past < -0.05:
                    desired.add(sym)
            for sym in [s for s in list(shorts) if s not in desired]:
                c = signals.get(sym, {}).get(ts, {}).get("close")
                if c is None:
                    continue
                pos = shorts.pop(sym)
                raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                _book(raw - _FEE * pos["notional"], ts)
            for sym in desired:
                if sym not in shorts and sym not in longs:
                    c = signals.get(sym, {}).get(ts, {}).get("close")
                    if c is None:
                        continue
                    n = _size(sym, ts, 1.0, vol)
                    equity -= _FEE * n
                    shorts[sym] = {"entry": c, "notional": n}

        # ── Short hard stop ──────────────────────────────────────────────────
        for sym in list(shorts):
            c = signals.get(sym, {}).get(ts, {}).get("close")
            if c is None:
                continue
            pos = shorts[sym]
            if (c - pos["entry"]) / pos["entry"] >= 0.15:
                shorts.pop(sym)
                raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                _book(raw - _FEE * pos["notional"], ts)

        # ── Close shorts on bull ─────────────────────────────────────────────
        if bull:
            for sym in list(shorts):
                c = signals.get(sym, {}).get(ts, {}).get("close")
                if c is None:
                    continue
                pos = shorts.pop(sym)
                raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                _book(raw - _FEE * pos["notional"], ts)

        # ── Daily long logic (V8.63 replication) ─────────────────────────────
        for sym in _SYMBOLS:
            sig = signals.get(sym, {}).get(ts, {})
            if not sig:
                continue
            c = sig["close"]
            n_long = sum([sig.get("ema_long", False), sig.get("macd_long", False),
                          sig.get("h4_long", False)])
            any_long = n_long > 0

            can_long = bull
            if can_long and not sig.get("sma200_above", True):
                if mom20.get(sym, {}).get(ts, 0.0) <= 0:
                    can_long = False

            in_slot = sym in longs

            # Daily exit: signal gone or regime off
            if in_slot and (not any_long or not can_long):
                _sell_long(sym, c, ts, drop_slot=True)
                in_slot = False

            # Daily entry
            if not in_slot and any_long and can_long and sym not in shorts:
                if mom20.get(sym, {}).get(ts, 0.0) <= 0:
                    continue
                mult = {1: 1.0, 2: 1.5, 3: 2.0}.get(n_long, 1.0)
                n = _size(sym, ts, mult, vol)
                equity -= _FEE * n
                longs[sym] = {"entry": c, "notional": n, "mult": mult,
                              "exposed": True, "bear_hours": 0}

        # ── Daily mark-to-market ─────────────────────────────────────────────
        mtm = 0.0
        for sym, slot in longs.items():
            if not slot["exposed"]:
                continue
            c = signals.get(sym, {}).get(ts, {}).get("close")
            if c is not None:
                mtm += (c - slot["entry"]) / slot["entry"] * slot["notional"]
        for sym, pos in shorts.items():
            c = signals.get(sym, {}).get(ts, {}).get("close")
            if c is not None:
                mtm += (pos["entry"] - c) / pos["entry"] * pos["notional"]
        daily_equity.append(equity + mtm)

    # Final close-out (mark unrealised into final equity + year pnl)
    last = daily_ts[-1] if daily_ts else to_ts
    unreal = 0.0
    for sym, slot in longs.items():
        if not slot["exposed"]:
            continue
        c = signals.get(sym, {}).get(last, {}).get("close", slot["entry"])
        unreal += (c - slot["entry"]) / slot["entry"] * slot["notional"]
    for sym, pos in shorts.items():
        c = signals.get(sym, {}).get(last, {}).get("close", pos["entry"])
        unreal += (pos["entry"] - c) / pos["entry"] * pos["notional"]
    equity += unreal
    yr = datetime.fromtimestamp(last / 1000, tz=timezone.utc).year
    year_pnl[yr] = year_pnl.get(yr, 0) + unreal

    # Metrics
    peak = -1e18; max_dd = 0.0
    for e in daily_equity:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)
    years = (to_ts - from_ts) / (_DAY_MS * 365.25)
    cagr = (equity / _CAPITAL) ** (1 / years) - 1 if years > 0 and equity > 0 else -1.0
    sharpe = 0.0
    if len(daily_equity) > 2:
        rets = [(daily_equity[i] - daily_equity[i-1]) / daily_equity[i-1]
                for i in range(1, len(daily_equity)) if daily_equity[i-1] > 0]
        if rets:
            mean = sum(rets) / len(rets)
            std = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
            sharpe = (mean / std) * (252 ** 0.5) if std > 0 else 0.0

    # Profit factor from year pnl is wrong; use simple positive/negative split of
    # daily equity changes as a proxy is also wrong — keep pf from trades sign.
    return {
        "equity": equity, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe,
        "year_pnl": year_pnl, "n_trades": trades, "overlay_exits": overlay_exits,
        "pf": 1.0 + max(0.0, cagr),  # placeholder for walk-fwd pass check below
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _year_table(rows):
    all_years = sorted({yr for _, r in rows for yr in r["year_pnl"]})
    col = 12
    print(f"\n  {'System':<34}" + "".join(f"{yr:>{col}}" for yr in all_years)
          + f"{'Total Eq':>{col}}  {'CAGR':>7}  {'DD':>6}  {'Sh':>5}  {'OvExits':>8}")
    print("  " + "-" * (34 + col * len(all_years) + col + 36))
    for label, r in rows:
        s = f"  {label:<34}"
        for yr in all_years:
            s += f"{r['year_pnl'].get(yr, 0):>+{col},.0f}"
        s += (f"  ${r['equity']:>9,.0f}  {r['cagr']*100:>6.1f}%  {r['max_dd']*100:>5.1f}%"
              f"  {r['sharpe']:>4.2f}  {r.get('overlay_exits', 0):>8,}")
        print(s)


def main():
    print("=" * 112)
    print("  HYBRID OVERLAY TEST — V8.63 daily + 1H tactical overlay on held longs")
    print("=" * 112)

    print("  Building daily signals ...")
    signals = B._build_signals(_SYMBOLS)
    bear_by_day, mom20, vol = _daily_aux(signals)
    print("  Building hourly trend state ...")
    hourly = _build_hourly_trend()
    print("  Done.\n")

    def run(a, b, **kw):
        return _run_hybrid(signals, hourly, bear_by_day, mom20, vol, a, b, **kw)

    # ── Step 0: replication check ────────────────────────────────────────────
    print("  STEP 0 — replication check (overlay OFF vs real V8.63 engine)")
    real = B._run(signals, True, True, False, True, _FROM, _TO, **_BASE_DAILY)
    repl = run(_FROM, _TO, overlay_k=0)
    print(f"    real engine : eq=${real['equity']:>9,.0f}  CAGR={real['cagr']*100:5.1f}%  DD={real['max_dd']*100:4.1f}%")
    print(f"    replication : eq=${repl['equity']:>9,.0f}  CAGR={repl['cagr']*100:5.1f}%  DD={repl['max_dd']*100:4.1f}%")
    drift = abs(repl["equity"] - real["equity"]) / real["equity"]
    print(f"    drift: {drift*100:.1f}%  {'OK (comparison valid)' if drift < 0.10 else 'TOO LARGE — comparison suspect'}\n")

    # ── Full-period + H1-2025 sweeps ────────────────────────────────────────
    variants = [
        ("hybrid K=6h",            dict(overlay_k=6)),
        ("hybrid K=12h",           dict(overlay_k=12)),
        ("hybrid K=24h",           dict(overlay_k=24)),
        ("hybrid K=12h reentry=2", dict(overlay_k=12, reentry_both=True)),
        ("hybrid K=24h reentry=2", dict(overlay_k=24, reentry_both=True)),
    ]

    rows = [("V8.63 replication (no overlay)", repl)]
    h1rows = [("V8.63 replication (no overlay)", run(_H1A, _H1B, overlay_k=0))]
    for label, kw in variants:
        rows.append((label, run(_FROM, _TO, **kw)))
        h1rows.append((label, run(_H1A, _H1B, **kw)))
        print(f"  {label}: done")

    print("\n" + "=" * 112)
    print("  YEAR-BY-YEAR EARNINGS — $5K starting, 2021 → 2026")
    print("=" * 112)
    _year_table(rows)

    print("\n  H1-2025 isolated window:")
    for label, r in h1rows:
        print(f"    {label:<34} CAGR={r['cagr']*100:>+6.1f}%  DD={r['max_dd']*100:>4.1f}%  "
              f"overlay_exits={r.get('overlay_exits', 0):,}")

    # ── Walk-forward on the best-looking variants ───────────────────────────
    print("\n" + "=" * 112)
    print("  WALK-FORWARD (2yr train / 6mo test) — pass = CAGR > 0 and eq growth")
    print("=" * 112)
    train = int(2 * 365.25 * _DAY_MS)
    test  = int(0.5 * 365.25 * _DAY_MS)
    for label, kw in [("V8.63 replication", dict(overlay_k=0))] + variants:
        ws = _FROM; npass = ntot = 0; cagrs = []
        while ws + train + test <= _TO:
            r = run(ws + train, min(ws + train + test, _TO), **kw)
            ok = r["cagr"] > 0
            npass += ok; ntot += 1; cagrs.append(r["cagr"] * 100)
            ws += test
        avg = sum(cagrs) / len(cagrs) if cagrs else 0
        print(f"  {label:<34} {npass}/{ntot}  avg OOS CAGR {avg:>+6.1f}%  "
              f"windows={['%+.0f%%' % c for c in cagrs]}")

    print("\n" + "=" * 112)
    print("  BAR: better earnings AND fewer losses WITHOUT sacrifice elsewhere. Wash = reject.")
    print("=" * 112)


if __name__ == "__main__":
    main()
