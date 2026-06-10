#!/usr/bin/env python3
"""Short-book research — the never-examined half of V8.63.

V8.63's TSMOM short side uses: 126d lookback, ret < -5% threshold, weekly
rebalance, 15% hard stop. These were picked once and never swept. The long
side has survived 13 rejection tests; the short side has survived zero.

Uses the validated replication engine approach (0.0% drift vs real engine,
see test_hybrid_overlay.py STEP 0) with the short parameters exposed.

Sweeps:
  S1. Momentum lookback: 60 / 90 / 126 / 180 days
  S2. Entry threshold: -2% / -5% / -10%
  S3. Rebalance cadence: 3 / 7 / 14 days
  S4. Short hard stop: 10% / 15% / 20%
  S5. Best combo + walk-forward + bear-window isolation

Judged on: 2022 PnL (the bear year), full-period equity/DD, and the
2025 bear stretch. The chosen V8.63 values must lose clearly before we
consider changing them.

Usage:
  python scripts/test_short_book.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import scripts.backtest_full_regime_system as B  # noqa: E402

B._CAPITAL = 5_000.0
_CAPITAL = 5_000.0
_DAY_MS = 86_400_000
_FEE = 0.0006
_FROM = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_TO   = int(datetime.now(timezone.utc).timestamp() * 1000)
# 2022 bear year isolation
_B22A = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_B22B = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

_SYMBOLS = B._SYMBOLS


def _daily_aux(signals: dict):
    btc_ts = sorted(signals["BTCUSDT"].keys())
    btc_closes = [signals["BTCUSDT"][t]["close"] for t in btc_ts]
    sma200 = B._sma_series(btc_closes, 200)
    bear_by_day: dict[int, bool] = {}
    streak = 0
    prev_bear = False
    for i, t in enumerate(btc_ts):
        above = sma200[i] is not None and btc_closes[i] > sma200[i]
        mom30 = ((btc_closes[i] - btc_closes[i-30]) / btc_closes[i-30]) if i >= 30 else 0.0
        enter_bear = (not above) and (mom30 < -0.20)
        streak = streak + 1 if above else 0
        bear = (not (streak >= 10)) if prev_bear else enter_bear
        prev_bear = bear
        bear_by_day[t] = bear

    mom20: dict[str, dict[int, float]] = {}
    vol:   dict[str, dict[int, float]] = {}
    for sym in _SYMBOLS:
        ts_list = sorted(signals.get(sym, {}).keys())
        closes = [signals[sym][t]["close"] for t in ts_list]
        mom20[sym] = {ts_list[i]: (closes[i] - closes[i-20]) / closes[i-20]
                      for i in range(20, len(ts_list))}
        vol[sym] = B._vol_series(signals, sym)
    return bear_by_day, mom20, vol


def _size(sym, ts, mult, vol):
    base = _CAPITAL / len(_SYMBOLS)
    v = vol.get(sym, {}).get(ts, 0.0)
    if v <= 0:
        return base * mult
    return min((0.01 * _CAPITAL) / v * mult, base * 2 * mult)


def _run(signals, bear_by_day, mom20, vol, from_ts, to_ts,
         short_lookback=126, short_thresh=-0.05, rebal_days=7,
         short_stop=0.15) -> dict:
    """V8.63 replication with parameterized short book (validated approach)."""
    daily_ts = sorted(set(t for s in _SYMBOLS for t in signals.get(s, {})
                          if from_ts <= t <= to_ts))
    equity = _CAPITAL
    longs:  dict[str, dict] = {}
    shorts: dict[str, dict] = {}
    last_rebal = 0
    year_pnl: dict[int, float] = {}
    daily_equity: list[float] = []
    short_pnl_total = 0.0
    n_short_trades = 0

    def _book(net, ts, is_short=False):
        nonlocal equity, short_pnl_total, n_short_trades
        equity += net
        yr = datetime.fromtimestamp(ts/1000, tz=timezone.utc).year
        year_pnl[yr] = year_pnl.get(yr, 0) + net
        if is_short:
            short_pnl_total += net
            n_short_trades += 1

    for ts in daily_ts:
        bear = bear_by_day.get(ts, False)
        bull = not bear

        if bear and (ts - last_rebal) >= rebal_days * _DAY_MS:
            last_rebal = ts
            desired = set()
            lb = short_lookback + 1
            for sym in _SYMBOLS:
                s_ts = sorted(signals.get(sym, {}).keys())
                past = [t for t in s_ts if t <= ts]
                if len(past) < lb:
                    continue
                c_now, c_past = signals[sym][past[-1]]["close"], signals[sym][past[-lb]]["close"]
                if (c_now - c_past) / c_past < short_thresh:
                    desired.add(sym)
            for sym in [s for s in list(shorts) if s not in desired]:
                c = signals.get(sym, {}).get(ts, {}).get("close")
                if c is None:
                    continue
                pos = shorts.pop(sym)
                raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                _book(raw - _FEE * pos["notional"], ts, is_short=True)
            for sym in desired:
                if sym not in shorts and sym not in longs:
                    c = signals.get(sym, {}).get(ts, {}).get("close")
                    if c is None:
                        continue
                    n = _size(sym, ts, 1.0, vol)
                    equity -= _FEE * n
                    shorts[sym] = {"entry": c, "notional": n}

        for sym in list(shorts):
            c = signals.get(sym, {}).get(ts, {}).get("close")
            if c is None:
                continue
            pos = shorts[sym]
            if (c - pos["entry"]) / pos["entry"] >= short_stop:
                shorts.pop(sym)
                raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                _book(raw - _FEE * pos["notional"], ts, is_short=True)

        if bull:
            for sym in list(shorts):
                c = signals.get(sym, {}).get(ts, {}).get("close")
                if c is None:
                    continue
                pos = shorts.pop(sym)
                raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                _book(raw - _FEE * pos["notional"], ts, is_short=True)

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
            in_pos = sym in longs
            if in_pos and (not any_long or not can_long):
                pos = longs.pop(sym)
                raw = (c - pos["entry"]) / pos["entry"] * pos["notional"]
                _book(raw - _FEE * pos["notional"], ts)
                in_pos = False
            if not in_pos and any_long and can_long and sym not in shorts:
                if mom20.get(sym, {}).get(ts, 0.0) <= 0:
                    continue
                mult = {1: 1.0, 2: 1.5, 3: 2.0}.get(n_long, 1.0)
                n = _size(sym, ts, mult, vol)
                equity -= _FEE * n
                longs[sym] = {"entry": c, "notional": n}

        mtm = 0.0
        for sym, pos in longs.items():
            c = signals.get(sym, {}).get(ts, {}).get("close")
            if c is not None:
                mtm += (c - pos["entry"]) / pos["entry"] * pos["notional"]
        for sym, pos in shorts.items():
            c = signals.get(sym, {}).get(ts, {}).get("close")
            if c is not None:
                mtm += (pos["entry"] - c) / pos["entry"] * pos["notional"]
        daily_equity.append(equity + mtm)

    last = daily_ts[-1] if daily_ts else to_ts
    unreal = 0.0
    for sym, pos in longs.items():
        c = signals.get(sym, {}).get(last, {}).get("close", pos["entry"])
        unreal += (c - pos["entry"]) / pos["entry"] * pos["notional"]
    for sym, pos in shorts.items():
        c = signals.get(sym, {}).get(last, {}).get("close", pos["entry"])
        unreal += (pos["entry"] - c) / pos["entry"] * pos["notional"]
    equity += unreal
    yr = datetime.fromtimestamp(last/1000, tz=timezone.utc).year
    year_pnl[yr] = year_pnl.get(yr, 0) + unreal

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
            std = (sum((r - mean)**2 for r in rets) / len(rets)) ** 0.5
            sharpe = (mean / std) * (252 ** 0.5) if std > 0 else 0.0

    return {"equity": equity, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe,
            "year_pnl": year_pnl, "short_pnl": short_pnl_total,
            "n_short_trades": n_short_trades}


def _row(label, r, r22):
    print(f"  {label:<36} eq=${r['equity']:>8,.0f} CAGR={r['cagr']*100:>5.1f}% "
          f"DD={r['max_dd']*100:>4.1f}% Sh={r['sharpe']:.2f} "
          f"shortPnL=${r['short_pnl']:>+8,.0f} ({r['n_short_trades']:>3} trades) | "
          f"2022: ${r22['year_pnl'].get(2022, 0):>+8,.0f}")


def main():
    print("=" * 110)
    print("  SHORT-BOOK RESEARCH — sweeping the never-examined half of V8.63 ($5K)")
    print("=" * 110)
    print("Building signals ...")
    signals = B._build_signals(_SYMBOLS)
    bear_by_day, mom20, vol = _daily_aux(signals)
    print("Done.\n")

    def run(a, b, **kw):
        return _run(signals, bear_by_day, mom20, vol, a, b, **kw)

    # Replication check vs real engine
    real = B._run(signals, True, True, False, True, _FROM, _TO,
                  hard_stop_pct=0.15, use_atr_sizing=True,
                  asymmetric_regime=True, and_entry=True,
                  bear_drop_pct=-0.20, confirm_days=10,
                  momentum_gate=True, momentum_gate_days=20)
    base = run(_FROM, _TO)
    drift = abs(base["equity"] - real["equity"]) / real["equity"]
    print(f"  Replication check: real=${real['equity']:,.0f} repl=${base['equity']:,.0f} "
          f"drift={drift*100:.1f}% {'OK' if drift < 0.05 else 'SUSPECT'}\n")

    base22 = run(_B22A, _B22B)
    print("  S1 — Momentum lookback (V8.63 chosen: 126d)")
    for lb in (60, 90, 126, 180):
        r, r22 = run(_FROM, _TO, short_lookback=lb), run(_B22A, _B22B, short_lookback=lb)
        _row(f"lookback={lb}d" + (" <- chosen" if lb == 126 else ""), r, r22)

    print("\n  S2 — Entry threshold (chosen: -5%)")
    for th in (-0.02, -0.05, -0.10):
        r, r22 = run(_FROM, _TO, short_thresh=th), run(_B22A, _B22B, short_thresh=th)
        _row(f"thresh={th:+.0%}" + (" <- chosen" if th == -0.05 else ""), r, r22)

    print("\n  S3 — Rebalance cadence (chosen: 7d)")
    for rd in (3, 7, 14):
        r, r22 = run(_FROM, _TO, rebal_days=rd), run(_B22A, _B22B, rebal_days=rd)
        _row(f"rebal={rd}d" + (" <- chosen" if rd == 7 else ""), r, r22)

    print("\n  S4 — Short hard stop (chosen: 15%)")
    for st in (0.10, 0.15, 0.20):
        r, r22 = run(_FROM, _TO, short_stop=st), run(_B22A, _B22B, short_stop=st)
        _row(f"stop={st:.0%}" + (" <- chosen" if st == 0.15 else ""), r, r22)

    print("\n  S5 — Promising combos")
    combos = {
        "fast: lb=60 th=-10% rebal=3":  dict(short_lookback=60, short_thresh=-0.10, rebal_days=3),
        "mid:  lb=90 th=-5%  rebal=7":  dict(short_lookback=90),
        "deep: lb=90 th=-10% rebal=7":  dict(short_lookback=90, short_thresh=-0.10),
    }
    for label, kw in combos.items():
        r, r22 = run(_FROM, _TO, **kw), run(_B22A, _B22B, **kw)
        _row(label, r, r22)

    # Walk-forward on baseline vs best
    print("\n  Walk-forward (2yr/6mo) — chosen vs alternatives")
    train = int(2 * 365.25 * _DAY_MS); test = int(0.5 * 365.25 * _DAY_MS)
    for label, kw in [("chosen (126/-5%/7d/15%)", {}),
                      ("lb=90", dict(short_lookback=90)),
                      ("lb=60 th=-10% rebal=3", dict(short_lookback=60, short_thresh=-0.10, rebal_days=3))]:
        ws = _FROM; npass = ntot = 0; cagrs = []
        while ws + train + test <= _TO:
            r = run(ws + train, min(ws + train + test, _TO), **kw)
            ok = r["cagr"] > 0
            npass += ok; ntot += 1; cagrs.append(r["cagr"] * 100)
            ws += test
        print(f"  {label:<36} {npass}/{ntot}  windows={['%+.0f%%' % c for c in cagrs]}")

    print("\n" + "=" * 110)
    print("  READ: chosen values should sit on a plateau. A clearly better cell that is")
    print("  ALSO robust across S1-S4 neighbours and walk-forward = candidate. Else keep.")
    print("=" * 110)


if __name__ == "__main__":
    main()
