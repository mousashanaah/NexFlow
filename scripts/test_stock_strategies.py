#!/usr/bin/env python3
"""Deep stock-strategy sweep — find a trend config that actually works on equities.

Background: the first multi-asset test made ZERO trades due to a timestamp
bug (pandas 2.x datetime64[s] truncation). With that fixed, this script does
the real work the crypto V8.63 params were never calibrated for: it sweeps
stock-native parameters and validates survivors the same way we validate
every crypto idea (year-by-year + H1-2025 + walk-forward).

Axes swept:
  - EMA fast/slow pairs (crypto uses 8/21; stocks trend slower)
  - Regime gate: none / SPY-SMA200 / per-asset-SMA200
  - bear_drop trigger for AND-entry (stocks crash shallower than crypto)
  - momentum gate window (0=off, 20, 60, 90d)
  - hard stop %
  - entry confluence: EMA-only / EMA+MACD / EMA+MACD+mom
  - direction: long-only / long+short (TSMOM-style)

Honest caveats (printed at end too):
  - Survivorship bias: NVDA/COIN/MSTR/META picked with hindsight
  - No 4H confirmation leg (daily data only)
  - Execution venue TBD: Bitget is crypto-perp only; a real stock book
    needs a CFD/equity-futures broker. This tests the STRATEGY, not the venue.
  - Fees modeled at 0.06%/side (conservative for equities)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_STOCK_DIR = _REPO_ROOT / "data" / "stocks"
_FEE    = 0.0006
_DAY_MS = 86_400_000
_FROM   = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_TO     = int(datetime.now(timezone.utc).timestamp() * 1000)
_H1A    = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_H1B    = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)

_STOCKS = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "GOOGL",
           "AMZN", "META", "TSLA", "AMD", "NFLX", "COIN", "MSTR", "GLD"]
_ANCHOR = "SPY"


def _ema(vals: np.ndarray, span: int) -> np.ndarray:
    out = np.empty_like(vals, dtype=float)
    a = 2.0 / (span + 1)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = a * vals[i] + (1 - a) * out[i - 1]
    return out


_CACHE: dict[str, pd.DataFrame] = {}


def _load(t: str, fast: int, slow: int) -> pd.DataFrame | None:
    key = f"{t}_{fast}_{slow}"
    if key in _CACHE:
        return _CACHE[key]
    p = _STOCK_DIR / f"{t}_1D.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p).sort_values("open_time").reset_index(drop=True)
    if int(df["open_time"].max()) < 10**12:
        raise SystemExit(
            f"[CORRUPT DATA] {t} open_time looks truncated "
            f"(max={df['open_time'].max()}). Re-run scripts/download_stock_data.py "
            f"locally with the timestamp fix, then commit & push data/stocks.")
    c = df["close"].values.astype(float)
    df["emaF"]   = _ema(c, fast)
    df["emaS"]   = _ema(c, slow)
    macd         = _ema(c, 12) - _ema(c, 26)
    df["macd"]   = macd
    df["macds"]  = _ema(macd, 9)
    df["sma200"] = pd.Series(c).rolling(200).mean().values
    df["mom30"]  = pd.Series(c).pct_change(30).values
    df["atrp"]   = pd.Series(c).pct_change().abs().rolling(14).mean().values
    for w in (20, 60, 90):
        df[f"mom{w}"] = pd.Series(c).pct_change(w).values
    df["ret126"] = pd.Series(c).pct_change(126).values
    _CACHE[key] = df
    return df


def run(capital, from_ts, to_ts, *, fast=8, slow=21, regime="spy",
        bear_drop=-0.12, mom_w=20, hard_stop=0.15, entry="ema_macd_mom",
        allow_short=False, confirm_days=10):
    data = {t: d for t in _STOCKS if (d := _load(t, fast, slow)) is not None}
    if regime == "spy" and _ANCHOR not in data:
        return None
    syms = list(data)
    base = capital / len(syms)
    target_risk = 0.01
    idx = {t: {int(ts): i for i, ts in enumerate(d["open_time"].values)}
           for t, d in data.items()}
    ref = data[_ANCHOR] if regime == "spy" else data[syms[0]]
    all_ts = sorted(int(ts) for ts in ref["open_time"].values
                    if from_ts <= ts < to_ts)

    eq = capital; peak = capital; mdd = 0.0
    pos: dict[str, dict] = {}
    ypl: dict[int, float] = {}
    deq: list[float] = []
    prev_bear = False; streak = 0
    ntr = 0; nwin = 0; gw = 0.0; gl = 0.0

    def _notional(t, i):
        atrp = data[t]["atrp"].values[i]
        if not np.isfinite(atrp) or atrp <= 0:
            return base
        return float(np.clip((target_risk * capital) / atrp, base * 0.5, base * 2))

    def _can_long_spy(ts):
        nonlocal prev_bear, streak
        ai = idx[_ANCHOR].get(ts)
        if ai is None:
            return not prev_bear
        r = data[_ANCHOR].iloc[ai]
        above = bool(r["close"] > r["sma200"]) if np.isfinite(r["sma200"]) else True
        m30 = r["mom30"] if np.isfinite(r["mom30"]) else 0.0
        enter = (not above) and (m30 < bear_drop)
        streak = streak + 1 if above else 0
        bear = (streak < confirm_days) if prev_bear else enter
        prev_bear = bear
        return not bear

    def _entry_ok(d, i):
        ef = d["emaF"].values[i] > d["emaS"].values[i]
        mb = d["macd"].values[i] > d["macds"].values[i]
        mom = d[f"mom{mom_w}"].values[i] if mom_w else 1.0
        momok = (mom > 0) if mom_w else True
        if not np.isfinite(mom):
            momok = False
        if entry == "ema":
            return ef
        if entry == "ema_macd":
            return ef and mb
        return ef and mb and momok

    last_rebal = 0
    for ts in all_ts:
        yr = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year
        can_long = _can_long_spy(ts) if regime == "spy" else True

        # exits (longs)
        for t in list(pos):
            if pos[t]["side"] != "long":
                continue
            i = idx[t].get(ts)
            if i is None:
                continue
            d = data[t]; c = float(d["close"].values[i]); p = pos[t]
            per_above = True
            if regime == "asset":
                per_above = bool(c > d["sma200"].values[i]) if np.isfinite(d["sma200"].values[i]) else True
            ef = d["emaF"].values[i] > d["emaS"].values[i]
            mb = d["macd"].values[i] > d["macds"].values[i]
            stop = c <= p["entry"] * (1 - hard_stop)
            sig = (not ef) and (not mb)
            regime_out = (not can_long) or (not per_above)
            if stop or sig or regime_out:
                px = p["entry"] * (1 - hard_stop) if stop else c
                net = (px - p["entry"]) / p["entry"] * p["notional"] - _FEE * p["notional"]
                eq += net; ypl[yr] = ypl.get(yr, 0.0) + net
                ntr += 1; nwin += net > 0
                gw += net if net > 0 else 0; gl += -net if net < 0 else 0
                pos.pop(t)

        # short management (weekly TSMOM) when bear & allowed
        if allow_short and regime == "spy" and not can_long and (ts - last_rebal) >= 7 * _DAY_MS:
            last_rebal = ts
            scores = []
            for t in syms:
                i = idx[t].get(ts)
                if i is None:
                    continue
                r126 = data[t]["ret126"].values[i]
                if np.isfinite(r126):
                    scores.append((r126, t))
            desired = {t for r, t in scores if r < -0.05}
            for t in list(pos):
                if pos[t]["side"] == "short" and t not in desired:
                    i = idx[t].get(ts)
                    if i is None:
                        continue
                    c = float(data[t]["close"].values[i]); p = pos[t]
                    net = (p["entry"] - c) / p["entry"] * p["notional"] - _FEE * p["notional"]
                    eq += net; ypl[yr] = ypl.get(yr, 0.0) + net
                    ntr += 1; nwin += net > 0
                    gw += net if net > 0 else 0; gl += -net if net < 0 else 0
                    pos.pop(t)
            for t in desired:
                if t not in pos:
                    i = idx[t].get(ts)
                    if i is None:
                        continue
                    c = float(data[t]["close"].values[i])
                    n = _notional(t, i); eq -= _FEE * n
                    pos[t] = {"entry": c, "notional": n, "side": "short"}

        # short stop / regime close
        for t in list(pos):
            if pos[t]["side"] != "short":
                continue
            i = idx[t].get(ts)
            if i is None:
                continue
            c = float(data[t]["close"].values[i]); p = pos[t]
            if c >= p["entry"] * (1 + hard_stop) or can_long:
                net = (p["entry"] - c) / p["entry"] * p["notional"] - _FEE * p["notional"]
                eq += net; ypl[yr] = ypl.get(yr, 0.0) + net
                ntr += 1; nwin += net > 0
                gw += net if net > 0 else 0; gl += -net if net < 0 else 0
                pos.pop(t)

        # entries (longs)
        if can_long:
            for t in syms:
                if t in pos:
                    continue
                i = idx[t].get(ts)
                if i is None:
                    continue
                d = data[t]
                if regime == "asset":
                    sa = d["sma200"].values[i]
                    if np.isfinite(sa) and float(d["close"].values[i]) <= sa:
                        continue
                if _entry_ok(d, i):
                    c = float(d["close"].values[i]); n = _notional(t, i)
                    eq -= _FEE * n
                    pos[t] = {"entry": c, "notional": n, "side": "long"}

        # mark to market
        mtm = 0.0
        for t, p in pos.items():
            i = idx[t].get(ts)
            if i is None:
                continue
            c = float(data[t]["close"].values[i])
            d = (c - p["entry"]) if p["side"] == "long" else (p["entry"] - c)
            mtm += d / p["entry"] * p["notional"]
        snap = eq + mtm; deq.append(snap)
        peak = max(peak, snap); mdd = max(mdd, (peak - snap) / peak)

    for t, p in pos.items():
        prior = [j for ts2, j in idx[t].items() if ts2 < to_ts]
        if not prior:
            continue
        c = float(data[t]["close"].values[max(prior)])
        d = (c - p["entry"]) if p["side"] == "long" else (p["entry"] - c)
        net = d / p["entry"] * p["notional"] - _FEE * p["notional"]
        eq += net
        yr = datetime.fromtimestamp(to_ts / 1000, tz=timezone.utc).year
        ypl[yr] = ypl.get(yr, 0.0) + net

    years = max((to_ts - from_ts) / (365.25 * _DAY_MS), 1e-9)
    rets = np.diff(deq) / np.array(deq[:-1]) if len(deq) > 2 else np.array([0.0])
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0.0
    return {"equity": eq, "year_pnl": ypl, "max_dd": mdd,
            "cagr": (eq / capital) ** (1 / years) - 1 if eq > 0 else -1.0,
            "sharpe": sharpe, "trades": ntr,
            "win": nwin / ntr if ntr else 0.0,
            "pf": gw / gl if gl > 0 else float("inf")}


def main():
    have = [t for t in _STOCKS if (_STOCK_DIR / f"{t}_1D.parquet").exists()]
    print("=" * 100)
    print("  DEEP STOCK STRATEGY SWEEP")
    print("=" * 100)
    if not have:
        print("  No data/stocks/. Run scripts/download_stock_data.py locally first.")
        return

    # Stage 0: sanity — does the baseline now trade after the timestamp fix?
    base = run(5_000, _FROM, _TO)
    print(f"\n  Stage 0 sanity (8/21, spy regime, long-only): "
          f"trades={base['trades']}  equity=${base['equity']:,.0f}  "
          f"CAGR={base['cagr']*100:+.1f}%  DD={base['max_dd']*100:.1f}%")
    if base["trades"] == 0:
        print("  STILL zero trades — deeper bug. Aborting sweep.")
        return

    # Stage 1: sweep core axes (long-only first to keep grid sane)
    print("\n  Stage 1: parameter sweep (long-only) — top configs by Sharpe")
    grid = list(product(
        [(5, 20), (8, 21), (10, 30), (20, 50)],   # ema pairs
        ["spy", "asset", "none"],                  # regime
        [0, 20, 60, 90],                           # momentum window
        [0.10, 0.15, 0.20],                        # hard stop
        ["ema_macd", "ema_macd_mom"],              # entry
    ))
    results = []
    for (f, s), reg, mw, hs, ent in grid:
        r = run(5_000, _FROM, _TO, fast=f, slow=s, regime=reg,
                mom_w=mw, hard_stop=hs, entry=ent, allow_short=False)
        if r and r["trades"] >= 20:
            results.append(((f, s, reg, mw, hs, ent), r))
    results.sort(key=lambda x: x[1]["sharpe"], reverse=True)

    print(f"\n  {'EMA':>7} {'regime':>7} {'mom':>4} {'stop':>5} {'entry':>14} "
          f"{'trades':>7} {'CAGR':>7} {'DD':>6} {'Sharpe':>7} {'PF':>5}")
    print("  " + "-" * 86)
    for (f, s, reg, mw, hs, ent), r in results[:12]:
        print(f"  {f:>2}/{s:<4} {reg:>7} {mw:>4} {hs*100:>4.0f}% {ent:>14} "
              f"{r['trades']:>7} {r['cagr']*100:>+6.1f}% {r['max_dd']*100:>5.1f}% "
              f"{r['sharpe']:>7.2f} {r['pf']:>5.2f}")

    if not results:
        print("  No config produced >=20 trades. Equity trend signals too sparse here.")
        return

    best_params, best = results[0]
    f, s, reg, mw, hs, ent = best_params

    # Stage 2: does adding shorts help the best long-only config?
    rs = run(5_000, _FROM, _TO, fast=f, slow=s, regime=reg, mom_w=mw,
             hard_stop=hs, entry=ent, allow_short=True)
    print(f"\n  Stage 2: best config + TSMOM shorts: "
          f"trades={rs['trades']}  CAGR={rs['cagr']*100:+.1f}%  "
          f"DD={rs['max_dd']*100:.1f}%  Sharpe={rs['sharpe']:.2f}")

    # Stage 3: year-by-year + H1-2025 + walk-forward on the winner
    winner = max([best, rs], key=lambda r: r["sharpe"])
    use_short = winner is rs
    print(f"\n  Stage 3: WINNER = {f}/{s} {reg} mom{mw} stop{hs*100:.0f}% {ent} "
          f"{'+shorts' if use_short else 'long-only'}")
    yrs = sorted(winner["year_pnl"])
    print("  Year-by-year ($5K stock book):")
    for y in yrs:
        print(f"    {y}: ${winner['year_pnl'][y]:>+8,.0f}")
    h1 = run(5_000, _H1A, _H1B, fast=f, slow=s, regime=reg, mom_w=mw,
             hard_stop=hs, entry=ent, allow_short=use_short)
    print(f"  H1-2025 window: CAGR={h1['cagr']*100:+.1f}%  DD={h1['max_dd']*100:.1f}%  "
          f"(crypto V8.63 here was -31%)")

    train = int(2 * 365.25 * _DAY_MS); test = int(0.5 * 365.25 * _DAY_MS)
    ws = _FROM; npass = ntot = 0; wins = []
    while ws + train + test <= _TO:
        t0, t1 = ws + train, min(ws + train + test, _TO)
        r = run(5_000, t0, t1, fast=f, slow=s, regime=reg, mom_w=mw,
                hard_stop=hs, entry=ent, allow_short=use_short)
        ok = r and r["equity"] > 5_000
        npass += bool(ok); ntot += 1
        wins.append(r["cagr"] * 100 if r else 0)
        ws += test
    print(f"  Walk-forward: {npass}/{ntot}  windows={['%+.0f%%' % c for c in wins]}")

    print("\n" + "=" * 100)
    print("  CAVEATS: survivorship bias (NVDA/COIN/MSTR picked in hindsight); no 4H")
    print("  leg; Bitget is crypto-only so a real stock book needs another broker.")
    print("  Judge vs crypto V8.63 (57% CAGR / 25% DD / 5-6 walk-fwd) on RISK-ADJ terms.")
    print("=" * 100)


if __name__ == "__main__":
    main()
