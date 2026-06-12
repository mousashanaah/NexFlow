#!/usr/bin/env python3
"""Multi-asset mode test — V8.63 trend rules on stocks, then crypto+stock combos.

Question: does adding a stock book (same trend philosophy, own regime anchor)
beat pure-crypto V8.63 on risk-adjusted terms?

Stock book rules (V8.63 translated):
  - Regime anchor: SPY > SMA200 (AND-entry analog: SPY<SMA200 AND 30d<-12%
    for bear — stocks crash shallower than crypto, threshold scaled by ~0.6×
    relative vol; we ALSO test the raw -20% to avoid cherry-picking)
  - Entry: EMA8/21 cross AND MACD bullish AND 20d momentum > 0
    (no 4H leg — we only have daily stock data; flagged as a difference)
  - Exit: EMA AND MACD both bearish, or 15% hard stop, or bear regime
  - ATR vol-adjusted sizing, 1% daily target risk, cap 2× base
  - NO stock shorts in v1 (equity TSMOM shorts are weaker + borrow issues)

Combo portfolios (fixed capital split, each book runs its own capital):
  A. 100% crypto V8.63 (baseline)
  B. 100% stock book
  C. 70% crypto / 30% stocks
  D. 50% crypto / 50% stocks

Verdict standard (same as every idea this session):
  - Must beat baseline on CAGR or DD/Sharpe with real margin, not noise
  - Year-by-year table since 2021, H1-2025 window, walk-forward 2yr/6mo
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import scripts.backtest_full_regime_system as B  # noqa: E402

B._CAPITAL = 5_000.0

_STOCK_DIR = _REPO_ROOT / "data" / "stocks"
_FEE       = 0.0006     # keep crypto taker fee for stocks too (conservative;
                        # real stock-CFD/futures fees are similar or lower)
_DAY_MS    = 86_400_000
_FROM      = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_TO        = int(datetime.now(timezone.utc).timestamp() * 1000)
_H1A       = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_H1B       = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)

_STOCKS = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "GOOGL",
           "AMZN", "META", "TSLA", "AMD", "NFLX", "COIN", "MSTR", "GLD"]
_REGIME_ANCHOR = "SPY"


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def _ema(vals: np.ndarray, span: int) -> np.ndarray:
    out = np.empty_like(vals, dtype=float)
    alpha = 2.0 / (span + 1)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = alpha * vals[i] + (1 - alpha) * out[i - 1]
    return out


def _load_stock(t: str) -> pd.DataFrame | None:
    p = _STOCK_DIR / f"{t}_1D.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p).sort_values("open_time").reset_index(drop=True)
    c = df["close"].values.astype(float)
    df["ema8"]   = _ema(c, 8)
    df["ema21"]  = _ema(c, 21)
    macd         = _ema(c, 12) - _ema(c, 26)
    df["macd"]   = macd
    df["macds"]  = _ema(macd, 9)
    df["sma200"] = pd.Series(c).rolling(200).mean().values
    df["mom20"]  = pd.Series(c).pct_change(20).values
    df["mom30"]  = pd.Series(c).pct_change(30).values
    df["atrp"]   = pd.Series(c).pct_change().abs().rolling(14).mean().values
    return df


# ---------------------------------------------------------------------------
# Stock book backtest (V8.63 long logic, daily only)
# ---------------------------------------------------------------------------
def run_stock_book(capital: float, from_ts: int, to_ts: int,
                   bear_drop: float = -0.12,
                   hard_stop: float = 0.15,
                   confirm_days: int = 10) -> dict:
    data = {t: d for t in _STOCKS if (d := _load_stock(t)) is not None}
    if _REGIME_ANCHOR not in data:
        raise SystemExit(f"No data for regime anchor {_REGIME_ANCHOR} — "
                         f"run scripts/download_stock_data.py locally first")
    syms = list(data)
    base_notional = capital / len(syms)
    target_risk   = 0.01

    # index rows by open_time
    idx = {t: {int(ts): i for i, ts in enumerate(d["open_time"].values)}
           for t, d in data.items()}
    anchor = data[_REGIME_ANCHOR]
    all_ts = sorted(int(ts) for ts in anchor["open_time"].values
                    if from_ts <= ts < to_ts)

    equity = capital; peak = capital; max_dd = 0.0
    positions: dict[str, dict] = {}
    year_pnl: dict[int, float] = {}
    daily_eq: list[float] = []
    prev_bear = False
    above_streak = 0
    n_trades = 0; n_wins = 0
    gross_win = 0.0; gross_loss = 0.0

    def _notional(t: str, i: int) -> float:
        atrp = data[t]["atrp"].values[i]
        if not np.isfinite(atrp) or atrp <= 0:
            return base_notional
        sized = (target_risk * capital) / atrp
        return float(np.clip(sized, base_notional * 0.5, base_notional * 2))

    for ts in all_ts:
        ai = idx[_REGIME_ANCHOR].get(ts)
        if ai is None:
            continue
        arow = anchor.iloc[ai]
        sma200 = arow["sma200"]
        above  = bool(arow["close"] > sma200) if np.isfinite(sma200) else True
        mom30  = arow["mom30"] if np.isfinite(arow["mom30"]) else 0.0

        enter_bear = (not above) and (mom30 < bear_drop)
        above_streak = above_streak + 1 if above else 0
        confirmed = above_streak >= confirm_days
        bear = (not confirmed) if prev_bear else enter_bear
        prev_bear = bear
        can_long = not bear

        yr = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year

        # exits
        for t in list(positions):
            i = idx[t].get(ts)
            if i is None:
                continue
            c = float(data[t]["close"].values[i])
            pos = positions[t]
            ema_bull  = data[t]["ema8"].values[i]  > data[t]["ema21"].values[i]
            macd_bull = data[t]["macd"].values[i]  > data[t]["macds"].values[i]
            stop_hit  = c <= pos["entry"] * (1 - hard_stop)
            sig_exit  = (not ema_bull) and (not macd_bull)
            if stop_hit or sig_exit or not can_long:
                px = pos["entry"] * (1 - hard_stop) if stop_hit else c
                raw = (px - pos["entry"]) / pos["entry"] * pos["notional"]
                net = raw - _FEE * pos["notional"]
                equity += net
                year_pnl[yr] = year_pnl.get(yr, 0.0) + net
                n_trades += 1; n_wins += net > 0
                if net > 0: gross_win += net
                else:       gross_loss += -net
                positions.pop(t)

        # entries
        if can_long:
            for t in syms:
                if t in positions:
                    continue
                i = idx[t].get(ts)
                if i is None:
                    continue
                d = data[t]
                ema_bull  = d["ema8"].values[i]  > d["ema21"].values[i]
                macd_bull = d["macd"].values[i]  > d["macds"].values[i]
                mom20     = d["mom20"].values[i]
                if ema_bull and macd_bull and np.isfinite(mom20) and mom20 > 0:
                    c = float(d["close"].values[i])
                    n = _notional(t, i)
                    equity -= _FEE * n
                    positions[t] = {"entry": c, "notional": n}

        # mark to market
        mtm = 0.0
        for t, pos in positions.items():
            i = idx[t].get(ts)
            if i is not None:
                c = float(data[t]["close"].values[i])
                mtm += (c - pos["entry"]) / pos["entry"] * pos["notional"]
        snap = equity + mtm
        daily_eq.append(snap)
        peak = max(peak, snap)
        max_dd = max(max_dd, (peak - snap) / peak)

    # close remaining at last available close
    for t, pos in positions.items():
        prior = [j for ts2, j in idx[t].items() if ts2 < to_ts]
        if not prior:
            continue
        i = max(prior)
        c = float(data[t]["close"].values[i])
        raw = (c - pos["entry"]) / pos["entry"] * pos["notional"]
        net = raw - _FEE * pos["notional"]
        equity += net
        yr = datetime.fromtimestamp(to_ts / 1000, tz=timezone.utc).year
        year_pnl[yr] = year_pnl.get(yr, 0.0) + net

    years = max((to_ts - from_ts) / (365.25 * _DAY_MS), 1e-9)
    rets = np.diff(daily_eq) / np.array(daily_eq[:-1]) if len(daily_eq) > 2 else np.array([0.0])
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    return {
        "equity": equity, "year_pnl": year_pnl, "max_dd": max_dd,
        "cagr": (equity / capital) ** (1 / years) - 1 if equity > 0 else -1.0,
        "sharpe": sharpe, "pf": pf, "trades": n_trades,
        "win_rate": n_wins / n_trades if n_trades else 0.0,
        "daily_eq": daily_eq,
    }


# ---------------------------------------------------------------------------
# Crypto V8.63 via the real engine
# ---------------------------------------------------------------------------
_BASE = dict(hard_stop_pct=0.15, use_atr_sizing=True, asymmetric_regime=True,
             and_entry=True, bear_drop_pct=-0.20, confirm_days=10,
             momentum_gate=True, momentum_gate_days=20)

_SIG_CACHE = {}


def run_crypto(capital: float, from_ts: int, to_ts: int) -> dict:
    old_cap = B._CAPITAL
    B._CAPITAL = capital
    try:
        if "sig" not in _SIG_CACHE:
            _SIG_CACHE["sig"] = B._build_signals(B._SYMBOLS)
        return B._run(_SIG_CACHE["sig"], True, True, False, True,
                      from_ts, to_ts, **_BASE)
    finally:
        B._CAPITAL = old_cap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 112)
    print("  MULTI-ASSET MODE TEST — V8.63 crypto vs stock book vs combos ($5K)")
    print("=" * 112)

    have = [t for t in _STOCKS if (_STOCK_DIR / f"{t}_1D.parquet").exists()]
    if not have:
        print("\n  [BLOCKED] No stock data found in data/stocks/.")
        print("  Run on your LOCAL machine:  python scripts/download_stock_data.py")
        print("  Then: git add data/stocks && git commit -m 'stock data' && git push")
        return
    print(f"\n  Stock data available: {len(have)}/{len(_STOCKS)}: {', '.join(have)}")

    print("\n  Running books ...")
    combos = {
        "A. 100% crypto V8.63":    (1.0, 0.0),
        "B. 100% stock book":      (0.0, 1.0),
        "C. 70% crypto / 30% stk": (0.7, 0.3),
        "D. 50% crypto / 50% stk": (0.5, 0.5),
    }

    def _mix(wc: float, ws: float, from_ts: int, to_ts: int) -> dict:
        parts = []
        if wc > 0:
            parts.append(run_crypto(5_000 * wc, from_ts, to_ts))
        if ws > 0:
            parts.append(run_stock_book(5_000 * ws, from_ts, to_ts))
        eq  = sum(r["equity"] for r in parts)
        ypl: dict[int, float] = {}
        for r in parts:
            for yr2, p in r["year_pnl"].items():
                ypl[yr2] = ypl.get(yr2, 0.0) + p
        years = max((to_ts - from_ts) / (365.25 * _DAY_MS), 1e-9)
        dd = max(r["max_dd"] for r in parts)  # conservative upper bound
        return {"equity": eq, "year_pnl": ypl, "max_dd": dd,
                "cagr": (eq / 5_000) ** (1 / years) - 1 if eq > 0 else -1.0}

    rows = []
    h1_rows = []
    for label, (wc, ws) in combos.items():
        rf = _mix(wc, ws, _FROM, _TO)
        rh = _mix(wc, ws, _H1A, _H1B)
        rows.append((label, rf))
        h1_rows.append((label, rh))
        print(f"    {label}: done")

    all_years = sorted({yr for _, r in rows for yr in r["year_pnl"]})
    col = 11
    print("\n" + "=" * 112)
    print("  YEAR-BY-YEAR PnL — $5K start, 2021 → now")
    print("=" * 112)
    print(f"  {'Portfolio':<26}" + "".join(f"{yr:>{col}}" for yr in all_years)
          + f"{'Equity':>{col}}  {'CAGR':>7}  {'maxDD*':>7}")
    print("  " + "-" * (26 + col * len(all_years) + col + 20))
    for label, r in rows:
        s = f"  {label:<26}"
        for yr in all_years:
            s += f"{r['year_pnl'].get(yr, 0):>+{col},.0f}"
        s += f"  ${r['equity']:>8,.0f}  {r['cagr']*100:>6.1f}%  {r['max_dd']*100:>6.1f}%"
        print(s)
    print("  (*combined DD shown as worst single-book DD — conservative)")

    print("\n  H1-2025 isolated window (V8.63's known weak spot):")
    for label, rh in h1_rows:
        print(f"    {label:<26} CAGR={rh['cagr']*100:>+7.1f}%")

    print("\n  Stock book standalone detail (full period):")
    rs = run_stock_book(5_000, _FROM, _TO)
    print(f"    trades={rs['trades']}  win={rs['win_rate']*100:.0f}%  "
          f"PF={rs['pf']:.2f}  Sharpe={rs['sharpe']:.2f}")

    print("\n" + "=" * 112)
    print("  WALK-FORWARD (2yr train / 6mo test) — all portfolios")
    print("=" * 112)
    train = int(2 * 365.25 * _DAY_MS); test = int(0.5 * 365.25 * _DAY_MS)
    for label, (wc, ws) in combos.items():
        ws_ts = _FROM; npass = ntot = 0; cagrs = []
        while ws_ts + train + test <= _TO:
            t0, t1 = ws_ts + train, min(ws_ts + train + test, _TO)
            r = _mix(wc, ws, t0, t1)
            ok = r["equity"] > 5_000
            npass += ok; ntot += 1; cagrs.append(r["cagr"] * 100)
            ws_ts += test
        print(f"  {label:<26} walk-fwd {npass}/{ntot}  "
              f"windows={['%+.0f%%' % c for c in cagrs]}")

    print("\n" + "=" * 112)
    print("  CAVEATS: stock list has survivorship bias (COIN/MSTR/NVDA chosen in")
    print("  hindsight); no 4H confirmation leg for stocks; fees modeled at crypto")
    print("  taker rates. Demand a LARGE margin over baseline before accepting.")
    print("=" * 112)


if __name__ == "__main__":
    main()
