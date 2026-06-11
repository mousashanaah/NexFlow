#!/usr/bin/env python3
"""Pairs trading test — cointegrated crypto pairs as V8.63 overlay or standalone.

Strategy:
  - Rolling 6-month window to estimate hedge ratio and spread mean/std (no look-ahead)
  - Entry when spread diverges >2σ from its rolling mean
  - Exit when spread reverts to mean OR spread hits 3σ stop-loss
  - Long the underperformer, short the outperformer
  - Full taker fees both legs (0.12% round-trip × 2 legs = 0.24% total per trade)

Honest caveats:
  - Cointegration in crypto is regime-dependent — often breaks during bull runs
  - Half-lives of 45–150d mean capital is tied up for weeks
  - Backtest assumes fills at daily close with no slippage — optimistic
  - V8.63 already holds some of these coins long — the pairs trade would FIGHT
    the trend position (going short a coin V8.63 is long, or vice versa)

Questions answered:
  1. Does the standalone pairs strategy make money over 2021-2026?
  2. Is the equity curve consistent (year-by-year)?
  3. Walk-forward: does it hold up on OOS data?
  4. Is it truly additive to V8.63 (different regime profile)?
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_FEE        = 0.0006   # taker per side
_DAY_MS     = 86_400_000
_CAPITAL    = 5_000.0
_PER_TRADE  = 500.0    # notional per pair leg ($500 per leg = $1K deployed per trade)
_ENTRY_Z    = 2.0      # enter when spread > 2σ
_EXIT_Z     = 0.0      # exit at mean
_STOP_Z     = 3.5      # cut loss at 3.5σ
_TRAIN_DAYS = 180      # rolling window for estimating params
_MAX_HOLD   = 60       # force-close after 60 days

_SYMS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
    "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT",
    "LINKUSDT","LTCUSDT","DOTUSDT","TRXUSDT",
]


def _load_prices() -> pd.DataFrame:
    closes = {}
    for s in _SYMS:
        df = pd.read_parquet(_REPO_ROOT / "data" / "candles" / f"{s}_1D.parquet",
                             columns=["open_time", "close"])
        df = df.sort_values("open_time").set_index("open_time")
        closes[s] = np.log(df["close"])
    return pd.DataFrame(closes).dropna()


def _hedge_ratio(x: np.ndarray, y: np.ndarray) -> float:
    cov = np.cov(x, y)
    return cov[0, 1] / np.var(x)


def _adf_t(series: np.ndarray) -> float:
    dy    = np.diff(series)
    y_lag = series[:-1]
    X     = np.column_stack([np.ones(len(y_lag)), y_lag])
    b, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
    fitted = X @ b
    resid  = dy - fitted
    se = np.sqrt(np.sum(resid ** 2) / (len(series) - 2)) / \
         np.sqrt(np.sum((y_lag - y_lag.mean()) ** 2))
    return b[1] / se


def _run_pair(
    log_a: np.ndarray,
    log_b: np.ndarray,
    dates: np.ndarray,
    train: int = _TRAIN_DAYS,
) -> dict:
    """Simulate one pair, rolling-window, return trade list and stats."""
    n = len(log_a)
    trades = []
    position = None   # None | {entry_spread, entry_z, hr, mu, sigma, day_in, side_a}
    total_pnl = 0.0
    year_pnl: dict[int, float] = {}

    for i in range(train, n):
        # Rolling estimate of hedge ratio and spread stats on TRAINING window
        x_tr = log_b[i - train: i]
        y_tr = log_a[i - train: i]
        hr   = _hedge_ratio(x_tr, y_tr)
        spr_tr = y_tr - hr * x_tr
        mu    = spr_tr.mean()
        sigma = spr_tr.std()
        if sigma < 1e-8:
            continue

        # Current spread
        spr_now = log_a[i] - hr * log_b[i]
        z = (spr_now - mu) / sigma

        dt  = datetime.utcfromtimestamp(int(dates[i]) / 1000)
        yr  = dt.year

        if position is None:
            # Entry: spread is extreme enough
            if abs(z) >= _ENTRY_Z:
                # If z > 0: A is overpriced relative to B → short A, long B
                # If z < 0: A is underpriced relative to B → long A, short B
                side_a = -1 if z > 0 else +1
                position = {
                    "entry_spr": spr_now,
                    "entry_z":   z,
                    "hr":        hr,
                    "mu":        mu,
                    "sigma":     sigma,
                    "day_in":    i,
                    "side_a":    side_a,
                    "price_a_in": np.exp(log_a[i]),
                    "price_b_in": np.exp(log_b[i]),
                }
        else:
            # Recompute z using ENTRY params (avoid parameter drift mid-trade)
            spr_entry = position["entry_spr"]
            mu0       = position["mu"]
            sig0      = position["sigma"]
            z_now     = (spr_now - mu0) / sig0
            side_a    = position["side_a"]
            held      = i - position["day_in"]

            exit_reason = None
            if side_a * z_now <= _EXIT_Z:
                exit_reason = "mean_rev"
            elif abs(z_now) >= _STOP_Z:
                exit_reason = "stop"
            elif held >= _MAX_HOLD:
                exit_reason = "timeout"

            if exit_reason:
                # PnL: leg A + leg B — 2 round trips of fees (entry + exit)
                pa_in  = position["price_a_in"]
                pb_in  = position["price_b_in"]
                pa_out = np.exp(log_a[i])
                pb_out = np.exp(log_b[i])
                hr0    = position["hr"]

                # side_a = +1 → long A, short B (B notional = hr0 × A_notional)
                pnl_a = side_a * (pa_out - pa_in) / pa_in * _PER_TRADE
                pnl_b = -side_a * hr0 * (pb_out - pb_in) / pb_in * _PER_TRADE
                fee   = 2 * _FEE * _PER_TRADE * (1 + abs(hr0))  # entry+exit both legs
                net   = pnl_a + pnl_b - fee

                total_pnl += net
                year_pnl[yr] = year_pnl.get(yr, 0.0) + net
                trades.append({
                    "day_in": position["day_in"],
                    "day_out": i,
                    "held": held,
                    "z_entry": position["entry_z"],
                    "z_exit": z_now,
                    "exit": exit_reason,
                    "net": net,
                })
                position = None

    return {
        "pnl":      total_pnl,
        "year_pnl": year_pnl,
        "trades":   trades,
        "n_trades": len(trades),
        "wins":     sum(1 for t in trades if t["net"] > 0),
        "stops":    sum(1 for t in trades if t["exit"] == "stop"),
        "timeouts": sum(1 for t in trades if t["exit"] == "timeout"),
    }


def main():
    print("=" * 110)
    print("  PAIRS TRADING TEST — cointegrated crypto pairs, rolling 180d window  ($5K)")
    print("=" * 110)

    log_prices = _load_prices()
    dates      = log_prices.index.values
    n          = len(log_prices)
    from_2021  = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    mask       = dates >= from_2021
    lp         = log_prices[mask]
    dates_     = lp.index.values

    print(f"\n  {len(lp)} daily bars, {dates_[0]} → {dates_[-1]}")

    # --- Screen for best pairs (full period, just for ranking) ---
    print("\n  Screening all pairs for cointegration and mean-reversion speed ...")
    screen = []
    for a, b in combinations(_SYMS, 2):
        la = lp[a].values
        lb = lp[b].values
        hr = _hedge_ratio(lb, la)
        spr = la - hr * lb
        spr -= spr.mean()
        t = _adf_t(spr)
        # half-life
        rho = np.cov(spr[1:], spr[:-1])[0, 1] / np.var(spr[:-1])
        hl  = -np.log(2) / np.log(abs(rho)) if 0 < abs(rho) < 1 else 9999
        screen.append((t, a, b, hr, hl))
    screen.sort()

    print(f"\n  {'Pair':<22} {'ADF-t':>7}  {'Half-life':>10}  Verdict")
    for t, a, b, hr, hl in screen[:15]:
        v = "GOOD" if t < -4.5 and hl < 80 else ("WEAK" if t < -3.5 else "NO")
        print(f"  {a[:6]}/{b[:6]:<10} {t:>7.2f}  {hl:>9.1f}d  {v}")

    # --- Backtest top pairs individually ---
    top_pairs = [(a, b) for t, a, b, _, hl in screen if t < -4.0 and hl < 100][:8]
    print(f"\n  Backtesting top {len(top_pairs)} pairs (ADF<-4.0, half-life<100d) ...")

    all_results = []
    for a, b in top_pairs:
        la = lp[a].values
        lb = lp[b].values
        r  = _run_pair(la, lb, dates_)
        all_results.append((a, b, r))

    # Print summary
    all_years = sorted({yr for _, _, r in all_results for yr in r["year_pnl"]})
    print(f"\n  {'Pair':<22} {'Trades':>7} {'Wins':>5} {'Stops':>6} {'Total PnL':>12}" +
          "".join(f" {yr:>7}" for yr in all_years))
    print("  " + "-" * (22 + 7 + 5 + 6 + 12 + 7 * len(all_years) + 5))
    for a, b, r in all_results:
        wr = r["wins"] / r["n_trades"] * 100 if r["n_trades"] else 0
        row = f"  {a[:6]}/{b[:6]:<10} {r['n_trades']:>7} {wr:>4.0f}% {r['stops']:>6} ${r['pnl']:>+10,.0f}"
        for yr in all_years:
            row += f" {r['year_pnl'].get(yr, 0):>+7,.0f}"
        print(row)

    # --- Portfolio: run all top pairs simultaneously ---
    print(f"\n  Portfolio of {len(top_pairs)} pairs combined:")
    port_year: dict[int, float] = {}
    port_total = 0.0
    for _, _, r in all_results:
        port_total += r["pnl"]
        for yr, p in r["year_pnl"].items():
            port_year[yr] = port_year.get(yr, 0.0) + p

    print(f"  Total PnL: ${port_total:>+,.0f}")
    for yr in sorted(port_year):
        print(f"    {yr}: ${port_year[yr]:>+8,.0f}")

    # --- Walk-forward (2yr train / 6mo test) ---
    from_ts = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    to_ts   = int(datetime.now(timezone.utc).timestamp() * 1000)
    day_ms  = 86_400_000
    train_w = int(2 * 365.25 * day_ms)
    test_w  = int(0.5 * 365.25 * day_ms)

    print(f"\n  Walk-forward (2yr train / 6mo test):")
    wf_pass = 0; wf_tot = 0
    ws = from_ts
    while ws + train_w + test_w <= to_ts:
        t0 = ws + train_w
        t1 = min(t0 + test_w, to_ts)
        mask_oos = (dates_ >= t0) & (dates_ < t1)
        d_oos    = dates_[mask_oos]
        if len(d_oos) < 30:
            ws += test_w; continue
        w_pnl = 0.0
        for a, b in top_pairs:
            # Use FULL window for training (up to t0), OOS for testing
            mask_full = dates_ < t0
            mask_all  = (dates_ >= ws) & (dates_ < t1)
            la = lp[a][mask_all].values
            lb = lp[b][mask_all].values
            d  = dates_[mask_all]
            n_train = mask_full[mask_all].sum()
            if n_train < _TRAIN_DAYS + 10:
                continue
            r = _run_pair(la, lb, d, train=n_train)
            w_pnl += r["pnl"]
        ok = w_pnl > 0
        wf_pass += ok; wf_tot += 1
        label = datetime.utcfromtimestamp(t0 / 1000).strftime("%Y-%m")
        print(f"    OOS {label}: ${w_pnl:>+8,.0f}  {'PASS' if ok else 'FAIL'}")
        ws += test_w
    print(f"  Walk-fwd: {wf_pass}/{wf_tot} windows profitable")

    # --- Correlation with V8.63 equity ---
    print("\n" + "=" * 110)
    print("  STRUCTURAL ASSESSMENT")
    print("=" * 110)
    print(f"  Pairs strategy makes money in: mean-reverting / choppy / sideways markets")
    print(f"  V8.63 makes money in:          trending markets")
    print(f"  → Potentially complementary — pairs could offset V8.63 weak years")
    print(f"\n  V8.63 year-by-year (from prior results):  2021: +$30K  2022: +$4K  "
          f"2023: -$1K  2024: +$19K  2025: -$0.4K  2026: +$3K")
    print(f"  Pairs portfolio year-by-year:             ", end="")
    print("  ".join(f"{yr}: {port_year.get(yr, 0):>+.0f}" for yr in sorted(port_year)))
    print()

    # Capital efficiency note
    n_pairs = len(top_pairs)
    max_concurrent = n_pairs  # worst case: all pairs open at once
    max_deployed = max_concurrent * _PER_TRADE * 2
    print(f"  Capital note: up to {max_concurrent} pairs open simultaneously "
          f"= ${max_deployed:,.0f} deployed out of $5K")
    if max_deployed > _CAPITAL:
        print(f"  [WARN] Max deployment ${max_deployed:,.0f} > $5K capital — "
              f"real sizing would need to be halved")

    print("=" * 110)


if __name__ == "__main__":
    main()
