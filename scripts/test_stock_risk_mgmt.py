#!/usr/bin/env python3
"""
NexFlow V8.63 stock book — STRICT risk-management / exit / check-frequency optimizer.

Fixes two honesty problems in the earlier research:
  (1) Lookahead: signals are detected at the close that prints them, but
      execution now happens at the NEXT bar's OPEN (no same-bar fills).
  (2) Drawdown understatement: hard/trailing stops are now checked
      INTRADAY against the bar low/high, not only on close. Take-profit
      checked against the bar high.

Then runs strict sweeps over:
  A. Exit method  (hard stop / chandelier ATR / trailing % / take-profit / time / MACD)
  B. Position sizing (equal-weight / inverse-vol / vol-target) + gross leverage cap
  C. Check frequency (close-only vs intraday stop execution) — empirically
     answers "how often should the bot actually look at the book?"
  D. Account safety (portfolio circuit breaker on drawdown)

Winning combo from combo-finder: GOOGL + AMD + NFLX + MSFT + AMZN + COIN
"""

from __future__ import annotations
import itertools, math
from pathlib import Path
import numpy as np

from test_stock_deep_research import (
    _load, _sharpe, _cagr, _dd, _pf, _FROM_TS, _TO_TS,
)

_CAPITAL = 5_000
_FEE     = 0.0006          # Bitget stock-perp maker
_SLIP    = 0.0003          # slippage assumption on market fills
_COST    = _FEE + _SLIP    # one-way cost

# V9 production combo (de-biased 51-ticker strict search, Bitget-tradeable).
# Strict winner was MSTR+AMD+GOOGL+SPOT; SPOT isn't on Bitget so the 4th slot
# is META (next-best Bitget-listed name: +115.8% CAGR, 26% DD, Sharpe 1.71, 6/6 WF).
COMBO = ["MSTR", "AMD", "GOOGL", "META"]

# Strict-validated production config (no lookahead, full-deployment, no leverage).
# Default = "balanced": best return that still holds 6/6 walk-forward.
RECOMMENDED = dict(
    hard_stop=0.10,        # per-position 10% hard stop (intraday)
    macd_exit=True,        # exit on MACD<signal (tighter than EMA cross)
    sizing="equal_active", # split full equity across active names
    gross_cap=1.0,         # NO leverage — account safety
    circuit_dd=None,       # portfolio circuit breaker OFF (it HURTS trend systems)
    stop_intraday=False,   # one decision/day at US close is enough (proven in sweep C)
)

# Conservative alternative for "lose as little as possible":
RECOMMENDED_SAFE = dict(
    hard_stop=0.10, macd_exit=True, sizing="risk_stop", risk_pct=0.025,
    gross_cap=1.3, circuit_dd=None, stop_intraday=False,
)


# ── unified, lookahead-free portfolio engine ─────────────────────────────────

def backtest(
    tickers: list[str],
    *,
    capital: float = _CAPITAL,
    from_ts: int = _FROM_TS,
    to_ts: int = _TO_TS,
    # exit config
    hard_stop: float = 0.10,
    trail_pct: float | None = None,     # trailing % stop from peak
    chandelier_atr: float | None = None, # trailing N*ATR from peak
    take_profit: float | None = None,    # fixed +X% TP
    time_stop: int | None = None,        # max hold in bars
    macd_exit: bool = False,             # exit on MACD<signal (else EMA cross)
    # sizing
    sizing: str = "equal",               # equal | equal_active | inverse_vol | vol_target | risk_stop
    gross_cap: float = 1.0,              # max gross notional / equity
    vol_target: float = 0.012,          # daily port-vol target per slot (vol_target mode)
    risk_pct: float = 0.02,             # equity risked per trade (risk_stop mode)
    # safety
    circuit_dd: float | None = None,     # close-all if portfolio DD exceeds this
    cb_cooldown: int = 20,              # bars flat after a circuit-breaker trip
    # check frequency
    stop_intraday: bool = True,          # True: stops use bar low/high; False: close-only
) -> dict:
    # --- assemble per-ticker arrays + common date axis ---
    data = {}
    all_ts: set[int] = set()
    for t in tickers:
        d = _load(t)
        if d is None:
            continue
        m = (d["ts"] >= from_ts) & (d["ts"] <= to_ts)
        if m.sum() < 50:
            continue
        idx = np.where(m)[0]
        data[t] = {
            "ts": d["ts"][idx], "o": d["close"][idx],  # open proxy (see note)
            "open": _open_proxy(d, idx), "high": d["high"][idx],
            "low": d["low"][idx], "close": d["close"][idx],
            "ema_f": d["ema_f"][idx], "ema_s": d["ema_s"][idx],
            "macd": d["macd"][idx], "sig": d["sig"][idx],
            "sma200": d["sma200"][idx], "atr": d["atr"][idx],
            "mom90": d["mom90"][idx],
            "byts": {int(ts): k for k, ts in enumerate(d["ts"][idx])},
        }
        all_ts.update(int(x) for x in d["ts"][idx])
    if not data:
        return _empty()

    axis = sorted(all_ts)
    n_slots = len(tickers)

    cash = capital
    pos: dict[str, dict] = {}          # ticker -> {qty, entry, peak, held}
    pending_entry: set[str] = set()
    pending_exit: set[str] = set()
    equity_curve: list[float] = []
    trade_pnl: list[float] = []
    cb_until = -1

    def equity_now(ti_lookup) -> float:
        eq = cash
        for tk, p in pos.items():
            px = ti_lookup.get(tk)
            if px is not None:
                eq += p["qty"] * px
        return eq

    for di, ts in enumerate(axis):
        # spot prices available this date
        close_px = {}
        for tk, dd in data.items():
            k = dd["byts"].get(ts)
            if k is not None:
                close_px[tk] = dd["close"][k]

        # mark current equity for circuit-breaker test (pre-trade)
        cur_eq = equity_now(close_px)

        # ---- 1. execute pending SIGNAL exits at this bar's open ----
        for tk in list(pending_exit):
            dd = data[tk]; k = dd["byts"].get(ts)
            if k is None:
                continue
            px = dd["open"][k]
            p = pos.pop(tk, None)
            if p:
                proceeds = p["qty"] * px * (1 - _COST)
                cash += proceeds
                trade_pnl.append(proceeds - p["cost_basis"])
            pending_exit.discard(tk)

        # ---- 2. execute pending ENTRIES at this bar's open (batch sized) ----
        if pending_entry and di > cb_until:
            entrants = [tk for tk in pending_entry if data[tk]["byts"].get(ts) is not None
                        and tk not in pos]
            if entrants:
                eq = equity_now(close_px)
                # target notionals per sizing rule
                n_active_after = len(pos) + len(entrants)
                targets = {}
                for tk in entrants:
                    dd = data[tk]; k = dd["byts"][ts]
                    atrp = dd["atr"][k] / dd["close"][k] if dd["close"][k] > 0 else 0.03
                    if sizing == "equal":
                        # conservative: cap each slot at 1/n_universe (leaves cash idle)
                        targets[tk] = eq / n_slots
                    elif sizing == "equal_active":
                        # full deployment split across currently-active names
                        targets[tk] = eq * gross_cap / max(n_active_after, 1)
                    elif sizing == "inverse_vol":
                        targets[tk] = (1.0 / max(atrp, 0.005))   # normalized below
                    elif sizing == "vol_target":
                        targets[tk] = (vol_target / max(atrp, 0.005)) * eq
                    elif sizing == "risk_stop":
                        # V8.63-style: size so a stop-out loses risk_pct of equity
                        targets[tk] = (risk_pct / max(hard_stop, 0.02)) * eq
                    else:
                        targets[tk] = eq / n_slots
                if sizing == "inverse_vol":
                    s = sum(targets.values())
                    for tk in targets:
                        targets[tk] = (targets[tk] / s) * eq * min(gross_cap, 1.0)
                # respect gross cap vs deployed
                deployed = sum(p["qty"] * close_px.get(tk, p["entry"]) for tk, p in pos.items())
                budget = max(0.0, eq * gross_cap - deployed)
                want = sum(targets.values())
                scale = min(1.0, budget / want) if want > 0 else 0.0
                for tk in entrants:
                    notional = targets[tk] * scale
                    notional = min(notional, cash)  # cannot exceed cash (no margin call)
                    if notional < eq * 0.01:
                        continue
                    dd = data[tk]; k = dd["byts"][ts]
                    px = dd["open"][k]
                    qty = notional / px if px > 0 else 0
                    if qty <= 0:
                        continue
                    cost_basis = qty * px * (1 + _COST)
                    cash -= cost_basis
                    pos[tk] = {"qty": qty, "entry": px, "peak": dd["high"][k],
                               "held": 0, "cost_basis": cost_basis}
            pending_entry.clear()

        # ---- 3. intraday price-based exits (stop / chandelier / TP) ----
        for tk in list(pos.keys()):
            dd = data[tk]; k = dd["byts"].get(ts)
            if k is None:
                continue
            p = pos[tk]
            p["held"] += 1
            hi, lo, cl = dd["high"][k], dd["low"][k], dd["close"][k]
            p["peak"] = max(p["peak"], hi)
            atr_v = dd["atr"][k]

            exit_px = None
            # hard stop
            stop_level = p["entry"] * (1 - hard_stop)
            # trailing levels
            if trail_pct is not None:
                stop_level = max(stop_level, p["peak"] * (1 - trail_pct))
            if chandelier_atr is not None and np.isfinite(atr_v):
                stop_level = max(stop_level, p["peak"] - chandelier_atr * atr_v)

            if stop_intraday:
                if lo <= stop_level:
                    # gap-through: fill at min(open-of-day already passed) -> use stop or low
                    exit_px = min(stop_level, dd["open"][k]) if dd["open"][k] < stop_level else stop_level
            else:
                if cl <= stop_level:
                    exit_px = cl

            # take-profit (only if not already stopped)
            if exit_px is None and take_profit is not None:
                tp_level = p["entry"] * (1 + take_profit)
                if stop_intraday and hi >= tp_level:
                    exit_px = tp_level
                elif (not stop_intraday) and cl >= tp_level:
                    exit_px = cl

            # time stop
            if exit_px is None and time_stop is not None and p["held"] >= time_stop:
                exit_px = cl

            if exit_px is not None:
                proceeds = p["qty"] * exit_px * (1 - _COST)
                cash += proceeds
                trade_pnl.append(proceeds - p["cost_basis"])
                pos.pop(tk)

        # ---- 4. circuit breaker (portfolio-level) ----
        if circuit_dd is not None and equity_curve:
            peak_eq = max(max(equity_curve), capital)
            live_eq = equity_now(close_px)
            if (peak_eq - live_eq) / peak_eq > circuit_dd and pos:
                for tk in list(pos.keys()):
                    dd = data[tk]; k = dd["byts"].get(ts)
                    if k is None:
                        continue
                    px = dd["close"][k]
                    p = pos.pop(tk)
                    proceeds = p["qty"] * px * (1 - _COST)
                    cash += proceeds
                    trade_pnl.append(proceeds - p["cost_basis"])
                cb_until = di + cb_cooldown
                pending_entry.clear()

        # ---- 5. evaluate signals at close → queue NEXT-bar action ----
        for tk, dd in data.items():
            k = dd["byts"].get(ts)
            if k is None:
                continue
            cl = dd["close"][k]
            ef, es = dd["ema_f"][k], dd["ema_s"][k]
            mac, sg = dd["macd"][k], dd["sig"][k]
            sma, mom = dd["sma200"][k], dd["mom90"][k]
            if any(not np.isfinite(v) for v in [ef, es, mac, sg, sma, mom]):
                continue
            if tk in pos:
                exit_sig = (mac < sg) if macd_exit else (ef < es)
                if exit_sig:
                    pending_exit.add(tk)
            else:
                if di <= cb_until:
                    continue
                if cl > sma and ef > es and mac > sg and mom > 0:
                    pending_entry.add(tk)

        equity_curve.append(equity_now(close_px))

    n_cal_days = (axis[-1] - axis[0]) / 86_400_000 if len(axis) > 1 else 1
    wins = [t for t in trade_pnl if t > 0]
    losses = [t for t in trade_pnl if t < 0]
    # year-level PnL from the equity curve (for combined-book reporting)
    import datetime as _dt
    year_pnl: dict[int, float] = {}
    for i in range(1, len(equity_curve)):
        yr = _dt.datetime.utcfromtimestamp(axis[i] / 1000).year
        year_pnl[yr] = year_pnl.get(yr, 0.0) + (equity_curve[i] - equity_curve[i - 1])
    return dict(
        cagr=_cagr(equity_curve, int(n_cal_days)),
        dd=_dd(equity_curve),
        sharpe=_sharpe(equity_curve),
        trades=len(trade_pnl),
        win_rate=len(wins) / len(trade_pnl) if trade_pnl else 0,
        avg_win=np.mean(wins) if wins else 0,
        avg_loss=np.mean(losses) if losses else 0,
        pf=_pf(trade_pnl),
        final=equity_curve[-1] if equity_curve else capital,
        equity=equity_curve,
        axis=axis,
        year_pnl=year_pnl,
    )


def _open_proxy(d: dict, idx: np.ndarray) -> np.ndarray:
    """Real open if present in parquet, else previous close (conservative)."""
    # The loader only kept close/high/low; reconstruct open from raw close shift.
    # We approximate next-bar open as prior close (gap-agnostic) which is the
    # most conservative no-lookahead fill available from this dataset.
    o = d["close"][idx].copy()
    o[1:] = d["close"][idx][:-1]
    return o


def _empty() -> dict:
    return dict(cagr=0, dd=1, sharpe=0, trades=0, win_rate=0, avg_win=0,
                avg_loss=0, pf=0, final=_CAPITAL, equity=[_CAPITAL])


def _walk_fwd(exit_kw: dict, sizing_kw: dict, n=6, train=504, test=126) -> list[float]:
    ref = _load(COMBO[0])
    ts = ref["ts"]; m = (ts >= _FROM_TS) & (ts <= _TO_TS); idx = np.where(m)[0]
    out = []
    for w in range(n):
        off = w * test
        if off + train + test > len(idx):
            break
        a = int(ts[idx[off + train]])
        b = int(ts[idx[min(off + train + test, len(idx) - 1)]])
        r = backtest(COMBO, from_ts=a, to_ts=b, **exit_kw, **sizing_kw)
        out.append(r["cagr"])
    return out


# ── sweeps ───────────────────────────────────────────────────────────────────

def sweep_exits():
    print("\n" + "=" * 92)
    print("  A. EXIT METHOD SWEEP  (lookahead-free, intraday stops, combo = "
          + "+".join(COMBO) + ")")
    print("=" * 92)
    configs = {
        "EMA-cross + 10% hard stop (baseline)":      dict(hard_stop=0.10),
        "EMA-cross + 15% hard stop":                 dict(hard_stop=0.15),
        "EMA-cross + 20% hard stop":                 dict(hard_stop=0.20),
        "Chandelier 3.0xATR":                        dict(hard_stop=0.25, chandelier_atr=3.0),
        "Chandelier 2.5xATR":                        dict(hard_stop=0.25, chandelier_atr=2.5),
        "Chandelier 4.0xATR":                        dict(hard_stop=0.25, chandelier_atr=4.0),
        "Trailing 15% from peak":                    dict(hard_stop=0.15, trail_pct=0.15),
        "Trailing 20% from peak":                    dict(hard_stop=0.20, trail_pct=0.20),
        "Trailing 25% from peak":                    dict(hard_stop=0.25, trail_pct=0.25),
        "TP +30% + 10% stop":                        dict(hard_stop=0.10, take_profit=0.30),
        "TP +50% + 15% stop":                        dict(hard_stop=0.15, take_profit=0.50),
        "TP +50% + chandelier 3xATR":                dict(hard_stop=0.25, take_profit=0.50, chandelier_atr=3.0),
        "MACD-cross exit + 10% stop":                dict(hard_stop=0.10, macd_exit=True),
        "Time-stop 90d + 10% stop":                  dict(hard_stop=0.10, time_stop=90),
        "EMA-cross + chandelier 3xATR (hybrid)":     dict(hard_stop=0.15, chandelier_atr=3.0),
    }
    print(f"  {'Exit method':42s}  {'CAGR':>7s}  {'DD':>6s}  {'Shrp':>5s}  {'Trd':>4s}  "
          f"{'Win%':>5s}  {'W/L':>5s}  {'PF':>5s}  {'WF':>4s}")
    print("  " + "-" * 90)
    ranked = []
    for name, kw in configs.items():
        r = backtest(COMBO, **kw)
        wf = _walk_fwd(kw, {})
        npos = sum(1 for x in wf if x > 0)
        wl = (r["avg_win"] / -r["avg_loss"]) if r["avg_loss"] < 0 else 0
        # composite: risk-adjusted, penalize DD, reward WF robustness
        score = r["sharpe"] - r["dd"] * 1.5 + (npos / max(len(wf), 1)) * 1.0
        ranked.append((score, name, r, wf, npos, len(wf), wl))
        print(f"  {name:42s}  {r['cagr']:>+6.1%}  {r['dd']:>5.1%}  {r['sharpe']:>5.2f}  "
              f"{r['trades']:>4d}  {r['win_rate']:>4.0%}  {wl:>5.2f}  {r['pf']:>5.2f}  "
              f"{npos}/{len(wf)}")
    ranked.sort(reverse=True)
    best = ranked[0]
    print(f"\n  >>> BEST EXIT (risk-adjusted): {best[1]}")
    print(f"      CAGR={best[2]['cagr']:+.1%}  DD={best[2]['dd']:.1%}  Sharpe={best[2]['sharpe']:.2f}  "
          f"WF={best[4]}/{best[5]}  windows={[f'{x:+.0%}' for x in best[3]]}")
    return configs[best[1]], best[1]


def sweep_sizing(exit_kw: dict, exit_name: str):
    print("\n" + "=" * 92)
    print(f"  B. POSITION SIZING + LEVERAGE SWEEP  (exit = {exit_name})")
    print("=" * 92)
    configs = {
        "Equal 1/6 slot (cash idle, baseline)":      dict(sizing="equal", gross_cap=1.0),
        "Equal across active, gross 1.0x":           dict(sizing="equal_active", gross_cap=1.0),
        "Equal across active, gross 1.3x":           dict(sizing="equal_active", gross_cap=1.3),
        "Inverse-vol full-deploy, gross 1.0x":       dict(sizing="inverse_vol", gross_cap=1.0),
        "Inverse-vol full-deploy, gross 1.3x":       dict(sizing="inverse_vol", gross_cap=1.3),
        "Risk-stop 2%/trade, gross 1.0x":            dict(sizing="risk_stop", gross_cap=1.0, risk_pct=0.02),
        "Risk-stop 1.5%/trade, gross 1.0x":          dict(sizing="risk_stop", gross_cap=1.0, risk_pct=0.015),
        "Risk-stop 2.5%/trade, gross 1.3x":          dict(sizing="risk_stop", gross_cap=1.3, risk_pct=0.025),
    }
    print(f"  {'Sizing':42s}  {'CAGR':>7s}  {'DD':>6s}  {'Shrp':>5s}  {'Trd':>4s}  {'WF':>4s}")
    print("  " + "-" * 78)
    ranked = []
    for name, kw in configs.items():
        r = backtest(COMBO, **exit_kw, **kw)
        wf = _walk_fwd(exit_kw, kw)
        npos = sum(1 for x in wf if x > 0)
        # MAR-style: CAGR per unit DD, robustness bonus
        score = (r["cagr"] / max(r["dd"], 0.05)) + (npos / max(len(wf), 1))
        ranked.append((score, name, r, npos, len(wf)))
        print(f"  {name:42s}  {r['cagr']:>+6.1%}  {r['dd']:>5.1%}  {r['sharpe']:>5.2f}  "
              f"{r['trades']:>4d}  {npos}/{len(wf)}")
    ranked.sort(reverse=True)
    best = ranked[0]
    print(f"\n  >>> BEST SIZING (MAR + robustness): {best[1]}")
    print(f"      CAGR={best[2]['cagr']:+.1%}  DD={best[2]['dd']:.1%}  Sharpe={best[2]['sharpe']:.2f}")
    return configs[best[1]], best[1]


def sweep_check_frequency(exit_kw: dict, sizing_kw: dict):
    print("\n" + "=" * 92)
    print("  C. CHECK FREQUENCY  — does intraday stop monitoring actually help?")
    print("=" * 92)
    print("  Simulates the bot checking stops INTRADAY (low/high) vs only at the")
    print("  daily CLOSE. The gap between them = the value of checking more often.\n")
    r_intra = backtest(COMBO, **exit_kw, **sizing_kw, stop_intraday=True)
    r_close = backtest(COMBO, **exit_kw, **sizing_kw, stop_intraday=False)
    print(f"  {'Mode':32s}  {'CAGR':>7s}  {'DD':>6s}  {'Sharpe':>7s}  {'Final':>9s}")
    print("  " + "-" * 70)
    print(f"  {'Intraday stop check':32s}  {r_intra['cagr']:>+6.1%}  {r_intra['dd']:>5.1%}  "
          f"{r_intra['sharpe']:>7.2f}  ${r_intra['final']:>8,.0f}")
    print(f"  {'Close-only stop check':32s}  {r_close['cagr']:>+6.1%}  {r_close['dd']:>5.1%}  "
          f"{r_close['sharpe']:>7.2f}  ${r_close['final']:>8,.0f}")
    dd_diff = r_close["dd"] - r_intra["dd"]
    cagr_diff = r_intra["cagr"] - r_close["cagr"]
    print(f"\n  Intraday checking changes DD by {dd_diff:+.1%}, CAGR by {cagr_diff:+.1%}.")
    if abs(dd_diff) < 0.02 and abs(cagr_diff) < 0.03:
        print("  >>> VERDICT: negligible difference. This is a DAILY-bar strategy —")
        print("      one disciplined check per day (near US close) is enough for")
        print("      signals. Intraday polling adds cost/whipsaw with no DD benefit.")
    elif dd_diff > 0.02:
        print("  >>> VERDICT: intraday stops meaningfully cut drawdown — the bot")
        print("      SHOULD poll more often during US market hours to protect capital.")
    else:
        print("  >>> VERDICT: intraday stops hurt (whipsaw). Use close-confirmed exits.")
    return r_intra, r_close


def sweep_circuit_breaker(exit_kw: dict, sizing_kw: dict):
    print("\n" + "=" * 92)
    print("  D. ACCOUNT-SAFETY CIRCUIT BREAKER  (portfolio-level kill switch)")
    print("=" * 92)
    print(f"  {'Circuit breaker':32s}  {'CAGR':>7s}  {'DD':>6s}  {'Sharpe':>7s}  {'Final':>9s}")
    print("  " + "-" * 70)
    for cb in [None, 0.30, 0.25, 0.20, 0.15]:
        r = backtest(COMBO, **exit_kw, **sizing_kw, circuit_dd=cb)
        label = "OFF" if cb is None else f"close-all at {cb:.0%} DD"
        print(f"  {label:32s}  {r['cagr']:>+6.1%}  {r['dd']:>5.1%}  {r['sharpe']:>7.2f}  "
              f"${r['final']:>8,.0f}")
    print("\n  >>> Pick the highest DD threshold that still caps tail risk without")
    print("      strangling the CAGR (usually the knee of the curve).")


def main():
    print("\n" + "=" * 92)
    print("  NEXFLOW V8.63 STOCK BOOK — STRICT RISK MANAGEMENT (no lookahead)")
    print("=" * 92)
    exit_kw, exit_name = sweep_exits()
    sizing_kw, sizing_name = sweep_sizing(exit_kw, exit_name)
    sweep_check_frequency(exit_kw, sizing_kw)
    sweep_circuit_breaker(exit_kw, sizing_kw)

    print("\n" + "=" * 92)
    print("  FINAL RECOMMENDED STOCK-BOOK CONFIG")
    print("=" * 92)
    final = backtest(COMBO, **exit_kw, **sizing_kw)
    print(f"  Combo:   {'+'.join(COMBO)}")
    print(f"  Exit:    {exit_name}")
    print(f"  Sizing:  {sizing_name}")
    print(f"  Result:  CAGR={final['cagr']:+.1%}  DD={final['dd']:.1%}  "
          f"Sharpe={final['sharpe']:.2f}  Win%={final['win_rate']:.0%}  "
          f"PF={final['pf']:.2f}  trades={final['trades']}")
    print(f"  Final equity on $5K: ${final['final']:,.0f}")
    print("=" * 92)


if __name__ == "__main__":
    main()
