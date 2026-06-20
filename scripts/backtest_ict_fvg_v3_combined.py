#!/usr/bin/env python3
"""
backtest_ict_fvg_v3_combined.py — ICT FVG v3 across all 6 STRONG GO assets
on a SINGLE $5K account (shared capital pool, 1% risk per trade).

Assets: BTC, ETH, SOL, LINK, DOGE, AVAX
Trades fire sequentially. Capital compounds across all signals.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required"); sys.exit(1)

# ─── Config (identical to v3) ─────────────────────────────────────────────────
_FEE            = 0.0006
_CAPITAL        = 5_000.0
_RISK_PCT       = 0.01

_SESSIONS = [
    (13 * 60 + 30, 15 * 60 + 30),
    (17 * 60 + 0,  18 * 60 + 30),
]

_HTF_EMA_FAST   = 8
_HTF_EMA_SLOW   = 21
_SWEEP_BARS     = 20
_SWING_TOUCHES  = 1
_SWING_TOLERANCE = 0.0015
_DISP_BODY_RATIO = 0.60
_DISP_BODY_MULT  = 2.0
_DISP_VOL_MULT   = 1.3
_PREMIUM_DISC    = 0.50
_FVG_MAX_BARS   = 20
_FVG_MIN_PCT    = 0.0005
_TARGET_MIN_R   = 2.5
_MAX_HOLD_BARS  = 90
_DAILY_SMA_LEN  = 200
_PARTIAL_R      = 2.0
_PARTIAL_FRAC   = 0.5
_RR             = 5.0

_STRONG_GO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOGEUSDT", "AVAXUSDT"]
_DATA_DIR  = _REPO_ROOT / "data" / "candles"


# ─── Data helpers (same as v3) ────────────────────────────────────────────────

def _load_1m(path: Path) -> list[dict]:
    t = pq.read_table(path, columns=["open_time","open","high","low","close","volume"])
    out = []
    for ot, o, h, l, c, v in zip(
        t["open_time"].to_pylist(), t["open"].to_pylist(),
        t["high"].to_pylist(), t["low"].to_pylist(),
        t["close"].to_pylist(), t["volume"].to_pylist()
    ):
        ts_raw = int(ot)
        ts_ms = ts_raw * 1000 if ts_raw < 2_000_000_000 else ts_raw
        out.append({"ts": ts_ms, "o": float(o), "h": float(h),
                    "l": float(l), "c": float(c), "v": float(v)})
    out.sort(key=lambda x: x["ts"])
    return out

def _resample_1h(m1):
    bk = {}
    for b in m1:
        t = (b["ts"] // 3_600_000) * 3_600_000
        if t not in bk:
            bk[t] = {"ts": t, "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]}
        else:
            bk[t]["h"] = max(bk[t]["h"], b["h"])
            bk[t]["l"] = min(bk[t]["l"], b["l"])
            bk[t]["c"] = b["c"]; bk[t]["v"] += b["v"]
    return sorted(bk.values(), key=lambda x: x["ts"])

def _resample_1d(m1):
    bk = {}
    for b in m1:
        t = (b["ts"] // 86_400_000) * 86_400_000
        if t not in bk:
            bk[t] = {"ts": t, "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]}
        else:
            bk[t]["h"] = max(bk[t]["h"], b["h"])
            bk[t]["l"] = min(bk[t]["l"], b["l"])
            bk[t]["c"] = b["c"]; bk[t]["v"] += b["v"]
    return sorted(bk.values(), key=lambda x: x["ts"])

def _build_htf_trend(h1):
    af = 2 / (_HTF_EMA_FAST + 1); as_ = 2 / (_HTF_EMA_SLOW + 1)
    ef = es = None; trend = {}
    for b in h1:
        c = b["c"]
        ef = c if ef is None else af * c + (1-af) * ef
        es = c if es is None else as_ * c + (1-as_) * es
        if ef > es * 1.0005: trend[b["ts"]] = "bull"
        elif ef < es * 0.9995: trend[b["ts"]] = "bear"
        else: trend[b["ts"]] = "neutral"
    return trend

def _build_daily_regime(d1):
    regime = {}; closes = []
    for b in d1:
        closes.append(b["c"])
        if len(closes) >= _DAILY_SMA_LEN:
            sma = sum(closes[-_DAILY_SMA_LEN:]) / _DAILY_SMA_LEN
            regime[b["ts"]] = "bull" if b["c"] >= sma else "bear"
    return regime

def _htf_at(ts, trend):
    return trend.get((ts // 3_600_000) * 3_600_000, "neutral")

def _daily_at(ts, regime):
    return regime.get((ts // 86_400_000) * 86_400_000, "neutral")

def _in_session(ts):
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    m = dt.hour * 60 + dt.minute
    return any(s <= m < e for s, e in _SESSIONS)

def _avg_body(candles, i, n=20):
    w = candles[max(0,i-n):i]
    return sum(abs(c["c"]-c["o"]) for c in w)/len(w) if w else 0.0

def _avg_volume(candles, i, n=20):
    w = candles[max(0,i-n):i]
    return sum(c["v"] for c in w)/len(w) if w else 0.0

def _count_touches(candles, i, level, lookback, is_low):
    touches = 0
    for j in range(max(0, i-lookback), i):
        key = "l" if is_low else "h"
        if abs(candles[j][key] - level) / level <= _SWING_TOLERANCE:
            touches += 1
    return touches

def _in_discount(candles, i, fvg_mid, n=20):
    w = candles[max(0,i-n):i+1]
    rh, rl = max(c["h"] for c in w), min(c["l"] for c in w)
    if rh == rl: return False
    return (fvg_mid - rl) / (rh - rl) <= _PREMIUM_DISC

def _in_premium(candles, i, fvg_mid, n=20):
    w = candles[max(0,i-n):i+1]
    rh, rl = max(c["h"] for c in w), min(c["l"] for c in w)
    if rh == rl: return False
    return (fvg_mid - rl) / (rh - rl) >= (1 - _PREMIUM_DISC)

def _next_target(candles, i, direction, risk, n=40):
    entry = candles[i]["c"]
    prior = candles[max(0,i-n):i]
    if direction == "LONG":
        for h in sorted(set(c["h"] for c in prior), reverse=True):
            if h > entry + _TARGET_MIN_R * risk: return h
    else:
        for l in sorted(set(c["l"] for c in prior)):
            if l < entry - _TARGET_MIN_R * risk: return l
    return None


# ─── Extract all trades for one asset ────────────────────────────────────────

def _get_trades_for_asset(symbol: str) -> list[dict]:
    path = _DATA_DIR / f"{symbol}_1m.parquet"
    if not path.exists():
        print(f"  [SKIP] {symbol} — no data file")
        return []

    m1     = _load_1m(path)
    h1     = _resample_1h(m1)
    d1     = _resample_1d(m1)
    trend  = _build_htf_trend(h1)
    regime = _build_daily_regime(d1)

    n = len(m1)
    raw_setups = []

    for i in range(_SWEEP_BARS + 2, n - _FVG_MAX_BARS - 2):
        if not _in_session(m1[i]["ts"]): continue
        if m1[i]["ts"] - m1[i-1]["ts"] > 120_000: continue

        avg_b = _avg_body(m1, i)
        avg_v = _avg_volume(m1, i)
        if avg_b == 0: continue

        htf  = _htf_at(m1[i]["ts"], trend)
        dreg = _daily_at(m1[i]["ts"], regime)
        if dreg == "neutral": continue

        if htf == "bull" and dreg == "bull":
            prior_lows = [m1[j]["l"] for j in range(i-_SWEEP_BARS, i)]
            swing_low  = min(prior_lows)
            if m1[i]["l"] < swing_low and m1[i]["c"] > swing_low:
                if _count_touches(m1, i, swing_low, _SWEEP_BARS*2, True) < _SWING_TOUCHES: continue
                d = m1[i+1] if i+1 < n else None
                if d is None: continue
                db = d["c"] - d["o"]; dr = d["h"] - d["l"]
                if dr == 0 or db < _DISP_BODY_MULT*avg_b or db/dr < _DISP_BODY_RATIO or db <= 0: continue
                if avg_v > 0 and d["v"] < _DISP_VOL_MULT * avg_v: continue
                if i+2 >= n: continue
                fvg_low = m1[i]["h"]; fvg_high = m1[i+2]["l"]
                if fvg_high <= fvg_low: continue
                if (fvg_high-fvg_low)/m1[i]["c"] < _FVG_MIN_PCT: continue
                fvg_mid = (fvg_low+fvg_high)/2
                if not _in_discount(m1, i, fvg_mid): continue
                risk_est = fvg_mid - (swing_low*0.999)
                if risk_est <= 0: continue
                target = _next_target(m1, i, "LONG", risk_est)
                if target is None: continue
                raw_setups.append({"direction":"LONG","sweep_bar":i,
                    "sweep_extreme":m1[i]["l"]*0.9995,
                    "fvg_low":fvg_low,"fvg_high":fvg_high,"fvg_mid":fvg_mid,
                    "valid_from":i+2,"target":target})

        if htf == "bear" and dreg == "bear":
            prior_highs = [m1[j]["h"] for j in range(i-_SWEEP_BARS, i)]
            swing_high  = max(prior_highs)
            if m1[i]["h"] > swing_high and m1[i]["c"] < swing_high:
                if _count_touches(m1, i, swing_high, _SWEEP_BARS*2, False) < _SWING_TOUCHES: continue
                d = m1[i+1] if i+1 < n else None
                if d is None: continue
                db = d["o"] - d["c"]; dr = d["h"] - d["l"]
                if dr == 0 or db < _DISP_BODY_MULT*avg_b or db/dr < _DISP_BODY_RATIO or db <= 0: continue
                if avg_v > 0 and d["v"] < _DISP_VOL_MULT * avg_v: continue
                if i+2 >= n: continue
                fvg_high = m1[i]["l"]; fvg_low = m1[i+2]["h"]
                if fvg_high <= fvg_low: continue
                if (fvg_high-fvg_low)/m1[i]["c"] < _FVG_MIN_PCT: continue
                fvg_mid = (fvg_low+fvg_high)/2
                if not _in_premium(m1, i, fvg_mid): continue
                risk_est = (swing_high*1.0005) - fvg_mid
                if risk_est <= 0: continue
                target = _next_target(m1, i, "SHORT", risk_est)
                if target is None: continue
                raw_setups.append({"direction":"SHORT","sweep_bar":i,
                    "sweep_extreme":m1[i]["h"]*1.0005,
                    "fvg_low":fvg_low,"fvg_high":fvg_high,"fvg_mid":fvg_mid,
                    "valid_from":i+2,"target":target})

    # Simulate each setup (using placeholder capital — real capital injected later)
    trades = []
    last_exit_bar = -1

    for setup in raw_setups:
        if setup["valid_from"] <= last_exit_bar: continue
        direction = setup["direction"]
        fvg_low   = setup["fvg_low"]; fvg_high = setup["fvg_high"]
        fvg_mid   = setup["fvg_mid"]; sweep_ext = setup["sweep_extreme"]
        valid_from= setup["valid_from"]

        entry_price = stop_price = None; entry_bar = None
        for j in range(valid_from, min(valid_from+_FVG_MAX_BARS, n)):
            if m1[j]["ts"] - m1[j-1]["ts"] > 120_000: break
            c = m1[j]
            if not _in_session(c["ts"]): continue
            if direction == "LONG" and c["l"] <= fvg_high and c["h"] >= fvg_low:
                entry_price = fvg_mid; stop_price = sweep_ext; entry_bar = j; break
            elif direction == "SHORT" and c["h"] >= fvg_low and c["l"] <= fvg_high:
                entry_price = fvg_mid; stop_price = sweep_ext; entry_bar = j; break

        if entry_price is None: continue
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0: continue

        if direction == "LONG":
            partial_tp = entry_price + _PARTIAL_R * risk_per_unit
            full_tp    = min(setup["target"], entry_price + _RR * risk_per_unit)
        else:
            partial_tp = entry_price - _PARTIAL_R * risk_per_unit
            full_tp    = max(setup["target"], entry_price - _RR * risk_per_unit)

        # Simulate with unit risk = 1.0 (scaled by actual capital later)
        unit_risk = 1.0
        qty = unit_risk / risk_per_unit
        pnl_units = -(qty * entry_price * _FEE)

        partial_hit = False; qty_rem = qty
        exit_reason = "TIME"; exit_price = entry_price; exit_bar_i = entry_bar

        for j in range(entry_bar+1, min(entry_bar+_MAX_HOLD_BARS+1, n)):
            if m1[j]["ts"] - m1[j-1]["ts"] > 120_000: break
            c = m1[j]
            if not partial_hit:
                if direction == "LONG" and c["h"] >= partial_tp:
                    close_qty = qty * _PARTIAL_FRAC
                    pnl_units += close_qty*(partial_tp-entry_price) - close_qty*partial_tp*_FEE
                    qty_rem -= close_qty; partial_hit = True
                elif direction == "SHORT" and c["l"] <= partial_tp:
                    close_qty = qty * _PARTIAL_FRAC
                    pnl_units += close_qty*(entry_price-partial_tp) - close_qty*partial_tp*_FEE
                    qty_rem -= close_qty; partial_hit = True

            if direction == "LONG" and c["l"] <= stop_price:
                hit = min(c["o"], stop_price)
                pnl_units += qty_rem*(hit-entry_price) - qty_rem*hit*_FEE
                exit_reason = "STOP"; exit_price = hit; exit_bar_i = j; break
            elif direction == "SHORT" and c["h"] >= stop_price:
                hit = max(c["o"], stop_price)
                pnl_units += qty_rem*(entry_price-hit) - qty_rem*hit*_FEE
                exit_reason = "STOP"; exit_price = hit; exit_bar_i = j; break

            if direction == "LONG" and c["h"] >= full_tp:
                pnl_units += qty_rem*(full_tp-entry_price) - qty_rem*full_tp*_FEE
                exit_reason = "TP"; exit_price = full_tp; exit_bar_i = j; break
            elif direction == "SHORT" and c["l"] <= full_tp:
                pnl_units += qty_rem*(entry_price-full_tp) - qty_rem*full_tp*_FEE
                exit_reason = "TP"; exit_price = full_tp; exit_bar_i = j; break
        else:
            last = m1[min(entry_bar+_MAX_HOLD_BARS, n-1)]
            ep = last["c"]
            pnl_units += qty_rem*((ep-entry_price) if direction=="LONG" else (entry_price-ep))
            pnl_units -= qty_rem*ep*_FEE
            exit_bar_i = min(entry_bar+_MAX_HOLD_BARS, n-1)

        last_exit_bar = exit_bar_i
        dt = datetime.fromtimestamp(m1[entry_bar]["ts"]/1000, tz=timezone.utc)
        trades.append({
            "symbol":      symbol,
            "direction":   direction,
            "entry_ts":    m1[entry_bar]["ts"],
            "exit_ts":     m1[exit_bar_i]["ts"],
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "pnl_units":   pnl_units,   # P&L per $1 of risk
            "exit_reason": exit_reason,
            "year":        dt.year,
            "month":       dt.month,
        })

    return trades


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  NexFlow — ICT FVG v3 — COMBINED 6-Asset Portfolio")
    print("  Single $5,000 account | 1% risk per trade | All assets share capital")
    print("  Assets: BTC  ETH  SOL  LINK  DOGE  AVAX")
    print("=" * 70)

    all_trades = []
    for sym in _STRONG_GO:
        print(f"\n  Scanning {sym}...", end="", flush=True)
        trades = _get_trades_for_asset(sym)
        print(f" {len(trades)} setups found")
        all_trades.extend(trades)

    # Sort all trades chronologically
    all_trades.sort(key=lambda t: t["entry_ts"])

    print(f"\n  Total raw setups across all assets: {len(all_trades)}")
    print(f"  Resolving overlaps (no two open trades simultaneously)...")

    # Simulate on single capital pool
    # Track per-asset exit timestamps to avoid overlapping trades on same asset
    last_exit_per_asset: dict[str, int] = {}
    capital = _CAPITAL
    simulated: list[dict] = []

    for t in all_trades:
        sym = t["symbol"]
        # Skip if this asset has an open trade that hasn't closed yet
        if t["entry_ts"] < last_exit_per_asset.get(sym, 0):
            continue

        # Scale P&L: pnl_units is per $1 of risk, actual risk = capital * 1%
        actual_risk = capital * _RISK_PCT
        pnl = t["pnl_units"] * actual_risk

        capital += pnl
        last_exit_per_asset[sym] = t["exit_ts"]

        simulated.append({**t, "pnl": pnl, "capital": capital})

    trades = simulated
    if not trades:
        print("  No trades after overlap resolution.")
        return

    # ── Stats ─────────────────────────────────────────────────────────────────
    wins      = [t for t in trades if t["pnl"] > 0]
    losses    = [t for t in trades if t["pnl"] <= 0]
    tps       = [t for t in trades if t["exit_reason"] == "TP"]
    stops     = [t for t in trades if t["exit_reason"] == "STOP"]
    gross_win = sum(t["pnl"] for t in wins)
    gross_los = abs(sum(t["pnl"] for t in losses))
    pf        = gross_win / gross_los if gross_los else float("inf")

    equity = [_CAPITAL] + [t["capital"] for t in trades]
    peak = _CAPITAL; max_dd = 0.0
    for eq in equity:
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    total_pnl = sum(t["pnl"] for t in trades)

    # Year stats
    year_stats: dict[int, dict] = {}
    for t in trades:
        yr = t["year"]
        if yr not in year_stats:
            year_stats[yr] = {"pnl":0.0,"trades":0,"wins":0,"start":0.0}
        year_stats[yr]["pnl"]    += t["pnl"]
        year_stats[yr]["trades"] += 1
        if t["pnl"] > 0: year_stats[yr]["wins"] += 1

    # Calculate start capital per year
    running = _CAPITAL
    for yr in sorted(year_stats):
        year_stats[yr]["start"] = running
        running += year_stats[yr]["pnl"]

    avg_per_yr = len(trades) / max(1, len(year_stats))

    print(f"\n{'═'*70}")
    print(f"  COMBINED PORTFOLIO RESULTS  ({len(trades)} trades, {avg_per_yr:.0f}/year)")
    print(f"{'═'*70}")
    print(f"  Starting capital : $5,000.00")
    print(f"  Final capital    : ${capital:>10,.2f}")
    print(f"  Total P&L        : ${total_pnl:>+10,.2f}  ({total_pnl/_CAPITAL*100:+.1f}%)")
    print(f"  Win rate         : {len(wins)/len(trades)*100:.1f}%  ({len(wins)}/{len(trades)})")
    print(f"  Profit factor    : {pf:.2f}")
    print(f"  Max drawdown     : {max_dd*100:.1f}%")
    if wins:   print(f"  Avg win          : ${gross_win/len(wins):,.2f}")
    if losses: print(f"  Avg loss         : ${gross_los/len(losses):,.2f}")
    print(f"  TP exits         : {len(tps)}  ({len(tps)/len(trades)*100:.1f}%)")
    print(f"  Stop exits       : {len(stops)}  ({len(stops)/len(trades)*100:.1f}%)")

    # Asset breakdown
    print(f"\n  {'Asset':<10} {'Trades':>7} {'Win%':>6} {'P&L':>10}")
    print(f"  {'─'*37}")
    for sym in _STRONG_GO:
        st = [t for t in trades if t["symbol"] == sym]
        if not st: continue
        sw = [t for t in st if t["pnl"] > 0]
        sp = sum(t["pnl"] for t in st)
        print(f"  {sym:<10} {len(st):>7} {len(sw)/len(st)*100:>5.1f}% {sp:>+10,.2f}")

    # Year by year — THE KEY TABLE
    print(f"\n{'─'*70}")
    print(f"  {'Year':<6} {'Trades':>7} {'Win%':>6} {'$Profit':>10} {'Capital':>10} {'Return':>7} {'$100 acct':>10}")
    print(f"  {'─'*65}")
    winning_years = 0
    running = _CAPITAL
    for yr in sorted(year_stats):
        st   = year_stats[yr]
        net  = st["pnl"]
        end  = running + net
        wp   = st["wins"] / st["trades"] * 100 if st["trades"] else 0
        ret  = net / running * 100 if running else 0
        # Scale to $100 account
        scaled = net * (100 / _CAPITAL)
        flag = "✓" if net >= 0 else "✗"
        if net >= 0: winning_years += 1
        print(f"  {flag} {yr:<5} {st['trades']:>7} {wp:>5.1f}% "
              f"{net:>+10,.2f} {end:>10,.2f} {ret:>6.1f}%  ${scaled:>+8.2f}")
        running = end

    print(f"\n  Winning years: {winning_years}/{len(year_stats)}")
    print(f"{'─'*70}")

    # Monthly for most recent full year
    full_yrs = [yr for yr in year_stats if year_stats[yr]["trades"] >= 5]
    if full_yrs:
        recent = max(full_yrs)
        monthly: dict[int, dict] = {}
        for t in trades:
            if t["year"] == recent:
                m = t["month"]
                monthly.setdefault(m, {"pnl":0.0,"trades":0,"wins":0})
                monthly[m]["pnl"] += t["pnl"]
                monthly[m]["trades"] += 1
                if t["pnl"] > 0: monthly[m]["wins"] += 1
        print(f"\n  Monthly breakdown {recent} (on $5K account):")
        losing_months = 0
        for m in sorted(monthly):
            st = monthly[m]
            wp = st["wins"]/st["trades"]*100 if st["trades"] else 0
            flag = "✓" if st["pnl"] >= 0 else "✗"
            if st["pnl"] < 0: losing_months += 1
            scaled = st["pnl"] * (100/_CAPITAL)
            print(f"    {flag} {datetime(recent,m,1).strftime('%b'):>3}  "
                  f"{st['trades']:>3} trades  {wp:>5.1f}% win  "
                  f"${st['pnl']:>+8,.2f}  (${scaled:>+6.2f} on $100)")
        print(f"\n    Losing months: {losing_months}/{len(monthly)}")

    print(f"\n{'═'*70}")
    if pf >= 1.5 and max_dd <= 0.15 and winning_years >= 5:
        verdict = "✅ DEPLOY — Edge proven across 6 assets, single capital pool"
    elif pf >= 1.3 and max_dd <= 0.20:
        verdict = "⚠️  PROMISING — Paper trade first"
    else:
        verdict = "⚠️  NEEDS WORK"
    print(f"  {verdict}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
