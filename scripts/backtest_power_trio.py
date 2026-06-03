#!/usr/bin/env python3
"""Power Trio portfolio backtest: EMA 8/21 + MACD + HTF 4H Trend.

Tests three strategies individually and combined:
  Strategy 1: EMA 8/21 Long-Only  (daily, 12 coins)
  Strategy 2: MACD Long-Only       (daily, 12 coins)
  Strategy 3: HTF Trend #7d        (4H Donchian, 4 coins: BTC/ETH/SOL/TRX)

For the combined portfolio, each strategy runs on 1/3 of total capital.
We measure whether combination beats any single strategy on risk-adjusted basis.

Usage:
    python scripts/backtest_power_trio.py
    python scripts/backtest_power_trio.py --capital 300000
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

_SYMBOLS_12 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]
_SYMBOLS_4 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "TRXUSDT"]
_TAKER_FEE  = 0.0006
_CANDLE_DIR = _REPO_ROOT / "data" / "candles"
_HOUR_MS    = 3_600_000
_DAY_MS     = 86_400_000


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def _ema(closes: list[float], period: int) -> list[float]:
    out = [float("nan")] * len(closes)
    alpha = 2.0 / (period + 1)
    val = closes[0] if closes else 0.0
    for i, c in enumerate(closes):
        val = alpha * c + (1 - alpha) * val
        out[i] = val if i >= period - 1 else float("nan")
    return out

def _sma(closes: list[float], period: int) -> list[float]:
    out = [float("nan")] * len(closes)
    s = 0.0
    for i, c in enumerate(closes):
        s += c
        if i >= period: s -= closes[i - period]
        if i >= period - 1: out[i] = s / period
    return out

def _atr_wilder(highs, lows, closes, period=14):
    out = [float("nan")] * len(closes)
    trs = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        trs.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    if len(trs) < period: return out
    out[period-1] = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        out[i] = (out[i-1] * (period-1) + trs[i]) / period
    return out

def _macd_signals(closes):
    """Returns list of 'LONG'|None per bar."""
    e12 = _ema(closes, 12)
    e26 = _ema(closes, 26)
    macd_line = [f - s if not (math.isnan(f) or math.isnan(s)) else float("nan")
                 for f, s in zip(e12, e26)]
    sig_alpha = 2.0 / 10
    sig_line = [float("nan")] * len(macd_line)
    fv = next((i for i, v in enumerate(macd_line) if not math.isnan(v)), None)
    if fv is None: return [None] * len(closes)
    sig_line[fv] = macd_line[fv]
    for i in range(fv + 1, len(macd_line)):
        if math.isnan(macd_line[i]): sig_line[i] = float("nan")
        elif math.isnan(sig_line[i-1]): sig_line[i] = macd_line[i]
        else: sig_line[i] = sig_alpha * macd_line[i] + (1 - sig_alpha) * sig_line[i-1]
    for i in range(fv, fv + 8): sig_line[i] = float("nan")
    state = [None] * len(closes)
    for i in range(1, len(closes)):
        if math.isnan(macd_line[i]) or math.isnan(sig_line[i]): continue
        state[i] = "LONG" if macd_line[i] > sig_line[i] else None
    return state

def _ema821_signals(closes):
    e8  = _ema(closes, 8)
    e21 = _ema(closes, 21)
    return ["LONG" if not (math.isnan(e8[i]) or math.isnan(e21[i])) and e8[i] > e21[i]
            else None for i in range(len(closes))]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_daily(symbols, from_ts, to_ts):
    data = {}
    for sym in symbols:
        path = _CANDLE_DIR / f"{sym}_1D.parquet"
        if not path.exists(): data[sym] = []; continue
        tbl = pq.read_table(path, columns=["open_time","open","high","low","close"])
        rows = sorted(zip(tbl.column("open_time").to_pylist(),
                          tbl.column("open").to_pylist(),
                          tbl.column("high").to_pylist(),
                          tbl.column("low").to_pylist(),
                          tbl.column("close").to_pylist()))
        data[sym] = [(ts,o,h,l,c) for ts,o,h,l,c in rows if from_ts <= ts <= to_ts]
    return data

def _load_1h(symbols, from_ts, to_ts):
    data = {}
    for sym in symbols:
        path = _CANDLE_DIR / f"{sym}_1H.parquet"
        if not path.exists(): data[sym] = []; continue
        tbl = pq.read_table(path, columns=["open_time","open","high","low","close","volume"])
        rows = sorted(zip(tbl.column("open_time").to_pylist(),
                          tbl.column("open").to_pylist(),
                          tbl.column("high").to_pylist(),
                          tbl.column("low").to_pylist(),
                          tbl.column("close").to_pylist(),
                          tbl.column("volume").to_pylist()))
        data[sym] = [(ts,o,h,l,c,v) for ts,o,h,l,c,v in rows if from_ts <= ts <= to_ts]
    return data


# ---------------------------------------------------------------------------
# Strategy simulators
# ---------------------------------------------------------------------------
@dataclass
class StratResult:
    name: str
    capital: float
    equity: float = 0.0
    peak: float = 0.0
    max_dd: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    total_fees: float = 0.0
    yearly: dict = field(default_factory=dict)
    is_gp: float = 0.0; is_gl: float = 0.0
    oos_gp: float = 0.0; oos_gl: float = 0.0

    def __post_init__(self):
        self.equity = self.capital
        self.peak   = self.capital

    @property
    def pf(self): return self.gross_profit / self.gross_loss if self.gross_loss > 0 else float("inf")
    @property
    def win_rate(self): return self.n_wins / self.n_trades if self.n_trades > 0 else 0.0
    @property
    def is_pf(self): return self.is_gp / self.is_gl if self.is_gl > 0 else float("inf")
    @property
    def oos_pf(self): return self.oos_gp / self.oos_gl if self.oos_gl > 0 else float("inf")

    def cagr(self, from_ts, to_ts):
        years = (to_ts - from_ts) / (1000 * 86400 * 365.25)
        if years <= 0 or self.equity <= 0: return -1.0
        return (self.equity / self.capital) ** (1 / years) - 1

    def record_trade(self, net, exit_ts):
        self.equity += net
        self.peak = max(self.peak, self.equity)
        dd = (self.peak - self.equity) / self.peak if self.peak > 0 else 0.0
        self.max_dd = max(self.max_dd, dd)
        if net > 0: self.gross_profit += net; self.n_wins += 1
        else: self.gross_loss += abs(net)
        self.n_trades += 1
        yr = datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc).year
        self.yearly[yr] = self.yearly.get(yr, 0.0) + net
        is_oos = yr >= 2023
        if is_oos:
            if net > 0: self.oos_gp += net
            else: self.oos_gl += abs(net)
        else:
            if net > 0: self.is_gp += net
            else: self.is_gl += abs(net)


def _sim_daily_longonly(name, signal_fn, daily_data, symbols, capital, from_ts, to_ts):
    r = StratResult(name=name, capital=capital)
    notional = capital / len(symbols)
    for sym in symbols:
        bars = daily_data.get(sym, [])
        if not bars: continue
        closes = [c for _,_,_,_,c in bars]
        timestamps = [ts for ts,_,_,_,_ in bars]
        signals = signal_fn(closes)
        prev_sig = None
        entry_price = None
        for i, (ts, sig) in enumerate(zip(timestamps, signals)):
            if sig == prev_sig: continue
            c = closes[i]
            # Close existing
            if entry_price is not None and sig != "LONG":
                pnl = (c - entry_price) / entry_price * notional
                fee = _TAKER_FEE * notional
                r.record_trade(pnl - fee, ts)
                r.total_fees += fee
                entry_price = None
            # Open new
            if sig == "LONG" and entry_price is None:
                fee = _TAKER_FEE * notional
                r.equity -= fee; r.total_fees += fee
                entry_price = c
            prev_sig = sig
        # EOD close
        if entry_price is not None and bars:
            last_ts, _, _, _, last_c = bars[-1]
            pnl = (last_c - entry_price) / entry_price * notional
            fee = _TAKER_FEE * notional
            r.record_trade(pnl - fee, last_ts)
    return r


def _sim_htf_trend(name, daily_data, symbols, capital, from_ts, to_ts):
    """Approximate HTF #7d using daily Donchian 30-bar + SMA(200) regime filter.
    Uses daily data as proxy for 4H (same signal, slightly different timing).
    Chandelier stop: 3× ATR(14) trailing stop."""
    r = StratResult(name=name, capital=capital)
    notional = capital / len(symbols)
    LOOKBACK = 30
    ATR_MULT = 3.0
    ATR_PERIOD = 14

    for sym in symbols:
        bars = daily_data.get(sym, [])
        if not bars: continue
        closes = [c for _,_,_,_,c in bars]
        highs  = [h for _,_,h,_,_ in bars]
        lows   = [l for _,_,_,l,_ in bars]
        timestamps = [ts for ts,_,_,_,_ in bars]
        sma200 = _sma(closes, 200)
        atrs   = _atr_wilder(highs, lows, closes, ATR_PERIOD)

        entry_price = None
        entry_side  = None
        trail_stop  = None
        prev_sig    = None

        for i in range(max(LOOKBACK, 200), len(bars)):
            ts    = timestamps[i]
            c     = closes[i]
            h     = highs[i]
            l     = lows[i]
            s200  = sma200[i]
            a     = atrs[i]
            if math.isnan(s200) or math.isnan(a): continue

            # Donchian signal
            don_high = max(highs[i - LOOKBACK: i])
            don_low  = min(lows[i - LOOKBACK: i])
            sig = None
            if c > don_high and c > s200: sig = "LONG"
            elif c < don_low and c < s200: sig = "SHORT"

            # Manage open position
            if entry_side is not None:
                # Update trail stop
                if entry_side == "LONG":
                    new_stop = h - ATR_MULT * a
                    trail_stop = max(trail_stop or new_stop, new_stop)
                    if c < trail_stop:
                        pnl = (c - entry_price) / entry_price * notional
                        fee = _TAKER_FEE * notional
                        r.record_trade(pnl - fee, ts); r.total_fees += fee
                        entry_price = entry_side = trail_stop = None
                        continue
                else:
                    new_stop = l + ATR_MULT * a
                    trail_stop = min(trail_stop or new_stop, new_stop)
                    if c > trail_stop:
                        pnl = (entry_price - c) / entry_price * notional
                        fee = _TAKER_FEE * notional
                        r.record_trade(pnl - fee, ts); r.total_fees += fee
                        entry_price = entry_side = trail_stop = None
                        continue

            # New entry
            if sig and sig != entry_side and entry_price is None:
                fee = _TAKER_FEE * notional
                r.equity -= fee; r.total_fees += fee
                entry_price = c
                entry_side  = sig
                trail_stop  = (c - ATR_MULT * a) if sig == "LONG" else (c + ATR_MULT * a)

        # EOD
        if entry_price is not None and bars:
            last_ts, _, _, _, last_c = bars[-1]
            if entry_side == "LONG":
                pnl = (last_c - entry_price) / entry_price * notional
            else:
                pnl = (entry_price - last_c) / entry_price * notional
            fee = _TAKER_FEE * notional
            r.record_trade(pnl - fee, last_ts)

    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _print_result(r, from_ts, to_ts, label=None):
    c = r.cagr(from_ts, to_ts)
    net = r.equity - r.capital
    label = label or r.name
    status = "✓ GO" if r.pf >= 1.30 and c >= 0.20 and r.max_dd <= 0.40 else \
             "MARG" if r.pf >= 1.10 and r.max_dd <= 0.40 and r.n_trades >= 60 else "KILL"
    print(f"\n  ── {label}  [{status}]")
    print(f"     Capital : ${r.capital:,.0f}  →  ${r.equity:,.0f}  (net ${net:+,.0f}, {net/r.capital*100:.1f}%)")
    print(f"     CAGR    : {c*100:.1f}%   PF: {r.pf:.2f}   DD: {r.max_dd*100:.1f}%   WR: {r.win_rate*100:.0f}%")
    print(f"     Trades  : {r.n_trades}   Fees: ${r.total_fees:,.0f}")
    print(f"     IS PF   : {r.is_pf:.2f}   OOS PF: {r.oos_pf:.2f}")
    yr_str = "  ".join(f"{yr}:{'+' if v>=0 else ''}{v/1000:.0f}K" for yr, v in sorted(r.yearly.items()))
    print(f"     By year : {yr_str}")
    return c, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=300_000.0,
                        help="Total capital split equally across 3 strategies (default $300K)")
    parser.add_argument("--from", dest="from_date", default="2021-01-01")
    parser.add_argument("--to",   dest="to_date",   default=None)
    args = parser.parse_args()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt   = (datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
               if args.to_date else datetime.now(timezone.utc))
    from_ts = int(from_dt.timestamp() * 1000)
    to_ts   = int(to_dt.timestamp() * 1000)

    slice_cap = args.capital / 3   # equal capital per strategy

    print("=" * 70)
    print("  POWER TRIO BACKTEST")
    print(f"  Period   : {from_dt.strftime('%Y-%m-%d')} → {to_dt.strftime('%Y-%m-%d')}")
    print(f"  Capital  : ${args.capital:,.0f} total  (${slice_cap:,.0f} per strategy)")
    print("=" * 70)

    print("\nLoading candles ...")
    daily12 = _load_daily(_SYMBOLS_12, from_ts, to_ts)
    daily4  = _load_daily(_SYMBOLS_4,  from_ts, to_ts)
    print(f"  Daily: {sum(len(v) for v in daily12.values()):,} bars (12 coins)")
    print(f"  Daily: {sum(len(v) for v in daily4.values()):,} bars (4 coins)")

    # Individual strategies
    ema = _sim_daily_longonly("EMA 8/21 Long-Only", _ema821_signals,
                               daily12, _SYMBOLS_12, slice_cap, from_ts, to_ts)
    mac = _sim_daily_longonly("MACD Long-Only",     _macd_signals,
                               daily12, _SYMBOLS_12, slice_cap, from_ts, to_ts)
    htf = _sim_htf_trend("HTF Trend #7d (daily proxy)",
                          daily4, _SYMBOLS_4, slice_cap, from_ts, to_ts)

    print("\n" + "=" * 70)
    print("  INDIVIDUAL STRATEGY RESULTS  (each on 1/3 capital)")
    print("=" * 70)
    c_ema, s_ema = _print_result(ema, from_ts, to_ts)
    c_mac, s_mac = _print_result(mac, from_ts, to_ts)
    c_htf, s_htf = _print_result(htf, from_ts, to_ts)

    # Combined portfolio
    print("\n" + "=" * 70)
    print("  COMBINED TRIO PORTFOLIO")
    print("=" * 70)

    # Merge yearly P&L
    all_years = sorted(set(list(ema.yearly) + list(mac.yearly) + list(htf.yearly)))
    trio_equity   = args.capital
    trio_peak     = args.capital
    trio_max_dd   = 0.0
    trio_gp = trio_gl = 0.0
    trio_fees = ema.total_fees + mac.total_fees + htf.total_fees
    trio_yearly: dict[int, float] = {}
    trio_is_gp = ema.is_gp + mac.is_gp + htf.is_gp
    trio_is_gl = ema.is_gl + mac.is_gl + htf.is_gl
    trio_oos_gp = ema.oos_gp + mac.oos_gp + htf.oos_gp
    trio_oos_gl = ema.oos_gl + mac.oos_gl + htf.oos_gl

    # Reconstruct DD from combined yearly P&L (approximation)
    running = args.capital
    peak_r  = args.capital
    for yr in all_years:
        yr_total = ema.yearly.get(yr, 0) + mac.yearly.get(yr, 0) + htf.yearly.get(yr, 0)
        running += yr_total
        peak_r   = max(peak_r, running)
        dd = (peak_r - running) / peak_r if peak_r > 0 else 0
        trio_max_dd = max(trio_max_dd, dd)
        trio_yearly[yr] = yr_total

    trio_equity = ema.equity + mac.equity + htf.equity
    trio_pf = (ema.gross_profit + mac.gross_profit + htf.gross_profit) / \
              max(ema.gross_loss + mac.gross_loss + htf.gross_loss, 0.01)
    trio_n  = ema.n_trades + mac.n_trades + htf.n_trades
    trio_wins = ema.n_wins + mac.n_wins + htf.n_wins
    trio_wr = trio_wins / trio_n if trio_n > 0 else 0
    trio_is_pf  = trio_is_gp  / trio_is_gl  if trio_is_gl  > 0 else float("inf")
    trio_oos_pf = trio_oos_gp / trio_oos_gl if trio_oos_gl > 0 else float("inf")

    years_trio = (to_ts - from_ts) / (1000 * 86400 * 365.25)
    trio_cagr = (trio_equity / args.capital) ** (1 / years_trio) - 1 if years_trio > 0 and trio_equity > 0 else -1

    net = trio_equity - args.capital
    status = "✓ GO" if trio_pf >= 1.30 and trio_cagr >= 0.20 and trio_max_dd <= 0.40 else \
             "MARG" if trio_pf >= 1.10 and trio_max_dd <= 0.40 else "KILL"

    print(f"\n  Combined capital: ${args.capital:,.0f}  →  ${trio_equity:,.0f}  [{status}]")
    print(f"  Net PnL : ${net:+,.0f}  ({net/args.capital*100:.1f}%)")
    print(f"  CAGR    : {trio_cagr*100:.1f}%   PF: {trio_pf:.2f}   DD: {trio_max_dd*100:.1f}%   WR: {trio_wr*100:.0f}%")
    print(f"  Trades  : {trio_n}   Fees: ${trio_fees:,.0f}")
    print(f"  IS PF   : {trio_is_pf:.2f}   OOS PF: {trio_oos_pf:.2f}")
    print(f"  By year :")
    for yr in all_years:
        e = ema.yearly.get(yr, 0)
        m = mac.yearly.get(yr, 0)
        h = htf.yearly.get(yr, 0)
        t = e + m + h
        print(f"    {yr}: EMA {e:+,.0f}  MACD {m:+,.0f}  HTF {h:+,.0f}  = TOTAL {t:+,.0f}")

    print()
    print("=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Strategy':<30} {'CAGR':>7} {'PF':>6} {'DD':>6} {'Verdict'}")
    print(f"  {'─'*30} {'─'*7} {'─'*6} {'─'*6} {'─'*7}")

    rows = [
        ("EMA 8/21 (1/3 capital)", c_ema, ema.pf, ema.max_dd, s_ema),
        ("MACD (1/3 capital)",     c_mac, mac.pf, mac.max_dd, s_mac),
        ("HTF Trend (1/3 capital)",c_htf, htf.pf, htf.max_dd, s_htf),
        ("─" * 30, None, None, None, None),
        ("TRIO COMBINED",           trio_cagr, trio_pf, trio_max_dd, status),
    ]
    for label, cagr, pf, dd, st in rows:
        if cagr is None:
            print(f"  {label}")
        else:
            print(f"  {label:<30} {cagr*100:>6.1f}% {pf:>6.2f} {dd*100:>5.1f}% {st}")

    print()
    print("  Key question: does TRIO beat the best individual strategy?")
    best_solo = max(c_ema, c_mac, c_htf)
    if trio_cagr > best_solo:
        print(f"  YES — Trio CAGR {trio_cagr*100:.1f}% > best solo {best_solo*100:.1f}%")
        print(f"        DD improvement: solo up to {max(ema.max_dd,mac.max_dd,htf.max_dd)*100:.1f}%  →  trio {trio_max_dd*100:.1f}%")
    else:
        print(f"  NO — Best solo ({best_solo*100:.1f}%) beats trio ({trio_cagr*100:.1f}%)")
        print(f"       Recommendation: run EMA 8/21 on full capital for max CAGR")
        print(f"       OR run EMA+MACD duo (similar signal, slightly different timing)")

    # Current signals
    print()
    print("=" * 70)
    print("  CURRENT SIGNAL STATE (as of today)")
    print("=" * 70)
    from nexflow.services.strategy.ema_trend_strategy import EMATrendStrategy

    strat = EMATrendStrategy(symbols=_SYMBOLS_12)
    today_ts = int(to_ts)
    seed_from = today_ts - 60 * _DAY_MS
    for sym, bars in daily12.items():
        recent = [(ts, c) for ts,_,_,_,c in bars if ts >= seed_from]
        for ts, c in recent:
            strat.on_daily_close(sym, c, ts)

    print("\n  EMA 8/21 signals:")
    for sym, state in strat.current_signals().items():
        icon = "📈 LONG" if state == "LONG" else "⏸  FLAT"
        print(f"    {sym:<12} {icon}")

    print()
    print("  MACD signals:")
    for sym in _SYMBOLS_12:
        bars = daily12.get(sym, [])
        if not bars: continue
        closes = [c for _,_,_,_,c in bars[-60:]]
        sigs = _macd_signals(closes)
        last = next((s for s in reversed(sigs) if s is not None), None)
        icon = "📈 LONG" if last == "LONG" else "⏸  FLAT"
        print(f"    {sym:<12} {icon}")

    print()
    print("  → ENTRY STRATEGY: wait for EMA 8/21 BUY signals (EMA8 crosses above EMA21)")
    print("    These fire when bull market resumes. Check daily after close.")
    print("=" * 70)


if __name__ == "__main__":
    main()
