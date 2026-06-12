#!/usr/bin/env python3
"""NexFlow Trio V8 — paper-trade all three GO strategies simultaneously.

  Strategy 1: EMA 8/21 Daily Long-Only     (CAGR 24%, DD 11%, PF 1.95)
  Strategy 2: MACD 12/26/9 Daily Long-Only (CAGR 23%, DD 17%, PF 1.59)
  Strategy 3: 4H EMA 5/13 Long-Only        (CAGR 20%, DD  9%, PF 1.46)

Confluence position sizing:
  1 strategy signals LONG  → 1.0× base notional per coin
  2 strategies agree       → 1.5× base notional per coin
  All 3 agree              → 2.0× base notional per coin

V8 upgrades over V7:
  - AND-entry asymmetric regime: enter bear only when BTC < SMA200
    AND 30d return < -20% (avoids false bear flips on normal corrections)
  - Slow bear exit: BTC must stay above SMA200 for 5 consecutive days
  - 20d momentum gate: skip new long entries if coin's 20d return ≤ 0

Base notional = capital / 12 coins.

Checks once per day at 00:05 UTC (5 min after daily candle close).
4H EMA checks are embedded in the same daily loop using latest 4H close.

Usage:
    # See what strategies would have done since 2024:
    python scripts/run_trio_paper.py --mode replay --from 2024-01-01 --capital 5000

    # Start live paper trading on Bitget demo:
    BITGET_PAPER=1 python scripts/run_trio_paper.py --mode live --capital 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pip install pyarrow"); sys.exit(1)

from nexflow.services.strategy.ema_trend_strategy import EMATrendStrategy

_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]
_CANDLE_DIR = _REPO_ROOT / "data" / "candles"
_TAKER_FEE      = 0.0006
_DAY_MS         = 86_400_000
_HOUR_MS        = 3_600_000
_HARD_STOP_PCT  = 0.15   # close a short if price rises 15% above entry
_STATE_FILE     = _REPO_ROOT / "data" / "nexflow_live_state.json"


# ---------------------------------------------------------------------------
# MACD strategy (stateful, per-symbol)
# ---------------------------------------------------------------------------
class MACDState:
    def __init__(self, fast=12, slow=26, signal=9):
        self._af = 2/(fast+1); self._as = 2/(slow+1); self._sg = 2/(signal+1)
        self._ema_fast = self._ema_slow = self._signal_ema = None
        self._prev_hist: Optional[float] = None
        self._position = False

    def update(self, close: float) -> Optional[str]:
        self._ema_fast = self._af*close + (1-self._af)*self._ema_fast if self._ema_fast else close
        self._ema_slow = self._as*close + (1-self._as)*self._ema_slow if self._ema_slow else close
        macd = self._ema_fast - self._ema_slow
        self._signal_ema = self._sg*macd + (1-self._sg)*self._signal_ema if self._signal_ema else macd
        hist = macd - self._signal_ema
        action = None
        if self._prev_hist is not None:
            if self._prev_hist <= 0 < hist and not self._position:
                action = "OPEN_LONG"; self._position = True
            elif self._prev_hist >= 0 > hist and self._position:
                action = "CLOSE_LONG"; self._position = False
        self._prev_hist = hist
        return action

    @property
    def position(self): return self._position

    def current_signal(self):
        if self._prev_hist is None: return "WARMUP"
        return "LONG" if self._position else "FLAT"


# ---------------------------------------------------------------------------
# 4H EMA strategy (stateful, per-symbol) — seeded from 1H parquet
# ---------------------------------------------------------------------------
class EMA4HState:
    def __init__(self, fast=5, slow=13):
        self._af = 2/(fast+1); self._as = 2/(slow+1)
        self._ema_fast = self._ema_slow = None
        self._bars_seen = 0
        self._prev_above: Optional[bool] = None
        self._position = False
        self._daily_trend: Optional[bool] = None  # True=bull, False=bear

    def update_daily_trend(self, ema_fast_d: float, ema_slow_d: float):
        self._daily_trend = ema_fast_d > ema_slow_d

    def update(self, close: float) -> Optional[str]:
        self._ema_fast = self._af*close + (1-self._af)*self._ema_fast if self._ema_fast else close
        self._ema_slow = self._as*close + (1-self._as)*self._ema_slow if self._ema_slow else close
        self._bars_seen += 1
        if self._bars_seen < 13:
            return None
        above = self._ema_fast > self._ema_slow
        action = None
        if self._prev_above is not None and above != self._prev_above:
            if above:
                # Only enter long if daily trend agrees (or unknown)
                if self._daily_trend is not False and not self._position:
                    action = "OPEN_LONG"; self._position = True
            else:
                if self._position:
                    action = "CLOSE_LONG"; self._position = False
        self._prev_above = above
        return action

    @property
    def position(self): return self._position

    def current_signal(self):
        if self._prev_above is None: return "WARMUP"
        return "LONG" if self._position else "FLAT"


# ---------------------------------------------------------------------------
# BTC regime tracker (V8: AND-entry asymmetric regime)
# ---------------------------------------------------------------------------
class BTCRegime:
    """V8 regime: enter bear only when BTC < SMA200 AND 30d drop > 20%.
    Exit bear slowly: BTC must stay above SMA200 for 5 consecutive days.
    """
    _BEAR_DROP_PCT  = -0.20   # 30d return threshold to trigger bear entry
    _CONFIRM_DAYS   = 10      # consecutive days above SMA200 needed to exit bear (V8.63)
    _MOM_GATE_DAYS  = 20      # lookback for momentum gate on longs (V8.63)

    def __init__(self):
        self._closes: list[float] = []
        self._bear = False
        self._above_streak = 0   # consecutive days BTC has been above SMA200

    def update(self, close: float) -> None:
        self._closes.append(close)
        if len(self._closes) < 200:
            return

        sma200 = sum(self._closes[-200:]) / 200
        below_sma200 = close < sma200

        # 30-day return
        mom30 = 0.0
        if len(self._closes) >= 31:
            mom30 = (self._closes[-1] - self._closes[-31]) / self._closes[-31]

        # Track streak of days above SMA200 (for slow exit)
        if not below_sma200:
            self._above_streak += 1
        else:
            self._above_streak = 0

        if self._bear:
            # Slow exit: need CONFIRM_DAYS consecutive days above SMA200
            if self._above_streak >= self._CONFIRM_DAYS:
                self._bear = False
        else:
            # AND-entry: only flip to bear if BOTH below SMA200 AND big 30d drop
            if below_sma200 and mom30 <= self._BEAR_DROP_PCT:
                self._bear = True

    @property
    def is_bear(self) -> bool:
        return self._bear

    def mom_return(self, closes: list[float], days: int) -> float:
        """N-day return for a coin. Returns 0.0 if not enough history."""
        if len(closes) < days + 1:
            return 0.0
        return (closes[-1] - closes[-(days+1)]) / closes[-(days+1)]

    def tsmom_return(self, closes_126: list[float]) -> float:
        """126-day return for TSMOM scoring. Needs 127+ items."""
        if len(closes_126) < 127:
            return 0.0
        return (closes_126[-1] - closes_126[-127]) / closes_126[-127]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_candles(symbol: str, tf: str, from_ts: int, to_ts: int) -> list[tuple]:
    path = _CANDLE_DIR / f"{symbol}_{tf}.parquet"
    if not path.exists():
        return []
    tbl = pq.read_table(path, columns=["open_time","close"])
    rows = sorted(zip(tbl.column("open_time").to_pylist(), tbl.column("close").to_pylist()))
    return [(int(ts), float(c)) for ts, c in rows if from_ts <= ts <= to_ts]


def _resample_4h(bars_1h: list[tuple]) -> list[tuple]:
    buckets: dict[int, list] = {}
    for ts, c in bars_1h:
        hour = (ts % _DAY_MS) // _HOUR_MS
        bts = (ts // _DAY_MS) * _DAY_MS + (hour // 4) * 4 * _HOUR_MS
        buckets.setdefault(bts, []).append((ts, c))
    result = []
    for bts in sorted(buckets):
        grp = sorted(buckets[bts])
        if len(grp) >= 4:
            result.append((bts, grp[-1][1]))
    return result


def _fetch_close(symbol: str) -> Optional[tuple[int, float]]:
    url = (f"https://api.bitget.com/api/v2/mix/market/history-candles"
           f"?symbol={symbol}&productType=USDT-FUTURES&granularity=1D&limit=3")
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"NexFlow/1.0","Accept":"application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        if data.get("code") != "00000" or not data.get("data"): return None
        rows = data["data"]
        # rows[0] = currently forming candle (not closed), rows[1] = last closed candle
        # Pick the most recent fully closed candle
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        for row in rows:
            candle_open = int(row[0])
            candle_close_ms = candle_open + 86_400_000
            if candle_close_ms <= now_ms:
                return candle_open, float(row[4])
        return int(rows[1][0]), float(rows[1][4])
    except Exception as e:
        print(f"  [WARN] {symbol}: {e}"); return None


def _fetch_1h_recent(symbol: str, n: int = 40) -> list[tuple[int, float]]:
    url = (f"https://api.bitget.com/api/v2/mix/market/history-candles"
           f"?symbol={symbol}&productType=USDT-FUTURES&granularity=1H&limit={n}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"NexFlow/1.0","Accept":"application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        if data.get("code") != "00000": return []
        return sorted([(int(row[0]), float(row[4])) for row in data["data"]])
    except Exception as e:
        print(f"  [WARN] 1H {symbol}: {e}"); return []


# ---------------------------------------------------------------------------
# Replay mode
# ---------------------------------------------------------------------------
def run_replay(symbols, capital, from_ts, to_ts):
    base_notional = capital / len(symbols)
    notional = base_notional / 3  # per-strategy slot (legacy sizing)
    print(f"NexFlow Trio V8.63 — REPLAY  |  capital=${capital:,.0f}  |  ${base_notional:,.0f}/coin")
    print(f"Period: {datetime.fromtimestamp(from_ts/1000,tz=timezone.utc).date()} → "
          f"{datetime.fromtimestamp(to_ts/1000,tz=timezone.utc).date()}")
    print(f"Regime: V8.63 AND-entry (BTC<SMA200 AND 30d<-20%, 10d confirm exit)")
    print(f"MomGate: skip longs if coin 20d return ≤ 0")
    print()

    ema_strat  = EMATrendStrategy(symbols=symbols, fast=8, slow=21)
    macd_strats = {sym: MACDState() for sym in symbols}
    ema4h_strats = {sym: EMA4HState() for sym in symbols}
    btc_regime = BTCRegime()
    coin_closes: dict[str, list[float]] = {sym: [] for sym in symbols}  # for TSMOM

    # Build event streams
    daily_events: list[tuple[int,str,float]] = []
    h4_events:    list[tuple[int,str,float]] = []

    for sym in symbols:
        for ts, c in _load_candles(sym, "1D", 0, to_ts):
            daily_events.append((ts, sym, c))
        h1 = _load_candles(sym, "1H", 0, to_ts)
        for ts, c in _resample_4h(h1):
            h4_events.append((ts, sym, c))

    daily_events.sort(); h4_events.sort()

    equity = capital; peak = capital; max_dd = 0.0
    # One confluence position per coin (long), separate TSMOM shorts
    long_pos: dict[str, tuple[float,float]] = {}    # sym → (entry, notional)
    coin_sigs: dict[str, set] = {sym: set() for sym in symbols}  # which strats signal LONG
    short_pos: dict[str, tuple[float,float]] = {}   # sym → (entry, notional) TSMOM shorts
    trades: list[dict] = []
    last_tsmom_rebal = 0
    year_pnl: dict[int,float] = {}

    def _confluence_size(sym: str) -> float:
        n = len(coin_sigs[sym])
        return base_notional * {0:0.0, 1:1.0, 2:1.5, 3:2.0}.get(n, 2.0)

    def _close_long(sym: str, close: float, ts: int, src: str) -> None:
        if sym not in long_pos: return
        ep, n = long_pos.pop(sym)
        pnl = (close-ep)/ep*n - _TAKER_FEE*n
        equity_ref[0] += pnl
        yr = datetime.fromtimestamp(ts/1000,tz=timezone.utc).year
        year_pnl[yr] = year_pnl.get(yr,0)+pnl
        trades.append({"ts":ts,"src":src,"sym":sym,"pnl":pnl})

    equity_ref = [equity]  # mutable ref so nested func can update

    # Merge daily + 4H events in time order
    all_events: list[tuple[int,str,str,float]] = (
        [(ts,sym,"1D",c) for ts,sym,c in daily_events] +
        [(ts,sym,"4H",c) for ts,sym,c in h4_events]
    )
    all_events.sort()

    for ts, sym, tf, close in all_events:
        in_range = ts >= from_ts
        equity = equity_ref[0]

        if tf == "1D":
            if sym == "BTCUSDT":
                btc_regime.update(close)
            coin_closes[sym].append(close)

            bear = btc_regime.is_bear

            # ── TSMOM short rebalance (weekly when BTC is bear) ──
            if sym == symbols[-1] and bear and in_range and (ts - last_tsmom_rebal) >= 7*_DAY_MS:
                last_tsmom_rebal = ts
                scores = [(btc_regime.tsmom_return(coin_closes[s]), s)
                          for s in symbols if len(coin_closes[s]) >= 127]
                scores.sort()
                desired = {s for ret, s in scores if ret < -0.05}

                for s in list(short_pos):
                    if s not in desired:
                        ep, n = short_pos.pop(s)
                        exit_c = coin_closes[s][-1] if coin_closes[s] else ep
                        pnl = (ep - exit_c) / ep * n - _TAKER_FEE * n
                        equity += pnl
                        yr = datetime.fromtimestamp(ts/1000,tz=timezone.utc).year
                        year_pnl[yr] = year_pnl.get(yr,0)+pnl
                        trades.append({"ts":ts,"src":"SHORT","sym":s,"pnl":pnl})

                for s in desired:
                    if s not in short_pos:
                        c = coin_closes[s][-1] if coin_closes[s] else 0
                        if c <= 0: continue
                        equity -= _TAKER_FEE * base_notional
                        short_pos[s] = (c, base_notional)

                equity_ref[0] = equity

            # Close shorts if returned to bull
            if sym == symbols[-1] and not bear and short_pos and in_range:
                for s in list(short_pos):
                    ep, n = short_pos.pop(s)
                    c = coin_closes[s][-1] if coin_closes[s] else ep
                    pnl = (ep - c) / ep * n - _TAKER_FEE * n
                    equity += pnl
                    yr = datetime.fromtimestamp(ts/1000,tz=timezone.utc).year
                    year_pnl[yr] = year_pnl.get(yr,0)+pnl
                    trades.append({"ts":ts,"src":"SHORT","sym":s,"pnl":pnl})
                equity_ref[0] = equity

            # Force-close long if bear triggered
            if bear and sym in long_pos and in_range:
                coin_sigs[sym].clear()
                _close_long(sym, close, ts, "REGIME")
                equity = equity_ref[0]

            # EMA 8/21 long signal
            prev_ema = "EMA" in coin_sigs[sym]
            sigs = ema_strat.on_daily_close(sym, close, ts)
            for sig in sigs:
                if sig.action == "OPEN_LONG":
                    coin_sigs[sym].add("EMA")
                elif sig.action == "CLOSE_LONG":
                    coin_sigs[sym].discard("EMA")
            new_ema = "EMA" in coin_sigs[sym]

            # MACD long signal
            prev_macd = "MACD" in coin_sigs[sym]
            action = macd_strats[sym].update(close)
            if action == "OPEN_LONG":
                coin_sigs[sym].add("MACD")
            elif action == "CLOSE_LONG":
                coin_sigs[sym].discard("MACD")
            new_macd = "MACD" in coin_sigs[sym]

            # Update 4H daily trend filter
            csigs = ema_strat.current_signals()
            if sym in csigs:
                ema4h_strats[sym].update_daily_trend(1.0 if csigs[sym]=="LONG" else 0.0, 0.5)

            if in_range and not bear:
                n_long = len(coin_sigs[sym])
                in_pos = sym in long_pos
                mom30 = btc_regime.mom_return(coin_closes[sym], BTCRegime._MOM_GATE_DAYS)
                if n_long == 0 and in_pos:
                    _close_long(sym, close, ts, "EMA/MACD")
                    equity = equity_ref[0]
                elif n_long > 0 and not in_pos and mom30 > 0:
                    size = _confluence_size(sym)
                    equity -= _TAKER_FEE * size
                    long_pos[sym] = (close, size)
                    equity_ref[0] = equity
                # no resize mid-trade — size is fixed at entry

        elif tf == "4H":
            bear = btc_regime.is_bear

            action = ema4h_strats[sym].update(close)
            if action == "OPEN_LONG":
                coin_sigs[sym].add("4H")
            elif action == "CLOSE_LONG":
                coin_sigs[sym].discard("4H")

            if in_range and not bear:
                n_long = len(coin_sigs[sym])
                in_pos = sym in long_pos
                mom30 = btc_regime.mom_return(coin_closes[sym], BTCRegime._MOM_GATE_DAYS)
                if n_long == 0 and in_pos:
                    _close_long(sym, close, ts, "4H")
                    equity = equity_ref[0]
                elif n_long > 0 and not in_pos and mom30 > 0:
                    size = _confluence_size(sym)
                    equity -= _TAKER_FEE * size
                    long_pos[sym] = (close, size)
                    equity_ref[0] = equity

        equity = equity_ref[0]

        if in_range:
            if equity > peak: peak = equity
            dd = (peak-equity)/peak
            if dd > max_dd: max_dd = dd

    # Mark open positions + shorts to market
    last_prices: dict[str,float] = {}
    for sym in symbols:
        d = _load_candles(sym, "1D", 0, to_ts)
        if d: last_prices[sym] = d[-1][1]

    equity = equity_ref[0]
    unrealised = 0.0
    last_yr = datetime.fromtimestamp(to_ts/1000,tz=timezone.utc).year
    for sym, (ep, n) in long_pos.items():
        p = last_prices.get(sym, ep)
        mtm = (p-ep)/ep*n
        unrealised += mtm
        year_pnl[last_yr] = year_pnl.get(last_yr, 0) + mtm
    for sym, (ep, n) in short_pos.items():
        p = last_prices.get(sym, ep)
        mtm = (ep-p)/ep*n
        unrealised += mtm
        year_pnl[last_yr] = year_pnl.get(last_yr, 0) + mtm

    total_eq = equity + unrealised
    net = total_eq - capital
    years = (to_ts - from_ts) / (1000*86400*365.25)
    cagr  = (total_eq/capital)**(1/years)-1 if years>0 and total_eq>0 else 0

    print(f"Trades    : {len(trades)}")
    print(f"Realised  : ${equity:,.0f}")
    print(f"Unrealised: ${unrealised:+,.0f}")
    print(f"Total eq  : ${total_eq:,.0f}  (net ${net:+,.0f}, {net/capital*100:.1f}%)")
    print(f"CAGR      : {cagr*100:.1f}%")
    print(f"Max DD    : {max_dd*100:.1f}%")
    print(f"Regime    : {'BEAR (BTC<SMA200) — TSMOM shorts active' if btc_regime.is_bear else 'BULL (BTC>SMA200) — longs active'}")
    print()
    print("Year-by-year:")
    for yr in sorted(year_pnl):
        p = year_pnl[yr]
        tag = " <<BEAR" if yr in [2022,2025,2026] else ""
        print(f"  {yr}: ${p:>+,.0f}{tag}")
    print()
    print("Current positions:")
    for sym, (ep, n) in long_pos.items():
        p = last_prices.get(sym, ep)
        pct = (p-ep)/ep*100
        sigs = "/".join(sorted(coin_sigs[sym])) or "?"
        print(f"  [LONG]  {sym:<12} entry={ep:,.4f}  now={p:,.4f}  {pct:+.1f}%  size=${n:,.0f}  [{sigs}]")
    for sym, (ep, n) in short_pos.items():
        p = last_prices.get(sym, ep)
        pct = (ep-p)/ep*100
        print(f"  [SHORT] {sym:<12} entry={ep:,.4f}  now={p:,.4f}  {pct:+.1f}%  size=${n:,.0f}")
    if not long_pos and not short_pos:
        print("  (all flat)")
    print()
    print("Current signals:")
    ema_sigs  = ema_strat.current_signals()
    for sym in symbols:
        e  = ema_sigs.get(sym,"?")
        m  = macd_strats[sym].current_signal()
        h4 = ema4h_strats[sym].current_signal()
        if e=="LONG" or m=="LONG" or h4=="LONG":
            print(f"  {sym:<12}  EMA:{e:<6}  MACD:{m:<6}  4H:{h4}")


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------
def run_live(symbols, capital):
    from nexflow.exchange.bitget_client import BitgetClient
    from nexflow.execution.adapter import BitgetPaperAdapter

    client  = BitgetClient.from_env()
    adapter = BitgetPaperAdapter(client)

    # Use real account balance if available, fall back to --capital arg
    try:
        from nexflow.exchange.bitget_order import get_account_balance
        live_balance = get_account_balance(client)
        if live_balance > 0:
            print(f"Account balance : ${live_balance:,.2f} (from exchange)")
            capital = live_balance
        else:
            print(f"Account balance : could not fetch — using --capital ${capital:,.0f}")
    except Exception as e:
        print(f"Account balance : fetch failed ({e}) — using --capital ${capital:,.0f}")

    base_notional    = capital / len(symbols)
    _TARGET_RISK     = 0.01    # 1% of capital daily risk per position (ATR sizing)
    _ATR_WINDOW      = 14      # rolling days for vol estimate
    _CIRCUIT_BREAKER_DD = 0.20
    portfolio_peak   = capital
    circuit_open     = False

    def _atr_notional(sym: str, mult: float = 1.0) -> float:
        """Return vol-adjusted notional for sym, capped at 2× base, floored at 0.5× base."""
        closes = coin_closes_live.get(sym, [])
        if len(closes) < _ATR_WINDOW + 1:
            return base_notional * mult  # not enough history yet — fall back to flat
        rets = [(closes[i] - closes[i-1]) / closes[i-1]
                for i in range(len(closes) - _ATR_WINDOW, len(closes))]
        mean_r = sum(rets) / len(rets)
        vol = (sum((r - mean_r)**2 for r in rets) / len(rets)) ** 0.5
        if vol <= 0:
            return base_notional * mult
        sized = (_TARGET_RISK * capital) / vol
        return max(base_notional * 0.5, min(sized * mult, base_notional * 2.0))

    print("NexFlow Trio V8.63 — LIVE PAPER MODE")
    print(f"Symbols        : {len(symbols)} coins")
    print(f"Capital        : ${capital:,.0f}  (${base_notional:.2f} base/coin)")
    print(f"Sizing         : ATR vol-adjusted (target risk {_TARGET_RISK*100:.0f}%/day, cap 2× base)")
    print(f"Confluence     : 1 strat=1×  2 strats=1.5×  3 strats=2×")
    print(f"Regime gate    : V8.63 AND-entry (BTC<SMA200 AND 30d<-20%, 10d confirm exit)")
    print(f"Momentum gate  : skip new longs if coin 20d return ≤ 0")
    print(f"Fee            : {_TAKER_FEE*100:.2f}% taker")
    print(f"Circuit breaker: pause new entries if portfolio DD ≥ {_CIRCUIT_BREAKER_DD*100:.0f}%")
    print()

    # Seed strategies from historical data
    today_ts = int(datetime.now(timezone.utc).replace(
        hour=0,minute=0,second=0,microsecond=0).timestamp()*1000)
    seed_from = today_ts - 210 * _DAY_MS  # 210 days to warm up SMA200

    ema_strat    = EMATrendStrategy(symbols=symbols, fast=8, slow=21)
    macd_strats  = {sym: MACDState() for sym in symbols}
    ema4h_strats = {sym: EMA4HState() for sym in symbols}
    btc_regime   = BTCRegime()
    coin_closes_live: dict[str, list[float]] = {sym: [] for sym in symbols}
    seed_last_4h_ts: dict[str, int] = {sym: 0 for sym in symbols}

    print("Seeding from historical data ...")
    for sym in symbols:
        daily = _load_candles(sym, "1D", seed_from, today_ts)
        for ts, c in daily:
            ema_strat.on_daily_close(sym, c, ts)
            macd_strats[sym].update(c)
            coin_closes_live[sym].append(c)
            if sym == "BTCUSDT":
                btc_regime.update(c)
        h1 = _load_candles(sym, "1H", seed_from, today_ts)
        seed_last_4h_ts[sym] = 0
        for ts, c in _resample_4h(h1):
            ema4h_strats[sym].update(c)
            seed_last_4h_ts[sym] = ts
        csigs = ema_strat.current_signals()
        if sym in csigs:
            ema4h_strats[sym].update_daily_trend(1.0 if csigs[sym]=="LONG" else 0.0, 0.5)

    print("Seed complete.")
    print()
    print("Current signals:")
    ema_sigs = ema_strat.current_signals()
    for sym in symbols:
        e  = ema_sigs.get(sym,"?")
        m  = macd_strats[sym].current_signal()
        h4 = ema4h_strats[sym].current_signal()
        print(f"  {sym:<12}  EMA:{e:<6}  MACD:{m:<6}  4H:{h4}")
    print()

    # Track current strategy agreement per coin for confluence sizing
    coin_longs: dict[str, set] = {sym: set() for sym in symbols}  # {sym: {src,...}}
    last_4h_ts: dict[str, int] = dict(seed_last_4h_ts)  # last processed 4H bar ts (seeded)
    news_suspend = [False]  # last news read: suspend new longs (shared with 6H checks)

    def _confluence_notional(sym: str) -> float:
        """Return ATR-adjusted notional scaled by confluence multiplier."""
        n = len(coin_longs[sym])
        mult = {0: 0.0, 1: 1.0, 2: 1.5, 3: 2.0}.get(n, 2.0)
        return _atr_notional(sym, mult)

    def _available_margin() -> float:
        """Free USDT margin on the exchange right now. 0.0 if fetch fails."""
        try:
            from nexflow.exchange.bitget_order import get_account_balance
            return max(0.0, get_account_balance(client))
        except Exception:
            return 0.0

    _MARGIN_BUFFER = 0.95  # never commit more than 95% of free margin

    def _exec(action, sym, price, src):
        try:
            if action == "OPEN_LONG":
                notional = _confluence_notional(sym)
                # Clamp to what the account can actually carry (1× cross margin)
                avail = _available_margin()
                if avail > 0 and notional > avail * _MARGIN_BUFFER:
                    clamped = avail * _MARGIN_BUFFER
                    if clamped < notional * 0.25:
                        print(f"  [{src}] {sym} skipped — free margin ${avail:,.0f} "
                              f"too low for ${notional:,.0f} position")
                        return
                    print(f"  [{src}] {sym} size clamped ${notional:,.0f} → ${clamped:,.0f} "
                          f"(free margin ${avail:,.0f})")
                    notional = clamped
                qty = notional / price if price > 0 else 0
                if qty <= 0: return
                res = adapter.on_entry(sym, "long", qty, 0.0, 0.0, 0.0)
                if res is not None and not getattr(res, "accepted", True):
                    print(f"  [ERROR] {src} BUY {sym} rejected by exchange: "
                          f"{getattr(res, 'note', '?')} — position NOT opened")
                    return
                n = len(coin_longs[sym])
                mult = {1:1.0, 2:1.5, 3:2.0}.get(n, 1.0)
                print(f"  [{src}] BUY  {sym:<12} @ {price:,.4f}  "
                      f"({n} strat{'s' if n>1 else ''} agree → {mult}× = ${notional:,.0f} ATR-sized)")
            elif action == "CLOSE_LONG":
                # Fetch actual size from exchange so we close exactly what's held
                from nexflow.exchange.bitget_order import get_position
                pos = get_position(adapter._client, sym)
                qty = float(pos.get("total", 0)) if pos else 0.0
                if qty <= 0:
                    coin_longs[sym].clear()
                    return
                adapter.on_close(sym, "long", qty, price, f"{src}_cross")
                print(f"  [{src}] SELL {sym:<12} @ {price:,.4f}  qty={qty}")
        except Exception as e:
            print(f"  [ERROR] {src} {sym} {action}: {e}")

    last_tsmom_rebal_live = [0]   # ts of last weekly short rebalance
    live_shorts: dict[str, float] = {}  # sym → entry_price (shorts open on exchange)

    def _save_state() -> None:
        import json
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STATE_FILE.write_text(json.dumps({
                "last_tsmom_rebal": last_tsmom_rebal_live[0],
            }))
        except Exception as e:
            print(f"  [WARN] Could not save state: {e}")

    def _load_state() -> None:
        import json
        if not _STATE_FILE.exists():
            return
        try:
            state = json.loads(_STATE_FILE.read_text())
            last_tsmom_rebal_live[0] = int(state.get("last_tsmom_rebal", 0))
            saved_dt = datetime.fromtimestamp(
                last_tsmom_rebal_live[0] / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC")
            print(f"  Restored last TSMOM rebal: {saved_dt}")
        except Exception as e:
            print(f"  [WARN] Could not load state: {e}")

    _load_state()

    # Reconcile with exchange on startup — restore live_shorts and coin_longs
    # from open positions so restarts don't lose state or create duplicates.
    if adapter is not None:
        try:
            from nexflow.exchange.bitget_order import get_position
            print("Reconciling open positions from exchange ...")
            restored_shorts = 0
            restored_longs  = 0
            for sym in symbols:
                pos = get_position(adapter._client, sym)
                if not pos:
                    time.sleep(0.15)
                    continue
                side  = pos.get("holdSide", "")
                entry = float(pos.get("openPriceAvg", 0))
                if entry <= 0:
                    time.sleep(0.15)
                    continue
                if side == "short":
                    live_shorts[sym] = entry
                    print(f"  Restored SHORT {sym:<12} entry={entry:,.4f}")
                    restored_shorts += 1
                elif side == "long":
                    # Mark as held by all three strategies so the bot treats it
                    # as a full confluence position and won't re-open or orphan it.
                    coin_longs[sym] = {"EMA", "MACD", "4H"}
                    print(f"  Restored LONG  {sym:<12} entry={entry:,.4f}")
                    restored_longs += 1
                time.sleep(0.15)
            print(f"  Restored {restored_shorts} short(s), {restored_longs} long(s).")
            print()
        except Exception as e:
            print(f"  [WARN] Could not reconcile positions: {e}")

    def _exec_short(action: str, sym: str, price: float,
                    notional: float | None = None) -> None:
        """Place or close a short position on the exchange.

        notional: pre-computed budget-aware size (from the rebalance batch).
        Falls back to standalone ATR sizing clamped to free margin.
        """
        try:
            if action == "OPEN_SHORT":
                if notional is None:
                    notional = _atr_notional(sym)
                    avail = _available_margin()
                    if avail > 0 and notional > avail * _MARGIN_BUFFER:
                        notional = avail * _MARGIN_BUFFER
                if notional < 10:  # below exchange minimum — don't bother
                    print(f"  [SHORT] {sym} skipped — size ${notional:,.0f} too small")
                    return
                qty = notional / price if price > 0 else 0
                if qty <= 0: return
                res = adapter.on_entry(sym, "short", qty, 0.0, 0.0, 0.0)
                if res is not None and not getattr(res, "accepted", True):
                    print(f"  [ERROR] SHORT SELL {sym} rejected by exchange: "
                          f"{getattr(res, 'note', '?')} — position NOT opened")
                    return
                live_shorts[sym] = price
                print(f"  [SHORT] SELL {sym:<12} @ {price:,.4f}  size=${notional:,.0f} (ATR-sized)")
            elif action == "CLOSE_SHORT":
                # Fetch actual size from exchange so we close exactly what's held
                from nexflow.exchange.bitget_order import get_position
                pos = get_position(adapter._client, sym)
                qty = float(pos.get("total", 0)) if pos else 0.0
                if qty <= 0:
                    live_shorts.pop(sym, None)
                    return
                adapter.on_close(sym, "short", qty, price, "tsmom_exit")
                live_shorts.pop(sym, None)
                print(f"  [SHORT] BUY  {sym:<12} @ {price:,.4f}  qty={qty}  (closed)")
        except Exception as e:
            print(f"  [ERROR] SHORT {sym} {action}: {e}")

    def _process_4h_signals(suspend: bool = False) -> bool:
        """Process only NEW 4H bars per symbol (dedup by timestamp).

        Safe to call multiple times per day — never re-feeds a bar into the
        stateful EMA, so it can run on the 6H checks as well as the daily check.
        Only acts when BTC regime is bull (longs allowed). Returns True if any
        signal fired.
        """
        if btc_regime.is_bear:
            return False
        fired = False
        for sym in symbols:
            bars_1h = _fetch_1h_recent(sym, 40)
            if not bars_1h:
                continue
            mom30 = btc_regime.mom_return(coin_closes_live[sym], BTCRegime._MOM_GATE_DAYS)
            mom_blocked = mom30 <= 0
            for ts4, c4 in _resample_4h(bars_1h):
                if ts4 <= last_4h_ts[sym]:
                    continue  # already processed this bar — skip to avoid corrupting EMA
                last_4h_ts[sym] = ts4
                action = ema4h_strats[sym].update(c4)
                if not action:
                    continue
                if action == "OPEN_LONG":
                    coin_longs[sym].add("4H")
                elif action == "CLOSE_LONG":
                    coin_longs[sym].discard("4H")
                if action == "OPEN_LONG" and (suspend or circuit_open):
                    print(f"  [4H] {sym} suppressed — {'extreme event' if suspend else 'circuit breaker'}")
                elif action == "OPEN_LONG" and mom_blocked:
                    print(f"  [4H] {sym} suppressed — momentum gate (20d={mom30*100:.1f}%≤0)")
                else:
                    _exec(action, sym, c4, "4H"); fired = True
            time.sleep(0.2)
        return fired

    def _run_daily_check():
        nonlocal capital, base_notional, portfolio_peak, circuit_open
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n[{ts_str}] ── Daily check ──")

        # Refresh capital from exchange so sizing stays accurate as account grows
        try:
            from nexflow.exchange.bitget_order import get_account_balance
            live_bal = get_account_balance(client)
            if live_bal > 0 and abs(live_bal - capital) / capital > 0.005:  # >0.5% drift
                print(f"  Capital updated: ${capital:,.2f} → ${live_bal:,.2f}")
                capital = live_bal
                base_notional = capital / len(symbols)
                if live_bal > portfolio_peak:
                    portfolio_peak = live_bal
        except Exception:
            pass

        # News sentiment
        try:
            from nexflow.services.news.fetcher import fetch_fear_greed, fetch_crypto_news
            from nexflow.services.news.analyzer import analyze_sentiment
            fg   = fetch_fear_greed()
            news = fetch_crypto_news(limit=15)
            sent = analyze_sentiment(news, fg)
            icon = {"BULLISH":"↑","BEARISH":"↓","NEUTRAL":"→"}.get(sent.overall_bias,"?")
            print(f"  {icon} News: {sent.overall_bias} ({sent.confidence:.0%})  "
                  f"F&G:{sent.fear_greed_value}({sent.fear_greed_label})")
            if sent.suspend_new_longs:
                print("  ⚠  EXTREME EVENT — new long entries suspended today")
            suspend = sent.suspend_new_longs
        except Exception:
            suspend = False
        news_suspend[0] = suspend

        # ── Fetch all closes first ──
        closes: dict[str, tuple[int, float]] = {}
        for sym in symbols:
            result = _fetch_close(sym)
            if result:
                closes[sym] = result
            time.sleep(0.2)

        # Update regime with BTC close
        if "BTCUSDT" in closes:
            btc_regime.update(closes["BTCUSDT"][1])
        for sym, (ts_ms, close) in closes.items():
            coin_closes_live[sym].append(close)

        bear = btc_regime.is_bear
        now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        if bear:
            regime_str = "BEAR (AND-entry: BTC<SMA200+30d<-20%) — shorts mode"
        else:
            regime_str = f"BULL — longs mode (confirm streak={btc_regime._above_streak}d)"
        print(f"  Regime: {regime_str}")

        # ── Circuit breaker: track portfolio mark-to-market equity ──
        try:
            portfolio_equity = 0.0
            for sym in symbols:
                pos_info = adapter.get_position_info(sym) if hasattr(adapter, "get_position_info") else None
                if pos_info:
                    portfolio_equity += float(pos_info.get("unrealizedPL", 0))
            portfolio_equity += capital  # base capital + open P&L
            if portfolio_equity > portfolio_peak:
                portfolio_peak = portfolio_equity
                if circuit_open:
                    circuit_open = False
                    print("  ✓ Circuit breaker RESET — portfolio recovered")
            dd = (portfolio_peak - portfolio_equity) / portfolio_peak
            print(f"  Portfolio: ${portfolio_equity:,.0f}  peak=${portfolio_peak:,.0f}  DD={dd*100:.1f}%")
            if dd >= _CIRCUIT_BREAKER_DD and not circuit_open:
                circuit_open = True
                print(f"  ⚠  CIRCUIT BREAKER TRIPPED — DD={dd*100:.1f}% ≥ {_CIRCUIT_BREAKER_DD*100:.0f}%"
                      f" — no new entries until recovery")
            elif circuit_open:
                print(f"  ⚠  Circuit breaker active (DD={dd*100:.1f}%) — new entries paused")
        except Exception:
            pass  # never block trading on a metric failure

        # ── BEAR REGIME: TSMOM short management ──
        if bear:
            # Close all open longs (regime switch)
            for sym in list(coin_longs.keys()):
                if coin_longs[sym] and sym in closes:
                    _, close = closes[sym]
                    print(f"  [REGIME] Closing long {sym} — bear regime")
                    _exec("CLOSE_LONG", sym, close, "REGIME")
                    coin_longs[sym].clear()

            # Weekly short rebalance
            if (now_ts - last_tsmom_rebal_live[0]) >= 7 * _DAY_MS:
                last_tsmom_rebal_live[0] = now_ts
                _save_state()
                print("  [TSMOM] Weekly rebalance ...")

                # Score all coins by 126-day return
                scores: list[tuple[float, str]] = []
                for sym in symbols:
                    cl = coin_closes_live[sym]
                    if len(cl) >= 127:
                        ret = (cl[-1] - cl[-127]) / cl[-127]
                        scores.append((ret, sym))
                scores.sort()
                desired = {sym for ret, sym in scores if ret < -0.05}

                # Close shorts no longer desired
                for sym in list(live_shorts.keys()):
                    if sym not in desired and sym in closes:
                        _, close = closes[sym]
                        _exec_short("CLOSE_SHORT", sym, close)
                    time.sleep(0.2)

                # Open new shorts (blocked if circuit breaker active)
                if circuit_open:
                    print("  [CIRCUIT] New short entries paused — portfolio DD too high")
                else:
                    # Budget-aware batch sizing: ATR sizes are computed per-coin,
                    # then scaled down together so the whole batch fits inside
                    # actual free margin. Weakest momentum (most negative 126d
                    # return) gets priority — scores is sorted ascending.
                    to_open = [sym for _, sym in scores
                               if sym in desired and sym not in live_shorts
                               and sym in closes]
                    if to_open:
                        wanted = {sym: _atr_notional(sym) for sym in to_open}
                        total_wanted = sum(wanted.values())
                        avail = _available_margin()
                        budget = avail * _MARGIN_BUFFER if avail > 0 else total_wanted
                        scale = min(1.0, budget / total_wanted) if total_wanted > 0 else 0.0
                        if scale < 1.0:
                            print(f"  [TSMOM] Batch scaled to fit margin: "
                                  f"wanted ${total_wanted:,.0f}, free ${avail:,.0f} "
                                  f"→ {scale*100:.0f}% of ATR size per coin")
                        for sym in to_open:
                            _, close = closes[sym]
                            _exec_short("OPEN_SHORT", sym, close,
                                        notional=wanted[sym] * scale)
                            time.sleep(0.2)

                if live_shorts:
                    print(f"  [TSMOM] Shorts active: {', '.join(sorted(live_shorts))}")
                    missing = desired - set(live_shorts)
                    if missing:
                        print(f"  [TSMOM] Qualified but NOT opened: {', '.join(sorted(missing))}")
                elif desired:
                    print(f"  [TSMOM] {len(desired)} coins qualified but no shorts opened")
                else:
                    print("  [TSMOM] No coins qualify for shorting")
            else:
                days_since = (now_ts - last_tsmom_rebal_live[0]) / _DAY_MS
                next_rebal = 7 - days_since
                shorts_str = ", ".join(f"{s}({(closes[s][1]-v)/v*-100:+.1f}%)"
                                       for s, v in live_shorts.items() if s in closes)
                print(f"  [TSMOM] Next rebalance in {next_rebal:.0f}d  |  "
                      f"Shorts: {shorts_str or 'none'}")

        # ── BULL REGIME: close any open shorts, run long trio ──
        else:
            # Close all shorts if back in bull
            if live_shorts:
                print("  [REGIME] Bull regime — closing all shorts")
                for sym in list(live_shorts.keys()):
                    if sym in closes:
                        _, close = closes[sym]
                        _exec_short("CLOSE_SHORT", sym, close)
                    time.sleep(0.2)

            # Process long signals
            if circuit_open:
                print("  [CIRCUIT] New long entries paused — portfolio DD too high")
            any_sig = False
            for sym, (ts_ms, close) in closes.items():
                mom30 = btc_regime.mom_return(coin_closes_live[sym], BTCRegime._MOM_GATE_DAYS)
                mom_blocked = mom30 <= 0

                # EMA daily
                for sig in ema_strat.on_daily_close(sym, close, ts_ms):
                    if sig.action == "OPEN_LONG":
                        coin_longs[sym].add("EMA")
                    elif sig.action == "CLOSE_LONG":
                        coin_longs[sym].discard("EMA")
                    if sig.action == "OPEN_LONG" and (suspend or circuit_open):
                        print(f"  [EMA] {sym} suppressed — {'extreme event' if suspend else 'circuit breaker'}")
                    elif sig.action == "OPEN_LONG" and mom_blocked:
                        print(f"  [EMA] {sym} suppressed — momentum gate (20d={mom30*100:.1f}%≤0)")
                    else:
                        _exec(sig.action, sym, close, "EMA"); any_sig = True

                # MACD daily
                action = macd_strats[sym].update(close)
                if action:
                    if action == "OPEN_LONG":
                        coin_longs[sym].add("MACD")
                    elif action == "CLOSE_LONG":
                        coin_longs[sym].discard("MACD")
                    if action == "OPEN_LONG" and (suspend or circuit_open):
                        print(f"  [MACD] {sym} suppressed — {'extreme event' if suspend else 'circuit breaker'}")
                    elif action == "OPEN_LONG" and mom_blocked:
                        print(f"  [MACD] {sym} suppressed — momentum gate (20d={mom30*100:.1f}%≤0)")
                    else:
                        _exec(action, sym, close, "MACD"); any_sig = True

                # Update daily trend for 4H filter
                csigs = ema_strat.current_signals()
                ema4h_strats[sym].update_daily_trend(
                    1.0 if csigs.get(sym) == "LONG" else 0.0, 0.5)
                time.sleep(0.1)

            # 4H check (dedup-safe, shared with 6H checks)
            if _process_4h_signals(suspend):
                any_sig = True

            if not any_sig:
                print("  No long signals today.")

            ema_sigs = ema_strat.current_signals()
            longs = []
            for sym in symbols:
                e  = ema_sigs.get(sym, "?")
                m  = macd_strats[sym].current_signal()
                h4 = ema4h_strats[sym].current_signal()
                if e == "LONG" or m == "LONG" or h4 == "LONG":
                    longs.append(f"{sym}(E:{e[0]} M:{m[0]} 4:{h4[0]})")
            print(f"  Active longs: {', '.join(longs) or 'none'}")

    def _run_stop_check():
        """Every 6H: process new 4H signals + check 15% hard stop on all positions."""
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ── 4H signals (can open/close positions intraday, dedup-safe) ──
        if _process_4h_signals(news_suspend[0]):
            print(f"[{ts_str}] 4H signal(s) acted on intraday")

        # ── Hard stops only matter if something is open ──
        if not live_shorts and not any(coin_longs[s] for s in symbols):
            return
        stopped = []

        # ── Hard stop on shorts: price rose 15%+ above entry ──
        for sym in list(live_shorts.keys()):
            result = _fetch_close(sym)
            if result is None:
                continue
            _, current_price = result
            entry_price = live_shorts[sym]
            loss_pct = (current_price - entry_price) / entry_price
            if loss_pct >= _HARD_STOP_PCT:
                print(f"\n[{ts_str}] ⚠  HARD STOP SHORT {sym}: entry={entry_price:,.4f}  "
                      f"now={current_price:,.4f}  loss={loss_pct*100:.1f}% ≥ {_HARD_STOP_PCT*100:.0f}%")
                _exec_short("CLOSE_SHORT", sym, current_price)
                stopped.append(f"{sym}(short)")
            time.sleep(0.2)

        # ── Hard stop on longs: price fell 15%+ below entry ──
        for sym in list(symbols):
            if not coin_longs[sym]:
                continue
            result = _fetch_close(sym)
            if result is None:
                continue
            _, current_price = result
            # Get entry price from exchange
            try:
                from nexflow.exchange.bitget_order import get_position
                pos = get_position(adapter._client, sym)
                if pos is None:
                    coin_longs[sym].clear()
                    continue
                entry_price = float(pos.get("openPriceAvg", current_price))
            except Exception:
                continue
            loss_pct = (entry_price - current_price) / entry_price
            if loss_pct >= _HARD_STOP_PCT:
                print(f"\n[{ts_str}] ⚠  HARD STOP LONG  {sym}: entry={entry_price:,.4f}  "
                      f"now={current_price:,.4f}  loss={loss_pct*100:.1f}% ≥ {_HARD_STOP_PCT*100:.0f}%")
                _exec("CLOSE_LONG", sym, current_price, "STOP")
                coin_longs[sym].clear()
                stopped.append(f"{sym}(long)")
            time.sleep(0.2)

        if stopped:
            print(f"[{ts_str}] Hard-stopped: {', '.join(stopped)}")

    def _next_daily_check_time(now: "datetime") -> "datetime":
        """Return next midnight+05min UTC after now."""
        candidate = now.replace(hour=0, minute=5, second=0, microsecond=0)
        if candidate <= now:
            candidate = candidate + timedelta(days=1)
        return candidate

    # Run immediately then loop: hard-stop check every 6H, full check at midnight+5min UTC
    _run_daily_check()
    next_daily = _next_daily_check_time(datetime.now(timezone.utc))
    while True:
        now = datetime.now(timezone.utc)
        # Next 6H boundary
        next_6h = now + timedelta(hours=6)
        # Wake at whichever comes first: next 6H check or next daily check
        wake_at = min(next_6h, next_daily)
        sleep_s = max((wake_at - now).total_seconds(), 60.0)
        print(f"\n  Next stop-check: {next_6h.strftime('%H:%M UTC')}  |  "
              f"Next daily: {next_daily.strftime('%Y-%m-%d %H:%M UTC')}  "
              f"(sleeping {sleep_s/3600:.1f}h)")
        time.sleep(sleep_s)

        now = datetime.now(timezone.utc)
        if now >= next_daily:
            _run_daily_check()
            next_daily = _next_daily_check_time(now)
        else:
            _run_stop_check()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NexFlow Trio paper trader")
    parser.add_argument("--mode",    choices=["replay","live"], default="replay")
    parser.add_argument("--symbols", nargs="+", default=_SYMBOLS)
    parser.add_argument("--capital", type=float, default=5_000.0)
    parser.add_argument("--from",    dest="from_date", default="2024-01-01")
    parser.add_argument("--to",      dest="to_date",   default=None)
    args = parser.parse_args()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt   = (datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
               if args.to_date else datetime.now(timezone.utc))
    from_ts = int(from_dt.timestamp()*1000)
    to_ts   = int(to_dt.timestamp()*1000)

    if args.mode == "replay":
        run_replay(args.symbols, args.capital, from_ts, to_ts)
    else:
        run_live(args.symbols, args.capital)


if __name__ == "__main__":
    main()
