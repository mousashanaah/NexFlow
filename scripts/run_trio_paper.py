#!/usr/bin/env python3
"""NexFlow Trio — paper-trade all three GO strategies simultaneously.

  Strategy 1: EMA 8/21 Daily Long-Only     (CAGR 24%, DD 11%, PF 1.95)
  Strategy 2: MACD 12/26/9 Daily Long-Only (CAGR 23%, DD 17%, PF 1.59)
  Strategy 3: 4H EMA 5/13 Long-Only        (CAGR 20%, DD  9%, PF 1.46)

Confluence position sizing (backtested CAGR 32.8%, DD 34.6%):
  1 strategy signals LONG  → 1.0× base notional per coin
  2 strategies agree       → 1.5× base notional per coin
  All 3 agree              → 2.0× base notional per coin

Base notional = capital / 12 coins.

Checks once per day at 00:05 UTC (5 min after daily candle close).
4H EMA checks are embedded in the same daily loop using latest 4H close.

News & sentiment: free rule-based engine (Fear&Greed + Google News).
Optional: set ANTHROPIC_API_KEY to upgrade to Claude analysis.

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
_TAKER_FEE  = 0.0006
_DAY_MS     = 86_400_000
_HOUR_MS    = 3_600_000


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
# BTC regime tracker (SMA200 master switch for V3 system)
# ---------------------------------------------------------------------------
class BTCRegime:
    """Tracks BTC SMA200. Returns bear=True when BTC < its 200-day SMA."""

    def __init__(self):
        self._closes: list[float] = []
        self._bear = False  # default: assume bull until enough data

    def update(self, close: float) -> None:
        self._closes.append(close)
        if len(self._closes) >= 200:
            sma200 = sum(self._closes[-200:]) / 200
            self._bear = close < sma200

    @property
    def is_bear(self) -> bool:
        return self._bear

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
    print(f"NexFlow Trio V3 — REPLAY  |  capital=${capital:,.0f}  |  ${base_notional:,.0f}/coin")
    print(f"Period: {datetime.fromtimestamp(from_ts/1000,tz=timezone.utc).date()} → "
          f"{datetime.fromtimestamp(to_ts/1000,tz=timezone.utc).date()}")
    print(f"Regime: BTC SMA200 master switch  |  TSMOM shorts in bear market")
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
                if n_long == 0 and in_pos:
                    _close_long(sym, close, ts, "EMA/MACD")
                    equity = equity_ref[0]
                elif n_long > 0 and not in_pos:
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
                if n_long == 0 and in_pos:
                    _close_long(sym, close, ts, "4H")
                    equity = equity_ref[0]
                elif n_long > 0 and not in_pos:
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

    base_notional = capital / len(symbols)  # confluence sizing base: capital/12
    print("NexFlow Trio V3 — LIVE PAPER MODE")
    print(f"Symbols        : {len(symbols)} coins")
    print(f"Capital        : ${capital:,.0f}  (${base_notional:.2f} base/coin)")
    print(f"Confluence     : 1 strat=1×  2 strats=1.5×  3 strats=2×")
    print(f"Regime gate    : BTC SMA200 — bear→short, bull→long")
    print(f"Fee            : {_TAKER_FEE*100:.2f}% taker")
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
        for ts, c in _resample_4h(h1):
            ema4h_strats[sym].update(c)
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

    def _confluence_notional(sym: str) -> float:
        """Return position size based on how many strategies agree on this coin."""
        n = len(coin_longs[sym])
        mult = {0: 0.0, 1: 1.0, 2: 1.5, 3: 2.0}.get(n, 2.0)
        return base_notional * mult

    def _exec(action, sym, price, src):
        notional = _confluence_notional(sym)
        qty = notional / price if price > 0 else 0
        if qty <= 0: return
        try:
            if action == "OPEN_LONG":
                adapter.on_entry(sym, "long", qty, 0.0, 0.0, 0.0)
                n = len(coin_longs[sym])
                mult = {1:1.0, 2:1.5, 3:2.0}.get(n, 1.0)
                print(f"  [{src}] BUY  {sym:<12} @ {price:,.4f}  "
                      f"({n} strat{'s' if n>1 else ''} agree → {mult}× = ${notional:,.0f})")
            elif action == "CLOSE_LONG":
                adapter.on_close(sym, "long", qty, 0.0, f"{src}_cross")
                print(f"  [{src}] SELL {sym:<12} @ {price:,.4f}")
        except Exception as e:
            print(f"  [ERROR] {src} {sym} {action}: {e}")

    last_tsmom_rebal_live = [0]   # ts of last weekly short rebalance
    live_shorts: dict[str, float] = {}  # sym → entry_price (shorts open on exchange)

    def _exec_short(action: str, sym: str, price: float) -> None:
        """Place or close a short position on the exchange."""
        qty = base_notional / price if price > 0 else 0
        if qty <= 0: return
        try:
            if action == "OPEN_SHORT":
                adapter.on_entry(sym, "short", qty, 0.0, 0.0, 0.0)
                live_shorts[sym] = price
                print(f"  [SHORT] SELL {sym:<12} @ {price:,.4f}  size=${base_notional:,.0f}")
            elif action == "CLOSE_SHORT":
                adapter.on_close(sym, "short", qty, 0.0, "tsmom_exit")
                live_shorts.pop(sym, None)
                print(f"  [SHORT] BUY  {sym:<12} @ {price:,.4f}  (closed)")
        except Exception as e:
            print(f"  [ERROR] SHORT {sym} {action}: {e}")

    def _run_daily_check():
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n[{ts_str}] ── Daily check ──")

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
        regime_str = "BEAR (BTC<SMA200) — shorts mode" if bear else "BULL (BTC>SMA200) — longs mode"
        print(f"  Regime: {regime_str}")

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

                # Open new shorts
                for sym in desired:
                    if sym not in live_shorts and sym in closes:
                        _, close = closes[sym]
                        _exec_short("OPEN_SHORT", sym, close)
                    time.sleep(0.2)

                if desired:
                    print(f"  [TSMOM] Shorts active: {', '.join(sorted(desired))}")
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
            any_sig = False
            for sym, (ts_ms, close) in closes.items():
                # EMA daily
                for sig in ema_strat.on_daily_close(sym, close, ts_ms):
                    if sig.action == "OPEN_LONG":
                        coin_longs[sym].add("EMA")
                    elif sig.action == "CLOSE_LONG":
                        coin_longs[sym].discard("EMA")
                    if sig.action == "OPEN_LONG" and suspend:
                        print(f"  [EMA] {sym} suppressed — extreme event")
                    else:
                        _exec(sig.action, sym, close, "EMA"); any_sig = True

                # MACD daily
                action = macd_strats[sym].update(close)
                if action:
                    if action == "OPEN_LONG":
                        coin_longs[sym].add("MACD")
                    elif action == "CLOSE_LONG":
                        coin_longs[sym].discard("MACD")
                    if action == "OPEN_LONG" and suspend:
                        print(f"  [MACD] {sym} suppressed — extreme event")
                    else:
                        _exec(action, sym, close, "MACD"); any_sig = True

                # Update daily trend for 4H filter
                csigs = ema_strat.current_signals()
                ema4h_strats[sym].update_daily_trend(
                    1.0 if csigs.get(sym) == "LONG" else 0.0, 0.5)
                time.sleep(0.1)

            # 4H check
            for sym in symbols:
                bars_1h = _fetch_1h_recent(sym, 40)
                if not bars_1h: continue
                for ts4, c4 in _resample_4h(bars_1h)[-3:]:
                    action = ema4h_strats[sym].update(c4)
                    if action:
                        if action == "OPEN_LONG":
                            coin_longs[sym].add("4H")
                        elif action == "CLOSE_LONG":
                            coin_longs[sym].discard("4H")
                        if action == "OPEN_LONG" and suspend:
                            print(f"  [4H] {sym} suppressed — extreme event")
                        else:
                            _exec(action, sym, c4, "4H"); any_sig = True
                time.sleep(0.2)

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

    # Run immediately then sleep until each midnight+5min UTC
    _run_daily_check()
    while True:
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
        sleep_s = max((tomorrow - now).total_seconds(), 60.0)
        print(f"\n  Next check: {tomorrow.strftime('%Y-%m-%d %H:%M UTC')}  "
              f"(sleeping {sleep_s/3600:.1f}h)")
        time.sleep(sleep_s)
        _run_daily_check()


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
