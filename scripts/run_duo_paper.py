#!/usr/bin/env python3
"""Paper-trade the EMA 8/21 + MACD Long-Only duo strategy.

Runs both strategies simultaneously on equal capital splits.
Each fires independent OPEN_LONG / CLOSE_LONG signals per coin.
A coin can be held by EMA, MACD, both, or neither.

REPLAY: uses cached daily parquet, logs all signals and equity.
LIVE  : polls Bitget daily close each hour, executes via BitgetPaperAdapter.

Deployed capital split:
  EMA 8/21 : half the capital, $cap/2/12 per coin
  MACD     : half the capital, $cap/2/12 per coin

Current signals (2026-06-03):
  EMA 8/21 : 1/12 LONG (BNB only) — slow, conservative
  MACD     : 10/12 LONG — already detecting recovery

Usage:
    python scripts/run_duo_paper.py --mode replay
    python scripts/run_duo_paper.py --mode replay --from 2024-01-01
    BITGET_PAPER=1 python scripts/run_duo_paper.py --mode live
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

from nexflow.services.strategy.ema_trend_strategy import EMATrendStrategy, EMASignal

_DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]
_CANDLE_DIR = _REPO_ROOT / "data" / "candles"
_DAY_MS     = 86_400_000
_TAKER_FEE  = 0.0006


# ---------------------------------------------------------------------------
# MACD signal engine (stateful per symbol)
# ---------------------------------------------------------------------------
class MACDStrategy:
    """Stateful MACD(12,26,9) long-only signal generator per symbol."""

    def __init__(self, symbols: list[str]) -> None:
        self._symbols  = symbols
        self._closes:   dict[str, list[float]] = {s: [] for s in symbols}
        self._positions: dict[str, bool] = {s: False for s in symbols}
        self._prev_above: dict[str, Optional[bool]] = {s: None for s in symbols}
        # EMA state
        self._e12: dict[str, float] = {}
        self._e26: dict[str, float] = {}
        self._sig: dict[str, float] = {}
        self._bars: dict[str, int]  = {s: 0 for s in symbols}

    def on_daily_close(self, symbol: str, close: float, ts: int) -> list:
        if symbol not in self._closes:
            return []
        self._bars[symbol] += 1
        n = self._bars[symbol]

        a12 = 2 / 13; a26 = 2 / 27; a9 = 2 / 10

        if symbol not in self._e12:
            self._e12[symbol] = close
            self._e26[symbol] = close
        else:
            self._e12[symbol] = a12 * close + (1 - a12) * self._e12[symbol]
            self._e26[symbol] = a26 * close + (1 - a26) * self._e26[symbol]

        macd_val = self._e12[symbol] - self._e26[symbol]

        if symbol not in self._sig:
            self._sig[symbol] = macd_val
        else:
            self._sig[symbol] = a9 * macd_val + (1 - a9) * self._sig[symbol]

        # Need warmup
        if n < 35:
            return []

        above = macd_val > self._sig[symbol]
        prev  = self._prev_above[symbol]
        signals = []

        if prev is not None and above != prev:
            if above and not self._positions[symbol]:
                signals.append({"symbol": symbol, "action": "OPEN_LONG",
                                 "price": close, "ts": ts, "reason": "MACD_cross_up"})
                self._positions[symbol] = True
            elif not above and self._positions[symbol]:
                signals.append({"symbol": symbol, "action": "CLOSE_LONG",
                                 "price": close, "ts": ts, "reason": "MACD_cross_down"})
                self._positions[symbol] = False

        self._prev_above[symbol] = above
        return signals

    def current_signals(self) -> dict[str, str]:
        out = {}
        for s in self._symbols:
            if self._bars[s] < 35:
                out[s] = "WARMUP"
            elif s in self._prev_above and self._prev_above[s] is not None:
                out[s] = "LONG" if self._prev_above[s] else "FLAT"
            else:
                out[s] = "WARMUP"
        return out

    @property
    def positions(self) -> dict[str, bool]:
        return dict(self._positions)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_daily(symbols, from_ts, to_ts):
    result = {}
    for sym in symbols:
        path = _CANDLE_DIR / f"{sym}_1D.parquet"
        if not path.exists():
            result[sym] = []; continue
        tbl = pq.read_table(path, columns=["open_time", "close"])
        rows = sorted(zip(tbl.column("open_time").to_pylist(),
                          tbl.column("close").to_pylist()))
        result[sym] = [(ts, float(c)) for ts, c in rows if from_ts <= ts <= to_ts]
    return result


def _fetch_daily_close(symbol: str) -> Optional[tuple[int, float]]:
    url = (f"https://api.bitget.com/api/v2/mix/market/history-candles"
           f"?symbol={symbol}&productType=USDT-FUTURES&granularity=1D&limit=2")
    headers = {"User-Agent": "NexFlow/1.0", "Accept": "application/json"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        rows = data.get("data", [])
        if not rows or data.get("code") != "00000": return None
        return int(rows[0][0]), float(rows[0][4])
    except Exception as exc:
        print(f"  [WARN] {symbol}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def run_replay(symbols, capital, from_ts, to_ts):
    print("Loading daily candles ...")
    history = _load_daily(symbols, from_ts, to_ts)
    n_bars = sum(len(v) for v in history.values())
    print(f"  {n_bars:,} bars across {len(symbols)} symbols")
    print()

    half_cap = capital / 2
    notional_ema  = half_cap / len(symbols)
    notional_macd = half_cap / len(symbols)

    ema_strat  = EMATrendStrategy(symbols=symbols)
    macd_strat = MACDStrategy(symbols=symbols)

    events = sorted(
        [(ts, sym, c) for sym, bars in history.items() for ts, c in bars]
    )

    # Track open positions per strategy
    ema_open:  dict[str, float] = {}   # {sym: entry_price}
    macd_open: dict[str, float] = {}

    ema_equity  = half_cap
    macd_equity = half_cap
    total_trades = 0

    for ts, sym, close in events:
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        for sig in ema_strat.on_daily_close(sym, close, ts):
            total_trades += 1
            if sig.action == "OPEN_LONG":
                ema_open[sym] = close
                ema_equity -= _TAKER_FEE * notional_ema
                print(f"  {dt}  [EMA]  BUY  {sym:<12} @ {close:>12,.4f}")
            elif sig.action == "CLOSE_LONG" and sym in ema_open:
                entry = ema_open.pop(sym)
                pnl   = (close - entry) / entry * notional_ema
                fee   = _TAKER_FEE * notional_ema
                ema_equity += pnl - fee
                pct = (close - entry) / entry * 100
                print(f"  {dt}  [EMA]  SELL {sym:<12} @ {close:>12,.4f}  {pct:+.1f}%  equity=${ema_equity:,.0f}")

        for sig in macd_strat.on_daily_close(sym, close, ts):
            total_trades += 1
            if sig["action"] == "OPEN_LONG":
                macd_open[sym] = close
                macd_equity -= _TAKER_FEE * notional_macd
                print(f"  {dt}  [MACD] BUY  {sym:<12} @ {close:>12,.4f}")
            elif sig["action"] == "CLOSE_LONG" and sym in macd_open:
                entry = macd_open.pop(sym)
                pnl   = (close - entry) / entry * notional_macd
                fee   = _TAKER_FEE * notional_macd
                macd_equity += pnl - fee
                pct = (close - entry) / entry * 100
                print(f"  {dt}  [MACD] SELL {sym:<12} @ {close:>12,.4f}  {pct:+.1f}%  equity=${macd_equity:,.0f}")

    # Mark to market
    last_prices = {sym: bars[-1][1] for sym, bars in history.items() if bars}
    ema_unreal  = sum((last_prices.get(s, e) - e) / e * notional_ema  for s, e in ema_open.items())
    macd_unreal = sum((last_prices.get(s, e) - e) / e * notional_macd for s, e in macd_open.items())

    total_equity = ema_equity + macd_equity + ema_unreal + macd_unreal
    net = total_equity - capital
    years = (to_ts - from_ts) / (1000 * 86400 * 365.25)
    cagr = (total_equity / capital) ** (1 / years) - 1 if years > 0 and total_equity > 0 else 0

    print()
    print("=" * 60)
    print(f"  EMA 8/21  : realised ${ema_equity:,.0f}  unrealised ${ema_unreal:+,.0f}")
    print(f"  MACD      : realised ${macd_equity:,.0f}  unrealised ${macd_unreal:+,.0f}")
    print(f"  TOTAL     : ${total_equity:,.0f}  (net ${net:+,.0f}, {net/capital*100:.1f}%)")
    print(f"  CAGR      : {cagr*100:.1f}%")
    print(f"  Signals   : {total_trades}")
    print()
    print("  EMA 8/21 current state:")
    for sym, state in ema_strat.current_signals().items():
        print(f"    {sym:<12} {'📈 LONG' if state=='LONG' else '⏸  FLAT'}")
    print()
    print("  MACD current state:")
    for sym, state in macd_strat.current_signals().items():
        print(f"    {sym:<12} {'📈 LONG' if state=='LONG' else '⏸  FLAT'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------
def run_live(symbols, capital):
    from nexflow.exchange.bitget_client import BitgetClient
    from nexflow.execution.adapter import BitgetPaperAdapter

    client  = BitgetClient.from_env()
    adapter = BitgetPaperAdapter(client)

    half_cap       = capital / 2
    notional_ema   = half_cap  / len(symbols)
    notional_macd  = half_cap  / len(symbols)

    print("EMA 8/21 + MACD Long-Only DUO — LIVE PAPER MODE")
    print(f"Capital: ${capital:,.0f}  (${half_cap:,.0f} per strategy, ${notional_ema:,.0f} per coin per strategy)")
    print()

    # Seed from history
    today_ts  = int(datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    seed_from = today_ts - 90 * _DAY_MS
    history   = _load_daily(symbols, seed_from, today_ts)

    ema_strat  = EMATrendStrategy(symbols=symbols)
    macd_strat = MACDStrategy(symbols=symbols)

    all_seed = sorted([(ts, sym, c) for sym, bars in history.items() for ts, c in bars])
    print(f"Seeding {len(all_seed)} historical bars ...")
    for ts, sym, c in all_seed:
        ema_strat.on_daily_close(sym, c, ts)
        macd_strat.on_daily_close(sym, c, ts)

    print("Seed complete. Current signals:")
    for sym in symbols:
        ema_state  = ema_strat.current_signals().get(sym, "?")
        macd_state = macd_strat.current_signals().get(sym, "?")
        print(f"  {sym:<12}  EMA:{ema_state:<6}  MACD:{macd_state}")
    print()

    def _exec(action, symbol, price, notional, source):
        try:
            qty = notional / price
            if action == "OPEN_LONG":
                adapter.on_entry(symbol, "long", qty, 0.0, 0.0, 0.0)
                print(f"  [{source}] BUY  {symbol} @ {price:,.4f}")
            elif action == "CLOSE_LONG":
                adapter.on_close(symbol, "long", qty, 0.0, f"{source}_cross")
                print(f"  [{source}] SELL {symbol} @ {price:,.4f}")
        except Exception as exc:
            print(f"  [ERROR] {source} {symbol} {action}: {exc}")

    def _seconds_until_next_daily_close(offset_minutes: int = 5) -> float:
        """Seconds until offset_minutes after next 00:00 UTC (daily candle close)."""
        now = datetime.now(timezone.utc)
        # Next midnight UTC
        from datetime import timedelta
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=offset_minutes, second=0, microsecond=0)
        return max((tomorrow - now).total_seconds(), 60.0)

    def _run_daily_check() -> None:
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n[{ts_str}] Daily candle close — fetching signals ...")

        # --- News & sentiment check ---
        sentiment = None
        try:
            from nexflow.services.news.fetcher import fetch_fear_greed, fetch_crypto_news
            from nexflow.services.news.analyzer import analyze_sentiment, sentiment_from_fear_greed_only
            fg   = fetch_fear_greed()
            news = fetch_crypto_news(limit=15)
            sentiment = analyze_sentiment(news, fg) or (sentiment_from_fear_greed_only(fg) if fg else None)
            if sentiment:
                bias_icon = {"BULLISH": "↑", "BEARISH": "↓", "NEUTRAL": "→"}.get(sentiment.overall_bias, "?")
                print(f"  {bias_icon} News: {sentiment.overall_bias} ({sentiment.confidence:.0%})  "
                      f"F&G:{sentiment.fear_greed_value}({sentiment.fear_greed_label})")
                print(f"    {sentiment.reasoning}")
                if sentiment.suspend_new_longs:
                    print("  ⚠  EXTREME EVENT — suspending new long entries today")
                if sentiment.key_events:
                    for ev in sentiment.key_events[:2]:
                        print(f"    • {ev}")
        except Exception as exc:
            print(f"  [news] skipped: {exc}")

        # --- Price signals ---
        any_signal = False
        for sym in symbols:
            result = _fetch_daily_close(sym)
            if result:
                ts_ms, close = result
                for sig in ema_strat.on_daily_close(sym, close, ts_ms):
                    if sig.action == "OPEN_LONG" and sentiment and sentiment.suspend_new_longs:
                        print(f"  [EMA] {sym} OPEN_LONG suppressed — extreme news event")
                    else:
                        _exec(sig.action, sym, close, notional_ema, "EMA")
                        any_signal = True
                for sig in macd_strat.on_daily_close(sym, close, ts_ms):
                    if sig["action"] == "OPEN_LONG" and sentiment and sentiment.suspend_new_longs:
                        print(f"  [MACD] {sym} OPEN_LONG suppressed — extreme news event")
                    else:
                        _exec(sig["action"], sym, close, notional_macd, "MACD")
                        any_signal = True
            time.sleep(0.25)  # gentle rate limiting

        if not any_signal:
            print("  No crossovers today. Positions unchanged.")

        print("  Current state:")
        for sym in symbols:
            e = ema_strat.current_signals().get(sym, "?")
            m = macd_strat.current_signals().get(sym, "?")
            if e == "LONG" or m == "LONG":
                print(f"    {sym:<12}  EMA:{e:<6}  MACD:{m}")
        longs = [s for s, v in ema_strat.positions.items() if v] + \
                [s for s, v in macd_strat.positions.items() if v]
        print(f"  Active longs: {sorted(set(longs)) or 'none (all flat)'}")

    # Run immediately on startup (catch up if missed today's close)
    _run_daily_check()

    # Then sleep until 5 minutes after each subsequent midnight UTC
    while True:
        sleep_secs = _seconds_until_next_daily_close(offset_minutes=5)
        next_run = datetime.now(timezone.utc)
        from datetime import timedelta
        next_run_dt = (next_run + timedelta(seconds=sleep_secs)).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n  Sleeping {sleep_secs/3600:.1f}h — next check at {next_run_dt}")
        time.sleep(sleep_secs)
        _run_daily_check()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["replay", "live"], default="replay")
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--capital", type=float, default=200_000.0)
    parser.add_argument("--from",    dest="from_date", default="2021-01-01")
    parser.add_argument("--to",      dest="to_date",   default=None)
    args = parser.parse_args()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt   = (datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
               if args.to_date else datetime.now(timezone.utc))
    from_ts = int(from_dt.timestamp() * 1000)
    to_ts   = int(to_dt.timestamp() * 1000)

    if args.mode == "replay":
        run_replay(args.symbols, args.capital, from_ts, to_ts)
    else:
        run_live(args.symbols, args.capital)


if __name__ == "__main__":
    main()
