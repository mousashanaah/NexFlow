#!/usr/bin/env python3
"""NexFlow V8.63 crypto engine — reusable class (full feature parity).

This is the EXACT V8.63 live crypto book extracted from run_trio_paper.run_live
so that V9 can drive it with confidence-allocated capital and there is ZERO
behavioural drift from the backtested system.

Features (all of V8.63):
  • BTC AND-entry asymmetric regime (BTC<SMA200 AND 30d<-20%, 10d confirm exit)
  • Confluence longs: EMA 8/21 daily + MACD daily + 4H EMA 5/13
  • ATR vol-adjusted sizing (target 1%/day risk, cap 2x base, floor 0.5x base)
  • Confluence multiplier: 1 strat=1x, 2=1.5x, 3=2x
  • 20d momentum gate on new longs
  • TSMOM shorts in bear regime (14d rebalance, 126d momentum, budget-aware,
    bearish-confluence sizing — backtested +$3,284/+Sharpe in full V9 vs weekly/flat)
  • 15% hard stops (intraday, checked on stop_check)
  • 20% portfolio circuit breaker (pause new entries)
  • Free-margin clamping on every entry

The class is capital-aware: daily_check(capital=...) re-bases sizing each call,
so V9's allocator can hand it crypto_capital that changes day to day.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from run_trio_paper import (
    MACDState, EMA4HState, BTCRegime,
    _load_candles, _resample_4h, _fetch_close, _fetch_1h_recent,
    _DAY_MS,
)
from nexflow.services.strategy.ema_trend_strategy import EMATrendStrategy

_TAKER_FEE = 0.0006
_HARD_STOP_PCT = 0.15
_TARGET_RISK = 0.01
_ATR_WINDOW = 14
_CIRCUIT_BREAKER_DD = 0.20
_MARGIN_BUFFER = 0.95
_DAY_MS_LOCAL = 86_400_000


def _fetch_daily_candles(symbol: str, limit: int = 400) -> list[tuple[int, float]]:
    """Fetch recent fully-CLOSED daily candles (ts_ms, close) from Bitget, asc.

    Used to backfill any gap between stale local parquet data and today so the
    regime / SMA / momentum series never has a hole. Mirrors _fetch_close's
    endpoint and 'fully closed' guard.
    """
    url = (f"https://api.bitget.com/api/v2/mix/market/history-candles"
           f"?symbol={symbol}&productType=USDT-FUTURES&granularity=1D&limit={limit}")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "NexFlow/1.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        if data.get("code") != "00000" or not data.get("data"):
            return []
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        out = []
        for row in data["data"]:
            ts = int(row[0])
            if ts + _DAY_MS_LOCAL <= now_ms:   # only fully-closed candles
                out.append((ts, float(row[4])))
        return sorted(out)
    except Exception as e:
        print(f"  [WARN] daily backfill {symbol}: {e}")
        return []


class CryptoBookV863:
    """Full V8.63 crypto engine as a reusable component."""

    def __init__(self, symbols, adapter, client, state_file: Path):
        self.symbols = symbols
        self.adapter = adapter
        self.client = client
        self.state_file = state_file

        self.ema_strat = EMATrendStrategy(symbols=symbols, fast=8, slow=21)
        self.macd = {s: MACDState() for s in symbols}
        self.ema4h = {s: EMA4HState() for s in symbols}
        self.regime = BTCRegime()
        self.coin_closes: dict[str, list[float]] = {s: [] for s in symbols}
        self.coin_longs: dict[str, set] = {s: set() for s in symbols}
        self.last_4h_ts: dict[str, int] = {s: 0 for s in symbols}
        self.last_daily_ts: dict[str, int] = {s: 0 for s in symbols}  # daily-bar dedup
        self.live_shorts: dict[str, float] = {}
        self.last_tsmom_rebal = 0
        self.portfolio_peak = 0.0
        self.circuit_open = False

        # set per daily_check
        self._capital = 0.0
        self._base_notional = 0.0

    # ── balance / sizing ──────────────────────────────────────────────────────
    def _available_margin(self) -> float:
        try:
            from nexflow.exchange.bitget_order import get_account_balance
            return max(0.0, get_account_balance(self.client))
        except Exception:
            return 0.0

    def _atr_notional(self, sym: str, mult: float = 1.0) -> float:
        closes = self.coin_closes.get(sym, [])
        if len(closes) < _ATR_WINDOW + 1:
            return self._base_notional * mult
        rets = [(closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(len(closes) - _ATR_WINDOW, len(closes))]
        mean_r = sum(rets) / len(rets)
        vol = (sum((r - mean_r) ** 2 for r in rets) / len(rets)) ** 0.5
        if vol <= 0:
            return self._base_notional * mult
        sized = (_TARGET_RISK * self._capital) / vol
        return max(self._base_notional * 0.5, min(sized * mult, self._base_notional * 2.0))

    def _confluence_notional(self, sym: str) -> float:
        n = len(self.coin_longs[sym])
        mult = {0: 0.0, 1: 1.0, 2: 1.5, 3: 2.0}.get(n, 2.0)
        return self._atr_notional(sym, mult)

    def _short_confluence_notional(self, sym: str) -> float:
        """ATR notional scaled by bearish confluence (mirror of long confluence).
        Counts how many of EMA(daily)/MACD/4H are NOT in LONG → size up shorts
        confirmed bearish by multiple signals. {1:1x, 2:1.5x, 3:2x}."""
        csigs = self.ema_strat.current_signals()
        n_short = sum([
            csigs.get(sym) != "LONG",          # daily EMA bearish/flat
            not self.macd[sym].position,        # MACD bearish/flat
            not self.ema4h[sym].position,       # 4H bearish/flat
        ])
        mult = {1: 1.0, 2: 1.5, 3: 2.0}.get(n_short, 1.0)
        return self._atr_notional(sym, mult)

    # ── state persistence ─────────────────────────────────────────────────────
    def load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            st = json.loads(self.state_file.read_text())
            self.last_tsmom_rebal = int(st.get("crypto_last_tsmom_rebal", 0))
        except Exception:
            pass

    def save_state(self, extra: dict | None = None) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if self.state_file.exists():
                try:
                    data = json.loads(self.state_file.read_text())
                except Exception:
                    data = {}
            data["crypto_last_tsmom_rebal"] = self.last_tsmom_rebal
            if extra:
                data.update(extra)
            self.state_file.write_text(json.dumps(data))
        except Exception:
            pass

    # ── seeding & reconciliation ──────────────────────────────────────────────
    def seed(self, seed_from: int, today: int) -> None:
        for s in self.symbols:
            last_ts = 0
            for ts, c in _load_candles(s, "1D", seed_from, today):
                self.ema_strat.on_daily_close(s, c, ts)
                self.macd[s].update(c)
                self.coin_closes[s].append(c)
                if s == "BTCUSDT":
                    self.regime.update(c)
                last_ts = ts

            # ── gap backfill: pull any CLOSED daily bars after the parquet end ──
            # so regime/SMA/momentum never run on a hole when local data is stale.
            n_filled = 0
            for ts, c in _fetch_daily_candles(s):
                if ts > last_ts:
                    self.ema_strat.on_daily_close(s, c, ts)
                    self.macd[s].update(c)
                    self.coin_closes[s].append(c)
                    if s == "BTCUSDT":
                        self.regime.update(c)
                    last_ts = ts
                    n_filled += 1
            self.last_daily_ts[s] = last_ts
            if n_filled:
                print(f"  [backfill] {s}: +{n_filled} daily bar(s) from API "
                      f"(local data was stale)")

            for ts, c in _resample_4h(_load_candles(s, "1H", seed_from, today)):
                self.ema4h[s].update(c)
                self.last_4h_ts[s] = ts
            csigs = self.ema_strat.current_signals()
            if s in csigs:
                self.ema4h[s].update_daily_trend(1.0 if csigs[s] == "LONG" else 0.0, 0.5)
            time.sleep(0.15)

    def reconcile(self) -> int:
        from nexflow.exchange.bitget_order import get_position
        n = 0
        for s in self.symbols:
            try:
                pos = get_position(self.client, s)
            except Exception:
                pos = None
            if not pos:
                time.sleep(0.1)
                continue
            side = pos.get("holdSide", "")
            entry = float(pos.get("openPriceAvg", 0))
            if entry <= 0:
                time.sleep(0.1)
                continue
            if side == "short":
                self.live_shorts[s] = entry
                print(f"  Restored CRYPTO short {s} @ {entry:,.4f}")
                n += 1
            elif side == "long":
                self.coin_longs[s] = {"EMA", "MACD", "4H"}
                print(f"  Restored CRYPTO long  {s} @ {entry:,.4f}")
                n += 1
            time.sleep(0.1)
        return n

    # ── execution ─────────────────────────────────────────────────────────────
    def _exec(self, action, sym, price, src):
        from nexflow.exchange.bitget_order import get_position
        try:
            if action == "OPEN_LONG":
                notional = self._confluence_notional(sym)
                avail = self._available_margin()
                if avail > 0 and notional > avail * _MARGIN_BUFFER:
                    clamped = avail * _MARGIN_BUFFER
                    if clamped < notional * 0.25:
                        print(f"  [{src}] {sym} skipped — free margin ${avail:,.2f} "
                              f"too low for ${notional:,.2f}")
                        return
                    notional = clamped
                if notional < 5.0:
                    print(f"  [{src}] {sym} skipped — ${notional:,.2f} < $5 min")
                    return
                qty = notional / price if price > 0 else 0
                if qty <= 0:
                    return
                res = self.adapter.on_entry(sym, "long", qty, 0.0, 0.0, 0.0)
                if res is not None and not getattr(res, "accepted", True):
                    print(f"  [ERROR] {src} BUY {sym} rejected: {getattr(res,'note','?')}")
                    return
                n = len(self.coin_longs[sym])
                mult = {1: 1.0, 2: 1.5, 3: 2.0}.get(n, 1.0)
                print(f"  [{src}] BUY  {sym:<10} @ {price:,.4f} "
                      f"({n} strat → {mult}× = ${notional:,.2f} ATR-sized)")
            elif action == "CLOSE_LONG":
                pos = get_position(self.client, sym)
                qty = float(pos.get("total", 0)) if pos else 0.0
                if qty <= 0:
                    self.coin_longs[sym].clear()
                    return
                self.adapter.on_close(sym, "long", qty, price, f"{src}_cross")
                print(f"  [{src}] SELL {sym:<10} @ {price:,.4f} qty={qty}")
        except Exception as e:
            print(f"  [ERROR] {src} {sym} {action}: {e}")

    def _exec_short(self, action, sym, price, notional=None):
        from nexflow.exchange.bitget_order import get_position
        try:
            if action == "OPEN_SHORT":
                if notional is None:
                    notional = self._atr_notional(sym)
                    avail = self._available_margin()
                    if avail > 0 and notional > avail * _MARGIN_BUFFER:
                        notional = avail * _MARGIN_BUFFER
                if notional < 5.0:
                    print(f"  [SHORT] {sym} skipped — ${notional:,.2f} < $5 min")
                    return
                qty = notional / price if price > 0 else 0
                if qty <= 0:
                    return
                res = self.adapter.on_entry(sym, "short", qty, 0.0, 0.0, 0.0)
                if res is not None and not getattr(res, "accepted", True):
                    print(f"  [ERROR] SHORT {sym} rejected: {getattr(res,'note','?')}")
                    return
                self.live_shorts[sym] = price
                print(f"  [SHORT] SELL {sym:<10} @ {price:,.4f} size=${notional:,.2f}")
            elif action == "CLOSE_SHORT":
                pos = get_position(self.client, sym)
                qty = float(pos.get("total", 0)) if pos else 0.0
                if qty <= 0:
                    self.live_shorts.pop(sym, None)
                    return
                self.adapter.on_close(sym, "short", qty, price, "tsmom_exit")
                self.live_shorts.pop(sym, None)
                print(f"  [SHORT] BUY  {sym:<10} @ {price:,.4f} (closed)")
        except Exception as e:
            print(f"  [ERROR] SHORT {sym} {action}: {e}")

    def _process_4h_signals(self, suspend: bool = False) -> bool:
        if self.regime.is_bear:
            return False
        fired = False
        for sym in self.symbols:
            bars = _fetch_1h_recent(sym, 40)
            if not bars:
                continue
            mom30 = self.regime.mom_return(self.coin_closes[sym], BTCRegime._MOM_GATE_DAYS)
            mom_blocked = mom30 <= 0
            for ts4, c4 in _resample_4h(bars):
                if ts4 <= self.last_4h_ts[sym]:
                    continue
                self.last_4h_ts[sym] = ts4
                action = self.ema4h[sym].update(c4)
                if not action:
                    continue
                if action == "OPEN_LONG":
                    self.coin_longs[sym].add("4H")
                elif action == "CLOSE_LONG":
                    self.coin_longs[sym].discard("4H")
                if action == "OPEN_LONG" and (suspend or self.circuit_open):
                    print(f"  [4H] {sym} suppressed — "
                          f"{'extreme event' if suspend else 'circuit breaker'}")
                elif action == "OPEN_LONG" and mom_blocked:
                    print(f"  [4H] {sym} suppressed — momentum gate ({mom30*100:.1f}%≤0)")
                else:
                    self._exec(action, sym, c4, "4H")
                    fired = True
            time.sleep(0.2)
        return fired

    # ── daily check (full V8.63) ──────────────────────────────────────────────
    def daily_check(self, capital: float, suspend: bool = False) -> None:
        self._capital = capital
        self._base_notional = capital / len(self.symbols)
        now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        # price = current close for ALL symbols (used for execution every run);
        # new_closes = only genuinely NEW daily bars (used to advance indicators
        # exactly once — prevents double-ingest on same-day re-runs/restarts).
        price: dict[str, float] = {}
        new_closes: dict[str, float] = {}
        for s in self.symbols:
            r = _fetch_close(s)
            if r:
                ts, c = int(r[0]), float(r[1])
                price[s] = c
                if ts > self.last_daily_ts.get(s, 0):
                    new_closes[s] = c
                    self.last_daily_ts[s] = ts
            time.sleep(0.2)
        if "BTCUSDT" in new_closes:
            self.regime.update(new_closes["BTCUSDT"])
        for s, c in new_closes.items():
            self.coin_closes[s].append(c)

        bear = self.regime.is_bear
        print(f"  Crypto regime: {'BEAR (shorts mode)' if bear else 'BULL (longs mode)'}")

        # circuit breaker (crypto book equity)
        try:
            eq = capital
            for s in self.symbols:
                info = (self.adapter.get_position_info(s)
                        if hasattr(self.adapter, "get_position_info") else None)
                if info:
                    eq += float(info.get("unrealizedPL", 0))
            if eq > self.portfolio_peak:
                self.portfolio_peak = eq
                if self.circuit_open:
                    self.circuit_open = False
                    print("  ✓ Circuit breaker RESET")
            dd = (self.portfolio_peak - eq) / self.portfolio_peak if self.portfolio_peak else 0
            if dd >= _CIRCUIT_BREAKER_DD and not self.circuit_open:
                self.circuit_open = True
                print(f"  ⚠ CIRCUIT BREAKER TRIPPED — DD={dd*100:.1f}%")
        except Exception:
            pass

        if bear:
            self._bear_logic(new_closes, now_ts, price)
        else:
            self._bull_logic(new_closes, suspend, price)
        self.save_state()

    def _bear_logic(self, new_closes, now_ts, price):
        # close longs — regime-driven, use current price so it runs every check
        for s in list(self.coin_longs):
            if self.coin_longs[s] and s in price:
                self._exec("CLOSE_LONG", s, price[s], "REGIME")
                self.coin_longs[s].clear()
        # TSMOM short rebalance (14d cadence — backtested best: less churn,
        # lets winning shorts run; vs weekly = +$3,284 / +Sharpe in full V9)
        if (now_ts - self.last_tsmom_rebal) >= 14 * _DAY_MS:
            self.last_tsmom_rebal = now_ts
            scores = []
            for s in self.symbols:
                cl = self.coin_closes[s]
                if len(cl) >= 127:
                    scores.append(((cl[-1] - cl[-127]) / cl[-127], s))
            scores.sort()
            desired = {s for ret, s in scores if ret < -0.05}
            for s in list(self.live_shorts):
                if s not in desired and s in price:
                    self._exec_short("CLOSE_SHORT", s, price[s])
                time.sleep(0.2)
            if self.circuit_open:
                print("  [CIRCUIT] short entries paused")
            else:
                to_open = [s for _, s in scores
                           if s in desired and s not in self.live_shorts and s in price]
                if to_open:
                    wanted = {s: self._short_confluence_notional(s) for s in to_open}
                    total = sum(wanted.values())
                    avail = self._available_margin()
                    budget = avail * _MARGIN_BUFFER if avail > 0 else total
                    scale = min(1.0, budget / total) if total > 0 else 0.0
                    for s in to_open:
                        self._exec_short("OPEN_SHORT", s, price[s], notional=wanted[s] * scale)
                        time.sleep(0.2)
            print(f"  [TSMOM] shorts: {', '.join(sorted(self.live_shorts)) or 'none'}")

    def _bull_logic(self, new_closes, suspend, price):
        # close any shorts — regime-driven, use current price so it runs every check
        if self.live_shorts:
            for s in list(self.live_shorts):
                if s in price:
                    self._exec_short("CLOSE_SHORT", s, price[s])
                time.sleep(0.2)
        if self.circuit_open:
            print("  [CIRCUIT] long entries paused")
        # entries — indicator-driven, only advance on genuinely new daily bars
        for s, c in new_closes.items():
            mom30 = self.regime.mom_return(self.coin_closes[s], BTCRegime._MOM_GATE_DAYS)
            mom_blocked = mom30 <= 0
            for sig in self.ema_strat.on_daily_close(s, c, int(datetime.now(timezone.utc).timestamp()*1000)):
                if sig.action == "OPEN_LONG":
                    self.coin_longs[s].add("EMA")
                elif sig.action == "CLOSE_LONG":
                    self.coin_longs[s].discard("EMA")
                if sig.action == "OPEN_LONG" and (suspend or self.circuit_open):
                    pass
                elif sig.action == "OPEN_LONG" and mom_blocked:
                    print(f"  [EMA] {s} suppressed — momentum gate")
                else:
                    self._exec(sig.action, s, c, "EMA")
            action = self.macd[s].update(c)
            if action:
                if action == "OPEN_LONG":
                    self.coin_longs[s].add("MACD")
                elif action == "CLOSE_LONG":
                    self.coin_longs[s].discard("MACD")
                if action == "OPEN_LONG" and (suspend or self.circuit_open):
                    pass
                elif action == "OPEN_LONG" and mom_blocked:
                    print(f"  [MACD] {s} suppressed — momentum gate")
                else:
                    self._exec(action, s, c, "MACD")
            csigs = self.ema_strat.current_signals()
            self.ema4h[s].update_daily_trend(1.0 if csigs.get(s) == "LONG" else 0.0, 0.5)
            time.sleep(0.1)
        self._process_4h_signals(suspend)

    # ── intraday stop check (every 6h) ────────────────────────────────────────
    def stop_check(self, suspend: bool = False) -> None:
        from nexflow.exchange.bitget_order import get_position
        self._process_4h_signals(suspend)
        if not self.live_shorts and not any(self.coin_longs[s] for s in self.symbols):
            return
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        for sym in list(self.live_shorts):
            r = _fetch_close(sym)
            if r is None:
                continue
            price = r[1]
            entry = self.live_shorts[sym]
            if (price - entry) / entry >= _HARD_STOP_PCT:
                print(f"[{ts_str}] ⚠ HARD STOP SHORT {sym}")
                self._exec_short("CLOSE_SHORT", sym, price)
            time.sleep(0.2)
        for sym in list(self.symbols):
            if not self.coin_longs[sym]:
                continue
            r = _fetch_close(sym)
            if r is None:
                continue
            price = r[1]
            try:
                pos = get_position(self.client, sym)
                if pos is None:
                    self.coin_longs[sym].clear()
                    continue
                entry = float(pos.get("openPriceAvg", price))
            except Exception:
                continue
            if (entry - price) / entry >= _HARD_STOP_PCT:
                print(f"[{ts_str}] ⚠ HARD STOP LONG {sym}")
                self._exec("CLOSE_LONG", sym, price, "STOP")
                self.coin_longs[sym].clear()
            time.sleep(0.2)
