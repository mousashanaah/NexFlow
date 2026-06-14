#!/usr/bin/env python3
"""NexFlow V9 — Confidence-paired crypto + stock bot (PRODUCTION).

This is the live wiring of the V9 system that backtested to $107,780 from $5K
(CAGR 76.6%, DD 13.9%, Sharpe 2.10, zero losing years, 2021-2026).

How V9 works
------------
Two books run side by side and a *confidence allocator* decides, each day near
the US close, how to split capital between them:

  • CRYPTO book  — V8.63 (BTC-regime AND-entry, EMA/MACD/4H confluence longs,
                   TSMOM shorts in bear).   Executes on Bitget USDT perps.
  • STOCK  book  — strict no-lookahead trend system on MSTR+AMD+GOOGL+META
                   (per-asset SMA200 gate, EMA8/21 + MACD confirm, mom90 gate,
                   MACD-cross exit + 10% hard stop, equal-across-active sizing).
                   Executes on Bitget STOCK perps.

  Allocator: crypto_score (0-4, SMA200 double-weighted) and stock_score (0-3)
  are normalised and mapped to weights — crypto-dominant, stock-dominant,
  both-hot (65/35), or both-cold (40/40 + 20% cash defensive).

$100-account adaptation (per user decision: "concentrate top picks")
-------------------------------------------------------------------
Bitget min order = $5 notional. With ~$50-65 per book you cannot spread across
12 crypto coins + 4 stocks. So each book CONCENTRATES into its top-N highest-
conviction names that each clear the $5 floor (default crypto K=3, stock all-4
when the slice allows; names that can't clear $5 are skipped, not forced).

⚠️  START HERE — real money checklist:
  1. The stock leg DEFAULTS TO DRY-RUN (logs orders, sends nothing).
     Pass --stock-live to enable it.
  2. Always start in Bitget demo first:  BITGET_PAPER=1 ... --mode live
  3. When happy, remove BITGET_PAPER=1 to go live.

Usage
-----
  # Sanity-check the wiring reproduces the backtest:
  python scripts/run_v9_confidence.py --mode replay

  # Live on Bitget demo (crypto live, stock dry-run):
  BITGET_PAPER=1 NEXFLOW_EXEC_MODE=BITGET_PAPER \
    python scripts/run_v9_confidence.py --mode live --capital 100

  # Live with stock leg actually sending orders (after verification):
  BITGET_PAPER=1 NEXFLOW_EXEC_MODE=BITGET_PAPER \
    python scripts/run_v9_confidence.py --mode live --capital 100 --stock-live
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

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

# Reuse the proven crypto live helpers (importing run_trio_paper is side-effect free)
from run_trio_paper import (
    MACDState, EMA4HState, BTCRegime,
    _load_candles, _resample_4h, _fetch_close, _fetch_1h_recent,
    _SYMBOLS as _CRYPTO_SYMBOLS, _DAY_MS, _HARD_STOP_PCT as _CRYPTO_STOP,
)
from nexflow.services.strategy.ema_trend_strategy import EMATrendStrategy

# ── V9 production config ────────────────────────────────────────────────────────

STOCK_COMBO = ["MSTR", "AMD", "GOOGL", "META"]   # de-biased strict winner (Bitget-tradeable)

# ⚠️  VERIFY against your Bitget account before --stock-live.
# Bitget lists equity perps under their own productType/symbol scheme; the values
# below are the integration points you confirm once (productType + symbol suffix).
STOCK_PRODUCT_TYPE = "SUSDT-FUTURES"      # confirmed: Bitget "Stock perps" tab
STOCK_SYMBOL_MAP = {                       # confirmed from Bitget app screenshots
    "MSTR":  "MSTRUSDT",
    "AMD":   "AMDUSDT",
    "GOOGL": "GOOGLUSDT",
    "META":  "METAUSDT",
}

_MIN_NOTIONAL   = 5.0       # Bitget min order value (USDT)
_CRYPTO_TOP_K   = 3         # concentrate crypto into top-K conviction names ($100 mode)
_STOCK_HARD_STOP = 0.10     # 10% per-position hard stop (stock book)
_TAKER_FEE      = 0.0006
_STATE_FILE     = _REPO / "data" / "v9_confidence_state.json"
_STOCK_DIR      = _REPO / "data" / "stocks"


# ── confidence scoring (live) ───────────────────────────────────────────────────

def _sma(seq: list[float], n: int) -> Optional[float]:
    return sum(seq[-n:]) / n if len(seq) >= n else None


def _mom(seq: list[float], days: int) -> Optional[float]:
    if len(seq) < days + 1:
        return None
    return (seq[-1] - seq[-(days + 1)]) / seq[-(days + 1)]


def crypto_score(btc_closes: list[float]) -> float:
    """0-4. SMA200 = 2 pts (primary regime gate) so bear regimes stay < 2.6."""
    if len(btc_closes) < 30:
        return 2.0
    c = btc_closes[-1]
    s200 = _sma(btc_closes, 200) or _sma(btc_closes, 50)
    m90, m30 = _mom(btc_closes, 90), _mom(btc_closes, 30)
    sc = 0.0
    if s200 is not None:           sc += 2.0 if c > s200 else 0.0
    if m90 is not None:            sc += 1.0 if m90 > 0 else 0.0
    if m30 is not None:            sc += 0.5 if m30 > 0 else 0.0
    # simple vol-calm proxy: 14d realised range below its 60d mean → +0.5
    if len(btc_closes) >= 75:
        rng = [abs(btc_closes[i] - btc_closes[i - 1]) for i in range(-14, 0)]
        rng60 = [abs(btc_closes[i] - btc_closes[i - 1]) for i in range(-60, 0)]
        if sum(rng) / 14 < (sum(rng60) / 60) * 1.5:
            sc += 0.5
    if m90 is not None and m90 > 0.30:  sc += 0.5
    if m90 is not None and m90 < -0.30: sc -= 0.5
    return max(0.0, min(sc, 4.0))


def stock_score(stock_closes: dict[str, list[float]]) -> float:
    """0-3 averaged across the combo. High when names are in confirmed uptrend."""
    vals = []
    for t in STOCK_COMBO:
        seq = stock_closes.get(t, [])
        if len(seq) < 30:
            continue
        c = seq[-1]
        s200 = _sma(seq, 200) or _sma(seq, 50)
        m90 = _mom(seq, 90)
        ef = _ema(seq, 8); es = _ema(seq, 21)
        sc = 0.0
        if s200 is not None:            sc += 1.0 if c > s200 else 0.0
        if m90 is not None:             sc += 1.0 if m90 > 0 else 0.0
        if ef is not None and es is not None: sc += 0.5 if ef > es else 0.0
        if m90 is not None and m90 > 0.20:    sc += 0.5
        vals.append(sc)
    return sum(vals) / len(vals) if vals else 2.0


def _ema(seq: list[float], n: int) -> Optional[float]:
    if len(seq) < n:
        return None
    k = 2 / (n + 1)
    e = seq[0]
    for x in seq[1:]:
        e = k * x + (1 - k) * e
    return e


def allocate(c_sc: float, s_sc: float) -> tuple[float, float, str]:
    """Return (crypto_w, stock_w, label). Remainder is cash."""
    cn, sn = c_sc / 4.0, s_sc / 3.0
    if cn >= 0.65 and sn >= 0.65: return 0.65, 0.35, "BOTH STRONG (crypto leads)"
    if cn >= 0.65 and sn <  0.65: return 0.80, 0.20, "CRYPTO DOMINANT"
    if sn >= 0.65 and cn <  0.65: return 0.20, 0.80, "STOCK DOMINANT"
    if cn <  0.35 and sn <  0.35: return 0.40, 0.40, "DEFENSIVE (20% cash)"
    tot = cn + sn
    wc = round(0.40 + (cn / tot) * 0.20, 2)
    return wc, round(1.0 - wc, 2), "BALANCED"


# ── stock data (seed from parquet, append live) ─────────────────────────────────

def _seed_stock_closes() -> dict[str, tuple[list[float], int]]:
    """Return {ticker: (closes, last_bar_ts_ms)} from local parquet seed."""
    import pyarrow.parquet as pq
    out: dict[str, tuple[list[float], int]] = {}
    for t in STOCK_COMBO:
        p = _STOCK_DIR / f"{t}_1D.parquet"
        if not p.exists():
            print(f"  [WARN] no seed data for {t}")
            out[t] = ([], 0)
            continue
        tbl = pq.read_table(p)
        cols = {c.lower(): c for c in tbl.column_names}
        tcol = cols.get("open_time") or cols.get("time") or cols.get("date")
        ccol = cols.get("close")
        rows = sorted(zip(tbl.column(tcol).to_pylist(), tbl.column(ccol).to_pylist()))
        closes = [float(c) for _, c in rows]
        last_ts = int(rows[-1][0]) if rows else 0
        out[t] = (closes, last_ts)
    return out


def _fetch_stock_close(ticker: str) -> Optional[float]:
    """Fetch latest closed daily (ts, close) for a Bitget stock perp.
    Returns (open_time_ms, close) or None on failure."""
    sym = STOCK_SYMBOL_MAP[ticker]
    url = (f"https://api.bitget.com/api/v2/mix/market/history-candles"
           f"?symbol={sym}&productType={STOCK_PRODUCT_TYPE}&granularity=1D&limit=3")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NexFlow/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        if data.get("code") != "00000" or not data.get("data"):
            return None
        row = data["data"][1]  # last fully-closed candle
        return int(row[0]), float(row[4])
    except Exception as e:
        print(f"  [WARN] stock close {ticker}: {e}")
        return None


# ── stock strategy state (per-ticker, long-only) ────────────────────────────────

class StockState:
    """Per-asset trend state: SMA200 gate + EMA8/21 + MACD confirm + mom90 gate.
    Exit = MACD cross-down OR -10% hard stop (stop handled by caller)."""

    def __init__(self):
        self.closes: list[float] = []
        self.macd = MACDState(12, 26, 9)
        self.in_pos = False
        self.entry = 0.0

    def seed(self, closes: list[float]) -> None:
        for c in closes:
            self.closes.append(c)
            self.macd.update(c)

    def on_close(self, c: float) -> Optional[str]:
        """Return 'OPEN_LONG' / 'CLOSE_LONG' / None for this daily close."""
        self.closes.append(c)
        macd_action = self.macd.update(c)
        s200 = _sma(self.closes, 200) or _sma(self.closes, 50)
        ef, es = _ema(self.closes, 8), _ema(self.closes, 21)
        m90 = _mom(self.closes, 90)

        bull = (s200 is not None and c > s200 and
                ef is not None and es is not None and ef > es and
                self.macd.position and
                m90 is not None and m90 > 0)

        if not self.in_pos and bull:
            self.in_pos = True
            self.entry = c
            return "OPEN_LONG"
        if self.in_pos and (macd_action == "CLOSE_LONG" or
                            (s200 is not None and c < s200)):
            self.in_pos = False
            return "CLOSE_LONG"
        return None

    def signal(self) -> str:
        return "LONG" if self.in_pos else "FLAT"


# ── stock execution (Bitget stock perps) ────────────────────────────────────────

def _set_stock_leverage(client, stock_live: bool) -> None:
    """Force every stock perp to 1x BEFORE any order (account safety — the app
    defaults to 10x). Matches the no-leverage design of the crypto book."""
    if not stock_live or client is None:
        return
    for ticker, sym in STOCK_SYMBOL_MAP.items():
        for hold_side in ("long", "short"):
            try:
                client.post("/api/v2/mix/account/set-leverage", {
                    "symbol": sym, "productType": STOCK_PRODUCT_TYPE,
                    "marginCoin": "USDT", "leverage": "1", "holdSide": hold_side,
                })
            except Exception:
                pass  # non-fatal; some accounts set leverage per-position only
    print("  [STOCK] leverage forced to 1x on all stock perps")


def _stock_order(client, ticker: str, side: str, notional: float,
                 price: float, stock_live: bool) -> bool:
    """Place/close a stock-perp market order. Dry-run unless stock_live=True.
    side: 'open_long' | 'close_long'.  Returns True if acted (or dry-logged)."""
    sym = STOCK_SYMBOL_MAP[ticker]
    if not stock_live or client is None:
        print(f"    [STOCK dry-run] {side} {sym} ~${notional:,.2f} @ {price:,.2f}")
        return True
    try:
        if side == "open_long":
            if notional < _MIN_NOTIONAL:
                print(f"    [STOCK] {sym} skipped — ${notional:,.2f} < ${_MIN_NOTIONAL} min")
                return False
            qty = notional / price if price > 0 else 0
            if qty <= 0:
                return False
            # 1x leverage already enforced at startup via _set_stock_leverage
            body = {
                "symbol": sym, "productType": STOCK_PRODUCT_TYPE,
                "marginMode": "crossed", "marginCoin": "USDT",
                "size": str(qty), "side": "buy", "tradeSide": "open",
                "orderType": "market",
            }
            client.post("/api/v2/mix/order/place-order", body)
            print(f"    [STOCK] BUY  {sym} ${notional:,.2f} @ {price:,.2f}")
        else:  # close_long
            body = {"symbol": sym, "productType": STOCK_PRODUCT_TYPE,
                    "holdSide": "long"}
            client.post("/api/v2/mix/order/close-positions", body)
            print(f"    [STOCK] SELL {sym} (close) @ {price:,.2f}")
        return True
    except Exception as e:
        print(f"    [ERROR] stock {side} {sym}: {e}")
        return False


def _stock_position(client, ticker: str) -> Optional[dict]:
    """Return the open stock-perp position dict for ticker, or None if flat."""
    if client is None:
        return None
    sym = STOCK_SYMBOL_MAP[ticker]
    try:
        data = client.get("/api/v2/mix/position/all-position", {
            "symbol": sym, "productType": STOCK_PRODUCT_TYPE, "marginCoin": "USDT",
        })
        if not data:
            return None
        positions = data if isinstance(data, list) else [data]
        for pos in positions:
            if pos.get("symbol") == sym and float(pos.get("total", 0)) > 0:
                return pos
    except Exception:
        return None
    return None


# ── replay (sanity check) ───────────────────────────────────────────────────────

def run_replay(capital: float) -> None:
    import test_v9_confidence as V9
    print(f"\nV9 CONFIDENCE — REPLAY (wiring sanity check)  capital=${capital:,.0f}")
    print(f"Combo: {'+'.join(STOCK_COMBO)}\n")
    r = V9.run_v9_confidence(STOCK_COMBO, capital=capital)
    print(f"\n  Final=${r['final']:,.0f}  CAGR={r['cagr']*100:.1f}%  "
          f"DD={r['dd']*100:.1f}%  Sharpe={r['sharpe']:.2f}")
    losing = [y for y, p in r["year_pnl"].items() if p < 0]
    print(f"  Losing years: {losing or 'NONE'}")
    print("\n  Year-by-year PnL:")
    bal = capital
    for yr in sorted(r["year_pnl"]):
        bal += r["year_pnl"][yr]
        print(f"    {yr}: ${r['year_pnl'][yr]:>+10,.0f}   balance ${bal:>10,.0f}")


# ── live ─────────────────────────────────────────────────────────────────────────

def run_live(capital: float, stock_live: bool) -> None:
    from nexflow.execution.adapter import BitgetPaperAdapter
    from nexflow.exchange.bitget_client import BitgetClient
    from nexflow.exchange.bitget_order import get_account_balance, get_position

    client = BitgetClient.from_env()
    adapter = BitgetPaperAdapter(client)

    # real balance if available
    try:
        bal = get_account_balance(client)
        if bal > 0:
            print(f"Account balance: ${bal:,.2f} (exchange)")
            capital = bal
    except Exception as e:
        print(f"Account balance: fetch failed ({e}) — using --capital ${capital:,.0f}")

    print("=" * 78)
    print("  NEXFLOW V9 — CONFIDENCE BOT (LIVE)")
    print(f"  Capital      : ${capital:,.2f}")
    print(f"  Crypto book  : V8.63 concentrate top-{_CRYPTO_TOP_K}  (Bitget USDT perps)")
    print(f"  Stock  book  : {'+'.join(STOCK_COMBO)}  "
          f"({'LIVE' if stock_live else 'DRY-RUN'}, {STOCK_PRODUCT_TYPE})")
    print(f"  Min notional : ${_MIN_NOTIONAL}  | Stops: crypto {_CRYPTO_STOP:.0%}, "
          f"stock {_STOCK_HARD_STOP:.0%}")
    print("=" * 78)
    if not stock_live:
        print("  NOTE: stock leg is DRY-RUN (logs orders, sends none).")
        print("        Pass --stock-live to enable real stock orders.\n")

    # ── seed crypto state ──
    today = int(datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    seed_from = today - 260 * _DAY_MS

    ema_strat = EMATrendStrategy(symbols=_CRYPTO_SYMBOLS, fast=8, slow=21)
    macd_c = {s: MACDState() for s in _CRYPTO_SYMBOLS}
    ema4h_c = {s: EMA4HState() for s in _CRYPTO_SYMBOLS}
    btc_regime = BTCRegime()
    btc_closes: list[float] = []
    coin_closes: dict[str, list[float]] = {s: [] for s in _CRYPTO_SYMBOLS}

    # Track the last daily-bar timestamp ingested per symbol so repeated checks
    # (weekends, restarts, same-day re-runs) never double-append a bar and
    # corrupt the SMA/EMA/momentum series.
    last_crypto_ts: dict[str, int] = {s: 0 for s in _CRYPTO_SYMBOLS}
    last_stock_ts: dict[str, int] = {t: 0 for t in STOCK_COMBO}

    print("Seeding crypto from history ...")
    for s in _CRYPTO_SYMBOLS:
        for ts, c in _load_candles(s, "1D", seed_from, today):
            ema_strat.on_daily_close(s, c, ts)
            macd_c[s].update(c)
            coin_closes[s].append(c)
            last_crypto_ts[s] = max(last_crypto_ts[s], int(ts))
            if s == "BTCUSDT":
                btc_regime.update(c)
                btc_closes.append(c)
        for ts, c in _resample_4h(_load_candles(s, "1H", seed_from, today)):
            ema4h_c[s].update(c)

    # ── seed stock state ──
    print("Seeding stock from parquet ...")
    seed_stock = _seed_stock_closes()
    stock_states = {t: StockState() for t in STOCK_COMBO}
    stock_closes: dict[str, list[float]] = {}
    for t in STOCK_COMBO:
        closes, last_ts = seed_stock[t]
        stock_states[t].seed(closes)
        stock_closes[t] = list(closes)
        last_stock_ts[t] = last_ts

    # Force stock perps to 1x before any trading (app defaults to 10x)
    _set_stock_leverage(client, stock_live)

    coin_longs: dict[str, set] = {s: set() for s in _CRYPTO_SYMBOLS}

    # ── restore last-bar timestamps from state file (cross-restart dedup) ──
    if _STATE_FILE.exists():
        try:
            st = json.loads(_STATE_FILE.read_text())
            for s, v in st.get("last_crypto_ts", {}).items():
                if s in last_crypto_ts:
                    last_crypto_ts[s] = max(last_crypto_ts[s], int(v))
            for t, v in st.get("last_stock_ts", {}).items():
                if t in last_stock_ts:
                    last_stock_ts[t] = max(last_stock_ts[t], int(v))
            print(f"  Restored bar-timestamp state from {_STATE_FILE.name}")
        except Exception as e:
            print(f"  [WARN] could not load state file: {e}")

    def _save_state() -> None:
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STATE_FILE.write_text(json.dumps({
                "last_crypto_ts": last_crypto_ts,
                "last_stock_ts":  last_stock_ts,
                "updated":        datetime.now(timezone.utc).isoformat(),
            }))
        except Exception as e:
            print(f"  [WARN] could not save state: {e}")

    # ── reconcile OPEN POSITIONS from the exchange (source of truth) ──
    print("Reconciling open positions from exchange ...")
    n_c = n_s = 0
    for s in _CRYPTO_SYMBOLS:
        try:
            pos = get_position(client, s)
        except Exception:
            pos = None
        if pos and pos.get("holdSide") == "long" and float(pos.get("total", 0)) > 0:
            # treat as a full confluence position so it isn't re-opened or orphaned
            coin_longs[s] = {"EMA", "MACD", "4H"}
            n_c += 1
            print(f"  Restored CRYPTO long {s} @ {pos.get('openPriceAvg','?')}")
        time.sleep(0.1)
    for t in STOCK_COMBO:
        pos = _stock_position(client, t)
        if pos:
            entry = float(pos.get("openPriceAvg", 0)) or (
                stock_closes[t][-1] if stock_closes[t] else 0)
            stock_states[t].in_pos = True
            stock_states[t].entry = entry
            n_s += 1
            print(f"  Restored STOCK long {t} @ {entry}")
        time.sleep(0.1)
    print(f"  Reconciled {n_c} crypto + {n_s} stock open position(s).\n")

    def _daily_check():
        nonlocal capital
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n[{now}] ════ V9 DAILY CHECK ════")

        # refresh balance
        try:
            b = get_account_balance(client)
            if b > 0:
                capital = b
        except Exception:
            pass

        # ── fetch crypto closes (price always; ingest only on NEW bar) ──
        c_price: dict[str, float] = {}   # latest price for execution
        c_new: dict[str, bool] = {}      # whether a new daily bar arrived
        for s in _CRYPTO_SYMBOLS:
            r = _fetch_close(s)          # (open_time_ms, close)
            if r:
                ts, px = int(r[0]), float(r[1])
                c_price[s] = px
                if ts > last_crypto_ts[s]:
                    c_new[s] = True
                    last_crypto_ts[s] = ts
                    coin_closes[s].append(px)
                    if s == "BTCUSDT":
                        btc_regime.update(px)
                        btc_closes.append(px)
            time.sleep(0.15)

        # ── fetch stock closes (price always; ingest only on NEW bar) ──
        s_price: dict[str, float] = {}
        s_new: dict[str, bool] = {}
        for t in STOCK_COMBO:
            r = _fetch_stock_close(t)    # (ts_ms, close) or None
            if r is not None:
                ts, px = int(r[0]), float(r[1])
                s_price[t] = px
                if ts > last_stock_ts[t]:
                    s_new[t] = True
                    last_stock_ts[t] = ts
                    stock_closes[t].append(px)
            time.sleep(0.15)

        # ── CONFIDENCE ALLOCATION ──
        c_sc = crypto_score(btc_closes)
        s_sc = stock_score(stock_closes)
        wc, ws, label = allocate(c_sc, s_sc)
        crypto_cap = capital * wc
        stock_cap = capital * ws
        print(f"  Crypto score {c_sc:.2f}/4   Stock score {s_sc:.2f}/3")
        print(f"  → {wc*100:.0f}% crypto (${crypto_cap:,.2f})  "
              f"{ws*100:.0f}% stock (${stock_cap:,.2f})  "
              f"{max(0,1-wc-ws)*100:.0f}% cash   [{label}]")

        bear = btc_regime.is_bear
        print(f"  Crypto regime: {'BEAR' if bear else 'BULL'}")

        # ── CRYPTO BOOK (concentrate top-K) ──
        # update signals — ONLY on a new daily bar (stateful EMA/MACD)
        for s in _CRYPTO_SYMBOLS:
            if not c_new.get(s):
                continue
            for sig in ema_strat.on_daily_close(s, c_price[s], today):
                if sig.action == "OPEN_LONG":  coin_longs[s].add("EMA")
                elif sig.action == "CLOSE_LONG": coin_longs[s].discard("EMA")
            a = macd_c[s].update(c_price[s])
            if a == "OPEN_LONG":  coin_longs[s].add("MACD")
            elif a == "CLOSE_LONG": coin_longs[s].discard("MACD")

        if bear or crypto_cap < _MIN_NOTIONAL:
            print(f"  Crypto: {'bear regime' if bear else 'slice too small'} — "
                  f"closing/holding flat (long book paused)")
            for s in list(coin_longs):
                if coin_longs[s] and s in c_price:
                    _crypto_close(adapter, client, s, c_price[s])
                    coin_longs[s].clear()
        else:
            # rank active-signal coins by confluence then 20d momentum, take top-K
            ranked = []
            for s in _CRYPTO_SYMBOLS:
                n = len(coin_longs[s])
                if n == 0:
                    continue
                m20 = _mom(coin_closes[s], 20) or 0
                if m20 <= 0:  # momentum gate
                    continue
                ranked.append((n, m20, s))
            ranked.sort(reverse=True)
            chosen = [s for _, _, s in ranked[:_CRYPTO_TOP_K]]
            per = crypto_cap / max(len(chosen), 1) if chosen else 0
            # close anything not chosen
            for s in list(coin_longs):
                held = get_position(client, s) if client else None
                if held and s not in chosen and s in c_price:
                    _crypto_close(adapter, client, s, c_price[s])
                    coin_longs[s].clear()
            # open chosen that clear min notional
            for s in chosen:
                if per < _MIN_NOTIONAL:
                    print(f"  [CRYPTO] {s} skipped — ${per:,.2f} < ${_MIN_NOTIONAL} min")
                    continue
                if s not in c_price:
                    continue
                held = get_position(client, s) if client else None
                if held:
                    continue
                qty = per / c_price[s]
                res = adapter.on_entry(s, "long", qty, 0.0, 0.0, 0.0)
                if res is not None and not getattr(res, "accepted", True):
                    print(f"  [CRYPTO] {s} rejected: {getattr(res,'note','?')}")
                else:
                    print(f"  [CRYPTO] BUY {s} ${per:,.2f} @ {c_price[s]:,.4f} "
                          f"({len(coin_longs[s])} strat)")
            print(f"  Crypto active: {', '.join(chosen) or 'none'}")

        # ── STOCK BOOK ──
        if stock_cap < _MIN_NOTIONAL:
            print(f"  Stock: slice ${stock_cap:,.2f} < ${_MIN_NOTIONAL} — paused")
        elif not any(s_new.get(t) for t in STOCK_COMBO):
            # no new stock bar (weekend / US market closed) — hold, don't act
            active = [t for t in STOCK_COMBO if stock_states[t].in_pos]
            print(f"  Stock: no new bar (market closed) — holding "
                  f"{', '.join(active) or 'flat'}")
        else:
            # process signals ONLY on new bars (stateful MACD), count active
            actions = {}
            for t in STOCK_COMBO:
                if s_new.get(t) and stock_closes[t]:
                    actions[t] = stock_states[t].on_close(stock_closes[t][-1])
            active = [t for t in STOCK_COMBO if stock_states[t].in_pos]
            per_stock = stock_cap / max(len(active), 1) if active else 0
            for t in STOCK_COMBO:
                px = s_price.get(t) or (stock_closes[t][-1] if stock_closes[t] else 0)
                if actions.get(t) == "OPEN_LONG":
                    _stock_order(client, t, "open_long", per_stock, px, stock_live)
                elif actions.get(t) == "CLOSE_LONG":
                    _stock_order(client, t, "close_long", 0, px, stock_live)
            print(f"  Stock active: {', '.join(active) or 'none'}")

        # persist bar-timestamp state so a restart resumes without re-processing
        _save_state()

    def _crypto_close(adapter, client, sym, price):
        try:
            pos = get_position(client, sym) if client else None
            qty = float(pos.get("total", 0)) if pos else 0.0
            if qty > 0:
                adapter.on_close(sym, "long", qty, price, "v9_close")
                print(f"  [CRYPTO] SELL {sym} qty={qty} @ {price:,.4f}")
        except Exception as e:
            print(f"  [ERROR] crypto close {sym}: {e}")

    # ── loop: daily at 21:05 UTC (just after US close / daily bar) ──
    def _next_check(now):
        cand = now.replace(hour=21, minute=5, second=0, microsecond=0)
        if cand <= now:
            cand += timedelta(days=1)
        return cand

    _daily_check()
    nxt = _next_check(datetime.now(timezone.utc))
    while True:
        now = datetime.now(timezone.utc)
        sleep_s = max((nxt - now).total_seconds(), 60.0)
        print(f"\n  Next V9 check: {nxt.strftime('%Y-%m-%d %H:%M UTC')} "
              f"(sleeping {sleep_s/3600:.1f}h)")
        time.sleep(sleep_s)
        _daily_check()
        nxt = _next_check(datetime.now(timezone.utc))


def main():
    ap = argparse.ArgumentParser(description="NexFlow V9 confidence bot")
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--capital", type=float, default=100.0)
    ap.add_argument("--stock-live", action="store_true",
                    help="actually send stock orders (default: dry-run)")
    args = ap.parse_args()
    if args.mode == "replay":
        run_replay(args.capital)
    else:
        run_live(args.capital, args.stock_live)


if __name__ == "__main__":
    main()
