"""
V9 Signal Engine — Module 4.

Produces a DailySignalRecord for every trading day.  Every field is
populated from core.py functions; nothing is computed here directly.
The record is the complete audit trail: given the same data it must
reproduce bit-for-bit identical outputs on replay.

Entry point:  compute_signals(dataset, regime_machine) -> DailySignalRecord
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import json

from nexflow.v9 import core
from nexflow.v9.data import V9DataSet


# ── Output record ─────────────────────────────────────────────────────────────

@dataclass
class StockSignal:
    ticker:      str
    close:       float
    sma200:      float
    mom90:       float
    ema_fast:    float
    ema_slow:    float
    pts_sma200:  float
    pts_mom90:   float
    pts_bonus:   float
    pts_ema:     float
    score:       float


@dataclass
class CryptoSignal:
    close:       float
    sma200:      float
    mom90:       float
    mom30:       float
    atr14:       float
    atr_avg:     float
    pts_sma200:  float
    pts_mom90:   float
    pts_mom30:   float
    pts_vol:     float
    pts_bonus:   float
    raw:         float
    score:       float


@dataclass
class DailySignalRecord:
    # ── Identity ──────────────────────────────────────────────────────────────
    date:            str          # YYYY-MM-DD (UTC)
    timestamp_ms:    int          # UTC midnight ms

    # ── Regime ────────────────────────────────────────────────────────────────
    in_bear:         bool

    # ── Crypto score ─────────────────────────────────────────────────────────
    btc:             CryptoSignal
    crypto_score:    float        # final normalised score (0–4)

    # ── Stock scores ─────────────────────────────────────────────────────────
    stocks:          list[StockSignal]
    stock_score:     float        # portfolio-level normalised score (0–3)

    # ── Allocation ───────────────────────────────────────────────────────────
    allocation_regime: str        # human label
    wc:              float        # crypto weight
    ws:              float        # stock weight
    cash:            float        # 1 - wc - ws

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DailySignalRecord":
        d = dict(d)
        d["btc"] = CryptoSignal(**d["btc"])
        d["stocks"] = [StockSignal(**s) for s in d["stocks"]]
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "DailySignalRecord":
        return cls.from_dict(json.loads(s))


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals(
    dataset: V9DataSet,
    regime_machine: core.RegimeMachine,
) -> DailySignalRecord:
    """
    Compute DailySignalRecord for the latest bar in *dataset*.

    The caller is responsible for:
    - Passing a RegimeMachine whose state is already stepped through all
      prior bars (loaded from state.py or replayed from history).
    - Calling regime_machine.step() BEFORE calling this function so that
      in_bear reflects the current bar.

    This function is pure: it reads dataset, calls core.py functions, and
    returns a record.  It has no side effects.
    """
    ts   = dataset.latest_ts()
    date = dataset.latest_date()
    btc  = dataset.btc()

    # ── BTC score ──────────────────────────────────────────────────────────
    idx = btc.by_ts[ts]
    c_sc, cb = core.crypto_score_breakdown(
        btc_close = float(btc.close[idx]),
        sma200    = float(btc.sma200[idx]),
        mom90     = float(btc.mom90[idx]),
        mom30     = float(btc.mom30[idx]),
        atr14     = float(btc.atr14[idx]),
        atr_avg   = float(btc.atr_avg[idx]),
    )

    btc_sig = CryptoSignal(
        close    = float(btc.close[idx]),
        sma200   = float(btc.sma200[idx]),
        mom90    = float(btc.mom90[idx]),
        mom30    = float(btc.mom30[idx]),
        atr14    = float(btc.atr14[idx]),
        atr_avg  = float(btc.atr_avg[idx]),
        pts_sma200 = cb["pts_sma200"],
        pts_mom90  = cb["pts_mom90"],
        pts_mom30  = cb["pts_mom30"],
        pts_vol    = cb["pts_vol"],
        pts_bonus  = cb["pts_bonus"],
        raw        = cb["raw"],
        score      = c_sc,
    )

    # ── Stock scores ───────────────────────────────────────────────────────
    stock_signals: list[StockSignal] = []
    ticker_scores: list[float] = []

    stocks_iter = (
        dataset.stocks.values()
        if isinstance(dataset.stocks, dict)
        else dataset.stocks
    )
    for stk in stocks_iter:
        if ts not in stk.by_ts:
            continue
        si = stk.by_ts[ts]
        s_sc_single, sb = core.stock_score_single_breakdown(
            close = float(stk.close[si]),
            s200  = float(stk.sma200[si]),
            mom90 = float(stk.mom90[si]),
            ema_f = float(stk.ema_f[si]),
            ema_s = float(stk.ema_s[si]),
        )
        stock_signals.append(StockSignal(
            ticker    = stk.ticker,
            close     = float(stk.close[si]),
            sma200    = float(stk.sma200[si]),
            mom90     = float(stk.mom90[si]),
            ema_fast  = float(stk.ema_f[si]),
            ema_slow  = float(stk.ema_s[si]),
            pts_sma200 = sb["pts_sma200"],
            pts_mom90  = sb["pts_mom90"],
            pts_bonus  = sb["pts_bonus"],
            pts_ema    = sb["pts_ema"],
            score      = s_sc_single,
        ))
        ticker_scores.append(s_sc_single)

    s_sc = core.stock_score_portfolio(ticker_scores) if ticker_scores else 0.0

    # ── Allocation ─────────────────────────────────────────────────────────
    wc, ws = core.allocate(c_sc, s_sc)
    regime_name = core.allocation_regime_name(c_sc, s_sc)

    return DailySignalRecord(
        date             = date,
        timestamp_ms     = ts,
        in_bear          = regime_machine.in_bear,
        btc              = btc_sig,
        crypto_score     = c_sc,
        stocks           = stock_signals,
        stock_score      = s_sc,
        allocation_regime = regime_name,
        wc               = wc,
        ws               = ws,
        cash             = round(1.0 - wc - ws, 10),
    )
