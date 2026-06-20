"""
V9 Signal Engine — Module 4 tests.

Test classes:
  TestDailySignalRecord   — dataclass round-trip, JSON serialisation
  TestComputeSignals      — compute_signals() with synthetic data
  TestSignalReplay        — exact equality against research engine fixtures
                            (4 historical dates; skip if parquet not present)
  TestAllocationRegimeName — allocation_regime_name() label coverage
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.v9.core import RegimeMachine, allocation_regime_name
from nexflow.v9.signals import (
    CryptoSignal,
    DailySignalRecord,
    StockSignal,
    compute_signals,
)
from nexflow.v9.data import load_v9_dataset


# ── helpers ───────────────────────────────────────────────────────────────────

def _ms(date_str: str) -> int:
    """Convert YYYY-MM-DD to UTC midnight milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _make_crypto_signal(**kwargs) -> CryptoSignal:
    defaults = dict(
        close=50000.0, sma200=40000.0, mom90=0.10, mom30=0.05,
        atr14=1000.0, atr_avg=800.0,
        pts_sma200=2.0, pts_mom90=1.0, pts_mom30=0.5, pts_vol=0.5,
        pts_bonus=0.0, raw=4.0, score=4.0,
    )
    defaults.update(kwargs)
    return CryptoSignal(**defaults)


def _make_stock_signal(ticker="AMD", **kwargs) -> StockSignal:
    defaults = dict(
        ticker=ticker, close=100.0, sma200=80.0, mom90=0.15,
        ema_fast=102.0, ema_slow=98.0,
        pts_sma200=1.0, pts_mom90=1.0, pts_bonus=0.0, pts_ema=0.5,
        score=2.5,
    )
    defaults.update(kwargs)
    return StockSignal(**defaults)


def _make_record(**kwargs) -> DailySignalRecord:
    defaults = dict(
        date="2024-01-15",
        timestamp_ms=_ms("2024-01-15"),
        in_bear=False,
        btc=_make_crypto_signal(),
        crypto_score=4.0,
        stocks=[_make_stock_signal()],
        stock_score=2.5,
        allocation_regime="CRYPTO_DOMINANT",
        wc=0.80,
        ws=0.20,
        cash=0.0,
    )
    defaults.update(kwargs)
    return DailySignalRecord(**defaults)


# ── DailySignalRecord round-trip ──────────────────────────────────────────────

class TestDailySignalRecord:
    def test_to_dict_round_trip(self):
        rec = _make_record()
        d = rec.to_dict()
        rec2 = DailySignalRecord.from_dict(d)
        assert rec2.date == rec.date
        assert rec2.crypto_score == rec.crypto_score
        assert rec2.stock_score == rec.stock_score
        assert rec2.wc == rec.wc
        assert rec2.ws == rec.ws

    def test_to_json_round_trip(self):
        rec = _make_record()
        j = rec.to_json()
        rec2 = DailySignalRecord.from_json(j)
        assert rec2.date == rec.date
        assert rec2.allocation_regime == rec.allocation_regime
        assert rec2.btc.score == rec.btc.score

    def test_json_is_valid_json(self):
        rec = _make_record()
        parsed = json.loads(rec.to_json())
        assert isinstance(parsed, dict)
        assert "date" in parsed
        assert "btc" in parsed
        assert "stocks" in parsed

    def test_stocks_list_preserved_in_round_trip(self):
        rec = _make_record(stocks=[
            _make_stock_signal("AMD", score=3.0),
            _make_stock_signal("GOOGL", score=1.5),
        ])
        rec2 = DailySignalRecord.from_dict(rec.to_dict())
        assert len(rec2.stocks) == 2
        assert rec2.stocks[0].ticker == "AMD"
        assert rec2.stocks[1].ticker == "GOOGL"

    def test_cash_field_computed_correctly(self):
        # wc=0.65, ws=0.35 → cash should be 0.0
        rec = _make_record(wc=0.65, ws=0.35, cash=0.0)
        assert abs(rec.wc + rec.ws + rec.cash - 1.0) < 1e-9

    def test_cash_defensive_regime(self):
        rec = _make_record(wc=0.40, ws=0.40, cash=0.20)
        assert abs(rec.wc + rec.ws + rec.cash - 1.0) < 1e-9

    def test_bear_flag_preserved(self):
        rec = _make_record(in_bear=True)
        rec2 = DailySignalRecord.from_json(rec.to_json())
        assert rec2.in_bear is True

    def test_btc_breakdown_preserved(self):
        btc = _make_crypto_signal(pts_sma200=0.0, pts_mom90=0.0, pts_mom30=0.5,
                                   pts_vol=0.5, pts_bonus=0.0, raw=1.0, score=1.0)
        rec = _make_record(btc=btc, crypto_score=1.0)
        rec2 = DailySignalRecord.from_json(rec.to_json())
        assert rec2.btc.pts_sma200 == 0.0
        assert rec2.btc.score == 1.0


# ── compute_signals() with synthetic dataset ───────────────────────────────────

class TestComputeSignals:
    """
    Verify compute_signals() plumbing with a mocked V9DataSet.
    Real computation is tested via exact-equality replay tests below.
    """

    def _make_mock_dataset(
        self,
        *,
        date="2024-01-15",
        btc_close=62000.0,
        btc_sma200=40000.0,
        btc_mom90=0.50,
        btc_mom30=0.05,
        btc_atr14=1000.0,
        btc_atr_avg=800.0,
        stock_close=150.0,
        stock_sma200=120.0,
        stock_mom90=0.20,
        stock_ema_f=152.0,
        stock_ema_s=148.0,
    ):
        ts = _ms(date)

        btc = MagicMock()
        btc.symbol = "BTCUSDT"
        btc.by_ts = {ts: 0}
        btc.close   = np.array([btc_close])
        btc.sma200  = np.array([btc_sma200])
        btc.mom90   = np.array([btc_mom90])
        btc.mom30   = np.array([btc_mom30])
        btc.atr14   = np.array([btc_atr14])
        btc.atr_avg = np.array([btc_atr_avg])

        stk = MagicMock()
        stk.ticker = "AMD"
        stk.by_ts  = {ts: 0}
        stk.close  = np.array([stock_close])
        stk.sma200 = np.array([stock_sma200])
        stk.mom90  = np.array([stock_mom90])
        stk.ema_f  = np.array([stock_ema_f])
        stk.ema_s  = np.array([stock_ema_s])

        ds = MagicMock()
        ds.latest_ts.return_value   = ts
        ds.latest_date.return_value = date
        ds.btc.return_value         = btc
        ds.stocks                   = {"AMD": stk}
        return ds

    def test_record_date_matches_dataset(self):
        ds  = self._make_mock_dataset(date="2024-03-01")
        rm  = RegimeMachine(in_bear=False)
        rec = compute_signals(ds, rm)
        assert rec.date == "2024-03-01"

    def test_timestamp_ms_is_utc_midnight(self):
        ds  = self._make_mock_dataset(date="2024-03-01")
        rm  = RegimeMachine()
        rec = compute_signals(ds, rm)
        assert rec.timestamp_ms == _ms("2024-03-01")

    def test_in_bear_taken_from_regime_machine(self):
        ds  = self._make_mock_dataset()
        rm  = RegimeMachine(in_bear=True)
        rec = compute_signals(ds, rm)
        assert rec.in_bear is True

    def test_in_bear_false_when_regime_not_bear(self):
        ds  = self._make_mock_dataset()
        rm  = RegimeMachine(in_bear=False)
        rec = compute_signals(ds, rm)
        assert rec.in_bear is False

    def test_crypto_score_above_zero_in_bull(self):
        ds  = self._make_mock_dataset(btc_close=62000.0, btc_sma200=40000.0)
        rm  = RegimeMachine()
        rec = compute_signals(ds, rm)
        assert rec.crypto_score > 0.0

    def test_btc_pts_sma200_positive_when_above_sma(self):
        ds  = self._make_mock_dataset(btc_close=62000.0, btc_sma200=40000.0)
        rm  = RegimeMachine()
        rec = compute_signals(ds, rm)
        assert rec.btc.pts_sma200 == 2.0

    def test_btc_pts_sma200_zero_when_below_sma(self):
        ds  = self._make_mock_dataset(btc_close=30000.0, btc_sma200=50000.0)
        rm  = RegimeMachine()
        rec = compute_signals(ds, rm)
        assert rec.btc.pts_sma200 == 0.0

    def test_stock_score_single_above_sma_positive_mom(self):
        ds  = self._make_mock_dataset(stock_close=150.0, stock_sma200=120.0, stock_mom90=0.25)
        rm  = RegimeMachine()
        rec = compute_signals(ds, rm)
        # pts_sma200=1 + pts_mom90=1 + pts_bonus=0.5 (mom90>0.20) + pts_ema=0.5 = 3.0
        assert rec.stocks[0].score == 3.0

    def test_stock_ticker_preserved(self):
        ds  = self._make_mock_dataset()
        rm  = RegimeMachine()
        rec = compute_signals(ds, rm)
        assert rec.stocks[0].ticker == "AMD"

    def test_allocation_weights_sum_to_one_or_less(self):
        ds  = self._make_mock_dataset()
        rm  = RegimeMachine()
        rec = compute_signals(ds, rm)
        assert rec.wc + rec.ws + rec.cash <= 1.0 + 1e-9

    def test_cash_equals_one_minus_wc_minus_ws(self):
        ds  = self._make_mock_dataset()
        rm  = RegimeMachine()
        rec = compute_signals(ds, rm)
        assert abs(rec.cash - (1.0 - rec.wc - rec.ws)) < 1e-9

    def test_allocation_regime_name_not_empty(self):
        ds  = self._make_mock_dataset()
        rm  = RegimeMachine()
        rec = compute_signals(ds, rm)
        assert rec.allocation_regime in {
            "BOTH_HOT", "CRYPTO_DOMINANT", "STOCK_DOMINANT", "DEFENSIVE", "NEUTRAL"
        }

    def test_both_hot_regime_when_scores_high(self):
        # c_sc=4.0 (cn=1.0≥0.65), s_sc from stock: above SMA, mom90=0.25, ema cross → 3.0 (sn=1.0≥0.65)
        ds  = self._make_mock_dataset(
            btc_close=62000.0, btc_sma200=40000.0, btc_mom90=0.50, btc_mom30=0.10,
            btc_atr14=800.0, btc_atr_avg=1200.0,  # vol low → pts_vol=0.5
            stock_close=150.0, stock_sma200=100.0, stock_mom90=0.25,
            stock_ema_f=152.0, stock_ema_s=148.0,
        )
        rm  = RegimeMachine()
        rec = compute_signals(ds, rm)
        assert rec.allocation_regime == "BOTH_HOT"
        assert rec.wc == 0.65
        assert rec.ws == 0.35

    def test_defensive_regime_when_both_cold(self):
        # c_sc near 0: below SMA200, negative mom, high vol
        # s_sc near 0: below SMA200, negative mom
        ds  = self._make_mock_dataset(
            btc_close=25000.0, btc_sma200=50000.0, btc_mom90=-0.40, btc_mom30=-0.10,
            btc_atr14=3000.0, btc_atr_avg=1000.0,
            stock_close=80.0, stock_sma200=120.0, stock_mom90=-0.15,
            stock_ema_f=78.0, stock_ema_s=85.0,
        )
        rm  = RegimeMachine()
        rec = compute_signals(ds, rm)
        assert rec.allocation_regime == "DEFENSIVE"
        assert rec.wc == 0.40
        assert rec.ws == 0.40
        assert abs(rec.cash - 0.20) < 1e-9


# ── Exact equality replay tests (require real parquet files) ──────────────────

_CANDLE_DIR = _REPO / "data" / "candles"
_STOCK_DIR  = _REPO / "data" / "stocks"
_HAS_DATA   = (_CANDLE_DIR / "BTCUSDT_1D.parquet").exists()


# Ground truth from research engine (test_v9_confidence.py)
# Extracted and frozen.  Changing these values requires a new research run
# and an authorised architecture change.
_FIXTURES = {
    "2022-03-01": {
        "in_bear":        False,
        "crypto_score":   1.0,
        "btc_pts_sma200": 0.0,
        "btc_pts_mom90":  0.0,
        "btc_pts_mom30":  0.5,
        "btc_pts_vol":    0.5,
        "btc_pts_bonus":  0.0,
        # Stock scores (sum/4 maps to s_sc)
        "amd_score":      1.0,
        "googl_score":    0.0,
        "mstr_score":     0.5,
        "spot_score":     0.0,
        "stock_score":    0.375,    # (1.0+0.0+0.5+0.0)/4 * STOCK_SCORE_MAX/STOCK_SCORE_MAX
        "allocation_regime": "DEFENSIVE",
        "wc": 0.40,
        "ws": 0.40,
    },
    "2023-06-01": {
        "in_bear":        False,
        "crypto_score":   3.5,
        "btc_pts_sma200": 2.0,
        "btc_pts_mom90":  1.0,
        "btc_pts_mom30":  0.0,
        "btc_pts_vol":    0.5,
        "btc_pts_bonus":  0.0,
        "amd_score":      3.0,
        "googl_score":    3.0,
        "mstr_score":     2.0,
        "spot_score":     3.0,
        "stock_score":    2.75,
        "allocation_regime": "BOTH_HOT",
        "wc": 0.65,
        "ws": 0.35,
    },
    "2024-03-01": {
        "in_bear":        False,
        "crypto_score":   4.0,
        "btc_pts_sma200": 2.0,
        "btc_pts_mom90":  1.0,
        "btc_pts_mom30":  0.5,
        "btc_pts_vol":    0.5,
        "btc_pts_bonus":  0.5,
        "amd_score":      3.0,
        "googl_score":    2.0,
        "mstr_score":     3.0,
        "spot_score":     3.0,
        "stock_score":    2.75,
        "allocation_regime": "BOTH_HOT",
        "wc": 0.65,
        "ws": 0.35,
    },
    "2025-05-01": {
        "in_bear":        False,
        "crypto_score":   3.0,
        "btc_pts_sma200": 2.0,
        "btc_pts_mom90":  0.0,
        "btc_pts_mom30":  0.5,
        "btc_pts_vol":    0.5,
        "btc_pts_bonus":  0.0,
        "amd_score":      0.5,
        "googl_score":    0.5,
        "mstr_score":     2.5,
        "spot_score":     3.0,
        "stock_score":    1.625,
        "allocation_regime": "CRYPTO_DOMINANT",
        "wc": 0.80,
        "ws": 0.20,
    },
}


def _load_regime_up_to(date_str: str) -> RegimeMachine:
    """Replay regime machine up to (and including) the given date."""
    from nexflow.v9.data import load_crypto
    btc = load_crypto("BTCUSDT", date_str)
    target_ts = _ms(date_str)
    rm = RegimeMachine()
    for i, ts in enumerate(btc.ts):
        if ts > target_ts:
            break
        rm.step(
            close  = float(btc.close[i]),
            sma200 = float(btc.sma200[i]),
            mom30  = float(btc.mom30[i]),
        )
    return rm


@pytest.mark.skipif(not _HAS_DATA, reason="parquet data not available")
class TestSignalReplay:
    """
    Exact equality against frozen research-engine ground truth.

    These tests define the parity contract for Module 4.
    A single failure means RISK 00 has materialised and deployment must halt.
    """

    @pytest.mark.parametrize("date", list(_FIXTURES.keys()))
    def test_crypto_score_exact(self, date):
        fix = _FIXTURES[date]
        ds  = load_v9_dataset(date)
        rm  = _load_regime_up_to(date)
        rec = compute_signals(ds, rm)
        assert rec.crypto_score == fix["crypto_score"], (
            f"{date}: crypto_score {rec.crypto_score} != {fix['crypto_score']}"
        )

    @pytest.mark.parametrize("date", list(_FIXTURES.keys()))
    def test_btc_breakdown_exact(self, date):
        fix = _FIXTURES[date]
        ds  = load_v9_dataset(date)
        rm  = _load_regime_up_to(date)
        rec = compute_signals(ds, rm)
        b   = rec.btc
        assert b.pts_sma200 == fix["btc_pts_sma200"], f"{date}: pts_sma200"
        assert b.pts_mom90  == fix["btc_pts_mom90"],  f"{date}: pts_mom90"
        assert b.pts_mom30  == fix["btc_pts_mom30"],  f"{date}: pts_mom30"
        assert b.pts_vol    == fix["btc_pts_vol"],    f"{date}: pts_vol"
        assert b.pts_bonus  == fix["btc_pts_bonus"],  f"{date}: pts_bonus"

    @pytest.mark.parametrize("date", list(_FIXTURES.keys()))
    def test_stock_score_exact(self, date):
        fix = _FIXTURES[date]
        ds  = load_v9_dataset(date)
        rm  = _load_regime_up_to(date)
        rec = compute_signals(ds, rm)
        assert rec.stock_score == fix["stock_score"], (
            f"{date}: stock_score {rec.stock_score} != {fix['stock_score']}"
        )

    @pytest.mark.parametrize("date", list(_FIXTURES.keys()))
    def test_individual_stock_scores_exact(self, date):
        fix    = _FIXTURES[date]
        ds     = load_v9_dataset(date)
        rm     = _load_regime_up_to(date)
        rec    = compute_signals(ds, rm)
        by_ticker = {s.ticker: s.score for s in rec.stocks}
        assert by_ticker.get("AMD")   == fix["amd_score"],   f"{date}: AMD"
        assert by_ticker.get("GOOGL") == fix["googl_score"], f"{date}: GOOGL"
        assert by_ticker.get("MSTR")  == fix["mstr_score"],  f"{date}: MSTR"
        assert by_ticker.get("SPOT")  == fix["spot_score"],  f"{date}: SPOT"

    @pytest.mark.parametrize("date", list(_FIXTURES.keys()))
    def test_allocation_regime_exact(self, date):
        fix = _FIXTURES[date]
        ds  = load_v9_dataset(date)
        rm  = _load_regime_up_to(date)
        rec = compute_signals(ds, rm)
        assert rec.allocation_regime == fix["allocation_regime"], (
            f"{date}: regime {rec.allocation_regime!r} != {fix['allocation_regime']!r}"
        )

    @pytest.mark.parametrize("date", list(_FIXTURES.keys()))
    def test_allocation_weights_exact(self, date):
        fix = _FIXTURES[date]
        ds  = load_v9_dataset(date)
        rm  = _load_regime_up_to(date)
        rec = compute_signals(ds, rm)
        assert rec.wc == fix["wc"], f"{date}: wc {rec.wc} != {fix['wc']}"
        assert rec.ws == fix["ws"], f"{date}: ws {rec.ws} != {fix['ws']}"

    @pytest.mark.parametrize("date", list(_FIXTURES.keys()))
    def test_record_is_json_serialisable(self, date):
        ds  = load_v9_dataset(date)
        rm  = _load_regime_up_to(date)
        rec = compute_signals(ds, rm)
        j   = rec.to_json()
        assert json.loads(j)["date"] == date

    @pytest.mark.parametrize("date", list(_FIXTURES.keys()))
    def test_record_round_trips_exactly(self, date):
        ds  = load_v9_dataset(date)
        rm  = _load_regime_up_to(date)
        rec = compute_signals(ds, rm)
        rec2 = DailySignalRecord.from_json(rec.to_json())
        assert rec2.crypto_score == rec.crypto_score
        assert rec2.stock_score  == rec.stock_score
        assert rec2.wc           == rec.wc


# ── allocation_regime_name() label coverage ───────────────────────────────────

class TestAllocationRegimeName:
    def test_both_hot(self):
        assert allocation_regime_name(4.0, 3.0) == "BOTH_HOT"

    def test_crypto_dominant(self):
        # cn≥0.65 but sn<0.65
        assert allocation_regime_name(4.0, 1.0) == "CRYPTO_DOMINANT"

    def test_stock_dominant(self):
        # sn≥0.65 but cn<0.65
        assert allocation_regime_name(1.0, 3.0) == "STOCK_DOMINANT"

    def test_defensive(self):
        # cn<0.35 and sn<0.35
        assert allocation_regime_name(0.5, 0.5) == "DEFENSIVE"

    def test_neutral(self):
        # neither hot nor cold on both axes
        assert allocation_regime_name(2.0, 1.5) == "NEUTRAL"

    def test_boundary_both_hot_exact(self):
        # cn = 0.65 exactly → hot
        assert allocation_regime_name(2.6, 1.95) == "BOTH_HOT"

    def test_boundary_cold_exact(self):
        # cn = 0.35 exactly → NOT cold (cold is strictly <0.35)
        # 0.35 * 4.0 = 1.4
        result = allocation_regime_name(1.4, 1.4)
        assert result != "DEFENSIVE"

    def test_all_labels_are_strings(self):
        for c_sc in [0.0, 1.0, 2.0, 3.0, 4.0]:
            for s_sc in [0.0, 0.75, 1.5, 2.25, 3.0]:
                name = allocation_regime_name(c_sc, s_sc)
                assert isinstance(name, str)
                assert len(name) > 0
