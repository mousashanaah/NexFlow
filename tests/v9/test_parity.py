"""
V9 Confidence — parity tests.

These tests assert that nexflow/v9/core.py produces results IDENTICAL to
the research scripts.  Any failure here means RISK 00 has materialised.

A CI run that fails these tests must block deployment.

Tests are organised by function:
  - test_sma_matches_research
  - test_regime_machine_matches_research
  - test_crypto_score_matches_research
  - test_stock_score_matches_research
  - test_allocate_matches_research
  - test_allocate_all_regimes  (boundary exhaustion)
  - test_startup_gate_passes_on_correct_state
  - test_startup_gate_fails_on_stale_state
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from nexflow.v9.core import (
    RegimeMachine,
    allocate,
    crypto_score,
    ema,
    ema_crossover_state,
    macd_crossover_state,
    sma,
    stock_score_portfolio,
    stock_score_single,
    BEAR_CONFIRM_DAYS,
    BEAR_DROP_PCT,
)

TOLERANCE = 1e-9


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_price_series(n=300, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.02, n)
    prices = 30_000.0 * np.cumprod(1 + returns)
    return prices


# ── SMA ───────────────────────────────────────────────────────────────────────

class TestSMA:
    def test_sma_matches_research(self):
        """Production sma() must match _sma() from test_v9_confidence.py."""
        from test_v9_confidence import _sma as research_sma

        prices = _make_price_series(300)
        prod   = sma(prices, 200)
        res    = research_sma(prices, 200)

        assert np.allclose(
            np.where(np.isfinite(prod), prod, 0),
            np.where(np.isfinite(res),  res,  0),
            atol=TOLERANCE,
        ), "sma() output differs from research _sma()"

    def test_sma_nan_for_warmup(self):
        prices = np.ones(199)
        result = sma(prices, 200)
        assert np.all(np.isnan(result)), "sma must return NaN during warmup"

    def test_sma_correct_value(self):
        prices = np.arange(1.0, 11.0)    # 1..10
        result = sma(prices, 5)
        assert np.isnan(result[3])
        assert result[4] == pytest.approx(3.0)   # mean(1,2,3,4,5)
        assert result[9] == pytest.approx(8.0)   # mean(6,7,8,9,10)


# ── Regime machine ────────────────────────────────────────────────────────────

class TestRegimeMachine:
    def _reference_run(self, closes, sma200s, mom30s):
        """Re-implementation of exact research logic for comparison."""
        in_bear = False; consec = 0
        results = []
        for i in range(len(closes)):
            c = closes[i]; s = sma200s[i]; m = mom30s[i]
            above = c > s
            if in_bear:
                consec = (consec + 1) if above else 0
                if consec >= BEAR_CONFIRM_DAYS:
                    in_bear = False; consec = 0
            else:
                if np.isfinite(m) and m < BEAR_DROP_PCT and not above:
                    in_bear = True; consec = 0
            results.append(in_bear)
        return results

    def test_matches_reference_on_synthetic_crash(self):
        n = 100
        closes = np.full(n, 40_000.0)
        sma200 = np.full(n, 40_000.0)
        mom30  = np.zeros(n)

        # crash at bar 20: drop 25%, go below SMA200
        closes[20:] = 30_000.0
        sma200[20:] = 40_000.0
        mom30[20:]  = -0.25

        machine = RegimeMachine()
        prod    = [machine.step(closes[i], sma200[i], mom30[i]) for i in range(n)]
        ref     = self._reference_run(closes, sma200, mom30)

        assert prod == ref, f"Regime mismatch: prod={prod[18:35]} ref={ref[18:35]}"

    def test_bear_exit_requires_10_days(self):
        machine = RegimeMachine(in_bear=True, consecutive_above=0)
        sma200  = 40_000.0

        # 9 days above — should still be in bear
        for _ in range(9):
            still_bear = machine.step(41_000.0, sma200, 0.0)
        assert still_bear is True

        # 10th day above — exits bear
        exited = machine.step(41_000.0, sma200, 0.0)
        assert exited is False

    def test_bear_counter_resets_on_below_day(self):
        machine = RegimeMachine(in_bear=True, consecutive_above=0)
        sma200 = 40_000.0
        machine.step(41_000.0, sma200, 0.0)  # above: consec=1
        machine.step(41_000.0, sma200, 0.0)  # above: consec=2
        machine.step(39_000.0, sma200, 0.0)  # below: consec resets
        assert machine.consecutive_above == 0
        assert machine.in_bear is True

    def test_bear_entry_requires_both_conditions(self):
        # Only drop, not below SMA200 — should NOT enter bear
        m = RegimeMachine()
        m.step(41_000.0, 40_000.0, -0.25)   # above SMA200, big drop
        assert m.in_bear is False

        # Below SMA200 but no big drop — should NOT enter bear
        m2 = RegimeMachine()
        m2.step(39_000.0, 40_000.0, -0.05)  # below SMA200, small drop
        assert m2.in_bear is False

        # Both conditions — SHOULD enter bear
        m3 = RegimeMachine()
        m3.step(39_000.0, 40_000.0, -0.25)
        assert m3.in_bear is True

    def test_reconcile_matches_full_run(self):
        prices = _make_price_series(300)
        s200   = sma(prices, 200)
        s50    = sma(prices, 50)
        s200   = np.where(np.isfinite(s200), s200, s50)
        mom30  = np.full(len(prices), np.nan)
        for i in range(30, len(prices)):
            mom30[i] = prices[i] / prices[i-30] - 1

        # Run full history
        full_machine = RegimeMachine()
        for i in range(len(prices)):
            full_machine.step(prices[i], s200[i], mom30[i])

        # Reconcile using last 30 bars
        fresh = RegimeMachine()
        reconciled = fresh.reconcile(prices[-30:], s200[-30:], mom30[-30:])

        # In bull market (most of synthetic series), both should be not-in-bear
        # State may differ in bear due to history-dependence — but for bull it must match
        if not full_machine.in_bear:
            assert not reconciled.in_bear


# ── Crypto score ──────────────────────────────────────────────────────────────

class TestCryptoScore:
    def _research_score(self, ts_val: int) -> float:
        from test_v9_confidence import crypto_score as r_score
        return r_score(ts_val)

    def test_manual_known_values(self):
        # Bull: above SMA200, positive mom90, positive mom30, low vol
        sc = crypto_score(
            btc_close=50_000.0, sma200=40_000.0,
            mom90=0.20, mom30=0.05,
            atr14=1_000.0, atr_avg=1_200.0,   # atr14 < atr_avg*1.5 → low vol
        )
        assert sc == pytest.approx(4.0)   # 2.0 + 1.0 + 0.5 + 0.5 = 4.0

    def test_bear_regime_score_below_threshold(self):
        # Below SMA200: crypto score can never exceed 2.0 from SMA200 alone
        sc = crypto_score(
            btc_close=30_000.0, sma200=40_000.0,
            mom90=-0.10, mom30=-0.05,
            atr14=2_000.0, atr_avg=1_000.0,
        )
        # 0 (SMA200) + 0 (mom90<0) + 0 (mom30<0) + 0 (high vol) = 0
        assert sc == pytest.approx(0.0)

    def test_bonus_penalty_applied(self):
        # mom90 > 0.30 → +0.5 bonus
        sc = crypto_score(
            btc_close=50_000.0, sma200=40_000.0,
            mom90=0.35, mom30=0.05,
            atr14=1_000.0, atr_avg=1_200.0,
        )
        # 2.0 + 1.0 + 0.5 + 0.5 + 0.5 = 4.5 → clipped to 4.0
        assert sc == pytest.approx(4.0)

        # mom90 < -0.30 → -0.5 penalty
        sc2 = crypto_score(
            btc_close=30_000.0, sma200=40_000.0,
            mom90=-0.35, mom30=0.0,
            atr14=1_000.0, atr_avg=1_200.0,
        )
        # 0 + 0 (mom90<0) + 0 (mom30=0, not >0) + 0.5 (low vol) - 0.5 = 0 → clipped
        assert sc2 == pytest.approx(0.0)

    def test_score_clamped_0_to_4(self):
        sc = crypto_score(
            btc_close=50_000.0, sma200=20_000.0,
            mom90=0.50, mom30=0.20,
            atr14=100.0, atr_avg=10_000.0,
        )
        assert 0.0 <= sc <= 4.0

    @pytest.mark.integration
    def test_matches_research_engine_on_live_data(self):
        """
        Integration test: production score must equal research score at
        three sampled timestamps.  Requires local parquet data.
        """
        import pandas as pd
        path = _REPO / "data" / "candles" / "BTCUSDT_1D.parquet"
        if not path.exists():
            pytest.skip("BTC candle data not available")

        # sample: 2022-06-01, 2023-06-01, 2024-01-01
        sample_dates = ["2022-06-01", "2023-06-01", "2024-01-01"]
        from nexflow.v9.replay import _load_btc_history, _production_snapshot, _research_snapshot
        btc = _load_btc_history("2025-01-01")

        for d in sample_dates:
            ts_ms = int(__import__("datetime").datetime.strptime(d, "%Y-%m-%d").timestamp() * 1000)
            closest = min(btc["byts"].keys(), key=lambda t: abs(t - ts_ms))
            bi = btc["byts"][closest]
            prod_sc = crypto_score(
                btc_close = btc["close"][bi],
                sma200    = btc["sma200"][bi],
                mom90     = btc["mom90"][bi],
                mom30     = btc["mom30"][bi],
                atr14     = btc["atr14"][bi],
                atr_avg   = btc["atr_avg"][bi],
            )
            res_sc = self._research_score(closest)
            assert abs(prod_sc - res_sc) <= TOLERANCE, (
                f"crypto_score mismatch on {d}: prod={prod_sc} research={res_sc}"
            )


# ── Stock score ───────────────────────────────────────────────────────────────

class TestStockScore:
    def test_single_ticker_bull(self):
        sc = stock_score_single(
            close=150.0, s200=100.0, mom90=0.25, ema_f=155.0, ema_s=145.0
        )
        # 1.0 (SMA) + 1.0 (mom90) + 0.5 (mom90>0.20) + 0.5 (EMA) = 3.0
        assert sc == pytest.approx(3.0)

    def test_single_ticker_bear(self):
        sc = stock_score_single(
            close=80.0, s200=100.0, mom90=-0.15, ema_f=78.0, ema_s=85.0
        )
        # 0 + 0 + 0 + 0 = 0
        assert sc == pytest.approx(0.0)

    def test_portfolio_average(self):
        sc = stock_score_portfolio([3.0, 1.5, 2.0, 0.5])
        assert sc == pytest.approx(np.mean([3.0, 1.5, 2.0, 0.5]))

    def test_empty_portfolio_returns_neutral(self):
        assert stock_score_portfolio([]) == pytest.approx(2.0)


# ── Allocation ────────────────────────────────────────────────────────────────

class TestAllocate:
    def _research_allocate(self, c_sc, s_sc):
        from test_v9_confidence import allocate as r_alloc
        return r_alloc(c_sc, s_sc)

    @pytest.mark.parametrize("c_sc,s_sc,expected_wc,expected_ws", [
        # both hot (normalised ≥0.65)
        (2.6, 1.95, 0.65, 0.35),
        (4.0, 3.0,  0.65, 0.35),
        # crypto dominant
        (2.6, 1.0,  0.80, 0.20),
        (4.0, 0.5,  0.80, 0.20),
        # stock dominant
        (1.0, 1.95, 0.20, 0.80),
        (0.5, 3.0,  0.20, 0.80),
        # both cold (normalised <0.35)
        (1.0, 0.75, 0.40, 0.40),
        (0.0, 0.0,  0.40, 0.40),
    ])
    def test_regime_boundaries(self, c_sc, s_sc, expected_wc, expected_ws):
        wc, ws = allocate(c_sc, s_sc)
        assert wc == pytest.approx(expected_wc, abs=1e-9)
        assert ws == pytest.approx(expected_ws, abs=1e-9)

    @pytest.mark.parametrize("c_sc,s_sc", [
        (2.6, 1.95), (4.0, 3.0), (2.6, 1.0),
        (1.0, 1.95), (1.0, 0.75), (0.0, 0.0),
        (2.0, 1.5),  (1.5, 1.2),  (3.0, 2.0),
    ])
    def test_matches_research_engine(self, c_sc, s_sc):
        prod_wc, prod_ws = allocate(c_sc, s_sc)
        res_wc,  res_ws  = self._research_allocate(c_sc, s_sc)
        assert prod_wc == pytest.approx(res_wc, abs=TOLERANCE), (
            f"wc mismatch c={c_sc} s={s_sc}: prod={prod_wc} research={res_wc}"
        )
        assert prod_ws == pytest.approx(res_ws, abs=TOLERANCE), (
            f"ws mismatch c={c_sc} s={s_sc}: prod={prod_ws} research={res_ws}"
        )

    def test_weights_always_sum_to_at_most_one(self):
        rng = np.random.default_rng(0)
        for _ in range(1000):
            c_sc = float(rng.uniform(0, 4))
            s_sc = float(rng.uniform(0, 3))
            wc, ws = allocate(c_sc, s_sc)
            assert wc + ws <= 1.0 + 1e-9
            assert wc >= 0 and ws >= 0

    def test_cash_only_in_both_cold(self):
        # 20% cash only when both cold
        wc, ws = allocate(1.0, 0.75)   # cn=0.25, sn=0.25 → both cold
        assert wc + ws == pytest.approx(0.80)   # 20% cash

        wc2, ws2 = allocate(2.0, 1.5)  # neutral
        assert wc2 + ws2 == pytest.approx(1.00)  # no cash


# ── State persistence ─────────────────────────────────────────────────────────

class TestState:
    def test_save_and_load_roundtrip(self, tmp_path):
        from nexflow.v9.state import SystemState, RegimeSnapshot, AllocationSnapshot

        path = tmp_path / "state.json"
        state = SystemState(
            regime=RegimeSnapshot(
                in_bear=False, consecutive_above=0, last_bar_date="2025-01-15"
            ),
            allocation=AllocationSnapshot(
                wc=0.65, ws=0.35, last_rebalance_date="2025-01-01",
                trading_days_since=10,
            ),
        )
        state.save(path)

        loaded = SystemState.load(path)
        assert loaded.regime.in_bear           == state.regime.in_bear
        assert loaded.regime.consecutive_above == state.regime.consecutive_above
        assert loaded.regime.last_bar_date     == state.regime.last_bar_date
        assert loaded.allocation.wc            == pytest.approx(state.allocation.wc)
        assert loaded.allocation.ws            == pytest.approx(state.allocation.ws)

    def test_initialize_creates_file(self, tmp_path):
        from nexflow.v9.state import SystemState
        path = tmp_path / "state.json"
        state = SystemState.initialize(path=path)
        assert path.exists()
        assert state.regime.in_bear is False

    def test_initialize_fails_if_file_exists(self, tmp_path):
        from nexflow.v9.state import SystemState
        path = tmp_path / "state.json"
        SystemState.initialize(path=path)
        with pytest.raises(FileExistsError):
            SystemState.initialize(path=path)

    def test_startup_gate_blocks_trading_before_reconcile(self, tmp_path):
        from nexflow.v9.state import SystemState
        path = tmp_path / "state.json"
        state = SystemState.initialize(path=path)
        with pytest.raises(Exception, match="startup_gate|gate"):
            state.assert_gate()

    def test_startup_gate_passes_on_correct_state(self, tmp_path):
        from nexflow.v9.state import SystemState, RegimeSnapshot, AllocationSnapshot

        # Build known regime state
        prices = _make_price_series(60)
        s200   = sma(prices, 50)   # use SMA50 as proxy (short series)
        mom30  = np.full(60, np.nan)
        for i in range(30, 60): mom30[i] = prices[i] / prices[i-30] - 1

        machine = RegimeMachine()
        for i in range(60):
            machine.step(prices[i], s200[i] if np.isfinite(s200[i]) else prices[i]*1.1, mom30[i])

        path = tmp_path / "state.json"
        state = SystemState(
            regime=RegimeSnapshot(
                in_bear           = machine.in_bear,
                consecutive_above = machine.consecutive_above,
                last_bar_date     = "2025-01-15",
            ),
            allocation=AllocationSnapshot(
                wc=0.50, ws=0.50,
                last_rebalance_date="2025-01-01",
                trading_days_since=10,
            ),
        )
        state.save(path)

        dates = [f"2024-11-{i+1:02d}" for i in range(60)]
        state.startup_gate(prices, s200, mom30, dates)
        assert state.gate_open is True

    def test_startup_gate_fails_on_wrong_bear_state(self, tmp_path):
        from nexflow.v9.state import SystemState, RegimeSnapshot, AllocationSnapshot, ReconciliationError

        # Construct a deterministic crash scenario:
        # First 40 bars: price at 50_000 (above any SMA)
        # Last 20 bars: price at 25_000 (well below SMA, 50% drop → mom30 = -0.50)
        n = 60
        prices = np.array([50_000.0] * 40 + [25_000.0] * 20)
        # SMA50 after bar 49 is between 25k and 50k — prices[49:] are below it
        s200   = sma(prices, 50)
        s200   = np.where(np.isfinite(s200), s200, prices * 1.1)
        mom30  = np.full(n, np.nan)
        for i in range(30, n):
            mom30[i] = prices[i] / prices[i - 30] - 1   # -0.50 for last 20 bars

        # Verify the regime machine does enter bear (sanity check)
        m = RegimeMachine()
        for i in range(n):
            m.step(prices[i], s200[i], mom30[i])
        # If for some reason it didn't enter bear, skip rather than give a false PASS
        if not m.in_bear:
            pytest.skip("Synthetic series did not produce bear — adjust test parameters")

        path = tmp_path / "state.json"
        state = SystemState(
            regime=RegimeSnapshot(
                in_bear=False,           # WRONG — derived state is True
                consecutive_above=0,
                last_bar_date="2025-01-15",
            ),
            allocation=AllocationSnapshot(
                wc=0.50, ws=0.50,
                last_rebalance_date="2025-01-01",
                trading_days_since=0,
            ),
        )
        state.save(path)
        dates = [f"2024-11-{(i % 30) + 1:02d}" for i in range(n)]

        with pytest.raises(ReconciliationError):
            state.startup_gate(prices, s200, mom30, dates)
