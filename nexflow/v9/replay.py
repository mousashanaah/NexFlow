"""
V9 Confidence — Module 2: Historical Replay Validator

Addresses RISK 00 (Backtest/Production Logic Drift).

Every night, this runs two engines against identical historical data:
  1. Research engine  — imported directly from scripts/test_v9_confidence.py
  2. Production engine — nexflow/v9/core.py (the single source of truth)

Any discrepancy in regime state, score, allocation, or signal:
  - Is logged in full detail
  - Triggers a ReplayDivergenceError
  - Blocks trading until resolved manually

Usage:
  python -m nexflow.v9.replay [--date YYYY-MM-DD] [--verbose]

  Or from code:
    from nexflow.v9.replay import run_replay
    report = run_replay(up_to_date="2025-01-15")
    if not report.passed:
        raise SystemExit("Replay divergence — trading blocked")
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

# ── Tolerance ─────────────────────────────────────────────────────────────────

# Floating-point differences below this are not divergence — they are noise
SCORE_TOLERANCE      = 1e-6
WEIGHT_TOLERANCE     = 1e-6
REGIME_TOLERANCE     = 0      # bool match is exact — no tolerance


# ── Report structures ─────────────────────────────────────────────────────────

@dataclass
class ReplayCheck:
    name:        str
    research:    object
    production:  object
    passed:      bool
    detail:      str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        msg    = f"  [{status}] {self.name}"
        if not self.passed:
            msg += f"\n         research={self.research!r}"
            msg += f"\n         production={self.production!r}"
            if self.detail:
                msg += f"\n         {self.detail}"
        return msg


@dataclass
class ReplayReport:
    run_date:    str
    up_to_date:  str
    checks:      list[ReplayCheck] = field(default_factory=list)
    passed:      bool              = True
    error:       Optional[str]     = None

    def add(self, check: ReplayCheck) -> None:
        self.checks.append(check)
        if not check.passed:
            self.passed = False

    def summary(self) -> str:
        total  = len(self.checks)
        failed = sum(1 for c in self.checks if not c.passed)
        status = "PASSED" if self.passed else f"FAILED ({failed}/{total} checks)"
        lines  = [
            "=" * 72,
            f"  REPLAY VALIDATOR — {status}",
            f"  Run at: {self.run_date}   Data up to: {self.up_to_date}",
            "=" * 72,
        ]
        if self.error:
            lines.append(f"  ERROR: {self.error}")
        for c in self.checks:
            lines.append(str(c))
        if not self.passed:
            lines += [
                "",
                "  !! DIVERGENCE DETECTED !!",
                "  Trading is BLOCKED until root cause is identified.",
                "  Compare nexflow/v9/core.py to scripts/test_v9_confidence.py",
                "  and scripts/backtest_full_regime_system.py.",
            ]
        lines.append("=" * 72)
        return "\n".join(lines)


class ReplayDivergenceError(RuntimeError):
    """Raised when any replay check fails. Trading must halt."""


# ── Data loader (shared by both engines) ─────────────────────────────────────

def _load_btc_history(up_to_date: str) -> dict:
    """
    Load BTC 1D candles from local parquet store up to and including up_to_date.
    Returns dict with arrays: ts, close, sma200, sma50, mom90, mom30, atr14, atr_avg, dates.
    """
    import pandas as pd

    path = _REPO / "data" / "candles" / "BTCUSDT_1D.parquet"
    df   = pd.read_parquet(path).sort_values("open_time").reset_index(drop=True)

    # filter to up_to_date
    cutoff_ms = int(datetime.strptime(up_to_date, "%Y-%m-%d").timestamp() * 1000) + 86_400_000
    df = df[df["open_time"] <= cutoff_ms].copy()

    c  = df["close"].values.astype(float)
    h  = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    ts = df["open_time"].values.astype(np.int64)
    n  = len(c)

    from nexflow.v9.core import sma as _sma
    sma200 = _sma(c, 200)
    sma50  = _sma(c, 50)
    # warmup proxy — matches test_v9_confidence.py lines 78-79
    sma200 = np.where(np.isfinite(sma200), sma200, sma50)

    mom90 = np.full(n, np.nan); mom30 = np.full(n, np.nan)
    for i in range(90, n): mom90[i] = c[i] / c[i - 90] - 1
    for i in range(30, n): mom30[i] = c[i] / c[i - 30] - 1
    m14 = np.full(n, np.nan)
    for i in range(14, n): m14[i]  = c[i] / c[i - 14] - 1
    mom90 = np.where(np.isfinite(mom90), mom90, np.where(np.isfinite(mom30), mom30, m14))
    mom30 = np.where(np.isfinite(mom30), mom30, m14)

    atr = np.full(n, np.nan)
    for i in range(1, n):
        atr[i] = max(h[i] - lo[i], abs(h[i] - c[i-1]), abs(lo[i] - c[i-1]))
    atr14   = _sma(atr, 14)
    atr_avg = _sma(atr14, 60)
    atr_avg = np.where(np.isfinite(atr_avg), atr_avg, _sma(atr14, 20))

    dates = [datetime.utcfromtimestamp(t / 1000).strftime("%Y-%m-%d") for t in ts]
    byts  = {int(t): i for i, t in enumerate(ts)}

    return dict(ts=ts, close=c, sma200=sma200, sma50=sma50,
                mom90=mom90, mom30=mom30, atr14=atr14, atr_avg=atr_avg,
                dates=dates, byts=byts)


def _load_stock_history(ticker: str, up_to_date: str) -> dict:
    """Load stock candles and compute all indicators via the research loader."""
    from test_stock_risk_mgmt import _load as _research_load
    from nexflow.v9.core import sma as _sma, ema as _ema

    d = _research_load(ticker)
    if not d:
        return {"close": np.array([]), "sma200": np.array([]),
                "mom90": np.array([]), "ema_f": np.array([]),
                "ema_s": np.array([]), "ts": np.array([], dtype=np.int64), "byts": {}}

    ts  = np.array(d["ts"],    dtype=np.int64)
    cl  = np.array(d["close"], dtype=float)

    cutoff_ms = int(datetime.strptime(up_to_date, "%Y-%m-%d").timestamp() * 1000) + 86_400_000
    mask = ts <= cutoff_ms
    ts  = ts[mask]; cl = cl[mask]

    n = len(cl)
    s200 = _sma(cl, 200); s50 = _sma(cl, 50)
    s200 = np.where(np.isfinite(s200), s200, s50)

    mom90 = np.full(n, np.nan)
    for i in range(90, n):
        if cl[i-90] > 0: mom90[i] = cl[i]/cl[i-90] - 1

    ema_f = _ema(cl, 8)
    ema_s = _ema(cl, 21)

    return {
        "close":  cl,
        "sma200": s200,
        "mom90":  mom90,
        "ema_f":  ema_f,
        "ema_s":  ema_s,
        "ts":     ts,
        "byts":   {int(t): i for i, t in enumerate(ts)},
    }


# ── Research engine snapshot ──────────────────────────────────────────────────

def _research_snapshot(btc: dict, stocks: dict[str, dict], ts_val: int) -> dict:
    """
    Compute scores using the research engine (test_v9_confidence.py functions).
    Returns dict with c_sc, s_sc, wc, ws.
    """
    from test_v9_confidence import crypto_score as _r_cscore, stock_score as _r_sscore, allocate as _r_alloc

    c_sc = _r_cscore(ts_val)
    s_sc = _r_sscore(list(stocks.keys()), ts_val)
    wc, ws = _r_alloc(c_sc, s_sc)
    return dict(c_sc=c_sc, s_sc=s_sc, wc=wc, ws=ws)


# ── Production engine snapshot ────────────────────────────────────────────────

def _production_snapshot(btc: dict, stocks: dict[str, dict], ts_val: int) -> dict:
    """
    Compute scores using the production engine (nexflow/v9/core.py).
    Returns dict with c_sc, s_sc, wc, ws.
    """
    from nexflow.v9.core import (
        crypto_score as _p_cscore,
        stock_score_single, stock_score_portfolio,
        allocate as _p_alloc,
    )

    bi = btc["byts"].get(ts_val)
    if bi is None:
        return dict(c_sc=2.0, s_sc=2.0, wc=0.50, ws=0.50)

    c_sc = _p_cscore(
        btc_close = btc["close"][bi],
        sma200    = btc["sma200"][bi],
        mom90     = btc["mom90"][bi],
        mom30     = btc["mom30"][bi],
        atr14     = btc["atr14"][bi],
        atr_avg   = btc["atr_avg"][bi],
    )

    ticker_scores = []
    for ticker, sd in stocks.items():
        si = sd["byts"].get(ts_val)
        if si is None:
            continue
        ticker_scores.append(stock_score_single(
            close = sd["close"][si],
            s200  = sd["sma200"][si],
            mom90 = sd["mom90"][si],
            ema_f = sd["ema_f"][si],
            ema_s = sd["ema_s"][si],
        ))
    s_sc = stock_score_portfolio(ticker_scores)
    wc, ws = _p_alloc(c_sc, s_sc)
    return dict(c_sc=c_sc, s_sc=s_sc, wc=wc, ws=ws)


# ── Regime parity check ───────────────────────────────────────────────────────

def _check_regime_parity(btc: dict, report: ReplayReport) -> None:
    """
    Re-run the bear regime state machine through all historical bars using both
    research-compatible logic and production RegimeMachine, then compare final state.
    """
    from nexflow.v9.core import RegimeMachine, BEAR_DROP_PCT, BEAR_CONFIRM_DAYS

    # Production: use RegimeMachine
    prod_machine = RegimeMachine()
    for i in range(len(btc["close"])):
        prod_machine.step(
            close  = btc["close"][i],
            sma200 = btc["sma200"][i],
            mom30  = btc["mom30"][i],
        )

    # Research: replicate exact logic from backtest_full_regime_system.py
    in_bear_r  = False
    consec_r   = 0
    for i in range(len(btc["close"])):
        c  = btc["close"][i]
        s  = btc["sma200"][i]
        m  = btc["mom30"][i]
        if not np.isfinite(s): s = c * 1.1  # warmup fallback → above
        above = c > s
        if in_bear_r:
            consec_r = (consec_r + 1) if above else 0
            if consec_r >= BEAR_CONFIRM_DAYS:
                in_bear_r = False; consec_r = 0
        else:
            if np.isfinite(m) and m < BEAR_DROP_PCT and not above:
                in_bear_r = True; consec_r = 0

    bear_match   = prod_machine.in_bear == in_bear_r
    consec_match = (prod_machine.consecutive_above == consec_r
                    or not in_bear_r)  # outside bear, counter is irrelevant

    report.add(ReplayCheck(
        name       = "Regime: in_bear matches research",
        research   = in_bear_r,
        production = prod_machine.in_bear,
        passed     = bear_match,
        detail     = f"Last date: {btc['dates'][-1]}",
    ))
    report.add(ReplayCheck(
        name       = "Regime: consecutive_above matches research",
        research   = consec_r,
        production = prod_machine.consecutive_above,
        passed     = consec_match,
        detail     = "Only checked when in_bear=True",
    ))


# ── Score and allocation parity checks ───────────────────────────────────────

def _check_score_parity(
    btc:     dict,
    stocks:  dict[str, dict],
    ts_val:  int,
    label:   str,
    report:  ReplayReport,
) -> None:
    r = _research_snapshot(btc, stocks, ts_val)
    p = _production_snapshot(btc, stocks, ts_val)

    report.add(ReplayCheck(
        name       = f"[{label}] crypto_score",
        research   = round(r["c_sc"], 6),
        production = round(p["c_sc"], 6),
        passed     = abs(r["c_sc"] - p["c_sc"]) <= SCORE_TOLERANCE,
    ))
    report.add(ReplayCheck(
        name       = f"[{label}] stock_score",
        research   = round(r["s_sc"], 6),
        production = round(p["s_sc"], 6),
        passed     = abs(r["s_sc"] - p["s_sc"]) <= SCORE_TOLERANCE,
    ))
    report.add(ReplayCheck(
        name       = f"[{label}] allocation wc",
        research   = r["wc"],
        production = p["wc"],
        passed     = abs(r["wc"] - p["wc"]) <= WEIGHT_TOLERANCE,
    ))
    report.add(ReplayCheck(
        name       = f"[{label}] allocation ws",
        research   = r["ws"],
        production = p["ws"],
        passed     = abs(r["ws"] - p["ws"]) <= WEIGHT_TOLERANCE,
    ))


# ── Main entry point ──────────────────────────────────────────────────────────

COMBO = ["AMD", "GOOGL", "MSTR", "SPOT"]

# Sample dates that span all known allocation regimes (from TEST 61 log)
_SAMPLE_DATES = [
    "2022-03-01",   # DEFENSIVE — both cold, 20% cash
    "2022-06-01",   # BOTH HOT
    "2023-06-01",   # BOTH HOT / CRYPTO DOMINANT period
    "2024-03-01",   # CRYPTO DOMINANT
    "2025-01-01",   # Various
]


def run_replay(
    up_to_date: Optional[str] = None,
    verbose:    bool          = False,
    raise_on_fail: bool       = True,
) -> ReplayReport:
    """
    Run full parity check between research engine and production engine.

    Args:
        up_to_date:    ISO date string; defaults to yesterday
        verbose:       print report to stdout
        raise_on_fail: raise ReplayDivergenceError if any check fails

    Returns:
        ReplayReport with .passed and full .checks list
    """
    if up_to_date is None:
        from datetime import timedelta
        up_to_date = (date.today() - timedelta(days=1)).isoformat()

    report = ReplayReport(
        run_date   = datetime.utcnow().isoformat() + "Z",
        up_to_date = up_to_date,
    )

    try:
        # Load shared data once
        btc    = _load_btc_history(up_to_date)
        stocks = {t: _load_stock_history(t, up_to_date) for t in COMBO}

        # 1. Regime parity (full history)
        _check_regime_parity(btc, report)

        # 2. Score + allocation parity at key sample dates
        for sample_date in _SAMPLE_DATES:
            if sample_date > up_to_date:
                continue
            ts_val = int(
                datetime.strptime(sample_date, "%Y-%m-%d").timestamp() * 1000
            )
            # find closest available BTC timestamp
            closest = min(btc["byts"].keys(), key=lambda t: abs(t - ts_val))
            actual_date = btc["dates"][btc["byts"][closest]]
            _check_score_parity(btc, stocks, closest, actual_date, report)

        # 3. Most recent rebalance date (last bar in data)
        last_ts = int(btc["ts"][-1])
        _check_score_parity(btc, stocks, last_ts, f"latest ({btc['dates'][-1]})", report)

    except Exception as e:
        report.passed = False
        report.error  = str(e)

    if verbose:
        print(report.summary())

    if raise_on_fail and not report.passed:
        raise ReplayDivergenceError(
            f"Replay divergence on {up_to_date}. Trading BLOCKED.\n"
            + report.summary()
        )

    return report


# ── Daily parity check (production operations mode) ───────────────────────────

@dataclass
class DailyParityResult:
    """
    Result of a single-day parity check run after live signal computation.
    Intended to be called EVERY DAY as part of normal operations.
    """
    date:              str
    regime_match:      bool
    crypto_score_match: bool
    stock_score_match: bool
    allocation_match:  bool
    research_state:    dict
    production_state:  dict
    passed:            bool

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL !! DIVERGENCE DETECTED !!"
        lines = [
            f"  Daily parity [{self.date}]: {status}",
            f"    Regime:    {'✓' if self.regime_match else '✗'}  "
            f"research={self.research_state.get('in_bear')}  "
            f"prod={self.production_state.get('in_bear')}",
            f"    CryptoSc:  {'✓' if self.crypto_score_match else '✗'}  "
            f"research={self.research_state.get('c_sc', '?'):.4f}  "
            f"prod={self.production_state.get('c_sc', '?'):.4f}",
            f"    StockSc:   {'✓' if self.stock_score_match else '✗'}  "
            f"research={self.research_state.get('s_sc', '?'):.4f}  "
            f"prod={self.production_state.get('s_sc', '?'):.4f}",
            f"    Alloc:     {'✓' if self.allocation_match else '✗'}  "
            f"research=({self.research_state.get('wc','?'):.0%}C/"
            f"{self.research_state.get('ws','?'):.0%}S)  "
            f"prod=({self.production_state.get('wc','?'):.0%}C/"
            f"{self.production_state.get('ws','?'):.0%}S)",
        ]
        return "\n".join(lines)


def run_daily_parity(
    as_of_date:    str,
    verbose:       bool = False,
    raise_on_fail: bool = True,
) -> DailyParityResult:
    """
    Run single-day parity check between research and production engines.

    Called EVERY DAY after signal computation, including after deployment.
    This is the operational guard against RISK 00 (Backtest/Production Logic Drift).

    Args:
        as_of_date:    ISO date of the bar to check (yesterday in normal ops)
        verbose:       print result to stdout
        raise_on_fail: raise ReplayDivergenceError on any mismatch

    Returns:
        DailyParityResult
    """
    from nexflow.v9.core import (
        crypto_score as _prod_cscore,
        stock_score_single, stock_score_portfolio,
        allocate as _prod_alloc,
        RegimeMachine, sma,
    )
    from nexflow.v9.data import load_v9_dataset, DataValidationError

    # Load data
    try:
        ds = load_v9_dataset(as_of_date)
    except DataValidationError as e:
        raise ReplayDivergenceError(f"Daily parity blocked by data error: {e}") from e

    btc    = _load_btc_history(as_of_date)
    stocks = {t: _load_stock_history(t, as_of_date) for t in COMBO}
    last_ts = int(btc["ts"][-1])

    # Research snapshot
    res = _research_snapshot(btc, stocks, last_ts)

    # Production snapshot
    prod = _production_snapshot(btc, stocks, last_ts)

    # Regime via production RegimeMachine (full history)
    btc_cs = ds.btc()
    machine = RegimeMachine()
    for i in range(len(btc_cs.ts)):
        machine.step(
            float(btc_cs.close[i]),
            float(btc_cs.sma200[i]) if np.isfinite(btc_cs.sma200[i]) else float(btc_cs.close[i]) * 1.1,
            float(btc_cs.mom30[i]),
        )

    # Research regime (re-derived from same data for comparison)
    from nexflow.v9.core import BEAR_DROP_PCT, BEAR_CONFIRM_DAYS
    in_bear_r = False; consec_r = 0
    for i in range(len(btc_cs.ts)):
        c = float(btc_cs.close[i])
        s = float(btc_cs.sma200[i]) if np.isfinite(btc_cs.sma200[i]) else c * 1.1
        m = float(btc_cs.mom30[i])
        above = c > s
        if in_bear_r:
            consec_r = (consec_r + 1) if above else 0
            if consec_r >= BEAR_CONFIRM_DAYS: in_bear_r = False; consec_r = 0
        else:
            if np.isfinite(m) and m < BEAR_DROP_PCT and not above:
                in_bear_r = True; consec_r = 0

    research_state = {
        "in_bear": in_bear_r,
        "c_sc": res["c_sc"],
        "s_sc": res["s_sc"],
        "wc":   res["wc"],
        "ws":   res["ws"],
    }
    production_state = {
        "in_bear": machine.in_bear,
        "c_sc": prod["c_sc"],
        "s_sc": prod["s_sc"],
        "wc":   prod["wc"],
        "ws":   prod["ws"],
    }

    regime_match  = research_state["in_bear"] == production_state["in_bear"]
    cscore_match  = abs(research_state["c_sc"] - production_state["c_sc"]) <= SCORE_TOLERANCE
    sscore_match  = abs(research_state["s_sc"] - production_state["s_sc"]) <= SCORE_TOLERANCE
    alloc_match   = (abs(research_state["wc"] - production_state["wc"]) <= WEIGHT_TOLERANCE and
                     abs(research_state["ws"] - production_state["ws"]) <= WEIGHT_TOLERANCE)

    result = DailyParityResult(
        date               = as_of_date,
        regime_match       = regime_match,
        crypto_score_match = cscore_match,
        stock_score_match  = sscore_match,
        allocation_match   = alloc_match,
        research_state     = research_state,
        production_state   = production_state,
        passed             = all([regime_match, cscore_match, sscore_match, alloc_match]),
    )

    if verbose:
        print(result.summary())

    if raise_on_fail and not result.passed:
        raise ReplayDivergenceError(
            f"Daily parity FAILED on {as_of_date}. Trading BLOCKED.\n"
            + result.summary()
        )

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="V9 Confidence — historical replay validator"
    )
    parser.add_argument("--date", default=None,
                        help="Validate up to this date (YYYY-MM-DD). Default: yesterday.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full report to stdout.")
    parser.add_argument("--no-raise", action="store_true",
                        help="Print result but do not raise on failure (for CI inspection).")
    parser.add_argument("--daily", action="store_true",
                        help="Run single-day parity check instead of full replay.")
    args = parser.parse_args()

    if args.daily:
        result = run_daily_parity(
            as_of_date    = args.date or (date.today() - __import__("datetime").timedelta(days=1)).isoformat(),
            verbose       = True,
            raise_on_fail = not args.no_raise,
        )
        sys.exit(0 if result.passed else 1)
    else:
        report = run_replay(
            up_to_date    = args.date,
            verbose       = True,
            raise_on_fail = not args.no_raise,
        )
        sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
