"""
V9 Confidence — canonical signal, scoring, and allocation logic.

This file is the single source of truth.

Rules:
  - No imports from research scripts.
  - No side effects at module level.
  - Every function here has a corresponding test in tests/v9/test_parity.py
    that asserts it produces identical output to the research engine.
  - Changing any threshold, formula, or branching logic here requires:
      1. Updating the corresponding test.
      2. Running the full historical replay and confirming no divergence.
      3. A deliberate version bump in __init__.py.
      4. A commit message explaining the change and why it was authorized.

Validated against:
  scripts/test_v9_confidence.py   (scoring, allocation)
  scripts/backtest_full_regime_system.py  (regime state machine)
"""
from __future__ import annotations

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

CRYPTO_SCORE_MAX  = 4.0
STOCK_SCORE_MAX   = 3.0

# Regime parameters — match _V863_KW in test_v9_confidence.py exactly
BEAR_DROP_PCT     = -0.20   # 30d BTC return threshold
BEAR_CONFIRM_DAYS = 10      # consecutive days above SMA200 to exit bear
MOM_GATE_DAYS     = 20      # per-coin momentum gate lookback

# Allocation thresholds — match allocate() in test_v9_confidence.py exactly
BOTH_HOT_THRESHOLD   = 0.65
BOTH_COLD_THRESHOLD  = 0.35
CRYPTO_LEAD_WC       = 0.65
CRYPTO_LEAD_WS       = 0.35
CRYPTO_DOM_WC        = 0.80
CRYPTO_DOM_WS        = 0.20
STOCK_DOM_WC         = 0.20
STOCK_DOM_WS         = 0.80
DEFENSIVE_WC         = 0.40
DEFENSIVE_WS         = 0.40
# cash = 1 - wc - ws = 0.20 in defensive


# ── Array helpers ─────────────────────────────────────────────────────────────

def sma(series: np.ndarray, n: int) -> np.ndarray:
    """Simple moving average.  Matches _sma() in test_v9_confidence.py."""
    out = np.full(len(series), np.nan)
    for i in range(n - 1, len(series)):
        out[i] = np.mean(series[i - n + 1 : i + 1])
    return out


def ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with SMA seed."""
    out = np.full(len(series), np.nan)
    k = 2.0 / (period + 1)
    # seed at first complete window
    if len(series) < period:
        return out
    out[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def ema_crossover_state(series: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """
    Returns boolean array: True = EMA-fast > EMA-slow (long state).

    Uses crossover state (transition-based), NOT just current relationship.
    A crossover triggers a state change; state persists until next crossover.
    This matches the EMA state logic in backtest_full_regime_system.py.
    """
    ema_f = ema(series, fast)
    ema_s = ema(series, slow)
    above = ema_f > ema_s
    state = np.zeros(len(series), dtype=bool)
    cur = False
    for i in range(len(series)):
        if not np.isfinite(ema_f[i]) or not np.isfinite(ema_s[i]):
            state[i] = cur
            continue
        if above[i] and not (above[i - 1] if i > 0 else above[i]):
            cur = True   # bullish crossover
        elif not above[i] and (above[i - 1] if i > 0 else not above[i]):
            cur = False  # bearish crossover
        state[i] = cur
    return state


def macd_crossover_state(
    series: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> np.ndarray:
    """
    Returns boolean array: True = MACD histogram > 0 (long state).
    Uses crossover transitions, not just sign of histogram.
    Matches _macd_state() in test30_signal_edge.py.
    """
    ema_f  = ema(series, fast)
    ema_s  = ema(series, slow)
    macd_l = ema_f - ema_s
    sig_l  = ema(np.where(np.isfinite(macd_l), macd_l, 0.0), signal)
    hist   = macd_l - sig_l

    state  = np.zeros(len(series), dtype=bool)
    cur    = False
    for i in range(len(series)):
        if not np.isfinite(hist[i]):
            state[i] = cur
            continue
        if hist[i] > 0 and (not state[i - 1] if i > 0 else True):
            cur = True
        elif hist[i] <= 0 and (state[i - 1] if i > 0 else False):
            cur = False
        state[i] = cur
    return state


# ── BTC Regime State Machine ──────────────────────────────────────────────────

class RegimeMachine:
    """
    Stateful BTC bear/bull regime detector.

    Matches the asymmetric_regime + and_entry logic in _run() of
    backtest_full_regime_system.py with bear_drop_pct=-0.20, confirm_days=10.

    Bear ENTRY:  btc_30d_return < -0.20  AND  btc_close < sma200
    Bear EXIT:   10 consecutive daily closes above sma200
    """

    def __init__(self, in_bear: bool = False, consecutive_above: int = 0) -> None:
        self.in_bear           = in_bear
        self.consecutive_above = consecutive_above

    def step(self, close: float, sma200: float, mom30: float) -> bool:
        """
        Advance the state machine by one daily bar.
        Returns True if in bear after this bar.

        Args:
            close:   BTC daily close price
            sma200:  BTC 200-day SMA (or SMA50 proxy during warmup)
            mom30:   BTC 30-day return  = close/close_30d_ago - 1
        """
        above = close > sma200

        if self.in_bear:
            self.consecutive_above = (self.consecutive_above + 1) if above else 0
            if self.consecutive_above >= BEAR_CONFIRM_DAYS:
                self.in_bear           = False
                self.consecutive_above = 0
        else:
            if np.isfinite(mom30) and mom30 < BEAR_DROP_PCT and not above:
                self.in_bear           = True
                self.consecutive_above = 0

        return self.in_bear

    def as_dict(self) -> dict:
        return {
            "in_bear":           self.in_bear,
            "consecutive_above": self.consecutive_above,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeMachine":
        return cls(
            in_bear           = bool(d["in_bear"]),
            consecutive_above = int(d["consecutive_above"]),
        )

    def reconcile(
        self,
        closes:  np.ndarray,
        sma200s: np.ndarray,
        mom30s:  np.ndarray,
    ) -> "RegimeMachine":
        """
        Re-derive state from a historical window and return a fresh machine
        whose state matches what a full-history run would have produced.

        Called at startup to validate stored state matches derived state.
        Use the last max(BEAR_CONFIRM_DAYS + 5, 20) bars for the window.
        """
        fresh = RegimeMachine()
        for i in range(len(closes)):
            fresh.step(closes[i], sma200s[i], mom30s[i])
        return fresh


# ── Confidence Scoring ────────────────────────────────────────────────────────

def crypto_score(
    btc_close: float,
    sma200:    float,
    mom90:     float,
    mom30:     float,
    atr14:     float,
    atr_avg:   float,   # 60d avg of atr14
) -> float:
    """
    BTC crypto confidence score.  Range 0–4.

    Exactly matches crypto_score() in test_v9_confidence.py.
    SMA200 carries 2 pts so bear regimes never trigger CRYPTO DOMINANT.
    """
    sc = 0.0

    if np.isfinite(sma200):
        sc += 2.0 if btc_close > sma200 else 0.0

    if np.isfinite(mom90):
        sc += 1.0 if mom90 > 0  else 0.0
        if mom90 >  0.30: sc += 0.5
        if mom90 < -0.30: sc -= 0.5

    if np.isfinite(mom30):
        sc += 0.5 if mom30 > 0 else 0.0

    if np.isfinite(atr14) and np.isfinite(atr_avg) and atr_avg > 0:
        sc += 0.5 if atr14 < atr_avg * 1.5 else 0.0

    return float(np.clip(sc, 0.0, CRYPTO_SCORE_MAX))


def stock_score_single(
    close: float,
    s200:  float,
    mom90: float,
    ema_f: float,
    ema_s: float,
) -> float:
    """
    Per-ticker stock score.  Range 0–3.
    Matches per-ticker logic inside stock_score() in test_v9_confidence.py.
    """
    sc = 0.0

    if np.isfinite(s200) and np.isfinite(close):
        sc += 1.0 if close > s200 else 0.0

    if np.isfinite(mom90):
        sc += 1.0 if mom90 > 0    else 0.0
        if mom90 > 0.20: sc += 0.5

    if np.isfinite(ema_f) and np.isfinite(ema_s):
        sc += 0.5 if ema_f > ema_s else 0.0

    return sc


def stock_score_portfolio(ticker_scores: list[float]) -> float:
    """
    Average per-ticker scores.  Returns 2.0 (neutral) if list is empty.
    Matches stock_score() in test_v9_confidence.py.
    """
    return float(np.mean(ticker_scores)) if ticker_scores else 2.0


# ── Allocation Engine ─────────────────────────────────────────────────────────

def allocate(c_sc: float, s_sc: float) -> tuple[float, float]:
    """
    Returns (crypto_weight, stock_weight).  sum ≤ 1.0; remainder is cash.

    Exactly matches allocate() in test_v9_confidence.py.
    Do not modify without version bump and full replay validation.
    """
    cn = c_sc / CRYPTO_SCORE_MAX   # normalise to 0–1
    sn = s_sc / STOCK_SCORE_MAX    # normalise to 0–1

    if   cn >= BOTH_HOT_THRESHOLD and sn >= BOTH_HOT_THRESHOLD:
        return (CRYPTO_LEAD_WC, CRYPTO_LEAD_WS)        # both hot  → 65/35

    elif cn >= BOTH_HOT_THRESHOLD and sn <  BOTH_HOT_THRESHOLD:
        return (CRYPTO_DOM_WC, CRYPTO_DOM_WS)           # crypto dominant → 80/20

    elif sn >= BOTH_HOT_THRESHOLD and cn <  BOTH_HOT_THRESHOLD:
        return (STOCK_DOM_WC, STOCK_DOM_WS)             # stock dominant → 20/80

    elif cn < BOTH_COLD_THRESHOLD and sn < BOTH_COLD_THRESHOLD:
        return (DEFENSIVE_WC, DEFENSIVE_WS)             # both cold → 40/40 + 20% cash

    else:                                               # neutral: proportional
        tot = cn + sn
        wc  = 0.40 + (cn / tot) * 0.20
        return (round(wc, 2), round(1.0 - wc, 2))


# ── Rebalance Gate ────────────────────────────────────────────────────────────

REBALANCE_DAYS = 21   # trading days between rebalances


def should_rebalance(trading_days_since_last: int) -> bool:
    return trading_days_since_last >= REBALANCE_DAYS
