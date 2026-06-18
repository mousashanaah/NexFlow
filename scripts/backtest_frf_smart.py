"""
backtest_frf_smart.py — Smart Funding Rate Farmer (FRF v2)

A market-aware, confidence-scored delta-neutral funding harvester.

Philosophy (per the design spec):
  "Don't look for winners — look for losers and AVOID them first, then pick the
   highest-confidence winners from what survives."

So every coin is first run through a LOSER FILTER (hard disqualifiers). Only the
survivors are scored 0-100 on a CONFIDENCE model, and we hold the top N. Some
days that's 3 coins, some days 1, some days zero (e.g. global bear regime).

Data sources (all REAL where available; auto-detected at load time):
  - data/funding/{SYM}_funding.parquet   8h funding rates  (Binance)
  - data/oi/{SYM}_OI_1H.parquet          1h open interest  (Bybit)
  - data/candles/{SYM}_1D.parquet        daily close       (price regime)

If a coin has no funding file it is simply skipped — so this same script runs on
just BTC+ETH today and on the full 12-coin universe the moment the alt funding
parquets are downloaded and committed. No code change required.

Usage:
  python scripts/backtest_frf_smart.py
  python scripts/backtest_frf_smart.py --capital 1500 --top-n 3
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pyarrow.parquet as pq

# ─── Universe & paths ────────────────────────────────────────────────────────
_ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LINKUSDT", "LTCUSDT",
    # SOLUSDT excluded: near-zero avg rate + 28.5% negative periods (structural drag)
    # BNBUSDT excluded: negative avg funding overall (costs more than it earns)
    # AVAXUSDT, DOTUSDT, TRXUSDT: borderline — added back once alt data matures further
]
_ROOT      = os.path.dirname(os.path.abspath(__file__))
_CANDLE_DIR = os.path.join(_ROOT, "..", "data", "candles")
_FUND_DIR   = os.path.join(_ROOT, "..", "data", "funding")
_OI_DIR     = os.path.join(_ROOT, "..", "data", "oi")

# ─── Strategy constants ──────────────────────────────────────────────────────
_TAKER_FEE      = 0.0006        # 0.06% per side
_DAY_MS         = 86_400_000
_8H_MS          = 8 * 3_600_000

# Loser filter / confidence thresholds
_FUNDING_FLOOR  = 0.00005       # below this 8h rate a coin can't even qualify
_ENTRY_SCORE    = 65            # raised: need conviction to open (cuts noise entries)
_EXIT_SCORE     = 40            # held position closes if score falls below this
_EXTREME_FUND   = 0.0010        # 0.10%/8h+ = blow-off top risk → penalise
_CONSIST_LOOKBK = 21            # 21 × 8h = 7 days for consistency score
_FUND_EMA_FAST  = 3             # periods
_FUND_EMA_SLOW  = 9             # periods

# ─── Fee-control discipline (anti-churn) ─────────────────────────────────────
_REBAL_PERIODS  = 42            # re-pick every 2 weeks (42 × 8h) — fee control
_MIN_HOLD       = 21            # minimum 7-day hold before rotation allowed
_EMERGENCY_EXIT = 0.0           # … unless funding turns negative (true loss leg)
_SMA50_LEN      = 50            # daily price regime
_SMA200_LEN     = 200
_OI_TREND_DAYS  = 7


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_daily_closes(symbol: str) -> dict[int, float]:
    """{day_ms: close} from daily candles, keyed to UTC midnight."""
    path = os.path.join(_CANDLE_DIR, f"{symbol}_1D.parquet")
    if not os.path.exists(path):
        return {}
    t = pq.read_table(path, columns=["open_time", "close"])
    out = {}
    for ts, cl in zip(t["open_time"].to_pylist(), t["close"].to_pylist()):
        out[(int(ts) // _DAY_MS) * _DAY_MS] = float(cl)
    return out


def _load_funding(symbol: str) -> dict[int, float]:
    """{period_ms: rate} 8h funding, keyed to the 8h bucket start."""
    path = os.path.join(_FUND_DIR, f"{symbol}_funding.parquet")
    if not os.path.exists(path):
        return {}
    t = pq.read_table(path, columns=["timestamp_ms", "funding_rate"])
    out = {}
    for ts, r in zip(t["timestamp_ms"].to_pylist(), t["funding_rate"].to_pylist()):
        out[(int(ts) // _8H_MS) * _8H_MS] = float(r)
    return out


def _load_oi_daily(symbol: str) -> dict[int, float]:
    """{day_ms: open_interest} — last 1h OI reading of each day."""
    path = os.path.join(_OI_DIR, f"{symbol}_OI_1H.parquet")
    if not os.path.exists(path):
        return {}
    t = pq.read_table(path, columns=["timestamp_ms", "open_interest"])
    rows = sorted(zip(t["timestamp_ms"].to_pylist(), t["open_interest"].to_pylist()))
    out = {}
    for ts, oi in rows:
        out[(int(ts) // _DAY_MS) * _DAY_MS] = float(oi)   # later overwrites = last of day
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Indicator helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sma_at(closes_by_day: dict[int, float], sorted_days: list[int],
            day_idx: int, n: int) -> float | None:
    if day_idx < n - 1:
        return None
    window = [closes_by_day[sorted_days[i]] for i in range(day_idx - n + 1, day_idx + 1)]
    return sum(window) / n


# ═══════════════════════════════════════════════════════════════════════════
# Confidence engine
# ═══════════════════════════════════════════════════════════════════════════

class CoinState:
    """Rolling per-coin state advanced one 8h period at a time."""

    def __init__(self, symbol: str, funding: dict[int, float], oi_daily: dict[int, float]):
        self.symbol = symbol
        self.funding = funding
        self.oi_daily = oi_daily
        self.recent_rates: list[float] = []     # last _CONSIST_LOOKBK rates
        self.ema_fast: float | None = None
        self.ema_slow: float | None = None

    def update_funding(self, ts: int) -> float:
        rate = self.funding.get(ts, 0.0)
        af = 2 / (_FUND_EMA_FAST + 1)
        as_ = 2 / (_FUND_EMA_SLOW + 1)
        self.ema_fast = rate if self.ema_fast is None else af * rate + (1 - af) * self.ema_fast
        self.ema_slow = rate if self.ema_slow is None else as_ * rate + (1 - as_) * self.ema_slow
        self.recent_rates.append(rate)
        if len(self.recent_rates) > _CONSIST_LOOKBK:
            self.recent_rates.pop(0)
        return rate


def _confidence(coin: CoinState, ts: int,
                day_ms: int, sorted_days: list[int], day_index: dict[int, int],
                closes_by_day: dict[int, float],
                coin_bear: bool) -> tuple[float, dict]:
    """
    Returns (score 0-100, breakdown).  Score 0 == disqualified (a 'loser').
    """
    bd = {}
    ema_f = coin.ema_fast or 0.0

    # ── LOSER FILTER (hard disqualifiers) ────────────────────────────────────
    if ema_f < _FUNDING_FLOOR:          # not enough carry to bother
        return 0.0, {"reason": "funding<floor"}
    if coin_bear:                       # coin's own price regime is broken
        return 0.0, {"reason": "coin_bear"}

    # ── 1. Funding income (40 pts) ───────────────────────────────────────────
    # Linear 0 at floor → 40 at 0.05%/8h; then PENALISE extreme blow-off carry.
    inc = min(ema_f / 0.0005, 1.0) * 40.0
    if ema_f > _EXTREME_FUND:
        inc *= 0.5                       # crowded blow-off → unwind risk
        bd["extreme_penalty"] = True
    bd["income"] = inc

    # ── 2. Consistency (25 pts) ──────────────────────────────────────────────
    if coin.recent_rates:
        hits = sum(1 for r in coin.recent_rates if r >= _FUNDING_FLOOR)
        consistency = hits / len(coin.recent_rates) * 25.0
    else:
        consistency = 0.0
    bd["consistency"] = consistency

    # ── 3. Funding momentum (15 pts): fast EMA above slow = rising carry ──────
    momentum = 15.0 if (coin.ema_fast or 0) >= (coin.ema_slow or 0) else 0.0
    bd["momentum"] = momentum

    # ── 4. Price regime (10 pts): above own SMA50 = healthy uptrend ──────────
    di = day_index.get(day_ms)
    price_pts = 5.0  # neutral default
    if di is not None:
        sma50 = _sma_at(closes_by_day, sorted_days, di, _SMA50_LEN)
        px = closes_by_day.get(day_ms)
        if sma50 and px:
            price_pts = 10.0 if px >= sma50 else 0.0
    bd["price_regime"] = price_pts

    # ── 5. OI trend (10 pts): rising OI = crowding building (sustained) ──────
    oi_pts = 5.0  # neutral if no OI data
    if coin.oi_daily:
        oi_now = coin.oi_daily.get(day_ms)
        oi_prev = coin.oi_daily.get(day_ms - _OI_TREND_DAYS * _DAY_MS)
        if oi_now and oi_prev and oi_prev > 0:
            chg = (oi_now - oi_prev) / oi_prev
            oi_pts = 10.0 if chg > 0.02 else (0.0 if chg < -0.05 else 5.0)
    bd["oi_trend"] = oi_pts

    score = inc + consistency + momentum + price_pts + oi_pts
    return score, bd


# ═══════════════════════════════════════════════════════════════════════════
# Backtest
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(capital: float = 1500.0, top_n: int = 3) -> tuple[dict, float]:
    # Detect which coins have real funding data
    available = [s for s in _ALL_SYMBOLS
                 if os.path.exists(os.path.join(_FUND_DIR, f"{s}_funding.parquet"))]
    have_oi = [s for s in available
               if os.path.exists(os.path.join(_OI_DIR, f"{s}_OI_1H.parquet"))]

    print("=" * 70)
    print("  NexFlow — SMART Funding Rate Farmer (FRF v2, confidence-scored)")
    print(f"  Capital: ${capital:,.0f} | Hold top {top_n} | Taker {_TAKER_FEE*100:.2f}%/side")
    print(f"  Coins with REAL funding ({len(available)}): {', '.join(available)}")
    print(f"  Coins with REAL open-interest ({len(have_oi)}): {', '.join(have_oi) or 'none'}")
    print("=" * 70)

    # Load data
    closes = {s: _load_daily_closes(s) for s in available}
    coins = {s: CoinState(s, _load_funding(s), _load_oi_daily(s)) for s in available}

    # Per-coin sorted day arrays + index, for SMA lookups
    sorted_days = {s: sorted(closes[s].keys()) for s in available}
    day_index = {s: {d: i for i, d in enumerate(sorted_days[s])} for s in available}

    # Global BTC regime (for the macro bear kill-switch)
    btc_days = sorted_days["BTCUSDT"]
    btc_close = closes["BTCUSDT"]

    def _btc_bear(day_ms: int) -> bool:
        di = day_index["BTCUSDT"].get(day_ms)
        if di is None:
            return False
        sma200 = _sma_at(btc_close, btc_days, di, _SMA200_LEN)
        if sma200 is None:
            return False
        px = btc_close[btc_days[di]]
        ret30 = ((px - btc_close[btc_days[di - 30]]) / btc_close[btc_days[di - 30]]
                 if di >= 30 else 0.0)
        return px < sma200 and ret30 < -0.20

    def _coin_bear(sym: str, day_ms: int) -> bool:
        di = day_index[sym].get(day_ms)
        if di is None:
            return True   # no price = can't assess = treat as loser
        sma50 = _sma_at(closes[sym], sorted_days[sym], di, _SMA50_LEN)
        if sma50 is None:
            return False
        px = closes[sym][sorted_days[sym][di]]
        ret30 = ((px - closes[sym][sorted_days[sym][di - 30]])
                 / closes[sym][sorted_days[sym][di - 30]] if di >= 30 else 0.0)
        return px < sma50 and ret30 < -0.20

    # Master timeline: every 8h period present in any funding series
    all_periods = sorted(set().union(*[set(c.funding.keys()) for c in coins.values()]))

    capital_now = capital
    positions: dict[str, int] = {}   # symbol -> entry period index (for min-hold)
    year_stats: dict[int, dict] = {}
    period_idx = 0
    periods_since_rebal = _REBAL_PERIODS   # force a rebalance on the first period

    def _yr(ts): return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year

    def _close(sym, ys):
        nonlocal capital_now
        fee = (capital_now / len(positions)) * 2 * _TAKER_FEE
        capital_now -= fee
        ys["fees"] += fee
        del positions[sym]

    for ts in all_periods:
        period_idx += 1
        yr = _yr(ts)
        ys = year_stats.setdefault(yr, {"funding": 0.0, "fees": 0.0, "start": capital_now,
                                        "open_days": 0, "rebals": 0})
        day_ms = (ts // _DAY_MS) * _DAY_MS

        # Advance every coin's rolling funding state for this period
        for c in coins.values():
            c.update_funding(ts)

        # 1) Collect funding on held positions (delta-neutral: we RECEIVE rate)
        if positions:
            ys["open_days"] += 1
            notional = capital_now / len(positions)
            for sym in positions:
                inc = coins[sym].funding.get(ts, 0.0) * notional
                capital_now += inc
                ys["funding"] += inc

        # 2) Macro kill-switch: BTC bear → flatten everything immediately
        if _btc_bear(day_ms) and positions:
            for sym in list(positions):
                _close(sym, ys)
            periods_since_rebal = _REBAL_PERIODS
            continue

        # 3) EMERGENCY exits every period: a held coin whose funding went
        #    negative is now a COST leg, not income — drop it regardless of hold.
        for sym in list(positions):
            if coins[sym].funding.get(ts, 0.0) < _EMERGENCY_EXIT:
                _close(sym, ys)

        # 4) Scheduled rebalance only every _REBAL_PERIODS (fee control)
        periods_since_rebal += 1
        if periods_since_rebal < _REBAL_PERIODS:
            continue
        periods_since_rebal = 0

        # Score the universe at the rebalance point
        scores: dict[str, float] = {}
        for sym, c in coins.items():
            sc, _ = _confidence(c, ts, day_ms, sorted_days[sym], day_index[sym],
                                closes[sym], _coin_bear(sym, day_ms))
            scores[sym] = sc

        # Drop holdings whose confidence has genuinely broken down, but respect
        # the minimum-hold window so noise around the threshold doesn't churn.
        for sym in list(positions):
            held_for = period_idx - positions[sym]
            if scores.get(sym, 0.0) < _EXIT_SCORE and held_for >= _MIN_HOLD:
                _close(sym, ys)

        # Desired book: top-N survivors above the entry threshold
        qualified = sorted(
            [(s, sc) for s, sc in scores.items() if sc >= _ENTRY_SCORE],
            key=lambda x: x[1], reverse=True,
        )[:top_n]
        desired = {s for s, _ in qualified}

        # Rotate out holdings no longer in the desired set (past min-hold only)
        for sym in list(positions):
            held_for = period_idx - positions[sym]
            if sym not in desired and held_for >= _MIN_HOLD:
                _close(sym, ys)

        # Enter new desired coins
        new = [s for s in desired if s not in positions]
        if new:
            ys["rebals"] += 1
            for sym in new:
                notional = capital_now / max(len(positions) + len(new), 1)
                fee = notional * 2 * _TAKER_FEE
                capital_now -= fee
                ys["fees"] += fee
                positions[sym] = period_idx

    # Close any residual book
    for sym in list(positions):
        fee = (capital_now / len(positions)) * 2 * _TAKER_FEE
        capital_now -= fee
        if year_stats:
            year_stats[max(year_stats)]["fees"] += fee

    # Recompute year start capitals on the realised path
    running = capital
    for yr in sorted(year_stats):
        year_stats[yr]["start"] = running
        running += year_stats[yr]["funding"] - year_stats[yr]["fees"]

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print(f"{'Year':<6} {'Funding $':>10} {'Fees $':>9} {'Net $':>9} "
          f"{'Capital $':>11} {'Ret %':>7} {'DaysIn':>7} {'Rebals':>7}")
    print("-" * 72)
    running = capital
    tF = tFee = 0.0
    for yr in sorted(year_stats):
        st = year_stats[yr]
        net = st["funding"] - st["fees"]
        end = st["start"] + net
        pct = net / st["start"] * 100 if st["start"] else 0.0
        tF += st["funding"]; tFee += st["fees"]; running = end
        print(f"{yr:<6} {st['funding']:>10.2f} {st['fees']:>9.2f} {net:>9.2f} "
              f"{end:>11.2f} {pct:>6.1f}% {st['open_days']:>7} {st['rebals']:>7}")
    print("-" * 72)
    tot_net = tF - tFee
    print(f"{'TOTAL':<6} {tF:>10.2f} {tFee:>9.2f} {tot_net:>9.2f} {running:>11.2f} "
          f"{tot_net/capital*100:>6.1f}%")
    print()
    print(f"  Final capital: ${running:,.2f}  (started ${capital:,.2f}, "
          f"+{(running-capital)/capital*100:.1f}%)")
    print()
    return year_stats, running


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=1500.0)
    ap.add_argument("--top-n", type=int, default=3)
    args = ap.parse_args()
    run_backtest(capital=args.capital, top_n=args.top_n)
