"""
backtest_frf.py — Funding Rate Farmer (FRF) Backtest
Strategy: delta-neutral funding rate harvesting on top 3 highest-funding coins.
Capital slice: $1,500 (30% of $5,000 base).
Suspend when BTC is in BEAR regime (BTC < SMA200 AND 30d return < -20%).

Data source:
  - BTC and ETH: data/funding/{SYMBOL}_funding.parquet (real Binance data)
  - Other 10 coins: synthetic funding derived from BTC rates with per-coin
    scaling factors drawn from empirical crypto market behaviour
    (alt coins carry ~1.2–2.5× BTC funding rate with added idiosyncratic noise)
"""

import math
import os
import random
from datetime import datetime, timezone

import pyarrow.parquet as pq

# ─── Config ─────────────────────────────────────────────────────────────────────
_CAPITAL         = 1_500.0
_TAKER_FEE       = 0.0006     # 0.06% per side
_FUNDING_ENTRY   = 0.0001     # 0.01% per 8h — minimum to open
_FUNDING_EXIT    = 0.00005    # 0.005% per 8h — close below this
_REBAL_DAYS      = 7
_TOP_N           = 3
_SMA200_LEN      = 200
_PERIODS_PER_DAY = 3          # 3 × 8h periods per day
_DATA_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "candles")
_FUND_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "funding")

# Per-coin funding multipliers relative to BTC (empirical alts carry premium)
# BTC=1.0, ETH=real data; rest are synthetic
_COIN_MULT = {
    "BTCUSDT":  1.00,
    "ETHUSDT":  1.00,   # real data — overridden
    "BNBUSDT":  1.30,
    "SOLUSDT":  1.60,
    "ADAUSDT":  1.45,
    "DOTUSDT":  1.55,
    "LINKUSDT": 1.50,
    "LTCUSDT":  1.20,
    "AVAXUSDT": 1.70,
    "DOGEUSDT": 1.80,
    "TRXUSDT":  1.25,
    "XRPUSDT":  1.35,
}

_SYMBOLS = list(_COIN_MULT.keys())


# ─── Data helpers ────────────────────────────────────────────────────────────────

def _load_daily(symbol):
    path = os.path.join(_DATA_DIR, f"{symbol}_1D.parquet")
    if not os.path.exists(path):
        return []
    table = pq.read_table(path, columns=["open_time", "close"])
    rows = [{"ts": int(ts), "close": float(cl)}
            for ts, cl in zip(table["open_time"].to_pylist(), table["close"].to_pylist())]
    rows.sort(key=lambda r: r["ts"])
    return rows


def _load_funding_parquet(symbol):
    """Load real funding rates from parquet; returns {day_ms: avg_rate_per_8h}."""
    path = os.path.join(_FUND_DIR, f"{symbol}_funding.parquet")
    if not os.path.exists(path):
        return {}
    table = pq.read_table(path, columns=["timestamp_ms", "funding_rate"])
    day_map = {}
    for ts, rate in zip(table["timestamp_ms"].to_pylist(), table["funding_rate"].to_pylist()):
        day_ms = (int(ts) // 86_400_000) * 86_400_000
        day_map.setdefault(day_ms, []).append(float(rate))
    return {day: sum(rates) / len(rates) for day, rates in day_map.items()}


def _sma(closes, n):
    out = [None] * len(closes)
    for i in range(n - 1, len(closes)):
        out[i] = sum(closes[i - n + 1: i + 1]) / n
    return out


# ─── Regime ──────────────────────────────────────────────────────────────────────

def _build_btc_day_regime(btc_rows):
    closes = [r["close"] for r in btc_rows]
    sma200 = _sma(closes, _SMA200_LEN)
    regime = {}
    for i, row in enumerate(btc_rows):
        day_ms = (row["ts"] // 86_400_000) * 86_400_000
        if sma200[i] is None:
            regime[day_ms] = False
            continue
        below_sma = closes[i] < sma200[i]
        ret30 = (closes[i] - closes[i - 30]) / closes[i - 30] if i >= 30 else 0.0
        regime[day_ms] = bool(below_sma and ret30 < -0.20)
    return regime


# ─── Synthetic funding for non-parquet coins ─────────────────────────────────────

def _synthesize_funding(btc_funding_map, multiplier, seed):
    """
    Create synthetic daily funding for a coin from BTC funding rates.
    Adds scaling by multiplier + small idiosyncratic noise (seeded for reproducibility).
    """
    rng = random.Random(seed)
    result = {}
    for day_ms, btc_rate in btc_funding_map.items():
        # Scale + add bounded noise (±20% of the scaled rate)
        noise = rng.uniform(-0.20, 0.20) * abs(btc_rate * multiplier)
        result[day_ms] = btc_rate * multiplier + noise
    return result


# ─── Backtest ────────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 65)
    print("  NexFlow — Funding Rate Farmer (FRF) Backtest")
    print(f"  Capital slice: ${_CAPITAL:,.0f} | Taker fee: {_TAKER_FEE*100:.2f}%/side")
    print("=" * 65)

    # BTC price + regime
    btc_rows = _load_daily("BTCUSDT")
    if not btc_rows:
        print("ERROR: Cannot load BTCUSDT_1D.parquet")
        return None, None
    btc_day_regime = _build_btc_day_regime(btc_rows)

    # Load real BTC funding
    btc_funding = _load_funding_parquet("BTCUSDT")
    eth_funding = _load_funding_parquet("ETHUSDT")

    print(f"\n  Loaded real funding: BTC ({len(btc_funding)} days), ETH ({len(eth_funding)} days)")
    print("  Synthesising funding for remaining 10 coins from BTC base rates...")

    # Build funding map for all symbols
    funding_daily = {}
    for sym in _SYMBOLS:
        if sym == "BTCUSDT":
            funding_daily[sym] = btc_funding
        elif sym == "ETHUSDT":
            funding_daily[sym] = eth_funding
        else:
            mult = _COIN_MULT[sym]
            seed = sum(ord(c) for c in sym)
            funding_daily[sym] = _synthesize_funding(btc_funding, mult, seed)

    all_days = sorted({(r["ts"] // 86_400_000) * 86_400_000 for r in btc_rows})

    capital = _CAPITAL
    positions = {}                # sym -> entry_day_ms
    days_since_rebal = _REBAL_DAYS
    year_stats = {}

    def _yr(day_ms):
        return datetime.fromtimestamp(day_ms / 1000, tz=timezone.utc).year

    for day_ms in all_days:
        yr = _yr(day_ms)
        if yr not in year_stats:
            year_stats[yr] = {"funding": 0.0, "fees": 0.0, "start": capital}

        bear = btc_day_regime.get(day_ms, False)

        # Collect daily funding on open positions
        for sym in list(positions.keys()):
            rate = funding_daily[sym].get(day_ms, 0.0)
            notional = capital / _TOP_N
            daily_f = rate * notional * _PERIODS_PER_DAY
            capital += daily_f
            year_stats[yr]["funding"] += daily_f

        # Close all if bear regime
        if bear and positions:
            for sym in list(positions.keys()):
                notional = capital / max(len(positions), 1)
                fee = notional * 2 * _TAKER_FEE
                capital -= fee
                year_stats[yr]["fees"] += fee
            positions.clear()
            days_since_rebal = _REBAL_DAYS
            continue

        # Exit any position whose funding drops below exit threshold
        for sym in list(positions.keys()):
            rate = funding_daily[sym].get(day_ms, 0.0)
            if rate < _FUNDING_EXIT:
                notional = capital / max(len(positions), 1)
                fee = notional * 2 * _TAKER_FEE
                capital -= fee
                year_stats[yr]["fees"] += fee
                del positions[sym]

        # Weekly rebalance
        days_since_rebal += 1
        if days_since_rebal >= _REBAL_DAYS:
            days_since_rebal = 0
            day_rates = sorted(
                [(sym, funding_daily[sym].get(day_ms, 0.0)) for sym in _SYMBOLS],
                key=lambda x: x[1], reverse=True
            )
            desired = set()
            for sym, rate in day_rates:
                if rate >= _FUNDING_ENTRY and len(desired) < _TOP_N:
                    desired.add(sym)

            # Exit not-desired
            for sym in list(positions.keys()):
                if sym not in desired:
                    notional = capital / max(len(positions), 1)
                    fee = notional * 2 * _TAKER_FEE
                    capital -= fee
                    year_stats[yr]["fees"] += fee
                    del positions[sym]

            # Enter new
            for sym in desired:
                if sym not in positions:
                    notional = capital / _TOP_N
                    fee = notional * 2 * _TAKER_FEE
                    capital -= fee
                    year_stats[yr]["fees"] += fee
                    positions[sym] = day_ms

    # Close remaining
    for sym in list(positions.keys()):
        notional = capital / max(len(positions), 1)
        fee = notional * 2 * _TAKER_FEE
        capital -= fee
        if year_stats:
            last_yr = max(year_stats)
            year_stats[last_yr]["fees"] += fee

    # Fix year start capitals forward
    running = _CAPITAL
    for yr in sorted(year_stats):
        year_stats[yr]["start"] = running
        running += year_stats[yr]["funding"] - year_stats[yr]["fees"]

    # ─── Print table ──────────────────────────────────────────────────────────
    print()
    print(f"{'Year':<6} {'Funding $':>10} {'Fees $':>10} {'Net P&L $':>11} {'Capital $':>11} {'Return %':>9}")
    print("-" * 62)

    running = _CAPITAL
    total_f = total_fees = 0.0
    for yr in sorted(year_stats):
        st = year_stats[yr]
        net = st["funding"] - st["fees"]
        end_cap = st["start"] + net
        pct = net / st["start"] * 100 if st["start"] else 0.0
        total_f += st["funding"]
        total_fees += st["fees"]
        running = end_cap
        print(f"{yr:<6} {st['funding']:>10.2f} {st['fees']:>10.2f} {net:>11.2f} {end_cap:>11.2f} {pct:>8.1f}%")

    total_net = total_f - total_fees
    total_pct = total_net / _CAPITAL * 100
    print("-" * 62)
    print(f"{'TOTAL':<6} {total_f:>10.2f} {total_fees:>10.2f} {total_net:>11.2f} {running:>11.2f} {total_pct:>8.1f}%")
    print()
    print(f"  Final FRF capital: ${running:,.2f}  (started ${_CAPITAL:,.2f})")
    print()
    print("  NOTE: BTC + ETH use real Binance funding rate history.")
    print("        Other 10 coins use BTC-derived synthetic rates with per-coin")
    print("        multipliers (alts typically carry 1.2–2.5× BTC funding premium).")
    print()

    return year_stats, running


if __name__ == "__main__":
    run_backtest()
