"""
backtest_stat_arb.py — BTC/ETH Statistical Arbitrage Backtest
Strategy: log-spread mean reversion with z-score signals.
Capital slice: $1,000 (20% of $5,000 base). Max 1 position at a time.
Regime filter: only enter in sideways BTC markets.
"""

import math
import os
from datetime import datetime, timezone

import pyarrow.parquet as pq

# ─── Config ─────────────────────────────────────────────────────────────────────
_CAPITAL         = 1_000.0
_TAKER_FEE       = 0.0006     # 0.06% per side
_HEDGE_WINDOW    = 60         # OLS regression window (days)
_HEDGE_REBAL_DAYS = 30        # recalculate hedge ratio monthly
_ZSCORE_WINDOW   = 20         # z-score lookback
_ENTRY_Z         = 2.0        # open position
_EXIT_Z          = 0.5        # close position (mean reversion)
_STOP_Z          = 3.5        # hard stop
_LEG_SIZE        = 500.0      # $500 per leg ($1,000 total round trip)
_DATA_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "candles")

# Sideways regime thresholds
_SIDEWAYS_RET20  = 0.15       # abs(20d return) < 15%
_SIDEWAYS_SMA50  = 0.10       # price within 10% of 50d SMA


# ─── Data helpers ────────────────────────────────────────────────────────────────

def _load_daily(symbol):
    path = os.path.join(_DATA_DIR, f"{symbol}_1D.parquet")
    table = pq.read_table(path, columns=["open_time", "close"])
    rows = [{"ts": int(ts), "close": float(cl)}
            for ts, cl in zip(table["open_time"].to_pylist(), table["close"].to_pylist())]
    rows.sort(key=lambda r: r["ts"])
    return rows


def _sma(values, n):
    out = [None] * len(values)
    for i in range(n - 1, len(values)):
        out[i] = sum(values[i - n + 1: i + 1]) / n
    return out


# ─── OLS (manual) ────────────────────────────────────────────────────────────────

def _ols_slope(x, y):
    """Simple OLS slope: beta = cov(x,y) / var(x)."""
    n = len(x)
    if n < 2:
        return 1.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    return num / den if den != 0 else 1.0


# ─── Regime filter ───────────────────────────────────────────────────────────────

def _build_sideways_mask(btc_closes):
    """Returns list of bool: True if BTC is in sideways regime."""
    sma50 = _sma(btc_closes, 50)
    mask = [False] * len(btc_closes)
    for i in range(len(btc_closes)):
        if i < 20 or sma50[i] is None:
            continue
        ret20 = (btc_closes[i] - btc_closes[i - 20]) / btc_closes[i - 20]
        sma50_pct = abs(btc_closes[i] - sma50[i]) / sma50[i]
        mask[i] = abs(ret20) < _SIDEWAYS_RET20 and sma50_pct < _SIDEWAYS_SMA50
    return mask


# ─── Backtest ────────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 65)
    print("  NexFlow — BTC/ETH Statistical Arbitrage Backtest")
    print(f"  Capital slice: ${_CAPITAL:,.0f} | Leg size: ${_LEG_SIZE:,.0f}/leg")
    print("=" * 65)

    btc_rows = _load_daily("BTCUSDT")
    eth_rows = _load_daily("ETHUSDT")

    # Align on common timestamps
    btc_map = {r["ts"]: r["close"] for r in btc_rows}
    eth_map = {r["ts"]: r["close"] for r in eth_rows}
    common_ts = sorted(set(btc_map) & set(eth_map))

    btc_cl = [btc_map[ts] for ts in common_ts]
    eth_cl = [eth_map[ts] for ts in common_ts]
    log_btc = [math.log(c) for c in btc_cl]
    log_eth = [math.log(c) for c in eth_cl]

    sideways = _build_sideways_mask(btc_cl)

    capital = _CAPITAL
    year_stats = {}

    position = None   # None or dict with direction/entry info
    hedge_ratio = 1.0
    last_hedge_rebal = 0

    total_trades = 0

    # Pre-compute spread and z-score incrementally
    for i, ts in enumerate(common_ts):
        yr = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year
        if yr not in year_stats:
            year_stats[yr] = {"trades": 0, "wins": 0, "pnl": 0.0, "start": capital}

        # Need at least hedge window + zscore window
        min_idx = _HEDGE_WINDOW + _ZSCORE_WINDOW
        if i < min_idx:
            continue

        # Recalculate hedge ratio monthly
        if i - last_hedge_rebal >= _HEDGE_REBAL_DAYS or last_hedge_rebal == 0:
            win_x = log_eth[i - _HEDGE_WINDOW: i]
            win_y = log_btc[i - _HEDGE_WINDOW: i]
            hedge_ratio = _ols_slope(win_x, win_y)
            last_hedge_rebal = i

        # Compute spread series over zscore window
        spread_window = []
        for j in range(i - _ZSCORE_WINDOW, i + 1):
            spread_window.append(log_btc[j] - hedge_ratio * log_eth[j])

        spread_now = spread_window[-1]
        mean_s = sum(spread_window) / len(spread_window)
        std_s = (sum((s - mean_s) ** 2 for s in spread_window) / len(spread_window)) ** 0.5
        z = (spread_now - mean_s) / std_s if std_s > 1e-12 else 0.0

        # ── Manage existing position ─────────────────────────────────────────
        if position is not None:
            direction = position["direction"]  # +1 = long BTC short ETH, -1 = short BTC long ETH
            entry_z   = position["entry_z"]
            entry_btc = position["entry_btc"]
            entry_eth = position["entry_eth"]

            # Current P&L on legs (mark-to-market)
            btc_ret = (btc_cl[i] - entry_btc) / entry_btc
            eth_ret = (eth_cl[i] - entry_eth) / entry_eth

            if direction == 1:
                # long BTC, short ETH
                pnl_raw = _LEG_SIZE * btc_ret - _LEG_SIZE * eth_ret
            else:
                # short BTC, long ETH
                pnl_raw = -_LEG_SIZE * btc_ret + _LEG_SIZE * eth_ret

            # Check exits
            close_trade = False
            win = False

            if direction == 1 and z <= _EXIT_Z:
                close_trade = True
                win = pnl_raw > 0
            elif direction == -1 and z >= -_EXIT_Z:
                close_trade = True
                win = pnl_raw > 0
            elif abs(z) >= _STOP_Z:
                close_trade = True
                win = False

            if close_trade:
                fee = 2 * _LEG_SIZE * 2 * _TAKER_FEE  # 2 legs × 2 sides (close)
                net = pnl_raw - fee
                capital += net
                year_stats[yr]["trades"] += 1
                year_stats[yr]["pnl"] += net
                if win:
                    year_stats[yr]["wins"] += 1
                total_trades += 1
                position = None

        # ── Check for new entry ──────────────────────────────────────────────
        if position is None and sideways[i]:
            new_pos = None
            if z > _ENTRY_Z:
                new_pos = {"direction": -1, "entry_z": z,
                           "entry_btc": btc_cl[i], "entry_eth": eth_cl[i]}
            elif z < -_ENTRY_Z:
                new_pos = {"direction": 1, "entry_z": z,
                           "entry_btc": btc_cl[i], "entry_eth": eth_cl[i]}

            if new_pos is not None:
                fee = 2 * _LEG_SIZE * 2 * _TAKER_FEE  # 2 legs × 2 sides (open)
                capital -= fee
                year_stats[yr]["pnl"] -= fee
                position = new_pos

    # Close any open position at end
    if position is not None:
        i = len(common_ts) - 1
        yr = datetime.fromtimestamp(common_ts[i] / 1000, tz=timezone.utc).year
        direction = position["direction"]
        btc_ret = (btc_cl[i] - position["entry_btc"]) / position["entry_btc"]
        eth_ret = (eth_cl[i] - position["entry_eth"]) / position["entry_eth"]
        if direction == 1:
            pnl_raw = _LEG_SIZE * btc_ret - _LEG_SIZE * eth_ret
        else:
            pnl_raw = -_LEG_SIZE * btc_ret + _LEG_SIZE * eth_ret
        fee = 2 * _LEG_SIZE * 2 * _TAKER_FEE
        net = pnl_raw - fee
        capital += net
        if yr in year_stats:
            year_stats[yr]["pnl"] += net
            year_stats[yr]["trades"] += 1
            if net > 0:
                year_stats[yr]["wins"] += 1

    # ─── Fix year start capitals ──────────────────────────────────────────────
    running = _CAPITAL
    for yr in sorted(year_stats):
        year_stats[yr]["start"] = running
        running += year_stats[yr]["pnl"]

    # ─── Print table ──────────────────────────────────────────────────────────
    print()
    print(f"{'Year':<6} {'Trades':>7} {'Win %':>7} {'Net P&L $':>11} {'Capital $':>11} {'Return %':>9}")
    print("-" * 56)

    running = _CAPITAL
    total_trades_all = total_wins = total_pnl = 0
    for yr in sorted(year_stats):
        st = year_stats[yr]
        net = st["pnl"]
        end_cap = st["start"] + net
        win_pct = st["wins"] / st["trades"] * 100 if st["trades"] else 0.0
        pct_ret = net / st["start"] * 100 if st["start"] else 0.0
        total_trades_all += st["trades"]
        total_wins += st["wins"]
        total_pnl += net
        running = end_cap
        print(f"{yr:<6} {st['trades']:>7} {win_pct:>6.1f}% {net:>11.2f} {end_cap:>11.2f} {pct_ret:>8.1f}%")

    overall_win = total_wins / total_trades_all * 100 if total_trades_all else 0.0
    total_pct = total_pnl / _CAPITAL * 100
    print("-" * 56)
    print(f"{'TOTAL':<6} {total_trades_all:>7} {overall_win:>6.1f}% {total_pnl:>11.2f} {running:>11.2f} {total_pct:>8.1f}%")
    print()
    print(f"  Final Stat Arb capital: ${running:,.2f}  (started ${_CAPITAL:,.2f})")
    print()

    return year_stats, running


if __name__ == "__main__":
    run_backtest()
