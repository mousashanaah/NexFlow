"""HTML report generator for paper trading analysis results.

Produces a single self-contained HTML file:
    - No external CDN dependencies
    - All charts rendered as inline SVG (computed from data in Python)
    - All CSS inline in <style> block
    - Works offline, over SSH, in CI

Chart inventory:
    equity_curve_svg    — area chart of portfolio equity over time
    drawdown_svg        — filled area chart of drawdown% over time
    pnl_histogram_svg   — binned bar chart of trade PnL distribution
    monthly_bar_svg     — signed bar chart of monthly PnL
    session_bar_svg     — horizontal bar chart of session win rates
"""

from __future__ import annotations

import math
import time
from typing import Sequence

from nexflow.analysis.analyze_paper_results import AnalysisResult, TradeRecord


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------

_W = 760       # chart width (inside SVG)
_PAD_L = 58    # left padding (y-axis labels)
_PAD_R = 20
_PAD_T = 18
_PAD_B = 44    # bottom padding (x-axis labels)
_PLOT_W = _W - _PAD_L - _PAD_R   # 682
_PLOT_H = 180  # plot area height

_TOTAL_H = _PLOT_H + _PAD_T + _PAD_B   # 242

_C_GREEN  = "#00d97e"
_C_RED    = "#ff4d4d"
_C_BLUE   = "#4d9fff"
_C_YELLOW = "#ffb547"
_C_PURPLE = "#9b59ff"
_C_MUTED  = "#3a3d4a"
_C_GRID   = "#23263a"
_C_TEXT   = "#8b8fa8"
_C_TEXT2  = "#e8eaf0"


def _fmt_num(v: float, decimals: int = 2) -> str:
    if math.isnan(v) or math.isinf(v):
        return "—"
    return f"{v:,.{decimals}f}"


def _scale_y(values: Sequence[float], out_top: float, out_bot: float) -> list[float]:
    """Scale values to SVG y-coordinates (top=small value visually)."""
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        mid = (out_top + out_bot) / 2
        return [mid] * len(values)
    return [out_top + (1.0 - (v - vmin) / (vmax - vmin)) * (out_bot - out_top) for v in values]


def _scale_x(n: int) -> list[float]:
    """Evenly distribute n points across the plot width."""
    if n <= 1:
        return [_PAD_L + _PLOT_W / 2]
    return [_PAD_L + i / (n - 1) * _PLOT_W for i in range(n)]


def _yticks(vmin: float, vmax: float, n: int = 5) -> list[tuple[float, str]]:
    """Return (value, label) tick pairs."""
    if vmax == vmin:
        return [(vmin, _fmt_num(vmin))]
    step = (vmax - vmin) / (n - 1)
    return [(vmin + i * step, _fmt_num(vmin + i * step)) for i in range(n)]


def _xtick_labels(timestamps: list[float], max_labels: int = 8) -> list[tuple[int, str]]:
    """Return (index, label) pairs for x-axis date ticks."""
    n = len(timestamps)
    if n == 0:
        return []
    step = max(1, n // max_labels)
    result = []
    for i in range(0, n, step):
        ts = timestamps[i]
        label = time.strftime("%m-%d", time.gmtime(ts)) if ts > 0 else str(i)
        result.append((i, label))
    return result


def _svg_wrap(body: str, height: int = _TOTAL_H, extra_height: int = 0) -> str:
    h = height + extra_height
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {h}" '
        f'width="100%" style="max-width:{_W}px;display:block;margin:0 auto;">'
        f'{body}</svg>'
    )


def _grid_lines(vmin: float, vmax: float) -> str:
    """Horizontal grid lines."""
    ticks = _yticks(vmin, vmax)
    ys = _scale_y([t[0] for t in ticks], _PAD_T, _PAD_T + _PLOT_H)
    lines = []
    for y, (_, label) in zip(ys, ticks):
        lines.append(
            f'<line x1="{_PAD_L}" y1="{y:.1f}" x2="{_PAD_L + _PLOT_W}" y2="{y:.1f}" '
            f'stroke="{_C_GRID}" stroke-width="1"/>'
            f'<text x="{_PAD_L - 6}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="{_C_TEXT}">{label}</text>'
        )
    return "".join(lines)


# ---------------------------------------------------------------------------
# Chart: Equity curve
# ---------------------------------------------------------------------------

def equity_curve_svg(equities: list[float], timestamps: list[float]) -> str:
    if len(equities) < 2:
        return _svg_wrap(_no_data_msg())

    xs = _scale_x(len(equities))
    top, bot = _PAD_T, _PAD_T + _PLOT_H
    ys = _scale_y(equities, top, bot)

    vmin, vmax = min(equities), max(equities)
    grid = _grid_lines(vmin, vmax)

    # Area fill
    pts_area = (
        f"{xs[0]:.1f},{bot} "
        + " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        + f" {xs[-1]:.1f},{bot}"
    )
    # Line
    pts_line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    # X-axis ticks
    tick_marks = ""
    for idx, label in _xtick_labels(timestamps):
        x = xs[idx]
        tick_marks += (
            f'<line x1="{x:.1f}" y1="{bot}" x2="{x:.1f}" y2="{bot + 4}" '
            f'stroke="{_C_TEXT}" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{bot + 16}" text-anchor="middle" '
            f'font-size="10" fill="{_C_TEXT}">{label}</text>'
        )

    # Zero line if range crosses zero
    zero_line = ""
    if vmin < 0 < vmax:
        zy = _scale_y([0.0], top, bot)[0]
        zero_line = (
            f'<line x1="{_PAD_L}" y1="{zy:.1f}" x2="{_PAD_L + _PLOT_W}" y2="{zy:.1f}" '
            f'stroke="{_C_MUTED}" stroke-width="1" stroke-dasharray="4,4"/>'
        )

    body = (
        f'{grid}{zero_line}'
        f'<polygon points="{pts_area}" fill="{_C_BLUE}" fill-opacity="0.15"/>'
        f'<polyline points="{pts_line}" fill="none" stroke="{_C_BLUE}" stroke-width="2"/>'
        f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3" fill="{_C_BLUE}"/>'
        f'{tick_marks}'
        f'<line x1="{_PAD_L}" y1="{top}" x2="{_PAD_L}" y2="{bot}" '
        f'stroke="{_C_MUTED}" stroke-width="1"/>'
        f'<line x1="{_PAD_L}" y1="{bot}" x2="{_PAD_L + _PLOT_W}" y2="{bot}" '
        f'stroke="{_C_MUTED}" stroke-width="1"/>'
    )
    return _svg_wrap(body)


# ---------------------------------------------------------------------------
# Chart: Drawdown
# ---------------------------------------------------------------------------

def drawdown_svg(drawdowns_pct: list[float], timestamps: list[float]) -> str:
    if len(drawdowns_pct) < 2:
        return _svg_wrap(_no_data_msg())

    # Drawdown: 0 at top, larger = worse (lower on chart)
    # Flip: chart y=top → dd=0, chart y=bot → dd=max
    dd_neg = [-d for d in drawdowns_pct]   # make negative for standard scaling
    xs = _scale_x(len(dd_neg))
    top, bot = _PAD_T, _PAD_T + _PLOT_H
    ys = _scale_y(dd_neg, top, bot)

    vmin = min(dd_neg)  # most negative = worst drawdown
    vmax = 0.0

    # Grid (show positive DD values on labels)
    ticks_raw = _yticks(vmin, vmax)
    ys_grid = _scale_y([t[0] for t in ticks_raw], top, bot)
    grid = ""
    for y, (v, _) in zip(ys_grid, ticks_raw):
        label = f"{abs(v)*100:.1f}%"
        grid += (
            f'<line x1="{_PAD_L}" y1="{y:.1f}" x2="{_PAD_L + _PLOT_W}" y2="{y:.1f}" '
            f'stroke="{_C_GRID}" stroke-width="1"/>'
            f'<text x="{_PAD_L - 6}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="{_C_TEXT}">{label}</text>'
        )

    zero_y = _scale_y([0.0], top, bot)[0]
    pts_area = (
        f"{xs[0]:.1f},{zero_y:.1f} "
        + " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        + f" {xs[-1]:.1f},{zero_y:.1f}"
    )
    pts_line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    tick_marks = ""
    for idx, label in _xtick_labels(timestamps):
        x = xs[idx]
        tick_marks += (
            f'<text x="{x:.1f}" y="{bot + 16}" text-anchor="middle" '
            f'font-size="10" fill="{_C_TEXT}">{label}</text>'
        )

    body = (
        f'{grid}'
        f'<polygon points="{pts_area}" fill="{_C_RED}" fill-opacity="0.20"/>'
        f'<polyline points="{pts_line}" fill="none" stroke="{_C_RED}" stroke-width="1.5"/>'
        f'{tick_marks}'
        f'<line x1="{_PAD_L}" y1="{top}" x2="{_PAD_L}" y2="{bot}" '
        f'stroke="{_C_MUTED}" stroke-width="1"/>'
        f'<line x1="{_PAD_L}" y1="{bot}" x2="{_PAD_L + _PLOT_W}" y2="{bot}" '
        f'stroke="{_C_MUTED}" stroke-width="1"/>'
    )
    return _svg_wrap(body)


# ---------------------------------------------------------------------------
# Chart: PnL histogram
# ---------------------------------------------------------------------------

def pnl_histogram_svg(pnls: list[float], bins: int = 20) -> str:
    if not pnls:
        return _svg_wrap(_no_data_msg())

    vmin, vmax = min(pnls), max(pnls)
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    bin_width = (vmax - vmin) / bins
    counts = [0] * bins
    for p in pnls:
        idx = min(int((p - vmin) / bin_width), bins - 1)
        counts[idx] += 1

    max_count = max(counts) or 1
    top, bot = _PAD_T, _PAD_T + _PLOT_H
    bar_total_w = _PLOT_W / bins
    bar_w = max(1.0, bar_total_w - 1.5)

    zero_bin = int((0.0 - vmin) / bin_width) if vmin < 0 < vmax else -1

    bars = ""
    for i, cnt in enumerate(counts):
        x = _PAD_L + i * bar_total_w
        bar_h = (cnt / max_count) * _PLOT_H
        y = bot - bar_h
        color = _C_RED if i < zero_bin else _C_GREEN
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" fill-opacity="0.85"/>'

    # Zero line
    zero_x = _PAD_L + (0.0 - vmin) / (vmax - vmin) * _PLOT_W if vmin < 0 < vmax else -1
    zero_line = ""
    if zero_x > 0:
        zero_line = (
            f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{bot}" '
            f'stroke="{_C_TEXT}" stroke-width="1" stroke-dasharray="4,3"/>'
        )

    # X-axis ticks (min, 0, max)
    tick_marks = ""
    for val, label in [(vmin, _fmt_num(vmin, 0)), (vmax, _fmt_num(vmax, 0))]:
        x = _PAD_L + (val - vmin) / (vmax - vmin) * _PLOT_W
        tick_marks += (
            f'<text x="{x:.1f}" y="{bot + 16}" text-anchor="middle" '
            f'font-size="10" fill="{_C_TEXT}">{label}</text>'
        )
    if zero_x > 0:
        tick_marks += (
            f'<text x="{zero_x:.1f}" y="{bot + 16}" text-anchor="middle" '
            f'font-size="10" fill="{_C_TEXT}">0</text>'
        )

    # Y-axis label (count)
    y_label = (
        f'<text x="{_PAD_L - 6}" y="{top + 4}" text-anchor="end" '
        f'font-size="10" fill="{_C_TEXT}">{max_count}</text>'
        f'<text x="{_PAD_L - 6}" y="{bot + 4}" text-anchor="end" '
        f'font-size="10" fill="{_C_TEXT}">0</text>'
    )

    body = (
        f'{bars}{zero_line}{tick_marks}{y_label}'
        f'<line x1="{_PAD_L}" y1="{top}" x2="{_PAD_L}" y2="{bot}" '
        f'stroke="{_C_MUTED}" stroke-width="1"/>'
        f'<line x1="{_PAD_L}" y1="{bot}" x2="{_PAD_L + _PLOT_W}" y2="{bot}" '
        f'stroke="{_C_MUTED}" stroke-width="1"/>'
    )
    return _svg_wrap(body)


# ---------------------------------------------------------------------------
# Chart: Monthly PnL bars
# ---------------------------------------------------------------------------

def monthly_bar_svg(monthly_pnl: dict[str, float]) -> str:
    if not monthly_pnl:
        return _svg_wrap(_no_data_msg())

    labels = list(monthly_pnl.keys())
    values = list(monthly_pnl.values())
    n = len(values)
    vmax = max(abs(v) for v in values) or 1.0
    top, bot = _PAD_T, _PAD_T + _PLOT_H
    zero_y = (top + bot) / 2   # 0 line in the middle
    bar_total_w = _PLOT_W / n
    bar_w = max(4.0, bar_total_w - 4.0)
    half_h = _PLOT_H / 2

    bars = ""
    for i, (val, label) in enumerate(zip(values, labels)):
        x = _PAD_L + i * bar_total_w + (bar_total_w - bar_w) / 2
        bar_h = abs(val) / vmax * half_h
        if val >= 0:
            y = zero_y - bar_h
            color = _C_GREEN
        else:
            y = zero_y
            color = _C_RED
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" fill-opacity="0.85"/>'
        # X label
        bars += (
            f'<text x="{x + bar_w/2:.1f}" y="{bot + 16}" text-anchor="middle" '
            f'font-size="9" fill="{_C_TEXT}" transform="rotate(-30,{x + bar_w/2:.1f},{bot + 16})">{label}</text>'
        )

    # Zero line
    zero_line = (
        f'<line x1="{_PAD_L}" y1="{zero_y:.1f}" x2="{_PAD_L + _PLOT_W}" y2="{zero_y:.1f}" '
        f'stroke="{_C_MUTED}" stroke-width="1"/>'
    )
    # Y-axis labels
    y_labels = (
        f'<text x="{_PAD_L - 6}" y="{top + 4}" text-anchor="end" font-size="10" fill="{_C_TEXT}">+{_fmt_num(vmax, 0)}</text>'
        f'<text x="{_PAD_L - 6}" y="{zero_y + 4:.1f}" text-anchor="end" font-size="10" fill="{_C_TEXT}">0</text>'
        f'<text x="{_PAD_L - 6}" y="{bot + 4}" text-anchor="end" font-size="10" fill="{_C_TEXT}">-{_fmt_num(vmax, 0)}</text>'
    )

    body = (
        f'{zero_line}{bars}{y_labels}'
        f'<line x1="{_PAD_L}" y1="{top}" x2="{_PAD_L}" y2="{bot}" '
        f'stroke="{_C_MUTED}" stroke-width="1"/>'
    )
    return _svg_wrap(body, extra_height=20)


# ---------------------------------------------------------------------------
# Chart: Session horizontal bars
# ---------------------------------------------------------------------------

def session_bar_svg(session_stats: dict) -> str:
    sessions_order = ["new_york", "london", "asia", "off_hours"]
    labels = {"new_york": "New York", "london": "London", "asia": "Asia", "off_hours": "Off-hours"}

    rows = [(labels.get(k, k), session_stats[k]) for k in sessions_order if k in session_stats]
    if not rows:
        return _svg_wrap(_no_data_msg(), height=80)

    n = len(rows)
    row_h = 36
    total_h = n * row_h + 20
    bar_max_w = _PLOT_W - 60
    max_trades = max(r[1].trades for r in rows) or 1

    body_parts = []
    for i, (label, ss) in enumerate(rows):
        y = 10 + i * row_h
        bar_w = (ss.trades / max_trades) * bar_max_w
        wr_pct = ss.win_rate * 100
        wr_fill = _C_GREEN if wr_pct >= 50 else _C_YELLOW if wr_pct >= 40 else _C_RED
        body_parts.append(
            f'<text x="{_PAD_L - 6}" y="{y + 14}" text-anchor="end" font-size="11" fill="{_C_TEXT2}">{label}</text>'
            f'<rect x="{_PAD_L}" y="{y + 2}" width="{bar_w:.1f}" height="22" rx="3" fill="{_C_BLUE}" fill-opacity="0.5"/>'
            f'<rect x="{_PAD_L}" y="{y + 2}" width="{(ss.wins / max_trades) * bar_max_w:.1f}" height="22" rx="3" fill="{_C_BLUE}" fill-opacity="0.9"/>'
            f'<text x="{_PAD_L + bar_w + 8:.1f}" y="{y + 14}" font-size="11" fill="{_C_TEXT}">'
            f'{ss.trades} trades  {wr_pct:.0f}% WR  {_fmt_num(ss.net_pnl, 0)} USD</text>'
        )

    body = "".join(body_parts)
    return _svg_wrap(body, height=total_h)


def _no_data_msg() -> str:
    cx = _W / 2
    return (
        f'<text x="{cx}" y="110" text-anchor="middle" font-size="14" '
        f'fill="{_C_TEXT}" font-style="italic">No data available</text>'
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f1117; color: #e8eaf0; font-family: 'Segoe UI', system-ui, sans-serif;
       font-size: 14px; line-height: 1.6; padding: 24px; }
h1 { font-size: 22px; font-weight: 600; color: #fff; margin-bottom: 4px; }
h2 { font-size: 15px; font-weight: 600; color: #a8abbe; text-transform: uppercase;
     letter-spacing: 0.08em; margin: 28px 0 12px; }
.subtitle { color: #5a5e78; font-size: 13px; margin-bottom: 24px; }
.card { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 8px; padding: 16px 20px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; margin-bottom: 28px; }
.card-label { font-size: 11px; color: #5a5e78; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
.card-value { font-size: 22px; font-weight: 700; color: #fff; }
.card-value.green { color: #00d97e; }
.card-value.red { color: #ff4d4d; }
.card-value.yellow { color: #ffb547; }
.chart-wrap { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 8px;
              padding: 16px; margin-bottom: 16px; }
.chart-title { font-size: 12px; color: #5a5e78; text-transform: uppercase;
               letter-spacing: 0.08em; margin-bottom: 10px; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; font-size: 11px; color: #5a5e78; text-transform: uppercase;
     letter-spacing: 0.06em; padding: 8px 12px; border-bottom: 1px solid #2a2d3a; }
td { padding: 8px 12px; border-bottom: 1px solid #1e2130; font-size: 13px; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #1e2234; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.pos { color: #00d97e; }
.neg { color: #ff4d4d; }
.neu { color: #8b8fa8; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
       font-size: 11px; font-weight: 600; }
.tag-green { background: rgba(0,217,126,0.15); color: #00d97e; }
.tag-red   { background: rgba(255,77,77,0.15);  color: #ff4d4d; }
.tag-blue  { background: rgba(77,159,255,0.15); color: #4d9fff; }
.tag-yellow{ background: rgba(255,181,71,0.15); color: #ffb547; }
.no-data { color: #5a5e78; font-style: italic; text-align: center; padding: 40px; }
.kill-row td { color: #ff4d4d; }
hr { border: none; border-top: 1px solid #2a2d3a; margin: 24px 0; }
footer { margin-top: 40px; color: #3a3d4a; font-size: 11px; text-align: center; }
"""


# ---------------------------------------------------------------------------
# HTML template helpers
# ---------------------------------------------------------------------------

def _pnl_class(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "neu")


def _pnl_fmt(v: float, decimals: int = 2) -> str:
    if math.isnan(v) or math.isinf(v):
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.{decimals}f}"


def _pct_fmt(v: float, decimals: int = 1) -> str:
    return f"{v * 100:.{decimals}f}%"


def _card(label: str, value: str, css_class: str = "") -> str:
    val_html = f'<div class="card-value {css_class}">{value}</div>'
    return f'<div class="card"><div class="card-label">{label}</div>{val_html}</div>'


def _stat_row(label: str, value: str, css_class: str = "") -> str:
    val_html = f'<td class="num {css_class}">{value}</td>'
    return f"<tr><td>{label}</td>{val_html}</tr>"


# ---------------------------------------------------------------------------
# Main report function
# ---------------------------------------------------------------------------

def generate_html_report(result: AnalysisResult) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    ts_str = f"Generated {ts}"

    if not result.has_data:
        return _no_data_report(ts_str)

    ts_ = result.trade_stats
    rs_ = result.risk_stats
    ex_ = result.execution_stats
    mk_ = result.market_stats

    # ---- Summary cards ----
    net_pnl = result.final_equity - result.initial_equity
    net_return_pct = net_pnl / result.initial_equity * 100 if result.initial_equity else 0
    pf_str = f"{ts_.profit_factor:.2f}" if not math.isinf(ts_.profit_factor) else "∞"

    cards_html = "".join([
        _card("Total Trades",     str(ts_.total)),
        _card("Win Rate",         _pct_fmt(ts_.win_rate),
              "green" if ts_.win_rate >= 0.5 else "yellow"),
        _card("Profit Factor",    pf_str,
              "green" if ts_.profit_factor > 1.5 else ("yellow" if ts_.profit_factor > 1.0 else "red")),
        _card("Net PnL",          _pnl_fmt(net_pnl, 0) + " USD",
              "green" if net_pnl > 0 else "red"),
        _card("Max Drawdown",     _pct_fmt(rs_.max_drawdown),
              "green" if rs_.max_drawdown < 0.05 else ("yellow" if rs_.max_drawdown < 0.10 else "red")),
        _card("Avg R Multiple",   f"{ts_.avg_r:+.3f}",
              "green" if ts_.avg_r > 0 else "red"),
        _card("Expectancy (R)",   f"{ts_.expectancy_r:+.3f}",
              "green" if ts_.expectancy_r > 0 else "red"),
        _card("Kill Switches",    str(rs_.kill_count),
              "red" if rs_.kill_count > 0 else "green"),
    ])

    # ---- Equity & Drawdown charts ----
    eq_timestamps = [p.ts for p in result.equity_curve]
    eq_values = [p.equity for p in result.equity_curve]
    dd_values = [p.drawdown for p in result.equity_curve]

    eq_chart = equity_curve_svg(eq_values, eq_timestamps)
    dd_chart = drawdown_svg(dd_values, eq_timestamps)

    # ---- PnL histogram ----
    pnls = [t.pnl for t in result.trades]
    hist_chart = pnl_histogram_svg(pnls)

    # ---- Monthly bar ----
    monthly_chart = monthly_bar_svg(result.monthly_pnl)

    # ---- Session chart ----
    sess_chart = session_bar_svg(mk_.by_session)

    # ---- Trade statistics table ----
    trade_table = f"""
    <table>
      {_stat_row("Total trades",       str(ts_.total))}
      {_stat_row("Wins / Losses",       f"{ts_.wins} / {ts_.losses}")}
      {_stat_row("Win rate",            _pct_fmt(ts_.win_rate), _pnl_class(ts_.win_rate - 0.5))}
      {_stat_row("Profit factor",       pf_str, "pos" if ts_.profit_factor > 1 else "neg")}
      {_stat_row("Expectancy (R)",      f"{ts_.expectancy_r:+.4f}", _pnl_class(ts_.expectancy_r))}
      {_stat_row("Avg R multiple",      f"{ts_.avg_r:+.4f}", _pnl_class(ts_.avg_r))}
      {_stat_row("Avg hold time",       f"{ts_.avg_hold_minutes:.1f} min")}
      {_stat_row("Avg win",             _pnl_fmt(ts_.avg_win) + " USD", "pos")}
      {_stat_row("Avg loss",            _pnl_fmt(-ts_.avg_loss) + " USD", "neg")}
      {_stat_row("Best trade",          _pnl_fmt(ts_.best_trade) + " USD", "pos")}
      {_stat_row("Worst trade",         _pnl_fmt(ts_.worst_trade) + " USD", "neg")}
      {_stat_row("TP exits",            str(ts_.tp_exits))}
      {_stat_row("Stop exits",          str(ts_.stop_exits))}
      {_stat_row("Forced closes",       str(ts_.forced_exits))}
      {_stat_row("Net PnL",             _pnl_fmt(ts_.net_pnl) + " USD", _pnl_class(ts_.net_pnl))}
      {_stat_row("Gross profit",        _pnl_fmt(ts_.gross_profit) + " USD", "pos")}
      {_stat_row("Gross loss",          _pnl_fmt(-ts_.gross_loss) + " USD", "neg")}
    </table>"""

    # ---- Risk statistics table ----
    risk_table = f"""
    <table>
      {_stat_row("Max drawdown",        _pct_fmt(rs_.max_drawdown), "neg" if rs_.max_drawdown > 0.05 else "pos")}
      {_stat_row("Avg drawdown",        _pct_fmt(rs_.avg_drawdown))}
      {_stat_row("Drawdown p50",        _pct_fmt(rs_.drawdown_p50))}
      {_stat_row("Drawdown p95",        _pct_fmt(rs_.drawdown_p95), "neg" if rs_.drawdown_p95 > 0.05 else "neu")}
      {_stat_row("Kill switch events",  str(rs_.kill_count), "neg" if rs_.kill_count > 0 else "pos")}
      {_stat_row("Rejected signals",    str(rs_.rejected_count))}
      {_stat_row("Max consec. losses",  str(rs_.max_consec_losses), "neg" if rs_.max_consec_losses >= 4 else "neu")}
      {_stat_row("Stale feed events",   str(rs_.stale_feed_events))}
      {_stat_row("Latency spikes",      str(rs_.latency_spike_events))}
    </table>"""

    # ---- Execution statistics cards ----
    exec_cards = "".join([
        _card("Avg Slippage",     f"{ex_.avg_slippage_pct * 100:.4f}%"),
        _card("Total Fees",       f"{ex_.total_fees:,.2f} USD"),
        _card("Fee Drag",         f"{ex_.fee_drag_gross_pct * 100:.2f}%"),
        _card("Avg Fee / Trade",  f"{ex_.avg_fee_per_trade:.2f} USD"),
        _card("Total Slippage",   f"{ex_.total_slippage_cost:.2f} USD"),
        _card("Spread Anomalies", str(ex_.spread_anomaly_events)),
    ])

    # ---- Symbol breakdown table ----
    sym_rows = ""
    for sym, ss in sorted(mk_.by_symbol.items(), key=lambda x: x[1].trades, reverse=True):
        pnl_cls = _pnl_class(ss.net_pnl)
        sym_rows += (
            f"<tr><td>{sym}</td>"
            f"<td class='num'>{ss.trades}</td>"
            f"<td class='num'>{_pct_fmt(ss.win_rate)}</td>"
            f"<td class='num {pnl_cls}'>{_pnl_fmt(ss.net_pnl)} USD</td>"
            f"<td class='num'>{ss.avg_r:+.3f}</td></tr>"
        )
    sym_table = f"""
    <table>
      <thead><tr><th>Symbol</th><th class='num'>Trades</th><th class='num'>Win Rate</th>
      <th class='num'>Net PnL</th><th class='num'>Avg R</th></tr></thead>
      <tbody>{sym_rows or "<tr><td colspan='5' class='no-data'>—</td></tr>"}</tbody>
    </table>"""

    # ---- Direction breakdown ----
    direction_table = f"""
    <table>
      <thead><tr><th>Direction</th><th class='num'>Trades</th><th class='num'>Win Rate</th><th class='num'>Net PnL</th></tr></thead>
      <tbody>
        <tr><td>Long</td><td class='num'>{mk_.long_trades}</td>
            <td class='num'>{_pct_fmt(mk_.long_win_rate)}</td>
            <td class='num {_pnl_class(mk_.long_pnl)}'>{_pnl_fmt(mk_.long_pnl)} USD</td></tr>
        <tr><td>Short</td><td class='num'>{mk_.short_trades}</td>
            <td class='num'>{_pct_fmt(mk_.short_win_rate)}</td>
            <td class='num {_pnl_class(mk_.short_pnl)}'>{_pnl_fmt(mk_.short_pnl)} USD</td></tr>
      </tbody>
    </table>"""

    # ---- Session breakdown table ----
    sess_order = ["new_york", "london", "asia", "off_hours"]
    sess_labels = {"new_york": "New York", "london": "London", "asia": "Asia", "off_hours": "Off-hours"}
    sess_rows = ""
    for sk in sess_order:
        if sk not in mk_.by_session:
            continue
        se = mk_.by_session[sk]
        pnl_cls = _pnl_class(se.net_pnl)
        sess_rows += (
            f"<tr><td>{sess_labels.get(sk, sk)}</td>"
            f"<td class='num'>{se.trades}</td>"
            f"<td class='num'>{_pct_fmt(se.win_rate)}</td>"
            f"<td class='num {pnl_cls}'>{_pnl_fmt(se.net_pnl)} USD</td></tr>"
        )
    sess_table = f"""
    <table>
      <thead><tr><th>Session</th><th class='num'>Trades</th><th class='num'>Win Rate</th><th class='num'>Net PnL</th></tr></thead>
      <tbody>{sess_rows or "<tr><td colspan='4' class='no-data'>—</td></tr>"}</tbody>
    </table>"""

    # ---- Regime breakdown table ----
    regime_order = ["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    regime_colors = {"LOW": "tag-blue", "MEDIUM": "tag-yellow", "HIGH": "tag-red", "UNKNOWN": ""}
    reg_rows = ""
    for rk in regime_order:
        if rk not in mk_.by_regime:
            continue
        re = mk_.by_regime[rk]
        tag_cls = regime_colors.get(rk, "")
        pnl_cls = _pnl_class(re.net_pnl)
        reg_rows += (
            f"<tr><td><span class='tag {tag_cls}'>{rk}</span></td>"
            f"<td class='num'>{re.trades}</td>"
            f"<td class='num'>{_pct_fmt(re.win_rate)}</td>"
            f"<td class='num {pnl_cls}'>{_pnl_fmt(re.net_pnl)} USD</td></tr>"
        )
    reg_table = f"""
    <table>
      <thead><tr><th>Vol. Regime</th><th class='num'>Trades</th><th class='num'>Win Rate</th><th class='num'>Net PnL</th></tr></thead>
      <tbody>{reg_rows or "<tr><td colspan='4' class='no-data'>—</td></tr>"}</tbody>
    </table>"""

    # ---- Kill switch log ----
    kill_rows = ""
    for ke in (result.risk_stats.kill_reasons or {}):
        pass  # already summarised above
    # Inline kill reason summary
    if rs_.kill_reasons:
        for reason, count in sorted(rs_.kill_reasons.items(), key=lambda x: x[1], reverse=True):
            kill_rows += f"<tr class='kill-row'><td>{reason}</td><td class='num'>{count}×</td></tr>"
    kill_section = f"""
    <table>
      <thead><tr><th>Kill Reason</th><th class='num'>Count</th></tr></thead>
      <tbody>{kill_rows or "<tr><td colspan='2' style='color:#5a5e78;text-align:center;padding:20px'>No kill switch events</td></tr>"}</tbody>
    </table>"""

    # ---- Rejection reason log ----
    reject_rows = ""
    for reason, count in sorted(rs_.rejected_reasons.items(), key=lambda x: x[1], reverse=True):
        reject_rows += f"<tr><td>{reason}</td><td class='num'>{count}</td></tr>"
    reject_section = f"""
    <table>
      <thead><tr><th>Rejection Reason</th><th class='num'>Count</th></tr></thead>
      <tbody>{reject_rows or "<tr><td colspan='2' style='color:#5a5e78;text-align:center;padding:20px'>No rejections</td></tr>"}</tbody>
    </table>"""

    # ---- Monthly PnL table ----
    monthly_rows = ""
    cum_pnl = 0.0
    for month, pnl in result.monthly_pnl.items():
        cum_pnl += pnl
        pnl_cls = _pnl_class(pnl)
        monthly_rows += (
            f"<tr><td>{month}</td>"
            f"<td class='num {pnl_cls}'>{_pnl_fmt(pnl)} USD</td>"
            f"<td class='num {_pnl_class(cum_pnl)}'>{_pnl_fmt(cum_pnl)} USD</td></tr>"
        )
    monthly_table = f"""
    <table>
      <thead><tr><th>Month</th><th class='num'>PnL</th><th class='num'>Cumulative</th></tr></thead>
      <tbody>{monthly_rows or "<tr><td colspan='3' class='no-data'>No trades</td></tr>"}</tbody>
    </table>"""

    # ---- Assemble full HTML ----
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NexFlow Paper Trading Report</title>
<style>{_CSS}</style>
</head>
<body>

<h1>NexFlow Paper Trading Report</h1>
<div class="subtitle">
  {result.date_range[0]} → {result.date_range[1]} &nbsp;·&nbsp;
  {result.session_count} session{'s' if result.session_count != 1 else ''} &nbsp;·&nbsp;
  {result.files_loaded} journal file{'s' if result.files_loaded != 1 else ''} &nbsp;·&nbsp;
  {result.total_events:,} events &nbsp;·&nbsp;
  {ts_str}
</div>

<div class="cards">{cards_html}</div>

<h2>Equity Curve</h2>
<div class="chart-wrap">
  <div class="chart-title">Portfolio Value (USDT) — Initial: {result.initial_equity:,.0f} → Final: {result.final_equity:,.0f}
  &nbsp; <span class="{'pos' if net_pnl >= 0 else 'neg'}">{_pnl_fmt(net_pnl, 0)} ({net_return_pct:+.2f}%)</span></div>
  {eq_chart}
</div>

<h2>Drawdown</h2>
<div class="chart-wrap">
  <div class="chart-title">Drawdown from Peak Equity — Max: {_pct_fmt(rs_.max_drawdown)} &nbsp; Avg: {_pct_fmt(rs_.avg_drawdown)}</div>
  {dd_chart}
</div>

<hr>

<div class="two-col">
  <div>
    <h2>Trade Statistics</h2>
    <div class="card">{trade_table}</div>
  </div>
  <div>
    <h2>Risk Statistics</h2>
    <div class="card">{risk_table}</div>
  </div>
</div>

<hr>

<h2>Execution Quality</h2>
<div class="cards">{exec_cards}</div>

<hr>

<h2>PnL Distribution</h2>
<div class="chart-wrap">
  <div class="chart-title">Trade PnL Histogram — {ts_.total} trades &nbsp; Median: {_pnl_fmt(_median(pnls))} USD</div>
  {hist_chart}
</div>

<hr>

<h2>Monthly Performance</h2>
<div class="chart-wrap">
  <div class="chart-title">Monthly Net PnL (USDT)</div>
  {monthly_chart}
</div>
<div class="card" style="margin-top:12px">{monthly_table}</div>

<hr>

<h2>Market Breakdown</h2>
<div class="two-col">
  <div>
    <h2>By Symbol</h2>
    <div class="card">{sym_table}</div>
  </div>
  <div>
    <h2>Long vs Short</h2>
    <div class="card">{direction_table}</div>
  </div>
</div>

<h2>By Session</h2>
<div class="two-col">
  <div class="card">{sess_table}</div>
  <div class="chart-wrap">{sess_chart}</div>
</div>

<h2>By Volatility Regime</h2>
<div class="card">{reg_table}</div>

<hr>

<h2>Risk Events</h2>
<div class="two-col">
  <div>
    <div style="font-size:12px;color:#5a5e78;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">Kill Switch Activations</div>
    <div class="card">{kill_section}</div>
  </div>
  <div>
    <div style="font-size:12px;color:#5a5e78;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">Signal Rejections</div>
    <div class="card">{reject_section}</div>
  </div>
</div>

<footer>NexFlow &nbsp;·&nbsp; Evidence-first paper trading analysis &nbsp;·&nbsp; {ts_str}</footer>
</body>
</html>"""


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2


def _no_data_report(ts_str: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NexFlow Paper Trading Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>NexFlow Paper Trading Report</h1>
<div class="subtitle">{ts_str}</div>
<div class="no-data" style="margin-top:80px;font-size:18px">
  No paper trading data found.<br>
  <span style="font-size:14px;margin-top:8px;display:block">
    Run <code>scripts/run_paper_trader.py</code> first to generate execution journals.
  </span>
</div>
<footer>NexFlow &nbsp;·&nbsp; {ts_str}</footer>
</body>
</html>"""
