"""
Single-venue LOB monitoring dashboard (Nobitex) — Plotly Dash.

Run:
    python -m monitoring.app
    # then open http://127.0.0.1:8050

Reads the CSVs written by data-collection/exchange-apis/nobitexCollector.py.
Point it elsewhere with the LOB_DATA_DIR env var. Refreshes every 10s to match
the collector's poll cadence.

Layout maps to lob_dashboard_metrics.md:
  * Section A -> the numeric tile grid,
  * Section B -> the plot grid + trade tape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output

from . import config, figures, metrics
from .data_loader import (data_health, latest_snapshot, load_orderbook,
                          load_trades)

app = Dash(__name__, title="LOB Monitor — Nobitex")
server = app.server  # for gunicorn / external WSGI if ever needed


# ── Formatting helpers ───────────────────────────────────────────────────────

def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "—"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return str(x)


def _tile(label: str, value: str, sub: str = "", color: str = "#e0e0e0") -> html.Div:
    children = [html.Div(label, className="tile-label"),
                html.Div(value, className="tile-value", style={"color": color})]
    if sub:
        children.append(html.Div(sub, className="tile-sub"))
    return html.Div(children, className="tile")


def _signed_color(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "#e0e0e0"
    return "#26a69a" if x >= 0 else "#ef5350"


# ── Layout ───────────────────────────────────────────────────────────────────

app.layout = html.Div([
    html.Div([
        html.H2("LOB Monitor", style={"margin": "0"}),
        html.Span("Nobitex · single venue", className="subtitle"),
        dcc.Dropdown(
            id="symbol", options=[{"label": s, "value": s} for s in config.SYMBOLS],
            value=config.DEFAULT_SYMBOL, clearable=False,
            style={"width": "200px", "color": "#000"},
        ),
        html.Div(id="status", className="status"),
    ], className="header"),

    dcc.Interval(id="tick", interval=config.REFRESH_MS, n_intervals=0),

    html.Div(id="tiles", className="tile-grid"),

    # Section B — plots
    html.Div([
        html.Div([dcc.Graph(id="g_price"), dcc.Graph(id="g_spread")], className="col"),
        html.Div([dcc.Graph(id="g_imb"), dcc.Graph(id="g_flow_imb")], className="col"),
    ], className="plot-row"),

    html.Div([
        html.Div([dcc.Graph(id="g_depth"), dcc.Graph(id="g_ladder")], className="col"),
        html.Div([dcc.Graph(id="g_heatmap"), dcc.Graph(id="g_flowbars")], className="col"),
    ], className="plot-row"),

    html.Div([
        html.Div([dcc.Graph(id="g_rvol")], className="col"),
        html.Div([dcc.Graph(id="g_vpin")], className="col"),
    ], className="plot-row"),

    html.Div([
        html.H4("Trade tape", style={"marginBottom": "6px"}),
        dash_table.DataTable(
            id="tape",
            columns=[{"name": c, "id": c} for c in ["time", "price", "volume", "direction"]],
            style_as_list_view=True,
            style_header={"backgroundColor": "#1e1e1e", "color": "#ccc", "fontWeight": "bold"},
            style_cell={"backgroundColor": "#121212", "color": "#ddd",
                        "fontFamily": "monospace", "fontSize": "12px", "padding": "4px"},
            style_data_conditional=[
                {"if": {"filter_query": '{direction} = "buy"', "column_id": "direction"},
                 "color": "#26a69a"},
                {"if": {"filter_query": '{direction} = "sell"', "column_id": "direction"},
                 "color": "#ef5350"},
            ],
        ),
    ], className="tape-wrap"),
], className="root")


# ── Callback ─────────────────────────────────────────────────────────────────

@app.callback(
    [Output("status", "children"),
     Output("tiles", "children"),
     Output("g_price", "figure"), Output("g_spread", "figure"),
     Output("g_imb", "figure"), Output("g_flow_imb", "figure"),
     Output("g_depth", "figure"), Output("g_ladder", "figure"),
     Output("g_heatmap", "figure"), Output("g_flowbars", "figure"),
     Output("g_rvol", "figure"), Output("g_vpin", "figure"),
     Output("tape", "data")],
    [Input("tick", "n_intervals"), Input("symbol", "value")],
)
def refresh(_n, symbol):
    # ── Load ──
    ob = load_orderbook(symbol, lookback_sec=config.PLOT_LOOKBACK_SEC)
    trades_plot = load_trades(symbol, lookback_sec=config.PLOT_LOOKBACK_SEC)
    trades_short = load_trades(symbol, lookback_sec=config.WINDOW_SHORT_SEC)
    trades_long = load_trades(symbol, lookback_sec=config.WINDOW_LONG_SEC)
    snap = latest_snapshot(symbol)

    # ── Series ──
    mid_df = metrics.mid_micro_series(ob)
    mid_short = mid_df[mid_df["time"] >= (mid_df["time"].iloc[-1] - pd.Timedelta(seconds=config.WINDOW_SHORT_SEC))]["mid"] if not mid_df.empty else pd.Series(dtype=float)
    mid_long = mid_df[mid_df["time"] >= (mid_df["time"].iloc[-1] - pd.Timedelta(seconds=config.WINDOW_LONG_SEC))]["mid"] if not mid_df.empty else pd.Series(dtype=float)
    bins = metrics.flow_bins(trades_plot, bin_sec=60)
    rvol_df = metrics.rolling_rvol(mid_df, window=30)
    vpin_df = metrics.vpin_series(
        trades_plot, config.VPIN_BUCKET_VOLUME.get(symbol, np.nan), config.VPIN_N_BUCKETS)

    # ── Tiles (Section A) ──
    t = metrics.snapshot_tiles(snap, trades_short, trades_long, mid_short, mid_long, symbol)
    tiles = _build_tiles(t)

    # ── Status banner ──
    h = data_health(symbol)
    status = _status_text(h)

    # ── Tape ──
    tape = _build_tape(trades_plot)

    return (
        status, tiles,
        figures.price_path(mid_df), figures.spread_path(mid_df),
        figures.imbalance_path(mid_df), figures.flow_imbalance_path(bins),
        figures.depth_chart(snap), figures.book_ladder(snap),
        figures.depth_heatmap(ob), figures.flow_bars(bins),
        figures.rvol_path(rvol_df), figures.vpin_path(vpin_df),
        tape,
    )


def _build_tiles(t: dict) -> list:
    imb = t.get("imbalance", {})
    flow = t.get("flow_short", {})
    tiles = [
        _tile("Best bid", _fmt(t.get("best_bid"), 0), color="#26a69a"),
        _tile("Best ask", _fmt(t.get("best_ask"), 0), color="#ef5350"),
        _tile("Mid", _fmt(t.get("mid"), 0)),
        _tile("Microprice", _fmt(t.get("microprice"), 0)),
        _tile("Spread", _fmt(t.get("spread_abs"), 0),
              sub=f"{_fmt(t.get('spread_bps'), 1)} bps"),
        _tile("Imbalance N=1", _fmt(imb.get(1), 3), color=_signed_color(imb.get(1))),
        _tile("Imbalance N=5", _fmt(imb.get(5), 3), color=_signed_color(imb.get(5))),
        _tile("Imbalance N=20", _fmt(imb.get(20), 3), color=_signed_color(imb.get(20))),
        _tile("Bid depth", _fmt(t.get("bid_depth"), 3), color="#26a69a"),
        _tile("Ask depth", _fmt(t.get("ask_depth"), 3), color="#ef5350"),
        _tile(f"Depth ±{config.DEPTH_WITHIN_BPS}bps", _fmt(t.get("depth_within_bps"), 3)),
        _tile("Last trade", _fmt(t.get("last_trade_price"), 0),
              sub=str(t.get("last_trade_dir", "")), color=_signed_color(
                  1 if t.get("last_trade_dir") == "buy" else -1)),
        _tile("Intensity", _fmt(t.get("intensity_short"), 1), sub="trades/min (1m)"),
        _tile("Net flow", _fmt(flow.get("net"), 3), sub="buy−sell (1m)",
              color=_signed_color(flow.get("net"))),
        _tile("Flow imbalance", _fmt(t.get("flow_imbalance_short"), 3),
              sub="(1m)", color=_signed_color(t.get("flow_imbalance_short"))),
        _tile("VWAP 1m", _fmt(t.get("vwap_short"), 0)),
        _tile("TWAP 1m", _fmt(t.get("twap_short"), 0)),
        _tile("Realized vol", _fmt(t.get("rvol_long"), 5), sub="30m, 10s mids"),
        _tile("VPIN", _fmt(t.get("vpin"), 3), sub="toxicity"),
    ]
    # Slippage tiles (a few representative sizes)
    for q, v in (t.get("buy_slippage") or {}).items():
        tiles.append(_tile(f"Buy slip {q}", _fmt(v, 1), sub="bps", color="#ef5350"))
    for q, v in (t.get("sell_slippage") or {}).items():
        tiles.append(_tile(f"Sell slip {q}", _fmt(v, 1), sub="bps", color="#26a69a"))
    return tiles


def _build_tape(trades: pd.DataFrame) -> list:
    if trades.empty:
        return []
    tail = trades.tail(config.TAPE_ROWS).iloc[::-1]
    return [{
        "time": ts.strftime("%H:%M:%S.%f")[:-3],
        "price": f"{p:,.0f}",
        "volume": f"{v:,.4f}",
        "direction": d,
    } for ts, p, v, d in zip(tail["trade_time"], tail["price"],
                             tail["volume"], tail["direction"])]


def _status_text(h: dict) -> str:
    if not h["has_data"]:
        return "⚠ no order-book data found — is the collector running?"
    ob_age = h["ob_age_sec"]
    fresh = "live" if (ob_age is not None and ob_age < 30) else "stale"
    return (f"{fresh} · book {_fmt(ob_age,0)}s ago · "
            f"{h['ob_rows']} snaps / {h['tr_rows']} trades in window")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
