"""
Plotly figure builders — Section B of the metric spec.

Each function returns a ``plotly.graph_objects.Figure`` from frames produced by
``metrics`` / ``data_loader``. All are defensive: empty input yields an empty
figure with an explanatory annotation rather than an exception.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config
from .data_loader import book_sides

_TEMPLATE = "plotly_dark"
_BUY = "#26a69a"
_SELL = "#ef5350"


def _empty(title: str, msg: str = "waiting for data…") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=_TEMPLATE, title=title, height=300,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    fig.add_annotation(text=msg, showarrow=False,
                       font=dict(size=14, color="#888"))
    return fig


def _base(fig: go.Figure, title: str, height: int = 300) -> go.Figure:
    fig.update_layout(
        template=_TEMPLATE, title=title, height=height,
        margin=dict(l=50, r=20, t=40, b=30), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ── Price / micro / spread / imbalance lines ─────────────────────────────────

def price_path(mid_df: pd.DataFrame) -> go.Figure:
    if mid_df.empty:
        return _empty("Mid / Microprice")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mid_df["time"], y=mid_df["mid"],
                             name="mid", line=dict(color="#90caf9", width=1.5)))
    fig.add_trace(go.Scatter(x=mid_df["time"], y=mid_df["microprice"],
                             name="microprice", line=dict(color="#ffb74d", width=1, dash="dot")))
    return _base(fig, "Mid / Microprice over time")


def spread_path(mid_df: pd.DataFrame) -> go.Figure:
    if mid_df.empty:
        return _empty("Spread (bps)")
    fig = go.Figure(go.Scatter(x=mid_df["time"], y=mid_df["spread_bps"],
                               name="spread", line=dict(color="#ce93d8", width=1.5)))
    return _base(fig, "Quoted spread (bps) over time")


def imbalance_path(mid_df: pd.DataFrame) -> go.Figure:
    if mid_df.empty:
        return _empty("Top-N imbalance")
    fig = go.Figure()
    for col, color in [("imb_1", "#ef9a9a"), ("imb_5", "#a5d6a7"), ("imb_20", "#90caf9")]:
        fig.add_trace(go.Scatter(x=mid_df["time"], y=mid_df[col],
                                 name=col.replace("imb_", "N="),
                                 line=dict(width=1)))
    fig.add_hline(y=0, line=dict(color="#666", width=1))
    fig.update_yaxes(range=[-1, 1])
    return _base(fig, "Top-N order-book imbalance over time")


# ── Book shape (snapshot) ────────────────────────────────────────────────────

def depth_chart(snapshot: pd.Series | None) -> go.Figure:
    """Cumulative staircase: cumulative volume vs price, each side."""
    if snapshot is None:
        return _empty("Order-book depth")
    bp, bv, ap, av = book_sides(snapshot)
    if not (len(bp) and len(ap)):
        return _empty("Order-book depth")
    fig = go.Figure()
    # bids: descending price, cumulative from best
    fig.add_trace(go.Scatter(x=bp, y=np.cumsum(bv), name="bids",
                             line=dict(color=_BUY, shape="hv"), fill="tozeroy"))
    fig.add_trace(go.Scatter(x=ap, y=np.cumsum(av), name="asks",
                             line=dict(color=_SELL, shape="hv"), fill="tozeroy"))
    fig.update_xaxes(title="price")
    fig.update_yaxes(title="cumulative volume")
    return _base(fig, "Order-book depth (cumulative)")


def book_ladder(snapshot: pd.Series | None) -> go.Figure:
    """Horizontal bars: size at each of the 20 levels per side."""
    if snapshot is None:
        return _empty("Book ladder")
    bp, bv, ap, av = book_sides(snapshot)
    if not (len(bp) and len(ap)):
        return _empty("Book ladder")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[f"B{i+1}" for i in range(len(bv))], x=bv,
                         orientation="h", name="bids", marker_color=_BUY))
    fig.add_trace(go.Bar(y=[f"A{i+1}" for i in range(len(av))], x=av,
                         orientation="h", name="asks", marker_color=_SELL))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, "Book ladder (size per level)", height=420)


def depth_heatmap(ob: pd.DataFrame) -> go.Figure:
    """Depth heatmap over time — price level (rows) × snapshot (cols), color = volume.

    The highest-value plot in the spec: shows walls building/pulling and the
    book "breathing". We bin prices into a fixed grid around the rolling mid so
    the y-axis is stable as price drifts.
    """
    if ob.empty or len(ob) < 2:
        return _empty("Depth heatmap")

    # Build a price grid spanning the observed best bid/ask range.
    times, prices_per_t, vols_per_t = [], [], []
    mids = []
    for _, r in ob.iterrows():
        bp, bv, ap, av = book_sides(r)
        if not (len(bp) and len(ap)):
            continue
        times.append(r["time"])
        prices_per_t.append(np.concatenate([bp, ap]))
        vols_per_t.append(np.concatenate([bv, av]))
        mids.append((bp[0] + ap[0]) / 2.0)
    if not times:
        return _empty("Depth heatmap")

    all_prices = np.concatenate(prices_per_t)
    pmin, pmax = np.percentile(all_prices, [1, 99])
    if pmax <= pmin:
        return _empty("Depth heatmap")
    n_bins = 60
    edges = np.linspace(pmin, pmax, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    grid = np.full((n_bins, len(times)), np.nan)
    for j, (p, v) in enumerate(zip(prices_per_t, vols_per_t)):
        idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        for i, vol in zip(idx, v):
            grid[i, j] = np.nansum([grid[i, j], vol])

    fig = go.Figure(go.Heatmap(
        x=times, y=centers, z=grid, colorscale="Viridis",
        colorbar=dict(title="vol"), hoverongaps=False,
    ))
    fig.add_trace(go.Scatter(x=times, y=mids, name="mid",
                             line=dict(color="white", width=1)))
    return _base(fig, "Depth heatmap over time (price × time, color = resting volume)", height=420)


# ── Flow ─────────────────────────────────────────────────────────────────────

def flow_bars(bins: pd.DataFrame) -> go.Figure:
    if bins.empty:
        return _empty("Buy/Sell volume")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bins["bin"], y=bins["buy"], name="buy", marker_color=_BUY))
    fig.add_trace(go.Bar(x=bins["bin"], y=-bins["sell"], name="sell", marker_color=_SELL))
    fig.update_layout(barmode="relative")
    return _base(fig, "Buy / Sell volume per bin")


def flow_imbalance_path(bins: pd.DataFrame) -> go.Figure:
    if bins.empty:
        return _empty("Trade-flow imbalance")
    fig = go.Figure(go.Scatter(x=bins["bin"], y=bins["imbalance"],
                               line=dict(color="#ffd54f", width=1.5)))
    fig.add_hline(y=0, line=dict(color="#666", width=1))
    fig.update_yaxes(range=[-1, 1])
    return _base(fig, "Trade-flow imbalance over time")


def rvol_path(rvol_df: pd.DataFrame) -> go.Figure:
    if rvol_df.empty:
        return _empty("Realized volatility")
    fig = go.Figure(go.Scatter(x=rvol_df["time"], y=rvol_df["rvol"],
                               line=dict(color="#f48fb1", width=1.5)))
    return _base(fig, "Realized volatility over time (10s mids)")


def vpin_path(vpin_df: pd.DataFrame) -> go.Figure:
    if vpin_df.empty:
        return _empty("VPIN", "needs more trade flow")
    fig = go.Figure(go.Scatter(x=vpin_df["time"], y=vpin_df["vpin"],
                               line=dict(color="#ff8a65", width=1.5)))
    fig.update_yaxes(range=[0, 1])
    return _base(fig, "VPIN (toxicity) over time")
