# -*- coding: utf-8 -*-
"""
Shifu DeepLOB + DLS Trading Engine Replay
UI language: English only.
Version: v13 hard-normalized DLS initial equity.

This is intentionally not a CSV dashboard. It builds a replayable event/state
machine from the exported notebook CSVs and visualizes the trading process as an
interactive engine: signals -> filters -> target weights -> orders -> executions
-> holdings -> portfolio state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover
    st_autorefresh = None

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Shifu Trading Engine Replay",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR = DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else Path(__file__).resolve().parent

# Normalize the reconstructed DLS equity path so the first valid replay day
# starts from the same 50M initial capital used by the backtest.
INITIAL_DLS_EQUITY = 50_000_000.0

FILES = {
    "comparison": "T001_comparison_original_deeplob_exact_vs_dls.csv",
    "dls_holding": "T001_dls_holding_snapshot_audit.csv",
    "dls_trade": "T001_dls_trade_audit.csv",
    "dls_weight_debug": "T001_dls_weight_debug_shifted_tplus2_conf_filter.csv",
    "dls_weight_step": "T001_dls_weight_step_audit.csv",
    "dls_weights_long": "T001_dls_weights_long_shifted_tplus2.csv",
    "base_daily_log": "T001_original_deeplob_exact_daily_log.csv",
    "raw_signals": "T001_original_deeplob_exact_raw_oos_signals.csv",
    "base_submission": "T001_oos_original_deeplob_exact_sell_close.csv",
    "dls_submission": "T001_oos_shifu_dls_colab_sell_open.csv",
    "seed_search": "shifu_dls_seed_search_summary.csv",
    "training_history": "shifu_dls_training_history_oos.csv",
}

# -----------------------------------------------------------------------------
# Terminal-style visual design - high contrast, no white-on-white states
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg0: #050816;
        --bg1: #08111f;
        --bg2: #0b1220;
        --panel: #0f172a;
        --panel2: #111827;
        --panel3: #172033;
        --line: rgba(148, 163, 184, 0.24);
        --line2: rgba(56, 189, 248, 0.32);
        --muted: #9ca3af;
        --txt: #e5e7eb;
        --txt2: #f8fafc;
        --cyan: #38bdf8;
        --cyan2: #22d3ee;
        --green: #22c55e;
        --red: #f87171;
        --amber: #fbbf24;
        --violet: #a78bfa;
        --blue: #60a5fa;
        --pink: #fb7185;
    }

    html, body, [class*="css"] { color: var(--txt) !important; }
    .stApp {
        background:
          radial-gradient(circle at 8% 0%, rgba(56,189,248,0.18), transparent 32%),
          radial-gradient(circle at 94% 10%, rgba(167,139,250,0.13), transparent 30%),
          linear-gradient(180deg, #050816 0%, #08111f 48%, #050816 100%) !important;
        color: var(--txt) !important;
    }
    .block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1640px; }
    h1, h2, h3, h4, h5, h6 { color: var(--txt2) !important; letter-spacing: -0.02em; }
    p, label, span, div, li { color: inherit; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050816 0%, #0b1220 58%, #101827 100%) !important;
        border-right: 1px solid var(--line);
    }
    div[data-testid="stSidebar"] * { color: #e5e7eb !important; }

    /* Streamlit widget contrast fixes */
    .stTextInput input, .stNumberInput input, .stDateInput input, .stTimeInput input, textarea {
        background: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid rgba(148,163,184,.32) !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[data-testid="stVirtualDropdown"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border-color: rgba(148,163,184,.32) !important;
    }
    div[data-baseweb="select"] span, div[data-baseweb="menu"] div, div[data-baseweb="popover"] div {
        color: #f8fafc !important;
    }
    .stSlider [data-baseweb="slider"] div { color: #e5e7eb !important; }
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(180deg, #172033 0%, #0f172a 100%) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(56,189,248,.38) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 20px rgba(0,0,0,.22);
        font-weight: 800;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(56,189,248,.75) !important;
        color: #ffffff !important;
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    }
    .stAlert {
        background-color: rgba(15,23,42,.92) !important;
        border: 1px solid rgba(148,163,184,.24) !important;
        color: #e5e7eb !important;
    }
    .stAlert * { color: #e5e7eb !important; }

    .engine-hero {
        position: relative;
        border: 1px solid rgba(56,189,248,0.42);
        border-radius: 22px;
        padding: 22px 24px;
        margin-bottom: 14px;
        background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(17,24,39,0.96)),
                    radial-gradient(circle at 12% 20%, rgba(56,189,248,0.24), transparent 25%);
        box-shadow: 0 18px 50px rgba(0,0,0,0.38), inset 0 0 40px rgba(56,189,248,0.05);
    }
    .engine-hero .kicker { color: var(--cyan); font-weight: 900; letter-spacing: .16em; font-size: .78rem; text-transform: uppercase; }
    .engine-hero h1 { margin: 3px 0 6px 0; font-size: 2.2rem; line-height: 1.05; }
    .engine-hero p { color: #cbd5e1 !important; margin: 0; max-width: 1150px; }

    .panel {
        background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(15,23,42,0.90));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 15px 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.24);
        margin-bottom: 14px;
    }
    .panel-title { color:#f8fafc !important; font-weight:950; font-size: 1.03rem; margin-bottom: 8px; display:flex; align-items:center; gap:8px; }
    .panel-sub { color:#aab6c7 !important; font-size: .84rem; margin-top: -3px; margin-bottom: 10px; }

    .status-grid { display:grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap: 10px; margin: 10px 0 2px 0; }
    .status-card {
        border: 1px solid rgba(148,163,184,0.26);
        border-radius: 16px;
        padding: 12px 12px;
        background: linear-gradient(180deg, rgba(15,23,42,.96), rgba(2,6,23,.62));
    }
    .status-card.good { border-color: rgba(34,197,94,.42); box-shadow: inset 0 0 0 1px rgba(34,197,94,.08); }
    .status-card.bad { border-color: rgba(248,113,113,.42); box-shadow: inset 0 0 0 1px rgba(248,113,113,.08); }
    .status-card.warn { border-color: rgba(251,191,36,.42); box-shadow: inset 0 0 0 1px rgba(251,191,36,.08); }
    .status-label { color:#9ca3af !important; font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; font-weight:900; }
    .status-value { color:#f8fafc !important; font-size:1.28rem; font-weight:950; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .status-note { color:#94a3b8 !important; font-size:.72rem; margin-top:2px; }

    .stage-line { display:grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin: 12px 0 2px 0; }
    .stage-box {
        position:relative; min-height: 94px; border-radius: 16px; padding: 12px 12px;
        border: 1px solid rgba(148,163,184,0.24);
        background: linear-gradient(180deg, rgba(15,23,42,.92), rgba(2,6,23,.58)); overflow:hidden;
    }
    .stage-box:after { content:""; position:absolute; top:0; right:0; width:4px; height:100%; background: var(--cyan); opacity:.85; }
    .stage-num { color:#94a3b8 !important; font-size:.72rem; font-weight:950; letter-spacing:.10em; }
    .stage-title { color:#e2e8f0 !important; font-size:.86rem; font-weight:950; margin-top:4px; }
    .stage-main { color:#38bdf8 !important; font-size:1.35rem; font-weight:950; margin-top:8px; }
    .stage-sub { color:#aab6c7 !important; font-size:.74rem; margin-top:2px; }

    .terminal {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        background: #020617;
        border: 1px solid rgba(56,189,248,0.22);
        border-radius: 14px;
        padding: 13px 14px;
        color: #d1fae5 !important;
        line-height: 1.55;
        font-size: .86rem;
        white-space: pre-wrap;
    }
    .buy { color: #86efac !important; font-weight:900; }
    .sell { color: #fca5a5 !important; font-weight:900; }
    .hold { color: #93c5fd !important; font-weight:900; }
    .warn { color: #fde68a !important; font-weight:900; }
    .chip { display:inline-block; padding:4px 9px; border-radius:999px; margin: 2px 5px 2px 0; font-size:.75rem; font-weight:900; background:rgba(96,165,250,.18); color:#bfdbfe !important; border:1px solid rgba(96,165,250,.32); }
    .chip-green { background:rgba(34,197,94,.16); color:#bbf7d0 !important; border-color:rgba(34,197,94,.35); }
    .chip-red { background:rgba(248,113,113,.16); color:#fecaca !important; border-color:rgba(248,113,113,.35); }
    .chip-amber { background:rgba(251,191,36,.16); color:#fde68a !important; border-color:rgba(251,191,36,.35); }

    .model-card {
        border-radius: 18px;
        padding: 16px 16px;
        background: linear-gradient(180deg, rgba(15,23,42,.98), rgba(17,24,39,.92));
        border: 1px solid rgba(148,163,184,.26);
        min-height: 120px;
    }
    .model-card.dls { border-color: rgba(56,189,248,.48); box-shadow: inset 0 0 28px rgba(56,189,248,.06); }
    .model-card.base { border-color: rgba(251,191,36,.46); box-shadow: inset 0 0 28px rgba(251,191,36,.05); }
    .model-name { color:#f8fafc !important; font-size:1.0rem; font-weight:950; margin-bottom:6px; }
    .model-kpi { color:#f8fafc !important; font-size:1.42rem; font-weight:950; }
    .model-sub { color:#aab6c7 !important; font-size:.78rem; margin-top:3px; }
    .delta-good { color:#86efac !important; font-weight:950; }
    .delta-bad { color:#fca5a5 !important; font-weight:950; }
    .delta-neutral { color:#bfdbfe !important; font-weight:950; }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(15,23,42,.96), rgba(2,6,23,.58));
        border: 1px solid rgba(148,163,184,0.26);
        border-radius: 16px;
        padding: 12px 13px;
    }
    div[data-testid="stMetric"] label { color:#aab6c7 !important; font-weight:900; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#f8fafc !important; }
    div[data-testid="stMetricDelta"] svg { display:none; }
    .stDataFrame { border-radius: 14px; overflow: hidden; border: 1px solid rgba(148,163,184,.20); }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(148,163,184,.18); }
    .stTabs [data-baseweb="tab"] {
        background: rgba(15,23,42,0.88);
        border: 1px solid rgba(148,163,184,0.22);
        border-bottom: none;
        border-radius: 14px 14px 0 0;
        padding: 10px 16px;
        color: #cbd5e1 !important;
        font-weight: 850;
    }
    .stTabs [aria-selected="true"] {
        border-color: rgba(56,189,248,0.55);
        color:#38bdf8 !important;
        background: rgba(15,23,42,1.0) !important;
    }
    .js-plotly-plot .plotly .modebar { background: rgba(15,23,42,.75) !important; }
    @media (max-width: 1200px) { .status-grid, .stage-line { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 800px) { .status-grid, .stage-line { grid-template-columns: repeat(2, 1fr); } }

    .signal-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 10px 0 14px 0; }
    .signal-card { border-radius: 16px; padding: 14px 16px; background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(17,24,39,.94)); border: 1px solid rgba(148,163,184,.22); box-shadow: 0 12px 26px rgba(0,0,0,.20); }
    .signal-card.down { border-color: rgba(248,113,113,.44); box-shadow: 0 0 0 1px rgba(248,113,113,.06), 0 14px 30px rgba(0,0,0,.24); }
    .signal-card.flat { border-color: rgba(251,191,36,.40); box-shadow: 0 0 0 1px rgba(251,191,36,.05), 0 14px 30px rgba(0,0,0,.24); }
    .signal-card.up { border-color: rgba(34,197,94,.46); box-shadow: 0 0 0 1px rgba(34,197,94,.06), 0 14px 30px rgba(0,0,0,.24); }
    .signal-label { color:#cbd5e1; font-size:.76rem; font-weight:900; letter-spacing:.12em; text-transform:uppercase; }
    .signal-value { color:#f8fafc; font-size:1.9rem; font-weight:950; line-height:1.15; margin-top:4px; }
    .signal-note { color:#94a3b8; font-size:.82rem; margin-top:4px; }
    .asset-chip-row { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 12px 0; }
    .asset-mini-chip { padding: 5px 9px; border-radius: 999px; background: rgba(15,23,42,.92); border: 1px solid rgba(148,163,184,.24); color:#dbeafe; font-size:.78rem; font-weight:800; }

    .explain-box {
        border: 1px solid rgba(56,189,248,0.24);
        background: linear-gradient(180deg, rgba(8,47,73,0.22), rgba(15,23,42,0.80));
        border-radius: 16px;
        padding: 12px 14px;
        margin: 8px 0 14px 0;
        color: #dbeafe !important;
    }
    .explain-box b { color:#f8fafc !important; }
    .explain-box .why { color:#93c5fd !important; font-weight: 900; letter-spacing:.04em; text-transform: uppercase; font-size:.73rem; }
    .explain-box p { color:#cbd5e1 !important; margin: 4px 0 0 0; line-height: 1.55; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_TEMPLATE = "plotly_dark"

# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------
def day_num(x: object) -> int:
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else -1


def ordered_day_list(values: Iterable[object]) -> list[str]:
    vals = pd.Series(list(values)).dropna().astype(str).unique().tolist()
    return sorted(vals, key=day_num)


def pct(x: object, decimals: int = 2, already_percent: bool = False) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    if not np.isfinite(v):
        return "—"
    if not already_percent and abs(v) <= 2:
        v *= 100
    return f"{v:.{decimals}f}%"


def money(x: object) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    if not np.isfinite(v):
        return "—"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e9:
        return f"{sign}¥{v/1e9:.2f}B"
    if v >= 1e6:
        return f"{sign}¥{v/1e6:.2f}M"
    if v >= 1e3:
        return f"{sign}¥{v/1e3:.1f}K"
    return f"{sign}¥{v:,.0f}"


def num(x: object, decimals: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    if not np.isfinite(v):
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{decimals}f}"


def as_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def add_day_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "trade_day_id" not in df.columns:
        return df
    df = df.copy()
    df["day_index"] = df["trade_day_id"].map(day_num)
    return df

# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Booting trading engine and loading CSV state tables...")
def load_tables() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for key, fname in FILES.items():
        path = DATA_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            if "trade_day_id" in df.columns:
                df["trade_day_id"] = df["trade_day_id"].astype(str)
                df = add_day_index(df)
            if "signal_day_id" in df.columns:
                df["signal_day_id"] = df["signal_day_id"].astype(str)
            if "asset_id" in df.columns:
                df["asset_id"] = df["asset_id"].astype(str)
            out[key] = df
        else:
            out[key] = empty_df()

    numeric_cols = {
        "dls_weight_step": [
            "prob_down", "prob_flat", "prob_up", "signal_score", "confidence",
            "target_weight", "pre_trade_weight", "post_trade_weight",
            "weight_gap_before_trade", "weight_gap_after_trade", "buy_shares", "sell_shares",
            "buy_turnover", "sell_turnover", "buy_cost", "sell_cost",
        ],
        "dls_trade": ["shares", "execution_price", "turnover", "cost", "order_percentage", "target_weight", "pre_trade_weight"],
        "dls_holding": ["shares", "close_price", "market_value", "eod_weight"],
        "dls_weight_debug": [
            "target_gross", "target_abs_gross", "num_target_names", "num_negative_names", "max_weight", "min_weight",
            "mean_positive_weight", "median_positive_weight", "num_entry_candidates", "num_quality_candidates",
            "min_confidence", "min_signal_score",
        ],
        "dls_weights_long": ["weight"],
        "base_daily_log": ["portfolio_value", "cash", "num_holdings", "num_buys", "num_sells", "buy_turnover", "sell_turnover", "total_costs"],
        "raw_signals": ["prob_down", "prob_flat", "prob_up", "signal"],
        "base_submission": ["buy_percentage", "sell_percentage"],
        "dls_submission": ["buy_percentage", "sell_percentage"],
        "comparison": ["total_return_%", "cagr_%", "sharpe", "max_drawdown_%", "score_proxy", "total_costs_RMB", "avg_turnover_RMB", "win_rate_%", "avg_holdings"],
        "seed_search": ["seed", "total_return_%", "cagr_%", "sharpe", "max_drawdown_%", "score_proxy", "total_costs_RMB", "avg_turnover_RMB", "win_rate_%", "avg_holdings"],
        "training_history": [
            "train_loss", "train_sharpe", "train_sortino", "train_turnover", "train_mean_net_return",
            "val_loss", "val_sharpe", "val_sortino", "val_turnover", "val_mean_net_return", "epoch",
            "best_val_loss_to_date", "best_epoch_to_date", "stale_epochs",
        ],
    }
    for key, cols in numeric_cols.items():
        if key in out and not out[key].empty:
            out[key] = as_numeric(out[key], cols)

    # Booleans may be loaded as bool or strings depending on environment.
    if not out["dls_weight_step"].empty:
        for c in ["entry_candidate", "quality_candidate"]:
            if c in out["dls_weight_step"].columns:
                out["dls_weight_step"][c] = out["dls_weight_step"][c].astype(str).str.lower().isin(["true", "1", "yes"])

    return out


def safe_group_sum(df: pd.DataFrame, group: str, col: str) -> pd.Series:
    if df.empty or group not in df.columns or col not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby(group)[col].sum(min_count=1)


@st.cache_data(show_spinner=False)
def build_daily_summary(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    day_sources = []
    for key in ["dls_weight_step", "dls_holding", "dls_trade", "dls_weight_debug", "base_daily_log", "raw_signals", "dls_submission", "base_submission"]:
        df = tables.get(key, empty_df())
        if not df.empty and "trade_day_id" in df.columns:
            day_sources.extend(df["trade_day_id"].dropna().astype(str).tolist())
    days = ordered_day_list(day_sources)
    summary = pd.DataFrame({"trade_day_id": days})
    summary["day_index"] = summary["trade_day_id"].map(day_num)

    step = tables.get("dls_weight_step", empty_df())
    if not step.empty:
        g = step.groupby("trade_day_id")
        step_sum = pd.DataFrame({
            "signal_day_id": g["signal_day_id"].first() if "signal_day_id" in step.columns else np.nan,
            "asset_universe": g["asset_id"].nunique(),
            "entry_candidates": g["entry_candidate"].sum() if "entry_candidate" in step.columns else np.nan,
            "quality_candidates": g["quality_candidate"].sum() if "quality_candidate" in step.columns else np.nan,
            "target_names": g["target_weight"].apply(lambda s: (pd.to_numeric(s, errors="coerce") > 0).sum()) if "target_weight" in step.columns else np.nan,
            "target_gross_from_steps": g["target_weight"].sum(min_count=1) if "target_weight" in step.columns else np.nan,
            "max_target_weight": g["target_weight"].max() if "target_weight" in step.columns else np.nan,
            "mean_confidence": g["confidence"].mean() if "confidence" in step.columns else np.nan,
            "max_confidence": g["confidence"].max() if "confidence" in step.columns else np.nan,
            "mean_signal_score": g["signal_score"].mean() if "signal_score" in step.columns else np.nan,
            "max_signal_score": g["signal_score"].max() if "signal_score" in step.columns else np.nan,
            "buy_names_step": g["buy_shares"].apply(lambda s: (pd.to_numeric(s, errors="coerce") > 0).sum()) if "buy_shares" in step.columns else np.nan,
            "sell_names_step": g["sell_shares"].apply(lambda s: (pd.to_numeric(s, errors="coerce") > 0).sum()) if "sell_shares" in step.columns else np.nan,
            "buy_turnover_step": g["buy_turnover"].sum(min_count=1) if "buy_turnover" in step.columns else np.nan,
            "sell_turnover_step": g["sell_turnover"].sum(min_count=1) if "sell_turnover" in step.columns else np.nan,
            "buy_cost_step": g["buy_cost"].sum(min_count=1) if "buy_cost" in step.columns else np.nan,
            "sell_cost_step": g["sell_cost"].sum(min_count=1) if "sell_cost" in step.columns else np.nan,
        }).reset_index()
        summary = summary.merge(step_sum, on="trade_day_id", how="left")

    debug = tables.get("dls_weight_debug", empty_df())
    if not debug.empty:
        keep = [c for c in [
            "trade_day_id", "mode", "target_gross", "target_abs_gross", "num_target_names",
            "num_negative_names", "max_weight", "mean_positive_weight", "median_positive_weight",
            "num_entry_candidates", "num_quality_candidates", "min_confidence", "min_signal_score", "mask_type"
        ] if c in debug.columns]
        summary = summary.merge(debug[keep].drop_duplicates("trade_day_id"), on="trade_day_id", how="left")

    trades = tables.get("dls_trade", empty_df())
    if not trades.empty:
        tr = trades.copy()
        tr["buy_turnover"] = np.where(tr.get("side", "") == "buy", tr.get("turnover", 0), 0)
        tr["sell_turnover"] = np.where(tr.get("side", "") == "sell", tr.get("turnover", 0), 0)
        g = tr.groupby("trade_day_id")
        trade_sum = pd.DataFrame({
            "executed_trades": g["asset_id"].count(),
            "executed_assets": g["asset_id"].nunique(),
            "executed_buys": g["side"].apply(lambda s: (s == "buy").sum()) if "side" in tr.columns else np.nan,
            "executed_sells": g["side"].apply(lambda s: (s == "sell").sum()) if "side" in tr.columns else np.nan,
            "executed_turnover": g["turnover"].sum(min_count=1) if "turnover" in tr.columns else np.nan,
            "executed_cost": g["cost"].sum(min_count=1) if "cost" in tr.columns else np.nan,
            "executed_buy_turnover": g["buy_turnover"].sum(min_count=1),
            "executed_sell_turnover": g["sell_turnover"].sum(min_count=1),
        }).reset_index()
        summary = summary.merge(trade_sum, on="trade_day_id", how="left")

    hold = tables.get("dls_holding", empty_df())
    if not hold.empty:
        g = hold.groupby("trade_day_id")
        hold_sum = pd.DataFrame({
            "eod_holdings": g["asset_id"].nunique(),
            "eod_market_value": g["market_value"].sum(min_count=1) if "market_value" in hold.columns else np.nan,
            "max_eod_weight": g["eod_weight"].max() if "eod_weight" in hold.columns else np.nan,
            "top10_eod_weight": g["eod_weight"].apply(lambda s: pd.to_numeric(s, errors="coerce").nlargest(10).sum()) if "eod_weight" in hold.columns else np.nan,
        }).reset_index()
        summary = summary.merge(hold_sum, on="trade_day_id", how="left")

    raw = tables.get("raw_signals", empty_df())
    if not raw.empty:
        g = raw.groupby("trade_day_id")
        raw_sum = pd.DataFrame({
            "raw_assets": g["asset_id"].nunique(),
            "raw_buy_signals": g["signal"].apply(lambda s: (pd.to_numeric(s, errors="coerce") == 1).sum()) if "signal" in raw.columns else np.nan,
            "raw_sell_signals": g["signal"].apply(lambda s: (pd.to_numeric(s, errors="coerce") == -1).sum()) if "signal" in raw.columns else np.nan,
            "raw_flat_signals": g["signal"].apply(lambda s: (pd.to_numeric(s, errors="coerce") == 0).sum()) if "signal" in raw.columns else np.nan,
            "avg_prob_up": g["prob_up"].mean() if "prob_up" in raw.columns else np.nan,
            "max_prob_up": g["prob_up"].max() if "prob_up" in raw.columns else np.nan,
        }).reset_index()
        summary = summary.merge(raw_sum, on="trade_day_id", how="left")

    for prefix, key in [("dls_order", "dls_submission"), ("base_order", "base_submission")]:
        df = tables.get(key, empty_df())
        if not df.empty:
            g = df.groupby("trade_day_id")
            order_sum = pd.DataFrame({
                f"{prefix}_names": g["asset_id"].nunique(),
                f"{prefix}_buy_names": g["buy_percentage"].apply(lambda s: (pd.to_numeric(s, errors="coerce") > 0).sum()) if "buy_percentage" in df.columns else np.nan,
                f"{prefix}_sell_names": g["sell_percentage"].apply(lambda s: (pd.to_numeric(s, errors="coerce") > 0).sum()) if "sell_percentage" in df.columns else np.nan,
                f"{prefix}_buy_pct": g["buy_percentage"].sum(min_count=1) if "buy_percentage" in df.columns else np.nan,
                f"{prefix}_sell_pct": g["sell_percentage"].sum(min_count=1) if "sell_percentage" in df.columns else np.nan,
            }).reset_index()
            summary = summary.merge(order_sum, on="trade_day_id", how="left")

    base = tables.get("base_daily_log", empty_df())
    if not base.empty:
        keep = [c for c in ["trade_day_id", "portfolio_value", "cash", "num_holdings", "num_buys", "num_sells", "buy_turnover", "sell_turnover", "total_costs"] if c in base.columns]
        base_small = base[keep].rename(columns={
            "portfolio_value": "base_portfolio_value",
            "cash": "base_cash",
            "num_holdings": "base_num_holdings",
            "num_buys": "base_num_buys",
            "num_sells": "base_num_sells",
            "buy_turnover": "base_buy_turnover",
            "sell_turnover": "base_sell_turnover",
            "total_costs": "base_total_costs",
        })
        summary = summary.merge(base_small, on="trade_day_id", how="left")

    # Fill and derived values.
    for c in [
        "entry_candidates", "quality_candidates", "target_names", "executed_trades", "executed_assets", "executed_buys",
        "executed_sells", "eod_holdings", "raw_assets", "raw_buy_signals", "raw_sell_signals", "raw_flat_signals",
        "dls_order_names", "dls_order_buy_names", "dls_order_sell_names", "base_order_names", "base_order_buy_names", "base_order_sell_names"
    ]:
        if c in summary.columns:
            summary[c] = summary[c].fillna(0).astype(int)

    if "target_gross" not in summary.columns:
        summary["target_gross"] = np.nan
    summary["engine_target_gross"] = summary["target_gross"].fillna(summary.get("target_gross_from_steps", np.nan)).fillna(0.95)

    # Keep the replay table in chronological order BEFORE creating the DLS
    # equity path. This removes any ambiguity from merge order and makes the
    # first visible replay day the anchor for the 50M normalization.
    summary = summary.sort_values("day_index").reset_index(drop=True)

    # The exported DLS audit CSVs do not contain one official daily
    # `portfolio_value` column. The dashboard therefore reconstructs a raw equity
    # proxy from EOD holdings and target gross, then hard-normalizes the entire
    # path so the first valid DLS replay value is exactly INITIAL_DLS_EQUITY
    # (= 50,000,000). The first valid row is also explicitly overwritten to
    # 50,000,000 so the status card and tables cannot display 49.7M on day one.
    summary["dls_equity_proxy_raw"] = summary.get("eod_market_value", np.nan) / summary["engine_target_gross"].replace(0, np.nan)
    summary["dls_equity_scale_factor"] = np.nan
    summary["dls_equity_proxy"] = np.nan

    first_valid_dls_idx = summary["dls_equity_proxy_raw"].first_valid_index()
    if first_valid_dls_idx is not None:
        first_dls_equity_proxy_raw = float(summary.loc[first_valid_dls_idx, "dls_equity_proxy_raw"])
        if np.isfinite(first_dls_equity_proxy_raw) and first_dls_equity_proxy_raw != 0:
            scale_factor = INITIAL_DLS_EQUITY / first_dls_equity_proxy_raw
        else:
            scale_factor = 1.0
        summary["dls_equity_scale_factor"] = scale_factor
        summary["dls_equity_proxy"] = summary["dls_equity_proxy_raw"] * scale_factor
        summary.loc[first_valid_dls_idx, "dls_equity_proxy"] = INITIAL_DLS_EQUITY

    summary["dls_eod_market_value_scaled"] = summary.get("eod_market_value", np.nan) * summary["dls_equity_scale_factor"]
    summary["dls_cash_proxy"] = summary["dls_equity_proxy"] - summary["dls_eod_market_value_scaled"]
    summary["dls_return_proxy_%"] = (summary["dls_equity_proxy"] / INITIAL_DLS_EQUITY - 1.0) * 100 if summary["dls_equity_proxy"].notna().any() else np.nan
    if "base_portfolio_value" in summary.columns and summary["base_portfolio_value"].notna().any():
        summary["base_return_%"] = (summary["base_portfolio_value"] / summary["base_portfolio_value"].dropna().iloc[0] - 1.0) * 100
    else:
        summary["base_return_%"] = np.nan
    summary["executed_total_cost"] = summary.get("executed_cost", np.nan).fillna(summary.get("buy_cost_step", 0).fillna(0) + summary.get("sell_cost_step", 0).fillna(0))
    summary["executed_total_turnover"] = summary.get("executed_turnover", np.nan).fillna(summary.get("buy_turnover_step", 0).fillna(0) + summary.get("sell_turnover_step", 0).fillna(0))

    # Daily model-comparison metrics used by the replay-level comparison view.
    # Base DeepLOB values come from the exported daily log. DeepLOB + DLS values
    # are reconstructed from the DLS holdings/trades because the CSV exports store
    # DLS state through holdings/trade audits rather than a single daily-log file.
    summary["dls_daily_return_%"] = summary["dls_equity_proxy"].pct_change() * 100
    if summary["dls_return_proxy_%"].notna().any():
        first_valid = summary["dls_return_proxy_%"].first_valid_index()
        if first_valid is not None:
            summary.loc[first_valid, "dls_daily_return_%"] = summary.loc[first_valid, "dls_return_proxy_%"]
    if "base_portfolio_value" in summary.columns:
        summary["base_daily_return_%"] = summary["base_portfolio_value"].pct_change() * 100
        if summary["base_return_%"].notna().any():
            first_valid_base = summary["base_return_%"].first_valid_index()
            if first_valid_base is not None:
                summary.loc[first_valid_base, "base_daily_return_%"] = summary.loc[first_valid_base, "base_return_%"]
    else:
        summary["base_daily_return_%"] = np.nan

    summary["dls_buy_turnover_day"] = summary.get("executed_buy_turnover", np.nan).fillna(summary.get("buy_turnover_step", 0).fillna(0))
    summary["dls_sell_turnover_day"] = summary.get("executed_sell_turnover", np.nan).fillna(summary.get("sell_turnover_step", 0).fillna(0))
    summary["base_total_turnover_day"] = summary.get("base_buy_turnover", 0).fillna(0) + summary.get("base_sell_turnover", 0).fillna(0)
    summary["dls_cum_turnover"] = summary["executed_total_turnover"].fillna(0).cumsum()
    summary["base_cum_turnover"] = summary["base_total_turnover_day"].fillna(0).cumsum()
    summary["dls_cum_cost"] = summary["executed_total_cost"].fillna(0).cumsum()
    summary["base_cum_cost"] = summary.get("base_total_costs", pd.Series(0, index=summary.index)).fillna(0).cumsum()
    summary["daily_return_delta_pp"] = summary["dls_daily_return_%"] - summary["base_daily_return_%"]
    summary["cum_return_delta_pp"] = summary["dls_return_proxy_%"] - summary["base_return_%"]
    summary["daily_turnover_delta"] = summary["executed_total_turnover"] - summary["base_total_turnover_day"]
    summary["daily_cost_delta"] = summary["executed_total_cost"] - summary.get("base_total_costs", pd.Series(0, index=summary.index)).fillna(0)
    summary["holdings_delta"] = summary["eod_holdings"] - summary.get("base_num_holdings", pd.Series(0, index=summary.index)).fillna(0)

    summary["quality_pass_rate"] = summary["quality_candidates"] / summary["entry_candidates"].replace(0, np.nan)
    summary["target_utilization"] = summary["target_names"] / summary["quality_candidates"].replace(0, np.nan)
    summary["execution_rate"] = summary["executed_assets"] / summary["target_names"].replace(0, np.nan)
    return summary.sort_values("day_index").reset_index(drop=True)

# -----------------------------------------------------------------------------
# Per-step state extraction
# -----------------------------------------------------------------------------
@dataclass
class EngineState:
    day: str
    index: int
    row: pd.Series
    raw: pd.DataFrame
    step: pd.DataFrame
    trades: pd.DataFrame
    holdings: pd.DataFrame
    debug: pd.DataFrame
    dls_orders: pd.DataFrame
    base_orders: pd.DataFrame


def slice_day(df: pd.DataFrame, day: str) -> pd.DataFrame:
    if df is None or df.empty or "trade_day_id" not in df.columns:
        return empty_df()
    return df[df["trade_day_id"].astype(str) == str(day)].copy()


def get_state(tables: Dict[str, pd.DataFrame], summary: pd.DataFrame, idx: int) -> EngineState:
    idx = max(0, min(idx, len(summary) - 1))
    row = summary.iloc[idx]
    day = str(row["trade_day_id"])
    return EngineState(
        day=day,
        index=idx,
        row=row,
        raw=slice_day(tables.get("raw_signals", empty_df()), day),
        step=slice_day(tables.get("dls_weight_step", empty_df()), day),
        trades=slice_day(tables.get("dls_trade", empty_df()), day),
        holdings=slice_day(tables.get("dls_holding", empty_df()), day),
        debug=slice_day(tables.get("dls_weight_debug", empty_df()), day),
        dls_orders=slice_day(tables.get("dls_submission", empty_df()), day),
        base_orders=slice_day(tables.get("base_submission", empty_df()), day),
    )

# -----------------------------------------------------------------------------
# Chart helpers
# -----------------------------------------------------------------------------
def apply_dark_layout(fig: go.Figure, height: int = 360, margin: Optional[dict] = None) -> go.Figure:
    """Apply one high-contrast theme to every Plotly figure.

    The previous version mixed transparent plots with Streamlit's default widget
    backgrounds, which could create low-contrast white-on-white areas on some
    local installations. This layout keeps every chart explicitly dark.
    """
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0f172a",
        margin=margin or dict(l=24, r=24, t=48, b=34),
        font=dict(color="#e5e7eb", family="Inter, Segoe UI, Arial, sans-serif"),
        title=dict(font=dict(color="#f8fafc", size=17), x=0.01, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(15,23,42,0.65)", bordercolor="rgba(148,163,184,0.18)", borderwidth=1),
        hoverlabel=dict(bgcolor="#020617", bordercolor="rgba(56,189,248,.45)", font_color="#f8fafc"),
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.16)",
        zerolinecolor="rgba(148,163,184,0.30)",
        linecolor="rgba(148,163,184,0.25)",
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#e5e7eb"),
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.16)",
        zerolinecolor="rgba(148,163,184,0.30)",
        linecolor="rgba(148,163,184,0.25)",
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#e5e7eb"),
    )
    return fig


def current_day_line(fig: go.Figure, day_index: int, yref: str = "paper") -> go.Figure:
    fig.add_vline(x=day_index, line_width=2, line_dash="dash", line_color="#22d3ee")
    return fig


def chart_returns(summary: pd.DataFrame, current_idx: int) -> go.Figure:
    x = summary["day_index"]
    fig = go.Figure()
    if "dls_return_proxy_%" in summary.columns:
        fig.add_trace(go.Scatter(x=x, y=summary["dls_return_proxy_%"], mode="lines", name="DLS equity proxy", line=dict(width=3, color="#22d3ee")))
    if "base_return_%" in summary.columns:
        fig.add_trace(go.Scatter(x=x, y=summary["base_return_%"], mode="lines", name="Base DeepLOB", line=dict(width=2, color="#f59e0b")))
    current_day_line(fig, int(summary.iloc[current_idx]["day_index"]))
    fig.update_layout(title="Engine Race: DLS vs Base Return Path", yaxis_title="Return (%)", xaxis_title="Trade day")
    return apply_dark_layout(fig, 340)


def chart_activity(summary: pd.DataFrame, current_idx: int) -> go.Figure:
    x = summary["day_index"]
    fig = go.Figure()
    for col, label, color in [
        ("entry_candidates", "Entry candidates", "#60a5fa"),
        ("quality_candidates", "Quality-filtered", "#22c55e"),
        ("target_names", "Target names", "#22d3ee"),
        ("executed_assets", "Executed assets", "#f59e0b"),
    ]:
        if col in summary.columns:
            fig.add_trace(go.Scatter(x=x, y=summary[col], mode="lines", name=label, line=dict(width=2, color=color)))
    current_day_line(fig, int(summary.iloc[current_idx]["day_index"]))
    fig.update_layout(title="Decision Activity Over Time", yaxis_title="Number of assets", xaxis_title="Trade day")
    return apply_dark_layout(fig, 310)


def chart_funnel(state: EngineState) -> go.Figure:
    r = state.row
    labels = ["Raw Signals", "Entry", "Quality Filter", "Target Weights", "Executed", "EOD Holdings"]
    values = [
        int(r.get("raw_assets", len(state.raw))),
        int(r.get("entry_candidates", 0)),
        int(r.get("quality_candidates", 0)),
        int(r.get("target_names", 0)),
        int(r.get("executed_assets", 0)),
        int(r.get("eod_holdings", 0)),
    ]
    fig = go.Figure(go.Funnel(
        y=labels,
        x=values,
        textinfo="value+percent previous",
        marker=dict(color=["#60a5fa", "#3b82f6", "#22c55e", "#22d3ee", "#f59e0b", "#a78bfa"]),
        connector=dict(line=dict(color="rgba(226,232,240,0.28)", width=1)),
    ))
    fig.update_layout(title=f"Decision Funnel — {state.day}")
    return apply_dark_layout(fig, 360)


def chart_sankey(state: EngineState) -> go.Figure:
    r = state.row
    raw = max(float(r.get("raw_assets", len(state.raw))), 0)
    entry = max(float(r.get("entry_candidates", 0)), 0)
    quality = max(float(r.get("quality_candidates", 0)), 0)
    target = max(float(r.get("target_names", 0)), 0)
    executed = max(float(r.get("executed_assets", 0)), 0)
    holdings = max(float(r.get("eod_holdings", 0)), 0)
    labels = ["Signal Universe", "Entry Mask", "Confidence Filter", "DLS Optimizer", "Execution Engine", "Portfolio State", "Rejected", "Not Executed"]
    source = [0, 0, 1, 1, 2, 2, 3, 3, 4]
    target_nodes = [1, 6, 2, 6, 3, 6, 4, 7, 5]
    values = [entry, max(raw-entry,0), quality, max(entry-quality,0), target, max(quality-target,0), executed, max(target-executed,0), holdings]
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=16, thickness=16, line=dict(color="rgba(226,232,240,0.35)", width=0.5), label=labels,
                  color=["#60a5fa", "#3b82f6", "#22c55e", "#22d3ee", "#f59e0b", "#a78bfa", "#ef4444", "#64748b"]),
        link=dict(source=source, target=target_nodes, value=values,
                  color=["rgba(96,165,250,.35)", "rgba(239,68,68,.22)", "rgba(34,197,94,.35)", "rgba(239,68,68,.18)", "rgba(34,211,238,.35)", "rgba(239,68,68,.16)", "rgba(245,158,11,.38)", "rgba(100,116,139,.25)", "rgba(167,139,250,.36)"])
    )])
    fig.update_layout(title=f"State Transition Flow — {state.day}")
    return apply_dark_layout(fig, 360, dict(l=10, r=10, t=42, b=10))


def chart_signal_distribution(state: EngineState) -> go.Figure:
    df = state.raw if not state.raw.empty else state.step
    fig = go.Figure()
    if not df.empty and "prob_up" in df.columns:
        fig.add_trace(go.Histogram(x=df["prob_up"], nbinsx=45, name="P(up)", marker_color="#22d3ee", opacity=0.72))
    if not state.step.empty and "confidence" in state.step.columns:
        fig.add_trace(go.Histogram(x=state.step["confidence"], nbinsx=45, name="Confidence", marker_color="#a78bfa", opacity=0.48))
    fig.update_layout(title=f"Signal and Confidence Distribution — {state.day}", barmode="overlay", xaxis_title="Score", yaxis_title="Asset count")
    return apply_dark_layout(fig, 310)



def prepare_deeplob_signal_frame(state: EngineState) -> pd.DataFrame:
    """Build an asset-level same-day DeepLOB 3-class signal table.

    The raw DeepLOB export contains the three model probabilities:
    prob_down / prob_flat / prob_up.  DLS step data is merged only to explain
    what the downstream trading engine did with those signals at the same
    replay clock.
    """
    src = state.raw.copy() if not state.raw.empty else state.step.copy()
    if src.empty:
        return empty_df()
    keep = [c for c in ["asset_id", "trade_day_id", "prob_down", "prob_flat", "prob_up", "signal"] if c in src.columns]
    df = src[keep].copy()
    for c in ["prob_down", "prob_flat", "prob_up", "signal"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["prob_down", "prob_flat", "prob_up"]:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize only for display safety when exported probabilities have tiny
    # floating-point drift. If a row has no valid probabilities, keep NaNs.
    prob_cols = ["prob_down", "prob_flat", "prob_up"]
    prob_sum = df[prob_cols].sum(axis=1).replace(0, np.nan)
    for c in prob_cols:
        df[c] = df[c] / prob_sum

    df["dominant_prob"] = df[prob_cols].max(axis=1)
    argmax = df[prob_cols].idxmax(axis=1).map({"prob_down": "Down", "prob_flat": "Flat", "prob_up": "Up"})
    if "signal" in df.columns:
        sig_label = df["signal"].map({-1: "Down", 0: "Flat", 1: "Up"})
        df["signal_label"] = sig_label.fillna(argmax)
    else:
        df["signal_label"] = argmax
    df["signal_margin"] = df[prob_cols].apply(lambda r: r.nlargest(2).iloc[0] - r.nlargest(2).iloc[1] if r.notna().sum() >= 2 else np.nan, axis=1)

    if not state.step.empty:
        step_cols = [
            "asset_id", "signal_day_id", "signal_score", "confidence", "entry_candidate", "quality_candidate",
            "target_weight", "pre_trade_weight", "post_trade_weight", "buy_shares", "sell_shares",
            "buy_turnover", "sell_turnover"
        ]
        stp = state.step[[c for c in step_cols if c in state.step.columns]].copy()
        df = df.merge(stp, on="asset_id", how="left", suffixes=("", "_dls"))

    for c in ["target_weight", "pre_trade_weight", "post_trade_weight", "buy_shares", "sell_shares", "buy_turnover", "sell_turnover", "confidence", "signal_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["engine_action"] = "NO TARGET"
    if "target_weight" in df.columns:
        df.loc[df["target_weight"].fillna(0) > 0, "engine_action"] = "TARGET"
    if "buy_shares" in df.columns:
        df.loc[df["buy_shares"].fillna(0) > 0, "engine_action"] = "BUY"
    if "sell_shares" in df.columns:
        df.loc[df["sell_shares"].fillna(0) > 0, "engine_action"] = "SELL"
    return df


def deeplob_signal_cards_html(sig_df: pd.DataFrame) -> str:
    if sig_df.empty:
        return "<div class='panel-sub'>No DeepLOB signal rows are available for this replay state.</div>"
    counts = sig_df["signal_label"].value_counts()
    total = max(len(sig_df), 1)
    avg_down = sig_df["prob_down"].mean()
    avg_flat = sig_df["prob_flat"].mean()
    avg_up = sig_df["prob_up"].mean()
    return f"""
    <div class="signal-grid">
      <div class="signal-card down"><div class="signal-label">DeepLOB Down Signal</div><div class="signal-value">{pct(avg_down, 1)}</div><div class="signal-note">dominant on {int(counts.get('Down', 0)):,} / {total:,} assets</div></div>
      <div class="signal-card flat"><div class="signal-label">DeepLOB Flat Signal</div><div class="signal-value">{pct(avg_flat, 1)}</div><div class="signal-note">dominant on {int(counts.get('Flat', 0)):,} / {total:,} assets</div></div>
      <div class="signal-card up"><div class="signal-label">DeepLOB Up Signal</div><div class="signal-value">{pct(avg_up, 1)}</div><div class="signal-note">dominant on {int(counts.get('Up', 0)):,} / {total:,} assets</div></div>
    </div>
    """


def top_deeplob_assets(sig_df: pd.DataFrame, sort_by: str, top_n: int) -> pd.DataFrame:
    if sig_df.empty:
        return sig_df
    sort_map = {
        "Highest P(up)": "prob_up",
        "Highest P(flat)": "prob_flat",
        "Highest P(down)": "prob_down",
        "Highest 3-class confidence": "dominant_prob",
        "Largest signal margin": "signal_margin",
        "Largest DLS target weight": "target_weight",
    }
    col = sort_map.get(sort_by, "prob_up")
    if col not in sig_df.columns:
        col = "prob_up"
    return sig_df.sort_values(col, ascending=False).head(top_n).copy()


def chart_deeplob_stacked_probs(sig_df: pd.DataFrame, sort_by: str = "Highest P(up)", top_n: int = 25) -> go.Figure:
    show = top_deeplob_assets(sig_df, sort_by, top_n)
    if show.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No DeepLOB 3-signal data"), 360)
    show = show.sort_values("prob_up")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=show["prob_down"], y=show["asset_id"], orientation="h", name="P(down)", marker_color="#f87171"))
    fig.add_trace(go.Bar(x=show["prob_flat"], y=show["asset_id"], orientation="h", name="P(flat)", marker_color="#fbbf24"))
    fig.add_trace(go.Bar(x=show["prob_up"], y=show["asset_id"], orientation="h", name="P(up)", marker_color="#22c55e"))
    fig.update_layout(title=f"Same-Day DeepLOB 3-Class Signal Stack — Top {len(show)}", barmode="stack", xaxis_title="Probability mass", yaxis_title="Asset")
    fig.update_xaxes(range=[0, 1])
    return apply_dark_layout(fig, 520)


def chart_deeplob_ternary(sig_df: pd.DataFrame, focus_asset: Optional[str] = None, max_points: int = 700) -> go.Figure:
    if sig_df.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No DeepLOB ternary signal map"), 420)
    df = sig_df.dropna(subset=["prob_down", "prob_flat", "prob_up"]).copy()
    if df.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No valid DeepLOB probabilities"), 420)
    df = df.sort_values("dominant_prob", ascending=False).head(max_points)
    color_map = {"Down": "#f87171", "Flat": "#fbbf24", "Up": "#22c55e"}
    hover_data = {
        "prob_up": ":.4f",
        "prob_flat": ":.4f",
        "prob_down": ":.4f",
        "dominant_prob": ":.4f",
        "signal_margin": ":.4f",
    }
    if "target_weight" in df.columns:
        hover_data["target_weight"] = ":.4f"
    if "engine_action" in df.columns:
        hover_data["engine_action"] = True
    fig = px.scatter_ternary(
        df,
        a="prob_up",
        b="prob_flat",
        c="prob_down",
        color="signal_label",
        color_discrete_map=color_map,
        hover_name="asset_id",
        hover_data=hover_data,
        title="DeepLOB Probability Simplex: Down / Flat / Up"
    )
    fig.update_traces(marker=dict(size=7, opacity=0.72, line=dict(width=0.4, color="rgba(248,250,252,.35)")))
    if focus_asset and focus_asset in set(df["asset_id"]):
        row = df[df["asset_id"] == focus_asset].iloc[0]
        fig.add_trace(go.Scatterternary(
            a=[row["prob_up"]], b=[row["prob_flat"]], c=[row["prob_down"]],
            mode="markers+text", text=[focus_asset], textposition="top center", name="Focus asset",
            marker=dict(size=16, color="#38bdf8", symbol="diamond", line=dict(width=2, color="#f8fafc"))
        ))
    fig.update_layout(
        ternary=dict(
            bgcolor="#0f172a",
            aaxis=dict(title="P(up)", gridcolor="rgba(148,163,184,.22)", linecolor="rgba(226,232,240,.28)", tickfont=dict(color="#cbd5e1")),
            baxis=dict(title="P(flat)", gridcolor="rgba(148,163,184,.22)", linecolor="rgba(226,232,240,.28)", tickfont=dict(color="#cbd5e1")),
            caxis=dict(title="P(down)", gridcolor="rgba(148,163,184,.22)", linecolor="rgba(226,232,240,.28)", tickfont=dict(color="#cbd5e1")),
        )
    )
    return apply_dark_layout(fig, 520, dict(l=12, r=12, t=48, b=12))


def chart_deeplob_focus_asset(sig_df: pd.DataFrame, asset: Optional[str]) -> go.Figure:
    if sig_df.empty or not asset or asset not in set(sig_df["asset_id"]):
        return apply_dark_layout(go.Figure().update_layout(title="Select an asset to inspect its three DeepLOB signals"), 300)
    row = sig_df[sig_df["asset_id"] == asset].iloc[0]
    labels = ["P(down)", "P(flat)", "P(up)"]
    vals = [row.get("prob_down", np.nan), row.get("prob_flat", np.nan), row.get("prob_up", np.nan)]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=["#f87171", "#fbbf24", "#22c55e"], text=[pct(v, 2) for v in vals], textposition="outside"))
    fig.update_layout(title=f"Focus Asset DeepLOB 3-Signal Vector — {asset}", xaxis_title="Signal class", yaxis_title="Probability")
    fig.update_yaxes(range=[0, 1])
    return apply_dark_layout(fig, 330)


def deeplob_signal_table_view(sig_df: pd.DataFrame, sort_by: str, top_n: int) -> pd.DataFrame:
    show = top_deeplob_assets(sig_df, sort_by, top_n)
    if show.empty:
        return show
    cols = [
        "asset_id", "signal_label", "prob_down", "prob_flat", "prob_up", "dominant_prob", "signal_margin",
        "signal_score", "confidence", "entry_candidate", "quality_candidate", "target_weight", "engine_action",
        "buy_shares", "sell_shares"
    ]
    out = show[[c for c in cols if c in show.columns]].copy()
    return out


def chart_order_books(state: EngineState) -> Tuple[go.Figure, go.Figure]:
    tr = state.trades.copy()
    if tr.empty:
        empty_fig = apply_dark_layout(go.Figure().update_layout(title="No executed trades for this day"), 320)
        return empty_fig, empty_fig
    buys = tr[tr["side"].astype(str).str.lower() == "buy"].sort_values("turnover", ascending=False).head(15)
    sells = tr[tr["side"].astype(str).str.lower() == "sell"].sort_values("turnover", ascending=False).head(15)
    fig_b = px.bar(buys.sort_values("turnover"), x="turnover", y="asset_id", orientation="h", title="Buy Execution Book", text="shares")
    fig_b.update_traces(marker_color="#22c55e", textposition="outside")
    fig_b.update_layout(xaxis_title="Turnover", yaxis_title="Asset")
    fig_s = px.bar(sells.sort_values("turnover"), x="turnover", y="asset_id", orientation="h", title="Sell Execution Book", text="shares")
    fig_s.update_traces(marker_color="#ef4444", textposition="outside")
    fig_s.update_layout(xaxis_title="Turnover", yaxis_title="Asset")
    return apply_dark_layout(fig_b, 360), apply_dark_layout(fig_s, 360)


def chart_weight_gap(state: EngineState) -> go.Figure:
    df = state.step.copy()
    if df.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No weight-step data"), 320)
    df["abs_gap"] = pd.to_numeric(df.get("weight_gap_before_trade", 0), errors="coerce").abs()
    show = df.sort_values("abs_gap", ascending=False).head(25).sort_values("abs_gap")
    fig = go.Figure()
    if not show.empty:
        fig.add_trace(go.Bar(x=show["pre_trade_weight"], y=show["asset_id"], orientation="h", name="Pre-trade", marker_color="#64748b"))
        fig.add_trace(go.Bar(x=show["target_weight"], y=show["asset_id"], orientation="h", name="Target", marker_color="#22d3ee"))
        fig.add_trace(go.Bar(x=show["post_trade_weight"], y=show["asset_id"], orientation="h", name="Post-trade", marker_color="#22c55e"))
    fig.update_layout(title="Largest Weight Transitions", xaxis_title="Portfolio weight", yaxis_title="Asset", barmode="group")
    return apply_dark_layout(fig, 500)


def chart_holdings_treemap(state: EngineState) -> go.Figure:
    df = state.holdings.copy()
    if df.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No holding snapshot"), 360)
    df = df.sort_values("market_value", ascending=False).head(80)
    fig = px.treemap(df, path=["asset_id"], values="market_value", color="eod_weight", color_continuous_scale="Viridis", title="EOD Portfolio Map")
    fig.update_traces(textinfo="label+value+percent entry")
    return apply_dark_layout(fig, 460, dict(l=10, r=10, t=42, b=10))




def chart_top_holdings_bar(state: EngineState, top_n: int = 18) -> go.Figure:
    df = state.holdings.copy()
    if df.empty or "market_value" not in df.columns:
        return apply_dark_layout(go.Figure().update_layout(title="No EOD holdings for this state"), 360)
    show = df.sort_values("market_value", ascending=False).head(top_n).sort_values("market_value")
    fig = go.Figure(go.Bar(
        x=show["market_value"],
        y=show["asset_id"],
        orientation="h",
        marker_color="#38bdf8",
        text=[pct(v, 2) for v in show.get("eod_weight", pd.Series(index=show.index, dtype=float))],
        textposition="outside",
        hovertemplate="Asset=%{y}<br>Market value=%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(title=f"Top {len(show)} EOD Holdings — {state.day}", xaxis_title="Market value", yaxis_title="Asset")
    return apply_dark_layout(fig, 420)


def chart_compact_weight_transition(state: EngineState, top_n: int = 18) -> go.Figure:
    df = state.step.copy()
    if df.empty or "target_weight" not in df.columns:
        return apply_dark_layout(go.Figure().update_layout(title="No target weight data for this state"), 340)
    df["transition_size"] = (pd.to_numeric(df.get("post_trade_weight", 0), errors="coerce") - pd.to_numeric(df.get("pre_trade_weight", 0), errors="coerce")).abs()
    show = df.sort_values(["transition_size", "target_weight"], ascending=False).head(top_n).sort_values("target_weight")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=show.get("target_weight", 0), y=show["asset_id"], orientation="h", name="Target", marker_color="#38bdf8"))
    if "post_trade_weight" in show.columns:
        fig.add_trace(go.Bar(x=show["post_trade_weight"], y=show["asset_id"], orientation="h", name="Post-trade", marker_color="#22c55e"))
    fig.update_layout(title=f"Compact Weight Transition — {state.day}", xaxis_title="Portfolio weight", yaxis_title="Asset", barmode="group")
    return apply_dark_layout(fig, 420)


def chart_compact_order_delta(state: EngineState, top_n: int = 18) -> go.Figure:
    df = build_order_diff(state)
    if df.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No order comparison data"), 360)
    show = df[df["order_overlap"] != "No active order"].head(top_n).sort_values("order_delta")
    if show.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No active orders for this replay day"), 360)
    colors = np.where(show["order_delta"] >= 0, "#22c55e", "#f87171")
    fig = go.Figure(go.Bar(x=show["order_delta"], y=show["asset_id"], orientation="h", marker_color=colors))
    fig.add_vline(x=0, line_color="rgba(226,232,240,.45)", line_width=1)
    fig.update_layout(title=f"Largest DLS vs Base Order Differences — {state.day}", xaxis_title="DLS net order − Base net order", yaxis_title="Asset")
    return apply_dark_layout(fig, 420)

def chart_heatmap_weights(tables: Dict[str, pd.DataFrame], summary: pd.DataFrame, idx: int, window: int = 22, top_n: int = 30) -> go.Figure:
    weights = tables.get("dls_weights_long", empty_df())
    if weights.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No long weights available"), 380)
    start = max(0, idx - window + 1)
    days = summary.iloc[start:idx+1]["trade_day_id"].tolist()
    df = weights[weights["trade_day_id"].isin(days)].copy()
    if df.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No weights in selected replay window"), 380)
    top_assets = df.groupby("asset_id")["weight"].max().nlargest(top_n).index.tolist()
    df = df[df["asset_id"].isin(top_assets)]
    pivot = df.pivot_table(index="asset_id", columns="trade_day_id", values="weight", aggfunc="sum").fillna(0)
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    pivot = pivot[[d for d in days if d in pivot.columns]]
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Viridis", colorbar=dict(title="Weight")))
    fig.update_layout(title=f"Weight Memory Window — last {len(days)} replay steps", xaxis_title="Trade day", yaxis_title="Asset")
    return apply_dark_layout(fig, 520)


def chart_asset_trace(tables: Dict[str, pd.DataFrame], summary: pd.DataFrame, asset: str, current_idx: int) -> go.Figure:
    step = tables.get("dls_weight_step", empty_df())
    hold = tables.get("dls_holding", empty_df())
    dls_orders = tables.get("dls_submission", empty_df())
    base_orders = tables.get("base_submission", empty_df())
    fig = go.Figure()

    if not step.empty and asset:
        s = step[step["asset_id"] == asset].copy().sort_values("day_index")
        if not s.empty:
            fig.add_trace(go.Scatter(x=s["day_index"], y=s["prob_up"], mode="lines", name="P(up)", line=dict(width=2, color="#60a5fa"), yaxis="y2"))
            fig.add_trace(go.Scatter(x=s["day_index"], y=s["confidence"], mode="lines", name="Confidence", line=dict(width=2, color="#a78bfa"), yaxis="y2"))
            fig.add_trace(go.Scatter(x=s["day_index"], y=s["target_weight"], mode="lines", name="Target weight", line=dict(width=3, color="#22d3ee")))
            fig.add_trace(go.Scatter(x=s["day_index"], y=s["post_trade_weight"], mode="lines", name="Post-trade weight", line=dict(width=2, color="#22c55e")))
    if not hold.empty and asset:
        h = hold[hold["asset_id"] == asset].copy().sort_values("day_index")
        if not h.empty:
            fig.add_trace(go.Scatter(x=h["day_index"], y=h["eod_weight"], mode="lines", name="EOD weight", line=dict(width=2, color="#f59e0b")))
    if not dls_orders.empty and asset:
        o = dls_orders[dls_orders["asset_id"] == asset].copy().sort_values("day_index")
        if not o.empty:
            fig.add_trace(go.Bar(x=o["day_index"], y=o["buy_percentage"], name="DLS buy order", marker_color="rgba(34,197,94,.45)"))
            fig.add_trace(go.Bar(x=o["day_index"], y=-o["sell_percentage"], name="DLS sell order", marker_color="rgba(239,68,68,.45)"))
    if not base_orders.empty and asset:
        b = base_orders[base_orders["asset_id"] == asset].copy().sort_values("day_index")
        if not b.empty:
            fig.add_trace(go.Scatter(x=b["day_index"], y=b["buy_percentage"], mode="markers", name="Base buy order", marker=dict(color="#fde68a", size=7, symbol="circle-open")))
            fig.add_trace(go.Scatter(x=b["day_index"], y=-b["sell_percentage"], mode="markers", name="Base sell order", marker=dict(color="#fecaca", size=7, symbol="x")))

    current_day_line(fig, int(summary.iloc[current_idx]["day_index"]))
    fig.update_layout(
        title=f"Asset State Trace — {asset}",
        xaxis_title="Trade day",
        yaxis=dict(title="Weight / order percentage"),
        yaxis2=dict(title="Signal score", overlaying="y", side="right", range=[0, 1]),
        barmode="relative",
    )
    return apply_dark_layout(fig, 430)


def chart_training(tables: Dict[str, pd.DataFrame]) -> go.Figure:
    hist = tables.get("training_history", empty_df())
    fig = go.Figure()
    if not hist.empty:
        x = hist["epoch"] if "epoch" in hist.columns else hist.index + 1
        for col, label, color in [("train_loss", "Train loss", "#60a5fa"), ("val_loss", "Val loss", "#f59e0b"), ("best_val_loss_to_date", "Best val loss", "#22c55e")]:
            if col in hist.columns:
                fig.add_trace(go.Scatter(x=x, y=hist[col], mode="lines", name=label, line=dict(width=2, color=color)))
    fig.update_layout(title="DLS Training Monitor", xaxis_title="Epoch", yaxis_title="Loss")
    return apply_dark_layout(fig, 330)


def chart_seed_search(tables: Dict[str, pd.DataFrame]) -> go.Figure:
    seed = tables.get("seed_search", empty_df())
    if seed.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No seed-search data"), 330)
    show = seed.copy().sort_values("score_proxy", ascending=False)
    fig = px.scatter(show, x="sharpe", y="total_return_%", size="avg_holdings", color="score_proxy", hover_name="seed", text="seed", title="Seed Search Map", color_continuous_scale="Viridis")
    fig.update_traces(textposition="top center")
    fig.update_layout(xaxis_title="Sharpe", yaxis_title="Total return (%)")
    return apply_dark_layout(fig, 390)


# -----------------------------------------------------------------------------
# Model comparison cockpit helpers
# -----------------------------------------------------------------------------
def get_model_pair(comp: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    if comp is None or comp.empty or "model" not in comp.columns:
        return None, None
    dls = comp[comp["model"].astype(str).str.contains("DLS", case=False, na=False)]
    base = comp[comp["model"].astype(str).str.contains("Original|DeepLOB-only|base", case=False, na=False, regex=True)]
    d = dls.iloc[0] if not dls.empty else None
    b = base.iloc[0] if not base.empty else None
    return d, b


def val(row: pd.Series | None, col: str, default=np.nan) -> float:
    if row is None:
        return default
    try:
        return float(row.get(col, default))
    except Exception:
        return default


def delta_class(delta: float, metric: str) -> str:
    # For costs, turnover, and drawdown magnitude, lower is usually better.
    lower_is_better = metric in {"total_costs_RMB", "avg_turnover_RMB"}
    if not np.isfinite(delta):
        return "delta-neutral"
    if lower_is_better:
        return "delta-good" if delta < 0 else "delta-bad" if delta > 0 else "delta-neutral"
    return "delta-good" if delta > 0 else "delta-bad" if delta < 0 else "delta-neutral"


def comparison_card_html(title: str, dls_value: str, base_value: str, delta_value: str, cls: str, note: str = "") -> str:
    return f"""
    <div class="model-card dls">
      <div class="model-name">{title}</div>
      <div class="model-kpi">{dls_value}</div>
      <div class="model-sub">Base: {base_value} · <span class="{cls}">Δ {delta_value}</span></div>
      <div class="model-sub">{note}</div>
    </div>
    """


def chart_comparison_delta(comp: pd.DataFrame) -> go.Figure:
    d, b = get_model_pair(comp)
    if d is None or b is None:
        return apply_dark_layout(go.Figure().update_layout(title="Comparison table is not available"), 360)
    specs = [
        ("total_return_%", "Return Δ (pp)", "higher"),
        ("cagr_%", "CAGR Δ (pp)", "higher"),
        ("sharpe", "Sharpe Δ", "higher"),
        ("max_drawdown_%", "Max DD Δ (pp)", "higher"),
        ("score_proxy", "Score Δ", "higher"),
        ("win_rate_%", "Win-rate Δ (pp)", "higher"),
        ("avg_holdings", "Avg holdings Δ", "lower"),
        ("total_costs_RMB", "Costs Δ (RMB)", "lower"),
    ]
    rows=[]
    colors=[]
    for col,label,direction in specs:
        dv=val(d,col); bv=val(b,col); delta=dv-bv
        rows.append((label,delta))
        good = (delta>=0) if direction=="higher" else (delta<=0)
        colors.append("#22c55e" if good else "#f87171")
    df=pd.DataFrame(rows,columns=["metric","delta"])
    fig=go.Figure(go.Bar(x=df["delta"], y=df["metric"], orientation="h", marker_color=colors, text=[num(x,2) for x in df["delta"]], textposition="outside"))
    fig.add_vline(x=0, line_color="rgba(226,232,240,.45)", line_width=1)
    fig.update_layout(title="DeepLOB + DLS Impact vs Base DeepLOB", xaxis_title="DLS minus Base", yaxis_title="")
    return apply_dark_layout(fig, 460, dict(l=120, r=35, t=48, b=34))


def chart_metric_radar(comp: pd.DataFrame) -> go.Figure:
    d, b = get_model_pair(comp)
    if d is None or b is None:
        return apply_dark_layout(go.Figure().update_layout(title="Comparison table is not available"), 360)
    # Normalize each final metric into a 0-1 score where higher is better.
    specs = [
        ("total_return_%", "Return", True),
        ("cagr_%", "CAGR", True),
        ("sharpe", "Sharpe", True),
        ("max_drawdown_%", "Drawdown", True),  # values are negative; higher means less severe
        ("score_proxy", "Score", True),
        ("win_rate_%", "Win-rate", True),
        ("avg_holdings", "Concentration", False),
        ("total_costs_RMB", "Cost efficiency", False),
    ]
    labels=[]; d_scores=[]; b_scores=[]
    for col,label,higher_better in specs:
        dv=val(d,col); bv=val(b,col)
        if not np.isfinite(dv) or not np.isfinite(bv):
            continue
        lo=min(dv,bv); hi=max(dv,bv)
        if abs(hi-lo)<1e-12:
            ds=bs=0.5
        else:
            ds=(dv-lo)/(hi-lo); bs=(bv-lo)/(hi-lo)
            if not higher_better:
                ds=1-ds; bs=1-bs
        labels.append(label); d_scores.append(ds); b_scores.append(bs)
    if labels:
        labels_closed=labels+[labels[0]]; d_closed=d_scores+[d_scores[0]]; b_closed=b_scores+[b_scores[0]]
    else:
        labels_closed=[]; d_closed=[]; b_closed=[]
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=d_closed, theta=labels_closed, fill="toself", name="DeepLOB + DLS", line=dict(color="#38bdf8", width=3), fillcolor="rgba(56,189,248,.18)"))
    fig.add_trace(go.Scatterpolar(r=b_closed, theta=labels_closed, fill="toself", name="Base DeepLOB", line=dict(color="#fbbf24", width=2), fillcolor="rgba(251,191,36,.13)"))
    fig.update_layout(title="Normalized Performance Profile", polar=dict(bgcolor="#0f172a", radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(148,163,184,.18)", tickfont=dict(color="#cbd5e1")), angularaxis=dict(gridcolor="rgba(148,163,184,.18)", tickfont=dict(color="#e5e7eb"))))
    return apply_dark_layout(fig, 460, dict(l=35, r=35, t=48, b=24))


def chart_drawdown_comparison(summary: pd.DataFrame, current_idx: int) -> go.Figure:
    fig=go.Figure()
    if "dls_equity_proxy" in summary.columns and summary["dls_equity_proxy"].notna().any():
        d_curve=summary["dls_equity_proxy"].astype(float)
        d_dd=(d_curve/d_curve.cummax()-1)*100
        fig.add_trace(go.Scatter(x=summary["day_index"], y=d_dd, mode="lines", name="DeepLOB + DLS drawdown", fill="tozeroy", line=dict(color="#38bdf8", width=2)))
    if "base_portfolio_value" in summary.columns and summary["base_portfolio_value"].notna().any():
        b_curve=summary["base_portfolio_value"].astype(float)
        b_dd=(b_curve/b_curve.cummax()-1)*100
        fig.add_trace(go.Scatter(x=summary["day_index"], y=b_dd, mode="lines", name="Base DeepLOB drawdown", fill="tozeroy", line=dict(color="#fbbf24", width=2)))
    current_day_line(fig, int(summary.iloc[current_idx]["day_index"]))
    fig.update_layout(title="Drawdown Replay: Risk Path Comparison", xaxis_title="Trade day", yaxis_title="Drawdown (%)")
    return apply_dark_layout(fig, 360)


def chart_cumulative_cost_turnover(summary: pd.DataFrame, current_idx: int) -> go.Figure:
    fig=go.Figure()
    x=summary["day_index"]
    if "executed_total_cost" in summary.columns:
        fig.add_trace(go.Scatter(x=x, y=summary["executed_total_cost"].fillna(0).cumsum(), mode="lines", name="DLS cumulative cost", line=dict(color="#38bdf8", width=3)))
    if "base_total_costs" in summary.columns:
        fig.add_trace(go.Scatter(x=x, y=summary["base_total_costs"].fillna(0).cumsum(), mode="lines", name="Base cumulative cost", line=dict(color="#fbbf24", width=2)))
    current_day_line(fig, int(summary.iloc[current_idx]["day_index"]))
    fig.update_layout(title="Cumulative Trading Cost", xaxis_title="Trade day", yaxis_title="Cost")
    return apply_dark_layout(fig, 340)


def chart_turnover_comparison(summary: pd.DataFrame, current_idx: int) -> go.Figure:
    fig=go.Figure()
    x=summary["day_index"]
    if "executed_total_turnover" in summary.columns:
        fig.add_trace(go.Scatter(x=x, y=summary["executed_total_turnover"].fillna(0), mode="lines", name="DLS turnover", line=dict(color="#38bdf8", width=2)))
    if "base_buy_turnover" in summary.columns and "base_sell_turnover" in summary.columns:
        base_turn=summary["base_buy_turnover"].fillna(0)+summary["base_sell_turnover"].fillna(0)
        fig.add_trace(go.Scatter(x=x, y=base_turn, mode="lines", name="Base turnover", line=dict(color="#fbbf24", width=2)))
    current_day_line(fig, int(summary.iloc[current_idx]["day_index"]))
    fig.update_layout(title="Daily Trading Intensity", xaxis_title="Trade day", yaxis_title="Turnover")
    return apply_dark_layout(fig, 340)


def build_order_diff(state: EngineState) -> pd.DataFrame:
    dls=state.dls_orders.copy()
    base=state.base_orders.copy()
    cols=["asset_id","buy_percentage","sell_percentage"]
    if dls.empty:
        dls=pd.DataFrame(columns=cols)
    if base.empty:
        base=pd.DataFrame(columns=cols)
    dls=dls[[c for c in cols if c in dls.columns]].rename(columns={"buy_percentage":"dls_buy","sell_percentage":"dls_sell"})
    base=base[[c for c in cols if c in base.columns]].rename(columns={"buy_percentage":"base_buy","sell_percentage":"base_sell"})
    out=pd.merge(dls,base,on="asset_id",how="outer").fillna(0)
    for c in ["dls_buy","dls_sell","base_buy","base_sell"]:
        if c not in out.columns:
            out[c]=0.0
        out[c]=pd.to_numeric(out[c],errors="coerce").fillna(0)
    out["dls_net_order"] = out["dls_buy"] - out["dls_sell"]
    out["base_net_order"] = out["base_buy"] - out["base_sell"]
    out["order_delta"] = out["dls_net_order"] - out["base_net_order"]
    out["abs_order_delta"] = out["order_delta"].abs()
    eps=1e-12
    out["order_overlap"] = np.select(
        [
            (out["dls_net_order"].abs()>eps) & (out["base_net_order"].abs()>eps),
            (out["dls_net_order"].abs()>eps) & (out["base_net_order"].abs()<=eps),
            (out["dls_net_order"].abs()<=eps) & (out["base_net_order"].abs()>eps),
        ],
        ["Both models", "DLS only", "Base only"],
        default="No active order"
    )
    out["direction_case"] = np.select(
        [
            (out["dls_net_order"]>eps) & (out["base_net_order"]>eps),
            (out["dls_net_order"]<-eps) & (out["base_net_order"]<-eps),
            (out["dls_net_order"]*out["base_net_order"]<0),
        ],
        ["Both buy", "Both sell", "Opposite direction"],
        default="Single-side / flat"
    )
    return out.sort_values("abs_order_delta", ascending=False)


def chart_order_divergence(state: EngineState) -> go.Figure:
    df=build_order_diff(state)
    if df.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No orders available for this replay state"), 360)
    show=df[df["order_overlap"]!="No active order"].copy()
    if show.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No active orders for this replay state"), 360)
    color_map={"Both models":"#38bdf8", "DLS only":"#22c55e", "Base only":"#fbbf24", "No active order":"#64748b"}
    fig=px.scatter(show, x="base_net_order", y="dls_net_order", size="abs_order_delta", color="order_overlap", color_discrete_map=color_map, hover_name="asset_id", hover_data=["direction_case","dls_buy","dls_sell","base_buy","base_sell","order_delta"], title=f"Daily Order Divergence — {state.day}")
    lim=max(show["base_net_order"].abs().max(), show["dls_net_order"].abs().max(), 0.001)*1.15
    fig.add_hline(y=0, line_color="rgba(226,232,240,.45)", line_width=1)
    fig.add_vline(x=0, line_color="rgba(226,232,240,.45)", line_width=1)
    fig.add_shape(type="line", x0=-lim, x1=lim, y0=-lim, y1=lim, line=dict(color="rgba(56,189,248,.32)", dash="dash"))
    fig.update_layout(xaxis_title="Base DeepLOB net order", yaxis_title="DeepLOB + DLS net order")
    fig.update_xaxes(range=[-lim,lim]); fig.update_yaxes(range=[-lim,lim])
    return apply_dark_layout(fig, 480)


def chart_order_overlap_counts(state: EngineState) -> go.Figure:
    df=build_order_diff(state)
    if df.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No order overlap data"), 310)
    active=df[df["order_overlap"]!="No active order"]
    counts=active["order_overlap"].value_counts().reindex(["Both models","DLS only","Base only"]).fillna(0).astype(int)
    fig=go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=["#38bdf8","#22c55e","#fbbf24"], text=counts.values, textposition="outside"))
    fig.update_layout(title=f"Order Overlap Counts — {state.day}", xaxis_title="Overlap bucket", yaxis_title="Assets")
    return apply_dark_layout(fig, 320)


def chart_holdings_comparison(summary: pd.DataFrame, current_idx: int) -> go.Figure:
    fig=go.Figure()
    x=summary["day_index"]
    if "eod_holdings" in summary.columns:
        fig.add_trace(go.Scatter(x=x, y=summary["eod_holdings"], mode="lines", name="DLS EOD holdings", line=dict(color="#38bdf8", width=2)))
    if "base_num_holdings" in summary.columns:
        fig.add_trace(go.Scatter(x=x, y=summary["base_num_holdings"], mode="lines", name="Base holdings", line=dict(color="#fbbf24", width=2)))
    current_day_line(fig, int(summary.iloc[current_idx]["day_index"]))
    fig.update_layout(title="Portfolio Breadth Through Time", xaxis_title="Trade day", yaxis_title="Number of holdings")
    return apply_dark_layout(fig, 340)


def comparison_table_view(comp: pd.DataFrame) -> pd.DataFrame:
    if comp.empty:
        return comp
    keep=["model","total_return_%","cagr_%","sharpe","max_drawdown_%","score_proxy","total_costs_RMB","avg_turnover_RMB","win_rate_%","avg_holdings"]
    return comp[[c for c in keep if c in comp.columns]].copy()

def daily_metric_rows(summary: pd.DataFrame, idx: int) -> pd.DataFrame:
    """Return a readable same-day metric comparison table.

    This table is intentionally string-formatted because it is used as an
    explanatory panel in the UI rather than as a downstream numeric dataset.
    """
    if summary.empty:
        return empty_df()
    r = summary.iloc[idx]

    def raw(name: str) -> float:
        try:
            return float(r.get(name, np.nan))
        except Exception:
            return np.nan

    rows = [
        {
            "Metric": "Daily return",
            "DeepLOB + DLS": pct(raw("dls_daily_return_%"), 3, already_percent=True),
            "Base DeepLOB": pct(raw("base_daily_return_%"), 3, already_percent=True),
            "Delta": f"{raw('daily_return_delta_pp'):.3f} pp" if np.isfinite(raw("daily_return_delta_pp")) else "—",
            "Better if": "Higher",
        },
        {
            "Metric": "Cumulative return",
            "DeepLOB + DLS": pct(raw("dls_return_proxy_%"), 2, already_percent=True),
            "Base DeepLOB": pct(raw("base_return_%"), 2, already_percent=True),
            "Delta": f"{raw('cum_return_delta_pp'):.2f} pp" if np.isfinite(raw("cum_return_delta_pp")) else "—",
            "Better if": "Higher",
        },
        {
            "Metric": "Portfolio value / equity proxy",
            "DeepLOB + DLS": money(raw("dls_equity_proxy")),
            "Base DeepLOB": money(raw("base_portfolio_value")),
            "Delta": money(raw("dls_equity_proxy") - raw("base_portfolio_value")) if np.isfinite(raw("dls_equity_proxy") - raw("base_portfolio_value")) else "—",
            "Better if": "Higher",
        },
        {
            "Metric": "Daily turnover",
            "DeepLOB + DLS": money(raw("executed_total_turnover")),
            "Base DeepLOB": money(raw("base_total_turnover_day")),
            "Delta": money(raw("daily_turnover_delta")),
            "Better if": "Context dependent",
        },
        {
            "Metric": "Cumulative turnover",
            "DeepLOB + DLS": money(raw("dls_cum_turnover")),
            "Base DeepLOB": money(raw("base_cum_turnover")),
            "Delta": money(raw("dls_cum_turnover") - raw("base_cum_turnover")) if np.isfinite(raw("dls_cum_turnover") - raw("base_cum_turnover")) else "—",
            "Better if": "Usually lower",
        },
        {
            "Metric": "Daily trading cost",
            "DeepLOB + DLS": money(raw("executed_total_cost")),
            "Base DeepLOB": money(raw("base_total_costs")),
            "Delta": money(raw("daily_cost_delta")),
            "Better if": "Lower",
        },
        {
            "Metric": "Cumulative trading cost",
            "DeepLOB + DLS": money(raw("dls_cum_cost")),
            "Base DeepLOB": money(raw("base_cum_cost")),
            "Delta": money(raw("dls_cum_cost") - raw("base_cum_cost")) if np.isfinite(raw("dls_cum_cost") - raw("base_cum_cost")) else "—",
            "Better if": "Lower",
        },
        {
            "Metric": "Holdings",
            "DeepLOB + DLS": num(raw("eod_holdings"), 0),
            "Base DeepLOB": num(raw("base_num_holdings"), 0),
            "Delta": num(raw("holdings_delta"), 0),
            "Better if": "Strategy dependent",
        },
        {
            "Metric": "Buy orders / executions",
            "DeepLOB + DLS": num(raw("executed_buys"), 0),
            "Base DeepLOB": num(raw("base_num_buys"), 0),
            "Delta": num(raw("executed_buys") - raw("base_num_buys"), 0) if np.isfinite(raw("executed_buys") - raw("base_num_buys")) else "—",
            "Better if": "Context dependent",
        },
        {
            "Metric": "Sell orders / executions",
            "DeepLOB + DLS": num(raw("executed_sells"), 0),
            "Base DeepLOB": num(raw("base_num_sells"), 0),
            "Delta": num(raw("executed_sells") - raw("base_num_sells"), 0) if np.isfinite(raw("executed_sells") - raw("base_num_sells")) else "—",
            "Better if": "Context dependent",
        },
    ]
    return pd.DataFrame(rows)


def chart_daily_return_window(summary: pd.DataFrame, current_idx: int, window: int = 24) -> go.Figure:
    if summary.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No daily return data"), 320)
    start = max(0, current_idx - window + 1)
    end = min(len(summary), current_idx + window // 3 + 1)
    df = summary.iloc[start:end].copy()
    fig = go.Figure()
    if "dls_daily_return_%" in df.columns:
        fig.add_trace(go.Scatter(x=df["day_index"], y=df["dls_daily_return_%"], mode="lines+markers", name="DeepLOB + DLS", line=dict(width=3, color="#22d3ee")))
    if "base_daily_return_%" in df.columns:
        fig.add_trace(go.Scatter(x=df["day_index"], y=df["base_daily_return_%"], mode="lines+markers", name="Base DeepLOB", line=dict(width=2, color="#fbbf24")))
    current_day_line(fig, int(summary.iloc[current_idx]["day_index"]))
    fig.add_hline(y=0, line_color="rgba(226,232,240,.36)", line_width=1)
    fig.update_layout(title="Daily Return Around Current Replay Day", xaxis_title="Trade day", yaxis_title="Daily return (%)")
    return apply_dark_layout(fig, 350)


def chart_daily_metric_delta(summary: pd.DataFrame, current_idx: int) -> go.Figure:
    if summary.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No daily metric delta data"), 320)
    r = summary.iloc[current_idx]
    data = pd.DataFrame([
        {"metric": "Daily return delta (pp)", "delta": r.get("daily_return_delta_pp", np.nan), "better": "higher"},
        {"metric": "Cumulative return delta (pp)", "delta": r.get("cum_return_delta_pp", np.nan), "better": "higher"},
        {"metric": "Holdings delta", "delta": r.get("holdings_delta", np.nan), "better": "neutral"},
        {"metric": "Buy count delta", "delta": r.get("executed_buys", np.nan) - r.get("base_num_buys", np.nan), "better": "neutral"},
        {"metric": "Sell count delta", "delta": r.get("executed_sells", np.nan) - r.get("base_num_sells", np.nan), "better": "neutral"},
    ]).dropna(subset=["delta"])
    if data.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No comparable daily deltas for this day"), 320)
    colors = np.where(data["delta"] >= 0, "#22c55e", "#f87171")
    fig = go.Figure(go.Bar(x=data["delta"], y=data["metric"], orientation="h", marker_color=colors, text=[num(v, 3) for v in data["delta"]], textposition="outside"))
    fig.add_vline(x=0, line_color="rgba(226,232,240,.45)", line_width=1)
    fig.update_layout(title=f"Current-Day Model Delta — {r.get('trade_day_id', '—')}", xaxis_title="DeepLOB + DLS minus Base DeepLOB", yaxis_title="")
    return apply_dark_layout(fig, 360)


def chart_daily_trading_intensity(summary: pd.DataFrame, current_idx: int) -> go.Figure:
    if summary.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No daily intensity data"), 320)
    r = summary.iloc[current_idx]
    data = pd.DataFrame([
        {"metric": "Buys", "DeepLOB + DLS": r.get("executed_buys", np.nan), "Base DeepLOB": r.get("base_num_buys", np.nan)},
        {"metric": "Sells", "DeepLOB + DLS": r.get("executed_sells", np.nan), "Base DeepLOB": r.get("base_num_sells", np.nan)},
        {"metric": "Holdings", "DeepLOB + DLS": r.get("eod_holdings", np.nan), "Base DeepLOB": r.get("base_num_holdings", np.nan)},
    ])
    long = data.melt(id_vars="metric", var_name="Model", value_name="Count").dropna(subset=["Count"])
    if long.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No daily activity counts available"), 320)
    fig = px.bar(long, x="metric", y="Count", color="Model", barmode="group", text="Count", title=f"Daily Trading Activity — {r.get('trade_day_id', '—')}", color_discrete_map={"DeepLOB + DLS": "#22d3ee", "Base DeepLOB": "#fbbf24"})
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Metric", yaxis_title="Count")
    return apply_dark_layout(fig, 350)


def chart_daily_money_comparison(summary: pd.DataFrame, current_idx: int) -> go.Figure:
    if summary.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No money metric data"), 320)
    r = summary.iloc[current_idx]
    # Use two separate units in one chart by converting turnover to millions and cost to thousands.
    data = pd.DataFrame([
        {"metric": "Turnover (¥M)", "DeepLOB + DLS": r.get("executed_total_turnover", np.nan) / 1e6, "Base DeepLOB": r.get("base_total_turnover_day", np.nan) / 1e6},
        {"metric": "Cost (¥K)", "DeepLOB + DLS": r.get("executed_total_cost", np.nan) / 1e3, "Base DeepLOB": r.get("base_total_costs", np.nan) / 1e3},
    ])
    long = data.melt(id_vars="metric", var_name="Model", value_name="Value").dropna(subset=["Value"])
    if long.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No daily money metrics available"), 320)
    fig = px.bar(long, x="metric", y="Value", color="Model", barmode="group", text="Value", title=f"Daily Money Metrics — {r.get('trade_day_id', '—')}", color_discrete_map={"DeepLOB + DLS": "#22d3ee", "Base DeepLOB": "#fbbf24"})
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(xaxis_title="Metric", yaxis_title="Scaled value")
    return apply_dark_layout(fig, 350)

# -----------------------------------------------------------------------------
# Tables and decision book
# -----------------------------------------------------------------------------
def build_decision_book(state: EngineState) -> pd.DataFrame:
    """Build the asset-level decision book used by Decision Inspector and Execution Tape.

    This table keeps the raw internal column names so charts and filters can still
    use them. Display-only functions rename the columns into human-readable labels.
    The important v11 addition is that the book now carries the complete DeepLOB
    probability vector, the full weight transition, submitted order percentages,
    and filled execution percentages in the same asset row.
    """
    step = state.step.copy()
    if step.empty:
        return empty_df()

    cols = [
        "asset_id", "prob_down", "prob_flat", "prob_up", "signal_score", "confidence",
        "entry_candidate", "quality_candidate", "target_weight", "pre_trade_weight",
        "post_trade_weight", "weight_gap_before_trade", "weight_gap_after_trade",
        "buy_shares", "sell_shares", "buy_turnover", "sell_turnover", "buy_cost", "sell_cost",
    ]
    book = step[[c for c in cols if c in step.columns]].copy()

    # Final end-of-day state from the DLS holding snapshot.
    if not state.holdings.empty:
        hcols = [c for c in ["asset_id", "shares", "close_price", "market_value", "eod_weight"] if c in state.holdings.columns]
        h = state.holdings[hcols].rename(columns={"shares": "eod_shares"})
        book = book.merge(h, on="asset_id", how="left")

    # Filled execution details from the DLS trade audit.
    # order_percentage is the filled percentage, which is different from the submitted order percentage.
    if not state.trades.empty and "asset_id" in state.trades.columns:
        tr = state.trades.copy()
        if "side" not in tr.columns:
            tr["side"] = ""
        for c in ["shares", "turnover", "cost", "order_percentage", "execution_price"]:
            if c not in tr.columns:
                tr[c] = np.nan
            tr[c] = pd.to_numeric(tr[c], errors="coerce")
        side = tr["side"].astype(str).str.lower()
        buy = tr[side == "buy"].groupby("asset_id").agg(
            filled_buy_shares=("shares", "sum"),
            filled_buy_pct=("order_percentage", "sum"),
            filled_buy_turnover=("turnover", "sum"),
            filled_buy_cost=("cost", "sum"),
            buy_execution_price=("execution_price", "mean"),
        ).reset_index()
        sell = tr[side == "sell"].groupby("asset_id").agg(
            filled_sell_shares=("shares", "sum"),
            filled_sell_pct=("order_percentage", "sum"),
            filled_sell_turnover=("turnover", "sum"),
            filled_sell_cost=("cost", "sum"),
            sell_execution_price=("execution_price", "mean"),
        ).reset_index()
        book = book.merge(buy, on="asset_id", how="left").merge(sell, on="asset_id", how="left")

    # Submitted DLS orders.
    if not state.dls_orders.empty:
        o = state.dls_orders[[c for c in ["asset_id", "buy_percentage", "sell_percentage"] if c in state.dls_orders.columns]].rename(columns={"buy_percentage": "dls_order_buy_pct", "sell_percentage": "dls_order_sell_pct"})
        book = book.merge(o, on="asset_id", how="left")

    # Submitted Base DeepLOB orders.
    if not state.base_orders.empty:
        b = state.base_orders[[c for c in ["asset_id", "buy_percentage", "sell_percentage"] if c in state.base_orders.columns]].rename(columns={"buy_percentage": "base_order_buy_pct", "sell_percentage": "base_order_sell_pct"})
        book = book.merge(b, on="asset_id", how="left")

    numeric_cols = [
        "prob_down", "prob_flat", "prob_up", "signal_score", "confidence",
        "target_weight", "pre_trade_weight", "post_trade_weight", "eod_weight",
        "buy_shares", "sell_shares", "buy_turnover", "sell_turnover", "buy_cost", "sell_cost",
        "filled_buy_shares", "filled_sell_shares", "filled_buy_pct", "filled_sell_pct",
        "filled_buy_turnover", "filled_sell_turnover", "filled_buy_cost", "filled_sell_cost",
        "dls_order_buy_pct", "dls_order_sell_pct", "base_order_buy_pct", "base_order_sell_pct",
    ]
    for c in numeric_cols:
        if c in book.columns:
            book[c] = pd.to_numeric(book[c], errors="coerce")

    # Fill execution/order quantities with zero where the asset has no activity.
    zero_cols = [
        "buy_turnover", "sell_turnover", "buy_cost", "sell_cost",
        "filled_buy_shares", "filled_sell_shares", "filled_buy_pct", "filled_sell_pct",
        "filled_buy_turnover", "filled_sell_turnover", "filled_buy_cost", "filled_sell_cost",
        "dls_order_buy_pct", "dls_order_sell_pct", "base_order_buy_pct", "base_order_sell_pct",
    ]
    for c in zero_cols:
        if c in book.columns:
            book[c] = book[c].fillna(0)

    # Prefer the trade-audit filled turnover when it exists. Fall back to the step-audit turnover.
    book["filled_total_turnover"] = book.get("filled_buy_turnover", 0) + book.get("filled_sell_turnover", 0)
    step_turnover = book.get("buy_turnover", 0) + book.get("sell_turnover", 0)
    book["engine_turnover"] = book["filled_total_turnover"].where(book["filled_total_turnover"].abs() > 0, step_turnover)

    book["engine_action"] = "HOLD"
    buy_signal = (book.get("filled_buy_turnover", 0) > 0) | (book.get("buy_turnover", 0) > 0) | (book.get("dls_order_buy_pct", 0) > 0)
    sell_signal = (book.get("filled_sell_turnover", 0) > 0) | (book.get("sell_turnover", 0) > 0) | (book.get("dls_order_sell_pct", 0) > 0)
    book.loc[buy_signal, "engine_action"] = "BUY"
    book.loc[sell_signal, "engine_action"] = "SELL"

    book["abs_target_weight"] = pd.to_numeric(book.get("target_weight", 0), errors="coerce").abs()
    sort_cols = [c for c in ["engine_turnover", "abs_target_weight", "confidence"] if c in book.columns]
    if sort_cols:
        book = book.sort_values(sort_cols, ascending=False)
    return book


def display_table(df: pd.DataFrame, height: int = 390):
    if df.empty:
        st.info("No rows available for the current replay state.")
    else:
        st.dataframe(df, use_container_width=True, height=height, hide_index=True)


def compact_book_view(book: pd.DataFrame, max_rows: int = 60) -> pd.DataFrame:
    """Display-ready Decision Inspector table with explicit signal, weight and order columns."""
    if book.empty:
        return book
    cols = [
        "asset_id", "engine_action",
        "prob_down", "prob_flat", "prob_up", "signal_score", "confidence",
        "entry_candidate", "quality_candidate",
        "pre_trade_weight", "target_weight", "post_trade_weight", "eod_weight",
        "dls_order_buy_pct", "dls_order_sell_pct", "filled_buy_pct", "filled_sell_pct",
        "filled_buy_shares", "filled_sell_shares", "engine_turnover",
        "base_order_buy_pct", "base_order_sell_pct",
    ]
    out = book[[c for c in cols if c in book.columns]].copy().head(max_rows)
    numeric_cols = [c for c in out.columns if c not in ["asset_id", "engine_action", "entry_candidate", "quality_candidate"]]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    rename = {
        "asset_id": "Asset",
        "engine_action": "Engine action",
        "prob_down": "DeepLOB P(down)",
        "prob_flat": "DeepLOB P(flat)",
        "prob_up": "DeepLOB P(up)",
        "signal_score": "Signal score",
        "confidence": "Confidence",
        "entry_candidate": "Entry passed",
        "quality_candidate": "Quality passed",
        "pre_trade_weight": "Previous weight",
        "target_weight": "DLS predicted target weight",
        "post_trade_weight": "Post-trade weight",
        "eod_weight": "Final EOD weight",
        "dls_order_buy_pct": "DLS submitted buy %",
        "dls_order_sell_pct": "DLS submitted sell %",
        "filled_buy_pct": "Filled buy %",
        "filled_sell_pct": "Filled sell %",
        "filled_buy_shares": "Filled buy shares",
        "filled_sell_shares": "Filled sell shares",
        "engine_turnover": "Executed turnover",
        "base_order_buy_pct": "Base submitted buy %",
        "base_order_sell_pct": "Base submitted sell %",
    }
    return out.rename(columns=rename)


def explain_asset(book: pd.DataFrame, asset: str) -> str:
    if book.empty or not asset:
        return "No asset selected."
    row_df = book[book["asset_id"] == asset]
    if row_df.empty:
        return f"Asset {asset} is not present in the current DLS decision state."
    r = row_df.iloc[0]
    action = str(r.get("engine_action", "HOLD"))
    icon = "BUY" if action == "BUY" else "SELL" if action == "SELL" else "HOLD"
    lines = [f"[{icon}] Asset {asset}"]
    lines.append(
        "Signals: "
        f"P(down)={num(r.get('prob_down'), 4)}, "
        f"P(flat)={num(r.get('prob_flat'), 4)}, "
        f"P(up)={num(r.get('prob_up'), 4)}, "
        f"signal_score={num(r.get('signal_score'), 4)}, confidence={num(r.get('confidence'), 4)}"
    )
    lines.append(f"Filter: entry={r.get('entry_candidate', '—')}, quality={r.get('quality_candidate', '—')}")
    lines.append(
        "Weights: "
        f"previous={pct(r.get('pre_trade_weight'), 3)}, "
        f"DLS target={pct(r.get('target_weight'), 3)}, "
        f"post-trade={pct(r.get('post_trade_weight'), 3)}, "
        f"final EOD={pct(r.get('eod_weight'), 3)}"
    )
    lines.append(
        "Submitted DLS order: "
        f"buy={pct(r.get('dls_order_buy_pct'), 3)}, sell={pct(r.get('dls_order_sell_pct'), 3)}"
    )
    lines.append(
        "Filled execution: "
        f"buy={pct(r.get('filled_buy_pct'), 3)} / {num(r.get('filled_buy_shares'), 0)} shares, "
        f"sell={pct(r.get('filled_sell_pct'), 3)} / {num(r.get('filled_sell_shares'), 0)} shares"
    )
    if pd.notna(r.get("base_order_buy_pct", np.nan)) or pd.notna(r.get("base_order_sell_pct", np.nan)):
        lines.append(f"Base submitted order: buy={pct(r.get('base_order_buy_pct'), 3)}, sell={pct(r.get('base_order_sell_pct'), 3)}")
    return "\n".join(lines)


def event_log_text(state: EngineState) -> str:
    r = state.row
    signal_day = r.get("signal_day_id", "—")
    lines = []
    lines.append(f"ENGINE CLOCK  trade_day={state.day} | signal_day={signal_day} | replay_step={state.index + 1}")
    lines.append(f"SIGNAL BUS    raw_assets={int(r.get('raw_assets', 0))} | buy_signals={int(r.get('raw_buy_signals', 0))} | avg_P_up={num(r.get('avg_prob_up'), 4)} | max_P_up={num(r.get('max_prob_up'), 4)}")
    lines.append(f"FILTER        entry={int(r.get('entry_candidates', 0))} | quality={int(r.get('quality_candidates', 0))} | pass_rate={pct(r.get('quality_pass_rate'), 2)} | mask={r.get('mask_type', '—')}")
    lines.append(f"OPTIMIZER     target_names={int(r.get('target_names', 0))} | target_gross={pct(r.get('engine_target_gross'), 2)} | max_weight={pct(r.get('max_target_weight', r.get('max_weight')), 2)}")
    lines.append(f"EXECUTION     buys={int(r.get('executed_buys', 0))} | sells={int(r.get('executed_sells', 0))} | turnover={money(r.get('executed_total_turnover'))} | cost={money(r.get('executed_total_cost'))}")
    lines.append(f"PORTFOLIO     holdings={int(r.get('eod_holdings', 0))} | equity_proxy={money(r.get('dls_equity_proxy'))} | cash_proxy={money(r.get('dls_cash_proxy'))} | top10_weight={pct(r.get('top10_eod_weight'), 2)}")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# App bootstrap
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Explained minimal UI helpers added in v9
# -----------------------------------------------------------------------------
def section_intro(title: str, body: str):
    """Small English explanation block shown at the start of each engine view."""
    st.markdown(
        f"""
        <div class="explain-box">
          <div class="why">How to read this section</div>
          <p><b>{title}</b> — {body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_signal_to_weight_scatter(state: EngineState) -> go.Figure:
    """Useful diagnostic: whether high-probability signals became real target weights."""
    book = build_decision_book(state)
    if book.empty:
        return apply_dark_layout(go.Figure().update_layout(title="No decision-book data for this replay state"), 360)
    df = book.copy()
    for c in ["prob_up", "target_weight", "confidence", "engine_turnover"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["engine_turnover"] = df["engine_turnover"].fillna(0)
    df["plot_size"] = df["engine_turnover"].abs()
    if df["plot_size"].max() <= 0 or not np.isfinite(df["plot_size"].max()):
        df["plot_size"] = 1.0
    # Keep the most informative points: traded/targeted assets plus high-confidence non-trades.
    df["importance"] = df["engine_turnover"].abs() + df["target_weight"].abs() * 1e6 + df["confidence"].fillna(0)
    show = df.sort_values("importance", ascending=False).head(350)
    color_map = {"BUY": "#22c55e", "SELL": "#f87171", "HOLD": "#60a5fa"}
    fig = px.scatter(
        show,
        x="prob_up",
        y="target_weight",
        size="plot_size",
        color="engine_action",
        color_discrete_map=color_map,
        hover_name="asset_id",
        hover_data=[c for c in ["confidence", "signal_score", "entry_candidate", "quality_candidate", "post_trade_weight", "eod_weight"] if c in show.columns],
        title=f"Signal-to-Weight Map — {state.day}",
    )
    fig.update_layout(xaxis_title="DeepLOB P(up)", yaxis_title="DLS target weight")
    return apply_dark_layout(fig, 420)


def chart_execution_by_side(state: EngineState) -> go.Figure:
    """One compact chart replacing separate buy/sell order-book charts."""
    tr = state.trades.copy()
    if tr.empty or "side" not in tr.columns:
        return apply_dark_layout(go.Figure().update_layout(title="No executed trades for this replay state"), 320)
    g = tr.groupby(tr["side"].astype(str).str.lower()).agg(
        assets=("asset_id", "nunique"),
        turnover=("turnover", "sum"),
        cost=("cost", "sum"),
    ).reset_index().rename(columns={"side": "execution_side"})
    g["execution_side"] = g["execution_side"].str.title()
    fig = go.Figure()
    colors = {"Buy": "#22c55e", "Sell": "#f87171"}
    fig.add_trace(go.Bar(
        x=g["execution_side"],
        y=g["turnover"],
        name="Turnover",
        marker_color=[colors.get(x, "#60a5fa") for x in g["execution_side"]],
        text=[money(v) for v in g["turnover"]],
        textposition="outside",
        customdata=np.stack([g["assets"], g["cost"]], axis=-1),
        hovertemplate="Side=%{x}<br>Turnover=%{y:,.0f}<br>Assets=%{customdata[0]}<br>Cost=%{customdata[1]:,.0f}<extra></extra>",
    ))
    fig.update_layout(title=f"Execution Summary by Side — {state.day}", xaxis_title="Side", yaxis_title="Turnover")
    return apply_dark_layout(fig, 330)


def build_execution_reconciliation(state: EngineState) -> pd.DataFrame:
    """Display-ready execution table combining signals, weights, orders and fills."""
    book = build_decision_book(state)
    diff = build_order_diff(state)
    if book.empty and diff.empty:
        return empty_df()
    if book.empty:
        out = diff.copy()
    else:
        out = book.copy()
        if not diff.empty:
            merge_cols = [c for c in ["asset_id", "dls_net_order", "base_net_order", "order_delta", "order_overlap", "direction_case"] if c in diff.columns]
            out = out.merge(diff[merge_cols], on="asset_id", how="outer")

    numeric_needed = [
        "prob_down", "prob_flat", "prob_up", "confidence", "pre_trade_weight", "target_weight", "post_trade_weight", "eod_weight",
        "dls_order_buy_pct", "dls_order_sell_pct", "filled_buy_pct", "filled_sell_pct",
        "dls_net_order", "base_net_order", "order_delta", "filled_buy_shares", "filled_sell_shares",
        "buy_execution_price", "sell_execution_price", "filled_buy_turnover", "filled_sell_turnover",
        "filled_buy_cost", "filled_sell_cost", "engine_turnover", "base_order_buy_pct", "base_order_sell_pct",
    ]
    for c in numeric_needed:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    if "engine_action" not in out.columns:
        out["engine_action"] = np.where(out.get("order_delta", 0).abs() > 0, "ORDER", "HOLD")
    if "order_overlap" not in out.columns:
        out["order_overlap"] = "—"
    if "direction_case" not in out.columns:
        out["direction_case"] = "—"

    active = out[
        (out["engine_action"].astype(str) != "HOLD")
        | (out["engine_turnover"].abs() > 0)
        | (out["dls_net_order"].abs() > 0)
        | (out["base_net_order"].abs() > 0)
        | (out["target_weight"].abs() > 0)
        | (out["filled_buy_pct"].abs() > 0)
        | (out["filled_sell_pct"].abs() > 0)
    ].copy()
    if active.empty:
        active = out.copy()

    active["sort_key"] = (
        active["engine_turnover"].abs()
        + active["order_delta"].abs() * 1e6
        + active["target_weight"].abs() * 1e5
        + active["filled_buy_pct"].abs() * 1e4
        + active["filled_sell_pct"].abs() * 1e4
    )
    cols = [
        "asset_id", "engine_action",
        "prob_down", "prob_flat", "prob_up", "confidence",
        "pre_trade_weight", "target_weight", "post_trade_weight", "eod_weight",
        "dls_order_buy_pct", "dls_order_sell_pct", "filled_buy_pct", "filled_sell_pct",
        "dls_net_order", "base_net_order", "order_delta", "order_overlap", "direction_case",
        "filled_buy_shares", "filled_sell_shares", "buy_execution_price", "sell_execution_price",
        "filled_buy_turnover", "filled_sell_turnover", "filled_buy_cost", "filled_sell_cost", "engine_turnover",
        "base_order_buy_pct", "base_order_sell_pct",
    ]
    show = active.sort_values("sort_key", ascending=False)[[c for c in cols if c in active.columns]].head(90).copy()
    rename = {
        "asset_id": "Asset",
        "engine_action": "Engine action",
        "prob_down": "DeepLOB P(down)",
        "prob_flat": "DeepLOB P(flat)",
        "prob_up": "DeepLOB P(up)",
        "confidence": "Confidence",
        "pre_trade_weight": "Previous weight",
        "target_weight": "DLS predicted target weight",
        "post_trade_weight": "Post-trade weight",
        "eod_weight": "Final EOD weight",
        "dls_order_buy_pct": "DLS submitted buy %",
        "dls_order_sell_pct": "DLS submitted sell %",
        "filled_buy_pct": "Filled buy %",
        "filled_sell_pct": "Filled sell %",
        "dls_net_order": "DLS net order %",
        "base_net_order": "Base net order %",
        "order_delta": "DLS minus Base order %",
        "order_overlap": "Order overlap",
        "direction_case": "Direction case",
        "filled_buy_shares": "Filled buy shares",
        "filled_sell_shares": "Filled sell shares",
        "buy_execution_price": "Buy execution price",
        "sell_execution_price": "Sell execution price",
        "filled_buy_turnover": "Buy turnover",
        "filled_sell_turnover": "Sell turnover",
        "filled_buy_cost": "Buy cost",
        "filled_sell_cost": "Sell cost",
        "engine_turnover": "Total executed turnover",
        "base_order_buy_pct": "Base submitted buy %",
        "base_order_sell_pct": "Base submitted sell %",
    }
    return show.rename(columns=rename)


def chart_replay_efficiency(summary: pd.DataFrame, current_idx: int) -> go.Figure:
    """Useful lifecycle chart: how many candidates survive through filters and execution over time."""
    x = summary["day_index"]
    fig = go.Figure()
    for col, label, color in [
        ("quality_pass_rate", "Quality pass rate", "#22c55e"),
        ("target_utilization", "Target utilization", "#22d3ee"),
        ("execution_rate", "Execution rate", "#f59e0b"),
    ]:
        if col in summary.columns:
            fig.add_trace(go.Scatter(x=x, y=summary[col] * 100, mode="lines", name=label, line=dict(width=2, color=color)))
    current_day_line(fig, int(summary.iloc[current_idx]["day_index"]))
    fig.update_layout(title="Filter and Execution Efficiency Over Time", xaxis_title="Trade day", yaxis_title="Rate (%)")
    return apply_dark_layout(fig, 320)


tables = load_tables()
summary = build_daily_summary(tables)

if summary.empty:
    st.error("Trading engine cannot boot: no replayable trade days were found in the bundled CSV exports.")
    st.stop()

# Sidebar replay controls
with st.sidebar:
    st.markdown("# ⚡ Engine Controls")
    st.caption("Replay the trading process as a state machine, not as static CSV views.")

    if "replay_idx" not in st.session_state:
        st.session_state.replay_idx = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if st.session_state.get("app_build_id") != "v10_daily_metrics_engine":
        st.session_state.app_build_id = "v10_daily_metrics_engine"
        st.session_state.playing = False
        st.session_state.replay_idx = 0
        st.session_state.last_engine_tick_v10 = None

    min_idx, max_idx = 0, len(summary) - 1
    c1, c2 = st.columns(2)
    with c1:
        if st.button("◀ Previous", use_container_width=True):
            st.session_state.replay_idx = max(min_idx, st.session_state.replay_idx - 1)
            st.session_state.playing = False
    with c2:
        if st.button("Next ▶", use_container_width=True):
            st.session_state.replay_idx = min(max_idx, st.session_state.replay_idx + 1)
            st.session_state.playing = False

    st.session_state.replay_idx = st.slider(
        "Replay step",
        min_value=min_idx,
        max_value=max_idx,
        value=int(st.session_state.replay_idx),
        format="Step %d",
    )

    play_cols = st.columns(2)
    with play_cols[0]:
        if st.button("▶ Play" if not st.session_state.playing else "⏸ Pause", use_container_width=True):
            st.session_state.playing = not st.session_state.playing
    with play_cols[1]:
        if st.button("⟲ Reset", use_container_width=True):
            st.session_state.replay_idx = 0
            st.session_state.playing = False
            st.rerun()

    speed = st.select_slider("Playback speed", options=["0.5x", "1x", "2x", "5x", "10x"], value="1x")
    # Safe intervals: very aggressive refresh rates can make Streamlit appear to spin forever,
    # especially on Windows/local browsers while Plotly figures are still rendering.
    speed_map = {"0.5x": 2400, "1x": 1500, "2x": 1000, "5x": 700, "10x": 500}
    loop_playback = st.toggle("Loop playback", value=True)
    enable_autoplay = st.toggle("Enable auto-play module", value=False, help="Keep this off for the fastest startup. Manual Previous/Next/Slider replay always works.")
    fast_render = st.toggle("Fast playback rendering", value=True, help="When playback is running, render only the lightweight engine state instead of heavy charts and large tables.")

    if not enable_autoplay and st.session_state.playing:
        st.session_state.playing = False

    if st.session_state.playing and enable_autoplay:
        if st_autorefresh is not None:
            tick = st_autorefresh(interval=speed_map[speed], key="engine_tick_v10_fast")
            last_tick = st.session_state.get("last_engine_tick_v10", None)
            if tick != last_tick:
                st.session_state.last_engine_tick_v10 = tick
                if st.session_state.replay_idx >= max_idx:
                    st.session_state.replay_idx = 0 if loop_playback else max_idx
                    if not loop_playback:
                        st.session_state.playing = False
                else:
                    st.session_state.replay_idx += 1
        else:
            st.warning("Auto-play package is not installed. Manual replay controls still work.")

    current_idx = int(st.session_state.replay_idx)
    current_day = str(summary.iloc[current_idx]["trade_day_id"])
    st.markdown("---")
    st.markdown("### Current State")
    st.markdown(f"<span class='chip'>Step {current_idx + 1}/{len(summary)}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='chip chip-green'>Trade Day {current_day}</span>", unsafe_allow_html=True)
    signal_day = summary.iloc[current_idx].get("signal_day_id", "—")
    st.markdown(f"<span class='chip chip-amber'>Signal Day {signal_day}</span>", unsafe_allow_html=True)
    st.markdown("---")
    view = st.radio(
        "Engine View",
        [
            "⚡ Engine Replay",
            "🧠 Decision Inspector",
            "📟 Execution Tape",
            "🧺 Portfolio State",
            "📅 Daily Metrics",
            "⚔ Model Comparison",
            "🧪 Training Lab",
        ],
        index=0,
    )
    st.caption("v13 uses explained minimal rendering and hard-normalizes the DLS equity path to ¥50M on the first valid replay day.")

state = get_state(tables, summary, current_idx)

# Header
st.markdown(
    """
    <div class="engine-hero">
      <div class="kicker">Event-driven trading simulation</div>
      <h1>Shifu DeepLOB + DLS Trading Engine Replay</h1>
      <p>Scrub through the out-of-sample run and inspect every state transition: raw DeepLOB signal, confidence filter, DLS target weights, submitted orders, executed trades, EOD holdings, and model comparison against the base DeepLOB strategy.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Status strip
r = state.row
st.markdown(
    f"""
    <div class="status-grid">
      <div class="status-card"><div class="status-label">Engine Clock</div><div class="status-value">{state.day}</div><div class="status-note">signal day {r.get('signal_day_id', '—')}</div></div>
      <div class="status-card"><div class="status-label">DLS Equity Proxy</div><div class="status-value">{money(r.get('dls_equity_proxy'))}</div><div class="status-note">normalized to ¥50M initial capital</div></div>
      <div class="status-card"><div class="status-label">Return Proxy</div><div class="status-value">{pct(r.get('dls_return_proxy_%'), 2, already_percent=True)}</div><div class="status-note">DLS replay path</div></div>
      <div class="status-card"><div class="status-label">Target Gross</div><div class="status-value">{pct(r.get('engine_target_gross'), 2)}</div><div class="status-note">optimizer exposure</div></div>
      <div class="status-card"><div class="status-label">Executed Turnover</div><div class="status-value">{money(r.get('executed_total_turnover'))}</div><div class="status-note">buy + sell notional</div></div>
      <div class="status-card"><div class="status-label">EOD Holdings</div><div class="status-value">{num(r.get('eod_holdings'), 0)}</div><div class="status-note">active names</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stage line
st.markdown(
    f"""
    <div class="stage-line">
      <div class="stage-box"><div class="stage-num">01</div><div class="stage-title">Signal Bus</div><div class="stage-main">{num(r.get('raw_assets', r.get('asset_universe', 0)),0)}</div><div class="stage-sub">all DeepLOB signals | up={num(r.get('raw_buy_signals'),0)} flat={num(r.get('raw_flat_signals'),0)} down={num(r.get('raw_sell_signals'),0)}</div></div>
      <div class="stage-box"><div class="stage-num">02</div><div class="stage-title">Entry Mask</div><div class="stage-main">{num(r.get('entry_candidates'),0)}</div><div class="stage-sub">candidate assets</div></div>
      <div class="stage-box"><div class="stage-num">03</div><div class="stage-title">Quality Filter</div><div class="stage-main">{num(r.get('quality_candidates'),0)}</div><div class="stage-sub">pass rate {pct(r.get('quality_pass_rate'),1)}</div></div>
      <div class="stage-box"><div class="stage-num">04</div><div class="stage-title">DLS Optimizer</div><div class="stage-main">{num(r.get('target_names'),0)}</div><div class="stage-sub">target names</div></div>
      <div class="stage-box"><div class="stage-num">05</div><div class="stage-title">Execution</div><div class="stage-main">{num(r.get('executed_assets'),0)}</div><div class="stage-sub">{num(r.get('executed_buys'),0)} buys / {num(r.get('executed_sells'),0)} sells</div></div>
      <div class="stage-box"><div class="stage-num">06</div><div class="stage-title">Portfolio State</div><div class="stage-main">{num(r.get('eod_holdings'),0)}</div><div class="stage-sub">top-10 weight {pct(r.get('top10_eod_weight'),1)}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)



# Main engine views: lean single-view rendering.
# v9 keeps only decision-useful charts and uses a fast playback render path.
fast_playback = bool(st.session_state.get("playing", False)) and bool(enable_autoplay) and bool(fast_render)

if fast_playback:
    st.markdown(
        "<div class='panel'><div class='panel-title'>⚡ Fast Playback Mode</div>"
        "<div class='panel-sub'>Heavy charts and large tables are paused while the replay is running. Pause playback to inspect charts and detailed rows for the selected day.</div>",
        unsafe_allow_html=True,
    )
    st.progress((current_idx + 1) / max(len(summary), 1), text=f"Replay step {current_idx + 1}/{len(summary)} · {state.day}")
    c1, c2 = st.columns([0.62, 0.38], gap="large")
    with c1:
        st.markdown(f"<div class='terminal'>{event_log_text(state)}</div>", unsafe_allow_html=True)
    with c2:
        sig_df_lite = prepare_deeplob_signal_frame(state)
        st.markdown(deeplob_signal_cards_html(sig_df_lite), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

if view == "⚡ Engine Replay":
    section_intro(
        "Engine Replay",
        "This is the high-level playback screen. It shows the current trading day as an engine state: portfolio path, event log, decision funnel, and the same-day DeepLOB three-class signal monitor. During Play mode, heavy charts are skipped; pause the replay for detailed inspection."
    )
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown("<div class='panel'><div class='panel-title'>📈 Replay Path</div><div class='panel-sub'>Tracks the reconstructed DLS equity path against the base DeepLOB portfolio. This is the main performance timeline used while scrubbing through the engine clock.</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_returns(summary, current_idx), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='panel'><div class='panel-title'>🖥 Engine Event Log</div><div class='panel-sub'>A compact text trace of what happened at this step: signal day, filters, executed orders, turnover, costs and portfolio state.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='terminal'>{event_log_text(state)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel'><div class='panel-title'>🔻 Current Decision Funnel</div><div class='panel-sub'>Shows how the universe shrinks from raw signals to entry candidates, quality-passed assets, target weights, executions and final holdings.</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_funnel(state), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    sig_df = prepare_deeplob_signal_frame(state)
    st.markdown(
        "<div class='panel'><div class='panel-title'>📡 DeepLOB 3-Signal Monitor</div>"
        "<div class='panel-sub'>For the replay day, this connects DeepLOB class probabilities — P(down), P(flat), P(up) — to the downstream DLS decision. The stacked chart is kept because it directly explains why a stock was considered bullish, neutral, or bearish.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(deeplob_signal_cards_html(sig_df), unsafe_allow_html=True)
    sctrl1, sctrl2 = st.columns([1.2, 0.8])
    with sctrl1:
        signal_sort = st.selectbox(
            "Signal ranking",
            ["Highest P(up)", "Highest P(flat)", "Highest P(down)", "Highest 3-class confidence", "Largest signal margin", "Largest DLS target weight"],
            index=0,
            key="signal_monitor_ranking_v9",
        )
    with sctrl2:
        signal_top_n = st.slider("Assets shown", 5, 30, 12, step=5, key="signal_monitor_top_n_v9")
    sm1, sm2 = st.columns([0.95, 1.05], gap="large")
    with sm1:
        st.plotly_chart(chart_deeplob_stacked_probs(sig_df, signal_sort, signal_top_n), use_container_width=True)
    with sm2:
        st.markdown("<div class='panel-sub'>Combined signal-to-decision table: probabilities, confidence, filter pass, target weight and executed action in one place.</div>", unsafe_allow_html=True)
        display_table(deeplob_signal_table_view(sig_df, signal_sort, signal_top_n), height=420)
    st.markdown("</div>", unsafe_allow_html=True)

elif view == "🧠 Decision Inspector":
    section_intro(
        "Decision Inspector",
        "This is the asset-level decision book. Use it to answer: which assets had strong signals, which passed filters, which received target weights, and which were actually bought or sold. Tables are combined so the full decision chain is visible in one row per asset."
    )
    book = build_decision_book(state)
    st.markdown("<div class='panel'><div class='panel-title'>🧠 Combined Decision Book</div><div class='panel-sub'>Full chain per asset: DeepLOB P(down/flat/up) → filters → previous/target/post/EOD weights → submitted order % → filled execution %.</div>", unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        action_filter = st.selectbox("Action filter", ["All", "BUY", "SELL", "HOLD"], index=0)
    with col_b:
        ranking = st.selectbox("Sort book by", ["Engine turnover", "Target weight", "Confidence", "P(up)", "Signal score"], index=0)
    with col_c:
        max_rows = st.slider("Rows", 20, 100, 45, step=5)
    with col_d:
        high_quality_only = st.toggle("Quality only", value=False)

    filtered_book = book.copy()
    if action_filter != "All" and not filtered_book.empty:
        filtered_book = filtered_book[filtered_book["engine_action"] == action_filter]
    if high_quality_only and not filtered_book.empty and "quality_candidate" in filtered_book.columns:
        filtered_book = filtered_book[filtered_book["quality_candidate"] == True]
    sort_map = {"Engine turnover": "engine_turnover", "Target weight": "target_weight", "Confidence": "confidence", "P(up)": "prob_up", "Signal score": "signal_score"}
    if not filtered_book.empty and sort_map[ranking] in filtered_book.columns:
        filtered_book = filtered_book.sort_values(sort_map[ranking], ascending=False)
    display_table(compact_book_view(filtered_book, max_rows=max_rows), height=410)
    st.markdown("</div>", unsafe_allow_html=True)

    d1, d2 = st.columns([1.0, 1.0], gap="large")
    with d1:
        st.markdown("<div class='panel'><div class='panel-title'>🧭 Signal-to-Weight Map</div><div class='panel-sub'>Useful diagnostic chart: a strong P(up) should generally map to larger target weights only after passing the confidence/quality filters. Point size indicates actual engine turnover.</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_signal_to_weight_scatter(state), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with d2:
        st.markdown("<div class='panel'><div class='panel-title'>⚖ Compact Weight Transition</div><div class='panel-sub'>Shows the assets with the largest practical move from pre-trade weight to target/post-trade weight. The large heatmap and duplicated transition charts were removed.</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_compact_weight_transition(state, top_n=14), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    assets_for_selector = []
    if not book.empty:
        traded_assets = book[book["engine_action"].isin(["BUY", "SELL"])]
        assets_for_selector = traded_assets["asset_id"].head(80).tolist() or book["asset_id"].head(80).tolist()
    focus_asset = st.selectbox("Focus asset", assets_for_selector, index=0 if assets_for_selector else None, placeholder="Select an asset")
    st.markdown("<div class='panel'><div class='panel-title'>🔎 Trade Explanation</div><div class='panel-sub'>Plain-English engine explanation for the selected asset.</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='terminal'>{explain_asset(book, focus_asset)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif view == "📟 Execution Tape":
    section_intro(
        "Execution Tape",
        "This section focuses only on executed trades and submitted orders. Instead of separate DLS order, base order and trade tables, the main reconciliation table combines them so you can compare what DLS wanted, what the base model wanted, and what was filled."
    )
    st.markdown("<div class='panel'><div class='panel-title'>📟 Execution Summary</div><div class='panel-sub'>One compact chart and KPI strip. Separate buy/sell order-book charts were removed because the reconciliation table is more useful.</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Executed buys", num(r.get("executed_buys"), 0))
    c2.metric("Executed sells", num(r.get("executed_sells"), 0))
    c3.metric("Turnover", money(r.get("executed_total_turnover")))
    c4.metric("Cost", money(r.get("executed_total_cost")))
    st.plotly_chart(chart_execution_by_side(state), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'><div class='panel-title'>🧾 Execution & Order Reconciliation Table</div><div class='panel-sub'>Execution audit per asset: three DeepLOB probabilities, previous/target/post/EOD weights, submitted buy/sell %, filled buy/sell %, shares, prices and costs.</div>", unsafe_allow_html=True)
    display_table(build_execution_reconciliation(state), height=520)
    st.markdown("</div>", unsafe_allow_html=True)

elif view == "🧺 Portfolio State":
    section_intro(
        "Portfolio State",
        "This section shows what the engine holds at the end of the selected day. The chart focuses on the largest holdings, while the table gives the exact shares, close prices, market values and EOD weights. Treemaps and large heatmaps were removed to keep the view readable."
    )
    p1, p2 = st.columns([1.0, 1.1], gap="large")
    with p1:
        st.markdown("<div class='panel'><div class='panel-title'>🧺 Top Holdings</div><div class='panel-sub'>Most useful portfolio chart: ranked exposure by market value, with EOD weight shown on the bar labels.</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_top_holdings_bar(state, top_n=16), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with p2:
        st.markdown("<div class='panel'><div class='panel-title'>📌 EOD Holding Snapshot</div><div class='panel-sub'>Exact end-of-day portfolio state for the replay day.</div>", unsafe_allow_html=True)
        hcols = ["asset_id", "shares", "close_price", "market_value", "eod_weight"]
        display_table(state.holdings[[c for c in hcols if c in state.holdings.columns]].sort_values("market_value", ascending=False) if not state.holdings.empty else state.holdings, height=430)
        st.markdown("</div>", unsafe_allow_html=True)

elif view == "📅 Daily Metrics":
    section_intro(
        "Daily Model Metrics",
        "This view compares the base DeepLOB strategy and DeepLOB + DLS at the current replay day. It is separate from the final model comparison: here you can inspect same-day return, cumulative return, turnover, costs, holdings and trade counts while moving through the replay timeline."
    )
    st.markdown(
        "<div class='panel'><div class='panel-title'>📅 Same-Day Scoreboard</div>"
        "<div class='panel-sub'>Current replay-day metrics. DeepLOB + DLS values are reconstructed from DLS trades and holding snapshots; Base DeepLOB values come from the original DeepLOB daily log. Deltas are always DLS minus Base.</div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(comparison_card_html("Daily Return", pct(r.get("dls_daily_return_%"), 3, already_percent=True), pct(r.get("base_daily_return_%"), 3, already_percent=True), f"{r.get('daily_return_delta_pp', np.nan):.3f} pp" if np.isfinite(r.get('daily_return_delta_pp', np.nan)) else "—", delta_class(r.get("daily_return_delta_pp", np.nan), "total_return_%"), "Same replay day"), unsafe_allow_html=True)
    with m2:
        st.markdown(comparison_card_html("Cumulative Return", pct(r.get("dls_return_proxy_%"), 2, already_percent=True), pct(r.get("base_return_%"), 2, already_percent=True), f"{r.get('cum_return_delta_pp', np.nan):.2f} pp" if np.isfinite(r.get('cum_return_delta_pp', np.nan)) else "—", delta_class(r.get("cum_return_delta_pp", np.nan), "total_return_%"), "Up to selected day"), unsafe_allow_html=True)
    with m3:
        st.markdown(comparison_card_html("Daily Cost", money(r.get("executed_total_cost")), money(r.get("base_total_costs")), money(r.get("daily_cost_delta")), delta_class(r.get("daily_cost_delta", np.nan), "total_costs_RMB"), "Lower is better"), unsafe_allow_html=True)
    with m4:
        st.markdown(comparison_card_html("Holdings", num(r.get("eod_holdings"), 0), num(r.get("base_num_holdings"), 0), num(r.get("holdings_delta"), 0), "delta-neutral", "Exposure breadth"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.05, 0.95], gap="large")
    with c1:
        st.markdown("<div class='panel'><div class='panel-title'>📈 Daily Return Window</div><div class='panel-sub'>Local return path around the selected replay day. This helps detect whether DLS improves one specific day or only the final aggregate score.</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_daily_return_window(summary, current_idx, window=24), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='panel-title'>⚖ Current-Day Delta Bar</div><div class='panel-sub'>A compact view of the same-day difference: positive bars mean DeepLOB + DLS is larger than Base DeepLOB for that metric.</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_daily_metric_delta(summary, current_idx), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("<div class='panel'><div class='panel-title'>🔄 Trading Activity</div><div class='panel-sub'>Compares how active the two strategies are on the selected day: buys, sells and number of holdings.</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_daily_trading_intensity(summary, current_idx), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='panel'><div class='panel-title'>💸 Turnover and Cost</div><div class='panel-sub'>Money metrics are scaled for readability: turnover is shown in millions of RMB and cost in thousands of RMB.</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_daily_money_comparison(summary, current_idx), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'><div class='panel-title'>📋 Same-Day Metric Table</div><div class='panel-sub'>One combined table for all daily model metrics, so the comparison is not scattered across several small tables.</div>", unsafe_allow_html=True)
    display_table(daily_metric_rows(summary, current_idx), height=420)
    st.markdown("</div>", unsafe_allow_html=True)

elif view == "⚔ Model Comparison":
    section_intro(
        "Model Comparison",
        "This cockpit compares DeepLOB + DLS against the base DeepLOB strategy. It keeps the charts that answer practical questions: did DLS improve return/risk, how did drawdown evolve, and how different were the submitted orders on the selected day?"
    )
    comp = tables.get("comparison", empty_df())
    dls_row, base_row = get_model_pair(comp)
    st.markdown(
        "<div class='panel'><div class='panel-title'>⚔ Model Comparison Cockpit</div>"
        "<div class='panel-sub'>Final scoreboard plus the minimum useful charts for model comparison. Redundant cost/turnover plots were removed; final costs remain in the KPI and impact chart.</div>",
        unsafe_allow_html=True,
    )
    if comp.empty or dls_row is None or base_row is None:
        st.info("Comparison table is not available or model names could not be detected.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        ret_delta = val(dls_row, "total_return_%") - val(base_row, "total_return_%")
        sharpe_delta = val(dls_row, "sharpe") - val(base_row, "sharpe")
        dd_delta = val(dls_row, "max_drawdown_%") - val(base_row, "max_drawdown_%")
        cost_delta = val(dls_row, "total_costs_RMB") - val(base_row, "total_costs_RMB")
        with k1:
            st.markdown(comparison_card_html("Total Return", pct(val(dls_row, "total_return_%"), already_percent=True), pct(val(base_row, "total_return_%"), already_percent=True), f"{ret_delta:.2f} pp", delta_class(ret_delta, "total_return_%"), "Higher is better"), unsafe_allow_html=True)
        with k2:
            st.markdown(comparison_card_html("Sharpe Ratio", num(val(dls_row, "sharpe"), 3), num(val(base_row, "sharpe"), 3), f"{sharpe_delta:.3f}", delta_class(sharpe_delta, "sharpe"), "Risk-adjusted performance"), unsafe_allow_html=True)
        with k3:
            st.markdown(comparison_card_html("Max Drawdown", pct(val(dls_row, "max_drawdown_%"), already_percent=True), pct(val(base_row, "max_drawdown_%"), already_percent=True), f"{dd_delta:.2f} pp", delta_class(dd_delta, "max_drawdown_%"), "Less negative is better"), unsafe_allow_html=True)
        with k4:
            st.markdown(comparison_card_html("Total Costs", money(val(dls_row, "total_costs_RMB")), money(val(base_row, "total_costs_RMB")), money(cost_delta), delta_class(cost_delta, "total_costs_RMB"), "Lower cost is better"), unsafe_allow_html=True)
        p1, p2 = st.columns(2, gap="large")
        with p1:
            st.plotly_chart(chart_returns(summary, current_idx), use_container_width=True)
        with p2:
            st.plotly_chart(chart_drawdown_comparison(summary, current_idx), use_container_width=True)
        p3, p4 = st.columns([0.95, 1.05], gap="large")
        with p3:
            st.plotly_chart(chart_comparison_delta(comp), use_container_width=True)
        with p4:
            st.plotly_chart(chart_compact_order_delta(state, top_n=14), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        diff = build_order_diff(state)
        top_diff = diff[diff["order_overlap"] != "No active order"].copy().head(50) if not diff.empty else diff
        cols = ["asset_id", "order_overlap", "direction_case", "dls_net_order", "base_net_order", "order_delta", "dls_buy", "dls_sell", "base_buy", "base_sell"]
        st.markdown("<div class='panel'><div class='panel-title'>📋 Combined Model Difference Table</div><div class='panel-sub'>Current-day order-level disagreement between DeepLOB + DLS and base DeepLOB.</div>", unsafe_allow_html=True)
        display_table(top_diff[[c for c in cols if c in top_diff.columns]] if not top_diff.empty else top_diff, height=360)
        with st.expander("Show final comparison table"):
            display_table(comparison_table_view(comp), height=160)
        st.markdown("</div>", unsafe_allow_html=True)

else:  # Training Lab
    section_intro(
        "Training Lab",
        "This view is for model-development diagnostics rather than trading replay. It keeps the training loss monitor visible and moves seed-search diagnostics into an optional expander, because these are not needed during normal trading playback."
    )
    st.markdown("<div class='panel'><div class='panel-title'>🧪 DLS Training Monitor</div><div class='panel-sub'>Training and validation loss across epochs. This helps check whether the DLS optimizer training was stable before the out-of-sample replay.</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_training(tables), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("Optional: show seed search diagnostics"):
        st.plotly_chart(chart_seed_search(tables), use_container_width=True)

st.caption("Engine note: v10 uses explained minimal single-view rendering for reliability. DLS equity is reconstructed as a replay proxy from EOD holdings and target gross because the exported files include the base daily log directly but DLS daily portfolio value is represented through holdings/trades/weights.")
