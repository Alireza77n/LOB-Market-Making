"""
Dashboard configuration — single venue (Nobitex) for now.

All paths and tunables live here so the app, metrics and figures modules stay
free of magic constants. Window sizes follow the spec in
``lob_dashboard_metrics.md`` (one short "now" window, one longer context window).
"""

from __future__ import annotations

import os

# ── Venue ────────────────────────────────────────────────────────────────────
VENUE = "nobitex"

# Symbols the Nobitex collector writes (see nobitexCollector.py SYMBOLS).
SYMBOLS = ["BTCIRT", "USDTIRT", "ETHIRT", "BNBIRT", "XRPIRT"]
DEFAULT_SYMBOL = "BTCIRT"

# Order-book depth stored per snapshot (20 levels each side).
LOB_DEPTH = 20

# ── Data location ────────────────────────────────────────────────────────────
# The collector writes ``nobitex_data/`` next to wherever it is launched. By
# default we look in the repo root; override with the LOB_DATA_DIR env var so the
# dashboard can point at wherever you actually run the collector.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get(
    "LOB_DATA_DIR",
    os.path.join(_REPO_ROOT, "data-collection", "exchange-apis", f"{VENUE}_data"),
)


def orderbook_csv(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"{symbol}_orderbook.csv")


def trades_csv(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"{symbol}_trades.csv")


# ── Rolling windows (seconds) ────────────────────────────────────────────────
# SHORT = "now" tiles; LONG = context. Tune per the spec's window-sizing note.
WINDOW_SHORT_SEC = 60          # 1 min
WINDOW_LONG_SEC = 30 * 60      # 30 min

# How much history to load into time-series plots (keeps the app responsive).
PLOT_LOOKBACK_SEC = 60 * 60    # 1 hour

# ── Imbalance / depth parameters ─────────────────────────────────────────────
IMBALANCE_LEVELS = [1, 5, 20]  # top-N imbalance tiles
DEPTH_WITHIN_BPS = 25          # "depth within X bps" tile

# Representative order sizes (in BASE units) for slippage tiles, per symbol.
# Calibrated loosely to each book's scale; adjust as you learn each venue.
SLIPPAGE_SIZES = {
    "BTCIRT":  [0.01, 0.05, 0.1],
    "ETHIRT":  [0.1, 0.5, 1.0],
    "BNBIRT":  [0.5, 2.0, 5.0],
    "XRPIRT":  [100, 500, 1000],
    "USDTIRT": [1000, 5000, 20000],
}

# VPIN equal-volume bucket size (BASE units) and number of buckets to average.
# USDT/IRT needs a far larger bucket than XRP/IRT — calibrate per symbol.
VPIN_BUCKET_VOLUME = {
    "BTCIRT":  0.5,
    "ETHIRT":  5.0,
    "BNBIRT":  20.0,
    "XRPIRT":  5000.0,
    "USDTIRT": 100000.0,
}
VPIN_N_BUCKETS = 20

# Trade tape: how many recent trades to show.
TAPE_ROWS = 25

# ── Refresh ──────────────────────────────────────────────────────────────────
REFRESH_MS = 10_000            # match the collector's 10s poll cadence
