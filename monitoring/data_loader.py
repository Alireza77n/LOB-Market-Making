"""
Data loading layer for the Nobitex dashboard.

Reads the two CSVs the collector writes per symbol and returns tidy pandas
frames. Robust to:
  * the file not existing yet (collector not started) -> empty frame,
  * millisecond *or* second timestamps (pd.to_datetime is format-tolerant),
  * partially written final rows (we drop rows that fail numeric coercion).

Order-book schema:  time, bid_price_1, bid_volume_1, ask_price_1, ask_volume_1, ... x20
Trade schema:       snapshot_time, trade_time, price, volume, direction
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config


# ── Order book ───────────────────────────────────────────────────────────────

def _orderbook_columns(depth: int) -> list[str]:
    cols = ["time"]
    for i in range(1, depth + 1):
        cols += [f"bid_price_{i}", f"bid_volume_{i}",
                 f"ask_price_{i}", f"ask_volume_{i}"]
    return cols


def load_orderbook(symbol: str, lookback_sec: int | None = None) -> pd.DataFrame:
    """Return order-book snapshots for *symbol*, newest last.

    Columns are parsed numeric; ``time`` is tz-aware UTC datetime. Empty frame
    (with correct columns) if the file is missing or unreadable.
    """
    path = config.orderbook_csv(symbol)
    cols = _orderbook_columns(config.LOB_DEPTH)
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(columns=cols)

    if df.empty or "time" not in df.columns:
        return pd.DataFrame(columns=cols)

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])

    price_vol_cols = [c for c in df.columns if c != "time"]
    df[price_vol_cols] = df[price_vol_cols].apply(pd.to_numeric, errors="coerce")

    df = df.sort_values("time").reset_index(drop=True)
    if lookback_sec is not None and not df.empty:
        cutoff = df["time"].iloc[-1] - pd.Timedelta(seconds=lookback_sec)
        df = df[df["time"] >= cutoff].reset_index(drop=True)
    return df


def latest_snapshot(symbol: str) -> pd.Series | None:
    """Most recent order-book row as a Series, or None if no data."""
    df = load_orderbook(symbol, lookback_sec=config.WINDOW_SHORT_SEC * 4)
    if df.empty:
        # fall back to a full read in case nothing is within the short window
        df = load_orderbook(symbol)
    if df.empty:
        return None
    return df.iloc[-1]


def book_sides(row: pd.Series, depth: int = config.LOB_DEPTH):
    """Extract (bid_prices, bid_vols, ask_prices, ask_vols) arrays from a snapshot row.

    Levels with NaN price/volume (book shallower than ``depth``) are dropped.
    Returned best-first on each side.
    """
    bp, bv, ap, av = [], [], [], []
    for i in range(1, depth + 1):
        p_b, v_b = row.get(f"bid_price_{i}"), row.get(f"bid_volume_{i}")
        p_a, v_a = row.get(f"ask_price_{i}"), row.get(f"ask_volume_{i}")
        if pd.notna(p_b) and pd.notna(v_b):
            bp.append(float(p_b)); bv.append(float(v_b))
        if pd.notna(p_a) and pd.notna(v_a):
            ap.append(float(p_a)); av.append(float(v_a))
    return np.array(bp), np.array(bv), np.array(ap), np.array(av)


# ── Trades ───────────────────────────────────────────────────────────────────

TRADE_COLS = ["snapshot_time", "trade_time", "price", "volume", "direction"]


def load_trades(symbol: str, lookback_sec: int | None = None) -> pd.DataFrame:
    """Return trades for *symbol*, newest last.

    ``trade_time`` is the authoritative event time (tz-aware UTC). Rows with a
    bad price/volume are dropped. Empty frame if the file is missing.
    """
    path = config.trades_csv(symbol)
    if not os.path.exists(path):
        return pd.DataFrame(columns=TRADE_COLS)

    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(columns=TRADE_COLS)

    if df.empty:
        return pd.DataFrame(columns=TRADE_COLS)

    # trade_time is the true event time; fall back to snapshot_time if absent.
    time_col = "trade_time" if "trade_time" in df.columns else "snapshot_time"
    df["trade_time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["trade_time"])

    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
    df = df.dropna(subset=["price", "volume"])

    df["direction"] = df.get("direction", "").astype(str).str.lower()

    df = df.sort_values("trade_time").reset_index(drop=True)
    if lookback_sec is not None and not df.empty:
        cutoff = df["trade_time"].iloc[-1] - pd.Timedelta(seconds=lookback_sec)
        df = df[df["trade_time"] >= cutoff].reset_index(drop=True)
    return df


def data_health(symbol: str) -> dict:
    """Lightweight freshness summary used by the status banner."""
    ob = load_orderbook(symbol, lookback_sec=config.WINDOW_SHORT_SEC * 6)
    tr = load_trades(symbol, lookback_sec=config.WINDOW_SHORT_SEC * 6)
    now = pd.Timestamp.now(tz="UTC")
    ob_age = (now - ob["time"].iloc[-1]).total_seconds() if not ob.empty else None
    tr_age = (now - tr["trade_time"].iloc[-1]).total_seconds() if not tr.empty else None
    return {
        "ob_rows": len(ob),
        "tr_rows": len(tr),
        "ob_age_sec": ob_age,
        "tr_age_sec": tr_age,
        "has_data": not ob.empty,
    }
