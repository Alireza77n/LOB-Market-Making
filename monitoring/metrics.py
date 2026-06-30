"""
Metric computations for the single-venue dashboard.

Every formula maps directly to a row in ``lob_dashboard_metrics.md``:
  * ``snapshot_tiles`` -> Section A (current-state numerical tiles),
  * the ``*_series`` helpers -> Section B (time series & book shape).

All functions are pure: they take frames/rows from ``data_loader`` and return
plain numbers / frames, so they are trivially testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data_loader import book_sides


# ── Order-book point metrics (one snapshot) ──────────────────────────────────

def mid_price(bid: float, ask: float) -> float:
    return (bid + ask) / 2.0


def microprice(bid: float, v_bid: float, ask: float, v_ask: float) -> float:
    """Volume-weighted mid; tilts toward the thin side. Falls back to mid if flat."""
    denom = v_bid + v_ask
    if denom <= 0:
        return mid_price(bid, ask)
    return (bid * v_ask + ask * v_bid) / denom


def spread_abs(bid: float, ask: float) -> float:
    return ask - bid


def spread_bps(bid: float, ask: float) -> float:
    m = mid_price(bid, ask)
    return (ask - bid) / m * 1e4 if m > 0 else np.nan


def topn_imbalance(bv: np.ndarray, av: np.ndarray, n: int) -> float:
    """(ΣV_bid − ΣV_ask)/(ΣV_bid + ΣV_ask) over top n levels. Range [−1, 1]."""
    b = bv[:n].sum()
    a = av[:n].sum()
    denom = b + a
    return (b - a) / denom if denom > 0 else np.nan


def depth_within_bps(bp, bv, ap, av, mid: float, x_bps: float) -> float:
    """Total resting volume inside mid ± x_bps (both sides)."""
    if mid <= 0:
        return np.nan
    lo = mid * (1 - x_bps / 1e4)
    hi = mid * (1 + x_bps / 1e4)
    bid_vol = bv[bp >= lo].sum()
    ask_vol = av[ap <= hi].sum()
    return float(bid_vol + ask_vol)


def slippage_bps(prices: np.ndarray, vols: np.ndarray, qty: float, mid: float, side: str) -> float:
    """Cost (bps) of market-filling *qty* by walking the book.

    side='buy' walks asks, side='sell' walks bids. Returns NaN if the visible
    book cannot fill the size.
    """
    if mid <= 0 or qty <= 0 or len(prices) == 0:
        return np.nan
    remaining = qty
    cost = 0.0
    for p, v in zip(prices, vols):
        take = min(remaining, v)
        cost += take * p
        remaining -= take
        if remaining <= 0:
            break
    if remaining > 0:
        return np.nan  # book too thin to fill
    avg_exec = cost / qty
    if side == "buy":
        return (avg_exec - mid) / mid * 1e4
    return (mid - avg_exec) / mid * 1e4


# ── Trade-flow metrics over a window ─────────────────────────────────────────

def _split_dir(trades: pd.DataFrame):
    buy = trades[trades["direction"] == "buy"]
    sell = trades[trades["direction"] == "sell"]
    return buy, sell


def trade_intensity(trades: pd.DataFrame, window_sec: int) -> float:
    """Trades per minute over the window (uses actual span if shorter)."""
    if trades.empty:
        return 0.0
    span = (trades["trade_time"].iloc[-1] - trades["trade_time"].iloc[0]).total_seconds()
    span = max(span, 1.0)
    return len(trades) / span * 60.0


def flow_volumes(trades: pd.DataFrame) -> dict:
    buy, sell = _split_dir(trades)
    v_buy = buy["volume"].sum()
    v_sell = sell["volume"].sum()
    return {"buy": float(v_buy), "sell": float(v_sell), "net": float(v_buy - v_sell)}


def flow_imbalance(trades: pd.DataFrame) -> float:
    f = flow_volumes(trades)
    denom = f["buy"] + f["sell"]
    return (f["buy"] - f["sell"]) / denom if denom > 0 else np.nan


def vwap(trades: pd.DataFrame) -> float:
    if trades.empty or trades["volume"].sum() <= 0:
        return np.nan
    return float((trades["price"] * trades["volume"]).sum() / trades["volume"].sum())


def twap_mid(mid_series: pd.Series) -> float:
    return float(mid_series.mean()) if len(mid_series) else np.nan


def realized_vol(mid_series: pd.Series) -> float:
    """√Σ rₜ² with rₜ = ln(midₜ / midₜ₋₁). Approximate at 10s resolution."""
    m = mid_series.dropna()
    if len(m) < 2:
        return np.nan
    rets = np.log(m / m.shift(1)).dropna()
    return float(np.sqrt((rets ** 2).sum()))


def vpin(trades: pd.DataFrame, bucket_volume: float, n_buckets: int) -> float:
    """VPIN over equal-volume buckets of size *bucket_volume*.

    Per bucket: |V_buy − V_sell| / V. Average over the last *n_buckets*.
    Direction is given, so no bulk-classification is needed. Returns NaN if
    there isn't enough flow to fill a single bucket.
    """
    if trades.empty or bucket_volume <= 0:
        return np.nan

    signed = np.where(trades["direction"].values == "buy", 1.0, -1.0)
    vols = trades["volume"].values

    buckets = []
    cur_vol = 0.0
    cur_signed = 0.0
    for v, s in zip(vols, signed):
        remaining = v
        while remaining > 0:
            space = bucket_volume - cur_vol
            take = min(remaining, space)
            cur_vol += take
            cur_signed += s * take
            remaining -= take
            if cur_vol >= bucket_volume - 1e-12:
                buckets.append(abs(cur_signed) / bucket_volume)
                cur_vol = 0.0
                cur_signed = 0.0
    if not buckets:
        return np.nan
    return float(np.mean(buckets[-n_buckets:]))


# ── Snapshot tiles (Section A) ───────────────────────────────────────────────

def snapshot_tiles(snapshot: pd.Series | None,
                   trades_short: pd.DataFrame,
                   trades_long: pd.DataFrame,
                   mid_series_short: pd.Series,
                   mid_series_long: pd.Series,
                   symbol: str) -> dict:
    """Compute every Section-A tile. Missing inputs yield NaN, never an error."""
    out: dict = {"symbol": symbol}

    if snapshot is not None:
        bp, bv, ap, av = book_sides(snapshot)
        if len(bp) and len(ap):
            bid, ask = bp[0], ap[0]
            v_bid, v_ask = bv[0], av[0]
            m = mid_price(bid, ask)
            out.update({
                "best_bid": bid, "best_ask": ask,
                "mid": m,
                "microprice": microprice(bid, v_bid, ask, v_ask),
                "spread_abs": spread_abs(bid, ask),
                "spread_bps": spread_bps(bid, ask),
                "bid_depth": float(bv.sum()), "ask_depth": float(av.sum()),
                "depth_within_bps": depth_within_bps(bp, bv, ap, av, m, config.DEPTH_WITHIN_BPS),
            })
            out["imbalance"] = {
                n: topn_imbalance(bv, av, n) for n in config.IMBALANCE_LEVELS
            }
            sizes = config.SLIPPAGE_SIZES.get(symbol, [])
            out["buy_slippage"] = {
                q: slippage_bps(ap, av, q, m, "buy") for q in sizes
            }
            out["sell_slippage"] = {
                q: slippage_bps(bp, bv, q, m, "sell") for q in sizes
            }

    # Last trade
    if not trades_long.empty:
        last = trades_long.iloc[-1]
        out["last_trade_price"] = float(last["price"])
        out["last_trade_time"] = last["trade_time"]
        out["last_trade_dir"] = last["direction"]

    # Flow tiles — short and long windows
    out["intensity_short"] = trade_intensity(trades_short, config.WINDOW_SHORT_SEC)
    out["flow_short"] = flow_volumes(trades_short)
    out["flow_imbalance_short"] = flow_imbalance(trades_short)
    out["vwap_short"] = vwap(trades_short)
    out["vwap_long"] = vwap(trades_long)
    out["twap_short"] = twap_mid(mid_series_short)
    out["rvol_long"] = realized_vol(mid_series_long)
    out["vpin"] = vpin(
        trades_long,
        config.VPIN_BUCKET_VOLUME.get(symbol, np.nan),
        config.VPIN_N_BUCKETS,
    )
    return out


# ── Time-series builders (Section B) ─────────────────────────────────────────

def mid_micro_series(ob: pd.DataFrame) -> pd.DataFrame:
    """Per-snapshot mid, microprice, spread(bps), and top-N imbalances."""
    if ob.empty:
        return pd.DataFrame(columns=["time", "mid", "microprice", "spread_bps",
                                     "imb_1", "imb_5", "imb_20"])
    rows = []
    for _, r in ob.iterrows():
        bp, bv, ap, av = book_sides(r)
        if not (len(bp) and len(ap)):
            continue
        bid, ask = bp[0], ap[0]
        m = mid_price(bid, ask)
        rows.append({
            "time": r["time"],
            "mid": m,
            "microprice": microprice(bid, bv[0], ask, av[0]),
            "spread_bps": spread_bps(bid, ask),
            "imb_1": topn_imbalance(bv, av, 1),
            "imb_5": topn_imbalance(bv, av, 5),
            "imb_20": topn_imbalance(bv, av, 20),
        })
    return pd.DataFrame(rows)


def flow_bins(trades: pd.DataFrame, bin_sec: int = 60) -> pd.DataFrame:
    """Buy/sell/net volume and signed flow imbalance per time bin."""
    cols = ["bin", "buy", "sell", "net", "imbalance"]
    if trades.empty:
        return pd.DataFrame(columns=cols)
    t = trades.copy()
    t["bin"] = t["trade_time"].dt.floor(f"{bin_sec}s")
    buy = t[t["direction"] == "buy"].groupby("bin")["volume"].sum()
    sell = t[t["direction"] == "sell"].groupby("bin")["volume"].sum()
    out = pd.DataFrame({"buy": buy, "sell": sell}).fillna(0.0)
    out["net"] = out["buy"] - out["sell"]
    tot = out["buy"] + out["sell"]
    out["imbalance"] = np.where(tot > 0, out["net"] / tot, np.nan)
    return out.reset_index()


def rolling_rvol(mid_df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Rolling realized vol over a sliding window of snapshots."""
    if mid_df.empty:
        return pd.DataFrame(columns=["time", "rvol"])
    m = mid_df[["time", "mid"]].copy()
    logret = np.log(m["mid"] / m["mid"].shift(1))
    m["rvol"] = logret.pow(2).rolling(window, min_periods=2).sum().pow(0.5)
    return m[["time", "rvol"]].dropna()


def vpin_series(trades: pd.DataFrame, bucket_volume: float, n_buckets: int) -> pd.DataFrame:
    """Time-stamped VPIN, one point per completed equal-volume bucket."""
    cols = ["time", "vpin"]
    if trades.empty or bucket_volume <= 0:
        return pd.DataFrame(columns=cols)

    signed = np.where(trades["direction"].values == "buy", 1.0, -1.0)
    vols = trades["volume"].values
    times = trades["trade_time"].values

    bucket_vals, bucket_times = [], []
    cur_vol = cur_signed = 0.0
    last_time = times[0]
    for v, s, ts in zip(vols, signed, times):
        remaining = v
        while remaining > 0:
            space = bucket_volume - cur_vol
            take = min(remaining, space)
            cur_vol += take
            cur_signed += s * take
            remaining -= take
            last_time = ts
            if cur_vol >= bucket_volume - 1e-12:
                bucket_vals.append(abs(cur_signed) / bucket_volume)
                bucket_times.append(last_time)
                cur_vol = cur_signed = 0.0

    if not bucket_vals:
        return pd.DataFrame(columns=cols)
    s = pd.Series(bucket_vals).rolling(n_buckets, min_periods=1).mean()
    return pd.DataFrame({"time": bucket_times, "vpin": s.values})
