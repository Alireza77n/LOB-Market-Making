"""
Convert Iranian-exchange order-book CSV files (Nobitex / Bitpin / Ramzinex / Tabdeal /
Wallex) into the *processed* LOBFrame format, writing directly into
./data/<dataset>/{scaled_data,unscaled_data}/{training,validation,test}.

NOTE: output paths are relative to the CURRENT WORKING DIRECTORY, so run this from the
LOBFrame repo root (not from inside data_processing/), e.g.:
    python data_processing/convert_exchange_data.py --input_csv ... --symbol ... --clean

Why this exists
---------------
LOBFrame's built-in `data_processing` stage (data_process.py) is hard-wired to the
LOBSTER NASDAQ format: it expects a separate `message` file per `orderbook` file,
integer tick prices, "seconds-since-midnight" timestamps, NASDAQ market hours
(09:30-16:00), a 5-day rolling z-score, etc. These exchanges export a single combined
snapshot CSV per symbol (datetime stamps, 20 levels, bid/ask interleaved, 24/7
crypto, ~10s cadence). Rather than fake LOBSTER message files, this script produces
the *final* CSVs that the downstream torch dataset loader (loaders/custom_dataset.py)
and evaluation code consume, so the `data_processing` stage can be skipped entirely.

The five exchanges share an identical header but use THREE different book conventions
(normal; nobitex's swapped bid/ask labels; ramzinex's reverse-sorted ask ladder). The
convention is AUTO-DETECTED per file by `detect_orientation` — no per-exchange flag needed.

Output contract (must match loaders/custom_dataset.py & utils.get_best_levels_prices_and_labels)
----------------------------------------------------------------------------------------------
Each processed CSV has columns, in this exact order:

    seconds,
    ASKp1, ASKs1, BIDp1, BIDs1, ... ASKp10, ASKs10, BIDp10, BIDs10,   # 40 features
    Raw_Target_<h1>, Raw_Target_<h2>, ...,                            # raw labels (one per horizon)
    Smooth_Target_<h1>, Smooth_Target_<h2>, ...                       # smooth labels (one per horizon)

  * The loader drops column 0 (`seconds`) via df.values[:, 1:], so features land at
    indices 0..39 and labels at index 40.. .
  * __getitem__ selects the label by *position* of `prediction_horizon` within
    `--horizons`, indexing into the slice [40:]. Hence the Raw_* block (the default
    targets_type) must come first so positions line up.
  * Prices are kept as integer ticks (like LOBSTER), mid-price = (ASKp1+BIDp1)//... ,
    labels are mid-price *differences* in ticks (Raw) / smoothed differences (Smooth).

Normalization
-------------
Per-symbol z-score using statistics computed ONLY on the training portion (no
look-ahead). The same mean/std are applied to validation and test. Unscaled copies
keep the raw integer prices/volumes (needed for the backtest price reconstruction).
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

LEVELS = 10  # LOBFrame's fixed feature width is 4 * 10 = 40.


def detect_orientation(df_raw: pd.DataFrame, levels: int) -> dict:
    """Auto-detect the order-book convention of a raw exchange export.

    These Iranian-exchange CSVs all share the same header
    (time, bid_price_i/bid_volume_i/ask_price_i/ask_volume_i, i=1..20) but use THREE
    different conventions, so a single hard-coded mapping is wrong. We detect, from a
    sample of rows, two independent facts:

      1. `swapped`  - whether the columns labelled `bid_*` actually hold the ASK side.
                      Determined by which labelled side has the HIGHER median price: the
                      bid (buy) side is always the lower-priced side, the ask (sell) side
                      the higher-priced one. If the `bid_*` columns are the higher-priced
                      side, the labels are swapped (Nobitex does this).
      2. ascending  - per true side, whether level 1 is already the BEST quote. We want
                      ASK level 1 = lowest ask, BID level 1 = highest bid. Some exchanges
                      (ramzinex) store the ask ladder reverse-sorted, so the best ask is at
                      level 10; we flip those ladders so level 1 is always best.

    Observed in this dataset:
      bitpin/tabdeal/wallex : normal labels, both ladders best-first  -> no change
      nobitex               : labels swapped, both ladders best-first  -> swap sides
      ramzinex              : normal labels, ASK ladder reverse-sorted -> reverse ask levels
    """
    n = min(len(df_raw), 3000)
    sample = df_raw.iloc[:n]
    bidcols = [f"bid_price_{i}" for i in range(1, levels + 1)]
    askcols = [f"ask_price_{i}" for i in range(1, levels + 1)]
    bid_med = sample[bidcols].to_numpy(dtype=float).mean()
    ask_med = sample[askcols].to_numpy(dtype=float).mean()
    # If the labelled-bid side is the more expensive one, the labels are swapped.
    swapped = bid_med > ask_med

    # After resolving the swap, decide for each TRUE side whether its ladder is best-first.
    # True ASK comes from labelled-bid if swapped else labelled-ask (and vice-versa).
    true_ask_cols = bidcols if swapped else askcols
    true_bid_cols = askcols if swapped else bidcols
    a = sample[true_ask_cols].to_numpy(dtype=float)
    b = sample[true_bid_cols].to_numpy(dtype=float)
    # Best ask = lowest price -> ladder should be ASCENDING (level1 smallest).
    ask_ascending = np.nanmean(np.all(np.diff(a, axis=1) >= 0, axis=1)) > 0.5
    # Best bid = highest price -> ladder should be DESCENDING (level1 largest).
    bid_descending = np.nanmean(np.all(np.diff(b, axis=1) <= 0, axis=1)) > 0.5
    return {
        "swapped": bool(swapped),
        "ask_needs_reverse": not ask_ascending,
        "bid_needs_reverse": not bid_descending,
    }


def build_feature_frame(df_raw: pd.DataFrame, levels: int, orient: dict | None = None) -> pd.DataFrame:
    """Re-shape a raw exchange export into LOBFrame order ASKp{i}, ASKs{i}, BIDp{i}, BIDs{i},
    with level 1 = best quote and the invariant ASKp1 > BIDp1 holding.

    The per-file convention is auto-detected by `detect_orientation` (pass `orient` to reuse a
    previously detected result, e.g. for consistency across day-files of the same symbol).
    """
    if orient is None:
        orient = detect_orientation(df_raw, levels)

    # Map labelled columns to TRUE sides.
    if orient["swapped"]:
        ask_price_src, ask_vol_src = "bid_price_{}", "bid_volume_{}"
        bid_price_src, bid_vol_src = "ask_price_{}", "ask_volume_{}"
    else:
        ask_price_src, ask_vol_src = "ask_price_{}", "ask_volume_{}"
        bid_price_src, bid_vol_src = "bid_price_{}", "bid_volume_{}"

    # Level indices, reversed if the source ladder is not best-first.
    ask_levels = list(range(1, levels + 1))
    bid_levels = list(range(1, levels + 1))
    if orient["ask_needs_reverse"]:
        ask_levels = ask_levels[::-1]
    if orient["bid_needs_reverse"]:
        bid_levels = bid_levels[::-1]

    out = pd.DataFrame()
    out["time"] = pd.to_datetime(df_raw["time"])
    for out_i, (a_lvl, b_lvl) in enumerate(zip(ask_levels, bid_levels), start=1):
        out[f"ASKp{out_i}"] = df_raw[ask_price_src.format(a_lvl)]
        out[f"ASKs{out_i}"] = df_raw[ask_vol_src.format(a_lvl)]
        out[f"BIDp{out_i}"] = df_raw[bid_price_src.format(b_lvl)]
        out[f"BIDs{out_i}"] = df_raw[bid_vol_src.format(b_lvl)]
    return out


def feature_names(levels: int) -> list[str]:
    names = []
    for i in range(1, levels + 1):
        names += [f"ASKp{i}", f"ASKs{i}", f"BIDp{i}", f"BIDs{i}"]
    return names


def make_labels(mid: np.ndarray, horizons: list[int]) -> dict[str, np.ndarray]:
    """Replicate LOBFrame's labelling in tick units.
    Raw_Target_h  = mid[t+h] - mid[t]
    Smooth_Target_h = mean(mid[t+1..t+h]) - mean(mid[t-h+1..t])
    Rows without a full forward/backward window are left as NaN and dropped later.
    """
    n = len(mid)
    mid_s = pd.Series(mid.astype(np.float64))
    labels = {}
    for h in horizons:
        raw = np.full(n, np.nan)
        raw[:-h] = mid[h:] - mid[:-h]
        labels[f"Raw_Target_{h}"] = raw
    for h in horizons:
        rolling_mean = mid_s.rolling(window=h, min_periods=h).mean()
        rolling_mid_minus = rolling_mean.shift(h).to_numpy()
        rolling_mid_plus = rolling_mean.to_numpy()
        labels[f"Smooth_Target_{h}"] = rolling_mid_plus - rolling_mid_minus
    return labels


def main():
    p = argparse.ArgumentParser(
        description="Convert an Iranian-exchange order-book CSV (Nobitex/Bitpin/Ramzinex/"
                    "Tabdeal/Wallex) to LOBFrame format. Book convention is auto-detected.")
    p.add_argument("--input_csv", required=True, help="Path to <SYMBOL>_orderbook.csv")
    p.add_argument("--symbol", required=True, help="Symbol name used in filenames, e.g. BTCIRT")
    p.add_argument("--dataset", default="nasdaq", help="LOBFrame dataset folder name (default: nasdaq)")
    p.add_argument("--horizons", default="10,50,100", help="Comma-separated horizons in snapshots")
    p.add_argument("--training_ratio", type=float, default=0.6)
    p.add_argument("--validation_ratio", type=float, default=0.2)
    p.add_argument("--clean", action="store_true", help="Wipe existing split folders for this dataset first")
    args = p.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    fnames = feature_names(LEVELS)

    print(f"Reading {args.input_csv} ...")
    df_raw = pd.read_csv(args.input_csv)
    orient = detect_orientation(df_raw, LEVELS)
    print(f"Detected book convention: swapped_labels={orient['swapped']}, "
          f"ask_ladder_reversed={orient['ask_needs_reverse']}, "
          f"bid_ladder_reversed={orient['bid_needs_reverse']}")
    df = build_feature_frame(df_raw, LEVELS, orient=orient).reset_index(drop=True)

    # Remove crossed / locked books (ASKp1 <= BIDp1) and any non-positive prices.
    before = len(df)
    df = df[df["ASKp1"] > df["BIDp1"]]
    df = df[(df["ASKp1"] > 0) & (df["BIDp1"] > 0)]
    df = df.reset_index(drop=True)
    print(f"Dropped {before - len(df)} crossed/invalid rows; {len(df)} remain.")

    # Mid-price in ticks (integer), exactly as LOBFrame derives it.
    mid = ((df["ASKp1"] + df["BIDp1"]) / 2).round().astype(np.int64).to_numpy()

    labels = make_labels(mid, horizons)
    for col, vals in labels.items():
        df[col] = vals

    # Chronological split.
    n = len(df)
    n_train = int(n * args.training_ratio)
    n_val = int(n * args.validation_ratio)
    splits = {
        "training": df.iloc[:n_train].copy(),
        "validation": df.iloc[n_train:n_train + n_val].copy(),
        "test": df.iloc[n_train + n_val:].copy(),
    }
    print(f"Split sizes -> training: {len(splits['training'])}, "
          f"validation: {len(splits['validation'])}, test: {len(splits['test'])}")

    # Per-symbol z-score stats from the TRAINING portion only (no look-ahead).
    train_feats = splits["training"][fnames]
    mu = train_feats.mean()
    sigma = train_feats.std().replace(0, 1.0)  # guard against zero-variance columns

    label_cols = [f"Raw_Target_{h}" for h in horizons] + [f"Smooth_Target_{h}" for h in horizons]
    ordered_cols = ["seconds"] + fnames + label_cols

    base = f"./data/{args.dataset}"
    for scaled in (True, False):
        root = f"{base}/{'scaled_data' if scaled else 'unscaled_data'}"
        for stage in ("training", "validation", "test"):
            folder = f"{root}/{stage}"
            if args.clean and os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)

            part = splits[stage].copy()
            # Drop rows with NaN labels (forward/backward window not available).
            part = part.dropna(subset=label_cols).reset_index(drop=True)

            # 'seconds' column carries a datetime string (loader drops it; evaluation
            # keeps it verbatim for plotting/backtest indexing).
            part.insert(0, "seconds", part["time"].astype(str))

            feat_block = part[fnames].astype(np.float64)
            if scaled:
                feat_block = (feat_block - mu) / sigma
            part[fnames] = feat_block

            out = part[ordered_cols]
            # One file per calendar day so the loader's per-file logic behaves naturally.
            part_dates = pd.to_datetime(part["time"]).dt.date
            for day, idx in part_dates.groupby(part_dates).groups.items():
                day_df = out.loc[idx]
                fname = f"{folder}/{args.symbol}_orderbooks_{day}.csv"
                day_df.to_csv(fname, index=False)
                print(f"  wrote {fname} ({len(day_df)} rows)")

    print("Done. Feature columns:", len(fnames), "| label columns:", len(label_cols))


if __name__ == "__main__":
    main()
