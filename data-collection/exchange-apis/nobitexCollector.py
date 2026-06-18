"""
Nobitex LOB & Trade Data Collector
====================================
Collects order book (depth 20) and recent trades for BTCIRT, USDTIRT,
ETHIRT, BNBIRT and XRPIRT every 10 seconds.

Output files (in nobitex_data/ folder):
  - BTCIRT_orderbook.csv
  - USDTIRT_orderbook.csv
  - ETHIRT_orderbook.csv
  - BNBIRT_orderbook.csv
  - XRPIRT_orderbook.csv
  - BTCIRT_trades.csv
  - USDTIRT_trades.csv
  - ETHIRT_trades.csv
  - BNBIRT_trades.csv
  - XRPIRT_trades.csv
"""

import requests
import csv
import time
import os
from datetime import datetime, timezone

# Config
SYMBOLS      = ["BTCIRT", "USDTIRT", "ETHIRT", "BNBIRT", "XRPIRT"]
INTERVAL_SEC = 10
LOB_DEPTH    = 20
BASE_URL     = "https://apiv2.nobitex.ir"
OUTPUT_DIR   = "nobitex_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# CSV helpers

def orderbook_csv_path(symbol):
    return os.path.join(OUTPUT_DIR, f"{symbol}_orderbook.csv")

def trades_csv_path(symbol):
    return os.path.join(OUTPUT_DIR, f"{symbol}_trades.csv")

def build_orderbook_header():
    header = ["time"]
    for i in range(1, LOB_DEPTH + 1):
        header += [f"bid_price_{i}", f"bid_volume_{i}",
                   f"ask_price_{i}", f"ask_volume_{i}"]
    return header

TRADES_HEADER = ["snapshot_time", "trade_time", "price", "volume", "direction"]

def ensure_headers():
    ob_header = build_orderbook_header()
    for sym in SYMBOLS:
        ob_path = orderbook_csv_path(sym)
        if not os.path.exists(ob_path):
            with open(ob_path, "w", newline="") as f:
                csv.writer(f).writerow(ob_header)
            print(f"[init] Created {ob_path}")

        tr_path = trades_csv_path(sym)
        if not os.path.exists(tr_path):
            with open(tr_path, "w", newline="") as f:
                csv.writer(f).writerow(TRADES_HEADER)
            print(f"[init] Created {tr_path}")


# API calls

def fetch_orderbook(symbol):
    try:
        r = requests.get(f"{BASE_URL}/v2/orderbook/{symbol}", timeout=8)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "ok":
            return data
        print(f"[warn] orderbook {symbol}: {data}")
    except Exception as e:
        print(f"[error] orderbook {symbol}: {e}")
    return None

def fetch_trades(symbol):
    """GET /v2/trades/{symbol} — returns recent trades with time, price, volume, type."""
    try:
        r = requests.get(f"{BASE_URL}/v2/trades/{symbol}", timeout=8)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "ok":
            return data
        print(f"[warn] trades {symbol}: {data}")
    except Exception as e:
        print(f"[error] trades {symbol}: {e}")
    return None


# Persistence helpers

# Track the last seen trade timestamp per symbol to avoid duplicates
# Nobitex trade 'time' is a Unix timestamp in milliseconds
_last_trade_time = {sym: 0 for sym in SYMBOLS}

def save_orderbook(symbol, data, ts):
    bids = data.get("bids", [])
    asks = data.get("asks", [])

    row = [ts]
    for i in range(LOB_DEPTH):
        bid = bids[i] if i < len(bids) else ["", ""]
        ask = asks[i] if i < len(asks) else ["", ""]
        row += [bid[0], bid[1], ask[0], ask[1]]

    with open(orderbook_csv_path(symbol), "a", newline="") as f:
        csv.writer(f).writerow(row)

def save_trades(symbol, data, snapshot_ts):
    trades = data.get("trades", [])
    if not trades:
        return

    last_t = _last_trade_time[symbol]

    # Keep only trades newer than what we've already saved
    new_trades = [t for t in trades if int(t.get("time", 0)) > last_t]

    if not new_trades:
        return

    # Update the watermark to the newest trade time seen
    _last_trade_time[symbol] = max(int(t["time"]) for t in new_trades)

    with open(trades_csv_path(symbol), "a", newline="") as f:
        writer = csv.writer(f)
        for t in new_trades:
            # Convert Unix-ms timestamp to readable UTC string
            trade_ts = datetime.fromtimestamp(
                int(t["time"]) / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]   # millisecond precision

            writer.writerow([
                snapshot_ts,            # when we polled
                trade_ts,               # when the trade actually happened
                t.get("price", ""),
                t.get("volume", ""),
                t.get("type", ""),      # "buy" or "sell"
            ])

    print(f"  [{symbol}] +{len(new_trades)} new trade(s)")


# Main loop

def collect_once():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{ts} UTC] Collecting ...")

    for sym in SYMBOLS:
        ob = fetch_orderbook(sym)
        if ob:
            save_orderbook(sym, ob, ts)
            bid1 = ob["bids"][0] if ob["bids"] else ["N/A", "N/A"]
            ask1 = ob["asks"][0] if ob["asks"] else ["N/A", "N/A"]
            print(f"  [{sym}] best bid={bid1[0]}  best ask={ask1[0]}")

        tr = fetch_trades(sym)
        if tr:
            save_trades(sym, tr, ts)


def main():
    print("=" * 55)
    print("  Nobitex Collector — BTCIRT, USDTIRT, ETHIRT, BNBIRT & XRPIRT")
    print(f"  Interval : {INTERVAL_SEC}s   |   LOB depth : {LOB_DEPTH}")
    print(f"  Output   : ./{OUTPUT_DIR}/")
    print("=" * 55)

    ensure_headers()

    try:
        # ── Deadline-based loop ──────────────────────────────────────
        # Record the target time for the NEXT tick before doing any work.
        # After collection, sleep only the *remaining* time to that target.
        # This keeps the wall-clock gap between records at exactly INTERVAL_SEC
        # regardless of how long the API calls take.
        next_tick = time.monotonic()

        while True:
            next_tick += INTERVAL_SEC          # advance the deadline first
            collect_once()

            # How long until the next tick?  (positive = still time to sleep)
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Collection overran the interval — skip sleeping, warn the user
                print(f"[warn] collection took >{INTERVAL_SEC}s "
                      f"(overran by {-sleep_for:.2f}s); next tick fires immediately")

    except KeyboardInterrupt:
        print("\n\nStopped by user. Data saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()