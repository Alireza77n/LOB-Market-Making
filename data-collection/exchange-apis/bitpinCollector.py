"""
Bitpin LOB & Trade Data Collector
====================================
Collects order book (depth 20) and recent trades for BTC_IRT, USDT_IRT,
ETH_IRT, BNB_IRT and XRP_IRT every 10 seconds. Press Ctrl+C to stop.

API docs: https://docs.bitpin.ir/
Base URL:  https://api.bitpin.ir  (also api.bitpin.org)

Endpoints used (no auth required):
  GET /v1/mth/orderbook/{symbol}/  → order book
  GET /v1/mth/matches/{symbol}/    → recent trades

Output files (in bitpin_data/ folder):
  - BTC_IRT_orderbook.csv
  - USDT_IRT_orderbook.csv
  - ETH_IRT_orderbook.csv
  - BNB_IRT_orderbook.csv
  - XRP_IRT_orderbook.csv
  - BTC_IRT_trades.csv
  - USDT_IRT_trades.csv
  - ETH_IRT_trades.csv
  - BNB_IRT_trades.csv
  - XRP_IRT_trades.csv
"""

import requests
import csv
import time
import os
from datetime import datetime, timezone

# Config
SYMBOLS      = ["BTC_IRT", "USDT_IRT", "ETH_IRT", "BNB_IRT", "XRP_IRT"]
INTERVAL_SEC = 10
LOB_DEPTH    = 20
BASE_URL     = "https://api.bitpin.org/api"
OUTPUT_DIR   = "bitpin_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Timestamp helper

def now_ms():
    """UTC wall-clock to millisecond precision, e.g. '2026-06-26 14:09:42.317'.

    Stamped at each fetch site (not once per loop) so every order-book and
    trade snapshot carries the instant it was actually observed — a
    prerequisite for intra-poll ordering and lead-lag analysis.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


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
    """
    GET /v1/mth/orderbook/{symbol}/
    Response: { "asks": [["price", "volume"], ...], "bids": [["price", "volume"], ...] }
    """
    try:
        r = requests.get(f"{BASE_URL}/v1/mth/orderbook/{symbol}/", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[error] orderbook {symbol}: {e}")
    return None

def fetch_trades(symbol):
    """
    GET /v1/mth/matches/{symbol}/
    Response: list of trades, each with: { "time": "ISO datetime", "price": "...", "amount": "...", "type": 0/1 }
    type: 0 = buy (taker), 1 = sell (taker)
    """
    try:
        r = requests.get(f"{BASE_URL}/v1/mth/matches/{symbol}/", timeout=8)
        r.raise_for_status()
        data = r.json()
        # Response may be paginated: { "results": [...] } or a plain list
        if isinstance(data, dict):
            return data.get("results", [])
        return data
    except Exception as e:
        print(f"[error] trades {symbol}: {e}")
    return None


# Persistence helpers

# Use (time, price, amount) as a deduplication key
_seen_trades: dict[str, set] = {sym: set() for sym in SYMBOLS}

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

def save_trades(symbol, trades, snapshot_ts):
    if not trades:
        return

    new_trades = []
    for t in trades:
        key = (t.get("time"), t.get("price"), t.get("amount"))
        if key not in _seen_trades[symbol]:
            _seen_trades[symbol].add(key)
            new_trades.append(t)

    if not new_trades:
        return

    if len(_seen_trades[symbol]) > 5000:
        _seen_trades[symbol] = set(
            (t.get("time"), t.get("price"), t.get("amount"))
            for t in new_trades
        )

    with open(trades_csv_path(symbol), "a", newline="") as f:
        writer = csv.writer(f)
        for t in new_trades:
            # type: 0 = buy-initiated, 1 = sell-initiated # type: ignore
            direction = "buy" if t.get("type") == 0 else "sell"
            writer.writerow([
                snapshot_ts,
                t.get("time", ""),
                t.get("price", ""),
                t.get("amount", ""),
                direction,
            ])

    print(f"  [{symbol}] +{len(new_trades)} new trade(s)")


# Main loop

def collect_once():
    print(f"\n[{now_ms()} UTC] Collecting ...")

    for sym in SYMBOLS:
        ob = fetch_orderbook(sym)
        ob_ts = now_ms()                      # stamp this fetch individually
        if ob:
            save_orderbook(sym, ob, ob_ts)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            bid1 = bids[0][0] if bids else "N/A"
            ask1 = asks[0][0] if asks else "N/A"
            print(f"  [{sym}] best bid={bid1}  best ask={ask1}")

        trades = fetch_trades(sym)
        tr_ts = now_ms()                      # stamp this fetch individually
        if trades is not None:
            save_trades(sym, trades, tr_ts)


def main():
    print("=" * 55)
    print("  Bitpin Collector — BTC_IRT, USDT_IRT, ETH_IRT, BNB_IRT & XRP_IRT")
    print(f"  Interval : {INTERVAL_SEC}s   |   LOB depth : {LOB_DEPTH}")
    print(f"  Output   : ./{OUTPUT_DIR}/")
    print("  Press Ctrl+C to stop.")
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
