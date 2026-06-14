"""
Wallex LOB & Trade Data Collector
====================================
Collects order book (depth 20) and recent trades for BTCTMN and USDTTMN
every 10 seconds. Press Ctrl+C to stop.

API docs: https://api-docs.wallex.ir/
Base URL:  https://api.wallex.ir

Endpoints used (no auth required):
  GET /v1/depth?symbol={symbol}   → order book
  GET /v1/trades?symbol={symbol}  → recent trades

Output files (in wallex_data/ folder):
  - BTCTMN_orderbook.csv
  - USDTTMN_orderbook.csv
  - BTCTMN_trades.csv
  - USDTTMN_trades.csv
"""

import requests
import csv
import time
import os
from datetime import datetime, timezone

# Config
SYMBOLS      = ["BTCTMN", "USDTTMN"]   # BTC/Toman and USDT/Toman
INTERVAL_SEC = 10
LOB_DEPTH    = 20
BASE_URL     = "https://api.wallex.ir"
OUTPUT_DIR   = "wallex_data"

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

TRADES_HEADER = ["snapshot_time", "trade_time", "direction", "price", "volume"]

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
    GET /v1/depth?symbol={symbol}
    Response: { "success": true, "result": { "ask": [...], "bid": [...] } }
    Each level: { "price": "...", "quantity": ..., "sum": "..." }
    NOTE: bids/asks are directly inside result, NOT nested under the symbol name.
    """
    try:
        r = requests.get(f"{BASE_URL}/v1/depth", params={"symbol": symbol}, timeout=8)
        r.raise_for_status()
        data = r.json()
        if data.get("success"):
            return data["result"]   # flat: {"ask": [...], "bid": [...]}
        print(f"[warn] orderbook {symbol}: {data}")
    except Exception as e:
        print(f"[error] orderbook {symbol}: {e}")
    return None

def fetch_trades(symbol):
    """
    GET /v1/trades?symbol={symbol}
    Response: { "success": true, "result": { "latestTrades": [
      { "symbol", "quantity", "price", "sum", "isBuyOrder", "timestamp" }, ...
    ] } }
    timestamp is an ISO-8601 string e.g. "2022-06-17T11:53:02Z"
    """
    try:
        r = requests.get(f"{BASE_URL}/v1/trades", params={"symbol": symbol}, timeout=8)
        r.raise_for_status()
        data = r.json()
        if data.get("success"):
            return data["result"].get("latestTrades", [])
        print(f"[warn] trades {symbol}: {data}")
    except Exception as e:
        print(f"[error] trades {symbol}: {e}")
    return None


# Persistence helpers

# Wallex trades have no numeric ID — deduplicate using ISO timestamp watermark.
# We track the latest timestamp seen and only save trades strictly newer than it.
_last_trade_ts: dict[str, str] = {sym: "" for sym in SYMBOLS}

def save_orderbook(symbol, data, ts):
    bids = data.get("bid", [])
    asks = data.get("ask", [])

    row = [ts]
    for i in range(LOB_DEPTH):
        bid = bids[i] if i < len(bids) else {}
        ask = asks[i] if i < len(asks) else {}
        row += [
            bid.get("price", ""), bid.get("quantity", ""),
            ask.get("price", ""), ask.get("quantity", ""),
        ]

    with open(orderbook_csv_path(symbol), "a", newline="") as f:
        csv.writer(f).writerow(row)

def save_trades(symbol, trades, snapshot_ts):
    if not trades:
        return

    last_ts = _last_trade_ts[symbol]
    # ISO strings sort lexicographically correctly (UTC Z format)
    new_trades = [t for t in trades if t.get("timestamp", "") > last_ts]

    if not new_trades:
        return

    _last_trade_ts[symbol] = max(t["timestamp"] for t in new_trades)

    with open(trades_csv_path(symbol), "a", newline="") as f:
        writer = csv.writer(f)
        for t in new_trades:
            direction = "buy" if t.get("isBuyOrder") else "sell"
            writer.writerow([
                snapshot_ts,
                t.get("timestamp", ""),  # ISO time of the trade itself
                direction,
                t.get("price", ""),
                t.get("quantity", ""),
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
            bid1 = ob["bid"][0] if ob.get("bid") else {}
            ask1 = ob["ask"][0] if ob.get("ask") else {}
            print(f"  [{sym}] best bid={bid1.get('price','N/A')}  best ask={ask1.get('price','N/A')}")

        trades = fetch_trades(sym)
        if trades is not None:
            save_trades(sym, trades, ts)


def main():
    print("=" * 55)
    print("  Wallex Collector — BTCTMN & USDTTMN")
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