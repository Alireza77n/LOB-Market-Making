"""
Tabdeal LOB & Trade Data Collector
====================================
Collects order book (depth 20) and recent trades for BTCIRT, USDTIRT,
ETHIRT, BNBIRT and XRPIRT every 10 seconds. Press Ctrl+C to stop.

API docs: https://docs.tabdeal.org/
Base URL:  https://api1.tabdeal.org

Endpoints used (no auth — security type [NONE]):
  GET /r/api/v1/depth?symbol={symbol}&limit={n}   → order book
  GET /r/api/v1/trades?symbol={symbol}&limit={n}  → recent trades

Symbol format: use tabdealSymbol with underscore e.g. BTCIRT, USDTIRT

Output files (in tabdeal_data/ folder):
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
BASE_URL     = "https://api1.tabdeal.org"
OUTPUT_DIR   = "tabdeal_data"

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

TRADES_HEADER = ["snapshot_time", "trade_id", "price", "volume", "direction"]

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
    GET /r/api/v1/depth?symbol={symbol}&limit={n}
    Response: {
      "bids": [["price", "qty"], ...],
      "asks": [["price", "qty"], ...]
    }
    """
    try:
        r = requests.get(
            f"{BASE_URL}/r/api/v1/depth",
            params={"symbol": symbol, "limit": LOB_DEPTH},
            timeout=8,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[error] orderbook {symbol}: {e}")
    return None

def fetch_trades(symbol):
    """
    GET /r/api/v1/trades?symbol={symbol}&limit={n}
    Response: list of trades, each with:
      { "id", "price", "qty", "time", "isBuyerMaker" }
    isBuyerMaker: true = seller-initiated (sell), false = buyer-initiated (buy)
    """
    try:
        r = requests.get(
            f"{BASE_URL}/r/api/v1/trades",
            params={"symbol": symbol, "limit": 50},
            timeout=8,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[error] trades {symbol}: {e}")
    return None


# Persistence helpers

# Tabdeal trades have an "id" field — use it for deduplication
_last_trade_id: dict[str, int] = {sym: 0 for sym in SYMBOLS}

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

    last_id = _last_trade_id[symbol]
    new_trades = [t for t in trades if int(t.get("id", 0)) > last_id]

    if not new_trades:
        return

    _last_trade_id[symbol] = max(int(t["id"]) for t in new_trades)

    with open(trades_csv_path(symbol), "a", newline="") as f:
        writer = csv.writer(f)
        for t in new_trades:
            # isBuyerMaker=True means sell-initiated; False means buy-initiated
            direction = "sell" if t.get("isBuyerMaker") else "buy"
            writer.writerow([
                snapshot_ts,
                t.get("id", ""),
                t.get("price", ""),
                t.get("qty", ""),
                direction,
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
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            bid1 = bids[0][0] if bids else "N/A"
            ask1 = asks[0][0] if asks else "N/A"
            print(f"  [{sym}] best bid={bid1}  best ask={ask1}")

        trades = fetch_trades(sym)
        if trades is not None:
            save_trades(sym, trades, snapshot_ts=ts)


def main():
    print("=" * 55)
    print("  Tabdeal Collector — BTCIRT, USDTIRT, ETHIRT, BNBIRT & XRPIRT")
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
