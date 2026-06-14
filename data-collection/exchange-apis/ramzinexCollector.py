"""
Ramzinex LOB & Trade Data Collector
====================================
Collects order book (depth 20) and recent trades for BTC/IRT and USDT/IRT
every 10 seconds. Press Ctrl+C to stop.

Pair IDs are hardcoded from the currencies endpoint:
  BTC  / IRT  → pair_id = 2   (bitcoin rial_related_pair)
  USDT / IRT  → pair_id = 11  (tether  rial_related_pair)

Endpoints (no auth required):
  GET https://publicapi.ramzinex.com/exchange/api/v1.0/exchange/orderbooks/{pair_id}/buys_sells
      → order book: { "status": 0, "data": { "buys": [...], "sells": [...] } }
        each entry: { "price": "...", "amount": "..." }

  GET https://publicapi.ramzinex.com/exchange/api/v1.0/exchange/orderbooks/{pair_id}/trades
      → trades: { "status": 0, "data": [ { "id", "price", "amount", "type": 0|1, "created_at" }, ... ] }
        type: 0 = sell, 1 = buy

Output files (in ramzinex_data/):
  BTC_IRT_orderbook.csv
  BTC_IRT_trades.csv
  USDT_IRT_orderbook.csv
  USDT_IRT_trades.csv
"""

import requests
import csv
import time
import os
import sys
from datetime import datetime, timezone

# Config
PAIRS = {
    "BTC_IRT":  2,
    "USDT_IRT": 11,
}
SYMBOLS      = list(PAIRS.keys())
INTERVAL_SEC = 10
LOB_DEPTH    = 20
BASE_URL     = "https://publicapi.ramzinex.com/exchange/api/v1.0/exchange"
OUTPUT_DIR   = "ramzinex_data"

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

TRADES_HEADER = ["snapshot_time", "trade_time", "trade_id", "direction", "price", "volume"]

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

def fetch_orderbook(pair_id):
    """
    GET /orderbooks/{pair_id}/buys_sells
    Response:
    {
      "status": 0,
      "data": {
        "buys":  [ {"price": "...", "amount": "..."}, ... ],   <- bids
        "sells": [ {"price": "...", "amount": "..."}, ... ]    <- asks
      }
    }
    """
    try:
        r = requests.get(f"{BASE_URL}/orderbooks/{pair_id}/buys_sells", timeout=8)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == 0:
            return data.get("data", {})
        print(f"[warn] orderbook pair {pair_id}: status={data.get('status')}")
    except Exception as e:
        print(f"[error] orderbook pair {pair_id}: {e}")
    return None

def fetch_trades(pair_id):
    """
    GET /orderbooks/{pair_id}/trades
    Response:
    {
      "status": 0,
      "data": [
        { "id": 123, "price": "...", "amount": "...", "type": 0|1, "created_at": "..." },
        ...
      ]
    }
    type: 0 = sell-initiated, 1 = buy-initiated
    """
    try:
        r = requests.get(f"{BASE_URL}/orderbooks/{pair_id}/trades", timeout=8)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == 0:
            return data.get("data") or []
        print(f"[warn] trades pair {pair_id}: status={data.get('status')}")
    except Exception as e:
        print(f"[error] trades pair {pair_id}: {e}")
    return []


# Persistence

_last_trade_id: dict[str, int] = {sym: 0 for sym in SYMBOLS}

def _parse_trade(t):
    """
    Ramzinex trade entries may be a dict or a list.
    Observed list format: [price, amount, created_at, direction_str, id, hash]
      direction_str: "buy" or "sell"
    Returns a normalised dict with a string "type" of "buy" or "sell".
    """
    if isinstance(t, dict):
        return t
    try:
        return {
            "id":         t[4],
            "price":      t[0],
            "amount":     t[1],
            "type":       t[3],   # "buy" or "sell" string
            "created_at": t[2] if len(t) > 2 else "",
        }
    except (IndexError, TypeError):
        return {}

def _price_vol(entry):
    """
    Ramzinex orderbook entries can be either:
      - a dict:  {"price": "...", "amount": "..."}
      - a list:  ["price_value", "amount_value"]
    Returns (price, volume) strings.
    """
    if isinstance(entry, dict):
        return entry.get("price", ""), entry.get("amount", "")
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return entry[0], entry[1]
    return "", ""

def save_orderbook(symbol, data, ts):
    bids = data.get("buys",  [])   # bid side
    asks = data.get("sells", [])   # ask side

    row = [ts]
    for i in range(LOB_DEPTH):
        bid = bids[i] if i < len(bids) else []
        ask = asks[i] if i < len(asks) else []
        bp, bv = _price_vol(bid)
        ap, av = _price_vol(ask)
        row += [bp, bv, ap, av]

    with open(orderbook_csv_path(symbol), "a", newline="") as f:
        csv.writer(f).writerow(row)

def save_trades(symbol, trades, snapshot_ts):
    if not trades:
        return

    # Normalise all entries to dicts
    trades = [_parse_trade(t) for t in trades]

    last_id    = _last_trade_id[symbol]
    new_trades = [t for t in trades if int(t.get("id", 0)) > last_id]

    if not new_trades:
        return

    _last_trade_id[symbol] = max(int(t["id"]) for t in new_trades)

    with open(trades_csv_path(symbol), "a", newline="") as f:
        writer = csv.writer(f)
        for t in new_trades:
            raw_type   = t.get("type", "")
            if raw_type in ("buy", "sell"):
                direction = raw_type
            else:
                try:
                    trade_type = int(raw_type)
                    direction  = "buy" if trade_type == 1 else ("sell" if trade_type == 0 else "?")
                except (ValueError, TypeError):
                    direction = "?"
            writer.writerow([
                snapshot_ts,
                t.get("created_at", ""),
                t.get("id", ""),
                direction,
                t.get("price", ""),
                t.get("amount", ""),
            ])

    print(f"  [{symbol}] +{len(new_trades)} new trade(s)")


# Main loop

def collect_once():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] Collecting ...")

    for sym, pair_id in PAIRS.items():
        ob = fetch_orderbook(pair_id)
        if ob is not None:
            save_orderbook(sym, ob, ts)
            bids = ob.get("buys",  [])
            asks = ob.get("sells", [])
            bid1 = _price_vol(bids[0])[0] if bids else "N/A"
            ask1 = _price_vol(asks[0])[0] if asks else "N/A"
            print(f"  [{sym}] best bid={bid1}  best ask={ask1}")

        trades = fetch_trades(pair_id)
        save_trades(sym, trades, snapshot_ts=ts)


def main():
    print("=" * 55)
    print("  Ramzinex Collector — BTC_IRT & USDT_IRT")
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