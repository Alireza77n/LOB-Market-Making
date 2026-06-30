# LOB Monitoring Dashboard (single venue — Nobitex)

A Plotly Dash dashboard over the 10s-polled L2 snapshots and direction-labelled
trades produced by `data-collection/exchange-apis/nobitexCollector.py`.

Implements the full v1 metric spec in
[`../lob_dashboard_metrics.md`](../lob_dashboard_metrics.md) — Section A
(numeric tiles) and Section B (time-series & book-shape plots), including the
depth heatmap.

## Quick start

```bash
# 1. install deps
pip install -r monitoring/requirements.txt

# 2. start collecting (in another terminal), from anywhere you like
cd data-collection/exchange-apis && python nobitexCollector.py

# 3. launch the dashboard (from the repo root)
python -m monitoring.app
# open http://127.0.0.1:8050
```

The app refreshes every 10s to match the collector's poll cadence. Pick a symbol
(BTCIRT / USDTIRT / ETHIRT / BNBIRT / XRPIRT) from the dropdown.

## Where it reads data

By default it looks in
`data-collection/exchange-apis/nobitex_data/` (where the collector writes when
run from that folder). If you run the collector elsewhere, point the dashboard
at the data with an env var:

```bash
LOB_DATA_DIR=/path/to/nobitex_data python -m monitoring.app
```

It expects the collector's schema:
- `{SYMBOL}_orderbook.csv` — `time` + 20 bid/ask price-volume pairs
- `{SYMBOL}_trades.csv` — `snapshot_time, trade_time, price, volume, direction`

Millisecond timestamps (current collector) and second timestamps (older data)
are both handled.

## What's on screen

**Tiles (Section A)** — best bid/ask, mid, microprice, quoted spread (abs &
bps), top-N imbalance (N=1/5/20), bid/ask depth, depth within ±X bps, buy/sell
slippage for representative sizes, last trade, trade intensity, net flow, flow
imbalance, VWAP, TWAP, realized vol, VPIN.

**Plots (Section B)** — mid/microprice, spread(bps), imbalance, trade-flow
imbalance, order-book depth (cumulative), book ladder, **depth heatmap**, buy/sell
volume bars, realized-vol, VPIN, plus a scrolling trade tape.

## Tuning

All windows and per-symbol parameters (slippage sizes, VPIN bucket volumes) live
in [`config.py`](config.py). The VPIN bucket size in particular must be
calibrated per symbol — USDT/IRT needs a far larger bucket than XRP/IRT.

## Module layout

| File | Role |
|---|---|
| `config.py` | paths, symbols, windows, per-symbol parameters |
| `data_loader.py` | CSV → tidy pandas frames; freshness/health |
| `metrics.py` | pure metric/series computations (Sections A & B) |
| `figures.py` | Plotly figure builders |
| `app.py` | Dash layout + refresh callback |
| `assets/style.css` | dark theme (auto-loaded by Dash) |

## Notes & caveats

- Metrics marked 🟡 in the spec (realized vol, VPIN, effective spread) are
  approximate at 10s resolution — they miss intra-poll moves.
- Single venue for now. The data/metrics/figures split is deliberately
  venue-agnostic so a second venue (and cross-venue Section C) can be added
  later without touching the compute layer.
- Uses the Dash development server. Fine for local monitoring; put a real WSGI
  server in front of `app.server` if you ever expose it.
