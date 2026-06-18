# Exchange API Scripts

Scripts for live LOB and trade data collection from Iranian crypto exchanges.
Each collector polls its exchange every 10 seconds and appends to CSV files.

## Structure

| File                    | Exchange  | Symbols                                              | Output dir       |
|-------------------------|-----------|-------------------------------------------------------|------------------|
| `nobitexCollector.py`   | Nobitex   | BTCIRT, USDTIRT, ETHIRT, BNBIRT, XRPIRT                | `nobitex_data/`  |
| `bitpinCollector.py`    | Bitpin    | BTC_IRT, USDT_IRT, ETH_IRT, BNB_IRT, XRP_IRT           | `bitpin_data/`   |
| `wallexCollector.py`    | Wallex    | BTCTMN, USDTTMN, ETHTMN, BNBTMN, XRPTMN                | `wallex_data/`   |
| `ramzinexCollector.py`  | Ramzinex  | BTC_IRT, USDT_IRT, ETH_IRT, BNB_IRT, XRP_IRT           | `ramzinex_data/` |
| `tabdealCollector.py`   | Tabdeal   | BTCIRT, USDTIRT, ETHIRT, BNBIRT, XRPIRT                | `tabdeal_data/`  |

## Output files

Each collector writes two CSV files per symbol (10 files per run — all five exchanges now track 5 symbols each):

```
{exchange}_data/
  {SYMBOL}_orderbook.csv   # LOB snapshot: time + 20 bid/ask price-volume pairs
  {SYMBOL}_trades.csv      # deduplicated recent trades with direction
```

## Running a collector

```bash
python nobitexCollector.py   # or any other collector
# Press Ctrl+C to stop
```

## Deduplication strategy

| Exchange  | Method                                      |
|-----------|---------------------------------------------|
| Nobitex   | Unix-ms timestamp watermark per symbol      |
| Bitpin    | (time, price, amount) set; capped at 5 000  |
| Wallex    | ISO-8601 timestamp watermark per symbol     |
| Ramzinex  | Integer trade ID watermark per symbol       |
| Tabdeal   | Integer trade ID watermark per symbol       |

## Notes

- No authentication required for any endpoint used.
- The polling loop is deadline-based: collection time is subtracted from the sleep, keeping snapshots exactly 10 s apart.
- Raw CSV files land in their respective output directories next to where the script is run.
