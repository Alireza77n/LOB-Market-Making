# LOB Market Making

Research repository on Limit Order Book (LOB) market-making, spanning three methodological tracks:

1. **Machine Learning**
2. **Optimal Control**
3. **Custom Algorithms**

---

## Repository Layout

```
LOB-Market-Making/
│
├── literature/                 # PDFs, summaries, and notes
│   ├── foundational/           # Glosten-Milgrom, Kyle, Ho-Stoll, A&S
│   ├── market-microstructure/  # Price impact, adverse selection, queue models
│   ├── optimal-control/        # HJB, Cartea-Jaimungal, inventory risk
│   ├── ml-approaches/          # RL, deep learning, supervised LOB models
│   ├── custom-notes/           # Internal derivations and research notes
│   └── INDEX.md                # Annotated bibliography
│
├── data-collection/            # Scripts that pull data from exchanges
│   ├── exchange-apis/          # Per-exchange WebSocket / REST scripts
│   ├── parsers/                # LOB diff → full snapshot reconstructors
│   └── storage/                # Writers for Parquet / HDF5 / TimescaleDB
│
├── data/
│   ├── raw/                    # Unprocessed exchange data (gitignored)
│   ├── processed/              # Cleaned, normalised LOB snapshots
│   └── snapshots/              # Point-in-time full book states
│
├── lob-simulator/              # Synthetic LOB / agent-based market simulator
│
├── models/
│   ├── baselines/              # Fixed-spread, symmetric, naive strategies
│   ├── optimal-control/        # A&S, Guéant-Lehalle, HJB solvers
│   ├── ml/                     # RL agents, feature models, predictors
│   └── custom/                 # Proprietary strategies
│
├── backtesting/
│   ├── engines/                # Event-driven backtesting infrastructure
│   ├── scenarios/              # Market regime / stress-test configurations
│   └── results/                # Saved backtest outputs (gitignored)
│
├── analysis/
│   ├── notebooks/              # Exploratory Jupyter notebooks
│   ├── plots/                  # Generated figures
│   └── metrics/                # PnL, Sharpe, fill rate, inventory trackers
│
├── monitoring/                 # Live dashboard for collected exchange data
│
├── utils/                      # Shared helpers (time, math, logging)
├── configs/                    # YAML/TOML strategy and data configs
└── tests/                      # Unit and integration tests
```