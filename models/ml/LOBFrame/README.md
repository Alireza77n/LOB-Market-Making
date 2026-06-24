# LOBFrame

We release `LOBFrame` (see the two papers [`Deep Limit Order Book Forecasting`](https://arxiv.org/abs/2403.09267) and [`HLOB - Information Persistence and Structure in Limit Order Books`](https://arxiv.org/abs/2405.18938)), a novel, open-source code base which presents a renewed way to process large-scale Limit Order Book (LOB) data. This framework integrates all the latest cutting-edge insights from scientific research (see [Lucchese et al.](https://www.sciencedirect.com/science/article/pii/S0169207024000062), [Prata et al.](https://arxiv.org/pdf/2308.01915.pdf)) into a cohesive system. Its strength lies in the comprehensive nature of the implemented pipeline, which includes the data transformation and processing stage, an ultra-fast implementation of the training, validation, and testing steps, as well as the evaluation of the quality of a model's outputs through trading simulations. Moreover, it offers flexibility by accommodating the integration of new models, ensuring adaptability to future advancements in the field.

## Introduction

In this tutorial, we show how to replicate the experiments presented in the two papers titled __"Deep Limit Order Book Forecasting: A microstructural guide"__ and __"HLOB - Information Persistence and Structure in Limit Order Books"__.

Before starting, please remember to **ALWAYS CITE OUR WORKS** as follows:

```
@article{briola2024deep,
  title={Deep Limit Order Book Forecasting},
  author={Briola, Antonio and Bartolucci, Silvia and Aste, Tomaso},
  journal={arXiv preprint arXiv:2403.09267},
  year={2024}
}
```

```
@misc{briola2024hlob,
      title={HLOB -- Information Persistence and Structure in Limit Order Books}, 
      author={Antonio Briola and Silvia Bartolucci and Tomaso Aste},
      year={2024},
      eprint={2405.18938},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR}
}
```

## Pre-requisites

Install the required packages:

```bash
pip3 install -r requirements.txt
```

If you are using a MacOS operating system, please proceed as follows:

```bash
pip3 install -r requirements_mac_os.txt
```

## Data
All the code in this repository exploits [LOBSTER](https://lobsterdata.com) data. To have an overview on their structure, please refer
to the official documentation available at the following [link](https://lobsterdata.com/info/DataStructure.php).

# Preliminary operations
Before starting any experiment:
- Open the ```lightning_batch_gd.py``` file and insert the [Weights & Biases](https://wandb.ai/site) project's name and API key (search for TODOs).
- Open the ```utils.py``` file and set the default values of the parameters.

## Usage
To start an experiment from scratch, you need to follow these steps:
- Place the raw data in the `data/nasdaq/raw` folder. The data must be in the LOBSTER format and each folder must be named with the asset's name (e.g. AAPL for Apple stock).
- Run the following command to pre-process data:
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --stages "data_processing"
  ```
- Run the following command to prepare the torch datasets (this allows to reduce the training time):
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --stages "torch_dataset_preparation" --prediction_horizon 10
  ```
  If you are interested also in performing the backtest stage, run the following command:
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --stages "torch_dataset_preparation,torch_dataset_preparation_backtest" --prediction_horizon 10
  ```
- If you are planning to use the HLOB model (see the paper titled [`HLOB - Structure and Persistence of Information in Limit Order Books`](https://arxiv.org/abs/2405.18938)), it is mandatory to execute the following command:
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --stages "complete_homological_structures_preparation"
  ```
- Run the following command to train the model:
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --stages "training"
  ```
  Currently available models are:
    - deeplob
    - transformer
    - itransformer
    - lobtransformer
    - dla
    - cnn1
    - cnn2
    - binbtabl
    - binctabl
    - axiallob
    - hlob
- Run the following command to evaluate the model:
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --experiment_id "<experiment_id_generated_in_the_training_stage>" --stages "evaluation"
  ```
- Run the following command to analyze the results:
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --experiment_id "<experiment_id_generated_in_the_training_stage>" --stages "backtest,post_trading_analysis"
  ```

Multiple (compatible) stages can be executed at the same time. Consider the following example:
```bash
python3 main --training_stocks "CSCO" --target_stocks "CSCO" --stages "data_processing,torch_dataset_preparation,torch_dataset_preparation_backtest,training,evaluation,backtest,post_trading_analysis"
```

Each experiment can be resumed and re-run by specifying its ID in the `experiment_id` parameter.

We now provide the typical structure of the folder in this fork, reflecting the
multi-exchange / offline (no Weights & Biases) setup actually used in this project.

A few notes on how it differs from the upstream layout:
- A Python 3.11 virtual environment lives in `.venv/` (CPU-only `torch`). All commands
  in this fork are run as `./.venv/Scripts/python.exe main.py ...`, **not** `python3 main`.
- `data_processing/convert_exchange_data.py` is a new converter that turns an
  Iranian-exchange order-book CSV directly into LOBFrame's *processed* format (the
  `scaled_data` / `unscaled_data` splits), so the LOBSTER-only `data_processing` stage
  is bypassed and there is no `raw_data` / message-file step for this data.
- Feature normalization is selectable across three methods (`global`, `rolling_1`,
  `rolling_5`) via `data_processing/normalization.py`; see
  [Feature normalization](#feature-normalization) below.
- `torch_datasets/` is keyed by `threshold_<thr>/batch_size_<bs>/training_<sym>_test_<sym>/<horizon>/`
  (one cache per symbol/horizon), so different exchanges/symbols never collide.
- Per-run outputs land in `loggers/results/<experiment_id>/` (checkpoint, `metrics.csv`,
  `prediction.pkl`, `data.yaml`, `lightning_logs/` via the Lightning `CSVLogger`).

```bash
.
├── README.md
├── .venv                                   # Python 3.11 venv (CPU-only torch) — use this
├── data
│   └── nasdaq
│        ├── scaled_data                    # z-scored features (written by convert_exchange_data.py)
│             ├── test
│             ├── training
│             └── validation
│        └── unscaled_data                  # raw prices/volumes (used by the backtest)
│             ├── test
│             ├── training
│             └── validation
├── data_processing
│   ├── convert_exchange_data.py            # exchange CSV -> processed LOBFrame format (auto-detects book convention, supports --data_representation)
│   ├── normalization.py                    # shared z-score methods (global / rolling_1 / rolling_5) + interactive selector
│   ├── data_process.py                     # LOBSTER-only (unused with exchange data)
│   ├── data_process_utils.py
│   └── complete_homological_utils.py
├── loaders
│   └── custom_dataset.py
├── loggers
│   ├── logger.py
│   └── results                             # one sub-folder per experiment_id
│        └── <experiment_id>
│             ├── best_val_model.ckpt
│             ├── metrics.csv
│             ├── prediction.pkl
│             ├── data.yaml
│             ├── hyperparameters.yaml
│             └── lightning_logs
├── main.py
├── models
│   ├── AxialLob
│   │     └── axiallob.py
│   ├── CNN1
│   │     └── cnn1.py
│   ├── CNN2
│   │     └── cnn2.py
│   ├── DeepLob
│   │     └── deeplob.py
│   ├── DLA
│   │     └── DLA.py
│   ├── iTransformer
│   │     └── itransformer.py
│   ├── LobTransformer
│   │     └── lobtransformer.py
│   ├── TABL
│   │     ├── bin_nn.py
│   │     ├── bin_tabl.py
│   │     ├── bl_layer.py
│   │     └── tabl_layer.py
│   ├── Transformer
│   │     └── transformer.py
│   └── CompleteHCNN
│         └── complete_hcnn.py
├── optimizers
│   ├── executor.py
│   └── lightning_batch_gd.py               # training loop (offline: CSVLogger, no wandb)
├── requirements.txt
├── simulator
│   ├── market_sim.py
│   ├── post_trading_analysis.py
│   └── trading_agent.py
├── torch_datasets
│   └── threshold_0.0
│       └── batch_size_32
│           └── training_<symbol>_test_<symbol>
│               └── 10
│                   ├── test_dataset.pt
│                   ├── test_dataset_backtest.pt
│                   ├── training_dataset.pt
│                   └── validation_dataset.pt
└── utils.py
```

# Running LOBFrame on the Iranian-exchange LOB data

> This section is specific to our dataset. It explains how
> to train any of the available models on every exchange, for **both BTC and Tether (USDT)**.
> The whole pipeline has been smoke-tested end-to-end on all 10 order-book files (5 exchanges
> × {BTC, USDT}) and on more than one model — every file converts, trains, and evaluates.

## 0. One-time setup

- **Always use the project venv** (Python 3.11, CPU-only torch). From inside the `LOBFrame/`
  folder, invoke Python as `./.venv/Scripts/python.exe`, **not** the system `python`/`python3`.
- **No Weights & Biases account is needed.** This fork logs offline with Lightning's
  `CSVLogger`; there is no API key to set, and the "Preliminary operations" wandb TODO above
  does **not** apply here.
- The raw exchange data lives one level up, in `../LOB/<exchange>_data/`. You do **not** run
  the upstream `data_processing` stage for it — `convert_exchange_data.py` replaces it.

## 1. Pick your data file and symbol label

The five exchanges use slightly different file names and three different book conventions
(the converter auto-detects the convention, so you don't pass any flag for it). Pick the
`--input_csv` for the exchange + coin you want, and choose a **distinct `--symbol` label**
(it becomes the filename prefix and the `torch_datasets` cache key, so keep exchanges apart):

| Exchange  | BTC order-book file (`--input_csv`)              | USDT / Tether order-book file (`--input_csv`)     | Suggested `--symbol` (BTC / USDT) |
|-----------|--------------------------------------------------|----------------------------------------------------|-----------------------------------|
| bitpin    | `../LOB/bitpin_data/BTC_IRT_orderbook.csv`       | `../LOB/bitpin_data/USDT_IRT_orderbook.csv`        | `BTC_bitpin` / `USDT_bitpin`      |
| nobitex   | `../LOB/nobitex_data/BTCIRT_orderbook.csv`       | `../LOB/nobitex_data/USDTIRT_orderbook.csv`        | `BTC_nobitex` / `USDT_nobitex`    |
| ramzinex  | `../LOB/ramzinex_data/BTC_IRT_orderbook.csv`     | `../LOB/ramzinex_data/USDT_IRT_orderbook.csv`      | `BTC_ramzinex` / `USDT_ramzinex`  |
| tabdeal   | `../LOB/tabdeal_data/BTCIRT_orderbook.csv`       | `../LOB/tabdeal_data/USDTIRT_orderbook.csv`        | `BTC_tabdeal` / `USDT_tabdeal`    |
| wallex    | `../LOB/wallex_data/BTCTMN_orderbook.csv`        | `../LOB/wallex_data/USDTTMN_orderbook.csv`         | `BTC_wallex` / `USDT_wallex`      |

> Note: wallex prices are quoted in **TMN (Toman)**, the others in **IRT (Rial)**; this does
> not change any command — only the file name differs.

## 2. The three-step run (per file)

Below, three placeholders: `<CSV>` (from the table), `<SYM>` (your `--symbol`), and
`<MODEL>` (see the list in step 3). The horizons `10,50,100` and `--prediction_horizon 10`
must be consistent across all three steps.

```bash
# Step A — convert the exchange CSV into LOBFrame's processed format (book convention auto-detected).
#          --clean wipes any previous split for THIS dataset folder first.
#          --normalization picks the z-score scheme; omit it to be asked interactively
#          (see "Feature normalization" below). global is the safe default for low volume.
#          --data_representation picks "lob" (40 raw LOB features) or "ofi" (10 pure OFI
#          features replacing LOB); omit it to be asked interactively (see below).
./.venv/Scripts/python.exe data_processing/convert_exchange_data.py \
  --input_csv "<CSV>" --symbol "<SYM>" --horizons "10,50,100" --normalization global --clean

# Step B — build the torch datasets. This prints/creates a NEW experiment_id folder under
#          loggers/results/ ; note it (it is "<SYM>_<MODEL>_<timestamp>_<rand>").
./.venv/Scripts/python.exe main.py --model <MODEL> \
  --training_stocks <SYM> --target_stocks <SYM> \
  --horizons "10,50,100" --prediction_horizon 10 \
  --stages "torch_dataset_preparation,torch_dataset_preparation_backtest" --num_workers 0

# Step C — train + evaluate, REUSING that experiment_id. Outputs land in
#          loggers/results/<experiment_id>/ (best_val_model.ckpt, metrics.csv, prediction.pkl).
./.venv/Scripts/python.exe main.py --experiment_id "<EXPERIMENT_ID>" --model <MODEL> \
  --training_stocks <SYM> --target_stocks <SYM> \
  --horizons "10,50,100" --prediction_horizon 10 \
  --stages "training,evaluation" --num_workers 0 --epochs 30 --patience 6
```

### Feature normalization

Step A z-scores the 40 order-book features (labels are **never** normalized). Choose the
scheme with `--normalization {global,rolling_1,rolling_5}`; if the flag is omitted you are
asked **interactively** — first which data you're using, then (for exchange data) which
method. Selecting LOBSTER auto-uses the original 5-day scheme without a second question.

| Method | What it does | Use when |
| --- | --- | --- |
| `global` *(default)* | Single z-score whose `mu`/`sigma` are fit on the **training portion only** (the first `--training_ratio` of rows, chronologically) and applied unchanged to validation/test. No look-ahead; drops no warm-up days. | **Low-volume** data — keeps the most samples. Does not adapt to drift over a long test period. |
| `rolling_1` | 1-day rolling z-score: each calendar day is normalized by the **previous 1 day**. Drops the first day. | **Moderate-volume** data; adapts quickly to drift. |
| `rolling_5` | Original LOBFrame scheme: each calendar day normalized by the **previous 5 days**. Drops the first 5 days. | **High-volume** data where 5 days is a stable estimate and per-day drift matters. |

Notes:
- The rolling methods are **causal** (only ever use strictly earlier calendar days), so
  they are applied across the whole dataset *before* the chronological split; warm-up days
  are dropped first and the train/val/test split is recomputed on the survivors. With short
  collections `rolling_5` can discard a large fraction of days — prefer `global` there.
- `global` fits its statistics on the training rows only, so there is **no train/test
  leakage**; its one weakness is staleness if the market drifts, which hurts (not inflates)
  test performance. The rolling methods exist for that case.
- The unscaled copy in `unscaled_data/` always keeps raw integer prices/volumes (the
  backtest needs them) and is restricted to the same surviving rows as the scaled copy.

### Data representation

Step A also accepts `--data_representation {lob,ofi}` to control which feature set is used as
the model input:

| `--data_representation` | What the model sees | When to use |
| --- | --- | --- |
| `lob` *(default)* | 40 raw LOB columns (ASKp1, ASKs1, ..., BIDp10, BIDs10) — the original LOBFrame behaviour. | General LOB modelling; any model is compatible. |
| `ofi` | 10 pure Order Flow Imbalance columns (one per level), **replacing** the raw LOB columns. OFI per level = bid order flow − ask order flow (Cont et al.) and is z-scored on its own. (ASKp1/BIDp1 are kept only in the unscaled file so the backtest can reconstruct trade prices.) | Field-standard microstructure signal; reduces feature count; requires a compatible model (see below). |

If the flag is omitted you are asked **interactively** (first in `convert_exchange_data.py`,
then in `main.py` for consistency).

**OFI-compatible models:** `transformer`, `dla`, `cnn1` — these accept an arbitrary feature
width. Models that assume a 40-wide LOB structure (`deeplob`, `lobtransformer`, `itransformer`,
`cnn2`, `binbtabl`, `binctabl`, `axiallob`, `hlob`) will fail with a clear error if selected
with `--data_representation ofi`.

**Worked example — DeepLOB on nobitex BTC:**

```bash
./.venv/Scripts/python.exe data_processing/convert_exchange_data.py \
  --input_csv "../LOB/nobitex_data/BTCIRT_orderbook.csv" --symbol BTC_nobitex --horizons "10,50,100" --clean

./.venv/Scripts/python.exe main.py --model deeplob \
  --training_stocks BTC_nobitex --target_stocks BTC_nobitex \
  --horizons "10,50,100" --prediction_horizon 10 \
  --stages "torch_dataset_preparation,torch_dataset_preparation_backtest" --num_workers 0
# -> note the printed experiment_id, e.g. BTC_nobitex_deeplob_2026-06-14_21_50_44_VdcgPBg

./.venv/Scripts/python.exe main.py --experiment_id "BTC_nobitex_deeplob_2026-06-14_21_50_44_VdcgPBg" \
  --model deeplob --training_stocks BTC_nobitex --target_stocks BTC_nobitex \
  --horizons "10,50,100" --prediction_horizon 10 \
  --stages "training,evaluation" --num_workers 0 --epochs 30 --patience 6
```

**Worked example — LobTransformer on wallex Tether (USDT):**

```bash
./.venv/Scripts/python.exe data_processing/convert_exchange_data.py \
  --input_csv "../LOB/wallex_data/USDTTMN_orderbook.csv" --symbol USDT_wallex --horizons "10,50,100" --clean

./.venv/Scripts/python.exe main.py --model lobtransformer \
  --training_stocks USDT_wallex --target_stocks USDT_wallex \
  --horizons "10,50,100" --prediction_horizon 10 \
  --stages "torch_dataset_preparation,torch_dataset_preparation_backtest" --num_workers 0
# -> note the printed experiment_id

./.venv/Scripts/python.exe main.py --experiment_id "<EXPERIMENT_ID>" \
  --model lobtransformer --training_stocks USDT_wallex --target_stocks USDT_wallex \
  --horizons "10,50,100" --prediction_horizon 10 \
  --stages "training,evaluation" --num_workers 0 --epochs 30 --patience 6
```

## 3. Choosing the model (`--model`)

Swap `<MODEL>` in steps B and C for any of the models the framework ships with:

| `--model`       | Notes                                                                 |
|-----------------|-----------------------------------------------------------------------|
| `deeplob`       | CNN + LSTM baseline (Zhang et al.).                                    |
| `lobtransformer`| Transformer over LOB snapshots (our default; smoke-tested on all 10 files). |
| `transformer`   | Vanilla transformer.                                                  |
| `itransformer`  | Inverted transformer.                                                 |
| `dla`           | Deep Layer Aggregation.                                               |
| `cnn1`          | CNN variant 1.                                                        |
| `cnn2`          | CNN variant 2.                                                        |
| `binbtabl`      | BiN + B(TABL).                                                        |
| `binctabl`      | BiN + C(TABL).                                                        |
| `axiallob`      | Axial-attention LOB model.                                            |
| `hlob`          | Homological model — needs the **extra stage** below before training.  |

**`hlob` only:** after Step A and before Step C, also run the homology preparation
(reusing the same `--symbol` and the experiment_id from Step B):

```bash
./.venv/Scripts/python.exe main.py --experiment_id "<EXPERIMENT_ID>" --model hlob \
  --training_stocks <SYM> --target_stocks <SYM> \
  --horizons "10,50,100" --prediction_horizon 10 \
  --stages "complete_homological_structures_preparation" --num_workers 0
```

**OFI models:** When you pick `--data_representation ofi`, only `transformer`, `dla`, and
`cnn1` are compatible. Use `lob` for all other models.

## 4. Tips & gotchas (our dataset)

- **`--num_workers 0`** on Windows avoids dataloader worker-spawn problems — keep it.
- **Re-running the same `--symbol`** with `--clean` is fine; just remember each model run in
  Steps B/C generates its own `experiment_id` folder, while the `torch_datasets` cache is
  shared per `symbol/horizon`, so you only convert + prep once per symbol, then reuse the id.
- **`--horizons` must match** what you converted with (`10,50,100` throughout these examples),
  and `--prediction_horizon` must be one of them.
- **`--threshold`** (default `0.0`) controls the up/flat/down labelling. At `0.0` the "flat"
  class is tiny; raise it (e.g. `--threshold 5`) if you want "flat" to be learnable.
- **Backtest / plots:** adding `--stages "backtest,post_trading_analysis"` (with the same
  experiment_id) opens blocking matplotlib windows — run it interactively, not headless.
- **More data is the real lever:** each file is only a few days of ~10s snapshots, so models
  overfit quickly; treat single-file accuracy as a sanity check, not a final number.

# License

Copyright 2024 Antonio Briola, Silvia Bartolucci, Tomaso Aste.

Licensed under the CC BY-NC-ND 4.0 Licence (the "Licence"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

```
https://creativecommons.org/licenses/by-nc-nd/4.0/
```

Software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the provided link for the specific language governing permissions and limitations under the License.