# Asset Allocation Module — DeepLOB + DLS

## 1. Overview

This branch adds an **Asset Allocation** module to the `LOB-Market-Making` repository.

The implemented pipeline combines **DeepLOB** and **DLS**:

- **DeepLOB** is used as a signal generation model.
- **DLS** is used as a portfolio allocation model.
- The final output is a set of target portfolio weights that can be passed to an execution and backtesting engine.

The main idea is that DeepLOB does not trade directly. Instead, it produces probability signals for each stock, and DLS converts these probability signals into portfolio weights.

```text
LOB + OHLCV Features
        ↓
DeepLOB Signal Model
        ↓
Down / Flat / Up Probabilities
        ↓
DLS Feature Panel
        ↓
DLS Portfolio Allocation Model
        ↓
Target Portfolio Weights
        ↓
Execution / Backtest Engine
        ↓
Portfolio Metrics + Dashboard Replay
```

---

## 2. Added Files and Folders

The following components were added in this branch:

```text
LOB-Market-Making/
│
├── analysis/
│   └── notebooks/
│       └── asset_allocation/
│           └── deeplob_dls_asset_allocation.ipynb
│
├── docs/
│   └── asset_allocation/
│       ├── DeepLOB_DLS_Report_documentation.pdf
│       └── README.md
│
├── dashboard/
│   └── asset_allocation/
│       ├── app.py
│       ├── requirements.txt
│       ├── .streamlit/
│       │   └── config.toml
│       └── data/
│           └── exported CSV files
```

---

## 3. Notebook

### Path

```text
analysis/notebooks/asset_allocation/deeplob_dls_asset_allocation.ipynb
```

### Description

This notebook contains the full DeepLOB + DLS asset allocation workflow.

Main tasks implemented in the notebook:

- load in-sample and out-of-sample OHLCV data;
- load and process LOB data;
- build DeepLOB input features;
- generate DeepLOB probability outputs;
- construct DLS input features from DeepLOB probabilities;
- train the DLS portfolio allocation model;
- apply tradability masks and signal-quality filters;
- generate target portfolio weights;
- run the execution and backtest engine;
- compare the DLS portfolio with the original DeepLOB-only baseline;
- export CSV files for the Streamlit dashboard replay.

---

## 4. Technical Report

### Path

```text
docs/asset_allocation/DeepLOB_DLS_Report_documentation.pdf
```

### Description

The technical report explains the methodology, model structure, objective function, execution logic, transaction costs, and final out-of-sample results.

Main topics covered in the report:

- role of DeepLOB as a signal generator;
- construction of DLS features;
- DLS model architecture;
- DLS portfolio objective function;
- no-lookahead trading convention;
- transaction-cost-aware execution;
- out-of-sample performance;
- comparison with the original DeepLOB-only baseline.

---

## 5. Streamlit Dashboard

### Path

```text
dashboard/asset_allocation/app.py
```

### Description

The Streamlit dashboard provides an interactive replay of the trading engine.

It is not only a CSV viewer. It reconstructs the trading process as a replayable state machine:

```text
signals → filters → target weights → orders → executions → holdings → portfolio state
```

The dashboard visualizes:

- DeepLOB probabilities;
- DLS target weights;
- tradability masks;
- confidence filters;
- buy and sell orders;
- executed trades;
- holdings;
- transaction costs;
- portfolio value path;
- drawdown behavior;
- turnover behavior;
- comparison between DeepLOB-only and DeepLOB + DLS.

### Dashboard Data Folder

The dashboard expects exported CSV files to be placed in:

```text
dashboard/asset_allocation/data/
```

---

## 6. Data Used in the Notebook

The notebook is designed for a Colab/Kaggle-style workflow.

The main input files are:

| File                                               | Role                                                          |
| -------------------------------------------------- | ------------------------------------------------------------- |
| `daily_data_in_sample.parquet`                   | Daily OHLCV data used as warmup and training history          |
| `lob_data_in_sample.parquet`                     | In-sample LOB data used to build historical DeepLOB features  |
| `daily_data_release_stage_out_of_sample.parquet` | Out-of-sample daily OHLCV data used for final backtesting     |
| `lob_data_release_stage_out_of_sample.parquet`   | Out-of-sample LOB data used for out-of-sample DeepLOB signals |
| `best_model_alpha_0015.pt`                       | Pretrained DeepLOB checkpoint                                 |

Executed notebook data summary:

| Item                      |        Value |
| ------------------------- | -----------: |
| Number of assets          |        2,306 |
| Total trading days        |          726 |
| Warmup / in-sample period | D001 to D484 |
| Out-of-sample period      | D485 to D726 |
| Out-of-sample days        |          242 |
| Combined OHLCV rows       |    1,606,720 |
| Processed LOB rows        |   37,572,229 |
| DeepLOB signal days       |          675 |

---

## 7. Model Structure

The implemented system contains two neural-network components:

1. **DeepLOB Signal Network**
2. **ShifuDLSNet Portfolio Allocation Network**

---

## 7.1 DeepLOB Signal Network

DeepLOB is used to predict the direction of future price movement for each asset.

For each asset $i$ and signal day $t$, DeepLOB outputs three probabilities:

$$
p_{\mathrm{down}}(t,i), 
\qquad
p_{\mathrm{flat}}(t,i), 
\qquad
p_{\mathrm{up}}(t,i)
$$

These three probabilities satisfy:

$$
p_{\mathrm{down}}(t,i)
+
p_{\mathrm{flat}}(t,i)
+
p_{\mathrm{up}}(t,i)
=
1
$$

Interpretation:

| Probability           | Meaning                               |
| --------------------- | ------------------------------------- |
| $p_{\mathrm{down}}$ | Probability of price decline          |
| $p_{\mathrm{flat}}$ | Probability of neutral price movement |
| $p_{\mathrm{up}}$   | Probability of price increase         |

### DeepLOB Input Shape

The input tensor of the DeepLOB model is:

$$
X_{\mathrm{LOB}}
\in
\mathbb{R}^{B \times 1 \times T \times N_F}
$$

In the executed notebook, the shape is:

$$
X_{\mathrm{LOB}}
\in
\mathbb{R}^{B \times 1 \times 50 \times 259}
$$

where:

| Symbol        | Meaning                  |
| ------------- | ------------------------ |
| $B$         | Batch size               |
| $T = 50$    | Time-window length       |
| $N_F = 259$ | Number of input features |

The final DeepLOB feature width is:

$$
N_F = 259
$$

The feature set includes:

- 10-level order book features;
- order flow imbalance features;
- OHLCV-derived features;
- rolling volatility;
- Amihud illiquidity;
- volume z-score;
- RSI;
- moving-average distance features;
- open-close return;
- high-low range;
- close-VWAP distance.

### DeepLOB Architecture

The DeepLOB model contains:

1. convolutional feature extraction blocks;
2. inception-style parallel convolution branches;
3. temporal LSTM layer;
4. fully connected classification head;
5. softmax output layer.

The loaded pretrained DeepLOB checkpoint contains approximately:

```text
199,203 trainable parameters
```

The DeepLOB output can be written as:

$$
\hat{\mathbf{p}}(t,i)
=
\mathrm{DeepLOB}
\left(
X_{\mathrm{LOB}}(t,i)
\right)
$$

or equivalently:

$$
\hat{\mathbf{p}}(t,i)
=
\begin{bmatrix}
p_{\mathrm{down}}(t,i) \\
p_{\mathrm{flat}}(t,i) \\
p_{\mathrm{up}}(t,i)
\end{bmatrix}
\in
\mathbb{R}^{3}
$$

More explicitly, the final classification head applies a softmax function:

$$
\hat{\mathbf{p}}(t,i)
=
\mathrm{softmax}
\left(
W_{\mathrm{DL}} h^{\mathrm{DL}}_{T}(t,i)
+
b_{\mathrm{DL}}
\right)
$$

---

## 7.2 DLS Portfolio Allocation Network

DLS receives DeepLOB probability outputs and converts them into portfolio weights.

For each asset, the notebook constructs two additional features from the DeepLOB probabilities.

The first feature is the directional signal score:

$$
\mathrm{signal\_score}(t,i)
=
p_{\mathrm{up}}(t,i)
-
p_{\mathrm{down}}(t,i)
$$

The second feature is the confidence score:

$$
\mathrm{confidence}(t,i)
=
\left|
\mathrm{signal\_score}(t,i)
\right|
\left(
1
-
p_{\mathrm{flat}}(t,i)
\right)
$$

Therefore, each asset has a 5-dimensional DLS input vector:

$$
\mathbf{z}(t,i)
=
\begin{bmatrix}
p_{\mathrm{down}}(t,i) \\
p_{\mathrm{flat}}(t,i) \\
p_{\mathrm{up}}(t,i) \\
\mathrm{signal\_score}(t,i) \\
\mathrm{confidence}(t,i)
\end{bmatrix}
\in
\mathbb{R}^{5}
$$

The full probability feature panel is:

$$
X_{\mathrm{prob}}
\in
\mathbb{R}^{D \times N \times 5}
$$

where:

| Symbol | Meaning                            |
| ------ | ---------------------------------- |
| $D$  | Number of signal days              |
| $N$  | Number of assets                   |
| $5$  | Number of DeepLOB-derived features |

The DLS lookback window is:

$$
L = 50
$$

So the DLS input for each day is:

$$
X_{\mathrm{DLS}}(t)
=
\left[
X_{\mathrm{prob}}(t-L+1),
X_{\mathrm{prob}}(t-L+2),
\ldots,
X_{\mathrm{prob}}(t)
\right]
$$

with shape:

$$
X_{\mathrm{DLS}}(t)
\in
\mathbb{R}^{L \times N \times 5}
$$

---

## 7.3 DLS Flattening Step

At each time step, the asset-feature panel is flattened.

Before flattening:

$$
X_{\mathrm{prob}}(t)
\in
\mathbb{R}^{N \times 5}
$$

After flattening:

$$
\mathrm{vec}
\left(
X_{\mathrm{prob}}(t)
\right)
\in
\mathbb{R}^{5N}
$$

In the executed notebook:

$$
N = 2306
$$

Therefore:

$$
5N
=
5 \times 2306
=
11530
$$

So the LSTM input size is:

$$
\mathrm{input\_size}_{\mathrm{LSTM}}
=
11530
$$

---

## 7.4 DLS Architecture

The DLS network structure is:

```text
DeepLOB probabilities
        ↓
5-feature probability panel
        ↓
Flatten asset-feature panel
        ↓
LSTM input size: 5N = 11,530
        ↓
LSTM hidden size: 64
        ↓
Linear layer: 64 → N
        ↓
Tradability mask
        ↓
Masked softmax
        ↓
Target portfolio weights
```

The DLS LSTM produces a hidden state:

$$
h^{\mathrm{DLS}}_{L}(t)
$$

The final linear layer maps this hidden state to asset-level logits:

$$
\mathbf{a}(t)
=
W_{\mathrm{DLS}}
h^{\mathrm{DLS}}_{L}(t)
+
b_{\mathrm{DLS}}
$$

where:

$$
\mathbf{a}(t)
\in
\mathbb{R}^{N}
$$

For non-tradable assets, the corresponding logit is masked before applying softmax:

$$
a_i(t)
=
-10^9,
\qquad
\text{if asset } i \text{ is not tradable}
$$

The final portfolio weight of asset $i$ is obtained using masked softmax:

$$
w_i(t)
=
\frac{
\exp
\left(
a_i(t)
\right)
}{
\sum_{j=1}^{N}
\exp
\left(
a_j(t)
\right)
}
$$

The DLS output is a long-only target weight vector:

$$
w_i(t) \geq 0
$$

and:

$$
\sum_{i=1}^{N}
w_i(t)
=
1
$$

The implemented ShifuDLSNet contains approximately:

```text
3,118,466 trainable parameters
```

---

## 8. No-Lookahead Trading Convention

The notebook uses a shifted no-lookahead convention.

```text
Day t     : DeepLOB generates signals
Day t + 1 : Portfolio is traded
Day t + 2 : Trade outcome is evaluated
```

The future return label is:

$$
R_{\mathrm{future}}(t,i)
=
\frac{
\mathrm{Close}(t+2,i)
}{
\mathrm{VWAP}_{\mathrm{entry}}(t+1,i)
}
-
1
$$

This convention prevents the model from using future information when making trading decisions.

---

## 9. DLS Objective Function

DLS is trained using a portfolio-level objective function.

Instead of using a classification loss, the model is trained based on the financial quality of the generated portfolio returns.

The gross portfolio return is:

$$
R_{\mathrm{gross}}(t)
=
\sum_{i=1}^{N}
w'_i(t)
R_{\mathrm{future}}(t,i)
$$

Turnover is measured as:

$$
\mathrm{Turnover}(t)
=
\sum_{i=1}^{N}
\left|
w'_i(t)
-
w'_i(t-1)
\right|
$$

The net portfolio return is:

$$
R_{\mathrm{net}}(t)
=
R_{\mathrm{gross}}(t)
-
C
\cdot
\mathrm{Turnover}(t)
$$

where:

$$
C
=
10^{-4}
$$

The complete DLS loss includes:

- Sharpe reward;
- Sortino reward;
- drawdown penalty;
- turnover penalty;
- concentration penalty;
- inventory penalty;
- commission penalty;
- optional sparsity penalty.

The general objective is:

$$
\mathcal{L}(\theta)
=
-
\mathrm{Sharpe}
\left(
R_{\mathrm{net}}
\right)
-
\lambda_s
\mathrm{Sortino}
\left(
R_{\mathrm{net}}
\right)
+
\lambda_{\mathrm{dd}}
\mathrm{Drawdown}
\left(
R_{\mathrm{net}}
\right)
+
\lambda_{\mathrm{to}}
\mathrm{Turnover}
+
\lambda_{\mathrm{conc}}
\mathrm{Concentration}
+
\lambda_{\mathrm{inv}}
\mathrm{Inventory}
+
\lambda_{\mathrm{comm}}
\mathrm{Commission}
+
\lambda_{\mathrm{sp}}
\mathrm{Sparsity}
$$

The negative signs before Sharpe and Sortino mean that minimizing the loss maximizes these two reward terms.

### Sharpe Term

The Sharpe term is defined as:

$$
\mathrm{Sharpe}
=
\frac{
\mu
}{
\sigma
+
\varepsilon
}
$$

where:

$$
\mu
=
\mathbb{E}
\left[
R_{\mathrm{net}}
\right]
$$

and:

$$
\sigma
=
\sqrt{
\mathbb{E}
\left[
R_{\mathrm{net}}^2
\right]
-
\mu^2
}
$$

### Sortino Term

The downside return is:

$$
R_{\mathrm{down}}(t)
=
\max
\left(
-
R_{\mathrm{net}}(t),
0
\right)
$$

The Sortino ratio is:

$$
\mathrm{Sortino}
=
\frac{
\mathbb{E}
\left[
R_{\mathrm{net}}
\right]
}{
\sqrt{
\mathbb{E}
\left[
R_{\mathrm{down}}^2
\right]
}
+
\varepsilon
}
$$

### Drawdown Term

The wealth path is:

$$
\mathrm{Wealth}(t)
=
\prod_{k=1}^{t}
\left(
1
+
R_{\mathrm{net}}(k)
\right)
$$

The running peak is:

$$
\mathrm{Peak}(t)
=
\max_{1 \leq k \leq t}
\mathrm{Wealth}(k)
$$

The drawdown at time $t$ is:

$$
\mathrm{Drawdown}(t)
=
\frac{
\mathrm{Peak}(t)
-
\mathrm{Wealth}(t)
}{
\mathrm{Peak}(t)
+
\varepsilon
}
$$

The drawdown penalty is:

$$
\mathrm{DD}
=
\max_t
\mathrm{Drawdown}(t)
$$

### Concentration Penalty

The concentration penalty is:

$$
\mathrm{Concentration}
=
\mathbb{E}_t
\left[
\sum_{i=1}^{N}
w_i(t)^2
\right]
$$

A larger value means that the portfolio is more concentrated in fewer assets.

### Inventory Penalty

The inventory penalty keeps the scaled gross exposure close to the target gross exposure:

$$
\mathrm{Inventory}
=
\mathbb{E}_t
\left[
\left(
\sum_{i=1}^{N}
\left|
w'_i(t)
\right|
-
G_{\mathrm{target}}
\right)^2
\right]
$$

where:

$$
G_{\mathrm{target}}
=
0.95
$$

### Commission Penalty

The commission penalty is:

$$
\mathrm{Commission}
=
c_{\mathrm{comm}}
\cdot
\mathbb{E}_t
\left[
\mathrm{Turnover}(t)
\right]
$$

where:

$$
c_{\mathrm{comm}}
=
10^{-4}
$$

### Loss Coefficients

| Component                 |                      Symbol | Value |
| ------------------------- | --------------------------: | ----: |
| Sortino coefficient       |               $\lambda_s$ |  0.25 |
| Drawdown coefficient      |   $\lambda_{\mathrm{dd}}$ |  0.10 |
| Turnover coefficient      |   $\lambda_{\mathrm{to}}$ |  2.00 |
| Concentration coefficient | $\lambda_{\mathrm{conc}}$ |  0.01 |
| Inventory coefficient     |  $\lambda_{\mathrm{inv}}$ |  0.02 |
| Commission coefficient    | $\lambda_{\mathrm{comm}}$ |  2.00 |
| Sparsity coefficient      |   $\lambda_{\mathrm{sp}}$ |  0.00 |

---

## 10. Execution Logic

After DLS produces target portfolio weights, the execution engine simulates real trading.

The execution engine applies:

- tradability masks;
- signal-quality filters;
- target gross exposure;
- maximum number of positions;
- minimum target weight;
- minimum holdings constraint;
- rebalance band;
- transaction costs;
- lot-size constraint;
- cash buffer.

### Main Execution Parameters

| Parameter                 |  Value |
| ------------------------- | -----: |
| `target_gross`          |   0.95 |
| `dls_max_positions`     |    500 |
| `dls_min_target_weight` | 0.0005 |
| `dls_rebalance_band`    |  0.006 |
| `commission_rate`       | 0.0001 |
| `stamp_duty_rate`       | 0.0005 |
| `min_commission`        |      5 |
| `lot_size`              |    100 |
| `cash_buffer`           |   0.05 |
| `min_holdings`          |     10 |

### Signal-Quality Filter

A stock is kept only if:

$$
\mathrm{confidence}(t,i)
\geq
0.02
$$

and:

$$
\mathrm{signal\_score}(t,i)
\geq
0.00
$$

If too few assets pass this filter, the notebook falls back to top-confidence tradable assets.

---

## 11. Training Configuration

The main DLS training hyperparameters are:

| Hyperparameter                  | Value |
| ------------------------------- | ----: |
| `dls_lookback`                |    50 |
| `dls_hidden_size`             |    64 |
| `dls_batch_size`              |    64 |
| `dls_epochs`                  |   100 |
| `dls_lr`                      | 0.001 |
| `dls_early_stopping_patience` |    15 |

A seed-search mechanism is used to select the best DLS checkpoint.

Tested seeds:

```text
7, 11, 22, 33, 42, 55, 77, 88, 101, 123
```

The best selected seed is:

```text
33
```

The final locked checkpoint is saved as:

```text
shifu_dls_model_locked_best_seed.pt
```

---

## 12. Final Out-of-Sample Results

The final DeepLOB + DLS model was evaluated on the out-of-sample period.

| Metric                   |          Value |
| ------------------------ | -------------: |
| Metric days              |            242 |
| Dataset share used       |         33.33% |
| Initial capital          | RMB 50,000,000 |
| Selected DLS seed        |             33 |
| Final value              |     RMB 73.05M |
| Total return             |         46.11% |
| CAGR                     |         46.11% |
| Annualized Sharpe ratio  |           1.20 |
| Maximum drawdown         |        -15.77% |
| Score proxy              |          20.41 |
| Annualized volatility    |         35.28% |
| Calmar ratio             |           2.92 |
| Total transaction costs  |  RMB 3,076,075 |
| Average daily turnover   | RMB 36,290,607 |
| Win rate                 |         53.72% |
| Average holdings per day |          235.3 |

---

## 13. Baseline Comparison

The notebook compares the final DLS strategy with the original DeepLOB-only baseline.

| Model                       | Total Return | Sharpe | Max Drawdown | Score Proxy | Avg Holdings |
| --------------------------- | -----------: | -----: | -----------: | ----------: | -----------: |
| Original DeepLOB-only exact |       41.16% |   0.97 |      -24.91% |       15.19 |        692.7 |
| DeepLOB + DLS               |       46.11% |   1.20 |      -15.77% |       20.41 |        235.3 |

Compared with the original DeepLOB-only baseline, the DeepLOB + DLS model achieved:

- higher total return;
- higher Sharpe ratio;
- lower maximum drawdown;
- better score proxy;
- fewer average holdings;
- more controlled portfolio allocation.

---

## 14. Generated Outputs

The notebook generates the following output files:

| Output File                                              | Purpose                          |
| -------------------------------------------------------- | -------------------------------- |
| `shifu_dls_model_locked_best_seed.pt`                  | Final selected DLS checkpoint    |
| `shifu_dls_seed_search_summary.csv`                    | Seed-search result summary       |
| `T001_dls_weight_debug_shifted_tplus2_conf_filter.csv` | DLS daily diagnostics            |
| `T001_dls_weights_long_shifted_tplus2.csv`             | Long-format DLS target weights   |
| `T001_dls_trade_audit.csv`                             | Executed trade audit             |
| `T001_dls_weight_step_audit.csv`                       | Weight adjustment audit          |
| `T001_dls_holding_snapshot_audit.csv`                  | Holdings snapshot audit          |
| `T001_oos_shifu_dls_colab_sell_open.csv`               | Final DLS submission / trade log |
| `T001_comparison_original_deeplob_exact_vs_dls.csv`    | Baseline comparison table        |

Final trade log summary:

| Item                 | Value |
| -------------------- | ----: |
| Trading days         |   242 |
| Trade log rows       | 7,495 |
| Unique traded assets |   459 |

---

## 15. Dashboard Usage

To run the dashboard from the project root:

```bash
streamlit run dashboard/asset_allocation/app.py
```

The dashboard expects CSV outputs in:

```text
dashboard/asset_allocation/data/
```

Required Python packages:

```text
streamlit
pandas
numpy
plotly
streamlit-autorefresh
```

---

## 16. Main Contributions in This Branch

This branch adds:

- DeepLOB + DLS asset allocation notebook;
- DLS portfolio allocation model;
- DeepLOB probability feature engineering;
- signal score and confidence feature construction;
- portfolio-level DLS objective function;
- Sharpe and Sortino-based optimization;
- drawdown, turnover, concentration, inventory, and commission penalties;
- shifted no-lookahead trading convention;
- tradability and quality filters;
- transaction-cost-aware execution engine;
- out-of-sample backtest;
- comparison with DeepLOB-only baseline;
- Streamlit replay dashboard;
- technical report explaining the model and objective function.

---

## 17. Notes

Non-tradable assets are masked before applying softmax. Final weights are post-processed using execution constraints such as target gross exposure, maximum number of positions, minimum target weight, and minimum holdings.

The dashboard is designed as a replay engine. It reconstructs the trading process from exported notebook outputs and helps inspect how signals are converted into target weights, trades, holdings, and portfolio-level metrics.
