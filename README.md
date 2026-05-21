# 🛡️ Neural Sentinel V1
**Advanced Crypto Lifecycle Automation & Predictive Intelligence**

Neural Sentinel is a high-performance deep learning pipeline designed to identify high-probability bullish rallies across the cryptocurrency market. It combines a global "General Intelligence" model trained on 100+ assets with a rigorous, production-aligned validation system to ensure your trading edge is real and stable.

---

## 🚀 1. Setup & Training
To build your own neural model, follow these three simple steps:

### A. Synchronize Data
Fetch the historical "Study Material" for the model.
```powershell
# Initial Download (Fetch data since 2025)
python src/data/fetcher.py --limit 100 --since 2025-01-01

# Periodical "Top Up" (Run this to get the latest candles)
python src/data/fetcher.py
```
*   **Intelligent Sync**: The fetcher automatically remembers which assets you have and only downloads the missing "Delta" candles.

### B. Train the Model
Train the Neural Network to recognize rally patterns.
```powershell
python src/train.py
```
*   **Hardware Optimized**: Uses an RTX 4070 (or similar) to process 1M+ samples in ~15 minutes.
*   **Smart Naming**: Saves models with timestamps (e.g., `sentinel_20240510.pth`) and updates the `best_model.pth` alias.

---

## 🧪 2. Verification & Comparison
Never trade a model without proving it works.

### A. Run the "Stress Test"
The Model Tester puts your new model through two distinct market regimes:
```powershell
python src/model_tester.py
```
*   **BEAR (2022-2023)**: A blind "Stress Test" on data the model has never seen. This prevents overfitting.
*   **LIVE (2025-Present)**: A "Recency Check" to ensure the model understands current market conditions.

### B. Compare Performance
See how your different models stack up against each other and the market.
```powershell
python src/compare_models.py
```
*   **Benchmark**: Compares your return against a "Buy & Hold" baseline.
*   **Metrics**: Shows Win Rate %, Total Trades, and Return % side-by-side.

---

## 🌳 3. GBDT Pipeline (HistGradientBoosting)
To train and test the fast Gradient Boosted Decision Tree (GBDT) model (a scikit-learn built-in alternative to LightGBM), follow these steps:

### A. Train the GBDT Classifier
Trains the GBDT on data from 2024-Present using raw approved assets:
```powershell
python src/train_gbdt.py
```
*   **CPU Optimized**: Trains in a few seconds on a single CPU thread.
*   **Balanced Classes**: Automatically uses `class_weight='balanced'` to offset class imbalances in bullish swing labels.
*   **Alias Save**: Saves model to `models/sentinel_gbdt.joblib` and updates the `best_gbdt_model.joblib` alias.

### B. Run GBDT Benchmark Tests
Runs the model on strict BEAR (2022-2023 Out-Of-Sample) and LIVE (2024-Present In-Sample) benchmarks:
```powershell
python src/model_tester_gbdt.py --model models/best_gbdt_model.joblib
```
*   **JSON Output**: Exports benchmarks to `reports/models/sentinel_gbdt_BEAR.json` and `sentinel_gbdt_LIVE.json`.
*   **Side-by-Side Evaluation**: After running, execute `python src/compare_models.py` to compare GBDT directly against your deep learning models.

### C. Run GBDT Dynamic Portfolio Backtest
Simulates the exact chronological portfolio strategy: 10% dynamic allocation, max 10 concurrent positions, cross-asset probability ranking. It includes whipsaw protection (minimum hold periods) and price-action stop-loss/take-profit boundaries:

Example with script defaults:
```powershell
python src/portfolio_backtest_gbdt.py --entry 0.70 --exit 0.40 --window oos
```

Example of recommended **Macro Low-Churn** settings:
```powershell
python src/portfolio_backtest_gbdt.py --entry 0.57 --exit 0.43 --min-hold 64 --window all
```

Example with Stop-Loss (e.g. 2.0%) and Take-Profit (e.g. 5.0%) limits:
```powershell
python src/portfolio_backtest_gbdt.py --entry 0.57 --exit 0.43 --min-hold 64 --sl 0.02 --tp 0.05
```

#### GBDT Command Line Parameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--model` | `models/best_gbdt_model.joblib` | Path to the saved GBDT model `.joblib` |
| `--entry` | `0.70` | Probability entry threshold to trigger buying (0.57 is recommended for the regularized model) |
| `--exit` | `0.40` | Probability exit threshold to trigger selling (0.43 is recommended for the regularized model) |
| `--min-hold` | `16` | Minimum holding duration in candles (15-min intervals) before allowing probability-based exits. Crucial for whipsaw protection (64 is recommended) |
| `--sl` | `0.0` | Stop-loss percentage (e.g., `0.02` for 2% immediate exit below entry price on low of candle). Set to `0.0` to disable |
| `--tp` | `0.0` | Take-profit percentage (e.g., `0.05` for 5% immediate exit above entry price on high of candle). Set to `0.0` to disable |
| `--fee` | `0.001` | Transaction fee percentage (0.001 represents 0.1%) |
| `--data-dir` | `data/raw` | Path to directory containing raw `.csv` candles data |
| `--window` | `oos` | Period split: `oos` (Out-of-sample before 2024), `is` (In-sample from 2024), or `all` |

---

## 🔌 4. Binance Integration
Neural Sentinel uses the **Binance API** via the CCXT library for two primary functions:

1.  **Data Acquisition**: The `fetcher.py` script scans the Top 100 assets by volume on Binance and downloads 15-minute OHLCV candles directly to your `data/raw/` folder.
2.  **Live Inference**: The live bot connects to Binance to get real-time price snapshots and execute trades based on the model's predictions.

---

## 📁 5. Repository Structure
*   `src/data/`: Data fetching and technical indicator processing (including `processor_gbdt.py`).
*   `src/models/`: Neural Network architectures and model definitions.
*   `reports/`: JSON and HTML audit logs of every backtest performed.
*   `models/`: Saved `.pth` and `.joblib` model versions.
*   `data/approved_assets.txt`: Your whitelist of coins the bot is allowed to trade.

