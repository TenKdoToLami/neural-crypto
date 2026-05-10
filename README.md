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

## 🔌 3. Binance Integration
Neural Sentinel uses the **Binance API** via the CCXT library for two primary functions:

1.  **Data Acquisition**: The `fetcher.py` script scans the Top 100 assets by volume on Binance and downloads 15-minute OHLCV candles directly to your `data/raw/` folder.
2.  **Live Inference**: The live bot connects to Binance to get real-time price snapshots and execute trades based on the model's predictions.

---

## 📁 Repository Structure
*   `src/data/`: Data fetching and technical indicator processing.
*   `src/models/`: Neural Network architecture (CNN-Transformer-GRU hybrid).
*   `reports/`: JSON audit logs of every backtest performed.
*   `models/`: All saved `.pth` model versions.
*   `data/approved_assets.txt`: Your whitelist of coins the bot is allowed to trade.
