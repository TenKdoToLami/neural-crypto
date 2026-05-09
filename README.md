# Neural Crypto Predictor

A deep learning system to predict bullish rallies across multiple crypto assets using 15m candles.

## 🚀 Quick Start

### 1. Synchronize Data
Fetch Top 100 volume leaders + your eToro whitelist. Skips stablecoins automatically.
```powershell
python src/data/fetcher.py --limit 100
```
*   **Whitelist**: Add symbols to `data/etoro_assets.txt`
*   **Blacklist**: Add stablecoins to `data/stables_ignore.txt`
*   **Cooldown**: Skips assets updated within the last 2 days.

### 2. Train the Model
Trains a hybrid CNN-Transformer-GRU model on your entire dataset.
```powershell
python src/train.py
```
*   **Slim-RAM**: Automatically uses `float16` compression to fit 10M+ samples into 32GB RAM.
*   **Output**: Saves weights to `models/sentinel_v1_slim.pth`.

### 3. Backtest a Single Asset
Test the strategy on one specific coin with realistic fees.
```powershell
python src/backtest.py BTC_USDT_15m.csv --entry 0.95 --exit 0.3 --fee 0.01
```

### 4. Grand Portfolio Audit
Scan your entire database and generate a sortable HTML dashboard.
```powershell
python src/portfolio_backtest.py --entry 0.95 --exit 0.3 --fee 0.01
```
*   **Output**: Open `reports/audit.html` in any browser.

---

## 🛠️ Components

### 🛰️ Data Fetcher (`src/data/fetcher.py`)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--limit` | 100 | Number of top volume assets to track from Binance. |
| `--days` | 365 | Days of history to download for new assets. |
| `--timeframe`| 15m | Candle interval. |

### 🧠 Data Processor (`src/data/processor.py`)
*   **Features**: Relative Returns, Log Volume, RSI, ATR, Trend Deviation, EMA Distance.
*   **Labeling**: 1 (Rally) if price increases 5% within the next 16 candles (4 hours).

### 📈 Backtester (`src/backtest.py`)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--entry` | 0.8 | Probability threshold to open a position (0.0 to 1.0). |
| `--exit` | 0.5 | Probability threshold to close a position. |
| `--fee` | 0.01 | Trading fee per side (0.01 = 1.0% eToro standard). |

---

## 💡 Strategy Tips
*   **eToro Users**: Because of the 1% fee, use a high `--entry` (0.90+) to ensure you only take the highest quality "Sniper" setups.
*   **Binance Users**: You can lower the `--fee` to 0.001 (0.1%) and the `--entry` to 0.8 for more frequent, high-volume trading.
*   **Time in Cash**: A healthy strategy should spend 70-90% of its time in cash, waiting for clear trend breakouts.
