# Neural Crypto Predictor

A deep learning system to predict bullish rallies across multiple crypto assets using 15m candles.

## Installation

1. Ensure you have Python 3.10+ installed.
2. Install dependencies:
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Acquisition
Fetch historical 15m candles for the top assets:
```powershell
python src/data/fetcher.py --limit 100 --timeframe 15m
```

### 2. Model Architecture: Neural Sentinel V1
Optimized for **RTX 4070 (Ada Lovelace)**:
- **Feature Extraction**: Dilated 1D Residual Blocks for multi-scale temporal patterns.
- **Attention**: Multi-Head Self-Attention (Transformer) for non-sequential event correlation.
- **Hardware Optimized**: Native `BF16` support and `torch.compile` compatibility.

### 3. GPU Benchmarking
Find the optimal batch size for your 4070:
```powershell
python src/benchmarks/gpu_test.py
```

### 4. Training
Template with Mixed Precision (BF16) and JIT compilation:
```powershell
python src/train.py
```

## Parameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--limit` | `100` | Number of top assets to fetch by volume. |
| `--timeframe` | `15m` | Candle timeframe (e.g., 1m, 15m, 1h). |
| `--days` | `365` | Number of days of history to download. |
