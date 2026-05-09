import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, lookback=100, horizon=16):
        self.lookback = lookback  # How many past candles to look at
        self.horizon = horizon    # How many candles into the future to predict
        self.scaler = StandardScaler()

    def add_indicators(self, df):
        """
        Calculates indicators and converts absolute prices into relative returns.
        This makes the data 'stationary' and asset-agnostic.
        """
        df = df.copy().sort_values('timestamp')
        
        # 1. Price Returns (Relative to previous close)
        # Using shift(1) to get the previous close for normalization
        prev_close = df['close'].shift(1)
        df['open_ret'] = (df['open'] / prev_close) - 1
        df['high_ret'] = (df['high'] / prev_close) - 1
        df['low_ret'] = (df['low'] / prev_close) - 1
        df['close_ret'] = (df['close'] / prev_close) - 1
        
        # 2. Volume Normalization (Log transform to handle spikes)
        # Volume can vary by 1000x, so log scale is more stable for neural nets
        df['volume_log'] = np.log1p(df['volume'])
        
        # 3. Technical Indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = (100 - (100 / (1 + rs))) / 100.0 # Scale to 0-1
        
        # EMAs
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        
        # Distance from EMA (already a percentage-like value)
        df['ema_diff'] = (ema_fast - ema_slow) / (df['close'] + 1e-9)

        # Volatility (Percentage-based ATR)
        df['volatility'] = (df['high'] - df['low']).rolling(window=14).mean() / (df['close'] + 1e-9)

        # Drop the first few rows that have NaNs due to rolling/diff
        return df.dropna()

    def create_labels(self, df, threshold=0.015):
        """
        Label as 1 if price increases by >threshold within the horizon.
        Default threshold 1.5% for 15m timeframe is a strong rally.
        """
        # Look ahead: find the maximum price in the next 'horizon' periods
        future_max = df['high'].shift(-self.horizon).rolling(window=self.horizon).max()
        df['target'] = (future_max > df['close'] * (1 + threshold)).astype(int)
        return df

    def prepare_features(self, df):
        """
        Convert dataframe into a normalized 2D feature matrix.
        Returns (features_np, labels_np)
        """
        features = ['open_ret', 'high_ret', 'low_ret', 'close_ret', 'volume_log', 'rsi', 'ema_diff', 'volatility']
        
        data = df[features].values
        labels = df['target'].values

        # Final Standardization (Z-score normalization)
        data = self.scaler.fit_transform(data)
        
        return data.astype(np.float32), labels.astype(np.int32)

    def prepare_sequences(self, df):
        """Convert dataframe into X (sequences) and y (labels) using sliding window."""
        data, labels = self.prepare_features(df)
        
        X, y = [], []
        # We stop early to avoid out-of-bounds due to lookback and horizon
        for i in range(len(data) - self.lookback - self.horizon):
            X.append(data[i:i + self.lookback])
            y.append(labels[i + self.lookback])

        return np.array(X), np.array(y)

if __name__ == "__main__":
    processor = DataProcessor()
    # Find raw data
    raw_path = 'data/raw/*.csv'
    files = glob(raw_path)
    
    if files:
        print(f"Loading {files[0]}...")
        df = pd.read_csv(files[0])
        df = processor.add_indicators(df)
        df = processor.create_labels(df)
        X, y = processor.prepare_sequences(df)
        
        print(f"--- Processing Complete ---")
        print(f"Asset: {os.path.basename(files[0])}")
        print(f"X shape (Samples, Lookback, Features): {X.shape}")
        print(f"y shape (Labels): {y.shape}")
        print(f"Positive class (Rallies): {y.sum()} ({y.mean():.2%})")
        print(f"Feature list: ['open_ret', 'high_ret', 'low_ret', 'close_ret', 'volume_log', 'rsi', 'ema_diff', 'volatility']")
    else:
        print(f"No data found in {raw_path}. Please run src/data/fetcher.py first.")
