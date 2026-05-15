import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.preprocessing import StandardScaler

class DataProcessorV2:
    def __init__(self, lookback=100, horizon=16):
        self.lookback = lookback
        self.horizon = horizon
        self.scaler = StandardScaler()

    def add_indicators(self, df):
        """
        V2: Includes Volume-Price Trend (VPT) + Original Indicators.
        """
        df = df.copy().sort_values('timestamp')
        
        # 1. Basic Price Returns
        prev_close = df['close'].shift(1)
        df['open_ret'] = (df['open'] / prev_close) - 1
        df['high_ret'] = (df['high'] / prev_close) - 1
        df['low_ret'] = (df['low'] / prev_close) - 1
        df['close_ret'] = (df['close'] / prev_close) - 1
        
        # 2. Volume Normalization
        df['volume_log'] = np.log1p(df['volume'])
        
        # 3. VOLUME-PRICE TREND (VPT) - NEW FEATURE
        # Formula: VPT = Prev_VPT + (Volume * (Close - Prev_Close) / Prev_Close)
        price_change_pct = (df['close'] - prev_close) / (prev_close + 1e-9)
        vpt_incremental = df['volume'] * price_change_pct
        df['vpt'] = vpt_incremental.fillna(0).cumsum()
        
        # Normalizing VPT: We use a rolling z-score style ROC for VPT
        # This tells the net if money flow is accelerating or decelerating
        vpt_ema = df['vpt'].ewm(span=14).mean()
        vpt_std = df['vpt'].rolling(window=14).std()
        df['vpt_signal'] = (df['vpt'] - vpt_ema) / (vpt_std + 1e-9)
        
        # 4. Legacy Technical Indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = (100 - (100 / (1 + rs))) / 100.0
        
        # EMAs
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['ema_diff'] = (ema_fast - ema_slow) / (df['close'] + 1e-9)

        # Volatility
        df['volatility'] = (df['high'] - df['low']).rolling(window=14).mean() / (df['close'] + 1e-9)

        return df.dropna()

    def create_labels(self, df, threshold=0.015):
        future_max = df['high'].shift(-self.horizon).rolling(window=self.horizon).max()
        df['target'] = (future_max > df['close'] * (1 + threshold)).astype(int)
        return df

    def prepare_features(self, df):
        # Now 9 features instead of 8
        features = [
            'open_ret', 'high_ret', 'low_ret', 'close_ret', 
            'volume_log', 'vpt_signal', 'rsi', 'ema_diff', 'volatility'
        ]
        
        rolling_mean = df[features].rolling(window=1000, min_periods=1).mean()
        rolling_std = df[features].rolling(window=1000, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1e-9)
        
        data_df = (df[features] - rolling_mean) / rolling_std
        data_df = data_df.ffill().bfill().fillna(0)
        
        data = data_df.values
        labels = df['target'].values.astype(np.int32) if 'target' in df.columns else None
        
        return data.astype(np.float32), labels

    def prepare_sequences(self, df):
        """Convert dataframe into X (sequences) and y (labels) using sliding window."""
        data, labels = self.prepare_features(df)
        X, y = [], []
        for i in range(len(data) - self.lookback - self.horizon):
            X.append(data[i:i + self.lookback])
            y.append(labels[i + self.lookback] if labels is not None else 0)
        return np.array(X), np.array(y)
