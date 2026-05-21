import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessorGBDT:
    def __init__(self, lookback=100, horizon=16):
        self.lookback = lookback
        self.horizon = horizon

    def add_indicators(self, df):
        """
        Computes base features and extends them with rolling statistics and lags
        to give the GBDT sequential memory.
        """
        df = df.copy().sort_values('timestamp')
        
        # 1. Base price returns
        prev_close = df['close'].shift(1)
        df['open_ret'] = (df['open'] / prev_close) - 1
        df['high_ret'] = (df['high'] / prev_close) - 1
        df['low_ret'] = (df['low'] / prev_close) - 1
        df['close_ret'] = (df['close'] / prev_close) - 1
        
        # 2. Base volume
        df['volume_log'] = np.log1p(df['volume'])
        
        # 3. Base Volume-Price Trend (VPT)
        price_change_pct = (df['close'] - prev_close) / (prev_close + 1e-9)
        vpt_incremental = df['volume'] * price_change_pct
        df['vpt'] = vpt_incremental.fillna(0).cumsum()
        
        vpt_ema = df['vpt'].ewm(span=14).mean()
        vpt_std = df['vpt'].rolling(window=14).std()
        df['vpt_signal'] = (df['vpt'] - vpt_ema) / (vpt_std + 1e-9)
        
        # 4. Base RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = (100 - (100 / (1 + rs))) / 100.0
        
        # 5. Base EMA Difference
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['ema_diff'] = (ema_fast - ema_slow) / (df['close'] + 1e-9)

        # 6. Base Volatility
        df['volatility'] = (df['high'] - df['low']).rolling(window=14).mean() / (df['close'] + 1e-9)

        # Drop rows with NaN from initial indicator calculations
        df = df.dropna().copy()

        # 7. Tree-specific Temporal Feature Extensions (Memory)
        windows = [4, 16, 64]
        for w in windows:
            # Returns stats
            df[f'close_ret_mean_{w}'] = df['close_ret'].rolling(window=w).mean()
            df[f'close_ret_std_{w}'] = df['close_ret'].rolling(window=w).std()
            df[f'close_ret_max_{w}'] = df['close_ret'].rolling(window=w).max()
            df[f'close_ret_min_{w}'] = df['close_ret'].rolling(window=w).min()
            
            # Volatility stats
            df[f'volatility_mean_{w}'] = df['volatility'].rolling(window=w).mean()
            df[f'volatility_max_{w}'] = df['volatility'].rolling(window=w).max()
            
            # Volume stats
            df[f'volume_mean_{w}'] = df['volume_log'].rolling(window=w).mean()
            
            # Momentum stats
            df[f'rsi_mean_{w}'] = df['rsi'].rolling(window=w).mean()

        # 8. Lag features to capture immediate sequence dynamics
        lags = [1, 2, 3, 4, 8, 12, 16]
        for lag in lags:
            df[f'close_ret_lag_{lag}'] = df['close_ret'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            df[f'vpt_signal_lag_{lag}'] = df['vpt_signal'].shift(lag)

        return df.dropna()

    def create_labels(self, df, threshold=0.015):
        """
        Creates binary target using Triple Barrier Method over self.horizon candles.
        A sample is bullish (1) if it rises +threshold% before falling -threshold/1.5% stop.
        """
        df = df.copy()
        future_highs = df['high'].shift(-self.horizon).rolling(window=self.horizon).max()
        future_lows = df['low'].shift(-self.horizon).rolling(window=self.horizon).min()
        
        # Bullish if future high hits threshold before future low hits stop loss
        tp_barrier = df['close'] * (1 + threshold)
        sl_barrier = df['close'] * (1 - threshold / 1.5)
        
        hit_tp = (future_highs >= tp_barrier)
        hit_sl = (future_lows <= sl_barrier)
        
        # Label 1 if TP is hit and SL is NOT hit first (simplified chronological proxy)
        df['target'] = np.where(hit_tp & ~hit_sl, 1, 0)
        return df

    def prepare_features(self, df):
        """
        Extracts feature array and label array.
        Standardizes features using a rolling 1000-candle window to prevent lookahead bias.
        """
        # Exclude metadata and raw price columns
        excluded = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'target', 'vpt'}
        features = [col for col in df.columns if col not in excluded]
        
        rolling_mean = df[features].rolling(window=1000, min_periods=1).mean()
        rolling_std = df[features].rolling(window=1000, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1e-9)
        
        data_df = (df[features] - rolling_mean) / rolling_std
        data_df = data_df.ffill().bfill().fillna(0)
        
        data = data_df.values.astype(np.float32)
        labels = df['target'].values.astype(np.int32) if 'target' in df.columns else None
        
        return data, labels, features
