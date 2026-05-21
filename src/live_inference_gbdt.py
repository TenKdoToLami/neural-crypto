import os
import joblib
import numpy as np
import pandas as pd
from src.data.processor_gbdt import DataProcessorGBDT
from src.utils.logger import logger

class LiveInferenceEngineGBDT:
    def __init__(self, model_path='models/best_gbdt_model.joblib'):
        self.processor = DataProcessorGBDT(lookback=100, horizon=16)
        
        if os.path.exists(model_path):
            logger.info(f"[Inference GBDT] Loading model from {model_path}...")
            try:
                model_artifact = joblib.load(model_path)
                self.model = model_artifact['model']
                self.feature_names = model_artifact['feature_names']
                logger.info(f"[Inference GBDT] Model loaded successfully.")
            except Exception as e:
                logger.error(f"[!] Warning: Could not load GBDT model: {e}")
                self.model = None
        else:
            logger.warning(f"[!] Warning: GBDT Model path {model_path} not found.")
            self.model = None
            
    def process_and_predict(self, data_dict):
        """
        Takes a dictionary of {symbol: DataFrame} and returns GBDT predictions.
        data_dict should contain the latest data (up to the lookback period).
        """
        predictions = []
        if self.model is None:
            logger.error("[Inference GBDT] Model is not loaded. Cannot generate predictions.")
            return pd.DataFrame()
            
        for symbol, df in data_dict.items():
            # Ensure we have enough history to compute rolling averages and standard scaling
            # For w=64 rolling stats and lookback=100, we need at least 150-200 candles
            if len(df) < 200:
                continue
                
            try:
                # Add indicators and temporal extensions
                processed_df = self.processor.add_indicators(df)
                
                # Check length again after dropping NaNs from indicator and shifting processes
                if len(processed_df) < 10:
                    continue
                    
                # Standardize features using chronological rolling window
                data_np, _, features = self.processor.prepare_features(processed_df)
                
                # Get the very last feature row (representing the current closed candle)
                last_row = data_np[-1]
                
                # Predict probability
                rally_prob = self.model.predict_proba(last_row.reshape(1, -1))[0, 1]
                
                predictions.append({
                    'symbol': symbol,
                    'rally_prob': float(rally_prob),
                    'last_close': float(df['close'].iloc[-1]),
                    'timestamp': df['timestamp'].iloc[-1]
                })
                
            except Exception as e:
                logger.error(f"[!] GBDT Inference error for {symbol}: {e}")
                
        # Sort predictions by probability (highest first)
        df_preds = pd.DataFrame(predictions)
        if not df_preds.empty:
            df_preds = df_preds.sort_values(by='rally_prob', ascending=False).reset_index(drop=True)
            
        return df_preds

if __name__ == "__main__":
    from src.live_data import LiveDataFetcher
    fetcher = LiveDataFetcher()
    data = fetcher.fetch_latest()
    
    engine = LiveInferenceEngineGBDT()
    preds = engine.process_and_predict(data)
    
    logger.info("\n[Live GBDT Predictions]\n" + preds.to_string())
