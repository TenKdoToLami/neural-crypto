import torch
import numpy as np
import pandas as pd
import os
from src.data.processor import DataProcessor
from src.models.classifier import NeuralSentinelV1

class LiveInferenceEngine:
    def __init__(self, model_path='models/best_model.pth', device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DataProcessor(lookback=100, horizon=16)
        
        self.model = NeuralSentinelV1().to(self.device)
        self.model.eval()
        
        if os.path.exists(model_path):
            print(f"[Inference] Loading model weights from {model_path}...")
            # Handling both full model and state_dict depending on how it was saved
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                if isinstance(state_dict, NeuralSentinelV1):
                    self.model = state_dict
                else:
                    self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"[!] Warning: Could not load model properly: {e}")
        else:
            print(f"[!] Warning: Model path {model_path} not found. Running with uninitialized weights.")
            
    def process_and_predict(self, data_dict):
        """
        Takes a dictionary of {symbol: DataFrame} and returns predictions.
        data_dict should contain the latest data (up to the lookback period).
        """
        predictions = []
        
        for symbol, df in data_dict.items():
            if len(df) < self.processor.lookback:
                print(f"[Inference] Not enough data for {symbol}. Needs {self.processor.lookback}, got {len(df)}")
                continue
                
            try:
                # Add indicators
                processed_df = self.processor.add_indicators(df)
                
                # Check length again after dropping NaNs from indicators
                if len(processed_df) < self.processor.lookback:
                    continue
                    
                # We do NOT use create_labels for live inference
                # Just get the features using the processor's prepare_features
                data_np, _ = self.processor.prepare_features(processed_df)
                
                # We only want the very last lookback window for the current prediction
                last_window = data_np[-self.processor.lookback:]
                
                # Ensure it has exactly `lookback` rows
                if len(last_window) != self.processor.lookback:
                    continue
                    
                # Convert to tensor: shape [1, seq_len, features]
                x_tensor = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Inference
                with torch.no_grad():
                    # We might use mixed precision for speed if needed
                    with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                        rally_logits, vol_est = self.model(x_tensor)
                        
                        # Apply sigmoid for classification probability
                        rally_prob = torch.sigmoid(rally_logits).item()
                        vol_pred = vol_est.item()
                        
                predictions.append({
                    'symbol': symbol,
                    'rally_prob': rally_prob,
                    'vol_pred': vol_pred,
                    'last_close': df['close'].iloc[-1],
                    'timestamp': df['timestamp'].iloc[-1]
                })
                
            except Exception as e:
                print(f"[!] Inference error for {symbol}: {e}")
                
        # Sort predictions by probability (highest first)
        df_preds = pd.DataFrame(predictions)
        if not df_preds.empty:
            df_preds = df_preds.sort_values(by='rally_prob', ascending=False).reset_index(drop=True)
            
        return df_preds

if __name__ == "__main__":
    from src.live_data import LiveDataFetcher
    fetcher = LiveDataFetcher()
    data = fetcher.fetch_latest()
    
    engine = LiveInferenceEngine()
    preds = engine.process_and_predict(data)
    
    print("\n[Live Predictions]")
    print(preds.to_string())
