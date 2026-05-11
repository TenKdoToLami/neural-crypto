import os
import torch
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import json
from datetime import datetime
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import NeuralSentinelV1
from src.data.processor import DataProcessor

class PortfolioSimulator:
    def __init__(self, model_path, device='cuda', cash=1000.0, max_allocation=0.096):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = DataProcessor()
        
        # Load Model
        self.model = NeuralSentinelV1(input_dim=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.initial_cash = cash
        self.max_allocation = max_allocation # 9.6%
        self.commission = 0.001 # 0.1% Binance fee

    def load_period_data(self, data_dir):
        """Loads and aligns only approved asset data for a specific directory."""
        # Load approved assets list
        approved_path = 'data/approved_assets.txt'
        approved_assets = []
        if os.path.exists(approved_path):
            with open(approved_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Convert BTC/USDC -> BTC
                        approved_assets.append(line.split('/')[0])
        
        files = glob(os.path.join(data_dir, "*.csv"))
        all_data = {}
        
        print(f"📂 Loading data from {data_dir} (Approved only)...")
        for f in tqdm(files, desc="Loading Assets"):
            asset = os.path.basename(f).split('_')[0]
            
            # Filter: If approved list exists, skip assets not on it
            if approved_assets and asset not in approved_assets:
                continue
                
            if os.path.getsize(f) < 1000: continue
            
            df = pd.read_csv(f)
            if len(df) < 500: continue
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Prepare features (normalized)
            df_proc = self.processor.add_indicators(df)
            features_np, _ = self.processor.prepare_features(df_proc)
            
            # We need to keep track of the price at each feature timestamp
            # The feature at index 'i' corresponds to the candle ending at df_proc.iloc[i]
            # The 'Open' price for the trade will be from the NEXT candle (t+1)
            all_data[asset] = {
                'features': features_np,
                'timestamps': df_proc['timestamp'].values,
                'opens': df_proc['open'].values,
                'closes': df_proc['close'].values
            }
            
        return all_data

    def run_simulation(self, data_dict, period_name="Test"):
        """Simulates trading step-by-step across all aligned assets."""
        if not data_dict:
            return {"error": "No data available for this period"}

        # Align all assets by timestamp
        all_ts = []
        for asset in data_dict:
            all_ts.extend(data_dict[asset]['timestamps'])
        unique_ts = np.unique(all_ts)
        unique_ts.sort()
        
        # Map timestamps to indices for each asset for fast lookup
        ts_to_idx = {asset: {ts: i for i, ts in enumerate(data_dict[asset]['timestamps'])} for asset in data_dict}
        
        cash = self.initial_cash
        portfolio = {} # asset -> {'units': float, 'entry_price': float}
        history = []
        trades = []
        
        # Buy & Hold Logic
        # Split $1000 across all assets available at the START
        available_at_start = [a for a in data_dict if unique_ts[0] in ts_to_idx[a]]
        bh_units = {}
        if available_at_start:
            allocation = self.initial_cash / len(available_at_start)
            for a in available_at_start:
                start_price = data_dict[a]['opens'][ts_to_idx[a][unique_ts[0]]]
                bh_units[a] = allocation / start_price

        # 1. PRE-INFERENCE (The Big Speedup)
        # We predict everything for all assets in one go using batches
        print(f"🧠 Pre-calculating predictions for {period_name}...")
        probs = {asset: np.zeros(len(data_dict[asset]['timestamps'])) for asset in data_dict}
        
        pbar_pred = tqdm(data_dict.keys(), desc="Predicting")
        for asset in pbar_pred:
            features = data_dict[asset]['features']
            if len(features) < 100: continue
            
            # Create all windows for this asset
            windows = []
            for j in range(len(features)):
                if j < 99:
                    windows.append(np.zeros((100, 8))) # Padding
                else:
                    windows.append(features[j-99 : j+1])
            
            windows_np = np.array(windows)
            windows_torch = torch.from_numpy(windows_np).float()
            
            # Batch inference
            asset_probs = []
            batch_size = 512 # Reduced from 4096 to avoid OOM while training
            with torch.no_grad():
                for b in range(0, len(windows_torch), batch_size):
                    batch_x = windows_torch[b : b + batch_size].to(self.device)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits, _ = self.model(batch_x)
                        p = torch.sigmoid(logits).float().cpu().numpy()
                    asset_probs.extend(p.flatten())
            
            probs[asset] = np.array(asset_probs)

        print(f"🚀 Simulating {period_name}...")
        
        # We start from 'lookback' to have enough data
        for i in tqdm(range(100, len(unique_ts)-1), desc="Trading Steps"):
            current_ts = unique_ts[i]
            next_ts = unique_ts[i+1]
            
            # 1. Evaluate Current Holdings (Check for Exits)
            to_sell = []
            for asset, pos in portfolio.items():
                if current_ts not in ts_to_idx[asset]: continue
                idx = ts_to_idx[asset][current_ts]
                
                # Get pre-calculated probability
                prob = probs[asset][idx]
                
                if prob < 0.35:
                    to_sell.append(asset)
            
            for asset in to_sell:
                if next_ts not in ts_to_idx[asset]: continue
                sell_price = data_dict[asset]['opens'][ts_to_idx[asset][next_ts]]
                val = portfolio[asset]['units'] * sell_price * (1 - self.commission)
                cash += val
                trades.append({
                    'asset': asset, 'type': 'SELL', 'price': sell_price, 
                    'time': str(next_ts), 'pnl': (sell_price / portfolio[asset]['entry_price']) - 1
                })
                del portfolio[asset]

            # 2. Evaluate Potential Buys
            if cash > (self.initial_cash * 0.05):
                potential_buys = []
                for asset in data_dict:
                    if asset in portfolio: continue
                    if current_ts not in ts_to_idx[asset]: continue
                    
                    idx = ts_to_idx[asset][current_ts]
                    prob = probs[asset][idx]
                    
                    if prob > 0.85: # High Confidence
                        potential_buys.append((asset, prob))
                
                # Sort by highest probability
                potential_buys.sort(key=lambda x: x[1], reverse=True)
                
                for asset, prob in potential_buys:
                    if next_ts not in ts_to_idx[asset]: continue
                    
                    target_alloc = self.initial_cash * self.max_allocation
                    if cash >= target_alloc:
                        buy_price = data_dict[asset]['opens'][ts_to_idx[asset][next_ts]]
                        units = (target_alloc * (1 - self.commission)) / buy_price
                        cash -= target_alloc
                        portfolio[asset] = {'units': units, 'entry_price': buy_price}
                        trades.append({'asset': asset, 'type': 'BUY', 'price': buy_price, 'time': str(next_ts)})
                    
            # Track Total Value
            total_val = cash
            for asset, pos in portfolio.items():
                if current_ts in ts_to_idx[asset]:
                    total_val += pos['units'] * data_dict[asset]['closes'][ts_to_idx[asset][current_ts]]
            history.append(total_val)

        # Final Results
        final_bh_val = 0
        for a, units in bh_units.items():
            last_ts = unique_ts[-1]
            if last_ts in ts_to_idx[a]:
                final_bh_val += units * data_dict[a]['closes'][ts_to_idx[a][last_ts]]
            else: # Use last available price if delisted
                final_bh_val += units * data_dict[a]['closes'][-1]
        
        final_val = history[-1] if history else self.initial_cash
        return {
            "period": period_name,
            "final_value": round(final_val, 2),
            "total_return": round((final_val / self.initial_cash - 1) * 100, 2),
            "buy_hold_return": round((final_bh_val / self.initial_cash - 1) * 100, 2),
            "num_trades": len(trades),
            "trades": trades[-20:] # Last 20 trades
        }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/best_model.pth')
    args = parser.parse_args()
    
    sim = PortfolioSimulator(args.model)
    
    # 1. Run Bear Market
    bear_data = sim.load_period_data('data/bear_market')
    bear_results = sim.run_simulation(bear_data, "Bear Market (2022-2023)")
    
    # 2. Run Recent Market
    recent_data = sim.load_period_data('data/raw')
    recent_results = sim.run_simulation(recent_data, "Recent Market (2024-Now)")
    
    summary = {
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": [bear_results, recent_results]
    }
    
    print("\n" + "="*50)
    print("🏆 FINAL SIMULATION RESULTS")
    print("="*50)
    for res in summary['results']:
        print(f"\n📈 Period: {res['period']}")
        print(f"   Final Portfolio: ${res.get('final_value', 'N/A')}")
        print(f"   Strategy Return: {res.get('total_return', 'N/A')}%")
        print(f"   Buy & Hold Return: {res.get('buy_hold_return', 'N/A')}%")
        print(f"   Total Trades: {res.get('num_trades', 'N/A')}")
    
    # Save to JSON
    out_file = f"reports/test_results_{datetime.now().strftime('%m%d_%H%M')}.json"
    os.makedirs('reports', exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\n📝 Full report saved to: {out_file}")

if __name__ == "__main__":
    main()
