import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import NeuralSentinelV1
from src.data.processor import DataProcessor

def backtest(csv_path, entry_threshold=0.8, exit_threshold=0.5, fee=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = NeuralSentinelV1(input_dim=8).to(device)
    model_path = "models/sentinel_v1_slim.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Load and Process Data
    print(f"⌛ Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    processor = DataProcessor(lookback=100, horizon=16)
    
    # Prepare features
    df = processor.add_indicators(df)
    df = processor.create_labels(df)
    X_np, _ = processor.prepare_sequences(df)
    X = torch.from_numpy(X_np).float().to(device)
    
    # 3. Generate Predictions (Probabilities)
    print("🧠 Generating model predictions...")
    probs = []
    with torch.no_grad():
        # Process in batches for speed
        batch_size = 2048
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i+batch_size]
            logits, _ = model(batch_x)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs.extend(prob.flatten())
    
    # 4. Simulation Loop
    # We start from the index where the first sequence ends
    # which is lookback - 1 in the original dataframe
    prices = df['close'].values[processor.lookback:]
    timestamps = df['timestamp'].values[processor.lookback:]
    
    balance = 100.0
    equity_curve = [balance]
    in_position = False
    entry_price = 0
    trades = []
    cash_count = 0
    
    print(f"📈 Simulating Strategy (Entry: {entry_threshold} | Exit: {exit_threshold} | Fee: {fee*100}%)...")
    
    for i in range(len(probs)):
        p = probs[i]
        current_price = prices[i]
        
        # Strategy Logic
        if not in_position:
            cash_count += 1
            if p >= entry_threshold:
                # BUY
                in_position = True
                entry_price = current_price
                balance *= (1.0 - fee)
                entry_time = timestamps[i]
                cash_count -= 1 # Buy happens at this candle
        
        elif in_position and p <= exit_threshold:
            # SELL
            in_position = False
            profit_pct = (current_price - entry_price) / entry_price
            balance *= (1.0 + profit_pct)
            balance *= (1.0 - fee)
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': timestamps[i],
                'profit_pct': profit_pct - (fee * 2)
            })
            
        equity_curve.append(balance if not in_position else balance * (prices[i]/entry_price))

    # 5. Report Results
    final_pnl = ((balance / 100.0) - 1.0) * 100
    win_rate = len([t for t in trades if t['profit_pct'] > 0]) / len(trades) if trades else 0
    cash_pct = (cash_count / len(probs)) * 100
    
    print("\n" + "="*30)
    print(f" BACKTEST REPORT: {os.path.basename(csv_path)}")
    print("="*30)
    print(f"Total Trades:    {len(trades)}")
    print(f"Win Rate:        {win_rate*100:.1f}%")
    print(f"Time in Cash:    {cash_pct:.1f}%")
    print(f"Final Balance:   ${balance:.2f}")
    print(f"Total PnL:       {final_pnl:.2f}%")
    
    if trades:
        best_trade = max(t['profit_pct'] for t in trades)
        worst_trade = min(t['profit_pct'] for t in trades)
        print(f"Best Trade:      {best_trade*100:.1f}%")
        print(f"Worst Trade:     {worst_trade*100:.1f}%")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to raw CSV file (e.g. data/raw/BTC_USDT_15m.csv)")
    parser.add_argument("--entry", type=float, default=0.8, help="Entry probability threshold")
    parser.add_argument("--exit", type=float, default=0.5, help="Exit probability threshold")
    parser.add_argument("--fee", type=float, default=0.01, help="Fee per trade (0.01 = 1%)")
    args = parser.parse_args()
    
    # Allow passing just the filename in the data/raw folder
    path = args.file
    if not os.path.exists(path):
        path = os.path.join("data/raw", args.file)
        
    backtest(path, entry_threshold=args.entry, exit_threshold=args.exit, fee=args.fee)
