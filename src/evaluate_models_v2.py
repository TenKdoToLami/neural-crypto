import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import sys
import re
import json

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both architectures
from src.models.classifier import NeuralSentinelV1
from src.models.classifier_v2 import NeuralSentinelV2
from src.data.processor import DataProcessor
from src.data.processor_v2 import DataProcessorV2
from src.data.fetcher import DataFetcher

def download_evaluation_data():
    eval_dir = 'data/evaluation'
    os.makedirs(eval_dir, exist_ok=True)
    symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT', 'TRX/USDT', 'DOGE/USDT']
    fetcher = DataFetcher()
    fetcher.data_dir = eval_dir
    print(f"📥 Checking evaluation data consistency...")
    for symbol in symbols:
        filename = os.path.join(eval_dir, f"{symbol.replace('/', '_')}_15m.csv")
        if not os.path.exists(filename):
            print(f"  [!] {symbol} missing. Downloading baseline...")
            fetcher.fetch_ohlcv(symbol, timeframe='15m', since_date='2020-01-01')
    return eval_dir

def run_backtest(probs, prices, timestamps, entry_t, exit_t):
    trades = []
    in_position = False
    entry_price = 0
    entry_idx = 0
    for i in range(len(probs)):
        if not in_position:
            if probs[i] >= entry_t:
                in_position = True
                entry_price = prices[i]
                entry_idx = i
        else:
            if probs[i] <= exit_t or i == len(probs) - 1:
                in_position = False
                exit_price = prices[i]
                profit = (exit_price / entry_price) - 1
                trades.append({
                    'entry_time': timestamps[entry_idx],
                    'year': pd.to_datetime(timestamps[entry_idx]).year,
                    'profit': profit,
                    'duration': i - entry_idx
                })
    return trades

def evaluate_model_detailed(model, processor, file_path, device):
    df = pd.read_csv(file_path)
    if len(df) < processor.lookback + processor.horizon + 100: return None
    
    asset_name = os.path.basename(file_path).split('_')[0]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df_p = processor.add_indicators(df)
    df_p = processor.create_labels(df_p)
    
    prices = df_p['close'].values[processor.lookback:]
    timestamps = df_p['timestamp'].iloc[processor.lookback:].values
    
    X, _ = processor.prepare_sequences(df_p)
    prices = prices[:len(X)]
    timestamps = timestamps[:len(X)]
    
    X_tensor = torch.from_numpy(X).float().to(device)
    model.eval()
    all_probs = []
    
    batch_size = 4096
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_x = X_tensor[i : i + batch_size]
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(batch_x)
                probs = torch.sigmoid(logits).to(torch.float32).cpu().numpy()
                all_probs.extend(probs)
    
    all_probs = np.array(all_probs).flatten()
    entry_thresholds = [0.85, 0.90, 0.95]
    exit_thresholds = [0.30, 0.35, 0.40, 0.45]
    
    asset_report = []
    all_trades = []
    
    for ent in entry_thresholds:
        for ext in exit_thresholds:
            trades = run_backtest(all_probs, prices, timestamps, ent, ext)
            if not trades: continue
            df_trades = pd.DataFrame(trades)
            for year, group in df_trades.groupby('year'):
                win_rate = (group['profit'] > 0).mean()
                avg_profit = group['profit'].mean()
                asset_report.append({
                    'asset': asset_name, 'year': year, 'entry_conf': ent, 'exit_conf': ext,
                    'trades': len(group), 'win_rate': win_rate, 'avg_profit': avg_profit
                })
                group['asset'] = asset_name
                group['entry_conf'] = ent
                group['exit_conf'] = ext
                all_trades.append(group)
    return asset_report, all_trades

def parse_neurons(filename):
    match = re.search(r'(\d+)N', filename)
    return int(match.group(1)) if match else 256

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Starting HYBRID (V1 & V2) BACKTEST Evaluation")
    
    summary_list = []
    eval_dir = download_evaluation_data()
    eval_files = glob(os.path.join(eval_dir, "*.csv"))
    model_paths = sorted(glob("models/*.pth"))
    
    # Processors
    proc_v1 = DataProcessor()
    proc_v2 = DataProcessorV2()
    
    thresholds = [0.85, 0.90, 0.95]
    
    for m_path in model_paths:
        m_name = os.path.basename(m_path).replace('.pth', '')
        hidden_dim = parse_neurons(m_name)
        is_v2 = "V2_" in m_name
        
        model_dir = os.path.join('reports/evaluation', m_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if report already exists
        stats_path = os.path.join(model_dir, 'asset_performance.csv')
        if os.path.exists(stats_path):
            print(f"⏭️ Skipping {m_name} (Already evaluated)")
            df_existing = pd.read_csv(stats_path)
            overall_row = {'Model': m_name, 'Type': 'V2' if is_v2 else 'V1'}
            for ent in thresholds:
                ent_group = df_existing[df_existing['entry_conf'] == ent]
                if not ent_group.empty:
                    avg_wr = ent_group['win_rate'].mean()
                    total_sigs = ent_group['trades'].sum()
                    sig_str = f"{total_sigs/1000:.1f}k" if total_sigs > 1000 else str(total_sigs)
                    overall_row[f'Acc_{int(ent*100)}%'] = f"{avg_wr:.1%} ({sig_str})"
                else: overall_row[f'Acc_{int(ent*100)}%'] = "0.0% (0)"
            summary_list.append(overall_row)
            continue
            
        print(f"\n🚀 Backtesting {'V2' if is_v2 else 'V1'} Model: {m_name}")
        
        # Initialize correct architecture
        if is_v2:
            model = NeuralSentinelV2(input_dim=9, hidden_dim=hidden_dim).to(device)
            processor = proc_v2
        else:
            model = NeuralSentinelV1(input_dim=8, hidden_dim=hidden_dim).to(device)
            processor = proc_v1

        try:
            model.load_state_dict(torch.load(m_path, map_location=device, weights_only=True))
        except Exception as e:
            print(f"  [!] Failed to load {m_name}: {e}")
            continue
        
        all_asset_stats = []
        all_cleaned_trades = []
        for f in tqdm(eval_files, desc=f"Scanning Markets"):
            res = evaluate_model_detailed(model, processor, f, device)
            if res:
                stats, trades = res
                all_asset_stats.extend(stats)
                all_cleaned_trades.extend(trades)
        
        if all_asset_stats:
            df_stats = pd.DataFrame(all_asset_stats)
            df_stats.to_csv(os.path.join(model_dir, 'asset_performance.csv'), index=False)
            
            # Update Leaderboard
            overall_row = {'Model': m_name, 'Type': 'V2' if is_v2 else 'V1'}
            for t in thresholds:
                ent_group = df_stats[df_stats['entry_conf'] == t]
                avg_wr = ent_group['win_rate'].mean() if not ent_group.empty else 0
                total_sigs = ent_group['trades'].sum() if not ent_group.empty else 0
                sig_str = f"{total_sigs/1000:.1f}k" if total_sigs > 1000 else str(total_sigs)
                overall_row[f'Acc_{int(t*100)}%'] = f"{avg_wr:.1%} ({sig_str})"
            summary_list.append(overall_row)

    print("\n🏆 HYBRID LEADERBOARD:")
    print(pd.DataFrame(summary_list).to_string(index=False))

if __name__ == "__main__":
    main()
