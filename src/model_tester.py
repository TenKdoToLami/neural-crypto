import os
import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
from datetime import datetime
from glob import glob
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import NeuralSentinelV1
from src.data.processor import DataProcessor
from src.prepare_bear_data import BearDataFetcher
from src.data.fetcher import DataFetcher

class ModelTester:
    def __init__(self, model_path, entry_threshold=0.95, exit_threshold=0.35, fee=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralSentinelV1(input_dim=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.processor = DataProcessor()
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.fee = fee
        self.model_name = os.path.basename(model_path)
        self.raw_fetcher = DataFetcher()
        self.bear_fetcher = BearDataFetcher()
        
    def get_approved_assets(self):
        assets = []
        path = 'data/approved_assets.txt'
        if not os.path.exists(path): return []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '/' not in line: line = f"{line}/USDT"
                    assets.append(line)
        return assets

    def run_backtest(self, start_date, end_date, report_name):
        assets = self.get_approved_assets()
        if not assets:
            print("No approved assets found.")
            return

        # Auto-fetch/update data before testing
        if report_name == "BEAR":
            self.bear_fetcher.run()
        elif report_name == "LIVE":
            print(f"[*] Updating live data for {len(assets)} approved assets...")
            for symbol in assets:
                self.raw_fetcher.fetch_ohlcv(symbol, days=500)

        all_signals = {}
        print(f"[*] Running Backtest for {self.model_name} on {report_name} period...")
        
        for symbol in tqdm(assets, desc="Generating Signals"):
            # Check specialized bear_market folder first for BEAR tests
            bear_file = os.path.join('data/bear_market', f"{symbol.replace('/', '_')}_15m.csv")
            raw_file = os.path.join('data/raw', f"{symbol.replace('/', '_')}_15m.csv")
            
            if report_name == "BEAR" and os.path.exists(bear_file):
                filename = bear_file
            elif os.path.exists(raw_file):
                filename = raw_file
            else:
                continue
            
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            if len(df) < self.processor.lookback + 10: continue
            
            # Process features
            processed_df = self.processor.add_indicators(df)
            data_np, _ = self.processor.prepare_features(processed_df)
            
            # Batch Inference
            probs = []
            with torch.no_grad():
                # Efficient batching could be done here, but sticking to sequential for simplicity
                for i in range(len(data_np) - self.processor.lookback):
                    window = data_np[i:i + self.processor.lookback]
                    x_tensor = torch.tensor(window).unsqueeze(0).to(self.device)
                    # Using modern autocast syntax
                    device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                    with torch.amp.autocast(device_type=device_type, enabled=(self.device.type == 'cuda')):
                        logits, _ = self.model(x_tensor)
                        probs.append(torch.sigmoid(logits).item())
            
            # Align signals with original DF (offset by lookback)
            signal_df = processed_df.iloc[self.processor.lookback:].copy()
            signal_df['prob'] = probs
            all_signals[symbol] = signal_df

        # --- Portfolio Simulation Loop ---
        print(f"[*] Simulating Portfolio Logic (20 assets, 4.8% alloc)...")
        
        # Combine all timestamps
        all_times = sorted(list(set().union(*[df.timestamp.tolist() for df in all_signals.values()])))
        
        if not all_times:
            print(f"[!] No data found for the period {start_date} to {end_date}. Skipping {report_name}.")
            return None

        portfolio_cash = 10000.0
        current_total_value = portfolio_cash
        active_trades = {} # {symbol: {'entry_price': p, 'amount': a}}
        stats = {'trades': [], 'equity_curve': []}
        
        for ts in tqdm(all_times, desc="Simulating Timeline"):
            current_total_value = portfolio_cash
            for sym, pos in active_trades.items():
                asset_df = all_signals[sym]
                row = asset_df[asset_df.timestamp == ts]
                if not row.empty:
                    current_total_value += pos['amount'] * row['close'].values[0]
                else:
                    # In a real backtest we'd use the last known price, simplifying here
                    pass
            
            stats['equity_curve'].append({'ts': str(ts), 'value': current_total_value})
            
            # 1. Check Exits
            to_exit = []
            for sym, pos in active_trades.items():
                asset_df = all_signals[sym]
                row = asset_df[asset_df.timestamp == ts]
                if not row.empty:
                    prob = row['prob'].values[0]
                    if prob < self.exit_threshold:
                        to_exit.append(sym)
            
            for sym in to_exit:
                asset_df = all_signals[sym]
                exit_price = asset_df[asset_df.timestamp == ts]['close'].values[0]
                pos = active_trades.pop(sym)
                sale_value = pos['amount'] * exit_price * (1 - self.fee)
                portfolio_cash += sale_value
                
                pnl_pct = (exit_price / pos['entry_price']) - 1
                stats['trades'].append({
                    'symbol': sym,
                    'entry': str(pos['entry_ts']),
                    'exit': str(ts),
                    'pnl_pct': pnl_pct,
                    'pnl_raw': sale_value - (pos['amount'] * pos['entry_price'])
                })

            # 2. Check Entries
            if len(active_trades) < 20:
                candidates = []
                for sym, asset_df in all_signals.items():
                    if sym in active_trades: continue
                    row = asset_df[asset_df.timestamp == ts]
                    if not row.empty:
                        prob = row['prob'].values[0]
                        if prob > self.entry_threshold:
                            candidates.append((sym, prob, row['close'].values[0]))
                
                candidates.sort(key=lambda x: x[1], reverse=True)
                for sym, prob, price in candidates:
                    if len(active_trades) >= 20: break
                    alloc_size = current_total_value * 0.048
                    if portfolio_cash >= alloc_size:
                        buy_value = alloc_size * (1 - self.fee)
                        amount = buy_value / price
                        portfolio_cash -= alloc_size
                        active_trades[sym] = {
                            'entry_price': price,
                            'amount': amount,
                            'entry_ts': ts
                        }

        # Final Summary
        final_value = current_total_value
        total_return = (final_value / 10000.0) - 1
        win_rate = len([t for t in stats['trades'] if t['pnl_pct'] > 0]) / max(1, len(stats['trades']))
        
        # Calculate Buy & Hold Benchmark (Equal Weight 5% each for the 20 assets)
        benchmark_returns = []
        for sym, asset_df in all_signals.items():
            if not asset_df.empty:
                start_p = asset_df.iloc[0]['close']
                end_p = asset_df.iloc[-1]['close']
                benchmark_returns.append((end_p / start_p) - 1)
        
        avg_benchmark_return = np.mean(benchmark_returns) if benchmark_returns else 0
        
        report = {
            'model': self.model_name,
            'period': report_name,
            'final_value': final_value,
            'total_return_pct': total_return * 100,
            'benchmark_return_pct': avg_benchmark_return * 100,
            'win_rate_pct': win_rate * 100,
            'total_trades': len(stats['trades']),
            'equity_curve': stats['equity_curve'],
            'trades': stats['trades']
        }
        
        os.makedirs('reports/models', exist_ok=True)
        report_path = f"reports/models/{self.model_name.replace('.pth', '')}_{report_name}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"✅ Report saved to {report_path}")
        print(f"   Final Value: ${final_value:.2f} ({total_return*100:.2f}%)")
        print(f"   Benchmark B&H: {avg_benchmark_return*100:.2f}%")
        print(f"   Trades: {len(stats['trades'])} | Win Rate: {win_rate*100:.1f}%")
        return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/best_model.pth', help="Path to model .pth file (default: models/best_model.pth)")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"[!] Error: Model file {args.model} not found.")
        sys.exit(1)
    
    tester = ModelTester(args.model)
    
    # 1. Bear Market Period (Historical Stability)
    tester.run_backtest("2022-01-01", "2023-12-31", "BEAR")
    
    # 2. Recent/Live Period
    tester.run_backtest("2025-01-01", "2099-01-01", "LIVE")
