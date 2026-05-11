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
    def __init__(self, model_path, entry_threshold=0.95, exit_threshold=0.35, fee=0.001, trail_pct=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralSentinelV1(input_dim=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.processor = DataProcessor()
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.fee = fee
        self.trail_pct = trail_pct  # e.g. 0.05 for 5% trailing stop from peak
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
            if report_name == "BEAR":
                filename = os.path.join('data/bear_market', f"{symbol.replace('/', '_')}_15m.csv")
            else:
                filename = os.path.join('data/raw', f"{symbol.replace('/', '_')}_15m.csv")
            
            if not os.path.exists(filename):
                continue
            
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            if len(df) < self.processor.lookback + 10: continue
            
            # Process features
            processed_df = self.processor.add_indicators(df)
            data_np, _ = self.processor.prepare_features(processed_df)
            
            # Batch Inference — sliding window via stride tricks (zero-copy)
            n_windows = len(data_np) - self.processor.lookback
            if n_windows <= 0:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(
                data_np, (self.processor.lookback, data_np.shape[1])
            ).squeeze(axis=1)[:n_windows]  # (n_windows, lookback, features)

            INFER_BATCH = 4096
            all_probs = []
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with torch.no_grad():
                for i in range(0, n_windows, INFER_BATCH):
                    batch = torch.from_numpy(
                        windows[i:i + INFER_BATCH].copy()
                    ).to(self.device, dtype=torch.float32)
                    with torch.amp.autocast(device_type=device_type, enabled=(self.device.type == 'cuda')):
                        logits, _ = self.model(batch)
                    all_probs.append(torch.sigmoid(logits).cpu())
            probs = torch.cat(all_probs).squeeze(-1).numpy()

            # Align signals with original DF (offset by lookback)
            signal_df = processed_df.iloc[self.processor.lookback:].copy()
            signal_df['prob'] = probs
            all_signals[symbol] = signal_df

        # --- Portfolio Simulation Loop ---
        MAX_POSITIONS = 10
        trail_label = f" | Trail: {self.trail_pct*100:.1f}%" if self.trail_pct else ""
        print(f"\n[*] Simulating Portfolio Logic ({MAX_POSITIONS} slots{trail_label})...")

        # Pre-index signals for O(1) timestamp lookups (replaces O(n) pandas filters)
        sig_index = {}  # {symbol: {timestamp: {'close': ..., 'prob': ...}}}
        all_ts_set = set()
        for sym, sdf in all_signals.items():
            idx = {}
            for _, row in sdf.iterrows():
                idx[row['timestamp']] = {'close': row['close'], 'prob': row['prob']}
            sig_index[sym] = idx
            all_ts_set.update(idx.keys())
        all_times = sorted(all_ts_set)

        if not all_times:
            print(f"[!] No data found for the period {start_date} to {end_date}. Skipping {report_name}.")
            return None

        portfolio_cash = 10000.0
        active_trades = {}  # {symbol: {'entry_price', 'amount', 'entry_ts', 'peak_price'}}
        last_prices = {}    # {symbol: last_known_close} for carry-forward
        stats = {'trades': [], 'equity_curve': []}

        for ts in tqdm(all_times, desc="Simulating Timeline"):
            # Update last known prices for all assets at this timestamp
            for sym in sig_index:
                if ts in sig_index[sym]:
                    last_prices[sym] = sig_index[sym][ts]['close']

            # Calculate correct total equity using carry-forward prices
            current_total_value = portfolio_cash
            for sym, pos in active_trades.items():
                if sym in last_prices:
                    current_total_value += pos['amount'] * last_prices[sym]

            stats['equity_curve'].append({'ts': str(ts), 'value': current_total_value})

            # 1. Check Exits
            to_exit = []
            for sym, pos in active_trades.items():
                if sym not in last_prices:
                    continue
                current_price = last_prices[sym]

                # Update peak price for trailing stop
                if current_price > pos['peak_price']:
                    pos['peak_price'] = current_price

                # Exit condition 1: Trailing stop hit
                if self.trail_pct is not None:
                    drawdown = 1 - (current_price / pos['peak_price'])
                    if drawdown >= self.trail_pct:
                        to_exit.append(sym)
                        continue

                # Exit condition 2: Model confidence dropped
                if ts in sig_index[sym]:
                    prob = sig_index[sym][ts]['prob']
                    if prob < self.exit_threshold:
                        to_exit.append(sym)

            for sym in to_exit:
                exit_price = last_prices[sym]
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
            if len(active_trades) < MAX_POSITIONS:
                candidates = []
                for sym in sig_index:
                    if sym in active_trades: continue
                    if ts in sig_index[sym]:
                        row = sig_index[sym][ts]
                        if row['prob'] > self.entry_threshold:
                            candidates.append((sym, row['prob'], row['close']))

                candidates.sort(key=lambda x: x[1], reverse=True)
                for sym, prob, price in candidates:
                    if len(active_trades) >= MAX_POSITIONS: break
                    # Equal-weight: each slot gets 1/MAX_POSITIONS of total equity
                    alloc_size = current_total_value / MAX_POSITIONS
                    if portfolio_cash >= alloc_size:
                        buy_value = alloc_size * (1 - self.fee)
                        amount = buy_value / price
                        portfolio_cash -= alloc_size
                        active_trades[sym] = {
                            'entry_price': price,
                            'amount': amount,
                            'entry_ts': ts,
                            'peak_price': price
                        }

        # Final Summary
        final_value = current_total_value
        total_return = (final_value / 10000.0) - 1
        win_rate = len([t for t in stats['trades'] if t['pnl_pct'] > 0]) / max(1, len(stats['trades']))
        
        # Calculate Buy & Hold Benchmark (Equal Weight across contributing assets)
        benchmark_returns = []
        print(f"   --- B&H per asset ({report_name}) ---")
        for sym, asset_df in all_signals.items():
            if not asset_df.empty:
                start_p = asset_df.iloc[0]['close']
                end_p = asset_df.iloc[-1]['close']
                ret = (end_p / start_p) - 1
                benchmark_returns.append(ret)
                print(f"     {sym:>15s}: {ret*100:+.2f}%  ({start_p:.4f} → {end_p:.4f})")

        avg_benchmark_return = np.mean(benchmark_returns) if benchmark_returns else 0
        print(f"   --- Avg B&H ({len(benchmark_returns)} assets): {avg_benchmark_return*100:.2f}% ---")
        
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
    parser = argparse.ArgumentParser(description="Backtest a trained model on BEAR and LIVE periods.")
    parser.add_argument('--model', type=str, default='models/best_model.pth', help="Path to model .pth file")
    parser.add_argument('--trail', type=float, default=None, help="Trailing stop percentage (e.g. 0.05 for 5%%)")
    parser.add_argument('--entry', type=float, default=0.95, help="Entry probability threshold (default: 0.95)")
    parser.add_argument('--exit', type=float, default=0.35, help="Exit probability threshold (default: 0.35)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[!] Error: Model file {args.model} not found.")
        sys.exit(1)

    tester = ModelTester(
        args.model,
        entry_threshold=args.entry,
        exit_threshold=args.exit,
        trail_pct=args.trail
    )

    # 1. Bear Market Period (Historical Stability)
    tester.run_backtest("2022-01-01", "2023-12-31", "BEAR")

    # 2. Recent/Live Period
    tester.run_backtest("2025-01-01", "2099-01-01", "LIVE")
