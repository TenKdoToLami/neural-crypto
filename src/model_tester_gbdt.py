import os
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import json
from datetime import datetime
import joblib

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor_gbdt import DataProcessorGBDT

class PortfolioSimulatorGBDT:
    def __init__(self, model_path, cash=10000.0, max_allocation=0.096, min_hold=16, sl=0.0, tp=0.0, entry=0.70, exit=0.40):
        # Load GBDT Model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
            
        print(f"[*] Loading GBDT Model from {model_path}...")
        model_artifact = joblib.load(model_path)
        self.model = model_artifact['model']
        self.feature_names = model_artifact['feature_names']
        
        self.processor = DataProcessorGBDT(lookback=100, horizon=16)
        self.initial_cash = cash
        self.max_allocation = max_allocation # 9.6%
        self.commission = 0.001 # 0.1% Binance fee
        self.min_hold = min_hold
        self.sl = sl
        self.tp = tp
        self.entry_threshold = entry
        self.exit_threshold = exit

    def load_period_data(self, data_dir, date_filter_fn=None):
        """Loads and aligns approved assets."""
        approved_path = 'data/approved_assets.txt'
        approved_assets = []
        if os.path.exists(approved_path):
            with open(approved_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        approved_assets.append(line.split('/')[0])
        
        files = glob(os.path.join(data_dir, "*.csv"))
        all_data = {}
        
        print(f"[*] Loading data from {data_dir}...")
        for f in tqdm(files, desc="Assets"):
            asset = os.path.basename(f).split('_')[0]
            
            if approved_assets and asset not in approved_assets:
                continue
                
            if os.path.getsize(f) < 1000: continue
            
            df = pd.read_csv(f)
            if len(df) < 500: continue
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Apply date filters if specified
            if date_filter_fn is not None:
                df = date_filter_fn(df)
                if len(df) < 200: continue
            
            # Add technical indicators
            df_proc = self.processor.add_indicators(df)
            features_np, _, _ = self.processor.prepare_features(df_proc)
            
            all_data[asset] = {
                'features': features_np,
                'timestamps': df_proc['timestamp'].values,
                'opens': df_proc['open'].values,
                'closes': df_proc['close'].values,
                'highs': df_proc['high'].values,
                'lows': df_proc['low'].values
            }
            
        return all_data

    def run_simulation(self, data_dict, period_name="Test"):
        """Runs the chronological portfolio simulation using the GBDT predictions."""
        if not data_dict:
            return {"error": f"No data available for {period_name}"}

        # Align assets chronologically
        all_ts = []
        for asset in data_dict:
            all_ts.extend(data_dict[asset]['timestamps'])
        unique_ts = np.unique(all_ts)
        unique_ts.sort()
        
        ts_to_idx = {asset: {ts: i for i, ts in enumerate(data_dict[asset]['timestamps'])} for asset in data_dict}
        
        cash = self.initial_cash
        portfolio = {} # asset -> {'units': float, 'entry_price': float, 'holding_duration': int}
        history = []
        trades = []
        
        # 1. Pre-calculate GBDT predictions in one fast pass
        print(f"[*] Scoring assets with GBDT for {period_name}...")
        probs = {}
        for asset in data_dict:
            features = data_dict[asset]['features']
            probs[asset] = self.model.predict_proba(features)[:, 1]

        # Buy & Hold Baseline calculation
        available_at_start = [a for a in data_dict if unique_ts[0] in ts_to_idx[a]]
        bh_units = {}
        if available_at_start:
            allocation = self.initial_cash / len(available_at_start)
            for a in available_at_start:
                start_price = data_dict[a]['opens'][ts_to_idx[a][unique_ts[0]]]
                bh_units[a] = allocation / start_price

        print(f"[*] Running chronological simulation...")
        # Start trading steps
        for i in tqdm(range(100, len(unique_ts)-1), desc="Simulation Steps"):
            current_ts = unique_ts[i]
            next_ts = unique_ts[i+1]
            
            # Increment holding duration for held assets
            for asset in portfolio:
                portfolio[asset]['holding_duration'] += 1
                
            # Update current portfolio valuation for history tracking
            total_val = cash
            for asset, pos in portfolio.items():
                if current_ts in ts_to_idx[asset]:
                    idx = ts_to_idx[asset][current_ts]
                    total_val += pos['units'] * data_dict[asset]['closes'][idx]
            history.append(total_val)

            # A. Evaluate exits
            to_sell = []
            for asset, pos in portfolio.items():
                if current_ts not in ts_to_idx[asset]: continue
                idx = ts_to_idx[asset][current_ts]
                prob = probs[asset][idx]
                
                # Check Stop-Loss (immediate execution on-candle)
                if self.sl > 0:
                    low_val = data_dict[asset]['lows'][idx]
                    sl_trigger = pos['entry_price'] * (1.0 - self.sl)
                    if low_val <= sl_trigger:
                        to_sell.append((asset, 'SL', sl_trigger, current_ts))
                        continue
                        
                # Check Take-Profit (immediate execution on-candle)
                if self.tp > 0:
                    high_val = data_dict[asset]['highs'][idx]
                    tp_trigger = pos['entry_price'] * (1.0 + self.tp)
                    if high_val >= tp_trigger:
                        to_sell.append((asset, 'TP', tp_trigger, current_ts))
                        continue
                        
                # Check Probability Exit (1-candle execution delay, must satisfy min_hold)
                if prob < self.exit_threshold:
                    if pos['holding_duration'] >= self.min_hold:
                        if next_ts not in ts_to_idx[asset]: continue
                        idx_next = ts_to_idx[asset][next_ts]
                        sell_price = data_dict[asset]['opens'][idx_next]
                        to_sell.append((asset, 'PROB', sell_price, next_ts))
            
            for asset, reason, sell_price, sell_ts in to_sell:
                val = portfolio[asset]['units'] * sell_price * (1 - self.commission)
                cash += val
                
                profit = (sell_price / portfolio[asset]['entry_price']) - 1
                trades.append({
                    'asset': asset, 'type': 'SELL', 'price': float(sell_price), 
                    'time': str(sell_ts), 'pnl': float(profit), 'reason': reason
                })
                del portfolio[asset]

            # B. Evaluate entries
            if len(portfolio) < 10 and cash > (self.initial_cash * 0.05):
                potential_buys = []
                for asset in data_dict:
                    if asset in portfolio: continue
                    if current_ts not in ts_to_idx[asset]: continue
                    
                    idx = ts_to_idx[asset][current_ts]
                    prob = probs[asset][idx]
                    
                    if prob > self.entry_threshold: # Entry threshold
                        potential_buys.append((asset, prob))
                
                # Cross-asset ranking
                potential_buys.sort(key=lambda x: x[1], reverse=True)
                
                for asset, prob in potential_buys:
                    if len(portfolio) >= 10: break
                    if next_ts not in ts_to_idx[asset]: continue
                    
                    target_alloc = self.initial_cash * self.max_allocation
                    if cash >= target_alloc:
                        idx_next = ts_to_idx[asset][next_ts]
                        buy_price = data_dict[asset]['opens'][idx_next]
                        
                        units = (target_alloc * (1 - self.commission)) / buy_price
                        cash -= target_alloc
                        portfolio[asset] = {'units': units, 'entry_price': buy_price, 'holding_duration': 0}
                        trades.append({'asset': asset, 'type': 'BUY', 'price': float(buy_price), 'time': str(next_ts)})

        # Final Liquidation
        final_val = history[-1] if history else self.initial_cash
        final_bh_val = 0
        for a, units in bh_units.items():
            last_ts = unique_ts[-1]
            if last_ts in ts_to_idx[a]:
                final_bh_val += units * data_dict[a]['closes'][ts_to_idx[a][last_ts]]
            else:
                final_bh_val += units * data_dict[a]['closes'][-1]
                
        # Calculate win rate from sells
        sells = [t for t in trades if t['type'] == 'SELL']
        win_rate = (len([t for t in sells if t['pnl'] > 0]) / len(sells) * 100) if sells else 0.0

        # Construct chronological equity curve for compare_models
        equity_curve = []
        for i, val in enumerate(history):
            equity_curve.append({
                "ts": str(unique_ts[100 + i]),
                "value": float(val)
            })

        return {
            "model": "sentinel_gbdt.joblib",
            "period": period_name,
            "final_value": float(final_val),
            "total_return_pct": float((final_val / self.initial_cash - 1) * 100),
            "benchmark_return_pct": float((final_bh_val / self.initial_cash - 1) * 100),
            "win_rate_pct": float(win_rate),
            "total_trades": len(trades),
            "equity_curve": equity_curve
        }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/best_gbdt_model.joblib')
    parser.add_argument("--entry", type=float, default=0.70, help="Entry probability threshold")
    parser.add_argument("--exit", type=float, default=0.40, help="Exit probability threshold")
    parser.add_argument("--min-hold", type=int, default=16, help="Minimum holding duration in candles")
    parser.add_argument("--sl", type=float, default=0.0, help="Stop-Loss percentage (0.0 to disable)")
    parser.add_argument("--tp", type=float, default=0.0, help="Take-Profit percentage (0.0 to disable)")
    args = parser.parse_args()
    
    sim = PortfolioSimulatorGBDT(
        args.model, 
        cash=10000.0, 
        min_hold=args.min_hold, 
        sl=args.sl, 
        tp=args.tp, 
        entry=args.entry, 
        exit=args.exit
    )
    
    # 1. Run BEAR Market split (data/bear_market)
    bear_data = sim.load_period_data('data/bear_market', date_filter_fn=None)
    bear_results = sim.run_simulation(bear_data, "BEAR")
    
    # 2. Run LIVE Market split (data/raw)
    live_data = sim.load_period_data('data/raw', date_filter_fn=None)
    live_results = sim.run_simulation(live_data, "LIVE")
    
    # 3. Export to reports/models for side-by-side compare
    os.makedirs('reports/models', exist_ok=True)
    
    bear_path = 'reports/models/sentinel_gbdt_BEAR.json'
    with open(bear_path, 'w') as f:
        json.dump(bear_results, f, indent=4)
        
    live_path = 'reports/models/sentinel_gbdt_LIVE.json'
    with open(live_path, 'w') as f:
        json.dump(live_results, f, indent=4)
        
    print("\n" + "="*50)
    print("GBDT BENCHMARK REPORT COMPLETED")
    print("="*50)
    print(f"[*] BEAR Market (OOS pre-2024):")
    print(f"   Return:     {bear_results['total_return_pct']:.2f}% (Buy & Hold: {bear_results['benchmark_return_pct']:.2f}%)")
    print(f"   Win Rate:   {bear_results['win_rate_pct']:.1f}%")
    print(f"   Trades:     {bear_results['total_trades']}")
    
    print(f"\n[*] LIVE Market (2024-Present):")
    print(f"   Return:     {live_results['total_return_pct']:.2f}% (Buy & Hold: {live_results['benchmark_return_pct']:.2f}%)")
    print(f"   Win Rate:   {live_results['win_rate_pct']:.1f}%")
    print(f"   Trades:     {live_results['total_trades']}")
    print("="*50)
    print(f"[+] Reports saved in reports/models/ allowing comparison via python src/compare_models.py\n")

if __name__ == "__main__":
    main()
