import os
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor_gbdt import DataProcessorGBDT

def get_approved_assets():
    approved_path = 'data/approved_assets.txt'
    if not os.path.exists(approved_path):
        return None
    
    approved = set()
    with open(approved_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                base = line.split('/')[0] # Get BTC from BTC/USDC
                approved.add(base)
    return approved

def prepare_asset_data(csv_path, lookback, horizon):
    """
    Worker function to process indicators on CPU for faster loading.
    """
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 500: return None
        
        processor = DataProcessorGBDT(lookback=lookback, horizon=horizon)
        df = processor.add_indicators(df)
        df = processor.create_labels(df, threshold=0.015)
        
        features, _, feature_names = processor.prepare_features(df)
        
        # Clean asset symbol naming (e.g., BTC_USDC_15m -> BTC)
        asset_name = os.path.basename(csv_path).split('_')[0]
        
        return {
            'Asset': asset_name,
            'features': features,
            'feature_names': feature_names,
            'prices': df['close'].values.astype(np.float32),
            'opens': df['open'].values.astype(np.float32),
            'highs': df['high'].values.astype(np.float32),
            'lows': df['low'].values.astype(np.float32),
            'timestamps': df['timestamp'].values
        }
    except Exception as e:
        print(f"Error preparing {csv_path}: {e}")
        return None

def generate_html(asset_summaries, trade_log, final_value, final_cash, chart_path, model_name, filename="reports/gbdt_audit.html"):
    os.makedirs("reports", exist_ok=True)
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Portfolio Audit - {model_name}</title>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; padding: 40px; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: #1e293b; padding: 20px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); }}
            h1, h2 {{ color: #38bdf8; text-align: center; }}
            .summary-box {{ background: #334155; padding: 20px; border-radius: 8px; margin-bottom: 20px; text-align: center; font-size: 1.2em; }}
            .chart-img {{ width: 100%; border-radius: 8px; margin-bottom: 40px; border: 1px solid #475569; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            .pnl-pos {{ color: #4ade80; font-weight: bold; }}
            .pnl-neg {{ color: #f87171; font-weight: bold; }}
            .dataTables_wrapper {{ color: #f8fafc; margin-bottom: 40px; }}
            select, input {{ background: #334155 !important; color: white !important; border: 1px solid #475569 !important; }}
            .dataTables_info {{ margin-top: 10px; }}
            table.dataTable tbody tr {{ background-color: #1e293b; }}
            table.dataTable tbody tr:hover {{ background-color: #334155; }}
            table.dataTable thead th {{ border-bottom: 2px solid #38bdf8; color: #f8fafc; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Chronological GBDT Portfolio Backtest Audit</h1>
            <h2 style="color: #94a3b8;">Model: {model_name}</h2>
            
            <div class="summary-box">
                <strong>Starting Balance:</strong> $1,000.00 &nbsp;|&nbsp; 
                <strong>Final Value:</strong> ${final_value:,.2f} &nbsp;|&nbsp; 
                <strong>Final Free Cash:</strong> ${final_cash:,.2f} &nbsp;|&nbsp; 
                <strong>Total Return:</strong> <span class="{pnl_class}">{total_return:,.2f}%</span>
            </div>
            
            <img src="{chart_name}" class="chart-img" alt="Equity Curve">

            <h2>Trade Log</h2>
            <table id="tradeTable" class="display">
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Profit (%)</th>
                        <th>Profit ($)</th>
                    </tr>
                </thead>
                <tbody>
                    {trade_rows}
                </tbody>
            </table>

            <h2>Asset Summary</h2>
            <table id="auditTable" class="display">
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th>Total Trades</th>
                        <th>Win Rate (%)</th>
                        <th>Avg Profit per Trade (%)</th>
                    </tr>
                </thead>
                <tbody>
                    {asset_rows}
                </tbody>
            </table>
        </div>
        <script type="text/javascript" src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script>
            $(document).ready(function() {{
                $('#auditTable').DataTable({{
                    order: [[1, 'desc']],
                    pageLength: 10
                }});
                $('#tradeTable').DataTable({{
                    order: [[2, 'desc']],
                    pageLength: 25
                }});
            }});
        </script>
    </body>
    </html>
    """
    asset_rows = ""
    for r in asset_summaries:
        pnl_class_row = "pnl-pos" if r['AvgProfit'] >= 0 else "pnl-neg"
        asset_rows += f"""
        <tr>
            <td>{r['Asset']}</td>
            <td>{r['Trades']}</td>
            <td>{r['WinRate']:.2f}%</td>
            <td class="{pnl_class_row}">{r['AvgProfit']:.2f}%</td>
        </tr>
        """
        
    trade_rows = ""
    for t in trade_log:
        pnl_class_row = "pnl-pos" if t['profit_usd'] >= 0 else "pnl-neg"
        trade_rows += f"""
        <tr>
            <td>{t['asset']}</td>
            <td>{t['entry_ts']}</td>
            <td>{t['exit_ts']}</td>
            <td>${t['entry_price']:.6f}</td>
            <td>${t['exit_price']:.6f}</td>
            <td class="{pnl_class_row}">{t['profit_pct']:.2f}%</td>
            <td class="{pnl_class_row}">${t['profit_usd']:.2f}</td>
        </tr>
        """
    
    total_pnl_class = "pnl-pos" if final_value >= 1000 else "pnl-neg"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_template.format(
            asset_rows=asset_rows, 
            trade_rows=trade_rows,
            final_value=final_value, 
            final_cash=final_cash,
            total_return=((final_value - 1000) / 1000 * 100),
            pnl_class=total_pnl_class,
            chart_name=os.path.basename(chart_path),
            model_name=model_name
        ))
    print(f"\n[*] Interactive Dashboard generated: {os.path.abspath(filename)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_gbdt_model.joblib", help="Path to the saved GBDT model")
    parser.add_argument("--entry", type=float, default=0.70, help="Bullish entry probability threshold (default: 0.70)")
    parser.add_argument("--exit", type=float, default=0.40, help="Bullish exit probability threshold (default: 0.40)")
    parser.add_argument("--fee", type=float, default=0.001, help="Transaction fee percentage (0.001 = 0.1%)")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing raw asset CSVs")
    parser.add_argument("--window", type=str, choices=['oos', 'is', 'all'], default='oos', help="Backtest split: oos (<2024), is (>=2024), or all")
    parser.add_argument("--min-hold", type=int, default=16, help="Minimum holding duration in candles to prevent churn (default: 16)")
    parser.add_argument("--sl", type=float, default=0.0, help="Stop-Loss percentage (e.g. 0.02 = 2.0% below entry price, 0.0 to disable)")
    parser.add_argument("--tp", type=float, default=0.0, help="Take-Profit percentage (e.g. 0.04 = 4.0% above entry price, 0.0 to disable)")
    args = parser.parse_args()

    # Load Model Artifact
    if not os.path.exists(args.model):
        print(f"[!] GBDT Model not found at: {args.model}. Please run src/train_gbdt.py first.")
        return
        
    print(f"[*] Loading GBDT model: {args.model}")
    model_artifact = joblib.load(args.model)
    model = model_artifact['model']
    feature_names = model_artifact['feature_names']

    # Load Approved Assets Filter
    approved_bases = get_approved_assets()
    
    # Locate all CSVs in data dir
    csv_files = []
    if os.path.exists(args.data_dir):
        for f in os.listdir(args.data_dir):
            if f.endswith(".csv"):
                base_symbol = f.split('_')[0]
                if approved_bases is None or base_symbol in approved_bases:
                    csv_files.append(os.path.join(args.data_dir, f))
                    
    if not csv_files:
        print(f"[!] No valid CSV assets found in {args.data_dir}")
        return
        
    print(f"[*] Found {len(csv_files)} relevant assets to process.")

    # PHASE 1: Parallel Data Processing
    num_workers = max(1, os.cpu_count() - 2)
    print(f"[*] Extracting features on {num_workers} CPU cores...")
    prepared_data = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(prepare_asset_data, f, 100, 16) for f in csv_files]
        for future in tqdm(as_completed(futures), total=len(csv_files), desc="CSVs"):
            res = future.result()
            if res:
                prepared_data.append(res)
                
    if not prepared_data:
        print("[!] No data prepared successfully.")
        return

    # PHASE 2: Fast Tabular GBDT Inference
    print("[*] Generating GBDT predictions across all timelines...")
    aligned_signals = {}
    
    for data in tqdm(prepared_data, desc="Predictions"):
        # Predict class probabilities
        probs = model.predict_proba(data['features'])[:, 1]
        
        aligned_signals[data['Asset']] = {
            'prices': data['prices'],
            'opens': data['opens'],
            'highs': data['highs'],
            'lows': data['lows'],
            'probs': probs,
            'timestamps': data['timestamps']
        }

    # PHASE 3: Chronological Portfolio Alignment
    print("[*] Aligning global timeline...")
    all_timestamps = set()
    for asset_data in aligned_signals.values():
        all_timestamps.update(asset_data['timestamps'])
        
    global_timeline = sorted(list(all_timestamps))
    
    df_prices = pd.DataFrame(index=global_timeline)
    df_opens = pd.DataFrame(index=global_timeline)
    df_highs = pd.DataFrame(index=global_timeline)
    df_lows = pd.DataFrame(index=global_timeline)
    df_probs = pd.DataFrame(index=global_timeline)
    
    for asset, asset_data in aligned_signals.items():
        df_prices[asset] = pd.Series(asset_data['prices'], index=asset_data['timestamps'])
        df_opens[asset] = pd.Series(asset_data['opens'], index=asset_data['timestamps'])
        df_highs[asset] = pd.Series(asset_data['highs'], index=asset_data['timestamps'])
        df_lows[asset] = pd.Series(asset_data['lows'], index=asset_data['timestamps'])
        df_probs[asset] = pd.Series(asset_data['probs'], index=asset_data['timestamps'])
        
    # Forward-fill prices to maintain accurate portfolio valuation
    df_prices = df_prices.ffill()
    df_opens = df_opens.ffill()
    df_highs = df_highs.ffill()
    df_lows = df_lows.ffill()

    # Apply strict date-based split
    # 'oos' = strict out-of-sample before 2024
    # 'is'  = training period from 2024 onwards
    if args.window == 'oos':
        print("[*] Filtering timeline to strict Out-Of-Sample window (< 2024-01-01)...")
        valid_idx = pd.to_datetime(df_prices.index) < '2024-01-01 00:00:00'
    elif args.window == 'is':
        print("[*] Filtering timeline to In-Sample / training window (>= 2024-01-01)...")
        valid_idx = pd.to_datetime(df_prices.index) >= '2024-01-01 00:00:00'
    else:
        print("[*] Using complete chronological window...")
        valid_idx = [True] * len(df_prices)
        
    df_prices = df_prices[valid_idx]
    df_opens = df_opens[valid_idx]
    df_highs = df_highs[valid_idx]
    df_lows = df_lows[valid_idx]
    df_probs = df_probs[valid_idx]
    global_timeline = list(df_prices.index)
    
    if not global_timeline:
        print("[!] No candles left in the timeline after split filtering.")
        return
        
    sorted_assets = sorted(list(aligned_signals.keys()))

    # PHASE 4: Chronological Portfolio Simulation
    print(f"[*] Running chronological GBDT simulation (10% allocation, Max 10 positions)...")
    cash = 1000.0
    # positions format: asset -> {'amount': float, 'entry_price': float, 'cost_basis': float, 'entry_ts': str, 'holding_duration': int}
    positions = {asset: {'amount': 0.0, 'entry_price': 0.0, 'cost_basis': 0.0, 'entry_ts': None, 'holding_duration': 0} for asset in sorted_assets}
    
    portfolio_history = []
    trade_log = []
    
    prices_mat = df_prices[sorted_assets].values
    opens_mat = df_opens[sorted_assets].values
    highs_mat = df_highs[sorted_assets].values
    lows_mat = df_lows[sorted_assets].values
    probs_mat = df_probs[sorted_assets].values

    for i in tqdm(range(len(global_timeline)), desc="Timeline Steps"):
        ts = global_timeline[i]
        current_prices = prices_mat[i]
        current_probs = probs_mat[i]
        
        # Increment holding duration for any held asset
        for asset in sorted_assets:
            if positions[asset]['amount'] > 0:
                positions[asset]['holding_duration'] += 1
        
        # 1. Update Portfolio Valuation
        pos_value = 0.0
        active_positions_count = 0
        for j, asset in enumerate(sorted_assets):
            if positions[asset]['amount'] > 0:
                active_positions_count += 1
                if not np.isnan(current_prices[j]):
                    pos_value += positions[asset]['amount'] * current_prices[j]
                    
        total_portfolio_value = cash + pos_value
        portfolio_history.append({'timestamp': ts, 'value': total_portfolio_value, 'cash': cash})

        # 2. Check Exits (Sell signals)
        to_sell = []
        for j, asset in enumerate(sorted_assets):
            pos = positions[asset]
            if pos['amount'] > 0:
                prob = current_probs[j]
                
                # Check Stop-Loss (immediate execution on-candle)
                if args.sl > 0 and not np.isnan(lows_mat[i][j]):
                    sl_trigger = pos['entry_price'] * (1.0 - args.sl)
                    if lows_mat[i][j] <= sl_trigger:
                        to_sell.append((asset, j, 'SL', sl_trigger))
                        continue
                        
                # Check Take-Profit (immediate execution on-candle)
                if args.tp > 0 and not np.isnan(highs_mat[i][j]):
                    tp_trigger = pos['entry_price'] * (1.0 + args.tp)
                    if highs_mat[i][j] >= tp_trigger:
                        to_sell.append((asset, j, 'TP', tp_trigger))
                        continue
                        
                # Check Probability-based exit (1-candle execution delay, must satisfy min_hold)
                if not np.isnan(prob) and prob < args.exit:
                    if pos['holding_duration'] >= args.min_hold:
                        # 1-candle delay: execute at NEXT candle's open price
                        exec_price = opens_mat[i+1][j] if i+1 < len(global_timeline) else current_prices[j]
                        to_sell.append((asset, j, 'PROB', exec_price))
                    
        for asset, j, reason, exec_price in to_sell:
            if np.isnan(exec_price): continue
            
            pos = positions[asset]
            revenue = pos['amount'] * exec_price
            revenue_after_fee = revenue * (1.0 - args.fee)
            cash += revenue_after_fee
            
            profit_pct = (exec_price - pos['entry_price']) / pos['entry_price']
            profit_usd = revenue_after_fee - pos['cost_basis']
            
            exit_ts = global_timeline[i+1] if (reason == 'PROB' and i+1 < len(global_timeline)) else ts
            
            trade_log.append({
                'asset': asset,
                'entry_ts': pos['entry_ts'],
                'exit_ts': exit_ts,
                'entry_price': pos['entry_price'],
                'exit_price': exec_price,
                'profit_pct': profit_pct * 100,
                'profit_usd': profit_usd,
                'exit_reason': reason
            })
            
            positions[asset] = {'amount': 0.0, 'entry_price': 0.0, 'cost_basis': 0.0, 'entry_ts': None, 'holding_duration': 0}
            active_positions_count -= 1
            
            # Recalculate immediate assets value & cash
            pos_value = sum(positions[a]['amount'] * prices_mat[i][k] for k, a in enumerate(sorted_assets) if positions[a]['amount'] > 0 and not np.isnan(prices_mat[i][k]))
            total_portfolio_value = cash + pos_value

        # 3. Check Entries (Buy signals)
        if active_positions_count < 10:
            potential_buys = []
            for j, asset in enumerate(sorted_assets):
                if positions[asset]['amount'] == 0:
                    prob = current_probs[j]
                    if not np.isnan(prob) and prob >= args.entry:
                        potential_buys.append((asset, j, prob))
            
            # Sort by descending probability (cross-asset ranking)
            potential_buys.sort(key=lambda x: x[2], reverse=True)
            
            for asset, j, prob in potential_buys:
                if active_positions_count >= 10:
                    break
                    
                trade_allocation = total_portfolio_value * 0.10
                if cash >= trade_allocation:
                    # 1-candle execution delay: buy at open price of NEXT candle
                    exec_price = opens_mat[i+1][j] if i+1 < len(global_timeline) else current_prices[j]
                    if np.isnan(exec_price): continue
                    
                    alloc_after_fee = trade_allocation * (1.0 - args.fee)
                    amount_bought = alloc_after_fee / exec_price
                    
                    cash -= trade_allocation
                    positions[asset] = {
                        'amount': amount_bought,
                        'entry_price': exec_price,
                        'cost_basis': trade_allocation,
                        'entry_ts': global_timeline[i+1] if i+1 < len(global_timeline) else ts,
                        'holding_duration': 0
                    }
                    active_positions_count += 1
                    
                    # Update valuation
                    pos_value = sum(positions[a]['amount'] * prices_mat[i][k] for k, a in enumerate(sorted_assets) if positions[a]['amount'] > 0 and not np.isnan(prices_mat[i][k]))
                    total_portfolio_value = cash + pos_value

    # Final Liquidation at the end of window
    final_ts = global_timeline[-1]
    for k, asset in enumerate(sorted_assets):
        pos = positions[asset]
        if pos['amount'] > 0:
            price = prices_mat[-1][k]
            if not np.isnan(price):
                revenue = pos['amount'] * price
                revenue_after_fee = revenue * (1.0 - args.fee)
                cash += revenue_after_fee
                
                profit_pct = (price - pos['entry_price']) / pos['entry_price']
                profit_usd = revenue_after_fee - pos['cost_basis']
                
                trade_log.append({
                    'asset': asset,
                    'entry_ts': pos['entry_ts'],
                    'exit_ts': final_ts,
                    'entry_price': pos['entry_price'],
                    'exit_price': price,
                    'profit_pct': profit_pct * 100,
                    'profit_usd': profit_usd,
                    'exit_reason': 'FORCE_LIQ'
                })
                positions[asset] = {'amount': 0.0, 'entry_price': 0.0, 'cost_basis': 0.0, 'entry_ts': None, 'holding_duration': 0}

    final_value = cash
    strategy_return = ((final_value - 1000) / 1000) * 100
    
    # 4. Generate Plot & Metrics
    df_history = pd.DataFrame(portfolio_history)
    df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
    df_history.set_index('timestamp', inplace=True)
    
    os.makedirs("reports", exist_ok=True)
    chart_path = "reports/gbdt_equity.png"
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_history.index, df_history['value'], label='GBDT Strategy Value', color='#38bdf8', linewidth=2)
    plt.plot(df_history.index, df_history['cash'], label='Free Cash', color='#475569', linestyle='--', alpha=0.6)
    plt.title(f"GBDT Portfolio Equity Curve ({args.window.upper()} Window)")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.15)
    plt.legend()
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Asset summaries for HTML report
    asset_summaries = []
    for asset in sorted_assets:
        asset_trades = [t['profit_pct'] for t in trade_log if t['asset'] == asset]
        if asset_trades:
            wins = len([t for t in asset_trades if t > 0])
            win_rate = (wins / len(asset_trades)) * 100
            asset_summaries.append({
                'Asset': asset,
                'Trades': len(asset_trades),
                'WinRate': win_rate,
                'AvgProfit': np.mean(asset_trades)
            })
            
    generate_html(asset_summaries, trade_log, final_value, cash, chart_path, os.path.basename(args.model))

    # Output Console Results
    print("\n" + "="*50)
    print("GBDT CHRONOLOGICAL PORTFOLIO SUMMARY")
    print("="*50)
    print(f"Backtest Window:    {args.window.upper()}")
    print(f"Starting Balance:   $1,000.00")
    print(f"Final Value:        ${final_value:,.2f}")
    print(f"Strategy Return:    {strategy_return:,.2f}%")
    print(f"Total Trades Taken: {len(trade_log)}")
    print("="*50)

    # Save summary report to JSON
    json_summary = {
        "model": args.model,
        "window": args.window,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "final_value": round(float(final_value), 2),
        "total_return": round(float(strategy_return), 2),
        "num_trades": len(trade_log),
        "trades": [{
            "asset": t['asset'],
            "entry_time": str(t['entry_ts']),
            "exit_time": str(t['exit_ts']),
            "profit_pct": round(float(t['profit_pct']), 2),
            "profit_usd": round(float(t['profit_usd']), 2)
        } for t in trade_log[-20:]]
    }
    
    json_out = "reports/test_results_gbdt.json"
    with open(json_out, 'w') as f:
        json.dump(json_summary, f, indent=4)
    print(f"[+] JSON metrics report saved to: {json_out}\n")

if __name__ == "__main__":
    main()
