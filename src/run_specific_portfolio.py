import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier_v2 import NeuralSentinelV2
from src.data.processor_v2 import DataProcessorV2

def prepare_asset_data(csv_path, lookback, horizon):
    """
    Worker function to process indicators on CPU using V2 processor.
    """
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 200: return None
        
        processor = DataProcessorV2(lookback=lookback, horizon=horizon)
        df = processor.add_indicators(df)
        df = processor.create_labels(df)
        
        features, _ = processor.prepare_features(df)
        
        return {
            'Asset': os.path.basename(csv_path).replace('_15m.csv', '').replace('_USDT', ''),
            'features': features,
            'prices': df['close'].values[lookback:].astype(np.float32),
            'opens': df['open'].values[lookback:].astype(np.float32),
            'timestamps': df['timestamp'].values[lookback:]
        }
    except Exception as e:
        print(f"Error preparing {csv_path}: {e}")
        return None

def generate_html(asset_summaries, trade_log, final_value, final_cash, chart_path, model_name, filename="reports/specific_audit.html"):
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
            <h1>🛡️ Portfolio Backtest Audit</h1>
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
    print(f"\n[*] Dashboard generated: {os.path.abspath(filename)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model .pth file")
    parser.add_argument("--entry", type=float, default=0.95)
    parser.add_argument("--exit", type=float, default=0.45)
    parser.add_argument("--fee", type=float, default=0.001) # Lower fee for realistic test
    parser.add_argument("--data-dir", type=str, default="data/evaluation", help="Directory containing OOS CSV data")
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--window", type=str, choices=['oos', 'is', 'all'], default='oos', help="Window to simulate: oos (<=2023), is (>=2024), or all")
    args = parser.parse_args()

    num_workers = max(1, os.cpu_count() - 2)
    print(f"[*] Starting specialized V2 portfolio simulation for {os.path.basename(args.model)}")
    
    csv_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".csv")]
    print(f"[*] Found {len(csv_files)} assets in {args.data_dir}")
    
    # PHASE 1: Parallel Data Preparation
    print("[*] Preparing data...")
    prepared_data = []
    lookback = 100
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(prepare_asset_data, f, lookback, 16) for f in csv_files]
        for future in tqdm(as_completed(futures), total=len(csv_files), desc="Processing CSVs"):
            res = future.result()
            if res:
                prepared_data.append(res)

    if not prepared_data:
        print("[!] No data prepared.")
        return

    # PHASE 2: PyTorch Inference
    print("[*] Running V2 PyTorch inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralSentinelV2(input_dim=9, hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    aligned_signals = {}
    for data in tqdm(prepared_data, desc="Generating Signals"):
        features = data['features']
        
        # Sliding window for inference
        X_view = np.lib.stride_tricks.sliding_window_view(features, (lookback, features.shape[1]))
        X_view = X_view.squeeze(1)
        X_np = X_view[:len(data['prices'])]
        
        batch_size = 4096
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X_np), batch_size):
                batch_x = torch.from_numpy(X_np[i : i + batch_size]).float().to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(batch_x)
                    batch_probs = torch.sigmoid(logits).to(torch.float32).cpu().numpy().flatten()
                    all_probs.append(batch_probs)
        
        probs = np.concatenate(all_probs)
        aligned_signals[data['Asset']] = {
            'prices': data['prices'],
            'opens': data['opens'],
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
    df_probs = pd.DataFrame(index=global_timeline)
    
    for asset, asset_data in aligned_signals.items():
        df_prices[asset] = pd.Series(asset_data['prices'], index=asset_data['timestamps'])
        df_opens[asset] = pd.Series(asset_data['opens'], index=asset_data['timestamps'])
        df_probs[asset] = pd.Series(asset_data['probs'], index=asset_data['timestamps'])
    
    df_prices = df_prices.ffill()
    df_opens = df_opens.ffill()
    
    # Filter for window
    if args.window == 'oos':
        print("[*] Filtering timeline to Out-Of-Sample window (<= 2023)...")
        valid_idx = pd.to_datetime(df_prices.index).year <= 2023
    elif args.window == 'is':
        print("[*] Filtering timeline to In-Sample window (>= 2024)...")
        valid_idx = pd.to_datetime(df_prices.index).year >= 2024
    else:
        print("[*] Using full timeline...")
        valid_idx = [True] * len(df_prices)
        
    df_prices = df_prices[valid_idx]
    df_opens = df_opens[valid_idx]
    df_probs = df_probs[valid_idx]
    global_timeline = list(df_prices.index)
    
    sorted_assets = sorted(list(aligned_signals.keys()))

    # PHASE 4: Chronological Simulation
    print("[*] Running realistic portfolio simulation (10% Allocation)...")
    cash = 1000.0
    positions = {asset: {'amount': 0.0, 'entry_price': 0.0, 'cost_basis': 0.0, 'entry_ts': None} for asset in sorted_assets}
    
    portfolio_history = []
    trade_log = []
    
    prices_mat = df_prices[sorted_assets].values
    opens_mat = df_opens[sorted_assets].values
    probs_mat = df_probs[sorted_assets].values
    
    for i in tqdm(range(len(global_timeline)), desc="Simulating Timeline"):
        ts = global_timeline[i]
        current_prices = prices_mat[i]
        current_probs = probs_mat[i]
        
        pos_value = 0.0
        for j, asset in enumerate(sorted_assets):
            if positions[asset]['amount'] > 0 and not np.isnan(current_prices[j]):
                pos_value += positions[asset]['amount'] * current_prices[j]
        
        total_portfolio_value = cash + pos_value
        portfolio_history.append({'timestamp': ts, 'value': total_portfolio_value, 'cash': cash})
        
        for j, asset in enumerate(sorted_assets):
            prob = current_probs[j]
            price = current_prices[j]
            if np.isnan(prob) or np.isnan(price): continue
                
            pos = positions[asset]
            
            # Exit
            if pos['amount'] > 0 and prob <= args.exit:
                exec_price = opens_mat[i+1][j] if i+1 < len(global_timeline) else current_prices[j]
                if np.isnan(exec_price): continue
                
                revenue = pos['amount'] * exec_price
                revenue_after_fee = revenue * (1.0 - args.fee)
                cash += revenue_after_fee
                
                profit_pct = (exec_price - pos['entry_price']) / pos['entry_price']
                profit_usd = revenue_after_fee - pos['cost_basis']
                
                trade_log.append({
                    'asset': asset, 
                    'entry_ts': pos['entry_ts'],
                    'exit_ts': global_timeline[i+1] if i+1 < len(global_timeline) else ts,
                    'entry_price': pos['entry_price'],
                    'exit_price': exec_price,
                    'profit_pct': profit_pct * 100, 
                    'profit_usd': profit_usd
                })
                pos['amount'] = 0.0
                
                # Immediate value update
                pos_value = sum(positions[a]['amount'] * prices_mat[i][k] for k, a in enumerate(sorted_assets) if positions[a]['amount'] > 0 and not np.isnan(prices_mat[i][k]))
                total_portfolio_value = cash + pos_value

            # Entry
            elif pos['amount'] == 0 and prob >= args.entry:
                trade_allocation = total_portfolio_value * 0.10
                if cash >= trade_allocation:
                    exec_price = opens_mat[i+1][j] if i+1 < len(global_timeline) else current_prices[j]
                    if np.isnan(exec_price): continue
                    
                    alloc_after_fee = trade_allocation * (1.0 - args.fee)
                    amount_bought = alloc_after_fee / exec_price
                    
                    cash -= trade_allocation
                    pos['amount'] = amount_bought
                    pos['entry_price'] = exec_price
                    pos['cost_basis'] = trade_allocation
                    pos['entry_ts'] = global_timeline[i+1] if i+1 < len(global_timeline) else ts
                    
                    pos_value = sum(positions[a]['amount'] * prices_mat[i][k] for k, a in enumerate(sorted_assets) if positions[a]['amount'] > 0 and not np.isnan(prices_mat[i][k]))
                    total_portfolio_value = cash + pos_value

    # Final Liquidation
    for k, asset in enumerate(sorted_assets):
        pos = positions[asset]
        if pos['amount'] > 0:
            price = prices_mat[-1][k]
            if not np.isnan(price):
                revenue = pos['amount'] * price
                revenue_after_fee = revenue * (1.0 - args.fee)
                cash += revenue_after_fee
                trade_log.append({
                    'asset': asset, 'entry_ts': pos['entry_ts'], 'exit_ts': global_timeline[-1],
                    'entry_price': pos['entry_price'], 'exit_price': price,
                    'profit_pct': ((price / pos['entry_price']) - 1) * 100, 
                    'profit_usd': revenue_after_fee - pos['cost_basis']
                })

    final_value = cash
    print(f"\n[+] Simulation Complete. Final Value: ${final_value:.2f}")
    
    # Generate Chart
    df_history = pd.DataFrame(portfolio_history)
    df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
    df_history.set_index('timestamp', inplace=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_history.index, df_history['value'], label='Total Portfolio Value', color='#38bdf8')
    plt.title(f'Portfolio Equity Curve: {os.path.basename(args.model)}')
    plt.ylabel('Value ($)')
    plt.grid(True, alpha=0.2)
    chart_path = "reports/specific_equity.png"
    plt.savefig(chart_path)
    
    # Asset Summary
    asset_summaries = []
    for asset in sorted_assets:
        asset_trades = [t['profit_pct'] for t in trade_log if t['asset'] == asset]
        if asset_trades:
            win_rate = (len([t for t in asset_trades if t > 0]) / len(asset_trades)) * 100
            asset_summaries.append({
                'Asset': asset, 'Trades': len(asset_trades),
                'WinRate': win_rate, 'AvgProfit': np.mean(asset_trades)
            })
            
    generate_html(asset_summaries, trade_log, final_value, cash, chart_path, os.path.basename(args.model))

if __name__ == "__main__":
    main()
