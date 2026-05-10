import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor import DataProcessor

def prepare_asset_data(csv_path, lookback, horizon):
    """
    Worker function to process indicators on CPU.
    """
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 200: return None
        
        processor = DataProcessor(lookback=lookback, horizon=horizon)
        df = processor.add_indicators(df)
        df = processor.create_labels(df)
        
        features, _ = processor.prepare_features(df)
        
        return {
            'Asset': os.path.basename(csv_path).replace('_15m.csv', ''),
            'features': features,
            'prices': df['close'].values[lookback:].astype(np.float32),
            'opens': df['open'].values[lookback:].astype(np.float32),
            'timestamps': df['timestamp'].values[lookback:]
        }
    except Exception as e:
        print(f"Error preparing {csv_path}: {e}")
        return None

def generate_html(asset_summaries, trade_log, final_value, final_cash, chart_path, filename="reports/audit.html"):
    os.makedirs("reports", exist_ok=True)
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Sentinel V1 - Chronological Audit</title>
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
            <h1>🛡️ Chronological Portfolio Backtest (5% Dynamic Allocation)</h1>
            
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
            chart_name=os.path.basename(chart_path)
        ))
    print(f"\n[*] Dashboard generated: {os.path.abspath(filename)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", type=float, default=0.95)
    parser.add_argument("--exit", type=float, default=0.3)
    parser.add_argument("--fee", type=float, default=0.01)
    parser.add_argument("--days", type=int, default=None, help="Only run on the last N days")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing CSV data")
    args = parser.parse_args()

    num_workers = max(1, os.cpu_count() - 2)
    print(f"[*] Using {num_workers} CPU cores for data preparation (leaving 2 free).")
    
    # Priority order from eToro assets
    priority_order = []
    if os.path.exists("data/etoro_assets.txt"):
        df_etoro = pd.read_csv("data/etoro_assets.txt")
        priority_order = df_etoro['Symbol'].tolist()
        
    # Ignored stablecoins
    ignored_stables = set()
    if os.path.exists("data/stables_ignore.txt"):
        with open("data/stables_ignore.txt", "r") as f:
            for line in f:
                pair = line.strip()
                if pair:
                    ignored_stables.add(pair.replace("/", "_"))

    # Anti-Cheating: Restrict strictly to top 10 assets
    top_10_assets = ['BTC', 'ETH', 'BCH', 'XRP', 'DASH', 'LTC', 'ETC', 'API3', 'CRO', 'ETHFI']
    
    csv_files = []
    if os.path.exists(args.data_dir):
        for f in os.listdir(args.data_dir):
            if f.endswith(".csv"):
                base_symbol = f.split('_')[0]
                if base_symbol in top_10_assets and not any(stable in f for stable in ignored_stables):
                    csv_files.append(os.path.join(args.data_dir, f))
                
    print(f"[*] Found {len(csv_files)} valid Top 10 assets to process in {args.data_dir}.")
    
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
    import torch
    from src.models.classifier import NeuralSentinelV1
    
    print("[*] Running PyTorch inference on GPU/CPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralSentinelV1(input_dim=8).to(device)
    model.load_state_dict(torch.load("models/sentinel_v1_slim.pth", map_location=device, weights_only=True))
    model.eval()

    aligned_signals = {}
    for data in tqdm(prepared_data, desc="Generating Signals"):
        features = data['features']
        
        X_view = np.lib.stride_tricks.sliding_window_view(features, (lookback, features.shape[1]))
        X_view = X_view.squeeze(1)
        X_np = X_view[:len(data['prices'])]
        
        X_np_copy = X_np.copy()
        batch_size = 1024
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X_np_copy), batch_size):
                batch_x = torch.from_numpy(np.copy(X_np_copy[i : i + batch_size])).float().to(device)
                logits, _ = model(batch_x)
                batch_probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.append(batch_probs)
        
        probs = np.concatenate(all_probs)
        aligned_signals[data['Asset']] = {
            'prices': data['prices'],
            'opens': data['opens'],
            'probs': probs,
            'timestamps': data['timestamps']
        }
        
        # Cleanup memory
        del data, X_np, X_view, probs, X_np_copy, all_probs

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
    
    # Forward fill prices so portfolio value is accurate even if an asset is temporarily missing a candle
    df_prices = df_prices.ffill()
    df_opens = df_opens.ffill()
    
    # Anti-Cheating: Hardcode Out-Of-Sample Timeline based on dataset
    if "2022" in args.data_dir:
        print("[*] Filtering timeline to strict Out-Of-Sample window (2022-01-01 to 2023-12-31)...")
        valid_idx = (df_prices.index >= '2022-01-01 00:00:00') & (df_prices.index <= '2023-12-31 23:59:59')
    else:
        print("[*] Filtering timeline to strict Out-Of-Sample window (2023-09-01 to 2023-12-01)...")
        valid_idx = (df_prices.index >= '2023-09-01 00:00:00') & (df_prices.index <= '2023-12-01 00:00:00')
        
    df_prices = df_prices[valid_idx]
    df_opens = df_opens[valid_idx]
    df_probs = df_probs[valid_idx]
    global_timeline = list(df_prices.index)
    
    # Determine eToro priority order
    def get_priority(asset_name):
        base_symbol = asset_name.split('_')[0]
        if base_symbol in top_10_assets:
            return top_10_assets.index(base_symbol)
        return 999999
        
    sorted_assets = sorted(list(aligned_signals.keys()), key=get_priority)

    # PHASE 4: Chronological Simulation
    print("[*] Running realistic out-of-sample simulation (10% Dynamic Allocation)...")
    cash = 1000.0
    positions = {asset: {'amount': 0.0, 'entry_price': 0.0, 'cost_basis': 0.0, 'entry_ts': None, 'trades': 0} for asset in sorted_assets}
    
    portfolio_history = []
    trade_log = []
    
    # Convert DataFrames to numpy arrays for much faster iteration
    prices_mat = df_prices[sorted_assets].values
    opens_mat = df_opens[sorted_assets].values
    probs_mat = df_probs[sorted_assets].values
    
    for i in tqdm(range(len(global_timeline)), desc="Simulating Timeline"):
        ts = global_timeline[i]
        current_prices = prices_mat[i]
        current_probs = probs_mat[i]
        
        # 1. Update Portfolio Value
        pos_value = 0.0
        for j, asset in enumerate(sorted_assets):
            if positions[asset]['amount'] > 0 and not np.isnan(current_prices[j]):
                pos_value += positions[asset]['amount'] * current_prices[j]
        
        total_portfolio_value = cash + pos_value
        portfolio_history.append({'timestamp': ts, 'value': total_portfolio_value, 'cash': cash})
        
        # 2. Process exits and entries in priority order
        for j, asset in enumerate(sorted_assets):
            prob = current_probs[j]
            price = current_prices[j]
            
            # Skip if missing data
            if np.isnan(prob) or np.isnan(price):
                continue
                
            pos = positions[asset]
            
            # Check Exit
            if pos['amount'] > 0 and prob <= args.exit:
                # Execution Delay: Execute at the open price of the NEXT candle (i+1)
                # If we are at the very last candle, we just use current close to force-liquidate later
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
                
                # Reset position
                pos['amount'] = 0.0
                
                # Update total portfolio value immediately so new cash can be used
                pos_value = sum(positions[a]['amount'] * prices_mat[i][k] for k, a in enumerate(sorted_assets) if positions[a]['amount'] > 0 and not np.isnan(prices_mat[i][k]))
                total_portfolio_value = cash + pos_value

            # Check Entry
            elif pos['amount'] == 0 and prob >= args.entry:
                trade_allocation = total_portfolio_value * 0.10
                # Check if we have enough cash
                if cash >= trade_allocation:
                    # Execution Delay: Execute at the open price of the NEXT candle
                    exec_price = opens_mat[i+1][j] if i+1 < len(global_timeline) else current_prices[j]
                    if np.isnan(exec_price): continue
                    
                    alloc_after_fee = trade_allocation * (1.0 - args.fee)
                    amount_bought = alloc_after_fee / exec_price
                    
                    cash -= trade_allocation
                    pos['amount'] = amount_bought
                    pos['entry_price'] = exec_price
                    pos['cost_basis'] = trade_allocation
                    pos['entry_ts'] = global_timeline[i+1] if i+1 < len(global_timeline) else ts
                    pos['trades'] += 1
                    
                    # Update total portfolio value
                    pos_value = sum(positions[a]['amount'] * prices_mat[i][k] for k, a in enumerate(sorted_assets) if positions[a]['amount'] > 0 and not np.isnan(prices_mat[i][k]))
                    total_portfolio_value = cash + pos_value

    # Force-liquidate all remaining positions at the very last timestamp
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
                    'profit_usd': profit_usd
                })
                pos['amount'] = 0.0

    final_value = cash
    
    print("\n" + "="*30)
    print(" CHRONOLOGICAL PORTFOLIO SUMMARY")
    print("="*30)
    print(f"Starting Balance:   $1000.00")
    print(f"Final Balance:      ${final_value:.2f}")
    print(f"Final Free Cash:    ${cash:.2f}")
    print(f"Total Return:       {((final_value - 1000) / 1000 * 100):.2f}%")
    print(f"Total Trades Taken: {len(trade_log)}")
    print("="*30)
    
    # Generate Chart
    df_history = pd.DataFrame(portfolio_history)
    df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
    df_history.set_index('timestamp', inplace=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_history.index, df_history['value'], label='Total Portfolio Value', color='blue')
    plt.plot(df_history.index, df_history['cash'], label='Free Cash', color='green', alpha=0.5)
    plt.title('Chronological Portfolio Backtest (5% Dynamic Allocation)')
    plt.ylabel('Value ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    chart_path = "reports/equity_curve.png"
    plt.savefig(chart_path)
    print(f"[*] Equity curve saved to {chart_path}")
    
    # Generate HTML summary
    asset_summaries = []
    for asset in sorted_assets:
        asset_trades = [t['profit_pct'] for t in trade_log if t['asset'] == asset]
        if len(asset_trades) > 0:
            wins = len([t for t in asset_trades if t > 0])
            win_rate = (wins / len(asset_trades)) * 100
            avg_profit = np.mean(asset_trades)
            asset_summaries.append({
                'Asset': asset,
                'Trades': len(asset_trades),
                'WinRate': win_rate,
                'AvgProfit': avg_profit
            })
            
    generate_html(asset_summaries, trade_log, final_value, cash, chart_path)

if __name__ == "__main__":
    main()
