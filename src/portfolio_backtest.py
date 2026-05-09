import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor import DataProcessor

def prepare_asset_data(csv_path, lookback, horizon):
    """
    Worker function to process indicators on CPU.
    Returns lightweight 2D features to prevent memory/pickling errors.
    NOTE: We DO NOT import torch here to prevent Windows from loading massive 
    CUDA DLLs into every worker process, which causes "Paging file too small" crashes.
    """
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 200: return None
        
        processor = DataProcessor(lookback=lookback, horizon=horizon)
        df = processor.add_indicators(df)
        df = processor.create_labels(df)
        
        # Returns 2D matrix (N, Features) instead of massive 3D array
        features, _ = processor.prepare_features(df)
        
        return {
            'Asset': os.path.basename(csv_path).replace('_15m.csv', ''),
            'features': features,
            'prices': df['close'].values[lookback:].astype(np.float32),
            'timestamps': df['timestamp'].values[lookback:]
        }
    except Exception as e:
        print(f"Error preparing {csv_path}: {e}")
        return None

def simulate_trades(data, probs, entry_threshold, exit_threshold, fee):
    """Worker function to run trading logic on CPU."""
    balance = 100.0
    in_position = False
    entry_price = 0
    trades = []
    cash_count = 0
    prices = data['prices']
    
    for i in range(len(probs)):
        p = probs[i]
        cur = prices[i]
        
        if not in_position:
            cash_count += 1
            if p >= entry_threshold:
                in_position = True
                entry_price = cur
                balance *= (1.0 - fee)
                cash_count -= 1
        elif in_position and p <= exit_threshold:
            in_position = False
            profit = (cur - entry_price) / entry_price
            balance *= (1.0 + profit)
            balance *= (1.0 - fee)
            trades.append(profit - (fee*2))

    win_rate = (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0
    pnl = (balance - 100.0)
    cash_pct = (cash_count / len(probs) * 100) if len(probs) > 0 else 100
    
    return {
        'Asset': data['Asset'],
        'Trades': len(trades),
        'WinRate': round(win_rate, 2),
        'CashPct': round(cash_pct, 1),
        'PnL': round(pnl, 2),
        'FinalBalance': round(balance, 2)
    }

def generate_html(results, filename="reports/audit.html"):
    os.makedirs("reports", exist_ok=True)
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Sentinel V1 - Audit Report</title>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; padding: 40px; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: #1e293b; padding: 20px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); }}
            h1 {{ color: #38bdf8; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            .pnl-pos {{ color: #4ade80; font-weight: bold; }}
            .pnl-neg {{ color: #f87171; font-weight: bold; }}
            #auditTable_wrapper {{ color: #f8fafc; }}
            select, input {{ background: #334155 !important; color: white !important; border: 1px solid #475569 !important; }}
            .dataTables_info {{ margin-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Neural Sentinel V1 - Portfolio Audit</h1>
            <table id="auditTable" class="display">
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th>Trades</th>
                        <th>Win Rate (%)</th>
                        <th>Time in Cash (%)</th>
                        <th>PnL (%)</th>
                        <th>Final Balance ($)</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        <script type="text/javascript" src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script>
            $(document).ready(function() {{
                $('#auditTable').DataTable({{
                    order: [[4, 'desc']],
                    pageLength: 25
                }});
            }});
        </script>
    </body>
    </html>
    """
    rows = ""
    for r in results:
        pnl_class = "pnl-pos" if r['PnL'] >= 0 else "pnl-neg"
        rows += f"""
        <tr>
            <td>{r['Asset']}</td>
            <td>{r['Trades']}</td>
            <td>{r['WinRate']}%</td>
            <td>{r['CashPct']}%</td>
            <td class="{pnl_class}">{r['PnL']}%</td>
            <td>${r['FinalBalance']:,.2f}</td>
        </tr>
        """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_template.format(rows=rows))
    print(f"\n[*] Dashboard generated: {os.path.abspath(filename)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", type=float, default=0.95)
    parser.add_argument("--exit", type=float, default=0.3)
    parser.add_argument("--fee", type=float, default=0.01)
    args = parser.parse_args()

    # Determine CPU resources (Leave 2 free as requested)
    num_workers = max(1, os.cpu_count() - 2)
    print(f"[*] Using {num_workers} CPU cores for data preparation (leaving 2 free).")
    
    csv_files = [os.path.join("data/raw", f) for f in os.listdir("data/raw") if f.endswith(".csv")]
    
    # 1. Parallel Data Preparation
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

    # 2. Main Process Inference
    # We import PyTorch ONLY in the main process. This prevents the WinError 1455 crash!
    import torch
    from src.models.classifier import NeuralSentinelV1
    
    print("[*] Running inference on GPU/CPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralSentinelV1(input_dim=8).to(device)
    model.load_state_dict(torch.load("models/sentinel_v1_slim.pth", map_location=device, weights_only=True))
    model.eval()

    results = []
    for data in tqdm(prepared_data, desc="Simulating Trades"):
        features = data['features']
        
        # Zero-copy memory efficient windowing
        X_view = np.lib.stride_tricks.sliding_window_view(features, (lookback, features.shape[1]))
        X_view = X_view.squeeze(1)
        X_np = X_view[:len(data['prices'])]
        
        # Inference in batches to prevent CUDA OutOfMemoryError
        X_np_copy = X_np.copy()
        batch_size = 1024
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X_np_copy), batch_size):
                # Copying the sliced batch view to completely silence PyTorch's "non-writable" warning
                batch_x = torch.from_numpy(np.copy(X_np_copy[i : i + batch_size])).float().to(device)
                logits, _ = model(batch_x)
                batch_probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.append(batch_probs)
        
        probs = np.concatenate(all_probs)
            
        # Simulation
        res = simulate_trades(data, probs, args.entry, args.exit, args.fee)
        results.append(res)
        
        # Clean up
        del data, X_np, X_view, probs, X_np_copy, all_probs

    # 3. Report
    generate_html(results)
    df_res = pd.DataFrame(results)
    print("\n" + "="*30)
    print(" PORTFOLIO SUMMARY")
    print("="*30)
    print(f"Total Assets:    {len(results)}")
    print(f"Avg Win Rate:    {df_res['WinRate'].mean():.2f}%")
    print(f"Profitable:      {len(df_res[df_res['PnL'] > 0])} / {len(results)}")
    print(f"Best Performer:  {df_res.iloc[df_res['PnL'].idxmax()]['Asset']} ({df_res['PnL'].max()}%)")
    print("="*30)

if __name__ == "__main__":
    main()
