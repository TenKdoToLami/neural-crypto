import ccxt
import pandas as pd
import os
import time
from datetime import datetime
from tqdm import tqdm

def fetch_historical_data():
    exchange = ccxt.binance({'enableRateLimit': True})
    data_dir = 'data/raw_2022'
    os.makedirs(data_dir, exist_ok=True)
    
    # Top 10 assets to fetch
    top_10_assets = ['BTC', 'ETH', 'BCH', 'XRP', 'DASH', 'LTC', 'ETC', 'API3', 'CRO', 'ETHFI']
    timeframe = '15m'
    
    start_dt = datetime(2022, 1, 1)
    end_dt = datetime(2023, 12, 31, 23, 59, 59)
    
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    print(f"[*] Starting historical fetch for {len(top_10_assets)} assets.")
    print(f"[*] Date Range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    print(f"[*] Saving to: {data_dir}/\n")
    
    for symbol_base in top_10_assets:
        symbol = f"{symbol_base}/USDT"
        filename = os.path.join(data_dir, f"{symbol_base}_USDT_{timeframe}.csv")
        
        # Check if file exists and has full data
        if os.path.exists(filename):
            print(f"  [=] {symbol_base}: File already exists. Skipping.")
            continue
            
        print(f"  [+] Fetching {symbol_base}...")
        
        # Check if market exists
        try:
            exchange.load_markets()
            if symbol not in exchange.markets:
                print(f"  [!] {symbol_base} not found on Binance USDT spot. Skipping.")
                continue
        except Exception as e:
            pass

        all_ohlcv = []
        current_ts = start_ts
        
        # Binance allows 1000 candles per request.
        # 2 years of 15m candles = ~70,080 candles = ~71 requests per asset.
        pbar = tqdm(total=end_ts, initial=start_ts, desc=f"{symbol_base}", leave=False)
        
        try:
            while current_ts < end_ts:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
                if not ohlcv:
                    break
                    
                # Filter out candles beyond our end date just in case
                valid_ohlcv = [c for c in ohlcv if c[0] <= end_ts]
                if not valid_ohlcv:
                    break
                    
                all_ohlcv.extend(valid_ohlcv)
                
                # Advance timestamp to the next candle after the last fetched
                last_fetched_ts = valid_ohlcv[-1][0]
                current_ts = last_fetched_ts + 1
                
                pbar.update(current_ts - pbar.n)
                time.sleep(exchange.rateLimit / 1000)
                
            pbar.close()
            
            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.to_csv(filename, index=False)
                print(f"  [✓] {symbol_base}: Saved {len(df)} candles.")
            else:
                print(f"  [!] {symbol_base}: No data returned for this period.")
                
        except Exception as e:
            pbar.close()
            print(f"  [!] Error fetching {symbol_base}: {e}")

if __name__ == "__main__":
    fetch_historical_data()
