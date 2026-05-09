import ccxt
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm

class DataFetcher:
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        self.data_dir = 'data/raw'
        self.stables_file = 'data/stables_ignore.txt'
        self.whitelist_file = 'data/etoro_assets.txt'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load ignored stables (Rule [3])
        self.ignored_stables = set()
        if os.path.exists(self.stables_file):
            with open(self.stables_file, 'r') as f:
                self.ignored_stables = set(line.strip() for line in f if line.strip())

    def get_top_assets(self, limit=100):
        """Fetch top assets by volume, PLUS whitelist, excluding known stables."""
        print(f"Fetching market data...")
        tickers = self.exchange.fetch_tickers()
        
        # 1. Load Whitelist from eToro list
        whitelist = set()
        if os.path.exists(self.whitelist_file):
            try:
                df_white = pd.read_csv(self.whitelist_file)
                for s in df_white['Symbol'].unique():
                    symbol = f"{s}/USDT"
                    # Only add if it actually exists on Binance
                    if symbol in tickers:
                        whitelist.add(symbol)
                    else:
                        # Silently skip CRO, etc.
                        pass
                print(f"Loaded {len(whitelist)} priority assets from whitelist.")
            except Exception as e:
                print(f"Warning: Could not load whitelist: {e}")

        # Only fetch /USDT pairs that are NOT in our blacklist
        usdt_pairs = []
        for symbol, t in tickers.items():
            if symbol.endswith('/USDT') and symbol not in self.ignored_stables:
                usdt_pairs.append(t)
        
        # Sort by volume
        sorted_pairs = sorted(usdt_pairs, key=lambda x: x['quoteVolume'], reverse=True)
        top_symbols = [p['symbol'] for p in sorted_pairs[:limit]]
        
        # Combine Whitelist + Top Volume (Unique set)
        final_list = list(set(top_symbols) | whitelist)
        
        # Final safety filter against blacklist (in case whitelist contained a stable)
        final_list = [s for s in final_list if s not in self.ignored_stables]
        
        print(f"Total targets: {len(final_list)} (Top {limit} + Whitelist)")
        return final_list

    def fetch_ohlcv(self, symbol, timeframe='15m', days=365):
        """Fetch historical OHLCV for a symbol with a stability check."""
        since = self.exchange.parse8601(str(datetime.now() - timedelta(days=days)))
        
        # 1. Fetch a small sample first to check if it's a stablecoin
        sample = self.exchange.fetch_ohlcv(symbol, timeframe, limit=200)
        if not sample:
            return None
            
        prices = [c[4] for c in sample] # Close prices
        avg_price = np.mean(prices)
        med_price = np.median(prices)
        price_spread = max(prices) / min(prices)
        p_high, p_low = max(prices), min(prices)
        
        print(f" Checking {symbol:10} | Avg: {avg_price:10.4f} | Med: {med_price:10.4f} | Range: [{p_low:.4f} - {p_high:.4f}] | Spread: {price_spread:.4f}")

        # Robust Filter: 
        # 1. Is it priced near $1.00? (0.98 - 1.02)
        # 2. Is the total movement range less than 3%? (spread < 1.03)
        if (0.98 < avg_price < 1.02) and (price_spread < 1.03):
            print(f" Skipping {symbol}: Stablecoin/Pegged asset detected.")
            # Add to ignore list for future runs (Rule [3])
            if symbol not in self.ignored_stables:
                with open(self.stables_file, 'a') as f:
                    f.write(f"{symbol}\n")
                self.ignored_stables.add(symbol)
            return None

        # 2. Proceed with full download if it's a volatile asset
        all_ohlcv = []
        
        filename = os.path.join(self.data_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")
        
        # Check if local data exists and determine start point (Rule [3])
        existing_data = None
        if os.path.exists(filename):
            try:
                existing_data = pd.read_csv(filename)
                if not existing_data.empty:
                    # Set 'since' to the timestamp of the last candle
                    last_ts_str = existing_data['timestamp'].iloc[-1]
                    last_ts_dt = datetime.strptime(last_ts_str, '%Y-%m-%d %H:%M:%S')
                    
                    # Check age: If less than 2 days old, skip to save time (Rule [3])
                    age = datetime.now() - last_ts_dt
                    if age < timedelta(days=2):
                        print(f"  [=] {symbol}: Data is fresh ({age.days}d {age.seconds//3600}h old). Skipping.")
                        return

                    since = self.exchange.parse8601(last_ts_dt.isoformat())
                    print(f"  [+] {symbol}: Resuming from {last_ts_str}")
            except Exception as e:
                print(f"  [!] Error reading existing file {filename}: {e}. Redownloading...")

        try:
            while since < self.exchange.milliseconds():
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                time.sleep(self.exchange.rateLimit / 1000)

            if all_ohlcv:
                df_new = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
                
                # If we have existing data, avoid duplicates and append
                if existing_data is not None:
                    # Filter out any candles we already have (just in case of overlap)
                    last_dt = pd.to_datetime(existing_data['timestamp'].iloc[-1])
                    df_new = df_new[df_new['timestamp'] > last_dt]
                    
                    if not df_new.empty:
                        df_new.to_csv(filename, mode='a', header=False, index=False)
                        print(f"  [▲] {symbol}: Appended {len(df_new)} new candles.")
                else:
                    df_new.to_csv(filename, index=False)
                    print(f"  [✓] {symbol}: Downloaded {len(df_new)} candles.")
            else:
                print(f"  [=] {symbol}: Already up to date.")
        except Exception as e:
            print(f"  [!] Error fetching {symbol}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=100, help='Number of top assets')
    parser.add_argument('--timeframe', type=str, default='15m', help='Candle timeframe')
    parser.add_argument('--days', type=int, default=365, help='Days of history')
    args = parser.parse_args()

    fetcher = DataFetcher()
    symbols = fetcher.get_top_assets(limit=args.limit)
    
    print(f"Starting download for {len(symbols)} assets...")
    for symbol in tqdm(symbols):
        fetcher.fetch_ohlcv(symbol, timeframe=args.timeframe, days=args.days)
