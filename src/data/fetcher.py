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
        self.whitelist_file = 'data/approved_assets.txt'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load ignored stables (Rule [3])
        self.ignored_stables = set()
        if os.path.exists(self.stables_file):
            with open(self.stables_file, 'r') as f:
                self.ignored_stables = set(line.strip() for line in f if line.strip())

    def get_top_assets(self, limit=100):
        """Fetch symbols from whitelist, existing files, or volume leaders."""
        # 1. Always load Whitelist from approved assets first
        whitelist = set()
        if os.path.exists(self.whitelist_file):
            try:
                # We need tickers to validate symbol existence
                tickers = self.exchange.fetch_tickers()
                with open(self.whitelist_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            symbol = f"{line}/USDT" if '/' not in line else line
                            if symbol in tickers:
                                whitelist.add(symbol)
                print(f"Loaded {len(whitelist)} priority assets from approved list.")
            except Exception as e:
                print(f"Warning: Could not load approved assets: {e}")

        # 2. Check for existing files in data/raw
        existing_symbols = []
        if os.path.exists(self.data_dir):
            for f in os.listdir(self.data_dir):
                if f.endswith('.csv'):
                    symbol = f.split('_15m.csv')[0].replace('_', '/')
                    existing_symbols.append(symbol)
        
        # 3. Decision Logic
        if existing_symbols:
            # Combine existing files with the whitelist (to catch any new additions)
            final_list = list(set(existing_symbols) | whitelist)
            print(f"Found {len(existing_symbols)} existing assets. Merged with whitelist -> {len(final_list)} targets.")
            return final_list

        # 4. Fallback to Volume Leaders (for initial setup)
        print(f"No existing data. Fetching Top {limit} volume leaders...")
        if 'tickers' not in locals():
            tickers = self.exchange.fetch_tickers()
            
        usdt_pairs = []
        for symbol, t in tickers.items():
            if symbol.endswith('/USDT') and symbol not in self.ignored_stables:
                usdt_pairs.append(t)
        
        sorted_pairs = sorted(usdt_pairs, key=lambda x: x['quoteVolume'], reverse=True)
        top_symbols = [p['symbol'] for p in sorted_pairs[:limit]]
        
        # Combine Whitelist + Top Volume (De-duplicate by base asset)
        final_list = []
        seen_bases = set()
        
        # 1. Priority: Whitelist (usually USDC)
        for s in whitelist:
            base = s.split('/')[0]
            if base not in seen_bases:
                final_list.append(s)
                seen_bases.add(base)
        
        # 2. Secondary: Top Volume (usually USDT)
        for s in top_symbols:
            base = s.split('/')[0]
            if base not in seen_bases:
                final_list.append(s)
                seen_bases.add(base)

        # Final safety filter against blacklist
        final_list = [s for s in final_list if s not in self.ignored_stables]
        
        print(f"Total unique targets: {len(final_list)} (Base de-duplication active)")
        return final_list

    def fetch_ohlcv(self, symbol, timeframe='15m', days=365, since_date=None):
        """Fetch historical OHLCV for a symbol with a stability check."""
        if since_date:
            since = self.exchange.parse8601(f"{since_date}T00:00:00Z")
        else:
            since = self.exchange.parse8601(str(datetime.now() - timedelta(days=days)))
        
        # 1. Fetch a small sample first to check if it's a stablecoin
        try:
            sample = self.exchange.fetch_ohlcv(symbol, timeframe, limit=200)
        except Exception as e:
            print(f"  [!] Connection error for {symbol}: {e}")
            return None

        if not sample:
            return None
            
        prices = [c[4] for c in sample] # Close prices
        avg_price = np.mean(prices)
        price_spread = max(prices) / min(prices)
        
        # Robust Filter for stablecoins
        if (0.98 < avg_price < 1.02) and (price_spread < 1.03):
            print(f" Skipping {symbol}: Stablecoin/Pegged asset detected.")
            if symbol not in self.ignored_stables:
                with open(self.stables_file, 'a') as f:
                    f.write(f"{symbol}\n")
                self.ignored_stables.add(symbol)
            return None

        # 2. Proceed with full download
        all_ohlcv = []
        filename = os.path.join(self.data_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")
        
        existing_data = None
        if os.path.exists(filename):
            try:
                existing_data = pd.read_csv(filename)
                if not existing_data.empty:
                    last_ts_str = existing_data['timestamp'].iloc[-1]
                    last_ts_dt = datetime.strptime(last_ts_str, '%Y-%m-%d %H:%M:%S')
                    
                    # If since_date is provided, we might need to fetch older data if missing
                    # but usually, we just resume from the end.
                    since = max(since, self.exchange.parse8601(last_ts_dt.isoformat()))
                    
                    # Check age: If less than 3 days old, skip (Rule [3])
                    age = datetime.now() - last_ts_dt
                    if age < timedelta(days=3) and not since_date:
                        print(f"  [=] {symbol}: Data is fresh ({age.days}d {age.seconds//3600}h old). Skipping.")
                        return
            except Exception as e:
                print(f"  [!] Error reading existing file {filename}: {e}. Redownloading...")

        try:
            while since < self.exchange.milliseconds():
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                if len(ohlcv) < 1000: break
                time.sleep(self.exchange.rateLimit / 1000)

            if all_ohlcv:
                df_new = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
                
                if existing_data is not None:
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
    parser.add_argument('--days', type=int, default=365, help='Days of history (ignored if --since is used)')
    parser.add_argument('--since', type=str, default=None, help='Start date (YYYY-MM-DD)')
    args = parser.parse_args()

    fetcher = DataFetcher()
    symbols = fetcher.get_top_assets(limit=args.limit)
    
    print(f"Starting download for {len(symbols)} assets...")
    for symbol in tqdm(symbols):
        fetcher.fetch_ohlcv(symbol, timeframe=args.timeframe, days=args.days, since_date=args.since)
