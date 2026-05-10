import os
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

class LiveDataFetcher:
    def __init__(self, exchange_id='binance', limit=20, timeframe='15m'):
        self.exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        self.data_dir = 'data/raw_live'
        self.limit = limit
        self.timeframe = timeframe
        self.approved_assets_file = 'data/approved_assets.txt'
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_approved_assets(self):
        """Read the manually approved asset list from a text file."""
        assets = []
        if not os.path.exists(self.approved_assets_file):
            # Create a default list if it doesn't exist
            default_assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
            with open(self.approved_assets_file, 'w') as f:
                f.write("\n".join(default_assets))
            return default_assets
            
        with open(self.approved_assets_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Ensure it has the /USDT suffix for ccxt if needed
                    if '/' not in line:
                        line = f"{line}/USDT"
                    assets.append(line)
                    
        # Remove duplicates while preserving original file order
        seen = set()
        ordered_assets = []
        for a in assets:
            if a not in seen:
                ordered_assets.append(a)
                seen.add(a)
                
        return ordered_assets

    def fetch_latest(self):
        """Fetch the latest candles for tracked assets, minimizing bandwidth."""
        assets = self.get_approved_assets()
        print(f"[Live Data] Tracking {len(assets)} manually approved assets.")
        updated_data = {}
        
        for symbol in assets:
            filename = os.path.join(self.data_dir, f"{symbol.replace('/', '_')}_{self.timeframe}.csv")
            
            # Check if file exists to calculate missing intervals
            fetch_limit = 250 # Default bootstrap
            last_dt = None
            if os.path.exists(filename):
                existing_data = pd.read_csv(filename)
                if not existing_data.empty:
                    last_dt = pd.to_datetime(existing_data['timestamp'].iloc[-1])
                    # Calculate missing intervals
                    time_diff = datetime.utcnow() - last_dt
                    missing_intervals = int(time_diff.total_seconds() / (15 * 60))
                    # Safely cap at 1000 (Binance single-request max)
                    fetch_limit = min(1000, max(5, missing_intervals + 5))

            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=fetch_limit)
                if not ohlcv:
                    continue
                
                df_new = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
                
                if last_dt is not None:
                    # Filter only strictly newer rows
                    df_new = df_new[df_new['timestamp'] > last_dt]
                    
                    if not df_new.empty:
                        df_new.to_csv(filename, mode='a', header=False, index=False)
                        # Reload the end to keep the cache size manageable (e.g. 500 max)
                        final_df = pd.read_csv(filename).tail(500)
                        final_df.to_csv(filename, index=False) 
                else:
                    # Initial save
                    df_new.to_csv(filename, index=False)
                    
                # Store in memory cache for immediate access by inference
                updated_data[symbol] = pd.read_csv(filename)
                
            except Exception as e:
                print(f"[!] Error fetching live data for {symbol}: {e}")
                
            time.sleep(self.exchange.rateLimit / 1000)
            
        return updated_data

if __name__ == "__main__":
    fetcher = LiveDataFetcher()
    data = fetcher.fetch_latest()
    print(f"Fetched live data for {len(data)} assets.")
