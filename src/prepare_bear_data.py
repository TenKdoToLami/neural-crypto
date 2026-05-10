import os
import pandas as pd
import ccxt
import time
from datetime import datetime
from tqdm import tqdm
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BearDataFetcher:
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        self.output_dir = 'data/bear_market'
        self.approved_file = 'data/approved_assets.txt'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.start_ts = self.exchange.parse8601("2022-01-01T00:00:00Z")
        self.end_ts = self.exchange.parse8601("2023-12-31T23:59:59Z")

    def get_symbols(self):
        assets = []
        if not os.path.exists(self.approved_file):
            return ["BTC/USDT", "ETH/USDT"]
        with open(self.approved_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '/' not in line: line = f"{line}/USDT"
                    assets.append(line)
        return assets

    def fetch_period(self, symbol):
        filename = os.path.join(self.output_dir, f"{symbol.replace('/', '_')}_15m.csv")
        
        # Check if already complete
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            if not df.empty:
                first = pd.to_datetime(df['timestamp'].iloc[0])
                last = pd.to_datetime(df['timestamp'].iloc[-1])
                # If it covers Jan 2023 and Dec 2024, we are good
                if first <= datetime(2023, 1, 2) and last >= datetime(2024, 12, 30):
                    print(f"  [=] {symbol}: Bear Market data already exists and is complete. Skipping.")
                    return

        print(f"  [+] {symbol}: Downloading 2023-2024 history...")
        all_ohlcv = []
        since = self.start_ts
        
        try:
            while since < self.end_ts:
                ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', since, limit=1000)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                if len(ohlcv) < 1000: break # End of available data
                time.sleep(self.exchange.rateLimit / 1000)

            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                # Filter strictly to the period
                df = df[df['timestamp'] <= "2024-12-31 23:59:59"]
                df.to_csv(filename, index=False)
                print(f"  [✓] {symbol}: Saved {len(df)} candles.")
        except Exception as e:
            print(f"  [!] Error fetching {symbol}: {e}")

    def run(self):
        symbols = self.get_symbols()
        print(f"🚀 Preparing BEAR MARKET data for {len(symbols)} assets...")
        for s in tqdm(symbols):
            self.fetch_period(s)

if __name__ == "__main__":
    BearDataFetcher().run()
