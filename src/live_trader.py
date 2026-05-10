import time
from datetime import datetime, timedelta
import os
import argparse

from src.live_data import LiveDataFetcher
from src.live_inference import LiveInferenceEngine
from src.live_execution import BinanceTrader
import argparse

def run_loop():
    print("=" * 60)
    print(" Neural Sentinel V1 - Live Execution Orchestrator (Cron Mode) ")
    print("=" * 60)
    
    # Sleep briefly to ensure Binance has closed the previous 15m candle
    # since cron runs exactly on the 00/15/30/45 second mark.
    print("[*] Waiting 5 seconds for exchange data to settle...")
    time.sleep(5)
    
    # Initialize engines
    fetcher = LiveDataFetcher()
    engine = LiveInferenceEngine()
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing run...")
    
    try:
        data = fetcher.fetch_latest()
        preds = engine.process_and_predict(data)
        
        print("\n[Latest Predictions]")
        if not preds.empty:
            print(preds.head(10).to_string())
            
            # Execute trades
            # Pass paper_trade=False when you are ready to use real money
            trader = BinanceTrader(paper_trade=True) 
            trader.execute_trades(preds, fetcher.get_approved_assets())
        else:
            print("No predictions generated.")
            
    except Exception as e:
        print(f"[!] Pipeline Error: {e}")
        
    print(f"{'=' * 40}")

if __name__ == "__main__":
    run_loop()
