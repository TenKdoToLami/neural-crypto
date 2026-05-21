import time
from datetime import datetime
import os
import argparse
import logging

from src.live_data import LiveDataFetcher
from src.live_inference_gbdt import LiveInferenceEngineGBDT
from src.live_execution import BinanceTrader
from src.utils.logger import logger

def run_loop():
    logger.info("=" * 60)
    logger.info(" Neural Sentinel V2 - Live Execution Orchestrator (Cron Mode) ")
    logger.info("=" * 60)
    
    # Sleep briefly to ensure Binance has closed the previous 15m candle
    # since cron runs exactly on the 00/15/30/45 second mark.
    logger.info("[*] Waiting 5 seconds for exchange data to settle...")
    time.sleep(5)
    
    # Initialize engines
    fetcher = LiveDataFetcher()
    engine = LiveInferenceEngineGBDT()
    
    logger.info(f"Executing run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    
    try:
        data = fetcher.fetch_latest()
        preds = engine.process_and_predict(data)
        
        if not preds.empty:
            logger.info("\n[Latest Predictions]\n" + preds.to_string())
            
            # Execute trades
            # Pass paper_trade=False when you are ready to use real money
            trader = BinanceTrader(paper_trade=False) 
            trader.execute_trades(preds, fetcher.get_approved_assets())
        else:
            logger.info("No predictions generated.")
            
    except Exception as e:
        logger.error(f"[!] Pipeline Error: {e}", exc_info=True)
        
    logger.info("=" * 40)

if __name__ == "__main__":
    run_loop()
