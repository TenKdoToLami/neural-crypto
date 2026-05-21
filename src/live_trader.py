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
        # Initialize trader early to get holdings & thresholds for display
        trader = BinanceTrader(paper_trade=False)
        total_value, free_quote, holdings, raw_balance = trader.get_portfolio_value()
        
        data = fetcher.fetch_latest()
        preds = engine.process_and_predict(data)
        
        if not preds.empty:
            # Add holding column for visualization
            preds_display = preds.copy()
            preds_display['holding'] = preds_display['symbol'].apply(
                lambda x: '[HOLDING]' if x in holdings else ''
            )
            
            # Format dataframe as strings to control output line-by-line
            lines = preds_display.to_string(index=True).split('\n')
            header = lines[0]
            row_lines = [line for line in lines[1:] if line.strip()]
            
            entry_threshold = trader.entry_threshold
            exit_threshold = trader.exit_threshold
            
            formatted_output = []
            formatted_output.append(header)
            
            buy_threshold_printed = False
            exit_threshold_printed = False
            
            # Line width for dividers based on the header length
            width = len(header) + 4
            
            for idx, line in enumerate(row_lines):
                if idx >= len(preds):
                    break
                prob = preds.iloc[idx]['rally_prob']
                
                # Check for buy threshold transition
                if not buy_threshold_printed and prob <= entry_threshold:
                    divider = f"--- BUY ZONE THRESHOLD ({entry_threshold}) ".ljust(width, '-')
                    formatted_output.append(divider)
                    buy_threshold_printed = True
                    
                # Check for liquidation threshold transition
                if not exit_threshold_printed and prob <= exit_threshold:
                    divider = f"--- LIQUIDATION ZONE THRESHOLD ({exit_threshold}) ".ljust(width, '-')
                    formatted_output.append(divider)
                    exit_threshold_printed = True
                    
                formatted_output.append(line)
                
            # If thresholds were not crossed during predictions list (e.g. all higher or lower)
            if not buy_threshold_printed:
                divider = f"--- BUY ZONE THRESHOLD ({entry_threshold}) ".ljust(width, '-')
                formatted_output.append(divider)
            if not exit_threshold_printed:
                divider = f"--- LIQUIDATION ZONE THRESHOLD ({exit_threshold}) ".ljust(width, '-')
                formatted_output.append(divider)
                
            logger.info("\n[Latest Predictions]\n" + "\n".join(formatted_output))
            
            # Execute trades using pre-initialized trader
            trader.execute_trades(preds, fetcher.get_approved_assets())
        else:
            logger.info("No predictions generated.")
            
    except Exception as e:
        logger.error(f"[!] Pipeline Error: {e}", exc_info=True)
        
    logger.info("=" * 40)

if __name__ == "__main__":
    run_loop()
