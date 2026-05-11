import os
import ccxt
import pandas as pd
import json
import time
from dotenv import load_dotenv
from datetime import datetime
from src.utils.logger import logger
from src.utils.db import db_manager

class BinanceTrader:
    def __init__(self, paper_trade=False):
        load_dotenv()
        
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret = os.getenv('BINANCE_SECRET')
        self.paper_trade = paper_trade
        
        # Configuration
        self.quote_currency = "USDC"
        self.target_allocation = 0.096
        self.entry_threshold = 0.95
        self.exit_threshold = 0.35
        
        if not self.api_key or not self.secret:
            logger.warning("[Trader] API keys not found in .env. Defaulting to paper trading.")
            self.paper_trade = True
            
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret,
            'enableRateLimit': True,
        })
        
        # Optionally load markets to get precision requirements
        try:
            if not self.paper_trade:
                self.exchange.load_markets()
        except Exception as e:
            logger.error(f"[Trader] Could not load markets: {e}")

    def get_portfolio_value(self):
        """Returns total portfolio value in USDC and a dict of current holdings."""
        if self.paper_trade:
            # Mock portfolio for paper trading
            return 1000.0, 1000.0, {}
            
        try:
            balance = self.exchange.fetch_balance()
            
            # Aggregate balances to handle Binance Earn (LD prefix for lending assets)
            total_balances = {}
            for raw_currency, amt in balance['total'].items():
                if amt <= 0:
                    continue
                
                currency = raw_currency
                # Strip LD prefix if it's a lending asset and we have a corresponding market or it's the quote
                if raw_currency.startswith('LD') and len(raw_currency) > 2:
                    potential = raw_currency[2:]
                    # Check if potential currency exists in markets or is the quote currency
                    if any(f"{potential}/" in m for m in self.exchange.markets) or potential == self.quote_currency:
                        currency = potential
                
                total_balances[currency] = total_balances.get(currency, 0.0) + amt
            
            free_quote = total_balances.get(self.quote_currency, 0.0)
            
            # Log if significant funds are in Earn
            ld_quote_sym = f"LD{self.quote_currency}"
            if ld_quote_sym in balance.get('total', {}):
                ld_amt = balance['total'][ld_quote_sym]
                if ld_amt > 1.0:
                    logger.info(f"[Trader] Note: Found ${ld_amt:.2f} in {ld_quote_sym} (Earn). Aggregated into total {self.quote_currency}.")

            holdings = {}
            total_crypto_value = 0.0
            
            # Find all held assets with non-zero balance
            for currency, amt in total_balances.items():
                if currency == self.quote_currency:
                    continue
                    
                # Map back to the symbol used by the inference engine
                symbol_quote = f"{currency}/{self.quote_currency}"
                
                # Estimate value using current ticker
                if symbol_quote in self.exchange.markets:
                    ticker = self.exchange.fetch_ticker(symbol_quote)
                    current_price = ticker['last']
                    value_quote = amt * current_price
                    
                    # Only track it if value is > $2 (ignores tiny dust)
                    if value_quote > 2.0:
                        holdings[symbol_quote] = {
                            'amount': amt,
                            'value': value_quote,
                            'price': current_price
                        }
                        total_crypto_value += value_quote
                            
            total_portfolio = free_quote + total_crypto_value
            
            # Add percentages
            for sym in holdings:
                holdings[sym]['pct'] = (holdings[sym]['value'] / total_portfolio) * 100
                
            return total_portfolio, free_quote, holdings, balance
            
        except Exception as e:
            logger.error(f"[Trader] Error fetching portfolio: {e}")
            return 0.0, 0.0, {}, {}

    def redeem_from_earn(self, asset, amount):
        """Redeems asset from Binance Flexible Earn to Spot wallet."""
        if self.paper_trade:
            return True
            
        try:
            # First find the productId for this flexible earn product
            products = self.exchange.sapiGetSimpleEarnFlexibleList({'asset': asset})
            product_id = None
            for row in products.get('rows', []):
                if row['asset'] == asset:
                    product_id = row['productId']
                    break
            
            if not product_id:
                logger.error(f"[Trader] Could not find Product ID for {asset} in Simple Earn.")
                return False

            logger.info(f"[Trader] Attempting to redeem {amount:.8f} {asset} from Earn (ID: {product_id})...")
            self.exchange.sapiPostSimpleEarnFlexibleRedeem({
                'productId': product_id,
                'amount': amount,
                'type': 'FAST'
            })
            # Wait a bit for balance to update
            time.sleep(2)
            return True
        except Exception as e:
            logger.error(f"[Trader] [!] Redemption Failed: {e}")
            return False

    def execute_trades(self, preds_df, approved_assets_ordered):
        """
        Executes buy/sell logic based on predictions.
        preds_df: DataFrame containing ['symbol', 'rally_prob', 'last_close']
        approved_assets_ordered: List of strings in exact order from the text file
        """
        if preds_df.empty:
            logger.info("[Trader] No predictions to act on.")
            return
            
        logger.info("-" * 40)
        logger.info(" PORTFOLIO EXECUTION ".center(40, "-"))
        
        total_value, free_quote, holdings, raw_balance = self.get_portfolio_value()
        logger.info(f"[Trader] Total Portfolio: ${total_value:.2f} | Free {self.quote_currency}: ${free_quote:.2f}")
        
        if holdings:
            logger.info("[Trader] Current Holdings (> $2):")
            for sym, info in holdings.items():
                # Try to get PnL from DB
                entry_price = db_manager.get_last_buy_price(sym)
                pnl_str = ""
                if entry_price:
                    pnl_pct = ((info['price'] - entry_price) / entry_price) * 100
                    pnl_raw = (info['price'] - entry_price) * info['amount']
                    pnl_str = f" | {pnl_pct:+6.2f}% (${pnl_raw:+7.2f})"
                
                logger.info(f"    - {sym:10} | {info['pct']:5.1f}% | ${info['value']:7.2f} | {info['amount']:12.6f} units{pnl_str}")
        else:
            logger.info("[Trader] Current Holdings: None")
        
        if total_value <= 0:
            logger.error("[Trader] Portfolio value is zero or fetch failed. Aborting execution.")
            return

        target_trade_quote = total_value * self.target_allocation
        logger.info(f"[Trader] Target Allocation Size ({self.target_allocation*100:.1f}%): ${target_trade_quote:.2f} {self.quote_currency}")

        # --- SQL Logging ---
        now_utc = datetime.utcnow()
        if now_utc.hour == 0 and now_utc.minute < 15:
            # Save portfolio snapshot to SQL (Once Daily at Midnight)
            db_manager.save_portfolio_snapshot(total_value, free_quote)
            logger.info(f"[Daily Report] Midnight SQL snapshot recorded: ${total_value:.2f} USDC")
        # ------------------------------

        # Convert predictions to a dict for fast lookup
        pred_map = preds_df.set_index('symbol').to_dict('index')

        # =====================================================================
        # STEP 1: LIQUIDATE (SELL)
        # =====================================================================
        for symbol, info in holdings.items():
            amount = info['amount']
            current_price = info['price']
            if symbol in pred_map:
                prob = pred_map[symbol]['rally_prob']
                if prob < self.exit_threshold:
                    logger.info(f"[-] LIQUIDATING {symbol} | Prob: {prob:.2f} < {self.exit_threshold}")
                    
                    # Calculate PnL if entry price exists in DB
                    entry_price = db_manager.get_last_buy_price(symbol)
                    pnl_pct = 0.0
                    pnl_raw = 0.0
                    if entry_price:
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        pnl_raw = (current_price - entry_price) * amount
                        logger.info(f"    -> Profit/Loss: {pnl_pct:+.2f}% | ${pnl_raw:+.2f}")
                    
                    status = "SUCCESS"
                    if not self.paper_trade:
                        try:
                            # Ensure funds are in Spot wallet
                            base_asset = symbol.split('/')[0]
                            spot_amt = raw_balance.get(base_asset, {}).get('free', 0.0)
                            if spot_amt < amount:
                                needed = amount - spot_amt
                                ld_asset = f"LD{base_asset}"
                                earn_amt = raw_balance.get(ld_asset, {}).get('free', 0.0)
                                if earn_amt >= needed:
                                    self.redeem_from_earn(base_asset, needed)
                                elif earn_amt > 0:
                                    # Redeem whatever is available
                                    self.redeem_from_earn(base_asset, earn_amt)
                            
                            order = self.exchange.create_market_sell_order(symbol, amount)
                            logger.info(f"    -> Sold {amount} of {symbol}")
                        except Exception as e:
                            status = "FAILED"
                            logger.error(f"    -> [!] Sell Failed: {e}")
                    
                    if status == "SUCCESS":
                        db_manager.record_trade(symbol, "SELL", amount, current_price, prob, pnl_pct, pnl_raw)
            else:
                # Not in active tracking
                pass

        # Refresh free quote after sells
        if not self.paper_trade:
            time.sleep(1) # Delay for balance update
            _, free_quote, holdings, raw_balance = self.get_portfolio_value()
            
        # =====================================================================
        # STEP 2: ALLOCATE (BUY)
        # =====================================================================
        for symbol in approved_assets_ordered:
            if symbol in pred_map:
                prob = pred_map[symbol]['rally_prob']
                price = pred_map[symbol]['last_close']
                
                if prob > self.entry_threshold:
                    # Check if already held
                    if symbol in holdings:
                        logger.info(f"[~] Skipping {symbol} | Prob {prob:.2f} | Already held.")
                        continue
                        
                    # Check cash
                    if free_quote >= target_trade_quote and target_trade_quote > 10.0:
                        amount_to_buy = target_trade_quote / price
                        
                        logger.info(f"[+] BUYING {symbol} | Prob: {prob:.2f} > {self.entry_threshold}")
                        logger.info(f"    -> Size: ${target_trade_quote:.2f} ({amount_to_buy:.6f} units)")
                        
                        status = "SUCCESS"
                        if not self.paper_trade:
                            try:
                                # Ensure USDC is in Spot wallet
                                spot_quote = raw_balance.get(self.quote_currency, {}).get('free', 0.0)
                                if spot_quote < target_trade_quote:
                                    needed = target_trade_quote - spot_quote
                                    ld_quote = f"LD{self.quote_currency}"
                                    earn_quote = raw_balance.get(ld_quote, {}).get('free', 0.0)
                                    if earn_quote >= needed:
                                        self.redeem_from_earn(self.quote_currency, needed)
                                    elif earn_quote > 0:
                                        self.redeem_from_earn(self.quote_currency, earn_quote)
                                
                                order = self.exchange.create_market_buy_order(symbol, amount_to_buy)
                                logger.info("    -> Order Success")
                            except Exception as e:
                                status = "FAILED"
                                logger.error(f"    -> [!] Buy Failed: {e}")
                                
                        if status == "SUCCESS":
                            db_manager.record_trade(symbol, "BUY", amount_to_buy, price, prob)
                            free_quote -= target_trade_quote
                            holdings[symbol] = amount_to_buy
                    else:
                        logger.warning(f"[!] Insufficient {self.quote_currency} to buy {symbol} (Need ${target_trade_quote:.2f}, Have ${free_quote:.2f})")
                        
        # =====================================================================
        # STEP 3: LOG RUN HISTORY (Sliding 7-day Window)
        # =====================================================================
        try:
            # 1. Format Predictions
            clean_preds = []
            for _, row in preds_df.iterrows():
                clean_preds.append({
                    "symbol": row['symbol'],
                    "prob": round(float(row['rally_prob']), 4),
                    "price": round(float(row['last_close']), 4)
                })
            
            # 2. Format Portfolio (Percentage based)
            portfolio_snapshot = {
                "total_value": round(total_value, 2),
                "cash_pct": round((free_quote / total_value) * 100, 2),
                "holdings": []
            }
            for sym, info in holdings.items():
                portfolio_snapshot["holdings"].append({
                    "symbol": sym,
                    "pct": round(info['pct'], 2),
                    "pnl_pct": round(((info['price'] - db_manager.get_last_buy_price(sym)) / db_manager.get_last_buy_price(sym)) * 100, 2) if db_manager.get_last_buy_price(sym) else 0.0
                })
            
            db_manager.save_run_history(clean_preds, portfolio_snapshot)
            logger.info("[Trader] Run history saved to SQL (7-day sliding window).")
        except Exception as e:
            logger.error(f"[Trader] Failed to save run history: {e}")

        logger.info("-" * 40)

if __name__ == "__main__":
    logger.info("Testing Binance API Connection...")
    trader = BinanceTrader(paper_trade=False)
    total, free, holdings, _ = trader.get_portfolio_value()
    logger.info(f"Total Portfolio: ${total:.2f}")
    logger.info(f"Free USDC: ${free:.2f}")
    logger.info(f"Current Holdings: {holdings}")
