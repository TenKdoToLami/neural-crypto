import os
import ccxt
import pandas as pd
from dotenv import load_dotenv

class BinanceTrader:
    def __init__(self, paper_trade=False):
        load_dotenv()
        
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret = os.getenv('BINANCE_SECRET')
        self.paper_trade = paper_trade
        
        # Target allocation per asset (4.8%)
        self.target_allocation = 0.048
        
        if not self.api_key or not self.secret:
            print("[Trader] WARNING: API keys not found in .env. Defaulting to paper trading.")
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
            print(f"[Trader] Could not load markets: {e}")

    def get_portfolio_value(self):
        """Returns total portfolio value in USDT and a dict of current holdings."""
        if self.paper_trade:
            # Mock portfolio for paper trading
            return 1000.0, 1000.0, {}
            
        try:
            balance = self.exchange.fetch_balance()
            free_usdt = balance.get('USDT', {}).get('free', 0.0)
            
            holdings = {}
            total_crypto_value = 0.0
            
            # Find all held assets with non-zero balance
            for currency, amt in balance['total'].items():
                if amt > 0 and currency != 'USDT':
                    symbol = f"{currency}/USDT"
                    # We only care about assets that actually trade against USDT
                    if symbol in self.exchange.markets:
                        # Estimate value using current ticker
                        ticker = self.exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                        value_usdt = amt * current_price
                        
                        # Only track it if value is > $2 (ignores tiny dust)
                        if value_usdt > 2.0:
                            holdings[symbol] = amt
                            total_crypto_value += value_usdt
                            
            total_portfolio = free_usdt + total_crypto_value
            return total_portfolio, free_usdt, holdings
            
        except Exception as e:
            print(f"[Trader] Error fetching portfolio: {e}")
            return 0.0, 0.0, {}

    def execute_trades(self, preds_df, approved_assets_ordered):
        """
        Executes buy/sell logic based on predictions.
        preds_df: DataFrame containing ['symbol', 'rally_prob', 'last_close']
        approved_assets_ordered: List of strings in exact order from the text file
        """
        if preds_df.empty:
            print("[Trader] No predictions to act on.")
            return
            
        print("\n" + "-"*40)
        print(" PORTFOLIO EXECUTION ".center(40, "-"))
        
        total_value, free_usdt, holdings = self.get_portfolio_value()
        print(f"[Trader] Total Portfolio: ${total_value:.2f} | Free USDT: ${free_usdt:.2f}")
        
        if total_value <= 0:
            print("[Trader] Portfolio value is zero or fetch failed. Aborting execution.")
            return

        target_trade_usdt = total_value * self.target_allocation
        print(f"[Trader] Target Allocation Size (4.8%): ${target_trade_usdt:.2f}")

        # Convert predictions to a dict for fast lookup
        # e.g. {'BTC/USDT': {'rally_prob': 0.8, 'last_close': 65000}}
        pred_map = preds_df.set_index('symbol').to_dict('index')

        # =====================================================================
        # STEP 1: LIQUIDATE (SELL)
        # =====================================================================
        for symbol, amount in holdings.items():
            if symbol in pred_map:
                prob = pred_map[symbol]['rally_prob']
                if prob < 0.3:
                    print(f"[-] LIQUIDATING {symbol} | Prob: {prob:.2f} < 0.3")
                    if not self.paper_trade:
                        try:
                            # Send market sell for the entire balance
                            order = self.exchange.create_market_sell_order(symbol, amount)
                            print(f"    -> Sold {amount} of {symbol}")
                        except Exception as e:
                            print(f"    -> [!] Sell Failed: {e}")
            else:
                # If an asset isn't in our active tracking, we might want to sell it,
                # but for safety, we just leave it alone here.
                pass

        # Refresh free USDT after sells
        if not self.paper_trade:
            # small delay to let Binance update balances
            import time
            time.sleep(1)
            _, free_usdt, holdings = self.get_portfolio_value()
            
        # =====================================================================
        # STEP 2: ALLOCATE (BUY)
        # =====================================================================
        for symbol in approved_assets_ordered:
            # We only buy if it scored > 0.95
            if symbol in pred_map:
                prob = pred_map[symbol]['rally_prob']
                
                if prob > 0.95:
                    # Check if we already hold it
                    if symbol in holdings:
                        print(f"[~] Skipping {symbol} | Prob {prob:.2f} | Already held.")
                        continue
                        
                    # Check if we have enough cash for a full allocation block
                    # (Binance typically has a $5-$10 min order size anyway)
                    if free_usdt >= target_trade_usdt and target_trade_usdt > 10.0:
                        price = pred_map[symbol]['last_close']
                        amount_to_buy = target_trade_usdt / price
                        
                        print(f"[+] BUYING {symbol} | Prob: {prob:.2f} > 0.95")
                        print(f"    -> Size: ${target_trade_usdt:.2f} ({amount_to_buy:.6f} units)")
                        
                        if not self.paper_trade:
                            try:
                                order = self.exchange.create_market_buy_order(symbol, amount_to_buy)
                                print("    -> Order Success")
                            except Exception as e:
                                print(f"    -> [!] Buy Failed: {e}")
                                
                        # Deduct from local free_usdt tracker so we don't overdraft (works for both live and paper)
                        free_usdt -= target_trade_usdt
                        
                    else:
                        print(f"[!] Insufficient USDT to buy {symbol} (Need ${target_trade_usdt:.2f}, Have ${free_usdt:.2f})")
                        
        print("-" * 40)

if __name__ == "__main__":
    print("Testing Binance API Connection...")
    trader = BinanceTrader(paper_trade=False)
    total, free, holdings = trader.get_portfolio_value()
    print(f"Total Portfolio: ${total:.2f}")
    print(f"Free USDT: ${free:.2f}")
    print("Current Holdings:", holdings)
