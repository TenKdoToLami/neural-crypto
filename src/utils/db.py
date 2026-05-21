import sqlite3
import os
import json
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="data/trading.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the database and creates tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table for portfolio snapshots (Daily Growth)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    free_cash REAL NOT NULL
                )
            ''')
            
            # Table for trade history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    amount REAL NOT NULL,
                    price REAL NOT NULL,
                    prob REAL NOT NULL,
                    pnl_pct REAL,
                    pnl_raw REAL
                )
            ''')
            
            # Table for high-detail run history (Sliding 7-day window)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS run_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    predictions_json TEXT NOT NULL,
                    portfolio_json TEXT NOT NULL
                )
            ''')
            conn.commit()

    def save_portfolio_snapshot(self, total_value, free_cash):
        """Saves a snapshot of the current portfolio total value."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO portfolio_history (timestamp, total_value, free_cash)
                VALUES (?, ?, ?)
            ''', (
                datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                total_value,
                free_cash
            ))
            conn.commit()

    def get_last_buy_price(self, symbol):
        """Retrieves the price of the most recent BUY action for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT price FROM trades 
                WHERE symbol = ? AND action = 'BUY' 
                ORDER BY timestamp DESC LIMIT 1
            ''', (symbol,))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_last_buy_timestamp(self, symbol):
        """Retrieves the timestamp of the most recent BUY action for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp FROM trades 
                WHERE symbol = ? AND action = 'BUY' 
                ORDER BY timestamp DESC LIMIT 1
            ''', (symbol,))
            row = cursor.fetchone()
            return row[0] if row else None

    def record_trade(self, symbol, action, amount, price, prob, pnl_pct=None, pnl_raw=None):
        """Records a trade action in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (timestamp, symbol, action, amount, price, prob, pnl_pct, pnl_raw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                symbol,
                action,
                amount,
                price,
                prob,
                pnl_pct,
                pnl_raw
            ))
            conn.commit()

    def save_run_history(self, predictions_list, portfolio_data):
        """Saves detailed run data and prunes history older than 7 days."""
        now = datetime.utcnow()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Insert new run data
            cursor.execute('''
                INSERT INTO run_history (timestamp, predictions_json, portfolio_json)
                VALUES (?, ?, ?)
            ''', (
                now.strftime('%Y-%m-%d %H:%M:%S'),
                json.dumps(predictions_list),
                json.dumps(portfolio_data)
            ))
            
            # 2. Prune data older than 7 days
            cursor.execute('''
                DELETE FROM run_history 
                WHERE timestamp < datetime('now', '-7 days')
            ''')
            conn.commit()

# Singleton instance
db_manager = DatabaseManager()
