import os
import time
import schedule
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class AdvancedTradingBot:
    def __init__(self):
        self.session = HTTP(
            api_key=os.getenv("BYBIT_API_KEY"),
            api_secret=os.getenv("BYBIT_API_SECRET"),
            testnet=True
        )
        self.symbol = "ETHUSDT"
        self.daily_trades = 0
        self.equity = self.get_equity()
        self.trade_log = []
        self.risk_per_trade = 0.0001  # 0.01% risk per trade
        self.setup_strategies()

    def get_equity(self):
        """Get current USDT balance with error handling"""
        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            print(response)
            return float(response['result']['list'][0]['coin'][0]['equity'])
        except Exception as e:
            print(f"Balance error: {e}")
            return 0

    def setup_strategies(self):
        """Initialize strategy parameters"""
        self.indicators = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'ema_short': 20,
            'ema_long': 50,
            'atr_period': 14
        }

    def fetch_market_data(self):
        """Get historical price data with multiple timeframes"""
        try:
            data = self.session.get_kline(
                category="linear",
                symbol=self.symbol,
                interval="15",
                limit=200
            )
            df = pd.DataFrame(data['result']['list'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df = df.astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.iloc[::-1].reset_index(drop=True)
        except Exception as e:
            print(f"Data error: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df):
        """Calculate technical indicators with triple confirmation"""
        # Momentum indicators
        df['RSI'] = ta.rsi(df['close'], length=self.indicators['rsi_period'])
        macd = ta.macd(df['close'], 
                      fast=self.indicators['macd_fast'],
                      slow=self.indicators['macd_slow'],
                      signal=self.indicators['macd_signal'])
        df = pd.concat([df, macd], axis=1)
        
        # Trend indicators
        df['EMA20'] = ta.ema(df['close'], length=self.indicators['ema_short'])
        df['EMA50'] = ta.ema(df['close'], length=self.indicators['ema_long'])
        
        # Volatility indicator
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], 
                          length=self.indicators['atr_period'])
        
        return df.dropna()

    def generate_signal(self, df):
        """Triple-confirmation trading signal with risk filter"""
        if len(df) < 50:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Momentum confirmation
        momentum_buy = (last['RSI'] < 30) and (last['MACD_12_26_9'] > last['MACDs_12_26_9'])
        momentum_sell = (last['RSI'] > 70) and (last['MACD_12_26_9'] < last['MACDs_12_26_9'])

        # Trend confirmation
        trend_buy = (last['EMA20'] > last['EMA50']) and (last['close'] > last['EMA20'])
        trend_sell = (last['EMA20'] < last['EMA50']) and (last['close'] < last['EMA20'])

        # Volatility filter
        volatility_ok = last['ATR'] < (last['close'] * 0.015)

        if momentum_buy and trend_buy and volatility_ok:
            return 'buy'
        elif momentum_sell and trend_sell and volatility_ok:
            return 'sell'
        return None

    def calculate_position_size(self):
        """Dynamic position sizing with compound growth"""
        return self.equity * self.risk_per_trade

    def execute_trade(self, signal):
        """Execute trade with tight risk controls"""
        if self.daily_trades >= 5 or self.equity < 10:
            return

        try:
            # Get precise pricing
            ticker = self.session.get_tickers(category="linear", symbol=self.symbol)
            price = float(ticker['result']['list'][0]['lastPrice'])
            
            # Calculate order parameters
            qty = self.calculate_position_size() / price
            stop_loss = price * 0.9999  # 0.01% loss
            take_profit = price * 1.0002  # 0.02% gain

            # Place market order
            order = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal == 'buy' else "Sell",
                orderType="Market",
                qty=f"{qty:.4f}",
                stopLoss=f"{stop_loss:.4f}",
                takeProfit=f"{take_profit:.4f}",
                timeInForce="GTC"
            )

            # Update tracking
            self.daily_trades += 1
            self.equity = self.get_equity()
            self.log_trade(order, signal, qty, price)

        except Exception as e:
            print(f"Trade error: {e}")

    def log_trade(self, order, signal, qty, price):
        """Maintain detailed trade records"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'qty': qty,
            'price': price,
            'equity': self.equity,
            'order_id': order['result']['orderId']
        }
        self.trade_log.append(log_entry)
        pd.DataFrame(self.trade_log).to_csv('trade_history.csv', index=False)

    def reset_daily_count(self):
        """Reset daily trade counter"""
        self.daily_trades = 0
        print(f"{datetime.now()} - Daily trades reset")

    def run_strategy(self):
        """Main trading cycle"""
        if self.daily_trades >= 5:
            return

        df = self.fetch_market_data()
        if df.empty:
            return

        df = self.calculate_indicators(df)
        signal = self.generate_signal(df)
        
        if signal:
            self.execute_trade(signal)

    def start(self):
        """Start automated trading"""
        schedule.every().day.at("00:00").do(self.reset_daily_count)
        schedule.every(1).hours.do(self.run_strategy)

        print(f"{datetime.now()} - Bot started with equity: ${self.equity:.2f}")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nBot shutdown successfully")

if __name__ == "__main__":
    bot = AdvancedTradingBot()
    bot.start()