import os
import time
import pandas as pd
import pandas_ta as ta
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
        self.total_trades = 0
        self.equity = self.get_equity()
        self.initial_equity = self.equity
        self.trade_log = []
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.setup_strategies()
        self.start_time = datetime.now()
        self.wins = 0
        self.losses = 0

    def get_equity(self):
        """Get current USDT balance"""
        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            return float(response['result']['list'][0]['coin'][0]['equity'])
        except Exception as e:
            print(f"Balance error: {e}")
            return 0

    def setup_strategies(self):
        """Initialize adaptive strategy parameters"""
        self.indicators = {
            'rsi_buy': 35,
            'rsi_sell': 65,
            'ema_short': 50,
            'ema_long': 200,
            'atr_threshold': 0.025,
            'min_position_size': 0.01
        }
        print("Adaptive strategy parameters initialized")

    def fetch_market_data(self):
        """Fetch optimized market data"""
        try:
            data = self.session.get_kline(
                category="linear",
                symbol=self.symbol,
                interval="60",
                limit=100
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
        """Calculate enhanced indicators"""
        try:
            # Momentum analysis
            df['RSI'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            
            # Trend analysis
            df['EMA50'] = ta.ema(df['close'], length=50)
            df['EMA200'] = ta.ema(df['close'], length=200)
            
            # Volatility analysis
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            return pd.concat([df, macd], axis=1).dropna()
        except Exception as e:
            print(f"Indicator error: {e}")
            return df

    def generate_signal(self, df):
        """Generate dynamic trading signals"""
        try:
            current = df.iloc[-1]
            
            print(f"\nMarket Analysis:")
            print(f"RSI: {current['RSI']:.2f}")
            print(f"MACD Histogram: {current['MACDh_12_26_9']:.4f}")
            print(f"EMA50/200: {current['EMA50']:.2f}/{current['EMA200']:.2f}")
            print(f"ATR/Price: {(current['ATR']/current['close']*100):.2f}%")

            # Trend and momentum conditions
            bullish_trend = current['EMA50'] > current['EMA200']
            bearish_trend = current['EMA50'] < current['EMA200']
            momentum_buy = current['RSI'] < self.indicators['rsi_buy']
            momentum_sell = current['RSI'] > self.indicators['rsi_sell']
            volatility_ok = current['ATR'] < current['close'] * self.indicators['atr_threshold']

            if bullish_trend and momentum_buy and volatility_ok:
                print("Bullish signal detected")
                return 'buy'
            elif bearish_trend and momentum_sell and volatility_ok:
                print("Bearish signal detected")
                return 'sell'
            print("No clear market direction")
            return None
            
        except Exception as e:
            print(f"Signal error: {e}")
            return None

    def execute_trade(self, signal):
        """Execute trade with dynamic sizing"""
        try:
            ticker = self.session.get_tickers(category="linear", symbol=self.symbol)
            price = float(ticker['result']['list'][0]['lastPrice'])
            
            position_size = max(
                (self.equity * self.risk_per_trade) / price,
                self.indicators['min_position_size']
            )
            
            order = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal == 'buy' else "Sell",
                orderType="Market",
                qty=f"{position_size:.4f}",
                stopLoss=f"{price * 0.995:.4f}",
                takeProfit=f"{price * 1.005:.4f}",
                timeInForce="GTC"
            )
            
            self.total_trades += 1
            self.equity = self.get_equity()
            
            print(f"\nExecuted {signal.upper()} trade:")
            print(f"Size: {position_size:.4f} ETH")
            print(f"Entry: ${price:.2f}")
            print(f"Current Equity: ${self.equity:.2f}")

        except Exception as e:
            print(f"Trade failed: {e}")

    def run_continuous(self):
        """Continuous trading loop"""
        print(f"Starting trading bot with ${self.equity:.2f}")
        while True:
            try:
                df = self.fetch_market_data()
                if not df.empty:
                    df = self.calculate_indicators(df)
                    signal = self.generate_signal(df)
                    if signal:
                        self.execute_trade(signal)
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                print("\nStopping bot...")
                print(f"Final equity: ${self.equity:.2f}")
                print(f"Total trades: {self.total_trades}")
                break

if __name__ == "__main__":
    bot = AdvancedTradingBot()
    bot.run_continuous()