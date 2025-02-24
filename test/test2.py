import os
import time
import schedule
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
        self.daily_trades = 0
        self.total_trades = 0
        self.equity = self.get_equity()
        self.initial_equity = self.equity
        self.trade_log = []
        self.risk_per_trade = 0.0001  # 0.01% risk per trade
        self.wins = 0
        self.losses = 0
        self.setup_strategies()
        self.start_time = datetime.now()
        
        # Verify initial setup
        if self.equity < 10:
            print("\n⚠️ WARNING: Low initial equity! Please fund your testnet account.")
            print("Visit https://testnet.bybit.com -> Assets -> Deposit\n")

    def get_equity(self):
        """Get current USDT balance with enhanced error handling"""
        try:
            response = self.session.get_wallet_balance(
                accountType="UNIFIED", 
                coin="USDT"
            )
            if response['retCode'] == 0:
                coins = response['result']['list'][0]['coin']
                usdt_info = next((c for c in coins if c['coin'] == 'USDT'), None)
                return float(usdt_info['equity']) if usdt_info else 0.0
            print(f"API Error: {response['retMsg']}")
            return 0.0
        except Exception as e:
            print(f"Critical balance error: {str(e)}")
            return 0.0

    def setup_strategies(self):
        """Initialize strategy parameters with validation"""
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
        """Fetch and format historical price data"""
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
            print(f"Data fetch error: {str(e)}")
            return pd.DataFrame()

    def calculate_indicators(self, df):
        """Calculate technical indicators with error handling"""
        try:
            df['RSI'] = ta.rsi(df['close'], length=self.indicators['rsi_period'])
            macd = ta.macd(df['close'],
                          fast=self.indicators['macd_fast'],
                          slow=self.indicators['macd_slow'],
                          signal=self.indicators['macd_signal'])
            df = pd.concat([df, macd], axis=1)
            df['EMA20'] = ta.ema(df['close'], length=self.indicators['ema_short'])
            df['EMA50'] = ta.ema(df['close'], length=self.indicators['ema_long'])
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], 
                             length=self.indicators['atr_period'])
            return df.dropna()
        except Exception as e:
            print(f"Indicator calculation error: {str(e)}")
            return df

    def generate_signal(self, df):
        """Generate trading signals with validation"""
        if len(df) < 50:
            return None

        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]

            momentum_buy = (last['RSI'] < 30) and (last['MACD_12_26_9'] > last['MACDs_12_26_9'])
            momentum_sell = (last['RSI'] > 70) and (last['MACD_12_26_9'] < last['MACDs_12_26_9'])
            trend_buy = (last['EMA20'] > last['EMA50']) and (last['close'] > last['EMA20'])
            trend_sell = (last['EMA20'] < last['EMA50']) and (last['close'] < last['EMA20'])
            volatility_ok = last['ATR'] < (last['close'] * 0.015)

            if momentum_buy and trend_buy and volatility_ok:
                return 'buy'
            elif momentum_sell and trend_sell and volatility_ok:
                return 'sell'
            return None
        except KeyError as e:
            print(f"Missing indicator in DataFrame: {str(e)}")
            return None

    def calculate_position_size(self):
        """Calculate position size with minimum order check"""
        size = self.equity * self.risk_per_trade
        return max(size, 10)  # Minimum $10 position

    def execute_trade(self, signal):
        """Execute trade with enhanced error handling"""
        if self.daily_trades >= 5 or self.equity < 10:
            return

        try:
            # Get price data
            ticker = self.session.get_tickers(category="linear", symbol=self.symbol)
            price = float(ticker['result']['list'][0]['lastPrice'])
            
            # Calculate order parameters
            position_size = self.calculate_position_size() / price
            if position_size < 0.001:
                print("Position size too small, skipping trade")
                return

            stop_loss = price * 0.9999
            take_profit = price * 1.0002

            # Execute order
            order = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal == 'buy' else "Sell",
                orderType="Market",
                qty=f"{position_size:.4f}",
                stopLoss=f"{stop_loss:.4f}",
                takeProfit=f"{take_profit:.4f}",
                timeInForce="GTC"
            )

            # Update tracking
            self.total_trades += 1
            self.daily_trades += 1
            new_equity = self.get_equity()
            profit = new_equity - self.equity
            self.equity = new_equity

            if profit > 0:
                self.wins += 1
            else:
                self.losses += 1

            self.log_trade(order, signal, position_size, price, profit)
            print(f"Trade executed: {signal.upper()} {position_size:.4f} ETH @ ${price:.2f}")

        except Exception as e:
            print(f"Trade execution failed: {str(e)}")

    def log_trade(self, order, signal, qty, price, profit):
        """Enhanced trade logging"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'qty': qty,
            'price': price,
            'profit': profit,
            'equity': self.equity,
            'order_id': order['result']['orderId'],
            'status': order['result']['orderStatus']
        }
        self.trade_log.append(log_entry)
        try:
            pd.DataFrame(self.trade_log).to_csv('trade_history.csv', index=False)
        except Exception as e:
            print(f"Logging error: {str(e)}")

    def print_performance(self):
        """Display performance metrics"""
        duration = datetime.now() - self.start_time
        win_rate = self.wins / self.total_trades * 100 if self.total_trades > 0 else 0
        profit = self.equity - self.initial_equity
        
        print("\nPerformance Summary:")
        print(f"Running Time: {duration}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Current Equity: ${self.equity:.2f}")
        print(f"Total Profit: ${profit:.2f}")
        print(f"Daily Trades Remaining: {5 - self.daily_trades}\n")

    def reset_daily_count(self):
        """Reset daily counters with performance update"""
        self.daily_trades = 0
        print("\nDaily reset performed")
        self.print_performance()

    def run_strategy(self):
        """Main trading cycle"""
        if self.daily_trades >= 5:
            return

        try:
            df = self.fetch_market_data()
            if df.empty:
                print("No market data received")
                return

            df = self.calculate_indicators(df)
            signal = self.generate_signal(df)
            
            if signal:
                print(f"Signal detected: {signal.upper()}")
                self.execute_trade(signal)
            
            self.print_performance()

        except Exception as e:
            print(f"Strategy error: {str(e)}")

    def start(self):
        """Start automated trading"""
        print(f"\n=== Trading Bot Initialized ===")
        print(f"Start Time: {datetime.now()}")
        print(f"Initial Equity: ${self.initial_equity:.2f}")
        print(f"Risk per Trade: {self.risk_per_trade*100:.2f}%")
        print(f"Symbol: {self.symbol}\n")

        schedule.every().day.at("00:00").do(self.reset_daily_count)
        schedule.every(30).minutes.do(self.run_strategy)

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.print_performance()
            pd.DataFrame(self.trade_log).to_csv('final_trade_history.csv', index=False)
            print("Trade history saved to final_trade_history.csv")

if __name__ == "__main__":
    bot = AdvancedTradingBot()
    bot.start()
    import socket
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# DNS resolution override
original_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args):
    try:
        return original_getaddrinfo(*args)
    except socket.gaierror:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('104.18.114.97', 443))]
socket.getaddrinfo = new_getaddrinfo

class AdvancedTradingBot:
    def fetch_market_data(self):
        try:
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1)
            session.mount('https://', HTTPAdapter(max_retries=retries))
            
            response = session.get(
                "https://api-testnet.bybit.com/v5/market/kline",
                params={
                    "category": "linear",
                    "symbol": self.symbol,
                    "interval": "15",
                    "limit": "200"
                },
                timeout=10
            )
            return response.json()['result']['list']
        except Exception as e:
            print(f"Robust data fetch error: {e}")
            return []