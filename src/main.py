# import os
# import time
# import schedule
# import pandas as pd
# import pandas_ta as ta
# from datetime import datetime
# from pybit.unified_trading import HTTP
# from dotenv import load_dotenv

# load_dotenv()

# class AdvancedTradingBot:
#     def __init__(self):
#         self.session = HTTP(
#             api_key=os.getenv("BYBIT_API_KEY"),
#             api_secret=os.getenv("BYBIT_API_SECRET"),
#             testnet=True
#         )
#         self.symbol = "ETHUSDT"
#         self.daily_trades = 0
#         self.total_trades = 0
#         self.equity = self.get_equity()
#         self.initial_equity = self.equity
#         self.trade_log = []
#         self.risk_per_trade = 0.0001  # 0.01% risk per trade
#         self.setup_strategies()
#         self.start_time = datetime.now()
        
#         # Performance tracking
#         self.wins = 0
#         self.losses = 0

#     def get_equity(self):
#         """Get current USDT balance with enhanced error handling"""
#         try:
#             response = self.session.get_wallet_balance(
#                 accountType="UNIFIED", 
#                 coin="USDT"
#             )
#             if response['retCode'] == 0:
#                 coins = response['result']['list'][0]['coin']
#                 usdt_balance = next(
#                     (c for c in coins if c['coin'] == 'USDT'), 
#                     {'equity': '0'}
#                 )
#                 return float(usdt_balance.get('equity', 0))
#             print(f"API Error: {response['retMsg']}")
#             return 0
#         except Exception as e:
#             print(f"Critical balance error: {str(e)}")
#             return 0

#     def setup_strategies(self):
#         """Initialize strategy parameters with validation"""
#         self.indicators = {
#             'rsi_period': 14,
#             'macd_fast': 12,
#             'macd_slow': 26,
#             'macd_signal': 9,
#             'ema_short': 20,
#             'ema_long': 50,
#             'atr_period': 14
#         }
#         print("🔧 Strategy parameters initialized:")
#         print(f"• RSI Period: {self.indicators['rsi_period']}")
#         print(f"• MACD Configuration: {self.indicators['macd_fast']}/{self.indicators['macd_slow']}/{self.indicators['macd_signal']}")
#         print(f"• EMA Configuration: {self.indicators['ema_short']}/{self.indicators['ema_long']}")
#         print(f"• ATR Period: {self.indicators['atr_period']}\n")

#     # ... (keep existing fetch_market_data, calculate_indicators, generate_signal methods)

#     def execute_trade(self, signal):
#         """Execute trade with enhanced logging and validation"""
#         if self.daily_trades >= 5 or self.equity < 10:
#             return

#         try:
#             # Get precise pricing with retry logic
#             ticker = self.session.get_tickers(category="linear", symbol=self.symbol)
#             price = float(ticker['result']['list'][0]['lastPrice'])
            
#             # Calculate position size with compound growth
#             position_size = self.calculate_position_size() / price
#             if position_size < 0.001:  # Minimum order size
#                 print("⚠️ Position size too small, skipping trade")
#                 return

#             # Calculate risk parameters
#             stop_loss = price * 0.9999  # 0.01% SL
#             take_profit = price * 1.0002  # 0.02% TP

#             # Execute order
#             order = self.session.place_order(
#                 category="linear",
#                 symbol=self.symbol,
#                 side="Buy" if signal == 'buy' else "Sell",
#                 orderType="Market",
#                 qty=f"{position_size:.4f}",
#                 stopLoss=f"{stop_loss:.4f}",
#                 takeProfit=f"{take_profit:.4f}",
#                 timeInForce="GTC"
#             )

#             # Update tracking
#             self.total_trades += 1
#             self.daily_trades += 1
#             new_equity = self.get_equity()
#             profit = new_equity - self.equity
#             self.equity = new_equity

#             # Update win/loss count
#             if profit > 0:
#                 self.wins += 1
#             else:
#                 self.losses += 1

#             # Log trade
#             self.log_trade(order, signal, position_size, price, profit)
            
#             # Print trade execution details
#             print(f"\n🎯 Trade Executed ({self.total_trades})")
#             print(f"• Direction: {'LONG' if signal == 'buy' else 'SHORT'}")
#             print(f"• Entry Price: ${price:.2f}")
#             print(f"• Position Size: {position_size:.4f} ETH")
#             print(f"• Stop Loss: ${stop_loss:.2f}")
#             print(f"• Take Profit: ${take_profit:.2f}")
#             print(f"• PnL: ${profit:.2f} ({profit/self.equity*100:.2f}%)\n")

#         except Exception as e:
#             print(f"🚨 Trade execution failed: {str(e)}")

#     def log_trade(self, order, signal, qty, price, profit):
#         """Enhanced trade logging with performance metrics"""
#         log_entry = {
#             'timestamp': datetime.now().isoformat(),
#             'signal': signal,
#             'qty': qty,
#             'price': price,
#             'profit': profit,
#             'equity': self.equity,
#             'order_id': order['result']['orderId'],
#             'status': order['result']['orderStatus']
#         }
#         self.trade_log.append(log_entry)
        
#         # Save to CSV with error handling
#         try:
#             pd.DataFrame(self.trade_log).to_csv('trade_history.csv', index=False)
#         except Exception as e:
#             print(f"📄 Logging error: {str(e)}")

#     def print_performance(self):
#         """Display real-time performance metrics"""
#         duration = datetime.now() - self.start_time
#         win_rate = self.wins / self.total_trades * 100 if self.total_trades > 0 else 0
#         profit = self.equity - self.initial_equity
#         roi = (profit / self.initial_equity) * 100
        
#         print("\n📊 Performance Summary")
#         print(f"• Running Time: {duration}")
#         print(f"• Total Trades: {self.total_trades}")
#         print(f"• Daily Trades: {self.daily_trades}/5")
#         print(f"• Win Rate: {win_rate:.1f}%")
#         print(f"• Current Equity: ${self.equity:.2f}")
#         print(f"• Total Profit: ${profit:.2f} ({roi:.1f}% ROI)")
#         print(f"• Risk Exposure: {self.risk_per_trade*100:.2f}% per trade\n")

#     def reset_daily_count(self):
#         """Reset daily trade counter with notification"""
#         self.daily_trades = 0
#         print(f"\n🔄 {datetime.now()} - Daily trades reset")
#         self.print_performance()

#     def run_strategy(self):
#         """Enhanced strategy runner with status updates"""
#         print(f"\n⏳ {datetime.now()} - Running strategy check...")
#         try:
#             df = self.fetch_market_data()
#             if df.empty:
#                 print("⚠️ No market data received")
#                 return

#             df = self.calculate_indicators(df)
#             signal = self.generate_signal(df)
            
#             if signal:
#                 print(f"📈 Signal detected: {signal.upper()}")
#                 self.execute_trade(signal)
#             else:
#                 print("⏸️ No trading signals found")
                
#             self.print_performance()
            
#         except Exception as e:
#             print(f"🚨 Strategy error: {str(e)}")

#     def start(self):
#         """Start automated trading with continuous operation"""
#         print(f"\n🚀 {datetime.now()} - Bot initializing...")
#         print(f"• Initial Equity: ${self.initial_equity:.2f}")
#         print(f"• Symbol: {self.symbol}")
#         print(f"• Risk Per Trade: {self.risk_per_trade*100:.2f}%\n")
        
#         # Schedule regular checks
#         schedule.every().day.at("00:00").do(self.reset_daily_count)
#         schedule.every(15).minutes.do(self.run_strategy)  # Check every 15 minutes
        
#         try:
#             while True:
#                 schedule.run_pending()
#                 time.sleep(1)  # Check scheduler every second
#         except KeyboardInterrupt:
#             print("\n🛑 Bot shutdown requested!")
#             self.print_performance()
#             print("💾 Saving trade history...")
#             pd.DataFrame(self.trade_log).to_csv('final_trade_history.csv', index=False)
#             print("✅ Shutdown complete")

# if __name__ == "__main__":
#     bot = AdvancedTradingBot()
#     bot.start()
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
        self.risk_per_trade = 0.02 #% risk per trade
        self.setup_strategies()
        self.start_time = datetime.now()
        self.wins = 0.05
        self.losses = 0.02

    def get_equity(self):
        """Get current USDT balance with enhanced error handling"""
        try:
            response = self.session.get_wallet_balance(
                accountType="UNIFIED", 
                coin="USDT"
            )
            if response['retCode'] == 0:
                coins = response['result']['list'][0]['coin']
                usdt_balance = next(
                    (c for c in coins if c['coin'] == 'USDT'), 
                    {'equity': '0'}
                )
                return float(usdt_balance.get('equity', 0))
            print(f"API Error: {response['retMsg']}")
            return 0
        except Exception as e:
            print(f"Critical balance error: {str(e)}")
            return 0

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
        print("🔧 Strategy parameters initialized:")
        print(f"• RSI Period: {self.indicators['rsi_period']}")
        print(f"• MACD Configuration: {self.indicators['macd_fast']}/{self.indicators['macd_slow']}/{self.indicators['macd_signal']}")
        print(f"• EMA Configuration: {self.indicators['ema_short']}/{self.indicators['ema_long']}")
        print(f"• ATR Period: {self.indicators['atr_period']}\n")

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
        """Calculate technical indicators"""
        try:
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
        except Exception as e:
            print(f"Indicator error: {str(e)}")
            return df

    def generate_signal(self, df):
        """Generate trading signals"""
        try:
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
        except Exception as e:
            print(f"Signal error: {str(e)}")
            return None

    def calculate_position_size(self):
        """Calculate position size based on current equity"""
        return self.equity * self.risk_per_trade

    def execute_trade(self, signal):
        """Execute trade with enhanced logging and validation"""
        if self.daily_trades >= 5 or self.equity < 10:
            return

        try:
            # Get precise pricing
            ticker = self.session.get_tickers(category="linear", symbol=self.symbol)
            price = float(ticker['result']['list'][0]['lastPrice'])
            
            # Calculate position size with compound growth
            position_size = self.calculate_position_size() / price
            if position_size < 0.001:  # Minimum order size
                print("⚠️ Position size too small, skipping trade")
                return

            # Calculate risk parameters
            stop_loss = price * 0.9999  # 0.01% SL
            take_profit = price * 1.0002  # 0.02% TP

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

            # Update win/loss count
            if profit > 0:
                self.wins += 1
            else:
                self.losses += 1

            # Log trade
            self.log_trade(order, signal, position_size, price, profit)
            
            # Print trade execution details
            print(f"\n🎯 Trade Executed ({self.total_trades})")
            print(f"• Direction: {'LONG' if signal == 'buy' else 'SHORT'}")
            print(f"• Entry Price: ${price:.2f}")
            print(f"• Position Size: {position_size:.4f} ETH")
            print(f"• Stop Loss: ${stop_loss:.2f}")
            print(f"• Take Profit: ${take_profit:.2f}")
            print(f"• PnL: ${profit:.2f} ({profit/self.equity*100:.2f}%)\n")

        except Exception as e:
            print(f"🚨 Trade execution failed: {str(e)}")

    def log_trade(self, order, signal, qty, price, profit):
        """Enhanced trade logging with performance metrics"""
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
        
        # Save to CSV with error handling
        try:
            pd.DataFrame(self.trade_log).to_csv('trade_history.csv', index=False)
        except Exception as e:
            print(f"📄 Logging error: {str(e)}")

    def print_performance(self):
        """Display real-time performance metrics"""
        duration = datetime.now() - self.start_time
        win_rate = self.wins / self.total_trades * 100 if self.total_trades > 0 else 0
        profit = self.equity - self.initial_equity
        roi = (profit / self.initial_equity) * 100
        
        print("\n📊 Performance Summary")
        print(f"• Running Time: {duration}")
        print(f"• Total Trades: {self.total_trades}")
        print(f"• Daily Trades: {self.daily_trades}/5")
        print(f"• Win Rate: {win_rate:.1f}%")
        print(f"• Current Equity: ${self.equity:.2f}")
        print(f"• Total Profit: ${profit:.2f} ({roi:.1f}% ROI)")
        print(f"• Risk Exposure: {self.risk_per_trade*100:.2f}% per trade\n")

    def reset_daily_count(self):
        """Reset daily trade counter with notification"""
        self.daily_trades = 0
        print(f"\n🔄 {datetime.now()} - Daily trades reset")
        self.print_performance()

    def run_strategy(self):
        """Enhanced strategy runner with status updates"""
        print(f"\n⏳ {datetime.now()} - Running strategy check...")
        try:
            df = self.fetch_market_data()
            if df.empty:
                print("⚠️ No market data received")
                return

            df = self.calculate_indicators(df)
            signal = self.generate_signal(df)
            
            if signal:
                print(f"📈 Signal detected: {signal.upper()}")
                self.execute_trade(signal)
            else:
                print("⏸️ No trading signals found")
                
            self.print_performance()
            
        except Exception as e:
            print(f"🚨 Strategy error: {str(e)}")

    def start(self):
        """Start automated trading with continuous operation"""
        print(f"\n🚀 {datetime.now()} - Bot initializing...")
        print(f"• Initial Equity: ${self.initial_equity:.2f}")
        print(f"• Symbol: {self.symbol}")
        print(f"• Risk Per Trade: {self.risk_per_trade*100:.2f}%\n")
        
        # Schedule regular checks
        schedule.every().day.at("00:00").do(self.reset_daily_count)
        schedule.every(1).hours.do(self.run_strategy)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Bot shutdown requested!")
            self.print_performance()
            print("💾 Saving trade history...")
            pd.DataFrame(self.trade_log).to_csv('final_trade_history.csv', index=False)
            print("✅ Shutdown complete")

if __name__ == "__main__":
    bot = AdvancedTradingBot()
    bot.start()