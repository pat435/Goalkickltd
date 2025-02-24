# import requests
# import time
# import hmac
# import hashlib
# import os
# from dotenv import load_dotenv

# load_dotenv()

# def apply_demo_funds(currency="USDT", amount=10000):
#     """
#     Request demo funds from Bybit testnet
#     :param currency: USDT, BTC, ETH, USDC
#     :param amount: Amount to request (max limits apply)
#     :return: API response
#     """
#     # API credentials from MAINNET account
#     api_key = os.getenv("BYBIT_MAINNET_API_KEY")
#     api_secret = os.getenv("BYBIT_MAINNET_API_SECRET")
    
#     # API configuration
#     endpoint = "/v5/account/demo-apply-money"
#     base_url = "https://api.bybit.com"
#     timestamp = str(int(time.time() * 1000))
    
#     # Prepare parameters
#     params = {
#         "currency": currency,
#         "amount": str(amount),
#         "adjustType": "0"  # 0: Add funds, 1: Reduce funds
#     }

#     # Generate signature
#     signature_payload = f"{timestamp}{api_key}5000{params}"
#     signature = hmac.new(
#         api_secret.encode("utf-8"),
#         signature_payload.encode("utf-8"),
#         hashlib.sha256
#     ).hexdigest()

#     # Prepare headers
#     headers = {
#         "X-BAPI-API-KEY": api_key,
#         "X-BAPI-SIGN": signature,
#         "X-BAPI-TIMESTAMP": timestamp,
#         "X-BAPI-RECV-WINDOW": "5000",
#         "Content-Type": "application/json"
#     }

#     # Send request
#     response = requests.post(
#         url=f"{base_url}{endpoint}",
#         headers=headers,
#         json=params
#     )

#     return response.json()

# # Usage example
# if __name__ == "__main__":
#     # Set your mainnet API keys in .env file first!
#     result = apply_demo_funds(currency="USDT", amount=10000)
#     print("Funds Request Result:", result)

# # from pybit.unified_trading import HTTP
# # from dotenv import load_dotenv
# # import os

# # load_dotenv()

# # # Bybit API credentials
# # BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "qPPPPsYu77bbczvD03")
# # BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "hkyOsWcsuNqzeTjAY9eFi0Ux5s4ybAHXnPcO")
# # TESTNET = True  # True means your API keys were generated on testnet.bybit.com

# # # Create a session
# # session = HTTP(
# #     api_key=BYBIT_API_KEY,
# #     api_secret=BYBIT_API_SECRET,
# #     testnet=TESTNET,
# # )

# # def get_account_details():
# #     # Get wallet balance
# #     balance = session.get_wallet_balance(accountType="UNIFIED", coin="ETH")
# #     print("Balance:", balance)

# #     # Get trade history
# #     trades = session.get_account_info(accountType="UNIFIED")
# #     print("Trades:", trades)

# # # Call the function to print account details
# # get_account_details()
# from pybit.unified_trading import HTTP
# from dotenv import load_dotenv
# import os
# import schedule
# import time
# import pandas as pd
# import pandas_ta as ta
# import numpy as np
# from datetime import datetime

# load_dotenv()

# class BybitTradingBot:
#     def __init__(self):
#         self.session = HTTP(
#             api_key=os.getenv("BYBIT_API_KEY"),
#             api_secret=os.getenv("BYBIT_API_SECRET"),
#             testnet=True
#         )
#         self.symbol = "ETHUSDT"
#         self.leverage = 10
#         self.trade_count = 0
#         self.daily_trade_limit = 5
#         self.equity = self.get_equity()
#         self.set_leverage()
        
#     def get_equity(self):
#         try:
#             response = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
#             return float(response['result']['list'][0]['coin'][0]['equity'])
#         except Exception as e:
#             print(f"Error getting balance: {e}")
#             return 0

#     def set_leverage(self):
#         try:
#             self.session.set_leverage(
#                 category="linear",
#                 symbol=self.symbol,
#                 buyLeverage=str(self.leverage),
#                 sellLeverage=str(self.leverage),
#             )
#         except Exception as e:
#             print(f"Error setting leverage: {e}")

#     def fetch_market_data(self):
#         try:
#             data = self.session.get_kline(
#                 category="linear",
#                 symbol=self.symbol,
#                 interval="15",
#                 limit=100
#             )
#             df = pd.DataFrame(data['result']['list'], columns=[
#                 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
#             df = df.astype({'open': float, 'high': float, 'low': float, 'close': float})
#             df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#             return df.iloc[::-1].reset_index(drop=True)
#         except Exception as e:
#             print(f"Error fetching data: {e}")
#             return pd.DataFrame()

#     def calculate_indicators(self, df):
#         df['RSI'] = ta.rsi(df['close'], length=14)
#         macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
#         df['MACD'] = macd['MACD_12_26_9']
#         df['MACD_Signal'] = macd['MACDs_12_26_9']
#         df['MA20'] = ta.sma(df['close'], length=20)
#         df['MA50'] = ta.sma(df['close'], length=50)
#         return df

#     def generate_signal(self, df):
#         if len(df) < 2:
#             return None

#         last_row = df.iloc[-1]
#         prev_row = df.iloc[-2]

#         # Trend-following strategy with confirmation
#         bullish = (
#             (last_row['RSI'] < 35) and
#             (last_row['MACD'] > last_row['MACD_Signal']) and
#             (last_row['MA20'] > last_row['MA50']) and
#             (last_row['close'] > prev_row['high'])
#         )

#         bearish = (
#             (last_row['RSI'] > 65) and
#             (last_row['MACD'] < last_row['MACD_Signal']) and
#             (last_row['MA20'] < last_row['MA50']) and
#             (last_row['close'] < prev_row['low'])
#         )

#         return 'buy' if bullish else 'sell' if bearish else None

#     def calculate_position_size(self, current_price):
#         risk_per_trade = self.equity * 0.02  # 2% risk per trade
#         price_diff = current_price * 0.02  # 2% stop loss
#         return round(risk_per_trade / price_diff, 3)

#     def execute_trade(self, signal):
#         if self.trade_count >= self.daily_trade_limit:
#             return

#         try:
#             ticker = self.session.get_tickers(category="linear", symbol=self.symbol)
#             current_price = float(ticker['result']['list'][0]['lastPrice'])
            
#             position_size = self.calculate_position_size(current_price)
#             if position_size <= 0:
#                 return

#             stop_loss = current_price * 0.98 if signal == 'buy' else current_price * 1.02
#             take_profit = current_price * 1.05 if signal == 'buy' else current_price * 0.95

#             self.session.place_order(
#                 category="linear",
#                 symbol=self.symbol,
#                 side="Buy" if signal == 'buy' else "Sell",
#                 orderType="Market",
#                 qty=str(position_size),
#                 stopLoss=str(stop_loss),
#                 takeProfit=str(take_profit),
#                 timeInForce="GTC"
#             )
            
#             self.trade_count += 1
#             self.equity = self.get_equity()
#             print(f"{datetime.now()} - Executed {signal} | Size: {position_size} | Equity: ${self.equity:.2f}")

#         except Exception as e:
#             print(f"Trade execution failed: {e}")

#     def reset_daily_trades(self):
#         self.trade_count = 0
#         print(f"{datetime.now()} - Daily trades reset")

#     def run_strategy(self):
#         if self.trade_count >= self.daily_trade_limit:
#             return

#         df = self.fetch_market_data()
#         if df.empty:
#             return

#         df = self.calculate_indicators(df)
#         signal = self.generate_signal(df)
        
#         if signal:
#             self.execute_trade(signal)

#     def start(self):
#         schedule.every().day.at("00:00").do(self.reset_daily_trades)
#         schedule.every(15).minutes.do(self.run_strategy)

#         print(f"{datetime.now()} - Bot started with initial equity: ${self.equity:.2f}")
#         while True:
#             schedule.run_pending()
#             time.sleep(1)

# if __name__ == "__main__":
#     bot = BybitTradingBot()
#     bot.start()