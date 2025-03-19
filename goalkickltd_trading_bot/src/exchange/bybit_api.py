"""
Bybit API wrapper for the Goalkick Ltd Trading Bot.
Handles all exchange-specific API interactions.
"""

import time
import hmac
import hashlib
import json
import requests
from urllib.parse import urlencode
import threading
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datetime import datetime, timedelta
import pandas as pd

from config.logging_config import get_logger
from config.bot_config import EXCHANGE_CONFIG
from src.utils.error_handling import handle_error, ExchangeError

logger = get_logger("bybit_api")

class BybitAPI:
    """Wrapper for Bybit API."""
    
    def __init__(self, use_testnet=None, api_key=None, api_secret=None):
        """
        Initialize the Bybit API wrapper.
        
        Args:
            use_testnet (bool): Whether to use testnet (default from config)
            api_key (str): API key (default from config)
            api_secret (str): API secret (default from config)
        """
        self.use_testnet = use_testnet if use_testnet is not None else EXCHANGE_CONFIG["testnet"]
        self.api_key = api_key or EXCHANGE_CONFIG["api_key"]
        self.api_secret = api_secret or EXCHANGE_CONFIG["api_secret"]
        
        # API URLs
        self.base_url = "https://api-testnet.bybit.com" if self.use_testnet else "https://api.bybit.com"
        self.v5_url = f"{self.base_url}/v5"
        
        # For rate limiting
        self.request_limit = 120  # Requests per minute
        self.request_count = 0
        self.reset_time = time.time() + 60
        self.lock = threading.RLock()
        
        # Cached exchange information
        self.exchange_info = None
        self.symbols_info = {}
        
        logger.info(f"Initialized Bybit API client (testnet: {self.use_testnet})")
    
    def _update_rate_limit(self):
        """Update rate limit counters."""
        with self.lock:
            current_time = time.time()
            
            # Reset counter if a minute has passed
            if current_time > self.reset_time:
                self.request_count = 0
                self.reset_time = current_time + 60
            
            # Increment counter
            self.request_count += 1
            
            # Check if we're approaching the limit
            if self.request_count > self.request_limit * 0.8:
                logger.warning(f"API rate limit at {self.request_count}/{self.request_limit}")
            
            # If we've hit the limit, sleep until reset
            if self.request_count >= self.request_limit:
                sleep_time = max(0, self.reset_time - current_time)
                logger.warning(f"API rate limit reached, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
                # Reset after sleeping
                self.request_count = 0
                self.reset_time = time.time() + 60
    
    def _generate_signature(self, parameters, timestamp):
        """
        Generate request signature for authentication.
        
        Args:
            parameters (dict): Request parameters
            timestamp (int): Current timestamp in milliseconds
            
        Returns:
            str: HMAC signature
        """
        param_str = ''
        
        if parameters:
            if isinstance(parameters, dict):
                # Sort parameters by key
                parameters = dict(sorted(parameters.items()))
                param_str = urlencode(parameters)
            else:
                # Json string
                param_str = parameters
        
        sign_str = f"{timestamp}{self.api_key}{param_str}"
        return hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(sign_str, 'utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()
    
    def _handle_response(self, response):
        """
        Handle API response.
        
        Args:
            response: Requests response object
            
        Returns:
            dict: Response data
            
        Raises:
            ExchangeError: If the response indicates an error
        """
        try:
            data = response.json()
            
            if not data:
                logger.error(f"Empty response: {response.status_code}")
                raise ExchangeError(f"Empty response with status code {response.status_code}")
            
            # Check Bybit API response format
            if 'retCode' in data:
                if data['retCode'] != 0:
                    # Handle specific error codes
                    logger.error(f"API error: {data}")
                    raise ExchangeError(f"API error {data['retCode']}: {data.get('retMsg', 'Unknown error')}")
                
                return data.get('result', {})
            
            # Fallback for other formats
            return data
        except ValueError as e:
            logger.error(f"Invalid JSON response: {response.text}")
            raise ExchangeError(f"Invalid JSON response: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.RequestException, ExchangeError)),
        reraise=True
    )
    def _request(self, method, endpoint, params=None, auth=False, api_version='v5'):
        """
        Make a request to the Bybit API.
        
        Args:
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint
            params (dict): Request parameters
            auth (bool): Whether authentication is required
            api_version (str): API version
            
        Returns:
            dict: Response data
        """
        # Update rate limit counters
        self._update_rate_limit()
        
        # Prepare URL
        if api_version == 'v5':
            url = f"{self.v5_url}/{endpoint}"
        else:
            url = f"{self.base_url}/{endpoint}"
        
        # Prepare headers
        headers = {"Content-Type": "application/json"}
        
        # Add authentication if required
        if auth:
            if not self.api_key or not self.api_secret:
                logger.error("API key or secret not configured")
                raise ExchangeError("API key or secret not configured")
            
            timestamp = str(int(time.time() * 1000))
            signature = self._generate_signature(params, timestamp)
            
            headers.update({
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-SIGN": signature
            })
        
        # Make request
        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=15)
            elif method == "POST":
                response = requests.post(url, json=params, headers=headers, timeout=15)
            elif method == "DELETE":
                response = requests.delete(url, json=params, headers=headers, timeout=15)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                raise ExchangeError(f"Unsupported HTTP method: {method}")
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Handle response
            return self._handle_response(response)
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
    
    def ping(self):
        """
        Check if the exchange API is reachable.
        
        Returns:
            bool: True if reachable, False otherwise
        """
        try:
            response = self._request("GET", "market/time", api_version='v5')
            return 'timeSecond' in response
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            return False
    
    def reconnect(self):
        """
        Attempt to reconnect to the exchange.
        
        Returns:
            bool: True if reconnected successfully, False otherwise
        """
        try:
            logger.info("Attempting to reconnect to Bybit")
            return self.ping()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False
    
    def get_server_time(self):
        """
        Get the exchange server time.
        
        Returns:
            int: Server time in milliseconds
        """
        try:
            response = self._request("GET", "market/time", api_version='v5')
            return int(response['timeSecond']) * 1000
        except Exception as e:
            logger.error(f"Failed to get server time: {e}")
            handle_error(e, "Failed to get server time")
            return int(time.time() * 1000)
    
    def get_exchange_info(self, force_update=False):
        """
        Get exchange information including trading rules.
        
        Args:
            force_update (bool): Whether to force an update from the API
            
        Returns:
            dict: Exchange information
        """
        if self.exchange_info is None or force_update:
            try:
                # Get linear (USDT) perpetual symbols
                response = self._request("GET", "market/instruments-info", {
                    "category": "linear",
                    "status": "Trading"
                })
                
                symbols = {}
                for symbol_info in response.get('list', []):
                    symbol = symbol_info.get('symbol')
                    if symbol:
                        symbols[symbol] = symbol_info
                
                self.exchange_info = {
                    "symbols": symbols,
                    "server_time": self.get_server_time(),
                    "rate_limits": {
                        "request_weight": {
                            "limit": self.request_limit,
                            "interval": "MINUTE"
                        }
                    }
                }
                
                # Cache individual symbol info
                self.symbols_info = symbols
                
                logger.info(f"Retrieved exchange info for {len(symbols)} symbols")
            except Exception as e:
                logger.error(f"Failed to get exchange info: {e}")
                handle_error(e, "Failed to get exchange info")
                if self.exchange_info is None:
                    self.exchange_info = {"symbols": {}, "server_time": int(time.time() * 1000)}
        
        return self.exchange_info
    
    def get_symbol_info(self, symbol):
        """
        Get information for a specific symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Symbol information
        """
        # Ensure exchange info is loaded
        if not self.symbols_info:
            self.get_exchange_info()
        
        # Check cache first
        if symbol in self.symbols_info:
            return self.symbols_info[symbol]
        
        # Fetch specifically if not in cache
        try:
            response = self._request("GET", "market/instruments-info", {
                "category": "linear",
                "symbol": symbol
            })
            
            symbols = response.get('list', [])
            if symbols:
                symbol_info = symbols[0]
                self.symbols_info[symbol] = symbol_info
                return symbol_info
            
            logger.warning(f"Symbol {symbol} not found")
            return None
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            handle_error(e, f"Failed to get symbol info for {symbol}")
            return None
    
    def get_ticker(self, symbol):
        """
        Get ticker information for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Ticker information
        """
        try:
            response = self._request("GET", "market/tickers", {
                "category": "linear",
                "symbol": symbol
            })
            
            tickers = response.get('list', [])
            if tickers:
                return tickers[0]
            
            logger.warning(f"No ticker found for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            handle_error(e, f"Failed to get ticker for {symbol}")
            return None
    
    def get_tickers(self):
        """
        Get ticker information for all symbols.
        
        Returns:
            dict: Dictionary of symbol -> ticker data
        """
        try:
            response = self._request("GET", "market/tickers", {
                "category": "linear"
            })
            
            tickers = {}
            for ticker in response.get('list', []):
                symbol = ticker.get('symbol')
                if symbol:
                    tickers[symbol] = ticker
            
            return tickers
        except Exception as e:
            logger.error(f"Failed to get tickers: {e}")
            handle_error(e, "Failed to get tickers")
            return {}
    
    def get_order_book(self, symbol, depth=50):
        """
        Get order book for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            depth (int): Depth of the order book
            
        Returns:
            dict: Order book with bids and asks
        """
        try:
            response = self._request("GET", "market/orderbook", {
                "category": "linear",
                "symbol": symbol,
                "limit": depth
            })
            
            result = {
                'bids': [],
                'asks': []
            }
            
            if 'b' in response:
                result['bids'] = [[float(price), float(qty)] for price, qty in response['b']]
            
            if 'a' in response:
                result['asks'] = [[float(price), float(qty)] for price, qty in response['a']]
            
            return result
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            handle_error(e, f"Failed to get order book for {symbol}")
            return {'bids': [], 'asks': []}
    
    def get_funding_rate(self, symbol):
        """
        Get funding rate for a perpetual futures symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Funding rate information
        """
        try:
            response = self._request("GET", "market/funding/history", {
                "category": "linear",
                "symbol": symbol,
                "limit": 1
            })
            
            rates = response.get('list', [])
            if rates:
                return rates[0]
            
            logger.warning(f"No funding rate found for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Failed to get funding rate for {symbol}: {e}")
            handle_error(e, f"Failed to get funding rate for {symbol}")
            return None
    
    def get_candles(self, symbol, timeframe, start_time=None, end_time=None, limit=1000):
        """
        Get OHLCV candles for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe of the candles
            start_time (int): Start timestamp in milliseconds
            end_time (int): End timestamp in milliseconds
            limit (int): Maximum number of candles
            
        Returns:
            list: List of candles (timestamp, open, high, low, close, volume)
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": timeframe,
                "limit": min(limit, 1000)  # Bybit max limit is 1000
            }
            
            if start_time:
                params["start"] = int(start_time)
            
            if end_time:
                params["end"] = int(end_time)
            
            response = self._request("GET", "market/kline", params)
            
            candles = []
            for candle in response.get('list', []):
                # Bybit returns [timestamp, open, high, low, close, volume, ...]
                if len(candle) >= 6:
                    # Timestamps are in seconds, convert to milliseconds
                    timestamp = int(candle[0]) * 1000 if int(candle[0]) < 10000000000 else int(candle[0])
                    candles.append([
                        timestamp,
                        float(candle[1]),  # open
                        float(candle[2]),  # high
                        float(candle[3]),  # low
                        float(candle[4]),  # close
                        float(candle[5])   # volume
                    ])
            
            # Sort by timestamp (ascending)
            candles.sort(key=lambda x: x[0])
            
            return candles
        except Exception as e:
            logger.error(f"Failed to get candles for {symbol} {timeframe}: {e}")
            handle_error(e, f"Failed to get candles for {symbol} {timeframe}")
            return []
    
    def get_account_info(self):
        """
        Get account information including balances.
        
        Returns:
            dict: Account information
        """
        try:
            response = self._request("GET", "account/wallet-balance", {
                "accountType": "UNIFIED"
            }, auth=True)
            
            if 'list' not in response or not response['list']:
                logger.warning("No account data returned")
                return {"balance": 0, "available": 0, "positions": []}
            
            account = response['list'][0]
            
            # Extract USDT coin balance
            usdt_balance = 0
            usdt_available = 0
            
            for coin in account.get('coin', []):
                if coin.get('coin') == 'USDT':
                    usdt_balance = float(coin.get('walletBalance', 0))
                    usdt_available = float(coin.get('availableToWithdraw', 0))
                    break
            
            return {
                "balance": usdt_balance,
                "available": usdt_available,
                "total_equity": float(account.get('totalEquity', 0)),
                "positions": []  # Will be populated separately
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            handle_error(e, "Failed to get account info")
            return {"balance": 0, "available": 0, "positions": []}
    
    def get_positions(self, symbol=None):
        """
        Get open positions.
        
        Args:
            symbol (str): Filter by symbol (optional)
            
        Returns:
            list: List of position dictionaries
        """
        try:
            params = {
                "category": "linear"
            }
            
            if symbol:
                params["symbol"] = symbol
            
            response = self._request("GET", "position/list", params, auth=True)
            
            positions = []
            for pos in response.get('list', []):
                size = float(pos.get('size', 0))
                
                if size == 0:
                    continue  # Skip empty positions
                
                positions.append({
                    "symbol": pos.get('symbol'),
                    "side": "LONG" if pos.get('side') == 'Buy' else "SHORT",
                    "size": size,
                    "entry_price": float(pos.get('avgPrice', 0)),
                    "mark_price": float(pos.get('markPrice', 0)),
                    "unrealized_pnl": float(pos.get('unrealisedPnl', 0)),
                    "leverage": float(pos.get('leverage', 1)),
                    "isolated": pos.get('positionMode', "MergedSingle") == "BothSides",
                    "position_value": float(pos.get('positionValue', 0)),
                    "position_idx": int(pos.get('positionIdx', 0)),
                    "risk_id": int(pos.get('riskId', 0)),
                    "stop_loss": float(pos.get('stopLoss', 0)),
                    "take_profit": float(pos.get('takeProfit', 0)),
                    "trailing_stop": float(pos.get('trailingStop', 0)),
                    "created_time": int(pos.get('createdTime', 0))
                })
            
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            handle_error(e, "Failed to get positions")
            return []
    
    def place_order(self, order_params):
        """
        Place a new order.
        
        Args:
            order_params (dict): Order parameters
                - symbol: Trading pair symbol
                - side: "Buy" or "Sell"
                - order_type: "Limit" or "Market"
                - qty: Order quantity
                - price: Order price (optional for Market orders)
                - time_in_force: "GTC", "IOC", "FOK", etc.
                - reduce_only: Whether to reduce position only
                - close_on_trigger: Close on trigger
                - position_idx: Position index (0: one-way, 1: hedge-buy, 2: hedge-sell)
                
        Returns:
            dict: Order information
        """
        try:
            # Map internal parameters to Bybit API format
            params = {
                "category": "linear",
                "symbol": order_params["symbol"],
                "side": order_params["side"],
                "orderType": order_params["order_type"],
                "qty": str(order_params["qty"]),
                "positionIdx": order_params.get("position_idx", 0)
            }
            
            # Add price for limit orders
            if order_params["order_type"].lower() == "limit" and "price" in order_params:
                params["price"] = str(order_params["price"])
            
            # Add time in force
            if "time_in_force" in order_params:
                params["timeInForce"] = order_params["time_in_force"]
            else:
                params["timeInForce"] = "GTC"  # Good Till Cancel
            
            # Add reduce only flag
            if "reduce_only" in order_params:
                params["reduceOnly"] = order_params["reduce_only"]
            
            # Add close on trigger flag
            if "close_on_trigger" in order_params:
                params["closeOnTrigger"] = order_params["close_on_trigger"]
            
            # Add client order ID if provided
            if "client_order_id" in order_params:
                params["orderLinkId"] = order_params["client_order_id"]
            
            # Add take profit and stop loss if provided
            if "take_profit" in order_params:
                params["takeProfit"] = str(order_params["take_profit"])
            
            if "stop_loss" in order_params:
                params["stopLoss"] = str(order_params["stop_loss"])
            
            response = self._request("POST", "order/create", params, auth=True)
            
            order_id = response.get('orderId')
            if not order_id:
                logger.warning(f"No order ID in response: {response}")
                raise ExchangeError("No order ID in response")
            
            # Return standardized order information
            return {
                "order_id": order_id,
                "client_order_id": response.get('orderLinkId'),
                "symbol": order_params["symbol"],
                "side": order_params["side"],
                "order_type": order_params["order_type"],
                "price": order_params.get("price"),
                "qty": order_params["qty"],
                "time_in_force": params["timeInForce"],
                "reduce_only": order_params.get("reduce_only", False),
                "close_on_trigger": order_params.get("close_on_trigger", False),
                "position_idx": params["positionIdx"],
                "status": "NEW",
                "created_time": int(time.time() * 1000)
            }
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            handle_error(e, "Failed to place order")
            raise
    
    def cancel_order(self, symbol, order_id=None, client_order_id=None):
        """
        Cancel an open order.
        
        Args:
            symbol (str): Trading pair symbol
            order_id (str): Order ID
            client_order_id (str): Client order ID
            
        Returns:
            dict: Cancellation response
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol
            }
            
            if order_id:
                params["orderId"] = order_id
            elif client_order_id:
                params["orderLinkId"] = client_order_id
            else:
                raise ValueError("Either order_id or client_order_id must be provided")
            
            response = self._request("POST", "order/cancel", params, auth=True)
            
            return {
                "order_id": response.get('orderId'),
                "client_order_id": response.get('orderLinkId'),
                "symbol": symbol,
                "status": "CANCELED"
            }
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            handle_error(e, "Failed to cancel order")
            raise
    
    def cancel_all_orders(self, symbol=None):
        """
        Cancel all open orders, optionally filtered by symbol.
        
        Args:
            symbol (str): Trading pair symbol (optional)
            
        Returns:
            list: List of canceled order IDs
        """
        try:
            params = {
                "category": "linear"
            }
            
            if symbol:
                params["symbol"] = symbol
            
            response = self._request("POST", "order/cancel-all", params, auth=True)
            
            return response.get('list', [])
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            handle_error(e, "Failed to cancel all orders")
            raise
    
    def get_order(self, symbol, order_id=None, client_order_id=None):
        """
        Get order information.
        
        Args:
            symbol (str): Trading pair symbol
            order_id (str): Order ID
            client_order_id (str): Client order ID
            
        Returns:
            dict: Order information
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol
            }
            
            if order_id:
                params["orderId"] = order_id
            elif client_order_id:
                params["orderLinkId"] = client_order_id
            else:
                raise ValueError("Either order_id or client_order_id must be provided")
            
            response = self._request("GET", "order/history", params, auth=True)
            
            orders = response.get('list', [])
            if not orders:
                logger.warning(f"Order not found: {order_id or client_order_id}")
                return None
            
            order = orders[0]
            
            # Map Bybit status to common status
            status_map = {
                "Created": "NEW",
                "New": "NEW",
                "Rejected": "REJECTED",
                "PartiallyFilled": "PARTIALLY_FILLED",
                "Filled": "FILLED",
                "Cancelled": "CANCELED",
                "PendingCancel": "PENDING_CANCEL"
            }
            
            # Return standardized order information
            return {
                "order_id": order.get('orderId'),
                "client_order_id": order.get('orderLinkId'),
                "symbol": order.get('symbol'),
                "side": order.get('side'),
                "order_type": order.get('orderType'),
                "price": float(order.get('price', 0)),
                "qty": float(order.get('qty', 0)),
                "executed_qty": float(order.get('execQty', 0)),
                "executed_price": float(order.get('avgPrice', 0)),
                "time_in_force": order.get('timeInForce'),
                "reduce_only": order.get('reduceOnly', False),
                "close_on_trigger": order.get('closeOnTrigger', False),
                "position_idx": int(order.get('positionIdx', 0)),
                "status": status_map.get(order.get('orderStatus'), order.get('orderStatus')),
                "created_time": int(order.get('createdTime', 0)),
                "updated_time": int(order.get('updatedTime', 0))
            }
        except Exception as e:
            logger.error(f"Failed to get order: {e}")
            handle_error(e, "Failed to get order")
            return None
    
    def get_open_orders(self, symbol=None):
        """
        Get all open orders, optionally filtered by symbol.
        
        Args:
            symbol (str): Trading pair symbol (optional)
            
        Returns:
            list: List of open orders
        """
        try:
            params = {
                "category": "linear",
                "settleCoin": "USDT",
                "limit": 50
            }
            
            if symbol:
                params["symbol"] = symbol
            
            response = self._request("GET", "order/realtime", params, auth=True)
            
            orders = []
            for order in response.get('list', []):
                # Map Bybit status to common status
                status_map = {
                    "Created": "NEW",
                    "New": "NEW",
                    "Rejected": "REJECTED",
                    "PartiallyFilled": "PARTIALLY_FILLED",
                    "Filled": "FILLED",
                    "Cancelled": "CANCELED",
                    "PendingCancel": "PENDING_CANCEL"
                }
                
                orders.append({
                    "order_id": order.get('orderId'),
                    "client_order_id": order.get('orderLinkId'),
                    "symbol": order.get('symbol'),
                    "side": order.get('side'),
                    "order_type": order.get('orderType'),
                    "price": float(order.get('price', 0)),
                    "qty": float(order.get('qty', 0)),
                    "executed_qty": float(order.get('cumExecQty', 0)),
                    "executed_price": float(order.get('avgPrice', 0)),
                    "time_in_force": order.get('timeInForce'),
                    "reduce_only": order.get('reduceOnly', False),
                    "close_on_trigger": order.get('closeOnTrigger', False),
                    "position_idx": int(order.get('positionIdx', 0)),
                    "status": status_map.get(order.get('orderStatus'), order.get('orderStatus')),
                    "created_time": int(order.get('createdTime', 0)),
                    "updated_time": int(order.get('updatedTime', 0))
                })
            
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            handle_error(e, "Failed to get open orders")
            return []
    
    def get_order_history(self, symbol=None, limit=50):
        """
        Get order history, optionally filtered by symbol.
        
        Args:
            symbol (str): Trading pair symbol (optional)
            limit (int): Maximum number of orders to retrieve
            
        Returns:
            list: List of historical orders
        """
        try:
            params = {
                "category": "linear",
                "limit": min(limit, 100)  # Bybit max limit is 100
            }
            
            if symbol:
                params["symbol"] = symbol
            
            response = self._request("GET", "order/history", params, auth=True)
            
            orders = []
            for order in response.get('list', []):
                # Map Bybit status to common status
                status_map = {
                    "Created": "NEW",
                    "New": "NEW",
                    "Rejected": "REJECTED",
                    "PartiallyFilled": "PARTIALLY_FILLED",
                    "Filled": "FILLED",
                    "Cancelled": "CANCELED",
                    "PendingCancel": "PENDING_CANCEL"
                }
                
                orders.append({
                    "order_id": order.get('orderId'),
                    "client_order_id": order.get('orderLinkId'),
                    "symbol": order.get('symbol'),
                    "side": order.get('side'),
                    "order_type": order.get('orderType'),
                    "price": float(order.get('price', 0)),
                    "qty": float(order.get('qty', 0)),
                    "executed_qty": float(order.get('cumExecQty', 0)),
                    "executed_price": float(order.get('avgPrice', 0)),
                    "time_in_force": order.get('timeInForce'),
                    "reduce_only": order.get('reduceOnly', False),
                    "close_on_trigger": order.get('closeOnTrigger', False),
                    "position_idx": int(order.get('positionIdx', 0)),
                    "status": status_map.get(order.get('orderStatus'), order.get('orderStatus')),
                    "created_time": int(order.get('createdTime', 0)),
                    "updated_time": int(order.get('updatedTime', 0))
                })
            
            return orders
        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
            handle_error(e, "Failed to get order history")
            return []
    
    def set_leverage(self, symbol, leverage):
        """
        Set leverage for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            leverage (float): Leverage value
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage)
            }
            
            self._request("POST", "position/set-leverage", params, auth=True)
            
            logger.info(f"Set leverage for {symbol} to {leverage}x")
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            handle_error(e, f"Failed to set leverage for {symbol}")
            return False