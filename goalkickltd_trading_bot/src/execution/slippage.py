"""
Slippage management module for the Goalkick Ltd Trading Bot.
Handles slippage estimation, monitoring, and mitigation strategies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading

from config.logging_config import get_logger
from config.bot_config import TRADING_CONFIG
from src.utils.error_handling import handle_error

logger = get_logger("execution.slippage")

class SlippageManager:
    """Class for managing slippage in order execution."""
    
    def __init__(self, exchange_api, datastore=None):
        """
        Initialize the SlippageManager.
        
        Args:
            exchange_api: Exchange API instance
            datastore: DataStore instance (optional)
        """
        self.exchange_api = exchange_api
        self.datastore = datastore
        self.slippage_history = {}  # symbol -> list of slippage data
        self.lock = threading.RLock()
        self.max_history_size = 100  # Max number of slippage records to keep per symbol
    
    def estimate_slippage(self, symbol, order_type, order_size, spread_factor=1.0):
        """
        Estimate potential slippage for a given order.
        
        Args:
            symbol (str): Trading pair symbol
            order_type (str): Order type (Market or Limit)
            order_size (float): Order size in base currency
            spread_factor (float): Multiplier to apply to spread for slippage estimation
            
        Returns:
            float: Estimated slippage as a percentage
        """
        try:
            # For Market orders, estimate slippage based on order book depth and historical data
            if order_type.upper() == "MARKET":
                # Get order book data
                order_book = self.exchange_api.get_order_book(symbol, depth=20)
                
                if not order_book or not order_book['bids'] or not order_book['asks']:
                    logger.warning(f"Could not get order book for {symbol}, using default slippage")
                    return self._get_default_slippage(symbol, order_type)
                
                # Calculate mid price
                best_bid = float(order_book['bids'][0][0])
                best_ask = float(order_book['asks'][0][0])
                mid_price = (best_bid + best_ask) / 2
                
                # Calculate current spread
                spread = (best_ask - best_bid) / mid_price
                
                # Estimate slippage based on order size and order book depth
                bids = order_book['bids']
                asks = order_book['asks']
                
                # Convert to pandas DataFrame for easier manipulation
                bids_df = pd.DataFrame(bids, columns=['price', 'quantity'])
                asks_df = pd.DataFrame(asks, columns=['price', 'quantity'])
                
                # Convert to float
                bids_df = bids_df.astype(float)
                asks_df = asks_df.astype(float)
                
                # Calculate cumulative quantity
                bids_df['cum_quantity'] = bids_df['quantity'].cumsum()
                asks_df['cum_quantity'] = asks_df['quantity'].cumsum()
                
                # Estimate slippage based on order size
                if order_size > 0:
                    # For buy orders, check ask side
                    if order_size > asks_df['cum_quantity'].max():
                        # Order size exceeds order book depth
                        logger.warning(f"Order size {order_size} exceeds order book depth for {symbol}")
                        return spread * 5  # Higher slippage estimate
                    
                    # Find the price level that can fulfill the order
                    execution_row = asks_df[asks_df['cum_quantity'] >= order_size].iloc[0]
                    execution_price = execution_row['price']
                    
                    # Calculate slippage as percentage from mid price
                    slippage = (execution_price - mid_price) / mid_price
                else:
                    # For sell orders, check bid side
                    abs_order_size = abs(order_size)
                    if abs_order_size > bids_df['cum_quantity'].max():
                        # Order size exceeds order book depth
                        logger.warning(f"Order size {abs_order_size} exceeds order book depth for {symbol}")
                        return spread * 5  # Higher slippage estimate
                    
                    # Find the price level that can fulfill the order
                    execution_row = bids_df[bids_df['cum_quantity'] >= abs_order_size].iloc[0]
                    execution_price = execution_row['price']
                    
                    # Calculate slippage as percentage from mid price
                    slippage = (mid_price - execution_price) / mid_price
                
                # Apply spread factor to adjust the slippage estimate
                slippage = slippage * spread_factor
                
                logger.debug(f"Estimated slippage for {symbol} market order of size {order_size}: {slippage:.4%}")
                return slippage
            
            # For Limit orders, slippage is typically negative (i.e., better than expected price)
            # or zero if the order is filled at the limit price
            elif order_type.upper() == "LIMIT":
                # For limit orders, we can use a small negative slippage to account for price improvement
                limit_slippage = -0.0005  # -0.05% (price improvement)
                return limit_slippage
            
            else:
                # Default slippage for unknown order types
                return self._get_default_slippage(symbol, order_type)
                
        except Exception as e:
            logger.error(f"Error estimating slippage for {symbol}: {e}")
            handle_error(e, f"Failed to estimate slippage for {symbol}")
            return self._get_default_slippage(symbol, order_type)
    
    def _get_default_slippage(self, symbol, order_type):
        """
        Get default slippage values based on order type and historical data.
        
        Args:
            symbol (str): Trading pair symbol
            order_type (str): Order type (Market or Limit)
            
        Returns:
            float: Default slippage as a percentage
        """
        # Check if we have historical slippage data for this symbol
        with self.lock:
            if symbol in self.slippage_history and self.slippage_history[symbol]:
                historical_slippage = [s['slippage'] for s in self.slippage_history[symbol]]
                # Use the 75th percentile for a more conservative estimate
                historical_estimate = np.percentile(historical_slippage, 75) if historical_slippage else 0.001
                
                # Add a safety margin
                if order_type.upper() == "MARKET":
                    return max(historical_estimate, 0.001)  # Minimum 0.1% slippage for market orders
                else:
                    return max(historical_estimate, 0.0)  # No minimum for limit orders
            
        # Default values if no historical data
        if order_type.upper() == "MARKET":
            return 0.001  # 0.1% for market orders
        else:
            return 0.0  # 0% for limit orders
    
    def record_slippage(self, symbol, order_type, expected_price, execution_price, order_size, timestamp=None):
        """
        Record actual slippage for a completed order.
        
        Args:
            symbol (str): Trading pair symbol
            order_type (str): Order type (Market or Limit)
            expected_price (float): Expected execution price
            execution_price (float): Actual execution price
            order_size (float): Order size
            timestamp (int): Execution timestamp (default: current time)
            
        Returns:
            float: Recorded slippage as a percentage
        """
        try:
            if timestamp is None:
                timestamp = int(datetime.now().timestamp() * 1000)
            
            # Calculate slippage as percentage
            if expected_price > 0:
                slippage = (execution_price - expected_price) / expected_price
            else:
                slippage = 0.0
            
            # Create slippage record
            slippage_record = {
                'timestamp': timestamp,
                'symbol': symbol,
                'order_type': order_type,
                'expected_price': expected_price,
                'execution_price': execution_price,
                'order_size': order_size,
                'slippage': slippage
            }
            
            # Store record
            with self.lock:
                if symbol not in self.slippage_history:
                    self.slippage_history[symbol] = []
                
                self.slippage_history[symbol].append(slippage_record)
                
                # Trim history if too large
                if len(self.slippage_history[symbol]) > self.max_history_size:
                    self.slippage_history[symbol] = self.slippage_history[symbol][-self.max_history_size:]
            
            # Log significant slippage
            if abs(slippage) > 0.01:  # More than 1%
                logger.warning(f"High slippage detected for {symbol}: {slippage:.4%}")
            
            # Save to datastore if available
            if self.datastore:
                self.datastore.save_slippage(slippage_record)
            
            return slippage
        except Exception as e:
            logger.error(f"Error recording slippage for {symbol}: {e}")
            handle_error(e, f"Failed to record slippage for {symbol}")
            return 0.0
    
    def get_slippage_stats(self, symbol=None, start_time=None, end_time=None):
        """
        Get slippage statistics.
        
        Args:
            symbol (str): Filter by symbol (optional)
            start_time (int): Filter by start time (optional)
            end_time (int): Filter by end time (optional)
            
        Returns:
            dict: Slippage statistics
        """
        try:
            with self.lock:
                # Filter by symbol if provided
                if symbol:
                    if symbol not in self.slippage_history:
                        return {
                            'count': 0,
                            'mean': 0.0,
                            'median': 0.0,
                            'std': 0.0,
                            'min': 0.0,
                            'max': 0.0,
                            'p90': 0.0
                        }
                    
                    history = self.slippage_history[symbol]
                else:
                    # Combine all histories
                    history = []
                    for sym_history in self.slippage_history.values():
                        history.extend(sym_history)
                
                # Filter by time if provided
                if start_time:
                    history = [h for h in history if h['timestamp'] >= start_time]
                
                if end_time:
                    history = [h for h in history if h['timestamp'] <= end_time]
                
                if not history:
                    return {
                        'count': 0,
                        'mean': 0.0,
                        'median': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'p90': 0.0
                    }
                
                # Calculate statistics
                slippage_values = [h['slippage'] for h in history]
                
                return {
                    'count': len(slippage_values),
                    'mean': np.mean(slippage_values),
                    'median': np.median(slippage_values),
                    'std': np.std(slippage_values),
                    'min': np.min(slippage_values),
                    'max': np.max(slippage_values),
                    'p90': np.percentile(slippage_values, 90)
                }
        except Exception as e:
            logger.error(f"Error getting slippage stats: {e}")
            handle_error(e, "Failed to get slippage stats")
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p90': 0.0
            }
    
    def adjust_order_price(self, symbol, base_price, order_type, side, order_size):
        """
        Adjust order price to account for expected slippage.
        
        Args:
            symbol (str): Trading pair symbol
            base_price (float): Base price to adjust
            order_type (str): Order type (Market or Limit)
            side (str): Order side (Buy or Sell)
            order_size (float): Order size
            
        Returns:
            float: Adjusted price
        """
        try:
            # If it's a market order, we don't need to adjust the price
            if order_type.upper() == "MARKET":
                return base_price
            
            # Estimate slippage
            expected_slippage = self.estimate_slippage(symbol, order_type, order_size)
            
            # Adjust price based on side
            if side.upper() == "BUY":
                # For buy limit orders, adjust price down to increase chance of execution
                adjustment_factor = -0.0005  # 0.05% below the base price
            else:
                # For sell limit orders, adjust price up to increase chance of execution
                adjustment_factor = 0.0005  # 0.05% above the base price
            
            # Apply adjustment
            adjusted_price = base_price * (1 + adjustment_factor)
            
            logger.debug(f"Adjusted {side} {order_type} price for {symbol} from {base_price} to {adjusted_price}")
            return adjusted_price
        except Exception as e:
            logger.error(f"Error adjusting order price for {symbol}: {e}")
            handle_error(e, f"Failed to adjust order price for {symbol}")
            return base_price
    
    def mitigate_slippage(self, symbol, order_size, side):
        """
        Recommend strategies to mitigate slippage.
        
        Args:
            symbol (str): Trading pair symbol
            order_size (float): Order size
            side (str): Order side (Buy or Sell)
            
        Returns:
            dict: Slippage mitigation recommendations
        """
        try:
            # Get order book to assess liquidity
            order_book = self.exchange_api.get_order_book(symbol, depth=20)
            
            if not order_book or not order_book['bids'] or not order_book['asks']:
                return {'recommendation': 'USE_LIMIT_ORDERS', 'details': 'Order book data not available'}
            
            # Calculate liquidity on the relevant side
            if side.upper() == "BUY":
                liquidity = sum(float(ask[1]) for ask in order_book['asks'])
            else:
                liquidity = sum(float(bid[1]) for bid in order_book['bids'])
            
            # Order size as percentage of available liquidity
            size_to_liquidity = order_size / liquidity if liquidity > 0 else float('inf')
            
            # Make recommendations based on order size and liquidity
            if size_to_liquidity > 0.2:  # Order size > 20% of available liquidity
                return {
                    'recommendation': 'SPLIT_ORDER',
                    'details': 'Order size is large relative to available liquidity. Split into multiple smaller orders.',
                    'suggested_splits': 3,
                    'time_interval': 60  # seconds between orders
                }
            elif size_to_liquidity > 0.1:  # Order size > 10% of available liquidity
                return {
                    'recommendation': 'TWAP',
                    'details': 'Use Time-Weighted Average Price (TWAP) to execute over time.',
                    'suggested_duration': 300,  # 5 minutes
                    'interval': 30  # seconds
                }
            elif TRADING_CONFIG.get("use_limit_orders", False):
                return {
                    'recommendation': 'LIMIT_ORDER',
                    'details': 'Use limit orders to avoid paying the spread.',
                    'price_adjustment': -0.0005 if side.upper() == "BUY" else 0.0005
                }
            else:
                return {
                    'recommendation': 'MARKET_ORDER',
                    'details': 'Market has sufficient liquidity for a market order.'
                }
        except Exception as e:
            logger.error(f"Error creating slippage mitigation recommendation for {symbol}: {e}")
            handle_error(e, f"Failed to create slippage mitigation recommendation for {symbol}")
            return {
                'recommendation': 'USE_LIMIT_ORDERS',
                'details': 'Error assessing liquidity'
            }
    
    def clear_history(self, symbol=None, days_to_keep=7):
        """
        Clear slippage history, optionally keeping recent data.
        
        Args:
            symbol (str): Symbol to clear history for (None for all)
            days_to_keep (int): Number of days of history to keep
            
        Returns:
            int: Number of records removed
        """
        try:
            removed_count = 0
            
            with self.lock:
                # Calculate cutoff timestamp
                cutoff_time = (datetime.now() - timedelta(days=days_to_keep)).timestamp() * 1000
                
                if symbol:
                    # Clear history for specific symbol
                    if symbol in self.slippage_history:
                        original_count = len(self.slippage_history[symbol])
                        self.slippage_history[symbol] = [
                            h for h in self.slippage_history[symbol] 
                            if h['timestamp'] >= cutoff_time
                        ]
                        removed_count = original_count - len(self.slippage_history[symbol])
                else:
                    # Clear history for all symbols
                    for sym in self.slippage_history:
                        original_count = len(self.slippage_history[sym])
                        self.slippage_history[sym] = [
                            h for h in self.slippage_history[sym] 
                            if h['timestamp'] >= cutoff_time
                        ]
                        removed_count += original_count - len(self.slippage_history[sym])
            
            logger.info(f"Cleared {removed_count} slippage history records")
            return removed_count
        except Exception as e:
            logger.error(f"Error clearing slippage history: {e}")
            handle_error(e, "Failed to clear slippage history")
            return 0