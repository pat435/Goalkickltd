"""
Account management module for the Goalkick Ltd Trading Bot.
Handles account balance, positions, and margin calculations.
"""

import time
from datetime import datetime
import threading

from config.logging_config import get_logger
from config.bot_config import TRADING_CONFIG, RISK_CONFIG
from src.utils.error_handling import handle_error

logger = get_logger("account_manager")

class AccountManager:
    """Class for managing account balance, positions, and risk."""
    
    def __init__(self, exchange_api):
        """
        Initialize the AccountManager.
        
        Args:
            exchange_api: Exchange API instance
        """
        self.exchange_api = exchange_api
        self.account_info = None
        self.positions = []
        self.open_orders = []
        self.balance_history = []
        self.last_update_time = 0
        self.lock = threading.RLock()
    
    def get_account_info(self, force_update=False):
        """
        Get account information including balance.
        
        Args:
            force_update (bool): Whether to force an update from the API
            
        Returns:
            dict: Account information
        """
        current_time = time.time()
        update_interval = 10  # Update every 10 seconds maximum
        
        if self.account_info is None or force_update or (current_time - self.last_update_time) > update_interval:
            try:
                with self.lock:
                    # Update account info
                    self.account_info = self.exchange_api.get_account_info()
                    
                    # Get open positions
                    self.positions = self.exchange_api.get_positions()
                    
                    # Add positions to account info
                    self.account_info['positions'] = self.positions
                    
                    # Get open orders
                    self.open_orders = self.exchange_api.get_open_orders()
                    
                    # Add open orders to account info
                    self.account_info['open_orders'] = self.open_orders
                    
                    # Track balance history
                    self.balance_history.append({
                        'timestamp': datetime.now().timestamp(),
                        'balance': self.account_info['balance'],
                        'equity': self.account_info.get('total_equity', self.account_info['balance'])
                    })
                    
                    # Keep only the last 1000 balance history points
                    if len(self.balance_history) > 1000:
                        self.balance_history = self.balance_history[-1000:]
                    
                    self.last_update_time = current_time
                    
                    logger.debug(f"Updated account info. Balance: {self.account_info['balance']} USDT")
            except Exception as e:
                logger.error(f"Failed to update account info: {e}")
                handle_error(e, "Failed to update account info")
                
                # If no account info yet, create a dummy one
                if self.account_info is None:
                    self.account_info = {
                        "balance": 0,
                        "available": 0,
                        "positions": [],
                        "open_orders": []
                    }
        
        return self.account_info
    
    def get_position(self, symbol):
        """
        Get position for a specific symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Position information or None if no position
        """
        # Update account info to get latest positions
        self.get_account_info()
        
        # Find position for the symbol
        for position in self.positions:
            if position['symbol'] == symbol:
                return position
        
        return None
    
    def get_all_positions(self):
        """
        Get all open positions.
        
        Returns:
            list: List of position dictionaries
        """
        # Update account info to get latest positions
        self.get_account_info()
        
        return self.positions
    
    def get_balance(self):
        """
        Get account balance.
        
        Returns:
            float: Account balance in USDT
        """
        account_info = self.get_account_info()
        return account_info['balance']
    
    def get_available_balance(self):
        """
        Get available balance for trading.
        
        Returns:
            float: Available balance in USDT
        """
        account_info = self.get_account_info()
        return account_info['available']
    
    def get_total_equity(self):
        """
        Get total account equity including unrealized PnL.
        
        Returns:
            float: Total equity in USDT
        """
        account_info = self.get_account_info()
        return account_info.get('total_equity', account_info['balance'])
    
    def calculate_position_size(self, symbol, price, risk_pct=None):
        """
        Calculate position size based on risk percentage and price.
        
        Args:
            symbol (str): Trading pair symbol
            price (float): Entry price
            risk_pct (float): Risk percentage (default from config)
            
        Returns:
            float: Position size in base currency
        """
        if risk_pct is None:
            risk_pct = TRADING_CONFIG["risk_per_trade"]
        
        try:
            # Get account balance
            balance = self.get_balance()
            
            # Calculate position value in USDT
            risk_amount = balance * risk_pct
            
            # Calculate quantity in base currency
            quantity = risk_amount / price
            
            # Round to appropriate precision
            symbol_info = self.exchange_api.get_symbol_info(symbol)
            if symbol_info and 'lotSizeFilter' in symbol_info:
                qty_step = float(symbol_info['lotSizeFilter'].get('qtyStep', 0.001))
                min_qty = float(symbol_info['lotSizeFilter'].get('minOrderQty', 0.001))
                
                # Round down to nearest step
                quantity = int(quantity / qty_step) * qty_step
                
                # Ensure minimum quantity
                quantity = max(quantity, min_qty)
            
            logger.debug(f"Calculated position size for {symbol}: {quantity} (risk: {risk_pct:.2%}, price: {price:.2f})")
            return quantity
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            handle_error(e, f"Failed to calculate position size for {symbol}")
            return 0
    
    def calculate_max_position_size(self, symbol, price):
        """
        Calculate maximum position size based on available balance.
        
        Args:
            symbol (str): Trading pair symbol
            price (float): Entry price
            
        Returns:
            float: Maximum position size in base currency
        """
        try:
            # Get available balance
            available = self.get_available_balance()
            
            # Calculate maximum quantity in base currency
            max_quantity = available / price
            
            # Round to appropriate precision
            symbol_info = self.exchange_api.get_symbol_info(symbol)
            if symbol_info and 'lotSizeFilter' in symbol_info:
                qty_step = float(symbol_info['lotSizeFilter'].get('qtyStep', 0.001))
                
                # Round down to nearest step
                max_quantity = int(max_quantity / qty_step) * qty_step
            
            logger.debug(f"Calculated max position size for {symbol}: {max_quantity}")
            return max_quantity
        except Exception as e:
            logger.error(f"Error calculating max position size for {symbol}: {e}")
            handle_error(e, f"Failed to calculate max position size for {symbol}")
            return 0
    
    def calculate_stop_loss_price(self, symbol, entry_price, side, pct=None):
        """
        Calculate stop loss price based on entry price and percentage.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            side (str): Position side ("Buy" or "Sell")
            pct (float): Stop loss percentage (default from config)
            
        Returns:
            float: Stop loss price
        """
        if pct is None:
            pct = RISK_CONFIG["stop_loss_pct"] / 100.0
        
        try:
            if side.upper() == "BUY":
                stop_price = entry_price * (1 - pct)
            else:
                stop_price = entry_price * (1 + pct)
            
            # Round to appropriate precision
            symbol_info = self.exchange_api.get_symbol_info(symbol)
            if symbol_info and 'priceFilter' in symbol_info:
                tick_size = float(symbol_info['priceFilter'].get('tickSize', 0.01))
                
                # Round to nearest tick
                stop_price = round(stop_price / tick_size) * tick_size
            
            logger.debug(f"Calculated stop loss for {symbol} {side}: {stop_price} (entry: {entry_price}, pct: {pct:.2%})")
            return stop_price
        except Exception as e:
            logger.error(f"Error calculating stop loss price for {symbol}: {e}")
            handle_error(e, f"Failed to calculate stop loss price for {symbol}")
            return 0
    
    def calculate_take_profit_price(self, symbol, entry_price, side, risk_reward=None):
        """
        Calculate take profit price based on entry price and risk/reward ratio.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            side (str): Position side ("Buy" or "Sell")
            risk_reward (float): Risk/reward ratio (default from config)
            
        Returns:
            float: Take profit price
        """
        if risk_reward is None:
            risk_reward = RISK_CONFIG["risk_reward_ratio"]
        
        try:
            # Get stop loss percentage
            stop_loss_pct = RISK_CONFIG["stop_loss_pct"] / 100.0
            
            # Calculate take profit percentage based on risk/reward ratio
            take_profit_pct = stop_loss_pct * risk_reward
            
            if side.upper() == "BUY":
                take_profit_price = entry_price * (1 + take_profit_pct)
            else:
                take_profit_price = entry_price * (1 - take_profit_pct)
            
            # Round to appropriate precision
            symbol_info = self.exchange_api.get_symbol_info(symbol)
            if symbol_info and 'priceFilter' in symbol_info:
                tick_size = float(symbol_info['priceFilter'].get('tickSize', 0.01))
                
                # Round to nearest tick
                take_profit_price = round(take_profit_price / tick_size) * tick_size
            
            logger.debug(f"Calculated take profit for {symbol} {side}: {take_profit_price} (entry: {entry_price}, R/R: {risk_reward})")
            return take_profit_price
        except Exception as e:
            logger.error(f"Error calculating take profit price for {symbol}: {e}")
            handle_error(e, f"Failed to calculate take profit price for {symbol}")
            return 0
    
    def calculate_position_exposure(self, symbol):
        """
        Calculate current exposure for a symbol as percentage of portfolio.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            float: Position exposure as percentage (0-1)
        """
        try:
            position = self.get_position(symbol)
            if not position:
                return 0
            
            # Get total equity
            total_equity = self.get_total_equity()
            if total_equity <= 0:
                return 0
            
            # Calculate position value
            position_value = abs(position['position_value'])
            
            # Calculate exposure as percentage
            exposure = position_value / total_equity
            
            # Calculate exposure as percentage
            exposure = position_value / total_equity
            
            logger.debug(f"Position exposure for {symbol}: {exposure:.2%}")
            return exposure
        except Exception as e:
            logger.error(f"Error calculating position exposure for {symbol}: {e}")
            handle_error(e, f"Failed to calculate position exposure for {symbol}")
            return 0
    
    def calculate_total_exposure(self):
        """
        Calculate total exposure across all positions.
        
        Returns:
            float: Total exposure as percentage (0-1)
        """
        try:
            # Get all positions
            positions = self.get_all_positions()
            if not positions:
                return 0
            
            # Get total equity
            total_equity = self.get_total_equity()
            if total_equity <= 0:
                return 0
            
            # Calculate total position value
            total_position_value = sum(abs(pos['position_value']) for pos in positions)
            
            # Calculate total exposure as percentage
            total_exposure = total_position_value / total_equity
            
            logger.debug(f"Total position exposure: {total_exposure:.2%}")
            return total_exposure
        except Exception as e:
            logger.error(f"Error calculating total exposure: {e}")
            handle_error(e, f"Failed to calculate total exposure")
            return 0
    
    def is_within_risk_limits(self, symbol, additional_exposure=0):
        """
        Check if adding a position would stay within risk limits.
        
        Args:
            symbol (str): Trading pair symbol
            additional_exposure (float): Additional exposure to consider
            
        Returns:
            bool: True if within limits, False otherwise
        """
        try:
            # Get current total exposure
            current_exposure = self.calculate_total_exposure()
            
            # Calculate new total exposure
            new_exposure = current_exposure + additional_exposure
            
            # Check against max drawdown percentage
            max_exposure = RISK_CONFIG["max_drawdown_pct"] / 100.0
            
            # Check if within limits
            within_limits = new_exposure <= max_exposure
            
            logger.debug(f"Risk check for {symbol}: current={current_exposure:.2%}, new={new_exposure:.2%}, max={max_exposure:.2%}, within_limits={within_limits}")
            return within_limits
        except Exception as e:
            logger.error(f"Error checking risk limits for {symbol}: {e}")
            handle_error(e, f"Failed to check risk limits for {symbol}")
            return False
    
    def get_position_count(self):
        """
        Get the number of open positions.
        
        Returns:
            int: Number of open positions
        """
        positions = self.get_all_positions()
        return len(positions)
    
    def get_realized_pnl(self, timeframe="day"):
        """
        Get realized profit and loss for a specific timeframe.
        
        Args:
            timeframe (str): Time period ("day", "week", "month", "all")
            
        Returns:
            float: Realized PnL in USDT
        """
        try:
            # For the simple implementation, we'll just return 0
            # In a real implementation, this would query the exchange API
            # to get the realized PnL for the specified timeframe
            logger.debug(f"Realized PnL calculation for {timeframe} not implemented")
            return 0
        except Exception as e:
            logger.error(f"Error getting realized PnL: {e}")
            handle_error(e, "Failed to get realized PnL")
            return 0
    
    def get_unrealized_pnl(self):
        """
        Get unrealized profit and loss across all open positions.
        
        Returns:
            float: Unrealized PnL in USDT
        """
        try:
            positions = self.get_all_positions()
            total_pnl = sum(pos['unrealized_pnl'] for pos in positions)
            
            logger.debug(f"Total unrealized PnL: {total_pnl}")
            return total_pnl
        except Exception as e:
            logger.error(f"Error getting unrealized PnL: {e}")
            handle_error(e, "Failed to get unrealized PnL")
            return 0
    
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
            return self.exchange_api.set_leverage(symbol, leverage)
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            handle_error(e, f"Failed to set leverage for {symbol}")
            return False