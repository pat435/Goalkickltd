"""
Stop loss module for the Goalkick Ltd Trading Bot.
Implements various stop loss strategies and manages stop loss orders.
"""

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.bot_config import RISK_CONFIG
from src.utils.error_handling import handle_error

logger = get_logger("risk.stop_loss")

class StopLossManager:
    """Class for managing stop loss strategies and orders."""
    
    def __init__(self, order_manager, account_manager):
        """
        Initialize the StopLossManager.
        
        Args:
            order_manager: Order Manager instance
            account_manager: Account Manager instance
        """
        self.order_manager = order_manager
        self.account_manager = account_manager
        self.active_stops = {}  # symbol -> {stop_type, price, order_id, trailing_offset, params}
    
    def set_fixed_stop_loss(self, symbol, entry_price, position_side, stop_percentage=None):
        """
        Calculate fixed percentage stop loss price.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            position_side (str): Position side ("LONG" or "SHORT")
            stop_percentage (float): Stop loss percentage (default from config)
            
        Returns:
            float: Stop loss price
        """
        try:
            if stop_percentage is None:
                stop_percentage = RISK_CONFIG["stop_loss_pct"] / 100.0
            
            if position_side.upper() == "LONG":
                stop_price = entry_price * (1 - stop_percentage)
            else:
                stop_price = entry_price * (1 + stop_percentage)
            
            # Round to the appropriate price precision
            symbol_info = self.account_manager.exchange_api.get_symbol_info(symbol)
            if symbol_info and 'priceFilter' in symbol_info:
                tick_size = float(symbol_info['priceFilter'].get('tickSize', 0.01))
                precision = len(str(tick_size).split('.')[-1])
                stop_price = round(stop_price, precision)
            
            logger.debug(f"Set fixed stop loss for {symbol} at {stop_price} ({stop_percentage:.2%} from {entry_price})")
            
            # Store stop loss info
            self.active_stops[symbol] = {
                'stop_type': 'fixed',
                'price': stop_price,
                'entry_price': entry_price,
                'position_side': position_side,
                'percentage': stop_percentage,
                'order_id': None  # Will be set when order is placed
            }
            
            return stop_price
        except Exception as e:
            logger.error(f"Error setting fixed stop loss for {symbol}: {e}")
            handle_error(e, f"Failed to set fixed stop loss for {symbol}")
            return 0
    
    def set_atr_stop_loss(self, symbol, entry_price, position_side, atr_value, multiplier=None):
        """
        Calculate ATR-based stop loss price.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            position_side (str): Position side ("LONG" or "SHORT")
            atr_value (float): ATR value
            multiplier (float): ATR multiplier (default from config)
            
        Returns:
            float: Stop loss price
        """
        try:
            if multiplier is None:
                multiplier = RISK_CONFIG["volatility_multiplier"]
            
            # Calculate ATR distance
            atr_distance = atr_value * multiplier
            
            if position_side.upper() == "LONG":
                stop_price = entry_price - atr_distance
            else:
                stop_price = entry_price + atr_distance
            
            # Round to the appropriate price precision
            symbol_info = self.account_manager.exchange_api.get_symbol_info(symbol)
            if symbol_info and 'priceFilter' in symbol_info:
                tick_size = float(symbol_info['priceFilter'].get('tickSize', 0.01))
                precision = len(str(tick_size).split('.')[-1])
                stop_price = round(stop_price, precision)
            
            # Calculate percentage for reference
            stop_percentage = abs(stop_price - entry_price) / entry_price
            
            logger.debug(f"Set ATR stop loss for {symbol} at {stop_price} (ATR: {atr_value:.4f}, multiplier: {multiplier})")
            
            # Store stop loss info
            self.active_stops[symbol] = {
                'stop_type': 'atr',
                'price': stop_price,
                'entry_price': entry_price,
                'position_side': position_side,
                'atr_value': atr_value,
                'multiplier': multiplier,
                'percentage': stop_percentage,
                'order_id': None  # Will be set when order is placed
            }
            
            return stop_price
        except Exception as e:
            logger.error(f"Error setting ATR stop loss for {symbol}: {e}")
            handle_error(e, f"Failed to set ATR stop loss for {symbol}")
            return 0
    
    def set_trailing_stop_loss(self, symbol, entry_price, position_side, trail_percentage=None):
        """
        Set up a trailing stop loss.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            position_side (str): Position side ("LONG" or "SHORT")
            trail_percentage (float): Trailing percentage (default from config)
            
        Returns:
            float: Initial stop loss price
        """
        try:
            if trail_percentage is None:
                trail_percentage = RISK_CONFIG["trailing_stop_pct"] / 100.0
            
            # Calculate initial stop price
            if position_side.upper() == "LONG":
                stop_price = entry_price * (1 - trail_percentage)
                activation_price = entry_price * 1.01  # 1% above entry
            else:
                stop_price = entry_price * (1 + trail_percentage)
                activation_price = entry_price * 0.99  # 1% below entry
            
            # Round to the appropriate price precision
            symbol_info = self.account_manager.exchange_api.get_symbol_info(symbol)
            if symbol_info and 'priceFilter' in symbol_info:
                tick_size = float(symbol_info['priceFilter'].get('tickSize', 0.01))
                precision = len(str(tick_size).split('.')[-1])
                stop_price = round(stop_price, precision)
                activation_price = round(activation_price, precision)
            
            logger.debug(f"Set trailing stop loss for {symbol} at {stop_price} (trail: {trail_percentage:.2%})")
            
            # Store stop loss info
            self.active_stops[symbol] = {
                'stop_type': 'trailing',
                'price': stop_price,
                'entry_price': entry_price,
                'position_side': position_side,
                'trail_percentage': trail_percentage,
                'activation_price': activation_price,
                'highest_price': entry_price if position_side.upper() == "LONG" else float('inf'),
                'lowest_price': entry_price if position_side.upper() == "SHORT" else 0,
                'activated': False,
                'order_id': None  # Will be set when order is placed
            }
            
            return stop_price
        except Exception as e:
            logger.error(f"Error setting trailing stop loss for {symbol}: {e}")
            handle_error(e, f"Failed to set trailing stop loss for {symbol}")
            return 0
    
    def set_chandelier_exit(self, symbol, entry_price, position_side, atr_value, periods=22, multiplier=3.0):
        """
        Set a Chandelier Exit stop loss (ATR-based trailing stop).
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            position_side (str): Position side ("LONG" or "SHORT")
            atr_value (float): ATR value
            periods (int): Number of periods for highest/lowest
            multiplier (float): ATR multiplier
            
        Returns:
            float: Initial stop loss price
        """
        try:
            # Calculate ATR distance
            atr_distance = atr_value * multiplier
            
            # Calculate initial stop price
            if position_side.upper() == "LONG":
                stop_price = entry_price - atr_distance
            else:
                stop_price = entry_price + atr_distance
            
            # Round to the appropriate price precision
            symbol_info = self.account_manager.exchange_api.get_symbol_info(symbol)
            if symbol_info and 'priceFilter' in symbol_info:
                tick_size = float(symbol_info['priceFilter'].get('tickSize', 0.01))
                precision = len(str(tick_size).split('.')[-1])
                stop_price = round(stop_price, precision)
            
            logger.debug(f"Set Chandelier Exit for {symbol} at {stop_price} (ATR: {atr_value:.4f}, multiplier: {multiplier})")
            
            # Store stop loss info
            self.active_stops[symbol] = {
                'stop_type': 'chandelier',
                'price': stop_price,
                'entry_price': entry_price,
                'position_side': position_side,
                'atr_value': atr_value,
                'multiplier': multiplier,
                'periods': periods,
                'highest_price': entry_price if position_side.upper() == "LONG" else float('inf'),
                'lowest_price': entry_price if position_side.upper() == "SHORT" else 0,
                'price_history': [],
                'order_id': None  # Will be set when order is placed
            }
            
            return stop_price
        except Exception as e:
            logger.error(f"Error setting Chandelier Exit for {symbol}: {e}")
            handle_error(e, f"Failed to set Chandelier Exit for {symbol}")
            return 0
    
    def set_support_resistance_stop(self, symbol, entry_price, position_side, support_level=None, resistance_level=None):
        """
        Set a stop loss based on support/resistance levels.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            position_side (str): Position side ("LONG" or "SHORT")
            support_level (float): Support level
            resistance_level (float): Resistance level
            
        Returns:
            float: Stop loss price
        """
        try:
            # Determine stop price based on position side
            if position_side.upper() == "LONG":
                if support_level is None:
                    # Use default percentage if no support level provided
                    return self.set_fixed_stop_loss(symbol, entry_price, position_side)
                stop_price = support_level
            else:
                if resistance_level is None:
                    # Use default percentage if no resistance level provided
                    return self.set_fixed_stop_loss(symbol, entry_price, position_side)
                stop_price = resistance_level
            
            # Round to the appropriate price precision
            symbol_info = self.account_manager.exchange_api.get_symbol_info(symbol)
            if symbol_info and 'priceFilter' in symbol_info:
                tick_size = float(symbol_info['priceFilter'].get('tickSize', 0.01))
                precision = len(str(tick_size).split('.')[-1])
                stop_price = round(stop_price, precision)
            
            # Calculate percentage for reference
            stop_percentage = abs(stop_price - entry_price) / entry_price
            
            logger.debug(f"Set S/R stop loss for {symbol} at {stop_price} ({stop_percentage:.2%} from {entry_price})")
            
            # Store stop loss info
            self.active_stops[symbol] = {
                'stop_type': 'support_resistance',
                'price': stop_price,
                'entry_price': entry_price,
                'position_side': position_side,
                'percentage': stop_percentage,
                'order_id': None  # Will be set when order is placed
            }
            
            return stop_price
        except Exception as e:
            logger.error(f"Error setting S/R stop loss for {symbol}: {e}")
            handle_error(e, f"Failed to set S/R stop loss for {symbol}")
            return 0
    
    def place_stop_loss_order(self, symbol, quantity, stop_price=None):
        """
        Place a stop loss order on the exchange.
        
        Args:
            symbol (str): Trading pair symbol
            quantity (float): Order quantity
            stop_price (float): Stop price (if None, use active stop)
            
        Returns:
            dict: Order information
        """
        try:
            if symbol not in self.active_stops:
                logger.warning(f"No active stop loss for {symbol}")
                return None
            
            stop_info = self.active_stops[symbol]
            
            if stop_price is None:
                stop_price = stop_info['price']
            
            # Determine order side (opposite of position)
            order_side = "Sell" if stop_info['position_side'].upper() == "LONG" else "Buy"
            
            # Place the stop loss order
            logger.info(f"Placing stop loss order for {symbol} at {stop_price}")
            order = self.order_manager.create_stop_loss_order(
                symbol=symbol,
                side=order_side,
                quantity=quantity,
                stop_price=stop_price,
                base_price=stop_info['entry_price']
            )
            
            # Update stop loss info with order ID
            if order:
                stop_info['order_id'] = order['order_id']
                
            return order
        except Exception as e:
            logger.error(f"Error placing stop loss order for {symbol}: {e}")
            handle_error(e, f"Failed to place stop loss order for {symbol}")
            return None
    
    def update_trailing_stop(self, symbol, current_price):
        """
        Update trailing stop loss based on current price.
        
        Args:
            symbol (str): Trading pair symbol
            current_price (float): Current market price
            
        Returns:
            float: New stop loss price or None if not updated
        """
        try:
            if symbol not in self.active_stops:
                return None
            
            stop_info = self.active_stops[symbol]
            
            if stop_info['stop_type'] not in ['trailing', 'chandelier']:
                return None
            
            # Determine if we need to update the stop
            update_stop = False
            new_stop_price = stop_info['price']
            
            if stop_info['stop_type'] == 'trailing':
                # Check if trailing stop is activated
                if not stop_info['activated']:
                    if (stop_info['position_side'].upper() == "LONG" and current_price >= stop_info['activation_price']) or \
                       (stop_info['position_side'].upper() == "SHORT" and current_price <= stop_info['activation_price']):
                        stop_info['activated'] = True
                        logger.debug(f"Trailing stop activated for {symbol} at {current_price}")
                
                # Update trailing stop if activated
                if stop_info['activated']:
                    if stop_info['position_side'].upper() == "LONG":
                        # Update highest price
                        if current_price > stop_info['highest_price']:
                            stop_info['highest_price'] = current_price
                            # Calculate new stop price
                            new_stop_price = current_price * (1 - stop_info['trail_percentage'])
                            update_stop = new_stop_price > stop_info['price']
                    else:
                        # Update lowest price
                        if current_price < stop_info['lowest_price']:
                            stop_info['lowest_price'] = current_price
                            # Calculate new stop price
                            new_stop_price = current_price * (1 + stop_info['trail_percentage'])
                            update_stop = new_stop_price < stop_info['price']
            
            elif stop_info['stop_type'] == 'chandelier':
                # Update price history
                stop_info['price_history'].append(current_price)
                if len(stop_info['price_history']) > stop_info['periods']:
                    stop_info['price_history'].pop(0)
                
                if stop_info['position_side'].upper() == "LONG":
                    # Calculate highest high
                    highest_high = max(stop_info['price_history']) if stop_info['price_history'] else stop_info['highest_price']
                    if highest_high > stop_info['highest_price']:
                        stop_info['highest_price'] = highest_high
                        # Calculate new stop price
                        new_stop_price = highest_high - (stop_info['atr_value'] * stop_info['multiplier'])
                        update_stop = new_stop_price > stop_info['price']
                else:
                    # Calculate lowest low
                    lowest_low = min(stop_info['price_history']) if stop_info['price_history'] else stop_info['lowest_price']
                    if lowest_low < stop_info['lowest_price']:
                        stop_info['lowest_price'] = lowest_low
                        # Calculate new stop price
                        new_stop_price = lowest_low + (stop_info['atr_value'] * stop_info['multiplier'])
                        update_stop = new_stop_price < stop_info['price']
            
            # Round to the appropriate price precision
            symbol_info = self.account_manager.exchange_api.get_symbol_info(symbol)
            if symbol_info and 'priceFilter' in symbol_info:
                tick_size = float(symbol_info['priceFilter'].get('tickSize', 0.01))
                precision = len(str(tick_size).split('.')[-1])
                new_stop_price = round(new_stop_price, precision)
            
            # Update stop price if needed
            if update_stop:
                logger.debug(f"Updated {stop_info['stop_type']} stop for {symbol} from {stop_info['price']} to {new_stop_price}")
                stop_info['price'] = new_stop_price
                
                # If there's an active order, update it
                if stop_info['order_id']:
                    # Cancel old order
                    self.order_manager.cancel_order(symbol, order_id=stop_info['order_id'])
                    
                    # Get position information to update quantity
                    position = self.account_manager.get_position(symbol)
                    if position:
                        # Place new order
                        order = self.place_stop_loss_order(symbol, position['size'], new_stop_price)
                        if order:
                            stop_info['order_id'] = order['order_id']
                
                return new_stop_price
            
            return None
        except Exception as e:
            logger.error(f"Error updating trailing stop for {symbol}: {e}")
            handle_error(e, f"Failed to update trailing stop for {symbol}")
            return None
    
    def check_stop_loss_hit(self, symbol, current_price):
        """
        Check if stop loss level has been hit.
        
        Args:
            symbol (str): Trading pair symbol
            current_price (float): Current market price
            
        Returns:
            bool: True if stop loss hit, False otherwise
        """
        try:
            if symbol not in self.active_stops:
                return False
            
            stop_info = self.active_stops[symbol]
            
            # Check if price has crossed the stop loss level
            if stop_info['position_side'].upper() == "LONG":
                if current_price <= stop_info['price']:
                    logger.info(f"Stop loss hit for {symbol} at {current_price} (stop: {stop_info['price']})")
                    return True
            else:
                if current_price >= stop_info['price']:
                    logger.info(f"Stop loss hit for {symbol} at {current_price} (stop: {stop_info['price']})")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking stop loss for {symbol}: {e}")
            handle_error(e, f"Failed to check stop loss for {symbol}")
            return False
    
    def remove_stop_loss(self, symbol):
        """
        Remove a stop loss for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            bool: True if removed, False otherwise
        """
        try:
            if symbol not in self.active_stops:
                return False
            
            stop_info = self.active_stops[symbol]
            
            # Cancel any active stop orders
            if stop_info.get('order_id'):
                try:
                    self.order_manager.cancel_order(symbol, order_id=stop_info['order_id'])
                except Exception as e:
                    logger.warning(f"Error cancelling stop loss order for {symbol}: {e}")
            
            # Remove stop loss info
            del self.active_stops[symbol]
            
            logger.debug(f"Removed stop loss for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error removing stop loss for {symbol}: {e}")
            handle_error(e, f"Failed to remove stop loss for {symbol}")
            return False
    
    def get_stop_loss_info(self, symbol):
        """
        Get stop loss information for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Stop loss information or None if not found
        """
        return self.active_stops.get(symbol)
    
    def update_all_trailing_stops(self, symbol_prices):
        """
        Update all trailing stops based on current prices.
        
        Args:
            symbol_prices (dict): Dictionary of symbol -> current price
            
        Returns:
            dict: Dictionary of updated stops
        """
        updated_stops = {}
        
        for symbol, price in symbol_prices.items():
            if symbol in self.active_stops:
                updated_price = self.update_trailing_stop(symbol, price)
                if updated_price:
                    updated_stops[symbol] = updated_price
        
        return updated_stops
    
    def check_all_stop_losses(self, symbol_prices):
        """
        Check all stop losses against current prices.
        
        Args:
            symbol_prices (dict): Dictionary of symbol -> current price
            
        Returns:
            list: List of symbols with triggered stop losses
        """
        triggered_stops = []
        
        for symbol, price in symbol_prices.items():
            if symbol in self.active_stops and self.check_stop_loss_hit(symbol, price):
                triggered_stops.append(symbol)
        
        return triggered_stops