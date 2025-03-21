"""
Order execution module for the Goalkick Ltd Trading Bot.
Handles the execution of trading signals into actual orders.
"""

import time
import threading
import uuid
from datetime import datetime

from config.logging_config import get_logger
from config.bot_config import TRADING_CONFIG
from src.utils.error_handling import handle_error, ExchangeError

logger = get_logger("execution.order_executor")

class OrderExecutor:
    """Class for executing trading signals as orders."""
    
    def __init__(self, order_manager, portfolio_manager):
        """
        Initialize the OrderExecutor.
        
        Args:
            order_manager: Order Manager instance
            portfolio_manager: Portfolio Manager instance
        """
        self.order_manager = order_manager
        self.portfolio_manager = portfolio_manager
        self.executed_signals = {}  # signal_id -> execution_info
        self.lock = threading.RLock()
        self.max_retries = 3
        self.retry_delay = 2  # seconds
    
    def execute_signal(self, signal):
        """
        Execute a trading signal by placing an order.
        
        Args:
            signal (dict): Signal dictionary with trading information
            
        Returns:
            dict: Execution information
        """
        try:
            signal_id = signal.get('id')
            symbol = signal.get('symbol')
            direction = signal.get('direction')
            
            logger.info(f"Executing {direction} signal for {symbol} (ID: {signal_id})")
            
            # Check if signal has already been executed
            with self.lock:
                if signal_id in self.executed_signals:
                    logger.warning(f"Signal {signal_id} already executed")
                    return self.executed_signals[signal_id]
            
            # Handle different signal types
            if direction == "ARBITRAGE":
                return self._execute_arbitrage_signal(signal)
            elif direction == "EXIT":
                return self._execute_exit_signal(signal)
            else:
                return self._execute_directional_signal(signal)
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            handle_error(e, "Failed to execute signal")
            return None
    
    def _execute_directional_signal(self, signal):
        """
        Execute a directional (BUY/SELL) trading signal.
        
        Args:
            signal (dict): Signal dictionary with trading information
            
        Returns:
            dict: Execution information
        """
        try:
            signal_id = signal.get('id')
            symbol = signal.get('symbol')
            direction = signal.get('direction')
            price = signal.get('price')
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            
            # Map signal direction to order side
            side = "Buy" if direction == "BUY" else "Sell"
            
            # Check if there is already a position
            position = self.portfolio_manager.get_position(symbol)
            if position:
                # Don't open a new position if one already exists in the same direction
                if (side == "Buy" and position['side'] == "LONG") or \
                   (side == "Sell" and position['side'] == "SHORT"):
                    logger.info(f"Position already exists for {symbol} in {direction} direction")
                    
                    execution_info = {
                        'signal_id': signal_id,
                        'symbol': symbol,
                        'direction': direction,
                        'status': 'IGNORED',
                        'reason': 'Position already exists',
                        'timestamp': int(time.time() * 1000)
                    }
                    
                    with self.lock:
                        self.executed_signals[signal_id] = execution_info
                    
                    return execution_info
            
            # Calculate position size
            position_size = self._calculate_position_size(symbol, price, stop_loss, direction)
            
            if position_size <= 0:
                logger.warning(f"Calculated position size for {symbol} is zero or negative")
                
                execution_info = {
                    'signal_id': signal_id,
                    'symbol': symbol,
                    'direction': direction,
                    'status': 'REJECTED',
                    'reason': 'Invalid position size',
                    'timestamp': int(time.time() * 1000)
                }
                
                with self.lock:
                    self.executed_signals[signal_id] = execution_info
                
                return execution_info
            
            # Place order
            for attempt in range(self.max_retries):
                try:
                    if TRADING_CONFIG.get("use_limit_orders", False):
                        # Place limit order
                        # Add a small buffer for limit orders to increase chance of execution
                        buffer = 0.001  # 0.1%
                        limit_price = price * (1 + buffer) if side == "Buy" else price * (1 - buffer)
                        
                        order = self.order_manager.create_limit_order(
                            symbol=symbol,
                            side=side,
                            quantity=position_size,
                            price=limit_price,
                            time_in_force="GTC"
                        )
                    else:
                        # Place market order
                        order = self.order_manager.create_market_order(
                            symbol=symbol,
                            side=side,
                            quantity=position_size
                        )
                    
                    # Place stop loss and take profit orders if provided
                    sl_order = None
                    tp_order = None
                    
                    if stop_loss:
                        sl_order = self._place_stop_loss(symbol, direction, position_size, stop_loss, price)
                    
                    if take_profit:
                        tp_order = self._place_take_profit(symbol, direction, position_size, take_profit, price)
                    
                    # Record execution
                    execution_info = {
                        'signal_id': signal_id,
                        'symbol': symbol,
                        'direction': direction,
                        'order_id': order['order_id'],
                        'position_size': position_size,
                        'price': price,
                        'status': 'EXECUTED',
                        'stop_loss': {
                            'price': stop_loss,
                            'order_id': sl_order['order_id'] if sl_order else None
                        } if stop_loss else None,
                        'take_profit': {
                            'price': take_profit,
                            'order_id': tp_order['order_id'] if tp_order else None
                        } if take_profit else None,
                        'timestamp': int(time.time() * 1000)
                    }
                    
                    # Open a trade in portfolio manager
                    position_side = "LONG" if direction == "BUY" else "SHORT"
                    
                    self.portfolio_manager.open_trade(
                        symbol=symbol,
                        side=position_side,
                        entry_price=price,
                        quantity=position_size,
                        strategy=signal.get('strategy', 'Unknown'),
                        timeframe=signal.get('timeframe', '1h'),
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    # Store execution info
                    with self.lock:
                        self.executed_signals[signal_id] = execution_info
                    
                    logger.info(f"Successfully executed {direction} signal for {symbol} with {position_size} units")
                    return execution_info
                
                except ExchangeError as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Retrying order execution for {symbol} after error: {e}")
                        time.sleep(self.retry_delay)
                    else:
                        raise
        
        except Exception as e:
            logger.error(f"Error executing directional signal for {symbol}: {e}")
            handle_error(e, f"Failed to execute directional signal for {symbol}")
            
            execution_info = {
                'signal_id': signal.get('id'),
                'symbol': symbol,
                'direction': direction,
                'status': 'FAILED',
                'reason': str(e),
                'timestamp': int(time.time() * 1000)
            }
            
            with self.lock:
                self.executed_signals[signal.get('id')] = execution_info
            
            return execution_info
    
    def _execute_arbitrage_signal(self, signal):
        """
        Execute an arbitrage trading signal.
        
        Args:
            signal (dict): Signal dictionary with arbitrage information
            
        Returns:
            dict: Execution information
        """
        try:
            signal_id = signal.get('id')
            symbol = signal.get('symbol')
            metadata = signal.get('metadata', {})
            
            # Get arbitrage details
            arb_type = metadata.get('type')
            
            if arb_type == 'triangular':
                return self._execute_triangular_arbitrage(signal)
            elif arb_type == 'statistical':
                # Execute statistical arbitrage (pairs trading)
                return self._execute_statistical_arbitrage(signal)
            else:
                logger.warning(f"Unknown arbitrage type: {arb_type}")
                
                execution_info = {
                    'signal_id': signal_id,
                    'symbol': symbol,
                    'status': 'REJECTED',
                    'reason': f'Unknown arbitrage type: {arb_type}',
                    'timestamp': int(time.time() * 1000)
                }
                
                with self.lock:
                    self.executed_signals[signal_id] = execution_info
                
                return execution_info
        
        except Exception as e:
            logger.error(f"Error executing arbitrage signal: {e}")
            handle_error(e, "Failed to execute arbitrage signal")
            
            execution_info = {
                'signal_id': signal.get('id'),
                'symbol': signal.get('symbol'),
                'status': 'FAILED',
                'reason': str(e),
                'timestamp': int(time.time() * 1000)
            }
            
            with self.lock:
                self.executed_signals[signal.get('id')] = execution_info
            
            return execution_info
    
    def _execute_triangular_arbitrage(self, signal):
        """
        Execute a triangular arbitrage.
        
        Args:
            signal (dict): Signal dictionary with triangular arbitrage information
            
        Returns:
            dict: Execution information
        """
        try:
            signal_id = signal.get('id')
            metadata = signal.get('metadata', {})
            
            pair1 = metadata.get('pair1')
            pair2 = metadata.get('pair2')
            pair3 = metadata.get('pair3')
            
            # Calculate position size (use a fixed percentage of available balance)
            available_balance = self.portfolio_manager.get_available_balance()
            position_value = available_balance * 0.1  # Use 10% of available balance
            
            # Execute the three legs of the arbitrage
            # This is a simplified implementation
            # In a real implementation, you would need to handle different base/quote currencies
            # and calculate the exact quantities for each leg
            
            # First leg
            order1 = self.order_manager.create_market_order(
                symbol=pair1,
                side="Buy",
                quantity=position_value / metadata.get('prices', {}).get(pair1, 1)
            )
            
            # Second leg
            order2 = self.order_manager.create_market_order(
                symbol=pair2,
                side="Sell",
                quantity=position_value / metadata.get('prices', {}).get(pair2, 1)
            )
            
            # Third leg
            order3 = self.order_manager.create_market_order(
                symbol=pair3,
                side="Buy",
                quantity=position_value / metadata.get('prices', {}).get(pair3, 1)
            )
            
            # Record execution
            execution_info = {
                'signal_id': signal_id,
                'type': 'triangular_arbitrage',
                'pairs': [pair1, pair2, pair3],
                'orders': [
                    order1['order_id'],
                    order2['order_id'],
                    order3['order_id']
                ],
                'status': 'EXECUTED',
                'timestamp': int(time.time() * 1000)
            }
            
            with self.lock:
                self.executed_signals[signal_id] = execution_info
            
            logger.info(f"Successfully executed triangular arbitrage: {pair1} → {pair2} → {pair3}")
            return execution_info
        
        except Exception as e:
            logger.error(f"Error executing triangular arbitrage: {e}")
            handle_error(e, "Failed to execute triangular arbitrage")
            
            execution_info = {
                'signal_id': signal.get('id'),
                'status': 'FAILED',
                'reason': str(e),
                'timestamp': int(time.time() * 1000)
            }
            
            with self.lock:
                self.executed_signals[signal.get('id')] = execution_info
            
            return execution_info
    
    def _execute_statistical_arbitrage(self, signal):
        """
        Execute a statistical arbitrage (pairs trading).
        
        Args:
            signal (dict): Signal dictionary with statistical arbitrage information
            
        Returns:
            dict: Execution information
        """
        try:
            signal_id = signal.get('id')
            symbol = signal.get('symbol')
            direction = signal.get('direction')
            metadata = signal.get('metadata', {})
            
            pair_symbol = metadata.get('pair_symbol')
            
            if not pair_symbol:
                logger.warning(f"Missing pair symbol for statistical arbitrage")
                
                execution_info = {
                    'signal_id': signal_id,
                    'symbol': symbol,
                    'status': 'REJECTED',
                    'reason': 'Missing pair symbol',
                    'timestamp': int(time.time() * 1000)
                }
                
                with self.lock:
                    self.executed_signals[signal_id] = execution_info
                
                return execution_info
            
            # Calculate position size for both legs
            available_balance = self.portfolio_manager.get_available_balance()
            position_value = available_balance * 0.05  # Use 5% of available balance for each leg
            
            # Direction mapping
            side1 = "Buy" if direction == "BUY" else "Sell"
            side2 = "Sell" if direction == "BUY" else "Buy"
            
            # Execute first leg
            order1 = self.order_manager.create_market_order(
                symbol=symbol,
                side=side1,
                quantity=position_value / signal.get('price', 1)
            )
            
            # Execute second leg
            order2 = self.order_manager.create_market_order(
                symbol=pair_symbol,
                side=side2,
                quantity=position_value / metadata.get('pair_price', 1)
            )
            
            # Record execution
            execution_info = {
                'signal_id': signal_id,
                'type': 'statistical_arbitrage',
                'symbols': [symbol, pair_symbol],
                'orders': [
                    order1['order_id'],
                    order2['order_id']
                ],
                'status': 'EXECUTED',
                'timestamp': int(time.time() * 1000)
            }
            
            with self.lock:
                self.executed_signals[signal_id] = execution_info
            
            logger.info(f"Successfully executed statistical arbitrage between {symbol} and {pair_symbol}")
            return execution_info
        
        except Exception as e:
            logger.error(f"Error executing statistical arbitrage: {e}")
            handle_error(e, "Failed to execute statistical arbitrage")
            
            execution_info = {
                'signal_id': signal.get('id'),
                'symbol': symbol,
                'status': 'FAILED',
                'reason': str(e),
                'timestamp': int(time.time() * 1000)
            }
            
            with self.lock:
                self.executed_signals[signal.get('id')] = execution_info
            
            return execution_info
    
    def _execute_exit_signal(self, signal):
        """
        Execute an exit signal to close a position.
        
        Args:
            signal (dict): Signal dictionary with exit information
            
        Returns:
            dict: Execution information
        """
        try:
            signal_id = signal.get('id')
            symbol = signal.get('symbol')
            
            # Check if there is a position to close
            position = self.portfolio_manager.get_position(symbol)
            
            if not position:
                logger.warning(f"No position to close for {symbol}")
                
                execution_info = {
                    'signal_id': signal_id,
                    'symbol': symbol,
                    'status': 'REJECTED',
                    'reason': 'No position to close',
                    'timestamp': int(time.time() * 1000)
                }
                
                with self.lock:
                    self.executed_signals[signal_id] = execution_info
                
                return execution_info
            
            # Get position details
            position_side = position['side']
            position_size = position['size']
            
            # Close the position
            result = self.order_manager.close_position(symbol, position_side, position_size)
            
            if not result:
                logger.error(f"Failed to close position for {symbol}")
                
                execution_info = {
                    'signal_id': signal_id,
                    'symbol': symbol,
                    'status': 'FAILED',
                    'reason': 'Failed to close position',
                    'timestamp': int(time.time() * 1000)
                }
                
                with self.lock:
                    self.executed_signals[signal_id] = execution_info
                
                return execution_info
            
            # Record execution
            execution_info = {
                'signal_id': signal_id,
                'symbol': symbol,
                'direction': 'EXIT',
                'order_id': result['order_id'],
                'position_side': position_side,
                'position_size': position_size,
                'status': 'EXECUTED',
                'timestamp': int(time.time() * 1000)
            }
            
            with self.lock:
                self.executed_signals[signal_id] = execution_info
            
            logger.info(f"Successfully closed {position_side} position for {symbol}")
            return execution_info
        
        except Exception as e:
            logger.error(f"Error executing exit signal for {signal.get('symbol')}: {e}")
            handle_error(e, f"Failed to execute exit signal for {signal.get('symbol')}")
            
            execution_info = {
                'signal_id': signal.get('id'),
                'symbol': signal.get('symbol'),
                'status': 'FAILED',
                'reason': str(e),
                'timestamp': int(time.time() * 1000)
            }
            
            with self.lock:
                self.executed_signals[signal.get('id')] = execution_info
            
            return execution_info
    
    def _calculate_position_size(self, symbol, price, stop_loss, direction):
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol (str): Trading pair symbol
            price (float): Entry price
            stop_loss (float): Stop loss price
            direction (str): Trade direction
            
        Returns:
            float: Position size
        """
        try:
            # If stop loss is provided, use risk-based position sizing
            if stop_loss and price:
                # For BUY orders, stop loss is below entry price
                # For SELL orders, stop loss is above entry price
                risk_per_unit = abs(price - stop_loss)
                
                # If zero risk (no stop loss), use a default risk percentage
                if risk_per_unit == 0:
                    return self.portfolio_manager.calculate_position_size(symbol, price)
                
                # Get risk amount based on account balance
                account_balance = self.portfolio_manager.get_balance()
                risk_amount = account_balance * TRADING_CONFIG["risk_per_trade"]
                
                # Calculate position size
                position_size = risk_amount / risk_per_unit
                
                # Round to appropriate precision
                return self._round_position_size(position_size, symbol)
            else:
                # Use default position sizing based on account balance
                return self.portfolio_manager.calculate_position_size(symbol, price)
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            handle_error(e, f"Failed to calculate position size for {symbol}")
            return 0
    
    def _round_position_size(self, position_size, symbol):
        """
        Round position size according to exchange requirements.
        
        Args:
            position_size (float): Raw position size
            symbol (str): Trading pair symbol
            
        Returns:
            float: Rounded position size
        """
        try:
            # Get symbol information for rounding precision
            symbol_info = self.order_manager.exchange_api.get_symbol_info(symbol)
            
            if symbol_info and 'lotSizeFilter' in symbol_info:
                qty_step = float(symbol_info['lotSizeFilter'].get('qtyStep', 0.001))
                min_qty = float(symbol_info['lotSizeFilter'].get('minOrderQty', 0.001))
                
                # Round down to nearest step
                position_size = int(position_size / qty_step) * qty_step
                
                # Ensure minimum quantity
                position_size = max(position_size, min_qty)
            
            return position_size
        except Exception as e:
            logger.error(f"Error rounding position size for {symbol}: {e}")
            handle_error(e, f"Failed to round position size for {symbol}")
            return position_size
    
    def _place_stop_loss(self, symbol, direction, quantity, stop_price, entry_price):
        """
        Place a stop loss order.
        
        Args:
            symbol (str): Trading pair symbol
            direction (str): Trade direction
            quantity (float): Position size
            stop_price (float): Stop loss price
            entry_price (float): Entry price
            
        Returns:
            dict: Order information
        """
        try:
            # Reverse the side for stop loss
            side = "Sell" if direction == "BUY" else "Buy"
            
            # Place stop loss order
            return self.order_manager.create_stop_loss_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                stop_price=stop_price,
                base_price=entry_price
            )
        except Exception as e:
            logger.error(f"Error placing stop loss for {symbol}: {e}")
            handle_error(e, f"Failed to place stop loss for {symbol}")
            return None
    
    def _place_take_profit(self, symbol, direction, quantity, take_profit_price, entry_price):
        """
        Place a take profit order.
        
        Args:
            symbol (str): Trading pair symbol
            direction (str): Trade direction
            quantity (float): Position size
            take_profit_price (float): Take profit price
            entry_price (float): Entry price
            
        Returns:
            dict: Order information
        """
        try:
            # Reverse the side for take profit
            side = "Sell" if direction == "BUY" else "Buy"
            
            # Place take profit order
            return self.order_manager.create_take_profit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                take_profit_price=take_profit_price,
                base_price=entry_price
            )
        except Exception as e:
            logger.error(f"Error placing take profit for {symbol}: {e}")
            handle_error(e, f"Failed to place take profit for {symbol}")
            return None
    
    def get_execution_info(self, signal_id):
        """
        Get execution information for a signal.
        
        Args:
            signal_id (str): Signal ID
            
        Returns:
            dict: Execution information or None if not found
        """
        with self.lock:
            return self.executed_signals.get(signal_id)
    
    def cancel_executions(self, symbol=None):
        """
        Cancel pending orders and executions.
        
        Args:
            symbol (str): Symbol to cancel orders for (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Cancelling pending executions" + (f" for {symbol}" if symbol else ""))
            
            # Cancel all open orders
            self.order_manager.cancel_all_orders(symbol)
            
            logger.info(f"All executions cancelled" + (f" for {symbol}" if symbol else ""))
            return True
        except Exception as e:
            logger.error(f"Error cancelling executions: {e}")
            handle_error(e, "Failed to cancel executions")
            return False