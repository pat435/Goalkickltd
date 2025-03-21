"""
Order management module for the Goalkick Ltd Trading Bot.
Handles order creation, modification, cancellation, and tracking.
"""

import time
import uuid
import threading
from datetime import datetime

from config.logging_config import get_logger
from config.bot_config import TRADING_CONFIG
from src.utils.error_handling import handle_error, ExchangeError

logger = get_logger("order_manager")

class OrderManager:
    """Class for managing orders."""
    
    def __init__(self, exchange_api, account_manager):
        """
        Initialize the OrderManager.
        
        Args:
            exchange_api: Exchange API instance
            account_manager: Account Manager instance
        """
        self.exchange_api = exchange_api
        self.account_manager = account_manager
        self.active_orders = {}  # order_id -> order_info
        self.order_history = {}  # order_id -> order_info
        self.lock = threading.RLock()
    
    def create_order(self, symbol, side, order_type, quantity, price=None, 
                    time_in_force="GTC", reduce_only=False, close_on_trigger=False, 
                    stop_loss=None, take_profit=None, client_order_id=None):
        """
        Create a new order.
        
        Args:
            symbol (str): Trading pair symbol
            side (str): Order side ("Buy" or "Sell")
            order_type (str): Order type ("Limit" or "Market")
            quantity (float): Order quantity in base currency
            price (float): Order price (required for Limit orders)
            time_in_force (str): Time in force ("GTC", "IOC", "FOK")
            reduce_only (bool): Whether to reduce position only
            close_on_trigger (bool): Close on trigger
            stop_loss (float): Stop loss price
            take_profit (float): Take profit price
            client_order_id (str): Client order ID
            
        Returns:
            dict: Order information
        """
        try:
            # Validate inputs
            if order_type.upper() == "LIMIT" and price is None:
                raise ValueError("Price is required for Limit orders")
            
            # Generate client order ID if not provided
            if not client_order_id:
                client_order_id = f"goalkick_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Prepare order parameters
            order_params = {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "qty": quantity,
                "client_order_id": client_order_id,
                "time_in_force": time_in_force,
                "reduce_only": reduce_only,
                "close_on_trigger": close_on_trigger
            }
            
            # Add price for limit orders
            if order_type.upper() == "LIMIT" and price is not None:
                order_params["price"] = price
            
            # Add stop loss and take profit if provided
            if stop_loss is not None:
                order_params["stop_loss"] = stop_loss
            
            if take_profit is not None:
                order_params["take_profit"] = take_profit
            
            # Place the order
            logger.info(f"Creating {order_type} {side} order for {symbol}: {quantity} @ {price if price else 'MARKET'}")
            order = self.exchange_api.place_order(order_params)
            
            # Save the order in active orders
            with self.lock:
                self.active_orders[order["order_id"]] = order
            
            logger.info(f"Order created: {order['order_id']}")
            return order
        except Exception as e:
            logger.error(f"Error creating order for {symbol}: {e}")
            handle_error(e, f"Failed to create order for {symbol}")
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
            if not order_id and not client_order_id:
                raise ValueError("Either order_id or client_order_id must be provided")
            
            # Look up order_id if only client_order_id is provided
            if not order_id and client_order_id:
                with self.lock:
                    for oid, order in self.active_orders.items():
                        if order.get("client_order_id") == client_order_id:
                            order_id = oid
                            break
            
            # If still no order_id, try to get it from the exchange
            if not order_id:
                open_orders = self.exchange_api.get_open_orders(symbol)
                for order in open_orders:
                    if order.get("client_order_id") == client_order_id:
                        order_id = order.get("order_id")
                        break
            
            if not order_id:
                logger.warning(f"Order with client ID {client_order_id} not found")
                return None
            
            # Cancel the order
            logger.info(f"Cancelling order {order_id}")
            result = self.exchange_api.cancel_order(symbol, order_id)
            
            # Update order status
            with self.lock:
                if order_id in self.active_orders:
                    order = self.active_orders.pop(order_id)
                    order["status"] = "CANCELED"
                    self.order_history[order_id] = order
            
            logger.info(f"Order cancelled: {order_id}")
            return result
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            handle_error(e, "Failed to cancel order")
            raise
    
    def cancel_all_orders(self, symbol=None):
        """
        Cancel all open orders, optionally filtered by symbol.
        
        Args:
            symbol (str): Trading pair symbol (optional)
            
        Returns:
            list: List of cancelled order IDs
        """
        try:
            logger.info(f"Cancelling all orders" + (f" for {symbol}" if symbol else ""))
            result = self.exchange_api.cancel_all_orders(symbol)
            
            # Update order statuses
            with self.lock:
                for order_id in list(self.active_orders.keys()):
                    order = self.active_orders[order_id]
                    if symbol is None or order.get("symbol") == symbol:
                        order["status"] = "CANCELED"
                        self.order_history[order_id] = order
                        self.active_orders.pop(order_id)
            
            logger.info(f"All orders cancelled" + (f" for {symbol}" if symbol else ""))
            return result
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
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
            if not order_id and not client_order_id:
                raise ValueError("Either order_id or client_order_id must be provided")
            
            # Look up order in active or historical orders
            with self.lock:
                if order_id:
                    if order_id in self.active_orders:
                        return self.active_orders[order_id]
                    elif order_id in self.order_history:
                        return self.order_history[order_id]
                elif client_order_id:
                    for orders in [self.active_orders, self.order_history]:
                        for order in orders.values():
                            if order.get("client_order_id") == client_order_id:
                                return order
            
            # If not found locally, query the exchange
            logger.debug(f"Order not found locally, querying exchange")
            return self.exchange_api.get_order(symbol, order_id, client_order_id)
        except Exception as e:
            logger.error(f"Error getting order: {e}")
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
            # Query the exchange to ensure we have the latest information
            open_orders = self.exchange_api.get_open_orders(symbol)
            
            # Update local cache
            with self.lock:
                # Clear existing open orders for the symbol or all symbols
                if symbol:
                    for order_id in list(self.active_orders.keys()):
                        if self.active_orders[order_id].get("symbol") == symbol:
                            self.active_orders.pop(order_id)
                else:
                    self.active_orders.clear()
                
                # Add new open orders to cache
                for order in open_orders:
                    self.active_orders[order["order_id"]] = order
            
            return open_orders
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            handle_error(e, "Failed to get open orders")
            
            # Return cached orders if available
            with self.lock:
                if symbol:
                    return [order for order in self.active_orders.values() 
                            if order.get("symbol") == symbol]
                else:
                    return list(self.active_orders.values())
    
    def update_order_status(self, order_id, new_status, executed_qty=None, executed_price=None):
        """
        Update the status of an order in local cache.
        
        Args:
            order_id (str): Order ID
            new_status (str): New order status
            executed_qty (float): Executed quantity
            executed_price (float): Executed price
            
        Returns:
            dict: Updated order information
        """
        with self.lock:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order["status"] = new_status
                
                if executed_qty is not None:
                    order["executed_qty"] = executed_qty
                
                if executed_price is not None:
                    order["executed_price"] = executed_price
                
                # Move to history if final status
                if new_status in ["FILLED", "CANCELED", "REJECTED"]:
                    self.order_history[order_id] = order
                    self.active_orders.pop(order_id)
                
                logger.debug(f"Updated order {order_id} status to {new_status}")
                return order
            elif order_id in self.order_history:
                order = self.order_history[order_id]
                order["status"] = new_status
                
                if executed_qty is not None:
                    order["executed_qty"] = executed_qty
                
                if executed_price is not None:
                    order["executed_price"] = executed_price
                
                logger.debug(f"Updated historical order {order_id} status to {new_status}")
                return order
            else:
                logger.warning(f"Order {order_id} not found in cache")
                return None
    
    def sync_orders(self):
        """
        Synchronize local order cache with exchange.
        
        Returns:
            int: Number of orders synced
        """
        try:
            # Get all open orders from exchange
            open_orders = self.exchange_api.get_open_orders()
            
            # Update local cache
            with self.lock:
                # Create sets for comparison
                exchange_order_ids = {order["order_id"] for order in open_orders}
                local_order_ids = set(self.active_orders.keys())
                
                # Orders that are in local cache but not in exchange (probably filled or cancelled)
                missing_orders = local_order_ids - exchange_order_ids
                
                # Handle missing orders
                for order_id in missing_orders:
                    # Get order details from exchange
                    order = self.active_orders[order_id]
                    symbol = order["symbol"]
                    
                    try:
                        updated_order = self.exchange_api.get_order(symbol, order_id)
                        
                        if updated_order:
                            # Update and move to history if final status
                            if updated_order["status"] in ["FILLED", "CANCELED", "REJECTED"]:
                                self.order_history[order_id] = updated_order
                                self.active_orders.pop(order_id)
                            else:
                                self.active_orders[order_id] = updated_order
                        else:
                            # Move to history as cancelled if not found
                            order["status"] = "CANCELED"
                            self.order_history[order_id] = order
                            self.active_orders.pop(order_id)
                    except Exception as e:
                        logger.error(f"Error updating order {order_id}: {e}")
                
                # Update existing orders and add new ones
                for order in open_orders:
                    self.active_orders[order["order_id"]] = order
            
            logger.debug(f"Synced {len(open_orders)} open orders")
            return len(open_orders)
        except Exception as e:
            logger.error(f"Error syncing orders: {e}")
            handle_error(e, "Failed to sync orders")
            return 0
    
    def create_market_order(self, symbol, side, quantity):
        """
        Create a market order.
        
        Args:
            symbol (str): Trading pair symbol
            side (str): Order side ("Buy" or "Sell")
            quantity (float): Order quantity
            
        Returns:
            dict: Order information
        """
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type="Market",
            quantity=quantity
        )
    
    def create_limit_order(self, symbol, side, quantity, price, time_in_force="GTC"):
        """
        Create a limit order.
        
        Args:
            symbol (str): Trading pair symbol
            side (str): Order side ("Buy" or "Sell")
            quantity (float): Order quantity
            price (float): Order price
            time_in_force (str): Time in force ("GTC", "IOC", "FOK")
            
        Returns:
            dict: Order information
        """
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type="Limit",
            quantity=quantity,
            price=price,
            time_in_force=time_in_force
        )
    
    def create_stop_loss_order(self, symbol, side, quantity, stop_price, base_price=None):
        """
        Create a stop loss order.
        
        Args:
            symbol (str): Trading pair symbol
            side (str): Order side ("Buy" or "Sell")
            quantity (float): Order quantity
            stop_price (float): Stop price
            base_price (float): Base price (optional)
            
        Returns:
            dict: Order information
        """
        # For Bybit, stop loss is set as a parameter in the main order
        # but we want to treat it like a separate order in our system
        
        # Calculate stop loss percentage from base price
        if base_price:
            stop_loss_pct = abs(stop_price - base_price) / base_price
            logger.debug(f"Stop loss percentage: {stop_loss_pct:.2%}")
        
        # Create a market order with reduce_only flag
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type="Market",
            quantity=quantity,
            reduce_only=True,
            close_on_trigger=True,
            stop_loss=stop_price
        )
    
    def create_take_profit_order(self, symbol, side, quantity, take_profit_price, base_price=None):
        """
        Create a take profit order.
        
        Args:
            symbol (str): Trading pair symbol
            side (str): Order side ("Buy" or "Sell")
            quantity (float): Order quantity
            take_profit_price (float): Take profit price
            base_price (float): Base price (optional)
            
        Returns:
            dict: Order information
        """
        # For Bybit, take profit is set as a parameter in the main order
        # but we want to treat it like a separate order in our system
        
        # Calculate take profit percentage from base price
        if base_price:
            take_profit_pct = abs(take_profit_price - base_price) / base_price
            logger.debug(f"Take profit percentage: {take_profit_pct:.2%}")
        
        # Create a market order with reduce_only flag
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type="Market",
            quantity=quantity,
            reduce_only=True,
            close_on_trigger=True,
            take_profit=take_profit_price
        )
    
    def close_position(self, symbol, position_side, position_size=None):
        """
        Close an open position.
        
        Args:
            symbol (str): Trading pair symbol
            position_side (str): Position side ("LONG" or "SHORT")
            position_size (float): Position size (if None, close entire position)
            
        Returns:
            dict: Order information
        """
        try:
            # Get position information
            position = self.account_manager.get_position(symbol)
            
            if not position or position['size'] == 0:
                logger.warning(f"No open position found for {symbol}")
                return None
            
            # Determine close side (opposite of position side)
            close_side = "Sell" if position_side.upper() == "LONG" else "Buy"
            
            # Determine quantity to close
            quantity = position_size if position_size else position['size']
            
            # Create market order to close position
            logger.info(f"Closing {position_side} position for {symbol}: {quantity}")
            return self.create_market_order(
                symbol=symbol,
                side=close_side,
                quantity=quantity
            )
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            handle_error(e, f"Failed to close position for {symbol}")
            raise