"""
Position sizing module for the Goalkick Ltd Trading Bot.
Implements various position sizing algorithms and risk management techniques.
"""

import numpy as np

from config.logging_config import get_logger
from config.bot_config import TRADING_CONFIG, RISK_CONFIG
from src.utils.error_handling import handle_error

logger = get_logger("risk.position_sizer")

class PositionSizer:
    """Class for calculating position sizes based on risk parameters."""
    
    def __init__(self, account_manager):
        """
        Initialize the PositionSizer.
        
        Args:
            account_manager: Account Manager instance
        """
        self.account_manager = account_manager
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_price, risk_percentage=None):
        """
        Calculate position size based on risk percentage and stop loss.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            stop_loss_price (float): Stop loss price
            risk_percentage (float): Risk percentage (default from config)
            
        Returns:
            float: Position size in base currency
        """
        try:
            if risk_percentage is None:
                risk_percentage = TRADING_CONFIG["risk_per_trade"]
            
            # Get account balance
            balance = self.account_manager.get_balance()
            
            # Calculate risk amount in quote currency
            risk_amount = balance * risk_percentage
            
            # Calculate stop loss distance in percentage
            stop_loss_pct = abs(entry_price - stop_loss_price) / entry_price
            
            # Calculate position size in quote currency
            position_value = risk_amount / stop_loss_pct
            
            # Calculate quantity in base currency
            quantity = position_value / entry_price
            
            # Apply symbol-specific rounding
            quantity = self._apply_lot_size_filter(symbol, quantity)
            
            logger.debug(f"Calculated position size for {symbol}: {quantity} (risk: {risk_percentage:.2%}, stop: {stop_loss_pct:.2%})")
            return quantity
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            handle_error(e, "Failed to calculate position size")
            return 0
    
    def calculate_position_size_atr(self, symbol, entry_price, atr_value, atr_multiplier=1.5, risk_percentage=None):
        """
        Calculate position size based on ATR for stop loss placement.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            atr_value (float): ATR value
            atr_multiplier (float): ATR multiplier for stop loss distance
            risk_percentage (float): Risk percentage (default from config)
            
        Returns:
            tuple: (Position size, Stop loss price)
        """
        try:
            if risk_percentage is None:
                risk_percentage = TRADING_CONFIG["risk_per_trade"]
            
            # Calculate stop loss distance based on ATR
            stop_distance = atr_value * atr_multiplier
            
            # Get account balance
            balance = self.account_manager.get_balance()
            
            # Calculate risk amount in quote currency
            risk_amount = balance * risk_percentage
            
            # Calculate position size in quote currency
            position_value = risk_amount / (stop_distance / entry_price)
            
            # Calculate quantity in base currency
            quantity = position_value / entry_price
            
            # Apply symbol-specific rounding
            quantity = self._apply_lot_size_filter(symbol, quantity)
            
            # Calculate stop loss price
            stop_loss_price = entry_price - stop_distance
            
            logger.debug(f"Calculated ATR position size for {symbol}: {quantity} (risk: {risk_percentage:.2%}, ATR: {atr_value:.2f})")
            return quantity, stop_loss_price
        except Exception as e:
            logger.error(f"Error calculating ATR position size: {e}")
            handle_error(e, "Failed to calculate ATR position size")
            return 0, 0
    
    def calculate_position_size_volatility(self, symbol, entry_price, volatility, vol_multiplier=2.0, risk_percentage=None):
        """
        Calculate position size based on volatility.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            volatility (float): Volatility value (standard deviation)
            vol_multiplier (float): Volatility multiplier for stop loss distance
            risk_percentage (float): Risk percentage (default from config)
            
        Returns:
            tuple: (Position size, Stop loss price)
        """
        try:
            if risk_percentage is None:
                risk_percentage = TRADING_CONFIG["risk_per_trade"]
            
            # Calculate stop loss distance based on volatility
            stop_distance = entry_price * volatility * vol_multiplier
            
            # Get account balance
            balance = self.account_manager.get_balance()
            
            # Calculate risk amount in quote currency
            risk_amount = balance * risk_percentage
            
            # Calculate position size in quote currency
            position_value = risk_amount / (stop_distance / entry_price)
            
            # Calculate quantity in base currency
            quantity = position_value / entry_price
            
            # Apply symbol-specific rounding
            quantity = self._apply_lot_size_filter(symbol, quantity)
            
            # Calculate stop loss price
            stop_loss_price = entry_price - stop_distance
            
            logger.debug(f"Calculated volatility position size for {symbol}: {quantity} (risk: {risk_percentage:.2%}, vol: {volatility:.4f})")
            return quantity, stop_loss_price
        except Exception as e:
            logger.error(f"Error calculating volatility position size: {e}")
            handle_error(e, "Failed to calculate volatility position size")
            return 0, 0
    
    def calculate_fixed_risk_position_size(self, symbol, entry_price, risk_amount, stop_loss_price):
        """
        Calculate position size based on fixed risk amount.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            risk_amount (float): Risk amount in quote currency
            stop_loss_price (float): Stop loss price
            
        Returns:
            float: Position size in base currency
        """
        try:
            # Calculate stop loss distance in percentage
            stop_loss_pct = abs(entry_price - stop_loss_price) / entry_price
            
            # Calculate position size in quote currency
            position_value = risk_amount / stop_loss_pct
            
            # Calculate quantity in base currency
            quantity = position_value / entry_price
            
            # Apply symbol-specific rounding
            quantity = self._apply_lot_size_filter(symbol, quantity)
            
            logger.debug(f"Calculated fixed risk position size for {symbol}: {quantity} (risk: {risk_amount}, stop: {stop_loss_pct:.2%})")
            return quantity
        except Exception as e:
            logger.error(f"Error calculating fixed risk position size: {e}")
            handle_error(e, "Failed to calculate fixed risk position size")
            return 0
    
    def calculate_kelly_position_size(self, symbol, entry_price, win_rate, reward_risk_ratio, max_risk_percentage=None):
        """
        Calculate position size based on Kelly Criterion.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            win_rate (float): Historical win rate (0-1)
            reward_risk_ratio (float): Reward to risk ratio
            max_risk_percentage (float): Maximum risk percentage (default from config)
            
        Returns:
            float: Position size in base currency
        """
        try:
            if max_risk_percentage is None:
                max_risk_percentage = TRADING_CONFIG["risk_per_trade"]
            
            # Calculate Kelly percentage (f = p - (1-p)/r, where p=win rate, r=reward/risk)
            kelly_pct = win_rate - ((1 - win_rate) / reward_risk_ratio)
            
            # Apply half-Kelly for more conservative sizing
            half_kelly = kelly_pct / 2
            
            # Cap at max risk percentage
            risk_percentage = min(half_kelly, max_risk_percentage)
            
            # Ensure non-negative
            risk_percentage = max(0, risk_percentage)
            
            # Get account balance
            balance = self.account_manager.get_balance()
            
            # Calculate position size in quote currency
            position_value = balance * risk_percentage
            
            # Calculate quantity in base currency
            quantity = position_value / entry_price
            
            # Apply symbol-specific rounding
            quantity = self._apply_lot_size_filter(symbol, quantity)
            
            logger.debug(f"Calculated Kelly position size for {symbol}: {quantity} (Kelly: {kelly_pct:.2%}, half-Kelly: {half_kelly:.2%})")
            return quantity
        except Exception as e:
            logger.error(f"Error calculating Kelly position size: {e}")
            handle_error(e, "Failed to calculate Kelly position size")
            return 0
    
    def calculate_optimal_f_position_size(self, symbol, entry_price, historic_trades, max_risk_percentage=None):
        """
        Calculate position size based on Optimal f (Ralph Vince).
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            historic_trades (list): List of historical trade results
            max_risk_percentage (float): Maximum risk percentage (default from config)
            
        Returns:
            float: Position size in base currency
        """
        try:
            if max_risk_percentage is None:
                max_risk_percentage = TRADING_CONFIG["risk_per_trade"]
            
            if not historic_trades:
                logger.warning(f"No historical trades provided for Optimal f calculation")
                return self.account_manager.calculate_position_size(symbol, entry_price, max_risk_percentage)
            
            # Find the worst loss in the historical trades
            worst_loss_pct = 0
            for trade in historic_trades:
                loss_pct = trade.get('loss_pct', 0)
                worst_loss_pct = min(worst_loss_pct, loss_pct)
            
            worst_loss_pct = abs(worst_loss_pct)
            
            if worst_loss_pct == 0:
                logger.warning(f"No losses found in historical trades for Optimal f calculation")
                return self.account_manager.calculate_position_size(symbol, entry_price, max_risk_percentage)
            
            # Calculate optimal f
            optimal_f = 0
            for trade in historic_trades:
                win_pct = trade.get('profit_pct', 0)
                if win_pct > 0:
                    optimal_f += (win_pct / worst_loss_pct)
            
            optimal_f = optimal_f / len(historic_trades)
            
            # Apply half-optimal f for more conservative sizing
            half_optimal_f = optimal_f / 2
            
            # Cap at max risk percentage
            risk_percentage = min(half_optimal_f, max_risk_percentage)
            
            # Ensure non-negative
            risk_percentage = max(0, risk_percentage)
            
            # Get account balance
            balance = self.account_manager.get_balance()
            
            # Calculate position size in quote currency
            position_value = balance * risk_percentage
            
            # Calculate quantity in base currency
            quantity = position_value / entry_price
            
            # Apply symbol-specific rounding
            quantity = self._apply_lot_size_filter(symbol, quantity)
            
            logger.debug(f"Calculated Optimal f position size for {symbol}: {quantity} (f: {optimal_f:.2%}, half-f: {half_optimal_f:.2%})")
            return quantity
        except Exception as e:
            logger.error(f"Error calculating Optimal f position size: {e}")
            handle_error(e, "Failed to calculate Optimal f position size")
            return 0
    
    def calculate_max_position_size(self, symbol, entry_price, max_percentage=None):
        """
        Calculate maximum position size based on available balance.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price
            max_percentage (float): Maximum percentage of available balance to use
            
        Returns:
            float: Position size in base currency
        """
        try:
            if max_percentage is None:
                max_percentage = 0.95  # 95% of available balance
            
            # Get available balance
            available = self.account_manager.get_available_balance()
            
            # Calculate position value in quote currency
            position_value = available * max_percentage
            
            # Calculate quantity in base currency
            quantity = position_value / entry_price
            
            # Apply symbol-specific rounding
            quantity = self._apply_lot_size_filter(symbol, quantity)
            
            logger.debug(f"Calculated max position size for {symbol}: {quantity} (max: {max_percentage:.2%})")
            return quantity
        except Exception as e:
            logger.error(f"Error calculating max position size: {e}")
            handle_error(e, "Failed to calculate max position size")
            return 0
    
    def _apply_lot_size_filter(self, symbol, quantity):
        """
        Apply symbol-specific lot size filter to round quantity.
        
        Args:
            symbol (str): Trading pair symbol
            quantity (float): Original quantity
            
        Returns:
            float: Rounded quantity
        """
        try:
            # Get symbol info from exchange
            symbol_info = self.account_manager.exchange_api.get_symbol_info(symbol)
            
            if not symbol_info or 'lotSizeFilter' not in symbol_info:
                logger.warning(f"No lot size filter found for {symbol}, using default precision")
                # Default rounding to 6 decimal places
                return round(float(quantity), 6)
            
            lot_size_filter = symbol_info['lotSizeFilter']
            
            # Get min quantity, max quantity, and step size
            min_qty = float(lot_size_filter.get('minOrderQty', 0.001))
            max_qty = float(lot_size_filter.get('maxOrderQty', 1000000))
            step_size = float(lot_size_filter.get('qtyStep', 0.001))
            
            # Round down to nearest step
            rounded_qty = int(quantity / step_size) * step_size
            
            # Ensure min and max limits
            rounded_qty = max(min_qty, min(rounded_qty, max_qty))
            
            # Round to the precision of step_size
            precision = len(str(step_size).split('.')[-1])
            rounded_qty = round(rounded_qty, precision)
            
            return rounded_qty
        except Exception as e:
            logger.error(f"Error applying lot size filter for {symbol}: {e}")
            handle_error(e, "Failed to apply lot size filter")
            # Default fallback
            return round(float(quantity), 6)
    
    def adjust_for_correlation(self, symbol, base_position_size, correlation_threshold=0.7):
        """
        Adjust position size based on correlation with existing positions.
        
        Args:
            symbol (str): Trading pair symbol
            base_position_size (float): Base position size
            correlation_threshold (float): Correlation threshold for adjustment
            
        Returns:
            float: Adjusted position size
        """
        try:
            # Get all open positions
            positions = self.account_manager.get_all_positions()
            
            if not positions:
                return base_position_size
            
            # Get the correlation factor (simple implementation)
            # In a real implementation, you would calculate actual correlations
            # between the symbol and existing positions
            correlated_positions = 0
            for position in positions:
                # Placeholder for correlation check
                # In reality, you would use historical price correlation
                if position['symbol'] == symbol:
                    return 0  # Already have a position in this symbol
                
                # Simplified correlation check based on base currency
                symbol_base = symbol.split('USDT')[0] if 'USDT' in symbol else symbol
                position_base = position['symbol'].split('USDT')[0] if 'USDT' in position['symbol'] else position['symbol']
                
                # Simplified correlation logic (in real implementation, use actual correlation matrix)
                if symbol_base == position_base:
                    correlated_positions += 1
            
            # Adjust position size based on correlation
            adjustment_factor = 1.0 - (correlated_positions * 0.2)
            adjustment_factor = max(0.2, adjustment_factor)  # At least 20% of base size
            
            adjusted_size = base_position_size * adjustment_factor
            
            logger.debug(f"Adjusted position size for {symbol}: {adjusted_size} (base: {base_position_size}, factor: {adjustment_factor:.2f})")
            return adjusted_size
        except Exception as e:
            logger.error(f"Error adjusting position size for correlation: {e}")
            handle_error(e, "Failed to adjust position size for correlation")
            return base_position_size