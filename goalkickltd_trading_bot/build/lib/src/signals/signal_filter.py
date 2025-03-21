"""
Signal filter module for the Goalkick Ltd Trading Bot.
Filters and validates signals based on various criteria.
"""

import pandas as pd
from datetime import datetime

from config.logging_config import get_logger
from config.bot_config import TRADING_CONFIG
from src.utils.error_handling import handle_error

logger = get_logger("signals.filter")

class SignalFilter:
    """Class for filtering and validating trading signals."""
    
    def __init__(self, portfolio_manager, datastore):
        """
        Initialize the SignalFilter.
        
        Args:
            portfolio_manager: PortfolioManager instance
            datastore: DataStore instance
        """
        self.portfolio_manager = portfolio_manager
        self.datastore = datastore
    
    def filter_signals(self, signals):
        """
        Filter and validate a list of signals.
        
        Args:
            signals (list): List of signals to filter
            
        Returns:
            list: List of filtered signals
        """
        try:
            if not signals:
                return []
            
            filtered_signals = []
            
            # Apply each filter
            for signal in signals:
                if self._validate_signal(signal):
                    filtered_signals.append(signal)
            
            if filtered_signals:
                logger.info(f"Filtered signals: {len(filtered_signals)} out of {len(signals)} passed")
            
            return filtered_signals
        except Exception as e:
            logger.error(f"Error filtering signals: {e}")
            handle_error(e, "Failed to filter signals")
            return []
    
    def _validate_signal(self, signal):
        """
        Validate a single signal against all filters.
        
        Args:
            signal (dict): Signal to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Apply filters in order
            # If any filter fails, return False
            
            # 1. Check if signal is already executed or cancelled
            if signal.get('status') in ['EXECUTED', 'CANCELLED', 'EXPIRED']:
                logger.debug(f"Signal {signal['id']} rejected: Already {signal['status']}")
                return False
            
            # 2. Check if signal is expired
            if 'expiry' in signal:
                current_time = datetime.now().timestamp() * 1000
                if signal['expiry'] < current_time:
                    logger.debug(f"Signal {signal['id']} rejected: Expired")
                    return False
            
            # 3. Check if signal strength is above threshold
            if signal.get('strength', 0) < 0.3:
                logger.debug(f"Signal {signal['id']} rejected: Low strength {signal.get('strength', 0)}")
                return False
            
            # 4. Check trade direction
            direction = signal.get('direction')
            if direction not in ['BUY', 'SELL', 'ARBITRAGE', 'EXIT']:
                logger.debug(f"Signal {signal['id']} rejected: Invalid direction {direction}")
                return False
            
            # 5. Check if symbol is valid and active
            symbol = signal.get('symbol')
            if not symbol:
                logger.debug(f"Signal {signal['id']} rejected: Missing symbol")
                return False
            
            # 6. Check if there's already a position in the opposite direction
            position = self.portfolio_manager.positions.get(symbol)
            if position:
                # Check if trying to open a position in the opposite direction
                position_side = position.get('side', '')
                if ((direction == 'BUY' and position_side == 'SHORT') or
                    (direction == 'SELL' and position_side == 'LONG')):
                    logger.debug(f"Signal {signal['id']} rejected: Opposite position exists")
                    return False
            
            # 7. Check max open positions
            max_open_trades = TRADING_CONFIG.get('max_open_trades', 5)
            if self.portfolio_manager.get_open_position_count() >= max_open_trades:
                # Only allow signals that close existing positions
                if not position:
                    logger.debug(f"Signal {signal['id']} rejected: Max open positions reached")
                    return False
            
            # 8. Check max positions per symbol
            max_per_symbol = TRADING_CONFIG.get('max_open_trades_per_pair', 1)
            open_trades = self.portfolio_manager.get_open_trades(symbol)
            if len(open_trades) >= max_per_symbol:
                logger.debug(f"Signal {signal['id']} rejected: Max positions per symbol reached")
                return False
            
            # 9. Check if similar signals have been executed recently
            recent_signals = self.datastore.get_signals(
                symbol=symbol,
                direction=direction,
                status='EXECUTED',
                start_time=datetime.now().timestamp() * 1000 - 4 * 60 * 60 * 1000  # Last 4 hours
            )
            
            if recent_signals:
                logger.debug(f"Signal {signal['id']} rejected: Similar signal executed recently")
                return False
            
            # 10. Check risk limits
            risk_check = self.portfolio_manager.check_risk_limits()
            if risk_check['status'] != 'OK':
                logger.debug(f"Signal {signal['id']} rejected: Risk limits exceeded")
                return False
            
            # 11. Check if the position is allowed based on risk rules
            is_allowed, reason = self.portfolio_manager.is_position_allowed(
                symbol=symbol, 
                side=direction,
                quantity=1.0,  # Placeholder
                entry_price=signal.get('price', 0)
            )
            
            if not is_allowed:
                logger.debug(f"Signal {signal['id']} rejected: {reason}")
                return False
            
            # All checks passed
            return True
        except Exception as e:
            logger.error(f"Error validating signal {signal.get('id', 'unknown')}: {e}")
            handle_error(e, f"Failed to validate signal {signal.get('id', 'unknown')}")
            return False
    
    def prioritize_signals(self, signals):
        """
        Prioritize a list of filtered signals.
        
        Args:
            signals (list): List of filtered signals
            
        Returns:
            list: List of signals sorted by priority
        """
        try:
            if not signals:
                return []
            
            # Sort by signal strength (descending)
            return sorted(signals, key=lambda s: s.get('strength', 0), reverse=True)
        except Exception as e:
            logger.error(f"Error prioritizing signals: {e}")
            handle_error(e, "Failed to prioritize signals")
            return signals  # Return unsorted signals as fallback
    
    def check_conflicting_signals(self, signals):
        """
        Check for conflicting signals and resolve conflicts.
        
        Args:
            signals (list): List of signals
            
        Returns:
            list: List of non-conflicting signals
        """
        try:
            if not signals:
                return []
            
            # Group signals by symbol
            symbol_signals = {}
            for signal in signals:
                symbol = signal.get('symbol')
                if symbol not in symbol_signals:
                    symbol_signals[symbol] = []
                symbol_signals[symbol].append(signal)
            
            # Resolve conflicts for each symbol
            resolved = []
            
            for symbol, sym_signals in symbol_signals.items():
                if len(sym_signals) <= 1:
                    # No conflicts
                    resolved.extend(sym_signals)
                    continue
                
                # Check for conflicting directions
                buy_signals = [s for s in sym_signals if s.get('direction') == 'BUY']
                sell_signals = [s for s in sym_signals if s.get('direction') == 'SELL']
                
                if buy_signals and sell_signals:
                    # Conflict detected
                    logger.warning(f"Conflicting signals for {symbol}: {len(buy_signals)} buy, {len(sell_signals)} sell")
                    
                    # Calculate total strength for each direction
                    buy_strength = sum(s.get('strength', 0) for s in buy_signals)
                    sell_strength = sum(s.get('strength', 0) for s in sell_signals)
                    
                    # Choose stronger direction
                    if buy_strength > sell_strength:
                        # Choose strongest buy signal
                        strongest = max(buy_signals, key=lambda s: s.get('strength', 0))
                        resolved.append(strongest)
                        logger.info(f"Resolved conflict for {symbol}: chose BUY (strength: {strongest.get('strength', 0)})")
                    else:
                        # Choose strongest sell signal
                        strongest = max(sell_signals, key=lambda s: s.get('strength', 0))
                        resolved.append(strongest)
                        logger.info(f"Resolved conflict for {symbol}: chose SELL (strength: {strongest.get('strength', 0)})")
                else:
                    # No direction conflicts, choose strongest signal
                    strongest = max(sym_signals, key=lambda s: s.get('strength', 0))
                    resolved.append(strongest)
            
            return resolved
        except Exception as e:
            logger.error(f"Error checking conflicting signals: {e}")
            handle_error(e, "Failed to check conflicting signals")
            return signals  # Return original signals as fallback
    
    def check_duplicate_signals(self, signals):
        """
        Check for duplicate signals and remove duplicates.
        
        Args:
            signals (list): List of signals
            
        Returns:
            list: List of unique signals
        """
        try:
            if not signals:
                return []
            
            # Track seen signal keys
            seen = set()
            unique = []
            
            for signal in signals:
                # Create a key for signal identity
                key = (
                    signal.get('symbol', ''),
                    signal.get('direction', ''),
                    signal.get('strategy', '')
                )
                
                if key in seen:
                    # Skip duplicate
                    continue
                
                seen.add(key)
                unique.append(signal)
            
            return unique
        except Exception as e:
            logger.error(f"Error checking duplicate signals: {e}")
            handle_error(e, "Failed to check duplicate signals")
            return signals  # Return original signals as fallback
    
    def check_signals_compatibility(self, signals):
        """
        Check if signals are compatible with current portfolio.
        
        Args:
            signals (list): List of signals
            
        Returns:
            list: List of compatible signals
        """
        try:
            if not signals:
                return []
            
            compatible = []
            
            for signal in signals:
                symbol = signal.get('symbol')
                direction = signal.get('direction')
                
                # Check if there's an existing position
                position = self.portfolio_manager.positions.get(symbol)
                
                if position:
                    position_side = position.get('side', '')
                    
                    # Check compatibility
                    if (direction == 'BUY' and position_side == 'LONG') or \
                       (direction == 'SELL' and position_side == 'SHORT'):
                        # Compatible with existing position
                        compatible.append(signal)
                    elif direction == 'EXIT':
                        # Exit signals are always compatible
                        compatible.append(signal)
                    else:
                        # Incompatible direction
                        logger.debug(f"Signal {signal.get('id')} incompatible with position {position_side}")
                else:
                    # No existing position
                    if direction != 'EXIT':
                        # New position signals are compatible
                        compatible.append(signal)
                    else:
                        # Exit with no position is incompatible
                        logger.debug(f"Signal {signal.get('id')} incompatible: EXIT with no position")
            
            return compatible
        except Exception as e:
            logger.error(f"Error checking signals compatibility: {e}")
            handle_error(e, "Failed to check signals compatibility")
            return signals  # Return original signals as fallback
    
    def process_signals(self, signals):
        """
        Process a list of signals through all filters.
        
        Args:
            signals (list): List of signals to process
            
        Returns:
            list: List of processed signals ready for execution
        """
        try:
            # Filter signals
            filtered = self.filter_signals(signals)
            
            # Check for duplicates
            unique = self.check_duplicate_signals(filtered)
            
            # Check for conflicts
            non_conflicting = self.check_conflicting_signals(unique)
            
            # Check compatibility
            compatible = self.check_signals_compatibility(non_conflicting)
            
            # Prioritize signals
            prioritized = self.prioritize_signals(compatible)
            
            # Limit number of signals
            max_signals = TRADING_CONFIG.get('max_trades_per_day', 5)
            limited = prioritized[:max_signals]
            
            if limited:
                logger.info(f"Processed {len(signals)} signals, resulting in {len(limited)} actionable signals")
            
            return limited
        except Exception as e:
            logger.error(f"Error processing signals: {e}")
            handle_error(e, "Failed to process signals")
            return []