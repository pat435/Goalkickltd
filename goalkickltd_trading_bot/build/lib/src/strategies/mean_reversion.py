"""
Mean reversion strategies for the Goalkick Ltd Trading Bot.
"""

import pandas as pd
import numpy as np

from config.logging_config import get_logger
from config.strategy_params import MEAN_REVERSION_PARAMS
from src.strategies.base_strategy import Strategy
from src.indicators.momentum import relative_strength_index, stochastic_oscillator, commodity_channel_index, williams_r
from src.indicators.volatility import bollinger_bands, average_true_range
from src.utils.error_handling import handle_error

logger = get_logger("strategies.mean_reversion")

class RSIStrategy(Strategy):
    """Relative Strength Index (RSI) mean reversion strategy."""
    
    def __init__(self, timeframes=None, symbols=None, params=None):
        """
        Initialize the strategy.
        
        Args:
            timeframes (list): List of timeframes to use
            symbols (list): List of symbols to trade
            params (dict): Strategy parameters
        """
        default_params = MEAN_REVERSION_PARAMS.get('rsi', {})
        params = {**default_params, **(params or {})}
        
        super().__init__('RSI', timeframes, symbols, params)
    
    def generate_signals(self, data, symbol, timeframe):
        """
        Generate trading signals for a symbol and timeframe.
        
        Args:
            data (pd.DataFrame): Historical price data
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            
        Returns:
            list: List of signal dictionaries
        """
        try:
            if data.empty:
                logger.warning(f"Empty data for {symbol} {timeframe}")
                return []
            
            # Preprocess data
            df = self.preprocess_data(data)
            
            if df.empty:
                return []
            
            # Get parameters
            rsi_length = self.params.get('length', 14)
            overbought = self.params.get('overbought', 70)
            oversold = self.params.get('oversold', 30)
            
            # Calculate RSI
            df['rsi'] = relative_strength_index(df, rsi_length)
            
            # Generate signals
            signals = []
            
            # Check if we have enough data
            if len(df) < rsi_length + 5:
                logger.warning(f"Not enough data for {symbol} {timeframe}")
                return []
            
            # Get the last two rows for comparison
            prev_row = df.iloc[-2]
            curr_row = df.iloc[-1]
            
            # Calculate price and other values
            current_price = curr_row['close']
            atr = average_true_range(df)
            atr_value = atr.iloc[-1]
            
            # Buy signal: RSI crosses above oversold level
            if prev_row['rsi'] <= oversold and curr_row['rsi'] > oversold:
                # Calculate stop loss and take profit
                stop_loss = current_price - atr_value * 1.5
                take_profit = current_price + atr_value * 1.5
                
                # Create signal
                signal = self.create_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction="BUY",
                    strength=0.7,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'rsi': curr_row['rsi'],
                        'oversold': oversold,
                        'atr': atr_value
                    }
                )
                
                signals.append(signal)
                logger.info(f"BUY signal for {symbol} at {current_price}")
            
            # Sell signal: RSI crosses below overbought level
            elif prev_row['rsi'] >= overbought and curr_row['rsi'] < overbought:
                # Calculate stop loss and take profit
                stop_loss = current_price + atr_value * 1.5
                take_profit = current_price - atr_value * 1.5
                
                # Create signal
                signal = self.create_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction="SELL",
                    strength=0.7,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'rsi': curr_row['rsi'],
                        'overbought': overbought,
                        'atr': atr_value
                    }
                )
                
                signals.append(signal)
                logger.info(f"SELL signal for {symbol} at {current_price}")
            
            return signals
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {e}")
            handle_error(e, f"Failed to generate signals for {symbol} {timeframe}")
            return []
    
    def validate_parameters(self, params=None):
        """
        Validate strategy parameters.
        
        Args:
            params (dict): Parameters to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        params = params or self.params
        
        # Check required parameters
        required = ['length', 'overbought', 'oversold']
        if not all(key in params for key in required):
            logger.error(f"Missing required parameters: {required}")
            return False
        
        # Validate parameter values
        if params['length'] < 2:
            logger.error(f"Length must be at least 2")
            return False
        
        if params['overbought'] <= params['oversold']:
            logger.error(f"Overbought must be greater than oversold")
            return False
        
        if params['overbought'] < 50 or params['overbought'] > 100:
            logger.error(f"Overbought should be between 50 and 100")
            return False
        
        if params['oversold'] < 0 or params['oversold'] > 50:
            logger.error(f"Oversold should be between 0 and 50")
            return False
        
        return True


class BollingerBandsStrategy(Strategy):
    """Bollinger Bands mean reversion strategy."""
    
    def __init__(self, timeframes=None, symbols=None, params=None):
        """
        Initialize the strategy.
        
        Args:
            timeframes (list): List of timeframes to use
            symbols (list): List of symbols to trade
            params (dict): Strategy parameters
        """
        default_params = MEAN_REVERSION_PARAMS.get('bollinger_bands', {})
        params = {**default_params, **(params or {})}
        
        super().__init__('BollingerBands', timeframes, symbols, params)
    
    def generate_signals(self, data, symbol, timeframe):
        """
        Generate trading signals for a symbol and timeframe.
        
        Args:
            data (pd.DataFrame): Historical price data
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            
        Returns:
            list: List of signal dictionaries
        """
        try:
            if data.empty:
                logger.warning(f"Empty data for {symbol} {timeframe}")
                return []
            
            # Preprocess data
            df = self.preprocess_data(data)
            
            if df.empty:
                return []
            
            # Get parameters
            bb_length = self.params.get('length', 20)
            bb_std_dev = self.params.get('std_dev', 2.0)
            
            # Calculate Bollinger Bands
            middle_band, upper_band, lower_band = bollinger_bands(df, bb_length, bb_std_dev)
            
            # Add Bollinger Bands to DataFrame
            df['bb_middle'] = middle_band
            df['bb_upper'] = upper_band
            df['bb_lower'] = lower_band
            
            # Calculate Bandwidth and %B
            df['bb_bandwidth'] = (upper_band - lower_band) / middle_band
            df['bb_percent_b'] = (df['close'] - lower_band) / (upper_band - lower_band)
            
            # Generate signals
            signals = []
            
            # Check if we have enough data
            if len(df) < bb_length + 5:
                logger.warning(f"Not enough data for {symbol} {timeframe}")
                return []
            
            # Get the last two rows for comparison
            prev_row = df.iloc[-2]
            curr_row = df.iloc[-1]
            
            # Calculate price and other values
            current_price = curr_row['close']
            
            # Buy signal: Price crosses above lower band and %B is low
            if (prev_row['close'] <= prev_row['bb_lower'] and curr_row['close'] > curr_row['bb_lower']) or \
               (curr_row['bb_percent_b'] < 0.05 and prev_row['bb_percent_b'] < curr_row['bb_percent_b']):
                # Calculate stop loss and take profit
                stop_loss = curr_row['bb_lower'] - (curr_row['bb_middle'] - curr_row['bb_lower']) * 0.5
                take_profit = curr_row['bb_middle']
                
                # Create signal
                signal = self.create_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction="BUY",
                    strength=0.8,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'bb_middle': curr_row['bb_middle'],
                        'bb_upper': curr_row['bb_upper'],
                        'bb_lower': curr_row['bb_lower'],
                        'bb_percent_b': curr_row['bb_percent_b'],
                        'bb_bandwidth': curr_row['bb_bandwidth']
                    }
                )
                
                signals.append(signal)
                logger.info(f"BUY signal for {symbol} at {current_price}")
            
            # Sell signal: Price crosses below upper band and %B is high
            elif (prev_row['close'] >= prev_row['bb_upper'] and curr_row['close'] < curr_row['bb_upper']) or \
                 (curr_row['bb_percent_b'] > 0.95 and prev_row['bb_percent_b'] > curr_row['bb_percent_b']):
                # Calculate stop loss and take profit
                stop_loss = curr_row['bb_upper'] + (curr_row['bb_upper'] - curr_row['bb_middle']) * 0.5
                take_profit = curr_row['bb_middle']
                
                # Create signal
                signal = self.create_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction="SELL",
                    strength=0.8,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'bb_middle': curr_row['bb_middle'],
                        'bb_upper': curr_row['bb_upper'],
                        'bb_lower': curr_row['bb_lower'],
                        'bb_percent_b': curr_row['bb_percent_b'],
                        'bb_bandwidth': curr_row['bb_bandwidth']
                    }
                )
                
                signals.append(signal)
                logger.info(f"SELL signal for {symbol} at {current_price}")
            
            return signals
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {e}")
            handle_error(e, f"Failed to generate signals for {symbol} {timeframe}")
            return []
    
    def validate_parameters(self, params=None):
        """
        Validate strategy parameters.
        
        Args:
            params (dict): Parameters to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        params = params or self.params
        
        # Check required parameters
        required = ['length', 'std_dev']
        if not all(key in params for key in required):
            logger.error(f"Missing required parameters: {required}")
            return False
        
        # Validate parameter values
        if params['length'] < 5:
            logger.error(f"Length must be at least 5")
            return False
        
        if params['std_dev'] <= 0:
            logger.error(f"Standard deviation must be positive")
            return False
        
        return True