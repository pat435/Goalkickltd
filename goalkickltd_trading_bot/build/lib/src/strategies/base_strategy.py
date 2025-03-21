"""
Base strategy module for the Goalkick Ltd Trading Bot.
Defines the base class for all trading strategies.
"""

from abc import ABC, abstractmethod
import uuid
import pandas as pd
import numpy as np

from config.logging_config import get_logger
from src.utils.error_handling import handle_error

logger = get_logger("strategies.base")

class Strategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, name, timeframes, symbols=None, params=None):
        """
        Initialize the strategy.
        
        Args:
            name (str): Strategy name
            timeframes (list): List of timeframes to use
            symbols (list): List of symbols to trade (optional)
            params (dict): Strategy parameters (optional)
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.timeframes = timeframes if isinstance(timeframes, list) else [timeframes]
        self.symbols = symbols or []
        self.params = params or {}
        self.position = {}  # Current positions: symbol -> position info
        self.signals = []  # Recent signals
        self.active = True  # Whether the strategy is active
    
    @abstractmethod
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
        pass
    
    def preprocess_data(self, data):
        """
        Preprocess data before signal generation.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        try:
            # Make a copy to avoid modifying the original data
            df = data.copy()
            
            # Ensure data has the correct columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing columns in data: {missing_columns}")
                return pd.DataFrame()
            
            # Remove rows with NaN values
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            # Remove duplicate indices
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by index
            df = df.sort_index()
            
            return df
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            handle_error(e, "Failed to preprocess data")
            return pd.DataFrame()
    
    def create_signal(self, symbol, timeframe, direction, strength=1.0, price=None, stop_loss=None, take_profit=None, metadata=None):
        """
        Create a trading signal.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            direction (str): Signal direction ("BUY", "SELL", "EXIT")
            strength (float): Signal strength (0.0-1.0)
            price (float): Signal price (optional)
            stop_loss (float): Stop loss price (optional)
            take_profit (float): Take profit price (optional)
            metadata (dict): Additional signal metadata (optional)
            
        Returns:
            dict: Signal dictionary
        """
        timestamp = pd.Timestamp.now()
        
        signal = {
            'id': str(uuid.uuid4()),
            'timestamp': timestamp.timestamp() * 1000,
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy': self.name,
            'direction': direction,
            'strength': strength,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'expiry': (timestamp + pd.Timedelta(hours=24)).timestamp() * 1000,  # 24-hour expiry by default
            'status': 'NEW',
            'metadata': metadata or {}
        }
        
        # Add to recent signals
        self.signals.append(signal)
        
        # Keep only the most recent 100 signals
        if len(self.signals) > 100:
            self.signals = self.signals[-100:]
        
        logger.debug(f"Created {direction} signal for {symbol} ({timeframe})")
        return signal
    
    def get_signals(self, max_age=None, status=None):
        """
        Get recent signals from this strategy.
        
        Args:
            max_age (float): Maximum age in seconds
            status (str): Filter by signal status
            
        Returns:
            list: List of signal dictionaries
        """
        current_time = pd.Timestamp.now().timestamp() * 1000
        
        if max_age is not None:
            max_age_ms = max_age * 1000
            filtered = [s for s in self.signals if (current_time - s['timestamp']) <= max_age_ms]
        else:
            filtered = self.signals.copy()
        
        if status is not None:
            filtered = [s for s in filtered if s['status'] == status]
        
        return filtered
    
    def validate_parameters(self, params=None):
        """
        Validate strategy parameters.
        
        Args:
            params (dict): Parameters to validate (defaults to self.params)
            
        Returns:
            bool: True if valid, False otherwise
        """
        params = params or self.params
        
        # By default, accept any parameters
        return True
    
    def update_parameters(self, params):
        """
        Update strategy parameters.
        
        Args:
            params (dict): New parameters
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            # Validate new parameters
            if not self.validate_parameters(params):
                logger.error(f"Invalid parameters for strategy {self.name}")
                return False
            
            # Update parameters
            self.params.update(params)
            
            logger.info(f"Updated parameters for strategy {self.name}")
            return True
        except Exception as e:
            logger.error(f"Error updating parameters for strategy {self.name}: {e}")
            handle_error(e, f"Failed to update parameters for strategy {self.name}")
            return False
    
    def serialize(self):
        """
        Serialize the strategy to a dictionary.
        
        Returns:
            dict: Serialized strategy
        """
        return {
            'id': self.id,
            'name': self.name,
            'timeframes': self.timeframes,
            'symbols': self.symbols,
            'params': self.params,
            'active': self.active
        }
    
    @classmethod
    def deserialize(cls, data):
        """
        Create a strategy from serialized data.
        
        Args:
            data (dict): Serialized strategy
            
        Returns:
            Strategy: Deserialized strategy
        """
        strategy = cls(
            name=data['name'],
            timeframes=data['timeframes'],
            symbols=data['symbols'],
            params=data['params']
        )
        
        strategy.id = data['id']
        strategy.active = data['active']
        
        return strategy


class StrategyManager:
    """Class for managing multiple trading strategies."""
    
    def __init__(self):
        """Initialize the StrategyManager."""
        self.strategies = {}  # id -> strategy
    
    def add_strategy(self, strategy):
        """
        Add a strategy to the manager.
        
        Args:
            strategy (Strategy): Strategy to add
            
        Returns:
            str: Strategy ID
        """
        try:
            strategy_id = strategy.id
            self.strategies[strategy_id] = strategy
            
            logger.info(f"Added strategy {strategy.name} with ID {strategy_id}")
            return strategy_id
        except Exception as e:
            logger.error(f"Error adding strategy: {e}")
            handle_error(e, "Failed to add strategy")
            return None
    
    def remove_strategy(self, strategy_id):
        """
        Remove a strategy from the manager.
        
        Args:
            strategy_id (str): ID of the strategy to remove
            
        Returns:
            bool: True if removed, False otherwise
        """
        try:
            if strategy_id in self.strategies:
                strategy = self.strategies.pop(strategy_id)
                
                logger.info(f"Removed strategy {strategy.name} with ID {strategy_id}")
                return True
            else:
                logger.warning(f"Strategy with ID {strategy_id} not found")
                return False
        except Exception as e:
            logger.error(f"Error removing strategy: {e}")
            handle_error(e, "Failed to remove strategy")
            return False
    
    def get_strategy(self, strategy_id):
        """
        Get a strategy by ID.
        
        Args:
            strategy_id (str): Strategy ID
            
        Returns:
            Strategy: Strategy instance or None if not found
        """
        return self.strategies.get(strategy_id)
    
    def get_all_strategies(self, active_only=False):
        """
        Get all strategies.
        
        Args:
            active_only (bool): Only return active strategies
            
        Returns:
            list: List of strategy instances
        """
        if active_only:
            return [s for s in self.strategies.values() if s.active]
        else:
            return list(self.strategies.values())
    
    def get_strategies_for_symbol(self, symbol, active_only=True):
        """
        Get strategies for a specific symbol.
        
        Args:
            symbol (str): Trading pair symbol
            active_only (bool): Only return active strategies
            
        Returns:
            list: List of strategy instances
        """
        if active_only:
            return [s for s in self.strategies.values() 
                   if s.active and (not s.symbols or symbol in s.symbols)]
        else:
            return [s for s in self.strategies.values() 
                   if not s.symbols or symbol in s.symbols]
    
    def get_strategies_for_timeframe(self, timeframe, active_only=True):
        """
        Get strategies for a specific timeframe.
        
        Args:
            timeframe (str): Timeframe
            active_only (bool): Only return active strategies
            
        Returns:
            list: List of strategy instances
        """
        if active_only:
            return [s for s in self.strategies.values() 
                   if s.active and timeframe in s.timeframes]
        else:
            return [s for s in self.strategies.values() 
                   if timeframe in s.timeframes]
    
    def generate_signals(self, data, symbol, timeframe):
        """
        Generate signals from all applicable strategies.
        
        Args:
            data (pd.DataFrame): Historical price data
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            
        Returns:
            list: List of signal dictionaries
        """
        signals = []
        
        # Get strategies for this symbol and timeframe
        strategies = [s for s in self.get_strategies_for_symbol(symbol) 
                     if timeframe in s.timeframes]
        
        if not strategies:
            return signals
        
        for strategy in strategies:
            try:
                strategy_signals = strategy.generate_signals(data, symbol, timeframe)
                
                if strategy_signals:
                    signals.extend(strategy_signals)
            except Exception as e:
                logger.error(f"Error generating signals for strategy {strategy.name}: {e}")
                handle_error(e, f"Failed to generate signals for strategy {strategy.name}")
        
        return signals
    
    def optimize_strategies(self, data, symbols, timeframes, metric='win_rate'):
        """
        Optimize strategy parameters.
        
        Args:
            data (dict): Historical price data (symbol -> timeframe -> DataFrame)
            symbols (list): List of symbols to optimize for
            timeframes (list): List of timeframes to optimize for
            metric (str): Metric to optimize for
            
        Returns:
            dict: Optimization results
        """
        results = {}
        
        for strategy_id, strategy in self.strategies.items():
            try:
                # Check if strategy implements the optimize method
                if hasattr(strategy, 'optimize') and callable(getattr(strategy, 'optimize')):
                    # Filter symbols and timeframes applicable to this strategy
                    strategy_symbols = strategy.symbols or symbols
                    strategy_timeframes = [tf for tf in timeframes if tf in strategy.timeframes]
                    
                    # Create data subset for this strategy
                    strategy_data = {}
                    for symbol in strategy_symbols:
                        if symbol in data:
                            strategy_data[symbol] = {tf: data[symbol][tf] for tf in strategy_timeframes if tf in data[symbol]}
                    
                    # Optimize strategy
                    optimization_result = strategy.optimize(strategy_data, metric)
                    
                    if optimization_result:
                        results[strategy_id] = optimization_result
                        
                        # Update strategy parameters if optimization succeeded
                        if 'best_params' in optimization_result:
                            strategy.update_parameters(optimization_result['best_params'])
                else:
                    logger.warning(f"Strategy {strategy.name} does not implement optimize method")
            except Exception as e:
                logger.error(f"Error optimizing strategy {strategy.name}: {e}")
                handle_error(e, f"Failed to optimize strategy {strategy.name}")
        
        return results
    
    def save_strategies(self, datastore):
        """
        Save strategies to datastore.
        
        Args:
            datastore: DataStore instance
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Serialize strategies
            serialized = {strategy_id: strategy.serialize() for strategy_id, strategy in self.strategies.items()}
            
            # Save to datastore
            datastore.save_strategies(serialized)
            
            logger.info(f"Saved {len(serialized)} strategies to datastore")
            return True
        except Exception as e:
            logger.error(f"Error saving strategies: {e}")
            handle_error(e, "Failed to save strategies")
            return False
    
    def load_strategies(self, datastore, strategy_classes):
        """
        Load strategies from datastore.
        
        Args:
            datastore: DataStore instance
            strategy_classes (dict): Dictionary of strategy class name -> class
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Load from datastore
            serialized = datastore.load_strategies()
            
            if not serialized:
                logger.warning("No strategies found in datastore")
                return False
            
            # Deserialize strategies
            for strategy_id, strategy_data in serialized.items():
                strategy_class = strategy_classes.get(strategy_data['name'])
                
                if strategy_class:
                    strategy = strategy_class.deserialize(strategy_data)
                    self.strategies[strategy_id] = strategy
                else:
                    logger.warning(f"Unknown strategy class: {strategy_data['name']}")
            
            logger.info(f"Loaded {len(self.strategies)} strategies from datastore")
            return True
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
            handle_error(e, "Failed to load strategies")
            return False