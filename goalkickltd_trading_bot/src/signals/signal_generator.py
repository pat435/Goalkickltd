"""
Signal generator module for the Goalkick Ltd Trading Bot.
Orchestrates signal generation from multiple strategies.
"""

import pandas as pd
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from config.logging_config import get_logger
from config.strategy_params import get_symbol_strategy
from src.utils.error_handling import handle_error
from src.strategies.base_strategy import StrategyManager
from src.strategies.trend_following import MovingAverageCrossStrategy, MACDStrategy, ParabolicSARStrategy, ADXTrendStrategy, IchimokuStrategy
from src.strategies.mean_reversion import RSIStrategy, BollingerBandsStrategy
from src.strategies.arbitrage import TriangularArbitrageStrategy, StatisticalArbitrageStrategy
from src.strategies.statistical import LinearRegressionStrategy, MachineLearningStrategy, KalmanFilterStrategy

logger = get_logger("signals.generator")

class SignalGenerator:
    """Class for generating and managing trading signals."""
    
    def __init__(self, data_fetcher, datastore):
        """
        Initialize the SignalGenerator.
        
        Args:
            data_fetcher: DataFetcher instance
            datastore: DataStore instance
        """
        self.data_fetcher = data_fetcher
        self.datastore = datastore
        self.strategy_manager = StrategyManager()
        self.strategy_classes = self._get_strategy_classes()
        self.lock = threading.RLock()
        self.active_signals = {}  # signal_id -> signal
        
        # Load strategies
        self._load_strategies()
    
    def _get_strategy_classes(self):
        """
        Get all available strategy classes.
        
        Returns:
            dict: Strategy name -> class
        """
        return {
            'MovingAverageCross': MovingAverageCrossStrategy,
            'MACD': MACDStrategy,
            'ParabolicSAR': ParabolicSARStrategy,
            'ADXTrend': ADXTrendStrategy,
            'Ichimoku': IchimokuStrategy,
            'RSI': RSIStrategy,
            'BollingerBands': BollingerBandsStrategy,
            'TriangularArbitrage': TriangularArbitrageStrategy,
            'StatisticalArbitrage': StatisticalArbitrageStrategy,
            'LinearRegression': LinearRegressionStrategy,
            'MachineLearning': MachineLearningStrategy,
            'KalmanFilter': KalmanFilterStrategy
        }
    
    def _load_strategies(self):
        """Load strategies from configuration or datastore."""
        try:
            # Try to load from datastore
            loaded = self.strategy_manager.load_strategies(self.datastore, self.strategy_classes)
            
            if not loaded or not self.strategy_manager.get_all_strategies():
                # If no strategies loaded, create default set
                self._create_default_strategies()
                
                # Save to datastore
                self.strategy_manager.save_strategies(self.datastore)
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
            handle_error(e, "Failed to load strategies")
            
            # Create default strategies as fallback
            self._create_default_strategies()
    
    def _create_default_strategies(self):
        """Create default set of strategies."""
        try:
            # Create trend following strategies
            self.strategy_manager.add_strategy(MovingAverageCrossStrategy(
                timeframes=['15m', '1h', '4h'],
                symbols=None
            ))
            
            self.strategy_manager.add_strategy(MACDStrategy(
                timeframes=['1h', '4h'],
                symbols=None
            ))
            
            self.strategy_manager.add_strategy(IchimokuStrategy(
                timeframes=['4h', '1d'],
                symbols=None
            ))
            
            # Create mean reversion strategies
            self.strategy_manager.add_strategy(RSIStrategy(
                timeframes=['1h', '4h'],
                symbols=None
            ))
            
            self.strategy_manager.add_strategy(BollingerBandsStrategy(
                timeframes=['1h', '4h'],
                symbols=None
            ))
            
            # Create statistical strategies
            self.strategy_manager.add_strategy(KalmanFilterStrategy(
                timeframes=['1h'],
                symbols=None
            ))
            
            logger.info("Created default strategy set")
        except Exception as e:
            logger.error(f"Error creating default strategies: {e}")
            handle_error(e, "Failed to create default strategies")
    
    def generate_signals(self, symbols=None, timeframes=None):
        """
        Generate trading signals for specified symbols and timeframes.
        
        Args:
            symbols (list): List of symbols to generate signals for
            timeframes (list): List of timeframes to use
            
        Returns:
            list: List of generated signals
        """
        try:
            # Use configured symbols if not specified
            if symbols is None:
                from config.trading_pairs import get_all_active_symbols
                symbols = get_all_active_symbols()
            
            # Use default timeframes if not specified
            if timeframes is None:
                from config.bot_config import DEFAULT_TIMEFRAMES
                timeframes = DEFAULT_TIMEFRAMES
            
            signals = []
            
            # Use thread pool for parallel signal generation
            with ThreadPoolExecutor(max_workers=min(10, len(symbols) * len(timeframes))) as executor:
                # Submit tasks
                futures = []
                
                for symbol in symbols:
                    for timeframe in timeframes:
                        futures.append(
                            executor.submit(self._generate_signals_for_symbol_timeframe, symbol, timeframe)
                        )
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        symbol_signals = future.result()
                        if symbol_signals:
                            signals.extend(symbol_signals)
                    except Exception as e:
                        logger.error(f"Error in signal generation task: {e}")
                        handle_error(e, "Signal generation task failed")
            
            # Save signals to datastore
            if signals:
                for signal in signals:
                    self.datastore.save_signal(signal)
                    
                    # Add to active signals
                    with self.lock:
                        self.active_signals[signal['id']] = signal
                
                logger.info(f"Generated {len(signals)} signals")
            
            return signals
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            handle_error(e, "Failed to generate signals")
            return []
    
    def _generate_signals_for_symbol_timeframe(self, symbol, timeframe):
        """
        Generate signals for a specific symbol and timeframe.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            
        Returns:
            list: Generated signals
        """
        try:
            # Get strategies for this symbol and timeframe
            strategies = self.strategy_manager.get_strategies_for_symbol(symbol)
            strategies = [s for s in strategies if timeframe in s.timeframes]
            
            if not strategies:
                return []
            
            # Get historical data
            data = self.data_fetcher.fetch_latest_data(symbol, timeframe)
            
            if data.empty:
                logger.warning(f"No data for {symbol} {timeframe}")
                return []
            
            # Generate signals for each strategy
            signals = []
            
            for strategy in strategies:
                try:
                    strategy_signals = strategy.generate_signals(data, symbol, timeframe)
                    if strategy_signals:
                        signals.extend(strategy_signals)
                except Exception as e:
                    logger.error(f"Error generating signals for strategy {strategy.name}: {e}")
                    handle_error(e, f"Failed to generate signals for strategy {strategy.name}")
            
            return signals
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {e}")
            handle_error(e, f"Failed to generate signals for {symbol} {timeframe}")
            return []
    
    def get_active_signals(self, symbol=None, max_age=None):
        """
        Get active signals, optionally filtered by symbol and age.
        
        Args:
            symbol (str): Filter by symbol
            max_age (float): Maximum age in seconds
            
        Returns:
            list: List of active signals
        """
        try:
            # Get current time
            current_time = datetime.now().timestamp() * 1000
            
            with self.lock:
                # Filter by symbol and age
                filtered = []
                
                for signal_id, signal in self.active_signals.items():
                    # Check expiry
                    if 'expiry' in signal and signal['expiry'] < current_time:
                        # Signal expired
                        continue
                    
                    # Check symbol
                    if symbol and signal.get('symbol') != symbol:
                        continue
                    
                    # Check age
                    if max_age:
                        signal_age = (current_time - signal.get('timestamp', 0)) / 1000
                        if signal_age > max_age:
                            continue
                    
                    filtered.append(signal)
                
                return filtered
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            handle_error(e, "Failed to get active signals")
            return []
    
    def update_signal_status(self, signal_id, new_status):
        """
        Update the status of a signal.
        
        Args:
            signal_id (str): Signal ID
            new_status (str): New status
            
        Returns:
            bool: True if updated, False otherwise
        """
        try:
            with self.lock:
                if signal_id in self.active_signals:
                    signal = self.active_signals[signal_id]
                    signal['status'] = new_status
                    
                    # Save to datastore
                    self.datastore.save_signal(signal)
                    
                    # Remove if final status
                    if new_status in ['EXECUTED', 'EXPIRED', 'CANCELLED']:
                        self.active_signals.pop(signal_id)
                    
                    logger.debug(f"Updated signal {signal_id} status to {new_status}")
                    return True
                else:
                    logger.warning(f"Signal {signal_id} not found")
                    return False
        except Exception as e:
            logger.error(f"Error updating signal status: {e}")
            handle_error(e, "Failed to update signal status")
            return False
    
    def create_manual_signal(self, symbol, direction, price=None, stop_loss=None, take_profit=None, timeframe='1h'):
        """
        Create a manual trading signal.
        
        Args:
            symbol (str): Trading pair symbol
            direction (str): Signal direction ("BUY", "SELL")
            price (float): Signal price
            stop_loss (float): Stop loss price
            take_profit (float): Take profit price
            timeframe (str): Timeframe
            
        Returns:
            dict: Created signal
        """
        try:
            # Get current price if not provided
            if price is None:
                ticker = self.data_fetcher.fetch_ticker_data([symbol]).get(symbol)
                if ticker:
                    price = float(ticker.get('lastPrice', 0))
            
            # Create signal
            signal_id = str(uuid.uuid4())
            timestamp = int(datetime.now().timestamp() * 1000)
            
            signal = {
                'id': signal_id,
                'timestamp': timestamp,
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy': 'Manual',
                'direction': direction,
                'strength': 1.0,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'expiry': timestamp + 24 * 60 * 60 * 1000,  # 24-hour expiry
                'status': 'NEW',
                'metadata': {
                    'manual': True,
                    'created_at': datetime.now().isoformat()
                }
            }
            
            # Save to datastore
            self.datastore.save_signal(signal)
            
            # Add to active signals
            with self.lock:
                self.active_signals[signal_id] = signal
            
            logger.info(f"Created manual {direction} signal for {symbol} at {price}")
            return signal
        except Exception as e:
            logger.error(f"Error creating manual signal: {e}")
            handle_error(e, "Failed to create manual signal")
            return None
    
    def cancel_signal(self, signal_id):
        """
        Cancel a signal.
        
        Args:
            signal_id (str): Signal ID
            
        Returns:
            bool: True if cancelled, False otherwise
        """
        return self.update_signal_status(signal_id, 'CANCELLED')
    
    def get_signal(self, signal_id):
        """
        Get a signal by ID.
        
        Args:
            signal_id (str): Signal ID
            
        Returns:
            dict: Signal or None if not found
        """
        with self.lock:
            # Check active signals
            if signal_id in self.active_signals:
                return self.active_signals[signal_id]
        
        # Check datastore
        signals = self.datastore.get_signals(id=signal_id)
        return signals[0] if signals else None
    
    def clean_expired_signals(self):
        """
        Remove expired signals.
        
        Returns:
            int: Number of removed signals
        """
        try:
            # Get current time
            current_time = datetime.now().timestamp() * 1000
            
            # Find expired signals
            expired = []
            
            with self.lock:
                for signal_id, signal in list(self.active_signals.items()):
                    if 'expiry' in signal and signal['expiry'] < current_time:
                        # Signal expired
                        expired.append(signal_id)
                        
                        # Update status
                        signal['status'] = 'EXPIRED'
                        self.datastore.save_signal(signal)
                        
                        # Remove from active signals
                        self.active_signals.pop(signal_id)
            
            if expired:
                logger.info(f"Removed {len(expired)} expired signals")
            
            return len(expired)
        except Exception as e:
            logger.error(f"Error cleaning expired signals: {e}")
            handle_error(e, "Failed to clean expired signals")
            return 0