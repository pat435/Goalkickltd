"""
Data fetching module for the Goalkick Ltd Trading Bot.
Handles retrieving historical and real-time market data from exchanges.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import uuid
import threading
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.logging_config import get_logger
from config.bot_config import DATA_CONFIG, EXCHANGE_CONFIG
from config.trading_pairs import get_all_active_symbols
from src.exchange.bybit_api import BybitAPI
from src.utils.error_handling import handle_error
from src.utils.time_utils import get_timeframe_ms, timeframe_to_milliseconds
from data.datastore import DataStore

logger = get_logger("data_fetcher")

class DataFetcher:
    """Class for fetching market data from exchanges."""
    
    def __init__(self, exchange_api=None, datastore=None):
        """
        Initialize the DataFetcher.
        
        Args:
            exchange_api: Exchange API instance (if None, creates a new one)
            datastore: DataStore instance (if None, creates a new one)
        """
        self.exchange_api = exchange_api or BybitAPI()
        self.datastore = datastore or DataStore()
        self.lock = threading.RLock()
        self._fetch_tasks = {}
        self._last_fetch_time = {}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def fetch_historical_data(self, symbol, timeframe, start_time=None, end_time=None, limit=1000, use_db_cache=True):
        """
        Fetch historical market data from the exchange.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe of the data
            start_time (int): Start timestamp in milliseconds
            end_time (int): End timestamp in milliseconds
            limit (int): Maximum number of candles to fetch
            use_db_cache (bool): Whether to use cached data from the database
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            now = int(datetime.now().timestamp() * 1000)
            end_time = end_time or now
            
            # Calculate default start time based on timeframe and limit if not provided
            if not start_time:
                timeframe_ms = timeframe_to_milliseconds(timeframe)
                start_time = end_time - (timeframe_ms * limit)
            
            # Check if we have data in the database if cache is enabled
            if use_db_cache:
                db_data = self.datastore.get_market_data(
                    symbol, timeframe, start_time, end_time, limit
                )
                
                if not db_data.empty:
                    logger.debug(f"Retrieved {len(db_data)} {timeframe} candles for {symbol} from database")
                    return db_data
            
            # Fetch data from exchange
            logger.debug(f"Fetching {timeframe} data for {symbol} from exchange")
            
            # Get OHLCV data from exchange
            candles = self.exchange_api.get_candles(
                symbol, timeframe, start_time, end_time, limit
            )
            
            if not candles or len(candles) == 0:
                logger.warning(f"No {timeframe} data available for {symbol}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Ensure timestamp is in milliseconds
            if df['timestamp'].iloc[0] < 10000000000:  # If in seconds
                df['timestamp'] = df['timestamp'] * 1000
            
            # Set timestamp as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Remove duplicates and sort
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            # Save to database
            if use_db_cache and not df.empty:
                # Reset index to include timestamp column in save operation
                save_df = df.reset_index()
                self.datastore.save_market_data(symbol, timeframe, save_df)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol} {timeframe}: {e}")
            handle_error(e, f"Failed to fetch historical data for {symbol} {timeframe}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def fetch_latest_data(self, symbol, timeframe, num_candles=100):
        """
        Fetch the latest market data for a symbol and timeframe.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe of the data
            num_candles (int): Number of candles to fetch
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            # Get current time
            now = int(datetime.now().timestamp() * 1000)
            
            # Fetch data
            return self.fetch_historical_data(
                symbol, timeframe, 
                start_time=None,  # Will be calculated based on timeframe and limit
                end_time=now,
                limit=num_candles,
                use_db_cache=True
            )
        except Exception as e:
            logger.error(f"Error fetching latest data for {symbol} {timeframe}: {e}")
            handle_error(e, f"Failed to fetch latest data for {symbol} {timeframe}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def update_market_data(self, symbols=None, timeframes=None):
        """
        Update market data for specified symbols and timeframes.
        
        Args:
            symbols (list): List of symbols to update
            timeframes (list): List of timeframes to update
            
        Returns:
            dict: Summary of updated data
        """
        try:
            # Use all active symbols if not specified
            symbols = symbols or get_all_active_symbols()
            
            # Use default timeframes if not specified
            timeframes = timeframes or ['5m', '15m', '1h', '4h', '1d']
            
            update_summary = {
                'symbols_updated': 0,
                'total_candles': 0,
                'errors': 0
            }
            
            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        # Fetch latest data
                        data = self.fetch_latest_data(symbol, timeframe)
                        
                        if not data.empty:
                            update_summary['symbols_updated'] += 1
                            update_summary['total_candles'] += len(data)
                            logger.debug(f"Updated {len(data)} {timeframe} candles for {symbol}")
                    except Exception as e:
                        update_summary['errors'] += 1
                        logger.error(f"Error updating {timeframe} data for {symbol}: {e}")
                        handle_error(e, f"Failed to update {timeframe} data for {symbol}")
            
            logger.info(f"Market data update completed: {update_summary}")
            return update_summary
        except Exception as e:
            logger.error(f"Error in update_market_data: {e}")
            handle_error(e, "Failed to update market data")
            return {'symbols_updated': 0, 'total_candles': 0, 'errors': 1}
    
    def schedule_data_updates(self, symbols=None, timeframes=None, interval=60):
        """
        Schedule regular data updates for specified symbols and timeframes.
        
        Args:
            symbols (list): List of symbols to update
            timeframes (list): List of timeframes to update
            interval (int): Update interval in seconds
            
        Returns:
            str: Task ID
        """
        with self.lock:
            task_id = str(uuid.uuid4())
            
            def update_task():
                """Task function to update data periodically"""
                while True:
                    try:
                        if task_id not in self._fetch_tasks:
                            logger.info(f"Task {task_id} has been cancelled")
                            break
                        
                        logger.debug(f"Running scheduled data update (task {task_id})")
                        self.update_market_data(symbols, timeframes)
                        self._last_fetch_time[task_id] = datetime.now()
                        
                        # Sleep for the interval
                        time.sleep(interval)
                    except Exception as e:
                        logger.error(f"Error in scheduled data update task {task_id}: {e}")
                        handle_error(e, "Scheduled data update failed")
                        time.sleep(max(5, interval // 10))  # Sleep less on error
            
            # Start the task in a new thread
            self._fetch_tasks[task_id] = threading.Thread(
                target=update_task, 
                name=f"data_fetch_{task_id}",
                daemon=True
            )
            self._fetch_tasks[task_id].start()
            
            logger.info(f"Scheduled data updates with task ID {task_id}")
            return task_id
    
    def cancel_data_updates(self, task_id):
        """
        Cancel a scheduled data update task.
        
        Args:
            task_id (str): ID of the task to cancel
            
        Returns:
            bool: True if cancelled successfully, False otherwise
        """
        with self.lock:
            if task_id in self._fetch_tasks:
                # Remove task from dictionary to signal it should stop
                thread = self._fetch_tasks.pop(task_id)
                
                # Wait for thread to finish naturally (in next iteration)
                if thread.is_alive():
                    logger.info(f"Cancelling data update task {task_id}")
                    # No need to join, we just let it exit gracefully
                
                if task_id in self._last_fetch_time:
                    del self._last_fetch_time[task_id]
                
                return True
            
            logger.warning(f"Task {task_id} not found or already cancelled")
            return False
    
    def get_task_status(self, task_id):
        """
        Get the status of a scheduled task.
        
        Args:
            task_id (str): ID of the task
            
        Returns:
            dict: Task status information
        """
        with self.lock:
            if task_id not in self._fetch_tasks:
                return {'status': 'not_found', 'task_id': task_id}
            
            thread = self._fetch_tasks[task_id]
            last_fetch = self._last_fetch_time.get(task_id)
            
            return {
                'status': 'running' if thread.is_alive() else 'stopped',
                'task_id': task_id,
                'thread_name': thread.name,
                'last_fetch_time': last_fetch.isoformat() if last_fetch else None,
                'last_fetch_ago_seconds': (datetime.now() - last_fetch).total_seconds() if last_fetch else None
            }
    
    def get_all_task_statuses(self):
        """
        Get the status of all scheduled tasks.
        
        Returns:
            list: List of task status dictionaries
        """
        with self.lock:
            return [self.get_task_status(task_id) for task_id in self._fetch_tasks]
    
    def fetch_ticker_data(self, symbols=None):
        """
        Fetch current ticker data for specified symbols.
        
        Args:
            symbols (list): List of symbols to fetch ticker data for
            
        Returns:
            dict: Dictionary of symbol -> ticker data
        """
        try:
            symbols = symbols or get_all_active_symbols()
            
            # Fetch all tickers at once if exchange supports it
            if hasattr(self.exchange_api, 'get_tickers'):
                all_tickers = self.exchange_api.get_tickers()
                
                # Filter for our symbols
                return {symbol: all_tickers.get(symbol) for symbol in symbols if symbol in all_tickers}
            
            # Fetch individually if batch fetching not supported
            tickers = {}
            for symbol in symbols:
                try:
                    ticker = self.exchange_api.get_ticker(symbol)
                    if ticker:
                        tickers[symbol] = ticker
                except Exception as e:
                    logger.error(f"Error fetching ticker for {symbol}: {e}")
                    handle_error(e, f"Failed to fetch ticker for {symbol}")
            
            return tickers
        except Exception as e:
            logger.error(f"Error fetching ticker data: {e}")
            handle_error(e, "Failed to fetch ticker data")
            return {}
    
    def fetch_order_book(self, symbol, depth=10):
        """
        Fetch order book data for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            depth (int): Depth of the order book to fetch
            
        Returns:
            dict: Order book data with bids and asks
        """
        try:
            return self.exchange_api.get_order_book(symbol, depth)
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            handle_error(e, f"Failed to fetch order book for {symbol}")
            return {'bids': [], 'asks': []}
    
    def fetch_multiple_timeframes(self, symbol, timeframes, limit=100):
        """
        Fetch data for multiple timeframes for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            timeframes (list): List of timeframes to fetch
            limit (int): Number of candles per timeframe
            
        Returns:
            dict: Dictionary of timeframe -> DataFrame with data
        """
        result = {}
        for tf in timeframes:
            try:
                df = self.fetch_latest_data(symbol, tf, limit)
                if not df.empty:
                    result[tf] = df
            except Exception as e:
                logger.error(f"Error fetching {tf} data for {symbol}: {e}")
                handle_error(e, f"Failed to fetch {tf} data for {symbol}")
        
        return result
    
    def fetch_funding_rate(self, symbol):
        """
        Fetch current funding rate for a perpetual futures symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Funding rate information
        """
        try:
            if hasattr(self.exchange_api, 'get_funding_rate'):
                return self.exchange_api.get_funding_rate(symbol)
            else:
                logger.warning(f"Funding rate not supported for exchange")
                return None
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            handle_error(e, f"Failed to fetch funding rate for {symbol}")
            return None
    
    def fetch_exchange_info(self):
        """
        Fetch exchange information including trading rules.
        
        Returns:
            dict: Exchange information
        """
        try:
            return self.exchange_api.get_exchange_info()
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            handle_error(e, "Failed to fetch exchange info")
            return {}
    
    def fetch_all_historical_data(self, symbol, timeframes, days_back=30, use_db_cache=True):
        """
        Fetch historical data for multiple timeframes for backtesting or analysis.
        
        Args:
            symbol (str): Trading pair symbol
            timeframes (list): List of timeframes to fetch
            days_back (int): Number of days of historical data to fetch
            use_db_cache (bool): Whether to use cached data
            
        Returns:
            dict: Dictionary of timeframe -> DataFrame with data
        """
        result = {}
        end_time = int(datetime.now().timestamp() * 1000)
        
        for tf in timeframes:
            try:
                # Calculate limit based on days and timeframe
                tf_ms = timeframe_to_milliseconds(tf)
                days_ms = days_back * 24 * 60 * 60 * 1000
                limit = min(days_ms // tf_ms + 10, 1000)  # Add some buffer, max 1000
                
                # Calculate start time
                start_time = end_time - days_ms
                
                # Fetch data
                df = self.fetch_historical_data(
                    symbol=symbol,
                    timeframe=tf,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit,
                    use_db_cache=use_db_cache
                )
                
                if not df.empty:
                    result[tf] = df
                    logger.info(f"Fetched {len(df)} historical {tf} candles for {symbol} ({days_back} days)")
            except Exception as e:
                logger.error(f"Error fetching historical {tf} data for {symbol}: {e}")
                handle_error(e, f"Failed to fetch historical {tf} data for {symbol}")
        
        return result
    
    def __del__(self):
        """Cleanup when the object is garbage collected."""
        # Cancel all scheduled tasks
        for task_id in list(self._fetch_tasks.keys()):
            self.cancel_data_updates(task_id)