"""
Data storage and management module for the Goalkick Ltd Trading Bot.
"""

import os
import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import threading

from config.logging_config import get_logger
from config.bot_config import DATA_CONFIG
from src.utils.error_handling import handle_error
from src.utils.error_handling import DataError, DataStoreError


logger = get_logger("datastore")

class DataStore:
    """Class for handling data storage and retrieval operations."""
    
    def __init__(self, db_path=None):
        """
        Initialize the DataStore with the specified database path.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path or DATA_CONFIG["db_path"]
        self.conn = None
        self.lock = threading.RLock()
        
        # Ensure the database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize the database
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database with required tables if they don't exist."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create market data table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT,
                    timeframe TEXT,
                    timestamp INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
                ''')
                
                # Create trades table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    order_id TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    entry_time INTEGER,
                    exit_time INTEGER,
                    status TEXT,
                    pnl REAL,
                    pnl_pct REAL,
                    fees REAL,
                    strategy TEXT,
                    timeframe TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    notes TEXT
                )
                ''')
                
                # Create performance metrics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    timestamp INTEGER PRIMARY KEY,
                    equity REAL,
                    balance REAL,
                    open_positions INTEGER,
                    daily_pnl REAL,
                    weekly_pnl REAL,
                    total_pnl REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    current_drawdown REAL,
                    metrics_json TEXT
                )
                ''').venv\Scriot\activate
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS strategies (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        parameters TEXT NOT NULL,
                        enabled BOOLEAN NOT NULL DEFAULT 1,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    ''')
                
                # Create signals table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    timeframe TEXT,
                    timestamp INTEGER,
                    strategy TEXT,
                    direction TEXT,
                    strength REAL,
                    expiry INTEGER,
                    status TEXT,
                    params TEXT
                )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS strategies (
                    name TEXT PRIMARY KEY,
                    parameters TEXT NOT NULL,
                    enabled BOOLEAN NOT NULL DEFAULT 1,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Create index for strategies
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategies_enabled ON strategies (enabled)')
            
  
                # Create index for faster queries
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_tf ON market_data (symbol, timeframe)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades (entry_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals (timestamp)')
                
                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            handle_error(e, "Database initialization failed")
    
    def get_connection(self):
        """Get a database connection with thread safety."""
        if self.conn is None:
            with self.lock:
                if self.conn is None:  # Double-check pattern
                    self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def close(self):
        """Close the database connection."""
        with self.lock:
            if self.conn:
                self.conn.close()
                self.conn = None
                logger.debug("Database connection closed")
    
    def save_market_data(self, symbol, timeframe, df):
        """
        Save market data to database.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe of the data
            df (pd.DataFrame): DataFrame with OHLCV data
        """
        try:
            if df.empty:
                logger.warning(f"Empty DataFrame provided for {symbol} {timeframe}")
                return
            
            # Ensure timestamp is in milliseconds
            if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                df = df.reset_index()
            
            with self.get_connection() as conn:
                # Convert DataFrame to list of tuples for bulk insert
                data = [(
                    symbol,
                    timeframe,
                    int(row['timestamp']) if isinstance(row['timestamp'], (int, float)) 
                        else int(pd.Timestamp(row['timestamp']).timestamp() * 1000),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume'])
                ) for _, row in df.iterrows()]
                
                # Use executemany for better performance
                conn.executemany(
                    '''INSERT OR REPLACE INTO market_data 
                       (symbol, timeframe, timestamp, open, high, low, close, volume)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                    data
                )
                conn.commit()
                logger.debug(f"Saved {len(data)} {timeframe} candles for {symbol}")
        except Exception as e:
            logger.error(f"Error saving market data for {symbol} {timeframe}: {e}")
            handle_error(e, "Failed to save market data")
    
    def get_market_data(self, symbol, timeframe, start_time=None, end_time=None, limit=None):
        """
        Retrieve market data from the database.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe of the data
            start_time (int): Start timestamp in milliseconds
            end_time (int): End timestamp in milliseconds
            limit (int): Maximum number of records to return
            
        Returns:
            pd.DataFrame: DataFrame with market data
        """
        try:
            query = '''
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ? AND timeframe = ?
            '''
            params = [symbol, timeframe]
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(int(start_time))
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(int(end_time))
            
            query += ' ORDER BY timestamp ASC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(int(limit))
            
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    logger.warning(f"No market data found for {symbol} {timeframe}")
                    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Set timestamp as index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
        except Exception as e:
            logger.error(f"Error retrieving market data for {symbol} {timeframe}: {e}")
            handle_error(e, "Failed to retrieve market data")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def save_trade(self, trade_data):
        """
        Save trade information to the database.
        
        Args:
            trade_data (dict): Trade data dictionary
        """
        try:
            with self.get_connection() as conn:
                # Check if trade exists
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM trades WHERE id = ?', (trade_data['id'],))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update existing trade
                    placeholders = ', '.join([f'{k} = ?' for k in trade_data.keys() if k != 'id'])
                    values = [trade_data[k] for k in trade_data.keys() if k != 'id']
                    values.append(trade_data['id'])
                    
                    cursor.execute(f'UPDATE trades SET {placeholders} WHERE id = ?', values)
                else:
                    # Insert new trade
                    placeholders = ', '.join(['?'] * len(trade_data))
                    columns = ', '.join(trade_data.keys())
                    values = list(trade_data.values())
                    
                    cursor.execute(f'INSERT INTO trades ({columns}) VALUES ({placeholders})', values)
                
                conn.commit()
                logger.debug(f"Saved trade {trade_data['id']} for {trade_data['symbol']}")
        except Exception as e:
            logger.error(f"Error saving trade data: {e}")
            handle_error(e, "Failed to save trade data")
    
    def get_trades(self, symbol=None, status=None, start_time=None, end_time=None, limit=None):
        """
        Retrieve trades from the database.
        
        Args:
            symbol (str): Filter by trading pair symbol
            status (str): Filter by trade status
            start_time (int): Filter by entry time
            end_time (int): Filter by entry time
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of trade dictionaries
        """
        try:
            query = 'SELECT * FROM trades'
            params = []
            conditions = []
            
            if symbol:
                conditions.append('symbol = ?')
                params.append(symbol)
            
            if status:
                conditions.append('status = ?')
                params.append(status)
            
            if start_time:
                conditions.append('entry_time >= ?')
                params.append(int(start_time))
            
            if end_time:
                conditions.append('entry_time <= ?')
                params.append(int(end_time))
            
            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)
            
            query += ' ORDER BY entry_time DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(int(limit))
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert rows to dictionaries
                trades = []
                for row in rows:
                    trade = {key: row[key] for key in row.keys()}
                    trades.append(trade)
                
                return trades
        except Exception as e:
            logger.error(f"Error retrieving trades: {e}")
            handle_error(e, "Failed to retrieve trades")
            return []
    
    def save_signal(self, signal_data):
        """
        Save signal information to the database.
        
        Args:
            signal_data (dict): Signal data dictionary
        """
        try:
            # Convert params dict to JSON string
            if 'params' in signal_data and isinstance(signal_data['params'], dict):
                signal_data = signal_data.copy()
                signal_data['params'] = json.dumps(signal_data['params'])
            
            with self.get_connection() as conn:
                # Check if signal exists
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM signals WHERE id = ?', (signal_data['id'],))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update existing signal
                    placeholders = ', '.join([f'{k} = ?' for k in signal_data.keys() if k != 'id'])
                    values = [signal_data[k] for k in signal_data.keys() if k != 'id']
                    values.append(signal_data['id'])
                    
                    cursor.execute(f'UPDATE signals SET {placeholders} WHERE id = ?', values)
                else:
                    # Insert new signal
                    placeholders = ', '.join(['?'] * len(signal_data))
                    columns = ', '.join(signal_data.keys())
                    values = list(signal_data.values())
                    
                    cursor.execute(f'INSERT INTO signals ({columns}) VALUES ({placeholders})', values)
                
                conn.commit()
                logger.debug(f"Saved signal {signal_data['id']} for {signal_data['symbol']}")
        except Exception as e:
            logger.error(f"Error saving signal data: {e}")
            handle_error(e, "Failed to save signal data")
    
    def get_signals(self, symbol=None, strategy=None, direction=None, status=None, 
                   start_time=None, end_time=None, limit=None):
        """
        Retrieve signals from the database.
        
        Args:
            symbol (str): Filter by trading pair symbol
            strategy (str): Filter by strategy name
            direction (str): Filter by signal direction
            status (str): Filter by signal status
            start_time (int): Filter by timestamp
            end_time (int): Filter by timestamp
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of signal dictionaries
        """
        try:
            query = 'SELECT * FROM signals'
            params = []
            conditions = []
            
            if symbol:
                conditions.append('symbol = ?')
                params.append(symbol)
            
            if strategy:
                conditions.append('strategy = ?')
                params.append(strategy)
            
            if direction:
                conditions.append('direction = ?')
                params.append(direction)
            
            if status:
                conditions.append('status = ?')
                params.append(status)
            
            if start_time:
                conditions.append('timestamp >= ?')
                params.append(int(start_time))
            
            if end_time:
                conditions.append('timestamp <= ?')
                params.append(int(end_time))
            
            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)
            
            query += ' ORDER BY timestamp DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(int(limit))
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert rows to dictionaries
                signals = []
                for row in rows:
                    signal = {key: row[key] for key in row.keys()}
                    
                    # Parse JSON params
                    if 'params' in signal and signal['params']:
                        try:
                            signal['params'] = json.loads(signal['params'])
                        except:
                            pass
                    
                    signals.append(signal)
                
                return signals
        except Exception as e:
            logger.error(f"Error retrieving signals: {e}")
            handle_error(e, "Failed to retrieve signals")
            return []
    
    def save_performance(self, performance_data):
        """
        Save performance metrics to the database.
        
        Args:
            performance_data (dict): Performance metrics dictionary
        """
        try:
            # Convert metrics dict to JSON string
            if 'metrics_json' in performance_data and isinstance(performance_data['metrics_json'], dict):
                performance_data = performance_data.copy()
                performance_data['metrics_json'] = json.dumps(performance_data['metrics_json'])
            
            with self.get_connection() as conn:
                # Check if record exists
                cursor = conn.cursor()
                cursor.execute('SELECT timestamp FROM performance WHERE timestamp = ?', 
                              (performance_data['timestamp'],))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update existing record
                    placeholders = ', '.join([f'{k} = ?' for k in performance_data.keys() if k != 'timestamp'])
                    values = [performance_data[k] for k in performance_data.keys() if k != 'timestamp']
                    values.append(performance_data['timestamp'])
                    
                    cursor.execute(f'UPDATE performance SET {placeholders} WHERE timestamp = ?', values)
                else:
                    # Insert new record
                    placeholders = ', '.join(['?'] * len(performance_data))
                    columns = ', '.join(performance_data.keys())
                    values = list(performance_data.values())
                    
                    cursor.execute(f'INSERT INTO performance ({columns}) VALUES ({placeholders})', values)
                
                conn.commit()
                logger.debug(f"Saved performance metrics for timestamp {performance_data['timestamp']}")
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
            handle_error(e, "Failed to save performance data")
    
    def get_performance(self, start_time=None, end_time=None, limit=None):
        """
        Retrieve performance metrics from the database.
        
        Args:
            start_time (int): Filter by timestamp
            end_time (int): Filter by timestamp
            limit (int): Maximum number of records to return
            
        Returns:
            pd.DataFrame: DataFrame with performance metrics
        """
        try:
            query = 'SELECT * FROM performance'
            params = []
            
            if start_time or end_time:
                query += ' WHERE'
                
                if start_time:
                    query += ' timestamp >= ?'
                    params.append(int(start_time))
                    
                    if end_time:
                        query += ' AND'
                
                if end_time:
                    query += ' timestamp <= ?'
                    params.append(int(end_time))
            
            query += ' ORDER BY timestamp ASC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(int(limit))
            
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    logger.warning("No performance data found")
                    return pd.DataFrame(columns=['timestamp', 'equity', 'balance'])
                
                # Parse JSON metrics
                if 'metrics_json' in df.columns:
                    df['metrics_json'] = df['metrics_json'].apply(
                        lambda x: json.loads(x) if isinstance(x, str) and x else {}
                    )
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
        except Exception as e:
            logger.error(f"Error retrieving performance data: {e}")
            handle_error(e, "Failed to retrieve performance data")
            return pd.DataFrame(columns=['timestamp', 'equity', 'balance'])
    
    def clean_old_data(self, days=None):
        """
        Remove data older than specified days.
        
        Args:
            days (int): Number of days to keep data (default from config)
        """
        try:
            days = days or DATA_CONFIG["clean_older_than_days"]
            threshold = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete old market data
                cursor.execute('DELETE FROM market_data WHERE timestamp < ?', (threshold,))
                market_rows = cursor.rowcount
                
                # Delete old signals
                cursor.execute('DELETE FROM signals WHERE timestamp < ?', (threshold,))
                signal_rows = cursor.rowcount
                
                # Keep all trades for historical analysis
                
                conn.commit()
                logger.info(f"Cleaned {market_rows} old market data rows and {signal_rows} old signal rows")
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")
            handle_error(e, "Failed to clean old data")
    def save_strategies(self, strategies):
        """
        Save trading strategies to the database.
        
        Args:
            strategies (list): List of strategy configurations to save
            
        Raises:
            DataStoreError: If there's an error saving the strategies
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for strategy in strategies:
                    cursor.execute("""
                        INSERT OR REPLACE INTO strategies 
                        (id, name, parameters, enabled, last_updated)
                        VALUES (?, ?, ?, ?, datetime('now'))
                    """, (
                        strategy['id'],
                        strategy['name'],
                        json.dumps(strategy['params']),
                        strategy['active']
                    ))
                conn.commit()
                logger.info(f"Successfully saved {len(strategies)} strategies")
        except Exception as e:
            error_msg = f"Failed to save strategies: {str(e)}"
            logger.error(error_msg)
            raise DataStoreError(error_msg)

    def load_strategies(self):
        """
        Load trading strategies from the database.
        
        Returns:
            list: List of strategy configurations
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name, parameters, enabled, last_updated FROM strategies")
                rows = cursor.fetchall()
                
                strategies = []
                for row in rows:
                    strategy = {
                        'name': row['name'],
                        'parameters': json.loads(row['parameters']),
                        'enabled': row['enabled'],
                        'last_updated': row['last_updated']
                    }
                    strategies.append(strategy)
                
                logger.info("Strategies loaded successfully")
                return strategies
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
            handle_error(e, "Failed to load strategies")
            return []


    def save_model(self, model_name, model_object, metadata=None):
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model
            model_object: The trained model object
            metadata (dict): Additional metadata about the model
        """
        try:
            model_dir = Path("data/models")
            model_dir.mkdir(exist_ok=True, parents=True)
            
            model_path = model_dir / f"{model_name}.pkl"
            meta_path = model_dir / f"{model_name}_meta.json"
            
            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(model_object, f)
            
            # Save metadata
            if metadata:
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
            
            logger.info(f"Model {model_name} saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            handle_error(e, f"Failed to save model {model_name}")
            return False
    
    def load_model(self, model_name):
        """
        Load a trained model from disk.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            tuple: (model_object, metadata)
        """
        try:
            model_dir = Path("data/models")
            model_path = model_dir / f"{model_name}.pkl"
            meta_path = model_dir / f"{model_name}_meta.json"
            
            if not model_path.exists():
                logger.warning(f"Model {model_name} not found")
                return None, None
            
            # Load the model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata if available
            metadata = None
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            
            logger.info(f"Model {model_name} loaded successfully")
            return model, metadata
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            handle_error(e, f"Failed to load model {model_name}")
            return None, None
    
    def __del__(self):
        """Close the database connection when the object is deleted."""
        self.close()