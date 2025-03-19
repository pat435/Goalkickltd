"""
Backtesting engine for the Goalkick Ltd Trading Bot.
Simulates trading strategies on historical data to evaluate performance.
"""

import pandas as pd
import numpy as np
import uuid
import time
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Union, Tuple, Optional, Callable

from config.logging_config import get_logger
from config.bot_config import TRADING_CONFIG, RISK_CONFIG
from src.strategies.base_strategy import Strategy
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossManager
from analysis.backtest.performance import PerformanceAnalyzer
from src.utils.error_handling import handle_error

logger = get_logger("backtest.engine")

class BacktestEngine:
    """Main backtesting engine for simulating trading strategies."""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1% fee
        slippage: float = 0.0005,  # 0.05% slippage
        risk_per_trade: float = None,
        use_position_sizing: bool = True,
        allow_shorting: bool = True
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Starting capital for the backtest
            commission_rate: Trading fee percentage (e.g., 0.001 for 0.1%)
            slippage: Slippage percentage (e.g., 0.0005 for 0.05%)
            risk_per_trade: Risk percentage per trade (default from config)
            use_position_sizing: Whether to use position sizing or fixed trade size
            allow_shorting: Whether to allow short positions
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade or TRADING_CONFIG["risk_per_trade"]
        self.use_position_sizing = use_position_sizing
        self.allow_shorting = allow_shorting
        
        # Performance analyzer
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Backtest state
        self.reset()
    
    def reset(self):
        """Reset the backtest state."""
        self.current_capital = self.initial_capital
        self.equity = [self.initial_capital]
        self.equity_timestamps = [0]  # Will be updated with actual timestamps
        self.positions = {}  # symbol -> position info
        self.trades = []
        self.orders = []
        self.current_time = None
        self.results = None
    
    def run_backtest(
        self, 
        strategies: List[Strategy], 
        data: Dict[str, pd.DataFrame], 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        run_once: bool = False
    ) -> Dict:
        """
        Run a backtest with the given strategies and data.
        
        Args:
            strategies: List of strategy instances to test
            data: Dictionary of {symbol: dataframe} with historical price data
            start_date: Start date for backtest (str format: 'YYYY-MM-DD')
            end_date: End date for backtest (str format: 'YYYY-MM-DD')
            run_once: If True, runs a single pass through the data without time-based iteration
            
        Returns:
            dict: Backtest results
        """
        try:
            logger.info(f"Starting backtest with {len(strategies)} strategies on {len(data)} symbols")
            
            # Reset state
            self.reset()
            
            # Validate input data
            if not data:
                logger.error("No data provided for backtest")
                return {"error": "No data provided"}
            
            if not strategies:
                logger.error("No strategies provided for backtest")
                return {"error": "No strategies provided"}
            
            # Prepare data
            prepared_data = self._prepare_data(data, start_date, end_date)
            
            if not prepared_data or all(df.empty for df in prepared_data.values()):
                logger.error("No valid data after preparation")
                return {"error": "No valid data for backtest"}
            
            # Set the initial timestamp
            first_symbol = next(iter(prepared_data))
            if not prepared_data[first_symbol].empty:
                self.current_time = prepared_data[first_symbol].index[0]
                self.equity_timestamps[0] = self.current_time
            
            # Initialize position sizer
            position_sizer = MockPositionSizer(self)
            
            if run_once:
                # Run each strategy once over the entire dataset
                result = self._run_once(strategies, prepared_data, position_sizer)
            else:
                # Run time-based simulation
                result = self._run_time_based(strategies, prepared_data, position_sizer)
            
            # Calculate performance metrics
            self.results = self.performance_analyzer.calculate_metrics(
                self.trades, self.equity, self.equity_timestamps, self.initial_capital
            )
            
            logger.info(f"Backtest completed. Final capital: {self.current_capital:.2f}")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            handle_error(e, "Backtest failed")
            return {"error": str(e)}
    
    def _prepare_data(
        self, 
        data: Dict[str, pd.DataFrame], 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for backtesting.
        
        Args:
            data: Dictionary of {symbol: dataframe} with historical price data
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            dict: Prepared data
        """
        prepared_data = {}
        
        for symbol, df in data.items():
            # Make a copy to avoid modifying original data
            prepared_df = df.copy()
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in prepared_df.columns for col in required_columns):
                logger.warning(f"Missing required columns in data for {symbol}, skipping")
                continue
            
            # Ensure DataFrame has a datetime index
            if not isinstance(prepared_df.index, pd.DatetimeIndex):
                if 'timestamp' in prepared_df.columns:
                    # Convert timestamp column to datetime index
                    if isinstance(prepared_df['timestamp'].iloc[0], (int, float)):
                        prepared_df.index = pd.to_datetime(prepared_df['timestamp'], unit='ms')
                    else:
                        prepared_df.index = pd.to_datetime(prepared_df['timestamp'])
                    prepared_df = prepared_df.drop('timestamp', axis=1)
                else:
                    logger.warning(f"No datetime index or timestamp column in data for {symbol}, skipping")
                    continue
            
            # Filter by date range if specified
            if start_date:
                start_date_dt = pd.to_datetime(start_date)
                prepared_df = prepared_df[prepared_df.index >= start_date_dt]
            
            if end_date:
                end_date_dt = pd.to_datetime(end_date)
                prepared_df = prepared_df[prepared_df.index <= end_date_dt]
            
            # Sort by index
            prepared_df = prepared_df.sort_index()
            
            # Verify there's still data after filtering
            if prepared_df.empty:
                logger.warning(f"No data for {symbol} after date filtering")
                continue
            
            prepared_data[symbol] = prepared_df
        
        return prepared_data
    
    def _run_once(self, strategies, data, position_sizer):
        """
        Run a single pass of each strategy over the entire dataset.
        
        Args:
            strategies: List of strategy instances
            data: Dictionary of prepared data
            position_sizer: Position sizer instance
            
        Returns:
            dict: Results summary
        """
        for strategy in strategies:
            for symbol, df in data.items():
                if strategy.symbols and symbol not in strategy.symbols:
                    continue
                
                for timeframe in strategy.timeframes:
                    try:
                        # Generate signals for the entire dataset
                        signals = strategy.generate_signals(df, symbol, timeframe)
                        
                        if not signals:
                            continue
                        
                        # Process signals
                        for signal in signals:
                            signal_time = pd.to_datetime(signal['timestamp'], unit='ms')
                            
                            # Find the corresponding row in the data
                            if signal_time in df.index:
                                idx = df.index.get_loc(signal_time)
                                if idx + 1 < len(df):  # Make sure we have a next candle
                                    # Use the next candle's open price for entry
                                    entry_price = df['open'].iloc[idx + 1]
                                    
                                    # Apply slippage
                                    if signal['direction'] == "BUY":
                                        entry_price *= (1 + self.slippage)
                                    elif signal['direction'] == "SELL":
                                        entry_price *= (1 - self.slippage)
                                    
                                    # Calculate position size
                                    if signal['direction'] in ["BUY", "SELL"]:
                                        if self.use_position_sizing and signal.get('stop_loss'):
                                            position_size = position_sizer.calculate_position_size(
                                                symbol, 
                                                entry_price, 
                                                signal.get('stop_loss'), 
                                                self.risk_per_trade
                                            )
                                        else:
                                            # Use a fixed percentage if no stop loss
                                            position_size = (self.current_capital * self.risk_per_trade) / entry_price
                                        
                                        # Open position
                                        self._open_position(
                                            symbol, 
                                            signal['direction'], 
                                            position_size, 
                                            entry_price, 
                                            signal_time,
                                            signal.get('stop_loss'),
                                            signal.get('take_profit'),
                                            strategy.name
                                        )
                    except Exception as e:
                        logger.error(f"Error processing signal for {symbol} {timeframe}: {e}")
                        handle_error(e, f"Failed to process signal for {symbol} {timeframe}")
        
        # Close all open positions at the end of the backtest
        for symbol, position in list(self.positions.items()):
            last_price = data[symbol]['close'].iloc[-1]
            self._close_position(symbol, last_price, data[symbol].index[-1], "End of backtest")
        
        return {
            "final_capital": self.current_capital,
            "total_trades": len(self.trades),
            "profit_factor": self._calculate_profit_factor()
        }
    
    def _run_time_based(self, strategies, data, position_sizer):
        """
        Run a time-based simulation, processing data chronologically.
        
        Args:
            strategies: List of strategy instances
            data: Dictionary of prepared data
            position_sizer: Position sizer instance
            
        Returns:
            dict: Results summary
        """
        # Combine all timestamps from all dataframes
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index)
        
        # Sort timestamps
        all_timestamps = sorted(all_timestamps)
        
        if not all_timestamps:
            logger.error("No timestamps found in data")
            return {"error": "No timestamps found in data"}
        
        # Process each timestamp
        for timestamp in all_timestamps:
            self.current_time = timestamp
            
            # Update equity curve
            self._update_equity_curve()
            
            # Check for stop losses and take profits on open positions
            self._check_position_exits(data, timestamp)
            
            # Generate and process signals for each strategy
            for strategy in strategies:
                for symbol, df in data.items():
                    if strategy.symbols and symbol not in strategy.symbols:
                        continue
                    
                    if timestamp not in df.index:
                        continue
                    
                    for timeframe in strategy.timeframes:
                        try:
                            # Get data up to current timestamp
                            current_idx = df.index.get_loc(timestamp)
                            historical_data = df.iloc[:current_idx+1].copy()
                            
                            # Skip if not enough data
                            if len(historical_data) < 50:  # Arbitrary minimum
                                continue
                            
                            # Generate signals
                            signals = strategy.generate_signals(historical_data, symbol, timeframe)
                            
                            # Process signals
                            for signal in signals:
                                # Only process fresh signals
                                signal_time = pd.to_datetime(signal['timestamp'], unit='ms')
                                if signal_time == timestamp:
                                    # Use current bar's close price for simulation
                                    # In real trading, you'd use the next bar's open
                                    entry_price = historical_data['close'].iloc[-1]
                                    
                                    # Apply slippage
                                    if signal['direction'] == "BUY":
                                        entry_price *= (1 + self.slippage)
                                    elif signal['direction'] == "SELL":
                                        entry_price *= (1 - self.slippage)
                                    
                                    # Check if we should enter a position
                                    should_enter = self._should_enter_position(symbol, signal['direction'])
                                    
                                    if should_enter and signal['direction'] in ["BUY", "SELL"]:
                                        # Calculate position size
                                        if self.use_position_sizing and signal.get('stop_loss'):
                                            position_size = position_sizer.calculate_position_size(
                                                symbol, 
                                                entry_price, 
                                                signal.get('stop_loss'), 
                                                self.risk_per_trade
                                            )
                                        else:
                                            # Use a fixed percentage if no stop loss
                                            position_size = (self.current_capital * self.risk_per_trade) / entry_price
                                        
                                        # Open position
                                        self._open_position(
                                            symbol, 
                                            signal['direction'], 
                                            position_size, 
                                            entry_price, 
                                            timestamp,
                                            signal.get('stop_loss'),
                                            signal.get('take_profit'),
                                            strategy.name
                                        )
                        except Exception as e:
                            logger.error(f"Error processing {symbol} {timeframe} at {timestamp}: {e}")
                            handle_error(e, f"Failed to process {symbol} {timeframe}")
        
        # Close all remaining positions at the end of the backtest
        for symbol, position in list(self.positions.items()):
            last_price = data[symbol]['close'].iloc[-1]
            self._close_position(symbol, last_price, all_timestamps[-1], "End of backtest")
        
        return {
            "final_capital": self.current_capital,
            "total_trades": len(self.trades),
            "profit_factor": self._calculate_profit_factor()
        }
    
    def _should_enter_position(self, symbol, direction):
        """
        Determine if a new position should be entered.
        
        Args:
            symbol: Trading pair symbol
            direction: Signal direction ("BUY" or "SELL")
            
        Returns:
            bool: True if should enter, False otherwise
        """
        # Check if we already have a position in this symbol
        if symbol in self.positions:
            existing_position = self.positions[symbol]
            
            # Allow reversing positions (close existing and open new)
            if (direction == "BUY" and existing_position['direction'] == "SELL") or \
               (direction == "SELL" and existing_position['direction'] == "BUY"):
                return True
            
            # Don't enter if we already have a position in the same direction
            if (direction == "BUY" and existing_position['direction'] == "BUY") or \
               (direction == "SELL" and existing_position['direction'] == "SELL"):
                return False
        
        # Check if we have enough capital
        if self.current_capital <= 0:
            return False
        
        # Check if shorting is allowed for sell signals
        if direction == "SELL" and not self.allow_shorting:
            return False
        
        return True
    
    def _open_position(
        self, 
        symbol, 
        direction, 
        size, 
        price, 
        timestamp, 
        stop_loss=None, 
        take_profit=None, 
        strategy_name="unknown"
    ):
        """
        Open a new position.
        
        Args:
            symbol: Trading pair symbol
            direction: "BUY" or "SELL"
            size: Position size in base currency
            price: Entry price
            timestamp: Entry timestamp
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy_name: Name of the strategy
        """
        # Close existing position if there is one
        if symbol in self.positions:
            self._close_position(symbol, price, timestamp, "Reversed position")
        
        # Calculate position value
        position_value = size * price
        
        # Ensure we have enough capital
        if position_value > self.current_capital:
            size = self.current_capital / price
            position_value = size * price
        
        # Apply commission
        commission = position_value * self.commission_rate
        self.current_capital -= commission
        
        # Record the order
        order_id = str(uuid.uuid4())
        order = {
            'id': order_id,
            'symbol': symbol,
            'direction': direction,
            'size': size,
            'price': price,
            'value': position_value,
            'commission': commission,
            'timestamp': timestamp,
            'strategy': strategy_name
        }
        self.orders.append(order)
        
        # Create the position
        position = {
            'symbol': symbol,
            'direction': direction,
            'size': size,
            'entry_price': price,
            'entry_time': timestamp,
            'current_price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy': strategy_name,
            'order_id': order_id
        }
        self.positions[symbol] = position
        
        # Update capital (for short positions, we don't reduce capital since we're selling)
        if direction == "BUY":
            self.current_capital -= position_value
        
        logger.debug(f"Opened {direction} position in {symbol}: {size} @ {price}")
    
    def _close_position(self, symbol, price, timestamp, reason=""):
        """
        Close an existing position.
        
        Args:
            symbol: Trading pair symbol
            price: Exit price
            timestamp: Exit timestamp
            reason: Reason for closing the position
        """
        if symbol not in self.positions:
            return
        
        position = self.positions.pop(symbol)
        
        # Calculate PnL
        if position['direction'] == "BUY":
            # Long position: sell at current price
            exit_value = position['size'] * price
            entry_value = position['size'] * position['entry_price']
            pnl = exit_value - entry_value
        else:
            # Short position: buy back at current price
            exit_value = position['size'] * price
            entry_value = position['size'] * position['entry_price']
            pnl = entry_value - exit_value
        
        # Apply commission
        commission = exit_value * self.commission_rate
        pnl -= commission
        
        # Update capital
        if position['direction'] == "BUY":
            # For long positions, we get back our position value + PnL
            self.current_capital += exit_value
        else:
            # For short positions, we reduce our capital by the exit value and add the PnL
            self.current_capital -= exit_value
            self.current_capital += pnl
        
        # Record the trade
        trade = {
            'id': str(uuid.uuid4()),
            'symbol': symbol,
            'direction': position['direction'],
            'size': position['size'],
            'entry_price': position['entry_price'],
            'exit_price': price,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'pnl': pnl,
            'pnl_pct': pnl / entry_value,
            'commission': commission,
            'strategy': position['strategy'],
            'stop_loss': position.get('stop_loss'),
            'take_profit': position.get('take_profit'),
            'holding_period': (timestamp - position['entry_time']).total_seconds() / 3600,  # hours
            'order_id': position.get('order_id'),
            'reason': reason
        }
        self.trades.append(trade)
        
        logger.debug(f"Closed {position['direction']} position in {symbol}: {position['size']} @ {price}, PnL: {pnl:.2f}")
    
    def _check_position_exits(self, data, timestamp):
        """
        Check if any positions should be closed due to stop loss or take profit.
        
        Args:
            data: Dictionary of price data
            timestamp: Current timestamp
        """
        for symbol, position in list(self.positions.items()):
            if symbol not in data or timestamp not in data[symbol].index:
                continue
            
            current_bar = data[symbol].loc[timestamp]
            position['current_price'] = current_bar['close']
            
            # Check if stop loss or take profit hit
            if position['stop_loss'] is not None:
                if position['direction'] == "BUY" and current_bar['low'] <= position['stop_loss']:
                    # Stop loss hit for long position
                    # For simulation accuracy, use the stop loss price, not the close
                    self._close_position(symbol, position['stop_loss'], timestamp, "Stop loss")
                    continue
                elif position['direction'] == "SELL" and current_bar['high'] >= position['stop_loss']:
                    # Stop loss hit for short position
                    self._close_position(symbol, position['stop_loss'], timestamp, "Stop loss")
                    continue
            
            if position['take_profit'] is not None:
                if position['direction'] == "BUY" and current_bar['high'] >= position['take_profit']:
                    # Take profit hit for long position
                    self._close_position(symbol, position['take_profit'], timestamp, "Take profit")
                    continue
                elif position['direction'] == "SELL" and current_bar['low'] <= position['take_profit']:
                    # Take profit hit for short position
                    self._close_position(symbol, position['take_profit'], timestamp, "Take profit")
                    continue
    
    def _update_equity_curve(self):
        """Update the equity curve with the current equity value."""
        # Calculate current equity
        current_equity = self.current_capital
        
        # Add unrealized PnL from open positions
        for symbol, position in self.positions.items():
            if position['direction'] == "BUY":
                # Long position
                unrealized_pnl = position['size'] * (position['current_price'] - position['entry_price'])
            else:
                # Short position
                unrealized_pnl = position['size'] * (position['entry_price'] - position['current_price'])
            
            current_equity += unrealized_pnl
        
        # Add to equity curve
        self.equity.append(current_equity)
        self.equity_timestamps.append(self.current_time)
    
    def _calculate_profit_factor(self):
        """Calculate the profit factor (gross profit / gross loss)."""
        if not self.trades:
            return 0
        
        gross_profit = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0)
        gross_loss = sum(abs(trade['pnl']) for trade in self.trades if trade['pnl'] < 0)
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    def get_equity_curve(self):
        """
        Get the equity curve as a DataFrame.
        
        Returns:
            pd.DataFrame: Equity curve with timestamp index
        """
        if not self.equity or not self.equity_timestamps:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'equity': self.equity,
            'timestamp': self.equity_timestamps
        })
        
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_trade_history(self):
        """
        Get the trade history as a DataFrame.
        
        Returns:
            pd.DataFrame: Trade history
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_order_history(self):
        """
        Get the order history as a DataFrame.
        
        Returns:
            pd.DataFrame: Order history
        """
        if not self.orders:
            return pd.DataFrame()
        
        return pd.DataFrame(self.orders)
    
    def get_results_summary(self):
        """
        Get a summary of the backtest results.
        
        Returns:
            dict: Summary of backtest results
        """
        if not self.results:
            return {"error": "No results available"}
        
        return self.results

    def plot_equity_curve(self, save_path=None):
        """
        Plot the equity curve.
        
        Args:
            save_path: Path to save the plot (optional)
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            equity_df = self.get_equity_curve()
            
            if equity_df.empty:
                logger.warning("No equity data to plot")
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(equity_df.index, equity_df['equity'], label='Equity', color='blue')
            
            # Add initial capital reference line
            ax.axhline(y=self.initial_capital, color='red', linestyle='--', label='Initial Capital')
            
            # Format the plot
            ax.set_title('Equity Curve')
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity')
            ax.legend()
            ax.grid(True)
            
            # Format date axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            
            return fig
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot")
            return None
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            handle_error(e, "Failed to plot equity curve")
            return None


class MockPositionSizer:
    """Mock position sizer for backtesting."""
    
    def __init__(self, backtest_engine):
        """
        Initialize the mock position sizer.
        
        Args:
            backtest_engine: BacktestEngine instance
        """
        self.backtest_engine = backtest_engine
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_price, risk_percentage=None):
        """
        Calculate position size based on risk percentage and stop loss.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_percentage: Risk percentage (optional)
            
        Returns:
            float: Position size in base currency
        """
        if risk_percentage is None:
            risk_percentage = self.backtest_engine.risk_per_trade
        
        # Calculate risk amount in quote currency
        risk_amount = self.backtest_engine.current_capital * risk_percentage
        
        # Calculate stop loss distance in percentage
        stop_loss_pct = abs(entry_price - stop_loss_price) / entry_price
        
        if stop_loss_pct == 0:
            logger.warning(f"Stop loss percentage is zero for {symbol}, using default 1%")
            stop_loss_pct = 0.01
        
        # Calculate position size in quote currency
        position_value = risk_amount / stop_loss_pct
        
        # Calculate quantity in base currency
        quantity = position_value / entry_price
        
        return quantity


class BacktestRunner:
    """Utility class for running multiple backtests and comparing results."""
    
    def __init__(self):
        """Initialize the backtest runner."""
        self.backtest_results = {}
    
    def run_backtest_suite(
        self, 
        strategies_list: List[List[Strategy]], 
        data: Dict[str, pd.DataFrame], 
        strategy_names: List[str] = None,
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        **backtest_params
    ) -> Dict:
        """
        Run multiple backtests and compare their performance.
        
        Args:
            strategies_list: List of strategy lists to test
            data: Dictionary of {symbol: dataframe} with historical price data
            strategy_names: Names for each strategy set (optional)
            start_date: Start date for backtest
            end_date: End date for backtest
            **backtest_params: Additional parameters for BacktestEngine
            
        Returns:
            dict: Comparison of backtest results
        """
        try:
            if not strategy_names:
                strategy_names = [f"Strategy Set {i}" for i in range(len(strategies_list))]
            
            if len(strategy_names) != len(strategies_list):
                strategy_names = [f"Strategy Set {i}" for i in range(len(strategies_list))]
            
            results = {}
            
            for i, strategies in enumerate(strategies_list):
                name = strategy_names[i]
                
                logger.info(f"Running backtest for {name}")
                
                # Create backtest engine
                backtest_engine = BacktestEngine(**backtest_params)
                
                # Run backtest
                result = backtest_engine.run_backtest(
                    strategies, data, start_date, end_date
                )
                
                # Store results
                results[name] = result
                self.backtest_results[name] = {
                    'result': result,
                    'engine': backtest_engine
                }
            
            # Compare results
            comparison = self._compare_results(results)
            return comparison
        except Exception as e:
            logger.error(f"Error running backtest suite: {e}")
            handle_error(e, "Backtest suite failed")
            return {"error": str(e)}
    
    def _compare_results(self, results):
        """
        Compare the results of multiple backtests.
        
        Args:
            results: Dictionary of backtest results
            
        Returns:
            dict: Comparison metrics
        """
        if not results:
            return {}
        
        # Extract key metrics for comparison
        comparison = {
            'summary': {},
            'metrics': {}
        }
        
        # Initialize metric categories
        metrics_categories = [
            'returns', 'risk', 'ratios', 'trade_stats', 'drawdowns'
        ]
        
        for category in metrics_categories:
            comparison['metrics'][category] = {}
        
        # Extract and organize metrics
        for name, result in results.items():
            # Overall summary
            comparison['summary'][name] = {
                'final_equity': result.get('final_equity', 0),
                'total_return_pct': result.get('total_return_pct', 0),
                'annual_return_pct': result.get('annual_return_pct', 0),
                'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'total_trades': result.get('total_trades', 0),
                'win_rate': result.get('win_rate', 0),
                'profit_factor': result.get('profit_factor', 0)
            }
            
            # Detailed metrics by category
            for category in metrics_categories:
                if category in result:
                    for metric, value in result[category].items():
                        if metric not in comparison['metrics'][category]:
                            comparison['metrics'][category][metric] = {}
                        comparison['metrics'][category][metric][name] = value
        
        # Determine best strategy by key metrics
        comparison['best_strategy'] = self._find_best_strategy(comparison['summary'])
        
        return comparison
    
    def _find_best_strategy(self, summary):
        """
        Find the best performing strategy based on key metrics.
        
        Args:
            summary: Summary of backtest results
            
        Returns:
            dict: Best strategy by different metrics
        """
        if not summary:
            return {}
        
        best = {
            'by_return': max(summary.items(), key=lambda x: x[1]['total_return_pct'])[0],
            'by_sharpe': max(summary.items(), key=lambda x: x[1]['sharpe_ratio'])[0],
            'by_profit_factor': max(summary.items(), key=lambda x: x[1]['profit_factor'])[0],
            'by_win_rate': max(summary.items(), key=lambda x: x[1]['win_rate'])[0],
            'by_drawdown': min(summary.items(), key=lambda x: x[1]['max_drawdown_pct'])[0],
            'overall': None
        }
        
        # Simple scoring system for overall best
        scores = {name: 0 for name in summary.keys()}
        
        for name in summary.keys():
            if name == best['by_return']:
                scores[name] += 3
            if name == best['by_sharpe']:
                scores[name] += 3
            if name == best['by_profit_factor']:
                scores[name] += 2
            if name == best['by_win_rate']:
                scores[name] += 1
            if name == best['by_drawdown']:
                scores[name] += 2
        
        best['overall'] = max(scores.items(), key=lambda x: x[1])[0]
        
        return best
    
    def plot_equity_comparison(self, save_path=None):
        """
        Plot a comparison of equity curves.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if not self.backtest_results:
                logger.warning("No backtest results to plot")
                return None
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Plot initial capital reference
            first_engine = next(iter(self.backtest_results.values()))['engine']
            initial_capital = first_engine.initial_capital
            ax.axhline(y=initial_capital, color='black', linestyle='--', label='Initial Capital')
            
            # Plot equity curves
            for name, result in self.backtest_results.items():
                engine = result['engine']
                equity_df = engine.get_equity_curve()
                
                if not equity_df.empty:
                    ax.plot(equity_df.index, equity_df['equity'], label=name)
            
            # Format the plot
            ax.set_title('Equity Curve Comparison')
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity')
            ax.legend()
            ax.grid(True)
            
            # Format date axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            
            return fig
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot")
            return None
        except Exception as e:
            logger.error(f"Error plotting equity comparison: {e}")
            handle_error(e, "Failed to plot equity comparison")
            return None