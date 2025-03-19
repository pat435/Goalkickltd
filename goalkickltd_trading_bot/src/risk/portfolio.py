"""
Portfolio management module for the Goalkick Ltd Trading Bot.
Handles overall portfolio risk, position tracking, and performance metrics.
"""

import time
import uuid
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import threading

from config.logging_config import get_logger
from config.bot_config import TRADING_CONFIG, RISK_CONFIG, PERFORMANCE_CONFIG
from config.trading_pairs import get_all_active_symbols
from src.utils.error_handling import handle_error

logger = get_logger("risk.portfolio")

class PortfolioManager:
    """Class for managing portfolio risk and tracking performance."""
    
    def __init__(self, account_manager, order_manager, datastore):
        """
        Initialize the PortfolioManager.
        
        Args:
            account_manager: Account Manager instance
            order_manager: Order Manager instance
            datastore: DataStore instance
        """
        self.account_manager = account_manager
        self.order_manager = order_manager
        self.datastore = datastore
        self.positions = {}  # Symbol -> position info
        self.open_trades = {}  # Trade ID -> trade info
        self.trade_history = []
        self.performance_metrics = {}
        self.equity_curve = []
        self.daily_pnl = 0
        self.drawdowns = []
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.last_update_time = 0
        self.lock = threading.RLock()
    
    def update_positions(self):
        """
        Update positions from the exchange.
        
        Returns:
            dict: Updated positions
        """
        try:
            # Get current positions from the exchange
            exchange_positions = self.account_manager.get_all_positions()
            
            with self.lock:
                # Clear existing positions
                self.positions = {}
                
                # Add current positions
                for position in exchange_positions:
                    symbol = position['symbol']
                    self.positions[symbol] = position
                
                # Update open trades with position info
                for trade_id, trade in list(self.open_trades.items()):
                    symbol = trade['symbol']
                    
                    if symbol in self.positions:
                        # Update trade with current position info
                        trade['current_price'] = self.positions[symbol]['mark_price']
                        trade['unrealized_pnl'] = self.calculate_unrealized_pnl(trade)
                        trade['unrealized_pnl_pct'] = self.calculate_unrealized_pnl_percentage(trade)
                        trade['duration'] = (datetime.now() - datetime.fromtimestamp(trade['entry_time'] / 1000)).total_seconds() / 3600  # Hours
                    else:
                        # Position closed, move to history
                        logger.info(f"Position for trade {trade_id} ({symbol}) has been closed outside the bot")
                        self.close_trade(trade_id, "Position closed externally")
            
            return self.positions
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            handle_error(e, "Failed to update positions")
            return {}
    
    def open_trade(self, symbol, side, entry_price, quantity, strategy, timeframe, stop_loss=None, take_profit=None):
        """
        Record a new trade entry.
        
        Args:
            symbol (str): Trading pair symbol
            side (str): Trade side ("LONG" or "SHORT")
            entry_price (float): Entry price
            quantity (float): Trade quantity
            strategy (str): Strategy name
            timeframe (str): Timeframe
            stop_loss (float): Stop loss price
            take_profit (float): Take profit price
            
        Returns:
            str: Trade ID
        """
        try:
            trade_id = str(uuid.uuid4())
            timestamp = int(time.time() * 1000)
            
            trade = {
                'id': trade_id,
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'quantity': quantity,
                'value': entry_price * quantity,
                'strategy': strategy,
                'timeframe': timeframe,
                'entry_time': timestamp,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'current_price': entry_price,
                'unrealized_pnl': 0,
                'unrealized_pnl_pct': 0,
                'status': 'OPEN',
                'exit_price': None,
                'exit_time': None,
                'realized_pnl': None,
                'realized_pnl_pct': None,
                'duration': 0,
                'notes': "",
                'tags': []
            }
            
            with self.lock:
                self.open_trades[trade_id] = trade
                
                # Save to datastore
                self.datastore.save_trade({
                    'id': trade_id,
                    'symbol': symbol,
                    'order_id': "",
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': None,
                    'quantity': quantity,
                    'entry_time': timestamp,
                    'exit_time': None,
                    'status': 'OPEN',
                    'pnl': None,
                    'pnl_pct': None,
                    'fees': 0,
                    'strategy': strategy,
                    'timeframe': timeframe,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'notes': ""
                })
            
            logger.info(f"Opened trade {trade_id}: {side} {quantity} {symbol} @ {entry_price}")
            return trade_id
        except Exception as e:
            logger.error(f"Error opening trade: {e}")
            handle_error(e, "Failed to open trade")
            return None
    
    def close_trade(self, trade_id, notes=""):
        """
        Record a trade exit.
        
        Args:
            trade_id (str): Trade ID
            notes (str): Additional notes
            
        Returns:
            dict: Closed trade info
        """
        try:
            with self.lock:
                if trade_id not in self.open_trades:
                    logger.warning(f"Trade {trade_id} not found in open trades")
                    return None
                
                trade = self.open_trades[trade_id]
                timestamp = int(time.time() * 1000)
                
                # Get current price (mark price from position or last trade price)
                symbol = trade['symbol']
                current_price = None
                
                if symbol in self.positions:
                    current_price = self.positions[symbol]['mark_price']
                else:
                    # Try to get ticker price
                    ticker = self.account_manager.exchange_api.get_ticker(symbol)
                    if ticker:
                        current_price = float(ticker.get('lastPrice', 0))
                
                # If we couldn't get a price, use the last known price or entry price
                if not current_price or current_price == 0:
                    current_price = trade.get('current_price', trade['entry_price'])
                
                # Calculate PnL
                exit_price = current_price
                realized_pnl = self.calculate_pnl(trade, exit_price)
                realized_pnl_pct = self.calculate_pnl_percentage(trade, exit_price)
                
                # Update trade
                trade['exit_price'] = exit_price
                trade['exit_time'] = timestamp
                trade['realized_pnl'] = realized_pnl
                trade['realized_pnl_pct'] = realized_pnl_pct
                trade['status'] = 'CLOSED'
                trade['duration'] = (datetime.fromtimestamp(timestamp / 1000) - datetime.fromtimestamp(trade['entry_time'] / 1000)).total_seconds() / 3600  # Hours
                trade['notes'] = notes
                
                # Update daily PnL
                self.daily_pnl += realized_pnl
                
                # Move to history
                self.trade_history.append(trade)
                del self.open_trades[trade_id]
                
                # Save to datastore
                self.datastore.save_trade({
                    'id': trade_id,
                    'symbol': symbol,
                    'order_id': "",
                    'side': trade['side'],
                    'entry_price': trade['entry_price'],
                    'exit_price': exit_price,
                    'quantity': trade['quantity'],
                    'entry_time': trade['entry_time'],
                    'exit_time': timestamp,
                    'status': 'CLOSED',
                    'pnl': realized_pnl,
                    'pnl_pct': realized_pnl_pct,
                    'fees': 0,
                    'strategy': trade['strategy'],
                    'timeframe': trade['timeframe'],
                    'stop_loss': trade['stop_loss'],
                    'take_profit': trade['take_profit'],
                    'notes': notes
                })
            
            logger.info(f"Closed trade {trade_id}: {trade['side']} {trade['quantity']} {symbol} @ {exit_price}, PnL: {realized_pnl:.2f} ({realized_pnl_pct:.2%})")
            return trade
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            handle_error(e, "Failed to close trade")
            return None
    
    def update_trade(self, trade_id, updates):
        """
        Update an existing trade.
        
        Args:
            trade_id (str): Trade ID
            updates (dict): Fields to update
            
        Returns:
            dict: Updated trade
        """
        try:
            with self.lock:
                if trade_id not in self.open_trades:
                    logger.warning(f"Trade {trade_id} not found in open trades")
                    return None
                
                trade = self.open_trades[trade_id]
                
                # Update fields
                for key, value in updates.items():
                    if key in trade:
                        trade[key] = value
                
                # Save to datastore
                if trade['status'] == 'OPEN':
                    self.datastore.save_trade({
                        'id': trade_id,
                        'symbol': trade['symbol'],
                        'order_id': "",
                        'side': trade['side'],
                        'entry_price': trade['entry_price'],
                        'exit_price': None,
                        'quantity': trade['quantity'],
                        'entry_time': trade['entry_time'],
                        'exit_time': None,
                        'status': 'OPEN',
                        'pnl': None,
                        'pnl_pct': None,
                        'fees': 0,
                        'strategy': trade['strategy'],
                        'timeframe': trade['timeframe'],
                        'stop_loss': trade['stop_loss'],
                        'take_profit': trade['take_profit'],
                        'notes': trade.get('notes', "")
                    })
            
            logger.debug(f"Updated trade {trade_id}")
            return trade
        except Exception as e:
            logger.error(f"Error updating trade: {e}")
            handle_error(e, "Failed to update trade")
            return None
    
    def get_trade(self, trade_id):
        """
        Get a trade by ID.
        
        Args:
            trade_id (str): Trade ID
            
        Returns:
            dict: Trade info
        """
        with self.lock:
            # Check open trades
            if trade_id in self.open_trades:
                return self.open_trades[trade_id]
            
            # Check trade history
            for trade in self.trade_history:
                if trade['id'] == trade_id:
                    return trade
            
            # Not found, try datastore
            trades = self.datastore.get_trades()
            for trade in trades:
                if trade['id'] == trade_id:
                    return trade
            
            return None
    
    def get_open_trades(self, symbol=None):
        """
        Get all open trades, optionally filtered by symbol.
        
        Args:
            symbol (str): Trading pair symbol (optional)
            
        Returns:
            list: List of open trades
        """
        with self.lock:
            if symbol:
                return [trade for trade in self.open_trades.values() if trade['symbol'] == symbol]
            else:
                return list(self.open_trades.values())
    
    def get_trade_history(self, symbol=None, limit=50):
        """
        Get trade history, optionally filtered by symbol.
        
        Args:
            symbol (str): Trading pair symbol (optional)
            limit (int): Maximum number of trades to return
            
        Returns:
            list: List of historical trades
        """
        with self.lock:
            if symbol:
                filtered = [trade for trade in self.trade_history if trade['symbol'] == symbol]
                return filtered[-limit:]
            else:
                return self.trade_history[-limit:]
    
    def get_open_position_count(self):
        """
        Get the number of open positions.
        
        Returns:
            int: Number of open positions
        """
        with self.lock:
            return len(self.positions)
    
    def get_open_trade_count(self):
        """
        Get the number of open trades.
        
        Returns:
            int: Number of open trades
        """
        with self.lock:
            return len(self.open_trades)
    
    def calculate_pnl(self, trade, exit_price=None):
        """
        Calculate realized or unrealized PnL for a trade.
        
        Args:
            trade (dict): Trade info
            exit_price (float): Exit price (if None, use current_price)
            
        Returns:
            float: PnL in quote currency
        """
        try:
            side = trade['side']
            entry_price = trade['entry_price']
            quantity = trade['quantity']
            
            if exit_price is None:
                exit_price = trade.get('current_price', entry_price)
            
            if side.upper() == "LONG":
                return (exit_price - entry_price) * quantity
            else:
                return (entry_price - exit_price) * quantity
        except Exception as e:
            logger.error(f"Error calculating PnL: {e}")
            return 0
    
    def calculate_pnl_percentage(self, trade, exit_price=None):
        """
        Calculate realized or unrealized PnL percentage for a trade.
        
        Args:
            trade (dict): Trade info
            exit_price (float): Exit price (if None, use current_price)
            
        Returns:
            float: PnL percentage
        """
        try:
            side = trade['side']
            entry_price = trade['entry_price']
            
            if exit_price is None:
                exit_price = trade.get('current_price', entry_price)
            
            if side.upper() == "LONG":
                return (exit_price - entry_price) / entry_price
            else:
                return (entry_price - exit_price) / entry_price
        except Exception as e:
            logger.error(f"Error calculating PnL percentage: {e}")
            return 0
    
    def calculate_unrealized_pnl(self, trade):
        """
        Calculate unrealized PnL for an open trade.
        
        Args:
            trade (dict): Trade info
            
        Returns:
            float: Unrealized PnL in quote currency
        """
        return self.calculate_pnl(trade)
    
    def calculate_unrealized_pnl_percentage(self, trade):
        """
        Calculate unrealized PnL percentage for an open trade.
        
        Args:
            trade (dict): Trade info
            
        Returns:
            float: Unrealized PnL percentage
        """
        return self.calculate_pnl_percentage(trade)
    
    def get_total_trade_value(self):
        """
        Get the total value of all open trades.
        
        Returns:
            float: Total trade value in quote currency
        """
        with self.lock:
            return sum(trade.get('value', 0) for trade in self.open_trades.values())
    
    def get_total_unrealized_pnl(self):
        """
        Get the total unrealized PnL across all open trades.
        
        Returns:
            float: Total unrealized PnL in quote currency
        """
        with self.lock:
            return sum(trade.get('unrealized_pnl', 0) for trade in self.open_trades.values())
    
    def update_performance_metrics(self):
        """
        Update performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        try:
            timestamp = int(time.time() * 1000)
            
            # Only update every 5 minutes to avoid excessive calculations
            if timestamp - self.last_update_time < 300000:  # 5 minutes in milliseconds
                return self.performance_metrics
            
            self.last_update_time = timestamp
            
            # Get account info
            account_info = self.account_manager.get_account_info()
            equity = account_info.get('total_equity', account_info['balance'])
            balance = account_info.get('balance', 0)
            
            # Get trade history from datastore
            trades = self.datastore.get_trades(status="CLOSED")
            
            if not trades:
                # No trade history yet
                with self.lock:
                    self.performance_metrics = {
                        'win_rate': 0,
                        'profit_factor': 0,
                        'average_win': 0,
                        'average_loss': 0,
                        'largest_win': 0,
                        'largest_loss': 0,
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'breakeven_trades': 0,
                        'average_trade': 0,
                        'average_bars_in_trade': 0,
                        'max_consecutive_wins': 0,
                        'max_consecutive_losses': 0,
                        'sharpe_ratio': 0,
                        'sortino_ratio': 0,
                        'max_drawdown': 0,
                        'current_drawdown': 0,
                        'recovery_factor': 0,
                        'expectancy': 0,
                        'average_rrr': 0
                    }
                    return self.performance_metrics
            
            # Calculate win rate
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            breakeven_trades = [t for t in trades if t.get('pnl', 0) == 0]
            
            total_trades = len(trades)
            winning_trades_count = len(winning_trades)
            losing_trades_count = len(losing_trades)
            
            win_rate = winning_trades_count / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            total_profit = sum(t.get('pnl', 0) for t in winning_trades)
            total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
            
            # Calculate average win/loss
            average_win = total_profit / winning_trades_count if winning_trades_count > 0 else 0
            average_loss = total_loss / losing_trades_count if losing_trades_count > 0 else 0
            
            # Find largest win/loss
            largest_win = max([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
            largest_loss = min([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0
            
            # Calculate average trade
            average_trade = sum(t.get('pnl', 0) for t in trades) / total_trades if total_trades > 0 else 0
            
            # Calculate average trade duration (in bars)
            # This is a placeholder, as we don't have access to the actual number of bars
            average_bars_in_trade = 0
            
            # Calculate consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            for t in sorted(trades, key=lambda x: x.get('entry_time', 0)):
                if t.get('pnl', 0) > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                elif t.get('pnl', 0) < 0:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_wins = 0
                    consecutive_losses = 0
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'balance': balance,
                'unrealized_pnl': self.get_total_unrealized_pnl(),
                'open_positions': self.get_open_position_count()
            })
            
            # Keep only the last 1000 points
            if len(self.equity_curve) > 1000:
                self.equity_curve = self.equity_curve[-1000:]
            
            # Calculate drawdown
            peak_equity = max([p.get('equity', 0) for p in self.equity_curve])
            current_drawdown_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            
            # Update max drawdown
            if current_drawdown_pct > self.max_drawdown:
                self.max_drawdown = current_drawdown_pct
            
            self.current_drawdown = current_drawdown_pct
            
            # Store drawdown point if significant
            if current_drawdown_pct > 0.01:  # More than 1%
                self.drawdowns.append({
                    'timestamp': timestamp,
                    'equity': equity,
                    'peak_equity': peak_equity,
                    'drawdown_pct': current_drawdown_pct,
                    'drawdown_value': peak_equity - equity
                })
            
            # Calculate recovery factor
            max_drawdown_value = self.max_drawdown * peak_equity
            total_net_profit = sum(t.get('pnl', 0) for t in trades)
            recovery_factor = total_net_profit / max_drawdown_value if max_drawdown_value > 0 else float('inf')
            
            # Calculate expectancy
            expectancy = (win_rate * average_win) - ((1 - win_rate) * average_loss)
            
            # Calculate average risk-reward ratio
            average_rrr = average_win / abs(average_loss) if average_loss != 0 else float('inf')
            
            # Calculate Sharpe ratio (simplified)
            returns = []
            
            # Calculate daily returns from equity curve
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1]['equity']
                curr_equity = self.equity_curve[i]['equity']
                
                if prev_equity > 0:
                    daily_return = (curr_equity - prev_equity) / prev_equity
                    returns.append(daily_return)
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                
                # Annualized Sharpe ratio (assuming daily returns)
                risk_free_rate = 0.02 / 252  # 2% annual risk-free rate, daily
                sharpe_ratio = (avg_return - risk_free_rate) / std_return * np.sqrt(252) if std_return > 0 else 0
                
                # Sortino ratio (only consider negative returns for risk)
                negative_returns = [r for r in returns if r < 0]
                std_negative = np.std(negative_returns) if negative_returns else 0
                sortino_ratio = (avg_return - risk_free_rate) / std_negative * np.sqrt(252) if std_negative > 0 else 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
            
            # Update performance metrics
            with self.lock:
                self.performance_metrics = {
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'average_win': average_win,
                    'average_loss': average_loss,
                    'largest_win': largest_win,
                    'largest_loss': largest_loss,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades_count,
                    'losing_trades': losing_trades_count,
                    'breakeven_trades': len(breakeven_trades),
                    'average_trade': average_trade,
                    'average_bars_in_trade': average_bars_in_trade,
                    'max_consecutive_wins': max_consecutive_wins,
                    'max_consecutive_losses': max_consecutive_losses,
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'max_drawdown': self.max_drawdown,
                    'current_drawdown': self.current_drawdown,
                    'recovery_factor': recovery_factor,
                    'expectancy': expectancy,
                    'average_rrr': average_rrr
                }
            
            # Save to datastore
            self.datastore.save_performance({
                'timestamp': timestamp,
                'equity': equity,
                'balance': balance,
                'open_positions': self.get_open_position_count(),
                'daily_pnl': self.daily_pnl,
                'weekly_pnl': sum(t.get('pnl', 0) for t in trades if t.get('exit_time', 0) > timestamp - 7 * 24 * 60 * 60 * 1000),
                'total_pnl': total_net_profit,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': self.max_drawdown,
                'current_drawdown': self.current_drawdown,
                'metrics_json': json.dumps(self.performance_metrics)
            })
            
            logger.debug(f"Updated performance metrics (win rate: {win_rate:.2%}, profit factor: {profit_factor:.2f}, max DD: {self.max_drawdown:.2%})")
            return self.performance_metrics
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
            handle_error(e, "Failed to update performance metrics")
            return {}
    
    def get_performance_metrics(self):
        """
        Get the latest performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        return self.performance_metrics
    
    def get_equity_curve(self):
        """
        Get the equity curve data.
        
        Returns:
            list: Equity curve data points
        """
        return self.equity_curve
    
    def get_daily_pnl(self):
        """
        Get the daily PnL.
        
        Returns:
            float: Daily PnL
        """
        return self.daily_pnl
    
    def get_total_pnl(self):
        """
        Get the total PnL (realized + unrealized).
        
        Returns:
            float: Total PnL
        """
        try:
            # Get realized PnL from trade history
            realized_pnl = sum(t.get('pnl', 0) for t in self.datastore.get_trades(status="CLOSED"))
            
            # Get unrealized PnL from open trades
            unrealized_pnl = self.get_total_unrealized_pnl()
            
            return realized_pnl + unrealized_pnl
        except Exception as e:
            logger.error(f"Error calculating total PnL: {e}")
            handle_error(e, "Failed to calculate total PnL")
            return 0
    
    def get_trade_distribution(self):
        """
        Get the distribution of trades by symbol.
        
        Returns:
            dict: Symbol -> trade count
        """
        try:
            trades = self.datastore.get_trades()
            
            distribution = {}
            for trade in trades:
                symbol = trade['symbol']
                if symbol not in distribution:
                    distribution[symbol] = 0
                distribution[symbol] += 1
            
            return distribution
        except Exception as e:
            logger.error(f"Error getting trade distribution: {e}")
            handle_error(e, "Failed to get trade distribution")
            return {}
    
    def get_strategy_performance(self):
        """
        Get performance metrics by strategy.
        
        Returns:
            dict: Strategy -> performance metrics
        """
        try:
            trades = self.datastore.get_trades()
            
            performance = {}
            
            for trade in trades:
                strategy = trade.get('strategy', 'unknown')
                
                if strategy not in performance:
                    performance[strategy] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'total_pnl': 0,
                        'win_rate': 0,
                        'profit_factor': 0,
                        'average_trade': 0
                    }
                
                performance[strategy]['total_trades'] += 1
                
                if trade.get('pnl', 0) > 0:
                    performance[strategy]['winning_trades'] += 1
                elif trade.get('pnl', 0) < 0:
                    performance[strategy]['losing_trades'] += 1
                
                performance[strategy]['total_pnl'] += trade.get('pnl', 0)
            
            # Calculate derived metrics
            for strategy, metrics in performance.items():
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
                
                total_profit = sum(t.get('pnl', 0) for t in trades 
                                  if t.get('strategy', 'unknown') == strategy and t.get('pnl', 0) > 0)
                
                total_loss = abs(sum(t.get('pnl', 0) for t in trades 
                                    if t.get('strategy', 'unknown') == strategy and t.get('pnl', 0) < 0))
                
                metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
                
                metrics['average_trade'] = metrics['total_pnl'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
            
            return performance
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            handle_error(e, "Failed to get strategy performance")
            return {}
    
    def check_risk_limits(self):
        """
        Check if portfolio risk limits are being observed.
        
        Returns:
            dict: Risk check results
        """
        try:
            # Get account info
            account_info = self.account_manager.get_account_info()
            equity = account_info.get('total_equity', account_info['balance'])
            
            # Check risk limits
            risk_checks = {
                'max_drawdown': {
                    'limit': RISK_CONFIG['max_drawdown_pct'] / 100,
                    'value': self.current_drawdown,
                    'status': 'OK' if self.current_drawdown < RISK_CONFIG['max_drawdown_pct'] / 100 else 'WARNING'
                },
                'max_daily_loss': {
                    'limit': RISK_CONFIG['max_daily_loss_pct'] / 100 * equity,
                    'value': -self.daily_pnl if self.daily_pnl < 0 else 0,
                    'status': 'OK' if self.daily_pnl >= -RISK_CONFIG['max_daily_loss_pct'] / 100 * equity else 'WARNING'
                },
                'max_open_trades': {
                    'limit': TRADING_CONFIG['max_open_trades'],
                    'value': self.get_open_trade_count(),
                    'status': 'OK' if self.get_open_trade_count() <= TRADING_CONFIG['max_open_trades'] else 'WARNING'
                }
            }
            
            # Determine overall status
            has_warning = any(check['status'] == 'WARNING' for check in risk_checks.values())
            overall_status = 'WARNING' if has_warning else 'OK'
            
            result = {
                'status': overall_status,
                'checks': risk_checks,
                'timestamp': int(time.time() * 1000)
            }
            
            return result
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            handle_error(e, "Failed to check risk limits")
            return {'status': 'ERROR', 'checks': {}, 'timestamp': int(time.time() * 1000)}
    
    def check_stop_losses(self):
        """
        Check and manage stop losses for all open positions.
        
        Returns:
            list: List of triggered stop losses
        """
        try:
            # Get current ticker data for all symbols with positions
            symbols = list(self.positions.keys())
            
            if not symbols:
                return []
            
            tickers = {}
            for symbol in symbols:
                ticker = self.account_manager.exchange_api.get_ticker(symbol)
                if ticker:
                    tickers[symbol] = float(ticker.get('lastPrice', 0))
            
            # Check stop losses
            triggered = []
            for trade_id, trade in list(self.open_trades.items()):
                symbol = trade['symbol']
                if symbol in tickers and trade.get('stop_loss'):
                    current_price = tickers[symbol]
                    stop_price = trade['stop_loss']
                    
                    # Check if stop loss is hit
                    if (trade['side'].upper() == "LONG" and current_price <= stop_price) or \
                       (trade['side'].upper() == "SHORT" and current_price >= stop_price):
                        # Stop loss hit
                        logger.info(f"Stop loss hit for trade {trade_id} ({symbol}) at {current_price}")
                        
                        triggered.append({
                            'trade_id': trade_id,
                            'symbol': symbol,
                            'side': trade['side'],
                            'entry_price': trade['entry_price'],
                            'current_price': current_price,
                            'stop_price': stop_price
                        })
                        
                        # Close the trade
                        self.close_trade(trade_id, f"Stop loss hit at {current_price}")
                        
                        # Close the position if needed
                        try:
                            # Determine close side (opposite of position)
                            close_side = "Sell" if trade['side'].upper() == "LONG" else "Buy"
                            
                            # Close the position on the exchange
                            self.order_manager.close_position(symbol, trade['side'], trade['quantity'])
                        except Exception as e:
                            logger.error(f"Error closing position for {symbol}: {e}")
                            handle_error(e, f"Failed to close position for {symbol}")
            
            return triggered
        except Exception as e:
            logger.error(f"Error checking stop losses: {e}")
            handle_error(e, "Failed to check stop losses")
            return []
    
    def reset_daily_stats(self):
        """
        Reset daily statistics (e.g., at the start of a new trading day).
        
        Returns:
            bool: True if reset successful
        """
        try:
            logger.info("Resetting daily statistics")
            
            with self.lock:
                # Save current daily PnL to datastore
                self.datastore.save_performance({
                    'timestamp': int(time.time() * 1000),
                    'daily_pnl': self.daily_pnl,
                    'equity': self.account_manager.get_total_equity(),
                    'balance': self.account_manager.get_balance()
                })
                
                # Reset daily PnL
                self.daily_pnl = 0
            
            return True
        except Exception as e:
            logger.error(f"Error resetting daily stats: {e}")
            handle_error(e, "Failed to reset daily stats")
            return False
    
    def get_position_risk(self, symbol):
        """
        Calculate risk metrics for a specific position.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Risk metrics
        """
        try:
            # Get position
            position = self.positions.get(symbol)
            
            if not position:
                return None
            
            # Get account info
            account_info = self.account_manager.get_account_info()
            equity = account_info.get('total_equity', account_info['balance'])
            
            # Calculate position value
            position_value = abs(position['position_value'])
            
            # Calculate risk metrics
            risk_metrics = {
                'position_value': position_value,
                'exposure_pct': position_value / equity if equity > 0 else 0,
                'unrealized_pnl': position['unrealized_pnl'],
                'unrealized_pnl_pct': position['unrealized_pnl'] / position_value if position_value > 0 else 0,
                'liquidation_price': position.get('liquidation_price', 0),
                'leverage': position.get('leverage', 1),
                'mark_price': position.get('mark_price', 0)
            }
            
            return risk_metrics
        except Exception as e:
            logger.error(f"Error calculating position risk for {symbol}: {e}")
            handle_error(e, f"Failed to calculate position risk for {symbol}")
            return None
    
    def check_correlation(self, symbols=None):
        """
        Check correlation between trading pairs.
        
        Args:
            symbols (list): List of symbols to check
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        try:
            if not symbols:
                symbols = list(self.positions.keys())
            
            if len(symbols) < 2:
                logger.warning("Not enough symbols to calculate correlation")
                return None
            
            # Get historical data
            price_data = {}
            
            for symbol in symbols:
                # Get daily candles for the past 30 days
                candles = self.account_manager.exchange_api.get_candles(
                    symbol, "1d", limit=30)
                
                if candles:
                    # Extract close prices
                    prices = [float(candle[4]) for candle in candles]  # Assuming close price is at index 4
                    price_data[symbol] = prices
            
            # Create DataFrame
            df = pd.DataFrame(price_data)
            
            # Calculate correlation matrix
            correlation = df.corr()
            
            return correlation
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            handle_error(e, "Failed to calculate correlation")
            return None
    
    def calculate_var(self, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) for the portfolio.
        
        Args:
            confidence_level (float): Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            float: VaR value
        """
        try:
            # Get account info
            account_info = self.account_manager.get_account_info()
            portfolio_value = account_info.get('total_equity', account_info['balance'])
            
            # Get historical daily returns
            returns = []
            
            # Calculate daily returns from equity curve
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1]['equity']
                curr_equity = self.equity_curve[i]['equity']
                
                if prev_equity > 0:
                    daily_return = (curr_equity - prev_equity) / prev_equity
                    returns.append(daily_return)
            
            if not returns:
                logger.warning("Not enough data to calculate VaR")
                return 0
            
            # Convert to numpy array
            returns = np.array(returns)
            
            # Calculate VaR
            var_pct = np.percentile(returns, (1 - confidence_level) * 100)
            var_value = abs(var_pct * portfolio_value)
            
            logger.debug(f"Calculated VaR ({confidence_level:.2%}): {var_value:.2f} ({var_pct:.2%})")
            
            return var_value
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            handle_error(e, "Failed to calculate VaR")
            return 0
    
    def is_position_allowed(self, symbol, side, quantity, entry_price):
        """
        Check if a new position is allowed based on risk rules.
        
        Args:
            symbol (str): Trading pair symbol
            side (str): Position side ("LONG" or "SHORT")
            quantity (float): Position quantity
            entry_price (float): Entry price
            
        Returns:
            tuple: (bool, str) - (allowed, reason)
        """
        try:
            # Check if we already have a position in this symbol
            if symbol in self.positions:
                existing_position = self.positions[symbol]
                existing_side = "LONG" if existing_position.get('side') == 'Buy' else "SHORT"
                
                # Check if trying to open opposite position
                if existing_side != side:
                    return False, f"Already have a {existing_side} position in {symbol}"
            
            # Check max open positions
            if self.get_open_position_count() >= TRADING_CONFIG['max_open_trades']:
                return False, f"Maximum open positions reached ({TRADING_CONFIG['max_open_trades']})"
            
            # Check max trades per symbol
            symbol_trades = [t for t in self.open_trades.values() if t['symbol'] == symbol]
            if len(symbol_trades) >= TRADING_CONFIG['max_open_trades_per_pair']:
                return False, f"Maximum positions per symbol reached ({TRADING_CONFIG['max_open_trades_per_pair']})"
            
            # Check if we are in a drawdown
            if self.current_drawdown > RISK_CONFIG['max_drawdown_pct'] / 100:
                return False, f"Drawdown limit reached ({self.current_drawdown:.2%})"
            
            # Check daily loss limit
            if self.daily_pnl < -RISK_CONFIG['max_daily_loss_pct'] / 100 * self.account_manager.get_balance():
                return False, f"Daily loss limit reached ({self.daily_pnl:.2f})"
            
            # Check position value
            position_value = quantity * entry_price
            account_equity = self.account_manager.get_total_equity()
            
            position_pct = position_value / account_equity if account_equity > 0 else 0
            
            if position_pct > TRADING_CONFIG['risk_per_trade']:
                return False, f"Position size exceeds risk limit ({position_pct:.2%} > {TRADING_CONFIG['risk_per_trade']:.2%})"
            
            return True, "Position allowed"
        except Exception as e:
            logger.error(f"Error checking if position is allowed: {e}")
            handle_error(e, "Failed to check if position is allowed")
            return False, f"Error: {str(e)}"