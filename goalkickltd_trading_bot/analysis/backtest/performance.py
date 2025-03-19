"""
Performance analysis module for the Goalkick Ltd Trading Bot.
Calculates performance metrics for backtesting and live trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Union, Optional

from config.logging_config import get_logger
from src.utils.error_handling import handle_error

logger = get_logger("backtest.performance")

class PerformanceAnalyzer:
    """Class for analyzing trading performance and calculating metrics."""
    
    def __init__(self):
        """Initialize the PerformanceAnalyzer."""
        pass
    
    def calculate_metrics(
        self, 
        trades: List[Dict], 
        equity: List[float], 
        timestamps: List[datetime], 
        initial_capital: float
    ) -> Dict:
        """
        Calculate comprehensive performance metrics from trade history and equity curve.
        
        Args:
            trades: List of trade dictionaries
            equity: List of equity values
            timestamps: List of timestamps corresponding to equity values
            initial_capital: Initial capital amount
            
        Returns:
            dict: Performance metrics
        """
        try:
            if not trades or not equity or not timestamps:
                logger.warning("Empty trade history or equity curve")
                return {"error": "Insufficient data for analysis"}
            
            # Convert to DataFrames
            trades_df = pd.DataFrame(trades)
            
            # Create equity curve DataFrame
            equity_df = pd.DataFrame({
                'equity': equity,
                'timestamp': timestamps
            })
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            returns_metrics = self.calculate_return_metrics(equity_df, initial_capital)
            
            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(equity_df, initial_capital)
            
            # Calculate trade statistics
            trade_metrics = self.calculate_trade_metrics(trades_df)
            
            # Calculate ratios
            ratio_metrics = self.calculate_ratio_metrics(
                returns_metrics, risk_metrics, trade_metrics, equity_df
            )
            
            # Calculate drawdowns
            drawdown_metrics = self.calculate_drawdown_metrics(equity_df)
            
            # Combine all metrics
            metrics = {
                **self.summarize_metrics(
                    returns_metrics, risk_metrics, ratio_metrics, 
                    trade_metrics, drawdown_metrics
                ),
                'returns': returns_metrics,
                'risk': risk_metrics,
                'ratios': ratio_metrics,
                'trade_stats': trade_metrics,
                'drawdowns': drawdown_metrics,
                'final_equity': equity[-1]
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            handle_error(e, "Failed to calculate performance metrics")
            return {"error": str(e)}
    
    def calculate_return_metrics(self, equity_df: pd.DataFrame, initial_capital: float) -> Dict:
        """
        Calculate return-related metrics.
        
        Args:
            equity_df: Equity curve DataFrame
            initial_capital: Initial capital amount
            
        Returns:
            dict: Return metrics
        """
        try:
            # Resample to daily returns for standard metrics
            if len(equity_df) > 1:
                # Ensure the DataFrame has a datetime index
                equity_df = equity_df.copy()
                if not isinstance(equity_df.index, pd.DatetimeIndex):
                    equity_df.index = pd.to_datetime(equity_df.index)
                
                # Resample to daily (last value of each day)
                try:
                    daily_equity = equity_df.resample('D').last().dropna()
                except:
                    # Fallback if resampling fails
                    daily_equity = equity_df
            else:
                daily_equity = equity_df
            
            # Calculate returns
            daily_equity['return'] = daily_equity['equity'].pct_change()
            
            # Filter out empty values
            daily_returns = daily_equity['return'].dropna()
            
            # Calculate total return
            final_equity = equity_df['equity'].iloc[-1]
            total_return = final_equity / initial_capital - 1
            
            # Calculate annualized return (if we have at least 2 days of data)
            if len(daily_equity) >= 2:
                days = (daily_equity.index[-1] - daily_equity.index[0]).days
                if days > 0:
                    years = days / 365.25
                    annual_return = (1 + total_return) ** (1 / years) - 1
                else:
                    annual_return = total_return
            else:
                annual_return = total_return
            
            # Calculate logarithmic returns
            if len(daily_returns) > 0:
                log_returns = np.log(1 + daily_returns)
                avg_daily_return = daily_returns.mean()
                avg_log_return = log_returns.mean()
                
                # Volatility (daily)
                volatility_daily = daily_returns.std()
                
                # Volatility (annualized)
                volatility_annual = volatility_daily * np.sqrt(252)
            else:
                avg_daily_return = 0
                avg_log_return = 0
                volatility_daily = 0
                volatility_annual = 0
            
            return {
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annual_return': annual_return,
                'annual_return_pct': annual_return * 100,
                'avg_daily_return': avg_daily_return,
                'avg_daily_return_pct': avg_daily_return * 100,
                'avg_log_return': avg_log_return,
                'volatility_daily': volatility_daily,
                'volatility_annual': volatility_annual,
                'volatility_annual_pct': volatility_annual * 100
            }
        except Exception as e:
            logger.error(f"Error calculating return metrics: {e}")
            handle_error(e, "Failed to calculate return metrics")
            return {}
    
    def calculate_risk_metrics(self, equity_df: pd.DataFrame, initial_capital: float) -> Dict:
        """
        Calculate risk-related metrics.
        
        Args:
            equity_df: Equity curve DataFrame
            initial_capital: Initial capital amount
            
        Returns:
            dict: Risk metrics
        """
        try:
            # Calculate drawdown series
            if len(equity_df) > 0:
                equity_series = equity_df['equity']
                rolling_max = equity_series.cummax()
                drawdown_series = (equity_series - rolling_max) / rolling_max
                
                # Maximum drawdown
                max_drawdown = drawdown_series.min()
                
                # Drawdown duration
                is_drawdown = drawdown_series < 0
                if is_drawdown.any():
                    drawdown_start = is_drawdown.idxmax()
                    drawdown_end = drawdown_series[drawdown_start:].idxmin()
                    max_drawdown_duration = (drawdown_end - drawdown_start).days
                else:
                    max_drawdown_duration = 0
                
                # Calculate underwater periods
                underwater_periods = []
                in_drawdown = False
                start_date = None
                
                for date, value in drawdown_series.items():
                    if not in_drawdown and value < 0:
                        # Start of drawdown
                        in_drawdown = True
                        start_date = date
                    elif in_drawdown and value >= 0:
                        # End of drawdown
                        in_drawdown = False
                        underwater_periods.append((start_date, date, (date - start_date).days))
                
                # Handle ongoing drawdown
                if in_drawdown:
                    underwater_periods.append((start_date, drawdown_series.index[-1], (drawdown_series.index[-1] - start_date).days))
                
                # Calculate average drawdown
                avg_drawdown = drawdown_series[drawdown_series < 0].mean() if (drawdown_series < 0).any() else 0
                
                # Calculate recovery factor
                if max_drawdown != 0:
                    recovery_factor = equity_df['equity'].iloc[-1] / (initial_capital * abs(max_drawdown))
                else:
                    recovery_factor = float('inf')
            else:
                max_drawdown = 0
                max_drawdown_duration = 0
                avg_drawdown = 0
                underwater_periods = []
                recovery_factor = 0
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'max_drawdown_duration': max_drawdown_duration,
                'avg_drawdown': avg_drawdown,
                'avg_drawdown_pct': avg_drawdown * 100,
                'underwater_periods': len(underwater_periods),
                'recovery_factor': recovery_factor
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            handle_error(e, "Failed to calculate risk metrics")
            return {}
    
    def calculate_trade_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """
        Calculate trade statistics.
        
        Args:
            trades_df: DataFrame of trades
            
        Returns:
            dict: Trade statistics
        """
        try:
            if trades_df.empty:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'avg_trade_pnl': 0,
                    'avg_winning_trade': 0,
                    'avg_losing_trade': 0,
                    'largest_winning_trade': 0,
                    'largest_losing_trade': 0,
                    'max_consecutive_wins': 0,
                    'max_consecutive_losses': 0,
                    'avg_holding_period': 0
                }
            
            # Total trades
            total_trades = len(trades_df)
            
            # Calculate win/loss statistics
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            # Win rate
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            # PnL statistics
            if win_count > 0:
                avg_winning_trade = winning_trades['pnl'].mean()
                largest_winning_trade = winning_trades['pnl'].max()
                total_profit = winning_trades['pnl'].sum()
            else:
                avg_winning_trade = 0
                largest_winning_trade = 0
                total_profit = 0
            
            if loss_count > 0:
                avg_losing_trade = losing_trades['pnl'].mean()
                largest_losing_trade = losing_trades['pnl'].min()
                total_loss = abs(losing_trades['pnl'].sum())
            else:
                avg_losing_trade = 0
                largest_losing_trade = 0
                total_loss = 0
            
            # Average trade PnL
            avg_trade_pnl = trades_df['pnl'].mean() if total_trades > 0 else 0
            
            # Profit factor
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
            
            # Calculate consecutive wins/losses
            if 'pnl' in trades_df.columns and len(trades_df) > 0:
                # Sort trades by time
                if 'exit_time' in trades_df.columns:
                    sorted_trades = trades_df.sort_values('exit_time')
                else:
                    sorted_trades = trades_df
                
                # Calculate whether each trade is a win or loss
                is_win = sorted_trades['pnl'] > 0
                
                # Calculate consecutive trade results
                consecutive_results = is_win.groupby((is_win != is_win.shift()).cumsum()).cumcount() + 1
                
                # Calculate max consecutive wins and losses
                max_consecutive_wins = consecutive_results[is_win].max() if any(is_win) else 0
                max_consecutive_losses = consecutive_results[~is_win].max() if any(~is_win) else 0
            else:
                max_consecutive_wins = 0
                max_consecutive_losses = 0
            
            # Calculate average holding period
            if 'holding_period' in trades_df.columns:
                avg_holding_period = trades_df['holding_period'].mean()
            else:
                avg_holding_period = 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'profit_factor': profit_factor,
                'avg_trade_pnl': avg_trade_pnl,
                'avg_winning_trade': avg_winning_trade,
                'avg_losing_trade': avg_losing_trade,
                'largest_winning_trade': largest_winning_trade,
                'largest_losing_trade': largest_losing_trade,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net_profit': total_profit - total_loss,
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                'avg_holding_period': avg_holding_period
            }
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            handle_error(e, "Failed to calculate trade metrics")
            return {}
    
    def calculate_ratio_metrics(
        self, 
        returns_metrics: Dict, 
        risk_metrics: Dict, 
        trade_metrics: Dict,
        equity_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate performance ratios and metrics.
        
        Args:
            returns_metrics: Dictionary of return metrics
            risk_metrics: Dictionary of risk metrics
            trade_metrics: Dictionary of trade metrics
            equity_df: Equity curve DataFrame
            
        Returns:
            dict: Ratio metrics
        """
        try:
            # Sharpe Ratio
            if 'volatility_annual' in returns_metrics and returns_metrics['volatility_annual'] > 0:
                # Assume risk-free rate of 0 for simplicity
                sharpe_ratio = returns_metrics['annual_return'] / returns_metrics['volatility_annual']
            else:
                sharpe_ratio = 0
            
            # Sortino Ratio (downside deviation)
            if len(equity_df) > 1:
                equity_df = equity_df.copy()
                if not isinstance(equity_df.index, pd.DatetimeIndex):
                    equity_df.index = pd.to_datetime(equity_df.index)
                
                try:
                    daily_equity = equity_df.resample('D').last().dropna()
                except:
                    daily_equity = equity_df
                
                daily_equity['return'] = daily_equity['equity'].pct_change()
                daily_returns = daily_equity['return'].dropna()
                
                # Calculate downside deviation (only negative returns)
                negative_returns = daily_returns[daily_returns < 0]
                if len(negative_returns) > 0:
                    downside_deviation = negative_returns.std() * np.sqrt(252)
                    
                    # Sortino Ratio
                    sortino_ratio = returns_metrics['annual_return'] / downside_deviation if downside_deviation > 0 else 0
                else:
                    sortino_ratio = float('inf') if returns_metrics['annual_return'] > 0 else 0
            else:
                sortino_ratio = 0
            
            # Calmar Ratio
            if 'max_drawdown' in risk_metrics and risk_metrics['max_drawdown'] != 0:
                calmar_ratio = returns_metrics['annual_return'] / abs(risk_metrics['max_drawdown'])
            else:
                calmar_ratio = float('inf') if returns_metrics['annual_return'] > 0 else 0
            
            # Profit-to-Loss Ratio (Average Win / Average Loss)
            if 'avg_losing_trade' in trade_metrics and trade_metrics['avg_losing_trade'] != 0:
                avg_win = trade_metrics.get('avg_winning_trade', 0)
                avg_loss = abs(trade_metrics.get('avg_losing_trade', 0))
                win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0
            else:
                win_loss_ratio = float('inf') if trade_metrics.get('avg_winning_trade', 0) > 0 else 0
            
            # Expectancy = (Win Rate * Average Win) - ((1 - Win Rate) * Average Loss)
            win_rate = trade_metrics.get('win_rate', 0)
            avg_win = trade_metrics.get('avg_winning_trade', 0)
            avg_loss = abs(trade_metrics.get('avg_losing_trade', 0))
            
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            expectancy_pct = win_rate * (avg_win / avg_loss) - (1 - win_rate) if avg_loss > 0 else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'win_loss_ratio': win_loss_ratio,
                'expectancy': expectancy,
                'expectancy_pct': expectancy_pct
            }
        except Exception as e:
            logger.error(f"Error calculating ratio metrics: {e}")
            handle_error(e, "Failed to calculate ratio metrics")
            return {}
    
    def calculate_drawdown_metrics(self, equity_df: pd.DataFrame) -> Dict:
        """
        Calculate detailed drawdown metrics.
        
        Args:
            equity_df: Equity curve DataFrame
            
        Returns:
            dict: Drawdown metrics
        """
        try:
            if len(equity_df) > 0:
                equity_series = equity_df['equity']
                peak = equity_series.expanding().max()
                drawdown = (equity_series - peak) / peak
                
                # Find drawdown periods
                drawdown_periods = []
                in_drawdown = False
                current_period = {}
                
                for date, value in drawdown.items():
                    if not in_drawdown and value < 0:
                        # Start of drawdown
                        in_drawdown = True
                        current_period = {
                            'start': date,
                            'peak_equity': peak[date],
                            'values': []
                        }
                    
                    if in_drawdown:
                        # Track drawdown values
                        current_period['values'].append(value)
                        
                        if value >= 0:
                            # End of drawdown
                            in_drawdown = False
                            current_period['end'] = date
                            current_period['duration_days'] = (date - current_period['start']).days
                            current_period['max_drawdown'] = min(current_period['values'])
                            current_period['recovery_days'] = (date - drawdown.loc[drawdown == current_period['max_drawdown']].index[0]).days
                            drawdown_periods.append(current_period)
                
                # Handle ongoing drawdown
                if in_drawdown:
                    current_period['end'] = drawdown.index[-1]
                    current_period['duration_days'] = (current_period['end'] - current_period['start']).days
                    current_period['max_drawdown'] = min(current_period['values'])
                    try:
                        current_period['recovery_days'] = None  # Still in drawdown
                        drawdown_periods.append(current_period)
                    except:
                        pass
                
                # Sort drawdown periods by magnitude
                sorted_periods = sorted(drawdown_periods, key=lambda x: x['max_drawdown'])
                
                # Get top 5 drawdowns
                top_drawdowns = sorted_periods[:5] if len(sorted_periods) >= 5 else sorted_periods
                
                # Calculate average drawdown statistics
                if drawdown_periods:
                    avg_drawdown_magnitude = np.mean([period['max_drawdown'] for period in drawdown_periods])
                    avg_drawdown_duration = np.mean([period['duration_days'] for period in drawdown_periods])
                    avg_recovery_duration = np.mean([period['recovery_days'] for period in drawdown_periods if period['recovery_days'] is not None])
                else:
                    avg_drawdown_magnitude = 0
                    avg_drawdown_duration = 0
                    avg_recovery_duration = 0
                
                return {
                    'top_drawdowns': top_drawdowns,
                    'avg_drawdown_magnitude': avg_drawdown_magnitude,
                    'avg_drawdown_duration': avg_drawdown_duration,
                    'avg_recovery_duration': avg_recovery_duration,
                    'total_drawdown_periods': len(drawdown_periods)
                }
            else:
                return {
                    'top_drawdowns': [],
                    'avg_drawdown_magnitude': 0,
                    'avg_drawdown_duration': 0,
                    'avg_recovery_duration': 0,
                    'total_drawdown_periods': 0
                }
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
            handle_error(e, "Failed to calculate drawdown metrics")
            return {}
    
    def summarize_metrics(
        self, 
        returns_metrics: Dict, 
        risk_metrics: Dict, 
        ratio_metrics: Dict, 
        trade_metrics: Dict, 
        drawdown_metrics: Dict
    ) -> Dict:
        """
        Create a summary of key performance metrics.
        
        Args:
            returns_metrics: Dictionary of return metrics
            risk_metrics: Dictionary of risk metrics
            ratio_metrics: Dictionary of ratio metrics
            trade_metrics: Dictionary of trade metrics
            drawdown_metrics: Dictionary of drawdown metrics
            
        Returns:
            dict: Summary metrics
        """
        return {
            'total_return_pct': returns_metrics.get('total_return_pct', 0),
            'annual_return_pct': returns_metrics.get('annual_return_pct', 0),
            'volatility_annual_pct': returns_metrics.get('volatility_annual_pct', 0),
            'max_drawdown_pct': risk_metrics.get('max_drawdown_pct', 0),
            'sharpe_ratio': ratio_metrics.get('sharpe_ratio', 0),
            'sortino_ratio': ratio_metrics.get('sortino_ratio', 0),
            'calmar_ratio': ratio_metrics.get('calmar_ratio', 0),
            'total_trades': trade_metrics.get('total_trades', 0),
            'win_rate': trade_metrics.get('win_rate', 0),
            'profit_factor': trade_metrics.get('profit_factor', 0),
            'expectancy': ratio_metrics.get('expectancy', 0)
        }
    
    def generate_performance_report(
        self, 
        metrics: Dict, 
        trades_df: pd.DataFrame, 
        equity_df: pd.DataFrame, 
        format_type: str = 'text'
    ) -> str:
        """
        Generate a performance report in the specified format.
        
        Args:
            metrics: Performance metrics dictionary
            trades_df: DataFrame of trades
            equity_df: Equity curve DataFrame
            format_type: Report format ('text', 'html', or 'markdown')
            
        Returns:
            str: Formatted performance report
        """
        try:
            if format_type == 'text':
                return self._generate_text_report(metrics, trades_df, equity_df)
            elif format_type == 'html':
                return self._generate_html_report(metrics, trades_df, equity_df)
            elif format_type == 'markdown':
                return self._generate_markdown_report(metrics, trades_df, equity_df)
            else:
                logger.warning(f"Unknown report format: {format_type}")
                return self._generate_text_report(metrics, trades_df, equity_df)
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            handle_error(e, "Failed to generate performance report")
            return f"Error generating report: {str(e)}"
    
    def _generate_text_report(self, metrics: Dict, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> str:
        """Generate a text-based performance report."""
        report = []
        
        # Title
        report.append("=" * 80)
        report.append("PERFORMANCE REPORT".center(80))
        report.append("=" * 80)
        report.append("")
        
        # Summary section
        report.append("-" * 80)
        report.append("SUMMARY".center(80))
        report.append("-" * 80)
        
        summary = [
            f"Total Return: {metrics.get('total_return_pct', 0):.2f}%",
            f"Annual Return: {metrics.get('annual_return_pct', 0):.2f}%",
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%",
            f"Total Trades: {metrics.get('total_trades', 0)}",
            f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%",
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
        ]
        
        # Format in columns
        for i in range(0, len(summary), 3):
            row = summary[i:i+3]
            report.append("  ".join(item.ljust(30) for item in row))
        
        report.append("")
        
        # Returns section
        report.append("-" * 80)
        report.append("RETURNS".center(80))
        report.append("-" * 80)
        
        returns_data = [
            f"Total Return: {metrics.get('total_return_pct', 0):.2f}%",
            f"Annual Return: {metrics.get('annual_return_pct', 0):.2f}%",
            f"Volatility (Annual): {metrics.get('volatility_annual_pct', 0):.2f}%",
            f"Avg Daily Return: {metrics.get('avg_daily_return_pct', 0):.4f}%",
        ]
        
        for item in returns_data:
            report.append(item)
        
        report.append("")
        
        # Risk section
        report.append("-" * 80)
        report.append("RISK METRICS".center(80))
        report.append("-" * 80)
        
        risk_data = [
            f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%",
            f"Max Drawdown Duration: {metrics.get('max_drawdown_duration', 0)} days",
            f"Avg Drawdown: {metrics.get('avg_drawdown_pct', 0):.2f}%",
            f"Recovery Factor: {metrics.get('recovery_factor', 0):.2f}",
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}",
            f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}",
        ]
        
        for item in risk_data:
            report.append(item)
        
        report.append("")
        
        # Trade statistics section
        report.append("-" * 80)
        report.append("TRADE STATISTICS".center(80))
        report.append("-" * 80)
        
        trade_data = [
            f"Total Trades: {metrics.get('total_trades', 0)}",
            f"Winning Trades: {metrics.get('winning_trades', 0)} ({metrics.get('win_rate', 0)*100:.2f}%)",
            f"Losing Trades: {metrics.get('losing_trades', 0)} ({(1-metrics.get('win_rate', 0))*100:.2f}%)",
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            f"Average Trade: {metrics.get('avg_trade_pnl', 0):.2f}",
            f"Average Winner: {metrics.get('avg_winning_trade', 0):.2f}",
            f"Average Loser: {metrics.get('avg_losing_trade', 0):.2f}",
            f"Largest Winner: {metrics.get('largest_winning_trade', 0):.2f}",
            f"Largest Loser: {metrics.get('largest_losing_trade', 0):.2f}",
            f"Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}",
            f"Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}",
            f"Win/Loss Ratio: {metrics.get('win_loss_ratio', 0):.2f}",
            f"Expectancy: {metrics.get('expectancy', 0):.2f}",
            f"Avg Holding Period: {metrics.get('avg_holding_period', 0):.2f} hours",
        ]
        
        for item in trade_data:
            report.append(item)
        
        report.append("")
        
        # Drawdown analysis
        report.append("-" * 80)
        report.append("DRAWDOWN ANALYSIS".center(80))
        report.append("-" * 80)
        
        report.append(f"Total Drawdown Periods: {metrics.get('total_drawdown_periods', 0)}")
        report.append(f"Avg Drawdown Magnitude: {metrics.get('avg_drawdown_magnitude', 0)*100:.2f}%")
        report.append(f"Avg Drawdown Duration: {metrics.get('avg_drawdown_duration', 0):.2f} days")
        report.append(f"Avg Recovery Duration: {metrics.get('avg_recovery_duration', 0):.2f} days")
        
        report.append("")
        report.append("Top Drawdowns:")
        
        top_drawdowns = metrics.get('top_drawdowns', [])
        if top_drawdowns:
            for i, dd in enumerate(top_drawdowns, 1):
                report.append(f"{i}. {dd.get('max_drawdown', 0)*100:.2f}% - Duration: {dd.get('duration_days', 0)} days - Recovery: {dd.get('recovery_days', 'N/A')} days")
        else:
            report.append("No significant drawdowns recorded.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _generate_html_report(self, metrics: Dict, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> str:
        """Generate an HTML performance report."""
        html = []
        
        # Document start
        html.append("<!DOCTYPE html>")
        html.append("<html lang='en'>")
        html.append("<head>")
        html.append("<meta charset='UTF-8'>")
        html.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html.append("<title>Trading Performance Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("h1, h2 { color: #333; }")
        html.append("h1 { text-align: center; }")
        html.append(".section { margin-bottom: 30px; }")
        html.append(".summary-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }")
        html.append(".metric { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }")
        html.append(".metric-title { font-weight: bold; font-size: 14px; }")
        html.append(".metric-value { font-size: 24px; margin: 5px 0; }")
        html.append(".positive { color: green; }")
        html.append(".negative { color: red; }")
        html.append("table { width: 100%; border-collapse: collapse; }")
        html.append("th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }")
        html.append("th { background-color: #f2f2f2; }")
        html.append("tr:hover { background-color: #f5f5f5; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Title
        html.append("<h1>Trading Performance Report</h1>")
        
        # Summary section
        html.append("<div class='section'>")
        html.append("<h2>Summary</h2>")
        html.append("<div class='summary-grid'>")
        
        # Key metrics
        summary_metrics = [
            {'title': 'Total Return', 'value': f"{metrics.get('total_return_pct', 0):.2f}%", 'class': 'positive' if metrics.get('total_return_pct', 0) > 0 else 'negative'},
            {'title': 'Annual Return', 'value': f"{metrics.get('annual_return_pct', 0):.2f}%", 'class': 'positive' if metrics.get('annual_return_pct', 0) > 0 else 'negative'},
            {'title': 'Sharpe Ratio', 'value': f"{metrics.get('sharpe_ratio', 0):.2f}", 'class': 'positive' if metrics.get('sharpe_ratio', 0) > 1 else ''},
            {'title': 'Max Drawdown', 'value': f"{metrics.get('max_drawdown_pct', 0):.2f}%", 'class': 'negative'},
            {'title': 'Win Rate', 'value': f"{metrics.get('win_rate', 0)*100:.2f}%", 'class': 'positive' if metrics.get('win_rate', 0) > 0.5 else ''},
            {'title': 'Profit Factor', 'value': f"{metrics.get('profit_factor', 0):.2f}", 'class': 'positive' if metrics.get('profit_factor', 0) > 1 else 'negative'},
            {'title': 'Total Trades', 'value': f"{metrics.get('total_trades', 0)}", 'class': ''},
            {'title': 'Expectancy', 'value': f"{metrics.get('expectancy', 0):.2f}", 'class': 'positive' if metrics.get('expectancy', 0) > 0 else 'negative'},
            {'title': 'Recovery Factor', 'value': f"{metrics.get('recovery_factor', 0):.2f}", 'class': 'positive' if metrics.get('recovery_factor', 0) > 1 else ''}
        ]
        
        for metric in summary_metrics:
            html.append(f"<div class='metric'>")
            html.append(f"<div class='metric-title'>{metric['title']}</div>")
            html.append(f"<div class='metric-value {metric['class']}'>{metric['value']}</div>")
            html.append(f"</div>")
        
        html.append("</div>") # End summary-grid
        html.append("</div>") # End section
        
        # Returns section
        html.append("<div class='section'>")
        html.append("<h2>Returns</h2>")
        html.append("<table>")
        html.append("<tr><th>Metric</th><th>Value</th></tr>")
        html.append(f"<tr><td>Total Return</td><td>{metrics.get('total_return_pct', 0):.2f}%</td></tr>")
        html.append(f"<tr><td>Annual Return</td><td>{metrics.get('annual_return_pct', 0):.2f}%</td></tr>")
        html.append(f"<tr><td>Volatility (Annual)</td><td>{metrics.get('volatility_annual_pct', 0):.2f}%</td></tr>")
        html.append(f"<tr><td>Avg Daily Return</td><td>{metrics.get('avg_daily_return_pct', 0):.4f}%</td></tr>")
        html.append("</table>")
        html.append("</div>") # End section
        
        # Risk section
        html.append("<div class='section'>")
        html.append("<h2>Risk Metrics</h2>")
        html.append("<table>")
        html.append("<tr><th>Metric</th><th>Value</th></tr>")
        html.append(f"<tr><td>Max Drawdown</td><td>{metrics.get('max_drawdown_pct', 0):.2f}%</td></tr>")
        html.append(f"<tr><td>Max Drawdown Duration</td><td>{metrics.get('max_drawdown_duration', 0)} days</td></tr>")
        html.append(f"<tr><td>Avg Drawdown</td><td>{metrics.get('avg_drawdown_pct', 0):.2f}%</td></tr>")
        html.append(f"<tr><td>Recovery Factor</td><td>{metrics.get('recovery_factor', 0):.2f}</td></tr>")
        html.append(f"<tr><td>Sharpe Ratio</td><td>{metrics.get('sharpe_ratio', 0):.2f}</td></tr>")
        html.append(f"<tr><td>Sortino Ratio</td><td>{metrics.get('sortino_ratio', 0):.2f}</td></tr>")
        html.append(f"<tr><td>Calmar Ratio</td><td>{metrics.get('calmar_ratio', 0):.2f}</td></tr>")
        html.append("</table>")
        html.append("</div>") # End section
        
        # Trade statistics section
        html.append("<div class='section'>")
        html.append("<h2>Trade Statistics</h2>")
        html.append("<table>")
        html.append("<tr><th>Metric</th><th>Value</th></tr>")
        html.append(f"<tr><td>Total Trades</td><td>{metrics.get('total_trades', 0)}</td></tr>")
        html.append(f"<tr><td>Winning Trades</td><td>{metrics.get('winning_trades', 0)} ({metrics.get('win_rate', 0)*100:.2f}%)</td></tr>")
        html.append(f"<tr><td>Losing Trades</td><td>{metrics.get('losing_trades', 0)} ({(1-metrics.get('win_rate', 0))*100:.2f}%)</td></tr>")
        html.append(f"<tr><td>Profit Factor</td><td>{metrics.get('profit_factor', 0):.2f}</td></tr>")
        html.append(f"<tr><td>Average Trade</td><td>{metrics.get('avg_trade_pnl', 0):.2f}</td></tr>")
        html.append(f"<tr><td>Average Winner</td><td>{metrics.get('avg_winning_trade', 0):.2f}</td></tr>")
        html.append(f"<tr><td>Average Loser</td><td>{metrics.get('avg_losing_trade', 0):.2f}</td></tr>")
        html.append(f"<tr><td>Largest Winner</td><td>{metrics.get('largest_winning_trade', 0):.2f}</td></tr>")
        html.append(f"<tr><td>Largest Loser</td><td>{metrics.get('largest_losing_trade', 0):.2f}</td></tr>")
        html.append(f"<tr><td>Max Consecutive Wins</td><td>{metrics.get('max_consecutive_wins', 0)}</td></tr>")
        html.append(f"<tr><td>Max Consecutive Losses</td><td>{metrics.get('max_consecutive_losses', 0)}</td></tr>")
        html.append(f"<tr><td>Win/Loss Ratio</td><td>{metrics.get('win_loss_ratio', 0):.2f}</td></tr>")
        html.append(f"<tr><td>Expectancy</td><td>{metrics.get('expectancy', 0):.2f}</td></tr>")
        html.append(f"<tr><td>Avg Holding Period</td><td>{metrics.get('avg_holding_period', 0):.2f} hours</td></tr>")
        html.append("</table>")
        html.append("</div>") # End section
        
        # Drawdown section
        html.append("<div class='section'>")
        html.append("<h2>Drawdown Analysis</h2>")
        html.append("<p>")
        html.append(f"Total Drawdown Periods: {metrics.get('total_drawdown_periods', 0)}<br>")
        html.append(f"Avg Drawdown Magnitude: {metrics.get('avg_drawdown_magnitude', 0)*100:.2f}%<br>")
        html.append(f"Avg Drawdown Duration: {metrics.get('avg_drawdown_duration', 0):.2f} days<br>")
        html.append(f"Avg Recovery Duration: {metrics.get('avg_recovery_duration', 0):.2f} days<br>")
        html.append("</p>")
        
        # Top drawdowns
        html.append("<h3>Top Drawdowns</h3>")
        html.append("<table>")
        html.append("<tr><th>#</th><th>Magnitude</th><th>Duration</th><th>Recovery</th></tr>")
        
        top_drawdowns = metrics.get('top_drawdowns', [])
        if top_drawdowns:
            for i, dd in enumerate(top_drawdowns, 1):
                recovery = f"{dd.get('recovery_days', 'N/A')} days" if dd.get('recovery_days') is not None else "Ongoing"
                html.append(f"<tr>")
                html.append(f"<td>{i}</td>")
                html.append(f"<td>{dd.get('max_drawdown', 0)*100:.2f}%</td>")
                html.append(f"<td>{dd.get('duration_days', 0)} days</td>")
                html.append(f"<td>{recovery}</td>")
                html.append(f"</tr>")
        else:
            html.append("<tr><td colspan='4'>No significant drawdowns recorded.</td></tr>")
        
        html.append("</table>")
        html.append("</div>") # End section
        
        # Document end
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def _generate_markdown_report(self, metrics: Dict, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> str:
        """Generate a markdown performance report."""
        md = []
        
        # Title
        md.append("# Trading Performance Report")
        md.append("")
        
        # Summary section
        md.append("## Summary")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        md.append(f"| Total Return | {metrics.get('total_return_pct', 0):.2f}% |")
        md.append(f"| Annual Return | {metrics.get('annual_return_pct', 0):.2f}% |")
        md.append(f"| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |")
        md.append(f"| Max Drawdown | {metrics.get('max_drawdown_pct', 0):.2f}% |")
        md.append(f"| Win Rate | {metrics.get('win_rate', 0)*100:.2f}% |")
        md.append(f"| Profit Factor | {metrics.get('profit_factor', 0):.2f} |")
        md.append(f"| Total Trades | {metrics.get('total_trades', 0)} |")
        md.append("")
        
        # Returns section
        md.append("## Returns")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        md.append(f"| Total Return | {metrics.get('total_return_pct', 0):.2f}% |")
        md.append(f"| Annual Return | {metrics.get('annual_return_pct', 0):.2f}% |")
        md.append(f"| Volatility (Annual) | {metrics.get('volatility_annual_pct', 0):.2f}% |")
        md.append(f"| Avg Daily Return | {metrics.get('avg_daily_return_pct', 0):.4f}% |")
        md.append("")
        
        # Risk section
        md.append("## Risk Metrics")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        md.append(f"| Max Drawdown | {metrics.get('max_drawdown_pct', 0):.2f}% |")
        md.append(f"| Max Drawdown Duration | {metrics.get('max_drawdown_duration', 0)} days |")
        md.append(f"| Avg Drawdown | {metrics.get('avg_drawdown_pct', 0):.2f}% |")
        md.append(f"| Recovery Factor | {metrics.get('recovery_factor', 0):.2f} |")
        md.append(f"| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |")
        md.append(f"| Sortino Ratio | {metrics.get('sortino_ratio', 0):.2f} |")
        md.append(f"| Calmar Ratio | {metrics.get('calmar_ratio', 0):.2f} |")
        md.append("")
        
        # Trade statistics section
        md.append("## Trade Statistics")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        md.append(f"| Total Trades | {metrics.get('total_trades', 0)} |")
        md.append(f"| Winning Trades | {metrics.get('winning_trades', 0)} ({metrics.get('win_rate', 0)*100:.2f}%) |")
        md.append(f"| Losing Trades | {metrics.get('losing_trades', 0)} ({(1-metrics.get('win_rate', 0))*100:.2f}%) |")
        md.append(f"| Profit Factor | {metrics.get('profit_factor', 0):.2f} |")
        md.append(f"| Average Trade | {metrics.get('avg_trade_pnl', 0):.2f} |")
        md.append(f"| Average Winner | {metrics.get('avg_winning_trade', 0):.2f} |")
        md.append(f"| Average Loser | {metrics.get('avg_losing_trade', 0):.2f} |")
        md.append(f"| Largest Winner | {metrics.get('largest_winning_trade', 0):.2f} |")
        md.append(f"| Largest Loser | {metrics.get('largest_losing_trade', 0):.2f} |")
        md.append(f"| Max Consecutive Wins | {metrics.get('max_consecutive_wins', 0)} |")
        md.append(f"| Max Consecutive Losses | {metrics.get('max_consecutive_losses', 0)} |")
        md.append(f"| Win/Loss Ratio | {metrics.get('win_loss_ratio', 0):.2f} |")
        md.append(f"| Expectancy | {metrics.get('expectancy', 0):.2f} |")
        md.append(f"| Avg Holding Period | {metrics.get('avg_holding_period', 0):.2f} hours |")
        md.append("")
        
        # Drawdown section
        md.append("## Drawdown Analysis")
        md.append("")
        md.append(f"- Total Drawdown Periods: {metrics.get('total_drawdown_periods', 0)}")
        md.append(f"- Avg Drawdown Magnitude: {metrics.get('avg_drawdown_magnitude', 0)*100:.2f}%")
        md.append(f"- Avg Drawdown Duration: {metrics.get('avg_drawdown_duration', 0):.2f} days")
        md.append(f"- Avg Recovery Duration: {metrics.get('avg_recovery_duration', 0):.2f} days")
        md.append("")
        
        # Top drawdowns
        md.append("### Top Drawdowns")
        md.append("")
        md.append("| # | Magnitude | Duration | Recovery |")
        md.append("|---|-----------|----------|----------|")
        
        top_drawdowns = metrics.get('top_drawdowns', [])
        if top_drawdowns:
            for i, dd in enumerate(top_drawdowns, 1):
                recovery = f"{dd.get('recovery_days', 'N/A')} days" if dd.get('recovery_days') is not None else "Ongoing"
                md.append(f"| {i} | {dd.get('max_drawdown', 0)*100:.2f}% | {dd.get('duration_days', 0)} days | {recovery} |")
        else:
            md.append("| - | No significant drawdowns recorded. | - | - |")
        
        return "\n".join(md)