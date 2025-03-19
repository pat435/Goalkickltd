"""
Visualization module for the Goalkick Ltd Trading Bot.
Creates visualizations for trading performance and market analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timedelta
import os
from pathlib import Path
import json

from config.logging_config import get_logger
from config.bot_config import PATHS
from src.utils.error_handling import handle_error

logger = get_logger("analysis.reporting.visualization")

class PerformanceVisualization:
    """Class for creating performance visualizations."""
    
    def __init__(self, datastore=None):
        """
        Initialize the PerformanceVisualization.
        
        Args:
            datastore: DataStore instance (optional)
        """
        self.datastore = datastore
        
        # Set default style
        plt.style.use('seaborn-darkgrid')
        sns.set_palette('deep')
        
        # Create reports directory if it doesn't exist
        reports_dir = Path(PATHS['reports'])
        reports_dir.mkdir(exist_ok=True, parents=True)
    
    def create_equity_curve(self, equity_data=None, start_date=None, end_date=None, 
                           include_drawdowns=True, show_trades=True, output_file=None):
        """
        Create equity curve visualization.
        
        Args:
            equity_data (list/dict): Equity curve data points
            start_date (datetime): Start date for filtering
            end_date (datetime): End date for filtering
            include_drawdowns (bool): Whether to highlight drawdowns
            show_trades (bool): Whether to mark individual trades
            output_file (str): Output file path
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Get equity data if not provided
            if equity_data is None and self.datastore:
                # Get performance metrics
                perf_data = self.datastore.get_performance()
                
                if not perf_data:
                    logger.warning("No performance data available for equity curve")
                    return None
                
                # Extract equity values
                equity_data = []
                for timestamp, row in perf_data.iterrows():
                    equity_data.append({
                        'timestamp': timestamp.timestamp() * 1000,  # Convert to milliseconds
                        'balance': row['equity']
                    })
            
            if not equity_data:
                logger.warning("No equity data provided for visualization")
                return None
            
            # Convert equity data to DataFrame
            if isinstance(equity_data, list):
                # Convert timestamp to datetime
                for point in equity_data:
                    if isinstance(point.get('timestamp'), (int, float)):
                        point['datetime'] = datetime.fromtimestamp(point['timestamp'] / 1000)
                
                df = pd.DataFrame(equity_data)
            else:
                df = pd.DataFrame(equity_data)
                if 'datetime' not in df.columns and 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Filter by date range if specified
            if start_date is not None:
                df = df[df['datetime'] >= start_date]
            
            if end_date is not None:
                df = df[df['datetime'] <= end_date]
            
            if df.empty:
                logger.warning("No data available after filtering")
                return None
            
            # Sort by time
            df = df.sort_values('datetime')
            
            # Create figure
            plt.figure(figsize=(12, 7))
            
            # Plot equity curve
            plt.plot(df['datetime'], df['balance'], linewidth=2, label='Equity')
            
            # Plot drawdowns if requested
            if include_drawdowns:
                # Calculate drawdowns
                df['peak'] = df['balance'].cummax()
                df['drawdown'] = df['peak'] - df['balance']
                df['drawdown_pct'] = df['drawdown'] / df['peak']
                
                # Highlight drawdown periods
                significant_dd = df['drawdown_pct'] > 0.05  # 5% drawdown threshold
                if significant_dd.any():
                    plt.fill_between(df['datetime'], df['balance'], df['peak'], 
                                    where=significant_dd, color='red', alpha=0.3, 
                                    label='Drawdowns > 5%')
            
            # Mark trades if requested and available
            if show_trades and 'pnl' in df.columns:
                # Mark winning trades
                winning_trades = df[df['pnl'] > 0]
                if not winning_trades.empty:
                    plt.scatter(winning_trades['datetime'], winning_trades['balance'], 
                               color='green', marker='^', s=50, label='Winning Trades')
                
                # Mark losing trades
                losing_trades = df[df['pnl'] < 0]
                if not losing_trades.empty:
                    plt.scatter(losing_trades['datetime'], losing_trades['balance'], 
                              color='red', marker='v', s=50, label='Losing Trades')
            
            # Configure the plot
            plt.title('Equity Curve', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Equity', fontsize=12)
            plt.legend()
            plt.grid(True)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
            
            # Format y-axis with comma separator
            plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
            
            plt.tight_layout()
            
            # Save figure if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Equity curve saved to {output_file}")
                return output_file
            else:
                # Create default output file
                reports_dir = Path(PATHS['reports'])
                output_file = reports_dir / f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Strategy comparison chart saved to {output_file}")
                return str(output_file)
        except Exception as e:
            logger.error(f"Error creating strategy comparison chart: {e}")
            handle_error(e, "Failed to create strategy comparison chart")
            plt.close()
            return None
    
    def create_full_report(self, report_data=None, start_date=None, end_date=None, output_dir=None):
        """
        Create a full suite of visualizations as a comprehensive report.
        
        Args:
            report_data (dict): Performance report data
            start_date (datetime): Start date for filtering
            end_date (datetime): End date for filtering
            output_dir (str): Output directory for report files
            
        Returns:
            dict: Dictionary of visualization paths
        """
        try:
            # Get report data if not provided
            if report_data is None and self.datastore:
                # Generate a comprehensive performance report
                from analysis.reporting.performance_report import PerformanceReport
                perf_report = PerformanceReport(self.datastore)
                report_data = perf_report.generate_report(
                    start_date=start_date,
                    end_date=end_date,
                    report_format='json'
                )
            
            if not report_data:
                logger.warning("No report data available for visualization")
                return {}
            
            # Set output directory
            if output_dir is None:
                reports_dir = Path(PATHS['reports'])
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = reports_dir / f"full_report_{timestamp}"
                output_dir.mkdir(exist_ok=True, parents=True)
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True, parents=True)
            
            # Create all visualizations
            visualizations = {}
            
            # 1. Equity curve
            equity_file = output_dir / "equity_curve.png"
            if self.create_equity_curve(
                equity_data=report_data.get('equity_curve'),
                start_date=start_date,
                end_date=end_date,
                output_file=equity_file
            ):
                visualizations['equity_curve'] = str(equity_file)
            
            # 2. Drawdown chart
            drawdown_file = output_dir / "drawdown_chart.png"
            if self.create_drawdown_chart(
                equity_data=report_data.get('equity_curve'),
                output_file=drawdown_file
            ):
                visualizations['drawdown_chart'] = str(drawdown_file)
            
            # 3. Win/loss ratio
            winloss_file = output_dir / "win_loss_ratio.png"
            if 'trades' in globals() and trades:
                # If trades were already fetched
                if self.create_win_loss_ratio_chart(
                    trades=trades,
                    output_file=winloss_file
                ):
                    visualizations['win_loss_ratio'] = str(winloss_file)
            else:
                # Fetch trades from datastore
                trades = self.datastore.get_trades(status="CLOSED") if self.datastore else None
                if self.create_win_loss_ratio_chart(
                    trades=trades,
                    output_file=winloss_file
                ):
                    visualizations['win_loss_ratio'] = str(winloss_file)
            
            # 4. Monthly performance
            monthly_file = output_dir / "monthly_performance.png"
            if self.create_monthly_performance(
                trades=trades,
                output_file=monthly_file
            ):
                visualizations['monthly_performance'] = str(monthly_file)
            
            # 5. Trade distribution
            distribution_file = output_dir / "trade_distribution.png"
            if self.create_trade_distribution(
                trades=trades,
                output_file=distribution_file
            ):
                visualizations['trade_distribution'] = str(distribution_file)
            
            # 6. Strategy comparison
            strategy_file = output_dir / "strategy_comparison.png"
            if self.create_strategy_comparison(
                strategies_data={"strategies": report_data.get('by_strategy', {})},
                output_file=strategy_file
            ):
                visualizations['strategy_comparison'] = str(strategy_file)
            
            # Create HTML report index
            html_report = output_dir / "report.html"
            self._create_html_report_index(
                report_data=report_data,
                visualizations=visualizations,
                output_file=html_report
            )
            
            visualizations['html_report'] = str(html_report)
            
            logger.info(f"Full report generated in {output_dir}")
            return visualizations
        except Exception as e:
            logger.error(f"Error creating full report: {e}")
            handle_error(e, "Failed to create full report")
            return {}
    
    def _create_html_report_index(self, report_data, visualizations, output_file):
        """
        Create an HTML index file for the full report.
        
        Args:
            report_data (dict): Performance report data
            visualizations (dict): Dictionary of visualization paths
            output_file (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a relative path for images
            rel_paths = {}
            for key, path in visualizations.items():
                if path and isinstance(path, str):
                    rel_paths[key] = os.path.basename(path)
            
            # Extract summary data
            summary = report_data.get('summary', {})
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Performance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .summary-box {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 20px; }}
                    .metric {{ display: inline-block; margin-right: 20px; margin-bottom: 10px; }}
                    .metric-value {{ font-size: 18px; font-weight: bold; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    .chart-container {{ margin-top: 30px; margin-bottom: 40px; }}
                    .chart {{ width: 100%; max-width: 1000px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Trading Performance Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="summary-box">
                        <h2>Summary</h2>
                        <div class="metric">
                            <div>Total Trades</div>
                            <div class="metric-value">{summary.get('total_trades', 0)}</div>
                        </div>
                        <div class="metric">
                            <div>Win Rate</div>
                            <div class="metric-value">{summary.get('win_rate', 0):.2%}</div>
                        </div>
                        <div class="metric">
                            <div>Net Profit</div>
                            <div class="metric-value {('positive' if summary.get('net_profit', 0) >= 0 else 'negative')}">${summary.get('net_profit', 0):.2f}</div>
                        </div>
                        <div class="metric">
                            <div>Profit Factor</div>
                            <div class="metric-value">{summary.get('profit_factor', 0):.2f}</div>
                        </div>
                        <div class="metric">
                            <div>Sharpe Ratio</div>
                            <div class="metric-value">{summary.get('sharpe_ratio', 0):.2f}</div>
                        </div>
                        <div class="metric">
                            <div>Max Drawdown</div>
                            <div class="metric-value negative">{summary.get('max_drawdown_percentage', 0):.2%}</div>
                        </div>
                    </div>
            """
            
            # Add visualizations
            if 'equity_curve' in rel_paths:
                html_content += f"""
                    <div class="chart-container">
                        <h2>Equity Curve</h2>
                        <img class="chart" src="{rel_paths['equity_curve']}" alt="Equity Curve">
                    </div>
                """
            
            if 'drawdown_chart' in rel_paths:
                html_content += f"""
                    <div class="chart-container">
                        <h2>Drawdown Chart</h2>
                        <img class="chart" src="{rel_paths['drawdown_chart']}" alt="Drawdown Chart">
                    </div>
                """
            
            if 'monthly_performance' in rel_paths:
                html_content += f"""
                    <div class="chart-container">
                        <h2>Monthly Performance</h2>
                        <img class="chart" src="{rel_paths['monthly_performance']}" alt="Monthly Performance">
                    </div>
                """
            
            if 'win_loss_ratio' in rel_paths:
                html_content += f"""
                    <div class="chart-container">
                        <h2>Win/Loss Analysis</h2>
                        <img class="chart" src="{rel_paths['win_loss_ratio']}" alt="Win/Loss Ratio">
                    </div>
                """
            
            if 'trade_distribution' in rel_paths:
                html_content += f"""
                    <div class="chart-container">
                        <h2>Trade Distribution</h2>
                        <img class="chart" src="{rel_paths['trade_distribution']}" alt="Trade Distribution">
                    </div>
                """
            
            if 'strategy_comparison' in rel_paths:
                html_content += f"""
                    <div class="chart-container">
                        <h2>Strategy Comparison</h2>
                        <img class="chart" src="{rel_paths['strategy_comparison']}" alt="Strategy Comparison">
                    </div>
                """
            
            # Add strategy performance table
            if 'by_strategy' in report_data and report_data['by_strategy']:
                html_content += """
                    <h2>Strategy Performance</h2>
                    <table>
                        <tr>
                            <th>Strategy</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Net Profit</th>
                            <th>Avg. Trade</th>
                        </tr>
                """
                
                for strategy, metrics in report_data['by_strategy'].items():
                    profit_class = "positive" if metrics.get('net_profit', 0) >= 0 else "negative"
                    html_content += f"""
                        <tr>
                            <td>{strategy}</td>
                            <td>{metrics.get('total_trades', 0)}</td>
                            <td>{metrics.get('win_rate', 0):.2%}</td>
                            <td class="{profit_class}">${metrics.get('net_profit', 0):.2f}</td>
                            <td>${metrics.get('average_trade', 0):.2f}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                """
            
            # Close HTML document
            html_content += """
                </div>
            </body>
            </html>
            """
            
            # Write HTML file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"HTML report index saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error creating HTML report index: {e}")
            handle_error(e, "Failed to create HTML report index")
            return False


class MarketVisualization:
    """Class for creating market-related visualizations."""
    
    def __init__(self, datastore=None):
        """
        Initialize the MarketVisualization.
        
        Args:
            datastore: DataStore instance (optional)
        """
        self.datastore = datastore
        
        # Set default style
        plt.style.use('seaborn-darkgrid')
        sns.set_palette('deep')
        
        # Create reports directory if it doesn't exist
        reports_dir = Path(PATHS['reports'])
        reports_dir.mkdir(exist_ok=True, parents=True)
    
    def create_price_chart(self, symbol, timeframe, data=None, indicators=None, 
                          start_date=None, end_date=None, output_file=None):
        """
        Create a price chart with optional technical indicators.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            data (pd.DataFrame): Price data (optional)
            indicators (dict): Dictionary of indicators to plot
            start_date (datetime): Start date for filtering
            end_date (datetime): End date for filtering
            output_file (str): Output file path
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Get data if not provided
            if data is None and self.datastore:
                data = self.datastore.get_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=int(start_date.timestamp() * 1000) if start_date else None,
                    end_time=int(end_date.timestamp() * 1000) if end_date else None
                )
            
            if data is None or data.empty:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return None
            
            # Convert index to datetime if not already
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Create figure
            fig, ax1 = plt.subplots(figsize=(14, 7))
            
            # Plot price
            ax1.plot(data.index, data['close'], linewidth=1.5, label='Price', color='black')
            
            # Add indicators if provided
            if indicators:
                for name, indicator_data in indicators.items():
                    if isinstance(indicator_data, tuple) and len(indicator_data) >= 2:
                        # Multiple lines (e.g., Bollinger Bands)
                        for i, (line_name, line_data) in enumerate(indicator_data):
                            if i == 0:
                                ax1.plot(data.index, line_data, label=f"{name} {line_name}", linewidth=1.5)
                            else:
                                ax1.plot(data.index, line_data, label=f"{name} {line_name}", linewidth=1)
                    else:
                        # Single line
                        ax1.plot(data.index, indicator_data, label=name, linewidth=1.5)
            
            # Configure the plot
            title = f"{symbol} {timeframe} Price Chart"
            if start_date and end_date:
                title += f" ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
            
            ax1.set_title(title, fontsize=16)
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Price', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(True)
            
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            
            # Save figure if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Price chart saved to {output_file}")
                return output_file
            else:
                # Create default output file
                reports_dir = Path(PATHS['reports'])
                output_file = reports_dir / f"{symbol}_{timeframe}_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Price chart saved to {output_file}")
                return str(output_file)
        except Exception as e:
            logger.error(f"Error creating price chart: {e}")
            handle_error(e, "Failed to create price chart")
            plt.close()
            return None
    
    def create_correlation_matrix(self, symbols=None, timeframe='1d', lookback_days=30, output_file=None):
        """
        Create a correlation matrix visualization for multiple symbols.
        
        Args:
            symbols (list): List of symbols to include
            timeframe (str): Timeframe for data
            lookback_days (int): Number of days to look back
            output_file (str): Output file path
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Default symbols if not provided
            if symbols is None or not symbols:
                from config.trading_pairs import get_all_active_symbols
                symbols = get_all_active_symbols()[:10]  # Limit to 10 symbols
            
            if not symbols:
                logger.warning("No symbols provided for correlation matrix")
                return None
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get data for all symbols
            price_data = {}
            
            for symbol in symbols:
                if self.datastore:
                    data = self.datastore.get_market_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=int(start_date.timestamp() * 1000),
                        end_time=int(end_date.timestamp() * 1000)
                    )
                    
                    if data is not None and not data.empty:
                        price_data[symbol] = data['close']
                else:
                    logger.warning(f"No data available for {symbol} {timeframe}")
            
            if not price_data:
                logger.warning("No price data available for correlation matrix")
                return None
            
            # Create DataFrame with closing prices for all symbols
            df = pd.DataFrame(price_data)
            
            # Calculate percentage returns
            returns = df.pct_change().dropna()
            
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                      square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .8})
            
            plt.title(f"Correlation Matrix ({timeframe} Returns, {lookback_days} days)", fontsize=16)
            plt.tight_layout()
            
            # Save figure if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Correlation matrix saved to {output_file}")
                return output_file
            else:
                # Create default output file
                reports_dir = Path(PATHS['reports'])
                output_file = reports_dir / f"correlation_matrix_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Correlation matrix saved to {output_file}")
                return str(output_file)
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            handle_error(e, "Failed to create correlation matrix")
            plt.close()
            return None
    
    def create_volatility_comparison(self, symbols=None, timeframe='1d', lookback_days=30, output_file=None):
        """
        Create a volatility comparison visualization for multiple symbols.
        
        Args:
            symbols (list): List of symbols to include
            timeframe (str): Timeframe for data
            lookback_days (int): Number of days to look back
            output_file (str): Output file path
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Default symbols if not provided
            if symbols is None or not symbols:
                from config.trading_pairs import get_all_active_symbols
                symbols = get_all_active_symbols()[:10]  # Limit to 10 symbols
            
            if not symbols:
                logger.warning("No symbols provided for volatility comparison")
                return None
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get data for all symbols
            volatility_data = {}
            
            for symbol in symbols:
                if self.datastore:
                    data = self.datastore.get_market_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=int(start_date.timestamp() * 1000),
                        end_time=int(end_date.timestamp() * 1000)
                    )
                    
                    if data is not None and not data.empty:
                        # Calculate returns
                        returns = data['close'].pct_change().dropna()
                        
                        # Calculate volatility (standard deviation of returns)
                        volatility = returns.std() * 100  # Convert to percentage
                        
                        # Calculate average true range
                        atr = (data['high'] - data['low']).mean() / data['close'].mean() * 100  # Normalized ATR
                        
                        volatility_data[symbol] = {
                            'returns_volatility': volatility,
                            'atr': atr
                        }
                else:
                    logger.warning(f"No data available for {symbol} {timeframe}")
            
            if not volatility_data:
                logger.warning("No volatility data available for comparison")
                return None
            
            # Create DataFrame for visualization
            df = pd.DataFrame(volatility_data).T
            df = df.sort_values('returns_volatility', ascending=False)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # Plot returns volatility
            bars1 = ax1.bar(df.index, df['returns_volatility'], color='blue', alpha=0.7)
            ax1.set_title(f"Returns Volatility Comparison ({timeframe}, {lookback_days} days)", fontsize=16)
            ax1.set_ylabel('Volatility (%)', fontsize=12)
            ax1.grid(True, axis='y')
            
            # Add values on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.annotate(f'{height:.2f}%',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')
            
            # Format x-axis labels
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Plot normalized ATR
            bars2 = ax2.bar(df.index, df['atr'], color='green', alpha=0.7)
            ax2.set_title(f"Normalized Average True Range", fontsize=16)
            ax2.set_ylabel('ATR/Price (%)', fontsize=12)
            ax2.grid(True, axis='y')
            
            # Add values on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}%',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')
            
            # Format x-axis labels
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save figure if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Volatility comparison saved to {output_file}")
                return output_file
            else:
                # Create default output file
                reports_dir = Path(PATHS['reports'])
                output_file = reports_dir / f"volatility_comparison_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Volatility comparison saved to {output_file}")
                return str(output_file)
        except Exception as e:
            logger.error(f"Error creating volatility comparison: {e}")
            handle_error(e, "Failed to create volatility comparison")
            plt.close()
            return None
    
    def create_drawdown_chart(self, equity_data=None, output_file=None):
        """
        Create drawdown visualization.
        
        Args:
            equity_data (list/dict): Equity curve data points
            output_file (str): Output file path
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Get equity data if not provided
            if equity_data is None and self.datastore:
                # Get performance metrics
                perf_data = self.datastore.get_performance()
                
                if not perf_data:
                    logger.warning("No performance data available for drawdown chart")
                    return None
                
                # Extract equity values
                equity_data = []
                for timestamp, row in perf_data.iterrows():
                    equity_data.append({
                        'timestamp': timestamp.timestamp() * 1000,  # Convert to milliseconds
                        'balance': row['equity']
                    })
            
            if not equity_data:
                logger.warning("No equity data provided for visualization")
                return None
            
            # Convert equity data to DataFrame
            if isinstance(equity_data, list):
                # Convert timestamp to datetime
                for point in equity_data:
                    if isinstance(point.get('timestamp'), (int, float)):
                        point['datetime'] = datetime.fromtimestamp(point['timestamp'] / 1000)
                
                df = pd.DataFrame(equity_data)
            else:
                df = pd.DataFrame(equity_data)
                if 'datetime' not in df.columns and 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sort by time
            df = df.sort_values('datetime')
            
            # Calculate drawdowns
            df['peak'] = df['balance'].cummax()
            df['drawdown'] = df['peak'] - df['balance']
            df['drawdown_pct'] = (df['drawdown'] / df['peak']) * 100  # Convert to percentage
            
            # Create figure
            plt.figure(figsize=(12, 7))
            
            # Plot drawdown percentage
            plt.fill_between(df['datetime'], 0, df['drawdown_pct'], color='red', alpha=0.5)
            plt.plot(df['datetime'], df['drawdown_pct'], color='darkred', linewidth=1)
            
            # Configure the plot
            plt.title('Drawdown Chart', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Drawdown (%)', fontsize=12)
            plt.grid(True)
            
            # Invert y-axis for better visualization (drawdowns are negative)
            plt.gca().invert_yaxis()
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            
            # Save figure if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Drawdown chart saved to {output_file}")
                return output_file
            else:
                # Create default output file
                reports_dir = Path(PATHS['reports'])
                output_file = reports_dir / f"drawdown_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Drawdown chart saved to {output_file}")
                return str(output_file)
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {e}")
            handle_error(e, "Failed to create drawdown chart")
            plt.close()
            return None
    
    def create_trade_distribution(self, trades=None, output_file=None):
        """
        Create trade distribution visualization.
        
        Args:
            trades (list): List of trade dictionaries
            output_file (str): Output file path
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Get trades if not provided
            if trades is None and self.datastore:
                trades = self.datastore.get_trades(status="CLOSED")
            
            if not trades:
                logger.warning("No trades available for distribution visualization")
                return None
            
            # Extract PnL from trades
            pnl_values = [t.get('pnl', 0) for t in trades]
            
            if not pnl_values:
                logger.warning("No PnL data available for distribution visualization")
                return None
            
            # Create figure with two subplots (histogram and box plot)
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Histogram of PnL
            sns.histplot(pnl_values, kde=True, bins=30, color='blue', ax=axes[0])
            axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1, label='Breakeven')
            
            # Add mean and median lines
            mean_pnl = np.mean(pnl_values)
            median_pnl = np.median(pnl_values)
            
            axes[0].axvline(x=mean_pnl, color='green', linestyle='-', linewidth=1, label=f'Mean: {mean_pnl:.2f}')
            axes[0].axvline(x=median_pnl, color='purple', linestyle='-', linewidth=1, label=f'Median: {median_pnl:.2f}')
            
            axes[0].set_title('Trade PnL Distribution', fontsize=16)
            axes[0].set_xlabel('PnL', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].legend()
            axes[0].grid(True)
            
            # Box plot of PnL
            sns.boxplot(x=pnl_values, ax=axes[1], color='skyblue')
            axes[1].set_title('PnL Box Plot', fontsize=14)
            axes[1].set_xlabel('PnL', fontsize=12)
            axes[1].grid(True)
            
            plt.tight_layout()
            
            # Save figure if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Trade distribution saved to {output_file}")
                return output_file
            else:
                # Create default output file
                reports_dir = Path(PATHS['reports'])
                output_file = reports_dir / f"trade_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Trade distribution saved to {output_file}")
                return str(output_file)
        except Exception as e:
            logger.error(f"Error creating trade distribution visualization: {e}")
            handle_error(e, "Failed to create trade distribution visualization")
            plt.close()
            return None
    
    def create_win_loss_ratio_chart(self, trades=None, output_file=None):
        """
        Create win/loss ratio visualization.
        
        Args:
            trades (list): List of trade dictionaries
            output_file (str): Output file path
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Get trades if not provided
            if trades is None and self.datastore:
                trades = self.datastore.get_trades(status="CLOSED")
            
            if not trades:
                logger.warning("No trades available for win/loss ratio visualization")
                return None
            
            # Categorize trades
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            breakeven_trades = [t for t in trades if t.get('pnl', 0) == 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            breakeven_count = len(breakeven_trades)
            total_count = len(trades)
            
            # Calculate average win and loss
            avg_win = np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0
            win_rate = win_count / total_count if total_count > 0 else 0
            
            # Create pie chart for win/loss ratio
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # Pie chart of trade outcomes
            labels = ['Winning', 'Losing', 'Breakeven']
            sizes = [win_count, loss_count, breakeven_count]
            colors = ['green', 'red', 'gray']
            explode = (0.1, 0, 0)  # Explode winning slice
            
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', 
                   shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax1.set_title('Trade Outcome Distribution', fontsize=14)
            
            # Bar chart of average win vs loss
            bar_labels = ['Average Win', 'Average Loss', 'Win Rate']
            bar_values = [avg_win, abs(avg_loss), win_rate * 100]  # Win rate as percentage
            bar_colors = ['green', 'red', 'blue']
            
            bars = ax2.bar(bar_labels, bar_values, color=bar_colors, alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            ax2.set_title('Performance Metrics', fontsize=14)
            ax2.set_ylabel('Value ($)', fontsize=12)
            ax2.grid(True, axis='y')
            
            # Add a second y-axis for win rate
            ax2_twin = ax2.twinx()
            ax2_twin.set_ylabel('Win Rate (%)', fontsize=12)
            ax2_twin.set_ylim(0, 100)
            
            plt.tight_layout()
            
            # Save figure if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Win/loss ratio chart saved to {output_file}")
                return output_file
            else:
                # Create default output file
                reports_dir = Path(PATHS['reports'])
                output_file = reports_dir / f"win_loss_ratio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Win/loss ratio chart saved to {output_file}")
                return str(output_file)
        except Exception as e:
            logger.error(f"Error creating win/loss ratio chart: {e}")
            handle_error(e, "Failed to create win/loss ratio chart")
            plt.close()
            return None
    
    def create_monthly_performance(self, trades=None, output_file=None):
        """
        Create monthly performance visualization.
        
        Args:
            trades (list): List of trade dictionaries
            output_file (str): Output file path
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Get trades if not provided
            if trades is None and self.datastore:
                trades = self.datastore.get_trades(status="CLOSED")
            
            if not trades:
                logger.warning("No trades available for monthly performance visualization")
                return None
            
            # Group trades by month
            monthly_pnl = {}
            monthly_win_rate = {}
            monthly_trade_count = {}
            
            for trade in trades:
                if trade.get('exit_time'):
                    exit_date = datetime.fromtimestamp(trade['exit_time'] / 1000)
                    month_key = exit_date.strftime('%Y-%m')
                    
                    if month_key not in monthly_pnl:
                        monthly_pnl[month_key] = 0
                        monthly_win_rate[month_key] = {'wins': 0, 'total': 0}
                        monthly_trade_count[month_key] = 0
                    
                    monthly_pnl[month_key] += trade.get('pnl', 0)
                    monthly_trade_count[month_key] += 1
                    
                    if trade.get('pnl', 0) > 0:
                        monthly_win_rate[month_key]['wins'] += 1
                    
                    monthly_win_rate[month_key]['total'] += 1
            
            # Calculate win rates
            for month, data in monthly_win_rate.items():
                if data['total'] > 0:
                    monthly_win_rate[month] = data['wins'] / data['total'] * 100
                else:
                    monthly_win_rate[month] = 0
            
            # Sort by month
            sorted_months = sorted(monthly_pnl.keys())
            sorted_pnl = [monthly_pnl[month] for month in sorted_months]
            sorted_win_rates = [monthly_win_rate[month] for month in sorted_months]
            sorted_trade_counts = [monthly_trade_count[month] for month in sorted_months]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # Bar chart of monthly PnL
            bars = ax1.bar(sorted_months, sorted_pnl, color=['green' if pnl >= 0 else 'red' for pnl in sorted_pnl])
            
            # Add trade count annotations
            for i, bar in enumerate(bars):
                ax1.text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + (1 if bar.get_height() >= 0 else -15),
                       f"{sorted_trade_counts[i]} trades", 
                       ha='center', va='bottom', rotation=0, fontsize=8)
            
            ax1.set_title('Monthly Performance', fontsize=16)
            ax1.set_xlabel('Month', fontsize=12)
            ax1.set_ylabel('Profit & Loss ($)', fontsize=12)
            ax1.grid(True, axis='y')
            
            # Format x-axis labels
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Line chart of monthly win rates
            ax2.plot(sorted_months, sorted_win_rates, marker='o', linestyle='-', color='blue', linewidth=2)
            ax2.set_xlabel('Month', fontsize=12)
            ax2.set_ylabel('Win Rate (%)', fontsize=12)
            ax2.set_ylim(0, 100)
            ax2.grid(True)
            
            # Format x-axis labels
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save figure if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Monthly performance chart saved to {output_file}")
                return output_file
            else:
                # Create default output file
                reports_dir = Path(PATHS['reports'])
                output_file = reports_dir / f"monthly_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Monthly performance chart saved to {output_file}")
                return str(output_file)
        except Exception as e:
            logger.error(f"Error creating monthly performance chart: {e}")
            handle_error(e, "Failed to create monthly performance chart")
            plt.close()
            return None
    
    def create_strategy_comparison(self, strategies_data=None, output_file=None):
        """
        Create strategy comparison visualization.
        
        Args:
            strategies_data (dict): Strategy comparison data
            output_file (str): Output file path
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Generate strategy data if not provided
            if strategies_data is None and self.datastore:
                # For this example, we'll create a simulated strategy comparison
                # In a real implementation, you would get this from a performance report
                strategies_data = {
                    "strategies": []
                }
                
                # Group trades by strategy
                trades = self.datastore.get_trades(status="CLOSED")
                
                if not trades:
                    logger.warning("No trades available for strategy comparison")
                    return None
                
                strategy_metrics = {}
                
                for trade in trades:
                    strategy = trade.get('strategy', 'Unknown')
                    
                    if strategy not in strategy_metrics:
                        strategy_metrics[strategy] = {
                            "name": strategy,
                            "total_trades": 0,
                            "winning_trades": 0,
                            "net_profit": 0,
                            "total_profit": 0,
                            "total_loss": 0
                        }
                    
                    strategy_metrics[strategy]['total_trades'] += 1
                    strategy_metrics[strategy]['net_profit'] += trade.get('pnl', 0)
                    
                    if trade.get('pnl', 0) > 0:
                        strategy_metrics[strategy]['winning_trades'] += 1
                        strategy_metrics[strategy]['total_profit'] += trade.get('pnl', 0)
                    else:
                        strategy_metrics[strategy]['total_loss'] += abs(trade.get('pnl', 0))
                
                # Calculate derived metrics
                for strategy, metrics in strategy_metrics.items():
                    metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
                    metrics['profit_factor'] = metrics['total_profit'] / metrics['total_loss'] if metrics['total_loss'] > 0 else float('inf')
                    metrics['average_trade'] = metrics['net_profit'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
                
                strategies_data['strategies'] = list(strategy_metrics.values())
            
            if not strategies_data or not strategies_data.get('strategies'):
                logger.warning("No strategy data available for comparison visualization")
                return None
            
            # Extract strategy data
            strategies = strategies_data['strategies']
            
            if not strategies:
                logger.warning("No strategies found in data")
                return None
            
            # Sort strategies by net profit
            strategies.sort(key=lambda x: x.get('net_profit', 0), reverse=True)
            
            # Extract metrics for visualization
            strategy_names = [s.get('name', 'Unknown') for s in strategies]
            net_profits = [s.get('net_profit', 0) for s in strategies]
            win_rates = [s.get('win_rate', 0) * 100 for s in strategies]  # Convert to percentage
            profit_factors = [min(s.get('profit_factor', 0), 5) for s in strategies]  # Cap at 5 for better visualization
            trade_counts = [s.get('total_trades', 0) for s in strategies]
            
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
            
            # Bar chart of net profits
            bars1 = ax1.bar(strategy_names, net_profits, color=['green' if p >= 0 else 'red' for p in net_profits])
            ax1.set_title('Strategy Net Profit Comparison', fontsize=16)
            ax1.set_xlabel('Strategy', fontsize=12)
            ax1.set_ylabel('Net Profit ($)', fontsize=12)
            ax1.grid(True, axis='y')
            
            # Add profit value annotations
            for bar in bars1:
                height = bar.get_height()
                ax1.annotate(f'${height:.2f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom' if height >= 0 else 'top')
            
            # Format x-axis labels
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Bar chart of win rates
            bars2 = ax2.bar(strategy_names, win_rates, color='blue')
            ax2.set_title('Strategy Win Rate Comparison', fontsize=16)
            ax2.set_xlabel('Strategy', fontsize=12)
            ax2.set_ylabel('Win Rate (%)', fontsize=12)
            ax2.set_ylim(0, 100)
            ax2.grid(True, axis='y')
            
            # Add win rate annotations
            for bar in bars2:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}%',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')
            
            # Format x-axis labels
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Scatter plot of profit factor vs. trade count
            sc = ax3.scatter(profit_factors, trade_counts, c=net_profits, cmap='RdYlGn', 
                          s=100, alpha=0.7, edgecolors='black')
            
            # Add strategy name annotations
            for i, name in enumerate(strategy_names):
                ax3.annotate(name,
                          xy=(profit_factors[i], trade_counts[i]),
                          xytext=(5, 5),
                          textcoords="offset points")
            
            ax3.set_title('Profit Factor vs. Trade Count', fontsize=16)
            ax3.set_xlabel('Profit Factor', fontsize=12)
            ax3.set_ylabel('Number of Trades', fontsize=12)
            ax3.grid(True)
            
            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax3)
            cbar.set_label('Net Profit ($)', fontsize=10)
            
            plt.tight_layout()
            
            # Save figure if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Strategy comparison chart saved to {output_file}")
                return output_file
            else:
                # Create default output file
                reports_dir = Path(PATHS['reports'])
                output_file = reports_dir / f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Strategy comparison chart saved to {output_file}")
                return str(output_file)
        except Exception as e:
            logger.error(f"Error creating strategy comparison chart: {e}")
            handle_error(e, "Failed to create strategy comparison chart")
            plt.close()
            return None