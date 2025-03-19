"""
Performance reporting module for the Goalkick Ltd Trading Bot.
Generates comprehensive performance reports and metrics for strategy evaluation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from config.logging_config import get_logger
from config.bot_config import PATHS, PERFORMANCE_CONFIG
from src.utils.error_handling import handle_error

logger = get_logger("analysis.reporting.performance")

class PerformanceReport:
    """Class for generating comprehensive performance reports."""
    
    def __init__(self, datastore=None):
        """
        Initialize the PerformanceReport.
        
        Args:
            datastore: DataStore instance (optional)
        """
        self.datastore = datastore
        
        # Create reports directory if it doesn't exist
        reports_dir = Path(PATHS['reports'])
        reports_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_report(self, start_date=None, end_date=None, symbols=None, 
                       strategies=None, timeframes=None, report_format='json'):
        """
        Generate a comprehensive performance report.
        
        Args:
            start_date (str/datetime): Start date for report (format: YYYY-MM-DD)
            end_date (str/datetime): End date for report (format: YYYY-MM-DD)
            symbols (list): List of symbols to include
            strategies (list): List of strategies to include
            timeframes (list): List of timeframes to include
            report_format (str): Report format ('json', 'csv', 'html')
            
        Returns:
            dict: Report data
        """
        try:
            # Convert string dates to datetime
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Set default date range if not specified
            if end_date is None:
                end_date = datetime.now()
            
            if start_date is None:
                start_date = end_date - timedelta(days=30)
            
            # Convert to timestamps for database queries
            start_time = int(start_date.timestamp() * 1000)
            end_time = int(end_date.timestamp() * 1000)
            
            logger.info(f"Generating performance report from {start_date.date()} to {end_date.date()}")
            
            # Get trade data
            trades = []
            if self.datastore:
                trades = self.datastore.get_trades(
                    start_time=start_time,
                    end_time=end_time,
                    status="CLOSED"
                )
            
            # Filter trades by symbols, strategies, and timeframes if specified
            if symbols:
                trades = [t for t in trades if t.get('symbol') in symbols]
            
            if strategies:
                trades = [t for t in trades if t.get('strategy') in strategies]
            
            if timeframes:
                trades = [t for t in trades if t.get('timeframe') in timeframes]
            
            # Build report
            report = self._build_report(trades, start_date, end_date)
            
            # Save report
            self._save_report(report, report_format)
            
            return report
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            handle_error(e, "Failed to generate performance report")
            return {"error": str(e)}
    
    def _build_report(self, trades, start_date, end_date):
        """
        Build a performance report from trade data.
        
        Args:
            trades (list): List of trade dictionaries
            start_date (datetime): Start date for report
            end_date (datetime): End date for report
            
        Returns:
            dict: Comprehensive performance report
        """
        # Initialize report structure
        report = {
            "summary": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_trades": len(trades),
                "winning_trades": 0,
                "losing_trades": 0,
                "breakeven_trades": 0,
                "win_rate": 0,
                "total_profit": 0,
                "total_loss": 0,
                "net_profit": 0,
                "profit_factor": 0,
                "average_win": 0,
                "average_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "average_trade": 0,
                "median_trade": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "max_drawdown_percentage": 0,
                "recovery_factor": 0,
                "expectancy": 0,
                "average_holding_time": 0,
            },
            "by_symbol": {},
            "by_strategy": {},
            "by_timeframe": {},
            "monthly_performance": {},
            "drawdowns": [],
            "equity_curve": []
        }
        
        if not trades:
            logger.warning("No trades found for performance report")
            return report
        
        # Calculate basic metrics
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        breakeven_trades = [t for t in trades if t.get('pnl', 0) == 0]
        
        report["summary"]["winning_trades"] = len(winning_trades)
        report["summary"]["losing_trades"] = len(losing_trades)
        report["summary"]["breakeven_trades"] = len(breakeven_trades)
        
        # Win rate
        if len(trades) > 0:
            report["summary"]["win_rate"] = len(winning_trades) / len(trades)
        
        # Profit and loss metrics
        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        
        report["summary"]["total_profit"] = total_profit
        report["summary"]["total_loss"] = total_loss
        report["summary"]["net_profit"] = total_profit - total_loss
        
        # Profit factor
        if total_loss > 0:
            report["summary"]["profit_factor"] = total_profit / total_loss
        elif total_profit > 0:
            report["summary"]["profit_factor"] = float('inf')
        
        # Average metrics
        if winning_trades:
            report["summary"]["average_win"] = total_profit / len(winning_trades)
            report["summary"]["largest_win"] = max(t.get('pnl', 0) for t in winning_trades)
        
        if losing_trades:
            report["summary"]["average_loss"] = total_loss / len(losing_trades)
            report["summary"]["largest_loss"] = min(t.get('pnl', 0) for t in losing_trades)
        
        if trades:
            report["summary"]["average_trade"] = sum(t.get('pnl', 0) for t in trades) / len(trades)
            
            # Get all PnLs and calculate median
            pnls = [t.get('pnl', 0) for t in trades]
            report["summary"]["median_trade"] = np.median(pnls)
        
        # Calculate holding time
        holding_times = []
        for trade in trades:
            if trade.get('entry_time') and trade.get('exit_time'):
                holding_time = (trade['exit_time'] - trade['entry_time']) / (1000 * 60 * 60)  # hours
                holding_times.append(holding_time)
        
        if holding_times:
            report["summary"]["average_holding_time"] = sum(holding_times) / len(holding_times)
        
        # Calculate drawdowns and max drawdown
        equity_curve, drawdowns = self._calculate_equity_curve_and_drawdowns(trades)
        
        report["equity_curve"] = equity_curve
        report["drawdowns"] = drawdowns
        
        if drawdowns:
            max_dd = max(drawdowns, key=lambda x: x['drawdown_pct'])
            report["summary"]["max_drawdown"] = max_dd['drawdown']
            report["summary"]["max_drawdown_percentage"] = max_dd['drawdown_pct']
            
            # Recovery factor
            if report["summary"]["max_drawdown"] > 0 and report["summary"]["net_profit"] > 0:
                report["summary"]["recovery_factor"] = report["summary"]["net_profit"] / report["summary"]["max_drawdown"]
        
        # Calculate expectancy
        if report["summary"]["win_rate"] > 0:
            report["summary"]["expectancy"] = (report["summary"]["win_rate"] * report["summary"]["average_win"]) - \
                                            ((1 - report["summary"]["win_rate"]) * abs(report["summary"]["average_loss"]))
        
        # Calculate Sharpe and Sortino ratios
        returns = self._calculate_daily_returns(trades)
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Annualized Sharpe ratio (assuming daily returns)
            risk_free_rate = 0.02 / 252  # 2% annual risk-free rate, daily
            if std_return > 0:
                report["summary"]["sharpe_ratio"] = (avg_return - risk_free_rate) / std_return * np.sqrt(252)
            
            # Sortino ratio (only consider negative returns for risk)
            negative_returns = [r for r in returns if r < 0]
            std_negative = np.std(negative_returns) if negative_returns else 0
            
            if std_negative > 0:
                report["summary"]["sortino_ratio"] = (avg_return - risk_free_rate) / std_negative * np.sqrt(252)
        
        # Calculate performance by symbol
        symbols = set(t.get('symbol') for t in trades)
        for symbol in symbols:
            symbol_trades = [t for t in trades if t.get('symbol') == symbol]
            symbol_wins = [t for t in symbol_trades if t.get('pnl', 0) > 0]
            
            symbol_report = {
                "total_trades": len(symbol_trades),
                "winning_trades": len(symbol_wins),
                "win_rate": len(symbol_wins) / len(symbol_trades) if symbol_trades else 0,
                "net_profit": sum(t.get('pnl', 0) for t in symbol_trades),
                "average_trade": sum(t.get('pnl', 0) for t in symbol_trades) / len(symbol_trades) if symbol_trades else 0,
            }
            
            report["by_symbol"][symbol] = symbol_report
        
        # Calculate performance by strategy
        strategies = set(t.get('strategy') for t in trades)
        for strategy in strategies:
            strategy_trades = [t for t in trades if t.get('strategy') == strategy]
            strategy_wins = [t for t in strategy_trades if t.get('pnl', 0) > 0]
            
            strategy_report = {
                "total_trades": len(strategy_trades),
                "winning_trades": len(strategy_wins),
                "win_rate": len(strategy_wins) / len(strategy_trades) if strategy_trades else 0,
                "net_profit": sum(t.get('pnl', 0) for t in strategy_trades),
                "average_trade": sum(t.get('pnl', 0) for t in strategy_trades) / len(strategy_trades) if strategy_trades else 0,
            }
            
            report["by_strategy"][strategy] = strategy_report
        
        # Calculate performance by timeframe
        timeframes = set(t.get('timeframe') for t in trades)
        for timeframe in timeframes:
            timeframe_trades = [t for t in trades if t.get('timeframe') == timeframe]
            timeframe_wins = [t for t in timeframe_trades if t.get('pnl', 0) > 0]
            
            timeframe_report = {
                "total_trades": len(timeframe_trades),
                "winning_trades": len(timeframe_wins),
                "win_rate": len(timeframe_wins) / len(timeframe_trades) if timeframe_trades else 0,
                "net_profit": sum(t.get('pnl', 0) for t in timeframe_trades),
                "average_trade": sum(t.get('pnl', 0) for t in timeframe_trades) / len(timeframe_trades) if timeframe_trades else 0,
            }
            
            report["by_timeframe"][timeframe] = timeframe_report
        
        # Calculate monthly performance
        for trade in trades:
            if trade.get('entry_time'):
                entry_date = datetime.fromtimestamp(trade['entry_time'] / 1000)
                month_key = entry_date.strftime('%Y-%m')
                
                if month_key not in report["monthly_performance"]:
                    report["monthly_performance"][month_key] = {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "net_profit": 0,
                        "win_rate": 0
                    }
                
                report["monthly_performance"][month_key]["total_trades"] += 1
                report["monthly_performance"][month_key]["net_profit"] += trade.get('pnl', 0)
                
                if trade.get('pnl', 0) > 0:
                    report["monthly_performance"][month_key]["winning_trades"] += 1
        
        # Calculate win rate for each month
        for month, data in report["monthly_performance"].items():
            if data["total_trades"] > 0:
                data["win_rate"] = data["winning_trades"] / data["total_trades"]
        
        return report
    
    def _calculate_equity_curve_and_drawdowns(self, trades):
        """
        Calculate equity curve and drawdowns from trade data.
        
        Args:
            trades (list): List of trade dictionaries
            
        Returns:
            tuple: (equity_curve, drawdowns)
        """
        if not trades:
            return [], []
        
        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda x: x.get('exit_time', 0))
        
        # Initialize equity curve
        equity_curve = []
        initial_balance = 10000  # Assume starting capital
        balance = initial_balance
        peak_balance = initial_balance
        drawdowns = []
        current_drawdown_start = None
        
        # Process each trade
        for trade in sorted_trades:
            # Update balance
            pnl = trade.get('pnl', 0)
            balance += pnl
            timestamp = trade.get('exit_time', 0)
            
            # Add point to equity curve
            equity_point = {
                "timestamp": timestamp,
                "balance": balance,
                "trade_id": trade.get('id'),
                "pnl": pnl
            }
            equity_curve.append(equity_point)
            
            # Update peak balance and check for drawdowns
            if balance > peak_balance:
                peak_balance = balance
                current_drawdown_start = None
            elif balance < peak_balance:
                # Calculate drawdown
                drawdown = peak_balance - balance
                drawdown_pct = drawdown / peak_balance
                
                # Check if this is a new drawdown
                if current_drawdown_start is None:
                    current_drawdown_start = timestamp
                
                # Record significant drawdowns (> 1%)
                if drawdown_pct > 0.01:
                    drawdown_info = {
                        "start_time": current_drawdown_start,
                        "current_time": timestamp,
                        "peak_balance": peak_balance,
                        "current_balance": balance,
                        "drawdown": drawdown,
                        "drawdown_pct": drawdown_pct,
                        "duration": (timestamp - current_drawdown_start) / (1000 * 60 * 60 * 24)  # days
                    }
                    drawdowns.append(drawdown_info)
        
        return equity_curve, drawdowns
    
    def _calculate_daily_returns(self, trades):
        """
        Calculate daily returns from trade data.
        
        Args:
            trades (list): List of trade dictionaries
            
        Returns:
            list: Daily returns
        """
        if not trades:
            return []
        
        # Group trades by day
        daily_pnl = {}
        
        for trade in trades:
            if trade.get('exit_time'):
                exit_date = datetime.fromtimestamp(trade['exit_time'] / 1000).date()
                day_key = exit_date.isoformat()
                
                if day_key not in daily_pnl:
                    daily_pnl[day_key] = 0
                
                daily_pnl[day_key] += trade.get('pnl', 0)
        
        # Calculate daily returns (simplified, assuming constant initial balance)
        initial_balance = 10000  # Assume starting capital
        daily_returns = []
        
        for day, pnl in sorted(daily_pnl.items()):
            daily_return = pnl / initial_balance
            daily_returns.append(daily_return)
        
        return daily_returns
    
    def _save_report(self, report, report_format='json'):
        """
        Save the performance report to disk.
        
        Args:
            report (dict): Report data
            report_format (str): Report format ('json', 'csv', 'html')
            
        Returns:
            str: Path to saved report
        """
        reports_dir = Path(PATHS['reports'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if report_format == 'json':
            report_path = reports_dir / f"performance_report_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4, default=str)
            
            logger.info(f"Performance report saved to {report_path}")
            return str(report_path)
        
        elif report_format == 'csv':
            # Save summary as CSV
            summary_path = reports_dir / f"performance_summary_{timestamp}.csv"
            summary_df = pd.DataFrame([report['summary']])
            summary_df.to_csv(summary_path, index=False)
            
            # Save equity curve as CSV
            if report['equity_curve']:
                equity_path = reports_dir / f"equity_curve_{timestamp}.csv"
                equity_df = pd.DataFrame(report['equity_curve'])
                equity_df.to_csv(equity_path, index=False)
            
            logger.info(f"Performance report saved to {summary_path}")
            return str(summary_path)
        
        elif report_format == 'html':
            # Generate HTML report
            report_path = reports_dir / f"performance_report_{timestamp}.html"
            
            # This is a simplified HTML report template
            html_content = f"""
            <html>
            <head>
                <title>Trading Performance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Trading Performance Report</h1>
                <h2>Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Start Date</td><td>{report['summary']['start_date']}</td></tr>
                    <tr><td>End Date</td><td>{report['summary']['end_date']}</td></tr>
                    <tr><td>Total Trades</td><td>{report['summary']['total_trades']}</td></tr>
                    <tr><td>Win Rate</td><td>{report['summary']['win_rate']:.2%}</td></tr>
                    <tr><td>Net Profit</td><td class="{'positive' if report['summary']['net_profit'] >= 0 else 'negative'}">{report['summary']['net_profit']:.2f}</td></tr>
                    <tr><td>Profit Factor</td><td>{report['summary']['profit_factor']:.2f}</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{report['summary']['sharpe_ratio']:.2f}</td></tr>
                    <tr><td>Max Drawdown</td><td class="negative">{report['summary']['max_drawdown_percentage']:.2%}</td></tr>
                </table>
                
                <h2>Performance by Symbol</h2>
                <table>
                    <tr><th>Symbol</th><th>Trades</th><th>Win Rate</th><th>Net Profit</th></tr>
            """
            
            # Add symbol performance
            for symbol, data in report['by_symbol'].items():
                html_content += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td>{data['total_trades']}</td>
                        <td>{data['win_rate']:.2%}</td>
                        <td class="{'positive' if data['net_profit'] >= 0 else 'negative'}">{data['net_profit']:.2f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h2>Performance by Strategy</h2>
                <table>
                    <tr><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>Net Profit</th></tr>
            """
            
            # Add strategy performance
            for strategy, data in report['by_strategy'].items():
                html_content += f"""
                    <tr>
                        <td>{strategy}</td>
                        <td>{data['total_trades']}</td>
                        <td>{data['win_rate']:.2%}</td>
                        <td class="{'positive' if data['net_profit'] >= 0 else 'negative'}">{data['net_profit']:.2f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h2>Monthly Performance</h2>
                <table>
                    <tr><th>Month</th><th>Trades</th><th>Win Rate</th><th>Net Profit</th></tr>
            """
            
            # Add monthly performance
            for month, data in sorted(report['monthly_performance'].items()):
                html_content += f"""
                    <tr>
                        <td>{month}</td>
                        <td>{data['total_trades']}</td>
                        <td>{data['win_rate']:.2%}</td>
                        <td class="{'positive' if data['net_profit'] >= 0 else 'negative'}">{data['net_profit']:.2f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"HTML performance report saved to {report_path}")
            return str(report_path)
        
        else:
            logger.warning(f"Unsupported report format: {report_format}")
            return None
    
    def generate_strategy_comparison(self, strategies=None, time_period='1m'):
        """
        Generate a comparison report between different strategies.
        
        Args:
            strategies (list): List of strategies to compare
            time_period (str): Time period for comparison ('1m', '3m', '6m', '1y', 'all')
            
        Returns:
            dict: Strategy comparison report
        """
        try:
            # Calculate date range based on time period
            end_date = datetime.now()
            
            if time_period == '1m':
                start_date = end_date - timedelta(days=30)
            elif time_period == '3m':
                start_date = end_date - timedelta(days=90)
            elif time_period == '6m':
                start_date = end_date - timedelta(days=180)
            elif time_period == '1y':
                start_date = end_date - timedelta(days=365)
            else:  # 'all'
                start_date = datetime(2000, 1, 1)  # Far in the past
            
            start_time = int(start_date.timestamp() * 1000)
            end_time = int(end_date.timestamp() * 1000)
            
            # Get trade data
            trades = []
            if self.datastore:
                trades = self.datastore.get_trades(
                    start_time=start_time,
                    end_time=end_time,
                    status="CLOSED"
                )
            
            if not trades:
                logger.warning("No trades found for strategy comparison")
                return {"strategies": [], "time_period": time_period}
            
            # Filter by strategies if specified
            if strategies:
                trades = [t for t in trades if t.get('strategy') in strategies]
                strategy_list = strategies
            else:
                # Get all unique strategies
                strategy_list = list(set(t.get('strategy') for t in trades if t.get('strategy')))
            
            # Initialize comparison report
            comparison = {
                "time_period": time_period,
                "strategies": [],
                "best_overall": None,
                "best_win_rate": None,
                "best_profit_factor": None,
                "lowest_drawdown": None
            }
            
            # Calculate metrics for each strategy
            best_profit = -float('inf')
            best_win_rate = -float('inf')
            best_profit_factor = -float('inf')
            lowest_drawdown = float('inf')
            
            for strategy in strategy_list:
                strategy_trades = [t for t in trades if t.get('strategy') == strategy]
                
                if not strategy_trades:
                    continue
                
                # Calculate basic metrics
                total_trades = len(strategy_trades)
                winning_trades = len([t for t in strategy_trades if t.get('pnl', 0) > 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                total_profit = sum(max(0, t.get('pnl', 0)) for t in strategy_trades)
                total_loss = abs(sum(min(0, t.get('pnl', 0)) for t in strategy_trades))
                net_profit = total_profit - total_loss
                
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
                
                # Calculate drawdown
                equity_curve, drawdowns = self._calculate_equity_curve_and_drawdowns(strategy_trades)
                max_drawdown_pct = max([d['drawdown_pct'] for d in drawdowns]) if drawdowns else 0
                
                # Create strategy metrics
                strategy_metrics = {
                    "name": strategy,
                    "total_trades": total_trades,
                    "win_rate": win_rate,
                    "net_profit": net_profit,
                    "profit_factor": profit_factor,
                    "max_drawdown_pct": max_drawdown_pct,
                    "average_trade": net_profit / total_trades if total_trades > 0 else 0,
                    "trades_per_day": total_trades / ((end_time - start_time) / (1000 * 60 * 60 * 24))
                }
                
                comparison["strategies"].append(strategy_metrics)
                
                # Update best performers
                if net_profit > best_profit:
                    best_profit = net_profit
                    comparison["best_overall"] = strategy
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    comparison["best_win_rate"] = strategy
                
                if profit_factor > best_profit_factor:
                    best_profit_factor = profit_factor
                    comparison["best_profit_factor"] = strategy
                
                if max_drawdown_pct < lowest_drawdown and max_drawdown_pct > 0:
                    lowest_drawdown = max_drawdown_pct
                    comparison["lowest_drawdown"] = strategy
            
            # Sort strategies by net profit (descending)
            comparison["strategies"].sort(key=lambda x: x["net_profit"], reverse=True)
            
            return comparison
        except Exception as e:
            logger.error(f"Error generating strategy comparison: {e}")
            handle_error(e, "Failed to generate strategy comparison")
            return {"error": str(e)}
    
    def generate_optimization_report(self, optimization_results):
        """
        Generate a report for strategy optimization results.
        
        Args:
            optimization_results (dict): Strategy optimization results
            
        Returns:
            dict: Optimization report
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "strategies": [],
                "best_parameters": {},
                "parameter_sensitivity": {}
            }
            
            if not optimization_results:
                logger.warning("No optimization results provided")
                return report
            
            # Process each strategy's optimization results
            for strategy_id, result in optimization_results.items():
                if not result or 'best_params' not in result:
                    continue
                
                strategy_report = {
                    "strategy_id": strategy_id,
                    "best_params": result['best_params'],
                    "best_score": result.get('best_score', 0),
                    "metric": result.get('metric', '')
                }
                
                report["strategies"].append(strategy_report)
                report["best_parameters"][strategy_id] = result['best_params']
            
            # Analyze parameter sensitivity if available
            # This would require more detailed optimization data
            
            # Save report to disk
            reports_dir = Path(PATHS['reports'])
            report_path = reports_dir / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4, default=str)

        except Exception as e:
            logger.error(f"Error generating optimization report: {e}")
            handle_error(e, "Failed to generate optimization report")
            return {"error": str(e)}