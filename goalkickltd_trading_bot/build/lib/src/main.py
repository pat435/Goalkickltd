"""
Main entry point for the Goalkick Ltd Trading Bot.
Orchestrates the entire trading system.
"""

import os
import sys
import signal
import time
import threading
import uuid
import argparse
from datetime import datetime
import schedule

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging, get_logger
from config.bot_config import BotMode, BotState, TRADING_CONFIG, SCHEDULE_CONFIG
from config.trading_pairs import get_all_active_symbols
from data.datastore import DataStore
from data.data_fetcher import DataFetcher
from src.exchange.bybit_api import BybitAPI
from src.exchange.account_manager import AccountManager
from src.exchange.order_manager import OrderManager
from src.signals.signal_generator import SignalGenerator
from src.signals.signal_filter import SignalFilter
from src.execution.order_execution import OrderExecutor
from src.risk.portfolio import PortfolioManager
from src.utils.error_handling import handle_error, setup_exception_handling
from src.utils.time_utils import get_current_time_ms
from src.utils.logging_utils import log_memory_usage

# Set up logging
logger = setup_logging()

class TradingBot:
    """Main trading bot class that orchestrates all components."""
    
    def __init__(self, mode=None, symbols=None, timeframes=None):
        """
        Initialize the trading bot.
        
        Args:
            mode (BotMode): Bot operation mode
            symbols (list): List of symbols to trade
            timeframes (list): List of timeframes to use
        """
        self.bot_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.mode = mode or TRADING_CONFIG["mode"]
        self.state = BotState.STARTING
        self.symbols = symbols or get_all_active_symbols()
        self.timeframes = timeframes or ["5m", "15m", "1h", "4h", "1d"]
        
        logger.info(f"Initializing Goalkick Trading Bot (ID: {self.bot_id}) in {self.mode.name} mode")
        
        # Initialize components
        self.exchange_api = BybitAPI()
        self.datastore = DataStore()
        self.data_fetcher = DataFetcher(self.exchange_api, self.datastore)
        self.account_manager = AccountManager(self.exchange_api)
        self.order_manager = OrderManager(self.exchange_api, self.account_manager)
        self.portfolio_manager = PortfolioManager(self.account_manager, self.order_manager, self.datastore)
        self.signal_generator = SignalGenerator(self.data_fetcher, self.datastore)
        self.signal_filter = SignalFilter(self.portfolio_manager, self.datastore)
        self.order_executor = OrderExecutor(self.order_manager, self.portfolio_manager)
        
        # Scheduling
        self.scheduler = schedule
        self.running = False
        self.data_update_task = None
        self.signal_check_task = None
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        logger.info("Trading bot initialization complete")
    
    def start(self):
        """Start the trading bot and its components."""
        try:
            logger.info("Starting trading bot...")
            self.state = BotState.RUNNING
            self.running = True
            
            # Connect to exchange and validate API credentials
            if not self.exchange_api.ping():
                logger.error("Cannot connect to exchange. Check API credentials and network connection.")
                self.state = BotState.ERROR
                return False
            
            # Log account information
            if self.mode in [BotMode.LIVE, BotMode.PAPER]:
                account_info = self.account_manager.get_account_info()
                logger.info(f"Account balance: {account_info['balance']} USDT")
            
            # Schedule tasks
            self._schedule_tasks()
            
            # Start with initial data update
            logger.info("Performing initial market data update...")
            self.data_fetcher.update_market_data(self.symbols, self.timeframes)
            
            # Run the scheduler in the main thread
            logger.info("Bot started successfully. Running scheduler...")
            while self.running:
                self.scheduler.run_pending()
                time.sleep(1)
            
            return True
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            handle_error(e, "Failed to start trading bot")
            self.state = BotState.ERROR
            return False
    
    def _schedule_tasks(self):
        """Schedule regular tasks."""
        # Schedule market data updates
        self.scheduler.every(SCHEDULE_CONFIG["data_update_interval"]).seconds.do(
            self._run_data_update
        ).tag("data_update")
        
        # Schedule signal checks
        self.scheduler.every(SCHEDULE_CONFIG["signal_check_interval"]).seconds.do(
            self._run_signal_check
        ).tag("signal_check")
        
        # Schedule position updates
        self.scheduler.every(SCHEDULE_CONFIG["position_update_interval"]).seconds.do(
            self._run_position_update
        ).tag("position_update")
        
        # Schedule performance tracking
        self.scheduler.every(SCHEDULE_CONFIG["performance_update_interval"]).seconds.do(
            self._run_performance_update
        ).tag("performance")
        
        # Schedule health check
        self.scheduler.every(SCHEDULE_CONFIG["health_check_interval"]).seconds.do(
            self._run_health_check
        ).tag("health_check")
        
        # Schedule daily cleanup at midnight
        self.scheduler.every().day.at("00:05").do(
            self._run_daily_cleanup
        ).tag("maintenance")
        
        logger.info("All tasks scheduled successfully")
    
    def _run_data_update(self):
        """Run the market data update task."""
        try:
            logger.debug("Running scheduled market data update")
            self.data_fetcher.update_market_data(self.symbols, self.timeframes)
            return True
        except Exception as e:
            logger.error(f"Error in scheduled market data update: {e}")
            handle_error(e, "Scheduled market data update failed")
            return False
    
    def _run_signal_check(self):
        """Run the signal check and order execution task."""
        try:
            if self.state != BotState.RUNNING:
                logger.debug(f"Skipping signal check (bot state: {self.state.name})")
                return False
            
            logger.debug("Running scheduled signal check")
            
            # Generate signals for all symbols and timeframes
            signals = self.signal_generator.generate_signals(self.symbols, self.timeframes)
            
            if not signals:
                logger.debug("No signals generated")
                return True
            
            # Filter signals
            filtered_signals = self.signal_filter.filter_signals(signals)
            
            if not filtered_signals:
                logger.debug("No signals passed filtering")
                return True
            
            # Execute orders based on signals
            for signal in filtered_signals:
                try:
                    # Skip execution in backtest mode
                    if self.mode == BotMode.BACKTEST:
                        logger.debug(f"Skipping execution in backtest mode: {signal}")
                        continue
                    
                    # Execute the order
                    self.order_executor.execute_signal(signal)
                except Exception as e:
                    logger.error(f"Error executing signal {signal['id']}: {e}")
                    handle_error(e, f"Failed to execute signal {signal['id']}")
            
            return True
        except Exception as e:
            logger.error(f"Error in scheduled signal check: {e}")
            handle_error(e, "Scheduled signal check failed")
            return False
    
    def _run_position_update(self):
        """Run the position update task."""
        try:
            if self.state != BotState.RUNNING:
                logger.debug(f"Skipping position update (bot state: {self.state.name})")
                return False
            
            logger.debug("Running scheduled position update")
            
            # Skip in backtest mode
            if self.mode == BotMode.BACKTEST:
                return True
            
            # Update positions
            self.portfolio_manager.update_positions()
            
            # Check for stop losses and trailing stops
            self.portfolio_manager.check_stop_losses()
            
            return True
        except Exception as e:
            logger.error(f"Error in scheduled position update: {e}")
            handle_error(e, "Scheduled position update failed")
            return False
    
    def _run_performance_update(self):
        """Run the performance tracking update."""
        try:
            logger.debug("Running scheduled performance update")
            
            # Skip in backtest mode
            if self.mode == BotMode.BACKTEST:
                return True
            
            # Update performance metrics
            self.portfolio_manager.update_performance_metrics()
            
            return True
        except Exception as e:
            logger.error(f"Error in scheduled performance update: {e}")
            handle_error(e, "Scheduled performance update failed")
            return False
    
    def _run_health_check(self):
        """Run the system health check."""
        try:
            logger.debug("Running scheduled health check")
            
            # Check exchange connection
            if not self.exchange_api.ping():
                logger.warning("Exchange connection failed in health check")
                
                # Try to reconnect
                if self.exchange_api.reconnect():
                    logger.info("Successfully reconnected to exchange")
                else:
                    logger.error("Failed to reconnect to exchange")
                    if self.state == BotState.RUNNING:
                        self.pause("Exchange connection lost")
            
            # Log memory usage
            log_memory_usage()
            
            # Check database connection
            try:
                self.datastore.get_connection()
                logger.debug("Database connection check passed")
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                handle_error(e, "Database connection check failed")
            
            # Record uptime
            uptime = datetime.now() - self.start_time
            logger.info(f"Bot uptime: {uptime}")
            
            return True
        except Exception as e:
            logger.error(f"Error in scheduled health check: {e}")
            handle_error(e, "Scheduled health check failed")
            return False
    
    def _run_daily_cleanup(self):
        """Run daily cleanup tasks."""
        try:
            logger.info("Running scheduled daily cleanup")
            
            # Clean old data
            self.datastore.clean_old_data()
            
            # Reset daily statistics
            self.portfolio_manager.reset_daily_stats()
            
            return True
        except Exception as e:
            logger.error(f"Error in scheduled daily cleanup: {e}")
            handle_error(e, "Scheduled daily cleanup failed")
            return False
    
    def pause(self, reason="User requested"):
        """
        Pause the trading bot.
        
        Args:
            reason (str): Reason for pausing
        """
        if self.state == BotState.RUNNING:
            logger.info(f"Pausing trading bot: {reason}")
            self.state = BotState.PAUSED
            
            # Cancel all open orders in paper/live mode
            if self.mode in [BotMode.LIVE, BotMode.PAPER]:
                try:
                    self.order_manager.cancel_all_orders()
                    logger.info("Cancelled all open orders")
                except Exception as e:
                    logger.error(f"Error cancelling orders during pause: {e}")
            
            return True
        return False
    
    def resume(self):
        """Resume the trading bot from a paused state."""
        if self.state == BotState.PAUSED:
            logger.info("Resuming trading bot")
            self.state = BotState.RUNNING
            return True
        return False
    
    def stop(self):
        """Stop the trading bot gracefully."""
        logger.info("Stopping trading bot...")
        self.state = BotState.STOPPED
        self.running = False
        
        # Clear scheduled tasks
        self.scheduler.clear()
        
        # Cancel all orders in paper/live mode
        if self.mode in [BotMode.LIVE, BotMode.PAPER]:
            try:
                self.order_manager.cancel_all_orders()
                logger.info("Cancelled all open orders")
            except Exception as e:
                logger.error(f"Error cancelling orders during shutdown: {e}")
        
        # Close database connection
        try:
            self.datastore.close()
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
        
        logger.info("Trading bot stopped")
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received shutdown signal {signum}")
        self.stop()
    
    def get_status(self):
        """Get the current status of the trading bot."""
        account_info = self.account_manager.get_account_info() if self.mode != BotMode.BACKTEST else {}
        
        return {
            "id": self.bot_id,
            "state": self.state.name,
            "mode": self.mode.name,
            "uptime": str(datetime.now() - self.start_time),
            "start_time": self.start_time.isoformat(),
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "account_balance": account_info.get("balance"),
            "open_positions": self.portfolio_manager.get_open_position_count(),
            "daily_pnl": self.portfolio_manager.get_daily_pnl(),
            "total_pnl": self.portfolio_manager.get_total_pnl(),
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Goalkick Ltd Trading Bot")
    
    parser.add_argument(
        "--mode", 
        choices=["live", "paper", "backtest", "optimize"],
        default="paper",
        help="Bot operation mode"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Trading symbols (e.g., BTCUSDT ETHUSDT)"
    )
    
    parser.add_argument(
        "--timeframes",
        nargs="+",
        help="Timeframes to use (e.g., 5m 1h 4h)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up exception handling
    setup_exception_handling()
    
    # Determine mode
    if args.mode == "live":
        mode = BotMode.LIVE
    elif args.mode == "paper":
        mode = BotMode.PAPER
    elif args.mode == "backtest":
        mode = BotMode.BACKTEST
    elif args.mode == "optimize":
        mode = BotMode.OPTIMIZE
    else:
        mode = TRADING_CONFIG["mode"]
    
    # Update log level if debug mode is enabled
    if args.debug:
        import logging
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create and start the bot
    bot = TradingBot(
        mode=mode,
        symbols=args.symbols,
        timeframes=args.timeframes
    )
    
    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        bot.stop()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        handle_error(e, "Unhandled exception in main")
        bot.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()