"""
Logging utilities for the Goalkick Ltd Trading Bot.
"""

import os
import sys
import logging
import time
import psutil
import platform
import json
from datetime import datetime, timedelta
import threading

from config.logging_config import get_logger
from config.bot_config import PATHS

logger = get_logger("utils.logging")

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Convert to MB for readability
    rss_mb = memory_info.rss / (1024 * 1024)
    vms_mb = memory_info.vms / (1024 * 1024)
    
    logger.info(f"Memory usage: RSS={rss_mb:.2f}MB, VMS={vms_mb:.2f}MB")
    
    # Log high memory usage as warning
    if rss_mb > 1000:  # Warning if using more than 1GB RAM
        logger.warning(f"High memory usage detected: RSS={rss_mb:.2f}MB")
    
    return {
        "rss_mb": rss_mb,
        "vms_mb": vms_mb,
        "percent": process.memory_percent()
    }

def log_system_info():
    """Log system information."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    system_memory = psutil.virtual_memory()
    disk_usage = psutil.disk_usage('/')
    
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU usage: {cpu_percent}%")
    logger.info(f"Memory: {system_memory.percent}% used ({system_memory.used / (1024 * 1024):.2f}MB / {system_memory.total / (1024 * 1024):.2f}MB)")
    logger.info(f"Disk: {disk_usage.percent}% used")
    
    return {
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
        "cpu_percent": cpu_percent,
        "memory_percent": system_memory.percent,
        "memory_used_mb": system_memory.used / (1024 * 1024),
        "memory_total_mb": system_memory.total / (1024 * 1024),
        "disk_percent": disk_usage.percent
    }

def create_performance_log(bot_instance):
    """
    Create a performance log entry.
    
    Args:
        bot_instance: The trading bot instance
        
    Returns:
        dict: Performance log data
    """
    status = bot_instance.get_status()
    
    performance_data = {
        "timestamp": datetime.now().isoformat(),
        "bot_id": status["id"],
        "uptime": status["uptime"],
        "state": status["state"],
        "mode": status["mode"],
        "account_balance": status["account_balance"],
        "open_positions": status["open_positions"],
        "daily_pnl": status["daily_pnl"],
        "total_pnl": status["total_pnl"],
        "memory_usage": log_memory_usage(),
        "system_info": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent
        }
    }
    
    # Save to log file
    performance_log_path = os.path.join(PATHS["logs"], "performance.log")
    
    with open(performance_log_path, "a") as f:
        f.write(json.dumps(performance_data) + "\n")
    
    return performance_data

def log_trade(trade_data, log_level=logging.INFO):
    """
    Log a trade with proper formatting.
    
    Args:
        trade_data (dict): Trade data
        log_level (int): Logging level
    """
    trade_logger = get_logger("trades")
    
    # Format trade information
    symbol = trade_data.get("symbol", "UNKNOWN")
    side = trade_data.get("side", "UNKNOWN")
    quantity = trade_data.get("quantity", 0)
    entry_price = trade_data.get("entry_price", 0)
    exit_price = trade_data.get("exit_price")
    pnl = trade_data.get("pnl")
    pnl_pct = trade_data.get("pnl_pct")
    
    if exit_price is not None and pnl is not None:
        # Closed trade
        trade_msg = (
            f"TRADE CLOSED - {symbol}: {side} {quantity} @ {entry_price} -> {exit_price} | "
            f"PNL: {pnl:.2f} ({pnl_pct:.2%})"
        )
    else:
        # New trade
        trade_msg = f"TRADE OPENED - {symbol}: {side} {quantity} @ {entry_price}"
    
    # Log the trade
    trade_logger.log(log_level, trade_msg)
    
    # Also save to trade log file
    trade_log_path = os.path.join(PATHS["logs"], "trades.log")
    
    timestamp = datetime.now().isoformat()
    trade_entry = f"{timestamp} - {trade_msg}\n"
    
    with open(trade_log_path, "a") as f:
        f.write(trade_entry)

class PerformanceMonitor:
    """Class for monitoring and logging system performance."""
    
    def __init__(self, interval=300):
        """
        Initialize the PerformanceMonitor.
        
        Args:
            interval (int): Monitoring interval in seconds
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.start_time = datetime.now()
        self.performance_history = []
        self.max_history_size = 1000  # Maximum number of history points to keep
    
    def start(self):
        """Start the performance monitoring."""
        if self.running:
            logger.warning("Performance monitor already running")
            return
        
        self.running = True
        self.start_time = datetime.now()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"Started performance monitoring with {self.interval}s interval")
    
    def stop(self):
        """Stop the performance monitoring."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        
        logger.info("Stopped performance monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect performance data
                performance_data = {
                    "timestamp": datetime.now().isoformat(),
                    "memory": log_memory_usage(),
                    "cpu": {
                        "percent": psutil.cpu_percent(interval=0.1),
                        "count": psutil.cpu_count(),
                        "freq": psutil.cpu_freq().current if psutil.cpu_freq() else None
                    },
                    "system_memory": {
                        "percent": psutil.virtual_memory().percent,
                        "available_mb": psutil.virtual_memory().available / (1024 * 1024)
                    },
                    "disk": {
                        "percent": psutil.disk_usage('/').percent,
                        "free_gb": psutil.disk_usage('/').free / (1024 * 1024 * 1024)
                    },
                    "uptime": str(datetime.now() - self.start_time)
                }
                
                # Add to history
                self.performance_history.append(performance_data)
                
                # Trim history if too large
                if len(self.performance_history) > self.max_history_size:
                    self.performance_history = self.performance_history[-self.max_history_size:]
                
                # Log if memory usage is high
                if performance_data["memory"]["percent"] > 80:
                    logger.warning(f"High memory usage: {performance_data['memory']['percent']}%")
                
                # Log if CPU usage is high
                if performance_data["cpu"]["percent"] > 80:
                    logger.warning(f"High CPU usage: {performance_data['cpu']['percent']}%")
                
                # Log if disk space is low
                if performance_data["disk"]["free_gb"] < 1:
                    logger.warning(f"Low disk space: {performance_data['disk']['free_gb']:.2f}GB free")
                
                # Save to performance log
                perf_log_path = os.path.join(PATHS["logs"], "system_performance.log")
                
                with open(perf_log_path, "a") as f:
                    f.write(json.dumps(performance_data) + "\n")
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
            
            # Sleep until next interval
            time.sleep(self.interval)
    
    def get_performance_stats(self, hours=24):
        """
        Get performance statistics for a time period.
        
        Args:
            hours (int): Number of hours to analyze
            
        Returns:
            dict: Performance statistics
        """
        try:
            # Calculate start time
            start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            # Filter history by time
            history = [h for h in self.performance_history if h["timestamp"] >= start_time]
            
            if not history:
                return {
                    "memory_avg": 0,
                    "memory_max": 0,
                    "cpu_avg": 0,
                    "cpu_max": 0,
                    "disk_free_min": 0
                }
            
            # Calculate stats
            memory_values = [h["memory"]["percent"] for h in history]
            cpu_values = [h["cpu"]["percent"] for h in history]
            disk_free_values = [h["disk"]["free_gb"] for h in history]
            
            stats = {
                "memory_avg": sum(memory_values) / len(memory_values),
                "memory_max": max(memory_values),
                "cpu_avg": sum(cpu_values) / len(cpu_values),
                "cpu_max": max(cpu_values),
                "disk_free_min": min(disk_free_values)
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return {
                "memory_avg": 0,
                "memory_max": 0,
                "cpu_avg": 0,
                "cpu_max": 0,
                "disk_free_min": 0
            }

class LogFilter(logging.Filter):
    """Custom log filter to add extra context to log records."""
    
    def __init__(self, bot_id=None):
        """
        Initialize the LogFilter.
        
        Args:
            bot_id (str): Trading bot ID
        """
        super().__init__()
        self.bot_id = bot_id
    
    def filter(self, record):
        """
        Filter log records and add extra context.
        
        Args:
            record: Log record
            
        Returns:
            bool: Always True to allow the record
        """
        # Add bot ID if available
        if self.bot_id:
            record.bot_id = self.bot_id
        
        # Add process information
        record.process_id = os.getpid()
        record.process_name = psutil.Process(os.getpid()).name()
        
        # Add thread information
        record.thread_name = threading.current_thread().name
        
        return True

def setup_file_logging(log_dir, bot_id=None):
    """
    Set up file logging with rotation.
    
    Args:
        log_dir (str): Log directory
        bot_id (str): Trading bot ID
        
    Returns:
        logging.Logger: Root logger
    """
    import logging.handlers
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Add log filter
    log_filter = LogFilter(bot_id)
    root_logger.addFilter(log_filter)
    
    # Create rotating file handler
    log_file = os.path.join(log_dir, "bot.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(bot_id)s] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(file_handler)
    
    # Create separate error log
    error_log_file = os.path.join(log_dir, "error.log")
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file, maxBytes=10*1024*1024, backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    
    # Add error handler to root logger
    root_logger.addHandler(error_handler)
    
    return root_logger

def log_api_request(exchange, endpoint, params, response=None, error=None):
    """
    Log an API request and response.
    
    Args:
        exchange (str): Exchange name
        endpoint (str): API endpoint
        params (dict): Request parameters
        response (dict): Response data (optional)
        error (Exception): Error that occurred (optional)
    """
    api_logger = get_logger("api")
    
    # Format request information
    req_msg = f"API REQUEST - {exchange} - {endpoint}"
    
    # Mask sensitive information
    masked_params = params.copy() if params else {}
    if "api_key" in masked_params:
        masked_params["api_key"] = "***"
    if "api_secret" in masked_params:
        masked_params["api_secret"] = "***"
    
    # Log request
    api_logger.debug(f"{req_msg} - Params: {masked_params}")
    
    # Log response or error
    if error:
        api_logger.error(f"API ERROR - {exchange} - {endpoint} - {str(error)}")
    elif response:
        # Truncate large responses for logging
        response_str = str(response)
        if len(response_str) > 500:
            response_str = response_str[:500] + "..."
        
        api_logger.debug(f"API RESPONSE - {exchange} - {endpoint} - {response_str}")

def rotate_logs(log_dir, max_age_days=30):
    """
    Rotate old log files.
    
    Args:
        log_dir (str): Log directory
        max_age_days (int): Maximum age of log files in days
        
    Returns:
        int: Number of files removed
    """
    try:
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0
        
        # Iterate over log files
        for filename in os.listdir(log_dir):
            if filename.endswith(".log") or filename.endswith(".log.gz"):
                file_path = os.path.join(log_dir, filename)
                
                # Get file modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Remove if older than cutoff
                if mod_time < cutoff_time:
                    os.remove(file_path)
                    removed_count += 1
        
        logger.info(f"Removed {removed_count} old log files")
        return removed_count
    except Exception as e:
        logger.error(f"Error rotating logs: {e}")
        return 0