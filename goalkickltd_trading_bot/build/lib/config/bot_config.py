"""
Main configuration for the Goalkick Ltd Trading Bot.
This module defines global settings for the bot's operation.
"""

import os
from enum import Enum, auto
from datetime import time, timezone
from dotenv import load_dotenv

load_dotenv()

class BotMode(Enum):
    """Enum for different bot operation modes."""
    LIVE = auto()
    PAPER = auto()
    BACKTEST = auto()
    OPTIMIZE = auto()
    DEMO = auto()

class BotState(Enum):
    """Enum for different bot states."""
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    ERROR = auto()

# Exchange configuration
EXCHANGE_CONFIG = {
    "name": "bybit",
    "testnet": os.getenv("BYBIT_TESTNET", "True").lower() in ("true", "1", "t"),
    "api_key": os.getenv("BYBIT_API_KEY", ""),
    "api_secret": os.getenv("BYBIT_API_SECRET", ""),
    "timeout": 10000,  # Milliseconds
    "retry_count": 3,
    "retry_delay": 1000,  # Milliseconds
    "use_ccxt": False,  # Whether to use CCXT library or direct API
}

# Trading parameters
TRADING_CONFIG = {
    "mode": BotMode.PAPER,  # Default to paper trading
    "risk_per_trade": float(os.getenv("RISK_PER_TRADE", "0.02")),  # 2% risk per trade
    "max_trades_per_day": int(os.getenv("MAX_TRADES_PER_DAY", "5")),
    "max_open_trades": 5,  # Maximum number of open positions at once
    "max_open_trades_per_pair": 1,  # Maximum positions per trading pair
    "compounding": os.getenv("COMPOUNDING", "True").lower() in ("true", "1", "t"),
    "start_capital": 10000,  # Starting capital for backtests
    "leverage": 1,  # Default leverage (1 = spot trading, >1 = margin/futures)
    "min_volume_btc": 10,  # Minimum 24h volume in BTC to consider trading a pair
}

# Risk management parameters
RISK_CONFIG = {
    "max_drawdown_pct": 10,  # Maximum acceptable drawdown (%)
    "max_daily_loss_pct": 5,  # Maximum acceptable daily loss (%)
    "trailing_stop_pct": 2,  # Default trailing stop percentage
    "stop_loss_pct": 5,  # Default stop loss percentage
    "risk_reward_ratio": 1.5,  # Minimum risk/reward ratio
    "use_dynamic_stops": True,  # Use volatility-based stop losses
    "volatility_lookback": 14,  # Lookback period for volatility calculation
    "volatility_multiplier": 1.5,  # Multiplier for volatility-based stops
}

# Performance tracking parameters
PERFORMANCE_CONFIG = {
    "track_drawdown": True,
    "track_win_rate": True,
    "track_profit_factor": True,
    "track_sharpe_ratio": True,
    "track_expectancy": True,
    "track_exposure": True,
}

# Market hours configuration (UTC)
MARKET_HOURS = {
    "Asian": {
        "start": time(hour=0, minute=0, tzinfo=timezone.utc),
        "end": time(hour=8, minute=0, tzinfo=timezone.utc),
    },
    "European": {
        "start": time(hour=8, minute=0, tzinfo=timezone.utc),
        "end": time(hour=16, minute=0, tzinfo=timezone.utc),
    },
    "US": {
        "start": time(hour=16, minute=0, tzinfo=timezone.utc),
        "end": time(hour=0, minute=0, tzinfo=timezone.utc),
    },
}

# Scheduling configuration
SCHEDULE_CONFIG = {
    "data_update_interval": 60,  # Seconds
    "signal_check_interval": 60,  # Seconds
    "position_update_interval": 60,  # Seconds
    "performance_update_interval": 300,  # 5 minutes
    "health_check_interval": 300,  # 5 minutes
}

# Notification settings
NOTIFICATION_CONFIG = {
    "enable_telegram": bool(os.getenv("TELEGRAM_BOT_TOKEN", "")),
    "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    "notify_on_trade": True,
    "notify_on_error": True,
    "notify_daily_summary": True,
    "quiet_hours_start": time(hour=22, minute=0),
    "quiet_hours_end": time(hour=7, minute=0),
    "NOTIFY_ON_ERROR": True,
    "TELEGRAM_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),  # Set this in your environment variables
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", ""),  # Set this in your environment variables
}


# Data storage configuration
DATA_CONFIG = {
    "db_path": os.getenv("DB_PATH", "./data/trading.db"),
    "use_sqlite": True,
    "use_csv": False,
    "historical_data_days": 90,  # Days of historical data to keep
    "clean_older_than_days": 90,  # Remove data older than this
}

# Feature flags for enabling/disabling parts of the system
FEATURE_FLAGS = {
    "enable_multi_timeframe": True,
    "enable_ml_models": True,
    "enable_arbitrage": False,
    "enable_telegram_bot": bool(os.getenv("TELEGRAM_BOT_TOKEN", "")),
    "enable_web_dashboard": False,
    "enable_performance_tracking": True,
    "enable_risk_management": True,
    "enable_dynamic_sizing": True,
    "debug_mode": os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "t"),
}

# API and Web Dashboard settings
API_CONFIG = {
    "enable_api": False,
    "api_host": "0.0.0.0",
    "api_port": 5000,
    "api_username": os.getenv("API_USERNAME", "admin"),
    "api_password": os.getenv("API_PASSWORD", ""),
    "jwt_secret": os.getenv("JWT_SECRET", "your-secret-key"),
    "jwt_expiration_days": 7,
}

# Default timeframes to use
DEFAULT_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# Recovery settings
RECOVERY_CONFIG = {
    "auto_restart": True,
    "max_retries": 3,
    "retry_delay": 60,  # Seconds
    "health_check_timeout": 30,  # Seconds
}

# System paths
PATHS = {
    "logs": "logs",
    "data": "data",
    "models": "data/models",
    "reports": "data/reports",
    "backtest_results": "data/backtest_results",
}

# Regional market weight settings - adjust based on specific assets
REGIONAL_MARKET_WEIGHTS = {
    "BTCUSDT": {"Asian": 1.0, "European": 1.0, "US": 1.0},
    "ETHUSDT": {"Asian": 1.0, "European": 1.0, "US": 1.0},
    "SOLUSDT": {"Asian": 0.8, "European": 1.0, "US": 1.2},
    "BNBUSDT": {"Asian": 1.2, "European": 1.0, "US": 0.9},
    "ADAUSDT": {"Asian": 0.9, "European": 1.0, "US": 1.1},
    "default": {"Asian": 1.0, "European": 1.0, "US": 1.0},
}

def get_bot_config():
    """Return a dictionary with all configuration settings."""
    return {
        "exchange": EXCHANGE_CONFIG,
        "trading": TRADING_CONFIG,
        "risk": RISK_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "market_hours": MARKET_HOURS,
        "schedule": SCHEDULE_CONFIG,
        "notifications": NOTIFICATION_CONFIG,
        "data": DATA_CONFIG,
        "features": FEATURE_FLAGS,
        "api": API_CONFIG,
        "timeframes": DEFAULT_TIMEFRAMES,
        "recovery": RECOVERY_CONFIG,
        "paths": PATHS,
        "regional_weights": REGIONAL_MARKET_WEIGHTS,
    }

def get_market_session(current_time=None):
    """
    Determine the current market session based on time.
    
    Args:
        current_time: Current datetime in UTC (if None, current time is used)
        
    Returns:
        str: Market session name ("Asian", "European", "US")
    """
    from datetime import datetime
    
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    
    current_time = current_time.time()
    
    if MARKET_HOURS["Asian"]["start"] <= current_time < MARKET_HOURS["Asian"]["end"]:
        return "Asian"
    elif MARKET_HOURS["European"]["start"] <= current_time < MARKET_HOURS["European"]["end"]:
        return "European"
    else:
        return "US"

def get_regional_weight(symbol, session=None):
    """
    Get the weight for a symbol based on the current market session.
    
    Args:
        symbol: Trading pair symbol
        session: Market session (if None, current session is determined)
        
    Returns:
        float: Weight for the symbol in the current session
    """
    if session is None:
        session = get_market_session()
    
    weights = REGIONAL_MARKET_WEIGHTS.get(symbol, REGIONAL_MARKET_WEIGHTS["default"])
    return weights.get(session, 1.0)