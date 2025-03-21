"""
Strategy parameters configuration for the Goalkick Ltd Trading Bot.
This module defines the parameters for different trading strategies.
"""

from enum import Enum, auto

class StrategyType(Enum):
    """Enum for different types of trading strategies."""
    TREND_FOLLOWING = auto()
    MEAN_REVERSION = auto()
    BREAKOUT = auto()
    ARBITRAGE = auto()
    STATISTICAL = auto()
    MULTI_TIMEFRAME = auto()

class TimeFrame(Enum):
    """Enum for different timeframes used in trading."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

# Default parameters for trend following strategies
TREND_FOLLOWING_PARAMS = {
    "moving_average": {
        "short_window": 9,
        "long_window": 21,
        "signal_window": 9,
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.3,  # Base position size as percentage of available capital
        "max_open_trades": 3,
    },
    "macd": {
        "fast_length": 12,
        "slow_length": 26,
        "signal_length": 9,
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.3,
        "max_open_trades": 3,
    },
    "parabolic_sar": {
        "step": 0.02,
        "max_step": 0.2,
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.25,
        "max_open_trades": 2,
    },
    "adx": {
        "length": 14,
        "threshold": 25,  # ADX value above which trend is considered strong
        "di_length": 14,
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.3,
        "max_open_trades": 3,
    },
    "ichimoku": {
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_span_b_period": 52,
        "displacement": 26,
        "timeframe": TimeFrame.HOUR_4.value,
        "base_qty_pct": 0.25,
        "max_open_trades": 2,
    }
}

# Default parameters for mean reversion strategies
MEAN_REVERSION_PARAMS = {
    "rsi": {
        "length": 14,
        "overbought": 70,
        "oversold": 30,
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.25,
        "max_open_trades": 2,
    },
    "bollinger_bands": {
        "length": 20,
        "std_dev": 2.0,
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.25,
        "max_open_trades": 2,
    },
    "stochastic": {
        "k_period": 14,
        "d_period": 3,
        "smooth_k": 3,
        "overbought": 80,
        "oversold": 20,
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.25,
        "max_open_trades": 2,
    },
    "cci": {
        "length": 20,
        "overbought": 100,
        "oversold": -100,
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.25,
        "max_open_trades": 2,
    },
    "williams_r": {
        "length": 14,
        "overbought": -20,
        "oversold": -80,
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.25,
        "max_open_trades": 2,
    }
}

# Default parameters for arbitrage strategies
ARBITRAGE_PARAMS = {
    "triangular": {
        "min_profit_pct": 0.5,  # Minimum profit percentage to execute trade
        "max_slippage_pct": 0.1,  # Maximum allowed slippage
        "check_interval_seconds": 5,  # How often to check for arbitrage opportunities
        "base_qty_pct": 0.2,
        "max_open_trades": 1,
    },
    "exchange": {
        "min_profit_pct": 0.7,
        "max_slippage_pct": 0.15,
        "check_interval_seconds": 10,
        "base_qty_pct": 0.2,
        "max_open_trades": 1,
    },
    "statistical": {
        "z_score_threshold": 2.0,  # Z-score threshold for pairs trading
        "correlation_threshold": 0.8,  # Minimum correlation between pairs
        "lookback_period": 100,  # Lookback period for statistical calculations
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.15,
        "max_open_trades": 2,
    }
}

# Default parameters for statistical strategies
STATISTICAL_PARAMS = {
    "linear_regression": {
        "lookback_period": 100,
        "prediction_periods": 20,
        "confidence_level": 0.95,
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.2,
        "max_open_trades": 2,
    },
    "machine_learning": {
        "model_type": "XGBoost",  # Options: RandomForest, XGBoost, LightGBM
        "training_window": 1000,  # Number of candles to use for training
        "features": [
            "open", "high", "low", "close", "volume",
            "rsi_14", "ma_9", "ma_21", "bbands_20_2", "atr_14"
        ],
        "target": "return_next_4h",  # Target variable to predict
        "train_test_split": 0.8,  # Proportion of data to use for training
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.15,
        "max_open_trades": 2,
    },
    "kalman_filter": {
        "process_variance": 1e-5,
        "measurement_variance": 0.1,
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.2,
        "max_open_trades": 2,
    }
}

# Combined strategy parameters for multi-strategy approaches
MULTI_STRATEGY_PARAMS = {
    "trend_rsi": {
        "trend_weight": 0.6,  # Weight for trend signals
        "rsi_weight": 0.4,  # Weight for RSI signals
        "trend_params": TREND_FOLLOWING_PARAMS["macd"],
        "rsi_params": MEAN_REVERSION_PARAMS["rsi"],
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.3,
        "max_open_trades": 3,
    },
    "volatility_trend": {
        "trend_weight": 0.5,
        "volatility_weight": 0.5,
        "trend_params": TREND_FOLLOWING_PARAMS["moving_average"],
        "volatility_params": MEAN_REVERSION_PARAMS["bollinger_bands"],
        "timeframe": TimeFrame.HOUR_1.value,
        "base_qty_pct": 0.25,
        "max_open_trades": 3,
    }
}

# Strategy configurations by symbol
SYMBOL_STRATEGY_MAP = {
    "BTCUSDT": {
        "primary": StrategyType.TREND_FOLLOWING.name,
        "secondary": StrategyType.MEAN_REVERSION.name,
        "params": {
            "trend_following": TREND_FOLLOWING_PARAMS["moving_average"],
            "mean_reversion": MEAN_REVERSION_PARAMS["bollinger_bands"],
        }
    },
    "ETHUSDT": {
        "primary": StrategyType.TREND_FOLLOWING.name,
        "secondary": StrategyType.MEAN_REVERSION.name,
        "params": {
            "trend_following": TREND_FOLLOWING_PARAMS["macd"],
            "mean_reversion": MEAN_REVERSION_PARAMS["rsi"],
        }
    },
    "SOLUSDT": {
        "primary": StrategyType.TREND_FOLLOWING.name,
        "secondary": StrategyType.STATISTICAL.name,
        "params": {
            "trend_following": TREND_FOLLOWING_PARAMS["ichimoku"],
            "statistical": STATISTICAL_PARAMS["linear_regression"],
        }
    },
    "BNBUSDT": {
        "primary": StrategyType.MEAN_REVERSION.name,
        "secondary": StrategyType.TREND_FOLLOWING.name,
        "params": {
            "mean_reversion": MEAN_REVERSION_PARAMS["stochastic"],
            "trend_following": TREND_FOLLOWING_PARAMS["adx"],
        }
    },
    "ADAUSDT": {
        "primary": StrategyType.MEAN_REVERSION.name,
        "secondary": StrategyType.STATISTICAL.name,
        "params": {
            "mean_reversion": MEAN_REVERSION_PARAMS["cci"],
            "statistical": STATISTICAL_PARAMS["machine_learning"],
        }
    }
}

def get_strategy_params(strategy_type, sub_strategy=None):
    """Get parameters for a specific strategy type and sub-strategy."""
    if strategy_type == StrategyType.TREND_FOLLOWING.name:
        return TREND_FOLLOWING_PARAMS.get(sub_strategy, TREND_FOLLOWING_PARAMS)
    elif strategy_type == StrategyType.MEAN_REVERSION.name:
        return MEAN_REVERSION_PARAMS.get(sub_strategy, MEAN_REVERSION_PARAMS)
    elif strategy_type == StrategyType.ARBITRAGE.name:
        return ARBITRAGE_PARAMS.get(sub_strategy, ARBITRAGE_PARAMS)
    elif strategy_type == StrategyType.STATISTICAL.name:
        return STATISTICAL_PARAMS.get(sub_strategy, STATISTICAL_PARAMS)
    elif strategy_type == StrategyType.MULTI_TIMEFRAME.name:
        return MULTI_STRATEGY_PARAMS.get(sub_strategy, MULTI_STRATEGY_PARAMS)
    return {}

def get_symbol_strategy(symbol):
    """Get strategy configuration for a specific symbol."""
    return SYMBOL_STRATEGY_MAP.get(symbol, SYMBOL_STRATEGY_MAP.get("BTCUSDT"))