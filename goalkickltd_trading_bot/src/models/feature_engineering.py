
"""
Feature engineering module for the Goalkick Ltd Trading Bot.
Transforms raw market data into features suitable for machine learning models.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from config.logging_config import get_logger
from src.indicators.momentum import relative_strength_index, moving_average_convergence_divergence
from src.indicators.volatility import bollinger_bands, average_true_range
from src.indicators.trend import simple_moving_average, exponential_moving_average
from src.indicators.volume import on_balance_volume, accumulation_distribution
from src.utils.error_handling import handle_error

logger = get_logger("models.feature_engineering")

class FeatureEngineer:
    """Class for transforming raw market data into features for machine learning models."""
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.feature_generators = {
            'technical_indicators': self._generate_technical_indicators,
            'price_patterns': self._generate_price_patterns,
            'volume_features': self._generate_volume_features,
            'time_features': self._generate_time_features,
            'statistical_features': self._generate_statistical_features,
            'trend_features': self._generate_trend_features
        }
    
    def generate_features(self, data, feature_sets=None):
        """
        Generate features from raw market data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            feature_sets (list): List of feature sets to generate (default: all)
            
        Returns:
            pd.DataFrame: DataFrame with generated features
        """
        try:
            if data.empty:
                logger.warning("Empty DataFrame provided for feature generation")
                return pd.DataFrame()
            
            # Make a copy to avoid modifying the original data
            df = data.copy()
            
            # Ensure data has the correct columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing columns in data: {missing_columns}")
                return pd.DataFrame()
            
            # Generate specified feature sets or all if not specified
            if feature_sets is None:
                feature_sets = list(self.feature_generators.keys())
            
            for feature_set in feature_sets:
                if feature_set in self.feature_generators:
                    logger.debug(f"Generating {feature_set} features")
                    df = self.feature_generators[feature_set](df)
                else:
                    logger.warning(f"Unknown feature set: {feature_set}")
            
            # Drop rows with NaN values
            df = df.dropna()
            
            return df
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            handle_error(e, "Failed to generate features")
            return pd.DataFrame()
    
    def _generate_technical_indicators(self, df):
        """
        Generate technical indicator features.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with technical indicator features
        """
        try:
            # RSI
            df['rsi_14'] = relative_strength_index(df, window=14)
            
            # MACD
            macd_line, signal_line, histogram = moving_average_convergence_divergence(df)
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_hist'] = histogram
            
            # Bollinger Bands
            middle_band, upper_band, lower_band = bollinger_bands(df)
            df['bb_middle'] = middle_band
            df['bb_upper'] = upper_band
            df['bb_lower'] = lower_band
            df['bb_width'] = (upper_band - lower_band) / middle_band
            
            # ATR
            df['atr_14'] = average_true_range(df, window=14)
            
            # Moving Averages
            df['sma_9'] = simple_moving_average(df, window=9)
            df['sma_21'] = simple_moving_average(df, window=21)
            df['sma_50'] = simple_moving_average(df, window=50)
            df['sma_200'] = simple_moving_average(df, window=200)
            
            df['ema_9'] = exponential_moving_average(df, window=9)
            df['ema_21'] = exponential_moving_average(df, window=21)
            
            # Moving Average Crossovers
            df['sma_cross_9_21'] = df['sma_9'] - df['sma_21']
            df['ema_cross_9_21'] = df['ema_9'] - df['ema_21']
            
            # Normalized indicators
            df['rsi_norm'] = (df['rsi_14'] - 50) / 25  # Normalize to [-2, 2]
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            return df
        except Exception as e:
            logger.error(f"Error generating technical indicators: {e}")
            handle_error(e, "Failed to generate technical indicators")
            return df
    
    def _generate_price_patterns(self, df):
        """
        Generate price pattern features.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with price pattern features
        """
        try:
            # Calculate price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_1d'] = df['close'].pct_change(periods=1)
            df['price_change_5d'] = df['close'].pct_change(periods=5)
            df['price_change_10d'] = df['close'].pct_change(periods=10)
            
            # Price volatility
            df['price_volatility_5d'] = df['price_change'].rolling(window=5).std()
            df['price_volatility_10d'] = df['price_change'].rolling(window=10).std()
            
            # High-Low range
            df['daily_range'] = (df['high'] - df['low']) / df['close']
            df['daily_range_ma_5'] = df['daily_range'].rolling(window=5).mean()
            
            # Candle patterns (basic)
            df['body_size'] = np.abs(df['close'] - df['open']) / df['close']
            df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
            df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
            
            # Candle color (1 for bullish, -1 for bearish)
            df['candle_color'] = np.where(df['close'] > df['open'], 1, -1)
            
            # Rolling candle patterns
            df['bullish_candles_3d'] = df['candle_color'].rolling(window=3).sum()
            df['bullish_candles_5d'] = df['candle_color'].rolling(window=5).sum()
            
            # Gap features
            df['gap_up'] = np.where(df['open'] > df['close'].shift(1), 
                                   (df['open'] - df['close'].shift(1)) / df['close'].shift(1), 0)
            df['gap_down'] = np.where(df['open'] < df['close'].shift(1), 
                                     (df['close'].shift(1) - df['open']) / df['close'].shift(1), 0)
            
            return df
        except Exception as e:
            logger.error(f"Error generating price patterns: {e}")
            handle_error(e, "Failed to generate price patterns")
            return df
    
    def _generate_volume_features(self, df):
        """
        Generate volume-based features.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with volume features
        """
        try:
            # Basic volume features
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            df['relative_volume'] = df['volume'] / df['volume_ma_10']
            
            # On-Balance Volume
            df['obv'] = on_balance_volume(df)
            df['obv_slope'] = df['obv'].diff(5) / 5
            
            # Accumulation/Distribution
            df['ad_line'] = accumulation_distribution(df)
            df['ad_slope'] = df['ad_line'].diff(5) / 5
            
            # Volume weighted by price change
            df['volume_price_trend'] = df['volume'] * df['price_change']
            df['vpt_ma_5'] = df['volume_price_trend'].rolling(window=5).mean()
            
            # Price-Volume relationship
            df['price_up_volume_up'] = np.where((df['candle_color'] > 0) & (df['volume_change'] > 0), 1, 0)
            df['price_down_volume_up'] = np.where((df['candle_color'] < 0) & (df['volume_change'] > 0), 1, 0)
            
            # Volume Oscillator (Percentage difference between fast and slow volume MAs)
            df['volume_osc'] = (df['volume_ma_5'] - df['volume_ma_10']) / df['volume_ma_10'] * 100
            
            return df
        except Exception as e:
            logger.error(f"Error generating volume features: {e}")
            handle_error(e, "Failed to generate volume features")
            return df
    
    def _generate_time_features(self, df):
        """
        Generate time-based features.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with time features
        """
        try:
            # Extract datetime components
            # Assume df.index is DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                df['hour'] = df.index.hour
                df['day_of_week'] = df.index.dayofweek
                df['day_of_month'] = df.index.day
                df['week_of_year'] = df.index.isocalendar().week
                df['month'] = df.index.month
                df['quarter'] = df.index.quarter
                df['year'] = df.index.year
                
                # Market session (simplified)
                df['session'] = df['hour'].apply(lambda x: 
                                              'Asian' if 0 <= x < 8 else 
                                              'European' if 8 <= x < 16 else 
                                              'US')
                
                # Market session encoding
                df['is_asian'] = np.where(df['session'] == 'Asian', 1, 0)
                df['is_european'] = np.where(df['session'] == 'European', 1, 0)
                df['is_us'] = np.where(df['session'] == 'US', 1, 0)
                
                # Weekend indicator
                df['is_weekend'] = np.where(df['day_of_week'] >= 5, 1, 0)
                
                # Time features based on sine/cosine transformation for cyclical features
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                
                df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
                
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            else:
                logger.warning("DataFrame index is not DatetimeIndex, skipping time features")
            
            return df
        except Exception as e:
            logger.error(f"Error generating time features: {e}")
            handle_error(e, "Failed to generate time features")
            return df
    
    def _generate_statistical_features(self, df):
        """
        Generate statistical features.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with statistical features
        """
        try:
            # Returns
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Rolling statistics
            for window in [5, 10, 20]:
                # Return statistics
                df[f'return_mean_{window}d'] = df['log_return'].rolling(window=window).mean()
                df[f'return_std_{window}d'] = df['log_return'].rolling(window=window).std()
                df[f'return_skew_{window}d'] = df['log_return'].rolling(window=window).skew()
                
                # Price statistics
                df[f'price_mean_{window}d'] = df['close'].rolling(window=window).mean()
                df[f'price_std_{window}d'] = df['close'].rolling(window=window).std()
                
                # Z-score
                df[f'price_zscore_{window}d'] = (df['close'] - df[f'price_mean_{window}d']) / df[f'price_std_{window}d']
                
                # Min/Max
                df[f'price_min_{window}d'] = df['low'].rolling(window=window).min()
                df[f'price_max_{window}d'] = df['high'].rolling(window=window).max()
                
                # Current price position within range
                price_range = df[f'price_max_{window}d'] - df[f'price_min_{window}d']
                df[f'price_position_{window}d'] = (df['close'] - df[f'price_min_{window}d']) / price_range
            
            # Serial correlation
            df['autocorr_1'] = df['log_return'].rolling(window=10).apply(
                lambda x: x.autocorr(lag=1), raw=False)
            
            # Quantile features
            df['return_quantile_10d'] = df['log_return'].rolling(window=10).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            
            return df
        except Exception as e:
            logger.error(f"Error generating statistical features: {e}")
            handle_error(e, "Failed to generate statistical features")
            return df
    
    def _generate_trend_features(self, df):
        """
        Generate trend-related features.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with trend features
        """
        try:
            # Price relative to moving averages
            df['price_to_sma_50'] = df['close'] / df['sma_50'] - 1
            df['price_to_sma_200'] = df['close'] / df['sma_200'] - 1
            
            # Golden/Death cross
            df['golden_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, 0)
            df['golden_cross_change'] = df['golden_cross'].diff()
            
            # Consecutive moves
            df['up_day'] = np.where(df['price_change'] > 0, 1, 0)
            df['consecutive_up_days'] = df['up_day'].groupby(
                (df['up_day'] != df['up_day'].shift()).cumsum()).cumcount() + 1
            df['consecutive_up_days'] = df['consecutive_up_days'] * df['up_day']
            
            df['down_day'] = np.where(df['price_change'] < 0, 1, 0)
            df['consecutive_down_days'] = df['down_day'].groupby(
                (df['down_day'] != df['down_day'].shift()).cumsum()).cumcount() + 1
            df['consecutive_down_days'] = df['consecutive_down_days'] * df['down_day']
            
            # Distance from 52-week high/low (approximated with 250 trading days)
            df['high_252d'] = df['high'].rolling(window=252).max()
            df['low_252d'] = df['low'].rolling(window=252).min()
            
            df['pct_from_high_252d'] = (df['close'] - df['high_252d']) / df['high_252d']
            df['pct_from_low_252d'] = (df['close'] - df['low_252d']) / df['low_252d']
            
            # Linear regression slope
            for window in [10, 20, 50]:
                df[f'slope_{window}d'] = self._calculate_slope(df['close'], window)
            
            # Directional Movement Index features
            df['dmi_plus'] = self._calculate_directional_index(df, positive=True)
            df['dmi_minus'] = self._calculate_directional_index(df, positive=False)
            df['dmi_diff'] = df['dmi_plus'] - df['dmi_minus']
            
            return df
        except Exception as e:
            logger.error(f"Error generating trend features: {e}")
            handle_error(e, "Failed to generate trend features")
            return df
    
    def _calculate_slope(self, series, window):
        """
        Calculate the slope of a linear regression line for a series.
        
        Args:
            series (pd.Series): Input series
            window (int): Window period
            
        Returns:
            pd.Series: Series of slope values
        """
        slopes = pd.Series(index=series.index, dtype=float)
        
        for i in range(window, len(series)):
            x = np.arange(window)
            y = series.iloc[i-window:i].values
            
            # Linear regression: y = ax + b
            a = np.polyfit(x, y, 1)[0]  # Get the slope coefficient
            slopes.iloc[i] = a
        
        return slopes
    
    def _calculate_directional_index(self, df, positive=True):
        """
        Calculate Directional Movement Index (DMI) components.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            positive (bool): If True, calculate +DI; otherwise -DI
            
        Returns:
            pd.Series: Series of DMI values
        """
        window = 14  # Default DMI period
        
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        if positive:
            dm = df['high'] - df['high'].shift(1)
            dm = np.where(dm > 0, dm, 0)
        else:
            dm = df['low'].shift(1) - df['low']
            dm = np.where(dm > 0, dm, 0)
        
        dm = pd.Series(dm, index=df.index)
        
        # Calculate smoothed values
        tr_smooth = tr.rolling(window=window).sum()
        dm_smooth = dm.rolling(window=window).sum()
        
        # Calculate DMI
        di = 100 * dm_smooth / tr_smooth
        
        # Handle NaN values
        di = di.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return di
    
    def normalize_features(self, df, method='z-score'):
        """
        Normalize features in the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            method (str): Normalization method ('z-score', 'min-max', or 'robust')
            
        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        try:
            # Create a copy of the DataFrame
            normalized_df = df.copy()
            
            # Get all numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            # Keep only desired columns for normalization (exclude OHLCV data)
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            cols_to_normalize = [col for col in numeric_cols if col not in ohlcv_cols]
            
            if method == 'z-score':
                # Z-score normalization
                for col in cols_to_normalize:
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        normalized_df[col] = (df[col] - mean) / std
                    else:
                        # Skip columns with zero standard deviation
                        logger.warning(f"Column {col} has zero standard deviation, skipping normalization")
            
            elif method == 'min-max':
                # Min-max normalization
                for col in cols_to_normalize:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:
                        normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        # Skip columns with no range
                        logger.warning(f"Column {col} has no range, skipping normalization")
            
            elif method == 'robust':
                # Robust scaling based on interquartile range
                for col in cols_to_normalize:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        normalized_df[col] = (df[col] - q1) / iqr
                    else:
                        # Skip columns with zero IQR
                        logger.warning(f"Column {col} has zero IQR, skipping normalization")
            
            else:
                logger.warning(f"Unknown normalization method: {method}")
            
            return normalized_df
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            handle_error(e, "Failed to normalize features")
            return df
    
    def select_features(self, df, features=None, method='correlation', target=None, threshold=0.1):
        """
        Select features based on a specified method.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            features (list): Specific features to select (if None, use method)
            method (str): Feature selection method ('correlation', 'importance', 'variance')
            target (str): Target column name for correlation-based selection
            threshold (float): Threshold for feature selection
            
        Returns:
            pd.DataFrame: DataFrame with selected features
        """
        try:
            if features is not None:
                # Select specified features
                valid_features = [f for f in features if f in df.columns]
                selected_df = df[valid_features].copy()
                logger.debug(f"Selected {len(valid_features)} specified features")
                return selected_df
            
            if method == 'correlation' and target is not None:
                if target not in df.columns:
                    logger.error(f"Target column {target} not found in DataFrame")
                    return df
                
                # Calculate correlation with target
                correlations = df.corr()[target].abs()
                
                # Select features with correlation above threshold
                selected_features = correlations[correlations > threshold].index.tolist()
                
                # Always include target
                if target not in selected_features:
                    selected_features.append(target)
                
                selected_df = df[selected_features].copy()
                logger.debug(f"Selected {len(selected_features)} features based on correlation")
                return selected_df
            
            elif method == 'variance':
                # Calculate variance for each column
                variances = df.var()
                
                # Select features with variance above threshold
                selected_features = variances[variances > threshold].index.tolist()
                
                selected_df = df[selected_features].copy()
                logger.debug(f"Selected {len(selected_features)} features based on variance")
                return selected_df
            
            elif method == 'importance':
                logger.warning("Importance-based feature selection requires a trained model")
                return df
            
            else:
                logger.warning(f"Unknown feature selection method: {method}")
                return df
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            handle_error(e, "Failed to select features")
            return df
    
    def prepare_training_data(self, df, target_col, forecast_horizon=1, split_ratio=0.8):
        """
        Prepare data for model training.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            target_col (str): Target column name
            forecast_horizon (int): Forecast horizon in periods
            split_ratio (float): Train/test split ratio
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            if target_col not in df.columns:
                # Create target column based on future price movement
                df[target_col] = df['close'].shift(-forecast_horizon) / df['close'] - 1
            
            # Drop NaN values
            df = df.dropna()
            
            # Get feature columns (exclude target)
            feature_cols = [col for col in df.columns if col != target_col]
            
            # Split data into features and target
            X = df[feature_cols]
            y = df[target_col]
            
            # Split into train and test sets
            split_idx = int(len(df) * split_ratio)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.debug(f"Prepared training data with {len(X_train)} training and {len(X_test)} testing samples")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            handle_error(e, "Failed to prepare training data")
            return None, None, None, None