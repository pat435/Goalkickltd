"""
Statistical and machine learning-based strategies for the Goalkick Ltd Trading Bot.
"""

import pandas as pd
import numpy as np

from config.logging_config import get_logger
from config.strategy_params import STATISTICAL_PARAMS
from src.strategies.base_strategy import Strategy
from src.models.feature_engineering import FeatureEngineer
from src.models.prediction import PredictionEngine
from src.utils.error_handling import handle_error

logger = get_logger("strategies.statistical")

class LinearRegressionStrategy(Strategy):
    """Linear regression-based trading strategy."""
    
    def __init__(self, timeframes=None, symbols=None, params=None):
        """
        Initialize the strategy.
        
        Args:
            timeframes (list): List of timeframes to use
            symbols (list): List of symbols to trade
            params (dict): Strategy parameters
        """
        default_params = STATISTICAL_PARAMS.get('linear_regression', {})
        params = {**default_params, **(params or {})}
        
        super().__init__('LinearRegression', timeframes, symbols, params)
        
        self.feature_engineer = FeatureEngineer()
    
    def generate_signals(self, data, symbol, timeframe):
        """
        Generate trading signals for a symbol and timeframe.
        
        Args:
            data (pd.DataFrame): Historical price data
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            
        Returns:
            list: List of signal dictionaries
        """
        try:
            if data.empty:
                logger.warning(f"Empty data for {symbol} {timeframe}")
                return []
            
            # Preprocess data
            df = self.preprocess_data(data)
            
            if df.empty:
                return []
            
            # Generate features
            feature_sets = ['technical_indicators', 'price_patterns', 'trend_features']
            features_df = self.feature_engineer.generate_features(df, feature_sets)
            
            if features_df.empty:
                logger.warning(f"Failed to generate features for {symbol} {timeframe}")
                return []
            
            # Get parameters
            lookback_period = self.params.get('lookback_period', 100)
            prediction_periods = self.params.get('prediction_periods', 20)
            confidence_level = self.params.get('confidence_level', 0.95)
            
            # Check if we have enough data
            if len(features_df) < lookback_period + prediction_periods:
                logger.warning(f"Not enough data for {symbol} {timeframe}")
                return []
            
            # Calculate linear regression on close prices
            X = np.arange(lookback_period).reshape(-1, 1)
            y = features_df['close'].iloc[-lookback_period:].values
            
            # Fit linear regression model
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            # Make prediction for next period
            next_period = np.array([[lookback_period]])
            predicted_price = model.predict(next_period)[0]
            
            # Calculate confidence interval
            from scipy import stats
            prediction_stddev = np.sqrt(np.sum((model.predict(X) - y) ** 2) / (lookback_period - 2))
            margin_error = stats.t.ppf((1 + confidence_level) / 2, lookback_period - 2) * prediction_stddev * np.sqrt(1 + 1/lookback_period)
            
            lower_bound = predicted_price - margin_error
            upper_bound = predicted_price + margin_error
            
            # Generate signals
            signals = []
            
            # Current price
            current_price = features_df['close'].iloc[-1]
            
            # Buy signal: Predicted price is significantly higher
            if predicted_price > current_price and lower_bound > current_price:
                # Determine signal strength based on predicted return
                predicted_return = (predicted_price - current_price) / current_price
                strength = min(1.0, predicted_return * 20)
                
                # Calculate stop loss and take profit
                stop_loss = current_price - (predicted_price - current_price) * 0.5
                take_profit = predicted_price
                
                # Create signal
                signal = self.create_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction="BUY",
                    strength=strength,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'predicted_price': predicted_price,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'predicted_return': predicted_return,
                        'confidence_level': confidence_level,
                        'lookback_period': lookback_period
                    }
                )
                
                signals.append(signal)
                logger.info(f"BUY signal for {symbol} at {current_price} (predicted: {predicted_price:.2f})")
            
            # Sell signal: Predicted price is significantly lower
            elif predicted_price < current_price and upper_bound < current_price:
                # Determine signal strength based on predicted return
                predicted_return = (current_price - predicted_price) / current_price
                strength = min(1.0, predicted_return * 20)
                
                # Calculate stop loss and take profit
                stop_loss = current_price + (current_price - predicted_price) * 0.5
                take_profit = predicted_price
                
                # Create signal
                signal = self.create_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction="SELL",
                    strength=strength,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'predicted_price': predicted_price,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'predicted_return': predicted_return,
                        'confidence_level': confidence_level,
                        'lookback_period': lookback_period
                    }
                )
                
                signals.append(signal)
                logger.info(f"SELL signal for {symbol} at {current_price} (predicted: {predicted_price:.2f})")
            
            return signals
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {e}")
            handle_error(e, f"Failed to generate signals for {symbol} {timeframe}")
            return []


class MachineLearningStrategy(Strategy):
    """Machine learning-based trading strategy."""
    
    def __init__(self, timeframes=None, symbols=None, params=None):
        """
        Initialize the strategy.
        
        Args:
            timeframes (list): List of timeframes to use
            symbols (list): List of symbols to trade
            params (dict): Strategy parameters
        """
        default_params = STATISTICAL_PARAMS.get('machine_learning', {})
        params = {**default_params, **(params or {})}
        
        super().__init__('MachineLearning', timeframes, symbols, params)
        
        self.feature_engineer = FeatureEngineer()
        self.prediction_engine = PredictionEngine()
        self.model_loaded = False
    
    def generate_signals(self, data, symbol, timeframe):
        """
        Generate trading signals for a symbol and timeframe.
        
        Args:
            data (pd.DataFrame): Historical price data
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            
        Returns:
            list: List of signal dictionaries
        """
        try:
            if data.empty:
                logger.warning(f"Empty data for {symbol} {timeframe}")
                return []
            
            # Preprocess data
            df = self.preprocess_data(data)
            
            if df.empty:
                return []
            
            # Generate features
            feature_sets = ['technical_indicators', 'price_patterns', 'volume_features', 'trend_features', 'statistical_features']
            features_df = self.feature_engineer.generate_features(df, feature_sets)
            
            if features_df.empty:
                logger.warning(f"Failed to generate features for {symbol} {timeframe}")
                return []
            
            # Load model if not already loaded
            model_name = f"{symbol}_{timeframe}_ml_model"
            
            if not self.model_loaded:
                if not self.prediction_engine.load_model(model_name):
                    logger.warning(f"Model {model_name} not found, skipping prediction")
                    return []
                self.model_loaded = True
            
            # Make prediction
            prediction = self.prediction_engine.predict(model_name, features_df)
            
            if prediction is None or len(prediction) == 0:
                logger.warning(f"Failed to make prediction for {symbol} {timeframe}")
                return []
            
            # Get the last prediction
            last_prediction = prediction[-1]
            
            # Get confidence scores
            confidence = self.prediction_engine.get_prediction_confidence(model_name, features_df)
            last_confidence = confidence[-1] if confidence is not None and len(confidence) > 0 else 0.5
            
            # Generate signals
            signals = []
            
            # Current price
            current_price = features_df['close'].iloc[-1]
            
            # For classification models
            if isinstance(last_prediction, (int, np.integer)):
                # 1 = buy, 0 = hold, -1 = sell
                if last_prediction == 1 and last_confidence > 0.6:
                    # Buy signal
                    # Calculate stop loss and take profit
                    atr = features_df['atr_14'].iloc[-1] if 'atr_14' in features_df.columns else current_price * 0.02
                    stop_loss = current_price - atr * 1.5
                    take_profit = current_price + atr * 2.0
                    
                    # Create signal
                    signal = self.create_signal(
                        symbol=symbol,
                        timeframe=timeframe,
                        direction="BUY",
                        strength=last_confidence,
                        price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'model': model_name,
                            'prediction': int(last_prediction),
                            'confidence': float(last_confidence)
                        }
                    )
                    
                    signals.append(signal)
                    logger.info(f"BUY signal for {symbol} at {current_price} (confidence: {last_confidence:.2f})")
                
                elif last_prediction == -1 and last_confidence > 0.6:
                    # Sell signal
                    # Calculate stop loss and take profit
                    atr = features_df['atr_14'].iloc[-1] if 'atr_14' in features_df.columns else current_price * 0.02
                    stop_loss = current_price + atr * 1.5
                    take_profit = current_price - atr * 2.0
                    
                    # Create signal
                    signal = self.create_signal(
                        symbol=symbol,
                        timeframe=timeframe,
                        direction="SELL",
                        strength=last_confidence,
                        price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'model': model_name,
                            'prediction': int(last_prediction),
                            'confidence': float(last_confidence)
                        }
                    )
                    
                    signals.append(signal)
                    logger.info(f"SELL signal for {symbol} at {current_price} (confidence: {last_confidence:.2f})")
            
            # For regression models (predicting price or return)
            else:
                # Predict future price
                predicted_return = float(last_prediction)
                
                # Buy signal: Predicted positive return
                if predicted_return > 0.01:  # 1% threshold
                    # Determine signal strength based on predicted return
                    strength = min(1.0, predicted_return * 10)
                    
                    # Calculate stop loss and take profit
                    atr = features_df['atr_14'].iloc[-1] if 'atr_14' in features_df.columns else current_price * 0.02
                    stop_loss = current_price - atr * 1.5
                    take_profit = current_price * (1 + predicted_return)
                    
                    # Create signal
                    signal = self.create_signal(
                        symbol=symbol,
                        timeframe=timeframe,
                        direction="BUY",
                        strength=strength,
                        price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'model': model_name,
                            'predicted_return': predicted_return
                        }
                    )
                    
                    signals.append(signal)
                    logger.info(f"BUY signal for {symbol} at {current_price} (predicted return: {predicted_return:.2%})")
                
                # Sell signal: Predicted negative return
                elif predicted_return < -0.01:  # -1% threshold
                    # Determine signal strength based on predicted return
                    strength = min(1.0, abs(predicted_return) * 10)
                    
                    # Calculate stop loss and take profit
                    atr = features_df['atr_14'].iloc[-1] if 'atr_14' in features_df.columns else current_price * 0.02
                    stop_loss = current_price + atr * 1.5
                    take_profit = current_price * (1 + predicted_return)
                    
                    # Create signal
                    signal = self.create_signal(
                        symbol=symbol,
                        timeframe=timeframe,
                        direction="SELL",
                        strength=strength,
                        price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'model': model_name,
                            'predicted_return': predicted_return
                        }
                    )
                    
                    signals.append(signal)
                    logger.info(f"SELL signal for {symbol} at {current_price} (predicted return: {predicted_return:.2%})")
            
            return signals
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {e}")
            handle_error(e, f"Failed to generate signals for {symbol} {timeframe}")
            return []


class KalmanFilterStrategy(Strategy):
    """Kalman filter-based trading strategy."""
    
    def __init__(self, timeframes=None, symbols=None, params=None):
        """
        Initialize the strategy.
        
        Args:
            timeframes (list): List of timeframes to use
            symbols (list): List of symbols to trade
            params (dict): Strategy parameters
        """
        default_params = STATISTICAL_PARAMS.get('kalman_filter', {})
        params = {**default_params, **(params or {})}
        
        super().__init__('KalmanFilter', timeframes, symbols, params)
        
        # Initialize Kalman filter state for each symbol and timeframe
        self.kalman_states = {}
    
    def generate_signals(self, data, symbol, timeframe):
        """
        Generate trading signals for a symbol and timeframe.
        
        Args:
            data (pd.DataFrame): Historical price data
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            
        Returns:
            list: List of signal dictionaries
        """
        try:
            if data.empty:
                logger.warning(f"Empty data for {symbol} {timeframe}")
                return []
            
            # Preprocess data
            df = self.preprocess_data(data)
            
            if df.empty:
                return []
            
            # Get parameters
            process_variance = self.params.get('process_variance', 1e-5)
            measurement_variance = self.params.get('measurement_variance', 0.1)
            
            # Get state key for this symbol and timeframe
            state_key = f"{symbol}_{timeframe}"
            
            # Initialize state if not already done
            if state_key not in self.kalman_states:
                # Initial state for the Kalman filter
                self.kalman_states[state_key] = {
                    'x': df['close'].iloc[0],  # Initial state (price)
                    'p': 1.0,                  # Initial uncertainty
                    'k': 0.0,                  # Initial Kalman gain
                    'history': []              # History of state estimates
                }
            
            # Get current state
            state = self.kalman_states[state_key]
            
            # Apply Kalman filter to the price series
            filtered_prices = []
            
            for _, row in df.iterrows():
                # Prediction step
                x = state['x']
                p = state['p'] + process_variance
                
                # Update step
                k = p / (p + measurement_variance)
                x = x + k * (row['close'] - x)
                p = (1 - k) * p
                
                # Update state
                state['x'] = x
                state['p'] = p
                state['k'] = k
                
                # Store filtered price
                filtered_prices.append(x)
            
            # Add filtered prices to the DataFrame
            df['kalman'] = filtered_prices
            
            # Save history
            state['history'] = filtered_prices[-100:]  # Keep last 100 values
            
            # Generate signals
            signals = []
            
            # Check if we have enough data
            if len(df) < 10:
                logger.warning(f"Not enough data for {symbol} {timeframe}")
                return []
            
            # Calculate price difference from Kalman estimate
            df['price_diff'] = df['close'] - df['kalman']
            
            # Calculate standard deviation of price difference
            price_diff_std = df['price_diff'].std()
            
            # Get the last row
            curr_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # Current price and Kalman estimate
            current_price = curr_row['close']
            kalman_price = curr_row['kalman']
            
            # Price difference in standard deviations
            diff_in_std = curr_row['price_diff'] / price_diff_std if price_diff_std > 0 else 0
            
            # Determine if price crossed the Kalman filter line
            crossed_above = prev_row['close'] <= prev_row['kalman'] and curr_row['close'] > curr_row['kalman']
            crossed_below = prev_row['close'] >= prev_row['kalman'] and curr_row['close'] < curr_row['kalman']
            
            # Buy signal: Price is significantly below Kalman estimate and crossing up
            if crossed_above or (diff_in_std < -2.0 and curr_row['price_diff'] > prev_row['price_diff']):
                # Calculate stop loss and take profit
                stop_loss = current_price - price_diff_std * 2
                take_profit = kalman_price
                
                # Create signal
                signal = self.create_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction="BUY",
                    strength=min(1.0, abs(diff_in_std) / 3),
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'kalman_price': kalman_price,
                        'diff_std': diff_in_std,
                        'crossed': crossed_above
                    }
                )
                
                signals.append(signal)
                logger.info(f"BUY signal for {symbol} at {current_price} (Kalman: {kalman_price:.2f})")
            
            # Sell signal: Price is significantly above Kalman estimate and crossing down
            elif crossed_below or (diff_in_std > 2.0 and curr_row['price_diff'] < prev_row['price_diff']):
                # Calculate stop loss and take profit
                stop_loss = current_price + price_diff_std * 2
                take_profit = kalman_price
                
                # Create signal
                signal = self.create_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction="SELL",
                    strength=min(1.0, abs(diff_in_std) / 3),
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'kalman_price': kalman_price,
                        'diff_std': diff_in_std,
                        'crossed': crossed_below
                    }
                )
                
                signals.append(signal)
                logger.info(f"SELL signal for {symbol} at {current_price} (Kalman: {kalman_price:.2f})")
            
            return signals
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {e}")
            handle_error(e, f"Failed to generate signals for {symbol} {timeframe}")
            return []