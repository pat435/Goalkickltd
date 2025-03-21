"""
Math utilities for the Goalkick Ltd Trading Bot.
Implements various mathematical functions used in trading algorithms.
"""

import numpy as np
import pandas as pd
from scipy import stats
import math

from config.logging_config import get_logger

logger = get_logger("utils.math")

def round_to_precision(value, precision):
    """
    Round a value to a specific precision.
    
    Args:
        value (float): Value to round
        precision (float): Precision to round to
        
    Returns:
        float: Rounded value
    """
    if precision == 0:
        return int(value)
    
    factor = 1.0 / precision
    return round(value * factor) / factor

def round_to_tick_size(value, tick_size):
    """
    Round a value to a specific tick size.
    
    Args:
        value (float): Value to round
        tick_size (float): Tick size
        
    Returns:
        float: Rounded value
    """
    return round(value / tick_size) * tick_size

def calculate_simple_returns(prices):
    """
    Calculate simple returns from a price series.
    
    Args:
        prices (array-like): Price series
        
    Returns:
        numpy.ndarray: Simple returns
    """
    return np.diff(prices) / prices[:-1]

def calculate_log_returns(prices):
    """
    Calculate logarithmic returns from a price series.
    
    Args:
        prices (array-like): Price series
        
    Returns:
        numpy.ndarray: Logarithmic returns
    """
    return np.diff(np.log(prices))

def calculate_volatility(returns, window=20, annualize=True, trading_periods=252):
    """
    Calculate volatility from returns.
    
    Args:
        returns (array-like): Return series
        window (int): Window size for rolling volatility
        annualize (bool): Whether to annualize the volatility
        trading_periods (int): Number of trading periods in a year
        
    Returns:
        pandas.Series: Volatility series
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    # Calculate rolling standard deviation
    volatility = returns.rolling(window=window).std()
    
    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(trading_periods)
    
    return volatility

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods=252):
    """
    Calculate Sharpe ratio.
    
    Args:
        returns (array-like): Return series
        risk_free_rate (float): Risk-free rate
        periods (int): Number of periods in a year
        
    Returns:
        float: Sharpe ratio
    """
    # Convert to numpy array if needed
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = returns.values
    
    # Calculate excess returns
    excess_returns = returns - risk_free_rate / periods
    
    # Calculate Sharpe ratio
    if len(excess_returns) > 1:
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(periods)
    else:
        sharpe_ratio = 0.0
    
    return sharpe_ratio

def calculate_sortino_ratio(returns, risk_free_rate=0.0, periods=252):
    """
    Calculate Sortino ratio.
    
    Args:
        returns (array-like): Return series
        risk_free_rate (float): Risk-free rate
        periods (int): Number of periods in a year
        
    Returns:
        float: Sortino ratio
    """
    # Convert to numpy array if needed
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = returns.values
    
    # Calculate excess returns
    excess_returns = returns - risk_free_rate / periods
    
    # Calculate downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) > 1 and len(excess_returns) > 1:
        downside_deviation = np.std(negative_returns, ddof=1)
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(periods)
    else:
        sortino_ratio = 0.0
    
    return sortino_ratio

def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown.
    
    Args:
        equity_curve (array-like): Equity curve
        
    Returns:
        tuple: (Maximum drawdown percentage, start index, end index)
    """
    # Convert to numpy array if needed
    if isinstance(equity_curve, (pd.Series, pd.DataFrame)):
        equity_curve = equity_curve.values
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Find maximum drawdown
    max_drawdown = np.min(drawdown)
    end_idx = np.argmin(drawdown)
    
    # Find start of the drawdown period
    start_idx = np.argmax(equity_curve[:end_idx+1])
    
    return max_drawdown, start_idx, end_idx

def calculate_linear_regression(x, y):
    """
    Calculate linear regression.
    
    Args:
        x (array-like): Independent variable
        y (array-like): Dependent variable
        
    Returns:
        tuple: (Slope, Intercept, R-squared)
    """
    # Convert to numpy arrays if needed
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.values
    
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values
    
    # Add constant term for statsmodels
    X = np.column_stack((np.ones(len(x)), x))
    
    # Calculate coefficients using least squares
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Extract intercept and slope
    intercept, slope = beta
    
    # Calculate predictions
    y_pred = intercept + slope * x
    
    # Calculate R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    return slope, intercept, r_squared

def calculate_z_score(series, window=20):
    """
    Calculate rolling z-score.
    
    Args:
        series (array-like): Data series
        window (int): Window size
        
    Returns:
        pandas.Series: Z-score series
    """
    # Convert to pandas Series if needed
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    # Calculate rolling mean and standard deviation
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Calculate z-score
    z_score = (series - rolling_mean) / rolling_std
    
    return z_score

def calculate_correlation_matrix(returns_df):
    """
    Calculate correlation matrix for asset returns.
    
    Args:
        returns_df (pandas.DataFrame): DataFrame of returns for multiple assets
        
    Returns:
        pandas.DataFrame: Correlation matrix
    """
    return returns_df.corr()

def calculate_portfolio_variance(weights, covariance_matrix):
    """
    Calculate portfolio variance.
    
    Args:
        weights (array-like): Portfolio weights
        covariance_matrix (array-like): Covariance matrix
        
    Returns:
        float: Portfolio variance
    """
    # Convert to numpy arrays if needed
    if isinstance(weights, (pd.Series, pd.DataFrame)):
        weights = weights.values
    
    if isinstance(covariance_matrix, pd.DataFrame):
        covariance_matrix = covariance_matrix.values
    
    # Calculate portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    
    return portfolio_variance

def calculate_portfolio_return(weights, returns):
    """
    Calculate portfolio return.
    
    Args:
        weights (array-like): Portfolio weights
        returns (array-like): Asset returns
        
    Returns:
        float: Portfolio return
    """
    # Convert to numpy arrays if needed
    if isinstance(weights, (pd.Series, pd.DataFrame)):
        weights = weights.values
    
    if isinstance(returns, pd.DataFrame):
        returns = returns.values
    
    # Calculate portfolio return
    portfolio_return = np.sum(returns * weights)
    
    return portfolio_return

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        prices (array-like): Price series
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period
        
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    # Convert to pandas Series if needed
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    
    # Calculate fast and slow EMAs
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices (array-like): Price series
        period (int): RSI period
        
    Returns:
        pandas.Series: RSI values
    """
    # Convert to pandas Series if needed
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()
    
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period, min_periods=1).mean()
    avg_losses = losses.rolling(window=period, min_periods=1).mean()
    
    # Calculate RS
    rs = avg_gains / avg_losses
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Handle nan and inf
    rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    
    return rsi

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        prices (array-like): Price series
        period (int): Period for moving average
        num_std_dev (float): Number of standard deviations
        
    Returns:
        tuple: (Middle Band, Upper Band, Lower Band)
    """
    # Convert to pandas Series if needed
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    
    # Calculate middle band (SMA)
    middle_band = prices.rolling(window=period).mean()
    
    # Calculate standard deviation
    std_dev = prices.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    
    return middle_band, upper_band, lower_band

def calculate_atr(high, low, close, period=14):
    """
    Calculate Average True Range (ATR).
    
    Args:
        high (array-like): High prices
        low (array-like): Low prices
        close (array-like): Close prices
        period (int): ATR period
        
    Returns:
        pandas.Series: ATR values
    """
    # Convert to pandas Series if needed
    if not isinstance(high, pd.Series):
        high = pd.Series(high)
    
    if not isinstance(low, pd.Series):
        low = pd.Series(low)
    
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    # Combine true ranges
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_kelly_criterion(win_rate, win_loss_ratio):
    """
    Calculate Kelly Criterion for optimal bet size.
    
    Args:
        win_rate (float): Win rate (0-1)
        win_loss_ratio (float): Ratio of average win to average loss
        
    Returns:
        float: Kelly percentage
    """
    # Calculate Kelly percentage
    kelly_percentage = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # Limit to range [0, 1]
    kelly_percentage = max(0, min(kelly_percentage, 1))
    
    return kelly_percentage

def calculate_expected_value(probabilities, outcomes):
    """
    Calculate expected value.
    
    Args:
        probabilities (array-like): Probability of each outcome
        outcomes (array-like): Value of each outcome
        
    Returns:
        float: Expected value
    """
    # Ensure arrays have the same length
    if len(probabilities) != len(outcomes):
        raise ValueError("Probabilities and outcomes must have the same length")
    
    # Calculate expected value
    expected_value = sum(p * o for p, o in zip(probabilities, outcomes))
    
    return expected_value

def calculate_profit_factor(wins, losses):
    """
    Calculate profit factor (sum of wins / sum of losses).
    
    Args:
        wins (array-like): Winning trades
        losses (array-like): Losing trades
        
    Returns:
        float: Profit factor
    """
    # Calculate sum of wins and losses
    sum_wins = sum(wins)
    sum_losses = sum(abs(l) for l in losses)
    
    # Calculate profit factor
    if sum_losses == 0:
        return float('inf') if sum_wins > 0 else 0
    
    profit_factor = sum_wins / sum_losses
    
    return profit_factor

def calculate_expectancy(win_rate, avg_win, avg_loss):
    """
    Calculate system expectancy.
    
    Args:
        win_rate (float): Win rate (0-1)
        avg_win (float): Average win
        avg_loss (float): Average loss (positive value)
        
    Returns:
        float: System expectancy
    """
    # Calculate expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    return expectancy

def calculate_risk_of_ruin(win_rate, risk_reward_ratio, trades=1000):
    """
    Calculate risk of ruin (probability of losing entire account).
    
    Args:
        win_rate (float): Win rate (0-1)
        risk_reward_ratio (float): Risk/reward ratio
        trades (int): Number of trades
        
    Returns:
        float: Risk of ruin
    """
    # Calculate single-trade probability of profit
    p = win_rate
    q = 1 - p
    r = risk_reward_ratio
    
    # Check if we expect to lose money
    if p * r <= q:
        return 1.0
    
    # Calculate risk of ruin
    ruin = ((q / (p * r))**trades)
    
    return ruin

def calculate_optimal_position_size(account_balance, risk_per_trade, stop_loss_percent):
    """
    Calculate optimal position size based on fixed percentage risk model.
    
    Args:
        account_balance (float): Account balance
        risk_per_trade (float): Risk per trade as a decimal (0-1)
        stop_loss_percent (float): Stop loss as a decimal (0-1)
        
    Returns:
        float: Optimal position size
    """
    # Calculate risk amount
    risk_amount = account_balance * risk_per_trade
    
    # Calculate position size
    if stop_loss_percent == 0:
        return 0
    
    position_size = risk_amount / stop_loss_percent
    
    return position_size

def normalize_data(data, method='z-score'):
    """
    Normalize data using various methods.
    
    Args:
        data (array-like): Data to normalize
        method (str): Normalization method ('z-score', 'min-max', 'decimal-scaling')
        
    Returns:
        array-like: Normalized data
    """
    # Convert to numpy array
    data_array = np.array(data)
    
    if method == 'z-score':
        # Z-score normalization
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        if std == 0:
            return np.zeros_like(data_array)
        
        normalized = (data_array - mean) / std
    
    elif method == 'min-max':
        # Min-max normalization
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        
        if max_val == min_val:
            return np.zeros_like(data_array)
        
        normalized = (data_array - min_val) / (max_val - min_val)
    
    elif method == 'decimal-scaling':
        # Decimal scaling
        max_abs = np.max(np.abs(data_array))
        decimal_places = len(str(int(max_abs)))
        
        normalized = data_array / (10 ** decimal_places)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

def ewma(data, alpha=0.5):
    """
    Calculate Exponentially Weighted Moving Average (EWMA).
    
    Args:
        data (array-like): Data series
        alpha (float): Smoothing factor (0-1)
        
    Returns:
        array-like: EWMA values
    """
    # Convert to numpy array
    data_array = np.array(data)
    
    # Initialize result array
    ewma_values = np.zeros_like(data_array)
    ewma_values[0] = data_array[0]
    
    # Calculate EWMA
    for i in range(1, len(data_array)):
        ewma_values[i] = alpha * data_array[i] + (1 - alpha) * ewma_values[i-1]
    
    return ewma_values

def calculate_drawdowns(equity_curve):
    """
    Calculate all drawdowns in an equity curve.
    
    Args:
        equity_curve (array-like): Equity curve
        
    Returns:
        list: List of drawdowns as (start_idx, end_idx, drawdown) tuples
    """
    # Convert to numpy array
    eq_array = np.array(equity_curve)
    
    # Find peaks
    peak_mask = np.r_[True, eq_array[1:] < eq_array[:-1]] & np.r_[eq_array[:-1] < eq_array[1:], True]
    peaks = np.where(peak_mask)[0]
    
    # Calculate drawdowns
    drawdowns = []
    
    for i in range(len(peaks)-1):
        peak_idx = peaks[i]
        peak_value = eq_array[peak_idx]
        
        # Find trough
        trough_idx = peak_idx + np.argmin(eq_array[peak_idx:peaks[i+1]+1])
        trough_value = eq_array[trough_idx]
        
        # Calculate drawdown
        drawdown = (trough_value - peak_value) / peak_value
        
        # Only include actual drawdowns
        if drawdown < 0:
            drawdowns.append((peak_idx, trough_idx, drawdown))
    
    return drawdowns

def calculate_underwater_curve(equity_curve):
    """
    Calculate underwater curve (drawdowns over time).
    
    Args:
        equity_curve (array-like): Equity curve
        
    Returns:
        array-like: Underwater curve
    """
    # Convert to numpy array
    eq_array = np.array(equity_curve)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(eq_array)
    
    # Calculate underwater curve
    underwater = (eq_array - running_max) / running_max
    
    return underwater

def fit_distribution(data, dist_name='norm'):
    """
    Fit a statistical distribution to data.
    
    Args:
        data (array-like): Data to fit
        dist_name (str): Distribution name from scipy.stats
        
    Returns:
        tuple: (Distribution parameters, Goodness of fit)
    """
    # Get distribution
    distribution = getattr(stats, dist_name)
    
    # Fit distribution
    params = distribution.fit(data)
    
    # Calculate goodness of fit (K-S test)
    ks_statistic, p_value = stats.kstest(data, dist_name, params)
    
    return params, (ks_statistic, p_value)

def calculate_var(returns, confidence_level=0.95, window=None):
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns (array-like): Return series
        confidence_level (float): Confidence level (0-1)
        window (int): Rolling window size (None for full series)
        
    Returns:
        float or array-like: VaR value(s)
    """
    # Convert to pandas Series if needed
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    if window is None:
        # Calculate VaR for full series
        var = np.percentile(returns, 100 * (1 - confidence_level))
        return var
    else:
        # Calculate rolling VaR
        rolling_var = returns.rolling(window=window).apply(
            lambda x: np.percentile(x, 100 * (1 - confidence_level)),
            raw=True
        )
        return rolling_var

def hull_moving_average(data, period=16):
    """
    Calculate Hull Moving Average.
    
    Args:
        data (array-like): Data series
        period (int): HMA period
        
    Returns:
        array-like: HMA values
    """
    # Convert to pandas Series if needed
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Calculate WMAs
    half_period = int(period / 2)
    sqrt_period = int(math.sqrt(period))
    
    wma1 = data.rolling(window=half_period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)),
        raw=True
    )
    
    wma2 = data.rolling(window=period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)),
        raw=True
    )
    
    # Calculate 2*WMA(n/2) - WMA(n)
    wma_diff = 2 * wma1 - wma2
    
    # Calculate WMA of the difference using sqrt(n)
    hma = wma_diff.rolling(window=sqrt_period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)),
        raw=True
    )
    
    return hma