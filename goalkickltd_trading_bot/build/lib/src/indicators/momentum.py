"""
Momentum indicators for the Goalkick Ltd Trading Bot.
Implements momentum-based technical indicators like RSI, MACD, Stochastic, etc.
"""

import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger("indicators.momentum")

def relative_strength_index(data, window=14, round_values=True):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): RSI period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of RSI values
    """
    # Calculate price changes
    delta = data['close'].diff()
    
    # Separate gains and losses
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    # Calculate relative strength
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Handle extremes (inf or nan)
    rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    
    if round_values:
        rsi = rsi.round(2)
    
    return rsi

def moving_average_convergence_divergence(data, fast_length=12, slow_length=26, signal_length=9, round_values=True):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        fast_length (int): Fast EMA period
        slow_length (int): Slow EMA period
        signal_length (int): Signal line period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    # Calculate EMAs
    ema_fast = data['close'].ewm(span=fast_length, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow_length, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_length, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    if round_values:
        macd_line = macd_line.round(2)
        signal_line = signal_line.round(2)
        histogram = histogram.round(2)
    
    return macd_line, signal_line, histogram

def stochastic_oscillator(data, k_period=14, d_period=3, smooth_k=3, round_values=True):
    """
    Calculate Stochastic Oscillator.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        k_period (int): %K period
        d_period (int): %D period
        smooth_k (int): %K smoothing period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        tuple: (%K, %D)
    """
    # Calculate %K
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    
    # Handle division by zero
    range_diff = high_max - low_min
    range_diff = range_diff.replace(0, np.nan)
    
    k = 100 * ((data['close'] - low_min) / range_diff)
    
    # Smooth %K if specified
    if smooth_k > 1:
        k = k.rolling(window=smooth_k).mean()
    
    # Calculate %D
    d = k.rolling(window=d_period).mean()
    
    # Replace nans with 50 (neutral)
    k = k.replace([np.inf, -np.inf], np.nan).fillna(50)
    d = d.replace([np.inf, -np.inf], np.nan).fillna(50)
    
    if round_values:
        k = k.round(2)
        d = d.round(2)
    
    return k, d

def commodity_channel_index(data, window=20, round_values=True):
    """
    Calculate Commodity Channel Index (CCI).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): CCI period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of CCI values
    """
    # Calculate typical price
    tp = (data['high'] + data['low'] + data['close']) / 3
    
    # Calculate simple moving average of typical price
    sma_tp = tp.rolling(window=window).mean()
    
    # Calculate mean deviation
    mean_dev = tp.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
    
    # Calculate CCI
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    
    # Replace nans with 0 (neutral)
    cci = cci.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if round_values:
        cci = cci.round(2)
    
    return cci

def rate_of_change(data, window=14, round_values=True):
    """
    Calculate Rate of Change (ROC).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): ROC period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of ROC values
    """
    # Calculate ROC
    roc = ((data['close'] / data['close'].shift(window)) - 1) * 100
    
    # Replace nans with 0 (neutral)
    roc = roc.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if round_values:
        roc = roc.round(2)
    
    return roc

def average_directional_index(data, window=14, round_values=True):
    """
    Calculate Average Directional Index (ADX).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): ADX period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        tuple: (ADX, +DI, -DI)
    """
    # Calculate True Range
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    # Calculate +DM and -DM
    plus_dm = data['high'].diff()
    minus_dm = data['low'].diff().multiply(-1)
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Convert to pandas series
    plus_dm = pd.Series(plus_dm, index=data.index)
    minus_dm = pd.Series(minus_dm, index=data.index)
    
    # Calculate smoothed values
    tr_smoothed = true_range.rolling(window=window).sum()
    plus_dm_smoothed = plus_dm.rolling(window=window).sum()
    minus_dm_smoothed = minus_dm.rolling(window=window).sum()
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smoothed / tr_smoothed)
    minus_di = 100 * (minus_dm_smoothed / tr_smoothed)
    
    # Calculate DX
    dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
    dx = dx.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate ADX
    adx = dx.rolling(window=window).mean()
    
    # Replace nans with 0 (neutral)
    adx = adx.replace([np.inf, -np.inf], np.nan).fillna(0)
    plus_di = plus_di.replace([np.inf, -np.inf], np.nan).fillna(0)
    minus_di = minus_di.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if round_values:
        adx = adx.round(2)
        plus_di = plus_di.round(2)
        minus_di = minus_di.round(2)
    
    return adx, plus_di, minus_di

def williams_r(data, window=14, round_values=True):
    """
    Calculate Williams %R.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): %R period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of Williams %R values
    """
    # Calculate highest high and lowest low
    highest_high = data['high'].rolling(window=window).max()
    lowest_low = data['low'].rolling(window=window).min()
    
    # Calculate Williams %R
    wr = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
    
    # Replace nans with -50 (neutral)
    wr = wr.replace([np.inf, -np.inf], np.nan).fillna(-50)
    
    if round_values:
        wr = wr.round(2)
    
    return wr

def money_flow_index(data, window=14, round_values=True):
    """
    Calculate Money Flow Index (MFI).
    
    Args:
        data (pd.DataFrame): DataFrame with price and volume data
        window (int): MFI period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of MFI values
    """
    # Calculate typical price
    tp = (data['high'] + data['low'] + data['close']) / 3
    
    # Calculate money flow
    money_flow = tp * data['volume']
    
    # Get positive and negative money flow
    positive_flow = pd.Series(np.where(tp > tp.shift(1), money_flow, 0), index=data.index)
    negative_flow = pd.Series(np.where(tp < tp.shift(1), money_flow, 0), index=data.index)
    
    # Calculate money flow ratio
    positive_sum = positive_flow.rolling(window=window).sum()
    negative_sum = negative_flow.rolling(window=window).sum()
    
    # Handle division by zero
    negative_sum = negative_sum.replace(0, np.nan)
    
    money_ratio = positive_sum / negative_sum
    
    # Calculate MFI
    mfi = 100 - (100 / (1 + money_ratio))
    
    # Replace nans with 50 (neutral)
    mfi = mfi.replace([np.inf, -np.inf], np.nan).fillna(50)
    
    if round_values:
        mfi = mfi.round(2)
    
    return mfi

def momentum(data, window=14, round_values=True):
    """
    Calculate Momentum.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): Momentum period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of Momentum values
    """
    # Calculate Momentum
    mom = data['close'] - data['close'].shift(window)
    
    # Replace nans with 0 (neutral)
    mom = mom.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if round_values:
        mom = mom.round(2)
    
    return mom

def tsi(data, long_window=25, short_window=13, signal_window=7, round_values=True):
    """
    Calculate True Strength Index (TSI).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        long_window (int): Long EMA period
        short_window (int): Short EMA period
        signal_window (int): Signal line period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        tuple: (TSI, Signal)
    """
    # Calculate momentum
    mom = data['close'].diff()
    
    # Calculate smoothed momentum
    mom_long_ema = mom.ewm(span=long_window, adjust=False).mean()
    mom_double_ema = mom_long_ema.ewm(span=short_window, adjust=False).mean()
    
    # Calculate absolute momentum
    abs_mom = mom.abs()
    
    # Calculate smoothed absolute momentum
    abs_mom_long_ema = abs_mom.ewm(span=long_window, adjust=False).mean()
    abs_mom_double_ema = abs_mom_long_ema.ewm(span=short_window, adjust=False).mean()
    
    # Calculate TSI
    tsi_value = (mom_double_ema / abs_mom_double_ema) * 100
    
    # Calculate signal line
    signal = tsi_value.ewm(span=signal_window, adjust=False).mean()
    
    # Replace nans with 0 (neutral)
    tsi_value = tsi_value.replace([np.inf, -np.inf], np.nan).fillna(0)
    signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if round_values:
        tsi_value = tsi_value.round(2)
        signal = signal.round(2)
    
    return tsi_value, signal