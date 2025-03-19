"""
Volatility indicators for the Goalkick Ltd Trading Bot.
Implements volatility-based technical indicators like Bollinger Bands, ATR, etc.
"""

import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger("indicators.volatility")

def bollinger_bands(data, window=20, num_std=2, round_values=True):
    """
    Calculate Bollinger Bands.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): Moving average period
        num_std (float): Number of standard deviations
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        tuple: (Middle Band, Upper Band, Lower Band)
    """
    # Calculate middle band (SMA)
    middle_band = data['close'].rolling(window=window).mean()
    
    # Calculate standard deviation
    std_dev = data['close'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    if round_values:
        middle_band = middle_band.round(2)
        upper_band = upper_band.round(2)
        lower_band = lower_band.round(2)
    
    return middle_band, upper_band, lower_band

def average_true_range(data, window=14, round_values=True):
    """
    Calculate Average True Range (ATR).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): ATR period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of ATR values
    """
    # Calculate True Range
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    # Calculate ATR
    atr = true_range.rolling(window=window).mean()
    
    # Replace nans with first valid value or zero
    atr = atr.fillna(method='bfill').fillna(0)
    
    if round_values:
        atr = atr.round(2)
    
    return atr

def keltner_channel(data, ema_window=20, atr_window=10, multiplier=2, round_values=True):
    """
    Calculate Keltner Channel.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        ema_window (int): EMA period
        atr_window (int): ATR period
        multiplier (float): ATR multiplier
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        tuple: (Middle Line, Upper Channel, Lower Channel)
    """
    # Calculate middle line (EMA)
    middle_line = data['close'].ewm(span=ema_window, adjust=False).mean()
    
    # Calculate ATR
    atr = average_true_range(data, window=atr_window, round_values=False)
    
    # Calculate upper and lower channels
    upper_channel = middle_line + (atr * multiplier)
    lower_channel = middle_line - (atr * multiplier)
    
    if round_values:
        middle_line = middle_line.round(2)
        upper_channel = upper_channel.round(2)
        lower_channel = lower_channel.round(2)
    
    return middle_line, upper_channel, lower_channel

def donchian_channel(data, window=20, round_values=True):
    """
    Calculate Donchian Channel.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): Window period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        tuple: (Upper Channel, Middle Channel, Lower Channel)
    """
    # Calculate upper and lower channels
    upper_channel = data['high'].rolling(window=window).max()
    lower_channel = data['low'].rolling(window=window).min()
    
    # Calculate middle channel
    middle_channel = (upper_channel + lower_channel) / 2
    
    if round_values:
        upper_channel = upper_channel.round(2)
        middle_channel = middle_channel.round(2)
        lower_channel = lower_channel.round(2)
    
    return upper_channel, middle_channel, lower_channel

def volatility_ratio(data, window=14, round_values=True):
    """
    Calculate Volatility Ratio.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): Window period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of Volatility Ratio values
    """
    # Calculate returns
    returns = data['close'].pct_change()
    
    # Calculate volatility (standard deviation of returns)
    volatility = returns.rolling(window=window).std()
    
    # Calculate average volatility
    avg_volatility = volatility.rolling(window=window*3).mean()
    
    # Calculate volatility ratio
    vol_ratio = volatility / avg_volatility
    
    # Replace nans with 1 (neutral)
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    if round_values:
        vol_ratio = vol_ratio.round(2)
    
    return vol_ratio

def normalized_average_true_range(data, window=14, round_values=True):
    """
    Calculate Normalized Average True Range (NATR).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): ATR period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of NATR values
    """
    # Calculate ATR
    atr = average_true_range(data, window=window, round_values=False)
    
    # Normalize ATR
    natr = (atr / data['close']) * 100
    
    # Replace nans with first valid value or zero
    natr = natr.fillna(method='bfill').fillna(0)
    
    if round_values:
        natr = natr.round(2)
    
    return natr

def chaikin_volatility(data, ema_period=10, window=10, round_values=True):
    """
    Calculate Chaikin Volatility.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        ema_period (int): EMA period for high-low range
        window (int): Window for rate of change
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of Chaikin Volatility values
    """
    # Calculate high-low range
    hl_range = data['high'] - data['low']
    
    # Calculate EMA of high-low range
    ema_hl_range = hl_range.ewm(span=ema_period, adjust=False).mean()
    
    # Calculate rate of change
    chaikin_vol = ((ema_hl_range - ema_hl_range.shift(window)) / ema_hl_range.shift(window)) * 100
    
    # Replace nans with 0 (neutral)
    chaikin_vol = chaikin_vol.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if round_values:
        chaikin_vol = chaikin_vol.round(2)
    
    return chaikin_vol

def historical_volatility(data, window=21, trading_periods=252, round_values=True):
    """
    Calculate Historical Volatility.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): Window period for standard deviation
        trading_periods (int): Number of trading periods in a year
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of Historical Volatility values (annualized)
    """
    # Calculate log returns
    log_returns = np.log(data['close'] / data['close'].shift(1))
    
    # Calculate standard deviation of log returns
    std_dev = log_returns.rolling(window=window).std()
    
    # Annualize volatility
    volatility = std_dev * np.sqrt(trading_periods) * 100
    
    # Replace nans with 0 (neutral)
    volatility = volatility.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if round_values:
        volatility = volatility.round(2)
    
    return volatility

def average_day_range(data, window=14, round_values=True):
    """
    Calculate Average Day Range.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): Window period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of Average Day Range values
    """
    # Calculate daily range
    day_range = (data['high'] - data['low'])
    
    # Calculate average day range
    adr = day_range.rolling(window=window).mean()
    
    if round_values:
        adr = adr.round(2)
    
    return adr