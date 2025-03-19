"""
Trend indicators for the Goalkick Ltd Trading Bot.
Implements trend-based technical indicators like Moving Averages, PSAR, etc.
"""

import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger("indicators.trend")

def simple_moving_average(data, window, round_values=True):
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): SMA period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of SMA values
    """
    sma = data['close'].rolling(window=window).mean()
    
    if round_values:
        sma = sma.round(2)
    
    return sma

def exponential_moving_average(data, window, round_values=True):
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): EMA period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of EMA values
    """
    ema = data['close'].ewm(span=window, adjust=False).mean()
    
    if round_values:
        ema = ema.round(2)
    
    return ema

def weighted_moving_average(data, window, round_values=True):
    """
    Calculate Weighted Moving Average (WMA).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): WMA period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of WMA values
    """
    weights = np.arange(1, window + 1)
    wma = data['close'].rolling(window=window).apply(
        lambda x: np.sum(weights * x) / weights.sum(), raw=True)
    
    if round_values:
        wma = wma.round(2)
    
    return wma

def hull_moving_average(data, window, round_values=True):
    """
    Calculate Hull Moving Average (HMA).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): HMA period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of HMA values
    """
    # Calculate WMA with period window/2
    half_window = int(window / 2)
    wma_half = weighted_moving_average(data, half_window, round_values=False)
    
    # Calculate WMA with period window
    wma_full = weighted_moving_average(data, window, round_values=False)
    
    # Calculate 2 * WMA(half_window) - WMA(window)
    diff = (2 * wma_half) - wma_full
    
    # Calculate WMA with period sqrt(window) on the difference
    sqrt_window = int(np.sqrt(window))
    
    # Create temporary DataFrame for the diff series
    temp_df = pd.DataFrame({'close': diff})
    
    # Calculate HMA
    hma = weighted_moving_average(temp_df, sqrt_window, round_values=False)
    
    if round_values:
        hma = hma.round(2)
    
    return hma

def parabolic_sar(data, af_start=0.02, af_step=0.02, af_max=0.2, round_values=True):
    """
    Calculate Parabolic Stop and Reverse (PSAR).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        af_start (float): Starting acceleration factor
        af_step (float): Acceleration factor step
        af_max (float): Maximum acceleration factor
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of PSAR values
    """
    # Initialize variables
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Initialize series for output
    psar = close.copy()
    bull = True  # Current trend (True for uptrend, False for downtrend)
    af = af_start  # Acceleration factor
    ep = low[0]  # Extreme point
    psar[0] = high[0]  # Start with high for downtrend (it will be reversed)
    
    # Calculate PSAR values
    for i in range(1, len(data)):
        # Determine trend
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            
            # Ensure PSAR is below the lows of the current and previous candle
            psar[i] = min(psar[i], low[i-1], low[i])
            
            # Check if trend reversed
            if high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
            
            # Check for trend reversal
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low[i]
                af = af_start
        else:
            psar[i] = psar[i-1] - af * (psar[i-1] - ep)
            
            # Ensure PSAR is above the highs of the current and previous candle
            psar[i] = max(psar[i], high[i-1], high[i])
            
            # Check if trend reversed
            if low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)
            
            # Check for trend reversal
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high[i]
                af = af_start
    
    if round_values:
        psar = psar.round(2)
    
    return psar

def ichimoku_cloud(data, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26, round_values=True):
    """
    Calculate Ichimoku Cloud components.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        tenkan_period (int): Tenkan-sen (Conversion Line) period
        kijun_period (int): Kijun-sen (Base Line) period
        senkou_b_period (int): Senkou Span B period
        displacement (int): Displacement period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        tuple: (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)
    """
    # Calculate Tenkan-sen (Conversion Line)
    tenkan_high = data['high'].rolling(window=tenkan_period).max()
    tenkan_low = data['low'].rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Calculate Kijun-sen (Base Line)
    kijun_high = data['high'].rolling(window=kijun_period).max()
    kijun_low = data['low'].rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Calculate Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # Calculate Senkou Span B (Leading Span B)
    senkou_b_high = data['high'].rolling(window=senkou_b_period).max()
    senkou_b_low = data['low'].rolling(window=senkou_b_period).min()
    senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
    
    # Calculate Chikou Span (Lagging Span)
    chikou_span = data['close'].shift(-displacement)
    
    if round_values:
        tenkan_sen = tenkan_sen.round(2)
        kijun_sen = kijun_sen.round(2)
        senkou_span_a = senkou_span_a.round(2)
        senkou_span_b = senkou_span_b.round(2)
        chikou_span = chikou_span.round(2)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def triple_exponential_moving_average(data, window, round_values=True):
    """
    Calculate Triple Exponential Moving Average (TEMA).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): TEMA period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of TEMA values
    """
    # Calculate EMA of close
    ema1 = data['close'].ewm(span=window, adjust=False).mean()
    
    # Calculate EMA of EMA
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    
    # Calculate EMA of EMA of EMA
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    
    # Calculate TEMA
    tema = 3 * ema1 - 3 * ema2 + ema3
    
    if round_values:
        tema = tema.round(2)
    
    return tema

def adaptive_moving_average(data, window, round_values=True):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): KAMA period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of KAMA values
    """
    # Calculate the efficiency ratio (ER)
    change = abs(data['close'] - data['close'].shift(window))
    volatility = abs(data['close'] - data['close'].shift(1)).rolling(window=window).sum()
    er = change / volatility
    
    # Replace NaN values and handle division by zero
    er = er.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate smoothing constant (SC)
    fast_sc = 2.0 / (2.0 + 1.0)
    slow_sc = 2.0 / (30.0 + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA with first price
    kama = data['close'].copy()
    
    # Calculate KAMA values
    for i in range(window, len(data)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data['close'].iloc[i] - kama.iloc[i-1])
    
    if round_values:
        kama = kama.round(2)
    
    return kama

def moving_average_convergence(data, fast_window=12, slow_window=26, round_values=True):
    """
    Calculate Moving Average Convergence.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        fast_window (int): Fast EMA period
        slow_window (int): Slow EMA period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of Moving Average Convergence values
    """
    # Calculate EMAs
    fast_ema = exponential_moving_average(data, fast_window, round_values=False)
    slow_ema = exponential_moving_average(data, slow_window, round_values=False)
    
    # Calculate convergence
    convergence = fast_ema - slow_ema
    
    if round_values:
        convergence = convergence.round(2)
    
    return convergence

def moving_average_crossover(data, fast_window=9, slow_window=21, round_values=True):
    """
    Calculate Moving Average Crossover signal.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        fast_window (int): Fast SMA period
        slow_window (int): Slow SMA period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of crossover signals (1: bullish, -1: bearish, 0: neutral)
    """
    # Calculate SMAs
    fast_sma = simple_moving_average(data, fast_window, round_values=False)
    slow_sma = simple_moving_average(data, slow_window, round_values=False)
    
    # Calculate crossover signal
    signal = pd.Series(0, index=data.index)
    
    # Bullish crossover (fast crosses above slow)
    signal[fast_sma > slow_sma] = 1
    
    # Bearish crossover (fast crosses below slow)
    signal[fast_sma < slow_sma] = -1
    
    return signal

def directional_movement_index(data, window=14, round_values=True):
    """
    Calculate Directional Movement Index (DMI).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): DMI period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        tuple: (DI+, DI-, DX)
    """
    # Calculate True Range
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift(1))
    low_close = np.abs(data['low'] - data['close'].shift(1))
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    # Calculate +DM and -DM
    plus_dm = data['high'].diff()
    minus_dm = data['low'].diff().multiply(-1)
    
    # Condition for +DM: if +DM > -DM and +DM > 0, then +DM; otherwise 0
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    
    # Condition for -DM: if -DM > +DM and -DM > 0, then -DM; otherwise 0
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Convert to Series
    plus_dm = pd.Series(plus_dm, index=data.index)
    minus_dm = pd.Series(minus_dm, index=data.index)
    
    # Calculate smoothed values
    smoothed_true_range = true_range.rolling(window=window).sum()
    smoothed_plus_dm = plus_dm.rolling(window=window).sum()
    smoothed_minus_dm = minus_dm.rolling(window=window).sum()
    
    # Calculate +DI and -DI
    plus_di = 100 * smoothed_plus_dm / smoothed_true_range
    minus_di = 100 * smoothed_minus_dm / smoothed_true_range
    
    # Handle division by zero
    plus_di = plus_di.replace([np.inf, -np.inf], np.nan).fillna(0)
    minus_di = minus_di.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Handle division by zero
    dx = dx.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if round_values:
        plus_di = plus_di.round(2)
        minus_di = minus_di.round(2)
        dx = dx.round(2)
    
    return plus_di, minus_di, dx
