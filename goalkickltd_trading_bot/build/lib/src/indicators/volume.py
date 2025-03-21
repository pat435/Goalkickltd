"""
Volume indicators for the Goalkick Ltd Trading Bot.
Implements volume-based technical indicators like OBV, A/D, VWAP, etc.
"""

import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger("indicators.volume")

def on_balance_volume(data, round_values=True):
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        data (pd.DataFrame): DataFrame with price and volume data
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of OBV values
    """
    # Calculate price changes
    price_change = data['close'].diff()
    
    # Initialize OBV
    obv = pd.Series(0, index=data.index)
    
    # Calculate OBV values
    for i in range(1, len(data)):
        if price_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
        elif price_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    if round_values:
        obv = obv.round(0)
    
    return obv

def accumulation_distribution(data, round_values=True):
    """
    Calculate Accumulation/Distribution Line (A/D).
    
    Args:
        data (pd.DataFrame): DataFrame with price and volume data
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of A/D values
    """
    # Calculate Money Flow Multiplier
    high_low = data['high'] - data['low']
    
    # Handle division by zero
    high_low = high_low.replace(0, np.nan)
    
    close_low = data['close'] - data['low']
    high_close = data['high'] - data['close']
    
    mfm = ((close_low - high_close) / high_low).fillna(0)
    
    # Calculate Money Flow Volume
    mfv = mfm * data['volume']
    
    # Calculate A/D
    ad = mfv.cumsum()
    
    if round_values:
        ad = ad.round(0)
    
    return ad

def chaikin_money_flow(data, window=20, round_values=True):
    """
    Calculate Chaikin Money Flow (CMF).
    
    Args:
        data (pd.DataFrame): DataFrame with price and volume data
        window (int): CMF period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of CMF values
    """
    # Calculate Money Flow Multiplier
    high_low = data['high'] - data['low']
    
    # Handle division by zero
    high_low = high_low.replace(0, np.nan)
    
    close_low = data['close'] - data['low']
    high_close = data['high'] - data['close']
    
    mfm = ((close_low - high_close) / high_low).fillna(0)
    
    # Calculate Money Flow Volume
    mfv = mfm * data['volume']
    
    # Calculate CMF
    cmf = mfv.rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
    
    # Handle division by zero
    cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if round_values:
        cmf = cmf.round(2)
    
    return cmf

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
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    
    # Calculate money flow
    money_flow = typical_price * data['volume']
    
    # Calculate price changes
    price_change = typical_price.diff()
    
    # Separate positive and negative money flow
    positive_flow = pd.Series(0, index=data.index)
    negative_flow = pd.Series(0, index=data.index)
    
    # Set initial values
    positive_flow.iloc[1:] = np.where(price_change.iloc[1:] > 0, money_flow.iloc[1:], 0)
    negative_flow.iloc[1:] = np.where(price_change.iloc[1:] < 0, money_flow.iloc[1:], 0)
    
    # Calculate sums
    positive_sum = positive_flow.rolling(window=window).sum()
    negative_sum = negative_flow.rolling(window=window).sum()
    
    # Calculate money ratio
    # Handle division by zero
    money_ratio = np.where(negative_sum != 0, positive_sum / negative_sum, 0)
    
    # Calculate MFI
    mfi = 100 - (100 / (1 + money_ratio))
    
    # Convert to Series
    mfi = pd.Series(mfi, index=data.index)
    
    if round_values:
        mfi = mfi.round(2)
    
    return mfi

def volume_weighted_average_price(data, window=14, round_values=True):
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Args:
        data (pd.DataFrame): DataFrame with price and volume data
        window (int): VWAP period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of VWAP values
    """
    # Calculate typical price
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    
    # Calculate VWAP
    vwap = (typical_price * data['volume']).rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
    
    # Handle division by zero
    vwap = vwap.replace([np.inf, -np.inf], np.nan).fillna(typical_price)
    
    if round_values:
        vwap = vwap.round(2)
    
    return vwap

def price_volume_trend(data, round_values=True):
    """
    Calculate Price Volume Trend (PVT).
    
    Args:
        data (pd.DataFrame): DataFrame with price and volume data
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of PVT values
    """
    # Calculate percentage price change
    pct_change = data['close'].pct_change()
    
    # Calculate PVT
    pvt = (pct_change * data['volume']).cumsum()
    
    # Handle NaN values
    pvt = pvt.fillna(0)
    
    if round_values:
        pvt = pvt.round(2)
    
    return pvt

def negative_volume_index(data, round_values=True):
    """
    Calculate Negative Volume Index (NVI).
    
    Args:
        data (pd.DataFrame): DataFrame with price and volume data
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of NVI values
    """
    # Calculate percentage price change
    pct_change = data['close'].pct_change()
    
    # Calculate volume change
    volume_change = data['volume'].pct_change()
    
    # Initialize NVI
    nvi = pd.Series(1000, index=data.index)
    
    # Calculate NVI values
    for i in range(1, len(data)):
        if volume_change.iloc[i] < 0:
            nvi.iloc[i] = nvi.iloc[i-1] * (1 + pct_change.iloc[i])
        else:
            nvi.iloc[i] = nvi.iloc[i-1]
    
    if round_values:
        nvi = nvi.round(2)
    
    return nvi

def positive_volume_index(data, round_values=True):
    """
    Calculate Positive Volume Index (PVI).
    
    Args:
        data (pd.DataFrame): DataFrame with price and volume data
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of PVI values
    """
    # Calculate percentage price change
    pct_change = data['close'].pct_change()
    
    # Calculate volume change
    volume_change = data['volume'].pct_change()
    
    # Initialize PVI
    pvi = pd.Series(1000, index=data.index)
    
    # Calculate PVI values
    for i in range(1, len(data)):
        if volume_change.iloc[i] > 0:
            pvi.iloc[i] = pvi.iloc[i-1] * (1 + pct_change.iloc[i])
        else:
            pvi.iloc[i] = pvi.iloc[i-1]
    
    if round_values:
        pvi = pvi.round(2)
    
    return pvi

def volume_oscillator(data, short_window=5, long_window=10, round_values=True):
    """
    Calculate Volume Oscillator.
    
    Args:
        data (pd.DataFrame): DataFrame with volume data
        short_window (int): Short MA period
        long_window (int): Long MA period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of Volume Oscillator values
    """
    # Calculate volume MAs
    short_ma = data['volume'].rolling(window=short_window).mean()
    long_ma = data['volume'].rolling(window=long_window).mean()
    
    # Calculate volume oscillator
    vol_osc = ((short_ma - long_ma) / long_ma) * 100
    
    # Handle division by zero
    vol_osc = vol_osc.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if round_values:
        vol_osc = vol_osc.round(2)
    
    return vol_osc

def ease_of_movement(data, window=14, round_values=True):
    """
    Calculate Ease of Movement (EOM).
    
    Args:
        data (pd.DataFrame): DataFrame with price and volume data
        window (int): EOM period
        round_values (bool): Whether to round values to 2 decimal places
        
    Returns:
        pd.Series: Series of EOM values
    """
    # Calculate the high-low difference
    hl_diff = data['high'] - data['low']
    
    # Calculate the yesterday-today high-low midpoints difference
    high_low_midpoint = (data['high'] + data['low']) / 2
    high_low_midpoint_diff = high_low_midpoint.diff()
    
    # Calculate box ratio (volume / high-low difference)
    box_ratio = data['volume'] / hl_diff
    
    # Handle division by zero
    box_ratio = box_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate single-period EOM
    eom_single = high_low_midpoint_diff / box_ratio
    
    # Calculate EOM
    eom = eom_single.rolling(window=window).mean()
    
    if round_values:
        eom = eom.round(6)  # EOM values are typically very small
    
    return eom