"""
Indicators Module
=================
All technical indicator calculations.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, length + 1)
    return series.rolling(window=length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def rma(series: pd.Series, length: int) -> pd.Series:
    """Running Moving Average (Wilder's smoothing)"""
    return series.ewm(alpha=1/length, adjust=False).mean()


def ma(series: pd.Series, length: int, ma_type: str = 'sma') -> pd.Series:
    """Generic Moving Average"""
    ma_funcs = {'sma': sma, 'ema': ema, 'wma': wma, 'rma': rma}
    return ma_funcs.get(ma_type.lower(), sma)(series, length)


# ═══════════════════════════════════════════════════════════════════════════════
# PAMRP (Price Action Mean Reversion Percentile)
# ═══════════════════════════════════════════════════════════════════════════════

def pamrp(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """
    Calculate PAMRP - where price is within its recent range.
    
    0 = at the low of the range (oversold)
    100 = at the high of the range (overbought)
    """
    highest = high.rolling(window=length).max()
    lowest = low.rolling(window=length).min()
    range_val = highest - lowest
    
    pamrp_val = np.where(
        range_val > 0,
        (close - lowest) / range_val * 100,
        50.0
    )
    
    return pd.Series(pamrp_val, index=close.index)


# ═══════════════════════════════════════════════════════════════════════════════
# BBWP (Bollinger Band Width Percentile)
# ═══════════════════════════════════════════════════════════════════════════════

def bollinger_width(close: pd.Series, length: int) -> pd.Series:
    """Calculate Bollinger Band Width"""
    basis = sma(close, length)
    std = close.rolling(window=length).std()
    upper = basis + std
    lower = basis - std
    
    width = np.where(basis > 0, (upper - lower) / basis * 100, 0)
    return pd.Series(width, index=close.index)


def percentile_rank(series: pd.Series, lookback: int) -> pd.Series:
    """Calculate percentile rank over lookback period"""
    result = np.zeros(len(series))
    
    for i in range(lookback, len(series)):
        current = series.iloc[i]
        historical = series.iloc[max(0, i-lookback):i]
        if len(historical) > 0:
            result[i] = (historical < current).sum() / len(historical) * 100
        else:
            result[i] = 50.0
    
    return pd.Series(result, index=series.index)


def bbwp(close: pd.Series, length: int, lookback: int) -> pd.Series:
    """
    Calculate BBWP - volatility percentile.
    
    Low values = volatility compression
    High values = volatility expansion
    """
    width = bollinger_width(close, length)
    return percentile_rank(width, lookback)


# ═══════════════════════════════════════════════════════════════════════════════
# RSI (Relative Strength Index)
# ═══════════════════════════════════════════════════════════════════════════════

def rsi(close: pd.Series, length: int) -> pd.Series:
    """Calculate RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    
    return rsi_val.fillna(50)


# ═══════════════════════════════════════════════════════════════════════════════
# Stochastic RSI
# ═══════════════════════════════════════════════════════════════════════════════

def stoch_rsi(close: pd.Series, rsi_length: int, k_length: int, d_length: int) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic RSI.
    
    Returns:
        Tuple of (%K, %D)
    """
    rsi_val = rsi(close, rsi_length)
    
    lowest_rsi = rsi_val.rolling(window=rsi_length).min()
    highest_rsi = rsi_val.rolling(window=rsi_length).max()
    rsi_range = highest_rsi - lowest_rsi
    
    stoch = np.where(rsi_range > 0, (rsi_val - lowest_rsi) / rsi_range * 100, 50)
    stoch = pd.Series(stoch, index=close.index)
    
    k = sma(stoch, k_length)
    d = sma(k, d_length)
    
    return k, d


# ═══════════════════════════════════════════════════════════════════════════════
# ADX (Average Directional Index)
# ═══════════════════════════════════════════════════════════════════════════════

def adx(high: pd.Series, low: pd.Series, close: pd.Series, 
        length: int, smoothing: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX with DI+ and DI-.
    
    Returns:
        Tuple of (DI+, DI-, ADX)
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    # Smoothed values
    atr = rma(tr, length)
    plus_di = 100 * rma(plus_dm, length) / atr
    minus_di = 100 * rma(minus_dm, length) / atr
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx_val = rma(dx.fillna(0), smoothing)
    
    return plus_di, minus_di, adx_val


# ═══════════════════════════════════════════════════════════════════════════════
# ATR (Average True Range)
# ═══════════════════════════════════════════════════════════════════════════════

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return rma(tr, length)


# ═══════════════════════════════════════════════════════════════════════════════
# Supertrend
# ═══════════════════════════════════════════════════════════════════════════════

def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int, multiplier: float) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Supertrend indicator.
    
    Returns:
        Tuple of (supertrend_value, direction)
        direction: -1 = bullish (price above), 1 = bearish (price below)
    """
    atr_val = atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val
    
    n = len(close)
    supertrend_val = np.zeros(n)
    direction = np.zeros(n)
    
    supertrend_val[0] = upper_band.iloc[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close.iloc[i] > supertrend_val[i-1]:
            direction[i] = -1  # Bullish
        elif close.iloc[i] < supertrend_val[i-1]:
            direction[i] = 1   # Bearish
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == -1:  # Bullish
            supertrend_val[i] = lower_band.iloc[i]
            if supertrend_val[i] < supertrend_val[i-1] and direction[i-1] == -1:
                supertrend_val[i] = supertrend_val[i-1]
        else:  # Bearish
            supertrend_val[i] = upper_band.iloc[i]
            if supertrend_val[i] > supertrend_val[i-1] and direction[i-1] == 1:
                supertrend_val[i] = supertrend_val[i-1]
    
    return pd.Series(supertrend_val, index=close.index), pd.Series(direction, index=close.index)


# ═══════════════════════════════════════════════════════════════════════════════
# VWAP (Volume Weighted Average Price)
# ═══════════════════════════════════════════════════════════════════════════════

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate VWAP (cumulative for the session).
    For daily data, this resets each day.
    """
    typical_price = (high + low + close) / 3
    cum_vol = volume.cumsum()
    cum_tp_vol = (typical_price * volume).cumsum()
    
    return cum_tp_vol / cum_vol


# ═══════════════════════════════════════════════════════════════════════════════
# MACD (Moving Average Convergence Divergence)
# ═══════════════════════════════════════════════════════════════════════════════

def macd(close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD.
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram
