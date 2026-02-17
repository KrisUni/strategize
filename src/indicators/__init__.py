"""
Indicators Module
=================
All technical indicator calculations.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    return series.rolling(window=length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1/length, adjust=False).mean()

def ma(series: pd.Series, length: int, ma_type: str = 'sma') -> pd.Series:
    ma_funcs = {'sma': sma, 'ema': ema, 'wma': wma, 'rma': rma}
    return ma_funcs.get(ma_type.lower(), sma)(series, length)

def pamrp(high, low, close, length):
    highest = high.rolling(window=length).max()
    lowest = low.rolling(window=length).min()
    range_val = highest - lowest
    pamrp_val = np.where(range_val > 0, (close - lowest) / range_val * 100, 50.0)
    return pd.Series(pamrp_val, index=close.index)

def bollinger_width(close, length):
    basis = sma(close, length)
    std = close.rolling(window=length).std()
    upper = basis + std
    lower = basis - std
    width = np.where(basis > 0, (upper - lower) / basis * 100, 0)
    return pd.Series(width, index=close.index)

def percentile_rank(series, lookback):
    """
    Percentile rank over rolling lookback window.

    FIXED: Returns NaN for bars before the lookback period.
    Previously returned 0.0, which passed 'bbwp < threshold' filters
    during warmup, generating false entry signals.
    """
    result = np.full(len(series), np.nan)  # NaN during warmup, not 0
    for i in range(lookback, len(series)):
        current = series.iloc[i]
        historical = series.iloc[max(0, i-lookback):i]
        if len(historical) > 0:
            result[i] = (historical < current).sum() / len(historical) * 100
        else:
            result[i] = np.nan
    return pd.Series(result, index=series.index)

def bbwp(close, length, lookback):
    width = bollinger_width(close, length)
    return percentile_rank(width, lookback)

def rsi(close, length):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def stoch_rsi(close, rsi_length, k_length, d_length):
    rsi_val = rsi(close, rsi_length)
    lowest_rsi = rsi_val.rolling(window=rsi_length).min()
    highest_rsi = rsi_val.rolling(window=rsi_length).max()
    rsi_range = highest_rsi - lowest_rsi
    stoch = np.where(rsi_range > 0, (rsi_val - lowest_rsi) / rsi_range * 100, 50)
    stoch = pd.Series(stoch, index=close.index)
    k = sma(stoch, k_length)
    d = sma(k, d_length)
    return k, d

def adx(high, low, close, length, smoothing):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    atr_val = rma(tr, length)
    plus_di = 100 * rma(plus_dm, length) / atr_val
    minus_di = 100 * rma(minus_dm, length) / atr_val
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx_val = rma(dx.fillna(0), smoothing)
    return plus_di, minus_di, adx_val

def atr(high, low, close, length):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return rma(tr, length)

def supertrend(high, low, close, period, multiplier):
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
            direction[i] = -1
        elif close.iloc[i] < supertrend_val[i-1]:
            direction[i] = 1
        else:
            direction[i] = direction[i-1]
        if direction[i] == -1:
            supertrend_val[i] = lower_band.iloc[i]
            if supertrend_val[i] < supertrend_val[i-1] and direction[i-1] == -1:
                supertrend_val[i] = supertrend_val[i-1]
        else:
            supertrend_val[i] = upper_band.iloc[i]
            if supertrend_val[i] > supertrend_val[i-1] and direction[i-1] == 1:
                supertrend_val[i] = supertrend_val[i-1]
    return pd.Series(supertrend_val, index=close.index), pd.Series(direction, index=close.index)

def vwap(high, low, close, volume):
    """
    Calculate VWAP with daily session reset.

    FIXED: Previously accumulated from bar 0 forever, making VWAP
    meaningless after the first day on intraday data. Now resets
    at each new calendar date (session boundary).

    For daily data, each bar is its own session so VWAP = typical price.
    """
    typical_price = (high + low + close) / 3

    # Group by calendar date so cumsum resets each session
    dates = close.index.date
    cum_tpv = (typical_price * volume).groupby(dates).cumsum()
    cum_vol = volume.groupby(dates).cumsum()

    return cum_tpv / cum_vol

def macd(close, fast, slow, signal):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram