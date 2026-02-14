"""
Data Module
===========
Handles data fetching from various sources.
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime


def fetch_yfinance(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance.
    
    Args:
        symbol: Ticker symbol (e.g., 'SPY', 'TSLA', 'BTC-USD')
        start: Start date 'YYYY-MM-DD'
        end: End date 'YYYY-MM-DD'
        interval: Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    
    Returns:
        DataFrame with columns: open, high, low, close, volume
    
    Notes:
        - 1m data: last 7 days only
        - 2m, 5m, 15m, 30m: last 60 days
        - 60m, 90m, 1h: last 730 days
        - 1d and above: full history
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")
    
    # Validate interval
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    if interval not in valid_intervals:
        raise ValueError(f"Invalid interval '{interval}'. Must be one of: {valid_intervals}")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # For intraday, use period instead of start/end for better reliability
        if interval in ['1m', '2m', '5m', '15m', '30m']:
            # Intraday - use period for reliability
            if interval == '1m':
                df = ticker.history(period='7d', interval=interval)
            else:
                df = ticker.history(period='60d', interval=interval)
        elif interval in ['60m', '90m', '1h']:
            df = ticker.history(period='730d', interval=interval)
        else:
            # Daily and above - use date range
            df = ticker.history(start=start, end=end, interval=interval)
        
        if df is None or df.empty:
            # Fallback: try with explicit date range
            df = ticker.history(start=start, end=end, interval=interval)
        
        if df is None or df.empty:
            raise ValueError(f"No data returned for {symbol} with interval {interval}")
        
    except Exception as e:
        raise ValueError(f"Error fetching {symbol}: {str(e)}")
    
    # Normalize column names
    df.columns = df.columns.str.lower()
    
    # Keep only OHLCV
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in required_cols if c in df.columns]]
    
    # Remove timezone info if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Drop any NaN rows
    df = df.dropna()
    
    return df


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Expected format: date,open,high,low,close,volume
    """
    df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
    df.columns = df.columns.str.lower()
    return df


def generate_sample_data(
    days: int = 500,
    volatility: float = 0.015,
    drift: float = 0.0002,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic price data for testing.
    
    Args:
        days: Number of trading days
        volatility: Daily volatility (e.g., 0.015 = 1.5%)
        drift: Daily drift/trend
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)
    
    n = days
    returns = np.random.randn(n) * volatility + drift
    
    # Add mean reversion
    for i in range(20, n):
        recent = returns[i-20:i].sum()
        if abs(recent) > volatility * 5:
            returns[i] -= np.sign(recent) * volatility * 0.3
    
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    daily_range = np.abs(np.random.randn(n)) * volatility * 0.5 + volatility * 0.3
    
    high = close * (1 + daily_range)
    low = close * (1 - daily_range)
    open_price = np.roll(close, 1) * (1 + np.random.randn(n) * 0.002)
    open_price[0] = close[0]
    
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    
    volume = np.random.randint(1000000, 5000000, n)
    
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    dates = dates[dates.dayofweek < 5][:n]  # Remove weekends
    
    if len(dates) < n:
        dates = pd.date_range(start='2020-01-01', periods=n, freq='B')
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates[:n])
    
    return df
