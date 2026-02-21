"""
Data Module - v2
================
Fixes from v1 audit:

- FIX #1: Intraday data no longer silently ignores user start/end.
  Previously used hardcoded periods ('7d', '60d', '730d') and discarded
  the user's date range entirely. Now respects start/end where the API
  allows, and clamps + warns clearly where yfinance has hard limits.

- FIX #2: Fallback logic replaced.
  The old fallback (try period first, then start/end on empty) was never
  reached for the common case because yfinance returns empty silently.
  Replaced with explicit limit-awareness and a clear error message that
  states the actual API limit for the chosen interval.

- FIX #3: Duplicate timestamps removed.
  yfinance intraday fetches occasionally return duplicate index entries.
  Now deduplicated (keep last) after fetch.

- FIX #4: Index guaranteed sorted ascending.
  yfinance does not guarantee sort order. Now explicit.

- FIX #5: Timezone handling corrected.
  Was using tz_localize(None) on a timezone-aware index, which raises
  in newer pandas. Corrected to tz_convert('UTC').tz_localize(None).

- FIX #6: auto_adjust and actions now explicit.
  yfinance defaults may change between versions. Now always fetches
  split/dividend-adjusted prices (correct for backtesting) and
  suppresses the Dividends/Stock Splits columns at source.

- FIX #7: Post-fetch data validation.
  Checks for: positive prices, high >= low, high >= open/close,
  low <= open/close. Raises ValueError with a clear message on failure.

- FIX #8: generate_sample_data weekend filter fixed.
  Dead-code first branch always fell through to the freq='B' fallback.
  Simplified to use freq='B' directly. Now always returns exactly n bars.

- FIX #9: generate_sample_data volume correlated with volatility.
  Previous uniform random volume made volume-based indicators meaningless
  on sample data. Volume now scales with daily return magnitude plus noise,
  producing realistic volume-volatility correlation.

- FIX #10: generate_sample_data start_date configurable.
  Hardcoded '2020-01-01' replaced with a parameter. Defaults to a fixed
  date for backward compatibility with existing tests.

New utility:
  validate_ohlcv(df) — standalone validation, usable from tests.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# yfinance API limits
# ─────────────────────────────────────────────────────────────────────────────

# Maximum lookback in calendar days per interval, enforced by Yahoo Finance API.
# Sources: yfinance docs + empirical testing.
_YFINANCE_MAX_DAYS: dict = {
    '1m':  7,
    '2m':  60,
    '5m':  60,
    '15m': 60,
    '30m': 60,
    '60m': 730,
    '90m': 60,
    '1h':  730,
    '1d':  None,   # No practical limit
    '5d':  None,
    '1wk': None,
    '1mo': None,
    '3mo': None,
}

_VALID_INTERVALS = list(_YFINANCE_MAX_DAYS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_ohlcv(df: pd.DataFrame) -> None:
    """
    Validate OHLCV DataFrame integrity.

    Checks
    ------
    - Required columns present: open, high, low, close
    - All prices strictly positive
    - high >= open, high >= close (high is the highest)
    - low  <= open, low  <= close (low is the lowest)
    - high >= low
    - Index is sorted ascending

    Raises
    ------
    ValueError with a descriptive message on first failure.
    Volume column is optional — its absence does not raise.
    """
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV validation failed: missing columns {missing}")

    if len(df) == 0:
        raise ValueError("OHLCV validation failed: DataFrame is empty")

    # Positive prices
    for col in required:
        if (df[col] <= 0).any():
            n_bad = int((df[col] <= 0).sum())
            raise ValueError(
                f"OHLCV validation failed: '{col}' has {n_bad} non-positive value(s). "
                f"Check for delisted symbols or bad data."
            )

    # OHLC consistency
    if (df['high'] < df['low']).any():
        n_bad = int((df['high'] < df['low']).sum())
        raise ValueError(
            f"OHLCV validation failed: high < low on {n_bad} bar(s). Data is corrupted."
        )
    if (df['high'] < df['open']).any():
        n_bad = int((df['high'] < df['open']).sum())
        raise ValueError(
            f"OHLCV validation failed: high < open on {n_bad} bar(s)."
        )
    if (df['high'] < df['close']).any():
        n_bad = int((df['high'] < df['close']).sum())
        raise ValueError(
            f"OHLCV validation failed: high < close on {n_bad} bar(s)."
        )
    if (df['low'] > df['open']).any():
        n_bad = int((df['low'] > df['open']).sum())
        raise ValueError(
            f"OHLCV validation failed: low > open on {n_bad} bar(s)."
        )
    if (df['low'] > df['close']).any():
        n_bad = int((df['low'] > df['close']).sum())
        raise ValueError(
            f"OHLCV validation failed: low > close on {n_bad} bar(s)."
        )

    # Sorted index
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "OHLCV validation failed: index is not sorted ascending. "
            "This will produce incorrect backtest results."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fetch from Yahoo Finance
# ─────────────────────────────────────────────────────────────────────────────

def _clamp_date_range(
    start: str,
    end: str,
    interval: str,
) -> Tuple[str, str, bool]:
    """
    Clamp the requested date range to yfinance API limits for the given interval.

    Returns
    -------
    (clamped_start, end, was_clamped)
    was_clamped is True if start was moved forward to fit within the API limit.
    """
    max_days = _YFINANCE_MAX_DAYS.get(interval)
    if max_days is None:
        return start, end, False

    end_dt = datetime.strptime(end, '%Y-%m-%d')
    earliest_allowed = end_dt - timedelta(days=max_days)
    requested_start = datetime.strptime(start, '%Y-%m-%d')

    if requested_start < earliest_allowed:
        clamped = earliest_allowed.strftime('%Y-%m-%d')
        return clamped, end, True

    return start, end, False


def fetch_yfinance(
    symbol: str,
    start: str,
    end: str,
    interval: str = '1d',
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. 'SPY', 'BTC-USD', 'AAPL'.
    start : str
        Start date 'YYYY-MM-DD'. For intraday intervals this may be clamped
        to the API maximum — a warning is logged and surfaced in the return
        metadata.
    end : str
        End date 'YYYY-MM-DD'.
    interval : str
        Bar size. Valid: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk,
        1mo, 3mo.

        yfinance API limits (hard limits enforced by Yahoo):
          1m  → last 7 calendar days only
          2m, 5m, 15m, 30m, 90m → last 60 calendar days
          60m, 1h → last 730 calendar days
          1d and above → full history

        When start is beyond the limit, it is clamped and a warning is raised.
        The returned DataFrame's metadata attribute (df.attrs) includes
        'date_range_clamped': bool and 'actual_start': str.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close[, volume]
        Index: DatetimeIndex, timezone-naive UTC, sorted ascending, no duplicates.

    Raises
    ------
    ValueError
        If interval is invalid, no data is returned, or OHLCV validation fails.
    ImportError
        If yfinance is not installed.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")

    # Validate interval
    if interval not in _VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. "
            f"Must be one of: {', '.join(_VALID_INTERVALS)}"
        )

    # FIX #1/#2: Clamp date range to API limits, don't silently discard dates
    clamped_start, end, was_clamped = _clamp_date_range(start, end, interval)
    max_days = _YFINANCE_MAX_DAYS.get(interval)

    if was_clamped:
        logger.warning(
            "Date range clamped for interval '%s': Yahoo Finance only provides "
            "the last %d calendar days for this interval. "
            "Requested start: %s → Actual start: %s",
            interval, max_days, start, clamped_start,
        )

    try:
        ticker = yf.Ticker(symbol)

        # FIX #6: Explicit auto_adjust and actions — never rely on defaults
        df = ticker.history(
            start=clamped_start,
            end=end,
            interval=interval,
            auto_adjust=True,   # Split/dividend-adjusted prices (correct for backtesting)
            actions=False,       # Don't fetch Dividends/Stock Splits columns
        )

    except Exception as e:
        raise ValueError(f"yfinance error fetching '{symbol}': {e}") from e

    if df is None or df.empty:
        limit_note = (
            f" Note: for '{interval}' data, Yahoo Finance only provides "
            f"the last {max_days} calendar days."
            if max_days else ""
        )
        raise ValueError(
            f"No data returned for '{symbol}' ({interval}, {clamped_start} → {end})."
            f"{limit_note}"
        )

    # FIX #5: Correct timezone strip — tz_localize(None) raises on aware index
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)

    # Normalize column names (handles yfinance version differences)
    df.columns = df.columns.str.lower()

    # Keep only OHLCV
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in required_cols if c in df.columns]]

    # FIX #4: Guarantee sorted ascending
    df = df.sort_index()

    # FIX #3: Remove duplicate timestamps (keep last — most recent tick wins)
    n_before = len(df)
    df = df[~df.index.duplicated(keep='last')]
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.warning(
            "%d duplicate timestamp(s) removed from '%s' (%s) data.",
            n_dupes, symbol, interval,
        )

    # Drop NaN rows
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    # FIX #7: Validate OHLCV integrity
    validate_ohlcv(df)

    # Attach metadata so the caller (app.py) can surface warnings to the user
    df.attrs['symbol'] = symbol
    df.attrs['interval'] = interval
    df.attrs['actual_start'] = str(df.index[0].date())
    df.attrs['actual_end'] = str(df.index[-1].date())
    df.attrs['requested_start'] = start
    df.attrs['date_range_clamped'] = was_clamped
    df.attrs['max_interval_days'] = max_days

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Load from CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.

    Expected format: date column (any name, parseable as datetime) + ohlcv columns.
    Column names are normalized to lowercase.

    Raises ValueError if the file cannot be parsed or OHLCV validation fails.
    """
    try:
        df = pd.read_csv(filepath, parse_dates=[0], index_col=0)
    except Exception as e:
        raise ValueError(f"Could not read CSV '{filepath}': {e}") from e

    df.columns = df.columns.str.lower()

    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    validate_ohlcv(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic sample data
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_data(
    days: int = 500,
    volatility: float = 0.015,
    drift: float = 0.0002,
    seed: int = 42,
    start_date: str = '2020-01-01',
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV price data for testing and development.

    The generated series has:
    - Geometric Brownian Motion close prices with mild mean-reversion overlay
    - Realistic intrabar ranges (high/low derived from close volatility)
    - Volume correlated with price volatility (high-vol bars = higher volume)
    - Exactly `days` trading days (Monday–Friday, no weekends)
    - OHLC consistency guaranteed: high = max(O,C,H), low = min(O,C,L)

    Parameters
    ----------
    days : int
        Number of trading bars to generate.
    volatility : float
        Daily return standard deviation (e.g. 0.015 = 1.5%).
    drift : float
        Daily expected return (e.g. 0.0002 ≈ 5% annual).
    seed : int
        Random seed for reproducibility.
    start_date : str
        Start date 'YYYY-MM-DD'. Defaults to '2020-01-01'.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume.
        Index: DatetimeIndex of business days, sorted ascending.
    """
    rng = np.random.default_rng(seed)

    n = days

    # ── Close prices: GBM with mild mean-reversion ────────────────────────────
    returns = rng.standard_normal(n) * volatility + drift

    # Mean-reversion: if recent cumulative return is large, push back gently
    for i in range(20, n):
        recent = returns[i - 20:i].sum()
        if abs(recent) > volatility * 5:
            returns[i] -= np.sign(recent) * volatility * 0.3

    close = 100.0 * np.exp(np.cumsum(returns))

    # ── Open: previous close ± small gap ─────────────────────────────────────
    open_price = np.empty(n)
    open_price[0] = close[0]
    open_price[1:] = close[:-1] * (1.0 + rng.standard_normal(n - 1) * 0.002)

    # ── Intrabar range ────────────────────────────────────────────────────────
    daily_range = np.abs(rng.standard_normal(n)) * volatility * 0.5 + volatility * 0.3
    high_raw = close * (1.0 + daily_range)
    low_raw  = close * (1.0 - daily_range)

    # OHLC consistency: high must contain open and close; low must contain both
    high = np.maximum(high_raw, np.maximum(open_price, close))
    low  = np.minimum(low_raw,  np.minimum(open_price, close))

    # ── FIX #9: Volume correlated with volatility ─────────────────────────────
    # Base volume: ~2M shares/day
    # Multiplier: higher on high-volatility bars (|return| drives up volume)
    abs_returns = np.abs(returns)
    vol_z = (abs_returns - abs_returns.mean()) / (abs_returns.std() + 1e-9)
    vol_multiplier = np.exp(vol_z * 0.5)   # Exponential scaling so spikes are visible
    base_volume = rng.integers(1_000_000, 3_000_000, n).astype(float)
    volume = (base_volume * vol_multiplier).astype(int)
    volume = np.clip(volume, 100_000, 20_000_000)

    # ── FIX #8: Business-day index — exactly n bars, no dead code ────────────
    # freq='B' (business days) directly gives weekday-only dates.
    # The previous approach generated calendar days then filtered weekends,
    # always producing fewer than n dates and falling through to this same call.
    dates = pd.bdate_range(start=start_date, periods=n)

    df = pd.DataFrame({
        'open':   open_price,
        'high':   high,
        'low':    low,
        'close':  close,
        'volume': volume,
    }, index=dates)

    # Paranoia: validate our own synthetic data
    validate_ohlcv(df)

    df.attrs['symbol'] = 'SAMPLE'
    df.attrs['interval'] = '1d'
    df.attrs['actual_start'] = str(df.index[0].date())
    df.attrs['actual_end'] = str(df.index[-1].date())
    df.attrs['date_range_clamped'] = False

    return df