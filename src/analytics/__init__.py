"""
Calendar Analytics Module — v2 (Fixed)
=======================================
Fixes from audit:
- CRITICAL: DOW and DOM stats now resample to daily before computing returns
  (previously, intraday data produced per-bar returns grouped by day name,
   which is statistically meaningless)
- Monthly stats already used resample('ME').last() — correct, unchanged
- Added hourly stats integration into main analyze_calendar

All statistics computed on the raw price data (not strategy returns),
giving the user a statistical edge map of the instrument itself.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
MONTH_NAMES = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]


@dataclass
class DayOfWeekStats:
    """Statistics for a single day of the week."""
    day_name: str
    count: int
    mean_return_pct: float
    median_return_pct: float
    std_return_pct: float
    positive_pct: float
    avg_positive_pct: float
    avg_negative_pct: float
    best_pct: float
    worst_pct: float
    total_return_pct: float


@dataclass
class MonthStats:
    """Statistics for a single month."""
    month_name: str
    count: int
    mean_return_pct: float
    median_return_pct: float
    positive_pct: float
    best_pct: float
    worst_pct: float


@dataclass
class CalendarAnalysis:
    """Complete calendar analysis result."""
    day_of_week: List[DayOfWeekStats]
    day_of_week_df: pd.DataFrame
    monthly: List[MonthStats]
    monthly_df: pd.DataFrame
    monthly_heatmap: pd.DataFrame
    day_of_month_df: pd.DataFrame
    hourly_df: Optional[pd.DataFrame]
    symbol: str
    start_date: str
    end_date: str
    total_bars: int
    is_intraday: bool


@dataclass
class TradeCalendarAnalysis:
    """Calendar analysis of trade performance (strategy-level)."""
    trades_by_day: pd.DataFrame
    trades_by_month: pd.DataFrame


def _is_intraday(df: pd.DataFrame) -> bool:
    """Detect whether data is intraday based on median bar interval."""
    if len(df) < 2:
        return False
    deltas = pd.Series(df.index).diff().dropna()
    median_seconds = deltas.median().total_seconds()
    return median_seconds < 86400


def _resample_to_daily(df: pd.DataFrame) -> pd.Series:
    """
    Convert any-frequency OHLCV data to daily close returns.

    For daily data: just pct_change on close.
    For intraday data: resample to daily (last close of each day),
    then compute daily returns. This gives true daily returns even
    from 1m/5m/1h bars.

    Returns:
        Series of daily returns in percent, with DatetimeIndex.
    """
    if _is_intraday(df):
        daily_close = df['close'].resample('D').last().dropna()
    else:
        daily_close = df['close']

    returns = daily_close.pct_change().dropna() * 100
    return returns


def compute_day_of_week_stats(df: pd.DataFrame) -> Tuple[List[DayOfWeekStats], pd.DataFrame]:
    """
    Compute return statistics for each day of the week.

    FIXED: Always uses daily returns regardless of input bar frequency.
    For 1h/5m/1m data, resamples to daily first.
    """
    returns = _resample_to_daily(df)
    dow = returns.index.dayofweek

    stats_list = []
    rows = []

    for day_idx in range(5):
        day_returns = returns[dow == day_idx]
        if len(day_returns) == 0:
            continue

        positive = day_returns[day_returns > 0]
        negative = day_returns[day_returns <= 0]

        s = DayOfWeekStats(
            day_name=DAY_NAMES[day_idx],
            count=len(day_returns),
            mean_return_pct=float(day_returns.mean()),
            median_return_pct=float(day_returns.median()),
            std_return_pct=float(day_returns.std()),
            positive_pct=float(len(positive) / len(day_returns) * 100),
            avg_positive_pct=float(positive.mean()) if len(positive) > 0 else 0.0,
            avg_negative_pct=float(negative.mean()) if len(negative) > 0 else 0.0,
            best_pct=float(day_returns.max()),
            worst_pct=float(day_returns.min()),
            total_return_pct=float(day_returns.sum()),
        )
        stats_list.append(s)
        rows.append({
            'Day': s.day_name, 'Count': s.count,
            'Avg %': round(s.mean_return_pct, 4),
            'Median %': round(s.median_return_pct, 4),
            'Std %': round(s.std_return_pct, 4),
            'Win %': round(s.positive_pct, 1),
            'Avg Up %': round(s.avg_positive_pct, 4),
            'Avg Down %': round(s.avg_negative_pct, 4),
            'Best %': round(s.best_pct, 3),
            'Worst %': round(s.worst_pct, 3),
            'Total %': round(s.total_return_pct, 3),
        })

    summary_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return stats_list, summary_df


def compute_monthly_stats(df: pd.DataFrame) -> Tuple[List[MonthStats], pd.DataFrame]:
    """Monthly returns — resample('ME').last() is correct for any frequency."""
    monthly_close = df['close'].resample('ME').last().dropna()
    monthly_returns = monthly_close.pct_change().dropna() * 100
    month_nums = monthly_returns.index.month

    stats_list = []
    rows = []

    for m in range(1, 13):
        m_rets = monthly_returns[month_nums == m]
        if len(m_rets) == 0:
            continue
        positive = m_rets[m_rets > 0]
        s = MonthStats(
            month_name=MONTH_NAMES[m - 1], count=len(m_rets),
            mean_return_pct=float(m_rets.mean()),
            median_return_pct=float(m_rets.median()),
            positive_pct=float(len(positive) / len(m_rets) * 100),
            best_pct=float(m_rets.max()), worst_pct=float(m_rets.min()),
        )
        stats_list.append(s)
        rows.append({
            'Month': s.month_name, 'Count': s.count,
            'Avg %': round(s.mean_return_pct, 3),
            'Median %': round(s.median_return_pct, 3),
            'Win %': round(s.positive_pct, 1),
            'Best %': round(s.best_pct, 2), 'Worst %': round(s.worst_pct, 2),
        })

    return stats_list, pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_monthly_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Year x Month heatmap of returns."""
    monthly_close = df['close'].resample('ME').last().dropna()
    monthly_returns = monthly_close.pct_change().dropna() * 100

    heatmap_data = {}
    for date, ret in monthly_returns.items():
        year = date.year
        month = date.month
        if year not in heatmap_data:
            heatmap_data[year] = {}
        heatmap_data[year][month] = ret

    if not heatmap_data:
        return pd.DataFrame()

    heatmap_df = pd.DataFrame(heatmap_data).T
    heatmap_df.columns = [MONTH_NAMES[m - 1] for m in sorted(heatmap_df.columns)]
    heatmap_df = heatmap_df.sort_index()
    return heatmap_df


def compute_day_of_month_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average return by day of month (1-31).
    FIXED: Uses daily returns regardless of input frequency.
    """
    returns = _resample_to_daily(df)
    dom = returns.index.day

    rows = []
    for d in range(1, 32):
        d_rets = returns[dom == d]
        if len(d_rets) == 0:
            continue
        rows.append({
            'Day': d, 'Count': len(d_rets),
            'Avg %': round(float(d_rets.mean()), 4),
            'Win %': round(float((d_rets > 0).sum() / len(d_rets) * 100), 1),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_hourly_stats(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Average return by hour (intraday only). Uses per-bar returns (correct)."""
    if not _is_intraday(df):
        return None

    returns = df['close'].pct_change().dropna() * 100
    hours = returns.index.hour

    rows = []
    for h in sorted(hours.unique()):
        h_rets = returns[hours == h]
        if len(h_rets) == 0:
            continue
        rows.append({
            'Hour': f"{h:02d}:00", 'Count': len(h_rets),
            'Avg %': round(float(h_rets.mean()), 4),
            'Win %': round(float((h_rets > 0).sum() / len(h_rets) * 100), 1),
            'Std %': round(float(h_rets.std()), 4),
        })
    return pd.DataFrame(rows) if rows else None


def analyze_calendar(df: pd.DataFrame, symbol: str = '') -> CalendarAnalysis:
    """Full calendar analysis — handles daily and intraday correctly."""
    intraday = _is_intraday(df)

    dow_stats, dow_df = compute_day_of_week_stats(df)
    monthly_stats, monthly_df = compute_monthly_stats(df)
    heatmap_df = compute_monthly_heatmap(df)
    dom_df = compute_day_of_month_stats(df)
    hourly_df = compute_hourly_stats(df) if intraday else None

    return CalendarAnalysis(
        day_of_week=dow_stats, day_of_week_df=dow_df,
        monthly=monthly_stats, monthly_df=monthly_df,
        monthly_heatmap=heatmap_df, day_of_month_df=dom_df,
        hourly_df=hourly_df, symbol=symbol,
        start_date=str(df.index[0].date()) if len(df) > 0 else '',
        end_date=str(df.index[-1].date()) if len(df) > 0 else '',
        total_bars=len(df), is_intraday=intraday,
    )


def analyze_trade_calendar(trades: List) -> TradeCalendarAnalysis:
    """Analyze trade performance by entry day-of-week and month."""
    if not trades:
        return TradeCalendarAnalysis(trades_by_day=pd.DataFrame(), trades_by_month=pd.DataFrame())

    trade_rows = []
    for t in trades:
        if t.entry_date is not None:
            trade_rows.append({
                'entry_date': t.entry_date, 'dow': t.entry_date.dayofweek,
                'month': t.entry_date.month, 'pnl': t.pnl,
                'pnl_pct': t.pnl_pct, 'direction': t.direction,
                'winner': 1 if t.pnl > 0 else 0,
            })

    if not trade_rows:
        return TradeCalendarAnalysis(trades_by_day=pd.DataFrame(), trades_by_month=pd.DataFrame())

    tdf = pd.DataFrame(trade_rows)

    day_rows = []
    for d in range(5):
        subset = tdf[tdf['dow'] == d]
        if len(subset) == 0:
            continue
        day_rows.append({
            'Day': DAY_NAMES[d], 'Trades': len(subset),
            'Win %': round(subset['winner'].mean() * 100, 1),
            'Avg PnL $': round(subset['pnl'].mean(), 2),
            'Avg PnL %': round(subset['pnl_pct'].mean(), 3),
            'Total PnL $': round(subset['pnl'].sum(), 2),
        })

    month_rows = []
    for m in range(1, 13):
        subset = tdf[tdf['month'] == m]
        if len(subset) == 0:
            continue
        month_rows.append({
            'Month': MONTH_NAMES[m - 1], 'Trades': len(subset),
            'Win %': round(subset['winner'].mean() * 100, 1),
            'Avg PnL $': round(subset['pnl'].mean(), 2),
            'Total PnL $': round(subset['pnl'].sum(), 2),
        })

    return TradeCalendarAnalysis(
        trades_by_day=pd.DataFrame(day_rows) if day_rows else pd.DataFrame(),
        trades_by_month=pd.DataFrame(month_rows) if month_rows else pd.DataFrame(),
    )