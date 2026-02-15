"""
Calendar Analytics Module
=========================
Analyzes historical return patterns by:
- Day of week (Monday through Friday)
- Month of year
- Day of month
- Intraday hour (for intraday data)
- Trade performance by entry day

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
    positive_pct: float        # % of days that were positive
    avg_positive_pct: float    # Average return on positive days
    avg_negative_pct: float    # Average return on negative days
    best_pct: float
    worst_pct: float
    total_return_pct: float    # Cumulative return


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
    # Day-of-week
    day_of_week: List[DayOfWeekStats]
    day_of_week_df: pd.DataFrame

    # Monthly seasonality
    monthly: List[MonthStats]
    monthly_df: pd.DataFrame

    # Year-over-year monthly returns heatmap data
    monthly_heatmap: pd.DataFrame  # rows=year, cols=month, values=return %

    # Day-of-month average return
    day_of_month_df: pd.DataFrame

    # Symbol and date range
    symbol: str
    start_date: str
    end_date: str
    total_bars: int


@dataclass
class TradeCalendarAnalysis:
    """Calendar analysis of trade performance (strategy-level)."""
    # Trades by entry day-of-week
    trades_by_day: pd.DataFrame    # day_name, count, win_rate, avg_pnl, total_pnl

    # Trades by entry month
    trades_by_month: pd.DataFrame


def compute_day_of_week_stats(df: pd.DataFrame) -> Tuple[List[DayOfWeekStats], pd.DataFrame]:
    """
    Compute return statistics for each day of the week.
    
    Args:
        df: DataFrame with DatetimeIndex and 'close' column
        
    Returns:
        Tuple of (list of DayOfWeekStats, summary DataFrame)
    """
    # Compute daily returns
    returns = df['close'].pct_change().dropna() * 100  # In percent

    # Add day-of-week
    dow = returns.index.dayofweek  # 0=Mon, 4=Fri
    
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
            'Day': s.day_name,
            'Count': s.count,
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
    """
    Compute return statistics for each calendar month.
    
    Uses monthly resampled returns (not daily aggregated).
    """
    # Monthly returns
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
            month_name=MONTH_NAMES[m - 1],
            count=len(m_rets),
            mean_return_pct=float(m_rets.mean()),
            median_return_pct=float(m_rets.median()),
            positive_pct=float(len(positive) / len(m_rets) * 100),
            best_pct=float(m_rets.max()),
            worst_pct=float(m_rets.min()),
        )
        stats_list.append(s)
        rows.append({
            'Month': s.month_name,
            'Count': s.count,
            'Avg %': round(s.mean_return_pct, 3),
            'Median %': round(s.median_return_pct, 3),
            'Win %': round(s.positive_pct, 1),
            'Best %': round(s.best_pct, 2),
            'Worst %': round(s.worst_pct, 2),
        })

    summary_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return stats_list, summary_df


def compute_monthly_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Year x Month heatmap of returns.
    
    Returns DataFrame with years as index, months (1-12) as columns,
    values are return percentages.
    """
    monthly_close = df['close'].resample('ME').last().dropna()
    monthly_returns = monthly_close.pct_change().dropna() * 100

    heatmap_data = {}
    for date, ret in monthly_returns.items():
        year = date.year
        month = date.month
        if year not in heatmap_data:
            heatmap_data[year] = {}
        heatmap_data[year][month] = ret

    heatmap_df = pd.DataFrame(heatmap_data).T
    heatmap_df.columns = [MONTH_NAMES[m - 1] for m in sorted(heatmap_df.columns)]
    heatmap_df = heatmap_df.sort_index()

    return heatmap_df


def compute_day_of_month_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Average return by day of month (1-31)."""
    returns = df['close'].pct_change().dropna() * 100
    dom = returns.index.day

    rows = []
    for d in range(1, 32):
        d_rets = returns[dom == d]
        if len(d_rets) == 0:
            continue
        rows.append({
            'Day': d,
            'Count': len(d_rets),
            'Avg %': round(float(d_rets.mean()), 4),
            'Win %': round(float((d_rets > 0).sum() / len(d_rets) * 100), 1),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_hourly_stats(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Average return by hour (for intraday data only).
    
    Returns None if data does not appear to be intraday.
    """
    if len(df) < 2:
        return None

    # Check if intraday
    deltas = pd.Series(df.index).diff().dropna()
    median_seconds = deltas.median().total_seconds()
    if median_seconds > 86400:  # More than a day between bars
        return None

    returns = df['close'].pct_change().dropna() * 100
    hours = returns.index.hour

    rows = []
    for h in sorted(hours.unique()):
        h_rets = returns[hours == h]
        if len(h_rets) == 0:
            continue
        rows.append({
            'Hour': f"{h:02d}:00",
            'Count': len(h_rets),
            'Avg %': round(float(h_rets.mean()), 4),
            'Win %': round(float((h_rets > 0).sum() / len(h_rets) * 100), 1),
            'Std %': round(float(h_rets.std()), 4),
        })

    return pd.DataFrame(rows) if rows else None


def analyze_calendar(
    df: pd.DataFrame,
    symbol: str = ''
) -> CalendarAnalysis:
    """
    Full calendar analysis of price data.
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        symbol: Ticker symbol for labeling
        
    Returns:
        CalendarAnalysis with all breakdowns
    """
    dow_stats, dow_df = compute_day_of_week_stats(df)
    monthly_stats, monthly_df = compute_monthly_stats(df)
    heatmap_df = compute_monthly_heatmap(df)
    dom_df = compute_day_of_month_stats(df)

    return CalendarAnalysis(
        day_of_week=dow_stats,
        day_of_week_df=dow_df,
        monthly=monthly_stats,
        monthly_df=monthly_df,
        monthly_heatmap=heatmap_df,
        day_of_month_df=dom_df,
        symbol=symbol,
        start_date=str(df.index[0].date()) if len(df) > 0 else '',
        end_date=str(df.index[-1].date()) if len(df) > 0 else '',
        total_bars=len(df),
    )


def analyze_trade_calendar(trades: List) -> TradeCalendarAnalysis:
    """
    Analyze trade performance by calendar dimensions.
    
    Groups trades by their entry date's day-of-week and month
    to find if certain days/months produce better trades.
    
    Args:
        trades: List of Trade objects with entry_date, pnl attributes
    """
    if not trades:
        return TradeCalendarAnalysis(
            trades_by_day=pd.DataFrame(),
            trades_by_month=pd.DataFrame(),
        )

    # Build trade DataFrame
    trade_rows = []
    for t in trades:
        if t.entry_date is not None:
            trade_rows.append({
                'entry_date': t.entry_date,
                'dow': t.entry_date.dayofweek,
                'month': t.entry_date.month,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'direction': t.direction,
                'winner': 1 if t.pnl > 0 else 0,
            })

    if not trade_rows:
        return TradeCalendarAnalysis(
            trades_by_day=pd.DataFrame(),
            trades_by_month=pd.DataFrame(),
        )

    tdf = pd.DataFrame(trade_rows)

    # ── By Day of Week ──
    day_rows = []
    for d in range(5):
        subset = tdf[tdf['dow'] == d]
        if len(subset) == 0:
            continue
        day_rows.append({
            'Day': DAY_NAMES[d],
            'Trades': len(subset),
            'Win %': round(subset['winner'].mean() * 100, 1),
            'Avg PnL $': round(subset['pnl'].mean(), 2),
            'Avg PnL %': round(subset['pnl_pct'].mean(), 3),
            'Total PnL $': round(subset['pnl'].sum(), 2),
        })
    by_day = pd.DataFrame(day_rows) if day_rows else pd.DataFrame()

    # ── By Month ──
    month_rows = []
    for m in range(1, 13):
        subset = tdf[tdf['month'] == m]
        if len(subset) == 0:
            continue
        month_rows.append({
            'Month': MONTH_NAMES[m - 1],
            'Trades': len(subset),
            'Win %': round(subset['winner'].mean() * 100, 1),
            'Avg PnL $': round(subset['pnl'].mean(), 2),
            'Total PnL $': round(subset['pnl'].sum(), 2),
        })
    by_month = pd.DataFrame(month_rows) if month_rows else pd.DataFrame()

    return TradeCalendarAnalysis(
        trades_by_day=by_day,
        trades_by_month=by_month,
    )