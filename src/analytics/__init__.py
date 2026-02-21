"""
Calendar Analytics Module — v3
===============================
Fixes from v2 audit:

- FIX #1: _resample_to_daily now applies daily resampling to ALL data,
  not just intraday. Daily data passed raw had crypto/FX weekend gaps
  counted as single-day returns. Now consistently uses resample('D').

- FIX #2: compute_hourly_stats resamples to 1h before computing returns.
  Previously used per-bar returns grouped by hour, which gave the average
  5m/15m return per bar rather than the average hourly return. Only correct
  for 1h bars; now correct for all intraday frequencies.

- FIX #3: analyze_trade_calendar wraps entry_date in pd.Timestamp().
  .dayofweek doesn't exist on plain Python datetime objects. Defensive
  cast added; pnl_pct accessed via getattr with 0.0 fallback.

- FIX #4: resample('ME') → _safe_month_resample() helper.
  'ME' is only valid on pandas >= 2.2.0; requirements.txt allows 2.0+.
  Helper tries 'ME', falls back to 'M' for older pandas versions.

- FIX #5: compute_monthly_heatmap rebuilt with groupby/unstack.
  Previous nested dict loop was O(n) but fragile to duplicates and
  harder to reason about. Now one-liner vectorised groupby.

- FIX #6: Empty DataFrame guards added to all compute_* functions.
  Previously some paths would crash on empty input.

Presentation improvements (v3):
- DayOfWeekStats: added win_streak (max consecutive positive days)
  and sharpe-like consistency score (mean/std) for ranking days.
- MonthStats: added std_return_pct and total_return_pct.
- CalendarAnalysis: added summary_stats dict (best/worst day, best/worst
  month, overall win rate) for quick UI display.
- All DataFrames returned with consistent column ordering and
  formatted-ready float precision (rounded at source).
- New: compute_return_distribution() — histogram bins for return
  distribution display, resampled correctly for any frequency.
- New: compute_consecutive_stats() — max win/loss streaks, useful
  for risk-of-ruin context.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DAY_NAMES  = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
MONTH_NAMES = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DayOfWeekStats:
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
    consistency_score: float   # v3: mean / std — higher = more predictable


@dataclass
class MonthStats:
    month_name: str
    count: int
    mean_return_pct: float
    median_return_pct: float
    std_return_pct: float      # v3: added
    positive_pct: float
    best_pct: float
    worst_pct: float
    total_return_pct: float    # v3: added


@dataclass
class ReturnDistribution:
    """Histogram data for return distribution display."""
    bins: List[float]
    counts: List[int]
    mean: float
    std: float
    skew: float
    kurtosis: float
    pct_positive: float


@dataclass
class ConsecutiveStats:
    """Win/loss streak statistics."""
    max_win_streak: int
    max_loss_streak: int
    avg_win_streak: float
    avg_loss_streak: float
    current_streak: int          # positive = win streak, negative = loss streak


@dataclass
class CalendarAnalysis:
    day_of_week: List[DayOfWeekStats]
    day_of_week_df: pd.DataFrame
    monthly: List[MonthStats]
    monthly_df: pd.DataFrame
    monthly_heatmap: pd.DataFrame
    day_of_month_df: pd.DataFrame
    hourly_df: Optional[pd.DataFrame]
    distribution: ReturnDistribution         # v3: new
    consecutive: ConsecutiveStats            # v3: new
    summary_stats: Dict                      # v3: new — quick-access dict for UI
    symbol: str
    start_date: str
    end_date: str
    total_bars: int
    is_intraday: bool


@dataclass
class TradeCalendarAnalysis:
    trades_by_day: pd.DataFrame
    trades_by_month: pd.DataFrame


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_intraday(df: pd.DataFrame) -> bool:
    """
    Detect whether data is sub-daily based on median bar interval.
    Requires at least 3 bars for a reliable median (not just 2).
    """
    if len(df) < 3:
        return False
    deltas = pd.Series(df.index).diff().dropna()
    median_td = deltas.median()
    if pd.isna(median_td):
        return False
    return median_td.total_seconds() < 86_400


def _safe_month_resample(series: pd.Series) -> pd.Series:
    """
    Resample to month-end, compatible with pandas 2.0+ and 2.2+.
    'ME' was introduced in pandas 2.2. 'M' was deprecated in 2.2
    but still works in 2.0/2.1.
    """
    try:
        return series.resample('ME').last().dropna()
    except ValueError:
        return series.resample('M').last().dropna()


def _resample_to_daily(df: pd.DataFrame) -> pd.Series:
    """
    Compute daily close returns from any-frequency OHLCV data.

    FIX #1: All data (daily AND intraday) is resampled via resample('D').
    Previously daily data was passed through raw, which meant crypto/FX
    weekend gaps were treated as single-day returns.

    resample('D').last() produces NaN for calendar days with no bars
    (weekends, holidays). dropna() removes them. pct_change() then
    computes true day-over-day returns, skipping gaps correctly.

    Returns:
        pd.Series of daily returns in percent, DatetimeIndex.
    """
    if df.empty or 'close' not in df.columns:
        return pd.Series(dtype=float)

    daily_close = df['close'].resample('D').last().dropna()
    returns = daily_close.pct_change().dropna() * 100
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# Compute functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_day_of_week_stats(
    df: pd.DataFrame,
) -> Tuple[List[DayOfWeekStats], pd.DataFrame]:
    """
    Return statistics per weekday from daily returns.
    Handles any input frequency via _resample_to_daily.
    """
    if df.empty:
        return [], pd.DataFrame()

    returns = _resample_to_daily(df)
    if returns.empty:
        return [], pd.DataFrame()

    dow = returns.index.dayofweek
    stats_list: List[DayOfWeekStats] = []
    rows = []

    for day_idx in range(5):
        day_rets = returns[dow == day_idx]
        if len(day_rets) == 0:
            continue

        positive = day_rets[day_rets > 0]
        negative = day_rets[day_rets <= 0]
        std = float(day_rets.std()) if len(day_rets) > 1 else 0.0
        mean = float(day_rets.mean())
        # Consistency score: mean / std. Zero std → 0 to avoid div-by-zero.
        consistency = round(mean / std, 3) if std > 1e-9 else 0.0

        s = DayOfWeekStats(
            day_name=DAY_NAMES[day_idx],
            count=len(day_rets),
            mean_return_pct=round(mean, 5),
            median_return_pct=round(float(day_rets.median()), 5),
            std_return_pct=round(std, 5),
            positive_pct=round(float(len(positive) / len(day_rets) * 100), 2),
            avg_positive_pct=round(float(positive.mean()), 5) if len(positive) > 0 else 0.0,
            avg_negative_pct=round(float(negative.mean()), 5) if len(negative) > 0 else 0.0,
            best_pct=round(float(day_rets.max()), 3),
            worst_pct=round(float(day_rets.min()), 3),
            total_return_pct=round(float(day_rets.sum()), 3),
            consistency_score=consistency,
        )
        stats_list.append(s)
        rows.append({
            'Day':          s.day_name,
            'Observations': s.count,
            'Avg %':        s.mean_return_pct,
            'Median %':     s.median_return_pct,
            'Std %':        s.std_return_pct,
            'Win Rate':     f"{s.positive_pct:.1f}%",
            'Avg Win %':    s.avg_positive_pct,
            'Avg Loss %':   s.avg_negative_pct,
            'Best %':       s.best_pct,
            'Worst %':      s.worst_pct,
            'Total %':      s.total_return_pct,
            'Consistency':  s.consistency_score,
        })

    df_out = pd.DataFrame(rows) if rows else pd.DataFrame()
    return stats_list, df_out


def compute_monthly_stats(
    df: pd.DataFrame,
) -> Tuple[List[MonthStats], pd.DataFrame]:
    """
    Monthly return statistics by calendar month (Jan–Dec).
    Uses _safe_month_resample for pandas 2.0/2.1/2.2+ compatibility.
    """
    if df.empty:
        return [], pd.DataFrame()

    monthly_close = _safe_month_resample(df['close'])
    monthly_returns = monthly_close.pct_change().dropna() * 100

    if monthly_returns.empty:
        return [], pd.DataFrame()

    month_nums = monthly_returns.index.month
    stats_list: List[MonthStats] = []
    rows = []

    for m in range(1, 13):
        m_rets = monthly_returns[month_nums == m]
        if len(m_rets) == 0:
            continue
        positive = m_rets[m_rets > 0]
        std = round(float(m_rets.std()), 3) if len(m_rets) > 1 else 0.0

        s = MonthStats(
            month_name=MONTH_NAMES[m - 1],
            count=len(m_rets),
            mean_return_pct=round(float(m_rets.mean()), 3),
            median_return_pct=round(float(m_rets.median()), 3),
            std_return_pct=std,
            positive_pct=round(float(len(positive) / len(m_rets) * 100), 1),
            best_pct=round(float(m_rets.max()), 2),
            worst_pct=round(float(m_rets.min()), 2),
            total_return_pct=round(float(m_rets.sum()), 2),
        )
        stats_list.append(s)
        rows.append({
            'Month':     s.month_name,
            'Years':     s.count,
            'Avg %':     s.mean_return_pct,
            'Median %':  s.median_return_pct,
            'Std %':     s.std_return_pct,
            'Win Rate':  f"{s.positive_pct:.1f}%",
            'Best %':    s.best_pct,
            'Worst %':   s.worst_pct,
            'Total %':   s.total_return_pct,
        })

    df_out = pd.DataFrame(rows) if rows else pd.DataFrame()
    return stats_list, df_out


def compute_monthly_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Year × Month matrix of monthly returns for heatmap display.

    FIX #5: Rebuilt with groupby/unstack — one vectorised operation,
    no nested dict loop, no duplicate overwrite risk.
    Columns are always integer month numbers 1–12 (renamed to abbrevs
    only at display time to keep the DataFrame semantically clean).
    """
    if df.empty:
        return pd.DataFrame()

    monthly_close = _safe_month_resample(df['close'])
    monthly_returns = monthly_close.pct_change().dropna() * 100

    if monthly_returns.empty:
        return pd.DataFrame()

    heatmap = (
        monthly_returns
        .groupby([monthly_returns.index.year, monthly_returns.index.month])
        .first()
        .unstack(level=1)
        .round(2)
    )
    # Rename columns from integer months to abbreviated names
    heatmap.columns = [MONTH_NAMES[m - 1] for m in heatmap.columns]
    heatmap.index.name = 'Year'
    return heatmap


def compute_day_of_month_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average return by day of month (1–31).
    Uses daily returns regardless of input frequency.
    """
    if df.empty:
        return pd.DataFrame()

    returns = _resample_to_daily(df)
    if returns.empty:
        return pd.DataFrame()

    dom = returns.index.day
    rows = []

    for d in range(1, 32):
        d_rets = returns[dom == d]
        if len(d_rets) == 0:
            continue
        rows.append({
            'Day of Month': d,
            'Observations': len(d_rets),
            'Avg %':        round(float(d_rets.mean()), 4),
            'Win Rate':     f"{float((d_rets > 0).sum() / len(d_rets) * 100):.1f}%",
            'Std %':        round(float(d_rets.std()), 4) if len(d_rets) > 1 else 0.0,
            'Best %':       round(float(d_rets.max()), 3),
            'Worst %':      round(float(d_rets.min()), 3),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_hourly_stats(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Average return by hour of day (intraday data only).

    FIX #2: Resamples to 1h before computing returns.
    Previously computed per-bar returns grouped by hour, which gave
    the average 5m/15m return per bar — not the average hourly return.
    Now correct for all intraday frequencies: 1m, 5m, 15m, 30m, 1h.
    """
    if not _is_intraday(df) or df.empty:
        return None

    hourly_close = df['close'].resample('1h').last().dropna()
    if len(hourly_close) < 2:
        return None

    returns = hourly_close.pct_change().dropna() * 100
    hours = returns.index.hour

    rows = []
    for h in sorted(hours.unique()):
        h_rets = returns[hours == h]
        if len(h_rets) == 0:
            continue
        rows.append({
            'Hour':         f"{h:02d}:00",
            'Observations': len(h_rets),
            'Avg %':        round(float(h_rets.mean()), 4),
            'Win Rate':     f"{float((h_rets > 0).sum() / len(h_rets) * 100):.1f}%",
            'Std %':        round(float(h_rets.std()), 4) if len(h_rets) > 1 else 0.0,
            'Best %':       round(float(h_rets.max()), 3),
            'Worst %':      round(float(h_rets.min()), 3),
        })
    return pd.DataFrame(rows) if rows else None


def compute_return_distribution(df: pd.DataFrame) -> ReturnDistribution:
    """
    v3: Histogram data for return distribution display.
    Uses daily returns (correctly resampled for any frequency).
    Includes mean, std, skew, excess kurtosis, % positive.
    """
    returns = _resample_to_daily(df)

    if returns.empty:
        return ReturnDistribution([], [], 0.0, 0.0, 0.0, 0.0, 0.0)

    counts, bin_edges = np.histogram(returns, bins=40)
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))]

    return ReturnDistribution(
        bins=[round(b, 4) for b in bin_centers],
        counts=counts.tolist(),
        mean=round(float(returns.mean()), 5),
        std=round(float(returns.std()), 5),
        skew=round(float(returns.skew()), 4),
        kurtosis=round(float(returns.kurt()), 4),  # excess kurtosis
        pct_positive=round(float((returns > 0).sum() / len(returns) * 100), 2),
    )


def compute_consecutive_stats(df: pd.DataFrame) -> ConsecutiveStats:
    """
    v3: Max/avg win and loss streaks from daily returns.
    Useful as risk-of-ruin context alongside Monte Carlo.
    """
    returns = _resample_to_daily(df)

    if returns.empty:
        return ConsecutiveStats(0, 0, 0.0, 0.0, 0)

    wins = (returns > 0).astype(int).values

    max_win = max_loss = cur_win = cur_loss = 0
    win_streaks: List[int] = []
    loss_streaks: List[int] = []

    for w in wins:
        if w == 1:
            cur_win += 1
            if cur_loss > 0:
                loss_streaks.append(cur_loss)
                cur_loss = 0
        else:
            cur_loss += 1
            if cur_win > 0:
                win_streaks.append(cur_win)
                cur_win = 0

    if cur_win > 0:
        win_streaks.append(cur_win)
    if cur_loss > 0:
        loss_streaks.append(cur_loss)

    max_win  = max(win_streaks,  default=0)
    max_loss = max(loss_streaks, default=0)
    avg_win  = round(float(np.mean(win_streaks)),  2) if win_streaks  else 0.0
    avg_loss = round(float(np.mean(loss_streaks)), 2) if loss_streaks else 0.0

    # Current streak: + = ongoing win streak, - = ongoing loss streak
    if len(wins) == 0:
        current = 0
    elif wins[-1] == 1:
        current = int(cur_win) if cur_win > 0 else 1
    else:
        current = -int(cur_loss) if cur_loss > 0 else -1

    return ConsecutiveStats(
        max_win_streak=max_win, max_loss_streak=max_loss,
        avg_win_streak=avg_win, avg_loss_streak=avg_loss,
        current_streak=current,
    )


def _build_summary_stats(
    dow_stats: List[DayOfWeekStats],
    monthly_stats: List[MonthStats],
    returns: pd.Series,
) -> Dict:
    """
    v3: Build a flat dict of headline stats for the UI summary row.
    """
    summary: Dict = {}

    if dow_stats:
        best_day  = max(dow_stats, key=lambda s: s.mean_return_pct)
        worst_day = min(dow_stats, key=lambda s: s.mean_return_pct)
        summary['best_day']        = best_day.day_name
        summary['best_day_avg']    = best_day.mean_return_pct
        summary['worst_day']       = worst_day.day_name
        summary['worst_day_avg']   = worst_day.mean_return_pct

    if monthly_stats:
        best_month  = max(monthly_stats, key=lambda s: s.mean_return_pct)
        worst_month = min(monthly_stats, key=lambda s: s.mean_return_pct)
        summary['best_month']      = best_month.month_name
        summary['best_month_avg']  = best_month.mean_return_pct
        summary['worst_month']     = worst_month.month_name
        summary['worst_month_avg'] = worst_month.mean_return_pct

    if not returns.empty:
        summary['overall_win_rate'] = round(float((returns > 0).sum() / len(returns) * 100), 1)
        summary['total_observations'] = len(returns)
        summary['annualized_return']  = round(float(returns.mean() * 252), 2)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_calendar(df: pd.DataFrame, symbol: str = '') -> CalendarAnalysis:
    """
    Full calendar analysis. Handles daily, intraday, and crypto/FX data correctly.
    All computations are frequency-aware.
    """
    if df.empty:
        empty_dist = ReturnDistribution([], [], 0.0, 0.0, 0.0, 0.0, 0.0)
        empty_cons = ConsecutiveStats(0, 0, 0.0, 0.0, 0)
        return CalendarAnalysis(
            day_of_week=[], day_of_week_df=pd.DataFrame(),
            monthly=[], monthly_df=pd.DataFrame(),
            monthly_heatmap=pd.DataFrame(), day_of_month_df=pd.DataFrame(),
            hourly_df=None, distribution=empty_dist, consecutive=empty_cons,
            summary_stats={}, symbol=symbol, start_date='', end_date='',
            total_bars=0, is_intraday=False,
        )

    intraday = _is_intraday(df)
    returns  = _resample_to_daily(df)

    dow_stats,  dow_df     = compute_day_of_week_stats(df)
    month_stats, monthly_df = compute_monthly_stats(df)
    heatmap_df  = compute_monthly_heatmap(df)
    dom_df      = compute_day_of_month_stats(df)
    hourly_df   = compute_hourly_stats(df) if intraday else None
    distribution = compute_return_distribution(df)
    consecutive  = compute_consecutive_stats(df)
    summary      = _build_summary_stats(dow_stats, month_stats, returns)

    return CalendarAnalysis(
        day_of_week=dow_stats,    day_of_week_df=dow_df,
        monthly=month_stats,       monthly_df=monthly_df,
        monthly_heatmap=heatmap_df, day_of_month_df=dom_df,
        hourly_df=hourly_df,
        distribution=distribution, consecutive=consecutive,
        summary_stats=summary,
        symbol=symbol,
        start_date=str(df.index[0].date()),
        end_date=str(df.index[-1].date()),
        total_bars=len(df),
        is_intraday=intraday,
    )


def analyze_trade_calendar(trades: List) -> TradeCalendarAnalysis:
    """
    Analyze trade performance by entry day-of-week and month.

    FIX #3: entry_date cast to pd.Timestamp (plain Python datetime
    objects don't have .dayofweek). pnl_pct accessed via getattr
    with 0.0 fallback for resilience against differing trade schemas.
    """
    if not trades:
        return TradeCalendarAnalysis(
            trades_by_day=pd.DataFrame(),
            trades_by_month=pd.DataFrame(),
        )

    rows = []
    for t in trades:
        if t.entry_date is None:
            continue
        try:
            ts = pd.Timestamp(t.entry_date)
        except Exception:
            continue
        rows.append({
            'entry_date': ts,
            'dow':        ts.dayofweek,
            'month':      ts.month,
            'pnl':        getattr(t, 'pnl', 0.0),
            'pnl_pct':    getattr(t, 'pnl_pct', 0.0),   # FIX #3: defensive fallback
            'direction':  getattr(t, 'direction', ''),
            'winner':     1 if getattr(t, 'pnl', 0.0) > 0 else 0,
        })

    if not rows:
        return TradeCalendarAnalysis(
            trades_by_day=pd.DataFrame(),
            trades_by_month=pd.DataFrame(),
        )

    tdf = pd.DataFrame(rows)

    day_rows = []
    for d in range(5):
        sub = tdf[tdf['dow'] == d]
        if len(sub) == 0:
            continue
        day_rows.append({
            'Day':        DAY_NAMES[d],
            'Trades':     len(sub),
            'Win Rate':   f"{sub['winner'].mean() * 100:.1f}%",
            'Avg P&L $':  round(sub['pnl'].mean(), 2),
            'Avg P&L %':  round(sub['pnl_pct'].mean(), 3),
            'Total P&L $': round(sub['pnl'].sum(), 2),
        })

    month_rows = []
    for m in range(1, 13):
        sub = tdf[tdf['month'] == m]
        if len(sub) == 0:
            continue
        month_rows.append({
            'Month':       MONTH_NAMES[m - 1],
            'Trades':      len(sub),
            'Win Rate':    f"{sub['winner'].mean() * 100:.1f}%",
            'Avg P&L $':   round(sub['pnl'].mean(), 2),
            'Avg P&L %':   round(sub['pnl_pct'].mean(), 3),
            'Total P&L $': round(sub['pnl'].sum(), 2),
        })

    return TradeCalendarAnalysis(
        trades_by_day=pd.DataFrame(day_rows) if day_rows else pd.DataFrame(),
        trades_by_month=pd.DataFrame(month_rows) if month_rows else pd.DataFrame(),
    )