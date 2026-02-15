"""
Backtest Module - v6
====================
Improvements over v5:
- Mark-to-market equity curve (tracks unrealized P&L every bar)
- Fixed position sizing (no circular dependency)
- CAGR, Calmar, Expectancy, Payoff Ratio metrics
- Proper annualization based on actual bar frequency
- Separate realized vs unrealized tracking
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from ..strategy import StrategyParams, SignalGenerator, TradeDirection


@dataclass
class Trade:
    """Single trade record"""
    entry_idx: int
    entry_date: pd.Timestamp
    entry_price: float
    direction: str  # 'long' or 'short'
    size_dollars: float = 0.0  # Dollar amount allocated to this trade
    exit_idx: Optional[int] = None
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    mae: float = 0.0   # Maximum Adverse Excursion (worst intra-trade drawdown %)
    mfe: float = 0.0   # Maximum Favorable Excursion (best intra-trade gain %)


@dataclass
class BacktestResults:
    """Backtest results with comprehensive metrics"""
    trades: List[Trade]
    equity_curve: pd.Series       # Mark-to-market (includes unrealized)
    realized_equity: pd.Series    # Only realized P&L
    total_return: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0            # Compound Annual Growth Rate
    num_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0      # Average $ per trade (risk-adjusted)
    payoff_ratio: float = 0.0    # avg_winner / abs(avg_loser)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0    # CAGR / Max Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    avg_winner_pct: float = 0.0
    avg_loser_pct: float = 0.0
    avg_trade: float = 0.0
    avg_bars_held: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    longest_drawdown_bars: int = 0  # Duration of longest drawdown
    pct_time_in_market: float = 0.0
    avg_mae: float = 0.0         # Average MAE across trades
    avg_mfe: float = 0.0         # Average MFE across trades
    # Settings used
    initial_capital: float = 10000.0
    commission_pct: float = 0.1
    bars_per_year: int = 252     # For annualization


def _estimate_bars_per_year(df: pd.DataFrame) -> int:
    """
    Estimate annualization factor from data frequency.
    
    Uses median time delta between bars to infer frequency.
    Returns approximate trading bars per year.
    """
    if len(df) < 2:
        return 252

    deltas = pd.Series(df.index).diff().dropna()
    median_delta = deltas.median()

    seconds = median_delta.total_seconds()

    if seconds <= 120:       # ~1-2 min
        return 252 * 390     # 390 mins per trading day
    elif seconds <= 600:     # ~5 min
        return 252 * 78
    elif seconds <= 1800:    # ~15 min
        return 252 * 26
    elif seconds <= 3600:    # ~30 min
        return 252 * 13
    elif seconds <= 7200:    # ~1 hour
        return 252 * 7       # ~7 hourly bars per day
    elif seconds <= 172800:  # ~1 day
        return 252
    elif seconds <= 864000:  # ~1 week
        return 52
    else:                    # monthly
        return 12


class BacktestEngine:
    """
    Backtest engine with mark-to-market equity tracking.
    
    Key improvements:
    - Equity curve reflects unrealized P&L every bar (not just on exits)
    - Position sizing uses dollar allocation, no circular dependency
    - MAE/MFE tracked per trade for trade quality analysis
    - Proper annualization based on data frequency
    """

    def __init__(
        self,
        params: StrategyParams,
        initial_capital: float = 10000,
        commission_pct: float = 0.1,
        slippage_pct: float = 0.0
    ):
        self.params = params
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.signal_gen = SignalGenerator(params)

    def _calculate_trade_size_dollars(
        self, available_capital: float, price: float,
        win_rate: float = 0.5
    ) -> float:
        """
        Calculate dollar amount to allocate to this trade.
        
        Returns dollar value (not shares), capped at available capital.
        No circular dependency: we simply compute a fraction of capital.
        """
        p = self.params

        if p.use_kelly and win_rate > 0:
            # Kelly criterion: f* = (b*p - q) / b
            # b = average win / average loss ratio
            avg_win = p.take_profit_pct_long if p.take_profit_enabled else 5.0
            avg_loss = p.stop_loss_pct_long if p.stop_loss_enabled else 3.0
            b = avg_win / max(avg_loss, 0.01)
            q = 1.0 - win_rate
            kelly = (b * win_rate - q) / b if b > 0 else 0.0
            kelly = max(0.0, min(kelly, 1.0))
            fraction = kelly * p.kelly_fraction
        else:
            fraction = p.position_size_pct / 100.0

        fraction = min(fraction, 1.0)
        return available_capital * fraction

    def _unrealized_pnl(self, trade: Trade, current_price: float) -> float:
        """Compute unrealized P&L for an open position at current_price."""
        if trade.direction == 'long':
            raw_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            raw_pct = (trade.entry_price - current_price) / trade.entry_price
        return trade.size_dollars * raw_pct

    def run(self, df: pd.DataFrame) -> BacktestResults:
        """Run backtest with mark-to-market equity tracking."""
        p = self.params
        bars_per_year = _estimate_bars_per_year(df)

        # Generate signals
        df = self.signal_gen.generate_all_signals(df)

        trades: List[Trade] = []
        cash = self.initial_capital      # Cash not in positions
        position: Optional[Trade] = None
        bars_in_trade = 0
        bars_in_market = 0
        highest_since_entry = 0.0
        lowest_since_entry = float('inf')

        # Track rolling win rate for Kelly
        recent_wins = 0
        recent_total = 0

        # Mark-to-market and realized equity, one value per bar
        mtm_equity = np.empty(len(df))
        realized_equity = np.empty(len(df))

        # Bar 0
        mtm_equity[0] = self.initial_capital
        realized_equity[0] = self.initial_capital

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            # ═══════════════════════════════════════════════════════════════
            # POSITION MANAGEMENT (exits)
            # ═══════════════════════════════════════════════════════════════
            if position is not None:
                bars_in_trade += 1
                bars_in_market += 1
                entry_price = position.entry_price

                # Track extremes
                if position.direction == 'long':
                    highest_since_entry = max(highest_since_entry, row['high'])
                    lowest_since_entry = min(lowest_since_entry, row['low'])
                    # MAE/MFE as percentages from entry
                    current_mae = (lowest_since_entry - entry_price) / entry_price * 100
                    current_mfe = (highest_since_entry - entry_price) / entry_price * 100
                else:
                    highest_since_entry = max(highest_since_entry, row['high'])
                    lowest_since_entry = min(lowest_since_entry, row['low'])
                    current_mae = (entry_price - highest_since_entry) / entry_price * 100
                    current_mfe = (entry_price - lowest_since_entry) / entry_price * 100

                exit_price = None
                exit_reason = None

                # ── STOP LOSS (uses high/low, not close) ──
                if p.stop_loss_enabled and exit_price is None:
                    if position.direction == 'long':
                        stop = entry_price * (1 - p.stop_loss_pct_long / 100)
                        if row['low'] <= stop:
                            exit_price = stop
                            exit_reason = 'stop_loss'
                    else:
                        stop = entry_price * (1 + p.stop_loss_pct_short / 100)
                        if row['high'] >= stop:
                            exit_price = stop
                            exit_reason = 'stop_loss'

                # ── TAKE PROFIT ──
                if p.take_profit_enabled and exit_price is None:
                    if position.direction == 'long':
                        tp = entry_price * (1 + p.take_profit_pct_long / 100)
                        if row['high'] >= tp:
                            exit_price = tp
                            exit_reason = 'take_profit'
                    else:
                        tp = entry_price * (1 - p.take_profit_pct_short / 100)
                        if row['low'] <= tp:
                            exit_price = tp
                            exit_reason = 'take_profit'

                # ── TRAILING STOP ──
                if p.trailing_stop_enabled and exit_price is None:
                    if position.direction == 'long':
                        activation = entry_price * (1 + p.trailing_stop_activation / 100)
                        if highest_since_entry >= activation:
                            trail = highest_since_entry * (1 - p.trailing_stop_pct / 100)
                            if row['low'] <= trail:
                                exit_price = trail
                                exit_reason = 'trailing_stop'
                    else:
                        activation = entry_price * (1 - p.trailing_stop_activation / 100)
                        if lowest_since_entry <= activation:
                            trail = lowest_since_entry * (1 + p.trailing_stop_pct / 100)
                            if row['high'] >= trail:
                                exit_price = trail
                                exit_reason = 'trailing_stop'

                # ── ATR TRAILING ──
                if p.atr_trailing_enabled and 'atr' in df.columns and exit_price is None:
                    atr_val = row['atr']
                    if position.direction == 'long':
                        atr_stop = highest_since_entry - atr_val * p.atr_multiplier
                        if row['low'] <= atr_stop:
                            exit_price = atr_stop
                            exit_reason = 'atr_trailing'
                    else:
                        atr_stop = lowest_since_entry + atr_val * p.atr_multiplier
                        if row['high'] >= atr_stop:
                            exit_price = atr_stop
                            exit_reason = 'atr_trailing'

                # ── TIME EXIT ──
                if p.time_exit_enabled and exit_price is None:
                    max_bars = (p.time_exit_bars_long
                                if position.direction == 'long'
                                else p.time_exit_bars_short)
                    if bars_in_trade >= max_bars:
                        exit_price = row['close']
                        exit_reason = 'time_exit'

                # ── SIGNAL EXIT ──
                if exit_price is None:
                    if position.direction == 'long' and row['exit_long_signal']:
                        exit_price = row['close']
                        exit_reason = 'signal'
                    elif position.direction == 'short' and row['exit_short_signal']:
                        exit_price = row['close']
                        exit_reason = 'signal'

                # ── EXECUTE EXIT ──
                if exit_price is not None:
                    # Slippage
                    if position.direction == 'long':
                        exit_price *= (1 - self.slippage_pct / 100)
                    else:
                        exit_price *= (1 + self.slippage_pct / 100)

                    # P&L calculation
                    if position.direction == 'long':
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price * 100

                    # Commission on both legs
                    pnl_pct -= self.commission_pct * 2

                    pnl = position.size_dollars * (pnl_pct / 100)

                    position.exit_idx = i
                    position.exit_date = row.name
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason
                    position.pnl = pnl
                    position.pnl_pct = pnl_pct
                    position.bars_held = bars_in_trade
                    position.mae = current_mae
                    position.mfe = current_mfe

                    trades.append(position)
                    cash += position.size_dollars + pnl  # Return allocation + profit/loss

                    recent_total += 1
                    if pnl > 0:
                        recent_wins += 1

                    position = None
                    bars_in_trade = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')

            # ═══════════════════════════════════════════════════════════════
            # ENTRY SIGNALS (on previous bar's signal, execute at this bar's open)
            # ═══════════════════════════════════════════════════════════════
            if position is None:
                win_rate = recent_wins / recent_total if recent_total > 0 else 0.5

                entered = False
                if prev_row['entry_long']:
                    entry_price = row['open'] * (1 + self.slippage_pct / 100)
                    size_dollars = self._calculate_trade_size_dollars(
                        cash, entry_price, win_rate
                    )
                    position = Trade(
                        entry_idx=i,
                        entry_date=row.name,
                        entry_price=entry_price,
                        direction='long',
                        size_dollars=size_dollars,
                    )
                    entered = True

                elif prev_row['entry_short']:
                    entry_price = row['open'] * (1 - self.slippage_pct / 100)
                    size_dollars = self._calculate_trade_size_dollars(
                        cash, entry_price, win_rate
                    )
                    position = Trade(
                        entry_idx=i,
                        entry_date=row.name,
                        entry_price=entry_price,
                        direction='short',
                        size_dollars=size_dollars,
                    )
                    entered = True

                if entered:
                    cash -= position.size_dollars  # Deduct allocation from cash
                    highest_since_entry = row['high']
                    lowest_since_entry = row['low']

            # ═══════════════════════════════════════════════════════════════
            # MARK-TO-MARKET EQUITY
            # ═══════════════════════════════════════════════════════════════
            unrealized = 0.0
            if position is not None:
                unrealized = self._unrealized_pnl(position, row['close'])

            mtm_equity[i] = cash + (position.size_dollars if position else 0.0) + unrealized
            realized_equity[i] = cash + (position.size_dollars if position else 0.0)

        # Build series
        eq_curve = pd.Series(mtm_equity, index=df.index)
        real_curve = pd.Series(realized_equity, index=df.index)

        return self._calculate_metrics(
            trades, eq_curve, real_curve,
            bars_per_year, bars_in_market, len(df)
        )

    def _calculate_metrics(
        self, trades: List[Trade],
        equity_curve: pd.Series, realized_equity: pd.Series,
        bars_per_year: int, bars_in_market: int, total_bars: int
    ) -> BacktestResults:
        """Calculate all performance metrics."""
        num_trades = len(trades)

        if num_trades == 0:
            return BacktestResults(
                trades=trades,
                equity_curve=equity_curve,
                realized_equity=realized_equity,
                initial_capital=self.initial_capital,
                commission_pct=self.commission_pct,
                bars_per_year=bars_per_year,
            )

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        total_return = equity_curve.iloc[-1] - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # ── CAGR ──
        # CAGR = (final/initial)^(bars_per_year / total_bars) - 1
        n_bars = len(equity_curve)
        if n_bars > 1 and equity_curve.iloc[-1] > 0:
            cagr = (equity_curve.iloc[-1] / self.initial_capital) ** (
                bars_per_year / n_bars
            ) - 1
            cagr *= 100  # As percentage
        else:
            cagr = 0.0

        win_rate = len(winners) / num_trades * 100 if num_trades > 0 else 0

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0

        # Profit factor
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 999.99
        else:
            profit_factor = 0
        profit_factor = min(profit_factor, 999.99)

        # Averages
        avg_winner = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loser = np.mean([t.pnl for t in losers]) if losers else 0
        avg_winner_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loser_pct = np.mean([t.pnl_pct for t in losers]) if losers else 0
        avg_trade = np.mean([t.pnl for t in trades])
        avg_bars = np.mean([t.bars_held for t in trades])

        # Payoff ratio: avg_winner / |avg_loser|
        payoff_ratio = (avg_winner / abs(avg_loser)) if avg_loser != 0 else 999.99
        payoff_ratio = min(payoff_ratio, 999.99)

        # Expectancy = (win_rate * avg_win) - (loss_rate * |avg_loss|)
        wr_frac = len(winners) / num_trades if num_trades > 0 else 0
        lr_frac = 1.0 - wr_frac
        expectancy = wr_frac * avg_winner - lr_frac * abs(avg_loser)

        # ── Drawdown ──
        peak = equity_curve.expanding().max()
        drawdown = equity_curve - peak
        max_drawdown = drawdown.min()
        max_drawdown_pct = (drawdown / peak).min() * 100 if peak.max() > 0 else 0

        # Longest drawdown duration (in bars)
        in_dd = drawdown < 0
        dd_groups = (~in_dd).cumsum()
        if in_dd.any():
            longest_dd_bars = int(in_dd.groupby(dd_groups).sum().max())
        else:
            longest_dd_bars = 0

        # ── Returns for Sharpe / Sortino ──
        returns = equity_curve.pct_change().dropna()

        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
        else:
            sharpe = 0.0

        neg_returns = returns[returns < 0]
        if len(neg_returns) > 1 and neg_returns.std() > 0:
            sortino = (returns.mean() / neg_returns.std()) * np.sqrt(bars_per_year)
        else:
            sortino = sharpe

        # ── Calmar = CAGR / |Max DD %| ──
        calmar = abs(cagr / max_drawdown_pct) if max_drawdown_pct != 0 else 0.0

        # ── Consecutive wins/losses ──
        max_consec_loss = 0
        max_consec_win = 0
        cur_loss = 0
        cur_win = 0
        for t in trades:
            if t.pnl <= 0:
                cur_loss += 1
                cur_win = 0
                max_consec_loss = max(max_consec_loss, cur_loss)
            else:
                cur_win += 1
                cur_loss = 0
                max_consec_win = max(max_consec_win, cur_win)

        # ── MAE / MFE averages ──
        avg_mae = np.mean([t.mae for t in trades]) if trades else 0
        avg_mfe = np.mean([t.mfe for t in trades]) if trades else 0

        # % time in market
        pct_in_market = (bars_in_market / total_bars * 100) if total_bars > 0 else 0

        return BacktestResults(
            trades=trades,
            equity_curve=equity_curve,
            realized_equity=realized_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            cagr=cagr,
            num_trades=num_trades,
            winners=len(winners),
            losers=len(losers),
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            payoff_ratio=payoff_ratio,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            avg_winner_pct=avg_winner_pct,
            avg_loser_pct=avg_loser_pct,
            avg_trade=avg_trade,
            avg_bars_held=avg_bars,
            max_consecutive_losses=max_consec_loss,
            max_consecutive_wins=max_consec_win,
            longest_drawdown_bars=longest_dd_bars,
            pct_time_in_market=pct_in_market,
            avg_mae=avg_mae,
            avg_mfe=avg_mfe,
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            bars_per_year=bars_per_year,
        )