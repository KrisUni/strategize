"""
Backtest Module - v8
====================
Fixes from v7 audit:
- CRITICAL #1: Gap-through fills — stop orders fill at open when open
  gaps past the order level (not at the order level). Applies to SL,
  TP, trailing stop, and ATR trailing. Matches NinjaTrader/Backtrader.
- CRITICAL #2: Entry-bar SL/TP — stops are now checked on the entry
  bar itself. If you enter at open and the bar's range hits your stop,
  you get stopped out immediately (same bar).
- HIGH #3: Time exit fills at open (was close — look-ahead bias).
- HIGH #4: Kelly criterion uses realized trade statistics (rolling
  avg win/loss %) after 20 trades, not theoretical SL/TP params.
- All prior fixes retained (signal timing, SL/TP priority, active
  Sharpe, end-of-data close, MTM equity, MAE/MFE).
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
    size_dollars: float = 0.0
    exit_idx: Optional[int] = None
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    mae: float = 0.0   # Maximum Adverse Excursion (%)
    mfe: float = 0.0   # Maximum Favorable Excursion (%)


@dataclass
class BacktestResults:
    """Backtest results with comprehensive metrics"""
    trades: List[Trade]
    equity_curve: pd.Series       # Mark-to-market (includes unrealized)
    realized_equity: pd.Series    # Only realized P&L
    total_return: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    num_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
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
    longest_drawdown_bars: int = 0
    pct_time_in_market: float = 0.0
    avg_mae: float = 0.0
    avg_mfe: float = 0.0
    initial_capital: float = 10000.0
    commission_pct: float = 0.1
    bars_per_year: int = 252


def _estimate_bars_per_year(df: pd.DataFrame) -> int:
    """
    Estimate annualization factor from data frequency.
    Uses median time delta between bars to infer frequency.
    """
    if len(df) < 2:
        return 252

    deltas = pd.Series(df.index).diff().dropna()
    median_delta = deltas.median()
    seconds = median_delta.total_seconds()

    if seconds <= 120:       # ~1-2 min
        return 252 * 390
    elif seconds <= 600:     # ~5 min
        return 252 * 78
    elif seconds <= 1800:    # ~15 min
        return 252 * 26
    elif seconds <= 3600:    # ~30 min
        return 252 * 13
    elif seconds <= 7200:    # ~1 hour
        return 252 * 7
    elif seconds <= 172800:  # ~1 day
        return 252
    elif seconds <= 864000:  # ~1 week
        return 52
    else:
        return 12


class BacktestEngine:
    """
    Backtest engine v8 — institutional-grade fill simulation.

    Key execution rules:
    - Entries: prev-bar signal → this bar's open (+ slippage).
    - Stop orders (SL/TP/trailing/ATR): trigger on high/low, fill at
      order level OR bar open if the open gaps past the level (whichever
      is worse for the trader). This is how real stop-market orders work.
    - SL/TP are checked on the ENTRY bar itself (bar[i] after entering
      at bar[i]'s open). If the bar's range hits the stop, exit same bar.
    - Signal exits: prev-bar signal → this bar's open (no look-ahead).
    - Time exits: execute at this bar's open (decision known at open).
    - SL/TP same-bar conflict: closer-to-open wins.
    - Open position at end of data is force-closed at last close.
    - Kelly uses rolling realized trade stats (after 20 trades).
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
        win_rate: float = 0.5,
        realized_avg_win_pct: float = 0.0,
        realized_avg_loss_pct: float = 0.0,
        realized_count: int = 0
    ) -> float:
        """
        Calculate dollar amount to allocate to this trade.
        Returns dollar value, capped at available capital.

        Kelly criterion uses REALIZED trade statistics once enough
        trades have accumulated (>= 20). Before that, falls back to
        parameter-based estimates. This prevents early over-sizing
        from inaccurate theoretical assumptions.
        """
        p = self.params
        KELLY_MIN_TRADES = 20

        if p.use_kelly and win_rate > 0:
            if realized_count >= KELLY_MIN_TRADES and realized_avg_loss_pct > 0:
                # Use actual realized performance
                avg_win = realized_avg_win_pct
                avg_loss = realized_avg_loss_pct
            else:
                # Fallback to parameter estimates until enough data
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
        """Compute unrealized P&L for an open position."""
        if trade.direction == 'long':
            raw_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            raw_pct = (trade.entry_price - current_price) / trade.entry_price
        return trade.size_dollars * raw_pct

    def _close_position(
        self, position: Trade, exit_price: float, exit_reason: str,
        bar_idx: int, bar_date: pd.Timestamp, bars_in_trade: int,
        current_mae: float, current_mfe: float
    ) -> Tuple[Trade, float]:
        """
        Close a position and compute final P&L.

        Returns:
            Tuple of (completed Trade, net P&L in dollars)
        """
        # Apply slippage to exit
        if position.direction == 'long':
            exit_price *= (1 - self.slippage_pct / 100)
        else:
            exit_price *= (1 + self.slippage_pct / 100)

        # P&L calculation
        if position.direction == 'long':
            pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price * 100

        # Commission on both legs
        pnl_pct -= self.commission_pct * 2

        pnl = position.size_dollars * (pnl_pct / 100)

        position.exit_idx = bar_idx
        position.exit_date = bar_date
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        position.pnl = pnl
        position.pnl_pct = pnl_pct
        position.bars_held = bars_in_trade
        position.mae = current_mae
        position.mfe = current_mfe

        return position, pnl

    def _gap_aware_fill(self, order_level: float, bar_open: float,
                        direction: str, order_type: str) -> float:
        """
        Compute realistic fill price for a stop-type order.

        Stop orders become market orders when triggered. If the bar's
        open has already gapped past the order level, the fill is at
        the open (the actual available market price), not the order level.

        Args:
            order_level: The SL/TP/trailing level
            bar_open: The bar's open price
            direction: 'long' or 'short'
            order_type: 'sl', 'tp', 'trail'

        Returns:
            Realistic fill price (worse of order_level vs open for trader)
        """
        if order_type == 'sl':
            # Stop-loss: trader wants OUT at order_level or worse
            if direction == 'long':
                # Long SL is below entry; gap down means open < SL → fill at open
                return min(order_level, bar_open)
            else:
                # Short SL is above entry; gap up means open > SL → fill at open
                return max(order_level, bar_open)
        elif order_type == 'tp':
            # Take-profit: fill at limit or better (for the trader)
            # TP is a limit order — if open gaps PAST the TP in the
            # trader's favor, fill at the open (which is BETTER).
            if direction == 'long':
                # Long TP is above entry; gap up means open > TP → fill at open
                return max(order_level, bar_open)
            else:
                # Short TP is below entry; gap down means open < TP → fill at open
                return min(order_level, bar_open)
        else:
            # Trailing / ATR trailing — same as stop-loss (worst-case)
            if direction == 'long':
                return min(order_level, bar_open)
            else:
                return max(order_level, bar_open)

    def _check_stop_exits(self, position: Trade, row, p,
                          highest_since_entry: float,
                          lowest_since_entry: float,
                          bars_in_trade: int,
                          df: pd.DataFrame) -> Tuple[Optional[float], Optional[str]]:
        """
        Check all stop-type exit conditions for a bar.

        This is factored out so it can be called BOTH on the entry bar
        (BUG 2 fix) and on subsequent bars without duplicating logic.

        Returns:
            (exit_price, exit_reason) or (None, None) if no exit.
        """
        entry_price = position.entry_price
        bar_open = row['open']
        exit_price = None
        exit_reason = None

        # ─────────────────────────────────────────────────────────
        # SL/TP with gap-through awareness and same-bar resolution
        # ─────────────────────────────────────────────────────────
        sl_price = None
        tp_price = None

        if p.stop_loss_enabled:
            if position.direction == 'long':
                stop = entry_price * (1 - p.stop_loss_pct_long / 100)
                if row['low'] <= stop:
                    sl_price = self._gap_aware_fill(stop, bar_open, 'long', 'sl')
            else:
                stop = entry_price * (1 + p.stop_loss_pct_short / 100)
                if row['high'] >= stop:
                    sl_price = self._gap_aware_fill(stop, bar_open, 'short', 'sl')

        if p.take_profit_enabled:
            if position.direction == 'long':
                tp = entry_price * (1 + p.take_profit_pct_long / 100)
                if row['high'] >= tp:
                    tp_price = self._gap_aware_fill(tp, bar_open, 'long', 'tp')
            else:
                tp = entry_price * (1 - p.take_profit_pct_short / 100)
                if row['low'] <= tp:
                    tp_price = self._gap_aware_fill(tp, bar_open, 'short', 'tp')

        # Resolve SL/TP conflict: closer to open wins
        if sl_price is not None and tp_price is not None:
            sl_dist = abs(bar_open - sl_price)
            tp_dist = abs(bar_open - tp_price)
            if tp_dist <= sl_dist:
                exit_price = tp_price
                exit_reason = 'take_profit'
            else:
                exit_price = sl_price
                exit_reason = 'stop_loss'
        elif sl_price is not None:
            exit_price = sl_price
            exit_reason = 'stop_loss'
        elif tp_price is not None:
            exit_price = tp_price
            exit_reason = 'take_profit'

        # ── TRAILING STOP ──
        if p.trailing_stop_enabled and exit_price is None:
            if position.direction == 'long':
                activation = entry_price * (1 + p.trailing_stop_activation / 100)
                if highest_since_entry >= activation:
                    trail = highest_since_entry * (1 - p.trailing_stop_pct / 100)
                    if row['low'] <= trail:
                        exit_price = self._gap_aware_fill(
                            trail, bar_open, 'long', 'trail')
                        exit_reason = 'trailing_stop'
            else:
                activation = entry_price * (1 - p.trailing_stop_activation / 100)
                if lowest_since_entry <= activation:
                    trail = lowest_since_entry * (1 + p.trailing_stop_pct / 100)
                    if row['high'] >= trail:
                        exit_price = self._gap_aware_fill(
                            trail, bar_open, 'short', 'trail')
                        exit_reason = 'trailing_stop'

        # ── ATR TRAILING ──
        if p.atr_trailing_enabled and 'atr' in df.columns and exit_price is None:
            atr_val = row['atr']
            if position.direction == 'long':
                atr_stop = highest_since_entry - atr_val * p.atr_multiplier
                if row['low'] <= atr_stop:
                    exit_price = self._gap_aware_fill(
                        atr_stop, bar_open, 'long', 'trail')
                    exit_reason = 'atr_trailing'
            else:
                atr_stop = lowest_since_entry + atr_val * p.atr_multiplier
                if row['high'] >= atr_stop:
                    exit_price = self._gap_aware_fill(
                        atr_stop, bar_open, 'short', 'trail')
                    exit_reason = 'atr_trailing'

        return exit_price, exit_reason

    def run(self, df: pd.DataFrame) -> BacktestResults:
        """Run backtest with v8 fixes (gap-through, entry-bar SL, time-exit, Kelly)."""
        p = self.params
        bars_per_year = _estimate_bars_per_year(df)

        # Generate signals
        df = self.signal_gen.generate_all_signals(df)

        trades: List[Trade] = []
        cash = self.initial_capital
        position: Optional[Trade] = None
        bars_in_trade = 0
        bars_in_market = 0
        highest_since_entry = 0.0
        lowest_since_entry = float('inf')
        current_mae = 0.0
        current_mfe = 0.0

        # ── BUG 4 FIX: Track rolling realized stats for Kelly ──
        recent_wins = 0
        recent_total = 0
        sum_win_pct = 0.0   # Sum of |pnl_pct| for winners
        count_wins = 0
        sum_loss_pct = 0.0  # Sum of |pnl_pct| for losers
        count_losses = 0

        # Mark-to-market and realized equity
        mtm_equity = np.empty(len(df))
        realized_equity = np.empty(len(df))

        mtm_equity[0] = self.initial_capital
        realized_equity[0] = self.initial_capital

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            intrabar_exit = False  # Reset each bar

            # ═══════════════════════════════════════════════════════════════
            # POSITION MANAGEMENT (exits)
            # ═══════════════════════════════════════════════════════════════
            if position is not None:
                bars_in_trade += 1
                bars_in_market += 1
                entry_price = position.entry_price

                # Track extremes for trailing stops and MAE/MFE
                highest_since_entry = max(highest_since_entry, row['high'])
                lowest_since_entry = min(lowest_since_entry, row['low'])

                if position.direction == 'long':
                    current_mae = (lowest_since_entry - entry_price) / entry_price * 100
                    current_mfe = (highest_since_entry - entry_price) / entry_price * 100
                else:
                    current_mae = (entry_price - highest_since_entry) / entry_price * 100
                    current_mfe = (entry_price - lowest_since_entry) / entry_price * 100

                # ─────────────────────────────────────────────────────────
                # Check all stop-type exits (SL/TP/trailing/ATR)
                # with gap-through-aware fill prices.
                # ─────────────────────────────────────────────────────────
                exit_price, exit_reason = self._check_stop_exits(
                    position, row, p,
                    highest_since_entry, lowest_since_entry,
                    bars_in_trade, df
                )

                # ── TIME EXIT (BUG 3 FIX: execute at open, not close) ──
                # The decision to time-exit is known at the start of the
                # bar (we know bars_in_trade). Execute at open to avoid
                # look-ahead bias. This is consistent with signal exits.
                if p.time_exit_enabled and exit_price is None:
                    max_bars = (p.time_exit_bars_long
                                if position.direction == 'long'
                                else p.time_exit_bars_short)
                    if bars_in_trade >= max_bars:
                        exit_price = row['open']
                        exit_reason = 'time_exit'

                # ─────────────────────────────────────────────────────────
                # Signal exit uses PREVIOUS bar's signal
                # Same logic as entries: signal on bar[i-1], execute at
                # bar[i]'s open. No look-ahead.
                # ─────────────────────────────────────────────────────────
                if exit_price is None:
                    if position.direction == 'long' and prev_row['exit_long_signal']:
                        exit_price = row['open']
                        exit_reason = 'signal'
                    elif position.direction == 'short' and prev_row['exit_short_signal']:
                        exit_price = row['open']
                        exit_reason = 'signal'

                # ── EXECUTE EXIT ──
                if exit_price is not None:
                    position, pnl = self._close_position(
                        position, exit_price, exit_reason,
                        i, row.name, bars_in_trade,
                        current_mae, current_mfe
                    )
                    trades.append(position)
                    cash += position.size_dollars + pnl

                    # Update rolling stats for Kelly (BUG 4 FIX)
                    recent_total += 1
                    if pnl > 0:
                        recent_wins += 1
                        sum_win_pct += abs(position.pnl_pct)
                        count_wins += 1
                    else:
                        sum_loss_pct += abs(position.pnl_pct)
                        count_losses += 1

                    # Flag: if exit was intrabar (SL/TP/trailing/ATR),
                    # we must NOT re-enter on this same bar's open because
                    # the stop was hit AFTER the open. Signal and time exits
                    # execute at open, so re-entry blocks until next bar.
                    intrabar_exit = exit_reason in (
                        'stop_loss', 'take_profit', 'trailing_stop',
                        'atr_trailing', 'time_exit'
                    )

                    position = None
                    bars_in_trade = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')

            # ═══════════════════════════════════════════════════════════════
            # ENTRY SIGNALS (previous bar's signal, execute at this bar's open)
            # ═══════════════════════════════════════════════════════════════
            if position is None and not intrabar_exit:
                win_rate = recent_wins / recent_total if recent_total > 0 else 0.5

                # Compute rolling realized stats for Kelly
                realized_avg_win = (sum_win_pct / count_wins) if count_wins > 0 else 0.0
                realized_avg_loss = (sum_loss_pct / count_losses) if count_losses > 0 else 0.0

                # Detect churn: if we just signal-exited a direction,
                # don't re-enter same direction on the same bar.
                just_exited_direction = None
                if (len(trades) > 0 and trades[-1].exit_idx == i
                        and trades[-1].exit_reason == 'signal'):
                    just_exited_direction = trades[-1].direction

                entered = False
                if prev_row['entry_long'] and just_exited_direction != 'long':
                    entry_price = row['open'] * (1 + self.slippage_pct / 100)
                    size_dollars = self._calculate_trade_size_dollars(
                        cash, entry_price, win_rate,
                        realized_avg_win, realized_avg_loss, recent_total
                    )
                    position = Trade(
                        entry_idx=i, entry_date=row.name,
                        entry_price=entry_price, direction='long',
                        size_dollars=size_dollars,
                    )
                    entered = True

                elif prev_row['entry_short'] and just_exited_direction != 'short':
                    entry_price = row['open'] * (1 - self.slippage_pct / 100)
                    size_dollars = self._calculate_trade_size_dollars(
                        cash, entry_price, win_rate,
                        realized_avg_win, realized_avg_loss, recent_total
                    )
                    position = Trade(
                        entry_idx=i, entry_date=row.name,
                        entry_price=entry_price, direction='short',
                        size_dollars=size_dollars,
                    )
                    entered = True

                if entered:
                    cash -= position.size_dollars
                    highest_since_entry = row['high']
                    lowest_since_entry = row['low']

                    # ─────────────────────────────────────────────────────
                    # BUG 2 FIX: Check SL/TP on the ENTRY bar.
                    #
                    # We just entered at this bar's open. The bar's
                    # high/low may already breach our stop or take-profit
                    # level. In reality, you'd get stopped out immediately.
                    #
                    # Update MAE/MFE for entry bar range.
                    # ─────────────────────────────────────────────────────
                    if position.direction == 'long':
                        current_mae = (lowest_since_entry - entry_price) / entry_price * 100
                        current_mfe = (highest_since_entry - entry_price) / entry_price * 100
                    else:
                        current_mae = (entry_price - highest_since_entry) / entry_price * 100
                        current_mfe = (entry_price - lowest_since_entry) / entry_price * 100

                    bars_in_trade = 0  # Will be incremented next iteration

                    entry_exit_price, entry_exit_reason = self._check_stop_exits(
                        position, row, p,
                        highest_since_entry, lowest_since_entry,
                        0, df  # bars_in_trade=0 on entry bar
                    )

                    if entry_exit_price is not None:
                        position, pnl = self._close_position(
                            position, entry_exit_price, entry_exit_reason,
                            i, row.name, 0,  # bars_held=0 (same bar)
                            current_mae, current_mfe
                        )
                        trades.append(position)
                        cash += position.size_dollars + pnl

                        recent_total += 1
                        if pnl > 0:
                            recent_wins += 1
                            sum_win_pct += abs(position.pnl_pct)
                            count_wins += 1
                        else:
                            sum_loss_pct += abs(position.pnl_pct)
                            count_losses += 1

                        position = None
                        bars_in_trade = 0
                        highest_since_entry = 0.0
                        lowest_since_entry = float('inf')
                        # intrabar_exit already False, no re-entry this bar
                        # because we're past the entry block

            # ═══════════════════════════════════════════════════════════════
            # MARK-TO-MARKET EQUITY
            # ═══════════════════════════════════════════════════════════════
            unrealized = 0.0
            if position is not None:
                unrealized = self._unrealized_pnl(position, row['close'])

            mtm_equity[i] = cash + (position.size_dollars if position else 0.0) + unrealized
            realized_equity[i] = cash + (position.size_dollars if position else 0.0)

        # ═══════════════════════════════════════════════════════════════════
        # FIX #8: Force-close any open position at end of data
        # ═══════════════════════════════════════════════════════════════════
        if position is not None:
            last_row = df.iloc[-1]
            last_close = last_row['close']

            position, pnl = self._close_position(
                position, last_close, 'end_of_data',
                len(df) - 1, last_row.name, bars_in_trade,
                current_mae, current_mfe
            )
            trades.append(position)
            cash += position.size_dollars + pnl

            # Update final equity bar to reflect the realized close
            mtm_equity[-1] = cash
            realized_equity[-1] = cash

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

        # ── CAGR = (final/initial)^(bars_per_year / n_bars) - 1 ──
        n_bars = len(equity_curve)
        if n_bars > 1 and equity_curve.iloc[-1] > 0:
            cagr = (equity_curve.iloc[-1] / self.initial_capital) ** (
                bars_per_year / n_bars
            ) - 1
            cagr *= 100
        else:
            cagr = 0.0

        win_rate = len(winners) / num_trades * 100

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 999.99
        else:
            profit_factor = 0
        profit_factor = min(profit_factor, 999.99)

        avg_winner = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loser = np.mean([t.pnl for t in losers]) if losers else 0
        avg_winner_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loser_pct = np.mean([t.pnl_pct for t in losers]) if losers else 0
        avg_trade = np.mean([t.pnl for t in trades])
        avg_bars = np.mean([t.bars_held for t in trades])

        payoff_ratio = (avg_winner / abs(avg_loser)) if avg_loser != 0 else 999.99
        payoff_ratio = min(payoff_ratio, 999.99)

        # Expectancy = (win_rate * avg_win) - (loss_rate * |avg_loss|)
        wr_frac = len(winners) / num_trades
        expectancy = wr_frac * avg_winner - (1.0 - wr_frac) * abs(avg_loser)

        # ── Drawdown ──
        peak = equity_curve.expanding().max()
        drawdown = equity_curve - peak
        max_drawdown = drawdown.min()
        max_drawdown_pct = (drawdown / peak).min() * 100 if peak.max() > 0 else 0

        in_dd = drawdown < 0
        dd_groups = (~in_dd).cumsum()
        if in_dd.any():
            longest_dd_bars = int(in_dd.groupby(dd_groups).sum().max())
        else:
            longest_dd_bars = 0

        # ─────────────────────────────────────────────────────────────
        # FIX #7: Sharpe / Sortino on ACTIVE returns only
        #
        # Filter to only bars where return != 0 (i.e. in-market).
        #
        # FIX #9 (v8): Correct annualization factor.
        # Active-only returns must be annualized using the number of
        # active observations per year, NOT bars_per_year (which is
        # the calendar frequency). Using bars_per_year inflates Sharpe
        # by sqrt(bars_per_year / active_bars_per_year) for strategies
        # with low market exposure.
        #
        # active_bars_per_year = n_active * (bars_per_year / n_total)
        # ─────────────────────────────────────────────────────────────
        returns = equity_curve.pct_change().dropna()
        active_returns = returns[returns != 0]
        n_total_rets = len(returns)
        n_active = len(active_returns)

        # Annualization factor based on active bar frequency
        if n_total_rets > 0 and n_active > 0:
            active_bars_per_year = n_active * (bars_per_year / n_total_rets)
        else:
            active_bars_per_year = bars_per_year

        if n_active > 1 and active_returns.std() > 0:
            sharpe = (active_returns.mean() / active_returns.std()) * np.sqrt(active_bars_per_year)
        else:
            sharpe = 0.0

        neg_active = active_returns[active_returns < 0]
        if len(neg_active) > 1 and neg_active.std() > 0:
            sortino = (active_returns.mean() / neg_active.std()) * np.sqrt(active_bars_per_year)
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

        avg_mae = np.mean([t.mae for t in trades]) if trades else 0
        avg_mfe = np.mean([t.mfe for t in trades]) if trades else 0
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