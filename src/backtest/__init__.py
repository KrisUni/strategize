"""
Backtest Module - v5 Fixed
==========================
- Stop loss checks high/low (not just close)
- Position sizing support
- Proper time exit tracking
- Fixed profit factor division by zero
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from ..strategy import StrategyParams, SignalGenerator, TradeDirection


@dataclass
class Trade:
    """Single trade record"""
    entry_idx: int
    entry_date: pd.Timestamp
    entry_price: float
    direction: str  # 'long' or 'short'
    position_size: float = 1.0  # Position size as fraction
    exit_idx: Optional[int] = None
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0


@dataclass
class BacktestResults:
    """Backtest results"""
    trades: List[Trade]
    equity_curve: pd.Series
    total_return: float = 0.0
    total_return_pct: float = 0.0
    num_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    avg_trade: float = 0.0
    avg_bars_held: float = 0.0
    max_consecutive_losses: int = 0
    # Settings used (for comparison)
    initial_capital: float = 10000.0
    commission_pct: float = 0.1


class BacktestEngine:
    """Backtest engine with proper stop loss and position sizing"""
    
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
    
    def _calculate_position_size(self, capital: float, price: float, win_rate: float = 0.5) -> float:
        """Calculate position size based on settings"""
        p = self.params
        
        if p.use_kelly and win_rate > 0:
            # Kelly criterion: f = (bp - q) / b
            # where b = win/loss ratio, p = win prob, q = 1-p
            avg_win = p.take_profit_pct_long if p.take_profit_enabled else 5.0
            avg_loss = p.stop_loss_pct_long if p.stop_loss_enabled else 3.0
            b = avg_win / avg_loss if avg_loss > 0 else 1.0
            q = 1 - win_rate
            kelly = (b * win_rate - q) / b if b > 0 else 0
            kelly = max(0, min(kelly, 1))  # Clamp 0-1
            position_pct = kelly * p.kelly_fraction * 100
        else:
            position_pct = p.position_size_pct
        
        position_pct = min(position_pct, 100.0)
        return (capital * position_pct / 100) / price
    
    def run(self, df: pd.DataFrame) -> BacktestResults:
        """Run backtest"""
        p = self.params
        
        # Generate signals
        df = self.signal_gen.generate_all_signals(df)
        
        trades = []
        equity = [self.initial_capital]
        capital = self.initial_capital
        position = None
        bars_in_trade = 0
        trailing_stop_price = None
        highest_since_entry = 0
        lowest_since_entry = float('inf')
        
        # Track win rate for Kelly
        recent_wins = 0
        recent_trades = 0
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            
            if position is not None:
                bars_in_trade += 1
                entry_price = position.entry_price
                
                # Track extremes for trailing stop
                if position.direction == 'long':
                    highest_since_entry = max(highest_since_entry, row['high'])
                    current_price = row['close']
                else:
                    lowest_since_entry = min(lowest_since_entry, row['low'])
                    current_price = row['close']
                
                exit_price = None
                exit_reason = None
                
                # ═══════════════════════════════════════════════════════════════
                # STOP LOSS - Check HIGH/LOW not just close!
                # ═══════════════════════════════════════════════════════════════
                if p.stop_loss_enabled and exit_price is None:
                    if position.direction == 'long':
                        stop_price = entry_price * (1 - p.stop_loss_pct_long / 100)
                        if row['low'] <= stop_price:  # LOW breaches stop
                            exit_price = stop_price  # Execute at stop level
                            exit_reason = 'stop_loss'
                    else:
                        stop_price = entry_price * (1 + p.stop_loss_pct_short / 100)
                        if row['high'] >= stop_price:  # HIGH breaches stop
                            exit_price = stop_price
                            exit_reason = 'stop_loss'
                
                # ═══════════════════════════════════════════════════════════════
                # TAKE PROFIT - Check HIGH/LOW
                # ═══════════════════════════════════════════════════════════════
                if p.take_profit_enabled and exit_price is None:
                    if position.direction == 'long':
                        tp_price = entry_price * (1 + p.take_profit_pct_long / 100)
                        if row['high'] >= tp_price:
                            exit_price = tp_price
                            exit_reason = 'take_profit'
                    else:
                        tp_price = entry_price * (1 - p.take_profit_pct_short / 100)
                        if row['low'] <= tp_price:
                            exit_price = tp_price
                            exit_reason = 'take_profit'
                
                # ═══════════════════════════════════════════════════════════════
                # TRAILING STOP
                # ═══════════════════════════════════════════════════════════════
                if p.trailing_stop_enabled and exit_price is None:
                    if position.direction == 'long':
                        activation_price = entry_price * (1 + p.trailing_stop_activation / 100)
                        if highest_since_entry >= activation_price:
                            trail_stop = highest_since_entry * (1 - p.trailing_stop_pct / 100)
                            if row['low'] <= trail_stop:
                                exit_price = trail_stop
                                exit_reason = 'trailing_stop'
                    else:
                        activation_price = entry_price * (1 - p.trailing_stop_activation / 100)
                        if lowest_since_entry <= activation_price:
                            trail_stop = lowest_since_entry * (1 + p.trailing_stop_pct / 100)
                            if row['high'] >= trail_stop:
                                exit_price = trail_stop
                                exit_reason = 'trailing_stop'
                
                # ═══════════════════════════════════════════════════════════════
                # ATR TRAILING
                # ═══════════════════════════════════════════════════════════════
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
                
                # ═══════════════════════════════════════════════════════════════
                # TIME EXIT - Proper tracking
                # ═══════════════════════════════════════════════════════════════
                if p.time_exit_enabled and exit_price is None:
                    max_bars = p.time_exit_bars_long if position.direction == 'long' else p.time_exit_bars_short
                    if bars_in_trade >= max_bars:
                        exit_price = row['close']
                        exit_reason = 'time_exit'
                
                # ═══════════════════════════════════════════════════════════════
                # SIGNAL EXIT
                # ═══════════════════════════════════════════════════════════════
                if exit_price is None:
                    if position.direction == 'long' and row['exit_long_signal']:
                        exit_price = row['close']
                        exit_reason = 'signal'
                    elif position.direction == 'short' and row['exit_short_signal']:
                        exit_price = row['close']
                        exit_reason = 'signal'
                
                # ═══════════════════════════════════════════════════════════════
                # EXECUTE EXIT
                # ═══════════════════════════════════════════════════════════════
                if exit_price is not None:
                    # Apply slippage
                    if position.direction == 'long':
                        exit_price *= (1 - self.slippage_pct / 100)
                    else:
                        exit_price *= (1 + self.slippage_pct / 100)
                    
                    # Calculate PnL
                    if position.direction == 'long':
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price * 100
                    
                    # Apply commission
                    pnl_pct -= self.commission_pct * 2  # Entry + exit
                    
                    # Position-sized PnL
                    trade_value = capital * position.position_size
                    pnl = trade_value * pnl_pct / 100
                    
                    position.exit_idx = i
                    position.exit_date = row.name
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason
                    position.pnl = pnl
                    position.pnl_pct = pnl_pct
                    position.bars_held = bars_in_trade
                    
                    trades.append(position)
                    capital += pnl
                    
                    # Update win rate tracking
                    recent_trades += 1
                    if pnl > 0:
                        recent_wins += 1
                    
                    position = None
                    bars_in_trade = 0
                    highest_since_entry = 0
                    lowest_since_entry = float('inf')
            
            # ═══════════════════════════════════════════════════════════════════
            # ENTRY SIGNALS
            # ═══════════════════════════════════════════════════════════════════
            if position is None:
                win_rate = recent_wins / recent_trades if recent_trades > 0 else 0.5
                
                if prev_row['entry_long']:
                    entry_price = row['open'] * (1 + self.slippage_pct / 100)
                    pos_size = self._calculate_position_size(capital, entry_price, win_rate)
                    position = Trade(
                        entry_idx=i,
                        entry_date=row.name,
                        entry_price=entry_price,
                        direction='long',
                        position_size=min(pos_size * entry_price / capital, 1.0)
                    )
                    highest_since_entry = row['high']
                    lowest_since_entry = row['low']
                    
                elif prev_row['entry_short']:
                    entry_price = row['open'] * (1 - self.slippage_pct / 100)
                    pos_size = self._calculate_position_size(capital, entry_price, win_rate)
                    position = Trade(
                        entry_idx=i,
                        entry_date=row.name,
                        entry_price=entry_price,
                        direction='short',
                        position_size=min(pos_size * entry_price / capital, 1.0)
                    )
                    highest_since_entry = row['high']
                    lowest_since_entry = row['low']
            
            equity.append(capital)
        
        # Create equity curve
        equity_curve = pd.Series(equity, index=df.index[:len(equity)])
        
        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve)
    
    def _calculate_metrics(self, trades: List[Trade], equity_curve: pd.Series) -> BacktestResults:
        """Calculate all metrics"""
        num_trades = len(trades)
        
        if num_trades == 0:
            return BacktestResults(
                trades=trades,
                equity_curve=equity_curve,
                total_return=0,
                total_return_pct=0,
                initial_capital=self.initial_capital,
                commission_pct=self.commission_pct
            )
        
        # Basic stats
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        
        total_return = equity_curve.iloc[-1] - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        win_rate = len(winners) / num_trades * 100 if num_trades > 0 else 0
        
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
        
        # FIXED: Profit factor - handle division by zero
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float('inf')  # All winners, no losers
        else:
            profit_factor = 0
        
        # Clamp profit factor for display
        profit_factor = min(profit_factor, 999.99)
        
        # Averages
        avg_winner = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loser = np.mean([t.pnl for t in losers]) if losers else 0
        avg_trade = np.mean([t.pnl for t in trades])
        avg_bars = np.mean([t.bars_held for t in trades])
        
        # Drawdown
        peak = equity_curve.expanding().max()
        drawdown = equity_curve - peak
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / peak.max()) * 100 if peak.max() > 0 else 0
        
        # Returns for Sharpe/Sortino
        returns = equity_curve.pct_change().dropna()
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Sortino ratio (only downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 1 and negative_returns.std() > 0:
            sortino = (returns.mean() / negative_returns.std()) * np.sqrt(252)
        else:
            sortino = sharpe  # Fall back to Sharpe if no negative returns
        
        # Max consecutive losses
        max_consec_loss = 0
        current_consec = 0
        for t in trades:
            if t.pnl <= 0:
                current_consec += 1
                max_consec_loss = max(max_consec_loss, current_consec)
            else:
                current_consec = 0
        
        return BacktestResults(
            trades=trades,
            equity_curve=equity_curve,
            total_return=total_return,
            total_return_pct=total_return_pct,
            num_trades=num_trades,
            winners=len(winners),
            losers=len(losers),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            avg_trade=avg_trade,
            avg_bars_held=avg_bars,
            max_consecutive_losses=max_consec_loss,
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct
        )
