"""
Optimization Module - v5 Fixed
==============================
- Properly passes capital/commission
- Walk-forward fold results for visualization
- Only optimizes enabled filters
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Install optuna: pip install optuna")

from ..strategy import StrategyParams, TradeDirection
from ..backtest import BacktestEngine, BacktestResults


@dataclass
class WalkForwardFold:
    """Single walk-forward fold result"""
    fold_num: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_value: float
    test_value: float
    train_trades: int
    test_trades: int


@dataclass
class OptimizationResult:
    best_params: Dict[str, Any]
    best_value: float
    best_results: BacktestResults
    all_trials: pd.DataFrame
    metric: str
    train_value: float = 0.0
    test_value: float = 0.0
    walkforward_folds: List[WalkForwardFold] = field(default_factory=list)
    # Settings used - CRITICAL for matching backtest
    initial_capital: float = 10000.0
    commission_pct: float = 0.1


class BayesianOptimizer:
    """Bayesian optimization with consistent settings"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        enabled_filters: Dict[str, bool],
        metric: str = 'sharpe_ratio',
        min_trades: int = 10,
        initial_capital: float = 10000,
        commission_pct: float = 0.1,
        trade_direction: str = 'long_only',
        train_pct: float = 0.7,
        use_walkforward: bool = False,
        n_folds: int = 5
    ):
        self.df = df
        self.enabled_filters = enabled_filters
        self.metric = metric
        self.min_trades = min_trades
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.train_pct = train_pct
        self.use_walkforward = use_walkforward
        self.n_folds = n_folds
        
        self.trade_direction_str = trade_direction
        self.trade_direction = {
            'long_only': TradeDirection.LONG_ONLY,
            'short_only': TradeDirection.SHORT_ONLY,
            'both': TradeDirection.BOTH
        }.get(trade_direction, TradeDirection.LONG_ONLY)
        
        split_idx = int(len(df) * train_pct)
        self.train_df = df.iloc[:split_idx].copy()
        self.test_df = df.iloc[split_idx:].copy()
        
        # Store walk-forward results
        self.wf_folds = []
    
    def _run_backtest(self, params: StrategyParams, df: pd.DataFrame) -> BacktestResults:
        # Use SAME capital and commission as will be used in UI
        engine = BacktestEngine(params, self.initial_capital, self.commission_pct)
        return engine.run(df.copy())
    
    def _get_metric(self, results: BacktestResults) -> float:
        if results.num_trades < self.min_trades:
            return float('-inf')
        value = getattr(results, self.metric, 0)
        if value is None or np.isnan(value) or np.isinf(value):
            return float('-inf')
        return value
    
    def _build_params_from_trial(self, trial: optuna.Trial) -> StrategyParams:
        """Build params - only optimize enabled filters"""
        ef = self.enabled_filters
        
        # PAMRP
        pamrp_enabled = ef.get('pamrp_enabled', True)
        if pamrp_enabled:
            pamrp_length = trial.suggest_int('pamrp_length', 10, 30)
            if self.trade_direction in [TradeDirection.LONG_ONLY, TradeDirection.BOTH]:
                pamrp_entry_long = trial.suggest_int('pamrp_entry_long', 10, 40)
                pamrp_exit_long = trial.suggest_int('pamrp_exit_long', 55, 90)
            else:
                pamrp_entry_long, pamrp_exit_long = 20, 70
            if self.trade_direction in [TradeDirection.SHORT_ONLY, TradeDirection.BOTH]:
                pamrp_entry_short = trial.suggest_int('pamrp_entry_short', 60, 90)
                pamrp_exit_short = trial.suggest_int('pamrp_exit_short', 10, 45)
            else:
                pamrp_entry_short, pamrp_exit_short = 80, 30
        else:
            pamrp_length, pamrp_entry_long, pamrp_entry_short = 21, 20, 80
            pamrp_exit_long, pamrp_exit_short = 70, 30
        
        # BBWP
        bbwp_enabled = ef.get('bbwp_enabled', True)
        if bbwp_enabled:
            bbwp_length = trial.suggest_int('bbwp_length', 8, 21)
            bbwp_lookback = trial.suggest_int('bbwp_lookback', 100, 300)
            bbwp_sma_length = trial.suggest_int('bbwp_sma_length', 3, 10)
            bbwp_ma_filter = trial.suggest_categorical('bbwp_ma_filter', ['disabled', 'decreasing'])
            if self.trade_direction in [TradeDirection.LONG_ONLY, TradeDirection.BOTH]:
                bbwp_threshold_long = trial.suggest_int('bbwp_threshold_long', 30, 70)
            else:
                bbwp_threshold_long = 50
            if self.trade_direction in [TradeDirection.SHORT_ONLY, TradeDirection.BOTH]:
                bbwp_threshold_short = trial.suggest_int('bbwp_threshold_short', 30, 70)
            else:
                bbwp_threshold_short = 50
        else:
            bbwp_length, bbwp_lookback, bbwp_sma_length = 13, 252, 5
            bbwp_threshold_long, bbwp_threshold_short = 50, 50
            bbwp_ma_filter = 'disabled'
        
        # ADX
        adx_enabled = ef.get('adx_enabled', False)
        if adx_enabled:
            adx_length = trial.suggest_int('adx_length', 10, 20)
            adx_smoothing = trial.suggest_int('adx_smoothing', 10, 20)
            adx_threshold = trial.suggest_int('adx_threshold', 15, 35)
        else:
            adx_length, adx_smoothing, adx_threshold = 14, 14, 20
        
        # MA Trend
        ma_trend_enabled = ef.get('ma_trend_enabled', False)
        if ma_trend_enabled:
            ma_fast_length = trial.suggest_int('ma_fast_length', 20, 100)
            ma_slow_length = trial.suggest_int('ma_slow_length', 100, 300)
            ma_type = trial.suggest_categorical('ma_type', ['sma', 'ema'])
        else:
            ma_fast_length, ma_slow_length, ma_type = 50, 200, 'sma'
        
        # RSI
        rsi_enabled = ef.get('rsi_enabled', False)
        if rsi_enabled:
            rsi_length = trial.suggest_int('rsi_length', 7, 21)
            rsi_oversold = trial.suggest_int('rsi_oversold', 20, 40)
            rsi_overbought = trial.suggest_int('rsi_overbought', 60, 80)
        else:
            rsi_length, rsi_oversold, rsi_overbought = 14, 30, 70
        
        # Volume
        volume_enabled = ef.get('volume_enabled', False)
        if volume_enabled:
            volume_ma_length = trial.suggest_int('volume_ma_length', 10, 30)
            volume_multiplier = trial.suggest_float('volume_multiplier', 0.8, 1.5)
        else:
            volume_ma_length, volume_multiplier = 20, 1.0
        
        # Supertrend
        supertrend_enabled = ef.get('supertrend_enabled', False)
        if supertrend_enabled:
            supertrend_period = trial.suggest_int('supertrend_period', 7, 14)
            supertrend_multiplier = trial.suggest_float('supertrend_multiplier', 2.0, 4.0)
        else:
            supertrend_period, supertrend_multiplier = 10, 3.0
        
        vwap_enabled = ef.get('vwap_enabled', False)
        
        # MACD
        macd_enabled = ef.get('macd_enabled', False)
        if macd_enabled:
            macd_fast = trial.suggest_int('macd_fast', 8, 15)
            macd_slow = trial.suggest_int('macd_slow', 20, 30)
            macd_signal = trial.suggest_int('macd_signal', 6, 12)
        else:
            macd_fast, macd_slow, macd_signal = 12, 26, 9
        
        # Exits
        stop_loss_enabled = ef.get('stop_loss_enabled', True)
        if stop_loss_enabled:
            stop_loss_pct_long = trial.suggest_float('stop_loss_pct_long', 1.0, 10.0)
            stop_loss_pct_short = trial.suggest_float('stop_loss_pct_short', 1.0, 10.0)
        else:
            stop_loss_pct_long, stop_loss_pct_short = 3.0, 3.0
        
        take_profit_enabled = ef.get('take_profit_enabled', False)
        if take_profit_enabled:
            take_profit_pct_long = trial.suggest_float('take_profit_pct_long', 2.0, 15.0)
            take_profit_pct_short = trial.suggest_float('take_profit_pct_short', 2.0, 15.0)
        else:
            take_profit_pct_long, take_profit_pct_short = 5.0, 5.0
        
        trailing_stop_enabled = ef.get('trailing_stop_enabled', False)
        trailing_stop_pct = trial.suggest_float('trailing_stop_pct', 1.0, 5.0) if trailing_stop_enabled else 2.0
        
        atr_trailing_enabled = ef.get('atr_trailing_enabled', False)
        if atr_trailing_enabled:
            atr_length = trial.suggest_int('atr_length', 10, 20)
            atr_multiplier = trial.suggest_float('atr_multiplier', 1.5, 4.0)
        else:
            atr_length, atr_multiplier = 14, 2.0
        
        pamrp_exit_enabled = ef.get('pamrp_exit_enabled', True)
        stoch_rsi_exit_enabled = ef.get('stoch_rsi_exit_enabled', False)
        
        time_exit_enabled = ef.get('time_exit_enabled', False)
        time_exit_bars = trial.suggest_int('time_exit_bars', 10, 50) if time_exit_enabled else 20
        
        ma_exit_enabled = ef.get('ma_exit_enabled', False)
        if ma_exit_enabled:
            ma_exit_fast = trial.suggest_int('ma_exit_fast', 5, 15)
            ma_exit_slow = trial.suggest_int('ma_exit_slow', 15, 30)
        else:
            ma_exit_fast, ma_exit_slow = 10, 20
        
        bbwp_exit_enabled = ef.get('bbwp_exit_enabled', False)
        bbwp_exit_threshold = trial.suggest_int('bbwp_exit_threshold', 70, 90) if bbwp_exit_enabled else 80
        
        # Ensure exit
        has_exit = any([stop_loss_enabled, take_profit_enabled, trailing_stop_enabled,
                       atr_trailing_enabled, pamrp_exit_enabled, stoch_rsi_exit_enabled,
                       time_exit_enabled, ma_exit_enabled, bbwp_exit_enabled])
        if not has_exit:
            pamrp_exit_enabled = True
        
        return StrategyParams(
            trade_direction=self.trade_direction,
            pamrp_enabled=pamrp_enabled,
            pamrp_length=pamrp_length,
            pamrp_entry_long=pamrp_entry_long,
            pamrp_entry_short=pamrp_entry_short,
            pamrp_exit_long=pamrp_exit_long,
            pamrp_exit_short=pamrp_exit_short,
            bbwp_enabled=bbwp_enabled,
            bbwp_length=bbwp_length,
            bbwp_lookback=bbwp_lookback,
            bbwp_sma_length=bbwp_sma_length,
            bbwp_threshold_long=bbwp_threshold_long,
            bbwp_threshold_short=bbwp_threshold_short,
            bbwp_ma_filter=bbwp_ma_filter,
            adx_enabled=adx_enabled,
            adx_length=adx_length,
            adx_smoothing=adx_smoothing,
            adx_threshold=adx_threshold,
            ma_trend_enabled=ma_trend_enabled,
            ma_fast_length=ma_fast_length,
            ma_slow_length=ma_slow_length,
            ma_type=ma_type,
            rsi_enabled=rsi_enabled,
            rsi_length=rsi_length,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            volume_enabled=volume_enabled,
            volume_ma_length=volume_ma_length,
            volume_multiplier=volume_multiplier,
            supertrend_enabled=supertrend_enabled,
            supertrend_period=supertrend_period,
            supertrend_multiplier=supertrend_multiplier,
            vwap_enabled=vwap_enabled,
            macd_enabled=macd_enabled,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            stop_loss_enabled=stop_loss_enabled,
            stop_loss_pct_long=stop_loss_pct_long,
            stop_loss_pct_short=stop_loss_pct_short,
            take_profit_enabled=take_profit_enabled,
            take_profit_pct_long=take_profit_pct_long,
            take_profit_pct_short=take_profit_pct_short,
            trailing_stop_enabled=trailing_stop_enabled,
            trailing_stop_pct=trailing_stop_pct,
            atr_trailing_enabled=atr_trailing_enabled,
            atr_length=atr_length,
            atr_multiplier=atr_multiplier,
            pamrp_exit_enabled=pamrp_exit_enabled,
            stoch_rsi_exit_enabled=stoch_rsi_exit_enabled,
            time_exit_enabled=time_exit_enabled,
            time_exit_bars_long=time_exit_bars,
            time_exit_bars_short=time_exit_bars,
            ma_exit_enabled=ma_exit_enabled,
            ma_exit_fast=ma_exit_fast,
            ma_exit_slow=ma_exit_slow,
            bbwp_exit_enabled=bbwp_exit_enabled,
            bbwp_exit_threshold=bbwp_exit_threshold,
        )
    
    def _objective(self, trial: optuna.Trial) -> float:
        try:
            params = self._build_params_from_trial(trial)
            
            if self.use_walkforward:
                return self._walkforward_objective(params)
            else:
                results = self._run_backtest(params, self.train_df)
                return self._get_metric(results)
                
        except Exception as e:
            return float('-inf')
    
    def _walkforward_objective(self, params: StrategyParams) -> float:
        fold_size = len(self.train_df) // self.n_folds
        scores = []
        for i in range(self.n_folds - 1):
            train_end = (i + 1) * fold_size
            test_end = (i + 2) * fold_size
            test_data = self.train_df.iloc[train_end:test_end]
            if len(test_data) < 20:
                continue
            results = self._run_backtest(params, test_data)
            score = self._get_metric(results)
            if score > float('-inf'):
                scores.append(score)
        return np.mean(scores) if scores else float('-inf')
    
    def optimize(self, n_trials: int = 200, timeout: Optional[int] = None, show_progress: bool = True) -> OptimizationResult:
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(self._objective, n_trials=n_trials, timeout=timeout, 
                      show_progress_bar=show_progress, n_jobs=1)
        
        valid_trials = [t for t in study.trials if t.value is not None and t.value > float('-inf')]
        
        if not valid_trials:
            return OptimizationResult(
                best_params={}, best_value=0,
                best_results=BacktestResults(trades=[], equity_curve=pd.Series()),
                all_trials=pd.DataFrame(), metric=self.metric,
                initial_capital=self.initial_capital,
                commission_pct=self.commission_pct
            )
        
        best_trial = max(valid_trials, key=lambda t: t.value)
        
        # Rebuild best params
        params = self._build_params_from_trial(best_trial)
        
        train_results = self._run_backtest(params, self.train_df)
        test_results = self._run_backtest(params, self.test_df)
        full_results = self._run_backtest(params, self.df)
        
        train_value = self._get_metric(train_results)
        test_value = self._get_metric(test_results)
        
        # Walk-forward fold visualization data
        wf_folds = []
        if self.use_walkforward:
            fold_size = len(self.df) // self.n_folds
            for i in range(self.n_folds - 1):
                train_start = 0
                train_end = (i + 1) * fold_size
                test_start = train_end
                test_end = (i + 2) * fold_size
                
                train_data = self.df.iloc[train_start:train_end]
                test_data = self.df.iloc[test_start:test_end]
                
                if len(test_data) < 20:
                    continue
                
                tr = self._run_backtest(params, train_data)
                te = self._run_backtest(params, test_data)
                
                wf_folds.append(WalkForwardFold(
                    fold_num=i+1,
                    train_start=train_data.index[0],
                    train_end=train_data.index[-1],
                    test_start=test_data.index[0],
                    test_end=test_data.index[-1],
                    train_value=self._get_metric(tr),
                    test_value=self._get_metric(te),
                    train_trades=tr.num_trades,
                    test_trades=te.num_trades
                ))
        
        trials_data = []
        for trial in valid_trials:
            trial_data = trial.params.copy()
            trial_data['value'] = trial.value
            trial_data['trial_number'] = trial.number
            trials_data.append(trial_data)
        
        trials_df = pd.DataFrame(trials_data) if trials_data else pd.DataFrame()
        
        final_params = params.to_dict()
        final_params['trade_direction_str'] = self.trade_direction_str
        
        return OptimizationResult(
            best_params=final_params,
            best_value=best_trial.value,
            best_results=full_results,
            all_trials=trials_df,
            metric=self.metric,
            train_value=train_value if train_value > float('-inf') else 0,
            test_value=test_value if test_value > float('-inf') else 0,
            walkforward_folds=wf_folds,
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct
        )


def optimize_strategy(
    df: pd.DataFrame,
    enabled_filters: Dict[str, bool],
    metric: str = 'sharpe_ratio',
    n_trials: int = 200,
    min_trades: int = 10,
    initial_capital: float = 10000,
    commission_pct: float = 0.1,
    trade_direction: str = 'long_only',
    train_pct: float = 0.7,
    use_walkforward: bool = False,
    n_folds: int = 5,
    show_progress: bool = True
) -> OptimizationResult:
    optimizer = BayesianOptimizer(
        df=df, enabled_filters=enabled_filters, metric=metric, min_trades=min_trades,
        initial_capital=initial_capital, commission_pct=commission_pct,
        trade_direction=trade_direction, train_pct=train_pct,
        use_walkforward=use_walkforward, n_folds=n_folds
    )
    return optimizer.optimize(n_trials=n_trials, show_progress=show_progress)
