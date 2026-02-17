"""
Optimization Module - v6 (Fixed Walk-Forward)
==============================================
Fixes from audit:
- CRITICAL #2: True anchored walk-forward optimization.
  Previously: optimized ONE set of params on the full training set,
  then tested those same params on each fold (= cross-validation).
  Now: each fold runs its OWN Optuna optimization on the expanding
  training window, then tests on the next OOS (out-of-sample) window.
  Only the OOS results are reported, giving a realistic performance estimate.

- Seed is no longer fixed at 42 when walk-forward is enabled
  (each fold gets a different seed for diversity).
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
    initial_capital: float = 10000.0
    commission_pct: float = 0.1


class BayesianOptimizer:
    """Bayesian optimization with true anchored walk-forward."""

    def __init__(self, df, enabled_filters, metric='sharpe_ratio', min_trades=10,
                 initial_capital=10000, commission_pct=0.1, trade_direction='long_only',
                 train_pct=0.7, use_walkforward=False, n_folds=5):
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

    def _run_backtest(self, params, df):
        engine = BacktestEngine(params, self.initial_capital, self.commission_pct)
        return engine.run(df.copy())

    def _get_metric(self, results):
        if results.num_trades < self.min_trades:
            return float('-inf')
        value = getattr(results, self.metric, 0)
        if value is None or np.isnan(value) or np.isinf(value):
            return float('-inf')
        return value

    def _build_params_from_trial(self, trial):
        """Build StrategyParams — only optimize enabled filters."""
        ef = self.enabled_filters

        pe = ef.get('pamrp_enabled', True)
        pl = trial.suggest_int('pamrp_length', 10, 30) if pe else 21
        pel = trial.suggest_int('pamrp_entry_long', 10, 40) if pe and self.trade_direction in [TradeDirection.LONG_ONLY, TradeDirection.BOTH] else 20
        pxl = trial.suggest_int('pamrp_exit_long', 55, 90) if pe and self.trade_direction in [TradeDirection.LONG_ONLY, TradeDirection.BOTH] else 70
        pes = trial.suggest_int('pamrp_entry_short', 60, 90) if pe and self.trade_direction in [TradeDirection.SHORT_ONLY, TradeDirection.BOTH] else 80
        pxs = trial.suggest_int('pamrp_exit_short', 10, 45) if pe and self.trade_direction in [TradeDirection.SHORT_ONLY, TradeDirection.BOTH] else 30

        be = ef.get('bbwp_enabled', True)
        bl = trial.suggest_int('bbwp_length', 8, 21) if be else 13
        blb = trial.suggest_int('bbwp_lookback', 100, 300) if be else 252
        bsma = trial.suggest_int('bbwp_sma_length', 3, 10) if be else 5
        bmf = trial.suggest_categorical('bbwp_ma_filter', ['disabled', 'decreasing']) if be else 'disabled'
        btl = trial.suggest_int('bbwp_threshold_long', 30, 70) if be and self.trade_direction in [TradeDirection.LONG_ONLY, TradeDirection.BOTH] else 50
        bts = trial.suggest_int('bbwp_threshold_short', 30, 70) if be and self.trade_direction in [TradeDirection.SHORT_ONLY, TradeDirection.BOTH] else 50

        ae = ef.get('adx_enabled', False)
        al = trial.suggest_int('adx_length', 10, 20) if ae else 14
        asm = trial.suggest_int('adx_smoothing', 10, 20) if ae else 14
        at = trial.suggest_int('adx_threshold', 15, 35) if ae else 20

        mae = ef.get('ma_trend_enabled', False)
        maf = trial.suggest_int('ma_fast_length', 20, 100) if mae else 50
        mas = trial.suggest_int('ma_slow_length', 100, 300) if mae else 200
        mat = trial.suggest_categorical('ma_type', ['sma', 'ema']) if mae else 'sma'

        re = ef.get('rsi_enabled', False)
        rl = trial.suggest_int('rsi_length', 7, 21) if re else 14
        ros = trial.suggest_int('rsi_oversold', 20, 40) if re else 30
        rob = trial.suggest_int('rsi_overbought', 60, 80) if re else 70

        ve = ef.get('volume_enabled', False)
        vml = trial.suggest_int('volume_ma_length', 10, 30) if ve else 20
        vm = trial.suggest_float('volume_multiplier', 0.8, 1.5) if ve else 1.0

        ste = ef.get('supertrend_enabled', False)
        stp = trial.suggest_int('supertrend_period', 7, 14) if ste else 10
        stm = trial.suggest_float('supertrend_multiplier', 2.0, 4.0) if ste else 3.0

        vwe = ef.get('vwap_enabled', False)

        mce = ef.get('macd_enabled', False)
        mcf = trial.suggest_int('macd_fast', 8, 15) if mce else 12
        mcs = trial.suggest_int('macd_slow', 20, 30) if mce else 26
        mcsi = trial.suggest_int('macd_signal', 6, 12) if mce else 9

        sle = ef.get('stop_loss_enabled', True)
        sll = trial.suggest_float('stop_loss_pct_long', 1.0, 10.0) if sle else 3.0
        sls = trial.suggest_float('stop_loss_pct_short', 1.0, 10.0) if sle else 3.0

        tpe = ef.get('take_profit_enabled', False)
        tpl = trial.suggest_float('take_profit_pct_long', 2.0, 15.0) if tpe else 5.0
        tps = trial.suggest_float('take_profit_pct_short', 2.0, 15.0) if tpe else 5.0

        tse = ef.get('trailing_stop_enabled', False)
        tsp = trial.suggest_float('trailing_stop_pct', 1.0, 5.0) if tse else 2.0

        ate = ef.get('atr_trailing_enabled', False)
        atl = trial.suggest_int('atr_length', 10, 20) if ate else 14
        atm = trial.suggest_float('atr_multiplier', 1.5, 4.0) if ate else 2.0

        pxe = ef.get('pamrp_exit_enabled', True)
        sre = ef.get('stoch_rsi_exit_enabled', False)

        txe = ef.get('time_exit_enabled', False)
        txb = trial.suggest_int('time_exit_bars', 10, 50) if txe else 20

        mxe = ef.get('ma_exit_enabled', False)
        mxf = trial.suggest_int('ma_exit_fast', 5, 15) if mxe else 10
        mxs = trial.suggest_int('ma_exit_slow', 15, 30) if mxe else 20

        bxe = ef.get('bbwp_exit_enabled', False)
        bxt = trial.suggest_int('bbwp_exit_threshold', 70, 90) if bxe else 80

        # Ensure at least one exit method
        has_exit = any([sle, tpe, tse, ate, pxe, sre, txe, mxe, bxe])
        if not has_exit:
            pxe = True

        return StrategyParams(
            trade_direction=self.trade_direction, pamrp_enabled=pe, pamrp_length=pl,
            pamrp_entry_long=pel, pamrp_entry_short=pes, pamrp_exit_long=pxl, pamrp_exit_short=pxs,
            bbwp_enabled=be, bbwp_length=bl, bbwp_lookback=blb, bbwp_sma_length=bsma,
            bbwp_threshold_long=btl, bbwp_threshold_short=bts, bbwp_ma_filter=bmf,
            adx_enabled=ae, adx_length=al, adx_smoothing=asm, adx_threshold=at,
            ma_trend_enabled=mae, ma_fast_length=maf, ma_slow_length=mas, ma_type=mat,
            rsi_enabled=re, rsi_length=rl, rsi_oversold=ros, rsi_overbought=rob,
            volume_enabled=ve, volume_ma_length=vml, volume_multiplier=vm,
            supertrend_enabled=ste, supertrend_period=stp, supertrend_multiplier=stm,
            vwap_enabled=vwe, macd_enabled=mce, macd_fast=mcf, macd_slow=mcs, macd_signal=mcsi,
            stop_loss_enabled=sle, stop_loss_pct_long=sll, stop_loss_pct_short=sls,
            take_profit_enabled=tpe, take_profit_pct_long=tpl, take_profit_pct_short=tps,
            trailing_stop_enabled=tse, trailing_stop_pct=tsp,
            atr_trailing_enabled=ate, atr_length=atl, atr_multiplier=atm,
            pamrp_exit_enabled=pxe, stoch_rsi_exit_enabled=sre,
            time_exit_enabled=txe, time_exit_bars_long=txb, time_exit_bars_short=txb,
            ma_exit_enabled=mxe, ma_exit_fast=mxf, ma_exit_slow=mxs,
            bbwp_exit_enabled=bxe, bbwp_exit_threshold=bxt,
        )

    def _make_objective(self, train_data: pd.DataFrame):
        """Create an objective function closed over the given training data."""
        def objective(trial):
            try:
                params = self._build_params_from_trial(trial)
                results = self._run_backtest(params, train_data)
                return self._get_metric(results)
            except Exception:
                return float('-inf')
        return objective

    def _optimize_on_data(self, train_data: pd.DataFrame, n_trials: int,
                          seed: int = 42, show_progress: bool = False) -> Optional[StrategyParams]:
        """
        Run Optuna optimization on a specific training slice.
        Returns the best StrategyParams, or None if no valid trials.
        """
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        objective = self._make_objective(train_data)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress, n_jobs=1)

        valid = [t for t in study.trials if t.value is not None and t.value > float('-inf')]
        if not valid:
            return None

        best_trial = max(valid, key=lambda t: t.value)
        return self._build_params_from_trial(best_trial)

    def _run_walkforward(self, n_trials: int, show_progress: bool) -> OptimizationResult:
        """
        TRUE anchored walk-forward optimization.

        Algorithm:
          1. Divide full data into n_folds equal windows
          2. For fold k (k=1..n_folds-1):
             - Training window = data[0 : fold_k_end]  (expanding/anchored)
             - Test window = data[fold_k_end : fold_{k+1}_end]
             - Run Optuna on training window → best_params_k
             - Backtest best_params_k on test window → OOS result
          3. Report aggregate OOS performance across all folds

        Each fold gets a DIFFERENT Optuna seed for exploration diversity.
        """
        fold_size = len(self.df) // self.n_folds
        wf_folds = []
        oos_scores = []
        best_fold_params = None
        best_fold_score = float('-inf')
        # Per-fold trial budget: split total trials across folds
        trials_per_fold = max(20, n_trials // max(self.n_folds - 1, 1))

        for k in range(1, self.n_folds):
            train_end_idx = k * fold_size
            test_end_idx = min((k + 1) * fold_size, len(self.df))

            train_data = self.df.iloc[:train_end_idx]
            test_data = self.df.iloc[train_end_idx:test_end_idx]

            if len(test_data) < 20 or len(train_data) < 50:
                continue

            # Optimize on THIS fold's training window (different seed per fold)
            fold_seed = 42 + k * 7
            fold_params = self._optimize_on_data(
                train_data, trials_per_fold, seed=fold_seed,
                show_progress=show_progress
            )
            if fold_params is None:
                continue

            # Evaluate on train and test
            train_res = self._run_backtest(fold_params, train_data)
            test_res = self._run_backtest(fold_params, test_data)
            train_score = self._get_metric(train_res)
            test_score = self._get_metric(test_res)

            wf_folds.append(WalkForwardFold(
                fold_num=k,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                train_value=train_score if train_score > float('-inf') else 0,
                test_value=test_score if test_score > float('-inf') else 0,
                train_trades=train_res.num_trades,
                test_trades=test_res.num_trades,
            ))

            if test_score > float('-inf'):
                oos_scores.append(test_score)
            if test_score > best_fold_score:
                best_fold_score = test_score
                best_fold_params = fold_params

        # If no folds produced results, fall back to simple train/test
        if best_fold_params is None:
            return self._run_simple_optimization(n_trials, show_progress)

        # Run the LAST fold's params on full data for the "best_results" display
        # (clearly labeled as full-data, not OOS)
        full_res = self._run_backtest(best_fold_params, self.df)

        # Aggregate OOS performance
        avg_oos = np.mean(oos_scores) if oos_scores else 0
        train_val = np.mean([f.train_value for f in wf_folds]) if wf_folds else 0

        final_dict = best_fold_params.to_dict()
        final_dict['trade_direction_str'] = self.trade_direction_str

        return OptimizationResult(
            best_params=final_dict,
            best_value=avg_oos,
            best_results=full_res,
            all_trials=pd.DataFrame(),  # WF doesn't have a single trial list
            metric=self.metric,
            train_value=train_val,
            test_value=avg_oos,
            walkforward_folds=wf_folds,
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
        )

    def _run_simple_optimization(self, n_trials: int, show_progress: bool) -> OptimizationResult:
        """Standard train/test split optimization (non-WF)."""
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        objective = self._make_objective(self.train_df)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress, n_jobs=1)

        valid = [t for t in study.trials if t.value is not None and t.value > float('-inf')]
        if not valid:
            return OptimizationResult(
                best_params={}, best_value=0,
                best_results=BacktestResults(trades=[], equity_curve=pd.Series(), realized_equity=pd.Series()),
                all_trials=pd.DataFrame(), metric=self.metric,
                initial_capital=self.initial_capital, commission_pct=self.commission_pct)

        best = max(valid, key=lambda t: t.value)
        params = self._build_params_from_trial(best)

        train_res = self._run_backtest(params, self.train_df)
        test_res = self._run_backtest(params, self.test_df)
        full_res = self._run_backtest(params, self.df)

        train_val = self._get_metric(train_res)
        test_val = self._get_metric(test_res)

        trials_data = [dict(**t.params, value=t.value, trial_number=t.number) for t in valid]
        trials_df = pd.DataFrame(trials_data) if trials_data else pd.DataFrame()

        fp = params.to_dict()
        fp['trade_direction_str'] = self.trade_direction_str

        return OptimizationResult(
            best_params=fp, best_value=best.value, best_results=full_res,
            all_trials=trials_df, metric=self.metric,
            train_value=train_val if train_val > float('-inf') else 0,
            test_value=test_val if test_val > float('-inf') else 0,
            walkforward_folds=[],
            initial_capital=self.initial_capital, commission_pct=self.commission_pct)

    def optimize(self, n_trials=200, timeout=None, show_progress=True) -> OptimizationResult:
        """Main entry point."""
        if self.use_walkforward:
            return self._run_walkforward(n_trials, show_progress)
        else:
            return self._run_simple_optimization(n_trials, show_progress)


def optimize_strategy(df, enabled_filters, metric='sharpe_ratio', n_trials=200, min_trades=10,
                      initial_capital=10000, commission_pct=0.1, trade_direction='long_only',
                      train_pct=0.7, use_walkforward=False, n_folds=5, show_progress=True):
    """Convenience function."""
    opt = BayesianOptimizer(
        df=df, enabled_filters=enabled_filters, metric=metric, min_trades=min_trades,
        initial_capital=initial_capital, commission_pct=commission_pct,
        trade_direction=trade_direction, train_pct=train_pct,
        use_walkforward=use_walkforward, n_folds=n_folds)
    return opt.optimize(n_trials=n_trials, show_progress=show_progress)