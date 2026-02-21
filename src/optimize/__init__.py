"""
Optimization Module - v8
========================
Fixes from v7 audit:

- FIX #1: `full_data_results` replaces `best_results`.
  Renamed to make unambiguous that this is an in-sample-contaminated
  full-data run used only for UI display, NOT a performance estimate.
  All callers (app.py) updated accordingly.

- FIX #2: Efficiency ratio handles zero and negative IS correctly.
  Previously silently returned 0.0 when avg_train <= 0.
  Negative IS performance with positive OOS is a red flag of a
  different kind and should be surfaced, not silenced.
  Now uses abs(avg_train) > 1e-9 guard and allows negative ratio.

- FIX #3: Parameter stability warning splits entry vs exit params.
  Exit params (SL%, TP%, ATR mult, etc.) adapting between regimes is
  EXPECTED behaviour — different volatility regimes require different
  risk management. Flagging them as "unstable" was a false positive.
  Now only entry-signal params are checked for stability.

- FIX #4: `trials_per_fold` floor raised from 30 → 50.
  TPE needs ~20 random warmup trials before making intelligent
  suggestions. A floor of 30 with 10+ dimensions = random search.

- FIX #5: `WalkForwardFold.oos_equity` replaces `oos_results`.
  Storing full BacktestResults per fold (trades list + equity curve)
  wastes significant memory on large datasets.
  Only the equity curve is needed for stitching. `test_trades` count
  is already stored separately on the fold.

- FIX #6: Module-level `warnings.filterwarnings('ignore')` removed.
  Previously suppressed ALL Python warnings globally on import.
  Now scoped via context manager only around Optuna study.optimize().

- FIX #7: `timeout` parameter restored to `optimize()`.
  Was silently dropped from v7. Restored for direct BayesianOptimizer
  users who rely on time-bounded runs.

All v7 fixes retained:
  - Rolling/anchored window choice
  - OOS selection bias fix (last fold's params, not best OOS)
  - Stitched OOS equity curve
  - Efficiency ratio, param stability CV, failed trial %, warnings
  - Trial budget warnings
  - Exception counter
  - study.best_trial canonical usage
"""

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Install optuna: pip install optuna")

from ..strategy import StrategyParams, TradeDirection
from ..backtest import BacktestEngine, BacktestResults


# ─────────────────────────────────────────────────────────────────────────────
# Context manager for scoped warning suppression
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _suppress_warnings():
    """Suppress noisy warnings only during Optuna study execution."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Parameter classification
# ─────────────────────────────────────────────────────────────────────────────

# Entry-signal params — instability HERE means no reliable edge
_ENTRY_PARAM_PREFIXES = (
    'pamrp_length', 'pamrp_entry', 'pamrp_exit',
    'bbwp_length', 'bbwp_lookback', 'bbwp_sma', 'bbwp_threshold', 'bbwp_ma',
    'adx_length', 'adx_smoothing', 'adx_threshold',
    'ma_fast_length', 'ma_slow_length', 'ma_type',
    'rsi_length', 'rsi_oversold', 'rsi_overbought',
    'volume_ma_length', 'volume_multiplier',
    'supertrend_period', 'supertrend_multiplier',
    'macd_fast', 'macd_slow', 'macd_signal',
)

# Exit/risk params — adapting between regimes is EXPECTED, not a red flag
_EXIT_PARAM_PREFIXES = (
    'stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct',
    'atr_length', 'atr_multiplier',
    'time_exit_bars', 'ma_exit_fast', 'ma_exit_slow',
    'bbwp_exit_threshold',
)


def _is_entry_param(name: str) -> bool:
    return any(name.startswith(p) for p in _ENTRY_PARAM_PREFIXES)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardFold:
    """
    Single walk-forward fold result.

    v7: added best_params, oos_results.
    v8: oos_results → oos_equity (pd.Series only) to reduce memory footprint.
        Full BacktestResults per fold is wasteful; only the equity curve
        is needed for stitching.
    """
    fold_num: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_value: float
    test_value: float
    train_trades: int
    test_trades: int
    best_params: Optional[StrategyParams] = None
    oos_equity: Optional[pd.Series] = None   # v8: was oos_results: BacktestResults


@dataclass
class OptimizationResult:
    """
    Full result from one optimization run.

    v8 rename: best_results → full_data_results.
    Rationale: 'best_results' implied it was the authoritative performance
    view. It is not — it runs the deployment params on the FULL dataset
    including in-sample periods. Renamed to make this unambiguous.

    For WFO, the authoritative performance view is `stitched_equity`.
    For simple split, the authoritative view is the test-set metrics
    in `test_value`.

    Fields
    ------
    full_data_results   : BacktestResults on full dataset (IS-contaminated,
                          for UI display / parameter inspection only).
    stitched_equity     : Continuous OOS equity curve (WFO only).
                          The ONLY unbiased performance representation for WFO.
    efficiency_ratio    : avg_oos_metric / avg_train_metric.
                          Rule of thumb: < 0.5 → suspect overfit.
                          Negative → IS losing money but OOS gaining (unusual).
    param_stability_cv  : CV per ENTRY param across folds.
                          High CV → unstable entry signal.
                          Exit params excluded (adapting = expected behaviour).
    failed_trial_pct    : Fraction of Optuna trials that threw exceptions.
                          > 0.20 → investigate backtest engine.
    warnings            : Human-readable list of robustness warnings.
    window_type         : 'rolling', 'anchored', or 'simple' (informational).
    """
    best_params: Dict[str, Any]
    best_value: float
    full_data_results: BacktestResults          # v8: renamed from best_results
    all_trials: pd.DataFrame
    metric: str
    train_value: float = 0.0
    test_value: float = 0.0
    walkforward_folds: List[WalkForwardFold] = field(default_factory=list)
    initial_capital: float = 10000.0
    commission_pct: float = 0.1
    stitched_equity: Optional[pd.Series] = None
    efficiency_ratio: float = 0.0
    param_stability_cv: Dict[str, float] = field(default_factory=dict)
    failed_trial_pct: float = 0.0
    warnings: List[str] = field(default_factory=list)
    window_type: str = 'rolling'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_active_params(enabled_filters: Dict[str, bool]) -> int:
    """
    Estimate numeric parameters that will be optimized.
    Each enabled indicator contributes an approximate dimension count.
    """
    dims_per_indicator = {
        'pamrp_enabled': 5,
        'bbwp_enabled': 6,
        'adx_enabled': 3,
        'ma_trend_enabled': 3,
        'rsi_enabled': 3,
        'volume_enabled': 2,
        'supertrend_enabled': 2,
        'vwap_enabled': 0,
        'macd_enabled': 3,
        'stop_loss_enabled': 2,
        'take_profit_enabled': 2,
        'trailing_stop_enabled': 1,
        'atr_trailing_enabled': 2,
        'time_exit_enabled': 1,
        'ma_exit_enabled': 2,
        'bbwp_exit_enabled': 1,
        'pamrp_exit_enabled': 0,
        'stoch_rsi_exit_enabled': 0,
    }
    total = sum(
        dims_per_indicator.get(k, 2)
        for k, v in enabled_filters.items() if v
    )
    return max(total, 1)


def _count_enabled_indicators(enabled_filters: Dict[str, bool]) -> int:
    """Count distinct entry-filter indicators only."""
    entry_keys = {
        'pamrp_enabled', 'bbwp_enabled', 'adx_enabled', 'ma_trend_enabled',
        'rsi_enabled', 'volume_enabled', 'supertrend_enabled', 'vwap_enabled',
        'macd_enabled',
    }
    return sum(1 for k, v in enabled_filters.items() if k in entry_keys and v)


def _build_trial_budget_warnings(
    n_trials: int,
    enabled_filters: Dict[str, bool],
) -> List[str]:
    msgs = []
    active_dims = _count_active_params(enabled_filters)
    min_recommended = active_dims * 10
    n_indicators = _count_enabled_indicators(enabled_filters)

    if n_trials < min_recommended:
        msgs.append(
            f"Trial budget may be insufficient: {n_trials} trials for ~{active_dims} "
            f"active parameters. Recommend ≥{min_recommended} trials for TPE to converge. "
            f"Results may reflect random search rather than true optimization."
        )
    if n_indicators > 3:
        msgs.append(
            f"{n_indicators} entry indicators active. For reliable optimization, "
            f"use ≤3 indicators simultaneously. More indicators increase curve-fitting "
            f"risk without a genuine edge."
        )
    return msgs


def _stitch_oos_equity(
    folds: List[WalkForwardFold],
    initial_capital: float,
) -> pd.Series:
    """
    Concatenate per-fold OOS equity curves into a single continuous series.

    Each fold's curve is normalized to start where the previous fold ended,
    preserving compounding. This is the only unbiased WFO equity view.

    Since BacktestEngine always resets to initial_capital, each fold's
    equity curve starts at initial_capital. We normalize each segment to
    start at `base` (where the previous segment ended).
    """
    segments = []
    base = initial_capital

    for fold in folds:
        eq = fold.oos_equity
        if eq is None or len(eq) == 0:
            continue
        start_val = eq.iloc[0]
        if start_val <= 0:
            continue
        normalized = eq / start_val * base
        segments.append(normalized)
        base = float(normalized.iloc[-1])

    if not segments:
        return pd.Series(dtype=float)

    stitched = pd.concat(segments)
    stitched = stitched[~stitched.index.duplicated(keep='first')]
    return stitched


def _compute_param_stability(
    folds: List[WalkForwardFold],
) -> Dict[str, float]:
    """
    Compute coefficient of variation per ENTRY param across folds.

    Exit/risk params (SL%, TP%, etc.) are intentionally excluded — they
    are expected to adapt between volatility regimes and flagging them
    as unstable produces false-positive overfit warnings.

    Returns {} if fewer than 2 valid folds.
    """
    param_values: Dict[str, List[float]] = {}

    for fold in folds:
        if fold.best_params is None:
            continue
        p_dict = fold.best_params.to_dict()
        for k, v in p_dict.items():
            if not _is_entry_param(k):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                param_values.setdefault(k, []).append(float(v))

    cv_dict = {}
    for k, vals in param_values.items():
        if len(vals) < 2:
            continue
        mean = np.mean(vals)
        std = np.std(vals)
        if abs(mean) > 1e-9:
            cv_dict[k] = round(std / abs(mean), 4)
        else:
            cv_dict[k] = 0.0

    return cv_dict


def _build_robustness_warnings(
    efficiency_ratio: float,
    param_stability_cv: Dict[str, float],
    failed_trial_pct: float,
    base_warnings: List[str],
    has_oos_data: bool = True,
) -> List[str]:
    msgs = list(base_warnings)

    if has_oos_data and efficiency_ratio != 0.0:
        if efficiency_ratio < 0:
            msgs.append(
                f"Efficiency ratio is {efficiency_ratio:.2f}: in-sample performance was "
                f"negative but OOS was positive. This is unusual and warrants scrutiny — "
                f"the strategy may be exploiting a data artefact."
            )
        elif efficiency_ratio < 0.5:
            msgs.append(
                f"Efficiency ratio is {efficiency_ratio:.2f} (OOS/IS = {efficiency_ratio:.0%}). "
                f"Values below 0.50 suggest significant overfitting. Consider fewer "
                f"indicators or a larger dataset."
            )

    unstable = [k for k, v in param_stability_cv.items() if v > 0.5]
    if unstable:
        msgs.append(
            f"Unstable entry parameters across folds (CV > 0.50): "
            f"{', '.join(unstable[:8])}. "
            f"These parameters vary significantly between regimes — the strategy "
            f"may lack a consistent entry edge."
        )

    if failed_trial_pct > 0.20:
        msgs.append(
            f"{failed_trial_pct:.0%} of optimization trials raised exceptions. "
            f"This is abnormally high and may indicate a bug in the backtest engine, "
            f"data quality issues, or incompatible parameter combinations."
        )

    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Main optimizer
# ─────────────────────────────────────────────────────────────────────────────

class BayesianOptimizer:
    """
    Bayesian optimization using Optuna TPE with optional walk-forward validation.

    Walk-forward modes
    ------------------
    rolling  (default):
        Fixed-size sliding training window of `train_window_bars` bars.
        Discards old data. Preferred when markets are non-stationary.

    anchored:
        Expanding window starting from bar 0.
        Preferred when you believe the edge is stable across all history.

    Parameters
    ----------
    df : pd.DataFrame
    enabled_filters : dict[str, bool]
    metric : str
        BacktestResults attribute to optimize.
    min_trades : int
        Trials producing fewer trades are penalized to -inf.
    initial_capital, commission_pct : float
    trade_direction : str  — 'long_only' | 'short_only' | 'both'
    train_pct : float  — fraction for training in simple (non-WF) mode.
    use_walkforward : bool
    n_folds : int
    window_type : str  — 'rolling' or 'anchored'
    train_window_bars : int | None
        Rolling window size. Defaults to int(len(df) * train_pct).
    """

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
        n_folds: int = 5,
        window_type: str = 'rolling',
        train_window_bars: Optional[int] = None,
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
        self.window_type = window_type
        self.trade_direction_str = trade_direction
        self.trade_direction = {
            'long_only': TradeDirection.LONG_ONLY,
            'short_only': TradeDirection.SHORT_ONLY,
            'both': TradeDirection.BOTH,
        }.get(trade_direction, TradeDirection.LONG_ONLY)

        split_idx = int(len(df) * train_pct)
        self.train_df = df.iloc[:split_idx].copy()
        self.test_df = df.iloc[split_idx:].copy()
        self.train_window_bars = (
            train_window_bars if train_window_bars is not None else split_idx
        )

    # ── Backtest helpers ──────────────────────────────────────────────────────

    def _run_backtest(self, params: StrategyParams, df: pd.DataFrame) -> BacktestResults:
        engine = BacktestEngine(params, self.initial_capital, self.commission_pct)
        return engine.run(df.copy())

    def _get_metric(self, results: BacktestResults) -> float:
        if results.num_trades < self.min_trades:
            return float('-inf')
        value = getattr(results, self.metric, 0)
        if value is None or np.isnan(value) or np.isinf(value):
            return float('-inf')
        return float(value)

    # ── Parameter builder ─────────────────────────────────────────────────────

    def _build_params_from_trial(self, trial) -> StrategyParams:
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

    # ── Optuna study runner ───────────────────────────────────────────────────

    def _make_objective(self, train_data: pd.DataFrame, exception_counter: list):
        def objective(trial):
            try:
                params = self._build_params_from_trial(trial)
                results = self._run_backtest(params, train_data)
                return self._get_metric(results)
            except Exception as exc:
                exception_counter[0] += 1
                logger.debug("Trial %d raised: %s", trial.number, exc)
                return float('-inf')
        return objective

    def _optimize_on_data(
        self,
        train_data: pd.DataFrame,
        n_trials: int,
        seed: int = 42,
        show_progress: bool = False,
        timeout: Optional[float] = None,
    ) -> Tuple[Optional[StrategyParams], float]:
        """
        Run an Optuna TPE study on train_data.

        Returns (best_params, failed_trial_pct).
        best_params is None if no valid trials found.
        """
        exception_counter = [0]
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        objective = self._make_objective(train_data, exception_counter)

        with _suppress_warnings():
            study.optimize(
                objective, n_trials=n_trials, timeout=timeout,
                show_progress_bar=show_progress, n_jobs=1,
            )

        total = len(study.trials)
        failed_pct = exception_counter[0] / total if total > 0 else 0.0
        valid = [t for t in study.trials if t.value is not None and t.value > float('-inf')]

        if not valid:
            return None, failed_pct

        return self._build_params_from_trial(study.best_trial), failed_pct

    # ── Fold window construction ──────────────────────────────────────────────

    def _get_fold_windows(self) -> List[Tuple[int, int, int, int]]:
        """
        Compute (train_start, train_end, test_start, test_end) index tuples.

        Rolling: train window is a fixed-size sliding window.
        Anchored: train window expands from bar 0.
        """
        n = len(self.df)
        fold_size = n // self.n_folds
        windows = []

        for k in range(1, self.n_folds):
            train_end_idx = k * fold_size
            test_start_idx = train_end_idx
            test_end_idx = min((k + 1) * fold_size, n)

            if self.window_type == 'rolling':
                train_start_idx = max(0, train_end_idx - self.train_window_bars)
            else:
                train_start_idx = 0

            windows.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))

        return windows

    # ── Walk-forward ──────────────────────────────────────────────────────────

    def _run_walkforward(
        self, n_trials: int, show_progress: bool, timeout: Optional[float]
    ) -> OptimizationResult:
        """
        Walk-forward optimization.

        Algorithm
        ---------
        1. Divide full data into n_folds equal windows.
        2. For each fold k:
           a. Construct training window (rolling or anchored).
           b. Run Optuna on training window → best_params_k.
           c. Backtest best_params_k on OOS window → store oos_equity.
        3. Deployment candidate = LAST fold's params.
           Rationale: most temporally recent, most regime-relevant.
           NOT selected by OOS score (that would re-introduce selection bias).
        4. Stitch per-fold OOS equity curves → stitched_equity.
        5. Compute efficiency ratio, param stability, and warnings.

        Per-fold trial budget: n_trials divided across folds, floor of 50.
        """
        budget_warnings = _build_trial_budget_warnings(n_trials, self.enabled_filters)

        fold_windows = self._get_fold_windows()
        n_active_folds = len(fold_windows)
        # v8: floor raised from 30 → 50 (TPE needs ~20 warmup + meaningful exploration)
        trials_per_fold = max(50, n_trials // max(n_active_folds, 1))

        wf_folds: List[WalkForwardFold] = []
        oos_scores: List[float] = []
        all_failed_pcts: List[float] = []

        for k, (tr_start, tr_end, ts_start, ts_end) in enumerate(fold_windows, start=1):
            train_data = self.df.iloc[tr_start:tr_end]
            test_data = self.df.iloc[ts_start:ts_end]

            if len(test_data) < 20 or len(train_data) < 50:
                logger.debug("Fold %d skipped: insufficient data", k)
                continue

            fold_seed = 42 + k * 7
            fold_params, failed_pct = self._optimize_on_data(
                train_data, trials_per_fold, seed=fold_seed,
                show_progress=show_progress, timeout=timeout,
            )
            all_failed_pcts.append(failed_pct)

            if fold_params is None:
                logger.debug("Fold %d: no valid trials", k)
                continue

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
                train_value=train_score if train_score > float('-inf') else 0.0,
                test_value=test_score if test_score > float('-inf') else 0.0,
                train_trades=train_res.num_trades,
                test_trades=test_res.num_trades,
                best_params=fold_params,
                oos_equity=test_res.equity_curve,   # v8: store only equity curve
            ))

            if test_score > float('-inf'):
                oos_scores.append(test_score)

        if not wf_folds:
            logger.warning("Walk-forward produced no valid folds; falling back to simple split.")
            return self._run_simple_optimization(n_trials, show_progress, timeout)

        # FIX #2 (v7): Last fold's params as deployment candidate — not best OOS
        deployment_params = wf_folds[-1].best_params

        # Stitch OOS equity
        stitched_equity = _stitch_oos_equity(wf_folds, self.initial_capital)

        # Efficiency ratio — FIX #2 (v8): use abs() guard, allow negative
        avg_oos = float(np.mean(oos_scores)) if oos_scores else 0.0
        avg_train = float(np.mean([f.train_value for f in wf_folds])) if wf_folds else 0.0
        if abs(avg_train) > 1e-9:
            efficiency_ratio = avg_oos / avg_train
        else:
            efficiency_ratio = 0.0

        # Param stability — entry params only (v8)
        param_stability_cv = _compute_param_stability(wf_folds)

        avg_failed_pct = float(np.mean(all_failed_pcts)) if all_failed_pcts else 0.0

        # Full-data run for UI display (IS-contaminated, labeled accordingly)
        full_res = self._run_backtest(deployment_params, self.df)

        all_warnings = _build_robustness_warnings(
            efficiency_ratio, param_stability_cv, avg_failed_pct,
            budget_warnings, has_oos_data=bool(oos_scores),
        )

        final_dict = deployment_params.to_dict()
        final_dict['trade_direction_str'] = self.trade_direction_str

        return OptimizationResult(
            best_params=final_dict,
            best_value=avg_oos,
            full_data_results=full_res,
            all_trials=pd.DataFrame(),
            metric=self.metric,
            train_value=avg_train,
            test_value=avg_oos,
            walkforward_folds=wf_folds,
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            stitched_equity=stitched_equity,
            efficiency_ratio=efficiency_ratio,
            param_stability_cv=param_stability_cv,
            failed_trial_pct=avg_failed_pct,
            warnings=all_warnings,
            window_type=self.window_type,
        )

    # ── Simple train/test split ───────────────────────────────────────────────

    def _run_simple_optimization(
        self, n_trials: int, show_progress: bool, timeout: Optional[float]
    ) -> OptimizationResult:
        """Standard single train/test split optimization (non-WF)."""
        budget_warnings = _build_trial_budget_warnings(n_trials, self.enabled_filters)

        exception_counter = [0]
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        objective = self._make_objective(self.train_df, exception_counter)

        with _suppress_warnings():
            study.optimize(
                objective, n_trials=n_trials, timeout=timeout,
                show_progress_bar=show_progress, n_jobs=1,
            )

        total = len(study.trials)
        failed_pct = exception_counter[0] / total if total > 0 else 0.0

        valid = [t for t in study.trials if t.value is not None and t.value > float('-inf')]
        if not valid:
            all_warnings = _build_robustness_warnings(0.0, {}, failed_pct, budget_warnings, False)
            return OptimizationResult(
                best_params={}, best_value=0.0,
                full_data_results=BacktestResults(
                    trades=[], equity_curve=pd.Series(), realized_equity=pd.Series()),
                all_trials=pd.DataFrame(), metric=self.metric,
                initial_capital=self.initial_capital, commission_pct=self.commission_pct,
                failed_trial_pct=failed_pct, warnings=all_warnings, window_type='simple',
            )

        params = self._build_params_from_trial(study.best_trial)

        train_res = self._run_backtest(params, self.train_df)
        test_res = self._run_backtest(params, self.test_df)
        full_res = self._run_backtest(params, self.df)

        train_val = self._get_metric(train_res)
        test_val = self._get_metric(test_res)

        # FIX #2 (v8): abs() guard, allow negative ratio
        if abs(train_val) > 1e-9 and train_val > float('-inf') and test_val > float('-inf'):
            efficiency_ratio = test_val / train_val
        else:
            efficiency_ratio = 0.0

        trials_data = [
            dict(**t.params, value=t.value, trial_number=t.number)
            for t in valid
        ]
        trials_df = pd.DataFrame(trials_data) if trials_data else pd.DataFrame()

        all_warnings = _build_robustness_warnings(
            efficiency_ratio, {}, failed_pct, budget_warnings, has_oos_data=True,
        )

        fp = params.to_dict()
        fp['trade_direction_str'] = self.trade_direction_str

        return OptimizationResult(
            best_params=fp,
            best_value=float(study.best_trial.value),
            full_data_results=full_res,
            all_trials=trials_df,
            metric=self.metric,
            train_value=train_val if train_val > float('-inf') else 0.0,
            test_value=test_val if test_val > float('-inf') else 0.0,
            walkforward_folds=[],
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            efficiency_ratio=efficiency_ratio,
            failed_trial_pct=failed_pct,
            warnings=all_warnings,
            window_type='simple',
        )

    # ── Entry point ───────────────────────────────────────────────────────────

    def optimize(
        self,
        n_trials: int = 200,
        timeout: Optional[float] = None,   # v8: restored
        show_progress: bool = True,
    ) -> OptimizationResult:
        """Run optimization. Dispatches to WFO or simple split."""
        if self.use_walkforward:
            return self._run_walkforward(n_trials, show_progress, timeout)
        else:
            return self._run_simple_optimization(n_trials, show_progress, timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function (backward compatible)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_strategy(
    df: pd.DataFrame,
    enabled_filters: Dict[str, bool],
    metric: str = 'sharpe_ratio',
    n_trials: int = 200,
    min_trades: int = 10,
    initial_capital: float = 10000,
    commission_pct: float = 1,
    trade_direction: str = 'long_only',
    train_pct: float = 0.7,
    use_walkforward: bool = True,
    n_folds: int = 5,
    show_progress: bool = True,
    window_type: str = 'rolling',
    train_window_bars: Optional[int] = None,
    timeout: Optional[float] = None,
) -> OptimizationResult:
    """
    Convenience wrapper around BayesianOptimizer.

    Parameters added in v7/v8
    -------------------------
    window_type : 'rolling' (default) or 'anchored'
    train_window_bars : int | None — rolling window size in bars
    timeout : float | None — per-study time limit in seconds

    All other parameters are backward compatible with v6.
    """
    opt = BayesianOptimizer(
        df=df,
        enabled_filters=enabled_filters,
        metric=metric,
        min_trades=min_trades,
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        trade_direction=trade_direction,
        train_pct=train_pct,
        use_walkforward=use_walkforward,
        n_folds=n_folds,
        window_type=window_type,
        train_window_bars=train_window_bars,
    )
    return opt.optimize(n_trials=n_trials, timeout=timeout, show_progress=show_progress)