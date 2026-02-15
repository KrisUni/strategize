"""
Monte Carlo Module
==================
Three simulation methods:
1. Trade Shuffle    – permute trade execution order
2. Return Bootstrap – resample daily returns with replacement
3. Noise Injection  – add Gaussian noise to trade P&L (stress test)

Outputs:
- Equity path distribution (for confidence band charts)
- Final equity distribution
- Risk of ruin probability
- Maximum drawdown distribution
- Confidence intervals for key metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo analysis result."""
    # Method used
    method: str

    # Equity paths: shape (n_simulations, n_steps)
    equity_paths: np.ndarray

    # Final equity per simulation
    final_equities: np.ndarray

    # Max drawdown per simulation (as negative %)
    max_drawdowns: np.ndarray

    # Risk of ruin: probability of equity falling below threshold
    risk_of_ruin: float          # P(equity < ruin_level)
    ruin_level: float            # Dollar level used

    # Percentile summaries
    equity_percentiles: Dict[str, float]   # 5th, 25th, 50th, 75th, 95th
    dd_percentiles: Dict[str, float]

    # Confidence bands for equity paths at each time step
    # Each key is a percentile string, value is array of length n_steps
    equity_bands: Dict[str, np.ndarray]

    # Sharpe distribution (if method is return_bootstrap)
    sharpe_distribution: Optional[np.ndarray] = None

    n_simulations: int = 0
    initial_capital: float = 10000.0


def _equity_from_pnls(pnls: np.ndarray, initial_capital: float) -> np.ndarray:
    """Build equity curve from P&L sequence. Returns array of length len(pnls)+1."""
    equity = np.empty(len(pnls) + 1)
    equity[0] = initial_capital
    np.cumsum(pnls, out=equity[1:])
    equity[1:] += initial_capital
    return equity


def _max_drawdown_pct(equity: np.ndarray) -> float:
    """Compute maximum drawdown as a negative percentage."""
    peak = np.maximum.accumulate(equity)
    # Avoid division by zero
    safe_peak = np.where(peak > 0, peak, 1.0)
    dd_pct = (equity - peak) / safe_peak * 100
    return dd_pct.min()


def trade_shuffle(
    trades: List,
    n_simulations: int = 1000,
    initial_capital: float = 10000,
    ruin_pct: float = 50.0,
    seed: int = 42
) -> MonteCarloResult:
    """
    Monte Carlo via trade order permutation.
    
    Shuffles the sequence of trade P&L values to explore how
    sensitive the equity path is to trade ordering.
    
    This tests: "Would a different sequence of the same trades
    have caused ruin or a much worse drawdown?"
    
    Args:
        trades: List of Trade objects (must have .pnl attribute)
        n_simulations: Number of permutations to run
        initial_capital: Starting equity
        ruin_pct: Ruin defined as losing this % of capital
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl for t in trades])
    n_trades = len(pnls)

    if n_trades < 2:
        empty = np.array([[initial_capital]])
        return MonteCarloResult(
            method='trade_shuffle', equity_paths=empty,
            final_equities=np.array([initial_capital]),
            max_drawdowns=np.array([0.0]),
            risk_of_ruin=0.0, ruin_level=initial_capital * (1 - ruin_pct / 100),
            equity_percentiles={'5%': initial_capital, '25%': initial_capital,
                                '50%': initial_capital, '75%': initial_capital,
                                '95%': initial_capital},
            dd_percentiles={'5%': 0, '50%': 0, '95%': 0},
            equity_bands={}, n_simulations=0,
            initial_capital=initial_capital,
        )

    ruin_level = initial_capital * (1 - ruin_pct / 100)

    # Pre-allocate: each path has n_trades+1 points (including start)
    paths = np.empty((n_simulations, n_trades + 1))
    finals = np.empty(n_simulations)
    dds = np.empty(n_simulations)

    for sim in range(n_simulations):
        shuffled = pnls.copy()
        rng.shuffle(shuffled)
        eq = _equity_from_pnls(shuffled, initial_capital)
        paths[sim] = eq
        finals[sim] = eq[-1]
        dds[sim] = _max_drawdown_pct(eq)

    # Risk of ruin: fraction of sims where equity ever dipped below ruin level
    ruin_count = 0
    for sim in range(n_simulations):
        if paths[sim].min() <= ruin_level:
            ruin_count += 1
    risk_of_ruin = ruin_count / n_simulations

    # Percentile bands at each time step
    bands = {}
    for pct_label, pct_val in [('5%', 5), ('25%', 25), ('50%', 50),
                                ('75%', 75), ('95%', 95)]:
        bands[pct_label] = np.percentile(paths, pct_val, axis=0)

    return MonteCarloResult(
        method='trade_shuffle',
        equity_paths=paths,
        final_equities=finals,
        max_drawdowns=dds,
        risk_of_ruin=risk_of_ruin,
        ruin_level=ruin_level,
        equity_percentiles={
            '5%': np.percentile(finals, 5),
            '25%': np.percentile(finals, 25),
            '50%': np.percentile(finals, 50),
            '75%': np.percentile(finals, 75),
            '95%': np.percentile(finals, 95),
        },
        dd_percentiles={
            '5%': np.percentile(dds, 5),
            '50%': np.percentile(dds, 50),
            '95%': np.percentile(dds, 95),
        },
        equity_bands=bands,
        n_simulations=n_simulations,
        initial_capital=initial_capital,
    )


def return_bootstrap(
    equity_curve: pd.Series,
    n_simulations: int = 1000,
    initial_capital: float = 10000,
    ruin_pct: float = 50.0,
    block_size: int = 5,
    seed: int = 42,
    bars_per_year: int = 252
) -> MonteCarloResult:
    """
    Monte Carlo via block bootstrap of returns.
    
    Resamples blocks of returns (preserving short-term autocorrelation)
    to generate synthetic equity paths of the same length.
    
    This tests: "Given the observed return distribution, what range
    of outcomes could we expect?"
    
    Args:
        equity_curve: Mark-to-market equity curve
        n_simulations: Number of bootstrap samples
        initial_capital: Starting equity
        ruin_pct: Ruin threshold as % of capital
        block_size: Block length for block bootstrap (preserves autocorrelation)
        seed: Random seed
        bars_per_year: For Sharpe calculation
    """
    rng = np.random.default_rng(seed)
    returns = equity_curve.pct_change().dropna().values
    n_returns = len(returns)

    if n_returns < block_size * 2:
        # Fall back to simple bootstrap if not enough data for blocks
        block_size = 1

    ruin_level = initial_capital * (1 - ruin_pct / 100)

    paths = np.empty((n_simulations, n_returns + 1))
    finals = np.empty(n_simulations)
    dds = np.empty(n_simulations)
    sharpes = np.empty(n_simulations)

    for sim in range(n_simulations):
        # Block bootstrap: sample blocks of consecutive returns
        sampled = np.empty(n_returns)
        idx = 0
        while idx < n_returns:
            start = rng.integers(0, max(1, n_returns - block_size + 1))
            end = min(start + block_size, n_returns)
            chunk = returns[start:end]
            take = min(len(chunk), n_returns - idx)
            sampled[idx:idx + take] = chunk[:take]
            idx += take

        # Build equity from bootstrapped returns
        eq = np.empty(n_returns + 1)
        eq[0] = initial_capital
        for j in range(n_returns):
            eq[j + 1] = eq[j] * (1 + sampled[j])

        paths[sim] = eq
        finals[sim] = eq[-1]
        dds[sim] = _max_drawdown_pct(eq)

        # Sharpe for this sim
        std = sampled.std()
        sharpes[sim] = (sampled.mean() / std * np.sqrt(bars_per_year)) if std > 0 else 0

    # Risk of ruin
    ruin_count = sum(1 for sim in range(n_simulations) if paths[sim].min() <= ruin_level)
    risk_of_ruin = ruin_count / n_simulations

    bands = {}
    for pct_label, pct_val in [('5%', 5), ('25%', 25), ('50%', 50),
                                ('75%', 75), ('95%', 95)]:
        bands[pct_label] = np.percentile(paths, pct_val, axis=0)

    return MonteCarloResult(
        method='return_bootstrap',
        equity_paths=paths,
        final_equities=finals,
        max_drawdowns=dds,
        risk_of_ruin=risk_of_ruin,
        ruin_level=ruin_level,
        equity_percentiles={
            '5%': np.percentile(finals, 5),
            '25%': np.percentile(finals, 25),
            '50%': np.percentile(finals, 50),
            '75%': np.percentile(finals, 75),
            '95%': np.percentile(finals, 95),
        },
        dd_percentiles={
            '5%': np.percentile(dds, 5),
            '50%': np.percentile(dds, 50),
            '95%': np.percentile(dds, 95),
        },
        equity_bands=bands,
        sharpe_distribution=sharpes,
        n_simulations=n_simulations,
        initial_capital=initial_capital,
    )


def noise_injection(
    trades: List,
    n_simulations: int = 1000,
    initial_capital: float = 10000,
    noise_pct: float = 20.0,
    ruin_pct: float = 50.0,
    seed: int = 42
) -> MonteCarloResult:
    """
    Monte Carlo via noise injection on trade P&L.
    
    Adds Gaussian noise proportional to each trade's P&L magnitude.
    This stress-tests: "What if our fills were slightly different,
    or market conditions caused small deviations in each trade?"
    
    Args:
        trades: List of Trade objects
        n_simulations: Number of simulations
        initial_capital: Starting equity
        noise_pct: Noise magnitude as % of each trade's |P&L|
        ruin_pct: Ruin threshold
        seed: Random seed
    """
    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl for t in trades])
    n_trades = len(pnls)

    if n_trades < 2:
        empty = np.array([[initial_capital]])
        return MonteCarloResult(
            method='noise_injection', equity_paths=empty,
            final_equities=np.array([initial_capital]),
            max_drawdowns=np.array([0.0]),
            risk_of_ruin=0.0, ruin_level=initial_capital * (1 - ruin_pct / 100),
            equity_percentiles={'5%': initial_capital, '25%': initial_capital,
                                '50%': initial_capital, '75%': initial_capital,
                                '95%': initial_capital},
            dd_percentiles={'5%': 0, '50%': 0, '95%': 0},
            equity_bands={}, n_simulations=0,
            initial_capital=initial_capital,
        )

    ruin_level = initial_capital * (1 - ruin_pct / 100)
    noise_scale = noise_pct / 100.0

    paths = np.empty((n_simulations, n_trades + 1))
    finals = np.empty(n_simulations)
    dds = np.empty(n_simulations)

    abs_pnls = np.abs(pnls)
    # Use median |pnl| as noise base for zero-pnl trades
    median_abs = np.median(abs_pnls[abs_pnls > 0]) if (abs_pnls > 0).any() else 1.0

    for sim in range(n_simulations):
        noise_base = np.where(abs_pnls > 0, abs_pnls, median_abs)
        noise = rng.normal(0, noise_base * noise_scale)
        noisy_pnls = pnls + noise
        eq = _equity_from_pnls(noisy_pnls, initial_capital)
        paths[sim] = eq
        finals[sim] = eq[-1]
        dds[sim] = _max_drawdown_pct(eq)

    ruin_count = sum(1 for sim in range(n_simulations) if paths[sim].min() <= ruin_level)
    risk_of_ruin = ruin_count / n_simulations

    bands = {}
    for pct_label, pct_val in [('5%', 5), ('25%', 25), ('50%', 50),
                                ('75%', 75), ('95%', 95)]:
        bands[pct_label] = np.percentile(paths, pct_val, axis=0)

    return MonteCarloResult(
        method='noise_injection',
        equity_paths=paths,
        final_equities=finals,
        max_drawdowns=dds,
        risk_of_ruin=risk_of_ruin,
        ruin_level=ruin_level,
        equity_percentiles={
            '5%': np.percentile(finals, 5),
            '25%': np.percentile(finals, 25),
            '50%': np.percentile(finals, 50),
            '75%': np.percentile(finals, 75),
            '95%': np.percentile(finals, 95),
        },
        dd_percentiles={
            '5%': np.percentile(dds, 5),
            '50%': np.percentile(dds, 50),
            '95%': np.percentile(dds, 95),
        },
        equity_bands=bands,
        n_simulations=n_simulations,
        initial_capital=initial_capital,
    )


def run_monte_carlo(
    trades: List,
    equity_curve: Optional[pd.Series] = None,
    method: str = 'trade_shuffle',
    n_simulations: int = 1000,
    initial_capital: float = 10000,
    ruin_pct: float = 50.0,
    noise_pct: float = 20.0,
    block_size: int = 5,
    bars_per_year: int = 252,
    seed: int = 42
) -> Optional[MonteCarloResult]:
    """
    Unified entry point for Monte Carlo simulation.
    
    Args:
        trades: List of Trade objects
        equity_curve: Mark-to-market equity (required for return_bootstrap)
        method: 'trade_shuffle', 'return_bootstrap', or 'noise_injection'
        n_simulations: Number of simulations
        initial_capital: Starting capital
        ruin_pct: % loss threshold for ruin calculation
        noise_pct: Noise level for noise_injection method
        block_size: Block size for return_bootstrap
        bars_per_year: Annualization factor
        seed: Random seed
    """
    if not trades or len(trades) < 3:
        return None

    if method == 'trade_shuffle':
        return trade_shuffle(trades, n_simulations, initial_capital, ruin_pct, seed)
    elif method == 'return_bootstrap':
        if equity_curve is None or len(equity_curve) < 20:
            return None
        return return_bootstrap(
            equity_curve, n_simulations, initial_capital,
            ruin_pct, block_size, seed, bars_per_year
        )
    elif method == 'noise_injection':
        return noise_injection(
            trades, n_simulations, initial_capital,
            noise_pct, ruin_pct, seed
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'trade_shuffle', 'return_bootstrap', or 'noise_injection'.")