"""
Trading Toolkit - Premium UI v5.1
==================================
Fixed: sidebar toggle, walk-forward chart, price chart with trades,
       optimization message, apply params sync.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import random

from src.data import fetch_yfinance, generate_sample_data
from src.strategy import StrategyParams, SignalGenerator, TradeDirection
from src.backtest import BacktestEngine, BacktestResults
from src.optimize import optimize_strategy

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Trading Toolkit Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — NO custom sidebar JS toggle (use Streamlit's native arrow)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    :root {
        --bg-primary: #0a0e14;
        --bg-secondary: #11151c;
        --bg-tertiary: #1a1f2e;
        --border-color: #2d3548;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --gradient-primary: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    }

    .stApp {
        background: var(--bg-primary);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #0a0e14 100%) !important;
    }

    [data-testid="stSidebar"] h1 {
        font-size: 1.2rem !important;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 0.7rem !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.6rem !important;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid var(--border-color);
    }

    .main .block-container {
        padding: 1rem 1.5rem;
        max-width: 100%;
    }

    .main h1 { font-size: 1.4rem !important; }
    .main h3 { font-size: 1rem !important; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-secondary);
        border-radius: 6px;
        padding: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 32px;
        padding: 0 12px;
        font-size: 0.8rem;
    }

    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
        border-radius: 4px;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #12161f 0%, #0d1117 100%);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.4rem;
    }

    [data-testid="stMetricLabel"] { font-size: 0.6rem !important; }
    [data-testid="stMetricValue"] { font-size: 1rem !important; }

    [data-testid="stHorizontalBlock"] {
        gap: 0.4rem;
        margin-bottom: 0.3rem;
    }

    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 5px;
        font-size: 0.8rem;
    }

    .streamlit-expanderHeader {
        font-size: 0.8rem !important;
        padding: 0.4rem 0.6rem !important;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 0.6rem 0;
    }

    /* Responsive: Tablet (768px - 1024px) */
    @media (max-width: 1024px) {
        .main .block-container {
            padding: 0.75rem 1rem;
        }
        [data-testid="column"] {
            min-width: 120px !important;
        }
        [data-testid="stMetric"] {
            padding: 0.3rem 0.5rem;
        }
        [data-testid="stMetricLabel"] { font-size: 0.55rem !important; }
        [data-testid="stMetricValue"] { font-size: 0.85rem !important; }
        .stTabs [data-baseweb="tab"] {
            padding: 0 8px;
            font-size: 0.7rem;
        }
    }

    /* Responsive: Small tablet / large phone (< 768px) */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem 0.5rem;
        }
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        [data-testid="stMetric"] {
            padding: 0.25rem 0.4rem;
            margin-bottom: 0.25rem;
        }
        [data-testid="stMetricLabel"] { font-size: 0.5rem !important; }
        [data-testid="stMetricValue"] { font-size: 0.75rem !important; }
        .stTabs [data-baseweb="tab"] {
            padding: 0 6px;
            font-size: 0.65rem;
        }
        .main h1 { font-size: 1.1rem !important; }
        .main h3 { font-size: 0.85rem !important; }
    }

    /* Hide menu & footer but NOT header — header contains the sidebar re-expand arrow */
    #MainMenu, footer {visibility: hidden;}
    /* Hide just the decorative header bar, not the collapse controls */
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def get_default_params() -> Dict[str, Any]:
    return {
        'trade_direction': 'Long Only',
        'position_size_pct': 100.0, 'use_kelly': False, 'kelly_fraction': 0.5,
        'pamrp_enabled': True, 'pamrp_length': 21, 'pamrp_entry_long': 20, 'pamrp_entry_short': 80,
        'pamrp_exit_long': 70, 'pamrp_exit_short': 30,
        'bbwp_enabled': True, 'bbwp_length': 13, 'bbwp_lookback': 252, 'bbwp_sma_length': 5,
        'bbwp_threshold_long': 50, 'bbwp_threshold_short': 50, 'bbwp_ma_filter': 'disabled',
        'adx_enabled': False, 'adx_length': 14, 'adx_smoothing': 14, 'adx_threshold': 20,
        'ma_trend_enabled': False, 'ma_fast_length': 50, 'ma_slow_length': 200, 'ma_type': 'sma',
        'rsi_enabled': False, 'rsi_length': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
        'volume_enabled': False, 'volume_ma_length': 20, 'volume_multiplier': 1.0,
        'supertrend_enabled': False, 'supertrend_period': 10, 'supertrend_multiplier': 3.0,
        'vwap_enabled': False,
        'macd_enabled': False, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
        'stop_loss_enabled': True, 'stop_loss_pct_long': 3.0, 'stop_loss_pct_short': 3.0,
        'take_profit_enabled': False, 'take_profit_pct_long': 5.0, 'take_profit_pct_short': 5.0,
        'trailing_stop_enabled': False, 'trailing_stop_pct': 2.0,
        'atr_trailing_enabled': False, 'atr_length': 14, 'atr_multiplier': 2.0,
        'pamrp_exit_enabled': True,
        'stoch_rsi_exit_enabled': False, 'stoch_rsi_length': 14, 'stoch_rsi_k': 3, 'stoch_rsi_d': 3,
        'stoch_rsi_overbought': 80, 'stoch_rsi_oversold': 20,
        'time_exit_enabled': False, 'time_exit_bars': 20,
        'ma_exit_enabled': False, 'ma_exit_fast': 10, 'ma_exit_slow': 20,
        'bbwp_exit_enabled': False, 'bbwp_exit_threshold': 80,
    }

if 'params' not in st.session_state:
    st.session_state.params = get_default_params()
if 'df' not in st.session_state:
    st.session_state.df = None
if 'multi_df' not in st.session_state:
    st.session_state.multi_df = {}
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'capital' not in st.session_state:
    st.session_state.capital = 10000
if 'commission' not in st.session_state:
    st.session_state.commission = 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def params_to_strategy(p: Dict) -> StrategyParams:
    direction_map = {'Long Only': TradeDirection.LONG_ONLY, 'Short Only': TradeDirection.SHORT_ONLY, 'Both': TradeDirection.BOTH}
    return StrategyParams(
        trade_direction=direction_map.get(p.get('trade_direction', 'Long Only'), TradeDirection.LONG_ONLY),
        position_size_pct=p.get('position_size_pct', 100.0),
        use_kelly=p.get('use_kelly', False),
        kelly_fraction=p.get('kelly_fraction', 0.5),
        pamrp_enabled=p.get('pamrp_enabled', True),
        pamrp_length=p.get('pamrp_length', 21),
        pamrp_entry_long=p.get('pamrp_entry_long', 20),
        pamrp_entry_short=p.get('pamrp_entry_short', 80),
        pamrp_exit_long=p.get('pamrp_exit_long', 70),
        pamrp_exit_short=p.get('pamrp_exit_short', 30),
        bbwp_enabled=p.get('bbwp_enabled', True),
        bbwp_length=p.get('bbwp_length', 13),
        bbwp_lookback=p.get('bbwp_lookback', 252),
        bbwp_sma_length=p.get('bbwp_sma_length', 5),
        bbwp_threshold_long=p.get('bbwp_threshold_long', 50),
        bbwp_threshold_short=p.get('bbwp_threshold_short', 50),
        bbwp_ma_filter=p.get('bbwp_ma_filter', 'disabled'),
        adx_enabled=p.get('adx_enabled', False),
        adx_length=p.get('adx_length', 14),
        adx_smoothing=p.get('adx_smoothing', 14),
        adx_threshold=p.get('adx_threshold', 20),
        ma_trend_enabled=p.get('ma_trend_enabled', False),
        ma_fast_length=p.get('ma_fast_length', 50),
        ma_slow_length=p.get('ma_slow_length', 200),
        ma_type=p.get('ma_type', 'sma'),
        rsi_enabled=p.get('rsi_enabled', False),
        rsi_length=p.get('rsi_length', 14),
        rsi_oversold=p.get('rsi_oversold', 30),
        rsi_overbought=p.get('rsi_overbought', 70),
        volume_enabled=p.get('volume_enabled', False),
        volume_ma_length=p.get('volume_ma_length', 20),
        volume_multiplier=p.get('volume_multiplier', 1.0),
        supertrend_enabled=p.get('supertrend_enabled', False),
        supertrend_period=p.get('supertrend_period', 10),
        supertrend_multiplier=p.get('supertrend_multiplier', 3.0),
        vwap_enabled=p.get('vwap_enabled', False),
        macd_enabled=p.get('macd_enabled', False),
        macd_fast=p.get('macd_fast', 12),
        macd_slow=p.get('macd_slow', 26),
        macd_signal=p.get('macd_signal', 9),
        stop_loss_enabled=p.get('stop_loss_enabled', True),
        stop_loss_pct_long=p.get('stop_loss_pct_long', 3.0),
        stop_loss_pct_short=p.get('stop_loss_pct_short', 3.0),
        take_profit_enabled=p.get('take_profit_enabled', False),
        take_profit_pct_long=p.get('take_profit_pct_long', 5.0),
        take_profit_pct_short=p.get('take_profit_pct_short', 5.0),
        trailing_stop_enabled=p.get('trailing_stop_enabled', False),
        trailing_stop_pct=p.get('trailing_stop_pct', 2.0),
        atr_trailing_enabled=p.get('atr_trailing_enabled', False),
        atr_length=p.get('atr_length', 14),
        atr_multiplier=p.get('atr_multiplier', 2.0),
        pamrp_exit_enabled=p.get('pamrp_exit_enabled', True),
        stoch_rsi_exit_enabled=p.get('stoch_rsi_exit_enabled', False),
        stoch_rsi_length=p.get('stoch_rsi_length', 14),
        stoch_rsi_k=p.get('stoch_rsi_k', 3),
        stoch_rsi_d=p.get('stoch_rsi_d', 3),
        stoch_rsi_overbought=p.get('stoch_rsi_overbought', 80),
        stoch_rsi_oversold=p.get('stoch_rsi_oversold', 20),
        time_exit_enabled=p.get('time_exit_enabled', False),
        time_exit_bars_long=p.get('time_exit_bars', 20),
        time_exit_bars_short=p.get('time_exit_bars', 20),
        ma_exit_enabled=p.get('ma_exit_enabled', False),
        ma_exit_fast=p.get('ma_exit_fast', 10),
        ma_exit_slow=p.get('ma_exit_slow', 20),
        bbwp_exit_enabled=p.get('bbwp_exit_enabled', False),
        bbwp_exit_threshold=p.get('bbwp_exit_threshold', 80),
    )


def get_active_filters_display(params: Dict) -> List[str]:
    """Get human-readable list of active entry filters and exit conditions."""
    entry_map = {
        'pamrp_enabled': 'PAMRP',
        'bbwp_enabled': 'BBWP',
        'adx_enabled': 'ADX',
        'ma_trend_enabled': 'MA Trend',
        'rsi_enabled': 'RSI',
        'volume_enabled': 'Volume',
        'supertrend_enabled': 'Supertrend',
        'vwap_enabled': 'VWAP',
        'macd_enabled': 'MACD',
    }
    exit_map = {
        'stop_loss_enabled': 'Stop Loss',
        'take_profit_enabled': 'Take Profit',
        'trailing_stop_enabled': 'Trailing Stop',
        'atr_trailing_enabled': 'ATR Trail',
        'pamrp_exit_enabled': 'PAMRP Exit',
        'stoch_rsi_exit_enabled': 'Stoch RSI Exit',
        'time_exit_enabled': 'Time Exit',
        'ma_exit_enabled': 'MA Exit',
        'bbwp_exit_enabled': 'BBWP Exit',
    }
    active = []
    for k, label in entry_map.items():
        if params.get(k, False):
            active.append(label)
    for k, label in exit_map.items():
        if params.get(k, False):
            active.append(label)
    return active


def run_monte_carlo(trades: List, n_simulations: int = 1000, initial_capital: float = 10000) -> Dict:
    """Monte Carlo simulation - randomize trade order"""
    if not trades or len(trades) < 5:
        return None

    pnls = [t.pnl for t in trades]
    final_equities = []
    max_dds = []

    for _ in range(n_simulations):
        shuffled = pnls.copy()
        random.shuffle(shuffled)

        equity = [initial_capital]
        for pnl in shuffled:
            equity.append(equity[-1] + pnl)

        peak = np.maximum.accumulate(equity)
        dd = (np.array(equity) - peak) / peak * 100

        final_equities.append(equity[-1])
        max_dds.append(dd.min())

    return {
        'final_equities': final_equities,
        'max_dds': max_dds,
        'percentiles': {
            '5%': np.percentile(final_equities, 5),
            '25%': np.percentile(final_equities, 25),
            '50%': np.percentile(final_equities, 50),
            '75%': np.percentile(final_equities, 75),
            '95%': np.percentile(final_equities, 95),
        },
        'dd_percentiles': {
            '5%': np.percentile(max_dds, 5),
            '50%': np.percentile(max_dds, 50),
            '95%': np.percentile(max_dds, 95),
        }
    }


def calculate_beta_alpha(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
    """Calculate Beta and Alpha vs benchmark"""
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return {'beta': 0, 'alpha': 0, 'correlation': 0}

    aligned.columns = ['strategy', 'benchmark']

    cov = aligned.cov().iloc[0, 1]
    var = aligned['benchmark'].var()
    beta = cov / var if var > 0 else 0

    alpha = (aligned['strategy'].mean() - beta * aligned['benchmark'].mean()) * 252

    corr = aligned.corr().iloc[0, 1]

    return {'beta': beta, 'alpha': alpha * 100, 'correlation': corr}


# ═══════════════════════════════════════════════════════════════════════════════
# PARAM ↔ WIDGET KEY MAPPING & APPLY CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

# Maps param dict keys → Streamlit widget keys used in the sidebar.
# This is needed because Streamlit widgets with a `key` read their value
# from st.session_state[key], NOT from the `value` argument after first render.
PARAM_TO_WIDGET_KEY = {
    'pamrp_enabled': 'pe', 'pamrp_length': 'pl',
    'pamrp_entry_long': 'pel', 'pamrp_entry_short': 'pes',
    'pamrp_exit_long': 'pxl', 'pamrp_exit_short': 'pxs',
    'bbwp_enabled': 'be', 'bbwp_length': 'bl',
    'bbwp_lookback': 'blb', 'bbwp_sma_length': 'bsma',
    'bbwp_threshold_long': 'btl', 'bbwp_threshold_short': 'bts',
    'bbwp_ma_filter': 'bmf',
    'adx_enabled': 'ae', 'adx_length': 'al', 'adx_threshold': 'at',
    'ma_trend_enabled': 'mae', 'ma_type': 'mat',
    'ma_fast_length': 'maf', 'ma_slow_length': 'mas',
    'rsi_enabled': 're', 'rsi_length': 'rl',
    'rsi_oversold': 'ros', 'rsi_overbought': 'rob',
    'volume_enabled': 've', 'volume_ma_length': 'vml',
    'volume_multiplier': 'vm',
    'supertrend_enabled': 'ste', 'supertrend_period': 'stp',
    'supertrend_multiplier': 'stm',
    'vwap_enabled': 'vwe',
    'macd_enabled': 'mce', 'macd_fast': 'mcf',
    'macd_slow': 'mcs', 'macd_signal': 'mcsi',
    'stop_loss_enabled': 'sle', 'stop_loss_pct_long': 'sll',
    'stop_loss_pct_short': 'sls',
    'take_profit_enabled': 'tpe', 'take_profit_pct_long': 'tpl',
    'take_profit_pct_short': 'tps',
    'trailing_stop_enabled': 'tse', 'trailing_stop_pct': 'tsp',
    'atr_trailing_enabled': 'ate', 'atr_length': 'atl',
    'atr_multiplier': 'atm',
    'pamrp_exit_enabled': 'pxe',
    'stoch_rsi_exit_enabled': 'sre', 'stoch_rsi_length': 'srl',
    'stoch_rsi_k': 'srk', 'stoch_rsi_d': 'srd',
    'stoch_rsi_overbought': 'srob', 'stoch_rsi_oversold': 'sros',
    'time_exit_enabled': 'txe', 'time_exit_bars': 'txb',
    'ma_exit_enabled': 'mxe', 'ma_exit_fast': 'mxf',
    'ma_exit_slow': 'mxs',
    'bbwp_exit_enabled': 'bxe', 'bbwp_exit_threshold': 'bxt',
}


def apply_best_params_callback():
    """on_click callback — runs BEFORE widgets render on next rerun.
    This is the only safe way to update widget keys in Streamlit."""
    res = st.session_state.get('optimization_results')
    if not res:
        return

    bp = res.best_params

    for k, v in bp.items():
        if isinstance(v, TradeDirection) or k == 'trade_direction':
            continue
        # Update params dict
        if k in st.session_state.params:
            st.session_state.params[k] = v
        # Update widget key so sidebar slider/toggle shows new value
        widget_key = PARAM_TO_WIDGET_KEY.get(k)
        if widget_key:
            st.session_state[widget_key] = v

    # Handle direction
    if 'trade_direction_str' in bp:
        dm = {'long_only': 'Long Only', 'short_only': 'Short Only', 'both': 'Both'}
        st.session_state.params['trade_direction'] = dm.get(bp['trade_direction_str'], 'Long Only')

    # Sync capital and commission
    st.session_state.capital = res.initial_capital
    st.session_state.commission = res.commission_pct

    # Clear old backtest so user re-runs with new params
    st.session_state.backtest_results = None

    # Flag so we can show success message after rerun
    st.session_state._apply_success = True


# ═══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_price_chart_with_trades(df: pd.DataFrame, trades: Optional[List] = None) -> go.Figure:
    """Candlestick price chart with optional trade entry/exit markers."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='#10b981', decreasing_line_color='#ef4444',
        increasing_fillcolor='rgba(16, 185, 129, 0.3)',
        decreasing_fillcolor='rgba(239, 68, 68, 0.3)',
        name='Price'))

    if trades:
        # Entry markers
        entry_dates = [t.entry_date for t in trades]
        entry_prices = [t.entry_price for t in trades]
        entry_colors = ['#10b981' if t.direction == 'long' else '#ef4444' for t in trades]
        entry_symbols = ['triangle-up' if t.direction == 'long' else 'triangle-down' for t in trades]
        entry_labels = [f"{'▲ Long' if t.direction == 'long' else '▼ Short'} @ ${t.entry_price:.2f}"
                        for t in trades]

        fig.add_trace(go.Scatter(
            x=entry_dates, y=entry_prices, mode='markers',
            marker=dict(color=entry_colors, size=9, symbol=entry_symbols,
                        line=dict(width=1, color='white')),
            text=entry_labels, hoverinfo='text', name='Entries'))

        # Exit markers
        exit_dates = [t.exit_date for t in trades if t.exit_date is not None]
        exit_prices = [t.exit_price for t in trades if t.exit_date is not None]
        exit_trades = [t for t in trades if t.exit_date is not None]
        exit_colors = ['#10b981' if t.pnl >= 0 else '#ef4444' for t in exit_trades]
        exit_labels = [f"Exit @ ${t.exit_price:.2f} | PnL: ${t.pnl:+.2f} ({t.exit_reason})"
                       for t in exit_trades]

        fig.add_trace(go.Scatter(
            x=exit_dates, y=exit_prices, mode='markers',
            marker=dict(color=exit_colors, size=8, symbol='x',
                        line=dict(width=1, color='white')),
            text=exit_labels, hoverinfo='text', name='Exits'))

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10, 14, 20, 0.8)', height=320, showlegend=False,
        margin=dict(l=50, r=20, t=10, b=30), font=dict(size=9),
        xaxis_rangeslider_visible=False)
    fig.update_xaxes(gridcolor='rgba(45, 53, 72, 0.3)')
    fig.update_yaxes(gridcolor='rgba(45, 53, 72, 0.3)')
    return fig


def create_equity_chart(results: BacktestResults) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=results.equity_curve.index, y=results.equity_curve.values, mode='lines',
        line=dict(color='#3b82f6', width=2), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'), row=1, col=1)
    peak = results.equity_curve.expanding().max()
    dd = (results.equity_curve - peak) / peak * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode='lines',
        line=dict(color='#ef4444', width=2), fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.15)'), row=2, col=1)
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10, 14, 20, 0.8)',
        height=280, showlegend=False, margin=dict(l=50, r=20, t=10, b=30), font=dict(size=9))
    fig.update_xaxes(gridcolor='rgba(45, 53, 72, 0.3)')
    fig.update_yaxes(gridcolor='rgba(45, 53, 72, 0.3)')
    return fig


def create_monte_carlo_chart(mc_results: Dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mc_results['final_equities'], nbinsx=50, marker_color='#3b82f6', opacity=0.7))
    for pct, val in mc_results['percentiles'].items():
        fig.add_vline(x=val, line_dash='dash', line_color='#f59e0b' if pct == '50%' else '#64748b',
                     annotation_text=f"{pct}: ${val:,.0f}")
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10, 14, 20, 0.8)',
        height=250, margin=dict(l=40, r=20, t=10, b=30), xaxis_title="Final Equity", font=dict(size=9))
    return fig


def create_walkforward_chart(wf_folds) -> go.Figure:
    """Walk-forward bar chart — ALL folds always visible, no dragging/scrolling."""
    if not wf_folds:
        return go.Figure()

    folds = [f"Fold {f.fold_num}" for f in wf_folds]
    train_vals = [f.train_value for f in wf_folds]
    test_vals = [f.test_value for f in wf_folds]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=folds, y=train_vals, name='Train', marker_color='#3b82f6', opacity=0.8))
    fig.add_trace(go.Bar(x=folds, y=test_vals, name='Test', marker_color='#10b981', opacity=0.8))

    # Dynamically adjust bar gap based on fold count for visual fit
    n = len(folds)
    bargap = max(0.05, 0.30 - n * 0.02)
    bargroupgap = max(0.02, 0.15 - n * 0.01)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10, 14, 20, 0.8)',
        height=250,
        margin=dict(l=40, r=20, t=10, b=40),
        barmode='group',
        font=dict(size=9),
        legend=dict(orientation='h', y=1.12),
        bargap=bargap,
        bargroupgap=bargroupgap,
        # Prevent any drag/zoom/pan
        dragmode=False,
    )
    # Force category axis so all folds show; fixedrange prevents drag
    fig.update_xaxes(
        type='category',
        fixedrange=True,
        categoryorder='array',
        categoryarray=folds,
        tickangle=-45 if n > 7 else 0,
    )
    fig.update_yaxes(fixedrange=True)

    return fig


def create_heatmap(df: pd.DataFrame, param1: str, param2: str, metric: str, params: Dict,
                   capital: float, commission: float) -> go.Figure:
    ranges = {
        'pamrp_entry_long': list(range(10, 45, 5)),
        'pamrp_exit_long': list(range(50, 95, 5)),
        'bbwp_threshold_long': list(range(30, 75, 5)),
        'stop_loss_pct_long': [1, 2, 3, 4, 5, 6, 7, 8],
        'take_profit_pct_long': [2, 4, 6, 8, 10, 12, 14],
    }

    r1 = ranges.get(param1, list(range(10, 50, 5)))
    r2 = ranges.get(param2, list(range(10, 50, 5)))

    results_matrix = np.zeros((len(r2), len(r1)))

    for i, v1 in enumerate(r1):
        for j, v2 in enumerate(r2):
            test_params = params.copy()
            test_params[param1] = v1
            test_params[param2] = v2
            try:
                engine = BacktestEngine(params_to_strategy(test_params), capital, commission)
                res = engine.run(df.copy())
                results_matrix[j, i] = getattr(res, metric, 0)
            except:
                results_matrix[j, i] = 0

    fig = go.Figure(data=go.Heatmap(z=results_matrix, x=r1, y=r2, colorscale='RdYlGn', showscale=True))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10, 14, 20, 0.8)',
        height=300, margin=dict(l=50, r=20, t=30, b=50), xaxis_title=param1, yaxis_title=param2, font=dict(size=9))
    return fig


def create_optimization_chart(trials_df: pd.DataFrame, metric: str) -> go.Figure:
    if trials_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trials_df['trial_number'], y=trials_df['value'], mode='markers',
        marker=dict(color='rgba(59, 130, 246, 0.4)', size=5)))
    best = trials_df['value'].expanding().max()
    fig.add_trace(go.Scatter(x=trials_df['trial_number'], y=best, mode='lines', line=dict(color='#10b981', width=2)))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10, 14, 20, 0.8)',
        height=200, margin=dict(l=50, r=20, t=10, b=30), xaxis_title="Trial", yaxis_title=metric, font=dict(size=9))
    return fig


def create_multi_asset_chart(results_dict: Dict[str, BacktestResults]) -> go.Figure:
    fig = go.Figure()
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    for i, (symbol, res) in enumerate(results_dict.items()):
        normalized = (res.equity_curve / res.equity_curve.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized.values, mode='lines',
            name=symbol, line=dict(color=colors[i % len(colors)], width=2)))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10, 14, 20, 0.8)',
        height=300, margin=dict(l=50, r=20, t=10, b=30), yaxis_title="Return %", font=dict(size=9),
        legend=dict(orientation='h', y=1.1))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

p = st.session_state.params

with st.sidebar:
    st.markdown("# 📊 Trading Toolkit Pro")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # DATA
    st.markdown("### 📈 Data")
    data_src = st.radio("Source", ["Yahoo Finance", "Sample"], horizontal=True, label_visibility="collapsed")

    INTERVALS = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]

    if data_src == "Yahoo Finance":
        symbol = st.text_input("Symbol", "SPY")
        c1, c2 = st.columns(2)
        interval = c1.selectbox("Interval", INTERVALS, index=5)
        days = c2.number_input("Days", 30, 2000, 730)
    else:
        symbol, interval, days = "SAMPLE", "1d", 500

    if st.button("📥 Load", use_container_width=True):
        with st.spinner("Loading..."):
            try:
                if data_src == "Yahoo Finance":
                    end = datetime.now()
                    start = end - timedelta(days=days)
                    df = fetch_yfinance(symbol, str(start.date()), str(end.date()), interval)
                else:
                    df = generate_sample_data(500)
                st.session_state.df = df
                st.success(f"✅ {len(df)} bars")
            except Exception as e:
                st.error(str(e)[:50])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # STRATEGY
    st.markdown("### ⚙️ Strategy")
    p['trade_direction'] = st.selectbox("Direction", ["Long Only", "Short Only", "Both"],
        index=["Long Only", "Short Only", "Both"].index(p['trade_direction']))

    # Position Sizing
    with st.expander("💰 Position Sizing", expanded=False):
        p['position_size_pct'] = st.slider("Position %", 10, 100, int(p['position_size_pct']), 10)
        p['use_kelly'] = st.toggle("Kelly Criterion", p['use_kelly'])
        if p['use_kelly']:
            p['kelly_fraction'] = st.slider("Kelly Fraction", 0.1, 1.0, p['kelly_fraction'], 0.1)

    # PAMRP
    with st.expander("📊 PAMRP", expanded=False):
        p['pamrp_enabled'] = st.toggle("Enable", p['pamrp_enabled'], key="pe")
        if p['pamrp_enabled']:
            p['pamrp_length'] = st.slider("Length", 5, 50, p['pamrp_length'], key="pl")
            p['pamrp_entry_long'] = st.slider("Entry Long", 5, 50, p['pamrp_entry_long'], key="pel")
            p['pamrp_entry_short'] = st.slider("Entry Short", 50, 95, p['pamrp_entry_short'], key="pes")
            p['pamrp_exit_long'] = st.slider("Exit Long", 50, 100, p['pamrp_exit_long'], key="pxl")
            p['pamrp_exit_short'] = st.slider("Exit Short", 0, 50, p['pamrp_exit_short'], key="pxs")

    # BBWP
    with st.expander("📊 BBWP", expanded=False):
        p['bbwp_enabled'] = st.toggle("Enable", p['bbwp_enabled'], key="be")
        if p['bbwp_enabled']:
            p['bbwp_length'] = st.slider("Length", 5, 30, p['bbwp_length'], key="bl")
            p['bbwp_lookback'] = st.slider("Lookback", 50, 400, p['bbwp_lookback'], key="blb")
            p['bbwp_sma_length'] = st.slider("SMA", 2, 15, p['bbwp_sma_length'], key="bsma")
            p['bbwp_threshold_long'] = st.slider("Thresh Long", 20, 80, p['bbwp_threshold_long'], key="btl")
            p['bbwp_threshold_short'] = st.slider("Thresh Short", 20, 80, p['bbwp_threshold_short'], key="bts")
            p['bbwp_ma_filter'] = st.selectbox("MA Filter", ["disabled", "decreasing", "increasing"],
                index=["disabled", "decreasing", "increasing"].index(p['bbwp_ma_filter']), key="bmf")

    # ADX
    with st.expander("📈 ADX", expanded=False):
        p['adx_enabled'] = st.toggle("Enable", p['adx_enabled'], key="ae")
        if p['adx_enabled']:
            p['adx_length'] = st.slider("Length", 7, 21, p['adx_length'], key="al")
            p['adx_threshold'] = st.slider("Threshold", 10, 40, p['adx_threshold'], key="at")

    # MA Trend
    with st.expander("📈 MA Trend", expanded=False):
        p['ma_trend_enabled'] = st.toggle("Enable", p['ma_trend_enabled'], key="mae")
        if p['ma_trend_enabled']:
            p['ma_type'] = st.selectbox("Type", ["sma", "ema"], index=["sma", "ema"].index(p['ma_type']), key="mat")
            p['ma_fast_length'] = st.slider("Fast", 10, 100, p['ma_fast_length'], key="maf")
            p['ma_slow_length'] = st.slider("Slow", 50, 400, p['ma_slow_length'], key="mas")

    # RSI
    with st.expander("📈 RSI", expanded=False):
        p['rsi_enabled'] = st.toggle("Enable", p['rsi_enabled'], key="re")
        if p['rsi_enabled']:
            p['rsi_length'] = st.slider("Length", 5, 21, p['rsi_length'], key="rl")
            p['rsi_oversold'] = st.slider("Oversold", 15, 45, p['rsi_oversold'], key="ros")
            p['rsi_overbought'] = st.slider("Overbought", 55, 85, p['rsi_overbought'], key="rob")

    # Volume
    with st.expander("📊 Volume", expanded=False):
        p['volume_enabled'] = st.toggle("Enable", p['volume_enabled'], key="ve")
        if p['volume_enabled']:
            p['volume_ma_length'] = st.slider("MA Length", 10, 50, p['volume_ma_length'], key="vml")
            p['volume_multiplier'] = st.slider("Multiplier", 0.5, 2.0, p['volume_multiplier'], 0.1, key="vm")

    # Supertrend
    with st.expander("📈 Supertrend", expanded=False):
        p['supertrend_enabled'] = st.toggle("Enable", p['supertrend_enabled'], key="ste")
        if p['supertrend_enabled']:
            p['supertrend_period'] = st.slider("Period", 5, 20, p['supertrend_period'], key="stp")
            p['supertrend_multiplier'] = st.slider("Mult", 1.0, 5.0, p['supertrend_multiplier'], 0.5, key="stm")

    # VWAP
    with st.expander("📈 VWAP", expanded=False):
        p['vwap_enabled'] = st.toggle("Enable", p['vwap_enabled'], key="vwe")

    # MACD
    with st.expander("📈 MACD", expanded=False):
        p['macd_enabled'] = st.toggle("Enable", p['macd_enabled'], key="mce")
        if p['macd_enabled']:
            p['macd_fast'] = st.slider("Fast", 5, 20, p['macd_fast'], key="mcf")
            p['macd_slow'] = st.slider("Slow", 15, 40, p['macd_slow'], key="mcs")
            p['macd_signal'] = st.slider("Signal", 5, 15, p['macd_signal'], key="mcsi")

    st.markdown("### 🚪 Exits")

    # Stop Loss
    with st.expander("🛑 Stop Loss", expanded=False):
        p['stop_loss_enabled'] = st.toggle("Enable", p['stop_loss_enabled'], key="sle")
        if p['stop_loss_enabled']:
            p['stop_loss_pct_long'] = st.slider("% Long", 0.5, 15.0, p['stop_loss_pct_long'], 0.5, key="sll")
            p['stop_loss_pct_short'] = st.slider("% Short", 0.5, 15.0, p['stop_loss_pct_short'], 0.5, key="sls")

    # Take Profit
    with st.expander("🎯 Take Profit", expanded=False):
        p['take_profit_enabled'] = st.toggle("Enable", p['take_profit_enabled'], key="tpe")
        if p['take_profit_enabled']:
            p['take_profit_pct_long'] = st.slider("% Long", 1.0, 30.0, p['take_profit_pct_long'], 0.5, key="tpl")
            p['take_profit_pct_short'] = st.slider("% Short", 1.0, 30.0, p['take_profit_pct_short'], 0.5, key="tps")

    # Trailing Stop
    with st.expander("📉 Trailing Stop", expanded=False):
        p['trailing_stop_enabled'] = st.toggle("Enable", p['trailing_stop_enabled'], key="tse")
        if p['trailing_stop_enabled']:
            p['trailing_stop_pct'] = st.slider("Trail %", 0.5, 10.0, p['trailing_stop_pct'], 0.5, key="tsp")

    # ATR Trailing
    with st.expander("📉 ATR Trail", expanded=False):
        p['atr_trailing_enabled'] = st.toggle("Enable", p['atr_trailing_enabled'], key="ate")
        if p['atr_trailing_enabled']:
            p['atr_length'] = st.slider("Length", 7, 21, p['atr_length'], key="atl")
            p['atr_multiplier'] = st.slider("Mult", 1.0, 5.0, p['atr_multiplier'], 0.5, key="atm")

    # PAMRP Exit
    with st.expander("📊 PAMRP Exit", expanded=False):
        p['pamrp_exit_enabled'] = st.toggle("Enable", p['pamrp_exit_enabled'], key="pxe")

    # Stoch RSI Exit
    with st.expander("📈 Stoch RSI Exit", expanded=False):
        p['stoch_rsi_exit_enabled'] = st.toggle("Enable", p['stoch_rsi_exit_enabled'], key="sre")
        if p['stoch_rsi_exit_enabled']:
            p['stoch_rsi_length'] = st.slider("Length", 7, 21, p['stoch_rsi_length'], key="srl")
            p['stoch_rsi_k'] = st.slider("K", 2, 5, p['stoch_rsi_k'], key="srk")
            p['stoch_rsi_d'] = st.slider("D", 2, 5, p['stoch_rsi_d'], key="srd")
            p['stoch_rsi_overbought'] = st.slider("OB", 60, 90, p['stoch_rsi_overbought'], key="srob")
            p['stoch_rsi_oversold'] = st.slider("OS", 10, 40, p['stoch_rsi_oversold'], key="sros")

    # Time Exit
    with st.expander("⏱️ Time Exit", expanded=False):
        p['time_exit_enabled'] = st.toggle("Enable", p['time_exit_enabled'], key="txe")
        if p['time_exit_enabled']:
            p['time_exit_bars'] = st.slider("Max Bars", 5, 100, p['time_exit_bars'], key="txb")

    # MA Exit
    with st.expander("📈 MA Exit", expanded=False):
        p['ma_exit_enabled'] = st.toggle("Enable", p['ma_exit_enabled'], key="mxe")
        if p['ma_exit_enabled']:
            p['ma_exit_fast'] = st.slider("Fast", 3, 20, p['ma_exit_fast'], key="mxf")
            p['ma_exit_slow'] = st.slider("Slow", 10, 40, p['ma_exit_slow'], key="mxs")

    # BBWP Exit
    with st.expander("📊 BBWP Exit", expanded=False):
        p['bbwp_exit_enabled'] = st.toggle("Enable", p['bbwp_exit_enabled'], key="bxe")
        if p['bbwp_exit_enabled']:
            p['bbwp_exit_threshold'] = st.slider("Threshold", 60, 95, p['bbwp_exit_threshold'], key="bxt")

st.session_state.params = p


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 📊 Trading Toolkit Pro")

tabs = st.tabs(["🔬 Backtest", "🎯 Optimize", "⚖️ Compare", "🎲 Monte Carlo", "🔥 Heatmap", "🌐 Multi-Asset", "📋 Trades"])

# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST TAB — price chart with trade markers below equity curve
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    st.session_state.capital = c1.number_input("Capital $", 1000, 1000000, st.session_state.capital, 1000)
    st.session_state.commission = c2.number_input("Comm %", 0.0, 1.0, st.session_state.commission, 0.01)
    slippage = c3.number_input("Slip %", 0.0, 0.5, 0.0, 0.01)
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🚀 Run", type="primary", use_container_width=True)

    if run:
        if st.session_state.df is None:
            st.warning("Load data first!")
        else:
            with st.spinner("Running..."):
                engine = BacktestEngine(params_to_strategy(st.session_state.params),
                    st.session_state.capital, st.session_state.commission, slippage)
                res = engine.run(st.session_state.df.copy())
                st.session_state.backtest_results = res
                st.success(f"✅ {res.num_trades} trades")

    if st.session_state.backtest_results:
        r = st.session_state.backtest_results

        c1, c2, c3, c4, c5 = st.columns(5)
        ret_delta = "normal" if r.total_return_pct >= 0 else "inverse"
        c1.metric("Return", f"{r.total_return_pct:.2f}%", delta_color=ret_delta)
        c2.metric("Sharpe", f"{r.sharpe_ratio:.3f}")
        c3.metric("Sortino", f"{r.sortino_ratio:.3f}")
        c4.metric("PF", f"{r.profit_factor:.2f}")
        c5.metric("Max DD", f"{r.max_drawdown_pct:.2f}%", delta_color="inverse")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trades", r.num_trades)
        c2.metric("Win%", f"{r.win_rate:.1f}%")
        c3.metric("Avg Win", f"${r.avg_winner:.2f}")
        c4.metric("Avg Loss", f"${abs(r.avg_loser):.2f}")
        c5.metric("Max Consec Loss", r.max_consecutive_losses)

        # Equity curve
        st.plotly_chart(create_equity_chart(r), use_container_width=True)

        # Price chart with trade entry/exit markers
        if st.session_state.df is not None:
            st.plotly_chart(
                create_price_chart_with_trades(st.session_state.df, r.trades),
                use_container_width=True)

    elif st.session_state.df is not None:
        # No backtest yet — just show plain candlestick
        st.plotly_chart(
            create_price_chart_with_trades(st.session_state.df, None),
            use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZE TAB — shows active params message, fixed Apply
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("### 🎯 Optimization")

    c1, c2, c3, c4 = st.columns(4)
    opt_metric = c1.selectbox("Metric", ["sharpe_ratio", "total_return_pct", "profit_factor", "sortino_ratio"])
    opt_dir = c2.selectbox("Dir", ["long_only", "short_only", "both"])
    opt_trials = c3.slider("Trials", 20, 300, 100)
    opt_min = c4.slider("Min Trades", 5, 30, 10)

    c1, c2, c3 = st.columns(3)
    train_pct = c1.slider("Train %", 50, 90, 70)
    use_wf = c2.toggle("Walk-Forward", False)
    n_folds = c3.slider("Folds", 3, 10, 5) if use_wf else 5

    # Show currently active filters as a preview
    active_display = get_active_filters_display(st.session_state.params)
    if active_display:
        st.caption(f"🔍 Active parameters: **{', '.join(active_display)}**")

    if st.button("🎯 Optimize", type="primary", use_container_width=True):
        if st.session_state.df is None:
            st.warning("Load data first!")
        else:
            ef = {k: v for k, v in st.session_state.params.items() if k.endswith('_enabled')}
            active_filters = [k.replace('_enabled', '') for k, v in ef.items() if v]
            # Show the "optimizing for" message
            if active_filters:
                readable = get_active_filters_display(st.session_state.params)
                st.info(f"🔍 Optimizing for: **{', '.join(readable)}**")
            with st.spinner("Optimizing..."):
                res = optimize_strategy(
                    df=st.session_state.df.copy(), enabled_filters=ef, metric=opt_metric,
                    n_trials=opt_trials, min_trades=opt_min,
                    initial_capital=st.session_state.capital,
                    commission_pct=st.session_state.commission,
                    trade_direction=opt_dir, train_pct=train_pct/100,
                    use_walkforward=use_wf, n_folds=n_folds, show_progress=False
                )
                st.session_state.optimization_results = res
                st.success("✅ Done!")

    if st.session_state.optimization_results:
        res = st.session_state.optimization_results

        c1, c2, c3 = st.columns(3)
        metric_label = res.metric.replace('_', ' ').title()
        c1.metric(f"Train ({metric_label})", f"{res.train_value:.4f}")
        c2.metric(f"Test ({metric_label})", f"{res.test_value:.4f}")
        if res.train_value > 0:
            deg = ((res.train_value - res.test_value) / res.train_value) * 100
            c3.metric("Degrad", f"{deg:.1f}%", delta_color="normal" if deg < 30 else "inverse")

        # Walk-forward visualization — FIXED chart
        if res.walkforward_folds:
            st.markdown("### 📊 Walk-Forward Results")
            st.plotly_chart(
                create_walkforward_chart(res.walkforward_folds),
                use_container_width=True,
                config={'displayModeBar': False, 'staticPlot': False,
                        'scrollZoom': False, 'doubleClick': False})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Best Params")
            params_list = [{"P": k, "V": str(v)[:20]} for k, v in res.best_params.items()
                if not k.startswith('trade_direction') and not isinstance(v, TradeDirection)]
            st.dataframe(pd.DataFrame(params_list), use_container_width=True, hide_index=True, height=250)
        with c2:
            st.markdown("### Performance (Full Data)")
            br = res.best_results
            st.dataframe(pd.DataFrame([
                {"Metric": "Return", "Value": f"{br.total_return_pct:.2f}%"},
                {"Metric": "Sharpe", "Value": f"{br.sharpe_ratio:.3f}"},
                {"Metric": "Max DD", "Value": f"{br.max_drawdown_pct:.2f}%"},
                {"Metric": "Trades", "Value": str(br.num_trades)},
            ]), use_container_width=True, hide_index=True)
            st.caption(f"Capital: ${res.initial_capital:,.0f} | Commission: {res.commission_pct}%")

        st.info("These are full-dataset results. Applying best params and running backtest with the same capital/commission will produce identical metrics.", icon="ℹ️")

        st.plotly_chart(create_optimization_chart(res.all_trials, res.metric), use_container_width=True)

        # Apply Best Parameters — uses on_click callback so it runs
        # BEFORE sidebar widgets render on the next rerun.
        st.button("📋 Apply Best Params", on_click=apply_best_params_callback,
                  use_container_width=True)

        # Show success message after callback triggered rerun
        if st.session_state.pop('_apply_success', False):
            st.success(f"✅ Applied! Capital: ${st.session_state.capital:,.0f} | "
                       f"Commission: {st.session_state.commission}% | "
                       f"Direction: {st.session_state.params.get('trade_direction', 'Long Only')}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARE TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("### ⚖️ Strategy vs Buy & Hold")

    if st.session_state.df is not None:
        if st.button("🔄 Compare", use_container_width=True):
            df = st.session_state.df.copy()
            engine = BacktestEngine(params_to_strategy(st.session_state.params),
                st.session_state.capital, st.session_state.commission)
            strat_res = engine.run(df)

            if strat_res.trades:
                first_trade = strat_res.trades[0]
                entry_price = first_trade.entry_price
                entry_date = first_trade.entry_date
                final_price = df['close'].iloc[-1]

                if first_trade.direction == 'short':
                    bh_return_pct = ((entry_price - final_price) / entry_price) * 100
                    bh_label = "📉 Sell & Hold (Short)"
                else:
                    bh_return_pct = ((final_price - entry_price) / entry_price) * 100
                    bh_label = "📈 Buy & Hold"

                bh_return = st.session_state.capital * (bh_return_pct / 100)

                mask = df.index >= entry_date
                prices = df.loc[mask, 'close']
                if first_trade.direction == 'short':
                    bottom = prices.expanding().min()
                    bh_dd = ((prices - bottom) / bottom * 100).max() * -1
                else:
                    peak = prices.expanding().max()
                    bh_dd = ((prices - peak) / peak * 100).min()

                comparison = pd.DataFrame([
                    {'Strategy': '📊 Your Strategy', 'Return %': f"{strat_res.total_return_pct:.2f}%",
                     'Max DD': f"{strat_res.max_drawdown_pct:.2f}%", 'Trades': strat_res.num_trades,
                     'Sharpe': f"{strat_res.sharpe_ratio:.3f}"},
                    {'Strategy': bh_label, 'Return %': f"{bh_return_pct:.2f}%",
                     'Max DD': f"{bh_dd:.2f}%", 'Trades': 1, 'Sharpe': "-"}
                ])
                st.dataframe(comparison, use_container_width=True, hide_index=True)

                diff = strat_res.total_return_pct - bh_return_pct
                if diff > 0:
                    st.success(f"🏆 Strategy beats B&H by {diff:.2f}%")
                else:
                    st.warning(f"📉 B&H beats strategy by {-diff:.2f}%")

                strat_rets = strat_res.equity_curve.pct_change().dropna()
                bench_rets = df.loc[mask, 'close'].pct_change().dropna()
                ba = calculate_beta_alpha(strat_rets, bench_rets)

                st.markdown("### 📈 Risk Metrics vs Benchmark")
                c1, c2, c3 = st.columns(3)
                c1.metric("Beta", f"{ba['beta']:.3f}")
                c2.metric("Alpha (ann)", f"{ba['alpha']:.2f}%")
                c3.metric("Correlation", f"{ba['correlation']:.3f}")
            else:
                st.warning("No trades generated")
    else:
        st.info("Load data first")


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("### 🎲 Monte Carlo Simulation")
    st.info("Randomizes trade order to test strategy robustness")

    c1, c2 = st.columns(2)
    n_sims = c1.slider("Simulations", 100, 5000, 1000)

    if st.button("🎲 Run Monte Carlo", use_container_width=True):
        if st.session_state.backtest_results and st.session_state.backtest_results.trades:
            with st.spinner(f"Running {n_sims} simulations..."):
                mc = run_monte_carlo(st.session_state.backtest_results.trades, n_sims, st.session_state.capital)

                if mc:
                    st.markdown("### 📊 Final Equity Distribution")
                    st.plotly_chart(create_monte_carlo_chart(mc), use_container_width=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("5th Pctl", f"${mc['percentiles']['5%']:,.0f}")
                    c2.metric("Median", f"${mc['percentiles']['50%']:,.0f}")
                    c3.metric("95th Pctl", f"${mc['percentiles']['95%']:,.0f}")

                    st.markdown("### 📉 Max Drawdown Distribution")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("5th Pctl DD", f"{mc['dd_percentiles']['5%']:.1f}%")
                    c2.metric("Median DD", f"{mc['dd_percentiles']['50%']:.1f}%")
                    c3.metric("95th Pctl DD", f"{mc['dd_percentiles']['95%']:.1f}%")
        else:
            st.warning("Run backtest first")


# ═══════════════════════════════════════════════════════════════════════════════
# HEATMAP TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("### 🔥 Parameter Sensitivity")
    st.info("Shows how results change with parameter variations")

    param_options = ['pamrp_entry_long', 'pamrp_exit_long', 'bbwp_threshold_long',
                    'stop_loss_pct_long', 'take_profit_pct_long']

    c1, c2, c3 = st.columns(3)
    p1 = c1.selectbox("Param X", param_options, index=0)
    p2 = c2.selectbox("Param Y", param_options, index=2)
    hm_metric = c3.selectbox("Metric", ["sharpe_ratio", "total_return_pct", "profit_factor"])

    if st.button("🔥 Generate Heatmap", use_container_width=True):
        if st.session_state.df is not None:
            with st.spinner("Calculating..."):
                fig = create_heatmap(st.session_state.df, p1, p2, hm_metric,
                    st.session_state.params, st.session_state.capital, st.session_state.commission)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Load data first")


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-ASSET TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown("### 🌐 Multi-Asset Portfolio")

    symbols_input = st.text_input("Symbols (comma-separated)", "SPY, QQQ, IWM")

    if st.button("📊 Run Multi-Asset", use_container_width=True):
        symbols = [s.strip().upper() for s in symbols_input.split(',')]

        results_dict = {}
        with st.spinner(f"Testing {len(symbols)} assets..."):
            for sym in symbols:
                try:
                    end = datetime.now()
                    start = end - timedelta(days=730)
                    df = fetch_yfinance(sym, str(start.date()), str(end.date()), '1d')
                    engine = BacktestEngine(params_to_strategy(st.session_state.params),
                        st.session_state.capital, st.session_state.commission)
                    res = engine.run(df)
                    results_dict[sym] = res
                except Exception as e:
                    st.warning(f"{sym}: {str(e)[:30]}")

        if results_dict:
            st.plotly_chart(create_multi_asset_chart(results_dict), use_container_width=True)

            summary = pd.DataFrame([{
                'Symbol': sym, 'Return %': f"{r.total_return_pct:.2f}%",
                'Sharpe': f"{r.sharpe_ratio:.3f}", 'Max DD': f"{r.max_drawdown_pct:.2f}%",
                'Trades': r.num_trades, 'Win %': f"{r.win_rate:.1f}%"
            } for sym, r in results_dict.items()])
            st.dataframe(summary, use_container_width=True, hide_index=True)

            total_ret = sum(r.total_return for r in results_dict.values())
            avg_sharpe = np.mean([r.sharpe_ratio for r in results_dict.values()])
            st.markdown(f"**Portfolio Total:** ${total_ret:,.2f} | **Avg Sharpe:** {avg_sharpe:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRADES TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown("### 📋 Trade Log")

    if st.session_state.backtest_results and st.session_state.backtest_results.trades:
        trades = st.session_state.backtest_results.trades

        c1, c2 = st.columns(2)
        dir_f = c1.selectbox("Dir", ["All", "Long", "Short"])
        res_f = c2.selectbox("Result", ["All", "Winners", "Losers"])

        flt = trades
        if dir_f != "All": flt = [t for t in flt if t.direction.lower() == dir_f.lower()]
        if res_f == "Winners": flt = [t for t in flt if t.pnl > 0]
        elif res_f == "Losers": flt = [t for t in flt if t.pnl <= 0]

        if flt:
            c1, c2, c3 = st.columns(3)
            c1.metric("Trades", len(flt))
            total_pnl = sum(t.pnl for t in flt)
            c2.metric("Total PnL", f"${total_pnl:,.2f}", delta_color="normal" if total_pnl >= 0 else "inverse")
            c3.metric("Avg", f"${np.mean([t.pnl for t in flt]):.2f}")

            trade_df = pd.DataFrame([{
                'Entry': t.entry_date, 'Exit': t.exit_date, 'Dir': t.direction,
                'Entry$': t.entry_price, 'Exit$': t.exit_price, 'PnL$': t.pnl,
                'PnL%': t.pnl_pct, 'Bars': t.bars_held, 'Reason': t.exit_reason
            } for t in flt])

            st.download_button("📥 Export CSV", trade_df.to_csv(index=False), "trades.csv", use_container_width=True)
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run backtest first")


st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#64748b;font-size:0.7rem;">Trading Toolkit Pro v5.1</p>', unsafe_allow_html=True)
