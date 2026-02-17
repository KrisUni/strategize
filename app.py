"""
Trading Toolkit - Premium UI v6
================================
Changes from v5.1:
- Mark-to-market equity curve with new metrics (CAGR, Calmar, Expectancy, MAE/MFE)
- Enhanced Monte Carlo: 3 methods, equity confidence bands, risk of ruin
- New Calendar Analytics tab (day-of-week, monthly seasonality, trade calendar)
- Backtest metrics row expanded
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from src.data import fetch_yfinance, generate_sample_data
from src.strategy import StrategyParams, SignalGenerator, TradeDirection
from src.backtest import BacktestEngine, BacktestResults
from src.optimize import optimize_strategy
from src.montecarlo import run_monte_carlo, MonteCarloResult
from src.analytics import analyze_calendar, analyze_trade_calendar

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
# CSS
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
    .stApp { background: var(--bg-primary); }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #0a0e14 100%) !important; }
    [data-testid="stSidebar"] h1 { font-size: 1.2rem !important; background: var(--gradient-primary); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    [data-testid="stSidebar"] h3 { font-size: 0.7rem !important; color: var(--text-secondary) !important; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.6rem !important; padding-bottom: 0.3rem; border-bottom: 1px solid var(--border-color); }
    .main .block-container { padding: 1rem 1.5rem; max-width: 100%; }
    .main h1 { font-size: 1.4rem !important; }
    .main h3 { font-size: 1rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: var(--bg-secondary); border-radius: 6px; padding: 2px; }
    .stTabs [data-baseweb="tab"] { height: 32px; padding: 0 12px; font-size: 0.8rem; }
    .stTabs [aria-selected="true"] { background: var(--gradient-primary) !important; color: white !important; border-radius: 4px; }
    [data-testid="stMetric"] { background: linear-gradient(145deg, #12161f 0%, #0d1117 100%); border: 1px solid var(--border-color); border-radius: 6px; padding: 0.5rem 0.7rem; margin-bottom: 0.4rem; }
    [data-testid="stMetricLabel"] { font-size: 0.6rem !important; }
    [data-testid="stMetricValue"] { font-size: 1rem !important; }
    [data-testid="stHorizontalBlock"] { gap: 0.4rem; margin-bottom: 0.3rem; }
    .stButton > button { background: var(--gradient-primary); color: white; border: none; border-radius: 5px; font-size: 0.8rem; }
    .streamlit-expanderHeader { font-size: 0.8rem !important; padding: 0.4rem 0.6rem !important; }
    .divider { height: 1px; background: linear-gradient(90deg, transparent, var(--border-color), transparent); margin: 0.6rem 0; }
    @media (max-width: 1024px) { .main .block-container { padding: 0.75rem 1rem; } [data-testid="stMetricLabel"] { font-size: 0.55rem !important; } [data-testid="stMetricValue"] { font-size: 0.85rem !important; } }
    @media (max-width: 768px) { .main .block-container { padding: 0.5rem; } [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; } }
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent !important; backdrop-filter: none !important; }
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

for key, default in [('params', get_default_params()), ('df', None), ('multi_df', {}),
                      ('backtest_results', None), ('optimization_results', None),
                      ('capital', 10000), ('commission', 0.1)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def params_to_strategy(p: Dict) -> StrategyParams:
    direction_map = {'Long Only': TradeDirection.LONG_ONLY, 'Short Only': TradeDirection.SHORT_ONLY, 'Both': TradeDirection.BOTH}
    return StrategyParams(
        trade_direction=direction_map.get(p.get('trade_direction', 'Long Only'), TradeDirection.LONG_ONLY),
        position_size_pct=p.get('position_size_pct', 100.0),
        use_kelly=p.get('use_kelly', False), kelly_fraction=p.get('kelly_fraction', 0.5),
        pamrp_enabled=p.get('pamrp_enabled', True), pamrp_length=p.get('pamrp_length', 21),
        pamrp_entry_long=p.get('pamrp_entry_long', 20), pamrp_entry_short=p.get('pamrp_entry_short', 80),
        pamrp_exit_long=p.get('pamrp_exit_long', 70), pamrp_exit_short=p.get('pamrp_exit_short', 30),
        bbwp_enabled=p.get('bbwp_enabled', True), bbwp_length=p.get('bbwp_length', 13),
        bbwp_lookback=p.get('bbwp_lookback', 252), bbwp_sma_length=p.get('bbwp_sma_length', 5),
        bbwp_threshold_long=p.get('bbwp_threshold_long', 50), bbwp_threshold_short=p.get('bbwp_threshold_short', 50),
        bbwp_ma_filter=p.get('bbwp_ma_filter', 'disabled'),
        adx_enabled=p.get('adx_enabled', False), adx_length=p.get('adx_length', 14),
        adx_smoothing=p.get('adx_smoothing', 14), adx_threshold=p.get('adx_threshold', 20),
        ma_trend_enabled=p.get('ma_trend_enabled', False), ma_fast_length=p.get('ma_fast_length', 50),
        ma_slow_length=p.get('ma_slow_length', 200), ma_type=p.get('ma_type', 'sma'),
        rsi_enabled=p.get('rsi_enabled', False), rsi_length=p.get('rsi_length', 14),
        rsi_oversold=p.get('rsi_oversold', 30), rsi_overbought=p.get('rsi_overbought', 70),
        volume_enabled=p.get('volume_enabled', False), volume_ma_length=p.get('volume_ma_length', 20),
        volume_multiplier=p.get('volume_multiplier', 1.0),
        supertrend_enabled=p.get('supertrend_enabled', False), supertrend_period=p.get('supertrend_period', 10),
        supertrend_multiplier=p.get('supertrend_multiplier', 3.0),
        vwap_enabled=p.get('vwap_enabled', False),
        macd_enabled=p.get('macd_enabled', False), macd_fast=p.get('macd_fast', 12),
        macd_slow=p.get('macd_slow', 26), macd_signal=p.get('macd_signal', 9),
        stop_loss_enabled=p.get('stop_loss_enabled', True),
        stop_loss_pct_long=p.get('stop_loss_pct_long', 3.0), stop_loss_pct_short=p.get('stop_loss_pct_short', 3.0),
        take_profit_enabled=p.get('take_profit_enabled', False),
        take_profit_pct_long=p.get('take_profit_pct_long', 5.0), take_profit_pct_short=p.get('take_profit_pct_short', 5.0),
        trailing_stop_enabled=p.get('trailing_stop_enabled', False), trailing_stop_pct=p.get('trailing_stop_pct', 2.0),
        atr_trailing_enabled=p.get('atr_trailing_enabled', False), atr_length=p.get('atr_length', 14),
        atr_multiplier=p.get('atr_multiplier', 2.0),
        pamrp_exit_enabled=p.get('pamrp_exit_enabled', True),
        stoch_rsi_exit_enabled=p.get('stoch_rsi_exit_enabled', False),
        stoch_rsi_length=p.get('stoch_rsi_length', 14), stoch_rsi_k=p.get('stoch_rsi_k', 3),
        stoch_rsi_d=p.get('stoch_rsi_d', 3), stoch_rsi_overbought=p.get('stoch_rsi_overbought', 80),
        stoch_rsi_oversold=p.get('stoch_rsi_oversold', 20),
        time_exit_enabled=p.get('time_exit_enabled', False),
        time_exit_bars_long=p.get('time_exit_bars', 20), time_exit_bars_short=p.get('time_exit_bars', 20),
        ma_exit_enabled=p.get('ma_exit_enabled', False), ma_exit_fast=p.get('ma_exit_fast', 10),
        ma_exit_slow=p.get('ma_exit_slow', 20),
        bbwp_exit_enabled=p.get('bbwp_exit_enabled', False), bbwp_exit_threshold=p.get('bbwp_exit_threshold', 80),
    )

def get_active_filters_display(params):
    entry = {'pamrp_enabled': 'PAMRP', 'bbwp_enabled': 'BBWP', 'adx_enabled': 'ADX',
             'ma_trend_enabled': 'MA Trend', 'rsi_enabled': 'RSI', 'volume_enabled': 'Volume',
             'supertrend_enabled': 'Supertrend', 'vwap_enabled': 'VWAP', 'macd_enabled': 'MACD'}
    exits = {'stop_loss_enabled': 'Stop Loss', 'take_profit_enabled': 'Take Profit',
             'trailing_stop_enabled': 'Trailing Stop', 'atr_trailing_enabled': 'ATR Trail',
             'pamrp_exit_enabled': 'PAMRP Exit', 'stoch_rsi_exit_enabled': 'Stoch RSI Exit',
             'time_exit_enabled': 'Time Exit', 'ma_exit_enabled': 'MA Exit', 'bbwp_exit_enabled': 'BBWP Exit'}
    return [l for k, l in {**entry, **exits}.items() if params.get(k, False)]

def calculate_beta_alpha(strategy_returns, benchmark_returns):
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10: return {'beta': 0, 'alpha': 0, 'correlation': 0}
    aligned.columns = ['strategy', 'benchmark']
    cov = aligned.cov().iloc[0, 1]; var = aligned['benchmark'].var()
    beta = cov / var if var > 0 else 0
    alpha = (aligned['strategy'].mean() - beta * aligned['benchmark'].mean()) * 252 * 100
    return {'beta': beta, 'alpha': alpha, 'correlation': aligned.corr().iloc[0, 1]}


# ═══════════════════════════════════════════════════════════════════════════════
# PARAM WIDGET KEY MAPPING (same as v5.1)
# ═══════════════════════════════════════════════════════════════════════════════

PARAM_TO_WIDGET_KEY = {
    'pamrp_enabled': 'pe', 'pamrp_length': 'pl', 'pamrp_entry_long': 'pel', 'pamrp_entry_short': 'pes',
    'pamrp_exit_long': 'pxl', 'pamrp_exit_short': 'pxs',
    'bbwp_enabled': 'be', 'bbwp_length': 'bl', 'bbwp_lookback': 'blb', 'bbwp_sma_length': 'bsma',
    'bbwp_threshold_long': 'btl', 'bbwp_threshold_short': 'bts', 'bbwp_ma_filter': 'bmf',
    'adx_enabled': 'ae', 'adx_length': 'al', 'adx_threshold': 'at',
    'ma_trend_enabled': 'mae', 'ma_type': 'mat', 'ma_fast_length': 'maf', 'ma_slow_length': 'mas',
    'rsi_enabled': 're', 'rsi_length': 'rl', 'rsi_oversold': 'ros', 'rsi_overbought': 'rob',
    'volume_enabled': 've', 'volume_ma_length': 'vml', 'volume_multiplier': 'vm',
    'supertrend_enabled': 'ste', 'supertrend_period': 'stp', 'supertrend_multiplier': 'stm',
    'vwap_enabled': 'vwe', 'macd_enabled': 'mce', 'macd_fast': 'mcf', 'macd_slow': 'mcs', 'macd_signal': 'mcsi',
    'stop_loss_enabled': 'sle', 'stop_loss_pct_long': 'sll', 'stop_loss_pct_short': 'sls',
    'take_profit_enabled': 'tpe', 'take_profit_pct_long': 'tpl', 'take_profit_pct_short': 'tps',
    'trailing_stop_enabled': 'tse', 'trailing_stop_pct': 'tsp',
    'atr_trailing_enabled': 'ate', 'atr_length': 'atl', 'atr_multiplier': 'atm',
    'pamrp_exit_enabled': 'pxe',
    'stoch_rsi_exit_enabled': 'sre', 'stoch_rsi_length': 'srl', 'stoch_rsi_k': 'srk', 'stoch_rsi_d': 'srd',
    'stoch_rsi_overbought': 'srob', 'stoch_rsi_oversold': 'sros',
    'time_exit_enabled': 'txe', 'time_exit_bars': 'txb',
    'ma_exit_enabled': 'mxe', 'ma_exit_fast': 'mxf', 'ma_exit_slow': 'mxs',
    'bbwp_exit_enabled': 'bxe', 'bbwp_exit_threshold': 'bxt',
}

def apply_best_params_callback():
    res = st.session_state.get('optimization_results')
    if not res: return
    bp = res.best_params
    for k, v in bp.items():
        if isinstance(v, TradeDirection) or k == 'trade_direction': continue
        if k in st.session_state.params: st.session_state.params[k] = v
        wk = PARAM_TO_WIDGET_KEY.get(k)
        if wk: st.session_state[wk] = v
    if 'trade_direction_str' in bp:
        dm = {'long_only': 'Long Only', 'short_only': 'Short Only', 'both': 'Both'}
        st.session_state.params['trade_direction'] = dm.get(bp['trade_direction_str'], 'Long Only')
    st.session_state.capital = res.initial_capital
    st.session_state.commission = res.commission_pct
    st.session_state.backtest_results = None
    st.session_state._apply_success = True


# ═══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _chart_layout(height=280, **kwargs):
    defaults = dict(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(10,14,20,0.8)', height=height, showlegend=False,
                    margin=dict(l=50, r=20, t=10, b=30), font=dict(size=9))
    defaults.update(kwargs)  # kwargs override defaults
    return defaults

def create_price_chart_with_trades(df, trades=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#10b981', decreasing_line_color='#ef4444',
        increasing_fillcolor='rgba(16,185,129,0.3)', decreasing_fillcolor='rgba(239,68,68,0.3)', name='Price'))
    if trades:
        ed = [t.entry_date for t in trades]; ep = [t.entry_price for t in trades]
        ec = ['#10b981' if t.direction=='long' else '#ef4444' for t in trades]
        es = ['triangle-up' if t.direction=='long' else 'triangle-down' for t in trades]
        el = [f"{'▲ Long' if t.direction=='long' else '▼ Short'} @ ${t.entry_price:.2f}" for t in trades]
        fig.add_trace(go.Scatter(x=ed, y=ep, mode='markers', marker=dict(color=ec, size=9, symbol=es, line=dict(width=1, color='white')), text=el, hoverinfo='text', name='Entries'))
        xt = [t for t in trades if t.exit_date]; xd = [t.exit_date for t in xt]; xp = [t.exit_price for t in xt]
        xc = ['#10b981' if t.pnl>=0 else '#ef4444' for t in xt]
        xl = [f"Exit @ ${t.exit_price:.2f} | ${t.pnl:+.2f} ({t.exit_reason})" for t in xt]
        fig.add_trace(go.Scatter(x=xd, y=xp, mode='markers', marker=dict(color=xc, size=8, symbol='x', line=dict(width=1, color='white')), text=xl, hoverinfo='text', name='Exits'))
    fig.update_layout(**_chart_layout(320), xaxis_rangeslider_visible=False)
    fig.update_xaxes(gridcolor='rgba(45,53,72,0.3)'); fig.update_yaxes(gridcolor='rgba(45,53,72,0.3)')
    return fig

def create_equity_chart(results):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=results.equity_curve.index, y=results.equity_curve.values, mode='lines',
        line=dict(color='#3b82f6', width=2), fill='tozeroy', fillcolor='rgba(59,130,246,0.1)', name='MTM Equity'), row=1, col=1)
    # Show realized equity as thinner line
    fig.add_trace(go.Scatter(x=results.realized_equity.index, y=results.realized_equity.values, mode='lines',
        line=dict(color='#64748b', width=1, dash='dot'), name='Realized'), row=1, col=1)
    peak = results.equity_curve.expanding().max()
    dd = (results.equity_curve - peak) / peak * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode='lines', line=dict(color='#ef4444', width=2),
        fill='tozeroy', fillcolor='rgba(239,68,68,0.15)'), row=2, col=1)
    fig.update_layout(**_chart_layout(280, showlegend=True, legend=dict(orientation='h', y=1.12, font=dict(size=8))))
    fig.update_xaxes(gridcolor='rgba(45,53,72,0.3)'); fig.update_yaxes(gridcolor='rgba(45,53,72,0.3)')
    return fig

def create_mc_confidence_chart(mc: MonteCarloResult):
    """Equity confidence band chart from Monte Carlo."""
    fig = go.Figure()
    n_steps = len(mc.equity_bands.get('50%', []))
    x = list(range(n_steps))
    # 5-95% band
    if '5%' in mc.equity_bands and '95%' in mc.equity_bands:
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['95%'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['5%'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(59,130,246,0.15)', name='5-95%'))
    # 25-75% band
    if '25%' in mc.equity_bands and '75%' in mc.equity_bands:
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['75%'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['25%'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(59,130,246,0.3)', name='25-75%'))
    # Median
    if '50%' in mc.equity_bands:
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['50%'], mode='lines', line=dict(color='#3b82f6', width=2), name='Median'))
    fig.update_layout(**_chart_layout(250, showlegend=True, legend=dict(orientation='h', y=1.12, font=dict(size=8))))
    fig.update_xaxes(title_text='Step', gridcolor='rgba(45,53,72,0.3)')
    fig.update_yaxes(title_text='Equity $', gridcolor='rgba(45,53,72,0.3)')
    return fig

def create_mc_histogram(values, title='', xaxis_title=''):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=50, marker_color='#3b82f6', opacity=0.7))
    p5 = np.percentile(values, 5); p50 = np.percentile(values, 50); p95 = np.percentile(values, 95)
    for val, label, color in [(p5, '5%', '#64748b'), (p50, '50%', '#f59e0b'), (p95, '95%', '#64748b')]:
        fig.add_vline(x=val, line_dash='dash', line_color=color, annotation_text=f"{label}: {val:,.0f}" if abs(val)>100 else f"{label}: {val:.1f}%")
    fig.update_layout(**_chart_layout(220), xaxis_title=xaxis_title)
    return fig

def create_dow_chart(dow_df):
    """Day-of-week bar chart."""
    if dow_df.empty: return go.Figure()
    colors = ['#ef4444' if v < 0 else '#10b981' for v in dow_df['Avg %']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dow_df['Day'], y=dow_df['Avg %'], marker_color=colors, text=[f"{v:+.4f}%" for v in dow_df['Avg %']], textposition='outside'))
    fig.update_layout(**_chart_layout(250), yaxis_title='Avg Return %')
    fig.update_xaxes(type='category', fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig

def create_monthly_heatmap(heatmap_df):
    """Year x Month heatmap."""
    if heatmap_df.empty: return go.Figure()
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values, x=heatmap_df.columns.tolist(),
        y=[str(y) for y in heatmap_df.index], colorscale='RdYlGn', zmid=0,
        text=np.round(heatmap_df.values, 2), texttemplate='%{text:.1f}%', showscale=True
    ))
    fig.update_layout(**_chart_layout(max(200, len(heatmap_df) * 35)))
    return fig

def create_walkforward_chart(wf_folds):
    if not wf_folds: return go.Figure()
    folds = [f"Fold {f.fold_num}" for f in wf_folds]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=folds, y=[f.train_value for f in wf_folds], name='Train', marker_color='#3b82f6', opacity=0.8))
    fig.add_trace(go.Bar(x=folds, y=[f.test_value for f in wf_folds], name='Test', marker_color='#10b981', opacity=0.8))
    fig.update_layout(**_chart_layout(250, showlegend=True, barmode='group', legend=dict(orientation='h', y=1.12), dragmode=False))
    fig.update_xaxes(type='category', fixedrange=True, categoryorder='array', categoryarray=folds)
    fig.update_yaxes(fixedrange=True)
    return fig

def create_optimization_chart(trials_df, metric):
    if trials_df.empty: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trials_df['trial_number'], y=trials_df['value'], mode='markers', marker=dict(color='rgba(59,130,246,0.4)', size=5)))
    fig.add_trace(go.Scatter(x=trials_df['trial_number'], y=trials_df['value'].expanding().max(), mode='lines', line=dict(color='#10b981', width=2)))
    fig.update_layout(**_chart_layout(200), xaxis_title="Trial", yaxis_title=metric)
    return fig

def create_heatmap(df, param1, param2, metric, params, capital, commission):
    ranges = {
        'pamrp_entry_long': list(range(10, 45, 5)), 'pamrp_exit_long': list(range(50, 95, 5)),
        'bbwp_threshold_long': list(range(30, 75, 5)), 'stop_loss_pct_long': [1,2,3,4,5,6,7,8],
        'take_profit_pct_long': [2,4,6,8,10,12,14],
    }
    r1 = ranges.get(param1, list(range(10, 50, 5))); r2 = ranges.get(param2, list(range(10, 50, 5)))
    m = np.zeros((len(r2), len(r1)))
    for i, v1 in enumerate(r1):
        for j, v2 in enumerate(r2):
            tp = params.copy(); tp[param1] = v1; tp[param2] = v2
            try:
                m[j, i] = getattr(BacktestEngine(params_to_strategy(tp), capital, commission).run(df.copy()), metric, 0)
            except: m[j, i] = 0
    fig = go.Figure(data=go.Heatmap(z=m, x=r1, y=r2, colorscale='RdYlGn', showscale=True))
    fig.update_layout(**_chart_layout(300), xaxis_title=param1, yaxis_title=param2)
    return fig

def create_multi_asset_chart(results_dict):
    fig = go.Figure()
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    for i, (sym, res) in enumerate(results_dict.items()):
        norm = (res.equity_curve / res.equity_curve.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode='lines', name=sym, line=dict(color=colors[i%len(colors)], width=2)))
    fig.update_layout(**_chart_layout(300, showlegend=True, legend=dict(orientation='h', y=1.1)), yaxis_title="Return %")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (same structure as v5.1 — abbreviated for brevity)
# ═══════════════════════════════════════════════════════════════════════════════

p = st.session_state.params
with st.sidebar:
    st.markdown("# 📊 Trading Toolkit Pro")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📈 Data")
    data_src = st.radio("Source", ["Yahoo Finance", "Sample"], horizontal=True, label_visibility="collapsed")
    INTERVALS = ["1m","5m","15m","30m","1h","1d","1wk","1mo"]
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
                    end = datetime.now(); start = end - timedelta(days=days)
                    df = fetch_yfinance(symbol, str(start.date()), str(end.date()), interval)
                else:
                    df = generate_sample_data(500)
                st.session_state.df = df; st.success(f"✅ {len(df)} bars")
            except Exception as e: st.error(str(e)[:50])
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ Strategy")
    p['trade_direction'] = st.selectbox("Direction", ["Long Only","Short Only","Both"], index=["Long Only","Short Only","Both"].index(p['trade_direction']))
    with st.expander("💰 Position Sizing", expanded=False):
        p['position_size_pct'] = st.slider("Position %", 10, 100, int(p['position_size_pct']), 10)
        p['use_kelly'] = st.toggle("Kelly Criterion", p['use_kelly'])
        if p['use_kelly']: p['kelly_fraction'] = st.slider("Kelly Fraction", 0.1, 1.0, p['kelly_fraction'], 0.1)
    with st.expander("📊 PAMRP", expanded=False):
        p['pamrp_enabled'] = st.toggle("Enable", p['pamrp_enabled'], key="pe")
        if p['pamrp_enabled']:
            p['pamrp_length'] = st.slider("Length", 5, 50, p['pamrp_length'], key="pl")
            p['pamrp_entry_long'] = st.slider("Entry Long", 5, 50, p['pamrp_entry_long'], key="pel")
            p['pamrp_entry_short'] = st.slider("Entry Short", 50, 95, p['pamrp_entry_short'], key="pes")
            p['pamrp_exit_long'] = st.slider("Exit Long", 50, 100, p['pamrp_exit_long'], key="pxl")
            p['pamrp_exit_short'] = st.slider("Exit Short", 0, 50, p['pamrp_exit_short'], key="pxs")
    with st.expander("📊 BBWP", expanded=False):
        p['bbwp_enabled'] = st.toggle("Enable", p['bbwp_enabled'], key="be")
        if p['bbwp_enabled']:
            p['bbwp_length'] = st.slider("Length", 5, 30, p['bbwp_length'], key="bl")
            p['bbwp_lookback'] = st.slider("Lookback", 50, 400, p['bbwp_lookback'], key="blb")
            p['bbwp_sma_length'] = st.slider("SMA", 2, 15, p['bbwp_sma_length'], key="bsma")
            p['bbwp_threshold_long'] = st.slider("Thresh Long", 20, 80, p['bbwp_threshold_long'], key="btl")
            p['bbwp_threshold_short'] = st.slider("Thresh Short", 20, 80, p['bbwp_threshold_short'], key="bts")
            p['bbwp_ma_filter'] = st.selectbox("MA Filter", ["disabled","decreasing","increasing"], index=["disabled","decreasing","increasing"].index(p['bbwp_ma_filter']), key="bmf")
    with st.expander("📈 ADX", expanded=False):
        p['adx_enabled'] = st.toggle("Enable", p['adx_enabled'], key="ae")
        if p['adx_enabled']:
            p['adx_length'] = st.slider("Length", 7, 21, p['adx_length'], key="al")
            p['adx_threshold'] = st.slider("Threshold", 10, 40, p['adx_threshold'], key="at")
    with st.expander("📈 MA Trend", expanded=False):
        p['ma_trend_enabled'] = st.toggle("Enable", p['ma_trend_enabled'], key="mae")
        if p['ma_trend_enabled']:
            p['ma_type'] = st.selectbox("Type", ["sma","ema"], index=["sma","ema"].index(p['ma_type']), key="mat")
            p['ma_fast_length'] = st.slider("Fast", 10, 100, p['ma_fast_length'], key="maf")
            p['ma_slow_length'] = st.slider("Slow", 50, 400, p['ma_slow_length'], key="mas")
    with st.expander("📈 RSI", expanded=False):
        p['rsi_enabled'] = st.toggle("Enable", p['rsi_enabled'], key="re")
        if p['rsi_enabled']:
            p['rsi_length'] = st.slider("Length", 5, 21, p['rsi_length'], key="rl")
            p['rsi_oversold'] = st.slider("Oversold", 15, 45, p['rsi_oversold'], key="ros")
            p['rsi_overbought'] = st.slider("Overbought", 55, 85, p['rsi_overbought'], key="rob")
    with st.expander("📊 Volume", expanded=False):
        p['volume_enabled'] = st.toggle("Enable", p['volume_enabled'], key="ve")
        if p['volume_enabled']:
            p['volume_ma_length'] = st.slider("MA Length", 10, 50, p['volume_ma_length'], key="vml")
            p['volume_multiplier'] = st.slider("Multiplier", 0.5, 2.0, p['volume_multiplier'], 0.1, key="vm")
    with st.expander("📈 Supertrend", expanded=False):
        p['supertrend_enabled'] = st.toggle("Enable", p['supertrend_enabled'], key="ste")
        if p['supertrend_enabled']:
            p['supertrend_period'] = st.slider("Period", 5, 20, p['supertrend_period'], key="stp")
            p['supertrend_multiplier'] = st.slider("Mult", 1.0, 5.0, p['supertrend_multiplier'], 0.5, key="stm")
    with st.expander("📈 VWAP", expanded=False): p['vwap_enabled'] = st.toggle("Enable", p['vwap_enabled'], key="vwe")
    with st.expander("📈 MACD", expanded=False):
        p['macd_enabled'] = st.toggle("Enable", p['macd_enabled'], key="mce")
        if p['macd_enabled']:
            p['macd_fast'] = st.slider("Fast", 5, 20, p['macd_fast'], key="mcf")
            p['macd_slow'] = st.slider("Slow", 15, 40, p['macd_slow'], key="mcs")
            p['macd_signal'] = st.slider("Signal", 5, 15, p['macd_signal'], key="mcsi")
    st.markdown("### 🚪 Exits")
    with st.expander("🛑 Stop Loss", expanded=False):
        p['stop_loss_enabled'] = st.toggle("Enable", p['stop_loss_enabled'], key="sle")
        if p['stop_loss_enabled']:
            p['stop_loss_pct_long'] = st.slider("% Long", 0.5, 15.0, p['stop_loss_pct_long'], 0.5, key="sll")
            p['stop_loss_pct_short'] = st.slider("% Short", 0.5, 15.0, p['stop_loss_pct_short'], 0.5, key="sls")
    with st.expander("🎯 Take Profit", expanded=False):
        p['take_profit_enabled'] = st.toggle("Enable", p['take_profit_enabled'], key="tpe")
        if p['take_profit_enabled']:
            p['take_profit_pct_long'] = st.slider("% Long", 1.0, 30.0, p['take_profit_pct_long'], 0.5, key="tpl")
            p['take_profit_pct_short'] = st.slider("% Short", 1.0, 30.0, p['take_profit_pct_short'], 0.5, key="tps")
    with st.expander("📉 Trailing Stop", expanded=False):
        p['trailing_stop_enabled'] = st.toggle("Enable", p['trailing_stop_enabled'], key="tse")
        if p['trailing_stop_enabled']: p['trailing_stop_pct'] = st.slider("Trail %", 0.5, 10.0, p['trailing_stop_pct'], 0.5, key="tsp")
    with st.expander("📉 ATR Trail", expanded=False):
        p['atr_trailing_enabled'] = st.toggle("Enable", p['atr_trailing_enabled'], key="ate")
        if p['atr_trailing_enabled']:
            p['atr_length'] = st.slider("Length", 7, 21, p['atr_length'], key="atl")
            p['atr_multiplier'] = st.slider("Mult", 1.0, 5.0, p['atr_multiplier'], 0.5, key="atm")
    with st.expander("📊 PAMRP Exit", expanded=False): p['pamrp_exit_enabled'] = st.toggle("Enable", p['pamrp_exit_enabled'], key="pxe")
    with st.expander("📈 Stoch RSI Exit", expanded=False):
        p['stoch_rsi_exit_enabled'] = st.toggle("Enable", p['stoch_rsi_exit_enabled'], key="sre")
        if p['stoch_rsi_exit_enabled']:
            p['stoch_rsi_length'] = st.slider("Length", 7, 21, p['stoch_rsi_length'], key="srl")
            p['stoch_rsi_k'] = st.slider("K", 2, 5, p['stoch_rsi_k'], key="srk")
            p['stoch_rsi_d'] = st.slider("D", 2, 5, p['stoch_rsi_d'], key="srd")
            p['stoch_rsi_overbought'] = st.slider("OB", 60, 90, p['stoch_rsi_overbought'], key="srob")
            p['stoch_rsi_oversold'] = st.slider("OS", 10, 40, p['stoch_rsi_oversold'], key="sros")
    with st.expander("⏱️ Time Exit", expanded=False):
        p['time_exit_enabled'] = st.toggle("Enable", p['time_exit_enabled'], key="txe")
        if p['time_exit_enabled']: p['time_exit_bars'] = st.slider("Max Bars", 5, 100, p['time_exit_bars'], key="txb")
    with st.expander("📈 MA Exit", expanded=False):
        p['ma_exit_enabled'] = st.toggle("Enable", p['ma_exit_enabled'], key="mxe")
        if p['ma_exit_enabled']:
            p['ma_exit_fast'] = st.slider("Fast", 3, 20, p['ma_exit_fast'], key="mxf")
            p['ma_exit_slow'] = st.slider("Slow", 10, 40, p['ma_exit_slow'], key="mxs")
    with st.expander("📊 BBWP Exit", expanded=False):
        p['bbwp_exit_enabled'] = st.toggle("Enable", p['bbwp_exit_enabled'], key="bxe")
        if p['bbwp_exit_enabled']: p['bbwp_exit_threshold'] = st.slider("Threshold", 60, 95, p['bbwp_exit_threshold'], key="bxt")
st.session_state.params = p


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 📊 Trading Toolkit Pro")
tabs = st.tabs(["🔬 Backtest", "🎯 Optimize", "⚖️ Compare", "🎲 Monte Carlo", "📅 Calendar", "🔥 Heatmap", "🌐 Multi-Asset", "📋 Trades"])

# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    c1, c2, c3, c4 = st.columns([2,1,1,1])
    st.session_state.capital = c1.number_input("Capital $", 1000, 1000000, st.session_state.capital, 1000)
    st.session_state.commission = c2.number_input("Comm %", 0.0, 1.0, st.session_state.commission, 0.01)
    slippage = c3.number_input("Slip %", 0.0, 0.5, 0.0, 0.01)
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🚀 Run", type="primary", use_container_width=True)
    if run:
        if st.session_state.df is None: st.warning("Load data first!")
        else:
            with st.spinner("Running..."):
                engine = BacktestEngine(params_to_strategy(st.session_state.params), st.session_state.capital, st.session_state.commission, slippage)
                st.session_state.backtest_results = engine.run(st.session_state.df.copy())
                st.success(f"✅ {st.session_state.backtest_results.num_trades} trades")
    if st.session_state.backtest_results:
        r = st.session_state.backtest_results
        # Row 1: Core performance
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Return", f"{r.total_return_pct:.2f}%")
        c2.metric("CAGR", f"{r.cagr:.2f}%")
        c3.metric("Sharpe", f"{r.sharpe_ratio:.3f}")
        c4.metric("Sortino", f"{r.sortino_ratio:.3f}")
        c5.metric("Calmar", f"{r.calmar_ratio:.3f}")
        c6.metric("Max DD", f"{r.max_drawdown_pct:.2f}%")
        # Row 2: Trade quality
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Trades", r.num_trades)
        c2.metric("Win%", f"{r.win_rate:.1f}%")
        c3.metric("PF", f"{r.profit_factor:.2f}")
        c4.metric("Expectancy", f"${r.expectancy:.2f}")
        c5.metric("Payoff", f"{r.payoff_ratio:.2f}")
        c6.metric("Mkt Time", f"{r.pct_time_in_market:.0f}%")
        # Row 3: Trade details
        with st.expander("📊 Detailed Metrics", expanded=False):
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Avg Win", f"${r.avg_winner:.2f}")
            c2.metric("Avg Loss", f"${abs(r.avg_loser):.2f}")
            c3.metric("Avg Win %", f"{r.avg_winner_pct:.2f}%")
            c4.metric("Avg Loss %", f"{abs(r.avg_loser_pct):.2f}%")
            c5.metric("Max Consec L", r.max_consecutive_losses)
            c6.metric("Max Consec W", r.max_consecutive_wins)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Avg Bars", f"{r.avg_bars_held:.1f}")
            c2.metric("Longest DD", f"{r.longest_drawdown_bars} bars")
            c3.metric("Avg MAE", f"{r.avg_mae:.2f}%")
            c4.metric("Avg MFE", f"{r.avg_mfe:.2f}%")
        st.plotly_chart(create_equity_chart(r), use_container_width=True)
        if st.session_state.df is not None:
            st.plotly_chart(create_price_chart_with_trades(st.session_state.df, r.trades), use_container_width=True)
    elif st.session_state.df is not None:
        st.plotly_chart(create_price_chart_with_trades(st.session_state.df, None), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZE TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### 🎯 Optimization")
    c1,c2,c3,c4 = st.columns(4)
    opt_metric = c1.selectbox("Metric", ["sharpe_ratio","total_return_pct","profit_factor","sortino_ratio"])
    opt_dir = c2.selectbox("Dir", ["long_only","short_only","both"])
    opt_trials = c3.slider("Trials", 20, 300, 100)
    opt_min = c4.slider("Min Trades", 5, 30, 10)
    c1,c2,c3 = st.columns(3)
    train_pct = c1.slider("Train %", 50, 90, 70)
    use_wf = c2.toggle("Walk-Forward", False)
    n_folds = c3.slider("Folds", 3, 10, 5) if use_wf else 5
    active_display = get_active_filters_display(st.session_state.params)
    if active_display: st.caption(f"🔍 Active: **{', '.join(active_display)}**")
    if st.button("🎯 Optimize", type="primary", use_container_width=True):
        if st.session_state.df is None: st.warning("Load data first!")
        else:
            ef = {k: v for k, v in st.session_state.params.items() if k.endswith('_enabled')}
            with st.spinner("Optimizing..."):
                res = optimize_strategy(df=st.session_state.df.copy(), enabled_filters=ef, metric=opt_metric,
                    n_trials=opt_trials, min_trades=opt_min, initial_capital=st.session_state.capital,
                    commission_pct=st.session_state.commission, trade_direction=opt_dir, train_pct=train_pct/100,
                    use_walkforward=use_wf, n_folds=n_folds, show_progress=False)
                st.session_state.optimization_results = res; st.success("✅ Done!")
    if st.session_state.optimization_results:
        res = st.session_state.optimization_results
        c1,c2,c3 = st.columns(3)
        ml = res.metric.replace('_',' ').title()
        c1.metric(f"Train ({ml})", f"{res.train_value:.4f}")
        c2.metric(f"Test ({ml})", f"{res.test_value:.4f}")
        if res.train_value > 0:
            deg = ((res.train_value - res.test_value) / res.train_value) * 100
            c3.metric("Degradation", f"{deg:.1f}%", delta_color="normal" if deg < 30 else "inverse")
        if res.walkforward_folds:
            st.plotly_chart(create_walkforward_chart(res.walkforward_folds), use_container_width=True, config={'displayModeBar': False})
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("### Best Params")
            st.dataframe(pd.DataFrame([{"P": k, "V": str(v)[:20]} for k, v in res.best_params.items()
                if not k.startswith('trade_direction') and not isinstance(v, TradeDirection)]), use_container_width=True, hide_index=True, height=250)
        with c2:
            br = res.best_results
            st.markdown("### Full-Data Performance")
            st.dataframe(pd.DataFrame([{"Metric": "Return", "Value": f"{br.total_return_pct:.2f}%"},
                {"Metric": "CAGR", "Value": f"{br.cagr:.2f}%"}, {"Metric": "Sharpe", "Value": f"{br.sharpe_ratio:.3f}"},
                {"Metric": "Max DD", "Value": f"{br.max_drawdown_pct:.2f}%"}, {"Metric": "Trades", "Value": str(br.num_trades)}]),
                use_container_width=True, hide_index=True)
        st.plotly_chart(create_optimization_chart(res.all_trials, res.metric), use_container_width=True)
        st.button("📋 Apply Best Params", on_click=apply_best_params_callback, use_container_width=True)
        if st.session_state.pop('_apply_success', False):
            st.success(f"✅ Applied! Capital: ${st.session_state.capital:,.0f} | Commission: {st.session_state.commission}%")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARE TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### ⚖️ Strategy vs Buy & Hold")
    if st.session_state.df is not None:
        if st.button("🔄 Compare", use_container_width=True):
            df = st.session_state.df.copy()
            strat_res = BacktestEngine(params_to_strategy(st.session_state.params), st.session_state.capital, st.session_state.commission).run(df)
            if strat_res.trades:
                ft = strat_res.trades[0]; ep = ft.entry_price; ed = ft.entry_date; fp = df['close'].iloc[-1]
                bh_pct = ((ep - fp) / ep * 100) if ft.direction == 'short' else ((fp - ep) / ep * 100)
                mask = df.index >= ed; prices = df.loc[mask, 'close']
                peak = prices.expanding().max(); bh_dd = ((prices - peak) / peak * 100).min()
                st.dataframe(pd.DataFrame([
                    {'Strategy': '📊 Yours', 'Return %': f"{strat_res.total_return_pct:.2f}%", 'CAGR': f"{strat_res.cagr:.2f}%", 'Max DD': f"{strat_res.max_drawdown_pct:.2f}%", 'Sharpe': f"{strat_res.sharpe_ratio:.3f}"},
                    {'Strategy': '📈 B&H', 'Return %': f"{bh_pct:.2f}%", 'CAGR': '-', 'Max DD': f"{bh_dd:.2f}%", 'Sharpe': '-'}
                ]), use_container_width=True, hide_index=True)
                diff = strat_res.total_return_pct - bh_pct
                (st.success if diff > 0 else st.warning)(f"{'🏆 Strategy beats' if diff > 0 else '📉 B&H beats strategy by'} {abs(diff):.2f}%")
                ba = calculate_beta_alpha(strat_res.equity_curve.pct_change().dropna(), df.loc[mask, 'close'].pct_change().dropna())
                c1,c2,c3 = st.columns(3)
                c1.metric("Beta", f"{ba['beta']:.3f}"); c2.metric("Alpha (ann)", f"{ba['alpha']:.2f}%"); c3.metric("Correlation", f"{ba['correlation']:.3f}")
            else: st.warning("No trades generated")
    else: st.info("Load data first")


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO TAB — ENHANCED
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 🎲 Monte Carlo Simulation")

    c1, c2, c3 = st.columns(3)
    mc_method = c1.selectbox("Method", ["Trade Shuffle", "Return Bootstrap", "Noise Injection"])
    n_sims = c2.slider("Simulations", 100, 5000, 1000)
    ruin_pct = c3.slider("Ruin Threshold %", 10, 90, 50)

    method_map = {"Trade Shuffle": "trade_shuffle", "Return Bootstrap": "return_bootstrap", "Noise Injection": "noise_injection"}
    method_key = method_map[mc_method]

    extra_kwargs = {}
    if method_key == 'noise_injection':
        extra_kwargs['noise_pct'] = st.slider("Noise %", 5.0, 50.0, 20.0, 5.0)
    if method_key == 'return_bootstrap':
        extra_kwargs['block_size'] = st.slider("Block Size", 1, 20, 5)

    if st.button("🎲 Run Monte Carlo", use_container_width=True):
        r = st.session_state.backtest_results
        if r and r.trades:
            with st.spinner(f"Running {n_sims} {mc_method} simulations..."):
                mc = run_monte_carlo(
                    trades=r.trades,
                    equity_curve=r.equity_curve,
                    method=method_key,
                    n_simulations=n_sims,
                    initial_capital=st.session_state.capital,
                    ruin_pct=ruin_pct,
                    bars_per_year=r.bars_per_year,
                    **extra_kwargs
                )
                if mc:
                    st.session_state._mc_result = mc

    mc = st.session_state.get('_mc_result')
    if mc:
        # Key stats
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Risk of Ruin", f"{mc.risk_of_ruin:.1%}")
        c2.metric("5th Pctl", f"${mc.equity_percentiles['5%']:,.0f}")
        c3.metric("Median", f"${mc.equity_percentiles['50%']:,.0f}")
        c4.metric("95th Pctl", f"${mc.equity_percentiles['95%']:,.0f}")

        # Confidence bands
        st.markdown("### 📈 Equity Confidence Bands")
        st.plotly_chart(create_mc_confidence_chart(mc), use_container_width=True)

        # Final equity distribution
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 💰 Final Equity Distribution")
            st.plotly_chart(create_mc_histogram(mc.final_equities, xaxis_title='Final Equity $'), use_container_width=True)
        with c2:
            st.markdown("### 📉 Max Drawdown Distribution")
            st.plotly_chart(create_mc_histogram(mc.max_drawdowns, xaxis_title='Max DD %'), use_container_width=True)

        # Sharpe distribution (bootstrap only)
        if mc.sharpe_distribution is not None:
            st.markdown("### 📊 Sharpe Ratio Distribution")
            st.plotly_chart(create_mc_histogram(mc.sharpe_distribution, xaxis_title='Sharpe'), use_container_width=True)

        # DD percentiles
        c1,c2,c3 = st.columns(3)
        c1.metric("5th Pctl DD", f"{mc.dd_percentiles['5%']:.1f}%")
        c2.metric("Median DD", f"{mc.dd_percentiles['50%']:.1f}%")
        c3.metric("95th Pctl DD", f"{mc.dd_percentiles['95%']:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# CALENDAR ANALYTICS TAB — NEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 📅 Calendar Analytics")

    if st.session_state.df is not None:
        if st.button("📅 Analyze Calendar", use_container_width=True):
            with st.spinner("Analyzing..."):
                cal = analyze_calendar(st.session_state.df)
                st.session_state._calendar = cal
                if st.session_state.backtest_results and st.session_state.backtest_results.trades:
                    st.session_state._trade_calendar = analyze_trade_calendar(st.session_state.backtest_results.trades)

        cal = st.session_state.get('_calendar')
        if cal:
            st.markdown("### 📊 Day-of-Week Returns")
            st.caption("Average daily return by day of week (historical price data)")
            st.plotly_chart(create_dow_chart(cal.day_of_week_df), use_container_width=True)
            st.dataframe(cal.day_of_week_df, use_container_width=True, hide_index=True)

            st.markdown("### 🗓️ Monthly Seasonality")
            st.dataframe(cal.monthly_df, use_container_width=True, hide_index=True)

            st.markdown("### 🌡️ Year × Month Heatmap")
            st.plotly_chart(create_monthly_heatmap(cal.monthly_heatmap), use_container_width=True)

            if not cal.day_of_month_df.empty:
                st.markdown("### 📆 Day-of-Month Average Return")
                dom = cal.day_of_month_df
                colors = ['#ef4444' if v < 0 else '#10b981' for v in dom['Avg %']]
                fig = go.Figure(go.Bar(x=dom['Day'], y=dom['Avg %'], marker_color=colors))
                fig.update_layout(**_chart_layout(200), xaxis_title='Day of Month', yaxis_title='Avg %')
                st.plotly_chart(fig, use_container_width=True)

            # Trade calendar (if backtest was run)
            tcal = st.session_state.get('_trade_calendar')

            # Hourly stats (only shown for intraday data)
            if cal.is_intraday and cal.hourly_df is not None and not cal.hourly_df.empty:
                st.markdown("### ⏰ Hourly Returns (Intraday)")
                st.caption("Average per-bar return by hour of day")
                hdf = cal.hourly_df
                colors = ['#ef4444' if v < 0 else '#10b981' for v in hdf['Avg %']]
                fig = go.Figure(go.Bar(x=hdf['Hour'], y=hdf['Avg %'], marker_color=colors))
                fig.update_layout(**_chart_layout(200), xaxis_title='Hour', yaxis_title='Avg %')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(hdf, use_container_width=True, hide_index=True)
            if tcal and not tcal.trades_by_day.empty:
                st.markdown("### 📊 Trade Performance by Entry Day")
                st.caption("How your strategy's trades performed based on entry day")
                st.dataframe(tcal.trades_by_day, use_container_width=True, hide_index=True)
            if tcal and not tcal.trades_by_month.empty:
                st.markdown("### 📊 Trade Performance by Entry Month")
                st.dataframe(tcal.trades_by_month, use_container_width=True, hide_index=True)
    else:
        st.info("Load data first")


# ═══════════════════════════════════════════════════════════════════════════════
# HEATMAP TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### 🔥 Parameter Sensitivity")
    param_options = ['pamrp_entry_long','pamrp_exit_long','bbwp_threshold_long','stop_loss_pct_long','take_profit_pct_long']
    c1,c2,c3 = st.columns(3)
    p1 = c1.selectbox("Param X", param_options, index=0)
    p2 = c2.selectbox("Param Y", param_options, index=2)
    hm_metric = c3.selectbox("Metric", ["sharpe_ratio","total_return_pct","profit_factor"])
    if st.button("🔥 Generate Heatmap", use_container_width=True):
        if st.session_state.df is not None:
            with st.spinner("Calculating..."):
                st.plotly_chart(create_heatmap(st.session_state.df, p1, p2, hm_metric, st.session_state.params, st.session_state.capital, st.session_state.commission), use_container_width=True)
        else: st.warning("Load data first")


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-ASSET TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 🌐 Multi-Asset Portfolio")
    symbols_input = st.text_input("Symbols (comma-separated)", "SPY, QQQ, IWM")
    if st.button("📊 Run Multi-Asset", use_container_width=True):
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        results_dict = {}
        with st.spinner(f"Testing {len(symbols)} assets..."):
            for sym in symbols:
                try:
                    end = datetime.now(); start = end - timedelta(days=730)
                    df = fetch_yfinance(sym, str(start.date()), str(end.date()), '1d')
                    results_dict[sym] = BacktestEngine(params_to_strategy(st.session_state.params), st.session_state.capital, st.session_state.commission).run(df)
                except Exception as e: st.warning(f"{sym}: {str(e)[:30]}")
        if results_dict:
            st.plotly_chart(create_multi_asset_chart(results_dict), use_container_width=True)
            st.dataframe(pd.DataFrame([{'Symbol': s, 'Return %': f"{r.total_return_pct:.2f}%", 'CAGR': f"{r.cagr:.2f}%",
                'Sharpe': f"{r.sharpe_ratio:.3f}", 'Max DD': f"{r.max_drawdown_pct:.2f}%", 'Trades': r.num_trades}
                for s, r in results_dict.items()]), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADES TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("### 📋 Trade Log")
    if st.session_state.backtest_results and st.session_state.backtest_results.trades:
        trades = st.session_state.backtest_results.trades
        c1,c2 = st.columns(2)
        dir_f = c1.selectbox("Dir", ["All","Long","Short"])
        res_f = c2.selectbox("Result", ["All","Winners","Losers"])
        flt = trades
        if dir_f != "All": flt = [t for t in flt if t.direction.lower() == dir_f.lower()]
        if res_f == "Winners": flt = [t for t in flt if t.pnl > 0]
        elif res_f == "Losers": flt = [t for t in flt if t.pnl <= 0]
        if flt:
            c1,c2,c3 = st.columns(3)
            c1.metric("Trades", len(flt))
            total_pnl = sum(t.pnl for t in flt)
            c2.metric("Total PnL", f"${total_pnl:,.2f}")
            c3.metric("Avg", f"${np.mean([t.pnl for t in flt]):.2f}")
            trade_df = pd.DataFrame([{
                'Entry': t.entry_date, 'Exit': t.exit_date, 'Dir': t.direction,
                'Entry$': round(t.entry_price, 2), 'Exit$': round(t.exit_price, 2) if t.exit_price else None,
                'Size$': round(t.size_dollars, 0), 'PnL$': round(t.pnl, 2), 'PnL%': round(t.pnl_pct, 3),
                'Bars': t.bars_held, 'MAE%': round(t.mae, 2), 'MFE%': round(t.mfe, 2), 'Reason': t.exit_reason
            } for t in flt])
            st.download_button("📥 Export CSV", trade_df.to_csv(index=False), "trades.csv", use_container_width=True)
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
    else: st.info("Run backtest first")


st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#64748b;font-size:0.7rem;">Trading Toolkit Pro v6.0</p>', unsafe_allow_html=True)