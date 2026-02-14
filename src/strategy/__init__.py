"""
Strategy Module - v5 Fixed
==========================
- All parameters exposed including Stoch RSI
- Proper signal generation
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum

from ..indicators import (
    pamrp, bbwp, sma, ema, rsi, stoch_rsi, adx, atr,
    supertrend, vwap, macd, ma
)


class TradeDirection(Enum):
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    BOTH = "both"


@dataclass
class StrategyParams:
    """All strategy parameters - fully exposed"""
    
    trade_direction: TradeDirection = TradeDirection.LONG_ONLY
    
    # Position Sizing
    position_size_pct: float = 100.0  # % of capital per trade
    use_kelly: bool = False  # Use Kelly criterion
    kelly_fraction: float = 0.5  # Fraction of Kelly (half-Kelly is safer)
    
    # PAMRP
    pamrp_enabled: bool = True
    pamrp_length: int = 21
    pamrp_entry_long: int = 20
    pamrp_entry_short: int = 80
    pamrp_exit_long: int = 70
    pamrp_exit_short: int = 30
    
    # BBWP
    bbwp_enabled: bool = True
    bbwp_length: int = 13
    bbwp_lookback: int = 252
    bbwp_sma_length: int = 5
    bbwp_threshold_long: int = 50
    bbwp_threshold_short: int = 50
    bbwp_ma_filter: str = "disabled"
    
    # ADX
    adx_enabled: bool = False
    adx_length: int = 14
    adx_smoothing: int = 14
    adx_threshold: int = 20
    adx_require_di: bool = False
    
    # MA Trend
    ma_trend_enabled: bool = False
    ma_fast_length: int = 50
    ma_slow_length: int = 200
    ma_type: str = "sma"
    
    # RSI
    rsi_enabled: bool = False
    rsi_length: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    
    # Volume
    volume_enabled: bool = False
    volume_ma_length: int = 20
    volume_multiplier: float = 1.0
    
    # Supertrend
    supertrend_enabled: bool = False
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    
    # VWAP
    vwap_enabled: bool = False
    
    # MACD
    macd_enabled: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_mode: str = "histogram"
    
    # Stop Loss
    stop_loss_enabled: bool = True
    stop_loss_pct_long: float = 3.0
    stop_loss_pct_short: float = 3.0
    
    # Take Profit
    take_profit_enabled: bool = False
    take_profit_pct_long: float = 5.0
    take_profit_pct_short: float = 5.0
    
    # Trailing Stop
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 2.0
    trailing_stop_activation: float = 1.0
    
    # ATR Trailing
    atr_trailing_enabled: bool = False
    atr_length: int = 14
    atr_multiplier: float = 2.0
    
    # PAMRP Exit
    pamrp_exit_enabled: bool = True
    
    # Stoch RSI Exit - ALL params exposed
    stoch_rsi_exit_enabled: bool = False
    stoch_rsi_length: int = 14
    stoch_rsi_k: int = 3
    stoch_rsi_d: int = 3
    stoch_rsi_overbought: int = 80
    stoch_rsi_oversold: int = 20
    
    # Time Exit
    time_exit_enabled: bool = False
    time_exit_bars_long: int = 20
    time_exit_bars_short: int = 20
    
    # MA Exit
    ma_exit_enabled: bool = False
    ma_exit_fast: int = 10
    ma_exit_slow: int = 20
    
    # BBWP Exit
    bbwp_exit_enabled: bool = False
    bbwp_exit_threshold: int = 80
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Enum):
                result[k] = v.value
            else:
                result[k] = v
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StrategyParams':
        d = d.copy()
        if 'trade_direction' in d:
            if isinstance(d['trade_direction'], str):
                d['trade_direction'] = TradeDirection(d['trade_direction'])
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


class SignalGenerator:
    """Generate entry and exit signals"""
    
    def __init__(self, params: StrategyParams):
        self.params = params
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params
        
        # PAMRP
        if p.pamrp_enabled:
            df['pamrp'] = pamrp(df['high'], df['low'], df['close'], p.pamrp_length)
        else:
            df['pamrp'] = 50.0
        
        # BBWP
        if p.bbwp_enabled:
            df['bbwp'] = bbwp(df['close'], p.bbwp_length, p.bbwp_lookback)
            df['bbwp_sma'] = sma(df['bbwp'], p.bbwp_sma_length)
        else:
            df['bbwp'] = 50.0
            df['bbwp_sma'] = 50.0
        
        # ADX
        if p.adx_enabled:
            df['di_plus'], df['di_minus'], df['adx'] = adx(
                df['high'], df['low'], df['close'], p.adx_length, p.adx_smoothing
            )
        
        # MA Trend
        if p.ma_trend_enabled:
            df['ma_fast'] = ma(df['close'], p.ma_fast_length, p.ma_type)
            df['ma_slow'] = ma(df['close'], p.ma_slow_length, p.ma_type)
        
        # RSI
        if p.rsi_enabled:
            df['rsi'] = rsi(df['close'], p.rsi_length)
        
        # Stoch RSI - with all params
        if p.stoch_rsi_exit_enabled:
            df['stoch_k'], df['stoch_d'] = stoch_rsi(
                df['close'], p.stoch_rsi_length, p.stoch_rsi_k, p.stoch_rsi_d
            )
        
        # Volume
        if p.volume_enabled and 'volume' in df.columns:
            df['volume_ma'] = sma(df['volume'], p.volume_ma_length)
        
        # Supertrend
        if p.supertrend_enabled:
            df['supertrend'], df['st_direction'] = supertrend(
                df['high'], df['low'], df['close'],
                p.supertrend_period, p.supertrend_multiplier
            )
        
        # VWAP
        if p.vwap_enabled and 'volume' in df.columns:
            df['vwap'] = vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # MACD
        if p.macd_enabled:
            df['macd'], df['macd_signal'], df['macd_hist'] = macd(
                df['close'], p.macd_fast, p.macd_slow, p.macd_signal
            )
        
        # ATR
        if p.atr_trailing_enabled:
            df['atr'] = atr(df['high'], df['low'], df['close'], p.atr_length)
        
        # MA Exit
        if p.ma_exit_enabled:
            df['exit_ma_fast'] = ema(df['close'], p.ma_exit_fast)
            df['exit_ma_slow'] = ema(df['close'], p.ma_exit_slow)
        
        return df
    
    def generate_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params
        
        long_signal = pd.Series(True, index=df.index)
        short_signal = pd.Series(True, index=df.index)
        has_any_filter = False
        
        if p.pamrp_enabled:
            has_any_filter = True
            long_signal = long_signal & (df['pamrp'] < p.pamrp_entry_long)
            short_signal = short_signal & (df['pamrp'] > p.pamrp_entry_short)
        
        if p.bbwp_enabled:
            has_any_filter = True
            long_signal = long_signal & (df['bbwp'] < p.bbwp_threshold_long)
            short_signal = short_signal & (df['bbwp'] < p.bbwp_threshold_short)
            
            if p.bbwp_ma_filter == 'decreasing':
                bbwp_ma_ok = df['bbwp_sma'] < df['bbwp_sma'].shift(1)
                long_signal = long_signal & bbwp_ma_ok
                short_signal = short_signal & bbwp_ma_ok
            elif p.bbwp_ma_filter == 'increasing':
                bbwp_ma_ok = df['bbwp_sma'] > df['bbwp_sma'].shift(1)
                long_signal = long_signal & bbwp_ma_ok
                short_signal = short_signal & bbwp_ma_ok
        
        if p.adx_enabled and 'adx' in df.columns:
            has_any_filter = True
            adx_ok = df['adx'] > p.adx_threshold
            long_signal = long_signal & adx_ok
            short_signal = short_signal & adx_ok
            if p.adx_require_di:
                long_signal = long_signal & (df['di_plus'] > df['di_minus'])
                short_signal = short_signal & (df['di_minus'] > df['di_plus'])
        
        if p.ma_trend_enabled and 'ma_fast' in df.columns:
            has_any_filter = True
            long_signal = long_signal & (df['ma_fast'] > df['ma_slow'])
            short_signal = short_signal & (df['ma_fast'] < df['ma_slow'])
        
        if p.rsi_enabled and 'rsi' in df.columns:
            has_any_filter = True
            long_signal = long_signal & (df['rsi'] < p.rsi_oversold)
            short_signal = short_signal & (df['rsi'] > p.rsi_overbought)
        
        if p.volume_enabled and 'volume_ma' in df.columns:
            has_any_filter = True
            vol_ok = df['volume'] > df['volume_ma'] * p.volume_multiplier
            long_signal = long_signal & vol_ok
            short_signal = short_signal & vol_ok
        
        if p.supertrend_enabled and 'st_direction' in df.columns:
            has_any_filter = True
            long_signal = long_signal & (df['st_direction'] < 0)
            short_signal = short_signal & (df['st_direction'] > 0)
        
        if p.vwap_enabled and 'vwap' in df.columns:
            has_any_filter = True
            long_signal = long_signal & (df['close'] > df['vwap'])
            short_signal = short_signal & (df['close'] < df['vwap'])
        
        if p.macd_enabled and 'macd_hist' in df.columns:
            has_any_filter = True
            if p.macd_mode == 'histogram':
                long_signal = long_signal & (df['macd_hist'] > 0)
                short_signal = short_signal & (df['macd_hist'] < 0)
            elif p.macd_mode == 'crossover':
                long_signal = long_signal & (df['macd'] > df['macd_signal'])
                short_signal = short_signal & (df['macd'] < df['macd_signal'])
            else:
                long_signal = long_signal & (df['macd'] > 0)
                short_signal = short_signal & (df['macd'] < 0)
        
        if not has_any_filter:
            long_signal = pd.Series(False, index=df.index)
            short_signal = pd.Series(False, index=df.index)
        
        if p.trade_direction == TradeDirection.LONG_ONLY:
            short_signal = pd.Series(False, index=df.index)
        elif p.trade_direction == TradeDirection.SHORT_ONLY:
            long_signal = pd.Series(False, index=df.index)
        
        df['entry_long'] = long_signal.fillna(False)
        df['entry_short'] = short_signal.fillna(False)
        
        return df
    
    def generate_exit_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params
        
        exit_long = pd.Series(False, index=df.index)
        exit_short = pd.Series(False, index=df.index)
        
        if p.pamrp_exit_enabled and p.pamrp_enabled:
            exit_long = exit_long | (df['pamrp'] > p.pamrp_exit_long)
            exit_short = exit_short | (df['pamrp'] < p.pamrp_exit_short)
        
        if p.stoch_rsi_exit_enabled and 'stoch_k' in df.columns:
            exit_long = exit_long | (df['stoch_k'] > p.stoch_rsi_overbought)
            exit_short = exit_short | (df['stoch_k'] < p.stoch_rsi_oversold)
        
        if p.ma_exit_enabled and 'exit_ma_fast' in df.columns:
            exit_long = exit_long | (df['exit_ma_fast'] < df['exit_ma_slow'])
            exit_short = exit_short | (df['exit_ma_fast'] > df['exit_ma_slow'])
        
        if p.bbwp_exit_enabled and p.bbwp_enabled:
            exit_long = exit_long | (df['bbwp'] > p.bbwp_exit_threshold)
            exit_short = exit_short | (df['bbwp'] > p.bbwp_exit_threshold)
        
        df['exit_long_signal'] = exit_long.fillna(False)
        df['exit_short_signal'] = exit_short.fillna(False)
        
        return df
    
    def generate_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df = self.generate_entry_signals(df)
        df = self.generate_exit_signals(df)
        return df
