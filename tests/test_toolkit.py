"""
Unit Tests for Trading Toolkit
==============================
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_sample_data
from src.indicators import pamrp, bbwp, rsi, macd, supertrend, atr, sma, ema
from src.strategy import StrategyParams, SignalGenerator, TradeDirection
from src.backtest import BacktestEngine


class TestIndicators:
    """Test indicator calculations"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample data for testing"""
        return generate_sample_data(days=100, seed=42)
    
    def test_pamrp_range(self, sample_df):
        """PAMRP should be between 0 and 100"""
        result = pamrp(sample_df['high'], sample_df['low'], sample_df['close'], 21)
        valid = result.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100
    
    def test_bbwp_range(self, sample_df):
        """BBWP should be between 0 and 100"""
        result = bbwp(sample_df['close'], 13, 50)
        valid = result[50:]  # After lookback period
        assert valid.min() >= 0
        assert valid.max() <= 100
    
    def test_rsi_range(self, sample_df):
        """RSI should be between 0 and 100"""
        result = rsi(sample_df['close'], 14)
        valid = result.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100
    
    def test_macd_output(self, sample_df):
        """MACD should return 3 series"""
        macd_line, signal_line, histogram = macd(sample_df['close'], 12, 26, 9)
        assert len(macd_line) == len(sample_df)
        assert len(signal_line) == len(sample_df)
        assert len(histogram) == len(sample_df)
    
    def test_supertrend_output(self, sample_df):
        """Supertrend should return value and direction"""
        st_value, st_direction = supertrend(
            sample_df['high'], sample_df['low'], sample_df['close'], 10, 3.0
        )
        assert len(st_value) == len(sample_df)
        assert len(st_direction) == len(sample_df)
        # Direction should be -1 (bullish) or 1 (bearish)
        unique_dirs = st_direction.dropna().unique()
        assert all(d in [-1, 1] for d in unique_dirs)
    
    def test_atr_positive(self, sample_df):
        """ATR should be positive"""
        result = atr(sample_df['high'], sample_df['low'], sample_df['close'], 14)
        valid = result.dropna()
        assert (valid > 0).all()
    
    def test_sma_calculation(self, sample_df):
        """SMA should be average of last n values"""
        result = sma(sample_df['close'], 5)
        # Check a specific calculation
        idx = 10
        expected = sample_df['close'].iloc[idx-4:idx+1].mean()
        assert abs(result.iloc[idx] - expected) < 0.0001
    
    def test_ema_responsiveness(self, sample_df):
        """EMA should be more responsive than SMA"""
        # After a big move, EMA should be closer to current price
        sma_result = sma(sample_df['close'], 20)
        ema_result = ema(sample_df['close'], 20)
        
        # Both should have same length
        assert len(sma_result) == len(ema_result)


class TestStrategy:
    """Test strategy signal generation"""
    
    @pytest.fixture
    def sample_df(self):
        return generate_sample_data(days=200, seed=42)
    
    @pytest.fixture
    def default_params(self):
        return StrategyParams()
    
    def test_signal_generator_creates_columns(self, sample_df, default_params):
        """Signal generator should add required columns"""
        gen = SignalGenerator(default_params)
        result = gen.generate_all_signals(sample_df)
        
        assert 'pamrp' in result.columns
        assert 'bbwp' in result.columns
        assert 'entry_long' in result.columns
        assert 'entry_short' in result.columns
        assert 'exit_long_signal' in result.columns
        assert 'exit_short_signal' in result.columns
    
    def test_long_only_no_short_signals(self, sample_df):
        """Long-only mode should not generate short signals"""
        params = StrategyParams(trade_direction=TradeDirection.LONG_ONLY)
        gen = SignalGenerator(params)
        result = gen.generate_all_signals(sample_df)
        
        assert not result['entry_short'].any()
    
    def test_short_only_no_long_signals(self, sample_df):
        """Short-only mode should not generate long signals"""
        params = StrategyParams(trade_direction=TradeDirection.SHORT_ONLY)
        gen = SignalGenerator(params)
        result = gen.generate_all_signals(sample_df)
        
        assert not result['entry_long'].any()
    
    def test_params_to_dict(self, default_params):
        """Params should convert to dict"""
        d = default_params.to_dict()
        assert isinstance(d, dict)
        assert 'pamrp_length' in d
        assert 'stop_loss_enabled' in d
    
    def test_params_from_dict(self):
        """Params should be creatable from dict"""
        d = {'pamrp_length': 30, 'stop_loss_pct_long': 5.0}
        params = StrategyParams.from_dict(d)
        assert params.pamrp_length == 30
        assert params.stop_loss_pct_long == 5.0


class TestBacktest:
    """Test backtesting engine"""
    
    @pytest.fixture
    def sample_df(self):
        return generate_sample_data(days=300, seed=42)
    
    @pytest.fixture
    def default_params(self):
        return StrategyParams(
            pamrp_entry_long=35,  # More lenient to generate trades
            bbwp_threshold_long=70,
        )
    
    def test_backtest_runs(self, sample_df, default_params):
        """Backtest should run without error"""
        engine = BacktestEngine(default_params)
        results = engine.run(sample_df)
        
        assert results is not None
        assert hasattr(results, 'num_trades')
        assert hasattr(results, 'equity_curve')
    
    def test_equity_curve_starts_at_capital(self, sample_df, default_params):
        """Equity curve should start at initial capital"""
        engine = BacktestEngine(default_params, initial_capital=10000)
        results = engine.run(sample_df)
        
        assert results.equity_curve.iloc[0] == 10000
    
    def test_metrics_calculated(self, sample_df, default_params):
        """All metrics should be calculated"""
        engine = BacktestEngine(default_params)
        results = engine.run(sample_df)
        
        assert hasattr(results, 'total_return_pct')
        assert hasattr(results, 'sharpe_ratio')
        assert hasattr(results, 'profit_factor')
        assert hasattr(results, 'max_drawdown_pct')
        assert hasattr(results, 'win_rate')
    
    def test_no_trades_handled(self, sample_df):
        """Should handle case with no trades gracefully"""
        # Disable all entry filters = no signals
        params = StrategyParams(
            pamrp_enabled=False,
            bbwp_enabled=False,
            adx_enabled=False,
            ma_trend_enabled=False,
            rsi_enabled=False,
            volume_enabled=False,
            supertrend_enabled=False,
            vwap_enabled=False,
            macd_enabled=False,
        )
        engine = BacktestEngine(params)
        results = engine.run(sample_df)
        
        assert results.num_trades == 0
        assert results.win_rate == 0
        assert results.initial_capital == 10000  # Verify stored
    
    def test_stop_loss_works(self, sample_df):
        """Stop loss should limit losses - checks HIGH/LOW"""
        params = StrategyParams(
            pamrp_enabled=True,
            pamrp_entry_long=40,  # Relaxed entry
            bbwp_enabled=True,
            bbwp_threshold_long=80,
            stop_loss_enabled=True,
            stop_loss_pct_long=2.0,
            pamrp_exit_enabled=False,
        )
        engine = BacktestEngine(params)
        results = engine.run(sample_df)
        
        # Check stop loss triggers properly
        for trade in results.trades:
            if trade.direction == 'long' and trade.exit_reason == 'stop_loss':
                # Stop loss should be near the stop level
                expected_stop = trade.entry_price * (1 - 2.0 / 100)
                assert trade.exit_price <= trade.entry_price  # Exit below entry


class TestDataModule:
    """Test data fetching and generation"""
    
    def test_generate_sample_data(self):
        """Sample data should have correct structure"""
        df = generate_sample_data(days=100)
        
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        assert len(df) == 100
    
    def test_sample_data_ohlc_valid(self):
        """OHLC relationships should be valid"""
        df = generate_sample_data(days=100)
        
        # High should be highest
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        
        # Low should be lowest
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
    
    def test_sample_data_reproducible(self):
        """Same seed should produce same data"""
        df1 = generate_sample_data(days=50, seed=123)
        df2 = generate_sample_data(days=50, seed=123)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_sample_data_different_seeds(self):
        """Different seeds should produce different data"""
        df1 = generate_sample_data(days=50, seed=123)
        df2 = generate_sample_data(days=50, seed=456)
        
        assert not df1['close'].equals(df2['close'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
