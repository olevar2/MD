"""
Unit tests for multi-timeframe integration components.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.multi_timeframe import (
    MultiTimeframeIndicator,
    TimeframeComparison,
    TimeframeConfluenceScanner
)
from core.base_indicator import BaseIndicator


class SimpleMAIndicator(BaseIndicator):
    """Simple moving average indicator for testing."""
    
    category = "moving_average"
    
    def __init__(self, period: int = 20, price_source: str = "close", **kwargs):
        """Initialize simple MA indicator."""
        self.period = period
        self.price_source = price_source
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate simple MA."""
        result = data.copy()
        result[f'sma_{self.period}'] = data[self.price_source].rolling(window=self.period).mean()
        return result


class TestMultiTimeframeIndicator(unittest.TestCase):
    """Test suite for multi-timeframe indicator wrapper."""

    def setUp(self):
        """Set up test data with multiple timeframes."""
        np.random.seed(42)
        
        # Create 1-minute data
        minutes = 1440  # 24 hours
        start_date = datetime(2023, 1, 1)
        date_range_1min = [start_date + timedelta(minutes=i) for i in range(minutes)]
        
        # Generate random price data
        price = 100
        prices_1min = [price]
        for _ in range(1, minutes):
            price += np.random.normal(0, 0.1)
            prices_1min.append(price)
        
        # Create DataFrame with 1-minute data
        self.data_1min = pd.DataFrame({
            'open': prices_1min,
            'high': [p + np.random.uniform(0, 0.05) for p in prices_1min],
            'low': [p - np.random.uniform(0, 0.05) for p in prices_1min],
            'close': prices_1min,
            'volume': [np.random.randint(1000, 10000) for _ in range(minutes)]
        }, index=date_range_1min)
        
        # Create 5-minute data
        self.data_5min = self._resample_ohlc(self.data_1min, '5T')
        
        # Create 15-minute data
        self.data_15min = self._resample_ohlc(self.data_1min, '15T')
        
        # Create hourly data
        self.data_1hour = self._resample_ohlc(self.data_1min, '1H')
        
        # Initialize multi-timeframe indicator with a simple MA
        self.base_indicator = SimpleMAIndicator(period=20)
        self.multi_tf_indicator = MultiTimeframeIndicator(
            indicator=self.base_indicator,
            timeframes=['5T', '15T', '1H']
        )
    
    def _resample_ohlc(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to a higher timeframe."""
        resampled = pd.DataFrame()
        resampled['open'] = df['open'].resample(timeframe).first()
        resampled['high'] = df['high'].resample(timeframe).max()
        resampled['low'] = df['low'].resample(timeframe).min()
        resampled['close'] = df['close'].resample(timeframe).last()
        resampled['volume'] = df['volume'].resample(timeframe).sum()
        return resampled
    
    def test_multi_timeframe_calculation(self):
        """Test calculation of indicator across multiple timeframes."""
        # Calculate indicator on 1-minute data with multi-timeframe integration
        result = self.multi_tf_indicator.calculate(self.data_1min)
        
        # Should have columns for each timeframe
        self.assertIn('sma_20_5T', result.columns)
        self.assertIn('sma_20_15T', result.columns)
        self.assertIn('sma_20_1H', result.columns)
        
        # Check that values are correctly aligned
        # 5-minute MA should change every 5 minutes
        changes_5min = result['sma_20_5T'].diff().dropna().abs()
        non_zero_changes_5min = changes_5min[changes_5min > 0]
        
        # Percentage of rows with changes should be close to 1/5 (20%)
        change_ratio_5min = len(non_zero_changes_5min) / len(changes_5min)
        self.assertAlmostEqual(change_ratio_5min, 0.2, delta=0.05)
        
        # 15-minute MA should change every 15 minutes
        changes_15min = result['sma_20_15T'].diff().dropna().abs()
        non_zero_changes_15min = changes_15min[changes_15min > 0]
        
        # Percentage of rows with changes should be close to 1/15 (≈6.67%)
        change_ratio_15min = len(non_zero_changes_15min) / len(changes_15min)
        self.assertAlmostEqual(change_ratio_15min, 1/15, delta=0.02)
        
        # Hourly MA should change every 60 minutes
        changes_1hour = result['sma_20_1H'].diff().dropna().abs()
        non_zero_changes_1hour = changes_1hour[changes_1hour > 0]
        
        # Percentage of rows with changes should be close to 1/60 (≈1.67%)
        change_ratio_1hour = len(non_zero_changes_1hour) / len(changes_1hour)
        self.assertAlmostEqual(change_ratio_1hour, 1/60, delta=0.01)
    
    def test_alignment_methods(self):
        """Test different alignment methods."""
        # Test forward fill alignment
        ff_indicator = MultiTimeframeIndicator(
            indicator=self.base_indicator,
            timeframes=['5T', '15T'],
            alignment_method='ffill'
        )
        
        result_ff = ff_indicator.calculate(self.data_1min)
        
        # Forward fill should not have NaN values after initial calculation period
        warmup_period = self.base_indicator.period * 15  # Worst case for 15-minute data
        self.assertEqual(
            result_ff['sma_20_15T'].iloc[warmup_period:].isna().sum(),
            0
        )
        
        # Test nearest alignment
        nearest_indicator = MultiTimeframeIndicator(
            indicator=self.base_indicator,
            timeframes=['5T', '15T'],
            alignment_method='nearest'
        )
        
        nearest_indicator.calculate(self.data_1min)  # Should not raise errors
    
    def test_comparison_with_native_calculation(self):
        """Test that multi-timeframe results match direct calculation on resampled data."""
        # Calculate with multi-timeframe indicator
        multi_result = self.multi_tf_indicator.calculate(self.data_1min)
        
        # Calculate directly on hourly data
        direct_result = self.base_indicator.calculate(self.data_1hour)
        
        # Get the values at hourly points
        hourly_indices = self.data_1hour.index
        multi_hourly_values = multi_result.loc[hourly_indices]['sma_20_1H'].dropna()
        direct_hourly_values = direct_result['sma_20'].dropna()
        
        # Values should match at hourly points
        pd.testing.assert_series_equal(
            multi_hourly_values, 
            direct_hourly_values,
            check_names=False
        )


class TestTimeframeComparison(unittest.TestCase):
    """Test suite for timeframe comparison functionality."""

    def setUp(self):
        """Set up test data with multiple timeframes."""
        np.random.seed(42)
        
        # Create 1-minute data for 3 days
        days = 3
        minutes_per_day = 1440
        minutes = days * minutes_per_day
        start_date = datetime(2023, 1, 1)
        date_range_1min = [start_date + timedelta(minutes=i) for i in range(minutes)]
        
        # Generate price with trend and reversal
        # Day 1: Uptrend
        # Day 2: Sideways
        # Day 3: Downtrend
        prices_1min = []
        price = 100
        
        # Day 1: Uptrend
        for i in range(minutes_per_day):
            price += 0.01 + np.random.normal(0, 0.05)
            prices_1min.append(price)
            
        # Day 2: Sideways
        for i in range(minutes_per_day):
            price += np.random.normal(0, 0.1)
            prices_1min.append(price)
            
        # Day 3: Downtrend
        for i in range(minutes_per_day):
            price -= 0.01 + np.random.normal(0, 0.05)
            prices_1min.append(price)
        
        # Create DataFrame with 1-minute data
        self.data_1min = pd.DataFrame({
            'open': [p - np.random.uniform(0, 0.03) for p in prices_1min],
            'high': [p + np.random.uniform(0, 0.05) for p in prices_1min],
            'low': [p - np.random.uniform(0, 0.05) for p in prices_1min],
            'close': prices_1min,
            'volume': [np.random.randint(1000, 10000) for _ in range(minutes)]
        }, index=date_range_1min)
        
        # Create higher timeframe data
        self.data_15min = self._resample_ohlc(self.data_1min, '15T')
        self.data_1hour = self._resample_ohlc(self.data_1min, '1H')
        self.data_4hour = self._resample_ohlc(self.data_1min, '4H')
        
        # Initialize timeframe comparison with moving averages
        self.tf_comparison = TimeframeComparison(
            indicator=SimpleMAIndicator(period=20),
            timeframes=['15T', '1H', '4H']
        )
    
    def _resample_ohlc(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to a higher timeframe."""
        resampled = pd.DataFrame()
        resampled['open'] = df['open'].resample(timeframe).first()
        resampled['high'] = df['high'].resample(timeframe).max()
        resampled['low'] = df['low'].resample(timeframe).min()
        resampled['close'] = df['close'].resample(timeframe).last()
        resampled['volume'] = df['volume'].resample(timeframe).sum()
        return resampled
    
    def test_trend_alignment_detection(self):
        """Test detection of trend alignment across timeframes."""
        # Calculate trend alignment
        result = self.tf_comparison.calculate_trend_alignment(self.data_1min)
        
        # Should have trend alignment column
        self.assertIn('trend_alignment', result.columns)
        
        # Check alignment values during different market phases
        # Day 1 (uptrend): Should have positive alignment for most of the day
        day1_data = result[result.index.date == datetime(2023, 1, 1).date()]
        day1_alignment = day1_data['trend_alignment'].dropna()
        self.assertGreater(
            (day1_alignment > 0).sum() / len(day1_alignment),
            0.6,  # At least 60% of the day should show positive alignment
            "Day 1 should show predominantly positive alignment during uptrend"
        )
        
        # Day 3 (downtrend): Should have negative alignment for most of the day
        day3_data = result[result.index.date == datetime(2023, 1, 3).date()]
        day3_alignment = day3_data['trend_alignment'].dropna()
        self.assertGreater(
            (day3_alignment < 0).sum() / len(day3_alignment),
            0.6,  # At least 60% of the day should show negative alignment
            "Day 3 should show predominantly negative alignment during downtrend"
        )
    
    def test_divergence_detection(self):
        """Test detection of divergences between timeframes."""
        # Calculate with divergence detection
        result = self.tf_comparison.calculate_divergences(self.data_1min)
        
        # Should have divergence columns
        self.assertIn('tf_divergence', result.columns)
        
        # There should be some divergences detected during trend changes
        self.assertGreater(
            (result['tf_divergence'] != 0).sum(),
            0,
            "Should detect some divergences during the 3-day period with trend changes"
        )
    
    def test_momentum_confluence(self):
        """Test detection of momentum confluence across timeframes."""
        # Calculate momentum confluence
        result = self.tf_comparison.calculate_momentum_confluence(self.data_1min)
        
        # Should have confluence column
        self.assertIn('momentum_confluence', result.columns)
        
        # Confluence values should range from -1 to 1
        max_confluence = result['momentum_confluence'].max()
        min_confluence = result['momentum_confluence'].min()
        
        self.assertLessEqual(max_confluence, 1.0)
        self.assertGreaterEqual(min_confluence, -1.0)
        
        # Should see strong confluence during trend periods
        # Day 1 (uptrend): Should have positive confluence
        day1_data = result[result.index.date == datetime(2023, 1, 1).date()]
        day1_confluence = day1_data['momentum_confluence'].dropna()
        
        self.assertGreater(
            day1_confluence.mean(),
            0,
            "Day 1 should show positive average momentum confluence during uptrend"
        )
        
        # Day 3 (downtrend): Should have negative confluence
        day3_data = result[result.index.date == datetime(2023, 1, 3).date()]
        day3_confluence = day3_data['momentum_confluence'].dropna()
        
        self.assertLess(
            day3_confluence.mean(),
            0,
            "Day 3 should show negative average momentum confluence during downtrend"
        )


class TestTimeframeConfluenceScanner(unittest.TestCase):
    """Test suite for multi-timeframe confluence scanner."""

    def setUp(self):
        """Set up test data with multiple indicators and timeframes."""
        np.random.seed(42)
        
        # Create 1-minute data for 2 days
        days = 2
        minutes_per_day = 1440
        minutes = days * minutes_per_day
        start_date = datetime(2023, 1, 1)
        date_range_1min = [start_date + timedelta(minutes=i) for i in range(minutes)]
        
        # Generate trending price data
        price = 100
        prices_1min = [price]
        
        # First day: uptrend
        for i in range(1, minutes_per_day):
            price += 0.01 + 0.005 * np.sin(i / 100) + np.random.normal(0, 0.03)
            prices_1min.append(price)
            
        # Second day: downtrend
        for i in range(minutes_per_day):
            price -= 0.01 + 0.005 * np.sin(i / 100) + np.random.normal(0, 0.03)
            prices_1min.append(price)
        
        # Create DataFrame with 1-minute data
        self.data_1min = pd.DataFrame({
            'open': [p - np.random.uniform(0, 0.02) for p in prices_1min],
            'high': [p + np.random.uniform(0, 0.04) for p in prices_1min],
            'low': [p - np.random.uniform(0, 0.04) for p in prices_1min],
            'close': prices_1min,
            'volume': [np.random.randint(1000, 10000) for _ in range(minutes)]
        }, index=date_range_1min)
        
        # Create test indicators
        self.ma20 = SimpleMAIndicator(period=20)
        self.ma50 = SimpleMAIndicator(period=50)
        
        # Custom trend indicator for testing
        class TrendIndicator(BaseIndicator):
            """Simple trend indicator for testing."""
            
            category = "trend"
            
            def __init__(self, period: int = 20, **kwargs):
                """Initialize trend indicator."""
                self.period = period
                
            def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                """Calculate trend indicator (positive when trending up)."""
                result = data.copy()
                ma_fast = data['close'].rolling(window=self.period).mean()
                ma_slow = data['close'].rolling(window=self.period*2).mean()
                result[f'trend_{self.period}'] = (ma_fast - ma_slow).apply(np.sign)
                return result
        
        self.trend_ind = TrendIndicator(period=30)
        
        # Initialize confluence scanner
        self.confluence_scanner = TimeframeConfluenceScanner(
            indicators=[self.ma20, self.ma50, self.trend_ind],
            timeframes=['15T', '1H', '4H']
        )
    
    def test_confluence_scanning(self):
        """Test multi-indicator confluence scanning."""
        # Calculate confluence scores
        result = self.confluence_scanner.scan_for_confluence(self.data_1min)
        
        # Should have confluence score columns
        self.assertIn('bullish_confluence', result.columns)
        self.assertIn('bearish_confluence', result.columns)
        
        # Scores should be between 0 and 1
        self.assertGreaterEqual(result['bullish_confluence'].min(), 0)
        self.assertLessEqual(result['bullish_confluence'].max(), 1)
        self.assertGreaterEqual(result['bearish_confluence'].min(), 0)
        self.assertLessEqual(result['bearish_confluence'].max(), 1)
        
        # Day 1 (uptrend): Should have higher bullish confluence
        day1_data = result[result.index.date == datetime(2023, 1, 1).date()]
        day1_bull = day1_data['bullish_confluence'].dropna().mean()
        day1_bear = day1_data['bearish_confluence'].dropna().mean()
        
        self.assertGreater(
            day1_bull, day1_bear,
            "Day 1 should have higher bullish confluence during uptrend"
        )
        
        # Day 2 (downtrend): Should have higher bearish confluence
        day2_data = result[result.index.date == datetime(2023, 1, 2).date()]
        day2_bull = day2_data['bullish_confluence'].dropna().mean()
        day2_bear = day2_data['bearish_confluence'].dropna().mean()
        
        self.assertGreater(
            day2_bear, day2_bull,
            "Day 2 should have higher bearish confluence during downtrend"
        )
    
    def test_signal_generation(self):
        """Test generation of trading signals based on confluence."""
        # Calculate signals with default threshold
        result = self.confluence_scanner.generate_signals(self.data_1min)
        
        # Should have signal column
        self.assertIn('confluence_signal', result.columns)
        
        # Signals should be -1, 0, or 1
        unique_signals = result['confluence_signal'].dropna().unique()
        for signal in unique_signals:
            self.assertIn(signal, [-1, 0, 1])
        
        # Should have some signals in each direction
        signals = result['confluence_signal'].dropna()
        self.assertGreater((signals == 1).sum(), 0, "Should have some bullish signals")
        self.assertGreater((signals == -1).sum(), 0, "Should have some bearish signals")
    
    def test_persistence_filter(self):
        """Test filtering of signals based on persistence."""
        # Calculate with high persistence requirement
        result = self.confluence_scanner.generate_signals(
            self.data_1min,
            min_persistence=5  # Require 5 consecutive periods of confluence
        )
        
        # Get signal transitions
        signal_changes = result['confluence_signal'].dropna().diff().abs()
        
        # Should have fewer signal changes with persistence filter
        self.assertLess(
            (signal_changes > 0).sum(),
            minutes_per_day / 100,  # Less than 1% of periods should have transitions
            "Persistence filter should reduce signal transitions"
        )


if __name__ == '__main__':
    unittest.main()
