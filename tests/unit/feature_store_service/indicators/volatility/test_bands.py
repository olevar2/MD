"""
Unit tests for volatility bands module.
"""

import unittest
import pandas as pd
import numpy as np
from feature_store_service.indicators.volatility.bands import (
    BollingerBands, KeltnerChannels, DonchianChannels
)


class TestVolatilityBands(unittest.TestCase):
    """Test cases for volatility bands indicators."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=100)
        self.df = pd.DataFrame({
            'open': np.random.normal(100, 1, 100),
            'high': np.random.normal(101, 1, 100),
            'low': np.random.normal(99, 1, 100),
            'close': np.random.normal(100, 1, 100)
        }, index=dates)
        
        # Ensure high is always >= close and open
        self.df['high'] = self.df[['high', 'open', 'close']].max(axis=1)
        
        # Ensure low is always <= close and open
        self.df['low'] = self.df[['low', 'open', 'close']].min(axis=1)
        
        # Create a trend for testing
        for i in range(30, 70):
            self.df.iloc[i, self.df.columns.get_loc('close')] = 100 + (i - 30) * 0.2
            self.df.iloc[i, self.df.columns.get_loc('high')] = 101 + (i - 30) * 0.2
            self.df.iloc[i, self.df.columns.get_loc('low')] = 99 + (i - 30) * 0.2
    
    def test_bollinger_bands(self):
        """Test BollingerBands indicator."""
        indicator = BollingerBands(window=20, num_std=2.0)
        result = indicator.calculate(self.df)
        
        # Check that the indicator added the expected columns
        self.assertIn("bb_middle_20", result.columns)
        self.assertIn("bb_upper_20_2.0", result.columns)
        self.assertIn("bb_lower_20_2.0", result.columns)
        self.assertIn("bb_width_20_2.0", result.columns)
        self.assertIn("bb_pct_b_20_2.0", result.columns)
        
        # Check that the middle band is the SMA of close
        expected_middle = self.df['close'].rolling(window=20).mean()
        pd.testing.assert_series_equal(
            result["bb_middle_20"].dropna(),
            expected_middle.dropna(),
            check_names=False
        )
        
        # Check that upper band is always >= middle band
        self.assertTrue(all(result["bb_upper_20_2.0"].dropna() >= result["bb_middle_20"].dropna()))
        
        # Check that lower band is always <= middle band
        self.assertTrue(all(result["bb_lower_20_2.0"].dropna() <= result["bb_middle_20"].dropna()))
    
    def test_keltner_channels(self):
        """Test KeltnerChannels indicator."""
        indicator = KeltnerChannels(window=20, atr_window=10, atr_multiplier=2.0)
        result = indicator.calculate(self.df)
        
        # Check that the indicator added the expected columns
        self.assertIn("kc_middle_20_ema", result.columns)
        self.assertIn("kc_upper_20_10_2.0", result.columns)
        self.assertIn("kc_lower_20_10_2.0", result.columns)
        
        # Check that the middle band is the EMA of close
        expected_middle = self.df['close'].ewm(span=20, adjust=False).mean()
        pd.testing.assert_series_equal(
            result["kc_middle_20_ema"].dropna(),
            expected_middle.dropna(),
            check_names=False
        )
        
        # Check that upper band is always >= middle band
        self.assertTrue(all(result["kc_upper_20_10_2.0"].dropna() >= result["kc_middle_20_ema"].dropna()))
        
        # Check that lower band is always <= middle band
        self.assertTrue(all(result["kc_lower_20_10_2.0"].dropna() <= result["kc_middle_20_ema"].dropna()))
    
    def test_donchian_channels(self):
        """Test DonchianChannels indicator."""
        indicator = DonchianChannels(window=20)
        result = indicator.calculate(self.df)
        
        # Check that the indicator added the expected columns
        self.assertIn("donchian_upper_20", result.columns)
        self.assertIn("donchian_lower_20", result.columns)
        self.assertIn("donchian_middle_20", result.columns)
        self.assertIn("donchian_width_20", result.columns)
        
        # Check that the upper band is the max high
        expected_upper = self.df['high'].rolling(window=20).max()
        pd.testing.assert_series_equal(
            result["donchian_upper_20"].dropna(),
            expected_upper.dropna(),
            check_names=False
        )
        
        # Check that the lower band is the min low
        expected_lower = self.df['low'].rolling(window=20).min()
        pd.testing.assert_series_equal(
            result["donchian_lower_20"].dropna(),
            expected_lower.dropna(),
            check_names=False
        )
        
        # Check that middle band is the average of upper and lower
        expected_middle = (expected_upper + expected_lower) / 2
        pd.testing.assert_series_equal(
            result["donchian_middle_20"].dropna(),
            expected_middle.dropna(),
            check_names=False
        )


if __name__ == "__main__":
    unittest.main()