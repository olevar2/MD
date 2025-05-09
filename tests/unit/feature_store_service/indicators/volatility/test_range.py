"""
Unit tests for volatility range module.
"""

import unittest
import pandas as pd
import numpy as np
from feature_store_service.indicators.volatility.range import AverageTrueRange


class TestVolatilityRange(unittest.TestCase):
    """Test cases for volatility range indicators."""
    
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
        
        # Create volatility clusters for testing
        for i in range(30, 40):
            self.df.iloc[i, self.df.columns.get_loc('high')] = 105
            self.df.iloc[i, self.df.columns.get_loc('low')] = 95
        
        for i in range(60, 70):
            self.df.iloc[i, self.df.columns.get_loc('high')] = 110
            self.df.iloc[i, self.df.columns.get_loc('low')] = 90
    
    def test_average_true_range(self):
        """Test AverageTrueRange indicator."""
        indicator = AverageTrueRange(window=14)
        result = indicator.calculate(self.df)
        
        # Check that the indicator added the expected column
        self.assertIn("atr_14", result.columns)
        
        # Check that ATR is higher during high volatility periods
        normal_atr = result.iloc[20:25]["atr_14"].mean()
        high_vol_atr1 = result.iloc[35:45]["atr_14"].mean()
        high_vol_atr2 = result.iloc[65:75]["atr_14"].mean()
        
        self.assertGreater(high_vol_atr1, normal_atr)
        self.assertGreater(high_vol_atr2, normal_atr)
        
        # Check that ATR is always positive
        self.assertTrue(all(result["atr_14"].dropna() >= 0))
        
        # Check that ATR calculation is correct for a simple case
        # Create a simple dataframe with known values
        simple_df = pd.DataFrame({
            'high': [110, 112, 108, 115],
            'low': [100, 102, 98, 105],
            'close': [105, 107, 103, 110]
        })
        
        # Calculate true range manually
        tr1 = 110 - 100  # high - low
        tr2 = max(112 - 102, abs(112 - 105), abs(102 - 105))  # max(high-low, abs(high-prev_close), abs(low-prev_close))
        tr3 = max(108 - 98, abs(108 - 107), abs(98 - 107))
        tr4 = max(115 - 105, abs(115 - 103), abs(105 - 103))
        
        # Calculate ATR manually (simple average for this test)
        expected_atr = pd.Series([np.nan, tr2, tr3, tr4]).rolling(window=3).mean()
        
        # Calculate using the indicator
        simple_result = AverageTrueRange(window=3).calculate(simple_df)
        
        # Compare the last value (should be the average of tr2, tr3, tr4)
        self.assertAlmostEqual(
            simple_result["atr_3"].iloc[-1],
            (tr2 + tr3 + tr4) / 3,
            places=4
        )


if __name__ == "__main__":
    unittest.main()