"""
Unit tests for Gann analysis tools.
"""

import unittest
import pandas as pd
import numpy as np
from feature_store_service.indicators.gann_tools import GannAngles, GannSquare9, GannFan


class TestGannTools(unittest.TestCase):
    """Test suite for Gann analysis tools."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 250
        
        # Generate price data with clear trend for Gann analysis
        date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # Create a strong trend with some retracements
        trend = np.linspace(0, 20, n_samples)
        base = np.cumsum(np.random.normal(0, 1, n_samples) * 0.3)  # Reduced noise
        
        # Add some cyclical component
        cycles = 5 * np.sin(np.linspace(0, 4*np.pi, n_samples))
        
        price = 100 + trend + base + cycles
        
        self.data = pd.DataFrame({
            'open': price * (1 + 0.003 * np.random.randn(n_samples)),
            'high': price * (1 + 0.006 * np.random.randn(n_samples)),
            'low': price * (1 - 0.006 * np.random.randn(n_samples)),
            'close': price,
            'volume': 1000000 * (1 + 0.1 * np.random.randn(n_samples))
        }, index=date_range)
        
        # Ensure high is highest and low is lowest
        self.data['high'] = np.maximum(
            np.maximum(self.data['high'], self.data['open']), 
            self.data['close']
        )
        self.data['low'] = np.minimum(
            np.minimum(self.data['low'], self.data['open']), 
            self.data['close']
        )
        
        # Create a pivot at index 100
        self.pivot_idx = 100
        self.pivot_price = self.data['close'].iloc[self.pivot_idx]
        self.pivot_date = self.data.index[self.pivot_idx]
    
    def test_gann_angles_calculation(self):
        """Test calculation of Gann angles from pivot point."""
        # Initialize Gann Angles with specified pivot
        gann_angles = GannAngles(pivot_price=self.pivot_price, pivot_date=self.pivot_date)
        
        # Calculate Gann angles
        angles = gann_angles.calculate_angles(self.data)
        
        # Should have 9 standard Gann angles
        self.assertEqual(len(angles), 9)
        
        # 1x1 angle should be present (45 degrees)
        self.assertIn('1x1', angles)
        
        # Check that 1x1 angle has the correct slope
        angle_1x1 = angles['1x1']
        
        # For a 1x1 angle, price should rise by 1 point for each time unit
        # Calculate expected price at end of series from pivot
        days_from_pivot = (self.data.index[-1] - self.pivot_date).days
        expected_final_price = self.pivot_price + days_from_pivot
        
        # Get the actual final price from the 1x1 angle
        actual_final_price = angle_1x1[-1]
        
        # They should be close (allowing for any time unit conversion)
        self.assertAlmostEqual(
            actual_final_price / expected_final_price,
            1.0,
            delta=0.5
        )
    
    def test_gann_angles_with_auto_pivot(self):
        """Test Gann angles with automatic pivot detection."""
        # Initialize Gann Angles with auto pivot detection
        gann_angles = GannAngles(auto_detect_pivot=True)
        
        # Calculate Gann angles
        angles = gann_angles.calculate_angles(self.data)
        
        # Should have 9 standard Gann angles
        self.assertEqual(len(angles), 9)
        
        # All angles should have the same length as the input data
        for angle_name, angle_values in angles.items():
            self.assertEqual(len(angle_values), len(self.data))
    
    def test_gann_square9_calculation(self):
        """Test Gann Square of 9 calculation."""
        # Initialize Gann Square of 9
        gann_square9 = GannSquare9(base_price=self.pivot_price)
        
        # Calculate Square of 9 levels
        levels = gann_square9.calculate_levels(n_levels=5)
        
        # Should have the requested number of levels
        self.assertEqual(len(levels), 5)
        
        # Each level should have multiple price points
        for level in levels:
            self.assertGreater(len(level), 4)
        
        # First level should contain the base price
        self.assertIn(self.pivot_price, levels[0])
        
        # Check cardinal directions (90-degree increments)
        cardinals = gann_square9.get_cardinal_levels(n_revolutions=3)
        
        # Should have 4 cardinal directions per revolution
        self.assertEqual(len(cardinals), 4 * 3)
    
    def test_gann_fan_calculation(self):
        """Test Gann Fan calculation."""
        # Initialize Gann Fan with specified pivot
        gann_fan = GannFan(pivot_price=self.pivot_price, pivot_date=self.pivot_date)
        
        # Calculate Gann Fan
        fan_lines = gann_fan.calculate_fan(self.data)
        
        # Should have multiple fan lines
        self.assertGreater(len(fan_lines), 5)
        
        # Each fan line should have the same length as input data
        for line_name, line_values in fan_lines.items():
            self.assertEqual(len(line_values), len(self.data))
        
        # Test specific fan line
        if '1x1' in fan_lines:
            # Price at pivot should match pivot price
            self.assertAlmostEqual(
                fan_lines['1x1'][self.pivot_idx],
                self.pivot_price,
                delta=0.01
            )
    
    def test_gann_support_resistance_levels(self):
        """Test Gann support and resistance level calculation."""
        # Initialize GannSquare9
        gann_square9 = GannSquare9(base_price=self.pivot_price)
        
        # Get support and resistance levels for current price
        current_price = self.data['close'].iloc[-1]
        levels = gann_square9.get_support_resistance_levels(current_price, n_levels=3)
        
        # Should have both support and resistance levels
        self.assertIn('support', levels)
        self.assertIn('resistance', levels)
        
        # Should have the requested number of levels in each direction
        self.assertEqual(len(levels['support']), 3)
        self.assertEqual(len(levels['resistance']), 3)
        
        # Support levels should be below current price
        for level in levels['support']:
            self.assertLess(level, current_price)
            
        # Resistance levels should be above current price
        for level in levels['resistance']:
            self.assertGreater(level, current_price)


if __name__ == '__main__':
    unittest.main()
