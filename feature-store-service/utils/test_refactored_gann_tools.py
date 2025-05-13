"""
Tests for Refactored Gann Tools.

This module contains tests for the refactored Gann analysis tools.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the refactored versions
from feature_store_service.indicators.gann import (
    GannAngles,
    GannFan,
    GannSquare,
    GannTimeCycles,
    GannGrid,
    GannBox,
    GannSquare144,
    GannHexagon
)


class TestRefactoredGannTools(unittest.TestCase):
    """Test case for Refactored Gann Tools."""

    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data
        dates = [datetime.now() - timedelta(days=i) for i in range(200)]
        dates.reverse()  # Oldest first
        
        # Create a simple uptrend with some volatility
        base_price = 100.0
        trend = np.linspace(0, 50, 200)  # Uptrend from 100 to 150
        volatility = np.random.normal(0, 2, 200)  # Add some noise
        
        prices = base_price + trend + volatility
        
        # Create OHLCV data
        self.data = pd.DataFrame({
            'open': prices - np.random.uniform(0, 1, 200),
            'high': prices + np.random.uniform(1, 2, 200),
            'low': prices - np.random.uniform(1, 2, 200),
            'close': prices + np.random.uniform(0, 1, 200),
            'volume': np.random.uniform(1000, 5000, 200)
        }, index=dates)

    def test_gann_angles(self):
        """Test GannAngles implementation."""
        # Initialize GannAngles
        gann_angles = GannAngles(
            pivot_type="swing_low",
            angle_types=["1x1", "1x2", "2x1"],
            lookback_period=100,
            price_scaling=1.0,
            projection_bars=50
        )
        
        # Calculate results
        result = gann_angles.calculate(self.data)
        
        # Check that pivot point is marked
        self.assertTrue(result['gann_angle_pivot_idx'].any())
        
        # Check that angle columns exist
        for angle_type in ["1x1", "1x2", "2x1"]:
            self.assertIn(f"gann_angle_up_{angle_type}", result.columns)
            self.assertIn(f"gann_angle_down_{angle_type}", result.columns)

    def test_gann_fan(self):
        """Test GannFan implementation."""
        # Initialize GannFan
        gann_fan = GannFan(
            pivot_type="swing_low",
            fan_angles=["1x1", "1x2", "2x1"],
            lookback_period=100,
            price_scaling=1.0,
            projection_bars=50
        )
        
        # Calculate results
        result = gann_fan.calculate(self.data)
        
        # Check that pivot point is marked
        self.assertTrue(result['gann_fan_pivot_idx'].any())
        
        # Check that fan columns exist
        for angle_type in ["1x1", "1x2", "2x1"]:
            self.assertIn(f"gann_fan_{angle_type}", result.columns)

    def test_gann_square(self):
        """Test GannSquare implementation."""
        # Initialize GannSquare
        gann_square = GannSquare(
            square_type="square_of_9",
            pivot_price=None,
            auto_detect_pivot=True,
            lookback_period=100,
            num_levels=2
        )
        
        # Calculate results
        result = gann_square.calculate(self.data)
        
        # Check that square columns exist
        for i in range(1, 3):  # num_levels=2
            for angle in [45, 90, 135, 180]:
                self.assertIn(f"gann_sq_sup_{angle}_{i}", result.columns)
                self.assertIn(f"gann_sq_res_{angle}_{i}", result.columns)

    def test_gann_time_cycles(self):
        """Test GannTimeCycles implementation."""
        # Initialize GannTimeCycles
        gann_time_cycles = GannTimeCycles(
            cycle_lengths=[30, 60, 90],
            starting_point_type="major_low",
            lookback_period=100,
            auto_detect_start=True,
            max_cycles=2
        )
        
        # Calculate results
        result = gann_time_cycles.calculate(self.data)
        
        # Check that starting point is marked
        self.assertTrue(result['gann_time_cycle_start'].any())
        
        # Check that cycle columns exist
        for length in [30, 60, 90]:
            for i in range(1, 3):  # max_cycles=2
                self.assertIn(f"gann_time_cycle_{length}_{i}", result.columns)

    def test_gann_grid(self):
        """Test GannGrid implementation."""
        # Initialize GannGrid
        gann_grid = GannGrid(
            pivot_type="swing_low",
            lookback_period=100,
            num_price_lines=3,
            num_time_lines=3
        )
        
        # Calculate results
        result = gann_grid.calculate(self.data)
        
        # Check that pivot point is marked
        self.assertTrue(result['gann_grid_pivot_idx'].any())
        
        # Check that grid columns exist
        for i in range(1, 4):  # num_price_lines=3
            self.assertIn(f"gann_grid_price_p{i}", result.columns)
            self.assertIn(f"gann_grid_price_m{i}", result.columns)

    def test_gann_box(self):
        """Test GannBox implementation."""
        # Initialize GannBox
        gann_box = GannBox(
            start_pivot_type="major_low",
            end_pivot_type="major_high",
            lookback_period=100,
            price_divisions=[0.5],
            time_divisions=[0.5]
        )
        
        # Calculate results
        result = gann_box.calculate(self.data)
        
        # Check that start and end points are marked
        self.assertTrue(result['gann_box_start_idx'].any())
        self.assertTrue(result['gann_box_end_idx'].any())
        
        # Check that division columns exist
        self.assertIn("gann_box_price_50", result.columns)
        self.assertIn("gann_box_time_50", result.columns)

    def test_gann_square144(self):
        """Test GannSquare144 implementation."""
        # Initialize GannSquare144
        gann_square144 = GannSquare144(
            pivot_type="major_low",
            lookback_period=100,
            num_levels=2
        )
        
        # Calculate results
        result = gann_square144.calculate(self.data)
        
        # Check that pivot point is marked
        self.assertTrue(result['gann_sq144_pivot_idx'].any())
        
        # Check that square columns exist
        for i in range(1, 3):  # num_levels=2
            self.assertIn(f"gann_sq144_sup_{i}", result.columns)
            self.assertIn(f"gann_sq144_res_{i}", result.columns)

    def test_gann_hexagon(self):
        """Test GannHexagon implementation."""
        # Initialize GannHexagon
        gann_hexagon = GannHexagon(
            pivot_type="major_low",
            lookback_period=100,
            degrees=[60, 120, 180],
            num_cycles=2
        )
        
        # Calculate results
        result = gann_hexagon.calculate(self.data)
        
        # Check that pivot point is marked
        self.assertTrue(result['gann_hexagon_pivot_idx'].any())
        
        # Check that hexagon columns exist
        for cycle in range(1, 3):  # num_cycles=2
            for degree in [60, 120, 180]:
                self.assertIn(f"gann_hexagon_price_c{cycle}_d{degree}", result.columns)


if __name__ == '__main__':
    unittest.main()
