"""
Unit tests for the VaRCalculator class in the risk management service.
"""
import pytest
import numpy as np
from risk_management_service.calculators.var_calculator import VaRCalculator


class TestVaRCalculator:
    """Test suite for the VaRCalculator class."""

    def test_calculate_parametric_var_single_asset_valid_input(self):
        """
        Test the parametric VaR calculation for a single asset with valid inputs.
        Verifies that the calculation produces expected results given known inputs.
        """
        # Arrange
        calculator = VaRCalculator(confidence_level=0.95, time_horizon_days=1)
        positions = {"EURUSD": 10000}
        market_data = {
            "price": {"EURUSD": 1.1},
            "volatility": {"EURUSD": 0.01}  # 1% daily volatility
        }
        
        # Expected result calculation:
        # position_value = 10000 * 1.1 = 11000
        # scaled_volatility = 0.01 * sqrt(1) = 0.01
        # z_score for 95% confidence = 1.645
        # VaR = 11000 * 0.01 * 1.645 = 180.95
        expected_var = 11000 * 0.01 * 1.645
        
        # Act
        result = calculator._calculate_parametric_var_single_asset(positions, market_data)
        
        # Assert
        assert isinstance(result, float)
        assert np.isclose(result, expected_var, rtol=1e-5)
    
    def test_calculate_parametric_var_missing_volatility(self):
        """
        Test the parametric VaR calculation when volatility data is missing.
        Should use default volatility value of 0.01 (1%).
        """
        # Arrange
        calculator = VaRCalculator(confidence_level=0.99, time_horizon_days=1)
        positions = {"GBPUSD": 5000}
        market_data = {
            "price": {"GBPUSD": 1.25}
            # Volatility intentionally missing
        }
        
        # Expected result calculation:
        # position_value = 5000 * 1.25 = 6250
        # default volatility = 0.01
        # z_score for 99% confidence = 2.326
        # VaR = 6250 * 0.01 * 2.326 = 145.375
        expected_var = 6250 * 0.01 * 2.326
        
        # Act
        result = calculator._calculate_parametric_var_single_asset(positions, market_data)
        
        # Assert
        assert isinstance(result, float)
        assert np.isclose(result, expected_var, rtol=1e-5)
