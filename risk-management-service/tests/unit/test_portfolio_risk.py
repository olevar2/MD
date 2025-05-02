"""
Tests for the Portfolio Risk calculations.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from risk_management_service.services.portfolio_risk_calculator import PortfolioRiskCalculator, PortfolioRiskMetrics
from risk_management_service.risk_manager import RiskManager, Position
from risk_management_service.stress_testing import StressTestingEngine

@pytest.fixture
def risk_calculator():
    """Create a PortfolioRiskCalculator instance for testing."""
    risk_manager = RiskManager(
        initial_balance=100000.0,
        max_position_size=10000.0,
        max_leverage=20.0,
        max_drawdown=0.20,
        risk_per_trade=0.02
    )
    return PortfolioRiskCalculator()

@pytest.fixture
def sample_portfolio_positions():
    """Provides sample portfolio positions for testing."""
    return {
        'EURUSD': {
            'symbol': 'EURUSD',
            'current_value': 100000,
            'size': 100000,
            'unrealized_pnl': 500,
            'direction': 'long'
        },
        'GBPUSD': {
            'symbol': 'GBPUSD',
            'current_value': 50000,
            'size': 50000,
            'unrealized_pnl': -200,
            'direction': 'long'
        },
        'USDJPY': {
            'symbol': 'USDJPY',
            'current_value': 80000,
            'size': 80000,
            'unrealized_pnl': 300,
            'direction': 'short'
        }
    }

@pytest.fixture
def sample_market_data():
    """Provides sample market data for testing."""
    dates = [datetime.now() - timedelta(days=x) for x in range(30)]
    return {
        'EURUSD': {
            'dates': dates,
            'prices': [1.1 + 0.001 * x for x in range(30)]
        },
        'GBPUSD': {
            'dates': dates,
            'prices': [1.3 + 0.002 * x for x in range(30)]
        },
        'USDJPY': {
            'dates': dates,
            'prices': [110.0 + 0.005 * x for x in range(30)]
        }
    }

class TestPortfolioRisk:
    """Test suite for portfolio risk calculations."""

    def test_calculate_portfolio_var_success(self, risk_calculator, sample_portfolio_positions, sample_market_data):
        """Test successful calculation of Portfolio Value at Risk (VaR)."""
        var = risk_calculator.calculate_portfolio_var(
            sample_portfolio_positions,
            sample_market_data,
            confidence_level=0.95,
            time_horizon_days=1
        )
        
        assert isinstance(var, float)
        assert var > 0
        assert var < 1.0  # VaR should be less than 100% of portfolio value

    def test_assess_concentration_risk_success(self, risk_calculator, sample_portfolio_positions):
        """Test successful assessment of portfolio concentration risk."""
        result = risk_calculator.assess_concentration_risk(sample_portfolio_positions)
        
        assert isinstance(result, dict)
        assert 'HHI' in result
        assert 'concentration_level' in result
        assert 'largest_positions' in result
        
        # Validate HHI calculation
        # HHI should be between 0 and 10000 (0.0 to 1.0 if expressed as ratio)
        assert 0 <= result['HHI'] <= 10000
        
        # Validate concentration level
        assert result['concentration_level'] in ['Low', 'Moderate', 'High']
        
        # Validate largest positions
        assert len(result['largest_positions']) <= 3
        assert all('symbol' in pos and 'share' in pos for pos in result['largest_positions'])

    def test_calculate_correlation_risk_success(self, risk_calculator, sample_portfolio_positions, sample_market_data):
        """Test successful calculation of correlation risk."""
        result = risk_calculator.calculate_correlation_risk(
            sample_portfolio_positions,
            sample_market_data,
            lookback_days=30
        )
        
        assert isinstance(result, dict)
        assert 'max_correlation' in result
        assert 'avg_correlation' in result
        
        # Validate correlation values
        assert -1 <= result['max_correlation'] <= 1
        assert -1 <= result['avg_correlation'] <= 1

    def test_calculate_portfolio_risk_metrics_success(self, risk_calculator, sample_portfolio_positions, sample_market_data):
        """Test successful calculation of comprehensive portfolio risk metrics."""
        metrics = risk_calculator.calculate_portfolio_risk_metrics(
            sample_portfolio_positions,
            sample_market_data,
            lookback_days=30
        )
        
        assert isinstance(metrics, PortfolioRiskMetrics)
        assert metrics.value_at_risk > 0
        assert metrics.expected_shortfall > 0
        assert 0 <= metrics.concentration_score <= 10000
        assert isinstance(metrics.correlation_exposure, dict)
        assert 0 <= metrics.max_drawdown <= 1
        assert isinstance(metrics.sharpe_ratio, float)
        assert metrics.total_exposure > 0

    def test_portfolio_risk_empty_portfolio(self, risk_calculator, sample_market_data):
        """Test risk calculations with an empty portfolio."""
        empty_portfolio = {}
        
        # Test VaR calculation
        var = risk_calculator.calculate_portfolio_var(empty_portfolio, sample_market_data)
        assert var == 0.0
        
        # Test concentration risk
        concentration = risk_calculator.assess_concentration_risk(empty_portfolio)
        assert concentration['HHI'] == 0
        assert concentration['concentration_level'] == 'None'
        assert len(concentration['largest_positions']) == 0
        
        # Test correlation risk
        correlation = risk_calculator.calculate_correlation_risk(empty_portfolio, sample_market_data)
        assert correlation['max_correlation'] == 0.0
        assert correlation['avg_correlation'] == 0.0
        
        # Test comprehensive metrics
        metrics = risk_calculator.calculate_portfolio_risk_metrics(empty_portfolio, sample_market_data)
        assert metrics.value_at_risk == 0.0
        assert metrics.expected_shortfall == 0.0
        assert metrics.concentration_score == 0.0
        assert metrics.total_exposure == 0.0

    def test_portfolio_risk_single_position(self, risk_calculator, sample_market_data):
        """Test risk calculations with a single position."""
        single_position = {
            'EURUSD': {
                'symbol': 'EURUSD',
                'current_value': 100000,
                'size': 100000,
                'unrealized_pnl': 500,
                'direction': 'long'
            }
        }
        
        # Test concentration risk
        concentration = risk_calculator.assess_concentration_risk(single_position)
        assert concentration['HHI'] == 10000  # Maximum concentration for single position
        assert concentration['concentration_level'] == 'High'
        assert len(concentration['largest_positions']) == 1
        assert concentration['largest_positions'][0]['share'] == 100.0

        # Test correlation risk
        correlation = risk_calculator.calculate_correlation_risk(single_position, sample_market_data)
        assert correlation['max_correlation'] == 0.0  # No correlation with single position
        assert correlation['avg_correlation'] == 0.0

    def test_risk_metrics_validation(self, risk_calculator, sample_portfolio_positions, sample_market_data):
        """Test validation of risk metrics calculations."""
        # Test with invalid confidence level
        with pytest.raises(ValueError):
            risk_calculator.calculate_portfolio_var(
                sample_portfolio_positions,
                sample_market_data,
                confidence_level=1.5  # Invalid confidence level
            )
            
        # Test with invalid time horizon
        with pytest.raises(ValueError):
            risk_calculator.calculate_portfolio_var(
                sample_portfolio_positions,
                sample_market_data,
                time_horizon_days=0  # Invalid time horizon
            )

        # Test with missing market data
        bad_market_data = {
            'EURUSD': sample_market_data['EURUSD']
            # Missing GBPUSD and USDJPY
        }
        
        # Should handle missing market data gracefully
        var = risk_calculator.calculate_portfolio_var(sample_portfolio_positions, bad_market_data)
        assert isinstance(var, float)
        assert var >= 0
