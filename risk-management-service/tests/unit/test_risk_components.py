"""
Tests for the enhanced risk management components.
"""
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from risk_management_service.risk_manager import RiskManager, Position
from risk_management_service.stress_testing import (
    StressTestingEngine,
    StressScenario
)
from risk_management_service.circuit_breaker import (
    CircuitBreakerManager,
    CircuitBreakerState,
    TriggerType,
    CircuitBreakerConfig
)
from risk_management_service.portfolio_risk import (
    PortfolioRiskCalculator,
    PortfolioMetrics
)

class MockHistoricalData:
    """Mock historical data provider for testing."""
    
    def get_prices(self, symbol: str, lookback_days: int) -> pd.Series:
        """Get mock historical prices."""
        dates = pd.date_range(end=datetime.utcnow(), periods=lookback_days)
        if symbol == "EUR/USD":
            # Generate slightly trending data
            prices = np.linspace(1.1, 1.2, lookback_days) + np.random.normal(0, 0.001, lookback_days)
        else:
            # Generate random walk
            prices = np.random.normal(0, 0.001, lookback_days).cumsum() + 1.0
        return pd.Series(prices, index=dates)

class TestStressTesting(unittest.TestCase):
    """Test suite for stress testing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager(
            initial_balance=100000.0,
            max_position_size=10000.0,
            max_leverage=20.0,
            max_drawdown=0.20,
            risk_per_trade=0.02
        )
        self.stress_testing = StressTestingEngine(self.risk_manager)
        
    def test_historical_scenarios(self):
        """Test historical stress scenarios."""
        # Add test positions
        self.risk_manager.add_position(
            symbol="EUR/USD",
            size=5000.0,
            price=1.2000,
            direction="long",
            leverage=5.0
        )
        
        current_prices = {"EUR/USD": 1.2000}
        results = self.stress_testing.run_stress_test(
            self.stress_testing.historical_scenarios["2008_crisis"],
            current_prices
        )
        
        self.assertIsNotNone(results)
        self.assertGreater(results['value_at_risk'], 0)
        self.assertIn('stressed_prices', results)
        self.assertIn('risk_metrics', results)
        
    def test_custom_scenario(self):
        """Test custom scenario creation and execution."""
        scenario = self.stress_testing.create_custom_scenario(
            name="Custom Test",
            description="Test scenario",
            price_shocks={"EUR/USD": -0.10},
            volatility_multiplier=2.0,
            liquidity_factor=1.5
        )
        
        current_prices = {"EUR/USD": 1.2000}
        results = self.stress_testing.run_stress_test(scenario, current_prices)
        
        self.assertEqual(results['scenario_name'], "Custom Test")
        self.assertGreater(results['value_at_risk'], 0)

class TestCircuitBreaker(unittest.TestCase):
    """Test suite for circuit breaker functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.circuit_breaker = CircuitBreakerManager()
        
    def test_drawdown_circuit_breaker(self):
        """Test drawdown-based circuit breaker."""
        metrics = {'drawdown': 0.25}  # 25% drawdown
        
        triggered = self.circuit_breaker.check_and_update(metrics)
        self.assertTrue(any(t['trigger_type'] == TriggerType.DRAWDOWN for t in triggered))
        self.assertEqual(
            self.circuit_breaker.states['global_drawdown'],
            CircuitBreakerState.TRIGGERED
        )
        
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery process."""
        # First trigger
        metrics = {'drawdown': 0.25}
        self.circuit_breaker.check_and_update(metrics)
        
        # Then improve conditions
        metrics = {'drawdown': 0.05}
        with unittest.mock.patch('datetime.datetime') as mock_datetime:
            # Simulate time passage
            mock_datetime.utcnow.return_value = datetime.utcnow() + timedelta(hours=2)
            triggered = self.circuit_breaker.check_and_update(metrics)
            
        self.assertEqual(
            self.circuit_breaker.states['global_drawdown'],
            CircuitBreakerState.RECOVERY
        )

class TestPortfolioRisk(unittest.TestCase):
    """Test suite for portfolio risk calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager(
            initial_balance=100000.0,
            max_position_size=10000.0,
            max_leverage=20.0,
            max_drawdown=0.20,
            risk_per_trade=0.02
        )
        self.stress_testing = StressTestingEngine(self.risk_manager)
        self.historical_data = MockHistoricalData()
        self.portfolio_risk = PortfolioRiskCalculator(
            self.risk_manager,
            self.stress_testing,
            self.historical_data
        )
        
    def test_risk_metrics_calculation(self):
        """Test calculation of portfolio risk metrics."""
        # Add test positions
        self.risk_manager.add_position(
            symbol="EUR/USD",
            size=5000.0,
            price=1.2000,
            direction="long",
            leverage=5.0
        )
        
        current_prices = {"EUR/USD": 1.2000}
        metrics = self.portfolio_risk.calculate_portfolio_metrics(current_prices)
        
        self.assertIsInstance(metrics, PortfolioMetrics)
        self.assertGreater(metrics.total_value, 0)
        self.assertGreater(metrics.exposure, 0)
        self.assertGreaterEqual(metrics.var_95, 0)
        self.assertGreaterEqual(metrics.var_99, 0)
        
    def test_risk_recommendations(self):
        """Test risk-based recommendations."""
        # Add high leverage position
        self.risk_manager.add_position(
            symbol="EUR/USD",
            size=10000.0,
            price=1.2000,
            direction="long",
            leverage=15.0  # High leverage
        )
        
        current_prices = {"EUR/USD": 1.2000}
        metrics = self.portfolio_risk.calculate_portfolio_metrics(current_prices)
        recommendations = self.portfolio_risk.get_risk_recommendations(metrics)
        
        self.assertTrue(len(recommendations) > 0)
        self.assertTrue(
            any(r['issue'] == 'High Leverage' for r in recommendations)
        )

if __name__ == '__main__':
    unittest.main()
