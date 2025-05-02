"""
Tests for the risk management service.

Tests risk limits, position management, and risk calculations.
"""
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from risk_management_service.risk_manager import (
    RiskManager,
    RiskLevel,
    RiskType,
    Position
)

class TestRiskManagement(unittest.TestCase):
    """Test suite for risk management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.initial_balance = 100000.0
        self.risk_manager = RiskManager(
            initial_balance=self.initial_balance,
            max_position_size=10000.0,
            max_leverage=20.0,
            max_drawdown=0.20,
            risk_per_trade=0.02
        )

    def test_position_validation(self):
        """Test position entry validation."""
        # Test valid position
        result = self.risk_manager.check_position_entry(
            symbol="EUR/USD",
            size=1000.0,
            price=1.2000,
            direction="long",
            leverage=10.0,
            stop_loss=1.1900
        )
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['warnings']), 0)
        
        # Test excessive leverage
        result = self.risk_manager.check_position_entry(
            symbol="EUR/USD",
            size=1000.0,
            price=1.2000,
            direction="long",
            leverage=25.0  # Exceeds max
        )
        self.assertFalse(result['valid'])
        self.assertTrue(any('leverage' in err.lower() for err in result['errors']))
        
        # Test excessive position size
        result = self.risk_manager.check_position_entry(
            symbol="EUR/USD",
            size=15000.0,  # Exceeds max
            price=1.2000,
            direction="long",
            leverage=10.0
        )
        self.assertFalse(result['valid'])
        self.assertTrue(any('size' in err.lower() for err in result['errors']))

    def test_risk_per_trade(self):
        """Test risk per trade limits."""
        # Calculate maximum allowed position size based on risk
        entry_price = 1.2000
        stop_loss = 1.1900  # 100 pip stop
        risk_amount = self.initial_balance * self.risk_manager.risk_per_trade
        max_loss_per_pip = risk_amount / 100
        
        # Test position within risk limit
        result = self.risk_manager.check_position_entry(
            symbol="EUR/USD",
            size=max_loss_per_pip,  # Size that risks exactly 2%
            price=entry_price,
            direction="long",
            leverage=1.0,
            stop_loss=stop_loss
        )
        self.assertTrue(result['valid'])
        
        # Test position exceeding risk limit
        result = self.risk_manager.check_position_entry(
            symbol="EUR/USD",
            size=max_loss_per_pip * 2,  # Double the allowed risk
            price=entry_price,
            direction="long",
            leverage=1.0,
            stop_loss=stop_loss
        )
        self.assertFalse(result['valid'])
        self.assertTrue(any('risk' in err.lower() for err in result['errors']))

    def test_position_management(self):
        """Test position management functionality."""
        # Add position
        self.risk_manager.add_position(
            symbol="EUR/USD",
            size=1000.0,
            price=1.2000,
            direction="long",
            leverage=10.0,
            stop_loss=1.1900
        )
        
        self.assertEqual(len(self.risk_manager.positions), 1)
        self.assertIn("EUR/USD", self.risk_manager.positions)
        
        # Update position
        self.risk_manager.update_position(
            symbol="EUR/USD",
            current_price=1.2100,  # Price moved in favor
            size=1200.0,  # Increase size
            stop_loss=1.2000  # Move stop loss up
        )
        
        position = self.risk_manager.positions["EUR/USD"]
        self.assertEqual(position.size, 1200.0)
        self.assertEqual(position.stop_loss, 1.2000)
        
        # Close position
        pnl = self.risk_manager.close_position(
            symbol="EUR/USD",
            current_price=1.2100
        )
        
        self.assertTrue(pnl > 0)  # Should be profitable
        self.assertEqual(len(self.risk_manager.positions), 0)

    def test_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        # Add some positions
        self.risk_manager.add_position(
            symbol="EUR/USD",
            size=1000.0,
            price=1.2000,
            direction="long",
            leverage=10.0
        )
        self.risk_manager.add_position(
            symbol="GBP/USD",
            size=800.0,
            price=1.4000,
            direction="short",
            leverage=5.0
        )
        
        metrics = self.risk_manager.get_portfolio_metrics()
        
        self.assertEqual(metrics['position_count'], 2)
        self.assertEqual(
            metrics['total_exposure'],
            (1000.0 * 1.2000 * 10.0) + (800.0 * 1.4000 * 5.0)
        )
        self.assertEqual(metrics['current_balance'], self.initial_balance)

    def test_risk_limit_breaches(self):
        """Test risk limit breach detection."""
        # Add position that causes high exposure
        self.risk_manager.add_position(
            symbol="EUR/USD",
            size=5000.0,
            price=1.2000,
            direction="long",
            leverage=15.0  # High leverage
        )
        
        breached_limits = self.risk_manager.check_risk_limits()
        
        self.assertTrue(len(breached_limits) > 0)
        self.assertTrue(
            any(limit['risk_type'] == RiskType.LEVERAGE for limit in breached_limits)
        )

    def test_metrics_history(self):
        """Test metrics history tracking."""
        # Add and update positions to generate metrics
        self.risk_manager.add_position(
            symbol="EUR/USD",
            size=1000.0,
            price=1.2000,
            direction="long",
            leverage=10.0
        )
        
        # Update position multiple times
        prices = [1.2010, 1.2020, 1.2030, 1.2040]
        for price in prices:
            self.risk_manager.update_position(
                symbol="EUR/USD",
                current_price=price
            )
        
        # Get metrics history
        history_df = self.risk_manager.get_metrics_history()
        
        self.assertEqual(len(history_df), len(prices) + 1)  # Initial + updates
        self.assertTrue(all(col in history_df.columns for col in [
            'timestamp', 'current_balance', 'total_exposure', 'drawdown'
        ]))

    def test_value_at_risk(self):
        """Test Value at Risk calculation."""
        # Generate some historical data
        for _ in range(50):
            self.risk_manager.current_balance += np.random.normal(0, 1000)
            self.risk_manager._update_metrics()
            
        var = self.risk_manager.calculate_var(
            confidence_level=0.95,
            lookback_days=30
        )
        
        self.assertGreater(var, 0)  # VaR should be positive
        
        # Test with higher confidence level
        var_99 = self.risk_manager.calculate_var(
            confidence_level=0.99,
            lookback_days=30
        )
        
        self.assertGreater(var_99, var)  # 99% VaR should be higher than 95% VaR

    def test_drawdown_monitoring(self):
        """Test drawdown monitoring and limits."""
        # Simulate losses to trigger drawdown
        initial_balance = self.risk_manager.current_balance
        loss_amount = initial_balance * 0.15  # 15% loss
        
        self.risk_manager.current_balance -= loss_amount
        self.risk_manager._update_metrics()
        
        metrics = self.risk_manager.get_portfolio_metrics()
        self.assertAlmostEqual(metrics['drawdown'], 0.15, places=2)
        
        breached_limits = self.risk_manager.check_risk_limits()
        drawdown_breaches = [
            limit for limit in breached_limits
            if limit['risk_type'] == RiskType.DRAWDOWN
        ]
        
        self.assertTrue(len(drawdown_breaches) > 0)
        self.assertTrue(
            any(
                limit['risk_level'] in [RiskLevel.MEDIUM, RiskLevel.HIGH]
                for limit in drawdown_breaches
            )
        )

if __name__ == '__main__':
    unittest.main()
