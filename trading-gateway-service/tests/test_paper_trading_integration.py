"""
Integration tests for the paper trading system.

Tests the complete integration between paper trading simulation
and risk management components.
"""
import unittest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trading_gateway_service.simulation.paper_trading_system import (
    PaperTradingSystem,
    TradingState
)
from trading_gateway_service.simulation.broker_simulator import OrderType
from risk_management_service.risk_manager import RiskLevel, RiskType

class TestPaperTradingIntegration(unittest.TestCase):
    """Test suite for paper trading integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.trading_system = PaperTradingSystem(
            initial_balance=100000.0,
            risk_params={
                'max_position_size': 10000.0,
                'max_leverage': 20.0,
                'max_drawdown': 0.20,
                'risk_per_trade': 0.02
            }
        )
        
        # Create event loop for async tests
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def tearDown(self):
        """Clean up after tests."""
        self.loop.close()
        
    def test_system_startup(self):
        """Test system startup and initialization."""
        async def startup_test():
            # Start the system
            await self.trading_system.start()
            
            # Check initial state
            self.assertEqual(self.trading_system.state, TradingState.ACTIVE)
            
            # Check component initialization
            self.assertIsNotNone(self.trading_system.broker)
            self.assertIsNotNone(self.trading_system.risk_manager)
            self.assertIsNotNone(self.trading_system.circuit_breaker)
            
            await self.trading_system.stop()
            
        self.loop.run_until_complete(startup_test())
        
    def test_order_submission_with_risk_checks(self):
        """Test order submission with integrated risk checks."""
        async def order_test():
            await self.trading_system.start()
            
            # Wait for market data initialization
            await asyncio.sleep(2)
            
            # Submit valid order
            result = await self.trading_system.submit_order(
                symbol="EUR/USD",
                order_type=OrderType.MARKET,
                direction="buy",
                size=1000.0,
                stop_loss=1.0800
            )
            
            self.assertTrue(result['success'])
            self.assertIsNotNone(result['order_id'])
            
            # Submit order exceeding risk limits
            result = await self.trading_system.submit_order(
                symbol="EUR/USD",
                order_type=OrderType.MARKET,
                direction="buy",
                size=50000.0  # Exceeds max position size
            )
            
            self.assertFalse(result['success'])
            self.assertIn('Risk check failed', result['error'])
            
            await self.trading_system.stop()
            
        self.loop.run_until_complete(order_test())
        
    def test_circuit_breaker_integration(self):
        """Test circuit breaker activation and trading pause."""
        async def circuit_breaker_test():
            await self.trading_system.start()
            
            # Create drawdown situation
            # First, take a large position
            await self.trading_system.submit_order(
                symbol="EUR/USD",
                order_type=OrderType.MARKET,
                direction="buy",
                size=5000.0
            )
            
            # Simulate market movement causing drawdown
            original_generate_tick = self.trading_system.market_simulator.generate_tick
            
            def modified_generate_tick():
                data = original_generate_tick()
                # Simulate 25% price drop
                for symbol in data:
                    data[symbol]['bid'] *= 0.75
                    data[symbol]['ask'] *= 0.75
                return data
                
            self.trading_system.market_simulator.generate_tick = modified_generate_tick
            
            # Wait for circuit breaker to trigger
            await asyncio.sleep(5)
            
            # Verify system state
            self.assertEqual(self.trading_system.state, TradingState.PAUSED)
            
            # Try to submit order during pause
            result = await self.trading_system.submit_order(
                symbol="EUR/USD",
                order_type=OrderType.MARKET,
                direction="buy",
                size=1000.0
            )
            
            self.assertFalse(result['success'])
            self.assertIn('System is in paused state', result['error'])
            
            await self.trading_system.stop()
            
        self.loop.run_until_complete(circuit_breaker_test())
        
    def test_risk_metrics_update(self):
        """Test risk metrics calculation and updates."""
        async def risk_metrics_test():
            await self.trading_system.start()
            
            # Take a position
            await self.trading_system.submit_order(
                symbol="EUR/USD",
                order_type=OrderType.MARKET,
                direction="buy",
                size=2000.0
            )
            
            # Wait for metrics updates
            await asyncio.sleep(3)
            
            # Check risk metrics
            status = self.trading_system.get_system_status()
            self.assertIsNotNone(status['risk_metrics'])
            self.assertIn('drawdown', status['risk_metrics'])
            self.assertIn('equity', status['risk_metrics'])
            
            await self.trading_system.stop()
            
        self.loop.run_until_complete(risk_metrics_test())
        
    def test_stress_testing_integration(self):
        """Test stress testing with portfolio positions."""
        async def stress_test():
            await self.trading_system.start()
            
            # Take positions in multiple pairs
            pairs = ["EUR/USD", "GBP/USD"]
            for pair in pairs:
                await self.trading_system.submit_order(
                    symbol=pair,
                    order_type=OrderType.MARKET,
                    direction="buy",
                    size=1000.0
                )
                
            # Wait for position establishment
            await asyncio.sleep(2)
            
            # Get current prices for stress test
            current_prices = self.trading_system._get_current_prices()
            
            # Run stress test
            stress_engine = self.trading_system.risk_manager.stress_testing_engine
            results = stress_engine.run_all_stress_tests(current_prices)
            
            self.assertIsNotNone(results)
            self.assertTrue(len(results) > 0)
            
            # Verify stress test results contain key metrics
            for scenario in results.values():
                self.assertIn('value_at_risk', scenario)
                self.assertIn('max_drawdown', scenario)
                
            await self.trading_system.stop()
            
        self.loop.run_until_complete(stress_test())
        
    def test_system_health_monitoring(self):
        """Test system health monitoring and alerts."""
        async def health_monitoring_test():
            await self.trading_system.start()
            
            # Initial health check
            self.assertEqual(len(self.trading_system.stats), 4)
            
            # Submit some orders to generate statistics
            for _ in range(3):
                await self.trading_system.submit_order(
                    symbol="EUR/USD",
                    order_type=OrderType.MARKET,
                    direction="buy",
                    size=1000.0
                )
                
            # Wait for health check cycle
            await asyncio.sleep(6)
            
            # Verify stats updated
            self.assertGreater(self.trading_system.stats['orders_submitted'], 0)
            self.assertGreater(self.trading_system.stats['risk_checks_performed'], 0)
            
            await self.trading_system.stop()
            
        self.loop.run_until_complete(health_monitoring_test())

if __name__ == '__main__':
    unittest.main()
