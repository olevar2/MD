"""
Integration tests for paper trading simulation components.
"""
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trading_gateway_service.simulation.broker_simulator import (
    SimulatedBroker,
    OrderType,
    OrderStatus
)
from trading_gateway_service.simulation.market_simulator import (
    MarketDataSimulator,
    MarketRegime,
    MarketProfile
)

class TestPaperTradingSimulation(unittest.TestCase):
    """Test suite for paper trading simulation."""
    
    def setUp(self):
        """Set up test environment."""
        # Initialize broker simulator
        self.broker = SimulatedBroker(
            initial_balance=100000.0,
            base_currency="USD",
            max_leverage=50.0,
            min_lot_size=0.01
        )
        
        # Initialize market simulator
        self.market = MarketDataSimulator(
            symbols=["EUR/USD", "GBP/USD"]
        )
        
        # Generate some initial market data
        self.initial_market_data = self.market.generate_tick()
        for symbol, data in self.initial_market_data.items():
            self.broker.update_market_data(
                symbol=symbol,
                bid=data['bid'],
                ask=data['ask'],
                timestamp=data['timestamp'],
                volume=data['volume']
            )

    def test_market_order_execution(self):
        """Test market order execution flow."""
        # Place market order
        order_result = self.broker.place_order(
            symbol="EUR/USD",
            order_type=OrderType.MARKET,
            direction="buy",
            size=1.0,
            stop_loss=1.0900,
            take_profit=1.1100
        )
        
        self.assertIsNotNone(order_result['order_id'])
        self.assertEqual(order_result['status'], OrderStatus.FILLED)
        
        # Verify position
        account = self.broker.get_account_summary()
        self.assertEqual(account['positions'], 1)
        self.assertGreater(account['margin_used'], 0)
        
    def test_limit_order_execution(self):
        """Test limit order execution."""
        current_price = self.initial_market_data["EUR/USD"]["ask"]
        limit_price = current_price - 0.0010  # 10 pips below current price
        
        # Place limit order
        order_result = self.broker.place_order(
            symbol="EUR/USD",
            order_type=OrderType.LIMIT,
            direction="buy",
            size=1.0,
            price=limit_price
        )
        
        self.assertIsNotNone(order_result['order_id'])
        self.assertEqual(order_result['status'], OrderStatus.PENDING)
        
        # Generate market data that triggers the limit order
        while True:
            market_data = self.market.generate_tick()
            self.broker.update_market_data(
                symbol="EUR/USD",
                bid=market_data["EUR/USD"]["bid"],
                ask=market_data["EUR/USD"]["ask"],
                timestamp=market_data["EUR/USD"]["timestamp"],
                volume=market_data["EUR/USD"]["volume"]
            )
            
            if market_data["EUR/USD"]["ask"] <= limit_price:
                break
                
        # Verify order execution
        order = self.broker.orders[order_result['order_id']]
        self.assertEqual(order['status'], OrderStatus.FILLED)
        
    def test_stop_loss_trigger(self):
        """Test stop loss execution."""
        # Place market order with stop loss
        entry_price = self.initial_market_data["EUR/USD"]["ask"]
        stop_loss = entry_price - 0.0020  # 20 pips below entry
        
        order_result = self.broker.place_order(
            symbol="EUR/USD",
            order_type=OrderType.MARKET,
            direction="buy",
            size=1.0,
            stop_loss=stop_loss
        )
        
        self.assertEqual(order_result['status'], OrderStatus.FILLED)
        
        # Generate market data that triggers stop loss
        while True:
            market_data = self.market.generate_tick()
            self.broker.update_market_data(
                symbol="EUR/USD",
                bid=market_data["EUR/USD"]["bid"],
                ask=market_data["EUR/USD"]["ask"],
                timestamp=market_data["EUR/USD"]["timestamp"],
                volume=market_data["EUR/USD"]["volume"]
            )
            
            if market_data["EUR/USD"]["bid"] <= stop_loss:
                break
                
        # Verify position closed
        account = self.broker.get_account_summary()
        self.assertEqual(account['positions'], 0)
        
    def test_market_regime_impact(self):
        """Test market regime impact on execution."""
        # Record normal regime spreads
        normal_spreads = {}
        for _ in range(10):
            market_data = self.market.generate_tick()
            for symbol, data in market_data.items():
                normal_spreads.setdefault(symbol, []).append(data['spread'])
                
        # Force high volatility regime
        self.market.current_regime = MarketRegime.HIGH_VOLATILITY
        
        # Record high volatility spreads
        high_vol_spreads = {}
        for _ in range(10):
            market_data = self.market.generate_tick()
            for symbol, data in market_data.items():
                high_vol_spreads.setdefault(symbol, []).append(data['spread'])
                
        # Verify spread widening
        for symbol in self.market.symbols:
            avg_normal = np.mean(normal_spreads[symbol])
            avg_high_vol = np.mean(high_vol_spreads[symbol])
            self.assertGreater(avg_high_vol, avg_normal)
            
    def test_session_based_spreads(self):
        """Test spread variation across sessions."""
        spreads = {
            "sydney": [],
            "tokyo": [],
            "london": [],
            "new_york": []
        }
        
        # Collect spreads across sessions
        for session in spreads.keys():
            # Mock session time
            session_times = {
                "sydney": "00:00",
                "tokyo": "03:00",
                "london": "10:00",
                "new_york": "15:00"
            }
            
            test_time = datetime.strptime(
                f"2025-01-01 {session_times[session]}",
                "%Y-%m-%d %H:%M"
            )
            
            # Generate ticks during session
            for _ in range(5):
                market_data = self.market.generate_tick(test_time)
                spreads[session].extend(
                    [data['spread'] for data in market_data.values()]
                )
                
        # Verify London/NY have tighter spreads than Sydney/Tokyo
        avg_spreads = {
            session: np.mean(spread_list)
            for session, spread_list in spreads.items()
        }
        
        self.assertLess(
            avg_spreads['london'],
            avg_spreads['sydney']
        )
        self.assertLess(
            avg_spreads['new_york'],
            avg_spreads['sydney']
        )
        
    def test_margin_requirements(self):
        """Test margin calculation and limits."""
        # Get initial margin available
        initial_account = self.broker.get_account_summary()
        initial_margin = initial_account['equity']
        
        # Try to place order using maximum leverage
        max_position_size = (
            initial_margin *
            self.broker.max_leverage /
            self.initial_market_data["EUR/USD"]["ask"]
        )
        
        order_result = self.broker.place_order(
            symbol="EUR/USD",
            order_type=OrderType.MARKET,
            direction="buy",
            size=max_position_size
        )
        
        self.assertEqual(order_result['status'], OrderStatus.FILLED)
        
        # Verify margin usage
        account = self.broker.get_account_summary()
        self.assertGreater(account['margin_used'], 0)
        self.assertLess(account['free_margin'], initial_margin)
        
        # Try to place another order - should fail due to margin
        order_result = self.broker.place_order(
            symbol="GBP/USD",
            order_type=OrderType.MARKET,
            direction="buy",
            size=max_position_size
        )
        
        self.assertEqual(order_result['status'], OrderStatus.REJECTED)
        
    def test_realistic_slippage(self):
        """Test realistic slippage simulation."""
        slippage_samples = []
        
        # Place multiple market orders and record slippage
        for _ in range(10):
            # Generate new market data
            market_data = self.market.generate_tick()
            self.broker.update_market_data(
                symbol="EUR/USD",
                bid=market_data["EUR/USD"]["bid"],
                ask=market_data["EUR/USD"]["ask"],
                timestamp=market_data["EUR/USD"]["timestamp"],
                volume=market_data["EUR/USD"]["volume"]
            )
            
            # Place order
            order_result = self.broker.place_order(
                symbol="EUR/USD",
                order_type=OrderType.MARKET,
                direction="buy",
                size=1.0
            )
            
            # Calculate slippage
            order = self.broker.orders[order_result['order_id']]
            expected_price = market_data["EUR/USD"]["ask"]
            actual_price = order['average_price']
            slippage = actual_price - expected_price
            slippage_samples.append(slippage)
            
            # Close position
            self.broker.close_position("EUR/USD", market_data["EUR/USD"]["bid"])
            
        # Verify slippage characteristics
        avg_slippage = np.mean(slippage_samples)
        self.assertGreater(avg_slippage, 0)  # Should have some slippage
        self.assertLess(avg_slippage, 0.0002)  # But not too much (2 pips)

if __name__ == '__main__':
    unittest.main()
