"""
Performance tests for Trading Gateway Service order execution.
"""

import pytest
import asyncio
import time
import random
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock, AsyncMock

from trading_gateway_service.order_management.order_manager import OrderManager
from trading_gateway_service.broker_adapters.adapter_factory import BrokerAdapterFactory
from trading_gateway_service.database import Database
from trading_gateway_service.service_clients import ServiceClients


class TestTradingGatewayOrderExecutionPerformance:
    """Performance tests for Trading Gateway Service order execution."""
    
    @pytest.fixture
    def mock_order_manager(self, performance_metrics, mock_database, mock_service_client):
        """Create a mock order manager."""
        # Create broker adapter factory
        broker_adapter_factory = BrokerAdapterFactory()
        
        # Create mock broker adapter
        mock_broker_adapter = MagicMock()
        mock_broker_adapter.submit_order = AsyncMock()
        mock_broker_adapter.submit_order.side_effect = lambda order: self._simulate_order_execution(order, performance_metrics)
        mock_broker_adapter.cancel_order = AsyncMock()
        mock_broker_adapter.cancel_order.side_effect = lambda order_id: self._simulate_order_cancellation(order_id, performance_metrics)
        mock_broker_adapter.get_order_status = AsyncMock()
        mock_broker_adapter.get_order_status.side_effect = lambda order_id: self._simulate_get_order_status(order_id, performance_metrics)
        
        # Mock the broker adapter factory
        broker_adapter_factory.create_adapter = MagicMock(return_value=mock_broker_adapter)
        
        # Create order manager
        order_manager = OrderManager(
            database=mock_database,
            broker_adapter_factory=broker_adapter_factory,
            service_clients=ServiceClients()
        )
        
        return order_manager
    
    async def _simulate_order_execution(self, order, performance_metrics):
        """Simulate order execution."""
        start_time = time.time()
        await asyncio.sleep(0.05)  # Simulate order execution time
        duration = time.time() - start_time
        performance_metrics.record("order_execution", duration)
        
        return {
            "order_id": str(uuid.uuid4()),
            "status": "submitted",
            "symbol": order["symbol"],
            "side": order["side"],
            "type": order["type"],
            "quantity": order["quantity"],
            "price": order.get("price"),
            "timestamp": time.time()
        }
    
    async def _simulate_order_cancellation(self, order_id, performance_metrics):
        """Simulate order cancellation."""
        start_time = time.time()
        await asyncio.sleep(0.03)  # Simulate order cancellation time
        duration = time.time() - start_time
        performance_metrics.record("order_cancellation", duration)
        
        return {
            "order_id": order_id,
            "status": "cancelled",
            "timestamp": time.time()
        }
    
    async def _simulate_get_order_status(self, order_id, performance_metrics):
        """Simulate get order status."""
        start_time = time.time()
        await asyncio.sleep(0.02)  # Simulate get order status time
        duration = time.time() - start_time
        performance_metrics.record("get_order_status", duration)
        
        return {
            "order_id": order_id,
            "status": random.choice(["submitted", "filled", "partially_filled", "cancelled", "rejected"]),
            "filled_quantity": random.randint(0, 10000),
            "average_price": random.uniform(1.0, 1.1),
            "timestamp": time.time()
        }
    
    @pytest.mark.asyncio
    async def test_market_order_performance(self, performance_metrics, time_it, mock_order_manager):
        """Test market order performance."""
        # Test performance
        @time_it("market_order_submission")
        async def submit_market_order():
            return await mock_order_manager.submit_order({
                "symbol": "EURUSD",
                "side": "buy",
                "type": "market",
                "quantity": 10000
            })
        
        # Run multiple times to get better statistics
        for _ in range(10):
            result = await submit_market_order()
        
        # Get statistics
        stats = performance_metrics.get_stats("market_order_submission")
        
        # Print statistics
        print(f"Market Order Submission Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify result
        assert "order_id" in result
        assert result["status"] == "submitted"
        assert result["symbol"] == "EURUSD"
        assert result["side"] == "buy"
        assert result["type"] == "market"
        assert result["quantity"] == 10000
        
        # Verify performance
        assert stats["mean"] < 0.1, "Market order submission is too slow"
    
    @pytest.mark.asyncio
    async def test_limit_order_performance(self, performance_metrics, time_it, mock_order_manager):
        """Test limit order performance."""
        # Test performance
        @time_it("limit_order_submission")
        async def submit_limit_order():
            return await mock_order_manager.submit_order({
                "symbol": "EURUSD",
                "side": "buy",
                "type": "limit",
                "quantity": 10000,
                "price": 1.1
            })
        
        # Run multiple times to get better statistics
        for _ in range(10):
            result = await submit_limit_order()
        
        # Get statistics
        stats = performance_metrics.get_stats("limit_order_submission")
        
        # Print statistics
        print(f"Limit Order Submission Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify result
        assert "order_id" in result
        assert result["status"] == "submitted"
        assert result["symbol"] == "EURUSD"
        assert result["side"] == "buy"
        assert result["type"] == "limit"
        assert result["quantity"] == 10000
        assert result["price"] == 1.1
        
        # Verify performance
        assert stats["mean"] < 0.1, "Limit order submission is too slow"
    
    @pytest.mark.asyncio
    async def test_order_cancellation_performance(self, performance_metrics, time_it, mock_order_manager):
        """Test order cancellation performance."""
        # Submit an order first
        order_result = await mock_order_manager.submit_order({
            "symbol": "EURUSD",
            "side": "buy",
            "type": "limit",
            "quantity": 10000,
            "price": 1.1
        })
        
        order_id = order_result["order_id"]
        
        # Test performance
        @time_it("order_cancellation_request")
        async def cancel_order():
            return await mock_order_manager.cancel_order(order_id)
        
        # Run multiple times to get better statistics
        for _ in range(10):
            result = await cancel_order()
        
        # Get statistics
        stats = performance_metrics.get_stats("order_cancellation_request")
        
        # Print statistics
        print(f"Order Cancellation Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify result
        assert result["order_id"] == order_id
        assert result["status"] == "cancelled"
        
        # Verify performance
        assert stats["mean"] < 0.1, "Order cancellation is too slow"
    
    @pytest.mark.asyncio
    async def test_get_order_status_performance(self, performance_metrics, time_it, mock_order_manager):
        """Test get order status performance."""
        # Submit an order first
        order_result = await mock_order_manager.submit_order({
            "symbol": "EURUSD",
            "side": "buy",
            "type": "market",
            "quantity": 10000
        })
        
        order_id = order_result["order_id"]
        
        # Test performance
        @time_it("get_order_status_request")
        async def get_order_status():
            return await mock_order_manager.get_order_status(order_id)
        
        # Run multiple times to get better statistics
        for _ in range(10):
            result = await get_order_status()
        
        # Get statistics
        stats = performance_metrics.get_stats("get_order_status_request")
        
        # Print statistics
        print(f"Get Order Status Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify result
        assert result["order_id"] == order_id
        
        # Verify performance
        assert stats["mean"] < 0.1, "Get order status is too slow"
    
    @pytest.mark.asyncio
    async def test_multiple_orders_performance(self, performance_metrics, time_it, mock_order_manager):
        """Test multiple orders performance."""
        # Test performance
        @time_it("multiple_orders_submission")
        async def submit_multiple_orders():
            # Submit orders in parallel
            tasks = [
                mock_order_manager.submit_order({
                    "symbol": "EURUSD",
                    "side": "buy",
                    "type": "market",
                    "quantity": 10000
                }),
                mock_order_manager.submit_order({
                    "symbol": "GBPUSD",
                    "side": "sell",
                    "type": "market",
                    "quantity": 10000
                }),
                mock_order_manager.submit_order({
                    "symbol": "USDJPY",
                    "side": "buy",
                    "type": "limit",
                    "quantity": 10000,
                    "price": 110.0
                })
            ]
            
            return await asyncio.gather(*tasks)
        
        # Run multiple times to get better statistics
        for _ in range(10):
            results = await submit_multiple_orders()
        
        # Get statistics
        stats = performance_metrics.get_stats("multiple_orders_submission")
        
        # Print statistics
        print(f"Multiple Orders Submission Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify results
        assert len(results) == 3
        for result in results:
            assert "order_id" in result
            assert result["status"] == "submitted"
        
        # Verify performance
        assert stats["mean"] < 0.2, "Multiple orders submission is too slow"
