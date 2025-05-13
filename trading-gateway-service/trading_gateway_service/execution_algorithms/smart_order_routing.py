"""
Smart Order Routing (SOR) Algorithm.

This module implements a Smart Order Routing algorithm that intelligently
splits and routes orders across available brokers based on price, speed,
and liquidity.
"""
import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from ..interfaces.broker_adapter_interface import BrokerAdapterInterface, OrderRequest, ExecutionReport, OrderStatus
from .base_algorithm import BaseExecutionAlgorithm, ExecutionResult
from trading_gateway_service.error.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class BrokerScore:
    """
    Represents a score for a broker based on various factors.
    
    Used by the SOR algorithm to rank brokers for order routing.
    """

    def __init__(self, broker_name: str, price_score: float=0.0,
        speed_score: float=0.0, liquidity_score: float=0.0,
        reliability_score: float=0.0):
        """
        Initialize a broker score.
        
        Args:
            broker_name: Name of the broker
            price_score: Score for price competitiveness (0-1)
            speed_score: Score for execution speed (0-1)
            liquidity_score: Score for available liquidity (0-1)
            reliability_score: Score for broker reliability (0-1)
        """
        self.broker_name = broker_name
        self.price_score = price_score
        self.speed_score = speed_score
        self.liquidity_score = liquidity_score
        self.reliability_score = reliability_score

    @property
    def total_score(self) ->float:
        """Calculate the total score for the broker."""
        return (self.price_score + self.speed_score + self.liquidity_score +
            self.reliability_score) / 4.0

    def __lt__(self, other):
        """Compare broker scores for sorting."""
        return self.total_score < other.total_score


class SmartOrderRoutingAlgorithm(BaseExecutionAlgorithm):
    """
    Smart Order Routing (SOR) algorithm.
    
    This algorithm intelligently splits and routes orders across available
    brokers based on price, speed, and liquidity to achieve the best overall
    execution quality.
    """

    def __init__(self, broker_adapters: Dict[str, BrokerAdapterInterface],
        logger: Optional[logging.Logger]=None, config: Optional[Dict[str,
        Any]]=None):
        """
        Initialize the SOR algorithm.
        
        Args:
            broker_adapters: Dictionary of broker adapters by name
            logger: Logger instance
            config: Algorithm-specific configuration
        """
        super().__init__(broker_adapters, logger, config)
        self.price_weight = self.config_manager.get('price_weight', 0.4)
        self.speed_weight = self.config_manager.get('speed_weight', 0.3)
        self.liquidity_weight = self.config_manager.get('liquidity_weight', 0.2)
        self.reliability_weight = self.config_manager.get('reliability_weight', 0.1)
        self.min_brokers = self.config_manager.get('min_brokers', 1)
        self.max_brokers = self.config_manager.get('max_brokers', len(broker_adapters))
        self.min_order_size = self.config_manager.get('min_order_size', 1000)
        self.original_order = None
        self.child_orders = {}
        self.execution_reports = {}
        self.is_executing = False
        self.is_cancelled = False

    @async_with_exception_handling
    async def execute(self, order: OrderRequest) ->ExecutionResult:
        """
        Execute the SOR algorithm for the given order.
        
        Args:
            order: The order to execute
            
        Returns:
            ExecutionResult with details of the execution
        """
        if self.is_executing:
            self.logger.warning('SOR algorithm is already executing an order')
            return ExecutionResult(algorithm_id=self.algorithm_id,
                original_order_id=order.client_order_id, status='FAILED',
                metrics={'error': 'Algorithm already executing an order'})
        self.is_executing = True
        self.original_order = order
        self.child_orders = {}
        self.execution_reports = {}
        try:
            self._trigger_callbacks('started', {'algorithm_id': self.
                algorithm_id, 'order_id': order.client_order_id,
                'timestamp': datetime.utcnow().isoformat()})
            broker_scores = await self._score_brokers(order)
            if not broker_scores:
                self.logger.error(
                    'No suitable brokers found for order execution')
                return ExecutionResult(algorithm_id=self.algorithm_id,
                    original_order_id=order.client_order_id, status=
                    'FAILED', metrics={'error': 'No suitable brokers found'})
            allocations = self._calculate_allocations(order, broker_scores)
            if not allocations:
                self.logger.error('Failed to calculate order allocations')
                return ExecutionResult(algorithm_id=self.algorithm_id,
                    original_order_id=order.client_order_id, status=
                    'FAILED', metrics={'error':
                    'Failed to calculate allocations'})
            tasks = []
            for broker_name, quantity in allocations.items():
                tasks.append(self._place_child_order(order, broker_name,
                    quantity))
            child_results = await asyncio.gather(*tasks, return_exceptions=True
                )
            success_count = 0
            for result in child_results:
                if isinstance(result, Exception):
                    self.logger.error(
                        f'Error placing child order: {str(result)}')
                elif isinstance(result, ExecutionReport):
                    self.execution_reports[result.client_order_id] = result
                    if result.status != OrderStatus.REJECTED:
                        success_count += 1
            if success_count == 0:
                status = 'FAILED'
            elif success_count < len(allocations):
                status = 'PARTIAL'
            else:
                status = 'COMPLETED'
            result = ExecutionResult(algorithm_id=self.algorithm_id,
                original_order_id=order.client_order_id, status=status,
                execution_reports=list(self.execution_reports.values()),
                metrics={'broker_allocations': allocations, 'broker_scores':
                {b.broker_name: b.total_score for b in broker_scores},
                'success_rate': success_count / len(allocations) if
                allocations else 0})
            self._trigger_callbacks('completed', result.to_dict())
            return result
        except Exception as e:
            self.logger.error(f'Error executing SOR algorithm: {str(e)}')
            self._trigger_callbacks('failed', {'algorithm_id': self.
                algorithm_id, 'order_id': order.client_order_id, 'error':
                str(e), 'timestamp': datetime.utcnow().isoformat()})
            return ExecutionResult(algorithm_id=self.algorithm_id,
                original_order_id=order.client_order_id, status='FAILED',
                metrics={'error': str(e)})
        finally:
            self.is_executing = False

    async def cancel(self) ->bool:
        """
        Cancel the current execution.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        if not self.is_executing:
            return True
        self.is_cancelled = True
        cancel_tasks = []
        for order_id, order_details in self.child_orders.items():
            broker_name = order_details.get('broker_name')
            if broker_name and broker_name in self.broker_adapters:
                adapter = self.broker_adapters[broker_name]
                cancel_tasks.append(adapter.cancel_order(order_id))
        if cancel_tasks:
            results = await asyncio.gather(*cancel_tasks, return_exceptions
                =True)
            success_count = sum(1 for r in results if r is True)
            self._trigger_callbacks('cancelled', {'algorithm_id': self.
                algorithm_id, 'order_id': self.original_order.
                client_order_id if self.original_order else None,
                'cancelled_orders': success_count, 'total_orders': len(
                cancel_tasks), 'timestamp': datetime.utcnow().isoformat()})
            return success_count == len(cancel_tasks)
        return True

    @with_broker_api_resilience('get_status')
    async def get_status(self) ->Dict[str, Any]:
        """
        Get the current status of the execution.
        
        Returns:
            Dictionary with the current status
        """
        total_quantity = (self.original_order.quantity if self.
            original_order else 0)
        filled_quantity = sum(report.filled_quantity for report in self.
            execution_reports.values() if hasattr(report, 'filled_quantity'))
        return {'algorithm_id': self.algorithm_id, 'is_executing': self.
            is_executing, 'is_cancelled': self.is_cancelled,
            'original_order_id': self.original_order.client_order_id if
            self.original_order else None, 'total_quantity': total_quantity,
            'filled_quantity': filled_quantity, 'completion_percentage': 
            filled_quantity / total_quantity * 100 if total_quantity > 0 else
            0, 'child_orders': len(self.child_orders), 'execution_reports':
            len(self.execution_reports)}

    async def _score_brokers(self, order: OrderRequest) ->List[BrokerScore]:
        """
        Score available brokers based on various factors.
        
        Args:
            order: The order to be executed
            
        Returns:
            List of BrokerScore objects sorted by total score (highest first)
        """
        scores = []
        quote_tasks = []
        for broker_name, adapter in self.broker_adapters.items():
            if hasattr(adapter, 'get_quote') and callable(adapter.get_quote):
                quote_tasks.append(self._get_broker_quote(broker_name,
                    adapter, order))
            else:
                self.logger.warning(
                    f'Broker {broker_name} does not support quotes')
        quotes = await asyncio.gather(*quote_tasks, return_exceptions=True)
        best_price = None
        best_liquidity = 0
        for quote in quotes:
            if isinstance(quote, Exception) or not isinstance(quote, dict):
                continue
            broker_name = quote.get('broker_name')
            price = quote.get('price')
            liquidity = quote.get('liquidity', 0)
            if price is not None:
                if best_price is None or (order.direction.value.upper() ==
                    'BUY' and price < best_price or order.direction.value.
                    upper() == 'SELL' and price > best_price):
                    best_price = price
            best_liquidity = max(best_liquidity, liquidity)
        for quote in quotes:
            if isinstance(quote, Exception) or not isinstance(quote, dict):
                continue
            broker_name = quote.get('broker_name')
            price = quote.get('price')
            liquidity = quote.get('liquidity', 0)
            speed = quote.get('response_time', 1000)
            reliability = quote.get('reliability', 0.5)
            if price is None:
                continue
            if best_price is not None and price is not None:
                if order.direction.value.upper() == 'BUY':
                    price_score = best_price / price if price > 0 else 0
                else:
                    price_score = price / best_price if best_price > 0 else 0
            else:
                price_score = 0.5
            speed_score = 1.0 - min(1.0, speed / 1000)
            liquidity_score = (liquidity / best_liquidity if best_liquidity >
                0 else 0)
            score = BrokerScore(broker_name=broker_name, price_score=
                price_score * self.price_weight, speed_score=speed_score *
                self.speed_weight, liquidity_score=liquidity_score * self.
                liquidity_weight, reliability_score=reliability * self.
                reliability_weight)
            scores.append(score)
        return sorted(scores, reverse=True)

    @async_with_exception_handling
    async def _get_broker_quote(self, broker_name: str, adapter:
        BrokerAdapterInterface, order: OrderRequest) ->Dict[str, Any]:
        """
        Get a quote from a broker.
        
        Args:
            broker_name: Name of the broker
            adapter: Broker adapter instance
            order: The order to get a quote for
            
        Returns:
            Dictionary with quote information
        """
        start_time = time.time()
        try:
            if hasattr(adapter, 'get_quote') and callable(adapter.get_quote):
                quote = await adapter.get_quote(order.instrument)
                response_time = (time.time() - start_time) * 1000
                if order.direction.value.upper() == 'BUY':
                    price = quote.get('ask')
                else:
                    price = quote.get('bid')
                liquidity = quote.get('liquidity', {}).get('ask' if order.
                    direction.value.upper() == 'BUY' else 'bid', 0)
                return {'broker_name': broker_name, 'price': price,
                    'liquidity': liquidity, 'response_time': response_time,
                    'reliability': 0.9, 'spread': quote.get('spread'),
                    'timestamp': quote.get('timestamp')}
            elif hasattr(adapter, 'get_price') and callable(adapter.get_price):
                price = await adapter.get_price(order.instrument)
                response_time = (time.time() - start_time) * 1000
                return {'broker_name': broker_name, 'price': price,
                    'liquidity': 0, 'response_time': response_time,
                    'reliability': 0.7, 'timestamp': datetime.utcnow().
                    isoformat()}
            else:
                return {'broker_name': broker_name, 'error':
                    'Broker does not support quotes or prices'}
        except Exception as e:
            self.logger.error(
                f'Error getting quote from broker {broker_name}: {str(e)}')
            return {'broker_name': broker_name, 'error': str(e)}

    def _calculate_allocations(self, order: OrderRequest, broker_scores:
        List[BrokerScore]) ->Dict[str, float]:
        """
        Calculate order allocations across brokers.
        
        Args:
            order: The order to allocate
            broker_scores: List of broker scores
            
        Returns:
            Dictionary mapping broker names to allocated quantities
        """
        total_quantity = order.quantity
        allocations = {}
        num_brokers = min(self.max_brokers, max(self.min_brokers, len(
            broker_scores)))
        top_brokers = broker_scores[:num_brokers]
        if not top_brokers:
            return {}
        total_score = sum(broker.total_score for broker in top_brokers)
        if total_score <= 0:
            base_quantity = total_quantity / len(top_brokers)
            for broker in top_brokers:
                allocations[broker.broker_name] = base_quantity
        else:
            remaining_quantity = total_quantity
            for i, broker in enumerate(top_brokers):
                if i == len(top_brokers) - 1:
                    allocation = remaining_quantity
                else:
                    allocation = total_quantity * (broker.total_score /
                        total_score)
                    allocation = round(allocation, 2)
                    allocation = max(self.min_order_size, allocation)
                    allocation = min(allocation, remaining_quantity)
                allocations[broker.broker_name] = allocation
                remaining_quantity -= allocation
                if remaining_quantity <= 0:
                    break
        return allocations

    @async_with_exception_handling
    async def _place_child_order(self, parent_order: OrderRequest,
        broker_name: str, quantity: float) ->ExecutionReport:
        """
        Place a child order with a specific broker.
        
        Args:
            parent_order: The original parent order
            broker_name: Name of the broker to use
            quantity: Quantity to allocate to this broker
            
        Returns:
            Execution report for the child order
        """
        if broker_name not in self.broker_adapters:
            raise ValueError(f'Unknown broker: {broker_name}')
        adapter = self.broker_adapters[broker_name]
        child_order_id = (
            f'{parent_order.client_order_id}_SOR_{broker_name}_{uuid.uuid4().hex[:8]}'
            )
        child_order = OrderRequest(instrument=parent_order.instrument,
            order_type=parent_order.order_type, direction=parent_order.
            direction, quantity=quantity, price=parent_order.price,
            stop_loss=parent_order.stop_loss, take_profit=parent_order.
            take_profit, client_order_id=child_order_id)
        self.child_orders[child_order_id] = {'broker_name': broker_name,
            'quantity': quantity, 'parent_order_id': parent_order.
            client_order_id, 'timestamp': datetime.utcnow().isoformat()}
        self.logger.info(
            f'Placing child order {child_order_id} with broker {broker_name}: {child_order.instrument} {child_order.direction.value} {child_order.quantity}'
            )
        self._trigger_callbacks('progress', {'algorithm_id': self.
            algorithm_id, 'order_id': parent_order.client_order_id,
            'child_order_id': child_order_id, 'broker': broker_name,
            'quantity': quantity, 'timestamp': datetime.utcnow().isoformat()})
        try:
            execution_report = await adapter.place_order(child_order)
            return execution_report
        except Exception as e:
            self.logger.error(
                f'Error placing child order with broker {broker_name}: {str(e)}'
                )
            raise
