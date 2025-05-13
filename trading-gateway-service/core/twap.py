"""
Time-Weighted Average Price (TWAP) Algorithm.

This module implements a TWAP algorithm that splits an order into smaller
chunks and executes them evenly over a specified time period to achieve
a price close to the time-weighted average price.
"""
import asyncio
import logging
import math
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from ..interfaces.broker_adapter_interface import BrokerAdapterInterface, OrderRequest, ExecutionReport, OrderStatus
from .base_algorithm import BaseExecutionAlgorithm, ExecutionResult
from core.exceptions_bridge_1 import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class TWAPAlgorithm(BaseExecutionAlgorithm):
    """
    Time-Weighted Average Price (TWAP) algorithm.
    
    This algorithm splits an order into smaller chunks and executes them
    evenly over a specified time period to achieve a price close to the
    time-weighted average price.
    """

    def __init__(self, broker_adapters: Dict[str, BrokerAdapterInterface],
        logger: Optional[logging.Logger]=None, config: Optional[Dict[str,
        Any]]=None):
        """
        Initialize the TWAP algorithm.
        
        Args:
            broker_adapters: Dictionary of broker adapters by name
            logger: Logger instance
            config: Algorithm-specific configuration
        """
        super().__init__(broker_adapters, logger, config)
        self.default_broker = self.config_manager.get('default_broker')
        self.duration_minutes = self.config_manager.get('duration_minutes', 60)
        self.num_slices = self.config_manager.get('num_slices', 12)
        self.min_slice_size = self.config_manager.get('min_slice_size', 1000)
        self.random_variance = self.config_manager.get('random_variance', 0.1)
        self.original_order = None
        self.execution_task = None
        self.child_orders = {}
        self.execution_reports = {}
        self.is_executing = False
        self.is_cancelled = False
        self.start_time = None
        self.end_time = None
        self.next_slice_time = None
        self.slices_executed = 0
        self.slices_total = 0

    @async_with_exception_handling
    async def execute(self, order: OrderRequest) ->ExecutionResult:
        """
        Execute the TWAP algorithm for the given order.
        
        Args:
            order: The order to execute
            
        Returns:
            ExecutionResult with details of the execution
        """
        if self.is_executing:
            self.logger.warning('TWAP algorithm is already executing an order')
            return ExecutionResult(algorithm_id=self.algorithm_id,
                original_order_id=order.client_order_id, status='FAILED',
                metrics={'error': 'Algorithm already executing an order'})
        self.is_executing = True
        self.original_order = order
        self.child_orders = {}
        self.execution_reports = {}
        self.is_cancelled = False
        self.start_time = datetime.utcnow()
        self.end_time = self.start_time + timedelta(minutes=self.
            duration_minutes)
        self.next_slice_time = self.start_time
        self.slices_total = self._calculate_num_slices(order)
        slice_interval = timedelta(minutes=self.duration_minutes / self.
            slices_total)
        self._trigger_callbacks('started', {'algorithm_id': self.
            algorithm_id, 'order_id': order.client_order_id, 'start_time':
            self.start_time.isoformat(), 'end_time': self.end_time.
            isoformat(), 'slices': self.slices_total,
            'slice_interval_seconds': slice_interval.total_seconds()})
        self.execution_task = asyncio.create_task(self._execute_slices(
            order, slice_interval))
        try:
            result = await self.execution_task
            return result
        except asyncio.CancelledError:
            self.logger.info('TWAP execution task was cancelled')
            return ExecutionResult(algorithm_id=self.algorithm_id,
                original_order_id=order.client_order_id, status='CANCELLED',
                execution_reports=list(self.execution_reports.values()),
                metrics={'slices_executed': self.slices_executed,
                'slices_total': self.slices_total, 'duration_minutes': self
                .duration_minutes, 'execution_time_minutes': (datetime.
                utcnow() - self.start_time).total_seconds() / 60})
        except Exception as e:
            self.logger.error(f'Error executing TWAP algorithm: {str(e)}')
            self._trigger_callbacks('failed', {'algorithm_id': self.
                algorithm_id, 'order_id': order.client_order_id, 'error':
                str(e), 'timestamp': datetime.utcnow().isoformat()})
            return ExecutionResult(algorithm_id=self.algorithm_id,
                original_order_id=order.client_order_id, status='FAILED',
                execution_reports=list(self.execution_reports.values()),
                metrics={'error': str(e)})
        finally:
            self.is_executing = False

    @async_with_exception_handling
    async def cancel(self) ->bool:
        """
        Cancel the current execution.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        if not self.is_executing:
            return True
        self.is_cancelled = True
        if self.execution_task and not self.execution_task.done():
            self.execution_task.cancel()
            try:
                await self.execution_task
            except asyncio.CancelledError:
                pass
        self._trigger_callbacks('cancelled', {'algorithm_id': self.
            algorithm_id, 'order_id': self.original_order.client_order_id if
            self.original_order else None, 'slices_executed': self.
            slices_executed, 'slices_total': self.slices_total, 'timestamp':
            datetime.utcnow().isoformat()})
        return True

    @with_broker_api_resilience('get_status')
    async def get_status(self) ->Dict[str, Any]:
        """
        Get the current status of the execution.
        
        Returns:
            Dictionary with the current status
        """
        now = datetime.utcnow()
        total_quantity = (self.original_order.quantity if self.
            original_order else 0)
        filled_quantity = sum(report.filled_quantity for report in self.
            execution_reports.values() if hasattr(report, 'filled_quantity'))
        time_elapsed = (now - self.start_time).total_seconds(
            ) if self.start_time else 0
        time_total = (self.end_time - self.start_time).total_seconds(
            ) if self.start_time and self.end_time else 0
        time_percentage = (time_elapsed / time_total * 100 if time_total > 
            0 else 0)
        return {'algorithm_id': self.algorithm_id, 'is_executing': self.
            is_executing, 'is_cancelled': self.is_cancelled,
            'original_order_id': self.original_order.client_order_id if
            self.original_order else None, 'start_time': self.start_time.
            isoformat() if self.start_time else None, 'end_time': self.
            end_time.isoformat() if self.end_time else None,
            'next_slice_time': self.next_slice_time.isoformat() if self.
            next_slice_time else None, 'slices_executed': self.
            slices_executed, 'slices_total': self.slices_total,
            'slices_percentage': self.slices_executed / self.slices_total *
            100 if self.slices_total > 0 else 0, 'time_elapsed_seconds':
            time_elapsed, 'time_total_seconds': time_total,
            'time_percentage': time_percentage, 'total_quantity':
            total_quantity, 'filled_quantity': filled_quantity,
            'quantity_percentage': filled_quantity / total_quantity * 100 if
            total_quantity > 0 else 0, 'child_orders': len(self.
            child_orders), 'execution_reports': len(self.execution_reports)}

    def _calculate_num_slices(self, order: OrderRequest) ->int:
        """
        Calculate the optimal number of slices for the order.
        
        Args:
            order: The order to slice
            
        Returns:
            Number of slices to use
        """
        num_slices = self.num_slices
        slice_size = order.quantity / num_slices
        if slice_size < self.min_slice_size:
            num_slices = max(1, int(order.quantity / self.min_slice_size))
        return num_slices

    def _calculate_slice_size(self, order: OrderRequest, slice_index: int,
        num_slices: int) ->float:
        """
        Calculate the size for a specific slice.
        
        Args:
            order: The original order
            slice_index: Index of the current slice (0-based)
            num_slices: Total number of slices
            
        Returns:
            Size for the slice
        """
        import random
        base_size = order.quantity / num_slices
        if self.random_variance > 0:
            random_factor = 1.0 + random.uniform(-self.random_variance,
                self.random_variance)
            size = base_size * random_factor
            remaining = order.quantity - sum(self.child_orders[order_id].
                get('quantity', 0) for order_id in self.child_orders)
            if slice_index == num_slices - 1:
                size = remaining
            else:
                remaining_slices = num_slices - slice_index - 1
                max_size = remaining - remaining_slices * base_size * (1 -
                    self.random_variance)
                size = min(size, max_size)
        elif slice_index == num_slices - 1:
            size = order.quantity - sum(self.child_orders[order_id].get(
                'quantity', 0) for order_id in self.child_orders)
        else:
            size = base_size
        return max(self.min_slice_size, round(size, 2))

    @async_with_exception_handling
    async def _execute_slices(self, order: OrderRequest, slice_interval:
        timedelta) ->ExecutionResult:
        """
        Execute all slices of the order over time.
        
        Args:
            order: The original order
            slice_interval: Time interval between slices
            
        Returns:
            ExecutionResult with details of the execution
        """
        for slice_index in range(self.slices_total):
            if self.is_cancelled:
                self.logger.info('TWAP execution cancelled')
                break
            now = datetime.utcnow()
            if now < self.next_slice_time:
                wait_time = (self.next_slice_time - now).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            try:
                slice_size = self._calculate_slice_size(order, slice_index,
                    self.slices_total)
                execution_report = await self._place_slice_order(order,
                    slice_index, slice_size)
                if execution_report:
                    self.execution_reports[execution_report.client_order_id
                        ] = execution_report
                self.slices_executed += 1
                self._trigger_callbacks('progress', {'algorithm_id': self.
                    algorithm_id, 'order_id': order.client_order_id,
                    'slice_index': slice_index, 'slices_total': self.
                    slices_total, 'slice_size': slice_size, 'timestamp':
                    datetime.utcnow().isoformat()})
            except Exception as e:
                self.logger.error(
                    f'Error executing slice {slice_index}: {str(e)}')
            self.next_slice_time = self.start_time + slice_interval * (
                slice_index + 1)
        if self.is_cancelled:
            status = 'CANCELLED'
        elif self.slices_executed == 0:
            status = 'FAILED'
        elif self.slices_executed < self.slices_total:
            status = 'PARTIAL'
        else:
            status = 'COMPLETED'
        execution_time = (datetime.utcnow() - self.start_time).total_seconds(
            ) / 60
        result = ExecutionResult(algorithm_id=self.algorithm_id,
            original_order_id=order.client_order_id, status=status,
            execution_reports=list(self.execution_reports.values()),
            metrics={'slices_executed': self.slices_executed,
            'slices_total': self.slices_total, 'duration_minutes': self.
            duration_minutes, 'execution_time_minutes': execution_time})
        self._trigger_callbacks('completed', result.to_dict())
        return result

    @async_with_exception_handling
    async def _place_slice_order(self, parent_order: OrderRequest,
        slice_index: int, quantity: float) ->Optional[ExecutionReport]:
        """
        Place an order for a single slice.
        
        Args:
            parent_order: The original parent order
            slice_index: Index of the current slice (0-based)
            quantity: Quantity for this slice
            
        Returns:
            Execution report for the slice order, or None if failed
        """
        broker_name = self.default_broker
        if not broker_name or broker_name not in self.broker_adapters:
            if not self.broker_adapters:
                self.logger.error('No broker adapters available')
                return None
            broker_name = next(iter(self.broker_adapters))
        adapter = self.broker_adapters[broker_name]
        child_order_id = (
            f'{parent_order.client_order_id}_TWAP_{slice_index}_{uuid.uuid4().hex[:8]}'
            )
        child_order = OrderRequest(instrument=parent_order.instrument,
            order_type=parent_order.order_type, direction=parent_order.
            direction, quantity=quantity, price=parent_order.price,
            stop_loss=parent_order.stop_loss, take_profit=parent_order.
            take_profit, client_order_id=child_order_id)
        self.child_orders[child_order_id] = {'broker_name': broker_name,
            'quantity': quantity, 'slice_index': slice_index,
            'parent_order_id': parent_order.client_order_id, 'timestamp':
            datetime.utcnow().isoformat()}
        self.logger.info(
            f'Placing TWAP slice {slice_index + 1}/{self.slices_total} order {child_order_id}: {child_order.instrument} {child_order.direction.value} {child_order.quantity}'
            )
        try:
            execution_report = await adapter.place_order(child_order)
            return execution_report
        except Exception as e:
            self.logger.error(f'Error placing TWAP slice order: {str(e)}')
            return None
