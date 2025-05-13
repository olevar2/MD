"""
Algorithm Execution Service.

This module provides a specialized service for executing orders using algorithms.
"""
import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from enum import Enum
from ...interfaces.broker_adapter_interface import BrokerAdapterInterface, OrderRequest, ExecutionReport, OrderStatus
from ...execution_algorithms import BaseExecutionAlgorithm, SmartOrderRoutingAlgorithm, TWAPAlgorithm, VWAPAlgorithm, ImplementationShortfallAlgorithm
from .base_execution_service import BaseExecutionService
from .execution_mode_handler import ExecutionModeHandler, ExecutionMode
from core.exceptions_bridge_1 import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class ExecutionAlgorithm(Enum):
    """Available execution algorithms."""
    DIRECT = 'direct'
    SOR = 'sor'
    TWAP = 'twap'
    VWAP = 'vwap'
    IMPLEMENTATION_SHORTFALL = 'implementation_shortfall'


class AlgorithmExecutionService(BaseExecutionService):
    """
    Service for executing orders using algorithms.

    Handles the execution of orders through various algorithms,
    with support for different execution modes.
    """

    def __init__(self, broker_adapters: Dict[str, BrokerAdapterInterface],
        mode_handler: ExecutionModeHandler, logger: Optional[logging.Logger
        ]=None):
        """
        Initialize the algorithm execution service.

        Args:
            broker_adapters: Dictionary of broker adapters by name
            mode_handler: Handler for different execution modes
            logger: Logger instance
        """
        super().__init__(broker_adapters, mode_handler, logger)
        self.active_algorithms: Dict[str, BaseExecutionAlgorithm] = {}
        self.callbacks.update({'algorithm_started': [],
            'algorithm_progress': [], 'algorithm_completed': [],
            'algorithm_failed': []})
        self.algorithm_configs: Dict[str, Dict[str, Any]] = {ExecutionAlgorithm
            .SOR.value: {}, ExecutionAlgorithm.TWAP.value: {
            'duration_minutes': 60, 'num_slices': 12}, ExecutionAlgorithm.
            VWAP.value: {'duration_minutes': 60, 'num_slices': 12},
            ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL.value: {
            'max_duration_minutes': 120, 'urgency': 0.5}}
        self.logger.info('AlgorithmExecutionService initialized')

    @with_exception_handling
    def place_order(self, order: OrderRequest, broker_name: Optional[str]=
        None, algorithm: Optional[Union[str, ExecutionAlgorithm]]=None,
        algorithm_config: Optional[Dict[str, Any]]=None, **kwargs
        ) ->ExecutionReport:
        """
        Place an order using an execution algorithm.

        Args:
            order: Order request to be placed
            broker_name: Name of the broker to use, or None for default
            algorithm: Execution algorithm to use
            algorithm_config: Configuration for the execution algorithm
            **kwargs: Additional arguments specific to the algorithm

        Returns:
            Execution report of the order placement
        """
        if isinstance(algorithm, str):
            try:
                algorithm = ExecutionAlgorithm(algorithm.lower())
            except ValueError:
                self.logger.warning(
                    f'Unknown algorithm: {algorithm}, using direct execution')
                algorithm = None
        if algorithm is None or algorithm == ExecutionAlgorithm.DIRECT:
            self.logger.warning(
                'No algorithm specified or direct execution requested, this should be handled by another service'
                )
            return ExecutionReport(order_id=str(uuid.uuid4()),
                client_order_id=order.client_order_id, instrument=order.
                instrument, status=OrderStatus.REJECTED, direction=order.
                direction, order_type=order.order_type, quantity=order.
                quantity, filled_quantity=0.0, price=order.price,
                rejection_reason=
                'Direct execution should be handled by another service')
        mode_report = self.mode_handler.handle_order_placement(order)
        if mode_report is not None:
            order_id = mode_report.order_id
            self.orders[order_id] = {'order': order, 'broker': broker_name or
                self.default_broker, 'execution_report': mode_report,
                'timestamp': datetime.utcnow(), 'algorithm': algorithm,
                'algorithm_config': algorithm_config}
            self._trigger_callbacks('order_placed', self.orders[order_id])
            return mode_report
        try:
            execute_result = self._execute_with_algorithm(order, algorithm,
                algorithm_config)
            if asyncio.iscoroutine(execute_result):
                execution_report = asyncio.run(execute_result)
            else:
                execution_report = execute_result
            order_id = execution_report.order_id
            self.orders[order_id] = {'order': order, 'broker': broker_name or
                self.default_broker, 'execution_report': execution_report,
                'timestamp': datetime.utcnow(), 'algorithm': algorithm,
                'algorithm_config': algorithm_config}
            self._trigger_callbacks('order_placed', self.orders[order_id])
            if execution_report.status == OrderStatus.FILLED:
                self._trigger_callbacks('order_filled', self.orders[order_id])
            elif execution_report.status == OrderStatus.REJECTED:
                self._trigger_callbacks('order_rejected', self.orders[order_id]
                    )
            return execution_report
        except Exception as e:
            rejection_report = self._handle_execution_error(error=e,
                client_order_id=order.client_order_id, instrument=order.
                instrument, direction=order.direction, order_type=order.
                order_type, quantity=order.quantity, price=order.price,
                error_prefix=
                f'Error executing order with algorithm {algorithm.value}')
            order_id = rejection_report.order_id
            self.orders[order_id] = {'order': order, 'broker': broker_name or
                self.default_broker, 'execution_report': rejection_report,
                'timestamp': datetime.utcnow(), 'algorithm': algorithm,
                'algorithm_config': algorithm_config, 'error': str(e)}
            self._trigger_callbacks('order_rejected', self.orders[order_id])
            return rejection_report

    @with_exception_handling
    def cancel_order(self, order_id: str) ->ExecutionReport:
        """
        Cancel an existing order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            Execution report of the cancellation
        """
        if order_id not in self.orders:
            error_msg = f'Order ID {order_id} not found'
            self.logger.error(error_msg)
            return ExecutionReport(order_id=order_id, client_order_id=
                'unknown', instrument='unknown', status=OrderStatus.
                REJECTED, direction=None, order_type=None, quantity=0.0,
                filled_quantity=0.0, price=None, rejection_reason=error_msg)
        order_info = self.orders[order_id]
        execution_report = order_info['execution_report']
        if execution_report.status in [OrderStatus.FILLED, OrderStatus.
            CANCELLED, OrderStatus.REJECTED]:
            error_msg = (
                f'Cannot cancel order with status {execution_report.status}')
            self.logger.error(error_msg)
            return ExecutionReport(order_id=order_id, client_order_id=
                execution_report.client_order_id, instrument=
                execution_report.instrument, status=OrderStatus.REJECTED,
                direction=execution_report.direction, order_type=
                execution_report.order_type, quantity=execution_report.
                quantity, filled_quantity=execution_report.filled_quantity,
                price=execution_report.price, rejection_reason=error_msg)
        mode_report = self.mode_handler.handle_order_cancellation(order_id,
            order_info)
        if mode_report is not None:
            self.orders[order_id]['execution_report'] = mode_report
            self._trigger_callbacks('order_cancelled', self.orders[order_id])
            return mode_report
        if 'algorithm' in order_info and order_info['algorithm'
            ] != ExecutionAlgorithm.DIRECT:
            algorithm_id = order_info.get('algorithm_id')
            if algorithm_id and algorithm_id in self.active_algorithms:
                try:
                    algorithm = self.active_algorithms[algorithm_id]
                    cancel_result = algorithm.cancel()
                    if asyncio.iscoroutine(cancel_result):
                        asyncio.run(cancel_result)
                    cancellation_report = ExecutionReport(order_id=order_id,
                        client_order_id=execution_report.client_order_id,
                        instrument=execution_report.instrument, status=
                        OrderStatus.CANCELLED, direction=execution_report.
                        direction, order_type=execution_report.order_type,
                        quantity=execution_report.quantity, filled_quantity
                        =execution_report.filled_quantity, price=
                        execution_report.price)
                    self.orders[order_id]['execution_report'
                        ] = cancellation_report
                    self._trigger_callbacks('order_cancelled', self.orders[
                        order_id])
                    return cancellation_report
                except Exception as e:
                    rejection_report = self._handle_execution_error(error=e,
                        order_id=order_id, client_order_id=execution_report
                        .client_order_id, instrument=execution_report.
                        instrument, direction=execution_report.direction,
                        order_type=execution_report.order_type, quantity=
                        execution_report.quantity, price=execution_report.
                        price, error_prefix='Error cancelling algorithm')
                    return rejection_report
        broker_name = order_info['broker']
        broker_adapter = self.broker_adapters[broker_name]
        try:
            cancel_order_result = broker_adapter.cancel_order(order_id)
            if asyncio.iscoroutine(cancel_order_result):
                cancellation_report = asyncio.run(cancel_order_result)
            else:
                cancellation_report = cancel_order_result
            self.orders[order_id]['execution_report'] = cancellation_report
            self._trigger_callbacks('order_cancelled', self.orders[order_id])
            return cancellation_report
        except Exception as e:
            rejection_report = self._handle_execution_error(error=e,
                order_id=order_id, client_order_id=execution_report.
                client_order_id, instrument=execution_report.instrument,
                direction=execution_report.direction, order_type=
                execution_report.order_type, quantity=execution_report.
                quantity, price=execution_report.price, error_prefix=
                'Error cancelling order')
            return rejection_report

    @with_exception_handling
    def modify_order(self, order_id: str, modifications: Dict[str, Any]
        ) ->ExecutionReport:
        """
        Modify an existing order.

        Args:
            order_id: ID of the order to modify
            modifications: Dictionary of parameters to modify

        Returns:
            Execution report of the modification
        """
        if order_id not in self.orders:
            error_msg = f'Order ID {order_id} not found'
            self.logger.error(error_msg)
            return ExecutionReport(order_id=order_id, client_order_id=
                'unknown', instrument='unknown', status=OrderStatus.
                REJECTED, direction=None, order_type=None, quantity=0.0,
                filled_quantity=0.0, price=None, rejection_reason=error_msg)
        order_info = self.orders[order_id]
        execution_report = order_info['execution_report']
        if execution_report.status in [OrderStatus.FILLED, OrderStatus.
            CANCELLED, OrderStatus.REJECTED]:
            error_msg = (
                f'Cannot modify order with status {execution_report.status}')
            self.logger.error(error_msg)
            return ExecutionReport(order_id=order_id, client_order_id=
                execution_report.client_order_id, instrument=
                execution_report.instrument, status=OrderStatus.REJECTED,
                direction=execution_report.direction, order_type=
                execution_report.order_type, quantity=execution_report.
                quantity, filled_quantity=execution_report.filled_quantity,
                price=execution_report.price, rejection_reason=error_msg)
        mode_report = self.mode_handler.handle_order_modification(order_id,
            order_info, modifications)
        if mode_report is not None:
            self.orders[order_id]['execution_report'] = mode_report
            self._trigger_callbacks('order_modified', self.orders[order_id])
            return mode_report
        if 'algorithm' in order_info and order_info['algorithm'
            ] != ExecutionAlgorithm.DIRECT:
            error_msg = 'Modifying algorithm-based orders is not supported'
            self.logger.error(error_msg)
            return ExecutionReport(order_id=order_id, client_order_id=
                execution_report.client_order_id, instrument=
                execution_report.instrument, status=OrderStatus.REJECTED,
                direction=execution_report.direction, order_type=
                execution_report.order_type, quantity=execution_report.
                quantity, filled_quantity=execution_report.filled_quantity,
                price=execution_report.price, rejection_reason=error_msg)
        broker_name = order_info['broker']
        broker_adapter = self.broker_adapters[broker_name]
        try:
            modify_order_result = broker_adapter.modify_order(order_id,
                modifications)
            if asyncio.iscoroutine(modify_order_result):
                modification_report = asyncio.run(modify_order_result)
            else:
                modification_report = modify_order_result
            self.orders[order_id]['execution_report'] = modification_report
            self._trigger_callbacks('order_modified', self.orders[order_id])
            return modification_report
        except Exception as e:
            rejection_report = self._handle_execution_error(error=e,
                order_id=order_id, client_order_id=execution_report.
                client_order_id, instrument=execution_report.instrument,
                direction=execution_report.direction, order_type=
                execution_report.order_type, quantity=execution_report.
                quantity, price=execution_report.price, error_prefix=
                'Error modifying order')
            return rejection_report

    @with_broker_api_resilience('get_algorithm_status')
    @with_exception_handling
    def get_algorithm_status(self, algorithm_id: str) ->Optional[Dict[str, Any]
        ]:
        """
        Get the status of an active execution algorithm.

        Args:
            algorithm_id: ID of the algorithm to get status for

        Returns:
            Dictionary with algorithm status, or None if not found
        """
        if algorithm_id not in self.active_algorithms:
            return None
        try:
            algo_instance = self.active_algorithms[algorithm_id]
            return asyncio.run(algo_instance.get_status())
        except Exception as e:
            self.logger.error(
                f'Error getting algorithm status for {algorithm_id}: {str(e)}')
            return None

    @with_broker_api_resilience('get_active_algorithms')
    def get_active_algorithms(self) ->List[str]:
        """
        Get a list of active algorithm IDs.

        Returns:
            List of active algorithm IDs
        """
        return list(self.active_algorithms.keys())

    @async_with_exception_handling
    async def _execute_with_algorithm(self, order: OrderRequest, algorithm:
        ExecutionAlgorithm, algorithm_config: Optional[Dict[str, Any]]=None
        ) ->ExecutionReport:
        """
        Execute an order using the specified algorithm.

        Args:
            order: Order to execute
            algorithm: Algorithm to use
            algorithm_config: Configuration for the algorithm

        Returns:
            Execution report for the order
        """
        config = self.algorithm_configs.get(algorithm.value, {}).copy()
        if algorithm_config:
            config.update(algorithm_config)
        algo_instance = self._create_algorithm_instance(algorithm, config)
        if algo_instance is None:
            return ExecutionReport(order_id=str(uuid.uuid4()),
                client_order_id=order.client_order_id, instrument=order.
                instrument, status=OrderStatus.REJECTED, direction=order.
                direction, order_type=order.order_type, quantity=order.
                quantity, filled_quantity=0.0, price=order.price,
                rejection_reason=
                f'Failed to create algorithm instance for {algorithm.value}')
        algo_instance.register_callback('started', lambda data: self.
            _trigger_callbacks('algorithm_started', data))
        algo_instance.register_callback('progress', lambda data: self.
            _trigger_callbacks('algorithm_progress', data))
        algo_instance.register_callback('completed', lambda data: self.
            _trigger_callbacks('algorithm_completed', data))
        algo_instance.register_callback('failed', lambda data: self.
            _trigger_callbacks('algorithm_failed', data))
        algorithm_id = algo_instance.algorithm_id
        self.active_algorithms[algorithm_id] = algo_instance
        try:
            result = await algo_instance.execute(order)
            if result.status == 'COMPLETED':
                status = OrderStatus.FILLED
            elif result.status == 'PARTIAL':
                status = OrderStatus.PARTIALLY_FILLED
            else:
                status = OrderStatus.REJECTED
            execution_report = ExecutionReport(order_id=str(uuid.uuid4()),
                client_order_id=order.client_order_id, instrument=order.
                instrument, status=status, direction=order.direction,
                order_type=order.order_type, quantity=order.quantity,
                filled_quantity=result.total_filled_quantity, price=order.
                price, executed_price=result.average_execution_price,
                rejection_reason=result.metrics.get('error') if status ==
                OrderStatus.REJECTED else None)
            if result.status in ['COMPLETED', 'FAILED']:
                del self.active_algorithms[algorithm_id]
            return execution_report
        except Exception as e:
            if algorithm_id in self.active_algorithms:
                del self.active_algorithms[algorithm_id]
            return ExecutionReport(order_id=str(uuid.uuid4()),
                client_order_id=order.client_order_id, instrument=order.
                instrument, status=OrderStatus.REJECTED, direction=order.
                direction, order_type=order.order_type, quantity=order.
                quantity, filled_quantity=0.0, price=order.price,
                rejection_reason=f'Algorithm execution failed: {str(e)}')

    @with_exception_handling
    def _create_algorithm_instance(self, algorithm: ExecutionAlgorithm,
        config: Dict[str, Any]) ->Optional[BaseExecutionAlgorithm]:
        """
        Create an instance of the specified algorithm.

        Args:
            algorithm: Algorithm to create
            config: Configuration for the algorithm

        Returns:
            Algorithm instance, or None if creation failed
        """
        try:
            if algorithm == ExecutionAlgorithm.SOR:
                return SmartOrderRoutingAlgorithm(broker_adapters=self.
                    broker_adapters, logger=self.logger, config=config)
            elif algorithm == ExecutionAlgorithm.TWAP:
                return TWAPAlgorithm(broker_adapters=self.broker_adapters,
                    logger=self.logger, config=config)
            elif algorithm == ExecutionAlgorithm.VWAP:
                return VWAPAlgorithm(broker_adapters=self.broker_adapters,
                    logger=self.logger, config=config)
            elif algorithm == ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL:
                return ImplementationShortfallAlgorithm(broker_adapters=
                    self.broker_adapters, logger=self.logger, config=config)
            else:
                self.logger.error(f'Unknown algorithm: {algorithm}')
                return None
        except Exception as e:
            self.logger.error(f'Error creating algorithm instance: {str(e)}')
            return None
