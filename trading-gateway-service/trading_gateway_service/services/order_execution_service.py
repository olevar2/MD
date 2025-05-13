"""
Order Execution Service for the trading gateway.

This service is responsible for processing order requests, routing them to appropriate
brokers, and managing the order lifecycle. It also provides advanced execution algorithms
for optimizing order execution.
"""
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from ..interfaces.broker_adapter_interface import BrokerAdapterInterface, OrderRequest, OrderType, ExecutionReport, OrderStatus
from .execution import ExecutionMode, ExecutionModeHandler, BaseExecutionService, MarketExecutionService, LimitExecutionService, StopExecutionService, ConditionalExecutionService, AlgorithmExecutionService, ExecutionAlgorithm
from trading_gateway_service.error.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class OrderExecutionService:
    """
    Service for executing orders through broker adapters.
    Handles order routing, execution, and state management.

    This is a facade that delegates to specialized execution services
    based on the order type and execution algorithm.
    """

    def __init__(self, mode: ExecutionMode=ExecutionMode.SIMULATED):
        """
        Initialize the order execution service.

        Args:
            mode: The execution mode to operate in
        """
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        self.broker_adapters: Dict[str, BrokerAdapterInterface] = {}
        self.default_broker = None
        self.mode_handler = ExecutionModeHandler(mode, self.logger)
        self.market_execution_service = None
        self.limit_execution_service = None
        self.stop_execution_service = None
        self.conditional_execution_service = None
        self.algorithm_execution_service = None
        self.algorithm_configs: Dict[str, Dict[str, Any]] = {ExecutionAlgorithm
            .SOR.value: {}, ExecutionAlgorithm.TWAP.value: {
            'duration_minutes': 60, 'num_slices': 12}, ExecutionAlgorithm.
            VWAP.value: {'duration_minutes': 60, 'num_slices': 12},
            ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL.value: {
            'max_duration_minutes': 120, 'urgency': 0.5}}
        self.logger.info(
            f'OrderExecutionService initialized in {mode.value} mode')

    def register_broker_adapter(self, name: str, adapter:
        BrokerAdapterInterface, default: bool=False) ->None:
        """
        Register a broker adapter with the service.

        Args:
            name: Name to register the adapter under
            adapter: The broker adapter instance
            default: Whether this should be the default broker
        """
        self.broker_adapters[name] = adapter
        if default or self.default_broker is None:
            self.default_broker = name
        self.logger.info(
            f"Registered broker adapter '{name}'{' (default)' if default else ''}"
            )
        self._initialize_execution_services()

    def set_default_broker(self, name: str) ->bool:
        """
        Set the default broker for order execution.

        Args:
            name: Name of the broker to set as default

        Returns:
            True if successful, False if broker not found
        """
        if name not in self.broker_adapters:
            self.logger.error(
                f"Cannot set default broker: '{name}' not registered")
            return False
        self.default_broker = name
        if self.market_execution_service:
            self.market_execution_service.set_default_broker(name)
        if self.limit_execution_service:
            self.limit_execution_service.set_default_broker(name)
        if self.stop_execution_service:
            self.stop_execution_service.set_default_broker(name)
        if self.conditional_execution_service:
            self.conditional_execution_service.set_default_broker(name)
        if self.algorithm_execution_service:
            self.algorithm_execution_service.set_default_broker(name)
        self.logger.info(f"Default broker set to '{name}'")
        return True

    def register_callback(self, event_type: str, callback: Callable) ->bool:
        """
        Register a callback for a specific event type.

        Args:
            event_type: Type of event to register for
            callback: Callback function to register

        Returns:
            True if registration successful, False otherwise
        """
        self._initialize_execution_services()
        success = True
        if self.market_execution_service:
            success = (success and self.market_execution_service.
                register_callback(event_type, callback))
        if self.limit_execution_service:
            success = (success and self.limit_execution_service.
                register_callback(event_type, callback))
        if self.stop_execution_service:
            success = (success and self.stop_execution_service.
                register_callback(event_type, callback))
        if self.conditional_execution_service:
            success = (success and self.conditional_execution_service.
                register_callback(event_type, callback))
        if self.algorithm_execution_service:
            success = (success and self.algorithm_execution_service.
                register_callback(event_type, callback))
        return success

    @with_exception_handling
    def connect(self, broker_name: Optional[str]=None, credentials:
        Optional[Dict[str, str]]=None) ->bool:
        """
        Connect to a broker or all registered brokers.

        Args:
            broker_name: Name of specific broker to connect to, or None for all
            credentials: Authentication credentials (if needed)

        Returns:
            True if connection successful, False if any connection failed
        """
        if self.mode == ExecutionMode.BACKTEST:
            self.logger.info('In backtest mode, no broker connection needed.')
            return True
        success = True
        if broker_name:
            if broker_name not in self.broker_adapters:
                self.logger.error(f"Broker '{broker_name}' not registered")
                return False
            adapters_to_connect = {broker_name: self.broker_adapters[
                broker_name]}
        else:
            adapters_to_connect = self.broker_adapters
        for name, adapter in adapters_to_connect.items():
            try:
                if hasattr(adapter, 'connect') and callable(adapter.connect):
                    if credentials:
                        adapter_success = adapter.connect(credentials)
                    else:
                        adapter_success = adapter.connect()
                    if not adapter_success:
                        self.logger.error(
                            f"Failed to connect to broker '{name}'")
                        success = False
                else:
                    self.logger.warning(
                        f"Broker '{name}' does not support connect method")
            except Exception as e:
                self.logger.error(
                    f"Error connecting to broker '{name}': {str(e)}")
                success = False
        return success

    @with_exception_handling
    def disconnect(self, broker_name: Optional[str]=None) ->bool:
        """
        Disconnect from a broker or all registered brokers.

        Args:
            broker_name: Name of specific broker to disconnect from, or None for all

        Returns:
            True if disconnection successful, False if any disconnection failed
        """
        if self.mode == ExecutionMode.BACKTEST:
            self.logger.info(
                'In backtest mode, no broker disconnection needed.')
            return True
        success = True
        if broker_name:
            if broker_name not in self.broker_adapters:
                self.logger.error(f"Broker '{broker_name}' not registered")
                return False
            adapters_to_disconnect = {broker_name: self.broker_adapters[
                broker_name]}
        else:
            adapters_to_disconnect = self.broker_adapters
        for name, adapter in adapters_to_disconnect.items():
            try:
                if hasattr(adapter, 'disconnect') and callable(adapter.
                    disconnect):
                    adapter_success = adapter.disconnect()
                    if not adapter_success:
                        self.logger.error(
                            f"Failed to disconnect from broker '{name}'")
                        success = False
                else:
                    self.logger.warning(
                        f"Broker '{name}' does not support disconnect method")
            except Exception as e:
                self.logger.error(
                    f"Error disconnecting from broker '{name}': {str(e)}")
                success = False
        return success

    @with_exception_handling
    def place_order(self, order: OrderRequest, broker_name: Optional[str]=
        None, algorithm: Optional[Union[str, ExecutionAlgorithm]]=None,
        algorithm_config: Optional[Dict[str, Any]]=None, **kwargs
        ) ->ExecutionReport:
        """
        Place an order with a specified broker or using an execution algorithm.

        Args:
            order: Order request to be placed
            broker_name: Name of the broker to use, or None for default
            algorithm: Execution algorithm to use (None for direct execution)
            algorithm_config: Configuration for the execution algorithm
            **kwargs: Additional arguments specific to the order type

        Returns:
            Execution report of the order placement
        """
        import asyncio
        self._initialize_execution_services()
        if isinstance(algorithm, str):
            try:
                algorithm = ExecutionAlgorithm(algorithm.lower())
            except ValueError:
                self.logger.warning(
                    f'Unknown algorithm: {algorithm}, using direct execution')
                algorithm = None
        if broker_name is None:
            broker_name = self.default_broker
        if broker_name in self.broker_adapters:
            broker = self.broker_adapters[broker_name]
            if hasattr(broker, 'is_connected') and callable(broker.is_connected
                ) and not broker.is_connected():
                self.logger.warning(
                    f"Broker '{broker_name}' not connected. Attempting to connect..."
                    )
                if hasattr(broker, 'connect') and callable(broker.connect):
                    try:
                        broker.connect()
                    except Exception as e:
                        self.logger.error(
                            f"Failed to connect to broker '{broker_name}': {str(e)}"
                            )
        if algorithm and algorithm != ExecutionAlgorithm.DIRECT:
            execution_report = self.algorithm_execution_service.place_order(
                order=order, broker_name=broker_name, algorithm=algorithm,
                algorithm_config=algorithm_config, **kwargs)
            if asyncio.iscoroutine(execution_report):
                try:
                    execution_report = asyncio.run(execution_report)
                except Exception as e:
                    self.logger.error(f'Error executing algorithm: {str(e)}')
                    return ExecutionReport(order_id='error',
                        client_order_id=order.client_order_id, instrument=
                        order.instrument, status=OrderStatus.REJECTED,
                        direction=order.direction, order_type=order.
                        order_type, quantity=order.quantity,
                        filled_quantity=0.0, price=order.price,
                        rejection_reason=f'Error executing algorithm: {str(e)}'
                        )
            return execution_report
        try:
            if order.order_type == OrderType.MARKET:
                execution_report = self.market_execution_service.place_order(
                    order=order, broker_name=broker_name, **kwargs)
            elif order.order_type == OrderType.LIMIT:
                execution_report = self.limit_execution_service.place_order(
                    order=order, broker_name=broker_name, **kwargs)
            elif order.order_type == OrderType.STOP:
                execution_report = self.stop_execution_service.place_order(
                    order=order, broker_name=broker_name, **kwargs)
            elif order.order_type == OrderType.CONDITIONAL:
                execution_report = (self.conditional_execution_service.
                    place_order(order=order, broker_name=broker_name, **kwargs)
                    )
            else:
                error_msg = f'Unsupported order type: {order.order_type}'
                self.logger.error(error_msg)
                return ExecutionReport(order_id='error', client_order_id=
                    order.client_order_id, instrument=order.instrument,
                    status=OrderStatus.REJECTED, direction=order.direction,
                    order_type=order.order_type, quantity=order.quantity,
                    filled_quantity=0.0, price=order.price,
                    rejection_reason=error_msg)
            if hasattr(execution_report, '__await__'):
                try:
                    execution_report = asyncio.run(execution_report)
                except Exception as e:
                    self.logger.error(f'Error placing order: {str(e)}')
                    return ExecutionReport(order_id='error',
                        client_order_id=order.client_order_id, instrument=
                        order.instrument, status=OrderStatus.REJECTED,
                        direction=order.direction, order_type=order.
                        order_type, quantity=order.quantity,
                        filled_quantity=0.0, price=order.price,
                        rejection_reason=f'Error placing order: {str(e)}')
            return execution_report
        except Exception as e:
            self.logger.error(f'Error placing order: {str(e)}')
            return ExecutionReport(order_id='error', client_order_id=order.
                client_order_id, instrument=order.instrument, status=
                OrderStatus.REJECTED, direction=order.direction, order_type
                =order.order_type, quantity=order.quantity, filled_quantity
                =0.0, price=order.price, rejection_reason=
                f'Error placing order: {str(e)}')

    @with_exception_handling
    def cancel_order(self, order_id: str) ->ExecutionReport:
        """
        Cancel an existing order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            Execution report of the cancellation
        """
        import asyncio
        self._initialize_execution_services()
        try:
            if (self.market_execution_service and order_id in self.
                market_execution_service.orders):
                execution_report = self.market_execution_service.cancel_order(
                    order_id)
            elif self.limit_execution_service and order_id in self.limit_execution_service.orders:
                execution_report = self.limit_execution_service.cancel_order(
                    order_id)
            elif self.stop_execution_service and order_id in self.stop_execution_service.orders:
                execution_report = self.stop_execution_service.cancel_order(
                    order_id)
            elif self.conditional_execution_service and order_id in self.conditional_execution_service.orders:
                execution_report = (self.conditional_execution_service.
                    cancel_order(order_id))
            elif self.algorithm_execution_service and order_id in self.algorithm_execution_service.orders:
                execution_report = (self.algorithm_execution_service.
                    cancel_order(order_id))
            else:
                error_msg = f'Order ID {order_id} not found'
                self.logger.error(error_msg)
                return ExecutionReport(order_id=order_id, client_order_id=
                    'unknown', instrument='unknown', status=OrderStatus.
                    REJECTED, direction=None, order_type=None, quantity=0.0,
                    filled_quantity=0.0, price=None, rejection_reason=error_msg
                    )
            if hasattr(execution_report, '__await__'):
                try:
                    execution_report = asyncio.run(execution_report)
                except Exception as e:
                    self.logger.error(f'Error cancelling order: {str(e)}')
                    return ExecutionReport(order_id=order_id,
                        client_order_id='unknown', instrument='unknown',
                        status=OrderStatus.REJECTED, direction=None,
                        order_type=None, quantity=0.0, filled_quantity=0.0,
                        price=None, rejection_reason=
                        f'Error cancelling order: {str(e)}')
            return execution_report
        except Exception as e:
            self.logger.error(f'Error cancelling order: {str(e)}')
            return ExecutionReport(order_id=order_id, client_order_id=
                'unknown', instrument='unknown', status=OrderStatus.
                REJECTED, direction=None, order_type=None, quantity=0.0,
                filled_quantity=0.0, price=None, rejection_reason=
                f'Error cancelling order: {str(e)}')

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
        import asyncio
        self._initialize_execution_services()
        try:
            if (self.market_execution_service and order_id in self.
                market_execution_service.orders):
                execution_report = self.market_execution_service.modify_order(
                    order_id, modifications)
            elif self.limit_execution_service and order_id in self.limit_execution_service.orders:
                execution_report = self.limit_execution_service.modify_order(
                    order_id, modifications)
            elif self.stop_execution_service and order_id in self.stop_execution_service.orders:
                execution_report = self.stop_execution_service.modify_order(
                    order_id, modifications)
            elif self.conditional_execution_service and order_id in self.conditional_execution_service.orders:
                execution_report = (self.conditional_execution_service.
                    modify_order(order_id, modifications))
            elif self.algorithm_execution_service and order_id in self.algorithm_execution_service.orders:
                execution_report = (self.algorithm_execution_service.
                    modify_order(order_id, modifications))
            else:
                error_msg = f'Order ID {order_id} not found'
                self.logger.error(error_msg)
                return ExecutionReport(order_id=order_id, client_order_id=
                    'unknown', instrument='unknown', status=OrderStatus.
                    REJECTED, direction=None, order_type=None, quantity=0.0,
                    filled_quantity=0.0, price=None, rejection_reason=error_msg
                    )
            if hasattr(execution_report, '__await__'):
                try:
                    execution_report = asyncio.run(execution_report)
                except Exception as e:
                    self.logger.error(f'Error modifying order: {str(e)}')
                    return ExecutionReport(order_id=order_id,
                        client_order_id='unknown', instrument='unknown',
                        status=OrderStatus.REJECTED, direction=None,
                        order_type=None, quantity=0.0, filled_quantity=0.0,
                        price=None, rejection_reason=
                        f'Error modifying order: {str(e)}')
            return execution_report
        except Exception as e:
            self.logger.error(f'Error modifying order: {str(e)}')
            return ExecutionReport(order_id=order_id, client_order_id=
                'unknown', instrument='unknown', status=OrderStatus.
                REJECTED, direction=None, order_type=None, quantity=0.0,
                filled_quantity=0.0, price=None, rejection_reason=
                f'Error modifying order: {str(e)}')

    @with_broker_api_resilience('get_orders')
    def get_orders(self, instrument: Optional[str]=None, status: Optional[
        OrderStatus]=None) ->List[Dict[str, Any]]:
        """
        Get orders matching specified criteria.

        Args:
            instrument: Filter orders by instrument
            status: Filter orders by status

        Returns:
            List of order dictionaries
        """
        self._initialize_execution_services()
        orders = []
        if self.market_execution_service:
            orders.extend(self.market_execution_service.get_orders(
                instrument, status))
        if self.limit_execution_service:
            orders.extend(self.limit_execution_service.get_orders(
                instrument, status))
        if self.stop_execution_service:
            orders.extend(self.stop_execution_service.get_orders(instrument,
                status))
        if self.conditional_execution_service:
            orders.extend(self.conditional_execution_service.get_orders(
                instrument, status))
        if self.algorithm_execution_service:
            orders.extend(self.algorithm_execution_service.get_orders(
                instrument, status))
        return orders

    @with_broker_api_resilience('get_order')
    def get_order(self, order_id: str) ->Optional[Dict[str, Any]]:
        """
        Get information about a specific order.

        Args:
            order_id: ID of the order to retrieve

        Returns:
            Order information dictionary, or None if not found
        """
        self._initialize_execution_services()
        if (self.market_execution_service and order_id in self.
            market_execution_service.orders):
            return self.market_execution_service.get_order(order_id)
        elif self.limit_execution_service and order_id in self.limit_execution_service.orders:
            return self.limit_execution_service.get_order(order_id)
        elif self.stop_execution_service and order_id in self.stop_execution_service.orders:
            return self.stop_execution_service.get_order(order_id)
        elif self.conditional_execution_service and order_id in self.conditional_execution_service.orders:
            return self.conditional_execution_service.get_order(order_id)
        elif self.algorithm_execution_service and order_id in self.algorithm_execution_service.orders:
            return self.algorithm_execution_service.get_order(order_id)
        else:
            return None

    @with_broker_api_resilience('update_execution_status')
    def update_execution_status(self, order_id: str, status_update: Dict[
        str, Any]) ->bool:
        """
        Update the status of an order (e.g., from external callbacks/websockets).

        Args:
            order_id: ID of the order to update
            status_update: New status information

        Returns:
            True if update was successful
        """
        self._initialize_execution_services()
        if (self.market_execution_service and order_id in self.
            market_execution_service.orders):
            return self.market_execution_service.update_execution_status(
                order_id, status_update)
        elif self.limit_execution_service and order_id in self.limit_execution_service.orders:
            return self.limit_execution_service.update_execution_status(
                order_id, status_update)
        elif self.stop_execution_service and order_id in self.stop_execution_service.orders:
            return self.stop_execution_service.update_execution_status(order_id
                , status_update)
        elif self.conditional_execution_service and order_id in self.conditional_execution_service.orders:
            return self.conditional_execution_service.update_execution_status(
                order_id, status_update)
        elif self.algorithm_execution_service and order_id in self.algorithm_execution_service.orders:
            return self.algorithm_execution_service.update_execution_status(
                order_id, status_update)
        else:
            self.logger.warning(
                f'Cannot update status for unknown order ID: {order_id}')
            return False

    @with_broker_api_resilience('get_algorithm_status')
    def get_algorithm_status(self, algorithm_id: str) ->Optional[Dict[str, Any]
        ]:
        """
        Get the status of an active execution algorithm.

        Args:
            algorithm_id: ID of the algorithm to get status for

        Returns:
            Dictionary with algorithm status, or None if not found
        """
        self._initialize_execution_services()
        if self.algorithm_execution_service:
            return self.algorithm_execution_service.get_algorithm_status(
                algorithm_id)
        else:
            return None

    @with_broker_api_resilience('get_active_algorithms')
    def get_active_algorithms(self) ->List[str]:
        """
        Get a list of active algorithm IDs.

        Returns:
            List of active algorithm IDs
        """
        self._initialize_execution_services()
        if self.algorithm_execution_service:
            return self.algorithm_execution_service.get_active_algorithms()
        else:
            return []

    def _initialize_execution_services(self) ->None:
        """
        Initialize the specialized execution services if they haven't been created yet.
        """
        if not self.market_execution_service:
            self.market_execution_service = MarketExecutionService(
                broker_adapters=self.broker_adapters, mode_handler=self.
                mode_handler, logger=self.logger)
        if not self.limit_execution_service:
            self.limit_execution_service = LimitExecutionService(
                broker_adapters=self.broker_adapters, mode_handler=self.
                mode_handler, logger=self.logger)
        if not self.stop_execution_service:
            self.stop_execution_service = StopExecutionService(broker_adapters
                =self.broker_adapters, mode_handler=self.mode_handler,
                logger=self.logger)
        if not self.conditional_execution_service:
            self.conditional_execution_service = ConditionalExecutionService(
                broker_adapters=self.broker_adapters, mode_handler=self.
                mode_handler, logger=self.logger)
        if not self.algorithm_execution_service:
            self.algorithm_execution_service = AlgorithmExecutionService(
                broker_adapters=self.broker_adapters, mode_handler=self.
                mode_handler, logger=self.logger)
            for algorithm, config in self.algorithm_configs.items():
                self.algorithm_execution_service.algorithm_configs[algorithm
                    ] = config
