"""
Base Execution Algorithm.

This module provides the base class for all execution algorithms, defining
the common interface and shared functionality.
"""
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from ..interfaces.broker_adapter_interface import BrokerAdapterInterface, OrderRequest, ExecutionReport, OrderStatus
from trading_gateway_service.error.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class ExecutionResult:
    """
    Represents the result of an execution algorithm.
    
    Contains information about the execution quality, filled orders,
    and any relevant metrics or statistics.
    """

    def __init__(self, algorithm_id: str, original_order_id: str, status:
        str, execution_reports: List[ExecutionReport]=None, metrics: Dict[
        str, Any]=None):
        """
        Initialize an execution result.
        
        Args:
            algorithm_id: ID of the execution algorithm
            original_order_id: ID of the original order
            status: Status of the execution (e.g., 'COMPLETED', 'PARTIAL', 'FAILED')
            execution_reports: List of execution reports for all child orders
            metrics: Dictionary of execution metrics and statistics
        """
        self.algorithm_id = algorithm_id
        self.original_order_id = original_order_id
        self.status = status
        self.execution_reports = execution_reports or []
        self.metrics = metrics or {}
        self.completion_time = datetime.utcnow()

    @property
    def total_filled_quantity(self) ->float:
        """Calculate the total filled quantity across all execution reports."""
        return sum(report.filled_quantity for report in self.execution_reports)

    @property
    def average_execution_price(self) ->Optional[float]:
        """Calculate the weighted average execution price."""
        total_value = 0.0
        total_quantity = 0.0
        for report in self.execution_reports:
            if report.executed_price and report.filled_quantity > 0:
                total_value += report.executed_price * report.filled_quantity
                total_quantity += report.filled_quantity
        return total_value / total_quantity if total_quantity > 0 else None

    @property
    def is_complete(self) ->bool:
        """Check if the execution is complete."""
        return self.status == 'COMPLETED'

    def to_dict(self) ->Dict[str, Any]:
        """Convert the execution result to a dictionary."""
        return {'algorithm_id': self.algorithm_id, 'original_order_id':
            self.original_order_id, 'status': self.status,
            'total_filled_quantity': self.total_filled_quantity,
            'average_execution_price': self.average_execution_price,
            'completion_time': self.completion_time.isoformat(), 'metrics':
            self.metrics, 'execution_reports': [{'order_id': report.
            order_id, 'status': report.status.value, 'filled_quantity':
            report.filled_quantity, 'executed_price': report.executed_price
            } for report in self.execution_reports]}


class BaseExecutionAlgorithm(ABC):
    """
    Base class for all execution algorithms.
    
    This abstract class defines the interface that all execution algorithms
    must implement, as well as providing common functionality.
    """

    def __init__(self, broker_adapters: Dict[str, BrokerAdapterInterface],
        logger: Optional[logging.Logger]=None, config: Optional[Dict[str,
        Any]]=None):
        """
        Initialize the execution algorithm.
        
        Args:
            broker_adapters: Dictionary of broker adapters by name
            logger: Logger instance
            config: Algorithm-specific configuration
        """
        self.broker_adapters = broker_adapters
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        self.algorithm_id = str(uuid.uuid4())
        self.status_callbacks: Dict[str, List[Callable]] = {'started': [],
            'progress': [], 'completed': [], 'failed': [], 'cancelled': []}

    def register_callback(self, event_type: str, callback: Callable) ->None:
        """
        Register a callback for a specific event type.
        
        Args:
            event_type: Type of event to register for
            callback: Callback function to call when the event occurs
        """
        if event_type in self.status_callbacks:
            self.status_callbacks[event_type].append(callback)
        else:
            self.logger.warning(f'Unknown event type: {event_type}')

    @with_exception_handling
    def _trigger_callbacks(self, event_type: str, data: Any=None) ->None:
        """
        Trigger all callbacks for a specific event type.
        
        Args:
            event_type: Type of event that occurred
            data: Data to pass to the callbacks
        """
        if event_type in self.status_callbacks:
            for callback in self.status_callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(
                        f'Error in {event_type} callback: {str(e)}')

    @abstractmethod
    async def execute(self, order: OrderRequest) ->ExecutionResult:
        """
        Execute the algorithm for the given order.
        
        Args:
            order: The order to execute
            
        Returns:
            ExecutionResult with details of the execution
        """
        pass

    @abstractmethod
    async def cancel(self) ->bool:
        """
        Cancel the current execution.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        pass

    @with_broker_api_resilience('get_status')
    @abstractmethod
    async def get_status(self) ->Dict[str, Any]:
        """
        Get the current status of the execution.
        
        Returns:
            Dictionary with the current status
        """
        pass
