"""
Risk Management Service adapters for decoupled service communication.
"""
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from common_lib.trading.interfaces import IRiskManager, OrderRequest
from common_lib.interfaces.trading import OrderType, OrderSide
from common_lib.interfaces.risk_management import RiskCheckResult, RiskLimitType
from common_lib.risk import RiskManagementAdapter as SharedRiskAdapter
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.events.event_bus import EventBus, EventType
logger = logging.getLogger(__name__)
from core.exceptions_bridge_1 import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class RiskManagerAdapter(IRiskManager):
    """
    Adapter for risk manager that implements the common interface.

    This adapter uses the standardized RiskManagementAdapter from common-lib
    to interact with the risk management service.
    """

    def __init__(self, base_url: Optional[str]=None):
        """
        Initialize the adapter.

        Args:
            base_url: Optional base URL for the Risk Management Service
        """
        self.event_bus = EventBus()
        client_config = ServiceClientConfig(base_url=base_url or
            'http://risk-management-service:8000/api/v1', timeout=30,
            retry_config={'max_retries': 3, 'backoff_factor': 0.5,
            'retry_statuses': [408, 429, 500, 502, 503, 504]})
        self.shared_adapter = SharedRiskAdapter(config=client_config)
        logger.info('Initialized Risk Manager Adapter with shared library')

    @with_broker_api_resilience('validate_order')
    @async_with_exception_handling
    async def validate_order(self, order: OrderRequest) ->Dict[str, Any]:
        """
        Validate an order against risk limits.

        Args:
            order: Order request

        Returns:
            Validation result
        """
        try:
            result = await self.shared_adapter.check_order(symbol=order.
                symbol, order_type=order.order_type, side=order.side,
                quantity=order.quantity, price=order.price)
            await self.event_bus.publish(EventType.RISK_VALIDATION, {
                'order': order.__dict__, 'validation': result.to_dict()})
            return result.to_dict()
        except Exception as e:
            logger.error(
                f'Error validating order with risk management service: {str(e)}'
                )
            validation_result = {'is_valid': False, 'message':
                f'Error validating order: {str(e)}'}
            await self.event_bus.publish(EventType.RISK_VALIDATION, {
                'order': order.__dict__, 'validation': validation_result})
            return validation_result

    @with_broker_api_resilience('get_position_risk')
    @async_with_exception_handling
    async def get_position_risk(self, symbol: str) ->Dict[str, Any]:
        """
        Get risk metrics for a position.

        Args:
            symbol: Trading symbol

        Returns:
            Risk metrics for the position
        """
        try:
            return await self.shared_adapter.get_position_risk(symbol=symbol)
        except Exception as e:
            logger.error(f'Error getting position risk: {str(e)}')
            return {'symbol': symbol, 'var': 0.0, 'expected_shortfall': 0.0}

    @with_risk_management_resilience('get_portfolio_risk')
    @async_with_exception_handling
    async def get_portfolio_risk(self) ->Dict[str, Any]:
        """
        Get risk metrics for the entire portfolio.

        Returns:
            Risk metrics for the portfolio
        """
        try:
            return await self.shared_adapter.get_portfolio_risk()
        except Exception as e:
            logger.error(f'Error getting portfolio risk: {str(e)}')
            return {'total_var': 0.0, 'total_exposure': 0.0}
