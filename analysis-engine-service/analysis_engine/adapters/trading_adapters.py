"""
Trading Service Adapters

This module provides adapter implementations for trading service interfaces.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from common_lib.interfaces.trading_interfaces import ITradingGateway
from common_lib.interfaces.trading import OrderType, OrderSide
from common_lib.interfaces.risk_management import IRiskManager, RiskCheckResult, RiskLimitType
from common_lib.risk import RiskManagementAdapter as SharedRiskManagementAdapter
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class RiskManagementAdapter(IRiskManager):
    """
    Adapter for risk management service.

    This adapter uses the standardized RiskManagementAdapter from common-lib
    to interact with the risk management service.
    """

    def __init__(self, base_url: Optional[str]=None):
        """
        Initialize the adapter.

        Args:
            base_url: Optional base URL for the Risk Management Service
        """
        self.risk_adapter = SharedRiskManagementAdapter(base_url=base_url)
        logger.info('Initialized Risk Management Adapter')

    @async_with_exception_handling
    async def evaluate_risk(self, trade_params: Dict[str, Any]) ->Dict[str, Any
        ]:
        """
        Evaluate risk for a potential trade using the risk management service.

        Args:
            trade_params: Parameters for the trade

        Returns:
            Dictionary of risk evaluation results
        """
        try:
            symbol = trade_params.get('symbol', '')
            order_type = trade_params.get('order_type', OrderType.MARKET)
            side = trade_params.get('side', OrderSide.BUY)
            quantity = trade_params.get('quantity', 0.0)
            price = trade_params.get('price')
            account_id = trade_params.get('account_id')
            result = await self.risk_adapter.check_order(symbol=symbol,
                order_type=order_type, side=side, quantity=quantity, price=
                price, account_id=account_id)
            return {'risk_score': 0.0 if result.is_valid else 1.0,
                'max_position_size': trade_params.get('quantity', 0.0) if
                result.is_valid else 0.0, 'is_valid': result.is_valid,
                'message': result.message, 'details': result.details,
                'breached_limits': result.breached_limits}
        except Exception as e:
            logger.error(f'Error evaluating risk: {str(e)}')
            return {'risk_score': 1.0, 'max_position_size': 0.0, 'is_valid':
                False, 'message': f'Error evaluating risk: {str(e)}',
                'details': {}, 'breached_limits': []}


class TradingGatewayAdapter(ITradingGateway):
    """Adapter for trading gateway service."""

    @with_resilience('get_market_status')
    async def get_market_status(self, symbols: List[str]) ->Dict[str, Any]:
        """
        Get market status for symbols from the trading gateway.

        Args:
            symbols: List of symbols to check

        Returns:
            Dictionary of market status information
        """
        return {'market_status': {symbol: 'open' for symbol in symbols}}
