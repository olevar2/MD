"""
Risk Management Adapters Module

This module provides adapter implementations for risk management interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
import asyncio
from common_lib.simulation.interfaces import IRiskManager
from common_lib.interfaces.trading import OrderType, OrderSide
from common_lib.interfaces.risk_management import RiskCheckResult, RiskLimitType
from common_lib.risk import RiskManagementClient, RiskManagementAdapter as SharedRiskAdapter
from common_lib.service_client.base_client import ServiceClientConfig
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class RiskManagerAdapter(IRiskManager):
    """
    Adapter for risk manager that implements the common interface.

    This adapter uses the standardized RiskManagementAdapter from common-lib
    to interact with the risk management service.
    """

    def __init__(self, risk_manager_instance=None, base_url: Optional[str]=None
        ):
        """
        Initialize the adapter.

        Args:
            risk_manager_instance: Optional actual risk manager instance to wrap
            base_url: Optional base URL for the Risk Management Service
        """
        self.risk_manager = risk_manager_instance
        if not self.risk_manager:
            client_config = ServiceClientConfig(base_url=base_url or
                'http://risk-management-service:8000/api/v1', timeout=30,
                retry_config={'max_retries': 3, 'backoff_factor': 0.5,
                'retry_statuses': [408, 429, 500, 502, 503, 504]})
            self.shared_adapter = SharedRiskAdapter(config=client_config)
            logger.info('Initialized shared Risk Management Adapter')
        self._positions = {}
        self._risk_limits = {'max_position_size': 10000.0, 'max_leverage': 
            20.0, 'max_drawdown': 0.2, 'risk_per_trade': 0.02}
        self.metrics_history = []
        self.peak_balance = 100000.0

    @async_with_exception_handling
    async def check_order(self, symbol: str, direction: str, size: float,
        current_price: Dict[str, float]) ->Dict[str, Any]:
        """
        Check if an order meets risk criteria.

        Args:
            symbol: The trading symbol
            direction: Order direction (buy or sell)
            size: Order size
            current_price: Current market price

        Returns:
            Risk check result with approval status
        """
        if self.risk_manager:
            try:
                return await self.risk_manager.check_order(symbol=symbol,
                    direction=direction, size=size, current_price=current_price
                    )
            except Exception as e:
                logger.warning(
                    f'Error checking order with risk manager: {str(e)}')
        if hasattr(self, 'shared_adapter'):
            try:
                side = OrderSide.BUY if direction.lower(
                    ) == 'buy' else OrderSide.SELL
                price = current_price.get('ask' if side == OrderSide.BUY else
                    'bid', current_price.get('mid', 0.0))
                result = await self.shared_adapter.check_order(symbol=
                    symbol, order_type=OrderType.MARKET, side=side,
                    quantity=size, price=price)
                return {'approved': result.is_valid, 'reason': result.
                    message, 'details': result.details, 'breached_limits':
                    result.breached_limits, 'timestamp': datetime.now().
                    isoformat()}
            except Exception as e:
                logger.warning(
                    f'Error checking order with shared adapter: {str(e)}')
        approved = True
        reason = ''
        if size > self._risk_limits['max_position_size']:
            approved = False
            reason = (
                f"Position size {size} exceeds maximum {self._risk_limits['max_position_size']}"
                )
        total_exposure = sum(pos['size'] for pos in self._positions.values())
        if total_exposure + size > self._risk_limits['max_position_size'] * 3:
            approved = False
            reason = f'Total exposure {total_exposure + size} exceeds maximum'
        return {'approved': approved, 'reason': reason, 'timestamp':
            datetime.now().isoformat()}

    @async_with_exception_handling
    async def add_position(self, symbol: str, size: float, price: float,
        direction: str, leverage: float=1.0) ->None:
        """
        Add a new position for risk tracking.

        Args:
            symbol: The trading symbol
            size: Position size
            price: Entry price
            direction: Position direction (long or short)
            leverage: Position leverage
        """
        if self.risk_manager:
            try:
                await self.risk_manager.add_position(symbol=symbol, size=
                    size, price=price, direction=direction, leverage=leverage)
                return
            except Exception as e:
                logger.warning(
                    f'Error adding position to risk manager: {str(e)}')
        if hasattr(self, 'shared_adapter'):
            try:
                side = OrderSide.BUY if direction.lower(
                    ) == 'buy' else OrderSide.SELL
                await self.shared_adapter.add_position(symbol=symbol,
                    quantity=size, price=price, side=side, leverage=leverage)
                logger.info(
                    f'Added position to shared adapter: {symbol} {direction} {size}'
                    )
                return
            except Exception as e:
                logger.warning(
                    f'Error adding position to shared adapter: {str(e)}')
        position_id = f'pos_{len(self._positions) + 1}'
        self._positions[position_id] = {'symbol': symbol, 'size': size,
            'price': price, 'direction': direction, 'leverage': leverage,
            'timestamp': datetime.now().isoformat()}
        logger.info(
            f'Added position {position_id}: {symbol} {direction} {size}')

    @async_with_exception_handling
    async def check_risk_limits(self) ->List[Dict[str, Any]]:
        """
        Check if any risk limits are breached.

        Returns:
            List of breached risk limits
        """
        if self.risk_manager:
            try:
                return await self.risk_manager.check_risk_limits()
            except Exception as e:
                logger.warning(
                    f'Error checking risk limits with risk manager: {str(e)}')
        if hasattr(self, 'shared_adapter'):
            try:
                breached_limits = await self.shared_adapter.check_risk_limits()
                return breached_limits
            except Exception as e:
                logger.warning(
                    f'Error checking risk limits with shared adapter: {str(e)}'
                    )
        breached_limits = []
        total_exposure = sum(pos['size'] for pos in self._positions.values())
        if total_exposure > self._risk_limits['max_position_size'] * 3:
            breached_limits.append({'limit_type': 'total_exposure',
                'current_value': total_exposure, 'limit_value': self.
                _risk_limits['max_position_size'] * 3, 'severity': 'high'})
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            current_drawdown = latest_metrics.get('drawdown', 0.0)
            if current_drawdown > self._risk_limits['max_drawdown']:
                breached_limits.append({'limit_type': 'drawdown',
                    'current_value': current_drawdown, 'limit_value': self.
                    _risk_limits['max_drawdown'], 'severity': 'high'})
        return breached_limits

    @async_with_exception_handling
    async def get_portfolio_metrics(self) ->Dict[str, Any]:
        """
        Get current portfolio risk metrics.

        Returns:
            Dictionary of risk metrics
        """
        if self.risk_manager:
            try:
                return await self.risk_manager.get_portfolio_metrics()
            except Exception as e:
                logger.warning(
                    f'Error getting portfolio metrics from risk manager: {str(e)}'
                    )
        if hasattr(self, 'shared_adapter'):
            try:
                portfolio_risk = await self.shared_adapter.get_portfolio_risk()
                metrics = {'total_exposure': portfolio_risk.get(
                    'total_exposure', 0.0), 'position_count': len(
                    portfolio_risk.get('positions', {})),
                    'max_position_size': max([pos.get('size', 0.0) for pos in
                    portfolio_risk.get('positions', {}).values()]) if
                    portfolio_risk.get('positions') else 0, 'drawdown':
                    portfolio_risk.get('risk_metrics', {}).get(
                    'portfolio_max_drawdown', 0.0), 'risk_per_trade': self.
                    _risk_limits['risk_per_trade'], 'var_95':
                    portfolio_risk.get('risk_metrics', {}).get(
                    'portfolio_var_95', 0.0), 'expected_shortfall':
                    portfolio_risk.get('risk_metrics', {}).get(
                    'portfolio_expected_shortfall', 0.0), 'sharpe_ratio':
                    portfolio_risk.get('risk_metrics', {}).get(
                    'portfolio_sharpe_ratio', 0.0), 'timestamp': datetime.
                    now().isoformat()}
                self.metrics_history.append(metrics)
                return metrics
            except Exception as e:
                logger.warning(
                    f'Error getting portfolio metrics from shared adapter: {str(e)}'
                    )
        total_exposure = sum(pos['size'] for pos in self._positions.values())
        metrics = {'total_exposure': total_exposure, 'position_count': len(
            self._positions), 'max_position_size': max([pos['size'] for pos in
            self._positions.values()]) if self._positions else 0,
            'drawdown': 0.0, 'risk_per_trade': self._risk_limits[
            'risk_per_trade'], 'timestamp': datetime.now().isoformat()}
        self.metrics_history.append(metrics)
        return metrics
