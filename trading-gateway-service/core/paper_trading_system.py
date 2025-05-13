"""
Integration layer between paper trading and risk management components.

Implements the full paper trading loop with risk management integration,
circuit breaker monitoring, and alerting system connectivity.
"""
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime, timedelta
import asyncio
from enum import Enum
from core.broker_simulator import SimulatedBroker, OrderType, OrderStatus
from core.market_simulator import MarketDataSimulator, MarketRegime
from adapters.risk_adapters import RiskManagerAdapter
from common_lib.simulation.interfaces import IRiskManager
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from risk_management_service.risk_manager import RiskManager
    from risk_management_service.circuit_breaker import CircuitBreakerManager
    from risk_management_service.stress_testing import StressTestingEngine
    from risk_management_service.portfolio_risk import PortfolioRiskCalculator
from core.performance_monitoring import TradingGatewayMonitoring
logger = logging.getLogger(__name__)
from core.exceptions_bridge_1 import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class TradingState(str, Enum):
    """Trading system states."""
    ACTIVE = 'active'
    PAUSED = 'paused'
    STOPPED = 'stopped'
    ERROR = 'error'


class PaperTradingSystem:
    """
    Integrated paper trading system with risk management.

    Coordinates the interaction between the broker simulator,
    risk management service, and monitoring/alerting components.
    """

    def __init__(self, initial_balance: float=100000.0, base_currency: str=
        'USD', risk_params: Optional[Dict[str, Any]]=None,
        monitoring_config: Optional[Dict[str, Any]]=None):
    """
      init  .
    
    Args:
        initial_balance: Description of initial_balance
        base_currency: Description of base_currency
        risk_params: Description of risk_params
        Any]]: Description of Any]]
        monitoring_config: Description of monitoring_config
        Any]]: Description of Any]]
    
    """

        self.monitoring = TradingGatewayMonitoring(base_dir=
            monitoring_config_manager.get('base_dir', 'monitoring/trading_gateway') if
            monitoring_config else 'monitoring/trading_gateway')
        self.risk_params = risk_params or {'max_position_size': 10000.0,
            'max_leverage': 20.0, 'max_drawdown': 0.2, 'risk_per_trade': 0.02}
        self.broker = SimulatedBroker(initial_balance=initial_balance,
            base_currency=base_currency)
        self.risk_manager = RiskManagerAdapter()
        self.circuit_breaker = None
        self.market_simulator = MarketDataSimulator(symbols=['EUR/USD',
            'GBP/USD', 'USD/JPY'])
        self.monitoring_config = monitoring_config or {}
        self.state = TradingState.STOPPED
        self.last_health_check = datetime.utcnow()
        self.health_check_interval = timedelta(seconds=5)
        self.stats = {'orders_submitted': 0, 'orders_rejected': 0,
            'risk_checks_performed': 0, 'circuit_breaker_triggers': 0}

    async def start(self) ->None:
        """Start the paper trading system."""
        if self.state != TradingState.STOPPED:
            raise RuntimeError('System is already running')
        self.state = TradingState.ACTIVE
        asyncio.create_task(self._market_data_loop())
        asyncio.create_task(self._health_check_loop())
        logger.info('Paper trading system started')

    async def stop(self) ->None:
        """Stop the paper trading system."""
        self.state = TradingState.STOPPED
        logger.info('Paper trading system stopped')

    async def submit_order(self, symbol: str, order_type: OrderType,
        direction: str, size: float, price: Optional[float]=None, stop_loss:
        Optional[float]=None, take_profit: Optional[float]=None) ->Dict[str,
        Any]:
        """Submit an order with full risk checks and performance monitoring."""

        @self.monitoring.track_order_submission
        async def _submit_order():
    """
     submit order.
    
    """

            self.stats['orders_submitted'] += 1
            if self.state != TradingState.ACTIVE:
                return {'success': False, 'error':
                    f'System is in {self.state} state'}
            risk_limits = self.risk_manager.check_risk_limits()
            active_breakers = [limit for limit in risk_limits if limit.get(
                'severity') == 'high']
            if active_breakers:
                self.stats['orders_rejected'] += 1
                return {'success': False, 'error': 'Risk limit breached',
                    'details': active_breakers}

            @self.monitoring.track_risk_check
            async def _check_risk():
    """
     check risk.
    
    """

                self.stats['risk_checks_performed'] += 1
                return await self.risk_manager.check_order(symbol=symbol,
                    direction=direction, size=size, current_price=self.
                    broker.get_current_price(symbol))
            risk_result = await _check_risk()
            if not risk_result['approved']:
                self.stats['orders_rejected'] += 1
                return {'success': False, 'error': 'Risk check failed',
                    'details': risk_result['reason']}
            order_id = await self.broker.submit_order(symbol=symbol,
                order_type=order_type, direction=direction, size=size,
                price=price, stop_loss=stop_loss, take_profit=take_profit)
            return {'success': True, 'order_id': order_id}
        return await _submit_order()

    async def _market_data_loop(self):
        """Process market data updates with performance monitoring."""
        while self.state == TradingState.ACTIVE:

            @self.monitoring.track_market_data
            async def _process_market_data():
    """
     process market data.
    
    """

                market_data = self.market_simulator.generate_tick()
                for symbol, data in market_data.items():
                    await self.broker.update_market_data(symbol=symbol, bid
                        =data['bid'], ask=data['ask'], timestamp=data[
                        'timestamp'], volume=data['volume'])
                return market_data
            await _process_market_data()
            await asyncio.sleep(0.1)

    @async_with_exception_handling
    async def _health_check_loop(self):
        """Monitor system health including performance metrics."""
        while self.state == TradingState.ACTIVE:
            try:
                health_status = self.monitoring.get_health_status()
                if not health_status['healthy']:
                    for issue in health_status['issues']:
                        logger.warning(f'Health check issue: {issue}')
                if len(health_status['issues']) > 3:
                    self.state = TradingState.ERROR
                    logger.error(
                        'System entering ERROR state due to multiple health issues'
                        )
                self.last_health_check = datetime.utcnow()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f'Health check error: {e}')
                await asyncio.sleep(30)

    def _update_risk_metrics(self, market_data: Dict[str, Dict[str, Any]]
        ) ->None:
        """Update risk metrics with new market data."""
        current_prices = {symbol: (data['bid'] if pos.direction == 'buy' else
            data['ask']) for symbol, data in market_data.items() for pos in
            self.broker.positions.values() if pos.symbol == symbol}
        for position in self.broker.positions.values():
            if position.symbol in current_prices:
                pnl = position.calculate_pnl(current_prices[position.symbol])
        metrics = {'equity': self.broker.equity, 'margin_used': self.broker
            .margin_used, 'unrealized_pnl': sum(pos.calculate_pnl(
            current_prices.get(pos.symbol, pos.entry_price)) for pos in
            self.broker.positions.values())}
        peak_equity = max(self.risk_manager.peak_balance, metrics['equity'])
        metrics['drawdown'] = (peak_equity - metrics['equity']) / peak_equity
        self.risk_manager.metrics_history.append(metrics)

    def _check_circuit_breakers(self) ->None:
        """Check and update circuit breaker status."""
        if not self.risk_manager.metrics_history:
            return
        risk_limits = self.risk_manager.check_risk_limits()
        if risk_limits:
            self.stats['circuit_breaker_triggers'] += 1
            if any(limit.get('severity') == 'high' for limit in risk_limits):
                self.state = TradingState.PAUSED
                logger.warning('Risk limit breached - Trading paused')

    def _check_system_health(self) ->None:
        """Perform system health checks."""
        if self.market_simulator.last_update:
            data_age = datetime.utcnow() - self.market_simulator.last_update
            if data_age > timedelta(seconds=5):
                logger.warning('Market data delay detected')
        account = self.broker.get_account_summary()
        if account['margin_level'] < 150:
            logger.warning('Low margin level detected')
        if self.risk_manager.metrics_history:
            latest_metrics = self.risk_manager.metrics_history[-1]
            if latest_metrics['drawdown'] > self.risk_params['max_drawdown'
                ] * 0.8:
                logger.warning('Approaching max drawdown limit')

    def _get_current_prices(self) ->Dict[str, Dict[str, float]]:
        """Get current market prices."""
        return {symbol: {'bid': data['bid'], 'ask': data['ask']} for symbol,
            data in self.market_simulator.current_prices.items()}

    def _update_risk_manager(self, order_result: Dict[str, Any]) ->None:
        """Update risk manager after order execution."""
        if 'executions' in order_result:
            for execution in order_result['executions']:
                pass

    @with_broker_api_resilience('get_system_status')
    def get_system_status(self) ->Dict[str, Any]:
        """Get current system status and statistics."""
        return {'state': self.state, 'stats': self.stats, 'account': self.
            broker.get_account_summary(), 'active_circuit_breakers': self.
            circuit_breaker.get_active_circuit_breakers(), 'risk_metrics': 
            self.risk_manager.metrics_history[-1] if self.risk_manager.
            metrics_history else None}
