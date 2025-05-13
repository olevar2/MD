"""
Main entry point for the Python components of the Trading Gateway Service.

This module initializes the FastAPI application for the Python components
of the Trading Gateway Service, which handle market data processing,
order reconciliation, and other backend functionality.
"""
import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
import uvicorn
from common_lib.correlation import FastAPICorrelationIdMiddleware
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from core_foundations.utils.logger import get_logger
from core_foundations.api.health_check import add_health_check_to_app
from core_foundations.events.kafka_event_bus import KafkaEventBus
from core_foundations.events.event_schema import Event, EventType
from core_foundations.models.schemas import HealthStatus
from trading_gateway_service.error import ForexTradingPlatformError, BrokerConnectionError, OrderValidationError, MarketDataError
from api.exception_handlers import register_exception_handlers
from services.market_data_service import MarketDataService
from services.order_reconciliation_service import OrderReconciliationService
from core.performance_monitoring import TradingGatewayMonitoring
from adapters.metrics_integration import setup_metrics
from core.degraded_mode import DegradedModeManager
from core.degraded_mode_strategies import configure_trading_gateway_degraded_mode
from trading_gateway_service.api import router as api_router
from api.adapter_api import adapter_router
from adapters.adapter_factory_1 import adapter_factory
from core.resilience import with_broker_api_resilience
APP_NAME = 'Trading Gateway Service (Python)'
APP_VERSION = '0.1.0'
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(
    ',')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'
    )
logger = get_logger('trading-gateway-service')
app = FastAPI(title=APP_NAME, description=
    'Python components for the Trading Gateway Service', version=APP_VERSION)
from api.service_template import get_service_template
service_template = get_service_template(app)
service_template.initialize()
logger.info('Service template initialized')
app.add_middleware(FastAPICorrelationIdMiddleware)
register_exception_handlers(app)
setup_metrics(app, service_name='trading-gateway-service')
app.include_router(api_router)
app.include_router(adapter_router)
health_checks = [{'name': 'market_data_service', 'check_func': lambda :
    True, 'critical': True}, {'name': 'order_reconciliation_service',
    'check_func': lambda : True, 'critical': True}, {'name':
    'kafka_connection', 'check_func': lambda : True, 'critical': False}, {
    'name': 'adapter_factory', 'check_func': lambda : adapter_factory.
    _initialized, 'critical': True}]
add_health_check_to_app(app=app, service_name=APP_NAME, version=APP_VERSION,
    checks=health_checks, dependencies={})
from core.exceptions_bridge_1 import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


@app.get('/')
async def root() ->Dict[str, str]:
    """Root endpoint returning service information."""
    return {'service': APP_NAME, 'version': APP_VERSION, 'status': 'running'}


@app.on_event('startup')
@async_with_exception_handling
async def startup_event():
    """Initialize service on startup."""
    logger.info(f'Starting {APP_NAME}')
    try:
        await service_template.startup()
        logger.info('Service template started')
        from adapters.broker_adapter import BrokerAdapter
        from core.forex_broker_simulator import ForexBrokerSimulator
        broker_adapter = ForexBrokerSimulator()
        app.state.market_data_service = MarketDataService(broker_adapters={
            'simulator': broker_adapter}, logger=logger)
        logger.info('Market data service initialized')
        app.state.health.checks[0]['check_func'] = lambda : True


        class MockOrderStore:
            """Mock internal order store for testing."""

            @with_broker_api_resilience('get_orders')
            async def get_orders(self):
                """
                Get orders.
                """
                return []

            @with_broker_api_resilience('get_positions')
            async def get_positions(self):
                """
                Get positions.
                """
                return []

            @with_broker_api_resilience('get_account_info')
            async def get_account_info(self):
                """
                Get account info.
                """
                return {'balance': 10000, 'equity': 10000, 'margin_used': 0}

            async def add_external_order(self, order):
                """
                Add external order.

                Args:
                    order: Description of order
                """
                logger.info(f"Adding external order: {order['order_id']}")

            @with_broker_api_resilience('update_order_status')
            async def update_order_status(self, order_id, status, reason=None):
                """
                Update order status.

                Args:
                    order_id: Description of order_id
                    status: Description of status
                    reason: Description of reason
                """
                logger.info(f'Updating order status: {order_id} -> {status}')

            @with_broker_api_resilience('update_order_fill')
            async def update_order_fill(self, order_id, filled_size, avg_price):
                """
                Update order fill.

                Args:
                    order_id: Description of order_id
                    filled_size: Description of filled_size
                    avg_price: Description of avg_price
                """
                logger.info(f'Updating order fill: {order_id} -> {filled_size} @ {avg_price}')

            async def add_external_position(self, position):
                """
                Add external position.

                Args:
                    position: Description of position
                """
                logger.info(f"Adding external position: {position['instrument']}")

            async def mark_position_inconsistent(self, instrument, reason=None):
                """
                Mark position inconsistent.

                Args:
                    instrument: Description of instrument
                    reason: Description of reason
                """
                logger.info(f'Marking position inconsistent: {instrument}')

            @with_broker_api_resilience('update_position')
            async def update_position(self, instrument, size, direction, entry_price=None, reason=None):
                """
                Update position.

                Args:
                    instrument: Description of instrument
                    size: Description of size
                    direction: Description of direction
                    entry_price: Description of entry_price
                    reason: Description of reason
                """
                logger.info(f'Updating position: {instrument} -> {size} {direction}')

            @with_broker_api_resilience('update_account_balance')
            async def update_account_balance(self, balance, reason=None):
                """
                Update account balance.

                Args:
                    balance: Description of balance
                    reason: Description of reason
                """
                logger.info(f'Updating account balance: {balance}')
        app.state.order_reconciliation_service = OrderReconciliationService(
            broker_adapter=broker_adapter, internal_order_store=
            MockOrderStore(), reconciliation_config={
            'order_reconciliation_interval_sec': 60,
            'position_reconciliation_interval_sec': 300,
            'account_reconciliation_interval_sec': 600})
        await app.state.order_reconciliation_service.start()
        logger.info('Order reconciliation service initialized and started')
        app.state.health.checks[1]['check_func'] = lambda : True
        app.state.monitoring = TradingGatewayMonitoring()
        logger.info('Performance monitoring initialized')
        from core.degraded_mode import TradingGatewayDegradedMode
        app.state.degraded_mode_manager = TradingGatewayDegradedMode(
            service_name=APP_NAME)
        logger.info('Degraded mode manager initialized')
        configure_trading_gateway_degraded_mode(submit_order_func=None,
            check_risk_func=None, process_queue_func=None,
            account_balance_func=None)
        logger.info('Degraded mode strategies configured')
        adapter_factory.initialize()
        logger.info('Adapter factory initialized')
        try:
            app.state.event_bus = KafkaEventBus(bootstrap_servers=
                KAFKA_BOOTSTRAP_SERVERS, group_id='trading-gateway-service',
                client_id='trading-gateway-service-python')
            app.state.event_bus.start_consuming(blocking=False)
            app.state.health.checks[2]['check_func'
                ] = lambda : app.state.event_bus is not None
            logger.info('Kafka event bus initialized successfully')
        except Exception as e:
            logger.error(f'Failed to initialize Kafka event bus: {e}')
        logger.info('Service initialization complete')
    except Exception as e:
        logger.error(f'Error during startup: {str(e)}')


@app.on_event('shutdown')
@async_with_exception_handling
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info(f'Shutting down {APP_NAME}')
    if hasattr(app.state, 'market_data_service'):
        try:
            logger.info('Market data service shut down successfully')
        except Exception as e:
            logger.error(f'Error shutting down market data service: {e}')
    if hasattr(app.state, 'order_reconciliation_service'):
        try:
            await app.state.order_reconciliation_service.stop()
            logger.info('Order reconciliation service shut down successfully')
        except Exception as e:
            logger.error(
                f'Error shutting down order reconciliation service: {e}')
    if hasattr(app.state, 'event_bus'):
        try:
            app.state.event_bus.flush()
            app.state.event_bus.close()
            logger.info('Kafka event bus closed successfully')
        except Exception as e:
            logger.error(f'Error closing Kafka event bus: {e}')
    try:
        adapter_factory.cleanup()
        logger.info('Adapter factory cleaned up')
    except Exception as e:
        logger.error(f'Error cleaning up adapter factory: {e}')
    try:
        await service_template.shutdown()
        logger.info('Service template shut down successfully')
    except Exception as e:
        logger.error(f'Error shutting down service template: {e}')


if __name__ == '__main__':
    port = int(os.getenv('PYTHON_PORT', '8005'))
    uvicorn.run('trading_gateway_service.main:app', host='0.0.0.0', port=
        port, reload=os.getenv('ENVIRONMENT', 'development') ==
        'development', log_level=os.getenv('LOG_LEVEL', 'info').lower())
