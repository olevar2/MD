"""
Portfolio Management Service Main Application.

This service handles portfolio state tracking, balance management, and position tracking.
"""
import os
import logging
import traceback
from typing import Dict, Optional, Union
import asyncio
import uvicorn
from common_lib.correlation import FastAPICorrelationIdMiddleware
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from prometheus_client import make_asgi_app
from core_foundations.utils.logger import get_logger
from core_foundations.api.health_check import add_health_check_to_app
from core_foundations.events.kafka_event_bus import KafkaEventBus
from core_foundations.events.event_schema import Event, EventType
from core_foundations.models.schemas import HealthStatus, HealthCheckResult
from portfolio_management_service.error import register_exception_handlers
from portfolio_management_service.api.router import api_router
from portfolio_management_service.db.connection import initialize_database, dispose_database, get_db_session, get_engine
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'
    )
APP_NAME = 'Portfolio Management Service'
APP_VERSION = '0.1.0'
logger = get_logger('portfolio-management-service')
app = FastAPI(title=APP_NAME, description=
    'Service for tracking portfolio state, positions, and balances',
    version=APP_VERSION)


from portfolio_management_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@app.on_event('startup')
@async_with_exception_handling
async def startup_event():
    """Application startup logic."""
    logger.info('Starting Portfolio Management Service...')
    try:
        await initialize_database()
        logger.info('Database initialized successfully.')
        logger.info('Kafka connection check function updated.')
    except Exception as e:
        logger.error(f'Error during startup: {e}', exc_info=True)


@app.on_event('shutdown')
async def shutdown_event():
    """Application shutdown logic."""
    logger.info('Shutting down Portfolio Management Service...')
    await dispose_database()
    logger.info('Service shutdown complete.')


@async_with_exception_handling
async def check_db_connection_async() ->HealthCheckResult:
    """Async check for database connection status."""
    try:
        async with get_db_session():
            return HealthCheckResult(status='OK', details=
                'Database connection successful.')
    except Exception as e:
        logger.error(f'Database health check failed: {e}', exc_info=True)
        error_detail = f'Database connection failed: {type(e).__name__}'
        return HealthCheckResult(status='ERROR', details=error_detail)


from portfolio_management_service.api.middleware import CorrelationIdMiddleware
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=
    True, allow_methods=['*'], allow_headers=['*'])
app.add_middleware(FastAPICorrelationIdMiddleware)
app.add_middleware(CorrelationIdMiddleware)
metrics_app = make_asgi_app()
app.mount('/metrics', metrics_app)
health_checks = [{'name': 'database_connection', 'check_func':
    check_db_connection_async, 'critical': True}, {'name':
    'kafka_connection', 'check_func': lambda : HealthCheckResult(status=
    'OK', details='Kafka connection placeholder OK.'), 'critical': True}]
dependencies: Dict[str, callable] = {}
add_health_check_to_app(app=app, service_name=APP_NAME, version=APP_VERSION,
    checks=health_checks, dependencies=dependencies)
register_exception_handlers(app)
app.include_router(api_router, prefix='/api/v1')
if __name__ == '__main__':
    logger.info(f'Starting {APP_NAME} v{APP_VERSION}')
    uvicorn.run('portfolio_management_service.main:app', host='0.0.0.0',
        port=8002, reload=True, log_level='info')
