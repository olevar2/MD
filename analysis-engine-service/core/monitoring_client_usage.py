"""
Example of using the Monitoring client

This module demonstrates how to use the standardized Monitoring client
to interact with the Analysis Engine Service API.
"""
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
from analysis_engine.clients.standardized import get_client_factory
from analysis_engine.monitoring.structured_logging import get_structured_logger
logger = get_structured_logger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@async_with_exception_handling
async def get_async_performance_metrics_example():
    """
    Example of getting async performance metrics using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_monitoring_client()
    try:
        all_metrics = await client.get_async_performance_metrics()
        logger.info(
            f'Got async performance metrics for all operations: {all_metrics}')
        if all_metrics.get('metrics') and len(all_metrics.get('metrics', {})
            ) > 0:
            operation_name = list(all_metrics.get('metrics', {}).keys())[0]
            operation_metrics = await client.get_async_performance_metrics(
                operation=operation_name)
            logger.info(
                f'Got async performance metrics for operation {operation_name}: {operation_metrics}'
                )
        return all_metrics
    except Exception as e:
        logger.error(f'Error getting async performance metrics: {str(e)}')
        raise


@async_with_exception_handling
async def get_memory_metrics_example():
    """
    Example of getting memory metrics using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_monitoring_client()
    try:
        metrics = await client.get_memory_metrics()
        logger.info(f'Got memory metrics: {metrics}')
        return metrics
    except Exception as e:
        logger.error(f'Error getting memory metrics: {str(e)}')
        raise


@async_with_exception_handling
async def trigger_async_performance_report_example():
    """
    Example of triggering an async performance report using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_monitoring_client()
    try:
        result = await client.trigger_async_performance_report()
        logger.info(f'Triggered async performance report: {result}')
        return result
    except Exception as e:
        logger.error(f'Error triggering async performance report: {str(e)}')
        raise


@async_with_exception_handling
async def get_service_health_example():
    """
    Example of getting service health using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_monitoring_client()
    try:
        health = await client.get_service_health()
        logger.info(f'Got service health: {health}')
        status = health.get('status')
        if status == 'healthy':
            logger.info('Service is healthy')
        elif status == 'warning':
            logger.warning('Service has warnings')
        elif status == 'critical':
            logger.error('Service is in critical state')
        components = health.get('components', {})
        for component_name, component_data in components.items():
            component_status = component_data.get('status')
            logger.info(
                f'Component {component_name} status: {component_status}')
        return health
    except Exception as e:
        logger.error(f'Error getting service health: {str(e)}')
        raise


async def main():
    """
    Run the examples.
    """
    logger.info('Running Monitoring client examples')
    await get_async_performance_metrics_example()
    await get_memory_metrics_example()
    await trigger_async_performance_report_example()
    await get_service_health_example()
    logger.info('Completed Monitoring client examples')


if __name__ == '__main__':
    asyncio.run(main())
