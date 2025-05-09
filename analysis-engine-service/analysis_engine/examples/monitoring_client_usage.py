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

async def get_async_performance_metrics_example():
    """
    Example of getting async performance metrics using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the monitoring client
    client = factory.get_monitoring_client()
    
    try:
        # Get async performance metrics for all operations
        all_metrics = await client.get_async_performance_metrics()
        logger.info(f"Got async performance metrics for all operations: {all_metrics}")
        
        # Get async performance metrics for a specific operation
        if all_metrics.get("metrics") and len(all_metrics.get("metrics", {})) > 0:
            # Get the first operation name from the metrics
            operation_name = list(all_metrics.get("metrics", {}).keys())[0]
            
            # Get metrics for that specific operation
            operation_metrics = await client.get_async_performance_metrics(operation=operation_name)
            logger.info(f"Got async performance metrics for operation {operation_name}: {operation_metrics}")
        
        return all_metrics
    except Exception as e:
        logger.error(f"Error getting async performance metrics: {str(e)}")
        raise

async def get_memory_metrics_example():
    """
    Example of getting memory metrics using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the monitoring client
    client = factory.get_monitoring_client()
    
    try:
        # Get memory metrics
        metrics = await client.get_memory_metrics()
        
        logger.info(f"Got memory metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error getting memory metrics: {str(e)}")
        raise

async def trigger_async_performance_report_example():
    """
    Example of triggering an async performance report using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the monitoring client
    client = factory.get_monitoring_client()
    
    try:
        # Trigger async performance report
        result = await client.trigger_async_performance_report()
        
        logger.info(f"Triggered async performance report: {result}")
        return result
    except Exception as e:
        logger.error(f"Error triggering async performance report: {str(e)}")
        raise

async def get_service_health_example():
    """
    Example of getting service health using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the monitoring client
    client = factory.get_monitoring_client()
    
    try:
        # Get service health
        health = await client.get_service_health()
        
        logger.info(f"Got service health: {health}")
        
        # Check overall status
        status = health.get("status")
        if status == "healthy":
            logger.info("Service is healthy")
        elif status == "warning":
            logger.warning("Service has warnings")
        elif status == "critical":
            logger.error("Service is in critical state")
        
        # Check component statuses
        components = health.get("components", {})
        for component_name, component_data in components.items():
            component_status = component_data.get("status")
            logger.info(f"Component {component_name} status: {component_status}")
        
        return health
    except Exception as e:
        logger.error(f"Error getting service health: {str(e)}")
        raise

async def main():
    """
    Run the examples.
    """
    logger.info("Running Monitoring client examples")
    
    # Get async performance metrics
    await get_async_performance_metrics_example()
    
    # Get memory metrics
    await get_memory_metrics_example()
    
    # Trigger async performance report
    await trigger_async_performance_report_example()
    
    # Get service health
    await get_service_health_example()
    
    logger.info("Completed Monitoring client examples")

if __name__ == "__main__":
    asyncio.run(main())
