#!/usr/bin/env python
"""
Start Services Script for Forex Trading Platform.

This script starts all services in the Forex Trading Platform.
"""

import os
import sys
import time
import logging
import subprocess
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("start_services")

# Service configurations
SERVICES = [
    {
        "name": "ML Workbench Service",
        "module": "ml_workbench_service.main:app",
        "host": "localhost",
        "port": 8030,
        "path": "D:/MD/forex_trading_platform/ml-workbench-service",
    },
    {
        "name": "Monitoring Alerting Service",
        "module": "monitoring_alerting_service.main:app",
        "host": "localhost",
        "port": 8009,
        "path": "D:/MD/forex_trading_platform/monitoring-alerting-service",
    },
    {
        "name": "Data Pipeline Service",
        "module": "data_pipeline_service.main:app",
        "host": "localhost",
        "port": 8010,
        "path": "D:/MD/forex_trading_platform/data-pipeline-service",
    },
    {
        "name": "ML Integration Service",
        "module": "ml_integration_service.main:app",
        "host": "localhost",
        "port": 8020,
        "path": "D:/MD/forex_trading_platform/ml-integration-service",
    },
]


def start_service(service: Dict[str, Any]) -> subprocess.Popen:
    """
    Start a service.

    Args:
        service: Service configuration

    Returns:
        Service process
    """
    logger.info(f"Starting {service['name']}...")
    
    # Start the service
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            service["module"],
            "--host",
            service["host"],
            "--port",
            str(service["port"]),
            "--reload",
        ],
        cwd=service["path"],
        env={
            **os.environ,
            "PORT": str(service["port"]),
            "HOST": service["host"],
            "ENVIRONMENT": "development",
            "LOG_LEVEL": "INFO",
        },
    )
    
    logger.info(f"{service['name']} started with PID {process.pid}")
    return process


def start_all_services() -> List[subprocess.Popen]:
    """
    Start all services.

    Returns:
        List of service processes
    """
    logger.info("Starting all services...")
    
    processes = []
    for service in SERVICES:
        process = start_service(service)
        processes.append(process)
        # Wait a bit to avoid port conflicts
        time.sleep(2)
    
    return processes


def main():
    """Main function."""
    processes = start_all_services()
    
    logger.info("All services started. Press Ctrl+C to stop...")
    
    try:
        # Wait for all processes to complete
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping all services...")
        
        # Stop all processes
        for process in processes:
            process.terminate()
        
        # Wait for all processes to terminate
        for process in processes:
            process.wait()
        
        logger.info("All services stopped")


if __name__ == "__main__":
    main()