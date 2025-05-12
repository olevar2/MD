#!/usr/bin/env python
"""
Platform Shutdown Script for Forex Trading Platform

This script stops all running services in the reverse order of their startup.
It handles graceful shutdown and cleanup of resources.

Usage:
    python stop_platform.py [--timeout TIMEOUT] [--services SERVICES] [--force] [--verbose]

Options:
    --timeout TIMEOUT          Shutdown timeout in seconds (default: 10)
    --services SERVICES        Comma-separated list of services to stop (default: all)
    --force                    Force kill services if they don't shut down gracefully
    --verbose                  Enable verbose output
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("stop_platform")

# Service shutdown order (reverse of startup order)
SERVICE_SHUTDOWN_ORDER = [
    "monitoring-alerting-service",
    "trading-gateway-service",
    "portfolio-management-service",
    "ml-integration-service",
    "analysis-engine-service",
    "feature-store-service",
    "data-pipeline-service",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stop the Forex Trading Platform")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Shutdown timeout in seconds")
    parser.add_argument("--services", type=str,
                        help="Comma-separated list of services to stop")
    parser.add_argument("--force", action="store_true",
                        help="Force kill services if they don't shut down gracefully")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    return parser.parse_args()


def get_service_processes() -> Dict[str, List[int]]:
    """
    Get running service processes.
    
    Returns:
        Dictionary of service name to list of PIDs
    """
    service_processes = {}
    
    # Get all Python processes
    try:
        result = subprocess.run(
            ["ps", "-ef"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to get process list: {result.stderr}")
            return {}
        
        # Parse process list
        for line in result.stdout.splitlines():
            if "python" in line and "start_service.py" in line:
                # Extract service name and PID
                parts = line.split()
                pid = int(parts[1])
                
                # Find service name
                for i, part in enumerate(parts):
                    if part == "--service" and i + 1 < len(parts):
                        service = parts[i + 1]
                        
                        if service not in service_processes:
                            service_processes[service] = []
                        
                        service_processes[service].append(pid)
                        break
    except Exception as e:
        logger.error(f"Failed to get service processes: {e}")
    
    return service_processes


def stop_service(service: str, pids: List[int], timeout: int, force: bool) -> bool:
    """
    Stop a service.
    
    Args:
        service: Service name
        pids: List of PIDs
        timeout: Shutdown timeout in seconds
        force: Force kill if service doesn't shut down gracefully
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Stopping service {service} (PIDs: {pids})...")
    
    # Send SIGTERM to all processes
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            logger.warning(f"Process {pid} not found, it may have already exited")
        except Exception as e:
            logger.error(f"Failed to send SIGTERM to process {pid}: {e}")
    
    # Wait for processes to exit
    start_time = time.time()
    remaining_pids = set(pids)
    
    while remaining_pids and time.time() - start_time < timeout:
        for pid in list(remaining_pids):
            try:
                # Check if process is still running
                os.kill(pid, 0)
            except ProcessLookupError:
                # Process has exited
                remaining_pids.remove(pid)
            except Exception as e:
                logger.error(f"Failed to check if process {pid} is running: {e}")
                remaining_pids.remove(pid)
        
        if remaining_pids:
            time.sleep(0.1)
    
    # Force kill remaining processes
    if remaining_pids:
        if force:
            logger.warning(f"Service {service} did not shut down gracefully, force killing {len(remaining_pids)} processes")
            
            for pid in remaining_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    logger.warning(f"Process {pid} not found, it may have already exited")
                except Exception as e:
                    logger.error(f"Failed to send SIGKILL to process {pid}: {e}")
        else:
            logger.error(f"Service {service} did not shut down gracefully, {len(remaining_pids)} processes still running")
            return False
    
    logger.info(f"Service {service} stopped successfully")
    return True


def main():
    """Main function."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get running service processes
    service_processes = get_service_processes()
    
    if not service_processes:
        logger.info("No running services found")
        return 0
    
    logger.info(f"Found running services: {', '.join(service_processes.keys())}")
    
    # Determine which services to stop
    if args.services:
        services_to_stop = args.services.split(",")
        # Validate services
        for service in services_to_stop:
            if service not in SERVICE_SHUTDOWN_ORDER:
                logger.error(f"Unknown service: {service}")
                return 1
        
        # Filter running services
        services_to_stop = [s for s in services_to_stop if s in service_processes]
    else:
        services_to_stop = [s for s in SERVICE_SHUTDOWN_ORDER if s in service_processes]
    
    if not services_to_stop:
        logger.info("No services to stop")
        return 0
    
    logger.info(f"Stopping services: {services_to_stop}")
    
    # Stop services in reverse order
    success = True
    for service in SERVICE_SHUTDOWN_ORDER:
        if service not in services_to_stop:
            continue
        
        if not stop_service(service, service_processes[service], args.timeout, args.force):
            logger.error(f"Failed to stop service {service}")
            success = False
    
    if success:
        logger.info("All services stopped successfully")
        return 0
    else:
        logger.error("Failed to stop all services")
        return 1


if __name__ == "__main__":
    sys.exit(main())
