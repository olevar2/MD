#!/usr/bin/env python
"""
Platform Startup Script for Forex Trading Platform

This script starts all services in the correct order and verifies that they are running correctly.
It handles environment validation, dependency checking, and health verification.

Usage:
    python start_platform.py [--env ENV] [--timeout TIMEOUT] [--skip-deps]
                            [--skip-health-check] [--services SERVICES] [--verbose]

Options:
    --env ENV                  Target environment (development, testing, production)
    --timeout TIMEOUT          Startup timeout in seconds (default: 30)
    --skip-deps                Skip dependency checking
    --skip-health-check        Skip health check verification
    --services SERVICES        Comma-separated list of services to start (default: all)
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
logger = logging.getLogger("start_platform")

# Service startup order
SERVICE_STARTUP_ORDER = [
    "data-pipeline-service",
    "feature-store-service",
    "analysis-engine-service",
    "ml-integration-service",
    "portfolio-management-service",
    "trading-gateway-service",
    "monitoring-alerting-service",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start the Forex Trading Platform")
    parser.add_argument("--env", type=str, default="development",
                        choices=["development", "testing", "production"],
                        help="Target environment")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Startup timeout in seconds")
    parser.add_argument("--skip-deps", action="store_true",
                        help="Skip dependency checking")
    parser.add_argument("--skip-health-check", action="store_true",
                        help="Skip health check verification")
    parser.add_argument("--services", type=str,
                        help="Comma-separated list of services to start")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    return parser.parse_args()


def generate_env_files(env: str) -> bool:
    """
    Generate environment files for all services.
    
    Args:
        env: Target environment
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Generating environment files for {env} environment...")
    
    # Run generate_env_files.py script
    result = subprocess.run(
        [sys.executable, "scripts/generate_env_files.py", "--env", env],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Failed to generate environment files: {result.stderr}")
        return False
    
    logger.info("Environment files generated successfully")
    return True


def validate_env_config() -> bool:
    """
    Validate environment configuration for all services.
    
    Returns:
        True if all required environment variables are set, False otherwise
    """
    logger.info("Validating environment configuration...")
    
    # Run validate_env_config.py script
    result = subprocess.run(
        [sys.executable, "scripts/validate_env_config.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Environment configuration validation failed: {result.stderr}")
        return False
    
    logger.info("Environment configuration validated successfully")
    return True


def start_service(service: str, timeout: int, skip_deps: bool, skip_health_check: bool, verbose: bool) -> Optional[subprocess.Popen]:
    """
    Start a service.
    
    Args:
        service: Service name
        timeout: Startup timeout in seconds
        skip_deps: Skip dependency checking
        skip_health_check: Skip health check verification
        verbose: Enable verbose output
        
    Returns:
        Subprocess object if successful, None otherwise
    """
    logger.info(f"Starting service {service}...")
    
    # Build command
    command = [
        sys.executable, "scripts/start_service.py",
        "--service", service,
        "--timeout", str(timeout)
    ]
    
    if skip_deps:
        command.append("--skip-deps")
    
    if skip_health_check:
        command.append("--skip-health-check")
    
    if verbose:
        command.append("--verbose")
    
    # Start service
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Service {service} started with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to start service {service}: {e}")
        return None


def monitor_processes(processes: Dict[str, subprocess.Popen]) -> None:
    """
    Monitor running processes and print their output.
    
    Args:
        processes: Dictionary of service name to subprocess object
    """
    try:
        while processes:
            for service, process in list(processes.items()):
                # Check if process is still running
                if process.poll() is not None:
                    logger.error(f"Service {service} exited unexpectedly with code {process.returncode}")
                    del processes[service]
                    continue
                
                # Print process output
                stdout_line = process.stdout.readline()
                if stdout_line:
                    print(f"[{service}] {stdout_line.strip()}")
                
                stderr_line = process.stderr.readline()
                if stderr_line:
                    print(f"[{service}] ERROR: {stderr_line.strip()}")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Stopping all services...")
        
        # Terminate all processes
        for service, process in processes.items():
            logger.info(f"Terminating service {service}...")
            process.terminate()
        
        # Wait for processes to terminate
        for service, process in processes.items():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Service {service} did not terminate gracefully, killing it")
                process.kill()


def main():
    """Main function."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Generate environment files
    if not generate_env_files(args.env):
        logger.error("Failed to generate environment files")
        return 1
    
    # Validate environment configuration
    if not validate_env_config():
        logger.error("Environment configuration validation failed")
        return 1
    
    # Determine which services to start
    if args.services:
        services_to_start = args.services.split(",")
        # Validate services
        for service in services_to_start:
            if service not in SERVICE_STARTUP_ORDER:
                logger.error(f"Unknown service: {service}")
                return 1
    else:
        services_to_start = SERVICE_STARTUP_ORDER
    
    logger.info(f"Starting services: {services_to_start}")
    
    # Start services in order
    processes = {}
    for service in SERVICE_STARTUP_ORDER:
        if service not in services_to_start:
            continue
        
        process = start_service(service, args.timeout, args.skip_deps, args.skip_health_check, args.verbose)
        
        if not process:
            logger.error(f"Failed to start service {service}")
            
            # Terminate all running processes
            for running_service, running_process in processes.items():
                logger.info(f"Terminating service {running_service}...")
                running_process.terminate()
                
                try:
                    running_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Service {running_service} did not terminate gracefully, killing it")
                    running_process.kill()
            
            return 1
        
        processes[service] = process
        
        # Wait a bit before starting the next service
        time.sleep(2)
    
    logger.info("All services started successfully")
    
    # Monitor processes
    monitor_processes(processes)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
