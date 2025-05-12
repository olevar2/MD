#!/usr/bin/env python
"""
Run All Services Script for Forex Trading Platform

This script starts all services in the forex trading platform in the correct order.
It handles dependency checking, environment validation, and proper service startup order.

Usage:
    python run_all_services.py [--env ENV] [--skip-deps] [--skip-health-check]
                              [--services SERVICES] [--verbose]

Options:
    --env ENV                  Target environment (development, testing, production)
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
logger = logging.getLogger("run_all_services")

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

# Service dependencies
SERVICE_DEPENDENCIES = {
    "data-pipeline-service": [],
    "feature-store-service": ["data-pipeline-service"],
    "analysis-engine-service": ["feature-store-service"],
    "ml-integration-service": ["analysis-engine-service", "feature-store-service"],
    "trading-gateway-service": ["analysis-engine-service", "portfolio-management-service"],
    "portfolio-management-service": ["data-pipeline-service"],
    "monitoring-alerting-service": [],
}

# Service ports
SERVICE_PORTS = {
    "data-pipeline-service": 8001,
    "feature-store-service": 8002,
    "analysis-engine-service": 8003,
    "ml-integration-service": 8004,
    "trading-gateway-service": 8005,
    "portfolio-management-service": 8006,
    "monitoring-alerting-service": 8007,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run all services for Forex Trading Platform")
    parser.add_argument("--env", type=str, default="development",
                        choices=["development", "testing", "production"],
                        help="Target environment")
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


def check_dependencies(service: str, skip_deps: bool = False) -> bool:
    """
    Check if service dependencies are running.
    
    Args:
        service: Service name
        skip_deps: Skip dependency checking
        
    Returns:
        True if all dependencies are running, False otherwise
    """
    if skip_deps:
        logger.info("Skipping dependency checking")
        return True
    
    dependencies = SERVICE_DEPENDENCIES.get(service, [])
    
    if not dependencies:
        logger.info(f"Service {service} has no dependencies")
        return True
    
    logger.info(f"Checking dependencies for {service}: {dependencies}")
    
    for dependency in dependencies:
        # Get dependency port
        port = SERVICE_PORTS.get(dependency, 8000)
        
        # Check if dependency is running
        health_endpoint = f"http://localhost:{port}/health"
        
        logger.info(f"Checking dependency {dependency} at {health_endpoint}...")
        
        # Run health check script
        result = subprocess.run(
            [sys.executable, "scripts/service_health_check.py", "--url", health_endpoint],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Dependency {dependency} is not running")
            logger.error(f"Health check output: {result.stdout}")
            logger.error(f"Health check error: {result.stderr}")
            return False
    
    logger.info(f"All dependencies for {service} are running")
    return True


def start_service(service: str, env: str, skip_deps: bool, skip_health_check: bool, verbose: bool) -> Optional[subprocess.Popen]:
    """
    Start a service.
    
    Args:
        service: Service name
        env: Target environment
        skip_deps: Skip dependency checking
        skip_health_check: Skip health check verification
        verbose: Enable verbose output
        
    Returns:
        Subprocess object if successful, None otherwise
    """
    logger.info(f"Starting service {service}...")
    
    # Check dependencies
    if not check_dependencies(service, skip_deps):
        logger.error(f"Dependency check failed for {service}")
        return None
    
    # Get service directory
    service_dir = Path(service)
    
    # Get environment file
    env_file = service_dir / ".env"
    
    if not env_file.exists():
        logger.error(f"Environment file not found for {service}: {env_file}")
        return None
    
    # Get service port
    port = SERVICE_PORTS.get(service, 8000)
    
    # Build command
    cmd = [
        sys.executable,
        "-m",
        f"{service.replace('-', '_')}.main"
    ]
    
    # Set environment variables
    env_vars = os.environ.copy()
    
    # Load environment variables from .env file
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
    
    # Start service
    try:
        process = subprocess.Popen(
            cmd,
            cwd=service_dir,
            env=env_vars,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None,
            text=True
        )
        
        logger.info(f"Service {service} started with PID {process.pid}")
        
        # Wait for service to start
        if not skip_health_check:
            health_endpoint = f"http://localhost:{port}/health"
            logger.info(f"Checking service health at {health_endpoint}...")
            
            # Wait for service to start
            start_time = time.time()
            while time.time() - start_time < 30:
                # Run health check script
                result = subprocess.run(
                    [sys.executable, "scripts/service_health_check.py", "--url", health_endpoint, "--retries", "1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info(f"Service {service} is healthy")
                    break
                
                logger.debug(f"Service {service} is not yet healthy, waiting...")
                time.sleep(1)
            else:
                logger.error(f"Service {service} failed to become healthy within 30 seconds")
                process.terminate()
                return None
        
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
                if process.stdout:
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        print(f"[{service}] {stdout_line.strip()}")
                
                if process.stderr:
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
        
        process = start_service(service, args.env, args.skip_deps, args.skip_health_check, args.verbose)
        
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
