#!/usr/bin/env python
"""
Service Startup Script for Forex Trading Platform

This script starts a service and verifies that it is running correctly.
It handles environment validation, dependency checking, and health verification.

Usage:
    python start_service.py --service SERVICE [--env-file ENV_FILE] [--port PORT]
                           [--host HOST] [--timeout TIMEOUT] [--skip-deps]
                           [--skip-health-check] [--verbose]

Options:
    --service SERVICE           Name of the service to start
    --env-file ENV_FILE         Path to the .env file (default: SERVICE_DIR/.env)
    --port PORT                 Port to bind to (overrides .env)
    --host HOST                 Host to bind to (overrides .env)
    --timeout TIMEOUT           Startup timeout in seconds (default: 30)
    --skip-deps                 Skip dependency checking
    --skip-health-check         Skip health check verification
    --verbose                   Enable verbose output
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("start_service")

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

# Service health check endpoints
SERVICE_HEALTH_ENDPOINTS = {
    "data-pipeline-service": "/health",
    "feature-store-service": "/health",
    "analysis-engine-service": "/health",
    "ml-integration-service": "/health",
    "trading-gateway-service": "/health",
    "portfolio-management-service": "/health",
    "monitoring-alerting-service": "/health",
}

# Service startup commands
SERVICE_STARTUP_COMMANDS = {
    "data-pipeline-service": ["python", "-m", "data_pipeline_service.main"],
    "feature-store-service": ["python", "-m", "feature_store_service.main"],
    "analysis-engine-service": ["python", "-m", "analysis_engine.main"],
    "ml-integration-service": ["python", "-m", "ml_integration_service.main"],
    "trading-gateway-service": ["python", "-m", "trading_gateway_service.main"],
    "portfolio-management-service": ["python", "-m", "portfolio_management_service.main"],
    "monitoring-alerting-service": ["python", "-m", "monitoring_alerting_service.main"],
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start a service")
    parser.add_argument("--service", type=str, required=True,
                        help="Name of the service to start")
    parser.add_argument("--env-file", type=str,
                        help="Path to the .env file")
    parser.add_argument("--port", type=int,
                        help="Port to bind to (overrides .env)")
    parser.add_argument("--host", type=str,
                        help="Host to bind to (overrides .env)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Startup timeout in seconds")
    parser.add_argument("--skip-deps", action="store_true",
                        help="Skip dependency checking")
    parser.add_argument("--skip-health-check", action="store_true",
                        help="Skip health check verification")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    return parser.parse_args()


def load_env_file(env_file: str) -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file: Path to the .env file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    
    try:
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                
                # Parse variable assignment
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    env_vars[key] = value
    except FileNotFoundError:
        logger.warning(f"Environment file not found: {env_file}")
    except Exception as e:
        logger.error(f"Error parsing environment file {env_file}: {e}")
    
    return env_vars


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
        # Get dependency port from .env file
        env_file = Path(dependency) / ".env"
        if env_file.exists():
            env_vars = load_env_file(str(env_file))
            port = env_vars.get("PORT", "8000")
        else:
            port = "8000"
        
        # Check if dependency is running
        health_endpoint = SERVICE_HEALTH_ENDPOINTS.get(dependency, "/health")
        url = f"http://localhost:{port}{health_endpoint}"
        
        logger.info(f"Checking dependency {dependency} at {url}...")
        
        # Run health check script
        result = subprocess.run(
            [sys.executable, "scripts/service_health_check.py", "--url", url],
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


def start_service(service: str, env_vars: Dict[str, str]) -> Optional[subprocess.Popen]:
    """
    Start a service.
    
    Args:
        service: Service name
        env_vars: Environment variables
        
    Returns:
        Subprocess object if successful, None otherwise
    """
    # Get startup command
    command = SERVICE_STARTUP_COMMANDS.get(service)
    
    if not command:
        logger.error(f"No startup command found for service {service}")
        return None
    
    # Create environment for subprocess
    env = os.environ.copy()
    env.update(env_vars)
    
    # Start service
    logger.info(f"Starting service {service} with command: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command,
            cwd=service,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Service {service} started with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to start service {service}: {e}")
        return None


def check_service_health(service: str, port: str, timeout: int = 30, skip_health_check: bool = False) -> bool:
    """
    Check if a service is healthy.
    
    Args:
        service: Service name
        port: Service port
        timeout: Timeout in seconds
        skip_health_check: Skip health check verification
        
    Returns:
        True if service is healthy, False otherwise
    """
    if skip_health_check:
        logger.info("Skipping health check verification")
        return True
    
    # Get health endpoint
    health_endpoint = SERVICE_HEALTH_ENDPOINTS.get(service, "/health")
    url = f"http://localhost:{port}{health_endpoint}"
    
    logger.info(f"Checking service health at {url}...")
    
    # Wait for service to start
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Run health check script
        result = subprocess.run(
            [sys.executable, "scripts/service_health_check.py", "--url", url, "--retries", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Service {service} is healthy")
            return True
        
        logger.debug(f"Service {service} is not yet healthy, waiting...")
        time.sleep(1)
    
    logger.error(f"Service {service} failed to become healthy within {timeout} seconds")
    return False


def main():
    """Main function."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate service
    if args.service not in SERVICE_STARTUP_COMMANDS:
        logger.error(f"Unknown service: {args.service}")
        return 1
    
    # Load environment variables
    env_file = args.env_file or Path(args.service) / ".env"
    env_vars = load_env_file(str(env_file))
    
    # Override environment variables
    if args.port:
        env_vars["PORT"] = str(args.port)
    
    if args.host:
        env_vars["HOST"] = args.host
    
    # Check dependencies
    if not check_dependencies(args.service, args.skip_deps):
        logger.error(f"Dependency check failed for {args.service}")
        return 1
    
    # Start service
    process = start_service(args.service, env_vars)
    
    if not process:
        logger.error(f"Failed to start service {args.service}")
        return 1
    
    # Check service health
    port = env_vars.get("PORT", "8000")
    if not check_service_health(args.service, port, args.timeout, args.skip_health_check):
        logger.error(f"Service {args.service} failed health check")
        
        # Terminate process
        logger.info(f"Terminating service {args.service} (PID {process.pid})")
        process.terminate()
        
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"Service {args.service} did not terminate gracefully, killing it")
            process.kill()
        
        return 1
    
    logger.info(f"Service {args.service} started successfully")
    
    # Keep process running
    try:
        while True:
            # Check if process is still running
            if process.poll() is not None:
                logger.error(f"Service {args.service} exited unexpectedly with code {process.returncode}")
                return 1
            
            # Print process output
            stdout_line = process.stdout.readline()
            if stdout_line:
                print(f"[{args.service}] {stdout_line.strip()}")
            
            stderr_line = process.stderr.readline()
            if stderr_line:
                print(f"[{args.service}] ERROR: {stderr_line.strip()}")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info(f"Stopping service {args.service}...")
        
        # Terminate process
        process.terminate()
        
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"Service {args.service} did not terminate gracefully, killing it")
            process.kill()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
