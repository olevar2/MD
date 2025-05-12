#!/usr/bin/env python
"""
Service Health Check Script for Forex Trading Platform

This script checks the health of a service by making a request to its health endpoint.
It can be used to verify that a service is running and ready to accept requests.

Usage:
    python service_health_check.py [--url URL] [--timeout TIMEOUT] [--retries RETRIES]
                                  [--delay DELAY] [--verbose]

Options:
    --url URL           URL of the service health endpoint (default: http://localhost:8000/health)
    --timeout TIMEOUT   Request timeout in seconds (default: 5)
    --retries RETRIES   Number of retries (default: 3)
    --delay DELAY       Delay between retries in seconds (default: 2)
    --verbose           Enable verbose output
"""

import argparse
import json
import logging
import sys
import time
from typing import Dict, Any, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("service_health_check")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Check service health")
    parser.add_argument("--url", type=str, default="http://localhost:8000/health",
                        help="URL of the service health endpoint")
    parser.add_argument("--timeout", type=int, default=5,
                        help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3,
                        help="Number of retries")
    parser.add_argument("--delay", type=int, default=2,
                        help="Delay between retries in seconds")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    return parser.parse_args()


def check_health(url: str, timeout: int = 5) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Check the health of a service.
    
    Args:
        url: URL of the service health endpoint
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success, response_data)
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Parse response
        try:
            data = response.json()
        except json.JSONDecodeError:
            data = {"status": response.text}
        
        # Check if status is healthy
        if "status" in data and data["status"].lower() == "healthy":
            return True, data
        else:
            return False, data
    except requests.RequestException as e:
        logger.debug(f"Health check failed: {e}")
        return False, None


def main():
    """Main function."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check service health with retries
    for attempt in range(args.retries + 1):
        if attempt > 0:
            logger.info(f"Retry {attempt}/{args.retries} after {args.delay} seconds...")
            time.sleep(args.delay)
        
        logger.info(f"Checking service health at {args.url}...")
        success, data = check_health(args.url, args.timeout)
        
        if success:
            logger.info(f"Service is healthy: {data}")
            return 0
        elif data:
            logger.warning(f"Service is not healthy: {data}")
        else:
            logger.warning(f"Service is not reachable")
    
    logger.error(f"Service health check failed after {args.retries} retries")
    return 1


if __name__ == "__main__":
    sys.exit(main())
