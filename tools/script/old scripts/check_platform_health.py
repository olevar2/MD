#!/usr/bin/env python
"""
Platform Health Check Script for Forex Trading Platform

This script checks the health of all services in the forex trading platform.
It verifies that all services are running and responding correctly.

Usage:
    python check_platform_health.py [--timeout TIMEOUT] [--services SERVICES]
                                   [--check-db] [--check-kafka] [--check-redis]
                                   [--verbose] [--output-format FORMAT]

Options:
    --timeout TIMEOUT          Request timeout in seconds (default: 5)
    --services SERVICES        Comma-separated list of services to check (default: all)
    --check-db                 Check database connectivity
    --check-kafka              Check Kafka connectivity
    --check-redis              Check Redis connectivity
    --verbose                  Enable verbose output
    --output-format FORMAT     Output format (text, json, csv) (default: text)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("check_platform_health")

# Service health endpoints
SERVICE_HEALTH_ENDPOINTS = {
    "data-pipeline-service": {
        "url": "http://localhost:8001/health",
        "dependencies": ["database"],
    },
    "feature-store-service": {
        "url": "http://localhost:8002/health",
        "dependencies": ["database", "data-pipeline-service"],
    },
    "analysis-engine-service": {
        "url": "http://localhost:8003/health",
        "dependencies": ["database", "feature-store-service"],
    },
    "ml-integration-service": {
        "url": "http://localhost:8004/health",
        "dependencies": ["database", "analysis-engine-service", "feature-store-service"],
    },
    "trading-gateway-service": {
        "url": "http://localhost:8005/health",
        "dependencies": ["database", "analysis-engine-service", "portfolio-management-service"],
    },
    "portfolio-management-service": {
        "url": "http://localhost:8006/health",
        "dependencies": ["database", "data-pipeline-service"],
    },
    "monitoring-alerting-service": {
        "url": "http://localhost:8007/health",
        "dependencies": ["database"],
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Check platform health")
    parser.add_argument("--timeout", type=int, default=5,
                        help="Request timeout in seconds")
    parser.add_argument("--services", type=str,
                        help="Comma-separated list of services to check")
    parser.add_argument("--check-db", action="store_true",
                        help="Check database connectivity")
    parser.add_argument("--check-kafka", action="store_true",
                        help="Check Kafka connectivity")
    parser.add_argument("--check-redis", action="store_true",
                        help="Check Redis connectivity")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--output-format", type=str, default="text",
                        choices=["text", "json", "csv"],
                        help="Output format (text, json, csv)")
    return parser.parse_args()


def check_service_health(service: str, endpoint: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    """
    Check the health of a service.
    
    Args:
        service: Service name
        endpoint: Service health endpoint configuration
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with health check results
    """
    url = endpoint["url"]
    dependencies = endpoint.get("dependencies", [])
    
    result = {
        "service": service,
        "url": url,
        "status": "unknown",
        "response_time": 0,
        "dependencies": dependencies,
        "details": {},
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        end_time = time.time()
        
        result["response_time"] = round((end_time - start_time) * 1000, 2)  # in milliseconds
        
        if response.status_code == 200:
            result["status"] = "healthy"
            
            # Parse response
            try:
                data = response.json()
                result["details"] = data
            except json.JSONDecodeError:
                result["details"] = {"raw_response": response.text}
        else:
            result["status"] = "unhealthy"
            result["details"] = {
                "status_code": response.status_code,
                "reason": response.reason,
                "text": response.text,
            }
    except requests.RequestException as e:
        result["status"] = "unreachable"
        result["details"] = {"error": str(e)}
    
    return result


def check_database_connectivity() -> Dict[str, Any]:
    """
    Check database connectivity.
    
    Returns:
        Dictionary with database connectivity check results
    """
    result = {
        "component": "database",
        "status": "unknown",
        "details": {},
        "timestamp": datetime.now().isoformat(),
    }
    
    # Use data-pipeline-service health endpoint to check database connectivity
    try:
        response = requests.get("http://localhost:8001/health/database", timeout=5)
        
        if response.status_code == 200:
            result["status"] = "healthy"
            
            # Parse response
            try:
                data = response.json()
                result["details"] = data
            except json.JSONDecodeError:
                result["details"] = {"raw_response": response.text}
        else:
            result["status"] = "unhealthy"
            result["details"] = {
                "status_code": response.status_code,
                "reason": response.reason,
                "text": response.text,
            }
    except requests.RequestException as e:
        result["status"] = "unreachable"
        result["details"] = {"error": str(e)}
    
    return result


def check_kafka_connectivity() -> Dict[str, Any]:
    """
    Check Kafka connectivity.
    
    Returns:
        Dictionary with Kafka connectivity check results
    """
    result = {
        "component": "kafka",
        "status": "unknown",
        "details": {},
        "timestamp": datetime.now().isoformat(),
    }
    
    # Use data-pipeline-service health endpoint to check Kafka connectivity
    try:
        response = requests.get("http://localhost:8001/health/kafka", timeout=5)
        
        if response.status_code == 200:
            result["status"] = "healthy"
            
            # Parse response
            try:
                data = response.json()
                result["details"] = data
            except json.JSONDecodeError:
                result["details"] = {"raw_response": response.text}
        else:
            result["status"] = "unhealthy"
            result["details"] = {
                "status_code": response.status_code,
                "reason": response.reason,
                "text": response.text,
            }
    except requests.RequestException as e:
        result["status"] = "unreachable"
        result["details"] = {"error": str(e)}
    
    return result


def check_redis_connectivity() -> Dict[str, Any]:
    """
    Check Redis connectivity.
    
    Returns:
        Dictionary with Redis connectivity check results
    """
    result = {
        "component": "redis",
        "status": "unknown",
        "details": {},
        "timestamp": datetime.now().isoformat(),
    }
    
    # Use feature-store-service health endpoint to check Redis connectivity
    try:
        response = requests.get("http://localhost:8002/health/redis", timeout=5)
        
        if response.status_code == 200:
            result["status"] = "healthy"
            
            # Parse response
            try:
                data = response.json()
                result["details"] = data
            except json.JSONDecodeError:
                result["details"] = {"raw_response": response.text}
        else:
            result["status"] = "unhealthy"
            result["details"] = {
                "status_code": response.status_code,
                "reason": response.reason,
                "text": response.text,
            }
    except requests.RequestException as e:
        result["status"] = "unreachable"
        result["details"] = {"error": str(e)}
    
    return result


def format_output(results: Dict[str, Any], output_format: str) -> str:
    """
    Format health check results.
    
    Args:
        results: Health check results
        output_format: Output format (text, json, csv)
        
    Returns:
        Formatted output
    """
    if output_format == "json":
        return json.dumps(results, indent=2)
    elif output_format == "csv":
        csv_lines = ["component,status,response_time,details"]
        
        # Add service results
        for service_result in results["services"]:
            csv_lines.append(f"{service_result['service']},{service_result['status']},{service_result['response_time']},{json.dumps(service_result['details'])}")
        
        # Add component results
        for component in ["database", "kafka", "redis"]:
            if component in results:
                csv_lines.append(f"{component},{results[component]['status']},0,{json.dumps(results[component]['details'])}")
        
        return "\n".join(csv_lines)
    else:  # text
        lines = []
        
        # Add summary
        lines.append("=== Forex Trading Platform Health Check ===")
        lines.append(f"Timestamp: {datetime.now().isoformat()}")
        lines.append(f"Overall Status: {results['overall_status']}")
        lines.append("")
        
        # Add service results
        lines.append("=== Services ===")
        for service_result in results["services"]:
            status_color = {
                "healthy": "\033[92m",  # green
                "unhealthy": "\033[91m",  # red
                "unreachable": "\033[91m",  # red
                "unknown": "\033[93m",  # yellow
            }.get(service_result["status"], "")
            reset_color = "\033[0m"
            
            lines.append(f"Service: {service_result['service']}")
            lines.append(f"  URL: {service_result['url']}")
            lines.append(f"  Status: {status_color}{service_result['status']}{reset_color}")
            lines.append(f"  Response Time: {service_result['response_time']} ms")
            
            if service_result["dependencies"]:
                lines.append(f"  Dependencies: {', '.join(service_result['dependencies'])}")
            
            if service_result["details"]:
                lines.append(f"  Details: {json.dumps(service_result['details'], indent=2)}")
            
            lines.append("")
        
        # Add component results
        for component in ["database", "kafka", "redis"]:
            if component in results:
                status_color = {
                    "healthy": "\033[92m",  # green
                    "unhealthy": "\033[91m",  # red
                    "unreachable": "\033[91m",  # red
                    "unknown": "\033[93m",  # yellow
                }.get(results[component]["status"], "")
                reset_color = "\033[0m"
                
                lines.append(f"=== {component.capitalize()} ===")
                lines.append(f"  Status: {status_color}{results[component]['status']}{reset_color}")
                
                if results[component]["details"]:
                    lines.append(f"  Details: {json.dumps(results[component]['details'], indent=2)}")
                
                lines.append("")
        
        return "\n".join(lines)


def main():
    """Main function."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Determine which services to check
    if args.services:
        services_to_check = args.services.split(",")
        # Validate services
        for service in services_to_check:
            if service not in SERVICE_HEALTH_ENDPOINTS:
                logger.error(f"Unknown service: {service}")
                return 1
    else:
        services_to_check = list(SERVICE_HEALTH_ENDPOINTS.keys())
    
    logger.info(f"Checking health of services: {services_to_check}")
    
    # Check service health
    service_results = []
    for service in services_to_check:
        logger.info(f"Checking health of service: {service}")
        result = check_service_health(service, SERVICE_HEALTH_ENDPOINTS[service], args.timeout)
        service_results.append(result)
        logger.info(f"Service {service} status: {result['status']}")
    
    # Check component health
    results = {
        "services": service_results,
        "timestamp": datetime.now().isoformat(),
    }
    
    if args.check_db:
        logger.info("Checking database connectivity")
        results["database"] = check_database_connectivity()
        logger.info(f"Database status: {results['database']['status']}")
    
    if args.check_kafka:
        logger.info("Checking Kafka connectivity")
        results["kafka"] = check_kafka_connectivity()
        logger.info(f"Kafka status: {results['kafka']['status']}")
    
    if args.check_redis:
        logger.info("Checking Redis connectivity")
        results["redis"] = check_redis_connectivity()
        logger.info(f"Redis status: {results['redis']['status']}")
    
    # Determine overall status
    service_statuses = [result["status"] for result in service_results]
    component_statuses = []
    
    if args.check_db:
        component_statuses.append(results["database"]["status"])
    
    if args.check_kafka:
        component_statuses.append(results["kafka"]["status"])
    
    if args.check_redis:
        component_statuses.append(results["redis"]["status"])
    
    all_statuses = service_statuses + component_statuses
    
    if "unreachable" in all_statuses:
        overall_status = "unreachable"
    elif "unhealthy" in all_statuses:
        overall_status = "unhealthy"
    elif "unknown" in all_statuses:
        overall_status = "unknown"
    else:
        overall_status = "healthy"
    
    results["overall_status"] = overall_status
    
    # Output results
    output = format_output(results, args.output_format)
    print(output)
    
    # Return exit code based on overall status
    if overall_status == "healthy":
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
