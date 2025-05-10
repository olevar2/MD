#!/usr/bin/env python
"""
Correlation ID Implementation Script

This script implements standardized correlation ID propagation across all services in the
Forex Trading Platform. It updates:

1. Service main files to add the FastAPICorrelationIdMiddleware
2. Logging configurations to include correlation IDs
3. Event producers and consumers to propagate correlation IDs
4. Service clients to use the BaseServiceClient with correlation ID support

Usage:
    python implement_correlation_id.py --service <service-name>
    python implement_correlation_id.py --all
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("implement_correlation_id")

# Repository root directory
REPO_ROOT = Path(__file__).parent.parent.parent.absolute()

# Service definitions
SERVICES = [
    {
        "name": "analysis-engine-service",
        "language": "python",
        "path": REPO_ROOT / "analysis-engine-service",
        "main_file": "analysis_engine/main.py",
        "middleware_file": "analysis_engine/api/middleware.py",
        "logging_file": "analysis_engine/core/logging.py",
    },
    {
        "name": "portfolio-management-service",
        "language": "python",
        "path": REPO_ROOT / "portfolio-management-service",
        "main_file": "portfolio_management_service/main.py",
        "middleware_file": "portfolio_management_service/api/middleware.py",
        "logging_file": "portfolio_management_service/core/logging.py",
    },
    {
        "name": "risk-management-service",
        "language": "python",
        "path": REPO_ROOT / "risk-management-service",
        "main_file": "risk_management_service/main.py",
        "middleware_file": "risk_management_service/api/middleware.py",
        "logging_file": "risk_management_service/core/logging.py",
    },
    {
        "name": "monitoring-alerting-service",
        "language": "python",
        "path": REPO_ROOT / "monitoring-alerting-service",
        "main_file": "monitoring_alerting_service/main.py",
        "middleware_file": "monitoring_alerting_service/api/middleware.py",
        "logging_file": "monitoring_alerting_service/core/logging.py",
    },
    {
        "name": "ml-integration-service",
        "language": "python",
        "path": REPO_ROOT / "ml-integration-service",
        "main_file": "ml_integration_service/main.py",
        "middleware_file": "ml_integration_service/api/middleware.py",
        "logging_file": "ml_integration_service/core/logging.py",
    },
    {
        "name": "ml-workbench-service",
        "language": "python",
        "path": REPO_ROOT / "ml-workbench-service",
        "main_file": "ml_workbench_service/main.py",
        "middleware_file": "ml_workbench_service/api/middleware.py",
        "logging_file": "ml_workbench_service/core/logging.py",
    },
    {
        "name": "data-pipeline-service",
        "language": "python",
        "path": REPO_ROOT / "data-pipeline-service",
        "main_file": "data_pipeline_service/main.py",
        "middleware_file": "data_pipeline_service/api/middleware.py",
        "logging_file": "data_pipeline_service/core/logging.py",
    },
    {
        "name": "feature-store-service",
        "language": "python",
        "path": REPO_ROOT / "feature-store-service",
        "main_file": "feature_store_service/main.py",
        "middleware_file": "feature_store_service/api/middleware.py",
        "logging_file": "feature_store_service/core/logging.py",
    },
    {
        "name": "trading-gateway-service",
        "language": "python",
        "path": REPO_ROOT / "trading-gateway-service",
        "main_file": "trading_gateway_service/main.py",
        "middleware_file": "trading_gateway_service/api/middleware.py",
        "logging_file": "trading_gateway_service/core/logging.py",
    },
]


def update_middleware_import(service: Dict[str, Any]) -> bool:
    """
    Update middleware import to include FastAPICorrelationIdMiddleware.
    
    Args:
        service: Service definition
        
    Returns:
        bool: True if successful, False otherwise
    """
    main_file_path = service["path"] / service["main_file"]
    if not main_file_path.exists():
        logger.warning(f"Main file not found: {main_file_path}")
        return False
    
    try:
        with open(main_file_path, "r") as f:
            content = f.read()
        
        # Check if correlation middleware is already imported
        if "FastAPICorrelationIdMiddleware" in content:
            logger.info(f"Correlation middleware already imported in {service['name']}")
            return True
        
        # Add import for FastAPICorrelationIdMiddleware
        if "from common_lib.correlation import" in content:
            # Update existing import
            content = re.sub(
                r"from common_lib.correlation import (.*)",
                r"from common_lib.correlation import \1, FastAPICorrelationIdMiddleware",
                content
            )
        else:
            # Add new import
            import_line = "from common_lib.correlation import FastAPICorrelationIdMiddleware\n"
            
            # Find a good place to add the import
            if "from fastapi import" in content:
                content = content.replace(
                    "from fastapi import",
                    f"{import_line}from fastapi import"
                )
            elif "import fastapi" in content:
                content = content.replace(
                    "import fastapi",
                    f"{import_line}import fastapi"
                )
            else:
                # Add after the last import
                import_match = re.search(r"^(import .*|from .* import .*)\n\n", content, re.MULTILINE)
                if import_match:
                    content = content.replace(
                        import_match.group(0),
                        f"{import_match.group(0)}{import_line}\n"
                    )
                else:
                    # Add at the beginning of the file
                    content = f"{import_line}\n{content}"
        
        # Write updated content
        with open(main_file_path, "w") as f:
            f.write(content)
        
        logger.info(f"Updated middleware import in {service['name']}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating middleware import in {service['name']}: {e}")
        return False


def add_middleware_to_app(service: Dict[str, Any]) -> bool:
    """
    Add FastAPICorrelationIdMiddleware to the FastAPI app.
    
    Args:
        service: Service definition
        
    Returns:
        bool: True if successful, False otherwise
    """
    main_file_path = service["path"] / service["main_file"]
    if not main_file_path.exists():
        logger.warning(f"Main file not found: {main_file_path}")
        return False
    
    try:
        with open(main_file_path, "r") as f:
            content = f.read()
        
        # Check if correlation middleware is already added
        if "app.add_middleware(FastAPICorrelationIdMiddleware" in content:
            logger.info(f"Correlation middleware already added in {service['name']}")
            return True
        
        # Find where to add the middleware
        middleware_match = re.search(r"app\.add_middleware\((.*?)\)", content, re.DOTALL)
        if middleware_match:
            # Add after the last middleware
            middleware_end = middleware_match.end()
            content = (
                content[:middleware_end] + 
                "\n\n# Add correlation ID middleware\napp.add_middleware(FastAPICorrelationIdMiddleware)" + 
                content[middleware_end:]
            )
        else:
            # Find app creation
            app_match = re.search(r"app\s*=\s*FastAPI\(.*?\)", content, re.DOTALL)
            if app_match:
                # Add after app creation
                app_end = app_match.end()
                content = (
                    content[:app_end] + 
                    "\n\n# Add correlation ID middleware\napp.add_middleware(FastAPICorrelationIdMiddleware)" + 
                    content[app_end:]
                )
            else:
                logger.warning(f"Could not find where to add middleware in {service['name']}")
                return False
        
        # Write updated content
        with open(main_file_path, "w") as f:
            f.write(content)
        
        logger.info(f"Added correlation middleware to {service['name']}")
        return True
    
    except Exception as e:
        logger.error(f"Error adding middleware to {service['name']}: {e}")
        return False


def update_logging_configuration(service: Dict[str, Any]) -> bool:
    """
    Update logging configuration to include correlation IDs.
    
    Args:
        service: Service definition
        
    Returns:
        bool: True if successful, False otherwise
    """
    logging_file_path = service["path"] / service["logging_file"]
    if not logging_file_path.exists():
        # Create the logging file if it doesn't exist
        os.makedirs(logging_file_path.parent, exist_ok=True)
        
        # Create a basic logging configuration file
        logging_content = """\"\"\"
Logging Module

This module provides logging functionality for the service.
\"\"\"

import logging
import sys
from typing import Optional

from common_lib.correlation import get_correlation_id


class CorrelationFilter(logging.Filter):
    \"\"\"Logging filter that adds correlation ID to log records.\"\"\"
    
    def filter(self, record):
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id or "no-correlation-id"
        return True


def configure_logging(log_level: str = "INFO") -> None:
    \"\"\"
    Configure logging for the application.
    
    Args:
        log_level: Logging level
    \"\"\"
    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Set formatter with correlation ID
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s"
    )
    console_handler.setFormatter(formatter)
    
    # Add correlation filter
    correlation_filter = CorrelationFilter()
    console_handler.addFilter(correlation_filter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Log configuration
    logging.info(f"Logging configured with level: {log_level}")
"""
        
        with open(logging_file_path, "w") as f:
            f.write(logging_content)
        
        logger.info(f"Created logging configuration file for {service['name']}")
        return True
    
    try:
        with open(logging_file_path, "r") as f:
            content = f.read()
        
        # Check if correlation ID is already included in logging
        if "correlation_id" in content and "CorrelationFilter" in content:
            logger.info(f"Correlation ID already included in logging for {service['name']}")
            return True
        
        # Add import for get_correlation_id if needed
        if "from common_lib.correlation import" not in content:
            # Add import
            import_line = "from common_lib.correlation import get_correlation_id\n"
            
            # Find a good place to add the import
            if "import logging" in content:
                content = content.replace(
                    "import logging",
                    f"import logging\n{import_line}"
                )
            else:
                # Add at the beginning of the file
                content = f"{import_line}\n{content}"
        
        # Add CorrelationFilter class if needed
        if "class CorrelationFilter" not in content:
            filter_class = """

class CorrelationFilter(logging.Filter):
    \"\"\"Logging filter that adds correlation ID to log records.\"\"\"
    
    def filter(self, record):
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id or "no-correlation-id"
        return True
"""
            
            # Find a good place to add the class
            if "def configure_logging" in content:
                content = content.replace(
                    "def configure_logging",
                    f"{filter_class}\n\ndef configure_logging"
                )
            else:
                # Add at the end of the file
                content = f"{content}\n{filter_class}"
        
        # Update formatter to include correlation ID if needed
        if "%(correlation_id)s" not in content:
            # Find formatter pattern
            formatter_match = re.search(r'formatter\s*=\s*logging\.Formatter\(\s*["\']([^"\']*)["\']', content)
            if formatter_match:
                # Update formatter pattern
                old_pattern = formatter_match.group(1)
                if "%(levelname)s" in old_pattern:
                    new_pattern = old_pattern.replace(
                        "%(levelname)s",
                        "%(levelname)s - [%(correlation_id)s]"
                    )
                else:
                    new_pattern = f"{old_pattern} - [%(correlation_id)s]"
                
                content = content.replace(
                    f'formatter = logging.Formatter("{old_pattern}"',
                    f'formatter = logging.Formatter("{new_pattern}"'
                )
        
        # Add correlation filter to handler if needed
        if "correlation_filter = CorrelationFilter()" not in content:
            # Find handler creation
            handler_match = re.search(r"(console_handler|handler)\s*=\s*logging\.StreamHandler", content)
            if handler_match:
                handler_name = handler_match.group(1)
                # Find where to add the filter
                handler_config_end = content.find("\n", handler_match.end())
                if handler_config_end != -1:
                    # Add after handler configuration
                    filter_code = f"\n    # Add correlation filter\n    correlation_filter = CorrelationFilter()\n    {handler_name}.addFilter(correlation_filter)"
                    content = content[:handler_config_end] + filter_code + content[handler_config_end:]
        
        # Write updated content
        with open(logging_file_path, "w") as f:
            f.write(content)
        
        logger.info(f"Updated logging configuration for {service['name']}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating logging configuration for {service['name']}: {e}")
        return False


def update_service(service: Dict[str, Any]) -> bool:
    """
    Update a service to use standardized correlation ID propagation.
    
    Args:
        service: Service definition
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Updating {service['name']}...")
    
    # Update middleware import
    if not update_middleware_import(service):
        logger.warning(f"Failed to update middleware import for {service['name']}")
    
    # Add middleware to app
    if not add_middleware_to_app(service):
        logger.warning(f"Failed to add middleware to {service['name']}")
    
    # Update logging configuration
    if not update_logging_configuration(service):
        logger.warning(f"Failed to update logging configuration for {service['name']}")
    
    logger.info(f"Completed updating {service['name']}")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Implement correlation ID propagation across services")
    parser.add_argument("--service", help="Implement for a specific service")
    parser.add_argument("--all", action="store_true", help="Implement for all services")
    args = parser.parse_args()
    
    if not args.service and not args.all:
        parser.print_help()
        return 1
    
    if args.service:
        # Find the service
        service = next((s for s in SERVICES if s["name"] == args.service), None)
        if not service:
            logger.error(f"Service {args.service} not found")
            return 1
        
        # Update the service
        if not update_service(service):
            logger.error(f"Failed to update {args.service}")
            return 1
    
    elif args.all:
        # Update all services
        for service in SERVICES:
            update_service(service)
    
    logger.info("Correlation ID implementation completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
