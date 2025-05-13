#!/usr/bin/env python3
"""
Forex Trading Platform Service Structure Improvement

This script implements the service structure recommendations from the optimization report.
It focuses on standardizing service structure, implementing clear separation of concerns,
and reducing coupling between services.

Usage:
python improve_service_structure.py [--project-root <project_root>] [--service <service_name>]
"""

import os
import sys
import re
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"

# Standard directory structure
STANDARD_DIRECTORIES = [
    "api",
    "config",
    "core",
    "models",
    "repositories",
    "services",
    "utils",
    "adapters",
    "interfaces",
    "tests"
]

# Standard files
STANDARD_FILES = {
    "config/__init__.py": """\"\"\"
Configuration module for the service.
\"\"\"

from .config import get_service_config, get_database_config
""",
    "config/config.py": """\"\"\"
Configuration management for the service.
\"\"\"

import os
from common_lib.config import ConfigManager
from .config_schema import ServiceConfig

# Create a singleton ConfigManager instance
config_manager = ConfigManager(
    config_path=os.environ.get("CONFIG_PATH", "config/config.yaml"),
    service_specific_model=ServiceConfig,
    env_prefix=os.environ.get("CONFIG_ENV_PREFIX", "SERVICE_"),
    default_config_path=os.environ.get("DEFAULT_CONFIG_PATH", "config/default/config.yaml")
)


# Helper functions to access configuration
def get_service_config() -> ServiceConfig:
    \"\"\"
    Get the service-specific configuration.

    Returns:
        Service-specific configuration
    \"\"\"
    return config_manager.get_service_specific_config()


def get_database_config():
    \"\"\"
    Get the database configuration.

    Returns:
        Database configuration
    \"\"\"
    return config_manager.get_database_config()
""",
    "config/config_schema.py": """\"\"\"
Configuration schema for the service.
\"\"\"

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class ServiceConfig(BaseModel):
    \"\"\"
    Service-specific configuration.
    \"\"\"

    name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    debug: bool = Field(False, description="Debug mode")
    log_level: str = Field("INFO", description="Logging level")
""",
    "core/__init__.py": """\"\"\"
Core module for the service.
\"\"\"
""",
    "models/__init__.py": """\"\"\"
Data models for the service.
\"\"\"
""",
    "services/__init__.py": """\"\"\"
Service implementations.
\"\"\"
""",
    "utils/__init__.py": """\"\"\"
Utility functions for the service.
\"\"\"
""",
    "adapters/__init__.py": """\"\"\"
Adapters for external services.
\"\"\"
""",
    "interfaces/__init__.py": """\"\"\"
Interface definitions for the service.
\"\"\"
""",
    "api/__init__.py": """\"\"\"
API endpoints for the service.
\"\"\"
""",
    "api/routes.py": """\"\"\"
API route definitions.
\"\"\"

from fastapi import APIRouter, Depends, FastAPI

def setup_routes(app: FastAPI):
    \"\"\"
    Set up API routes.

    Args:
        app: FastAPI application
    \"\"\"
    # Create routers
    main_router = APIRouter()

    # Add routes to main router

    # Include routers in app
    app.include_router(main_router, prefix="/api")
""",
    "repositories/__init__.py": """\"\"\"
Data repositories for the service.
\"\"\"
""",
    "tests/__init__.py": """\"\"\"
Tests for the service.
\"\"\"
""",
    "tests/conftest.py": """\"\"\"
Test fixtures for the service.
\"\"\"

import pytest
"""
}

class ServiceStructureImprover:
    """Improves service structure based on recommendations."""

    def __init__(self, project_root: str, service_name: str = None):
        """
        Initialize the improver.

        Args:
            project_root: Root directory of the project
            service_name: Name of the service to improve (None for all services)
        """
        self.project_root = project_root
        self.service_name = service_name
        self.services = []
        self.improvements = []

    def identify_services(self) -> None:
        """Identify services in the project based on directory structure."""
        logger.info("Identifying services...")

        # Look for service directories
        for item in os.listdir(self.project_root):
            item_path = os.path.join(self.project_root, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it's likely a service
                if (
                    item.endswith('-service') or
                    item.endswith('_service') or
                    item.endswith('-api') or
                    item.endswith('-engine') or
                    'service' in item.lower() or
                    'api' in item.lower()
                ):
                    self.services.append(item)

        logger.info(f"Identified {len(self.services)} services")

    def standardize_service_structure(self, service_name: str) -> List[str]:
        """
        Standardize the structure of a service.

        Args:
            service_name: Name of the service to standardize

        Returns:
            List of improvements made
        """
        logger.info(f"Standardizing structure for {service_name}...")

        service_path = os.path.join(self.project_root, service_name)
        improvements = []

        # Check if service directory exists
        if not os.path.exists(service_path):
            logger.error(f"Service directory not found: {service_path}")
            return improvements

        # Create standard directories
        for directory in STANDARD_DIRECTORIES:
            directory_path = os.path.join(service_path, directory)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path, exist_ok=True)
                improvements.append(f"Created directory: {directory_path}")

        # Create standard files
        for file_path, content in STANDARD_FILES.items():
            full_path = os.path.join(service_path, file_path)

            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Only create file if it doesn't exist
            if not os.path.exists(full_path):
                with open(full_path, 'w', encoding='utf-8') as f:
                    # Replace SERVICE with actual service name
                    service_env_prefix = service_name.replace('-', '_').upper()
                    updated_content = content.replace("SERVICE_", f"{service_env_prefix}_")

                    f.write(updated_content)

                improvements.append(f"Created file: {full_path}")

        return improvements

    def improve_service_structure(self) -> List[str]:
        """
        Improve service structure based on recommendations.

        Returns:
            List of improvements made
        """
        logger.info("Starting service structure improvements...")

        # Identify services
        self.identify_services()

        if not self.services:
            logger.info("No services found")
            return []

        # Filter services if a specific service was specified
        if self.service_name:
            if self.service_name in self.services:
                services_to_improve = [self.service_name]
            else:
                logger.error(f"Service not found: {self.service_name}")
                return []
        else:
            services_to_improve = self.services

        # Improve each service
        for service in services_to_improve:
            improvements = self.standardize_service_structure(service)
            self.improvements.extend(improvements)

        logger.info("Service structure improvements complete")
        return self.improvements

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Improve service structure")
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--service",
        help="Name of the service to improve (default: all services)"
    )
    args = parser.parse_args()

    # Improve service structure
    improver = ServiceStructureImprover(args.project_root, args.service)
    improvements = improver.improve_service_structure()

    # Print summary
    print("\nService Structure Improvement Summary:")
    print(f"- Applied {len(improvements)} improvements")

    if improvements:
        print("\nImprovements:")
        for i, improvement in enumerate(improvements):
            print(f"  {i+1}. {improvement}")

    # Save results to file
    output_dir = os.path.join(args.project_root, 'tools', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'service_structure_improvements.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_improvements': len(improvements),
            'improvements': improvements
        }, f, indent=2)

    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
