#!/usr/bin/env python
"""
Environment Configuration Validation Script for Forex Trading Platform

This script validates that all required environment variables are set for each service.
It reads the .env.example files for each service and checks if the corresponding
environment variables are set in the current environment or in the .env file.

Usage:
    python validate_env_config.py [--service SERVICE] [--env-file ENV_FILE]

Options:
    --service SERVICE     Validate only the specified service
    --env-file ENV_FILE   Path to the .env file to validate against
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("validate_env_config")

# Service directories
SERVICES = [
    "data-pipeline-service",
    "feature-store-service",
    "analysis-engine-service",
    "ml-integration-service",
    "trading-gateway-service",
    "portfolio-management-service",
    "monitoring-alerting-service",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate environment configuration")
    parser.add_argument("--service", type=str, help="Validate only the specified service")
    parser.add_argument("--env-file", type=str, help="Path to the .env file to validate against")
    return parser.parse_args()


def parse_env_file(file_path: str) -> Dict[str, str]:
    """
    Parse a .env file and return a dictionary of environment variables.
    
    Args:
        file_path: Path to the .env file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    
    try:
        with open(file_path, "r") as f:
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
        logger.warning(f"Environment file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error parsing environment file {file_path}: {e}")
    
    return env_vars


def extract_required_vars(example_file: str) -> Set[str]:
    """
    Extract required environment variables from a .env.example file.
    
    Args:
        example_file: Path to the .env.example file
        
    Returns:
        Set of required environment variable names
    """
    required_vars = set()
    
    try:
        with open(example_file, "r") as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                
                # Parse variable assignment
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    
                    # Check if variable is marked as required
                    # We consider a variable required if it doesn't have a default value
                    # or if it's explicitly marked as required in a comment
                    prev_line = ""
                    if "required" in prev_line.lower() or not value.strip():
                        required_vars.add(key)
                    
                    prev_line = line
    except FileNotFoundError:
        logger.warning(f"Example environment file not found: {example_file}")
    except Exception as e:
        logger.error(f"Error parsing example environment file {example_file}: {e}")
    
    return required_vars


def validate_service_env(service: str, env_vars: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Validate environment variables for a service.
    
    Args:
        service: Service name
        env_vars: Dictionary of environment variables
        
    Returns:
        Tuple of (missing variables, optional variables)
    """
    example_file = Path(service) / ".env.example"
    
    if not example_file.exists():
        logger.warning(f"No .env.example file found for service: {service}")
        return [], []
    
    required_vars = extract_required_vars(str(example_file))
    
    # Check which required variables are missing
    missing_vars = []
    for var in required_vars:
        if var not in env_vars:
            missing_vars.append(var)
    
    # Check which optional variables are missing
    all_vars = set()
    with open(example_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _ = line.split("=", 1)
                all_vars.add(key.strip())
    
    optional_vars = all_vars - required_vars
    missing_optional_vars = [var for var in optional_vars if var not in env_vars]
    
    return missing_vars, missing_optional_vars


def main():
    """Main function."""
    args = parse_args()
    
    # Get environment variables
    env_vars = dict(os.environ)
    
    # Add variables from .env file if specified
    if args.env_file:
        env_file_vars = parse_env_file(args.env_file)
        env_vars.update(env_file_vars)
    
    # Determine which services to validate
    services_to_validate = [args.service] if args.service else SERVICES
    
    # Validate each service
    all_missing_vars = []
    all_missing_optional_vars = []
    
    for service in services_to_validate:
        logger.info(f"Validating environment configuration for {service}...")
        
        missing_vars, missing_optional_vars = validate_service_env(service, env_vars)
        
        if missing_vars:
            logger.error(f"Missing required environment variables for {service}: {', '.join(missing_vars)}")
            all_missing_vars.extend(missing_vars)
        else:
            logger.info(f"All required environment variables are set for {service}")
        
        if missing_optional_vars:
            logger.warning(f"Missing optional environment variables for {service}: {', '.join(missing_optional_vars)}")
            all_missing_optional_vars.extend(missing_optional_vars)
    
    # Print summary
    if all_missing_vars:
        logger.error(f"Total missing required environment variables: {len(all_missing_vars)}")
        return 1
    else:
        logger.info("All required environment variables are set for all services")
        return 0


if __name__ == "__main__":
    sys.exit(main())
