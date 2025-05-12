#!/usr/bin/env python
"""
Environment File Generator for Forex Trading Platform

This script generates .env files for all services based on the .env.example files
and a configuration file for the target environment.

Usage:
    python generate_env_files.py [--env ENV] [--config CONFIG_FILE] [--output-dir OUTPUT_DIR]

Options:
    --env ENV                 Target environment (development, testing, production)
    --config CONFIG_FILE      Path to the environment configuration file
    --output-dir OUTPUT_DIR   Directory to write the generated .env files
"""

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("generate_env_files")

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

# Default environment configurations
DEFAULT_CONFIGS = {
    "development": {
        "database": {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "password": "",
        },
        "kafka": {
            "bootstrap_servers": "localhost:9092",
        },
        "services": {
            "data_pipeline": {
                "host": "localhost",
                "port": 8001,
            },
            "feature_store": {
                "host": "localhost",
                "port": 8002,
            },
            "analysis_engine": {
                "host": "localhost",
                "port": 8003,
            },
            "ml_integration": {
                "host": "localhost",
                "port": 8004,
            },
            "trading_gateway": {
                "host": "localhost",
                "port": 8005,
            },
            "portfolio_management": {
                "host": "localhost",
                "port": 8006,
            },
            "monitoring_alerting": {
                "host": "localhost",
                "port": 8007,
            },
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "features": {
            "enable_gpu_acceleration": "false",
            "enable_distributed_computing": "false",
            "enable_advanced_indicators": "true",
        },
    },
    "testing": {
        "database": {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "password": "",
        },
        "kafka": {
            "bootstrap_servers": "localhost:9092",
        },
        "services": {
            "data_pipeline": {
                "host": "localhost",
                "port": 8001,
            },
            "feature_store": {
                "host": "localhost",
                "port": 8002,
            },
            "analysis_engine": {
                "host": "localhost",
                "port": 8003,
            },
            "ml_integration": {
                "host": "localhost",
                "port": 8004,
            },
            "trading_gateway": {
                "host": "localhost",
                "port": 8005,
            },
            "portfolio_management": {
                "host": "localhost",
                "port": 8006,
            },
            "monitoring_alerting": {
                "host": "localhost",
                "port": 8007,
            },
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "features": {
            "enable_gpu_acceleration": "false",
            "enable_distributed_computing": "false",
            "enable_advanced_indicators": "true",
        },
    },
    "production": {
        "database": {
            "host": "db.example.com",
            "port": 5432,
            "user": "forex_user",
            "password": "secure_password",
        },
        "redis": {
            "host": "redis.example.com",
            "port": 6379,
            "password": "secure_password",
        },
        "kafka": {
            "bootstrap_servers": "kafka.example.com:9092",
        },
        "services": {
            "data_pipeline": {
                "host": "data-pipeline.example.com",
                "port": 8001,
            },
            "feature_store": {
                "host": "feature-store.example.com",
                "port": 8002,
            },
            "analysis_engine": {
                "host": "analysis-engine.example.com",
                "port": 8003,
            },
            "ml_integration": {
                "host": "ml-integration.example.com",
                "port": 8004,
            },
            "trading_gateway": {
                "host": "trading-gateway.example.com",
                "port": 8005,
            },
            "portfolio_management": {
                "host": "portfolio-management.example.com",
                "port": 8006,
            },
            "monitoring_alerting": {
                "host": "monitoring-alerting.example.com",
                "port": 8007,
            },
        },
        "logging": {
            "level": "WARNING",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "features": {
            "enable_gpu_acceleration": "true",
            "enable_distributed_computing": "true",
            "enable_advanced_indicators": "true",
        },
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate environment files")
    parser.add_argument("--env", type=str, default="development", choices=["development", "testing", "production"],
                        help="Target environment")
    parser.add_argument("--config", type=str, help="Path to the environment configuration file")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to write the generated .env files")
    return parser.parse_args()


def load_config(config_file: str, env: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_file: Path to the configuration file
        env: Target environment
        
    Returns:
        Configuration dictionary
    """
    if not config_file:
        logger.info(f"Using default configuration for environment: {env}")
        return DEFAULT_CONFIGS[env]
    
    try:
        with open(config_file, "r") as f:
            if config_file.endswith(".json"):
                config = json.load(f)
            elif config_file.endswith((".yaml", ".yml")):
                config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_file}")
                return DEFAULT_CONFIGS[env]
            
            # Get configuration for the specified environment
            if env in config:
                return config[env]
            else:
                logger.warning(f"Environment {env} not found in configuration file, using default")
                return DEFAULT_CONFIGS[env]
    except Exception as e:
        logger.error(f"Error loading configuration file {config_file}: {e}")
        return DEFAULT_CONFIGS[env]


def parse_env_example(example_file: str) -> Dict[str, str]:
    """
    Parse a .env.example file and return a dictionary of environment variables.
    
    Args:
        example_file: Path to the .env.example file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    
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
                    value = value.strip()
                    
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    env_vars[key] = value
    except FileNotFoundError:
        logger.warning(f"Example environment file not found: {example_file}")
    except Exception as e:
        logger.error(f"Error parsing example environment file {example_file}: {e}")
    
    return env_vars


def generate_env_file(service: str, config: Dict[str, Any], output_dir: str) -> bool:
    """
    Generate a .env file for a service.
    
    Args:
        service: Service name
        config: Configuration dictionary
        output_dir: Output directory
        
    Returns:
        True if successful, False otherwise
    """
    example_file = Path(service) / ".env.example"
    
    if not example_file.exists():
        logger.warning(f"No .env.example file found for service: {service}")
        return False
    
    # Parse example file
    env_vars = parse_env_example(str(example_file))
    
    # Update environment variables with configuration values
    service_key = service.replace("-", "_").replace("service", "").strip("_")
    
    # Database configuration
    if "database" in config:
        env_vars["DB_HOST"] = config["database"]["host"]
        env_vars["DB_PORT"] = str(config["database"]["port"])
        env_vars["DB_USER"] = config["database"]["user"]
        env_vars["DB_PASSWORD"] = config["database"]["password"]
        env_vars["DB_NAME"] = f"{service_key}_db"
    
    # Redis configuration
    if "redis" in config:
        env_vars["REDIS_HOST"] = config["redis"]["host"]
        env_vars["REDIS_PORT"] = str(config["redis"]["port"])
        env_vars["REDIS_PASSWORD"] = config["redis"]["password"]
    
    # Kafka configuration
    if "kafka" in config:
        env_vars["KAFKA_BOOTSTRAP_SERVERS"] = config["kafka"]["bootstrap_servers"]
    
    # Service configuration
    if "services" in config and service_key in config["services"]:
        env_vars["HOST"] = config["services"][service_key]["host"]
        env_vars["PORT"] = str(config["services"][service_key]["port"])
    
    # Logging configuration
    if "logging" in config:
        env_vars["LOG_LEVEL"] = config["logging"]["level"]
        env_vars["LOG_FORMAT"] = config["logging"]["format"]
    
    # Feature flags
    if "features" in config:
        for key, value in config["features"].items():
            env_key = f"FEATURE_{key.upper()}"
            env_vars[env_key] = value
    
    # Write .env file
    output_file = Path(output_dir) / service / ".env"
    os.makedirs(output_file.parent, exist_ok=True)
    
    try:
        with open(output_file, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"Generated .env file for {service}: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error writing .env file for {service}: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args.env)
    
    # Generate .env files for each service
    success_count = 0
    for service in SERVICES:
        if generate_env_file(service, config, args.output_dir):
            success_count += 1
    
    # Print summary
    logger.info(f"Generated {success_count} out of {len(SERVICES)} .env files")
    
    return 0 if success_count == len(SERVICES) else 1


if __name__ == "__main__":
    sys.exit(main())
