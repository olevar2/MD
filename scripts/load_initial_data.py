#!/usr/bin/env python
"""
Initial Data Loading Script for Forex Trading Platform

This script loads initial data into the forex trading platform databases.
It orchestrates the loading of various data types from sample files.

Usage:
    python load_initial_data.py [--data-dir DATA_DIR] [--skip-types SKIP_TYPES]
                               [--only-types ONLY_TYPES] [--verify] [--verbose]

Options:
    --data-dir DATA_DIR        Directory containing sample data files (default: data/sample)
    --skip-types SKIP_TYPES    Comma-separated list of data types to skip
    --only-types ONLY_TYPES    Comma-separated list of data types to load (overrides skip-types)
    --verify                   Verify data after loading
    --verbose                  Enable verbose output
"""

import argparse
import importlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("load_initial_data")

# Data types and their loaders
DATA_LOADERS = {
    "symbols": "symbol_loader",
    "historical": "historical_data_loader",
    "indicators": "indicator_loader",
    "alternative": "alternative_data_loader",
    "accounts": "account_loader",
    "models": "model_loader",
}

# Data loading order
DATA_LOADING_ORDER = [
    "symbols",
    "historical",
    "indicators",
    "alternative",
    "accounts",
    "models",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load initial data into the forex trading platform")
    parser.add_argument("--data-dir", type=str, default="data/sample",
                        help="Directory containing sample data files")
    parser.add_argument("--skip-types", type=str,
                        help="Comma-separated list of data types to skip")
    parser.add_argument("--only-types", type=str,
                        help="Comma-separated list of data types to load (overrides skip-types)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify data after loading")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    return parser.parse_args()


def load_data_type(data_type: str, data_dir: str, verify: bool, verbose: bool) -> bool:
    """
    Load a specific data type.
    
    Args:
        data_type: Data type to load
        data_dir: Directory containing sample data files
        verify: Verify data after loading
        verbose: Enable verbose output
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Loading {data_type} data...")
    
    # Get loader module
    loader_module_name = DATA_LOADERS.get(data_type)
    if not loader_module_name:
        logger.error(f"No loader found for data type: {data_type}")
        return False
    
    # Import loader module
    try:
        loader_module = importlib.import_module(f"data_loaders.{loader_module_name}")
    except ImportError as e:
        logger.error(f"Failed to import loader module for {data_type}: {e}")
        return False
    
    # Load data
    try:
        result = loader_module.load_data(data_dir, verify=verify, verbose=verbose)
        
        if result:
            logger.info(f"{data_type} data loaded successfully")
            return True
        else:
            logger.error(f"Failed to load {data_type} data")
            return False
    except Exception as e:
        logger.error(f"Error loading {data_type} data: {e}")
        return False


def verify_data_consistency() -> bool:
    """
    Verify data consistency across services.
    
    Returns:
        True if data is consistent, False otherwise
    """
    logger.info("Verifying data consistency across services...")
    
    # TODO: Implement data consistency verification
    # This would involve checking that data is consistent across different services
    # For example, ensuring that symbols in the data pipeline service match those in the feature store service
    
    logger.info("Data consistency verification not implemented yet")
    return True


def main():
    """Main function."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Determine which data types to load
    if args.only_types:
        data_types_to_load = args.only_types.split(",")
        # Validate data types
        for data_type in data_types_to_load:
            if data_type not in DATA_LOADERS:
                logger.error(f"Unknown data type: {data_type}")
                return 1
    elif args.skip_types:
        skip_types = args.skip_types.split(",")
        data_types_to_load = [dt for dt in DATA_LOADING_ORDER if dt not in skip_types]
    else:
        data_types_to_load = DATA_LOADING_ORDER
    
    logger.info(f"Loading data types: {data_types_to_load}")
    
    # Check if data directory exists
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Add scripts directory to Python path
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, scripts_dir)
    
    # Load data types in order
    success = True
    for data_type in DATA_LOADING_ORDER:
        if data_type not in data_types_to_load:
            continue
        
        if not load_data_type(data_type, data_dir, args.verify, args.verbose):
            logger.error(f"Failed to load {data_type} data")
            success = False
    
    # Verify data consistency
    if args.verify and success:
        if not verify_data_consistency():
            logger.error("Data consistency verification failed")
            success = False
    
    if success:
        logger.info("All data loaded successfully")
        return 0
    else:
        logger.error("Failed to load all data")
        return 1


if __name__ == "__main__":
    sys.exit(main())
