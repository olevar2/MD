#!/usr/bin/env python
"""
Create the necessary directories for the Historical Data Management service.

This script creates the necessary directories for the service.
"""

import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create directories
    dirs = [
        os.path.join(base_dir, "data_management_service"),
        os.path.join(base_dir, "data_management_service", "historical"),
        os.path.join(base_dir, "tests"),
        os.path.join(base_dir, "scripts"),
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            logger.info(f"Creating directory: {dir_path}")
            os.makedirs(dir_path)
        else:
            logger.info(f"Directory already exists: {dir_path}")
    
    # Create __init__.py files
    init_files = [
        os.path.join(base_dir, "data_management_service", "__init__.py"),
        os.path.join(base_dir, "data_management_service", "historical", "__init__.py"),
        os.path.join(base_dir, "tests", "__init__.py"),
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            logger.info(f"Creating file: {init_file}")
            with open(init_file, "w") as f:
                f.write('"""Module initialization."""\n')
        else:
            logger.info(f"File already exists: {init_file}")
    
    logger.info("Directory creation complete")


if __name__ == "__main__":
    main()
