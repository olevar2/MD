#!/usr/bin/env python
"""
Make scripts executable.

This script makes the scripts in the scripts directory executable.
"""

import logging
import os
import stat
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    # Get the scripts directory
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all Python scripts
    scripts = [
        os.path.join(scripts_dir, f)
        for f in os.listdir(scripts_dir)
        if f.endswith(".py")
    ]
    
    # Make scripts executable
    for script in scripts:
        logger.info(f"Making executable: {script}")
        
        # Get current permissions
        current_mode = os.stat(script).st_mode
        
        # Add execute permission
        os.chmod(script, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    
    logger.info("Script permissions updated")


if __name__ == "__main__":
    main()
