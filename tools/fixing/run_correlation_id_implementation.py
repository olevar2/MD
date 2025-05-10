#!/usr/bin/env python
"""
Run Correlation ID Implementation

This script runs the correlation ID implementation for all services.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_correlation_id_implementation")

# Repository root directory
REPO_ROOT = Path(__file__).parent.parent.parent.absolute()


def run_implementation():
    """Run the correlation ID implementation."""
    logger.info("Running correlation ID implementation...")

    # Run the implementation script
    implementation_script = REPO_ROOT / "tools" / "fixing" / "implement_correlation_id.py"

    try:
        result = subprocess.run(
            [sys.executable, str(implementation_script), "--all"],
            check=True,
            capture_output=True,
            text=True
        )

        logger.info(f"Implementation output:\n{result.stdout}")

        if result.stderr:
            logger.warning(f"Implementation warnings:\n{result.stderr}")

        logger.info("Correlation ID implementation completed successfully.")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Implementation failed with exit code {e.returncode}")
        logger.error(f"Output:\n{e.stdout}")
        logger.error(f"Error:\n{e.stderr}")
        return False


def run_tests():
    """Run the correlation ID tests."""
    logger.info("Running correlation ID tests...")

    # Run the tests
    test_dir = REPO_ROOT / "common-lib" / "tests"

    try:
        # Run unit tests
        logger.info("Running unit tests...")
        unit_test_result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_dir / "correlation")],
            check=True,
            capture_output=True,
            text=True
        )

        logger.info(f"Unit test output:\n{unit_test_result.stdout}")

        if unit_test_result.stderr:
            logger.warning(f"Unit test warnings:\n{unit_test_result.stderr}")

        # Skip integration tests as they require external dependencies
        logger.info("Skipping integration tests as they require external dependencies.")

        logger.info("Correlation ID tests completed successfully.")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed with exit code {e.returncode}")
        logger.error(f"Output:\n{e.stdout}")
        logger.error(f"Error:\n{e.stderr}")
        return False


def main():
    """Main function."""
    logger.info("Starting correlation ID implementation and testing...")

    # Run the implementation
    if not run_implementation():
        logger.error("Implementation failed.")
        return 1

    # Run the tests
    if not run_tests():
        logger.error("Tests failed.")
        return 1

    logger.info("Correlation ID implementation and testing completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
