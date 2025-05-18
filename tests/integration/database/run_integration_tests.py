#!/usr/bin/env python
"""
Script to run database integration tests.

This script sets up the test environment and runs all database integration tests.
"""
import os
import sys
import argparse
import subprocess
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run database integration tests")
    parser.add_argument(
        "--test-db-host",
        default="localhost",
        help="Test database host",
    )
    parser.add_argument(
        "--test-db-port",
        type=int,
        default=5432,
        help="Test database port",
    )
    parser.add_argument(
        "--test-db-user",
        default="postgres",
        help="Test database user",
    )
    parser.add_argument(
        "--test-db-password",
        default="postgres",
        help="Test database password",
    )
    parser.add_argument(
        "--test-db-name",
        default="test_forex_platform",
        help="Test database name",
    )
    parser.add_argument(
        "--test-redis-host",
        default="localhost",
        help="Test Redis host",
    )
    parser.add_argument(
        "--test-redis-port",
        type=int,
        default=6379,
        help="Test Redis port",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Use Docker for test environment",
    )
    parser.add_argument(
        "--test-file",
        help="Run a specific test file",
    )
    parser.add_argument(
        "--test-case",
        help="Run a specific test case",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--use-mocks",
        action="store_true",
        default=True,
        help="Use mocks instead of real database connections",
    )
    return parser.parse_args()


def setup_docker_environment():
    """Set up Docker environment for testing."""
    logger.info("Setting up Docker environment...")
    
    # Check if Docker is installed
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker is not installed or not in PATH")
        sys.exit(1)
    
    # Check if Docker Compose is installed
    try:
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker Compose is not installed or not in PATH")
        sys.exit(1)
    
    # Create a docker-compose.yml file for testing
    with open("docker-compose.test.yml", "w") as f:
        f.write("""
version: '3'

services:
  postgres:
    image: timescale/timescaledb:latest-pg13
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: test_forex_platform
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:latest
    ports:
      - "6379:6379"

volumes:
  postgres_data:
""")
    
    # Start the Docker containers
    subprocess.run(
        ["docker-compose", "-f", "docker-compose.test.yml", "up", "-d"],
        check=True,
    )
    
    # Wait for the containers to start
    logger.info("Waiting for containers to start...")
    time.sleep(10)
    
    # Check if the containers are running
    result = subprocess.run(
        ["docker-compose", "-f", "docker-compose.test.yml", "ps"],
        check=True,
        capture_output=True,
        text=True,
    )
    
    if "Up" not in result.stdout:
        logger.error("Containers failed to start")
        sys.exit(1)
    
    logger.info("Docker environment is ready")


def teardown_docker_environment():
    """Tear down Docker environment."""
    logger.info("Tearing down Docker environment...")
    
    # Stop the Docker containers
    subprocess.run(
        ["docker-compose", "-f", "docker-compose.test.yml", "down", "-v"],
        check=True,
    )
    
    # Remove the docker-compose.yml file
    os.remove("docker-compose.test.yml")
    
    logger.info("Docker environment has been torn down")


def run_tests(args):
    """Run the integration tests."""
    logger.info("Running integration tests...")
    
    # Set environment variables for the tests
    env = os.environ.copy()
    env["TEST_DB_HOST"] = args.test_db_host
    env["TEST_DB_PORT"] = str(args.test_db_port)
    env["TEST_DB_USER"] = args.test_db_user
    env["TEST_DB_PASSWORD"] = args.test_db_password
    env["TEST_DB_NAME"] = args.test_db_name
    env["TEST_REDIS_HOST"] = args.test_redis_host
    env["TEST_REDIS_PORT"] = str(args.test_redis_port)
    
    # Enable mocks if requested
    if args.use_mocks:
        logger.info("Using mocks for database connections")
        # Run the enable_mocks.py script
        subprocess.run([sys.executable, "enable_mocks.py"], check=True)
    
    # Build the pytest command
    cmd = ["pytest", "-xvs" if args.verbose else "-v"]
    
    if args.test_file:
        cmd.append(args.test_file)
    else:
        cmd.append(".")
    
    if args.test_case:
        cmd.append(f"-k {args.test_case}")
    
    # Run the tests
    result = subprocess.run(cmd, env=env)
    
    return result.returncode


def main():
    """Main function."""
    args = parse_args()
    
    # Change to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Set up the test environment
        if args.docker and not args.use_mocks:
            setup_docker_environment()
        
        # Run the tests
        exit_code = run_tests(args)
        
        # Exit with the test result
        sys.exit(exit_code)
    finally:
        # Tear down the test environment
        if args.docker and not args.use_mocks:
            teardown_docker_environment()


if __name__ == "__main__":
    main()