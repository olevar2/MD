#!/usr/bin/env python
"""
Run all examples with a database in a Docker container.

This script runs all the example scripts to demonstrate the functionality of the
Historical Data Management service, with a database in a Docker container.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run all examples with a database in a Docker container"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8000)),
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--db-port",
        type=int,
        default=5432,
        help="Database port (default: 5432)"
    )
    
    parser.add_argument(
        "--db-user",
        type=str,
        default="postgres",
        help="Database user (default: postgres)"
    )
    
    parser.add_argument(
        "--db-password",
        type=str,
        default="postgres",
        help="Database password (default: postgres)"
    )
    
    parser.add_argument(
        "--db-name",
        type=str,
        default="forex_platform",
        help="Database name (default: forex_platform)"
    )
    
    parser.add_argument(
        "--db-container",
        type=str,
        default="historical-data-db",
        help="Database container name (default: historical-data-db)"
    )
    
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the database container (default: False)"
    )
    
    parser.add_argument(
        "--examples",
        type=str,
        nargs="+",
        choices=["backtest", "ml", "correction", "quality"],
        default=["backtest", "ml", "correction", "quality"],
        help="Examples to run (default: all)"
    )
    
    return parser.parse_args()


def run_command(cmd: list, env: dict = None) -> int:
    """
    Run a command.
    
    Args:
        cmd: Command to run
        env: Environment variables
        
    Returns:
        Exit code
    """
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Create environment
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)
    
    process = subprocess.Popen(
        cmd,
        env=cmd_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Print output
    for line in process.stdout:
        print(line, end="")
    
    # Wait for process to complete
    process.wait()
    
    # Print errors
    if process.returncode != 0:
        logger.error(f"Command failed with exit code {process.returncode}")
        for line in process.stderr:
            print(line, end="")
    
    return process.returncode


def start_db_container(args):
    """
    Start the database container.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting the database container")
    
    # Check if container exists
    result = subprocess.run(
        ["docker", "ps", "-a", "-q", "-f", f"name={args.db_container}"],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip():
        # Container exists, start it
        logger.info(f"Container {args.db_container} exists, starting it")
        run_command(["docker", "start", args.db_container])
    else:
        # Container doesn't exist, create it
        logger.info(f"Container {args.db_container} doesn't exist, creating it")
        run_command([
            "docker", "run", "-d",
            "--name", args.db_container,
            "-p", f"{args.db_port}:5432",
            "-e", f"POSTGRES_USER={args.db_user}",
            "-e", f"POSTGRES_PASSWORD={args.db_password}",
            "-e", f"POSTGRES_DB={args.db_name}",
            "timescale/timescaledb:latest-pg13"
        ])
    
    # Wait for the database to be ready
    logger.info("Waiting for the database to be ready")
    time.sleep(5)


def stop_db_container(args):
    """
    Stop the database container.
    
    Args:
        args: Command line arguments
    """
    logger.info("Stopping the database container")
    run_command(["docker", "stop", args.db_container])


def init_sample_data(api_url):
    """
    Initialize sample data.
    
    Args:
        api_url: API URL
    """
    logger.info("Initializing sample data")
    
    # Wait for the service to be ready
    logger.info("Waiting for the service to be ready")
    time.sleep(5)
    
    # Initialize sample data
    env = {"API_URL": api_url}
    run_command([sys.executable, "scripts/init_sample_data.py"], env)


def run_examples(args, api_url):
    """
    Run examples.
    
    Args:
        args: Command line arguments
        api_url: API URL
    """
    logger.info("Running examples")
    
    # Set environment variables
    env = {"API_URL": api_url}
    
    # Run examples
    for example in args.examples:
        if example == "backtest":
            logger.info("Running backtest example")
            run_command([sys.executable, "scripts/backtest_example.py"], env)
        
        elif example == "ml":
            logger.info("Running ML example")
            run_command([sys.executable, "scripts/ml_example.py"], env)
        
        elif example == "correction":
            logger.info("Running correction example")
            run_command([sys.executable, "scripts/correction_example.py"], env)
        
        elif example == "quality":
            logger.info("Running quality report example")
            run_command([sys.executable, "scripts/quality_report_example.py"], env)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Change to the base directory
    os.chdir(base_dir)
    
    # Stop the database container
    if args.stop:
        stop_db_container(args)
        return 0
    
    # Start the database container
    start_db_container(args)
    
    # Set environment variables
    db_url = f"postgresql+asyncpg://{args.db_user}:{args.db_password}@localhost:{args.db_port}/{args.db_name}"
    api_url = f"http://localhost:{args.port}"
    
    env = {
        "DATABASE_URL": db_url,
        "PORT": str(args.port)
    }
    
    # Initialize the database
    logger.info("Initializing the database")
    run_command([sys.executable, "scripts/init_db.py"], env)
    
    # Start the service
    logger.info(f"Running the service on {args.host}:{args.port}")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "data_management_service.main:app",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    # Start the service in a separate thread
    service_thread = threading.Thread(
        target=run_command,
        args=(cmd,),
        kwargs={"env": env}
    )
    service_thread.daemon = True
    service_thread.start()
    
    # Initialize sample data
    init_sample_data(api_url)
    
    # Run examples
    run_examples(args, api_url)
    
    # Stop the service
    logger.info("Examples completed, stopping the service")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
