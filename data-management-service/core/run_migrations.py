#!/usr/bin/env python
"""
Run database migrations.

This script runs database migrations using Alembic.
"""

import argparse
import logging
import os
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run database migrations")
    
    parser.add_argument(
        "--alembic-dir",
        type=str,
        default="alembic",
        help="Alembic directory (default: alembic)"
    )
    
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/forex_platform"
        ),
        help="Database connection URL"
    )
    
    parser.add_argument(
        "--revision",
        type=str,
        default="head",
        help="Revision to migrate to (default: head)"
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


def run_migrations(alembic_dir, db_url, revision):
    """
    Run database migrations.
    
    Args:
        alembic_dir: Alembic directory
        db_url: Database connection URL
        revision: Revision to migrate to
    """
    logger.info(f"Running migrations to {revision}")
    
    # Set environment variables
    env = {"DATABASE_URL": db_url}
    
    # Run migrations
    cmd = [
        sys.executable, "-m", "alembic",
        "-c", os.path.join(alembic_dir, "alembic.ini"),
        "upgrade", revision
    ]
    
    run_command(cmd, env)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Change to the base directory
    os.chdir(base_dir)
    
    # Run migrations
    run_migrations(args.alembic_dir, args.db_url, args.revision)
    
    logger.info("Migrations complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
