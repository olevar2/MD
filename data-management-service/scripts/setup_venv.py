#!/usr/bin/env python
"""
Set up a virtual environment and install the required packages.

This script creates a new virtual environment and installs the required packages.
"""

import argparse
import logging
import os
import subprocess
import sys
import venv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Set up a virtual environment and install the required packages"
    )
    
    parser.add_argument(
        "--venv-dir",
        type=str,
        default=".venv",
        help="Virtual environment directory (default: .venv)"
    )
    
    parser.add_argument(
        "--requirements",
        type=str,
        default="requirements.txt",
        help="Requirements file (default: requirements.txt)"
    )
    
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade packages (default: False)"
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


def create_venv(venv_dir):
    """
    Create a virtual environment.
    
    Args:
        venv_dir: Virtual environment directory
    """
    logger.info(f"Creating virtual environment in {venv_dir}")
    
    # Create virtual environment
    venv.create(venv_dir, with_pip=True)


def install_packages(venv_dir, requirements, upgrade):
    """
    Install packages.
    
    Args:
        venv_dir: Virtual environment directory
        requirements: Requirements file
        upgrade: Whether to upgrade packages
    """
    logger.info(f"Installing packages from {requirements}")
    
    # Get the Python executable
    if os.name == "nt":
        python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python = os.path.join(venv_dir, "bin", "python")
    
    # Install packages
    cmd = [python, "-m", "pip", "install", "-r", requirements]
    
    if upgrade:
        cmd.append("--upgrade")
    
    run_command(cmd)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Change to the base directory
    os.chdir(base_dir)
    
    # Create virtual environment
    venv_dir = os.path.join(base_dir, args.venv_dir)
    
    if not os.path.exists(venv_dir):
        create_venv(venv_dir)
    else:
        logger.info(f"Virtual environment already exists in {venv_dir}")
    
    # Install packages
    install_packages(venv_dir, args.requirements, args.upgrade)
    
    logger.info("Setup complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
