#!/usr/bin/env python3
"""
Deployment script for the Enhanced API Gateway.

This script deploys the Enhanced API Gateway by:
1. Copying the enhanced API Gateway files to the appropriate locations
2. Setting up the environment variables
3. Starting the API Gateway
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from typing import Dict, Any, Optional, List


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deploy_api_gateway")


class APIGatewayDeployer:
    """
    Deployer for the Enhanced API Gateway.
    """

    def __init__(self, source_dir: str, target_dir: str, config_file: str, env_file: str):
        """
        Initialize the deployer.

        Args:
            source_dir: Source directory containing the enhanced API Gateway files
            target_dir: Target directory to deploy the API Gateway to
            config_file: Path to the API Gateway configuration file
            env_file: Path to the environment variables file
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.config_file = config_file
        self.env_file = env_file

    def copy_files(self) -> bool:
        """
        Copy the enhanced API Gateway files to the target directory.

        Returns:
            True if the files were copied successfully, False otherwise
        """
        logger.info(f"Copying files from {self.source_dir} to {self.target_dir}...")

        try:
            # Create target directory if it doesn't exist
            os.makedirs(self.target_dir, exist_ok=True)

            # Copy files
            for root, dirs, files in os.walk(self.source_dir):
                # Get relative path
                rel_path = os.path.relpath(root, self.source_dir)
                if rel_path == ".":
                    rel_path = ""

                # Create target directory
                target_path = os.path.join(self.target_dir, rel_path)
                os.makedirs(target_path, exist_ok=True)

                # Copy files
                for file in files:
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_path, file)
                    shutil.copy2(source_file, target_file)

            logger.info("Files copied successfully")
            return True
        except Exception as e:
            logger.error(f"Error copying files: {str(e)}")
            return False

    def setup_environment(self) -> bool:
        """
        Set up the environment variables.

        Returns:
            True if the environment variables were set up successfully, False otherwise
        """
        logger.info(f"Setting up environment variables from {self.env_file}...")

        try:
            # Check if environment file exists
            if not os.path.isfile(self.env_file):
                logger.error(f"Environment file {self.env_file} not found")
                return False

            # Read environment variables
            env_vars = {}
            with open(self.env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    key, value = line.split("=", 1)
                    env_vars[key] = value

            # Set environment variables
            for key, value in env_vars.items():
                os.environ[key] = value

            logger.info("Environment variables set up successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting up environment variables: {str(e)}")
            return False

    def start_api_gateway(self) -> bool:
        """
        Start the API Gateway.

        Returns:
            True if the API Gateway was started successfully, False otherwise
        """
        logger.info("Starting API Gateway...")

        try:
            # Check if config file exists
            if not os.path.isfile(os.path.join(self.target_dir, self.config_file)):
                logger.error(f"Config file {self.config_file} not found in {self.target_dir}")
                return False

            # Start API Gateway
            cmd = [
                "uvicorn",
                "api.app_enhanced:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ]
            process = subprocess.Popen(
                cmd,
                cwd=self.target_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Wait for API Gateway to start
            logger.info("API Gateway starting...")
            logger.info("Press Ctrl+C to stop")

            # Print output
            while True:
                output = process.stdout.readline()
                if output:
                    logger.info(output.strip())
                else:
                    break

            return True
        except Exception as e:
            logger.error(f"Error starting API Gateway: {str(e)}")
            return False

    def deploy(self) -> bool:
        """
        Deploy the API Gateway.

        Returns:
            True if the API Gateway was deployed successfully, False otherwise
        """
        # Copy files
        if not self.copy_files():
            return False

        # Set up environment
        if not self.setup_environment():
            return False

        # Start API Gateway
        if not self.start_api_gateway():
            return False

        return True


def main():
    """
    Main function.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Deploy the Enhanced API Gateway")
    parser.add_argument("--source-dir", default="api-gateway", help="Source directory containing the enhanced API Gateway files")
    parser.add_argument("--target-dir", default="deployed/api-gateway", help="Target directory to deploy the API Gateway to")
    parser.add_argument("--config-file", default="config/api-gateway-enhanced.yaml", help="Path to the API Gateway configuration file")
    parser.add_argument("--env-file", default=".env", help="Path to the environment variables file")
    args = parser.parse_args()

    # Create deployer
    deployer = APIGatewayDeployer(args.source_dir, args.target_dir, args.config_file, args.env_file)

    # Deploy
    success = deployer.deploy()

    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()