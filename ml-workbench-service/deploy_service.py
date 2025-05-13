#!/usr/bin/env python
"""
Deployment script for ML Workbench Service.

This script deploys the ML Workbench Service to the specified environment.
"""

import os
import sys
import argparse
import subprocess
import logging
import json
import time
import requests
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("deploy_ml_workbench_service")

# Service configuration
SERVICE_NAME = "ML Workbench Service"
SERVICE_PACKAGE = "ml_workbench_service"
SERVICE_MODULE = "ml_workbench_service.main"
SERVICE_PORT = 8030
SERVICE_HOST = "0.0.0.0"

# Environment configurations
ENVIRONMENTS = {
    "development": {
        "host": "localhost",
        "port": SERVICE_PORT,
        "log_level": "DEBUG",
        "environment": "development",
        "debug": True,
    },
    "testing": {
        "host": "localhost",
        "port": SERVICE_PORT,
        "log_level": "INFO",
        "environment": "testing",
        "debug": False,
    },
    "staging": {
        "host": SERVICE_HOST,
        "port": SERVICE_PORT,
        "log_level": "INFO",
        "environment": "staging",
        "debug": False,
    },
    "production": {
        "host": SERVICE_HOST,
        "port": SERVICE_PORT,
        "log_level": "WARNING",
        "environment": "production",
        "debug": False,
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=f"Deploy {SERVICE_NAME}")
    parser.add_argument(
        "--environment",
        "-e",
        choices=ENVIRONMENTS.keys(),
        default="development",
        help="Deployment environment",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Port to run the service on (overrides environment config)",
    )
    parser.add_argument(
        "--host",
        "-H",
        help="Host to run the service on (overrides environment config)",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (overrides environment config)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode (overrides environment config)",
    )
    parser.add_argument(
        "--no-debug",
        "-n",
        action="store_true",
        help="Disable debug mode (overrides environment config)",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run tests before deployment",
    )
    parser.add_argument(
        "--skip-dependencies",
        "-s",
        action="store_true",
        help="Skip installing dependencies",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Deploy using Docker",
    )
    parser.add_argument(
        "--kubernetes",
        "-k",
        action="store_true",
        help="Deploy to Kubernetes",
    )
    parser.add_argument(
        "--namespace",
        default="default",
        help="Kubernetes namespace",
    )
    return parser.parse_args()


def install_dependencies():
    """Install dependencies."""
    logger.info("Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )
    logger.info("Dependencies installed successfully")


def run_tests():
    """Run tests."""
    logger.info("Running tests...")
    result = subprocess.run(
        [sys.executable, "test_service.py"],
        check=False,
    )
    if result.returncode != 0:
        logger.error("Tests failed")
        sys.exit(1)
    logger.info("Tests passed successfully")


def deploy_local(config: Dict[str, Any]):
    """
    Deploy the service locally.

    Args:
        config: Deployment configuration
    """
    logger.info(f"Deploying {SERVICE_NAME} locally...")
    
    # Set environment variables
    env = os.environ.copy()
    env["HOST"] = config["host"]
    env["PORT"] = str(config["port"])
    env["LOG_LEVEL"] = config["log_level"]
    env["ENVIRONMENT"] = config["environment"]
    
    # Start the service
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        SERVICE_MODULE + ":app",
        "--host",
        config["host"],
        "--port",
        str(config["port"]),
        "--log-level",
        config["log_level"].lower(),
    ]
    
    if config["debug"]:
        cmd.append("--reload")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)


def deploy_docker(config: Dict[str, Any]):
    """
    Deploy the service using Docker.

    Args:
        config: Deployment configuration
    """
    logger.info(f"Deploying {SERVICE_NAME} using Docker...")
    
    # Build Docker image
    image_name = f"{SERVICE_PACKAGE}:{config['environment']}"
    build_cmd = [
        "docker",
        "build",
        "-t",
        image_name,
        ".",
    ]
    logger.info(f"Building Docker image: {' '.join(build_cmd)}")
    subprocess.run(build_cmd, check=True)
    
    # Run Docker container
    container_name = f"{SERVICE_PACKAGE}-{config['environment']}"
    run_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{config['port']}:{config['port']}",
        "-e",
        f"HOST={config['host']}",
        "-e",
        f"PORT={config['port']}",
        "-e",
        f"LOG_LEVEL={config['log_level']}",
        "-e",
        f"ENVIRONMENT={config['environment']}",
        image_name,
    ]
    logger.info(f"Running Docker container: {' '.join(run_cmd)}")
    subprocess.run(run_cmd, check=True)
    
    logger.info(f"Docker container {container_name} started successfully")


def deploy_kubernetes(config: Dict[str, Any], namespace: str):
    """
    Deploy the service to Kubernetes.

    Args:
        config: Deployment configuration
        namespace: Kubernetes namespace
    """
    logger.info(f"Deploying {SERVICE_NAME} to Kubernetes...")
    
    # Build Docker image
    image_name = f"{SERVICE_PACKAGE}:{config['environment']}"
    build_cmd = [
        "docker",
        "build",
        "-t",
        image_name,
        ".",
    ]
    logger.info(f"Building Docker image: {' '.join(build_cmd)}")
    subprocess.run(build_cmd, check=True)
    
    # Create Kubernetes deployment YAML
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": SERVICE_PACKAGE,
            "namespace": namespace,
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": SERVICE_PACKAGE,
                },
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": SERVICE_PACKAGE,
                    },
                },
                "spec": {
                    "containers": [
                        {
                            "name": SERVICE_PACKAGE,
                            "image": image_name,
                            "ports": [
                                {
                                    "containerPort": config["port"],
                                },
                            ],
                            "env": [
                                {
                                    "name": "HOST",
                                    "value": config["host"],
                                },
                                {
                                    "name": "PORT",
                                    "value": str(config["port"]),
                                },
                                {
                                    "name": "LOG_LEVEL",
                                    "value": config["log_level"],
                                },
                                {
                                    "name": "ENVIRONMENT",
                                    "value": config["environment"],
                                },
                            ],
                            "resources": {
                                "limits": {
                                    "cpu": "500m",
                                    "memory": "512Mi",
                                },
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "128Mi",
                                },
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": config["port"],
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": config["port"],
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                            },
                        },
                    ],
                },
            },
        },
    }
    
    # Create Kubernetes service YAML
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": SERVICE_PACKAGE,
            "namespace": namespace,
        },
        "spec": {
            "selector": {
                "app": SERVICE_PACKAGE,
            },
            "ports": [
                {
                    "port": config["port"],
                    "targetPort": config["port"],
                },
            ],
            "type": "ClusterIP",
        },
    }
    
    # Write deployment YAML to file
    with open("deployment.yaml", "w") as f:
        json.dump(deployment, f, indent=2)
    
    # Write service YAML to file
    with open("service.yaml", "w") as f:
        json.dump(service, f, indent=2)
    
    # Apply deployment YAML
    apply_cmd = [
        "kubectl",
        "apply",
        "-f",
        "deployment.yaml",
    ]
    logger.info(f"Applying Kubernetes deployment: {' '.join(apply_cmd)}")
    subprocess.run(apply_cmd, check=True)
    
    # Apply service YAML
    apply_cmd = [
        "kubectl",
        "apply",
        "-f",
        "service.yaml",
    ]
    logger.info(f"Applying Kubernetes service: {' '.join(apply_cmd)}")
    subprocess.run(apply_cmd, check=True)
    
    logger.info(f"Kubernetes deployment and service created successfully")


def main():
    """Main function."""
    args = parse_args()
    
    # Get environment configuration
    config = ENVIRONMENTS[args.environment].copy()
    
    # Override configuration with command line arguments
    if args.port:
        config["port"] = args.port
    if args.host:
        config["host"] = args.host
    if args.log_level:
        config["log_level"] = args.log_level
    if args.debug:
        config["debug"] = True
    if args.no_debug:
        config["debug"] = False
    
    logger.info(f"Deploying {SERVICE_NAME} to {args.environment} environment")
    logger.info(f"Configuration: {config}")
    
    # Install dependencies
    if not args.skip_dependencies:
        install_dependencies()
    
    # Run tests
    if args.test:
        run_tests()
    
    # Deploy the service
    if args.kubernetes:
        deploy_kubernetes(config, args.namespace)
    elif args.docker:
        deploy_docker(config)
    else:
        deploy_local(config)


if __name__ == "__main__":
    main()