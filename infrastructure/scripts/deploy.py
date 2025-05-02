#!/usr/bin/env python3
"""
Deployment script for the Forex Trading Platform services.
This script handles deployment of microservices to different environments.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
import yaml
import json
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Service definitions - which services to deploy and their dependencies
SERVICES = {
    "core-foundations": {
        "path": "core-foundations",
        "dependencies": [],
        "deployment_order": 1
    },
    "data-pipeline-service": {
        "path": "data-pipeline-service",
        "dependencies": ["core-foundations"],
        "deployment_order": 2
    },
    "feature-store-service": {
        "path": "feature-store-service",
        "dependencies": ["core-foundations", "data-pipeline-service"],
        "deployment_order": 3
    },
    "analysis-engine-service": {
        "path": "analysis-engine-service",
        "dependencies": ["core-foundations", "feature-store-service"],
        "deployment_order": 4
    },
    "strategy-execution-engine": {
        "path": "strategy-execution-engine",
        "dependencies": ["core-foundations", "analysis-engine-service"],
        "deployment_order": 5
    },
    "portfolio-management-service": {
        "path": "portfolio-management-service",
        "dependencies": ["core-foundations", "strategy-execution-engine"],
        "deployment_order": 6
    },
    "risk-management-service": {
        "path": "risk-management-service",
        "dependencies": ["core-foundations", "portfolio-management-service"],
        "deployment_order": 7
    },
    "trading-gateway-service": {
        "path": "trading-gateway-service",
        "dependencies": ["core-foundations", "portfolio-management-service", "risk-management-service"],
        "deployment_order": 8
    },
    "monitoring-alerting-service": {
        "path": "monitoring-alerting-service",
        "dependencies": [],
        "deployment_order": 9
    },
    "ml-workbench-service": {
        "path": "ml-workbench-service",
        "dependencies": ["core-foundations", "feature-store-service"],
        "deployment_order": 10
    },
    "ui-service": {
        "path": "ui-service",
        "dependencies": [],
        "deployment_order": 11
    }
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deploy Forex Trading Platform services')
    
    parser.add_argument('--environment', type=str, required=True,
                        choices=['dev', 'staging', 'production'],
                        help='Environment to deploy to')
    
    parser.add_argument('--version', type=str, required=True,
                        help='Version or commit SHA to deploy')
    
    parser.add_argument('--services', type=str, nargs='+',
                        help='List of services to deploy. If not specified, all services will be deployed')
    
    parser.add_argument('--skip-tests', action='store_true',
                        help='Skip running tests before deployment')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Print deployment commands without executing them')
    
    parser.add_argument('--notify', action='store_true',
                        help='Send notifications about deployment')
    
    return parser.parse_args()


def get_service_env_vars(service: str, environment: str) -> Dict[str, str]:
    """Get environment variables for a service based on the environment."""
    # This would typically read from a secure parameter store or secrets manager
    # For this example, we're just simulating the behavior
    
    env_vars = {
        "ENVIRONMENT": environment,
        "LOG_LEVEL": "INFO" if environment == "production" else "DEBUG",
    }
    
    if environment == "production":
        env_vars["ENABLE_FEATURE_X"] = "false"
        env_vars["AUTO_SCALING"] = "true"
    elif environment == "staging":
        env_vars["ENABLE_FEATURE_X"] = "true"
        env_vars["AUTO_SCALING"] = "true"
        env_vars["USE_MOCK_DATA"] = "false"
    else:  # dev
        env_vars["ENABLE_FEATURE_X"] = "true"
        env_vars["AUTO_SCALING"] = "false"
        env_vars["USE_MOCK_DATA"] = "true"
    
    # Service-specific variables
    if service == "data-pipeline-service":
        env_vars["DATA_RETENTION_DAYS"] = "30" if environment == "production" else "7"
    elif service == "trading-gateway-service":
        env_vars["TRADING_MODE"] = "LIVE" if environment == "production" else "PAPER"
    
    return env_vars


def update_kubernetes_manifests(service: str, environment: str, version: str) -> None:
    """Update Kubernetes manifests for a service with the new version."""
    logger.info(f"Updating Kubernetes manifests for {service} in {environment}")
    
    manifest_dir = os.path.join("infrastructure", "kubernetes", environment)
    manifest_path = os.path.join(manifest_dir, f"{service}-deployment.yaml")
    
    # Ensure the manifest directory exists
    if not os.path.exists(manifest_dir):
        os.makedirs(manifest_dir)
    
    # If the manifest doesn't exist, create a template
    if not os.path.exists(manifest_path):
        create_k8s_manifest_template(manifest_path, service, environment, version)
    else:
        # Update the existing manifest
        with open(manifest_path, 'r') as file:
            manifest = yaml.safe_load(file)
        
        # Update container image version
        for container in manifest['spec']['template']['spec']['containers']:
            if container['name'] == service:
                container['image'] = f"{service}:{version}"
        
        # Update environment variables
        env_vars = get_service_env_vars(service, environment)
        for container in manifest['spec']['template']['spec']['containers']:
            if container['name'] == service:
                container['env'] = [{"name": k, "value": v} for k, v in env_vars.items()]
        
        # Write updated manifest
        with open(manifest_path, 'w') as file:
            yaml.dump(manifest, file)


def create_k8s_manifest_template(path: str, service: str, environment: str, version: str) -> None:
    """Create a template Kubernetes manifest for a service."""
    env_vars = get_service_env_vars(service, environment)
    
    manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{service}",
            "namespace": environment
        },
        "spec": {
            "replicas": 3 if environment == "production" else 1,
            "selector": {
                "matchLabels": {
                    "app": service
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": service
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": service,
                            "image": f"{service}:{version}",
                            "ports": [
                                {
                                    "containerPort": 8000
                                }
                            ],
                            "resources": {
                                "limits": {
                                    "cpu": "1000m",
                                    "memory": "1Gi"
                                },
                                "requests": {
                                    "cpu": "500m",
                                    "memory": "512Mi"
                                }
                            },
                            "env": [{"name": k, "value": v} for k, v in env_vars.items()],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 10
                            }
                        }
                    ]
                }
            }
        }
    }
    
    # Write the manifest
    with open(path, 'w') as file:
        yaml.dump(manifest, file)


def deploy_service(service: str, environment: str, version: str, dry_run: bool) -> bool:
    """Deploy a service to the specified environment."""
    logger.info(f"Deploying {service} to {environment} with version {version}")
    
    # Update Kubernetes manifests with new version
    update_kubernetes_manifests(service, environment, version)
    
    # Deploy to Kubernetes
    cmd = f"kubectl apply -f infrastructure/kubernetes/{environment}/{service}-deployment.yaml"
    
    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {cmd}")
        return True
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Successfully deployed {service} to {environment}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to deploy {service} to {environment}: {e}")
        return False


def run_tests_for_service(service: str, environment: str, dry_run: bool) -> bool:
    """Run tests for a specific service against the environment."""
    logger.info(f"Running tests for {service} against {environment} environment")
    
    test_cmd = f"cd {service} && python -m pytest tests/ -v --environment {environment}"
    
    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {test_cmd}")
        return True
    
    try:
        subprocess.run(test_cmd, shell=True, check=True)
        logger.info(f"All tests passed for {service}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed for {service}: {e}")
        return False


def rollback_service(service: str, environment: str, previous_version: str, dry_run: bool) -> bool:
    """Rollback a service to the previous version."""
    logger.warning(f"Rolling back {service} to version {previous_version} in {environment}")
    
    rollback_cmd = f"kubectl rollout undo deployment/{service} -n {environment}"
    
    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {rollback_cmd}")
        return True
    
    try:
        subprocess.run(rollback_cmd, shell=True, check=True)
        logger.info(f"Successfully rolled back {service} in {environment}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to rollback {service} in {environment}: {e}")
        return False


def notify_deployment_status(environment: str, services: List[str], status: str, version: str) -> None:
    """Send a notification about deployment status."""
    message = {
        "environment": environment,
        "services": services,
        "status": status,
        "version": version,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    logger.info(f"Notification: {json.dumps(message)}")
    # In a real implementation, this would send to a notification service:
    # - Slack webhook
    # - Email notification
    # - Microsoft Teams webhook
    # - SMS alert for critical environments


def seed_test_data(environment: str, dry_run: bool) -> bool:
    """Seed test data into the environment if needed."""
    if environment not in ["staging"]:
        logger.info(f"Skipping test data seeding for {environment} environment")
        return True
    
    logger.info(f"Seeding test data into {environment} environment")
    
    seed_cmd = f"python infrastructure/scripts/seed_data.py --environment {environment}"
    
    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {seed_cmd}")
        return True
    
    try:
        subprocess.run(seed_cmd, shell=True, check=True)
        logger.info(f"Successfully seeded test data into {environment}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to seed test data into {environment}: {e}")
        return False


def deploy_all_services(args: argparse.Namespace) -> None:
    """Deploy all services in the correct order."""
    environment = args.environment
    version = args.version
    dry_run = args.dry_run
    skip_tests = args.skip_tests
    services_to_deploy = args.services or sorted(SERVICES.keys(), key=lambda s: SERVICES[s]["deployment_order"])
    
    # Create Kubernetes namespace if it doesn't exist
    if not dry_run:
        subprocess.run(f"kubectl get namespace {environment} || kubectl create namespace {environment}", 
                      shell=True, check=False)
    
    # If we're deploying to staging, seed test data
    if environment == "staging":
        if not seed_test_data(environment, dry_run):
            logger.error("Failed to seed test data, aborting deployment")
            return
    
    # Track deployed services for notification
    deployed_services = []
    failed_services = []
    
    # Deploy each service in order
    for service in services_to_deploy:
        if service not in SERVICES:
            logger.warning(f"Unknown service: {service}, skipping")
            continue
        
        # Check if dependencies have been deployed first
        dependencies_ok = True
        for dependency in SERVICES[service]["dependencies"]:
            if dependency in services_to_deploy and dependency not in deployed_services:
                logger.warning(f"Dependency {dependency} for {service} has not been deployed yet")
                dependencies_ok = False
        
        if not dependencies_ok:
            logger.error(f"Cannot deploy {service} due to unmet dependencies")
            failed_services.append(service)
            continue
        
        # Run tests if not skipped
        if not skip_tests:
            if not run_tests_for_service(service, environment, dry_run):
                logger.error(f"Tests failed for {service}, skipping deployment")
                failed_services.append(service)
                continue
        
        # Deploy the service
        if deploy_service(service, environment, version, dry_run):
            deployed_services.append(service)
        else:
            failed_services.append(service)
    
    # Send notification
    if args.notify:
        if failed_services:
            status = f"PARTIAL - {len(deployed_services)} succeeded, {len(failed_services)} failed"
        else:
            status = "SUCCESS"
        
        notify_deployment_status(environment, deployed_services, status, version)
    
    # Summarize deployment
    logger.info("=== Deployment Summary ===")
    logger.info(f"Environment: {environment}")
    logger.info(f"Version: {version}")
    logger.info(f"Successfully deployed: {', '.join(deployed_services) if deployed_services else 'None'}")
    
    if failed_services:
        logger.error(f"Failed to deploy: {', '.join(failed_services)}")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting deployment to {args.environment} environment")
    deploy_all_services(args)
    logger.info("Deployment completed")
