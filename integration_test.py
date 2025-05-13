#!/usr/bin/env python
"""
Integration Test Script for Forex Trading Platform.

This script tests the interactions between ML Workbench Service, Monitoring Alerting Service,
Data Pipeline Service, and ML Integration Service.
"""

import os
import sys
import time
import json
import logging
import asyncio
import subprocess
import requests
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("integration_test")

# Service configurations
SERVICES = [
    {
        "name": "ML Workbench Service",
        "host": "localhost",
        "port": 8030,
        "base_url": "http://localhost:8030",
        "api_prefix": "/api/v1",
        "health_endpoint": "/health",
        "deploy_script": "D:/MD/forex_trading_platform/ml-workbench-service/deploy_service.py",
        "process": None,
    },
    {
        "name": "Monitoring Alerting Service",
        "host": "localhost",
        "port": 8009,
        "base_url": "http://localhost:8009",
        "api_prefix": "/api/v1",
        "health_endpoint": "/health",
        "deploy_script": "D:/MD/forex_trading_platform/monitoring-alerting-service/deploy_service.py",
        "process": None,
    },
    {
        "name": "Data Pipeline Service",
        "host": "localhost",
        "port": 8010,
        "base_url": "http://localhost:8010",
        "api_prefix": "/api/v1",
        "health_endpoint": "/health",
        "deploy_script": "D:/MD/forex_trading_platform/data-pipeline-service/deploy_service.py",
        "process": None,
    },
    {
        "name": "ML Integration Service",
        "host": "localhost",
        "port": 8020,
        "base_url": "http://localhost:8020",
        "api_prefix": "/api/v1",
        "health_endpoint": "/health",
        "deploy_script": "D:/MD/forex_trading_platform/ml-integration-service/deploy_service.py",
        "process": None,
    },
]


def start_service(service: Dict[str, Any]) -> subprocess.Popen:
    """
    Start a service.

    Args:
        service: Service configuration

    Returns:
        Service process
    """
    logger.info(f"Starting {service['name']}...")

    # Start the service
    process = subprocess.Popen(
        [
            sys.executable,
            service["deploy_script"],
            "--environment",
            "development",
            "--port",
            str(service["port"]),
            "--host",
            service["host"],
            "--log-level",
            "INFO",
            "--debug",
            "--skip-dependencies",
        ],
        env={
            **os.environ,
            "PORT": str(service["port"]),
            "HOST": service["host"],
            "ENVIRONMENT": "development",
            "LOG_LEVEL": "INFO",
        },
    )

    # Wait for the service to start
    logger.info(f"Waiting for {service['name']} to start...")
    for _ in range(30):
        try:
            response = requests.get(f"{service['base_url']}{service['health_endpoint']}")
            if response.status_code == 200:
                logger.info(f"{service['name']} started successfully")
                return process
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)

    logger.error(f"Failed to start {service['name']}")
    return process


def stop_service(service: Dict[str, Any]) -> None:
    """
    Stop a service.

    Args:
        service: Service configuration
    """
    if service["process"]:
        logger.info(f"Stopping {service['name']}...")
        service["process"].terminate()
        service["process"].wait()
        service["process"] = None
        logger.info(f"{service['name']} stopped successfully")


def start_all_services() -> bool:
    """
    Start all services.

    Returns:
        True if all services started successfully, False otherwise
    """
    logger.info("Starting all services...")

    all_started = True
    for service in SERVICES:
        process = start_service(service)
        service["process"] = process
        if not process:
            all_started = False

    return all_started


def stop_all_services() -> None:
    """Stop all services."""
    logger.info("Stopping all services...")

    for service in SERVICES:
        stop_service(service)


def check_health(service: Dict[str, Any]) -> bool:
    """
    Check the health of a service.

    Args:
        service: Service configuration

    Returns:
        True if the service is healthy, False otherwise
    """
    logger.info(f"Checking health of {service['name']}...")

    try:
        response = requests.get(f"{service['base_url']}{service['health_endpoint']}")
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                logger.info(f"{service['name']} is healthy")
                return True

        logger.error(f"{service['name']} is not healthy: {response.text}")
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"{service['name']} is not reachable")
        return False


def check_all_health() -> bool:
    """
    Check the health of all services.

    Returns:
        True if all services are healthy, False otherwise
    """
    logger.info("Checking health of all services...")

    all_healthy = True
    for service in SERVICES:
        if not check_health(service):
            all_healthy = False

    return all_healthy


def test_ml_workbench_service() -> bool:
    """
    Test ML Workbench Service.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing ML Workbench Service...")

    service = next(s for s in SERVICES if s["name"] == "ML Workbench Service")
    base_url = service["base_url"]
    api_prefix = service["api_prefix"]

    # Test model registry
    logger.info("Testing model registry...")
    model_data = {
        "name": "test_model",
        "version": "1.0.0",
        "description": "Test model",
        "framework": "TensorFlow",
        "input_schema": {"features": ["price_open", "price_high", "price_low", "price_close", "volume"]},
        "output_schema": {"prediction": "float"},
        "metrics": {"accuracy": 0.85, "precision": 0.82, "recall": 0.79, "f1_score": 0.80},
        "tags": ["test", "tensorflow"],
        "created_by": "test_user",
    }

    try:
        response = requests.post(
            f"{base_url}{api_prefix}/model-registry/models",
            json=model_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to create model: {response.text}")
            return False

        model_id = response.json()["data"]["id"]
        logger.info(f"Model created with ID: {model_id}")

        # Test model training
        logger.info("Testing model training...")
        job_data = {
            "model_name": "test_model",
            "model_version": "1.0.0",
            "dataset_id": "test_dataset",
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "optimizer": "adam",
            },
            "created_by": "test_user",
        }

        response = requests.post(
            f"{base_url}{api_prefix}/model-training/jobs",
            json=job_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to create training job: {response.text}")
            return False

        job_id = response.json()["data"]["job_id"]
        logger.info(f"Training job created with ID: {job_id}")

        # Test model serving
        logger.info("Testing model serving...")
        endpoint_data = {
            "model_name": "test_model",
            "model_version": "1.0.0",
            "endpoint_name": "test_endpoint",
            "created_by": "test_user",
        }

        response = requests.post(
            f"{base_url}{api_prefix}/model-serving/endpoints",
            json=endpoint_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to create endpoint: {response.text}")
            return False

        endpoint_id = response.json()["data"]["endpoint_id"]
        logger.info(f"Endpoint created with ID: {endpoint_id}")

        logger.info("ML Workbench Service tests passed")
        return True
    except Exception as e:
        logger.exception(f"Error testing ML Workbench Service: {str(e)}")
        return False


def test_monitoring_alerting_service() -> bool:
    """
    Test Monitoring Alerting Service.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing Monitoring Alerting Service...")

    service = next(s for s in SERVICES if s["name"] == "Monitoring Alerting Service")
    base_url = service["base_url"]
    api_prefix = service["api_prefix"]

    # Test alerts
    logger.info("Testing alerts...")
    alert_data = {
        "name": "test_alert",
        "description": "Test alert",
        "query": "cpu_usage_percent > 90",
        "severity": "high",
        "labels": {"service": "trading-gateway", "environment": "testing"},
        "annotations": {"summary": "High CPU usage", "description": "CPU usage is above 90%"},
    }

    try:
        response = requests.post(
            f"{base_url}{api_prefix}/alerts",
            json=alert_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to create alert: {response.text}")
            return False

        alert_id = response.json()["data"]["id"]
        logger.info(f"Alert created with ID: {alert_id}")

        # Test dashboards
        logger.info("Testing dashboards...")
        dashboard_data = {
            "title": "Test Dashboard",
            "description": "Test dashboard",
            "created_by": "test_user",
            "tags": ["test", "dashboard"],
            "data": {
                "panels": [
                    {
                        "id": 1,
                        "title": "CPU Usage",
                        "type": "graph",
                        "datasource": "prometheus",
                        "targets": [
                            {
                                "expr": 'cpu_usage_percent{service="trading-gateway"}',
                                "legendFormat": "CPU Usage",
                            }
                        ],
                    },
                ],
            },
        }

        response = requests.post(
            f"{base_url}{api_prefix}/dashboards",
            json=dashboard_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to create dashboard: {response.text}")
            return False

        dashboard_uid = response.json()["data"]["uid"]
        logger.info(f"Dashboard created with UID: {dashboard_uid}")

        # Test notifications
        logger.info("Testing notifications...")
        notification_data = {
            "channel": "email",
            "recipient": "test@example.com",
            "message": "Test notification",
        }

        response = requests.post(
            f"{base_url}{api_prefix}/notifications/test",
            json=notification_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to send notification: {response.text}")
            return False

        logger.info("Notification sent successfully")

        logger.info("Monitoring Alerting Service tests passed")
        return True
    except Exception as e:
        logger.exception(f"Error testing Monitoring Alerting Service: {str(e)}")
        return False


def test_data_pipeline_service() -> bool:
    """
    Test Data Pipeline Service.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing Data Pipeline Service...")

    service = next(s for s in SERVICES if s["name"] == "Data Pipeline Service")
    base_url = service["base_url"]
    api_prefix = service["api_prefix"]

    # Test pipeline creation
    logger.info("Testing pipeline creation...")
    pipeline_data = {
        "name": "test_pipeline",
        "description": "Test pipeline",
        "source": "test_source",
        "destination": "test_destination",
        "schedule": "0 0 * * *",
        "enabled": True,
        "config": {
            "batch_size": 1000,
            "timeout": 3600,
            "retry_count": 3,
        },
    }

    try:
        response = requests.post(
            f"{base_url}{api_prefix}/pipelines",
            json=pipeline_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to create pipeline: {response.text}")
            return False

        pipeline_id = response.json()["data"]["id"]
        logger.info(f"Pipeline created with ID: {pipeline_id}")

        # Test pipeline execution
        logger.info("Testing pipeline execution...")
        execution_data = {
            "pipeline_id": pipeline_id,
            "parameters": {
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",
            },
        }

        response = requests.post(
            f"{base_url}{api_prefix}/pipelines/{pipeline_id}/executions",
            json=execution_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to execute pipeline: {response.text}")
            return False

        execution_id = response.json()["data"]["execution_id"]
        logger.info(f"Pipeline execution started with ID: {execution_id}")

        logger.info("Data Pipeline Service tests passed")
        return True
    except Exception as e:
        logger.exception(f"Error testing Data Pipeline Service: {str(e)}")
        return False


def test_ml_integration_service() -> bool:
    """
    Test ML Integration Service.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing ML Integration Service...")

    service = next(s for s in SERVICES if s["name"] == "ML Integration Service")
    base_url = service["base_url"]
    api_prefix = service["api_prefix"]

    # Test model prediction
    logger.info("Testing model prediction...")
    prediction_data = {
        "model_name": "test_model",
        "model_version": "1.0.0",
        "features": {
            "price_open": 1.1234,
            "price_high": 1.1345,
            "price_low": 1.1200,
            "price_close": 1.1300,
            "volume": 1000,
        },
    }

    try:
        response = requests.post(
            f"{base_url}{api_prefix}/predict",
            json=prediction_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to make prediction: {response.text}")
            return False

        prediction = response.json()["data"]["prediction"]
        logger.info(f"Prediction: {prediction}")

        # Test feature importance
        logger.info("Testing feature importance...")
        feature_importance_data = {
            "model_name": "test_model",
            "model_version": "1.0.0",
        }

        response = requests.post(
            f"{base_url}{api_prefix}/feature-importance",
            json=feature_importance_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to get feature importance: {response.text}")
            return False

        feature_importance = response.json()["data"]["feature_importance"]
        logger.info(f"Feature importance: {feature_importance}")

        logger.info("ML Integration Service tests passed")
        return True
    except Exception as e:
        logger.exception(f"Error testing ML Integration Service: {str(e)}")
        return False


def test_service_interactions() -> bool:
    """
    Test interactions between services.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing service interactions...")

    # Test ML Workbench Service -> ML Integration Service
    logger.info("Testing ML Workbench Service -> ML Integration Service...")
    ml_workbench_service = next(s for s in SERVICES if s["name"] == "ML Workbench Service")
    ml_integration_service = next(s for s in SERVICES if s["name"] == "ML Integration Service")

    # Create a model in ML Workbench Service
    model_data = {
        "name": "interaction_test_model",
        "version": "1.0.0",
        "description": "Interaction test model",
        "framework": "TensorFlow",
        "input_schema": {"features": ["price_open", "price_high", "price_low", "price_close", "volume"]},
        "output_schema": {"prediction": "float"},
        "metrics": {"accuracy": 0.85, "precision": 0.82, "recall": 0.79, "f1_score": 0.80},
        "tags": ["test", "tensorflow"],
        "created_by": "test_user",
    }

    try:
        response = requests.post(
            f"{ml_workbench_service['base_url']}{ml_workbench_service['api_prefix']}/model-registry/models",
            json=model_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to create model: {response.text}")
            return False

        model_id = response.json()["data"]["id"]
        logger.info(f"Model created with ID: {model_id}")

        # Create an endpoint in ML Workbench Service
        endpoint_data = {
            "model_name": "interaction_test_model",
            "model_version": "1.0.0",
            "endpoint_name": "interaction_test_endpoint",
            "created_by": "test_user",
        }

        response = requests.post(
            f"{ml_workbench_service['base_url']}{ml_workbench_service['api_prefix']}/model-serving/endpoints",
            json=endpoint_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to create endpoint: {response.text}")
            return False

        endpoint_id = response.json()["data"]["endpoint_id"]
        logger.info(f"Endpoint created with ID: {endpoint_id}")

        # Make a prediction using ML Integration Service
        prediction_data = {
            "model_name": "interaction_test_model",
            "model_version": "1.0.0",
            "features": {
                "price_open": 1.1234,
                "price_high": 1.1345,
                "price_low": 1.1200,
                "price_close": 1.1300,
                "volume": 1000,
            },
        }

        response = requests.post(
            f"{ml_integration_service['base_url']}{ml_integration_service['api_prefix']}/predict",
            json=prediction_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to make prediction: {response.text}")
            return False

        prediction = response.json()["data"]["prediction"]
        logger.info(f"Prediction: {prediction}")

        # Test Data Pipeline Service -> ML Integration Service
        logger.info("Testing Data Pipeline Service -> ML Integration Service...")
        data_pipeline_service = next(s for s in SERVICES if s["name"] == "Data Pipeline Service")

        # Create a pipeline in Data Pipeline Service
        pipeline_data = {
            "name": "interaction_test_pipeline",
            "description": "Interaction test pipeline",
            "source": "test_source",
            "destination": "test_destination",
            "schedule": "0 0 * * *",
            "enabled": True,
            "config": {
                "batch_size": 1000,
                "timeout": 3600,
                "retry_count": 3,
            },
        }

        response = requests.post(
            f"{data_pipeline_service['base_url']}{data_pipeline_service['api_prefix']}/pipelines",
            json=pipeline_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to create pipeline: {response.text}")
            return False

        pipeline_id = response.json()["data"]["id"]
        logger.info(f"Pipeline created with ID: {pipeline_id}")

        # Create a pipeline execution that uses ML Integration Service
        execution_data = {
            "pipeline_id": pipeline_id,
            "parameters": {
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",
                "model_name": "interaction_test_model",
                "model_version": "1.0.0",
            },
        }

        response = requests.post(
            f"{data_pipeline_service['base_url']}{data_pipeline_service['api_prefix']}/pipelines/{pipeline_id}/executions",
            json=execution_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to execute pipeline: {response.text}")
            return False

        execution_id = response.json()["data"]["execution_id"]
        logger.info(f"Pipeline execution started with ID: {execution_id}")

        # Test Monitoring Alerting Service
        logger.info("Testing Monitoring Alerting Service...")
        monitoring_alerting_service = next(s for s in SERVICES if s["name"] == "Monitoring Alerting Service")

        # Create an alert for ML Integration Service
        alert_data = {
            "name": "interaction_test_alert",
            "description": "Interaction test alert",
            "query": 'ml_integration_model_predictions_total{model_name="interaction_test_model"} > 100',
            "severity": "high",
            "labels": {"service": "ml-integration-service", "environment": "testing"},
            "annotations": {"summary": "High prediction count", "description": "Prediction count is above 100"},
        }

        response = requests.post(
            f"{monitoring_alerting_service['base_url']}{monitoring_alerting_service['api_prefix']}/alerts",
            json=alert_data,
        )

        if response.status_code != 200:
            logger.error(f"Failed to create alert: {response.text}")
            return False

        alert_id = response.json()["data"]["id"]
        logger.info(f"Alert created with ID: {alert_id}")

        logger.info("Service interaction tests passed")
        return True
    except Exception as e:
        logger.exception(f"Error testing service interactions: {str(e)}")
        return False


def run_tests() -> bool:
    """
    Run all tests.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Running integration tests...")

    # Start all services
    if not start_all_services():
        logger.error("Failed to start all services")
        stop_all_services()
        return False

    # Check health of all services
    if not check_all_health():
        logger.error("Not all services are healthy")
        stop_all_services()
        return False

    # Run tests
    try:
        # Test individual services
        if not test_ml_workbench_service():
            logger.error("ML Workbench Service tests failed")
            stop_all_services()
            return False

        if not test_monitoring_alerting_service():
            logger.error("Monitoring Alerting Service tests failed")
            stop_all_services()
            return False

        if not test_data_pipeline_service():
            logger.error("Data Pipeline Service tests failed")
            stop_all_services()
            return False

        if not test_ml_integration_service():
            logger.error("ML Integration Service tests failed")
            stop_all_services()
            return False

        # Test service interactions
        if not test_service_interactions():
            logger.error("Service interaction tests failed")
            stop_all_services()
            return False

        logger.info("All tests passed")
        return True
    finally:
        # Stop all services
        stop_all_services()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)