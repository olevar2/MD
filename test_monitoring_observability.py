#!/usr/bin/env python
"""
Test script for the standardized monitoring and observability system.

This script tests the standardized monitoring and observability system to ensure it is correctly
implemented and functioning as expected.
"""

import os
import sys
import unittest
import logging
import json
import tempfile
import time
import asyncio
import threading
from typing import Dict, Any, Optional, List
import importlib.util
import inspect
import requests
from fastapi import FastAPI
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_monitoring_observability")

# Services to test
SERVICES = [
    {
        "name": "ML Workbench Service",
        "path": "ml-workbench-service",
        "package": "ml_workbench_service",
        "monitoring_module": "ml_workbench_service.monitoring",
        "logging_module": "ml_workbench_service.logging_setup",
    },
    {
        "name": "Monitoring Alerting Service",
        "path": "monitoring-alerting-service",
        "package": "monitoring_alerting_service",
        "monitoring_module": "monitoring_alerting_service.monitoring",
        "logging_module": "monitoring_alerting_service.logging_setup",
    },
]

# Test server
test_app = FastAPI()
test_server = None
test_server_port = 8099
test_server_thread = None


@test_app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello World"}


def run_test_server():
    """Run the test server."""
    uvicorn.run(test_app, host="127.0.0.1", port=test_server_port, log_level="error")


def start_test_server():
    """Start the test server."""
    global test_server_thread
    test_server_thread = threading.Thread(target=run_test_server)
    test_server_thread.daemon = True
    test_server_thread.start()
    
    # Wait for the server to start
    for _ in range(30):
        try:
            response = requests.get(f"http://127.0.0.1:{test_server_port}/")
            if response.status_code == 200:
                logger.info("Test server started successfully")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.1)
    
    logger.error("Failed to start test server")
    return False


def stop_test_server():
    """Stop the test server."""
    global test_server_thread
    if test_server_thread:
        test_server_thread.join(timeout=1)
        test_server_thread = None
        logger.info("Test server stopped successfully")


class MonitoringObservabilityTestCase(unittest.TestCase):
    """Test case for monitoring and observability system."""

    @classmethod
    def setUpClass(cls):
        """Set up the test case."""
        # Add service paths to sys.path
        cls.original_path = sys.path.copy()
        for service in SERVICES:
            service_path = os.path.join(os.getcwd(), service["path"])
            if service_path not in sys.path:
                sys.path.append(service_path)
        
        # Start test server
        if not start_test_server():
            raise Exception("Failed to start test server")

    @classmethod
    def tearDownClass(cls):
        """Tear down the test case."""
        # Restore original sys.path
        sys.path = cls.original_path
        
        # Stop test server
        stop_test_server()

    def test_monitoring_module_existence(self):
        """Test that all required monitoring modules exist."""
        for service in SERVICES:
            logger.info(f"Testing monitoring module existence for {service['name']}")
            try:
                module = importlib.import_module(service["monitoring_module"])
                self.assertIsNotNone(module, f"Module {service['monitoring_module']} should not be None")
                logger.info(f"Module {service['monitoring_module']} exists")
            except ImportError as e:
                self.fail(f"Module {service['monitoring_module']} does not exist: {str(e)}")

    def test_logging_module_existence(self):
        """Test that all required logging modules exist."""
        for service in SERVICES:
            logger.info(f"Testing logging module existence for {service['name']}")
            try:
                module = importlib.import_module(service["logging_module"])
                self.assertIsNotNone(module, f"Module {service['logging_module']} should not be None")
                logger.info(f"Module {service['logging_module']} exists")
            except ImportError as e:
                self.fail(f"Module {service['logging_module']} does not exist: {str(e)}")

    def test_monitoring_module_attributes(self):
        """Test that all required monitoring module attributes exist."""
        for service in SERVICES:
            logger.info(f"Testing monitoring module attributes for {service['name']}")
            module = importlib.import_module(service["monitoring_module"])
            
            # Check for required attributes
            self.assertTrue(hasattr(module, "setup_monitoring"), "Module should have setup_monitoring function")
            self.assertTrue(hasattr(module, "register_health_check"), "Module should have register_health_check function")
            self.assertTrue(hasattr(module, "health_check"), "Module should have health_check attribute")
            self.assertTrue(hasattr(module, "metrics_registry"), "Module should have metrics_registry attribute")
            self.assertTrue(hasattr(module, "start_metrics_collection"), "Module should have start_metrics_collection function")
            self.assertTrue(hasattr(module, "stop_metrics_collection"), "Module should have stop_metrics_collection function")
            
            # Check for metric decorators
            self.assertTrue(hasattr(module, "track_database_query"), "Module should have track_database_query decorator")
            self.assertTrue(hasattr(module, "track_service_client_request"), "Module should have track_service_client_request decorator")
            
            logger.info(f"Module {service['monitoring_module']} has all required attributes")

    def test_logging_module_attributes(self):
        """Test that all required logging module attributes exist."""
        for service in SERVICES:
            logger.info(f"Testing logging module attributes for {service['name']}")
            module = importlib.import_module(service["logging_module"])
            
            # Check for required attributes
            self.assertTrue(hasattr(module, "configure_logging"), "Module should have configure_logging function")
            self.assertTrue(hasattr(module, "get_logger"), "Module should have get_logger function")
            self.assertTrue(hasattr(module, "set_correlation_id"), "Module should have set_correlation_id function")
            self.assertTrue(hasattr(module, "get_correlation_id"), "Module should have get_correlation_id function")
            
            logger.info(f"Module {service['logging_module']} has all required attributes")

    def test_health_check(self):
        """Test that health check works correctly."""
        for service in SERVICES:
            logger.info(f"Testing health check for {service['name']}")
            module = importlib.import_module(service["monitoring_module"])
            
            # Check health check
            health_result = module.health_check.check()
            self.assertIsNotNone(health_result, "Health check should return a result")
            self.assertIsInstance(health_result, dict, "Health check should return a dictionary")
            self.assertIn("status", health_result, "Health check result should include status")
            self.assertEqual(health_result["status"], "ok", "Health check status should be 'ok'")
            
            # Register a health check
            module.register_health_check(
                name="test_health_check",
                check_func=lambda: True,
                description="Test health check",
            )
            
            # Check health check again
            health_result = module.health_check.check()
            self.assertIn("checks", health_result, "Health check result should include checks")
            self.assertIn("test_health_check", health_result["checks"], "Health check result should include test_health_check")
            self.assertEqual(health_result["checks"]["test_health_check"]["status"], "ok", "Test health check status should be 'ok'")
            
            logger.info(f"Health check for {service['name']} works correctly")

    def test_metrics_registry(self):
        """Test that metrics registry works correctly."""
        for service in SERVICES:
            logger.info(f"Testing metrics registry for {service['name']}")
            module = importlib.import_module(service["monitoring_module"])
            
            # Check metrics registry
            self.assertIsNotNone(module.metrics_registry, "Metrics registry should not be None")
            
            # Create a test counter
            test_counter = module.metrics_registry.counter(
                name="test_counter",
                description="Test counter",
                labels=["label1", "label2"],
            )
            self.assertIsNotNone(test_counter, "Test counter should not be None")
            
            # Increment the counter
            test_counter.labels(label1="value1", label2="value2").inc()
            
            # Create a test gauge
            test_gauge = module.metrics_registry.gauge(
                name="test_gauge",
                description="Test gauge",
                labels=["label1", "label2"],
            )
            self.assertIsNotNone(test_gauge, "Test gauge should not be None")
            
            # Set the gauge
            test_gauge.labels(label1="value1", label2="value2").set(123)
            
            # Create a test histogram
            test_histogram = module.metrics_registry.histogram(
                name="test_histogram",
                description="Test histogram",
                labels=["label1", "label2"],
                buckets=[0.1, 0.5, 1.0, 5.0],
            )
            self.assertIsNotNone(test_histogram, "Test histogram should not be None")
            
            # Observe the histogram
            test_histogram.labels(label1="value1", label2="value2").observe(0.5)
            
            logger.info(f"Metrics registry for {service['name']} works correctly")

    def test_setup_monitoring(self):
        """Test that setup_monitoring works correctly."""
        for service in SERVICES:
            logger.info(f"Testing setup_monitoring for {service['name']}")
            module = importlib.import_module(service["monitoring_module"])
            
            # Create a test app
            test_app = FastAPI()
            
            # Set up monitoring
            module.setup_monitoring(test_app)
            
            # Check that health check endpoint was added
            self.assertTrue(any(route.path == "/health" for route in test_app.routes), "Health check endpoint should be added")
            
            # Check that readiness check endpoint was added
            self.assertTrue(any(route.path == "/ready" for route in test_app.routes), "Readiness check endpoint should be added")
            
            # Check that metrics middleware was added
            self.assertTrue(len(test_app.middleware) > 0, "Metrics middleware should be added")
            
            logger.info(f"setup_monitoring for {service['name']} works correctly")

    def test_logging_setup(self):
        """Test that logging setup works correctly."""
        for service in SERVICES:
            logger.info(f"Testing logging setup for {service['name']}")
            module = importlib.import_module(service["logging_module"])
            
            # Configure logging
            module.configure_logging(
                service_name=service["name"],
                log_level="INFO",
                enable_json_logging=False,
            )
            
            # Get a logger
            test_logger = module.get_logger("test_logger")
            self.assertIsNotNone(test_logger, "Test logger should not be None")
            
            # Set correlation ID
            test_correlation_id = "test-correlation-id"
            module.set_correlation_id(test_correlation_id)
            
            # Get correlation ID
            correlation_id = module.get_correlation_id()
            self.assertEqual(correlation_id, test_correlation_id, "Correlation ID should match")
            
            logger.info(f"Logging setup for {service['name']} works correctly")

    def test_metric_decorators(self):
        """Test that metric decorators work correctly."""
        for service in SERVICES:
            logger.info(f"Testing metric decorators for {service['name']}")
            module = importlib.import_module(service["monitoring_module"])
            
            # Define a test function with database query decorator
            @module.track_database_query("test_operation")
            async def test_db_query():
                return "test"
            
            # Define a test function with service client request decorator
            @module.track_service_client_request("test_service", "GET")
            async def test_service_request():
                return "test"
            
            # Run the test functions
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result1 = loop.run_until_complete(test_db_query())
                self.assertEqual(result1, "test", "Database query decorator should not affect function result")
                
                result2 = loop.run_until_complete(test_service_request())
                self.assertEqual(result2, "test", "Service client request decorator should not affect function result")
            finally:
                loop.close()
            
            logger.info(f"Metric decorators for {service['name']} work correctly")


def run_tests():
    """Run the tests."""
    logger.info("Running monitoring and observability tests")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


if __name__ == "__main__":
    run_tests()