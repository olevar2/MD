#!/usr/bin/env python
"""
Test script for verifying the standardized service template modules.

This script tests the standardized modules in the ML Workbench Service and
Monitoring Alerting Service to ensure they are correctly implemented and
functioning as expected.
"""

import os
import sys
import importlib
import inspect
import unittest
import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_service_template")

# Services to test
SERVICES = [
    {
        "name": "ML Workbench Service",
        "path": "ml-workbench-service",
        "package": "ml_workbench_service",
        "modules": [
            "config.standardized_config",
            "logging_setup",
            "service_clients",
            "database",
            "error_handlers",
            "monitoring",
            "main",
        ],
    },
    {
        "name": "Monitoring Alerting Service",
        "path": "monitoring-alerting-service",
        "package": "monitoring_alerting_service",
        "modules": [
            "config.standardized_config",
            "logging_setup",
            "service_clients",
            "database",
            "error_handlers",
            "monitoring",
            "main",
        ],
    },
]


class ServiceTemplateTestCase(unittest.TestCase):
    """Test case for service template modules."""

    def setUp(self):
        """Set up the test case."""
        # Add service paths to sys.path
        self.original_path = sys.path.copy()
        for service in SERVICES:
            service_path = os.path.join(os.getcwd(), service["path"])
            if service_path not in sys.path:
                sys.path.append(service_path)

    def tearDown(self):
        """Tear down the test case."""
        # Restore original sys.path
        sys.path = self.original_path

    def test_module_existence(self):
        """Test that all required modules exist."""
        for service in SERVICES:
            logger.info(f"Testing module existence for {service['name']}")
            for module_name in service["modules"]:
                full_module_name = f"{service['package']}.{module_name}"
                try:
                    module = importlib.import_module(full_module_name)
                    self.assertIsNotNone(module, f"Module {full_module_name} should not be None")
                    logger.info(f"Module {full_module_name} exists")
                except ImportError as e:
                    self.fail(f"Module {full_module_name} does not exist: {str(e)}")

    def test_config_module(self):
        """Test the configuration module."""
        for service in SERVICES:
            logger.info(f"Testing configuration module for {service['name']}")
            full_module_name = f"{service['package']}.config.standardized_config"
            try:
                module = importlib.import_module(full_module_name)
                
                # Check for required attributes
                self.assertTrue(hasattr(module, "settings"), "Module should have settings attribute")
                self.assertTrue(hasattr(module, "get_settings"), "Module should have get_settings function")
                self.assertTrue(hasattr(module, "get_db_url"), "Module should have get_db_url function")
                self.assertTrue(hasattr(module, "get_api_settings"), "Module should have get_api_settings function")
                self.assertTrue(hasattr(module, "get_security_settings"), "Module should have get_security_settings function")
                self.assertTrue(hasattr(module, "get_monitoring_settings"), "Module should have get_monitoring_settings function")
                
                # Check settings class
                settings_class = module.settings.__class__
                self.assertTrue(hasattr(settings_class, "SERVICE_NAME"), "Settings should have SERVICE_NAME attribute")
                self.assertTrue(hasattr(settings_class, "API_VERSION"), "Settings should have API_VERSION attribute")
                self.assertTrue(hasattr(settings_class, "HOST"), "Settings should have HOST attribute")
                self.assertTrue(hasattr(settings_class, "PORT"), "Settings should have PORT attribute")
                self.assertTrue(hasattr(settings_class, "ENVIRONMENT"), "Settings should have ENVIRONMENT attribute")
                
                logger.info(f"Configuration module {full_module_name} is valid")
            except ImportError as e:
                self.fail(f"Module {full_module_name} does not exist: {str(e)}")
            except Exception as e:
                self.fail(f"Error testing configuration module {full_module_name}: {str(e)}")

    def test_logging_module(self):
        """Test the logging module."""
        for service in SERVICES:
            logger.info(f"Testing logging module for {service['name']}")
            full_module_name = f"{service['package']}.logging_setup"
            try:
                module = importlib.import_module(full_module_name)
                
                # Check for required attributes
                self.assertTrue(hasattr(module, "configure_logging"), "Module should have configure_logging function")
                self.assertTrue(hasattr(module, "get_logger"), "Module should have get_logger function")
                self.assertTrue(hasattr(module, "set_correlation_id"), "Module should have set_correlation_id function")
                self.assertTrue(hasattr(module, "get_correlation_id"), "Module should have get_correlation_id function")
                
                # Test get_logger function
                test_logger = module.get_logger("test")
                self.assertIsNotNone(test_logger, "get_logger should return a logger")
                self.assertEqual(test_logger.name, "test", "Logger name should be 'test'")
                
                # Test correlation ID functions
                test_correlation_id = "test-correlation-id"
                module.set_correlation_id(test_correlation_id)
                self.assertEqual(module.get_correlation_id(), test_correlation_id, "get_correlation_id should return the set correlation ID")
                
                logger.info(f"Logging module {full_module_name} is valid")
            except ImportError as e:
                self.fail(f"Module {full_module_name} does not exist: {str(e)}")
            except Exception as e:
                self.fail(f"Error testing logging module {full_module_name}: {str(e)}")

    def test_service_clients_module(self):
        """Test the service clients module."""
        for service in SERVICES:
            logger.info(f"Testing service clients module for {service['name']}")
            full_module_name = f"{service['package']}.service_clients"
            try:
                module = importlib.import_module(full_module_name)
                
                # Check for required attributes
                self.assertTrue(hasattr(module, "BaseServiceClient"), "Module should have BaseServiceClient class")
                self.assertTrue(hasattr(module, "close_all_clients"), "Module should have close_all_clients function")
                
                # Check BaseServiceClient class
                self.assertTrue(inspect.isclass(module.BaseServiceClient), "BaseServiceClient should be a class")
                self.assertTrue(hasattr(module.BaseServiceClient, "get"), "BaseServiceClient should have get method")
                self.assertTrue(hasattr(module.BaseServiceClient, "post"), "BaseServiceClient should have post method")
                self.assertTrue(hasattr(module.BaseServiceClient, "put"), "BaseServiceClient should have put method")
                self.assertTrue(hasattr(module.BaseServiceClient, "delete"), "BaseServiceClient should have delete method")
                
                # Check close_all_clients function
                self.assertTrue(asyncio.iscoroutinefunction(module.close_all_clients), "close_all_clients should be a coroutine function")
                
                logger.info(f"Service clients module {full_module_name} is valid")
            except ImportError as e:
                self.fail(f"Module {full_module_name} does not exist: {str(e)}")
            except Exception as e:
                self.fail(f"Error testing service clients module {full_module_name}: {str(e)}")

    def test_database_module(self):
        """Test the database module."""
        for service in SERVICES:
            logger.info(f"Testing database module for {service['name']}")
            full_module_name = f"{service['package']}.database"
            try:
                module = importlib.import_module(full_module_name)
                
                # Check for required attributes
                self.assertTrue(hasattr(module, "Base"), "Module should have Base class")
                self.assertTrue(hasattr(module, "init_db"), "Module should have init_db function")
                self.assertTrue(hasattr(module, "create_tables"), "Module should have create_tables function")
                self.assertTrue(hasattr(module, "get_sync_session"), "Module should have get_sync_session function")
                self.assertTrue(hasattr(module, "get_async_session"), "Module should have get_async_session function")
                self.assertTrue(hasattr(module, "BaseRepository"), "Module should have BaseRepository class")
                
                # Check BaseRepository class
                self.assertTrue(inspect.isclass(module.BaseRepository), "BaseRepository should be a class")
                self.assertTrue(hasattr(module.BaseRepository, "get_by_id"), "BaseRepository should have get_by_id method")
                self.assertTrue(hasattr(module.BaseRepository, "get_all"), "BaseRepository should have get_all method")
                self.assertTrue(hasattr(module.BaseRepository, "create"), "BaseRepository should have create method")
                self.assertTrue(hasattr(module.BaseRepository, "update"), "BaseRepository should have update method")
                self.assertTrue(hasattr(module.BaseRepository, "delete"), "BaseRepository should have delete method")
                
                logger.info(f"Database module {full_module_name} is valid")
            except ImportError as e:
                self.fail(f"Module {full_module_name} does not exist: {str(e)}")
            except Exception as e:
                self.fail(f"Error testing database module {full_module_name}: {str(e)}")

    def test_error_handlers_module(self):
        """Test the error handlers module."""
        for service in SERVICES:
            logger.info(f"Testing error handlers module for {service['name']}")
            full_module_name = f"{service['package']}.error_handlers"
            try:
                module = importlib.import_module(full_module_name)
                
                # Check for required attributes
                self.assertTrue(hasattr(module, "setup_error_handlers"), "Module should have setup_error_handlers function")
                self.assertTrue(hasattr(module, "get_error_response"), "Module should have get_error_response function")
                
                # Check setup_error_handlers function
                self.assertTrue(callable(module.setup_error_handlers), "setup_error_handlers should be callable")
                
                # Check get_error_response function
                self.assertTrue(callable(module.get_error_response), "get_error_response should be callable")
                
                logger.info(f"Error handlers module {full_module_name} is valid")
            except ImportError as e:
                self.fail(f"Module {full_module_name} does not exist: {str(e)}")
            except Exception as e:
                self.fail(f"Error testing error handlers module {full_module_name}: {str(e)}")

    def test_monitoring_module(self):
        """Test the monitoring module."""
        for service in SERVICES:
            logger.info(f"Testing monitoring module for {service['name']}")
            full_module_name = f"{service['package']}.monitoring"
            try:
                module = importlib.import_module(full_module_name)
                
                # Check for required attributes
                self.assertTrue(hasattr(module, "setup_monitoring"), "Module should have setup_monitoring function")
                self.assertTrue(hasattr(module, "register_health_check"), "Module should have register_health_check function")
                self.assertTrue(hasattr(module, "health_check"), "Module should have health_check attribute")
                self.assertTrue(hasattr(module, "metrics_registry"), "Module should have metrics_registry attribute")
                self.assertTrue(hasattr(module, "start_metrics_collection"), "Module should have start_metrics_collection function")
                self.assertTrue(hasattr(module, "stop_metrics_collection"), "Module should have stop_metrics_collection function")
                
                # Check setup_monitoring function
                self.assertTrue(callable(module.setup_monitoring), "setup_monitoring should be callable")
                
                # Check register_health_check function
                self.assertTrue(callable(module.register_health_check), "register_health_check should be callable")
                
                logger.info(f"Monitoring module {full_module_name} is valid")
            except ImportError as e:
                self.fail(f"Module {full_module_name} does not exist: {str(e)}")
            except Exception as e:
                self.fail(f"Error testing monitoring module {full_module_name}: {str(e)}")

    def test_main_module(self):
        """Test the main module."""
        for service in SERVICES:
            logger.info(f"Testing main module for {service['name']}")
            full_module_name = f"{service['package']}.main"
            try:
                module = importlib.import_module(full_module_name)
                
                # Check for required attributes
                self.assertTrue(hasattr(module, "app"), "Module should have app attribute")
                self.assertTrue(hasattr(module, "startup_event"), "Module should have startup_event function")
                self.assertTrue(hasattr(module, "shutdown_event"), "Module should have shutdown_event function")
                
                # Check app attribute
                self.assertTrue(hasattr(module.app, "title"), "app should have title attribute")
                self.assertTrue(hasattr(module.app, "description"), "app should have description attribute")
                self.assertTrue(hasattr(module.app, "version"), "app should have version attribute")
                
                # Check startup_event function
                self.assertTrue(asyncio.iscoroutinefunction(module.startup_event), "startup_event should be a coroutine function")
                
                # Check shutdown_event function
                self.assertTrue(asyncio.iscoroutinefunction(module.shutdown_event), "shutdown_event should be a coroutine function")
                
                logger.info(f"Main module {full_module_name} is valid")
            except ImportError as e:
                self.fail(f"Module {full_module_name} does not exist: {str(e)}")
            except Exception as e:
                self.fail(f"Error testing main module {full_module_name}: {str(e)}")


def run_tests():
    """Run the tests."""
    logger.info("Running service template tests")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


if __name__ == "__main__":
    run_tests()