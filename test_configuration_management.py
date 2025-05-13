#!/usr/bin/env python
"""
Test script for the standardized configuration management system.

This script tests the standardized configuration management system to ensure it is correctly
implemented and functioning as expected.
"""

import os
import sys
import unittest
import logging
import json
import tempfile
from typing import Dict, Any, Optional, List
import importlib.util
import inspect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_configuration_management")

# Services to test
SERVICES = [
    {
        "name": "ML Workbench Service",
        "path": "ml-workbench-service",
        "package": "ml_workbench_service",
        "config_module": "ml_workbench_service.config.standardized_config",
    },
    {
        "name": "Monitoring Alerting Service",
        "path": "monitoring-alerting-service",
        "package": "monitoring_alerting_service",
        "config_module": "monitoring_alerting_service.config.standardized_config",
    },
]


class ConfigurationManagementTestCase(unittest.TestCase):
    """Test case for configuration management system."""

    def setUp(self):
        """Set up the test case."""
        # Add service paths to sys.path
        self.original_path = sys.path.copy()
        for service in SERVICES:
            service_path = os.path.join(os.getcwd(), service["path"])
            if service_path not in sys.path:
                sys.path.append(service_path)
        
        # Save original environment variables
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Tear down the test case."""
        # Restore original sys.path
        sys.path = self.original_path
        
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_config_module_existence(self):
        """Test that all required config modules exist."""
        for service in SERVICES:
            logger.info(f"Testing config module existence for {service['name']}")
            try:
                module = importlib.import_module(service["config_module"])
                self.assertIsNotNone(module, f"Module {service['config_module']} should not be None")
                logger.info(f"Module {service['config_module']} exists")
            except ImportError as e:
                self.fail(f"Module {service['config_module']} does not exist: {str(e)}")

    def test_config_module_attributes(self):
        """Test that all required config module attributes exist."""
        for service in SERVICES:
            logger.info(f"Testing config module attributes for {service['name']}")
            module = importlib.import_module(service["config_module"])
            
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
            
            logger.info(f"Module {service['config_module']} has all required attributes")

    def test_environment_variable_loading(self):
        """Test that configuration can be loaded from environment variables."""
        for service in SERVICES:
            logger.info(f"Testing environment variable loading for {service['name']}")
            
            # Set environment variables
            os.environ["HOST"] = "test-host"
            os.environ["PORT"] = "9999"
            os.environ["LOG_LEVEL"] = "DEBUG"
            os.environ["ENVIRONMENT"] = "testing"
            os.environ["API_VERSION"] = "v2"
            
            # Reload module to pick up environment variables
            module_name = service["config_module"]
            if module_name in sys.modules:
                del sys.modules[module_name]
            module = importlib.import_module(module_name)
            
            # Check that environment variables were loaded
            self.assertEqual(module.settings.HOST, "test-host", "HOST should be loaded from environment variable")
            self.assertEqual(module.settings.PORT, 9999, "PORT should be loaded from environment variable")
            self.assertEqual(module.settings.LOG_LEVEL, "DEBUG", "LOG_LEVEL should be loaded from environment variable")
            self.assertEqual(module.settings.ENVIRONMENT, "testing", "ENVIRONMENT should be loaded from environment variable")
            self.assertEqual(module.settings.API_VERSION, "v2", "API_VERSION should be loaded from environment variable")
            
            logger.info(f"Module {module_name} loads configuration from environment variables")

    def test_config_file_loading(self):
        """Test that configuration can be loaded from a config file."""
        for service in SERVICES:
            logger.info(f"Testing config file loading for {service['name']}")
            
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                config_file = f.name
                config = {
                    "HOST": "config-file-host",
                    "PORT": 8888,
                    "LOG_LEVEL": "INFO",
                    "ENVIRONMENT": "staging",
                    "API_VERSION": "v3",
                }
                json.dump(config, f)
            
            try:
                # Set environment variable to point to config file
                os.environ["CONFIG_FILE"] = config_file
                
                # Reload module to pick up config file
                module_name = service["config_module"]
                if module_name in sys.modules:
                    del sys.modules[module_name]
                module = importlib.import_module(module_name)
                
                # Check that config file was loaded
                self.assertEqual(module.settings.HOST, "config-file-host", "HOST should be loaded from config file")
                self.assertEqual(module.settings.PORT, 8888, "PORT should be loaded from config file")
                self.assertEqual(module.settings.LOG_LEVEL, "INFO", "LOG_LEVEL should be loaded from config file")
                self.assertEqual(module.settings.ENVIRONMENT, "staging", "ENVIRONMENT should be loaded from config file")
                self.assertEqual(module.settings.API_VERSION, "v3", "API_VERSION should be loaded from config file")
                
                logger.info(f"Module {module_name} loads configuration from config file")
            finally:
                # Clean up
                os.unlink(config_file)

    def test_default_values(self):
        """Test that default values are used when no configuration is provided."""
        for service in SERVICES:
            logger.info(f"Testing default values for {service['name']}")
            
            # Clear environment variables
            for key in list(os.environ.keys()):
                if key in ["HOST", "PORT", "LOG_LEVEL", "ENVIRONMENT", "API_VERSION", "CONFIG_FILE"]:
                    del os.environ[key]
            
            # Reload module to use default values
            module_name = service["config_module"]
            if module_name in sys.modules:
                del sys.modules[module_name]
            module = importlib.import_module(module_name)
            
            # Check that default values are used
            self.assertIsNotNone(module.settings.HOST, "HOST should have a default value")
            self.assertIsNotNone(module.settings.PORT, "PORT should have a default value")
            self.assertIsNotNone(module.settings.LOG_LEVEL, "LOG_LEVEL should have a default value")
            self.assertIsNotNone(module.settings.ENVIRONMENT, "ENVIRONMENT should have a default value")
            self.assertIsNotNone(module.settings.API_VERSION, "API_VERSION should have a default value")
            
            logger.info(f"Module {module_name} uses default values when no configuration is provided")

    def test_helper_functions(self):
        """Test that helper functions work correctly."""
        for service in SERVICES:
            logger.info(f"Testing helper functions for {service['name']}")
            module = importlib.import_module(service["config_module"])
            
            # Test get_settings function
            settings = module.get_settings()
            self.assertIsNotNone(settings, "get_settings should return a settings object")
            self.assertEqual(settings, module.settings, "get_settings should return the same settings object")
            
            # Test get_db_url function
            db_url = module.get_db_url()
            self.assertIsNotNone(db_url, "get_db_url should return a database URL")
            
            # Test get_api_settings function
            api_settings = module.get_api_settings()
            self.assertIsNotNone(api_settings, "get_api_settings should return API settings")
            self.assertIsInstance(api_settings, dict, "get_api_settings should return a dictionary")
            self.assertIn("api_prefix", api_settings, "API settings should include api_prefix")
            
            # Test get_security_settings function
            security_settings = module.get_security_settings()
            self.assertIsNotNone(security_settings, "get_security_settings should return security settings")
            self.assertIsInstance(security_settings, dict, "get_security_settings should return a dictionary")
            self.assertIn("cors_origins", security_settings, "Security settings should include cors_origins")
            
            # Test get_monitoring_settings function
            monitoring_settings = module.get_monitoring_settings()
            self.assertIsNotNone(monitoring_settings, "get_monitoring_settings should return monitoring settings")
            self.assertIsInstance(monitoring_settings, dict, "get_monitoring_settings should return a dictionary")
            self.assertIn("enable_metrics", monitoring_settings, "Monitoring settings should include enable_metrics")
            
            logger.info(f"Helper functions for {module.__name__} work correctly")


def run_tests():
    """Run the tests."""
    logger.info("Running configuration management tests")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


if __name__ == "__main__":
    run_tests()