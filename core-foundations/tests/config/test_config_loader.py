"""
Unit tests for configuration loader.
"""

import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, ValidationError

from core_foundations.config.config_loader import ConfigLoader, ConfigLoadError


class TestConfig(BaseModel):
    """Test configuration model for unit tests."""
    
    app_name: str = Field(..., description="Name of the application")
    api_key: str = Field(..., description="API key for external services")
    debug: bool = Field(False, description="Debug mode flag")
    log_level: str = Field("INFO", description="Logging level")
    timeout: float = Field(30.0, description="Request timeout in seconds")
    retry_count: int = Field(3, description="Number of retry attempts")


class TestConfigWithNested(BaseModel):
    """Test configuration with nested objects."""
    
    app_name: str = Field(..., description="Name of the application")
    database: Dict[str, Any] = Field(..., description="Database settings")
    services: Dict[str, Dict[str, Any]] = Field(..., description="External services configuration")


class TestConfigLoader(unittest.TestCase):
    """Tests for ConfigLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.yaml_config_path = os.path.join(self.temp_dir.name, "config.yaml")
        self.json_config_path = os.path.join(self.temp_dir.name, "config.json")
        
        # Create a valid YAML config
        with open(self.yaml_config_path, "w") as f:
            f.write("""
app_name: TestApp
api_key: abc123xyz789
debug: true
log_level: DEBUG
timeout: 45.5
retry_count: 5
            """)
        
        # Create a valid JSON config
        with open(self.json_config_path, "w") as f:
            f.write("""
{
    "app_name": "TestAppJSON",
    "api_key": "json_api_key",
    "debug": false,
    "log_level": "WARNING",
    "timeout": 20.0,
    "retry_count": 2
}
            """)
        
        # Create the loader
        self.config_loader = ConfigLoader(TestConfig)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_load_yaml_config(self):
        """Test loading configuration from YAML file."""
        config = self.config_loader.load_from_file(self.yaml_config_path)
        
        self.assertEqual(config.app_name, "TestApp")
        self.assertEqual(config.api_key, "abc123xyz789")
        self.assertEqual(config.debug, True)
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.timeout, 45.5)
        self.assertEqual(config.retry_count, 5)
    
    def test_load_json_config(self):
        """Test loading configuration from JSON file."""
        config = self.config_loader.load_from_file(self.json_config_path)
        
        self.assertEqual(config.app_name, "TestAppJSON")
        self.assertEqual(config.api_key, "json_api_key")
        self.assertEqual(config.debug, False)
        self.assertEqual(config.log_level, "WARNING")
        self.assertEqual(config.timeout, 20.0)
        self.assertEqual(config.retry_count, 2)
    
    def test_invalid_file_extension(self):
        """Test loading from a file with unsupported extension."""
        invalid_path = os.path.join(self.temp_dir.name, "config.txt")
        with open(invalid_path, "w") as f:
            f.write("app_name: TestApp\napi_key: test_key")
        
        with self.assertRaises(ConfigLoadError) as cm:
            self.config_loader.load_from_file(invalid_path)
        
        self.assertIn("Unsupported file extension", str(cm.exception))
    
    def test_file_not_found(self):
        """Test handling of non-existent config file."""
        non_existent_path = os.path.join(self.temp_dir.name, "non_existent.yaml")
        
        with self.assertRaises(ConfigLoadError) as cm:
            self.config_loader.load_from_file(non_existent_path)
        
        self.assertIn("Config file not found", str(cm.exception))
    
    def test_invalid_yaml_syntax(self):
        """Test handling of YAML file with invalid syntax."""
        invalid_yaml_path = os.path.join(self.temp_dir.name, "invalid.yaml")
        
        with open(invalid_yaml_path, "w") as f:
            f.write("""
app_name: TestApp
api_key: test_key
invalid yaml: : :
            """)
        
        with self.assertRaises(ConfigLoadError) as cm:
            self.config_loader.load_from_file(invalid_yaml_path)
        
        self.assertIn("Failed to parse YAML config", str(cm.exception))
    
    def test_invalid_json_syntax(self):
        """Test handling of JSON file with invalid syntax."""
        invalid_json_path = os.path.join(self.temp_dir.name, "invalid.json")
        
        with open(invalid_json_path, "w") as f:
            f.write("""
{
    "app_name": "TestApp",
    "api_key": "test_key",
    invalid json here
}
            """)
        
        with self.assertRaises(ConfigLoadError) as cm:
            self.config_loader.load_from_file(invalid_json_path)
        
        self.assertIn("Failed to parse JSON config", str(cm.exception))
    
    def test_validation_error(self):
        """Test handling of validation errors."""
        invalid_config_path = os.path.join(self.temp_dir.name, "invalid_schema.yaml")
        
        with open(invalid_config_path, "w") as f:
            f.write("""
app_name: TestApp
# Missing required api_key field
debug: true
            """)
        
        with self.assertRaises(ConfigLoadError) as cm:
            self.config_loader.load_from_file(invalid_config_path)
        
        self.assertIn("Validation error", str(cm.exception))
    
    def test_load_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "app_name": "DictApp",
            "api_key": "dict_key",
            "debug": True,
            "log_level": "ERROR",
            "timeout": 15.0,
            "retry_count": 1
        }
        
        config = self.config_loader.load_from_dict(config_dict)
        
        self.assertEqual(config.app_name, "DictApp")
        self.assertEqual(config.api_key, "dict_key")
        self.assertEqual(config.debug, True)
        self.assertEqual(config.log_level, "ERROR")
        self.assertEqual(config.timeout, 15.0)
        self.assertEqual(config.retry_count, 1)
    
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ["APP_NAME"] = "EnvApp"
        os.environ["API_KEY"] = "env_key"
        os.environ["DEBUG"] = "true"
        os.environ["LOG_LEVEL"] = "CRITICAL"
        os.environ["TIMEOUT"] = "10.5"
        os.environ["RETRY_COUNT"] = "7"
        
        # Create config loader with prefix
        env_config_loader = ConfigLoader(TestConfig, env_prefix="")
        config = env_config_loader.load_from_env()
        
        self.assertEqual(config.app_name, "EnvApp")
        self.assertEqual(config.api_key, "env_key")
        self.assertEqual(config.debug, True)
        self.assertEqual(config.log_level, "CRITICAL")
        self.assertEqual(config.timeout, 10.5)
        self.assertEqual(config.retry_count, 7)
    
    def test_merge_configs(self):
        """Test merging configurations from multiple sources."""
        # Base config with some values
        base_config = {
            "app_name": "BaseApp",
            "api_key": "base_key",
            "debug": False,
            "log_level": "INFO",
            "timeout": 30.0,
            "retry_count": 3
        }
        
        # Override config with some values
        override_config = {
            "app_name": "OverrideApp",
            "debug": True,
            "timeout": 45.0
        }
        
        # Expected merged result
        expected = {
            "app_name": "OverrideApp",  # Overridden
            "api_key": "base_key",       # From base
            "debug": True,               # Overridden
            "log_level": "INFO",         # From base
            "timeout": 45.0,             # Overridden
            "retry_count": 3             # From base
        }
        
        merged = self.config_loader.merge_configs(base_config, override_config)
        
        # Convert to TestConfig and verify
        config = TestConfig(**merged)
        
        self.assertEqual(config.app_name, expected["app_name"])
        self.assertEqual(config.api_key, expected["api_key"])
        self.assertEqual(config.debug, expected["debug"])
        self.assertEqual(config.log_level, expected["log_level"])
        self.assertEqual(config.timeout, expected["timeout"])
        self.assertEqual(config.retry_count, expected["retry_count"])
    
    def test_nested_config(self):
        """Test loading configuration with nested objects."""
        # Create a loader for nested config
        nested_loader = ConfigLoader(TestConfigWithNested)
        
        # Create a nested config file
        nested_config_path = os.path.join(self.temp_dir.name, "nested.yaml")
        with open(nested_config_path, "w") as f:
            f.write("""
app_name: NestedApp
database:
  host: localhost
  port: 5432
  username: user
  password: pass
services:
  auth:
    url: https://auth.example.com
    timeout: 10.0
  storage:
    url: https://storage.example.com
    region: us-west
            """)
        
        config = nested_loader.load_from_file(nested_config_path)
        
        self.assertEqual(config.app_name, "NestedApp")
        self.assertEqual(config.database["host"], "localhost")
        self.assertEqual(config.database["port"], 5432)
        self.assertEqual(config.services["auth"]["url"], "https://auth.example.com")
        self.assertEqual(config.services["storage"]["region"], "us-west")


if __name__ == "__main__":
    unittest.main()
