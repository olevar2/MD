"""
Simple test script for the configuration management module.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the configuration management module
from common_lib.config import (
    ConfigManager,
    DictConfigSource,
    EnvVarConfigSource,
    JsonFileConfigSource,
)
from common_lib.errors import ConfigurationError


def test_dict_config_source():
    """Test the dictionary configuration source."""
    print("Testing dictionary configuration source...")

    # Create a dictionary configuration source
    config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "postgres",
            "password": "postgres"
        },
        "api": {
            "port": 8000,
            "debug": True
        }
    }
    source = DictConfigSource(config)

    # Test get method
    assert source.get("database.host") == "localhost"
    assert source.get("database.port") == 5432
    assert source.get("api.debug") is True
    assert source.get("nonexistent") is None
    assert source.get("nonexistent", "default") == "default"

    # Test has method
    assert source.has("database.host") is True
    assert source.has("nonexistent") is False

    # Test get_all method
    assert source.get_all() == config

    print("Dictionary configuration source test passed!")
    return True


def test_env_var_config_source():
    """Test the environment variable configuration source."""
    print("Testing environment variable configuration source...")

    # Set environment variables for testing
    os.environ["TEST_DB_HOST"] = "localhost"
    os.environ["TEST_DB_PORT"] = "5432"
    os.environ["TEST_API_DEBUG"] = "true"

    # Create an environment variable configuration source
    source = EnvVarConfigSource(prefix="TEST_")

    # Test get method
    assert source.get("db_host") == "localhost"
    assert source.get("db_port") == 5432  # Should be parsed as int
    assert source.get("api_debug") is True  # Should be parsed as bool
    assert source.get("nonexistent") is None
    assert source.get("nonexistent", "default") == "default"

    # Test has method
    assert source.has("db_host") is True
    assert source.has("nonexistent") is False

    # Clean up environment variables
    del os.environ["TEST_DB_HOST"]
    del os.environ["TEST_DB_PORT"]
    del os.environ["TEST_API_DEBUG"]

    print("Environment variable configuration source test passed!")
    return True


def test_json_file_config_source():
    """Test the JSON file configuration source."""
    print("Testing JSON file configuration source...")

    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "postgres",
                "password": "postgres"
            },
            "api": {
                "port": 8000,
                "debug": True
            }
        }, f)

    try:
        # Create a JSON file configuration source
        source = JsonFileConfigSource(f.name)

        # Test get method
        assert source.get("database.host") == "localhost"
        assert source.get("database.port") == 5432
        assert source.get("api.debug") is True
        assert source.get("nonexistent") is None
        assert source.get("nonexistent", "default") == "default"

        # Test has method
        assert source.has("database.host") is True
        assert source.has("nonexistent") is False

        # Test get_all method
        assert source.get_all() == {
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "postgres",
                "password": "postgres"
            },
            "api": {
                "port": 8000,
                "debug": True
            }
        }
    finally:
        # Clean up temporary file
        os.unlink(f.name)

    print("JSON file configuration source test passed!")
    return True


def test_config_manager():
    """Test the configuration manager."""
    print("Testing configuration manager...")

    # Create configuration sources
    dict_source = DictConfigSource({
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "postgres",
            "password": "postgres"
        },
        "api": {
            "port": 8000,
            "debug": True
        }
    })

    # Set environment variables for testing
    os.environ["TEST_DB_HOST"] = "db.example.com"
    os.environ["TEST_API_PORT"] = "9000"

    env_source = EnvVarConfigSource(prefix="TEST_")

    # Create a configuration manager with sources
    # Note: env_source has higher precedence than dict_source
    manager = ConfigManager([env_source, dict_source])

    # Test get method
    assert manager.get("db_host") == "db.example.com"  # From env_source
    assert manager.get("database.port") == 5432  # From dict_source
    assert manager.get("api_port") == 9000  # From env_source
    assert manager.get("api.debug") is True  # From dict_source
    assert manager.get("nonexistent") is None
    assert manager.get("nonexistent", "default") == "default"

    # Test has method
    assert manager.has("db_host") is True
    assert manager.has("database.port") is True
    assert manager.has("nonexistent") is False

    # Test get_all method
    all_config = manager.get_all()
    assert all_config["db_host"] == "db.example.com"
    assert all_config["database"]["port"] == 5432
    assert all_config["api_port"] == 9000
    assert all_config["api"]["debug"] is True

    # Test type-specific getters
    assert manager.get_int("api_port") == 9000
    assert manager.get_bool("api.debug") is True
    assert manager.get_float("database.port") == 5432.0

    # Test require method
    assert manager.require("db_host") == "db.example.com"
    try:
        manager.require("nonexistent")
        assert False, "require() should raise ConfigurationError for nonexistent keys"
    except ConfigurationError:
        pass

    # Test type-specific require methods
    assert manager.require_int("api_port") == 9000
    assert manager.require_bool("api.debug") is True
    assert manager.require_float("database.port") == 5432.0

    # Clean up environment variables
    del os.environ["TEST_DB_HOST"]
    del os.environ["TEST_API_PORT"]

    print("Configuration manager test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("Running all configuration management tests...")

    test_dict_config_source()
    test_env_var_config_source()
    test_json_file_config_source()
    test_config_manager()

    print("All configuration management tests passed!")
    return True


if __name__ == "__main__":
    run_all_tests()