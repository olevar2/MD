"""
Test script to verify that the deployed modules can be imported correctly.
"""
import os
import sys
import importlib
from pathlib import Path


def test_import(module_path, module_name):
    """Test importing a module."""
    try:
        # Add the parent directory to sys.path
        parent_dir = str(Path(module_path).parent.parent)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        # Import the module
        module = importlib.import_module(module_name)
        print(f"[OK] Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to import {module_name}: {e}")
        return False


def test_data_pipeline_service():
    """Test importing modules from the Data Pipeline Service."""
    print("\n=== Testing Data Pipeline Service Imports ===\n")
    
    base_path = "D:/MD/forex_trading_platform/data-pipeline-service"
    
    modules = [
        (f"{base_path}/data_pipeline_service/config/config.py", "data_pipeline_service.config.config"),
        (f"{base_path}/data_pipeline_service/logging_setup.py", "data_pipeline_service.logging_setup"),
        (f"{base_path}/data_pipeline_service/service_clients.py", "data_pipeline_service.service_clients"),
        (f"{base_path}/data_pipeline_service/database.py", "data_pipeline_service.database"),
        (f"{base_path}/data_pipeline_service/error_handling.py", "data_pipeline_service.error_handling")
    ]
    
    success = True
    for module_path, module_name in modules:
        if not test_import(module_path, module_name):
            success = False
    
    return success


def test_ml_integration_service():
    """Test importing modules from the ML Integration Service."""
    print("\n=== Testing ML Integration Service Imports ===\n")
    
    base_path = "D:/MD/forex_trading_platform/ml-integration-service"
    
    modules = [
        (f"{base_path}/ml_integration_service/config/config.py", "ml_integration_service.config.config"),
        (f"{base_path}/ml_integration_service/logging_setup.py", "ml_integration_service.logging_setup"),
        (f"{base_path}/ml_integration_service/service_clients.py", "ml_integration_service.service_clients"),
        (f"{base_path}/ml_integration_service/error_handling.py", "ml_integration_service.error_handling")
    ]
    
    success = True
    for module_path, module_name in modules:
        if not test_import(module_path, module_name):
            success = False
    
    return success


def main():
    """Main entry point."""
    print("Testing imports of deployed modules...")
    
    data_pipeline_success = test_data_pipeline_service()
    ml_integration_success = test_ml_integration_service()
    
    if data_pipeline_success and ml_integration_success:
        print("\n=== All imports successful! ===")
        return 0
    else:
        print("\n=== Some imports failed! ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
