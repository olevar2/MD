"""
Script to verify the standardized modules.
"""
import os
import sys
import importlib.util
from pathlib import Path


def check_module_exists(module_path):
    """Check if a module exists."""
    return os.path.exists(module_path)


def print_module_content(module_path):
    """Print the content of a module."""
    try:
        with open(module_path, 'r') as f:
            content = f.read()
        print(f"Content of {module_path}:")
        print("-" * 80)
        print(content[:500] + "..." if len(content) > 500 else content)
        print("-" * 80)
        return True
    except Exception as e:
        print(f"Error reading {module_path}: {e}")
        return False


def verify_data_pipeline_service():
    """Verify the Data Pipeline Service standardized modules."""
    print("\n=== Verifying Data Pipeline Service ===\n")

    base_path = Path("D:/MD/forex_trading_platform/data-pipeline-service/data_pipeline_service")

    modules = [
        base_path / "config" / "standardized_config.py",
        base_path / "logging_setup_standardized.py",
        base_path / "service_clients_standardized.py",
        base_path / "database_standardized.py",
        base_path / "error_handling_standardized.py",
        base_path / "main.py",
        base_path / "api" / "v1" / "ohlcv.py"
    ]

    for module_path in modules:
        if check_module_exists(module_path):
            print(f"[OK] {module_path} exists")
            print_module_content(module_path)
        else:
            print(f"[ERROR] {module_path} does not exist")


def verify_ml_integration_service():
    """Verify the ML Integration Service standardized modules."""
    print("\n=== Verifying ML Integration Service ===\n")

    base_path = Path("D:/MD/forex_trading_platform/ml-integration-service/ml_integration_service")

    modules = [
        base_path / "config" / "standardized_config.py",
        base_path / "logging_setup_standardized.py",
        base_path / "service_clients_standardized.py",
        base_path / "error_handling_standardized.py",
        base_path / "main.py",
        base_path / "api" / "v1" / "health_api.py"
    ]

    for module_path in modules:
        if check_module_exists(module_path):
            print(f"[OK] {module_path} exists")
            print_module_content(module_path)
        else:
            print(f"[ERROR] {module_path} does not exist")


def main():
    """Main entry point."""
    print("Verifying standardized modules...")

    verify_data_pipeline_service()
    verify_ml_integration_service()

    print("\nVerification complete!")


if __name__ == "__main__":
    main()
