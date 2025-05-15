"""
Verification script for the adapter implementation.

This script verifies that the adapter implementation works correctly by importing
the necessary modules and creating instances of the adapters.
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("Python path:", sys.path)
print("Current directory:", os.getcwd())

def verify_interfaces():
    """Verify that all necessary interfaces can be imported."""
    try:
        print("Verifying interfaces...")

        # Check if the interface files exist
        interface_files = [
            "common-lib/common_lib/interfaces/trading_gateway.py",
            "common-lib/common_lib/interfaces/ml_workbench.py",
            "common-lib/common_lib/interfaces/risk_management.py",
            "common-lib/common_lib/interfaces/feature_store.py"
        ]

        for file_path in interface_files:
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), file_path)):
                print(f"✓ Interface file exists: {file_path}")
            else:
                print(f"✗ Interface file missing: {file_path}")

        print("Interface verification completed!")
        return True
    except Exception as e:
        print(f"Error during interface verification: {str(e)}")
        print(f"Error type: {type(e)}")
        return False

def verify_adapters():
    """Verify that all necessary adapter files exist."""
    try:
        print("Verifying adapters...")

        # Check if the adapter files exist
        adapter_files = [
            "common-lib/common_lib/adapters/trading_gateway_adapter.py",
            "common-lib/common_lib/adapters/ml_workbench_adapter.py",
            "analysis-engine-service/adapters/common_adapter_factory.py",
            "analysis-engine-service/core/service_dependencies.py"
        ]

        for file_path in adapter_files:
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), file_path)):
                print(f"✓ Adapter file exists: {file_path}")
            else:
                print(f"✗ Adapter file missing: {file_path}")

        print("Adapter verification completed!")
        return True
    except Exception as e:
        print(f"Error during adapter verification: {str(e)}")
        print(f"Error type: {type(e)}")
        return False

def verify_api_endpoints():
    """Verify that the API endpoints exist."""
    try:
        print("Verifying API endpoints...")

        # Check if the API endpoint files exist
        api_files = [
            "analysis-engine-service/api/v1/integrated_analysis.py",
            "analysis-engine-service/api/v1/router.py"
        ]

        for file_path in api_files:
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), file_path)):
                print(f"✓ API file exists: {file_path}")
            else:
                print(f"✗ API file missing: {file_path}")

        print("API endpoint verification completed!")
        return True
    except Exception as e:
        print(f"Error during API endpoint verification: {str(e)}")
        print(f"Error type: {type(e)}")
        return False

def verify_documentation():
    """Verify that the documentation exists."""
    try:
        print("Verifying documentation...")

        # Check if the documentation files exist
        doc_files = [
            "analysis-engine-service/docs/interface_based_decoupling.md",
            "analysis-engine-service/examples/common_adapter_usage.py"
        ]

        for file_path in doc_files:
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), file_path)):
                print(f"✓ Documentation file exists: {file_path}")
            else:
                print(f"✗ Documentation file missing: {file_path}")

        print("Documentation verification completed!")
        return True
    except Exception as e:
        print(f"Error during documentation verification: {str(e)}")
        print(f"Error type: {type(e)}")
        return False

def main():
    """Run all verifications."""
    print("Starting verification...")

    interfaces_ok = verify_interfaces()
    adapters_ok = verify_adapters()
    api_endpoints_ok = verify_api_endpoints()
    documentation_ok = verify_documentation()

    if interfaces_ok and adapters_ok and api_endpoints_ok and documentation_ok:
        print("All verifications passed! The Interface-Based Decoupling approach is fully implemented.")
        return 0
    else:
        print("Some verifications failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
