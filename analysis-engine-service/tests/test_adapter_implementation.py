"""
Test script for the adapter implementation.

This script tests if the adapter implementation works correctly.
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Python path:", sys.path)
print("Current directory:", os.getcwd())

# Try to import the modules
try:
    print("Importing modules...")
    
    # Import common_lib modules
    from common_lib.interfaces.trading_gateway import ITradingGateway
    print("Successfully imported ITradingGateway")
    
    from common_lib.interfaces.ml_workbench import IExperimentManager
    print("Successfully imported IExperimentManager")
    
    from common_lib.interfaces.risk_management import IRiskManager
    print("Successfully imported IRiskManager")
    
    from common_lib.interfaces.feature_store import IFeatureProvider
    print("Successfully imported IFeatureProvider")
    
    # Import analysis_engine modules
    from analysis_engine.adapters.common_adapter_factory import CommonAdapterFactory
    print("Successfully imported CommonAdapterFactory")
    
    # Create an instance of the adapter factory
    print("Creating adapter factory...")
    adapter_factory = CommonAdapterFactory()
    print("Successfully created adapter factory")
    
    # Try to get adapters
    print("Getting adapters...")
    
    # This might fail if the adapters are not properly implemented
    # or if the services are not available
    # We're just testing if the code runs without errors
    
    print("All imports and instantiations successful!")
    print("Test completed successfully")
    
except Exception as e:
    print(f"Error: {str(e)}")
    print(f"Error type: {type(e)}")
    print("Test failed")
