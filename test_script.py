"""
Test script to verify the Analysis Engine Adapter implementation.
"""

import sys
import os

# Add the necessary directories to the Python path
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./data-pipeline-service'))
sys.path.append(os.path.abspath('./feature-store-service'))
sys.path.append(os.path.abspath('./trading-gateway-service'))
sys.path.append(os.path.abspath('./common-lib'))

# Try to import the adapter factory from each service
try:
    from data_pipeline_service.adapters.adapter_factory import adapter_factory as dp_adapter_factory
    print("Successfully imported data_pipeline_service.adapters.adapter_factory")
except ImportError as e:
    print(f"Failed to import data_pipeline_service.adapters.adapter_factory: {e}")

try:
    from feature_store_service.adapters.adapter_factory import adapter_factory as fs_adapter_factory
    print("Successfully imported feature_store_service.adapters.adapter_factory")
except ImportError as e:
    print(f"Failed to import feature_store_service.adapters.adapter_factory: {e}")

try:
    from trading_gateway_service.adapters.adapter_factory import adapter_factory as tg_adapter_factory
    print("Successfully imported trading_gateway_service.adapters.adapter_factory")
except ImportError as e:
    print(f"Failed to import trading_gateway_service.adapters.adapter_factory: {e}")

# Try to import the analysis engine adapter from each service
try:
    from data_pipeline_service.adapters.analysis_engine_adapter import AnalysisEngineAdapter as DPAnalysisEngineAdapter
    print("Successfully imported data_pipeline_service.adapters.analysis_engine_adapter.AnalysisEngineAdapter")
except ImportError as e:
    print(f"Failed to import data_pipeline_service.adapters.analysis_engine_adapter.AnalysisEngineAdapter: {e}")

try:
    from feature_store_service.adapters.analysis_engine_adapter import AnalysisEngineAdapter as FSAnalysisEngineAdapter
    print("Successfully imported feature_store_service.adapters.analysis_engine_adapter.AnalysisEngineAdapter")
except ImportError as e:
    print(f"Failed to import feature_store_service.adapters.analysis_engine_adapter.AnalysisEngineAdapter: {e}")

try:
    from trading_gateway_service.adapters.analysis_engine_adapter import AnalysisEngineAdapter as TGAnalysisEngineAdapter
    print("Successfully imported trading_gateway_service.adapters.analysis_engine_adapter.AnalysisEngineAdapter")
except ImportError as e:
    print(f"Failed to import trading_gateway_service.adapters.analysis_engine_adapter.AnalysisEngineAdapter: {e}")

# Print the Python path
print("\nPython path:")
for path in sys.path:
    print(f"  {path}")

# Print the current directory
print(f"\nCurrent directory: {os.getcwd()}")

# List the directories in the current directory
print("\nDirectories in the current directory:")
for item in os.listdir('.'):
    if os.path.isdir(item):
        print(f"  {item}")

# List the files in the data-pipeline-service/adapters directory
print("\nFiles in data-pipeline-service/adapters directory:")
try:
    for item in os.listdir('data-pipeline-service/data_pipeline_service/adapters'):
        print(f"  {item}")
except FileNotFoundError:
    print("  Directory not found")

# List the files in the feature-store-service/adapters directory
print("\nFiles in feature-store-service/adapters directory:")
try:
    for item in os.listdir('feature-store-service/feature_store_service/adapters'):
        print(f"  {item}")
except FileNotFoundError:
    print("  Directory not found")

# List the files in the trading-gateway-service/adapters directory
print("\nFiles in trading-gateway-service/adapters directory:")
try:
    for item in os.listdir('trading-gateway-service/trading_gateway_service/adapters'):
        print(f"  {item}")
except FileNotFoundError:
    print("  Directory not found")
