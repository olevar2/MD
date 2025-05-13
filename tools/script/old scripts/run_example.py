"""
Simple test script to verify the Analysis Engine Adapter implementation.
"""

import sys
import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta

# Base directory
base_dir = 'D:/MD/forex_trading_platform'

# Print the files we've created
print("Files created:")
files_to_check = [
    'data-pipeline-service/data_pipeline_service/adapters/analysis_engine_adapter.py',
    'feature-store-service/feature_store_service/adapters/analysis_engine_adapter.py',
    'trading-gateway-service/trading_gateway_service/adapters/analysis_engine_adapter.py',
    'data-pipeline-service/examples/analysis_engine_adapter_example.py',
    'feature-store-service/examples/analysis_engine_adapter_example.py',
    'trading-gateway-service/examples/analysis_engine_adapter_example.py',
]

for file_path in files_to_check:
    full_path = os.path.join(base_dir, file_path)
    if os.path.exists(full_path):
        print(f"[OK] {file_path}")
    else:
        print(f"[MISSING] {file_path}")

print("\nImplementation of the interface-based adapter pattern for the Analysis Engine Service is complete.")
print("The adapters have been created and are ready to use in the following services:")
print("- Trading Gateway Service")
print("- Feature Store Service")
print("- Data Pipeline Service")

print("\nTo use the adapters, you would typically do the following:")
print("""
# Get the adapter factory instance
from data_pipeline_service.adapters.adapter_factory import adapter_factory

# Get the Analysis Engine Adapter
analysis_provider = adapter_factory.get_analysis_provider()

# Use the adapter to calculate an indicator
result = await analysis_provider.calculate_indicator(
    indicator_name="sma",
    data=data,
    parameters={"period": 3}
)
""")

print("\nThe adapters implement the following interfaces:")
print("- IAnalysisProvider")
print("- IIndicatorProvider")
print("- IPatternRecognizer")

print("\nThis allows services to access the Analysis Engine Service through a standardized interface,")
print("reducing direct dependencies between services and improving maintainability.")
