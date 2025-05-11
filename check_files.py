"""
Check if the files exist.
"""

import os

# Base directory
base_dir = 'D:/MD/forex_trading_platform'

# Check if the files exist
files_to_check = [
    'data-pipeline-service/data_pipeline_service/adapters/analysis_engine_adapter.py',
    'feature-store-service/feature_store_service/adapters/analysis_engine_adapter.py',
    'trading-gateway-service/trading_gateway_service/adapters/analysis_engine_adapter.py',
    'data-pipeline-service/examples/analysis_engine_adapter_example.py',
    'feature-store-service/examples/analysis_engine_adapter_example.py',
    'trading-gateway-service/examples/analysis_engine_adapter_example.py',
]

print(f"Base directory: {base_dir}")
print(f"Current directory: {os.getcwd()}")

for file_path in files_to_check:
    full_path = os.path.join(base_dir, file_path)
    if os.path.exists(full_path):
        print(f"File exists: {file_path}")
        # Print the first few lines of the file
        try:
            with open(full_path, 'r') as f:
                first_lines = ''.join(f.readlines()[:5])
                print(f"First few lines:\n{first_lines}")
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print(f"File does not exist: {file_path}")
