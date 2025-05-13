#!/usr/bin/env python3
"""
Analyze the common library structure.
"""

import os
import json
from pathlib import Path

# Define the common library path
common_lib_path = Path("D:/MD/forex_trading_platform/common-lib")

# Check if the common library exists
if not common_lib_path.exists():
    print(f"Common library not found at {common_lib_path}")
    exit(1)

# Analyze the directory structure
def analyze_directory(path, prefix=""):
    """
    Analyze directory.
    
    Args:
        path: Description of path
        prefix: Description of prefix
    
    """

    result = []
    for item in sorted(os.listdir(path)):
        item_path = path / item
        if item_path.is_dir() and not item.startswith((".", "__")):
            result.append(f"{prefix}{item}/")
            result.extend(analyze_directory(item_path, prefix + "  "))
        elif item_path.is_file() and item.endswith(".py") and not item.startswith("__"):
            result.append(f"{prefix}{item}")
    return result

# Analyze interfaces and adapters
def analyze_interfaces_adapters():
    """
    Analyze interfaces adapters.
    
    """

    interfaces_path = common_lib_path / "common_lib" / "interfaces"
    adapters_path = common_lib_path / "common_lib" / "adapters"
    
    interfaces = []
    if interfaces_path.exists():
        print("\nInterfaces:")
        for item in sorted(os.listdir(interfaces_path)):
            if item.endswith(".py") and not item.startswith("__"):
                interfaces.append(item[:-3])  # Remove .py extension
                print(f"  {item[:-3]}")
    
    adapters = []
    if adapters_path.exists():
        print("\nAdapters:")
        for item in sorted(os.listdir(adapters_path)):
            if item.endswith(".py") and not item.startswith("__"):
                adapters.append(item[:-3])  # Remove .py extension
                print(f"  {item[:-3]}")
    
    # Check for adapter implementation completeness
    if interfaces and adapters:
        print("\nAdapter Implementation Completeness:")
        for interface in interfaces:
            if interface + "_adapter" in adapters:
                print(f"  {interface}: Implemented")
            else:
                print(f"  {interface}: Not implemented")

# Analyze error handling
def analyze_error_handling():
    """
    Analyze error handling.
    
    """

    errors_path = common_lib_path / "common_lib" / "errors"
    if not errors_path.exists():
        errors_path = common_lib_path / "common_lib" / "exceptions"
    
    if errors_path.exists():
        print("\nError Handling:")
        for item in sorted(os.listdir(errors_path)):
            if item.endswith(".py") and not item.startswith("__"):
                print(f"  {item[:-3]}")

# Analyze resilience patterns
def analyze_resilience():
    """
    Analyze resilience.
    
    """

    resilience_path = common_lib_path / "common_lib" / "resilience"
    if resilience_path.exists():
        print("\nResilience Patterns:")
        for item in sorted(os.listdir(resilience_path)):
            if item.endswith(".py") and not item.startswith("__"):
                print(f"  {item[:-3]}")

# Main analysis
print(f"Common Library Structure:")
structure = analyze_directory(common_lib_path)
for item in structure[:20]:  # Show only the first 20 items to avoid overwhelming output
    print(item)
if len(structure) > 20:
    print(f"... and {len(structure) - 20} more items")

# Analyze interfaces and adapters
analyze_interfaces_adapters()

# Analyze error handling
analyze_error_handling()

# Analyze resilience patterns
analyze_resilience()