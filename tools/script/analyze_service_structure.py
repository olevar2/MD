#!/usr/bin/env python3
"""
Analyze the service structure and integration.
"""

import os
import json
from pathlib import Path
import re

# Define the project root
project_root = Path("D:/MD/forex_trading_platform")

# Load the architecture analysis
with open(project_root / 'tools/output/architecture_analysis.json', 'r') as f:
    arch_data = json.load(f)

# Get services
services = arch_data['services']

# Analyze service structure
def analyze_service_structure(service_name):
    """
    Analyze service structure.
    
    Args:
        service_name: Description of service_name
    
    """

    service_data = services[service_name]
    service_path = Path(service_data['path'])
    
    # Check for key components
    components = {
        'api': False,
        'models': False,
        'services': False,
        'repositories': False,
        'config': False,
        'utils': False,
        'tests': False,
        'adapters': False
    }
    
    # Check directory structure
    for root, dirs, files in os.walk(service_path):
        root_path = Path(root)
        rel_path = root_path.relative_to(service_path)
        
        # Check for key directories
        for component in components:
            if component in str(rel_path).lower():
                components[component] = True
        
        # Check for adapter pattern implementation
        if 'adapters' in str(rel_path).lower():
            components['adapters'] = True
    
    return components

# Analyze service integration
def analyze_service_integration(service_name):
    """
    Analyze service integration.
    
    Args:
        service_name: Description of service_name
    
    """

    service_data = services[service_name]
    
    # Check for API endpoints
    api_count = len(service_data['apis'])
    
    # Check for events
    event_count = len(service_data['events'])
    
    # Check for database models
    db_model_count = len(service_data['database_models'])
    
    # Check for dependencies
    dependencies = arch_data['service_dependencies'].get(service_name, [])
    
    return {
        'api_count': api_count,
        'event_count': event_count,
        'db_model_count': db_model_count,
        'dependencies': dependencies
    }

# Analyze all services
print("Service Structure and Integration Analysis:")
print("===========================================")

for service_name in sorted(services.keys()):
    print(f"\nService: {service_name}")
    
    # Analyze structure
    structure = analyze_service_structure(service_name)
    print("  Structure:")
    for component, exists in structure.items():
        status = "Yes" if exists else "No"
        print(f"    {component}: {status}")
    
    # Analyze integration
    integration = analyze_service_integration(service_name)
    print("  Integration:")
    print(f"    API Endpoints: {integration['api_count']}")
    print(f"    Events: {integration['event_count']}")
    print(f"    Database Models: {integration['db_model_count']}")
    print(f"    Dependencies: {', '.join(integration['dependencies']) if integration['dependencies'] else 'None'}")

# Check for adapter pattern implementation across services
print("\nAdapter Pattern Implementation:")
print("===============================")

adapter_implementation = {}
for service_name, service_data in services.items():
    service_path = Path(service_data['path'])
    adapters_found = False
    
    # Check for adapters directory
    for root, dirs, files in os.walk(service_path):
        if 'adapters' in os.path.basename(root).lower():
            adapters_found = True
            break
    
    adapter_implementation[service_name] = adapters_found

# Print results
for service_name, implemented in adapter_implementation.items():
    status = "Implemented" if implemented else "Not implemented"
    print(f"  {service_name}: {status}")

# Check for error handling implementation across services
print("\nError Handling Implementation:")
print("==============================")

error_handling = {}
for service_name, service_data in services.items():
    service_path = Path(service_data['path'])
    error_handling_found = False
    
    # Check for error handling files
    for root, dirs, files in os.walk(service_path):
        for file in files:
            if file.endswith('.py') and ('error' in file.lower() or 'exception' in file.lower()):
                error_handling_found = True
                break
    
    error_handling[service_name] = error_handling_found

# Print results
for service_name, implemented in error_handling.items():
    status = "Implemented" if implemented else "Not implemented"
    print(f"  {service_name}: {status}")

# Check for resilience pattern implementation across services
print("\nResilience Pattern Implementation:")
print("==================================")

resilience_patterns = {}
for service_name, service_data in services.items():
    service_path = Path(service_data['path'])
    resilience_found = False
    
    # Check for resilience pattern files
    for root, dirs, files in os.walk(service_path):
        for file in files:
            if file.endswith('.py') and ('circuit' in file.lower() or 'retry' in file.lower() or 'timeout' in file.lower() or 'resilience' in file.lower()):
                resilience_found = True
                break
    
    resilience_patterns[service_name] = resilience_found

# Print results
for service_name, implemented in resilience_patterns.items():
    status = "Implemented" if implemented else "Not implemented"
    print(f"  {service_name}: {status}")