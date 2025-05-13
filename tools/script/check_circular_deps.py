#!/usr/bin/env python3
"""
Check for circular dependencies in the architecture analysis.
"""

import json
import os

# Load the architecture analysis
with open('d:/MD/forex_trading_platform/tools/output/architecture_analysis.json', 'r') as f:
    data = json.load(f)

# Get service dependencies
deps = data['service_dependencies']

# Check for circular dependencies
circular = []
for service, dependencies in deps.items():
    for dep in dependencies:
        if service in deps.get(dep, []):
            circular.append((service, dep))

print('Circular Dependencies:')
for service1, service2 in circular:
    print(f"  {service1} <-> {service2}")

# Check for duplicate services with different naming conventions
services = list(data['services'].keys())
service_names = {}
for service in services:
    # Normalize name (remove hyphens, underscores, convert to lowercase)
    normalized = service.lower().replace('-', '').replace('_', '')
    if normalized in service_names:
        print(f"Duplicate service names: {service} and {service_names[normalized]}")
    else:
        service_names[normalized] = service

# Check for naming convention inconsistencies
kebab_case = [s for s in services if '-' in s]
snake_case = [s for s in services if '_' in s]
camel_case = [s for s in services if not '-' in s and not '_' in s and any(c.isupper() for c in s)]

print(f"\nNaming Conventions:")
print(f"  Kebab-case services: {len(kebab_case)} ({', '.join(kebab_case)})")
print(f"  Snake_case services: {len(snake_case)} ({', '.join(snake_case)})")
print(f"  CamelCase services: {len(camel_case)} ({', '.join(camel_case)})")

# Check for common interfaces and adapters
common_lib_files = [f for f in data['module_dependencies'].keys() if 'common-lib' in f]
interfaces = [f for f in common_lib_files if 'interfaces' in f]
adapters = [f for f in common_lib_files if 'adapters' in f]

print(f"\nCommon Library:")
print(f"  Interface files: {len(interfaces)}")
print(f"  Adapter files: {len(adapters)}")

# Check for API endpoints by service
api_counts = {}
for service, service_data in data['services'].items():
    api_counts[service] = len(service_data['apis'])

print(f"\nAPI Endpoints by Service:")
for service, count in sorted(api_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {service}: {count} endpoints")