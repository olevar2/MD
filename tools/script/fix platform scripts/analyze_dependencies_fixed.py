#!/usr/bin/env python3
"""
Analyze Dependencies in the Forex Trading Platform

This script analyzes the dependencies between services in the forex trading platform.
It identifies direct dependencies, circular dependencies, and generates a dependency report.

Usage:
    python analyze_dependencies_fixed.py [--output-file OUTPUT_FILE]

Options:
    --output-file OUTPUT_FILE    Output file for the dependency report (default: dependency-report.json)
"""

import os
import sys
import json
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Root directory of the forex trading platform
ROOT_DIR = "D:/MD/forex_trading_platform"

# Service directories to analyze
SERVICE_DIRS = [
    'analysis-engine-service',
    'api-gateway',
    'data-management-service',
    'data-pipeline-service',
    'feature-store-service',
    'ml-integration-service',
    'ml-workbench-service',
    'model-registry-service',
    'monitoring-alerting-service',
    'portfolio-management-service',
    'risk-management-service',
    'strategy-execution-engine',
    'trading-gateway-service',
    'ui-service',
    'common-lib',
    'common-js-lib'
]

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in a directory recursively."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_imports(file_path: str) -> List[str]:
    """Extract import statements from a Python file."""
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract import statements
            import_pattern = r'^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)'
            imports = re.findall(import_pattern, content, re.MULTILINE)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return imports

def map_imports_to_services(imports: List[str], service_dirs: List[str]) -> Set[str]:
    """Map import statements to service directories."""
    service_imports = set()
    for imp in imports:
        for service in service_dirs:
            # Convert kebab-case to snake_case for import comparison
            service_snake = service.replace('-', '_')
            if imp.startswith(service_snake) or imp == service_snake:
                service_imports.add(service)
                break
    return service_imports

def analyze_service_dependencies() -> Dict[str, List[str]]:
    """Analyze dependencies between services."""
    dependencies = defaultdict(set)
    
    for service in SERVICE_DIRS:
        service_dir = os.path.join(ROOT_DIR, service)
        if not os.path.isdir(service_dir):
            print(f"Warning: Service directory {service_dir} not found")
            continue
        
        python_files = find_python_files(service_dir)
        for file in python_files:
            imports = extract_imports(file)
            service_imports = map_imports_to_services(imports, SERVICE_DIRS)
            
            # Remove self-imports
            if service in service_imports:
                service_imports.remove(service)
            
            dependencies[service].update(service_imports)
    
    # Convert sets to lists for JSON serialization
    return {k: list(v) for k, v in dependencies.items()}

def find_circular_dependencies(dependencies: Dict[str, List[str]]) -> List[List[str]]:
    """Find circular dependencies in the dependency graph."""
    circular_deps = []
    
    def dfs(node, path, visited):
        if node in path:
            # Found a cycle
            cycle_start = path.index(node)
            circular_deps.append(path[cycle_start:] + [node])
            return
        
        if node in visited:
            return
        
        visited.add(node)
        path.append(node)
        
        for neighbor in dependencies.get(node, []):
            dfs(neighbor, path.copy(), visited)
    
    for node in dependencies:
        dfs(node, [], set())
    
    # Remove duplicate cycles
    unique_cycles = []
    for cycle in circular_deps:
        # Normalize cycle by rotating to start with the smallest element
        min_idx = cycle.index(min(cycle))
        normalized = cycle[min_idx:] + cycle[:min_idx]
        if normalized not in unique_cycles:
            unique_cycles.append(normalized)
    
    return unique_cycles

def generate_dependency_report(dependencies: Dict[str, List[str]], circular_deps: List[List[str]]) -> Dict:
    """Generate a comprehensive dependency report."""
    # Count incoming dependencies
    incoming_deps = defaultdict(int)
    for service, deps in dependencies.items():
        for dep in deps:
            incoming_deps[dep] += 1
    
    # Identify services with most dependencies
    most_deps = sorted(dependencies.items(), key=lambda x: len(x[1]), reverse=True)
    most_deps = [(service, deps) for service, deps in most_deps if deps]
    
    # Identify most depended-on services
    most_depended = sorted(incoming_deps.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "dependencies": dependencies,
        "circular_dependencies": circular_deps,
        "services_with_most_dependencies": [
            {"service": service, "dependencies": deps, "count": len(deps)}
            for service, deps in most_deps[:5]  # Top 5
        ],
        "most_depended_on_services": [
            {"service": service, "count": count}
            for service, count in most_depended[:5]  # Top 5
        ],
        "total_services": len(dependencies),
        "total_dependencies": sum(len(deps) for deps in dependencies.values()),
        "services_with_circular_dependencies": list(set(sum(circular_deps, [])))
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze dependencies in the forex trading platform')
    parser.add_argument('--output-file', default='dependency-report.json', help='Output file for the dependency report')
    args = parser.parse_args()
    
    print("Analyzing service dependencies...")
    dependencies = analyze_service_dependencies()
    
    print("Finding circular dependencies...")
    circular_deps = find_circular_dependencies(dependencies)
    
    print("Generating dependency report...")
    report = generate_dependency_report(dependencies, circular_deps)
    
    # Save report to file
    output_path = os.path.join(ROOT_DIR, 'tools', 'output', args.output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Dependency report saved to {output_path}")
    
    # Print summary
    print("\nDependency Analysis Summary:")
    print(f"Total services: {report['total_services']}")
    print(f"Total dependencies: {report['total_dependencies']}")
    print(f"Circular dependencies: {len(report['circular_dependencies'])}")
    
    if report['circular_dependencies']:
        print("\nCircular Dependencies:")
        for i, cycle in enumerate(report['circular_dependencies'], 1):
            print(f"{i}. {' -> '.join(cycle)} -> {cycle[0]}")
    
    print("\nServices with Most Dependencies:")
    for item in report['services_with_most_dependencies']:
        print(f"{item['service']}: {item['count']} dependencies")
    
    print("\nMost Depended-on Services:")
    for item in report['most_depended_on_services']:
        print(f"{item['service']}: {item['count']} dependents")

if __name__ == "__main__":
    main()