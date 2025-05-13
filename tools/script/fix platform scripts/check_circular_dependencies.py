#!/usr/bin/env python3
"""
Check for Circular Dependencies

This script analyzes the dependency report and provides detailed information
about circular dependencies in the codebase, including specific modules involved.
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple

def load_report(report_path: str) -> Dict[str, Any]:
    """
    Load the dependency report.
    
    Args:
        report_path: Path to the dependency report
        
    Returns:
        Dependency report data
    """
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading report: {e}")
        sys.exit(1)

def find_circular_dependencies(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Find circular dependencies between services.
    
    Args:
        data: Dependency report data
        
    Returns:
        List of circular dependencies
    """
    service_dependencies = data.get('service_dependencies', {})
    circular = []
    
    for service, dependencies in service_dependencies.items():
        for dep in dependencies:
            if service in service_dependencies.get(dep, []):
                circular.append((service, dep))
    
    return circular

def find_module_dependencies(data: Dict[str, Any], service1: str, service2: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Find module dependencies between two services.
    
    Args:
        data: Dependency report data
        service1: First service
        service2: Second service
        
    Returns:
        Tuple of (service1 -> service2 dependencies, service2 -> service1 dependencies)
    """
    module_dependencies = data.get('module_dependencies', {})
    
    service1_to_service2 = []
    service2_to_service1 = []
    
    # Convert service names to module names (kebab-case to snake_case)
    service1_module = service1.replace('-', '_')
    service2_module = service2.replace('-', '_')
    
    for module, imports in module_dependencies.items():
        # Check if module belongs to service1
        if service1_module in module:
            for imp in imports:
                if service2_module in imp:
                    service1_to_service2.append((module, imp))
        
        # Check if module belongs to service2
        if service2_module in module:
            for imp in imports:
                if service1_module in imp:
                    service2_to_service1.append((module, imp))
    
    return service1_to_service2, service2_to_service1

def suggest_fixes(service1: str, service2: str, s1_to_s2: List[Tuple[str, str]], s2_to_s1: List[Tuple[str, str]]) -> List[str]:
    """
    Suggest fixes for circular dependencies.
    
    Args:
        service1: First service
        service2: Second service
        s1_to_s2: Service1 to Service2 dependencies
        s2_to_s1: Service2 to Service1 dependencies
        
    Returns:
        List of suggested fixes
    """
    suggestions = []
    
    # Check if the circular dependency is between a service and its alternate naming
    if service1.replace('-', '_') == service2 or service2.replace('-', '_') == service1:
        suggestions.append(f"This appears to be a naming issue. '{service1}' and '{service2}' seem to be the same service with different naming conventions.")
        suggestions.append(f"Standardize the service naming to use either kebab-case or snake_case consistently.")
        return suggestions
    
    # Group imports by module type
    s1_to_s2_types = defaultdict(list)
    s2_to_s1_types = defaultdict(list)
    
    for module, imp in s1_to_s2:
        module_type = imp.split('.')[-2] if len(imp.split('.')) > 1 else 'unknown'
        s1_to_s2_types[module_type].append((module, imp))
    
    for module, imp in s2_to_s1:
        module_type = imp.split('.')[-2] if len(imp.split('.')) > 1 else 'unknown'
        s2_to_s1_types[module_type].append((module, imp))
    
    # Suggest fixes based on module types
    if 'models' in s1_to_s2_types or 'models' in s2_to_s1_types:
        suggestions.append("Consider moving shared models to a common library.")
    
    if 'config' in s1_to_s2_types or 'config' in s2_to_s1_types:
        suggestions.append("Move shared configuration to a common configuration service or library.")
    
    if 'api' in s1_to_s2_types or 'api' in s2_to_s1_types:
        suggestions.append("Use interface-based adapters for API communication between services.")
    
    if 'client' in s1_to_s2_types or 'client' in s2_to_s1_types:
        suggestions.append("Implement client interfaces in a common library and provide concrete implementations in each service.")
    
    # General suggestions
    suggestions.append("Consider using an event-based communication pattern to decouple the services.")
    suggestions.append("Extract shared functionality into a common library that both services can depend on.")
    
    return suggestions

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check for circular dependencies")
    parser.add_argument(
        "--report-path",
        default="tools/output/dependency-report.json",
        help="Path to the dependency report"
    )
    parser.add_argument(
        "--output-path",
        default="tools/output/circular_dependencies_report.json",
        help="Path to save the circular dependencies report"
    )
    args = parser.parse_args()
    
    # Load report
    data = load_report(args.report_path)
    
    # Find circular dependencies
    circular = find_circular_dependencies(data)
    
    if not circular:
        print("No circular dependencies found.")
        return
    
    print(f"Found {len(circular)} circular dependencies:")
    
    # Analyze each circular dependency
    circular_report = []
    
    for service1, service2 in circular:
        print(f"\n{service1} <-> {service2}")
        
        # Find module dependencies
        s1_to_s2, s2_to_s1 = find_module_dependencies(data, service1, service2)
        
        # Print examples
        if s1_to_s2:
            print(f"  {service1} imports from {service2}:")
            for i, (module, imp) in enumerate(s1_to_s2[:5]):
                print(f"    {os.path.basename(module)} imports {imp}")
            if len(s1_to_s2) > 5:
                print(f"    ... and {len(s1_to_s2) - 5} more")
        
        if s2_to_s1:
            print(f"  {service2} imports from {service1}:")
            for i, (module, imp) in enumerate(s2_to_s1[:5]):
                print(f"    {os.path.basename(module)} imports {imp}")
            if len(s2_to_s1) > 5:
                print(f"    ... and {len(s2_to_s1) - 5} more")
        
        # Suggest fixes
        suggestions = suggest_fixes(service1, service2, s1_to_s2, s2_to_s1)
        
        print("  Suggested fixes:")
        for suggestion in suggestions:
            print(f"    - {suggestion}")
        
        # Add to report
        circular_report.append({
            'service1': service1,
            'service2': service2,
            'service1_to_service2': [{'module': m, 'import': i} for m, i in s1_to_s2[:10]],
            'service2_to_service1': [{'module': m, 'import': i} for m, i in s2_to_s1[:10]],
            'suggestions': suggestions
        })
    
    # Save report
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'circular_dependencies': circular_report
        }, f, indent=2)
    
    print(f"\nCircular dependencies report saved to {args.output_path}")

if __name__ == "__main__":
    main()
