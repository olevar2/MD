"""
Script to identify direct service calls that bypass the adapter layer.

This script scans the codebase for imports from other services or direct HTTP calls
to other service endpoints, which might indicate bypassing the adapter layer.
"""

import os
import re
import json
from typing import Dict, List, Set, Tuple

# Service names
SERVICE_NAMES = [
    "analysis-engine-service",
    "data-pipeline-service",
    "feature-store-service",
    "ml-integration-service",
    "ml-workbench-service",
    "monitoring-alerting-service",
    "portfolio-management-service",
    "strategy-execution-engine",
    "trading-gateway-service",
    "ui-service"
]

# Patterns to look for
IMPORT_PATTERN = re.compile(r'^\s*(?:from|import)\s+([\w\d_.-]+)')
HTTP_PATTERN = re.compile(r'(?:requests\.(?:get|post|put|delete)|http\.(?:get|post|put|delete))\s*\(\s*[\'"](?:https?://|/)([^/\'"]+)')
URL_PATTERN = re.compile(r'[\'"](?:https?://|/)([^/\'"]+)')

# Directories to skip
SKIP_DIRS = ['.git', '.venv', 'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist']

def is_service_import(import_name: str, current_service: str) -> bool:
    """Check if an import is from another service."""
    for service in SERVICE_NAMES:
        service_module = service.replace('-', '_')
        if service != current_service and (
            import_name.startswith(service) or 
            import_name.startswith(service_module)
        ):
            return True
    return False

def find_direct_service_calls(root_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Find direct service calls in the codebase.
    
    Args:
        root_dir: Root directory to scan
        
    Returns:
        Dictionary mapping service names to lists of files with direct service calls
    """
    results = {}
    
    for service in SERVICE_NAMES:
        service_dir = os.path.join(root_dir, service)
        if not os.path.exists(service_dir):
            continue
            
        service_results = {
            'direct_imports': [],
            'direct_http_calls': []
        }
        
        for dirpath, dirnames, filenames in os.walk(service_dir):
            # Skip directories
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            
            for filename in filenames:
                if not filename.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                    continue
                    
                file_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(file_path, root_dir)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # Check for direct imports
                        for line in content.split('\n'):
                            match = IMPORT_PATTERN.match(line)
                            if match:
                                import_name = match.group(1)
                                if is_service_import(import_name, service):
                                    service_results['direct_imports'].append({
                                        'file': rel_path,
                                        'line': line.strip(),
                                        'import': import_name
                                    })
                        
                        # Check for direct HTTP calls
                        http_matches = HTTP_PATTERN.findall(content)
                        url_matches = URL_PATTERN.findall(content)
                        
                        for match in http_matches + url_matches:
                            for service_name in SERVICE_NAMES:
                                service_host = service_name.replace('-', '')
                                if service_name != service and (
                                    match.startswith(service_name) or
                                    match.startswith(service_host) or
                                    match.endswith(service_name) or
                                    match.endswith(service_host)
                                ):
                                    service_results['direct_http_calls'].append({
                                        'file': rel_path,
                                        'url': match
                                    })
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
        
        if service_results['direct_imports'] or service_results['direct_http_calls']:
            results[service] = service_results
    
    return results

def main():
    """Main function."""
    root_dir = '.'
    results = find_direct_service_calls(root_dir)
    
    # Print results
    print("Direct Service Calls Analysis")
    print("============================")
    
    for service, service_results in results.items():
        print(f"\n{service}:")
        
        if service_results['direct_imports']:
            print("\n  Direct Imports:")
            for item in service_results['direct_imports']:
                print(f"    {item['file']}: {item['line']}")
        
        if service_results['direct_http_calls']:
            print("\n  Direct HTTP Calls:")
            for item in service_results['direct_http_calls']:
                print(f"    {item['file']}: {item['url']}")
    
    # Save results to file
    with open('direct_service_calls.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to direct_service_calls.json")

if __name__ == "__main__":
    main()
