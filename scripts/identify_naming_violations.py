#!/usr/bin/env python3
"""
Script to identify naming convention violations in the codebase.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Constants
SNAKE_CASE_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')
KEBAB_CASE_PATTERN = re.compile(r'^[a-z][a-z0-9-]*$')
CAMEL_CASE_PATTERN = re.compile(r'^[a-z][a-zA-Z0-9]*$')
PASCAL_CASE_PATTERN = re.compile(r'^[A-Z][a-zA-Z0-9]*$')

# Directories to exclude from analysis
EXCLUDE_DIRS = {
    '.git', '.github', '.vscode', '.pytest_cache', '__pycache__', 
    'node_modules', 'venv', '.venv', 'corrupted_backups'
}

# File extensions to check
PYTHON_EXTENSIONS = {'.py'}
SERVICE_DIRS = {
    'analysis-engine-service', 'data-pipeline-service', 'feature-store-service',
    'ml-integration-service', 'ml-workbench-service', 'monitoring-alerting-service',
    'portfolio-management-service', 'strategy-execution-engine', 'trading-gateway-service',
    'ui-service', 'api-gateway', 'model-registry-service', 'data-management-service'
}

def is_snake_case(name: str) -> bool:
    """Check if a name follows snake_case convention."""
    return bool(SNAKE_CASE_PATTERN.match(name))

def is_kebab_case(name: str) -> bool:
    """Check if a name follows kebab-case convention."""
    return bool(KEBAB_CASE_PATTERN.match(name))

def is_camel_case(name: str) -> bool:
    """Check if a name follows camelCase convention."""
    return bool(CAMEL_CASE_PATTERN.match(name))

def is_pascal_case(name: str) -> bool:
    """Check if a name follows PascalCase convention."""
    return bool(PASCAL_CASE_PATTERN.match(name))

def to_snake_case(name: str) -> str:
    """Convert a name to snake_case."""
    # Handle kebab-case
    name = name.replace('-', '_')
    
    # Handle camelCase and PascalCase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def to_kebab_case(name: str) -> str:
    """Convert a name to kebab-case."""
    # First convert to snake_case
    snake = to_snake_case(name)
    # Then replace underscores with hyphens
    return snake.replace('_', '-')

def find_python_files(root_dir: str) -> List[str]:
    """Find all Python files in the given directory."""
    python_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        
        for filename in filenames:
            if Path(filename).suffix in PYTHON_EXTENSIONS:
                python_files.append(os.path.join(dirpath, filename))
    
    return python_files

def find_service_dirs(root_dir: str) -> List[str]:
    """Find all service directories in the given directory."""
    service_dirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.') and item not in EXCLUDE_DIRS:
            service_dirs.append(item_path)
    
    return service_dirs

def check_python_file_naming(file_path: str) -> Dict:
    """Check if a Python file follows snake_case naming convention."""
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    
    if not is_snake_case(name):
        return {
            'file': file_path,
            'current_name': name,
            'suggested_name': to_snake_case(name),
            'type': 'file'
        }
    
    return None

def check_service_dir_naming(dir_path: str) -> Dict:
    """Check if a service directory follows kebab-case naming convention."""
    dirname = os.path.basename(dir_path)
    
    if not is_kebab_case(dirname):
        return {
            'directory': dir_path,
            'current_name': dirname,
            'suggested_name': to_kebab_case(dirname),
            'type': 'directory'
        }
    
    return None

def main():
    """Main function to identify naming convention violations."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Find all Python files
    python_files = find_python_files(root_dir)
    
    # Find all service directories
    service_dirs = find_service_dirs(root_dir)
    
    # Check Python file naming
    file_violations = []
    for file_path in python_files:
        violation = check_python_file_naming(file_path)
        if violation:
            file_violations.append(violation)
    
    # Check service directory naming
    dir_violations = []
    for dir_path in service_dirs:
        violation = check_service_dir_naming(dir_path)
        if violation:
            dir_violations.append(violation)
    
    # Combine violations
    violations = {
        'file_violations': file_violations,
        'directory_violations': dir_violations,
        'stats': {
            'total_files_checked': len(python_files),
            'total_directories_checked': len(service_dirs),
            'file_violations_found': len(file_violations),
            'directory_violations_found': len(dir_violations)
        }
    }
    
    # Write violations to file
    with open(os.path.join(root_dir, 'naming_violations.json'), 'w') as f:
        json.dump(violations, f, indent=2)
    
    print(f"Found {len(file_violations)} file naming violations and {len(dir_violations)} directory naming violations.")
    print(f"Results written to {os.path.join(root_dir, 'naming_violations.json')}")

if __name__ == '__main__':
    main()
