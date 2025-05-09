#!/usr/bin/env python
"""
Check Feature Store Tests Dependency

This script checks for circular dependencies between feature-store-service and its tests.
"""
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Regular expressions for import statements
IMPORT_PATTERNS = [
    r'^\s*import\s+([a-zA-Z0-9_.,\s]+)(?:\s+as\s+[a-zA-Z0-9_]+)?',  # import module
    r'^\s*from\s+([a-zA-Z0-9_.]+)\s+import\s+',  # from module import ...
]


def find_python_files(directory: str) -> List[str]:
    """
    Find all Python files in a directory recursively.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of Python file paths
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def extract_imports(file_path: str) -> List[str]:
    """
    Extract import statements from a Python file.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        List of imported module names
    """
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split into lines and remove comments
        lines = []
        for line in content.split('\n'):
            if '#' in line:
                line = line[:line.index('#')]
            lines.append(line)
        
        # Join lines with line continuations
        joined_lines = []
        current_line = ''
        for line in lines:
            if line.endswith('\\'):
                current_line += line[:-1].strip() + ' '
            else:
                current_line += line.strip()
                if current_line:
                    joined_lines.append(current_line)
                current_line = ''
        
        # Extract imports using regex patterns
        for line in joined_lines:
            for pattern in IMPORT_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    modules = match.group(1).split(',')
                    for module in modules:
                        module = module.strip()
                        if module:
                            imports.append(module)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    
    return imports


def check_feature_store_tests_dependency(root_dir: str) -> Dict:
    """
    Check for circular dependencies between feature-store-service and its tests.
    
    Args:
        root_dir: Root directory of the codebase
        
    Returns:
        Dictionary with dependency information
    """
    feature_store_dir = os.path.join(root_dir, 'feature-store-service')
    feature_store_tests_dir = os.path.join(feature_store_dir, 'tests')
    
    # Find all Python files
    feature_store_files = find_python_files(feature_store_dir)
    feature_store_tests_files = [f for f in feature_store_files if f.startswith(feature_store_tests_dir)]
    feature_store_main_files = [f for f in feature_store_files if not f.startswith(feature_store_tests_dir)]
    
    # Extract imports
    tests_imports = []
    for file_path in feature_store_tests_files:
        imports = extract_imports(file_path)
        for module in imports:
            if module.startswith('feature_store_service'):
                tests_imports.append((file_path, module))
    
    main_imports = []
    for file_path in feature_store_main_files:
        imports = extract_imports(file_path)
        for module in imports:
            if module.startswith('tests'):
                main_imports.append((file_path, module))
    
    # Check for circular dependencies
    circular_deps = []
    for main_file, main_import in main_imports:
        for test_file, test_import in tests_imports:
            if main_import.startswith('tests') and test_import.startswith('feature_store_service'):
                # Check if the test file imports from the main file
                main_module = os.path.relpath(main_file, feature_store_dir).replace(os.sep, '.').replace('.py', '')
                if test_import == main_module or test_import.startswith(main_module + '.'):
                    circular_deps.append({
                        'main_file': main_file,
                        'main_import': main_import,
                        'test_file': test_file,
                        'test_import': test_import
                    })
    
    return {
        'feature_store_files': len(feature_store_main_files),
        'feature_store_tests_files': len(feature_store_tests_files),
        'tests_importing_feature_store': len(tests_imports),
        'feature_store_importing_tests': len(main_imports),
        'circular_dependencies': circular_deps,
        'tests_imports': tests_imports,
        'main_imports': main_imports
    }


def main():
    root_dir = '.'
    output_file = 'tools/reports/feature_store_tests_dependency_report.json'
    
    print("Checking feature-store-service and tests dependencies...")
    result = check_feature_store_tests_dependency(root_dir)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the report to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"Report generated: {output_file}")
    print(f"Feature store files: {result['feature_store_files']}")
    print(f"Feature store tests files: {result['feature_store_tests_files']}")
    print(f"Tests importing feature_store: {result['tests_importing_feature_store']}")
    print(f"Feature store importing tests: {result['feature_store_importing_tests']}")
    print(f"Circular dependencies: {len(result['circular_dependencies'])}")
    
    if result['circular_dependencies']:
        print("\nCircular Dependencies:")
        for i, dep in enumerate(result['circular_dependencies'], 1):
            print(f"{i}. {dep['main_file']} imports {dep['main_import']}")
            print(f"   {dep['test_file']} imports {dep['test_import']}")
    else:
        print("\nNo circular dependencies found!")


if __name__ == "__main__":
    main()
