#!/usr/bin/env python3
"""
Script to implement documentation improvements across the codebase.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

def find_python_files(root_dir: str) -> List[str]:
    """Find all Python files in the given directory."""
    python_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']

        for filename in filenames:
            if filename.endswith('.py'):
                python_files.append(os.path.join(dirpath, filename))

    return python_files

def find_functions_without_docstrings(file_path: str) -> List[Dict]:
    """Find functions without docstrings in a Python file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Find function definitions
    functions_without_docstrings = []

    # Pattern to match function definitions
    function_pattern = r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?\s*:'

    for match in re.finditer(function_pattern, content, re.DOTALL):
        function_name = match.group(1)
        function_args = match.group(2)
        function_return = match.group(3)

        # Check if function has a docstring
        function_start = match.end()
        next_lines = content[function_start:function_start + 200]

        # Skip if it's a test function
        if function_name.startswith('test_'):
            continue

        # Check for docstring
        docstring_pattern = r'^\s*"""'
        if not re.search(docstring_pattern, next_lines, re.MULTILINE):
            functions_without_docstrings.append({
                'file': file_path,
                'function_name': function_name,
                'function_args': function_args,
                'function_return': function_return,
                'start': match.start(),
                'end': match.end()
            })

    return functions_without_docstrings

def find_classes_without_docstrings(file_path: str) -> List[Dict]:
    """Find classes without docstrings in a Python file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Find class definitions
    classes_without_docstrings = []

    # Pattern to match class definitions
    class_pattern = r'class\s+(\w+)(?:\(([^)]+)\))?\s*:'

    for match in re.finditer(class_pattern, content, re.DOTALL):
        class_name = match.group(1)
        class_parent = match.group(2)

        # Check if class has a docstring
        class_start = match.end()
        next_lines = content[class_start:class_start + 200]

        # Check for docstring
        docstring_pattern = r'^\s*"""'
        if not re.search(docstring_pattern, next_lines, re.MULTILINE):
            classes_without_docstrings.append({
                'file': file_path,
                'class_name': class_name,
                'class_parent': class_parent,
                'start': match.start(),
                'end': match.end()
            })

    return classes_without_docstrings

def find_modules_without_docstrings(file_path: str) -> List[Dict]:
    """Find modules without docstrings in a Python file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Check if module has a docstring
    module_docstring_pattern = r'^"""'
    if not re.search(module_docstring_pattern, content, re.MULTILINE):
        return [{
            'file': file_path,
            'start': 0,
            'end': 0
        }]

    return []

def generate_function_docstring(function_name: str, function_args: str, function_return: Optional[str] = None) -> str:
    """Generate a docstring for a function."""
    # Parse arguments
    args = [arg.strip() for arg in function_args.split(',') if arg.strip()]
    arg_names = []

    for arg in args:
        # Handle default values and type hints
        arg_parts = arg.split('=')[0].split(':')[0].strip()
        # Handle *args and **kwargs
        if arg_parts.startswith('*'):
            arg_parts = arg_parts.lstrip('*')
        # Handle self and cls
        if arg_parts not in ['self', 'cls']:
            arg_names.append(arg_parts)

    # Generate docstring
    docstring = f'    """\n    {function_name.replace("_", " ").capitalize()}.\n    \n'

    if arg_names:
        docstring += '    Args:\n'
        for arg_name in arg_names:
            docstring += f'        {arg_name}: Description of {arg_name}\n'
        docstring += '    \n'

    if function_return and function_return.strip() != 'None':
        docstring += '    Returns:\n'
        docstring += f'        {function_return.strip()}: Description of return value\n'
        docstring += '    \n'

    docstring += '    """\n'

    return docstring

def generate_class_docstring(class_name: str, class_parent: Optional[str] = None) -> str:
    """Generate a docstring for a class."""
    # Generate docstring
    docstring = f'    """\n    {class_name} class'

    if class_parent:
        docstring += f' that inherits from {class_parent}'

    docstring += '.\n    \n    Attributes:\n        Add attributes here\n    """\n'

    return docstring

def generate_module_docstring(file_path: str) -> str:
    """Generate a docstring for a module."""
    # Extract module name from file path
    module_name = os.path.basename(file_path).replace('.py', '')

    # Generate docstring
    docstring = f'"""\n{module_name.replace("_", " ").capitalize()} module.\n\nThis module provides functionality for...\n"""\n\n'

    return docstring

def add_docstrings_to_file(file_path: str, functions: List[Dict], classes: List[Dict], module: List[Dict]) -> None:
    """Add docstrings to a file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Add docstrings to functions and classes
    # We need to add them in reverse order to avoid messing up the positions
    all_items = functions + classes + module
    all_items.sort(key=lambda x: x['start'], reverse=True)

    for item in all_items:
        if 'function_name' in item:
            docstring = generate_function_docstring(
                item['function_name'],
                item['function_args'],
                item['function_return']
            )
            content = content[:item['end']] + '\n' + docstring + content[item['end']:]
        elif 'class_name' in item:
            docstring = generate_class_docstring(
                item['class_name'],
                item['class_parent']
            )
            content = content[:item['end']] + '\n' + docstring + content[item['end']:]
        else:
            docstring = generate_module_docstring(file_path)
            content = docstring + content

    # Write updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Main function to implement documentation improvements."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Find Python files
    python_files = find_python_files(root_dir)

    # Find functions, classes, and modules without docstrings
    functions_without_docstrings = []
    classes_without_docstrings = []
    modules_without_docstrings = []

    for file_path in python_files:
        functions = find_functions_without_docstrings(file_path)
        classes = find_classes_without_docstrings(file_path)
        modules = find_modules_without_docstrings(file_path)

        functions_without_docstrings.extend(functions)
        classes_without_docstrings.extend(classes)
        modules_without_docstrings.extend(modules)

    # Write results to file
    with open(os.path.join(root_dir, 'documentation_improvements.json'), 'w') as f:
        json.dump({
            'functions_without_docstrings': functions_without_docstrings,
            'classes_without_docstrings': classes_without_docstrings,
            'modules_without_docstrings': modules_without_docstrings
        }, f, indent=2)

    print(f"Found {len(functions_without_docstrings)} functions, {len(classes_without_docstrings)} classes, and {len(modules_without_docstrings)} modules without docstrings.")
    print(f"Results written to {os.path.join(root_dir, 'documentation_improvements.json')}")

    # Automatically confirm
    print("Automatically adding docstrings to all files...")
    confirm = 'y'

    # Add docstrings to files
    for file_path in set([item['file'] for item in functions_without_docstrings + classes_without_docstrings + modules_without_docstrings]):
        functions = [item for item in functions_without_docstrings if item['file'] == file_path]
        classes = [item for item in classes_without_docstrings if item['file'] == file_path]
        modules = [item for item in modules_without_docstrings if item['file'] == file_path]

        add_docstrings_to_file(file_path, functions, classes, modules)
        print(f"Added docstrings to {file_path}")

    print("Documentation improvements completed.")

if __name__ == '__main__':
    main()
