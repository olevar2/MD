#!/usr/bin/env python3
"""
Script to rename files and update imports according to naming conventions.
"""

import os
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set

def load_violations(file_path: str) -> Dict:
    """Load naming violations from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_rename_map(violations: Dict) -> Dict[str, str]:
    """Create a map of old paths to new paths."""
    rename_map = {}

    # Add file violations
    for violation in violations.get('file_violations', []):
        old_path = violation['file']
        dir_path = os.path.dirname(old_path)
        new_name = f"{violation['suggested_name']}{os.path.splitext(old_path)[1]}"
        new_path = os.path.join(dir_path, new_name)
        rename_map[old_path] = new_path

    # Add directory violations
    for violation in violations.get('directory_violations', []):
        old_path = violation['directory']
        parent_dir = os.path.dirname(old_path)
        new_name = violation['suggested_name']
        new_path = os.path.join(parent_dir, new_name)
        rename_map[old_path] = new_path

    return rename_map

def rename_files_and_dirs(rename_map: Dict[str, str], dry_run: bool = False) -> None:
    """Rename files and directories according to the rename map."""
    # Sort paths by length in descending order to handle nested paths correctly
    paths = sorted(rename_map.keys(), key=len, reverse=True)

    for old_path in paths:
        new_path = rename_map[old_path]

        if os.path.isfile(old_path):
            if not dry_run:
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.move(old_path, new_path)
            print(f"Renamed file: {old_path} -> {new_path}")
        elif os.path.isdir(old_path):
            if not dry_run:
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.move(old_path, new_path)
            print(f"Renamed directory: {old_path} -> {new_path}")

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

def update_imports(python_files: List[str], rename_map: Dict[str, str], dry_run: bool = False) -> None:
    """Update imports in Python files to reflect renamed files and directories."""
    # Create a map of old module paths to new module paths
    module_map = {}
    for old_path, new_path in rename_map.items():
        if os.path.isfile(old_path) and old_path.endswith('.py'):
            # Convert file paths to module paths
            old_module = os.path.splitext(old_path)[0].replace(os.path.sep, '.')
            new_module = os.path.splitext(new_path)[0].replace(os.path.sep, '.')
            module_map[old_module] = new_module
        elif os.path.isdir(old_path):
            # Convert directory paths to package paths
            old_package = old_path.replace(os.path.sep, '.')
            new_package = new_path.replace(os.path.sep, '.')
            module_map[old_package] = new_package

    # Update imports in Python files
    for file_path in python_files:
        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Create a copy of the original content
        new_content = content

        # Update imports
        for old_module, new_module in module_map.items():
            # Handle different import patterns
            patterns = [
                rf'from\s+{re.escape(old_module)}\s+import',  # from module import
                rf'import\s+{re.escape(old_module)}(\s+as)?',  # import module [as]
                rf'from\s+{re.escape(old_module)}\.([a-zA-Z0-9_]+)\s+import',  # from module.submodule import
            ]

            for pattern in patterns:
                new_content = re.sub(
                    pattern,
                    lambda m: m.group(0).replace(old_module, new_module),
                    new_content
                )

        # Write updated content back to the file
        if new_content != content and not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            if new_content != content:
                print(f"Updated imports in: {file_path}")

def main():
    """Main function to rename files and update imports."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    violations_file = os.path.join(root_dir, 'naming_violations.json')

    # Check if violations file exists
    if not os.path.exists(violations_file):
        print(f"Violations file not found: {violations_file}")
        print("Please run identify_naming_violations.py first.")
        return

    # Load violations
    violations = load_violations(violations_file)

    # Create rename map
    rename_map = create_rename_map(violations)

    # Automatically confirm
    print(f"Found {len(rename_map)} files and directories to rename.")
    print("Automatically proceeding with renaming...")
    confirm = 'y'

    # Rename files and directories
    rename_files_and_dirs(rename_map, dry_run=False)

    # Find all Python files
    python_files = find_python_files(root_dir)

    # Update imports
    update_imports(python_files, rename_map, dry_run=False)

    print("Renaming and import updates completed.")

if __name__ == '__main__':
    main()
