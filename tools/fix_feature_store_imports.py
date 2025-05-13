#!/usr/bin/env python3
"""
Fix Feature Store Import Statements

This script fixes the circular dependency between feature-store-service and feature_store_service
by standardizing import statements to use a consistent naming convention.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple

# Configure paths
PROJECT_ROOT = Path("D:/MD/forex_trading_platform")
FEATURE_STORE_DIR = PROJECT_ROOT / "feature-store-service"
FEATURE_STORE_PACKAGE = FEATURE_STORE_DIR / "feature_store_service"

# Patterns to search for
IMPORT_PATTERNS = [
    r'from\s+(feature[-_]store[-_]service)\.([^\s]+)\s+import\s+([^\n]+)',
    r'import\s+(feature[-_]store[-_]service)\.([^\s]+)(?:\s+as\s+([^\s]+))?'
]

def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in a directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)
    return python_files

def fix_imports_in_file(file_path: Path) -> Tuple[int, List[str]]:
    """
    Fix imports in a single file.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (number of replacements, list of changes)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = []

        # Fix imports
        for pattern in IMPORT_PATTERNS:
            matches = re.finditer(pattern, content)
            for match in matches:
                module_name = match.group(1)

                # Standardize to snake_case
                if module_name != "feature_store_service":
                    replacement = match.group(0).replace(module_name, "feature_store_service")
                    content = content.replace(match.group(0), replacement)
                    changes.append(f"Changed '{match.group(0)}' to '{replacement}'")

        # Write changes back to file if content was modified
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return len(changes), changes

        return 0, []

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, [f"Error: {e}"]

def main():
    """Main entry point."""
    print(f"Fixing feature store imports in {PROJECT_ROOT}...", flush=True)

    # Find all Python files in the project
    python_files = find_python_files(PROJECT_ROOT)
    print(f"Found {len(python_files)} Python files", flush=True)

    # Print the first 10 files for debugging
    print("First 10 files:", flush=True)
    for file in python_files[:10]:
        print(f"  {file}", flush=True)

    # Fix imports in each file
    total_replacements = 0
    files_modified = 0

    print("Starting to process files...", flush=True)
    file_count = 0

    for file_path in python_files:
        file_count += 1
        if file_count % 100 == 0:
            print(f"Processed {file_count}/{len(python_files)} files...", flush=True)

        replacements, changes = fix_imports_in_file(file_path)

        if replacements > 0:
            files_modified += 1
            total_replacements += replacements

            print(f"\nFixed {replacements} imports in {file_path}:", flush=True)
            for change in changes:
                print(f"  - {change}", flush=True)

    print(f"\nSummary: Fixed {total_replacements} imports in {files_modified} files", flush=True)

    # Create a log file with all changes
    log_path = PROJECT_ROOT / "tools" / "output" / "feature_store_import_fixes.log"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Fixed {total_replacements} imports in {files_modified} files\n")
        f.write(f"Timestamp: {os.path.getmtime(log_path)}\n\n")

        for file_path in python_files:
            replacements, changes = fix_imports_in_file(file_path)

            if replacements > 0:
                f.write(f"\nFixed {replacements} imports in {file_path}:\n")
                for change in changes:
                    f.write(f"  - {change}\n")

    print(f"Log file created at {log_path}")

if __name__ == "__main__":
    main()
