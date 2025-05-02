#!/usr/bin/env python
"""
API Router Migration Tool

This tool helps migrate from the deprecated analysis_engine.api.router module
to the recommended analysis_engine.api.routes module.

Usage:
    python migrate_router_imports.py [--path PATH] [--dry-run] [--verbose]

Options:
    --path PATH     Path to search for Python files [default: .]
    --dry-run       Show changes without applying them
    --verbose       Show detailed information
"""

import os
import re
import argparse
import sys
from typing import List, Dict, Tuple, Set


def find_python_files(path: str) -> List[str]:
    """Find all Python files in the given path."""
    python_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def process_file(file_path: str, dry_run: bool = False, verbose: bool = False) -> Tuple[bool, List[str]]:
    """Process a Python file to migrate router imports."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Skip files that can't be decoded as UTF-8
        if verbose:
            print(f"Skipping {file_path} due to encoding issues")
        return False, []
    
    original_content = content
    changes = []
    
    # Patterns to search for
    patterns = [
        # From analysis_engine.api.router import api_router
        (
            r"from\s+analysis_engine\.api\.router\s+import\s+api_router",
            lambda match: "from analysis_engine.api.routes import setup_routes  # Migrated from api_router"
        ),
        # import analysis_engine.api.router
        (
            r"import\s+analysis_engine\.api\.router(?:\s+as\s+(\w+))?",
            lambda match: f"from analysis_engine.api.routes import setup_routes  # Migrated from api_router"
        ),
        # app.include_router(api_router)
        (
            r"(app|application)\.include_router\s*\(\s*api_router\s*\)",
            lambda match: f"{match.group(1)}.include_router(api_router)  # TODO: Replace with setup_routes({match.group(1)})"
        ),
    ]
    
    # Apply patterns
    for pattern, replacement_func in patterns:
        for match in re.finditer(pattern, content):
            old_text = match.group(0)
            new_text = replacement_func(match)
            content = content.replace(old_text, new_text)
            changes.append(f"Changed '{old_text}' to '{new_text}'")
    
    # Only write changes if content has changed and not in dry run mode
    if content != original_content and not dry_run:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    return content != original_content, changes


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Migrate API router imports")
    parser.add_argument("--path", default=".", help="Path to search for Python files")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying them")
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    args = parser.parse_args()
    
    # Find Python files
    python_files = find_python_files(args.path)
    print(f"Found {len(python_files)} Python files")
    
    # Process files
    changed_files = 0
    total_changes = 0
    
    for file_path in python_files:
        changed, changes = process_file(file_path, args.dry_run, args.verbose)
        if changed:
            changed_files += 1
            total_changes += len(changes)
            if args.verbose:
                print(f"Changes in {file_path}:")
                for change in changes:
                    print(f"  - {change}")
    
    # Print summary
    action = "Would change" if args.dry_run else "Changed"
    print(f"{action} {total_changes} imports in {changed_files} files")
    
    if args.dry_run and changed_files > 0:
        print("\nRun without --dry-run to apply these changes")


if __name__ == "__main__":
    main()
