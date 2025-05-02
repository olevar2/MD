#!/usr/bin/env python
"""
Configuration Import Migration Tool

This script helps automate the migration from deprecated configuration modules
to the new consolidated module.

Usage:
    python migrate_config_imports.py [--path PATH] [--dry-run] [--verbose]

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
    """Process a Python file to migrate configuration imports."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # Patterns to search for
    patterns = [
        # From analysis_engine.core.config import ...
        (
            r"from\s+analysis_engine\.core\.config\s+import\s+(.*)",
            lambda match: process_core_config_import(match.group(1), file_path)
        ),
        # import analysis_engine.core.config
        (
            r"import\s+analysis_engine\.core\.config(?:\s+as\s+(\w+))?",
            lambda match: process_core_config_module_import(match.group(0), match.group(1))
        ),
        # From config.config import ...
        (
            r"from\s+config\.config\s+import\s+(.*)",
            lambda match: process_config_config_import(match.group(1), file_path)
        ),
        # import config.config
        (
            r"import\s+config\.config(?:\s+as\s+(\w+))?",
            lambda match: process_config_config_module_import(match.group(0), match.group(1))
        ),
        # settings.host -> settings.HOST
        (
            r"(\w+)\.([a-z][a-z0-9_]*)",
            lambda match: process_settings_attribute(match.group(0), match.group(1), match.group(2))
        ),
        # config_manager.get("host") -> config_manager.get("HOST")
        (
            r"(\w+)\.get\(\s*[\"']([a-z][a-z0-9_]*)[\"']\s*(?:,\s*(.*))?\)",
            lambda match: process_config_manager_get(
                match.group(0), match.group(1), match.group(2), match.group(3)
            )
        ),
    ]
    
    # Apply patterns
    for pattern, processor in patterns:
        matches = list(re.finditer(pattern, content))
        for match in reversed(matches):  # Process in reverse to avoid offset issues
            replacement, change_desc = processor(match)
            if replacement and replacement != match.group(0):
                start, end = match.span()
                content = content[:start] + replacement + content[end:]
                changes.append(change_desc)
    
    # Write changes if needed
    if content != original_content and not dry_run:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    changed = content != original_content
    
    if verbose and changed:
        print(f"Changes in {file_path}:")
        for change in changes:
            print(f"  - {change}")
    
    return changed, changes


def process_core_config_import(imports: str, file_path: str) -> Tuple[str, str]:
    """Process imports from analysis_engine.core.config."""
    # Split imports
    import_items = [item.strip() for item in imports.split(",")]
    new_imports = []
    
    # Process each import
    for item in import_items:
        if item == "Settings":
            new_imports.append("AnalysisEngineSettings as Settings")
        else:
            new_imports.append(item)
    
    # Create new import statement
    new_import = f"from analysis_engine.config import {', '.join(new_imports)}"
    
    return new_import, f"Changed 'from analysis_engine.core.config import {imports}' to '{new_import}'"


def process_core_config_module_import(import_stmt: str, alias: str) -> Tuple[str, str]:
    """Process import of analysis_engine.core.config module."""
    if alias:
        new_import = f"import analysis_engine.config as {alias}"
    else:
        new_import = "import analysis_engine.config"
    
    return new_import, f"Changed '{import_stmt}' to '{new_import}'"


def process_config_config_import(imports: str, file_path: str) -> Tuple[str, str]:
    """Process imports from config.config."""
    # Split imports
    import_items = [item.strip() for item in imports.split(",")]
    
    # Check if we're importing specific settings or functions
    if any(item not in ["get_settings"] and not item.isupper() for item in import_items):
        # We're importing a mix of things, need to handle differently
        return process_mixed_config_imports(imports, import_items)
    
    # Process settings constants
    settings_imports = [item for item in import_items if item.isupper()]
    function_imports = [item for item in import_items if item == "get_settings"]
    
    new_imports = []
    
    # Add settings import if needed
    if settings_imports:
        new_imports.append("settings")
    
    # Add function imports
    new_imports.extend(function_imports)
    
    # Create new import statement
    new_import = f"from analysis_engine.config import {', '.join(new_imports)}"
    
    # Create additional statements for settings constants
    additional_stmts = [f"{item} = settings.{item}" for item in settings_imports]
    
    if additional_stmts:
        new_import = f"{new_import}\n{chr(10).join(additional_stmts)}"
    
    return new_import, f"Changed 'from config.config import {imports}' to use analysis_engine.config"


def process_mixed_config_imports(imports: str, import_items: List[str]) -> Tuple[str, str]:
    """Process mixed imports from config.config."""
    # This is a more complex case, we'll need to add comments
    new_import = f"from analysis_engine.config import settings, get_settings  # Migrated from config.config\n"
    new_import += "# TODO: Review the following settings access after migration\n"
    
    # Add comments for each imported item
    for item in import_items:
        if item.isupper():
            new_import += f"# {item} = settings.{item}\n"
        elif item == "get_settings":
            new_import += f"# {item} is imported directly\n"
        else:
            new_import += f"# {item} needs to be accessed through settings or other means\n"
    
    return new_import, f"Added migration comments for 'from config.config import {imports}'"


def process_config_config_module_import(import_stmt: str, alias: str) -> Tuple[str, str]:
    """Process import of config.config module."""
    if alias:
        new_import = f"import analysis_engine.config as {alias}  # Migrated from config.config"
    else:
        new_import = "import analysis_engine.config  # Migrated from config.config"
    
    return new_import, f"Changed '{import_stmt}' to '{new_import}'"


def process_settings_attribute(full_expr: str, obj_name: str, attr_name: str) -> Tuple[str, str]:
    """Process settings attribute access."""
    # Only process if it looks like a settings attribute
    if obj_name not in ["settings", "Settings"] or not attr_name[0].islower():
        return full_expr, ""
    
    # Convert attribute name to uppercase
    new_attr_name = attr_name.upper()
    new_expr = f"{obj_name}.{new_attr_name}"
    
    return new_expr, f"Changed '{full_expr}' to '{new_expr}'"


def process_config_manager_get(full_expr: str, obj_name: str, key: str, default: str) -> Tuple[str, str]:
    """Process config_manager.get() calls."""
    # Only process if it looks like a config manager and lowercase key
    if not obj_name.endswith("config_manager") or not key[0].islower():
        return full_expr, ""
    
    # Convert key to uppercase
    new_key = key.upper()
    
    # Reconstruct the expression
    if default:
        new_expr = f"{obj_name}.get(\"{new_key}\", {default})"
    else:
        new_expr = f"{obj_name}.get(\"{new_key}\")"
    
    return new_expr, f"Changed '{full_expr}' to '{new_expr}'"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Migrate configuration imports")
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
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Files processed: {len(python_files)}")
    print(f"  Files changed: {changed_files}")
    print(f"  Total changes: {total_changes}")
    
    if args.dry_run:
        print("\nThis was a dry run. No files were modified.")
        print("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
