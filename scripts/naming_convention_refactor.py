#!/usr/bin/env python3
"""
Naming Convention Refactoring Script

This script helps refactor files and directories to follow the standardized
naming conventions. It reads the naming convention analysis report and
provides options to refactor specific files or directories.
"""

import os
import re
import json
import argparse
import logging
import shutil
from typing import Dict, List, Any, Tuple, Set
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("naming-convention-refactor")

# Define naming convention patterns
KEBAB_CASE_PATTERN = re.compile(r'^[a-z]+(-[a-z0-9]+)*$')
SNAKE_CASE_PATTERN = re.compile(r'^[a-z]+(_[a-z0-9]+)*$')
CAMEL_CASE_PATTERN = re.compile(r'^[a-z]+([A-Z][a-z0-9]*)*$')
PASCAL_CASE_PATTERN = re.compile(r'^[A-Z][a-z0-9]*([A-Z][a-z0-9]*)*$')
UPPER_SNAKE_CASE_PATTERN = re.compile(r'^[A-Z]+(_[A-Z0-9]+)*$')


def to_kebab_case(name: str) -> str:
    """
    Convert a name to kebab-case.
    
    Args:
        name: Name to convert
        
    Returns:
        Name in kebab-case
    """
    # Handle PascalCase
    if is_pascal_case(name):
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', name)
    
    # Handle camelCase
    if is_camel_case(name):
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', name)
    
    # Handle snake_case
    if is_snake_case(name):
        name = name.replace('_', '-')
    
    # Handle UPPER_SNAKE_CASE
    if is_upper_snake_case(name):
        name = name.replace('_', '-').lower()
    
    # Handle other cases
    name = re.sub(r'[^a-zA-Z0-9-]', '-', name)
    name = re.sub(r'-+', '-', name)
    name = name.strip('-').lower()
    
    return name


def to_snake_case(name: str) -> str:
    """
    Convert a name to snake_case.
    
    Args:
        name: Name to convert
        
    Returns:
        Name in snake_case
    """
    # Handle PascalCase
    if is_pascal_case(name):
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    
    # Handle camelCase
    if is_camel_case(name):
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    
    # Handle kebab-case
    if is_kebab_case(name):
        name = name.replace('-', '_')
    
    # Handle UPPER_SNAKE_CASE
    if is_upper_snake_case(name):
        name = name.lower()
    
    # Handle other cases
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_').lower()
    
    return name


def to_pascal_case(name: str) -> str:
    """
    Convert a name to PascalCase.
    
    Args:
        name: Name to convert
        
    Returns:
        Name in PascalCase
    """
    # Handle snake_case
    if is_snake_case(name):
        name = ''.join(word.capitalize() for word in name.split('_'))
    
    # Handle kebab-case
    if is_kebab_case(name):
        name = ''.join(word.capitalize() for word in name.split('-'))
    
    # Handle camelCase
    if is_camel_case(name):
        name = name[0].upper() + name[1:]
    
    # Handle UPPER_SNAKE_CASE
    if is_upper_snake_case(name):
        name = ''.join(word.capitalize() for word in name.lower().split('_'))
    
    # Handle other cases
    name = re.sub(r'[^a-zA-Z0-9]', ' ', name)
    name = ''.join(word.capitalize() for word in name.split())
    
    return name


def is_kebab_case(name: str) -> bool:
    """Check if a name follows kebab-case convention."""
    return bool(KEBAB_CASE_PATTERN.match(name))


def is_snake_case(name: str) -> bool:
    """Check if a name follows snake_case convention."""
    return bool(SNAKE_CASE_PATTERN.match(name))


def is_camel_case(name: str) -> bool:
    """Check if a name follows camelCase convention."""
    return bool(CAMEL_CASE_PATTERN.match(name))


def is_pascal_case(name: str) -> bool:
    """Check if a name follows PascalCase convention."""
    return bool(PASCAL_CASE_PATTERN.match(name))


def is_upper_snake_case(name: str) -> bool:
    """Check if a name follows UPPER_SNAKE_CASE convention."""
    return bool(UPPER_SNAKE_CASE_PATTERN.match(name))


def get_actual_convention(name: str) -> str:
    """
    Determine the actual naming convention used for a name.
    
    Args:
        name: Name to check
        
    Returns:
        String representing the actual naming convention
    """
    if is_kebab_case(name):
        return "kebab-case"
    elif is_snake_case(name):
        return "snake_case"
    elif is_camel_case(name):
        return "camelCase"
    elif is_pascal_case(name):
        return "PascalCase"
    elif is_upper_snake_case(name):
        return "UPPER_SNAKE_CASE"
    else:
        return "unknown"


def load_analysis_report(report_file: str) -> Dict[str, Any]:
    """
    Load the naming convention analysis report.
    
    Args:
        report_file: Path to the report file
        
    Returns:
        Dictionary with analysis results
    """
    with open(report_file, 'r') as f:
        return json.load(f)


def refactor_directory(dir_path: str, is_service_dir: bool = False, dry_run: bool = False) -> None:
    """
    Refactor a directory to follow the standardized naming convention.
    
    Args:
        dir_path: Path to the directory
        is_service_dir: Whether the directory is a top-level service directory
        dry_run: Whether to perform a dry run (no actual changes)
    """
    dir_name = os.path.basename(dir_path)
    parent_dir = os.path.dirname(dir_path)
    
    if is_service_dir:
        # Service directories should use kebab-case
        new_name = to_kebab_case(dir_name)
        expected_convention = "kebab-case"
    else:
        # Module directories should use snake_case
        new_name = to_snake_case(dir_name)
        expected_convention = "snake_case"
    
    if new_name != dir_name:
        new_path = os.path.join(parent_dir, new_name)
        
        logger.info(f"Refactoring directory: {dir_path}")
        logger.info(f"  Current name: {dir_name} ({get_actual_convention(dir_name)})")
        logger.info(f"  New name: {new_name} ({expected_convention})")
        
        if not dry_run:
            try:
                # Check if the new path already exists
                if os.path.exists(new_path):
                    logger.warning(f"  Cannot refactor: {new_path} already exists")
                    return
                
                # Rename the directory
                shutil.move(dir_path, new_path)
                logger.info(f"  Refactored: {dir_path} -> {new_path}")
            except Exception as e:
                logger.error(f"  Error refactoring directory: {e}")
        else:
            logger.info(f"  Dry run: Would refactor {dir_path} -> {new_path}")


def refactor_file(file_path: str, dry_run: bool = False) -> None:
    """
    Refactor a file to follow the standardized naming convention.
    
    Args:
        file_path: Path to the file
        dry_run: Whether to perform a dry run (no actual changes)
    """
    file_name = os.path.basename(file_path)
    parent_dir = os.path.dirname(file_path)
    file_name_without_ext, file_ext = os.path.splitext(file_name)
    
    # Determine the expected naming convention based on the file extension
    if file_ext.lower() in {'.py', '.pyi', '.pyx', '.pyd'}:
        # Python files should use snake_case
        new_name_without_ext = to_snake_case(file_name_without_ext)
        expected_convention = "snake_case"
    elif file_ext.lower() in {'.js', '.jsx', '.ts', '.tsx'}:
        # JavaScript/TypeScript files should use kebab-case
        new_name_without_ext = to_kebab_case(file_name_without_ext)
        expected_convention = "kebab-case"
    elif file_ext.lower() in {'.yml', '.yaml', '.json', '.toml', '.ini', '.cfg'}:
        # Configuration files should use kebab-case
        new_name_without_ext = to_kebab_case(file_name_without_ext)
        expected_convention = "kebab-case"
    elif file_ext.lower() in {'.md', '.rst', '.txt'}:
        # Documentation files should use kebab-case
        new_name_without_ext = to_kebab_case(file_name_without_ext)
        expected_convention = "kebab-case"
    else:
        # Default to snake_case for other files
        new_name_without_ext = to_snake_case(file_name_without_ext)
        expected_convention = "snake_case"
    
    new_name = new_name_without_ext + file_ext
    
    if new_name != file_name:
        new_path = os.path.join(parent_dir, new_name)
        
        logger.info(f"Refactoring file: {file_path}")
        logger.info(f"  Current name: {file_name} ({get_actual_convention(file_name_without_ext)})")
        logger.info(f"  New name: {new_name} ({expected_convention})")
        
        if not dry_run:
            try:
                # Check if the new path already exists
                if os.path.exists(new_path):
                    logger.warning(f"  Cannot refactor: {new_path} already exists")
                    return
                
                # Rename the file
                shutil.move(file_path, new_path)
                logger.info(f"  Refactored: {file_path} -> {new_path}")
            except Exception as e:
                logger.error(f"  Error refactoring file: {e}")
        else:
            logger.info(f"  Dry run: Would refactor {file_path} -> {new_path}")


def refactor_invalid_directories(results: Dict[str, Any], dry_run: bool = False) -> None:
    """
    Refactor invalid directories to follow the standardized naming conventions.
    
    Args:
        results: Analysis results
        dry_run: Whether to perform a dry run (no actual changes)
    """
    logger.info("Refactoring invalid directories...")
    
    # Get top-level directories to identify service directories
    root_dir = os.path.abspath('.')
    top_level_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    service_dirs = [d for d in top_level_dirs if d.endswith('-service') or d.endswith('_service')]
    
    for dir_check in results["invalid_directories"]:
        dir_path = dir_check["path"]
        dir_name = os.path.basename(dir_path)
        
        # Check if this is a service directory
        is_service_dir = dir_name in service_dirs
        
        # Refactor the directory
        refactor_directory(dir_path, is_service_dir, dry_run)


def refactor_invalid_files(results: Dict[str, Any], dry_run: bool = False) -> None:
    """
    Refactor invalid files to follow the standardized naming conventions.
    
    Args:
        results: Analysis results
        dry_run: Whether to perform a dry run (no actual changes)
    """
    logger.info("Refactoring invalid files...")
    
    for file_check in results["invalid_files"]:
        file_path = file_check["path"]
        
        # Refactor the file
        refactor_file(file_path, dry_run)


def refactor_duplicate_directories(results: Dict[str, Any], dry_run: bool = False) -> None:
    """
    Refactor duplicate directories to follow the standardized naming conventions.
    
    Args:
        results: Analysis results
        dry_run: Whether to perform a dry run (no actual changes)
    """
    logger.info("Refactoring duplicate directories...")
    
    # Get top-level directories to identify service directories
    root_dir = os.path.abspath('.')
    top_level_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    service_dirs = [d for d in top_level_dirs if d.endswith('-service') or d.endswith('_service')]
    
    for normalized_name, paths in results["duplicate_directories"].items():
        logger.info(f"Duplicate directories for '{normalized_name}':")
        
        for path in paths:
            dir_name = os.path.basename(path)
            
            # Check if this is a service directory
            is_service_dir = dir_name in service_dirs
            
            # Log the directory
            logger.info(f"  {path} ({get_actual_convention(dir_name)})")
        
        logger.info("  Skipping refactoring of duplicate directories (manual intervention required)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Refactor files and directories to follow the standardized naming conventions.")
    parser.add_argument("--report-file", default="reports/naming_convention_analysis.json", help="Path to the naming convention analysis report")
    parser.add_argument("--refactor-dirs", action="store_true", help="Refactor invalid directories")
    parser.add_argument("--refactor-files", action="store_true", help="Refactor invalid files")
    parser.add_argument("--refactor-duplicates", action="store_true", help="Refactor duplicate directories")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run (no actual changes)")
    args = parser.parse_args()
    
    # Convert to absolute path
    report_file = os.path.abspath(args.report_file)
    
    logger.info(f"Loading naming convention analysis report: {report_file}")
    results = load_analysis_report(report_file)
    
    if args.refactor_dirs:
        refactor_invalid_directories(results, args.dry_run)
    
    if args.refactor_files:
        refactor_invalid_files(results, args.dry_run)
    
    if args.refactor_duplicates:
        refactor_duplicate_directories(results, args.dry_run)
    
    if not (args.refactor_dirs or args.refactor_files or args.refactor_duplicates):
        logger.info("No refactoring options specified. Use --refactor-dirs, --refactor-files, or --refactor-duplicates.")


if __name__ == "__main__":
    main()