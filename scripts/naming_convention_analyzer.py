#!/usr/bin/env python3
"""
Naming Convention Analyzer

This script analyzes the codebase for naming convention inconsistencies
and generates a report of files and directories that don't follow the
standardized naming conventions.
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Set, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("naming-convention-analyzer")

# Define naming convention patterns
KEBAB_CASE_PATTERN = re.compile(r'^[a-z]+(-[a-z0-9]+)*$')
SNAKE_CASE_PATTERN = re.compile(r'^[a-z]+(_[a-z0-9]+)*$')
CAMEL_CASE_PATTERN = re.compile(r'^[a-z]+([A-Z][a-z0-9]*)*$')
PASCAL_CASE_PATTERN = re.compile(r'^[A-Z][a-z0-9]*([A-Z][a-z0-9]*)*$')
UPPER_SNAKE_CASE_PATTERN = re.compile(r'^[A-Z]+(_[A-Z0-9]+)*$')

# Define file extensions for different languages
PYTHON_EXTENSIONS = {'.py', '.pyi', '.pyx', '.pyd'}
JS_TS_EXTENSIONS = {'.js', '.jsx', '.ts', '.tsx'}
CONFIG_EXTENSIONS = {'.yml', '.yaml', '.json', '.toml', '.ini', '.cfg'}
DOC_EXTENSIONS = {'.md', '.rst', '.txt'}

# Define directories to ignore
IGNORE_DIRS = {
    '.git',
    '__pycache__',
    'node_modules',
    'venv',
    'env',
    '.venv',
    '.env',
    'dist',
    'build',
    '.idea',
    '.vscode',
    '.pytest_cache',
    '.mypy_cache',
    '.coverage',
    '.tox',
    '.eggs',
    '*.egg-info',
}

# Define files to ignore
IGNORE_FILES = {
    '.gitignore',
    '.dockerignore',
    'LICENSE',
    'README.md',
    'CHANGELOG.md',
    'CONTRIBUTING.md',
    'CODE_OF_CONDUCT.md',
    'requirements.txt',
    'setup.py',
    'setup.cfg',
    'pyproject.toml',
    'package.json',
    'package-lock.json',
    'yarn.lock',
    'Dockerfile',
    '.env',
    '.env.example',
    '.env.local',
    '.env.development',
    '.env.test',
    '.env.production',
}


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


def should_ignore_dir(dir_name: str) -> bool:
    """Check if a directory should be ignored."""
    for pattern in IGNORE_DIRS:
        if '*' in pattern:
            if re.match(pattern.replace('*', '.*'), dir_name):
                return True
        elif dir_name == pattern:
            return True
    return False


def should_ignore_file(file_name: str) -> bool:
    """Check if a file should be ignored."""
    for pattern in IGNORE_FILES:
        if '*' in pattern:
            if re.match(pattern.replace('*', '.*'), file_name):
                return True
        elif file_name == pattern:
            return True
    return False


def check_directory_naming(dir_path: str, is_service_dir: bool = False) -> Dict[str, Any]:
    """
    Check if a directory follows the standardized naming convention.

    Args:
        dir_path: Path to the directory
        is_service_dir: Whether the directory is a top-level service directory

    Returns:
        Dictionary with naming convention check results
    """
    dir_name = os.path.basename(dir_path)

    if is_service_dir:
        # Service directories should use kebab-case
        is_valid = is_kebab_case(dir_name)
        expected_convention = "kebab-case"
    else:
        # Module directories should use snake_case
        is_valid = is_snake_case(dir_name)
        expected_convention = "snake_case"

    return {
        "path": dir_path,
        "name": dir_name,
        "is_valid": is_valid,
        "expected_convention": expected_convention,
        "actual_convention": get_actual_convention(dir_name),
    }


def check_file_naming(file_path: str) -> Dict[str, Any]:
    """
    Check if a file follows the standardized naming convention.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with naming convention check results
    """
    file_name = os.path.basename(file_path)
    file_name_without_ext, file_ext = os.path.splitext(file_name)

    if file_ext.lower() in PYTHON_EXTENSIONS:
        # Python files should use snake_case
        is_valid = is_snake_case(file_name_without_ext)
        expected_convention = "snake_case"
    elif file_ext.lower() in JS_TS_EXTENSIONS:
        # JavaScript/TypeScript files should use kebab-case
        is_valid = is_kebab_case(file_name_without_ext)
        expected_convention = "kebab-case"
    elif file_ext.lower() in CONFIG_EXTENSIONS:
        # Configuration files should use kebab-case
        is_valid = is_kebab_case(file_name_without_ext)
        expected_convention = "kebab-case"
    elif file_ext.lower() in DOC_EXTENSIONS:
        # Documentation files should use kebab-case
        is_valid = is_kebab_case(file_name_without_ext)
        expected_convention = "kebab-case"
    else:
        # Default to snake_case for other files
        is_valid = is_snake_case(file_name_without_ext)
        expected_convention = "snake_case"

    return {
        "path": file_path,
        "name": file_name,
        "is_valid": is_valid,
        "expected_convention": expected_convention,
        "actual_convention": get_actual_convention(file_name_without_ext),
    }


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


def find_duplicate_directories(root_dir: str) -> Dict[str, List[str]]:
    """
    Find duplicate directories with different naming conventions.

    Args:
        root_dir: Root directory to search

    Returns:
        Dictionary mapping normalized directory names to lists of actual directory paths
    """
    duplicates: Dict[str, List[str]] = {}

    for dir_path, dir_names, _ in os.walk(root_dir):
        for dir_name in dir_names:
            if should_ignore_dir(dir_name):
                continue

            # Normalize the directory name by removing special characters and converting to lowercase
            normalized_name = re.sub(r'[-_]', '', dir_name.lower())

            if normalized_name not in duplicates:
                duplicates[normalized_name] = []

            duplicates[normalized_name].append(os.path.join(dir_path, dir_name))

    # Filter out non-duplicates
    return {k: v for k, v in duplicates.items() if len(v) > 1}


def analyze_codebase(root_dir: str) -> Dict[str, Any]:
    """
    Analyze the codebase for naming convention inconsistencies.

    Args:
        root_dir: Root directory of the codebase

    Returns:
        Dictionary with analysis results
    """
    results = {
        "invalid_directories": [],
        "invalid_files": [],
        "duplicate_directories": {},
        "summary": {
            "total_directories": 0,
            "invalid_directories": 0,
            "total_files": 0,
            "invalid_files": 0,
            "total_duplicates": 0,
        }
    }

    # Get top-level directories to identify service directories
    top_level_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    service_dirs = [d for d in top_level_dirs if d.endswith('-service') or d.endswith('_service')]

    # Walk the directory tree
    for dir_path, dir_names, file_names in os.walk(root_dir):
        # Skip ignored directories
        dir_names[:] = [d for d in dir_names if not should_ignore_dir(d)]

        # Check directory naming
        for dir_name in dir_names:
            results["summary"]["total_directories"] += 1

            # Check if this is a service directory
            is_service_dir = dir_name in service_dirs

            # Check directory naming
            dir_check = check_directory_naming(os.path.join(dir_path, dir_name), is_service_dir)

            if not dir_check["is_valid"]:
                results["invalid_directories"].append(dir_check)
                results["summary"]["invalid_directories"] += 1

        # Check file naming
        for file_name in file_names:
            if should_ignore_file(file_name):
                continue

            results["summary"]["total_files"] += 1

            # Check file naming
            file_check = check_file_naming(os.path.join(dir_path, file_name))

            if not file_check["is_valid"]:
                results["invalid_files"].append(file_check)
                results["summary"]["invalid_files"] += 1

    # Find duplicate directories
    results["duplicate_directories"] = find_duplicate_directories(root_dir)
    results["summary"]["total_duplicates"] = len(results["duplicate_directories"])

    return results


def generate_report(results: Dict[str, Any], output_file: str) -> None:
    """
    Generate a report of naming convention inconsistencies.

    Args:
        results: Analysis results
        output_file: Path to the output file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Report generated: {output_file}")

    # Print summary
    logger.info("Summary:")
    logger.info(f"  Total directories: {results['summary']['total_directories']}")
    logger.info(f"  Invalid directories: {results['summary']['invalid_directories']}")
    logger.info(f"  Total files: {results['summary']['total_files']}")
    logger.info(f"  Invalid files: {results['summary']['invalid_files']}")
    logger.info(f"  Total duplicates: {results['summary']['total_duplicates']}")


def generate_markdown_report(results: Dict[str, Any], output_file: str) -> None:
    """
    Generate a Markdown report of naming convention inconsistencies.

    Args:
        results: Analysis results
        output_file: Path to the output file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("# Naming Convention Analysis Report\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Total directories: {results['summary']['total_directories']}\n")
        f.write(f"- Invalid directories: {results['summary']['invalid_directories']}\n")
        f.write(f"- Total files: {results['summary']['total_files']}\n")
        f.write(f"- Invalid files: {results['summary']['invalid_files']}\n")
        f.write(f"- Total duplicates: {results['summary']['total_duplicates']}\n\n")

        f.write("## Invalid Directories\n\n")
        if results["invalid_directories"]:
            f.write("| Path | Expected Convention | Actual Convention |\n")
            f.write("|------|---------------------|-------------------|\n")
            for dir_check in results["invalid_directories"]:
                f.write(f"| {dir_check['path']} | {dir_check['expected_convention']} | {dir_check['actual_convention']} |\n")
        else:
            f.write("No invalid directories found.\n")

        f.write("\n## Invalid Files\n\n")
        if results["invalid_files"]:
            f.write("| Path | Expected Convention | Actual Convention |\n")
            f.write("|------|---------------------|-------------------|\n")
            for file_check in results["invalid_files"]:
                f.write(f"| {file_check['path']} | {file_check['expected_convention']} | {file_check['actual_convention']} |\n")
        else:
            f.write("No invalid files found.\n")

        f.write("\n## Duplicate Directories\n\n")
        if results["duplicate_directories"]:
            for normalized_name, paths in results["duplicate_directories"].items():
                f.write(f"### {normalized_name}\n\n")
                for path in paths:
                    f.write(f"- {path}\n")
                f.write("\n")
        else:
            f.write("No duplicate directories found.\n")

    logger.info(f"Markdown report generated: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze naming convention inconsistencies in the codebase.")
    parser.add_argument("--root-dir", default=".", help="Root directory of the codebase")
    parser.add_argument("--output-file", default="reports/naming_convention_analysis.json", help="Path to the output JSON file")
    parser.add_argument("--markdown-output", default="reports/naming_convention_analysis.md", help="Path to the output Markdown file")
    args = parser.parse_args()

    # Convert to absolute paths
    root_dir = os.path.abspath(args.root_dir)
    output_file = os.path.join(root_dir, args.output_file)
    markdown_output = os.path.join(root_dir, args.markdown_output)

    logger.info(f"Analyzing codebase in {root_dir}...")
    results = analyze_codebase(root_dir)

    generate_report(results, output_file)
    generate_markdown_report(results, markdown_output)


if __name__ == "__main__":
    main()