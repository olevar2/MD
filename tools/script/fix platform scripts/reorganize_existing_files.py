#!/usr/bin/env python3
"""
Forex Trading Platform File Reorganization

This script reorganizes existing files in the forex trading platform to follow the standardized
directory structure. It moves files to their appropriate locations, updates imports, and removes
redundant files.

Usage:
python reorganize_existing_files.py [--project-root <project_root>] [--service <service_name>] [--dry-run]
"""

import os
import sys
import re
import ast
import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"

# Standard directory structure
STANDARD_DIRECTORIES = [
    "api",
    "config",
    "core",
    "models",
    "repositories",
    "services",
    "utils",
    "adapters",
    "interfaces",
    "tests"
]

# File classification patterns
FILE_PATTERNS = {
    "api": [
        r"api\.py$",
        r"routes\.py$",
        r"endpoints\.py$",
        r"controllers\.py$",
        r"views\.py$",
        r"app\.py$",
        r"server\.py$",
        r"rest\.py$",
        r"http\.py$",
        r"handlers\.py$"
    ],
    "config": [
        r"config\.py$",
        r"settings\.py$",
        r"environment\.py$",
        r"constants\.py$",
        r"parameters\.py$"
    ],
    "core": [
        r"core\.py$",
        r"main\.py$",
        r"engine\.py$",
        r"processor\.py$",
        r"manager\.py$",
        r"director\.py$",
        r"coordinator\.py$"
    ],
    "models": [
        r"models\.py$",
        r"schemas\.py$",
        r"entities\.py$",
        r"data_models\.py$",
        r"dto\.py$",
        r"domain\.py$"
    ],
    "repositories": [
        r"repositories\.py$",
        r"repo\.py$",
        r"storage\.py$",
        r"dao\.py$",
        r"data_access\.py$",
        r"persistence\.py$",
        r"database\.py$",
        r"db\.py$"
    ],
    "services": [
        r"services\.py$",
        r"service\.py$",
        r"business\.py$",
        r"logic\.py$",
        r"operations\.py$",
        r"workflow\.py$",
        r"process\.py$"
    ],
    "utils": [
        r"utils\.py$",
        r"helpers\.py$",
        r"common\.py$",
        r"tools\.py$",
        r"utilities\.py$",
        r"functions\.py$",
        r"lib\.py$",
        r"misc\.py$"
    ],
    "adapters": [
        r"adapters\.py$",
        r"adapter\.py$",
        r"clients\.py$",
        r"client\.py$",
        r"connectors\.py$",
        r"connector\.py$",
        r"external\.py$",
        r"integration\.py$"
    ],
    "interfaces": [
        r"interfaces\.py$",
        r"interface\.py$",
        r"contracts\.py$",
        r"contract\.py$",
        r"protocols\.py$",
        r"protocol\.py$",
        r"abstracts\.py$",
        r"abstract\.py$"
    ],
    "tests": [
        r"test_.*\.py$",
        r".*_test\.py$",
        r"tests\.py$",
        r"conftest\.py$",
        r"fixtures\.py$",
        r"mocks\.py$"
    ]
}

class FileReorganizer:
    """Reorganizes files in the forex trading platform."""

    def __init__(self, project_root: str, service_name: Optional[str] = None, dry_run: bool = False):
        """
        Initialize the reorganizer.

        Args:
            project_root: Root directory of the project
            service_name: Name of the service to reorganize (None for all services)
            dry_run: If True, only print what would be done without making changes
        """
        self.project_root = project_root
        self.service_name = service_name
        self.dry_run = dry_run
        self.services = []
        self.changes = []
        self.import_updates = {}

    def identify_services(self) -> None:
        """Identify services in the project based on directory structure."""
        logger.info("Identifying services...")

        # Look for service directories
        for item in os.listdir(self.project_root):
            item_path = os.path.join(self.project_root, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it's likely a service
                if (
                    item.endswith('-service') or
                    item.endswith('_service') or
                    item.endswith('-api') or
                    item.endswith('-engine') or
                    'service' in item.lower() or
                    'api' in item.lower()
                ):
                    self.services.append(item)

        logger.info(f"Identified {len(self.services)} services")

    def classify_file(self, file_path: str) -> str:
        """
        Classify a file based on its name and content.

        Args:
            file_path: Path to the file

        Returns:
            Directory where the file should be placed
        """
        file_name = os.path.basename(file_path)

        # Check file name patterns
        for directory, patterns in FILE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, file_name, re.IGNORECASE):
                    return directory

        # If no match by name, try to analyze content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for API-related imports
            if re.search(r'import (fastapi|flask|django|tornado|aiohttp)', content, re.IGNORECASE):
                return "api"

            # Check for model-related imports
            if re.search(r'import (pydantic|sqlalchemy|dataclasses)', content, re.IGNORECASE):
                return "models"

            # Check for repository-related imports
            if re.search(r'import (sqlalchemy|pymongo|redis|elasticsearch)', content, re.IGNORECASE):
                return "repositories"

            # Check for class definitions
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name.lower()

                        if any(pattern in class_name for pattern in ['api', 'controller', 'endpoint', 'route']):
                            return "api"

                        if any(pattern in class_name for pattern in ['model', 'schema', 'entity']):
                            return "models"

                        if any(pattern in class_name for pattern in ['repository', 'repo', 'dao']):
                            return "repositories"

                        if any(pattern in class_name for pattern in ['service', 'manager', 'handler']):
                            return "services"

                        if any(pattern in class_name for pattern in ['adapter', 'client', 'connector']):
                            return "adapters"

                        if any(pattern in class_name for pattern in ['interface', 'protocol', 'abstract']):
                            return "interfaces"
            except SyntaxError:
                # If we can't parse the file, just continue
                pass
        except Exception as e:
            logger.warning(f"Error analyzing file content: {e}")

        # Default to core if we can't classify
        return "core"

    def update_imports(self, file_path: str, old_to_new_paths: Dict[str, str]) -> bool:
        """
        Update import statements in a file.

        Args:
            file_path: Path to the file
            old_to_new_paths: Mapping of old paths to new paths

        Returns:
            True if the file was updated, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            updated_content = content

            # Get the service name from the file path
            service_path = None
            for service in self.services:
                if service in file_path:
                    service_path = os.path.join(self.project_root, service)
                    break

            if not service_path:
                return False

            # Update import statements
            for old_path, new_path in old_to_new_paths.items():
                # Skip if the old path is not in the same service
                if service_path not in old_path:
                    continue

                # Get the module paths relative to the service
                old_rel_path = os.path.relpath(old_path, service_path)
                new_rel_path = os.path.relpath(new_path, service_path)

                # Convert to module paths
                old_module = os.path.splitext(old_rel_path)[0].replace(os.path.sep, '.')
                new_module = os.path.splitext(new_rel_path)[0].replace(os.path.sep, '.')

                # Skip if the module paths are the same
                if old_module == new_module:
                    continue

                # Update import statements
                updated_content = re.sub(
                    r'from\s+' + re.escape(old_module) + r'\s+import',
                    f'from {new_module} import',
                    updated_content
                )
                updated_content = re.sub(
                    r'import\s+' + re.escape(old_module) + r'(\s|$)',
                    f'import {new_module}\\1',
                    updated_content
                )

            # Write updated content if changes were made
            if updated_content != content:
                if not self.dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)

                self.changes.append(f"Updated imports in {file_path}")
                return True

        except Exception as e:
            logger.error(f"Error updating imports in {file_path}: {e}")

        return False

    def reorganize_service(self, service_name: str) -> List[str]:
        """
        Reorganize files in a service.

        Args:
            service_name: Name of the service to reorganize

        Returns:
            List of changes made
        """
        logger.info(f"Reorganizing files in {service_name}...")

        service_path = os.path.join(self.project_root, service_name)
        changes = []

        # Check if service directory exists
        if not os.path.exists(service_path):
            logger.error(f"Service directory not found: {service_path}")
            return changes

        # Create standard directories if they don't exist
        for directory in STANDARD_DIRECTORIES:
            directory_path = os.path.join(service_path, directory)
            if not os.path.exists(directory_path):
                if not self.dry_run:
                    os.makedirs(directory_path, exist_ok=True)
                changes.append(f"Created directory: {directory_path}")

        # Find Python files in the service
        python_files = []
        for root, _, files in os.walk(service_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    # Skip files that are already in standard directories
                    rel_path = os.path.relpath(root, service_path)
                    if rel_path in STANDARD_DIRECTORIES:
                        continue

                    python_files.append(file_path)

        logger.info(f"Found {len(python_files)} Python files to reorganize in {service_name}")

        # Classify and move files
        old_to_new_paths = {}

        for file_path in python_files:
            # Classify the file
            directory = self.classify_file(file_path)

            # Determine new path
            file_name = os.path.basename(file_path)
            new_directory = os.path.join(service_path, directory)
            new_path = os.path.join(new_directory, file_name)

            # Skip if the file is already in the right place
            if file_path == new_path:
                continue

            # Check if the destination file already exists
            if os.path.exists(new_path):
                # If it's the same file (e.g., case-insensitive filesystem), just skip
                if os.path.samefile(file_path, new_path):
                    continue

                # Otherwise, use a different name
                base, ext = os.path.splitext(file_name)
                counter = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(new_directory, f"{base}_{counter}{ext}")
                    counter += 1

            # Move the file
            if not self.dry_run:
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.copy2(file_path, new_path)

                # Don't delete the original file yet, as we need to update imports first

            changes.append(f"Moved file: {file_path} -> {new_path}")
            old_to_new_paths[file_path] = new_path

        # Update imports in all Python files
        if old_to_new_paths:
            self.import_updates[service_name] = old_to_new_paths

            # Find all Python files in the service (including those in standard directories)
            all_python_files = []
            for root, _, files in os.walk(service_path):
                for file in files:
                    if file.endswith('.py'):
                        all_python_files.append(os.path.join(root, file))

            # Update imports in all files
            for file_path in all_python_files:
                if self.update_imports(file_path, old_to_new_paths):
                    changes.append(f"Updated imports in {file_path}")

            # Now we can delete the original files
            if not self.dry_run:
                for old_path in old_to_new_paths:
                    try:
                        os.remove(old_path)
                        changes.append(f"Deleted original file: {old_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {old_path}: {e}")

        return changes

    def reorganize_files(self) -> List[str]:
        """
        Reorganize files in the forex trading platform.

        Returns:
            List of changes made
        """
        logger.info("Starting file reorganization...")

        # Identify services
        self.identify_services()

        if not self.services:
            logger.info("No services found")
            return []

        # Filter services if a specific service was specified
        if self.service_name:
            if self.service_name in self.services:
                services_to_reorganize = [self.service_name]
            else:
                logger.error(f"Service not found: {self.service_name}")
                return []
        else:
            services_to_reorganize = self.services

        # Reorganize each service
        for service in services_to_reorganize:
            changes = self.reorganize_service(service)
            self.changes.extend(changes)

        logger.info("File reorganization complete")
        return self.changes

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Reorganize files in the forex trading platform")
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--service",
        help="Name of the service to reorganize (default: all services)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done without making changes"
    )
    args = parser.parse_args()

    # Reorganize files
    reorganizer = FileReorganizer(args.project_root, args.service, args.dry_run)
    changes = reorganizer.reorganize_files()

    # Print summary
    print("\nFile Reorganization Summary:")
    print(f"- {'Would apply' if args.dry_run else 'Applied'} {len(changes)} changes")

    if changes:
        print("\nChanges:")
        for i, change in enumerate(changes):
            print(f"  {i+1}. {change}")

    # Save results to file
    output_dir = os.path.join(args.project_root, 'tools', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'file_reorganization_changes.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        import json
        json.dump({
            'total_changes': len(changes),
            'dry_run': args.dry_run,
            'changes': changes
        }, f, indent=2)

    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
