#!/usr/bin/env python3
"""
Forex Trading Platform Circular Dependency Fixer

This script fixes circular dependencies in the forex trading platform by standardizing
service naming conventions and updating import statements.

Usage:
python fix_circular_dependencies.py [--project-root <project_root>]
"""

import os
import sys
import re
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_REPORT_PATH = r"D:\MD\forex_trading_platform\tools\output\circular_dependencies_report.json"

class CircularDependencyFixer:
    """Fixes circular dependencies in the forex trading platform."""

    def __init__(self, project_root: str, report_path: str):
        """
        Initialize the fixer.

        Args:
            project_root: Root directory of the project
            report_path: Path to the circular dependencies report
        """
        self.project_root = project_root
        self.report_path = report_path
        self.circular_dependencies = []
        self.fixes = []

    def load_report(self) -> None:
        """Load the circular dependencies report."""
        try:
            with open(self.report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
                self.circular_dependencies = report.get('circular_dependencies', [])

            logger.info(f"Loaded {len(self.circular_dependencies)} circular dependencies from {self.report_path}")
        except Exception as e:
            logger.error(f"Error loading circular dependencies report: {e}")
            self.circular_dependencies = []

    def fix_naming_inconsistencies(self) -> None:
        """Fix naming inconsistencies between kebab-case and snake_case."""
        logger.info("Fixing naming inconsistencies...")

        for dep in self.circular_dependencies:
            service1 = dep.get('service1', '')
            service2 = dep.get('service2', '')

            # Check if this is a naming inconsistency
            if service1.replace('-', '_') == service2 or service2.replace('-', '_') == service1:
                logger.info(f"Found naming inconsistency: {service1} <-> {service2}")

                # Determine which service to standardize
                if '-' in service1 and '_' in service2:
                    # Standardize to kebab-case
                    old_name = service2
                    new_name = service1
                elif '_' in service1 and '-' in service2:
                    # Standardize to kebab-case
                    old_name = service1
                    new_name = service2
                else:
                    # Default to kebab-case
                    old_name = service2 if '_' in service2 else service1
                    new_name = service1 if '-' in service1 else service2

                # Check if directories exist
                old_path = os.path.join(self.project_root, old_name)
                new_path = os.path.join(self.project_root, new_name)

                if os.path.exists(old_path) and os.path.exists(new_path):
                    logger.info(f"Both directories exist: {old_path} and {new_path}")

                    # This is likely a false positive due to import inconsistencies
                    # Update imports in all Python files
                    self.update_imports(old_name, new_name)
                elif os.path.exists(old_path):
                    logger.info(f"Renaming directory: {old_path} -> {new_path}")

                    # Rename directory
                    try:
                        shutil.move(old_path, new_path)
                        self.fixes.append(f"Renamed directory: {old_path} -> {new_path}")

                        # Update imports in all Python files
                        self.update_imports(old_name, new_name)
                    except Exception as e:
                        logger.error(f"Error renaming directory: {e}")
                elif os.path.exists(new_path):
                    logger.info(f"Directory already exists: {new_path}")

                    # Update imports in all Python files
                    self.update_imports(old_name, new_name)
                else:
                    logger.warning(f"Neither directory exists: {old_path} or {new_path}")

    def update_imports(self, old_name: str, new_name: str) -> None:
        """
        Update import statements in all Python files.

        Args:
            old_name: Old service name
            new_name: New service name
        """
        logger.info(f"Updating imports: {old_name} -> {new_name}")

        # Convert to module names
        old_module = old_name.replace('-', '_')
        new_module = new_name.replace('-', '_')

        # Find all Python files
        python_files = []
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))

        logger.info(f"Found {len(python_files)} Python files to update")

        # Update imports in each file
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check if file contains imports from the old module
                if f"from {old_module}" in content or f"import {old_module}" in content:
                    # Replace imports
                    updated_content = content.replace(f"from {old_module}", f"from {new_module}")
                    updated_content = updated_content.replace(f"import {old_module}", f"import {new_module}")

                    # Write updated content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)

                    self.fixes.append(f"Updated imports in {file_path}")
            except Exception as e:
                logger.error(f"Error updating imports in {file_path}: {e}")

    def fix_circular_dependencies(self) -> List[str]:
        """
        Fix circular dependencies.

        Returns:
            List of fixes applied
        """
        logger.info("Starting circular dependency fixes...")

        # Load the circular dependencies report
        self.load_report()

        if not self.circular_dependencies:
            logger.info("No circular dependencies to fix")
            return []

        # Fix naming inconsistencies
        self.fix_naming_inconsistencies()

        logger.info("Circular dependency fixes complete")
        return self.fixes

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix circular dependencies")
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help="Path to the circular dependencies report"
    )
    args = parser.parse_args()

    # Fix circular dependencies
    fixer = CircularDependencyFixer(args.project_root, args.report_path)
    fixes = fixer.fix_circular_dependencies()

    # Print summary
    print("\nCircular Dependency Fixes Summary:")
    print(f"- Applied {len(fixes)} fixes")

    if fixes:
        print("\nFixes:")
        for i, fix in enumerate(fixes):
            print(f"  {i+1}. {fix}")

    # Save results to file
    output_dir = os.path.join(args.project_root, 'tools', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'circular_dependency_fixes.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_fixes': len(fixes),
            'fixes': fixes
        }, f, indent=2)

    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
