#!/usr/bin/env python3
"""
Forex Trading Platform Service Naming Standardization

This script standardizes service naming conventions across the forex trading platform.
It resolves naming inconsistencies between kebab-case and snake_case by updating
import statements and directory references.

Usage:
python standardize_service_naming.py [--project-root <project_root>] [--naming-convention <kebab|snake>]
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import concurrent.futures
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"

# Directories to exclude from analysis
EXCLUDE_DIRS = {
    ".git", ".github", ".pytest_cache", "__pycache__", 
    "node_modules", ".venv", "venv", "env", ".vscode"
}

class ServiceNamingStandardizer:
    """Standardizes service naming conventions across the forex trading platform."""
    
    def __init__(self, project_root: str, naming_convention: str = 'kebab'):
        """
        Initialize the standardizer.
        
        Args:
            project_root: Root directory of the project
            naming_convention: Naming convention to use ('kebab' or 'snake')
        """
        self.project_root = project_root
        self.naming_convention = naming_convention
        self.files = []
        self.services = {}
        self.service_mappings = {}
        self.changes = []
    
    def find_files(self) -> None:
        """Find all relevant files in the project."""
        logger.info(f"Finding files in {self.project_root}...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_root)
                    
                    # Skip files in excluded directories
                    if any(part in EXCLUDE_DIRS for part in Path(rel_path).parts):
                        continue
                    
                    self.files.append(file_path)
        
        logger.info(f"Found {len(self.files)} Python files to analyze")
    
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
                    self.services[item] = item_path
        
        logger.info(f"Identified {len(self.services)} services")
    
    def create_service_mappings(self) -> None:
        """Create mappings between current and standardized service names."""
        logger.info("Creating service name mappings...")
        
        for service_name in self.services:
            # Determine standardized name
            if self.naming_convention == 'kebab':
                # Convert to kebab-case
                standardized_name = service_name.replace('_', '-')
            else:
                # Convert to snake_case
                standardized_name = service_name.replace('-', '_')
            
            # Add to mappings if different
            if service_name != standardized_name:
                self.service_mappings[service_name] = standardized_name
                logger.info(f"Mapping: {service_name} -> {standardized_name}")
        
        logger.info(f"Created {len(self.service_mappings)} service name mappings")
    
    def update_imports(self, file_path: str) -> List[Tuple[str, str, str]]:
        """
        Update import statements in a file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of changes made (original, replacement, line)
        """
        changes = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            updated_content = content
            
            # Update import statements
            for old_name, new_name in self.service_mappings.items():
                # Convert to module name (kebab-case to snake_case for imports)
                old_module = old_name.replace('-', '_')
                new_module = new_name.replace('-', '_')
                
                # Find and replace import statements
                import_patterns = [
                    (f"from {old_module} import", f"from {new_module} import"),
                    (f"from {old_module}.", f"from {new_module}."),
                    (f"import {old_module}", f"import {new_module}"),
                    (f"import {old_module} as", f"import {new_module} as")
                ]
                
                for pattern, replacement in import_patterns:
                    if pattern in updated_content:
                        # Find all occurrences with line context
                        for match in re.finditer(f"^{re.escape(pattern)}.*$", updated_content, re.MULTILINE):
                            original = match.group(0)
                            updated = original.replace(pattern, replacement)
                            changes.append((original, updated, original))
                        
                        # Replace in content
                        updated_content = updated_content.replace(pattern, replacement)
            
            # Write updated content if changes were made
            if updated_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
        
        except Exception as e:
            logger.error(f"Error updating imports in {file_path}: {e}")
        
        return changes
    
    def rename_directories(self) -> List[Tuple[str, str]]:
        """
        Rename service directories to follow the standardized naming convention.
        
        Returns:
            List of directory renames (old_path, new_path)
        """
        renames = []
        
        for old_name, new_name in self.service_mappings.items():
            old_path = os.path.join(self.project_root, old_name)
            new_path = os.path.join(self.project_root, new_name)
            
            # Skip if source doesn't exist or destination already exists
            if not os.path.exists(old_path) or os.path.exists(new_path):
                continue
            
            try:
                # Rename directory
                shutil.move(old_path, new_path)
                renames.append((old_path, new_path))
                logger.info(f"Renamed directory: {old_path} -> {new_path}")
            except Exception as e:
                logger.error(f"Error renaming directory {old_path}: {e}")
        
        return renames
    
    def update_module_references(self, file_path: str) -> List[Tuple[str, str, str]]:
        """
        Update module references in a file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of changes made (original, replacement, line)
        """
        changes = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            updated_content = content
            
            # Update module references
            for old_name, new_name in self.service_mappings.items():
                # Find and replace string references to module names
                if f"'{old_name}'" in updated_content or f'"{old_name}"' in updated_content:
                    # Find all occurrences with line context
                    for match in re.finditer(f"['\"]({re.escape(old_name)})['\"]", updated_content):
                        line_start = updated_content.rfind('\n', 0, match.start()) + 1
                        line_end = updated_content.find('\n', match.end())
                        if line_end == -1:
                            line_end = len(updated_content)
                        
                        original_line = updated_content[line_start:line_end]
                        updated_line = original_line.replace(f"'{old_name}'", f"'{new_name}'").replace(f'"{old_name}"', f'"{new_name}"')
                        
                        changes.append((original_line, updated_line, original_line))
                    
                    # Replace in content
                    updated_content = updated_content.replace(f"'{old_name}'", f"'{new_name}'").replace(f'"{old_name}"', f'"{new_name}"')
            
            # Write updated content if changes were made
            if updated_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
        
        except Exception as e:
            logger.error(f"Error updating module references in {file_path}: {e}")
        
        return changes
    
    def standardize(self) -> Dict[str, Any]:
        """
        Standardize service naming conventions.
        
        Returns:
            Standardization results
        """
        logger.info("Starting service naming standardization...")
        
        # Find all files
        self.find_files()
        
        # Identify services
        self.identify_services()
        
        # Create service mappings
        self.create_service_mappings()
        
        if not self.service_mappings:
            logger.info("No service name standardization needed")
            return {
                'services': list(self.services.keys()),
                'mappings': {},
                'import_changes': [],
                'reference_changes': [],
                'directory_renames': []
            }
        
        # Update imports
        logger.info("Updating import statements...")
        import_changes = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(self.update_imports, file): file for file in self.files}
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    changes = future.result()
                    if changes:
                        import_changes.extend([(file, original, replacement) for original, replacement, _ in changes])
                except Exception as e:
                    logger.error(f"Error processing result for {file}: {e}")
        
        logger.info(f"Updated {len(import_changes)} import statements")
        
        # Update module references
        logger.info("Updating module references...")
        reference_changes = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(self.update_module_references, file): file for file in self.files}
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    changes = future.result()
                    if changes:
                        reference_changes.extend([(file, original, replacement) for original, replacement, _ in changes])
                except Exception as e:
                    logger.error(f"Error processing result for {file}: {e}")
        
        logger.info(f"Updated {len(reference_changes)} module references")
        
        # Rename directories
        logger.info("Renaming directories...")
        directory_renames = self.rename_directories()
        
        logger.info(f"Renamed {len(directory_renames)} directories")
        
        # Generate summary
        summary = {
            'services': list(self.services.keys()),
            'mappings': self.service_mappings,
            'import_changes': import_changes,
            'reference_changes': reference_changes,
            'directory_renames': directory_renames
        }
        
        logger.info("Service naming standardization complete")
        return summary

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Standardize service naming conventions")
    parser.add_argument(
        "--project-root", 
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--naming-convention",
        choices=["kebab", "snake"],
        default="kebab",
        help="Naming convention to use ('kebab' for kebab-case, 'snake' for snake_case)"
    )
    args = parser.parse_args()
    
    # Standardize service naming
    standardizer = ServiceNamingStandardizer(
        args.project_root,
        args.naming_convention
    )
    results = standardizer.standardize()
    
    # Print summary
    print("\nService Naming Standardization Summary:")
    print(f"- Analyzed {len(results['services'])} services")
    print(f"- Created {len(results['mappings'])} service name mappings")
    print(f"- Updated {len(results['import_changes'])} import statements")
    print(f"- Updated {len(results['reference_changes'])} module references")
    print(f"- Renamed {len(results['directory_renames'])} directories")
    
    if results['mappings']:
        print("\nService Name Mappings:")
        for old_name, new_name in results['mappings'].items():
            print(f"  {old_name} -> {new_name}")

if __name__ == "__main__":
    main()
