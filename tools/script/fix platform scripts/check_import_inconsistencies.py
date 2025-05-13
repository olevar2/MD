#!/usr/bin/env python3
"""
Forex Trading Platform Import Inconsistency Checker

This script checks for inconsistencies in import statements across the forex trading platform.
It identifies cases where the same service is imported using different naming conventions.

Usage:
python check_import_inconsistencies.py [--project-root <project_root>]
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import concurrent.futures

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

class ImportInconsistencyChecker:
    """Checks for inconsistencies in import statements across the forex trading platform."""
    
    def __init__(self, project_root: str):
        """
        Initialize the checker.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.files = []
        self.services = {}
        self.imports = {}
        self.inconsistencies = []
    
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
                    # Store both kebab-case and snake_case versions
                    kebab_name = item
                    snake_name = item.replace('-', '_')
                    
                    self.services[kebab_name] = {
                        'path': item_path,
                        'kebab_name': kebab_name,
                        'snake_name': snake_name
                    }
        
        logger.info(f"Identified {len(self.services)} services")
    
    def extract_imports(self, file_path: str) -> Dict[str, List[str]]:
        """
        Extract import statements from a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary of imported modules and their import statements
        """
        imports = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract import statements
            import_pattern = r'^(?:from\s+([a-zA-Z0-9_.]+)\s+import|import\s+([a-zA-Z0-9_.]+)).*$'
            
            for line in content.splitlines():
                match = re.match(import_pattern, line)
                if match:
                    module = match.group(1) or match.group(2)
                    if module:
                        # Get the top-level module
                        top_module = module.split('.')[0]
                        
                        if top_module not in imports:
                            imports[top_module] = []
                        
                        imports[top_module].append(line)
        
        except Exception as e:
            logger.error(f"Error extracting imports from {file_path}: {e}")
        
        return imports
    
    def check_inconsistencies(self) -> None:
        """Check for inconsistencies in import statements."""
        logger.info("Checking for import inconsistencies...")
        
        # Extract imports from all files
        all_imports = {}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(self.extract_imports, file): file for file in self.files}
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    imports = future.result()
                    all_imports[file] = imports
                except Exception as e:
                    logger.error(f"Error processing result for {file}: {e}")
        
        # Check for inconsistencies
        for service_info in self.services.values():
            kebab_name = service_info['kebab_name']
            snake_name = service_info['snake_name']
            
            # Skip if kebab and snake names are the same
            if kebab_name == snake_name:
                continue
            
            # Check for files that import both kebab and snake versions
            for file_path, imports in all_imports.items():
                kebab_imports = imports.get(kebab_name.replace('-', '_'), [])
                snake_imports = imports.get(snake_name, [])
                
                if kebab_imports and snake_imports:
                    self.inconsistencies.append({
                        'file': file_path,
                        'service': kebab_name,
                        'kebab_imports': kebab_imports,
                        'snake_imports': snake_imports
                    })
        
        logger.info(f"Found {len(self.inconsistencies)} import inconsistencies")
    
    def check(self) -> Dict[str, Any]:
        """
        Check for import inconsistencies.
        
        Returns:
            Check results
        """
        logger.info("Starting import inconsistency check...")
        
        # Find all files
        self.find_files()
        
        # Identify services
        self.identify_services()
        
        # Check for inconsistencies
        self.check_inconsistencies()
        
        # Generate summary
        summary = {
            'services': list(self.services.keys()),
            'inconsistencies': self.inconsistencies
        }
        
        logger.info("Import inconsistency check complete")
        return summary

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check for import inconsistencies")
    parser.add_argument(
        "--project-root", 
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    args = parser.parse_args()
    
    # Check for import inconsistencies
    checker = ImportInconsistencyChecker(args.project_root)
    results = checker.check()
    
    # Print summary
    print("\nImport Inconsistency Check Summary:")
    print(f"- Analyzed {len(results['services'])} services")
    print(f"- Found {len(results['inconsistencies'])} import inconsistencies")
    
    if results['inconsistencies']:
        print("\nInconsistencies:")
        for i, inconsistency in enumerate(results['inconsistencies']):
            print(f"\n{i+1}. File: {inconsistency['file']}")
            print(f"   Service: {inconsistency['service']}")
            print("   Kebab-case imports:")
            for imp in inconsistency['kebab_imports']:
                print(f"     {imp}")
            print("   Snake_case imports:")
            for imp in inconsistency['snake_imports']:
                print(f"     {imp}")

if __name__ == "__main__":
    main()
