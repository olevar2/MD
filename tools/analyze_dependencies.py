#!/usr/bin/env python3
"""
Forex Trading Platform Dependency Analyzer

This script analyzes dependencies between modules and services in the forex trading platform.
It identifies:
1. Direct imports between services
2. Circular dependencies
3. Dependency chains
4. Common dependencies

Output is a comprehensive JSON file that maps the dependency relationships.
"""

import os
import sys
import json
import re
import ast
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
import concurrent.futures
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output"

# Directories to exclude from analysis
EXCLUDE_DIRS = {
    ".git", ".github", ".pytest_cache", "__pycache__", 
    "node_modules", ".venv", "venv", "env", ".vscode"
}

# File extensions to analyze
PYTHON_EXTENSIONS = {".py"}
JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
ALL_EXTENSIONS = PYTHON_EXTENSIONS | JS_EXTENSIONS

class DependencyAnalyzer:
    """Analyzes dependencies between modules and services in the forex trading platform."""
    
    def __init__(self, project_root: Path):
        """
        Initialize the analyzer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.files = []
        self.modules = {}
        self.services = {}
        self.dependencies = defaultdict(set)
        self.service_dependencies = defaultdict(set)
        self.circular_dependencies = []
        
    def find_files(self) -> None:
        """Find all relevant files in the project."""
        logger.info(f"Finding files in {self.project_root}...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.project_root)
                
                # Skip files in excluded directories
                if any(part in EXCLUDE_DIRS for part in Path(rel_path).parts):
                    continue
                
                # Only include files with relevant extensions
                ext = os.path.splitext(file)[1].lower()
                if ext in ALL_EXTENSIONS:
                    self.files.append(file_path)
        
        logger.info(f"Found {len(self.files)} files to analyze")
    
    def identify_services(self) -> None:
        """Identify services in the project based on directory structure."""
        logger.info("Identifying services...")
        
        # Look for service directories
        service_dirs = []
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
                    service_dirs.append(item)
        
        # Create service objects
        for service_dir in service_dirs:
            service_path = os.path.join(self.project_root, service_dir)
            service_files = []
            
            # Find all files in this service
            for root, _, files in os.walk(service_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    if ext in ALL_EXTENSIONS:
                        service_files.append(file_path)
            
            # Create service object
            self.services[service_dir] = {
                'name': service_dir,
                'path': service_path,
                'files': service_files,
                'dependencies': [],
                'dependents': []
            }
        
        logger.info(f"Identified {len(self.services)} services")
    
    def analyze_python_imports(self, file_path: str) -> List[str]:
        """
        Analyze Python imports in a file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of imported modules
        """
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the Python file
            try:
                tree = ast.parse(content)
                
                # Extract imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for name in node.names:
                                imports.append(f"{node.module}.{name.name}")
            except SyntaxError:
                # Fall back to regex for files with syntax errors
                import_regex = r'^\s*(?:from\s+([a-zA-Z0-9_.]+)\s+import|import\s+([a-zA-Z0-9_.]+))'
                for line in content.splitlines():
                    match = re.match(import_regex, line)
                    if match:
                        module = match.group(1) or match.group(2)
                        if module:
                            imports.append(module)
        except Exception as e:
            logger.error(f"Error analyzing imports in {file_path}: {e}")
        
        return imports
    
    def analyze_js_imports(self, file_path: str) -> List[str]:
        """
        Analyze JavaScript/TypeScript imports in a file.
        
        Args:
            file_path: Path to the JS/TS file
            
        Returns:
            List of imported modules
        """
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract ES6 imports
            import_regex = r'import\s+(?:{[^}]*}|[^{;]+)\s+from\s+[\'"]([^\'"]+)[\'"]'
            for match in re.finditer(import_regex, content):
                imports.append(match.group(1))
            
            # Extract require imports
            require_regex = r'(?:const|let|var)\s+(?:{[^}]*}|[^{;]+)\s+=\s+require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
            for match in re.finditer(require_regex, content):
                imports.append(match.group(1))
        except Exception as e:
            logger.error(f"Error analyzing imports in {file_path}: {e}")
        
        return imports
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Analysis results for the file
        """
        result = {
            'path': file_path,
            'imports': []
        }
        
        # Determine file type
        ext = os.path.splitext(file_path)[1].lower()
        
        # Analyze imports
        if ext in PYTHON_EXTENSIONS:
            result['imports'] = self.analyze_python_imports(file_path)
        elif ext in JS_EXTENSIONS:
            result['imports'] = self.analyze_js_imports(file_path)
        
        return result
    
    def map_to_services(self) -> None:
        """Map analysis results to services."""
        logger.info("Mapping analysis results to services...")
        
        # Map files to services
        file_to_service = {}
        for service_name, service in self.services.items():
            service_path = service['path']
            for file_path in self.files:
                if file_path.startswith(service_path):
                    file_to_service[file_path] = service_name
        
        # Map modules to services
        for module_path, module in self.modules.items():
            if module_path in file_to_service:
                service_name = file_to_service[module_path]
                
                # Map dependencies between services
                for imported_module in module['imports']:
                    # Try to find which service this import belongs to
                    for other_service in self.services:
                        # Convert kebab-case to snake_case for import comparison
                        other_service_import = other_service.replace('-', '_')
                        
                        # Check if the import is from the other service
                        if (imported_module == other_service_import or 
                            imported_module.startswith(f"{other_service_import}.") or
                            imported_module.startswith(f"{other_service}.")):
                            
                            if other_service != service_name:
                                self.service_dependencies[service_name].add(other_service)
                                self.dependencies[module_path].add(imported_module)
                                break
    
    def detect_circular_dependencies(self) -> None:
        """Detect circular dependencies between services."""
        logger.info("Detecting circular dependencies...")
        
        for service, dependencies in self.service_dependencies.items():
            for dep in dependencies:
                if service in self.service_dependencies.get(dep, []):
                    self.circular_dependencies.append((service, dep))
        
        logger.info(f"Found {len(self.circular_dependencies)} circular dependencies")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the project dependencies.
        
        Returns:
            Analysis results
        """
        logger.info("Starting dependency analysis...")
        
        # Find all files
        self.find_files()
        
        # Identify services
        self.identify_services()
        
        # Analyze files
        logger.info("Analyzing files...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(self.analyze_file, file): file for file in self.files}
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    self.modules[file] = result
                    
                    # Add to dependencies
                    for imported_module in result['imports']:
                        self.dependencies[file].add(imported_module)
                except Exception as e:
                    logger.error(f"Error processing result for {file}: {e}")
        
        # Map results to services
        self.map_to_services()
        
        # Detect circular dependencies
        self.detect_circular_dependencies()
        
        # Update service dependencies and dependents
        for service, dependencies in self.service_dependencies.items():
            self.services[service]['dependencies'] = list(dependencies)
            for dep in dependencies:
                if dep in self.services:
                    if 'dependents' not in self.services[dep]:
                        self.services[dep]['dependents'] = []
                    self.services[dep]['dependents'].append(service)
        
        # Generate summary
        summary = {
            'services': self.services,
            'service_dependencies': {k: list(v) for k, v in self.service_dependencies.items()},
            'circular_dependencies': self.circular_dependencies,
            'module_dependencies': {k: list(v) for k, v in self.dependencies.items()}
        }
        
        logger.info("Dependency analysis complete")
        return summary

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze forex trading platform dependencies")
    parser.add_argument(
        "--project-root", 
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--output-dir", 
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--report-file",
        default="dependency-report.json",
        help="Name of the report file"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze dependencies
    analyzer = DependencyAnalyzer(Path(args.project_root))
    results = analyzer.analyze()
    
    # Save results
    output_path = os.path.join(args.output_dir, args.report_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Dependency analysis saved to {output_path}")
    
    # Print summary
    print("\nDependency Analysis Summary:")
    print(f"- Analyzed {len(analyzer.files)} files")
    print(f"- Found {len(analyzer.services)} services")
    print(f"- Detected {sum(len(deps) for deps in analyzer.service_dependencies.values())} service dependencies")
    print(f"- Found {len(analyzer.circular_dependencies)} circular dependencies")
    
    if analyzer.circular_dependencies:
        print("\nCircular Dependencies:")
        for service1, service2 in analyzer.circular_dependencies:
            print(f"  {service1} <-> {service2}")

if __name__ == "__main__":
    main()
