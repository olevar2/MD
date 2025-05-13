#!/usr/bin/env python3
"""
Architecture Analyzer Script

This script analyzes the codebase to identify architectural issues such as:
- Circular dependencies
- High coupling between services
- Inconsistent naming conventions
- Duplicate component implementations
- MCP-related components
"""

import os
import re
import json
import sys
from pathlib import Path
from collections import defaultdict
import importlib.util
import ast
from typing import Dict, List, Set, Tuple, Any, Optional

# Constants
IGNORED_DIRS = {'.git', '.github', '.venv', '.pytest_cache', '__pycache__', 'node_modules'}
SERVICE_DIRS = {
    'analysis-engine-service', 
    'analysis_engine',
    'api-gateway', 
    'common-lib', 
    'common-js-lib',
    'core-foundations',
    'data-management-service',
    'data-pipeline-service',
    'feature-store-service',
    'feature_store_service',
    'ml-integration-service',
    'ml_workbench-service',
    'model-registry-service',
    'monitoring-alerting-service',
    'portfolio-management-service',
    'risk-management-service',
    'strategy-execution-engine',
    'trading-gateway-service',
    'ui-service'
}

class ArchitectureAnalyzer:
    """
    ArchitectureAnalyzer class.
    
    Attributes:
        Add attributes here
    """

    def __init__(self, root_dir: str):
    """
      init  .
    
    Args:
        root_dir: Description of root_dir
    
    """

        self.root_dir = Path(root_dir)
        self.dependencies = defaultdict(set)
        self.imports = defaultdict(list)
        self.files_by_service = defaultdict(list)
        self.service_dependencies = defaultdict(set)
        self.circular_dependencies = []
        self.naming_issues = []
        self.duplicate_components = []
        self.mcp_components = []
        
    def analyze(self):
        """Run the full analysis"""
        print("Analyzing codebase architecture...")
        self._collect_files()
        self._analyze_dependencies()
        self._detect_circular_dependencies()
        self._analyze_naming_conventions()
        self._detect_duplicate_components()
        self._find_mcp_components()
        
    def _collect_files(self):
        """Collect all Python files in the codebase"""
        print("Collecting files...")
        for root, dirs, files in os.walk(self.root_dir):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            
            path = Path(root)
            relative_path = path.relative_to(self.root_dir)
            
            # Determine which service this file belongs to
            service = None
            for part in relative_path.parts:
                if part in SERVICE_DIRS:
                    service = part
                    break
            
            for file in files:
                if file.endswith('.py'):
                    file_path = path / file
                    if service:
                        self.files_by_service[service].append(file_path)
    
    def _analyze_dependencies(self):
        """Analyze dependencies between modules"""
        print("Analyzing dependencies...")
        for service, files in self.files_by_service.items():
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse the Python file
                    tree = ast.parse(content)
                    
                    # Extract imports
                    module_path = self._get_module_path(file_path)
                    imports = self._extract_imports(tree)
                    self.imports[module_path] = imports
                    
                    # Track service dependencies
                    for imp in imports:
                        # Check if import is from another service
                        for other_service in SERVICE_DIRS:
                            if imp.startswith(other_service.replace('-', '_')):
                                if other_service != service:
                                    self.service_dependencies[service].add(other_service)
                                    self.dependencies[module_path].add(imp)
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
    
    def _get_module_path(self, file_path: Path) -> str:
        """Convert file path to module path"""
        rel_path = file_path.relative_to(self.root_dir)
        module_path = str(rel_path).replace('/', '.').replace('\\', '.').replace('.py', '')
        return module_path
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract imports from AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for name in node.names:
                        imports.append(f"{node.module}.{name.name}")
        return imports
    
    def _detect_circular_dependencies(self):
        """Detect circular dependencies between modules"""
        print("Detecting circular dependencies...")
        visited = set()
        path = []
        
        def dfs(module):
    """
    Dfs.
    
    Args:
        module: Description of module
    
    """

            if module in path:
                # Found a cycle
                cycle_start = path.index(module)
                self.circular_dependencies.append(path[cycle_start:] + [module])
                return
            
            if module in visited:
                return
            
            visited.add(module)
            path.append(module)
            
            for dep in self.dependencies.get(module, []):
                dfs(dep)
            
            path.pop()
        
        for module in self.dependencies:
            dfs(module)
    
    def _analyze_naming_conventions(self):
        """Analyze naming conventions"""
        print("Analyzing naming conventions...")
        # Check for inconsistent directory naming
        kebab_case_dirs = []
        snake_case_dirs = []
        
        for item in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, item)) and item not in IGNORED_DIRS:
                if '-' in item:
                    kebab_case_dirs.append(item)
                elif '_' in item:
                    snake_case_dirs.append(item)
        
        # Check for duplicate directories with different naming conventions
        kebab_to_snake = {d.replace('-', '_'): d for d in kebab_case_dirs}
        snake_to_kebab = {d: kebab_to_snake.get(d) for d in snake_case_dirs if d in kebab_to_snake}
        
        for snake, kebab in snake_to_kebab.items():
            self.naming_issues.append({
                'type': 'duplicate_directory',
                'snake_case': snake,
                'kebab_case': kebab
            })
        
        # Check for inconsistent file naming
        for service, files in self.files_by_service.items():
            for file_path in files:
                file_name = file_path.name
                if not file_name.islower() and not file_name.startswith('__'):
                    self.naming_issues.append({
                        'type': 'inconsistent_file_naming',
                        'file': str(file_path),
                        'service': service
                    })
    
    def _detect_duplicate_components(self):
        """Detect potential duplicate component implementations"""
        print("Detecting duplicate components...")
        # Look for similar class names across services
        class_definitions = defaultdict(list)
        
        for service, files in self.files_by_service.items():
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_name = node.name
                            class_definitions[class_name].append({
                                'service': service,
                                'file': str(file_path),
                                'line': node.lineno
                            })
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
        
        # Find classes with multiple implementations
        for class_name, implementations in class_definitions.items():
            if len(implementations) > 1 and len(set(impl['service'] for impl in implementations)) > 1:
                # Skip common names like "Config", "Settings", "BaseModel", etc.
                if class_name not in {'Config', 'Settings', 'BaseModel', 'Base', 'Test', 'Client'}:
                    self.duplicate_components.append({
                        'class_name': class_name,
                        'implementations': implementations
                    })
    
    def _find_mcp_components(self):
        """Find MCP-related components"""
        print("Finding MCP-related components...")
        for service, files in self.files_by_service.items():
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'mcp' in content.lower():
                        self.mcp_components.append({
                            'service': service,
                            'file': str(file_path)
                        })
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a report of the analysis results"""
        return {
            'circular_dependencies': self.circular_dependencies,
            'service_dependencies': {k: list(v) for k, v in self.service_dependencies.items()},
            'naming_issues': self.naming_issues,
            'duplicate_components': self.duplicate_components,
            'mcp_components': self.mcp_components,
            'stats': {
                'total_files': sum(len(files) for files in self.files_by_service.values()),
                'total_services': len(self.files_by_service),
                'total_dependencies': sum(len(deps) for deps in self.dependencies.values()),
                'circular_dependencies_count': len(self.circular_dependencies),
                'naming_issues_count': len(self.naming_issues),
                'duplicate_components_count': len(self.duplicate_components),
                'mcp_components_count': len(self.mcp_components)
            }
        }

def main():
    """
    Main.
    
    """

    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = os.getcwd()
    
    analyzer = ArchitectureAnalyzer(root_dir)
    analyzer.analyze()
    
    report = analyzer.generate_report()
    
    # Save the report to a file
    output_file = os.path.join(root_dir, 'architecture_analysis_report.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Analysis complete. Report saved to {output_file}")
    
    # Print summary
    print("\nSummary:")
    for key, value in report['stats'].items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
