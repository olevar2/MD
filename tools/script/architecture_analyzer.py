#!/usr/bin/env python3
"""
Forex Trading Platform Architecture Analyzer

This script performs deep analysis of the forex trading platform architecture:
1. Import statements and dependencies between modules
2. Service-to-service communication patterns
3. API contracts and data flows
4. Event patterns and message passing
5. Database schema relationships
6. Configuration settings that define component interactions

Output is a comprehensive JSON file that maps the relationships between components.
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
CONFIG_EXTENSIONS = {".json", ".yml", ".yaml", ".toml", ".ini", ".env"}
SQL_EXTENSIONS = {".sql"}
ALL_EXTENSIONS = PYTHON_EXTENSIONS | JS_EXTENSIONS | CONFIG_EXTENSIONS | SQL_EXTENSIONS

# Patterns for detecting different architectural elements
API_PATTERNS = {
    'fastapi': [
        r'@(app|router|api_router)\.(get|post|put|delete|patch|options|head)\s*\(\s*[\'"]([^\'"]+)[\'"]',
        r'@(app|router|api_router)\.([a-z]+)\s*\(\s*[\'"]([^\'"]+)[\'"]'
    ],
    'flask': [
        r'@(app|blueprint)\.route\s*\(\s*[\'"]([^\'"]+)[\'"]',
        r'@(app|blueprint)\.(get|post|put|delete|patch|options|head)\s*\(\s*[\'"]([^\'"]+)[\'"]'
    ],
    'django': [
        r'path\s*\(\s*[\'"]([^\'"]+)[\'"],\s*([^,]+)',
        r'url\s*\(\s*[\'"]([^\'"]+)[\'"],\s*([^,]+)'
    ],
    'rest_client': [
        r'(get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]',
        r'requests\.(get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]',
        r'client\.(get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]'
    ]
}

EVENT_PATTERNS = [
    r'emit\s*\(\s*[\'"]([^\'"]+)[\'"]',
    r'publish\s*\(\s*[\'"]([^\'"]+)[\'"]',
    r'subscribe\s*\(\s*[\'"]([^\'"]+)[\'"]',
    r'on\s*\(\s*[\'"]([^\'"]+)[\'"]',
    r'dispatch\s*\(\s*[\'"]([^\'"]+)[\'"]',
    r'EventEmitter',
    r'event_bus',
    r'message_broker',
    r'kafka',
    r'rabbitmq',
    r'pubsub',
    r'KafkaProducer',
    r'KafkaConsumer'
]

DATABASE_PATTERNS = [
    r'(SQLAlchemy|sqlalchemy|create_engine|session|Session|Base|Column|relationship|backref)',
    r'(psycopg2|pymysql|sqlite3|mysql|postgresql|mongodb|pymongo|redis)',
    r'(Model|models|schema|Schema|Table|table|field|Field|ForeignKey|primary_key)',
    r'(query|Query|filter|filter_by|join|order_by|group_by|having|limit|offset)',
    r'(insert|update|delete|select|where|from|join|on|and|or|not|like|in|between)',
    r'(commit|rollback|flush|refresh|expire|expunge|merge|add|remove|execute)'
]

CONFIG_PATTERNS = [
    r'config\s*\[\s*[\'"]([^\'"]+)[\'"]\s*\]',
    r'settings\s*\.\s*([A-Za-z0-9_]+)',
    r'os\.environ\s*\[\s*[\'"]([^\'"]+)[\'"]\s*\]',
    r'os\.getenv\s*\(\s*[\'"]([^\'"]+)[\'"]',
    r'app\.config\s*\[\s*[\'"]([^\'"]+)[\'"]\s*\]',
    r'ConfigParser',
    r'load_dotenv'
]

class ArchitectureAnalyzer:
    """Analyzes the architecture of the forex trading platform."""
    
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
        self.apis = {}
        self.events = {}
        self.database_models = {}
        self.configs = {}
        self.dependencies = defaultdict(set)
        self.service_dependencies = defaultdict(set)
        
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
                'apis': [],
                'events': [],
                'database_models': [],
                'configs': []
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
    
    def detect_apis(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Detect API endpoints in a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of API endpoints
        """
        apis = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for API patterns
            for api_type, patterns in API_PATTERNS.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, content):
                        if api_type == 'fastapi':
                            decorator, method, path = match.groups()
                            apis.append({
                                'type': api_type,
                                'method': method,
                                'path': path,
                                'file': file_path
                            })
                        elif api_type == 'flask':
                            if len(match.groups()) == 2:
                                decorator, path = match.groups()
                                method = "get"  # Default method
                            else:
                                decorator, method, path = match.groups()
                            apis.append({
                                'type': api_type,
                                'method': method,
                                'path': path,
                                'file': file_path
                            })
                        elif api_type == 'django':
                            path, view = match.groups()
                            apis.append({
                                'type': api_type,
                                'method': 'any',
                                'path': path,
                                'file': file_path
                            })
                        elif api_type == 'rest_client':
                            if len(match.groups()) == 2:
                                method, url = match.groups()
                                apis.append({
                                    'type': 'api_client',
                                    'method': method,
                                    'url': url,
                                    'file': file_path
                                })
        except Exception as e:
            logger.error(f"Error detecting APIs in {file_path}: {e}")
        
        return apis
    
    def detect_events(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Detect event patterns in a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of events
        """
        events = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for event patterns
            for pattern in EVENT_PATTERNS:
                for match in re.finditer(pattern, content):
                    if len(match.groups()) > 0:
                        event_name = match.group(1)
                        events.append({
                            'name': event_name,
                            'pattern': pattern,
                            'file': file_path
                        })
                    else:
                        # Just record that this file uses events
                        events.append({
                            'pattern': pattern,
                            'file': file_path
                        })
        except Exception as e:
            logger.error(f"Error detecting events in {file_path}: {e}")
        
        return events
    
    def detect_database_models(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Detect database models in a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of database models
        """
        models = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for SQLAlchemy models
            model_regex = r'class\s+([A-Za-z0-9_]+)\s*\(\s*(?:Base|db\.Model|Model)\s*\)'
            for match in re.finditer(model_regex, content):
                model_name = match.group(1)
                models.append({
                    'name': model_name,
                    'type': 'sqlalchemy',
                    'file': file_path
                })
            
            # Check for database patterns
            for pattern in DATABASE_PATTERNS:
                if re.search(pattern, content):
                    models.append({
                        'pattern': pattern,
                        'file': file_path
                    })
                    break
        except Exception as e:
            logger.error(f"Error detecting database models in {file_path}: {e}")
        
        return models
    
    def detect_configs(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Detect configuration settings in a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of configuration settings
        """
        configs = []
        
        try:
            # Check if it's a config file based on extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext in CONFIG_EXTENSIONS:
                configs.append({
                    'type': ext[1:],  # Remove the dot
                    'file': file_path
                })
                return configs
            
            # Check for config patterns in code files
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in CONFIG_PATTERNS:
                for match in re.finditer(pattern, content):
                    if len(match.groups()) > 0:
                        config_name = match.group(1)
                        configs.append({
                            'name': config_name,
                            'pattern': pattern,
                            'file': file_path
                        })
                    else:
                        # Just record that this file uses configs
                        configs.append({
                            'pattern': pattern,
                            'file': file_path
                        })
        except Exception as e:
            logger.error(f"Error detecting configs in {file_path}: {e}")
        
        return configs
    
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
            'imports': [],
            'apis': [],
            'events': [],
            'database_models': [],
            'configs': []
        }
        
        # Determine file type
        ext = os.path.splitext(file_path)[1].lower()
        
        # Analyze imports
        if ext in PYTHON_EXTENSIONS:
            result['imports'] = self.analyze_python_imports(file_path)
        elif ext in JS_EXTENSIONS:
            result['imports'] = self.analyze_js_imports(file_path)
        
        # Detect architectural elements
        result['apis'] = self.detect_apis(file_path)
        result['events'] = self.detect_events(file_path)
        result['database_models'] = self.detect_database_models(file_path)
        result['configs'] = self.detect_configs(file_path)
        
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
                
                # Add APIs to service
                for api in module['apis']:
                    self.services[service_name]['apis'].append(api)
                
                # Add events to service
                for event in module['events']:
                    self.services[service_name]['events'].append(event)
                
                # Add database models to service
                for model in module['database_models']:
                    self.services[service_name]['database_models'].append(model)
                
                # Add configs to service
                for config in module['configs']:
                    self.services[service_name]['configs'].append(config)
                
                # Map dependencies between services
                for imported_module in module['imports']:
                    # Try to find which service this import belongs to
                    for imported_file, imported_service in file_to_service.items():
                        if imported_module in imported_file:
                            if imported_service != service_name:
                                self.service_dependencies[service_name].add(imported_service)
                                break
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the project architecture.
        
        Returns:
            Analysis results
        """
        logger.info("Starting architecture analysis...")
        
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
                    
                    # Add to global collections
                    for api in result['apis']:
                        self.apis[f"{api.get('method', 'unknown')}:{api.get('path', api.get('url', 'unknown'))}"] = api
                    
                    for event in result['events']:
                        if 'name' in event:
                            self.events[event['name']] = event
                    
                    for model in result['database_models']:
                        if 'name' in model:
                            self.database_models[model['name']] = model
                    
                    for config in result['configs']:
                        if 'name' in config:
                            self.configs[config['name']] = config
                    
                    # Add to dependencies
                    for imported_module in result['imports']:
                        self.dependencies[file].add(imported_module)
                except Exception as e:
                    logger.error(f"Error processing result for {file}: {e}")
        
        # Map results to services
        self.map_to_services()
        
        # Generate summary
        summary = {
            'services': self.services,
            'apis': self.apis,
            'events': self.events,
            'database_models': self.database_models,
            'configs': self.configs,
            'service_dependencies': {k: list(v) for k, v in self.service_dependencies.items()},
            'module_dependencies': {k: list(v) for k, v in self.dependencies.items()}
        }
        
        logger.info("Architecture analysis complete")
        return summary

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze forex trading platform architecture")
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
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze architecture
    analyzer = ArchitectureAnalyzer(Path(args.project_root))
    results = analyzer.analyze()
    
    # Save results
    output_path = os.path.join(args.output_dir, "architecture_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Architecture analysis saved to {output_path}")

if __name__ == "__main__":
    main()
