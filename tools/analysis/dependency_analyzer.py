#!/usr/bin/env python
"""
Dependency Analyzer

This script analyzes the codebase for circular dependencies between modules.
"""
import os
import sys
import re
import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Regular expressions for import statements
IMPORT_PATTERNS = [
    r'^\s*import\s+([a-zA-Z0-9_.,\s]+)(?:\s+as\s+[a-zA-Z0-9_]+)?',  # import module
    r'^\s*from\s+([a-zA-Z0-9_.]+)\s+import\s+',  # from module import ...
]

# Service directories to analyze
SERVICE_DIRS = [
    'analysis-engine-service',
    'api-gateway-service',
    'auth-service',
    'backtesting-service',
    'common-lib',
    'data-ingestion-service',
    'feature-store-service',
    'market-data-service',
    'ml-workbench-service',
    'notification-service',
    'order-execution-service',
    'portfolio-service',
    'risk-management-service',
    'strategy-service',
    'trading-service',
    'user-service',
]


def find_python_files(directory: str) -> List[str]:
    """
    Find all Python files in a directory recursively.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of Python file paths
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def extract_imports(file_path: str) -> List[str]:
    """
    Extract import statements from a Python file.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        List of imported module names
    """
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split into lines and remove comments
        lines = []
        for line in content.split('\n'):
            if '#' in line:
                line = line[:line.index('#')]
            lines.append(line)
        
        # Join lines with line continuations
        joined_lines = []
        current_line = ''
        for line in lines:
            if line.endswith('\\'):
                current_line += line[:-1].strip() + ' '
            else:
                current_line += line.strip()
                if current_line:
                    joined_lines.append(current_line)
                current_line = ''
        
        # Extract imports using regex patterns
        for line in joined_lines:
            for pattern in IMPORT_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    modules = match.group(1).split(',')
                    for module in modules:
                        module = module.strip()
                        if module:
                            imports.append(module)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    
    return imports


def normalize_module_name(module_name: str, file_path: str) -> str:
    """
    Normalize a module name to a standard format.
    
    Args:
        module_name: Module name to normalize
        file_path: Path to the file containing the import
        
    Returns:
        Normalized module name
    """
    # Handle relative imports
    if module_name.startswith('.'):
        # Get the package of the current file
        file_dir = os.path.dirname(file_path)
        package_parts = []
        
        # Walk up the directory tree for each dot
        dot_count = 0
        for char in module_name:
            if char == '.':
                dot_count += 1
            else:
                break
        
        # Get the current package
        current_package = file_dir.replace(os.sep, '.')
        package_parts = current_package.split('.')
        
        # Remove parts based on dot count
        if dot_count <= len(package_parts):
            package_parts = package_parts[:-dot_count]
        
        # Add the rest of the module name
        if len(module_name) > dot_count:
            package_parts.append(module_name[dot_count:])
        
        return '.'.join(package_parts)
    
    return module_name


def map_module_to_service(module_name: str) -> str:
    """
    Map a module name to a service.
    
    Args:
        module_name: Module name
        
    Returns:
        Service name or 'external' if not in our services
    """
    for service in SERVICE_DIRS:
        service_prefix = service.replace('-', '_')
        if module_name.startswith(service_prefix):
            return service
    
    # Check for common external modules
    common_external = [
        'os', 'sys', 'json', 're', 'time', 'datetime', 'logging',
        'numpy', 'pandas', 'matplotlib', 'sklearn', 'tensorflow',
        'torch', 'fastapi', 'pydantic', 'sqlalchemy', 'asyncio',
        'aiohttp', 'requests', 'pytest', 'unittest', 'typing',
        'collections', 'itertools', 'functools', 'math', 'random',
        'hashlib', 'uuid', 'pathlib', 'shutil', 'tempfile', 'io',
        'csv', 'xml', 'html', 'yaml', 'toml', 'configparser',
        'argparse', 'subprocess', 'multiprocessing', 'threading',
        'queue', 'socket', 'ssl', 'email', 'smtplib', 'ftplib',
        'http', 'urllib', 'base64', 'zlib', 'gzip', 'zipfile',
        'tarfile', 'pickle', 'shelve', 'dbm', 'sqlite3', 'psycopg2',
        'mysql', 'redis', 'pymongo', 'elasticsearch', 'boto3',
        'azure', 'google', 'flask', 'django', 'celery', 'kafka',
        'rabbitmq', 'zeromq', 'grpc', 'protobuf', 'avro', 'thrift',
        'jwt', 'bcrypt', 'cryptography', 'ssl', 'openssl', 'pytest',
        'nose', 'coverage', 'tox', 'sphinx', 'docutils', 'jinja2',
        'mako', 'pillow', 'opencv', 'scipy', 'statsmodels', 'sympy',
        'networkx', 'nltk', 'spacy', 'gensim', 'transformers',
        'keras', 'xgboost', 'lightgbm', 'catboost', 'dask', 'ray',
        'pyspark', 'airflow', 'prefect', 'dagster', 'mlflow', 'wandb',
        'tensorboard', 'streamlit', 'dash', 'bokeh', 'plotly',
        'seaborn', 'altair', 'folium', 'geopandas', 'shapely',
        'fiona', 'rasterio', 'gdal', 'pyproj', 'cartopy', 'basemap',
        'h5py', 'netcdf4', 'xarray', 'zarr', 'numba', 'cython',
        'pyarrow', 'parquet', 'feather', 'hdf5', 'tables', 'onnx',
        'tflite', 'torchscript', 'triton', 'tvm', 'talib', 'backtrader',
        'zipline', 'pyfolio', 'alphalens', 'empyrical', 'pymc3',
        'prophet', 'statsforecast', 'skforecast', 'tsfresh', 'tslearn',
        'pmdarima', 'fbprophet', 'kats', 'neuralprophet', 'gluonts',
        'sktime', 'tsai', 'pyts', 'stumpy', 'cesium', 'tsflex',
        'tsfel', 'seglearn', 'pykalman', 'filterpy', 'pyro', 'numpyro',
        'jax', 'flax', 'haiku', 'optax', 'dm_control', 'gym',
        'stable_baselines3', 'rllib', 'dopamine', 'acme', 'rlpyt',
        'tianshou', 'cleanrl', 'spinningup', 'rlkit', 'garage',
        'pfrl', 'rl_coach', 'ray.rllib', 'tensorflow_probability',
        'pystan', 'cmdstanpy', 'arviz', 'bambi', 'edward', 'pymc',
        'pyro', 'numpyro', 'tensorflow_probability', 'torch.distributions',
        'pomegranate', 'pgmpy', 'bayespy', 'pyAgrum', 'causalnex',
        'dowhy', 'econml', 'causalml', 'causalimpact', 'gcastle',
        'pgmpy', 'bnlearn', 'causality', 'pycausal', 'causalinference',
        'causallib', 'causalimpact', 'causalml', 'econml', 'dowhy',
        'causalnex', 'gcastle', 'pgmpy', 'bnlearn', 'causality',
        'pycausal', 'causalinference', 'causallib', 'causalimpact',
    ]
    
    for external in common_external:
        if module_name == external or module_name.startswith(external + '.'):
            return 'external'
    
    # If we can't determine the service, mark as unknown
    return 'unknown'


def build_dependency_graph(root_dir: str) -> Dict[str, Dict[str, Set[str]]]:
    """
    Build a dependency graph for all services.
    
    Args:
        root_dir: Root directory of the codebase
        
    Returns:
        Dictionary mapping services to their dependencies
    """
    dependency_graph = {}
    
    for service in SERVICE_DIRS:
        service_dir = os.path.join(root_dir, service)
        if not os.path.isdir(service_dir):
            continue
        
        service_files = find_python_files(service_dir)
        service_imports = defaultdict(set)
        
        for file_path in service_files:
            imports = extract_imports(file_path)
            for module in imports:
                normalized_module = normalize_module_name(module, file_path)
                target_service = map_module_to_service(normalized_module)
                if target_service != 'external' and target_service != 'unknown':
                    service_imports[target_service].add(normalized_module)
        
        dependency_graph[service] = dict(service_imports)
    
    return dependency_graph


def find_circular_dependencies(dependency_graph: Dict[str, Dict[str, Set[str]]]) -> List[List[str]]:
    """
    Find circular dependencies in the dependency graph.
    
    Args:
        dependency_graph: Dependency graph
        
    Returns:
        List of circular dependency chains
    """
    circular_deps = []
    
    def dfs(service, path, visited):
        if service in path:
            # Found a cycle
            cycle_start = path.index(service)
            circular_deps.append(path[cycle_start:] + [service])
            return
        
        if service in visited:
            return
        
        visited.add(service)
        path.append(service)
        
        for dep_service in dependency_graph.get(service, {}):
            dfs(dep_service, path.copy(), visited)
    
    for service in dependency_graph:
        dfs(service, [], set())
    
    return circular_deps


def generate_report(dependency_graph: Dict[str, Dict[str, Set[str]]], circular_deps: List[List[str]], output_file: str):
    """
    Generate a dependency analysis report.
    
    Args:
        dependency_graph: Dependency graph
        circular_deps: List of circular dependency chains
        output_file: Output file path
    """
    report = {
        'timestamp': str(datetime.now()),
        'services': list(dependency_graph.keys()),
        'dependencies': {},
        'circular_dependencies': circular_deps,
        'summary': {
            'total_services': len(dependency_graph),
            'total_dependencies': sum(len(deps) for deps in dependency_graph.values()),
            'total_circular_dependencies': len(circular_deps)
        }
    }
    
    # Format dependencies for the report
    for service, deps in dependency_graph.items():
        report['dependencies'][service] = {
            target: list(modules)
            for target, modules in deps.items()
        }
    
    # Write the report to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report generated: {output_file}")
    print(f"Total services: {report['summary']['total_services']}")
    print(f"Total dependencies: {report['summary']['total_dependencies']}")
    print(f"Total circular dependencies: {report['summary']['total_circular_dependencies']}")
    
    if circular_deps:
        print("\nCircular Dependencies:")
        for i, cycle in enumerate(circular_deps, 1):
            print(f"{i}. {' -> '.join(cycle)}")


def main():
    parser = argparse.ArgumentParser(description='Analyze dependencies between services')
    parser.add_argument('--root-dir', default='.', help='Root directory of the codebase')
    parser.add_argument('--output', default='dependency_analysis_report.json', help='Output file path')
    args = parser.parse_args()
    
    print("Building dependency graph...")
    dependency_graph = build_dependency_graph(args.root_dir)
    
    print("Finding circular dependencies...")
    circular_deps = find_circular_dependencies(dependency_graph)
    
    print("Generating report...")
    generate_report(dependency_graph, circular_deps, args.output)


if __name__ == "__main__":
    from datetime import datetime
    main()
