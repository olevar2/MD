"""
Apply Standardized Resilience Patterns

This script analyzes the codebase and applies standardized resilience patterns to functions
that need them. It identifies functions that make external calls, database operations, or
other operations that should have resilience patterns applied.
"""

import os
import re
import ast
import argparse
import json
from typing import Dict, List, Set, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("apply-standardized-resilience")

# Directories to skip
SKIP_DIRS = ['.git', '.venv', 'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist']

# Resilience decorators to look for
RESILIENCE_DECORATORS = [
    "circuit_breaker",
    "retry_with_policy",
    "timeout_handler",
    "bulkhead",
    "with_resilience",
    "resilient",
    "with_broker_api_resilience",
    "with_market_data_resilience",
    "with_order_execution_resilience",
    "with_risk_management_resilience",
    "with_database_resilience",
    "with_standard_resilience",
    "with_standard_circuit_breaker",
    "with_standard_retry",
    "with_standard_bulkhead",
    "with_standard_timeout",
    "with_external_api_resilience",
    "with_critical_resilience",
    "with_high_throughput_resilience"
]

# Critical method names that should have resilience patterns
CRITICAL_METHOD_PREFIXES = [
    "get_",
    "fetch_",
    "load_",
    "query_",
    "execute_",
    "process_",
    "analyze_",
    "calculate_",
    "validate_",
    "check_",
    "update_",
    "create_",
    "delete_",
    "send_",
    "receive_"
]

# External API call indicators
EXTERNAL_API_INDICATORS = [
    "requests.",
    "aiohttp.",
    "httpx.",
    "urllib.",
    "http_client",
    "api_client",
    "client.",
    "service_client",
    "broker_client",
    "market_data_client"
]

# Database operation indicators
DATABASE_INDICATORS = [
    "session.",
    "connection.",
    "cursor.",
    "query(",
    "execute(",
    "commit(",
    "rollback(",
    "fetchone(",
    "fetchall(",
    "fetchmany(",
    "sqlalchemy.",
    "db.",
    "database.",
    "repository.",
    "dao."
]

# Market data indicators
MARKET_DATA_INDICATORS = [
    "market_data",
    "price_data",
    "ohlc",
    "candle",
    "tick",
    "quote",
    "historical_data",
    "get_market_data",
    "fetch_market_data",
    "get_price",
    "fetch_price"
]

# Broker API indicators
BROKER_API_INDICATORS = [
    "broker_api",
    "place_order",
    "cancel_order",
    "modify_order",
    "get_positions",
    "get_orders",
    "get_account",
    "execute_trade",
    "broker_client"
]

# Service types based on directory names
SERVICE_TYPE_DIRS = {
    "trading-gateway-service": "broker-api",
    "data-pipeline-service": "market-data",
    "feature-store-service": "high-throughput",
    "analysis-engine-service": "critical",
    "ml-integration-service": "standard",
    "ml-workbench-service": "standard",
    "portfolio-management-service": "critical",
    "risk-management-service": "critical",
    "strategy-execution-engine": "critical",
    "model-registry-service": "standard",
    "monitoring-alerting-service": "standard",
    "api-gateway": "high-throughput",
    "ui-service": "standard",
    "data-management-service": "database"
}


class ResilienceAnalyzer(ast.NodeVisitor):
    """AST visitor that analyzes functions for resilience patterns."""
    
    def __init__(self, filename: str, service_name: str, service_type: str):
        """
        Initialize the analyzer.
        
        Args:
            filename: Name of the file being analyzed
            service_name: Name of the service
            service_type: Type of service
        """
        self.filename = filename
        self.service_name = service_name
        self.service_type = service_type
        self.functions_needing_resilience: List[Dict[str, Any]] = []
        self.functions_with_resilience: List[Dict[str, Any]] = []
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.imports: Set[str] = set()
    
    def visit_Import(self, node: ast.Import) -> None:
        """
        Visit an import node.
        
        Args:
            node: Import node
        """
        for name in node.names:
            self.imports.add(name.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Visit an import from node.
        
        Args:
            node: ImportFrom node
        """
        if node.module:
            module = node.module
            for name in node.names:
                self.imports.add(f"{module}.{name.name}")
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Visit a class definition node.
        
        Args:
            node: ClassDef node
        """
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visit a function definition node.
        
        Args:
            node: FunctionDef node
        """
        self.current_function = node.name
        
        # Check if the function has resilience decorators
        has_resilience = False
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                if decorator.func.id in RESILIENCE_DECORATORS:
                    has_resilience = True
                    break
            elif isinstance(decorator, ast.Name) and decorator.id in RESILIENCE_DECORATORS:
                has_resilience = True
                break
            elif isinstance(decorator, ast.Attribute) and isinstance(decorator.value, ast.Name):
                decorator_name = f"{decorator.value.id}.{decorator.attr}"
                if any(resilience_decorator in decorator_name for resilience_decorator in RESILIENCE_DECORATORS):
                    has_resilience = True
                    break
        
        # Check if the function needs resilience
        needs_resilience = False
        operation_type = None
        
        # Check if the function name indicates it might need resilience
        if any(node.name.startswith(prefix) for prefix in CRITICAL_METHOD_PREFIXES):
            # Analyze the function body to determine if it needs resilience
            function_code = ast.unparse(node)
            
            # Check for external API calls
            if any(indicator in function_code for indicator in EXTERNAL_API_INDICATORS):
                needs_resilience = True
                operation_type = "external-api"
            
            # Check for database operations
            elif any(indicator in function_code for indicator in DATABASE_INDICATORS):
                needs_resilience = True
                operation_type = "database"
            
            # Check for market data operations
            elif any(indicator in function_code for indicator in MARKET_DATA_INDICATORS):
                needs_resilience = True
                operation_type = "market-data"
            
            # Check for broker API operations
            elif any(indicator in function_code for indicator in BROKER_API_INDICATORS):
                needs_resilience = True
                operation_type = "broker-api"
        
        # If no specific operation type was detected but the function needs resilience,
        # use the service type
        if needs_resilience and not operation_type:
            operation_type = self.service_type
        
        # Record the function
        function_info = {
            "filename": self.filename,
            "class_name": self.current_class,
            "function_name": node.name,
            "line_number": node.lineno,
            "end_line_number": node.end_lineno,
            "has_resilience": has_resilience,
            "needs_resilience": needs_resilience,
            "operation_type": operation_type,
            "is_async": isinstance(node, ast.AsyncFunctionDef)
        }
        
        if has_resilience:
            self.functions_with_resilience.append(function_info)
        elif needs_resilience:
            self.functions_needing_resilience.append(function_info)
        
        self.generic_visit(node)
        self.current_function = None
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """
        Visit an async function definition node.
        
        Args:
            node: AsyncFunctionDef node
        """
        # Use the same logic as for regular functions
        self.visit_FunctionDef(node)


def analyze_file(filepath: str, service_name: str, service_type: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Analyze a file for functions that need resilience patterns.
    
    Args:
        filepath: Path to the file
        service_name: Name of the service
        service_type: Type of service
        
    Returns:
        Tuple of (functions needing resilience, functions with resilience)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code)
        analyzer = ResilienceAnalyzer(filepath, service_name, service_type)
        analyzer.visit(tree)
        
        return analyzer.functions_needing_resilience, analyzer.functions_with_resilience
    except Exception as e:
        logger.error(f"Error analyzing file {filepath}: {str(e)}")
        return [], []


def analyze_directory(directory: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Analyze a directory for functions that need resilience patterns.
    
    Args:
        directory: Path to the directory
        
    Returns:
        Tuple of (functions needing resilience, functions with resilience)
    """
    functions_needing_resilience = []
    functions_with_resilience = []
    
    # Determine service name and type
    service_name = os.path.basename(directory)
    service_type = "standard"  # Default
    
    for service_dir, service_type_value in SERVICE_TYPE_DIRS.items():
        if service_dir in directory:
            service_type = service_type_value
            break
    
    for root, dirs, files in os.walk(directory):
        # Skip directories in SKIP_DIRS
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                needing, with_resilience = analyze_file(filepath, service_name, service_type)
                functions_needing_resilience.extend(needing)
                functions_with_resilience.extend(with_resilience)
    
    return functions_needing_resilience, functions_with_resilience


def generate_resilience_decorator(function_info: Dict[str, Any]) -> str:
    """
    Generate a resilience decorator for a function.
    
    Args:
        function_info: Information about the function
        
    Returns:
        Resilience decorator code
    """
    operation_type = function_info["operation_type"]
    is_async = function_info["is_async"]
    
    if not is_async:
        logger.warning(f"Function {function_info['function_name']} in {function_info['filename']} is not async, skipping")
        return ""
    
    # Determine the appropriate decorator based on operation type
    if operation_type == "database":
        decorator = f"@with_database_resilience(\n    service_name=\"{function_info['class_name'] or 'service'}\",\n    operation_name=\"{function_info['function_name']}\"\n)"
    elif operation_type == "broker-api":
        decorator = f"@with_broker_api_resilience(\n    service_name=\"{function_info['class_name'] or 'service'}\",\n    operation_name=\"{function_info['function_name']}\"\n)"
    elif operation_type == "market-data":
        decorator = f"@with_market_data_resilience(\n    service_name=\"{function_info['class_name'] or 'service'}\",\n    operation_name=\"{function_info['function_name']}\"\n)"
    elif operation_type == "external-api":
        decorator = f"@with_external_api_resilience(\n    service_name=\"{function_info['class_name'] or 'service'}\",\n    operation_name=\"{function_info['function_name']}\"\n)"
    elif operation_type == "critical":
        decorator = f"@with_critical_resilience(\n    service_name=\"{function_info['class_name'] or 'service'}\",\n    operation_name=\"{function_info['function_name']}\"\n)"
    elif operation_type == "high-throughput":
        decorator = f"@with_high_throughput_resilience(\n    service_name=\"{function_info['class_name'] or 'service'}\",\n    operation_name=\"{function_info['function_name']}\"\n)"
    else:
        decorator = f"@with_standard_resilience(\n    service_name=\"{function_info['class_name'] or 'service'}\",\n    operation_name=\"{function_info['function_name']}\",\n    service_type=\"{operation_type}\"\n)"
    
    return decorator


def apply_resilience_to_file(filepath: str, functions_needing_resilience: List[Dict[str, Any]]) -> bool:
    """
    Apply resilience patterns to a file.
    
    Args:
        filepath: Path to the file
        functions_needing_resilience: List of functions needing resilience
        
    Returns:
        True if the file was modified, False otherwise
    """
    if not functions_needing_resilience:
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified = False
        
        # Sort functions by line number in descending order to avoid line number changes
        functions_needing_resilience.sort(key=lambda x: x["line_number"], reverse=True)
        
        for function_info in functions_needing_resilience:
            if function_info["filename"] != filepath:
                continue
            
            # Generate the resilience decorator
            decorator = generate_resilience_decorator(function_info)
            if not decorator:
                continue
            
            # Add the decorator before the function definition
            line_number = function_info["line_number"] - 1
            indent = re.match(r'^(\s*)', lines[line_number]).group(1)
            lines.insert(line_number, f"{indent}{decorator}\n")
            
            modified = True
        
        if modified:
            # Check if we need to add the import
            import_line = "from common_lib.resilience import (with_standard_resilience, with_database_resilience, with_broker_api_resilience, with_market_data_resilience, with_external_api_resilience, with_critical_resilience, with_high_throughput_resilience)\n"
            
            # Find the last import line
            last_import_line = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    last_import_line = i
            
            # Add the import after the last import line
            lines.insert(last_import_line + 1, import_line)
            
            # Write the modified file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            logger.info(f"Applied resilience patterns to {filepath}")
            return True
        
        return False
    except Exception as e:
        logger.error(f"Error applying resilience to file {filepath}: {str(e)}")
        return False


def apply_resilience_to_directory(directory: str, functions_needing_resilience: List[Dict[str, Any]]) -> int:
    """
    Apply resilience patterns to a directory.
    
    Args:
        directory: Path to the directory
        functions_needing_resilience: List of functions needing resilience
        
    Returns:
        Number of files modified
    """
    modified_files = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip directories in SKIP_DIRS
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if apply_resilience_to_file(filepath, functions_needing_resilience):
                    modified_files += 1
    
    return modified_files


def generate_report(functions_needing_resilience: List[Dict[str, Any]], functions_with_resilience: List[Dict[str, Any]], output_file: str) -> None:
    """
    Generate a report of functions needing resilience patterns.
    
    Args:
        functions_needing_resilience: List of functions needing resilience
        functions_with_resilience: List of functions with resilience
        output_file: Path to the output file
    """
    report = {
        "summary": {
            "total_functions_analyzed": len(functions_needing_resilience) + len(functions_with_resilience),
            "functions_with_resilience": len(functions_with_resilience),
            "functions_needing_resilience": len(functions_needing_resilience),
            "resilience_coverage_percentage": round(len(functions_with_resilience) / (len(functions_needing_resilience) + len(functions_with_resilience)) * 100, 2) if len(functions_needing_resilience) + len(functions_with_resilience) > 0 else 0
        },
        "functions_needing_resilience": functions_needing_resilience,
        "functions_with_resilience": functions_with_resilience
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Generated report at {output_file}")
    logger.info(f"Summary: {report['summary']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Apply standardized resilience patterns to functions that need them.')
    parser.add_argument('--directory', '-d', type=str, required=True, help='Directory to analyze')
    parser.add_argument('--output', '-o', type=str, default='resilience_report.json', help='Output file for the report')
    parser.add_argument('--apply', '-a', action='store_true', help='Apply resilience patterns to functions that need them')
    args = parser.parse_args()
    
    logger.info(f"Analyzing directory: {args.directory}")
    functions_needing_resilience, functions_with_resilience = analyze_directory(args.directory)
    
    logger.info(f"Found {len(functions_needing_resilience)} functions needing resilience patterns")
    logger.info(f"Found {len(functions_with_resilience)} functions with resilience patterns")
    
    generate_report(functions_needing_resilience, functions_with_resilience, args.output)
    
    if args.apply:
        logger.info("Applying resilience patterns to functions that need them")
        modified_files = apply_resilience_to_directory(args.directory, functions_needing_resilience)
        logger.info(f"Modified {modified_files} files")


if __name__ == "__main__":
    main()