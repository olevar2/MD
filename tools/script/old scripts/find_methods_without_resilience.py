"""
Script to identify methods that don't use resilience patterns.

This script scans the codebase for methods that don't use resilience patterns
such as circuit breaker, retry, timeout, and bulkhead.
"""

import os
import re
import json
import ast
from typing import Dict, List, Set, Tuple, Optional, Any

# Service directories to scan
SERVICE_DIRS = [
    "analysis-engine-service",
    "data-pipeline-service",
    "feature-store-service",
    "ml-integration-service",
    "ml-workbench-service",
    "monitoring-alerting-service",
    "portfolio-management-service",
    "strategy-execution-engine",
    "trading-gateway-service",
    "ui-service"
]

# Directories to skip
SKIP_DIRS = ['.git', '.venv', 'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist']

# Resilience patterns to look for
RESILIENCE_DECORATORS = [
    "circuit_breaker",
    "retry_with_policy",
    "timeout_handler",
    "bulkhead",
    "with_resilience",
    "resilient"
]

# Resilience method calls to look for
RESILIENCE_METHOD_CALLS = [
    "execute",  # Circuit breaker execute
    "with_timeout",
    "with_retry",
    "with_bulkhead",
    "with_circuit_breaker"
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


class ResiliencePatternFinder:
    """
    Class to find methods that don't use resilience patterns.
    """
    
    def __init__(self, root_dir: str = '.'):
        """
        Initialize the finder.
        
        Args:
            root_dir: Root directory to scan
        """
        self.root_dir = root_dir
        self.results = {}
    
    def scan_codebase(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan the codebase for methods that don't use resilience patterns.
        
        Returns:
            Dictionary mapping service names to lists of methods without resilience patterns
        """
        for service_dir in SERVICE_DIRS:
            service_path = os.path.join(self.root_dir, service_dir)
            if not os.path.exists(service_path):
                continue
            
            self.results[service_dir] = []
            
            for dirpath, dirnames, filenames in os.walk(service_path):
                # Skip directories
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                
                for filename in filenames:
                    if not filename.endswith('.py'):
                        continue
                    
                    file_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(file_path, self.root_dir)
                    
                    try:
                        self._analyze_file(file_path, rel_path, service_dir)
                    except Exception as e:
                        print(f"Error analyzing {rel_path}: {str(e)}")
        
        return self.results
    
    def _analyze_file(self, file_path: str, rel_path: str, service_dir: str) -> None:
        """
        Analyze a file for methods that don't use resilience patterns.
        
        Args:
            file_path: Path to the file
            rel_path: Relative path to the file
            service_dir: Service directory
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check if the file uses resilience patterns
        uses_resilience_patterns = any(decorator in content for decorator in RESILIENCE_DECORATORS) or \
                                  any(method_call in content for method_call in RESILIENCE_METHOD_CALLS)
        
        # Parse the file with AST
        try:
            tree = ast.parse(content)
            visitor = MethodVisitor(rel_path, service_dir, uses_resilience_patterns)
            visitor.visit(tree)
            
            # Add methods without resilience patterns to results
            if visitor.methods_without_resilience:
                self.results[service_dir].extend(visitor.methods_without_resilience)
        except SyntaxError:
            # Skip files with syntax errors
            print(f"Syntax error in {rel_path}, skipping")


class MethodVisitor(ast.NodeVisitor):
    """
    AST visitor to find methods that don't use resilience patterns.
    """
    
    def __init__(self, file_path: str, service_dir: str, uses_resilience_patterns: bool):
        """
        Initialize the visitor.
        
        Args:
            file_path: Path to the file
            service_dir: Service directory
            uses_resilience_patterns: Whether the file uses resilience patterns
        """
        self.file_path = file_path
        self.service_dir = service_dir
        self.uses_resilience_patterns = uses_resilience_patterns
        self.methods_without_resilience = []
        self.current_class = None
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Visit a class definition.
        
        Args:
            node: AST node
        """
        old_class = self.current_class
        self.current_class = node.name
        
        # Continue visiting child nodes
        self.generic_visit(node)
        
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visit a function definition.
        
        Args:
            node: AST node
        """
        # Skip if the file already uses resilience patterns
        if self.uses_resilience_patterns:
            # Check if this specific function uses resilience patterns
            has_resilience_decorator = any(
                isinstance(decorator, ast.Name) and decorator.id in RESILIENCE_DECORATORS
                for decorator in node.decorator_list
            ) or any(
                isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id in RESILIENCE_DECORATORS
                for decorator in node.decorator_list
            )
            
            # If the function has a resilience decorator, skip it
            if has_resilience_decorator:
                return
        
        # Check if the function is a critical method
        is_critical_method = any(node.name.startswith(prefix) for prefix in CRITICAL_METHOD_PREFIXES)
        
        # Check if the function is async
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        # Check if the function is a method (has self parameter)
        is_method = bool(node.args.args) and node.args.args[0].arg == 'self'
        
        # If the function is a critical method and doesn't have resilience patterns, add it to the list
        if is_critical_method and (is_async or is_method):
            self.methods_without_resilience.append({
                'file': self.file_path,
                'class': self.current_class,
                'method': node.name,
                'is_async': is_async,
                'line': node.lineno
            })
        
        # Continue visiting child nodes
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """
        Visit an async function definition.
        
        Args:
            node: AST node
        """
        # Skip if the file already uses resilience patterns
        if self.uses_resilience_patterns:
            # Check if this specific function uses resilience patterns
            has_resilience_decorator = any(
                isinstance(decorator, ast.Name) and decorator.id in RESILIENCE_DECORATORS
                for decorator in node.decorator_list
            ) or any(
                isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id in RESILIENCE_DECORATORS
                for decorator in node.decorator_list
            )
            
            # If the function has a resilience decorator, skip it
            if has_resilience_decorator:
                return
        
        # Check if the function is a critical method
        is_critical_method = any(node.name.startswith(prefix) for prefix in CRITICAL_METHOD_PREFIXES)
        
        # Check if the function is a method (has self parameter)
        is_method = bool(node.args.args) and node.args.args[0].arg == 'self'
        
        # If the function is a critical method and doesn't have resilience patterns, add it to the list
        if is_critical_method and is_method:
            self.methods_without_resilience.append({
                'file': self.file_path,
                'class': self.current_class,
                'method': node.name,
                'is_async': True,
                'line': node.lineno
            })
        
        # Continue visiting child nodes
        self.generic_visit(node)


def main():
    """Main function."""
    finder = ResiliencePatternFinder()
    results = finder.scan_codebase()
    
    # Print results
    print("Methods Without Resilience Patterns Analysis")
    print("===========================================")
    
    total_methods = 0
    
    for service_dir, methods in results.items():
        if not methods:
            continue
        
        print(f"\n{service_dir}:")
        print(f"  Found {len(methods)} methods without resilience patterns:")
        
        for method in methods[:10]:  # Show only the first 10 methods
            class_name = method.get('class', 'N/A')
            method_name = method.get('method', 'N/A')
            is_async = method.get('is_async', False)
            line = method.get('line', 'unknown')
            
            print(f"    {method['file']}: Line {line}, {'Async ' if is_async else ''}Method: {class_name}.{method_name}")
        
        if len(methods) > 10:
            print(f"    ... and {len(methods) - 10} more")
        
        total_methods += len(methods)
    
    print(f"\nTotal methods without resilience patterns found: {total_methods}")
    
    # Save results to file
    with open('methods_without_resilience.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to methods_without_resilience.json")


if __name__ == "__main__":
    main()
