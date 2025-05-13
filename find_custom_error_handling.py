"""
Script to identify custom try/catch blocks that don't use standardized error handling.

This script scans the codebase for try/catch blocks that don't use the standardized
error handling utilities from common-lib.
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

# Standardized error handling patterns
STANDARD_DECORATORS = [
    "with_exception_handling",
    "async_with_exception_handling",
    "with_error_handling",
    "async_with_error_handling"
]

# Common-lib error classes
COMMON_LIB_ERRORS = [
    "ForexTradingPlatformError",
    "ValidationError",
    "DatabaseError",
    "APIError",
    "ServiceError",
    "DataError",
    "BusinessError",
    "SecurityError",
    "TimeoutError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError"
]


class CustomErrorHandlingFinder:
    """
    Class to find custom try/catch blocks that don't use standardized error handling.
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
        Scan the codebase for custom try/catch blocks.
        
        Returns:
            Dictionary mapping service names to lists of custom try/catch blocks
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
        Analyze a file for custom try/catch blocks.
        
        Args:
            file_path: Path to the file
            rel_path: Relative path to the file
            service_dir: Service directory
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check if the file uses standardized error handling
        uses_standard_decorators = any(decorator in content for decorator in STANDARD_DECORATORS)
        
        # Parse the file with AST
        try:
            tree = ast.parse(content)
            visitor = TryExceptVisitor(rel_path, service_dir, uses_standard_decorators)
            visitor.visit(tree)
            
            # Add custom try/catch blocks to results
            if visitor.custom_try_except_blocks:
                self.results[service_dir].extend(visitor.custom_try_except_blocks)
        except SyntaxError:
            # Fall back to regex-based analysis for files with syntax errors
            self._analyze_file_with_regex(content, rel_path, service_dir, uses_standard_decorators)
    
    def _analyze_file_with_regex(
        self,
        content: str,
        rel_path: str,
        service_dir: str,
        uses_standard_decorators: bool
    ) -> None:
        """
        Analyze a file for custom try/catch blocks using regex.
        
        Args:
            content: File content
            rel_path: Relative path to the file
            service_dir: Service directory
            uses_standard_decorators: Whether the file uses standardized error handling decorators
        """
        # Find all try-except blocks
        try_except_pattern = r'try:.*?except\s+([A-Za-z0-9_\.]+(?:Error|Exception))?(?:\s+as\s+([A-Za-z0-9_]+))?:'
        try_except_blocks = re.findall(try_except_pattern, content, re.DOTALL)
        
        for exception_type, exception_var in try_except_blocks:
            exception_type = exception_type.strip() if exception_type else 'Exception'
            
            # Check if the exception type is from common-lib
            is_common_lib_error = any(error in exception_type for error in COMMON_LIB_ERRORS)
            
            # If the file doesn't use standardized decorators and the exception type is not from common-lib,
            # consider it a custom try/catch block
            if not uses_standard_decorators and not is_common_lib_error:
                self.results[service_dir].append({
                    'file': rel_path,
                    'exception_type': exception_type,
                    'exception_var': exception_var.strip() if exception_var else None,
                    'uses_standard_decorators': uses_standard_decorators,
                    'is_common_lib_error': is_common_lib_error
                })


class TryExceptVisitor(ast.NodeVisitor):
    """
    AST visitor to find custom try/catch blocks.
    """
    
    def __init__(self, file_path: str, service_dir: str, uses_standard_decorators: bool):
        """
        Initialize the visitor.
        
        Args:
            file_path: Path to the file
            service_dir: Service directory
            uses_standard_decorators: Whether the file uses standardized error handling decorators
        """
        self.file_path = file_path
        self.service_dir = service_dir
        self.uses_standard_decorators = uses_standard_decorators
        self.custom_try_except_blocks = []
    
    def visit_Try(self, node: ast.Try) -> None:
        """
        Visit a try/except block.
        
        Args:
            node: AST node
        """
        for handler in node.handlers:
            exception_type = None
            exception_var = None
            
            if handler.type:
                if isinstance(handler.type, ast.Name):
                    exception_type = handler.type.id
                elif isinstance(handler.type, ast.Attribute):
                    exception_type = self._get_attribute_name(handler.type)
            
            if handler.name:
                exception_var = handler.name
            
            # Check if the exception type is from common-lib
            is_common_lib_error = exception_type and any(error in exception_type for error in COMMON_LIB_ERRORS)
            
            # If the file doesn't use standardized decorators and the exception type is not from common-lib,
            # consider it a custom try/catch block
            if not self.uses_standard_decorators and not is_common_lib_error:
                self.custom_try_except_blocks.append({
                    'file': self.file_path,
                    'exception_type': exception_type or 'Exception',
                    'exception_var': exception_var,
                    'uses_standard_decorators': self.uses_standard_decorators,
                    'is_common_lib_error': is_common_lib_error,
                    'line': node.lineno
                })
        
        # Continue visiting child nodes
        self.generic_visit(node)
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """
        Get the full name of an attribute.
        
        Args:
            node: AST node
            
        Returns:
            Full attribute name
        """
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        return node.attr


def main():
    """Main function."""
    finder = CustomErrorHandlingFinder()
    results = finder.scan_codebase()
    
    # Print results
    print("Custom Try/Catch Blocks Analysis")
    print("================================")
    
    total_custom_blocks = 0
    
    for service_dir, blocks in results.items():
        if not blocks:
            continue
        
        print(f"\n{service_dir}:")
        print(f"  Found {len(blocks)} custom try/catch blocks:")
        
        for block in blocks[:10]:  # Show only the first 10 blocks
            print(f"    {block['file']}: Line {block.get('line', 'unknown')}, Exception: {block['exception_type']}")
        
        if len(blocks) > 10:
            print(f"    ... and {len(blocks) - 10} more")
        
        total_custom_blocks += len(blocks)
    
    print(f"\nTotal custom try/catch blocks found: {total_custom_blocks}")
    
    # Save results to file
    with open('custom_error_handling.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to custom_error_handling.json")


if __name__ == "__main__":
    main()
