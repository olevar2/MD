"""
Script to apply resilience patterns to methods in the codebase.

This script applies resilience patterns to methods in the codebase
that don't already use resilience patterns.
"""

import os
import re
import json
import ast
import astor
from typing import Dict, List, Set, Tuple, Optional, Any

# Service directories to process
SERVICE_DIRS = [
    "trading-gateway-service",
    "analysis-engine-service"
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
    "resilient",
    "with_broker_api_resilience",
    "with_market_data_resilience",
    "with_order_execution_resilience",
    "with_risk_management_resilience",
    "with_database_resilience"
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


class ResiliencePatternApplier:
    """
    Class to apply resilience patterns to methods in the codebase.
    """
    
    def __init__(self, root_dir: str = '.'):
        """
        Initialize the applier.
        
        Args:
            root_dir: Root directory to process
        """
        self.root_dir = root_dir
        self.results = {}
    
    def process_codebase(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process the codebase to apply resilience patterns.
        
        Returns:
            Dictionary mapping service names to lists of processed files
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
                        self._process_file(file_path, rel_path, service_dir)
                    except Exception as e:
                        print(f"Error processing {rel_path}: {str(e)}")
        
        return self.results
    
    def _process_file(self, file_path: str, rel_path: str, service_dir: str) -> None:
        """
        Process a file to apply resilience patterns.
        
        Args:
            file_path: Path to the file
            rel_path: Relative path to the file
            service_dir: Service directory
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check if the file already uses resilience patterns
        uses_resilience_patterns = any(decorator in content for decorator in RESILIENCE_DECORATORS)
        
        # Parse the file with AST
        try:
            tree = ast.parse(content)
            transformer = MethodTransformer(rel_path, service_dir, uses_resilience_patterns)
            new_tree = transformer.visit(tree)
            
            # If the transformer made changes, write the new content to the file
            if transformer.made_changes:
                new_content = astor.to_source(new_tree)
                
                # Add imports if needed
                if transformer.needs_resilience_imports:
                    new_content = self._add_imports(new_content, service_dir)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                self.results[service_dir].append({
                    'file': rel_path,
                    'changes_made': transformer.changes_made
                })
        except SyntaxError:
            # Skip files with syntax errors
            print(f"Syntax error in {rel_path}, skipping")
    
    def _add_imports(self, content: str, service_dir: str) -> str:
        """
        Add imports for resilience patterns.
        
        Args:
            content: File content
            service_dir: Service directory
            
        Returns:
            Updated file content with imports
        """
        # Check if the file already has the imports
        if "from trading_gateway_service.resilience" in content and service_dir == "trading-gateway-service":
            return content
        elif "from analysis_engine.resilience" in content and service_dir == "analysis-engine-service":
            return content
        
        # Add imports based on service directory
        if service_dir == "trading-gateway-service":
            import_statement = (
                "from trading_gateway_service.resilience.utils import (\n"
                "    with_broker_api_resilience,\n"
                "    with_market_data_resilience,\n"
                "    with_order_execution_resilience,\n"
                "    with_risk_management_resilience,\n"
                "    with_database_resilience\n"
                ")\n\n"
            )
        elif service_dir == "analysis-engine-service":
            import_statement = (
                "from analysis_engine.resilience.utils import (\n"
                "    with_resilience,\n"
                "    with_analysis_resilience,\n"
                "    with_database_resilience\n"
                ")\n\n"
            )
        else:
            # Skip other services for now
            return content
        
        # Add the import statement after the existing imports
        import_match = re.search(r'((?:from|import).*?\n)(?:\s*\n)+', content, re.DOTALL)
        if import_match:
            # Add after the last import
            pos = import_match.end()
            return content[:pos] + import_statement + content[pos:]
        else:
            # Add at the beginning of the file
            return import_statement + content


class MethodTransformer(ast.NodeTransformer):
    """
    AST transformer to apply resilience patterns to methods.
    """
    
    def __init__(self, file_path: str, service_dir: str, uses_resilience_patterns: bool):
        """
        Initialize the transformer.
        
        Args:
            file_path: Path to the file
            service_dir: Service directory
            uses_resilience_patterns: Whether the file uses resilience patterns
        """
        self.file_path = file_path
        self.service_dir = service_dir
        self.uses_resilience_patterns = uses_resilience_patterns
        self.made_changes = False
        self.changes_made = []
        self.needs_resilience_imports = False
        self.current_class = None
    
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """
        Visit a class definition.
        
        Args:
            node: AST node
            
        Returns:
            Transformed AST node
        """
        old_class = self.current_class
        self.current_class = node.name
        
        # Continue visiting child nodes
        result = self.generic_visit(node)
        
        self.current_class = old_class
        return result
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """
        Visit a function definition.
        
        Args:
            node: AST node
            
        Returns:
            Transformed AST node
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
                return node
        
        # Check if the function is a critical method
        is_critical_method = any(node.name.startswith(prefix) for prefix in CRITICAL_METHOD_PREFIXES)
        
        # Check if the function is a method (has self parameter)
        is_method = bool(node.args.args) and node.args.args[0].arg == 'self'
        
        # If the function is a critical method and is a method, add resilience decorator
        if is_critical_method and is_method:
            # Determine the appropriate resilience decorator
            decorator_name = self._get_resilience_decorator(node.name)
            
            # Create the decorator
            decorator = ast.Name(id=decorator_name, ctx=ast.Load())
            
            # Add operation name argument
            decorator = ast.Call(
                func=decorator,
                args=[ast.Str(s=node.name)],
                keywords=[]
            )
            
            # Add the decorator to the function
            node.decorator_list.insert(0, decorator)
            
            # Mark that we made changes
            self.made_changes = True
            self.needs_resilience_imports = True
            self.changes_made.append({
                'function': node.name,
                'decorator': decorator_name
            })
        
        # Continue visiting child nodes
        return self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """
        Visit an async function definition.
        
        Args:
            node: AST node
            
        Returns:
            Transformed AST node
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
                return node
        
        # Check if the function is a critical method
        is_critical_method = any(node.name.startswith(prefix) for prefix in CRITICAL_METHOD_PREFIXES)
        
        # Check if the function is a method (has self parameter)
        is_method = bool(node.args.args) and node.args.args[0].arg == 'self'
        
        # If the function is a critical method and is a method, add resilience decorator
        if is_critical_method and is_method:
            # Determine the appropriate resilience decorator
            decorator_name = self._get_resilience_decorator(node.name)
            
            # Create the decorator
            decorator = ast.Name(id=decorator_name, ctx=ast.Load())
            
            # Add operation name argument
            decorator = ast.Call(
                func=decorator,
                args=[ast.Str(s=node.name)],
                keywords=[]
            )
            
            # Add the decorator to the function
            node.decorator_list.insert(0, decorator)
            
            # Mark that we made changes
            self.made_changes = True
            self.needs_resilience_imports = True
            self.changes_made.append({
                'function': node.name,
                'decorator': decorator_name
            })
        
        # Continue visiting child nodes
        return self.generic_visit(node)
    
    def _get_resilience_decorator(self, method_name: str) -> str:
        """
        Get the appropriate resilience decorator for a method.
        
        Args:
            method_name: Name of the method
            
        Returns:
            Name of the resilience decorator to use
        """
        if self.service_dir == "trading-gateway-service":
            # Determine the appropriate decorator based on method name
            if any(term in method_name for term in ["broker", "order", "trade", "position"]):
                return "with_broker_api_resilience"
            elif any(term in method_name for term in ["market", "price", "quote", "tick", "candle", "ohlc"]):
                return "with_market_data_resilience"
            elif any(term in method_name for term in ["execute", "submit", "cancel", "modify"]):
                return "with_order_execution_resilience"
            elif any(term in method_name for term in ["risk", "limit", "exposure", "margin"]):
                return "with_risk_management_resilience"
            elif any(term in method_name for term in ["db", "database", "query", "save", "load", "store", "retrieve"]):
                return "with_database_resilience"
            else:
                return "with_broker_api_resilience"  # Default for trading gateway
        elif self.service_dir == "analysis-engine-service":
            # Determine the appropriate decorator based on method name
            if any(term in method_name for term in ["analyze", "calculate", "compute", "predict", "forecast"]):
                return "with_analysis_resilience"
            elif any(term in method_name for term in ["db", "database", "query", "save", "load", "store", "retrieve"]):
                return "with_database_resilience"
            else:
                return "with_resilience"  # Default for analysis engine
        else:
            return "with_resilience"  # Default for other services


def main():
    """Main function."""
    applier = ResiliencePatternApplier()
    results = applier.process_codebase()
    
    # Print results
    print("Resilience Patterns Application")
    print("===============================")
    
    total_files_changed = 0
    total_methods_changed = 0
    
    for service_dir, files in results.items():
        if not files:
            continue
        
        print(f"\n{service_dir}:")
        print(f"  Changed {len(files)} files:")
        
        for file_info in files[:10]:  # Show only the first 10 files
            file_path = file_info['file']
            changes = file_info['changes_made']
            print(f"    {file_path}: {len(changes)} methods changed")
            
            for change in changes[:3]:  # Show only the first 3 changes per file
                print(f"      - Added {change['decorator']} to {change['function']}")
            
            if len(changes) > 3:
                print(f"      - ... and {len(changes) - 3} more methods")
            
            total_methods_changed += len(changes)
        
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more files")
        
        total_files_changed += len(files)
    
    print(f"\nTotal files changed: {total_files_changed}")
    print(f"Total methods changed: {total_methods_changed}")
    
    # Save results to file
    with open('resilience_patterns_changes.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to resilience_patterns_changes.json")


if __name__ == "__main__":
    main()
