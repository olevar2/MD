"""
Script to replace custom try/catch blocks with standardized error handling.

This script replaces custom try/catch blocks with standardized error handling
decorators from common-lib.
"""

import os
import re
import json
import ast
import astor
from typing import Dict, List, Set, Tuple, Optional, Any

# Service directories to process
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


class CustomErrorHandlingReplacer:
    """
    Class to replace custom try/catch blocks with standardized error handling.
    """
    
    def __init__(self, root_dir: str = '.'):
        """
        Initialize the replacer.
        
        Args:
            root_dir: Root directory to process
        """
        self.root_dir = root_dir
        self.results = {}
    
    def process_codebase(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process the codebase to replace custom try/catch blocks.
        
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
        Process a file to replace custom try/catch blocks.
        
        Args:
            file_path: Path to the file
            rel_path: Relative path to the file
            service_dir: Service directory
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check if the file already uses standardized error handling
        uses_standard_decorators = any(decorator in content for decorator in STANDARD_DECORATORS)
        
        # If the file already uses standardized error handling, skip it
        if uses_standard_decorators:
            return
        
        # Parse the file with AST
        try:
            tree = ast.parse(content)
            transformer = TryExceptTransformer(rel_path, service_dir)
            new_tree = transformer.visit(tree)
            
            # If the transformer made changes, write the new content to the file
            if transformer.made_changes:
                new_content = astor.to_source(new_tree)
                
                # Add imports if needed
                if transformer.needs_common_lib_imports:
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
        Add imports for standardized error handling.
        
        Args:
            content: File content
            service_dir: Service directory
            
        Returns:
            Updated file content with imports
        """
        # Check if the file already has the imports
        if "from common_lib.errors import" in content:
            return content
        
        # Add imports based on service directory
        if service_dir == "analysis-engine-service":
            import_statement = (
                "from analysis_engine.core.exceptions_bridge import (\n"
                "    with_exception_handling,\n"
                "    async_with_exception_handling,\n"
                "    ForexTradingPlatformError,\n"
                "    ServiceError,\n"
                "    DataError,\n"
                "    ValidationError\n"
                ")\n\n"
            )
        else:
            # Use the service-specific exception bridge if available
            service_name = service_dir.replace('-', '_')
            import_statement = (
                f"from {service_name}.error.exceptions_bridge import (\n"
                "    with_exception_handling,\n"
                "    async_with_exception_handling,\n"
                "    ForexTradingPlatformError,\n"
                "    ServiceError,\n"
                "    DataError,\n"
                "    ValidationError\n"
                ")\n\n"
            )
        
        # Add the import statement after the existing imports
        import_match = re.search(r'((?:from|import).*?\n)(?:\s*\n)+', content, re.DOTALL)
        if import_match:
            # Add after the last import
            pos = import_match.end()
            return content[:pos] + import_statement + content[pos:]
        else:
            # Add at the beginning of the file
            return import_statement + content


class TryExceptTransformer(ast.NodeTransformer):
    """
    AST transformer to replace custom try/catch blocks with standardized error handling.
    """
    
    def __init__(self, file_path: str, service_dir: str):
        """
        Initialize the transformer.
        
        Args:
            file_path: Path to the file
            service_dir: Service directory
        """
        self.file_path = file_path
        self.service_dir = service_dir
        self.made_changes = False
        self.changes_made = []
        self.needs_common_lib_imports = False
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """
        Visit a function definition.
        
        Args:
            node: AST node
            
        Returns:
            Transformed AST node
        """
        # Check if the function has a try/except block
        has_try_except = False
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                has_try_except = True
                break
        
        # If the function has a try/except block, add the decorator
        if has_try_except:
            # Check if the function already has the decorator
            has_decorator = False
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id in STANDARD_DECORATORS:
                    has_decorator = True
                    break
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id in STANDARD_DECORATORS:
                    has_decorator = True
                    break
            
            # If the function doesn't have the decorator, add it
            if not has_decorator:
                # Determine the decorator to use
                if node.name.startswith('async_') or node.name.startswith('await_') or isinstance(node, ast.AsyncFunctionDef):
                    decorator_name = 'async_with_exception_handling'
                else:
                    decorator_name = 'with_exception_handling'
                
                # Add the decorator
                node.decorator_list.append(ast.Name(id=decorator_name, ctx=ast.Load()))
                
                # Mark that we made changes
                self.made_changes = True
                self.needs_common_lib_imports = True
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
        # Check if the function has a try/except block
        has_try_except = False
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                has_try_except = True
                break
        
        # If the function has a try/except block, add the decorator
        if has_try_except:
            # Check if the function already has the decorator
            has_decorator = False
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id in STANDARD_DECORATORS:
                    has_decorator = True
                    break
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id in STANDARD_DECORATORS:
                    has_decorator = True
                    break
            
            # If the function doesn't have the decorator, add it
            if not has_decorator:
                # Add the async decorator
                node.decorator_list.append(ast.Name(id='async_with_exception_handling', ctx=ast.Load()))
                
                # Mark that we made changes
                self.made_changes = True
                self.needs_common_lib_imports = True
                self.changes_made.append({
                    'function': node.name,
                    'decorator': 'async_with_exception_handling'
                })
        
        # Continue visiting child nodes
        return self.generic_visit(node)


def main():
    """Main function."""
    replacer = CustomErrorHandlingReplacer()
    results = replacer.process_codebase()
    
    # Print results
    print("Custom Try/Catch Blocks Replacement")
    print("==================================")
    
    total_files_changed = 0
    total_functions_changed = 0
    
    for service_dir, files in results.items():
        if not files:
            continue
        
        print(f"\n{service_dir}:")
        print(f"  Changed {len(files)} files:")
        
        for file_info in files[:10]:  # Show only the first 10 files
            file_path = file_info['file']
            changes = file_info['changes_made']
            print(f"    {file_path}: {len(changes)} functions changed")
            
            for change in changes[:3]:  # Show only the first 3 changes per file
                print(f"      - Added {change['decorator']} to {change['function']}")
            
            if len(changes) > 3:
                print(f"      - ... and {len(changes) - 3} more functions")
            
            total_functions_changed += len(changes)
        
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more files")
        
        total_files_changed += len(files)
    
    print(f"\nTotal files changed: {total_files_changed}")
    print(f"Total functions changed: {total_functions_changed}")
    
    # Save results to file
    with open('error_handling_changes.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to error_handling_changes.json")


if __name__ == "__main__":
    main()
