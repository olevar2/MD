"""
Apply Caching to Read Repositories

This script applies caching to all read repositories in the forex trading platform.
"""
import os
import sys
import logging
import re
from pathlib import Path
import ast
import astor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define the base directory
BASE_DIR = Path("D:/MD/forex_trading_platform")

class ReadRepositoryTransformer(ast.NodeTransformer):
    """
    AST transformer to add caching to read repository methods.
    """
    
    def __init__(self, service_name):
        self.service_name = service_name
        self.module_name = service_name.replace("-", "_")
        self.has_cache_import = False
        self.has_cache_factory_import = False
        self.has_cache_attribute = False
        self.has_cache_init = False
    
    def visit_Import(self, node):
        """Visit Import nodes to check for cache imports."""
        for name in node.names:
            if name.name == 'common_lib.caching.decorators':
                self.has_cache_import = True
            elif name.name == f'{self.module_name}.utils.cache_factory':
                self.has_cache_factory_import = True
        return node
    
    def visit_ImportFrom(self, node):
        """Visit ImportFrom nodes to check for cache imports."""
        if node.module == 'common_lib.caching.decorators':
            self.has_cache_import = True
        elif node.module == f'{self.module_name}.utils.cache_factory':
            self.has_cache_factory_import = True
        return node
    
    def visit_ClassDef(self, node):
        """Visit ClassDef nodes to add caching to repository classes."""
        # Check if this is a read repository class
        is_read_repo = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == 'ReadRepository':
                is_read_repo = True
            elif isinstance(base, ast.Attribute) and base.attr == 'ReadRepository':
                is_read_repo = True
        
        if is_read_repo:
            # Process the class
            self.generic_visit(node)
            
            # Check if the class already has a cache attribute
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == 'cache':
                            self.has_cache_attribute = True
            
            # Check if __init__ method initializes cache
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    for stmt in item.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self' and target.attr == 'cache':
                                    self.has_cache_init = True
            
            # Add cache initialization to __init__ if needed
            if not self.has_cache_init:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        # Add cache initialization to existing __init__
                        cache_init = ast.parse("self.cache = cache_factory.get_cache()").body[0]
                        item.body.append(cache_init)
                        self.has_cache_init = True
                        break
                
                # If no __init__ method exists, create one
                if not self.has_cache_init:
                    init_method = ast.parse("""
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cache = cache_factory.get_cache()
""").body[0]
                    node.body.insert(0, init_method)
                    self.has_cache_init = True
            
            # Add caching to get methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and (item.name.startswith('get_') or item.name == 'find'):
                    # Check if method already has caching decorator
                    has_cache_decorator = False
                    for decorator in item.decorator_list:
                        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == 'cached':
                            has_cache_decorator = True
                        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == 'cached':
                            has_cache_decorator = True
                    
                    if not has_cache_decorator:
                        # Add caching decorator
                        cache_key = item.name.lower()
                        decorator = ast.parse(f"@cached(cache_factory.get_cache(), '{cache_key}', ttl=3600)").body[0].decorator_list[0]
                        item.decorator_list.insert(0, decorator)
            
            return node
        else:
            return node
    
    def get_imports_to_add(self):
        """Get imports that need to be added to the file."""
        imports_to_add = []
        
        if not self.has_cache_import:
            imports_to_add.append(ast.parse("from common_lib.caching.decorators import cached").body[0])
        
        if not self.has_cache_factory_import:
            imports_to_add.append(ast.parse(f"from {self.module_name}.utils.cache_factory import cache_factory").body[0])
        
        return imports_to_add

def apply_caching_to_repository(repo_path, service_name):
    """
    Apply caching to a read repository.
    
    Args:
        repo_path: Path to the repository file
        service_name: Name of the service
        
    Returns:
        True if caching was applied, False otherwise
    """
    try:
        # Parse the file
        with open(repo_path, "r", encoding="utf-8") as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        # Transform the AST
        transformer = ReadRepositoryTransformer(service_name)
        transformed_tree = transformer.visit(tree)
        
        # Add imports if needed
        imports_to_add = transformer.get_imports_to_add()
        if imports_to_add:
            for import_stmt in reversed(imports_to_add):
                transformed_tree.body.insert(0, import_stmt)
        
        # Generate the modified source code
        modified_source = astor.to_source(transformed_tree)
        
        # Check if the file was actually modified
        if source == modified_source:
            logger.info(f"No changes needed for {repo_path}")
            return False
        
        # Create a backup of the original file
        backup_path = f"{repo_path}.bak"
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(source)
        
        # Write the modified source back to the file
        with open(repo_path, "w", encoding="utf-8") as f:
            f.write(modified_source)
        
        logger.info(f"Applied caching to {repo_path}")
        return True
    except Exception as e:
        logger.error(f"Error applying caching to {repo_path}: {e}")
        return False

def main():
    """Main function to apply caching to read repositories."""
    logger.info("Starting to apply caching to read repositories")
    
    # Check if read_repositories.txt exists
    repos_file = BASE_DIR / "tools" / "script" / "fix platform scripts" / "read_repositories.txt"
    if not repos_file.exists():
        logger.error(f"File {repos_file} does not exist. Run identify_read_repositories.py first.")
        return
    
    # Read the list of repositories
    with open(repos_file, "r") as f:
        repo_paths = [line.strip() for line in f.readlines()]
    
    logger.info(f"Found {len(repo_paths)} read repositories to process")
    
    # Apply caching to each repository
    modified_count = 0
    for repo_path in repo_paths:
        # Determine the service name from the path
        path_parts = repo_path.split(os.sep)
        service_idx = -1
        for i, part in enumerate(path_parts):
            if part in ["causal-analysis-service", "backtesting-service", "market-analysis-service", "analysis-coordinator-service"]:
                service_idx = i
                break
        
        if service_idx == -1:
            logger.warning(f"Could not determine service name for {repo_path}")
            continue
        
        service_name = path_parts[service_idx]
        
        # Apply caching
        if apply_caching_to_repository(repo_path, service_name):
            modified_count += 1
    
    logger.info(f"Applied caching to {modified_count} out of {len(repo_paths)} read repositories")

if __name__ == "__main__":
    main()