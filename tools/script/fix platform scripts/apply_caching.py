#!/usr/bin/env python3
"""
Script to apply caching to all read repository methods in the services.
"""

import os
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Base directory - hardcoded absolute path
BASE_DIR = Path("D:/MD/forex_trading_platform")

# Services to process
SERVICES = [
    "causal-analysis-service",
    "backtesting-service",
    "market-analysis-service",
    "analysis-coordinator-service"
]

# Methods that should be cached
CACHEABLE_METHOD_PATTERNS = [
    r"^get_all$",
    r"^find_by_",
    r"^list_",
    r"^search_",
    r"^query_",
    r"^lookup_",
    r"^retrieve_"
]

# TTL values for different method types (in seconds)
TTL_VALUES = {
    "get_all": 1800,
    "find_by": 1800,
    "list": 1800,
    "default": 1800
}

def find_read_repositories(service_dir):
    """
    Find all read repository files in a service directory.
    """
    read_repos = []
    
    # Debug output
    logger.info(f"Looking for read repositories in {service_dir}")
    
    # Look for read repositories in the repositories/read_repositories directory
    read_repos_dir = service_dir / "repositories" / "read_repositories"
    if read_repos_dir.exists():
        logger.info(f"Directory exists: {read_repos_dir}")
        for file_path in read_repos_dir.glob("**/*.py"):
            if not file_path.name.startswith("__"):  # Skip __init__.py and similar
                read_repos.append(str(file_path))
    else:
        logger.warning(f"Directory does not exist: {read_repos_dir}")
    
    # Also look in the service_name/repositories/read_repositories directory
    service_name = service_dir.name.replace("-", "_")
    alt_read_repos_dir = service_dir / service_name / "repositories" / "read_repositories"
    if alt_read_repos_dir.exists():
        logger.info(f"Directory exists: {alt_read_repos_dir}")
        for file_path in alt_read_repos_dir.glob("**/*.py"):
            if not file_path.name.startswith("__"):  # Skip __init__.py and similar
                read_repos.append(str(file_path))
    else:
        logger.warning(f"Directory does not exist: {alt_read_repos_dir}")
    
    return read_repos

def apply_caching_to_repository(repo_path, service_name):
    """
    Apply caching to a read repository.
    """
    try:
        # Read the file
        with open(repo_path, "r") as f:
            content = f.read()
        
        # Check if this is a read repository
        if "class" in content and "ReadRepository" in content:
            # Extract the module name
            module_name = service_name.replace("-", "_")
            
            # Add caching imports
            if "from common_lib.caching.decorators import cached" not in content:
                # Find the last import statement
                import_match = re.search(r"^(from|import).*$", content, re.MULTILINE)
                if import_match:
                    last_import_pos = content.rindex(import_match.group(0)) + len(import_match.group(0))
                    content = content[:last_import_pos] + "\nfrom common_lib.caching.decorators import cached" + content[last_import_pos:]
            
            # Add cache factory import
            if f"from {module_name}.utils.cache_factory import cache_factory" not in content:
                # Find the last import statement
                import_match = re.search(r"^(from|import).*$", content, re.MULTILINE)
                if import_match:
                    last_import_pos = content.rindex(import_match.group(0)) + len(import_match.group(0))
                    content = content[:last_import_pos] + f"\nfrom {module_name}.utils.cache_factory import cache_factory" + content[last_import_pos:]
            
            # Add cache initialization to __init__ method
            init_match = re.search(r"def __init__\(self.*?\):.*?(?=\n\s*def|$)", content, re.DOTALL)
            if init_match:
                init_content = init_match.group(0)
                if "self.cache = cache_factory.get_cache()" not in init_content:
                    # Find the end of the __init__ method
                    init_end_pos = content.find(init_match.group(0)) + len(init_match.group(0))
                    # Add cache initialization
                    content = content[:init_end_pos] + "\n        self.cache = cache_factory.get_cache()" + content[init_end_pos:]
            
            # Find all cacheable methods
            modified_methods = []
            
            # First, check for get_all method
            get_all_match = re.search(r"(async )?def get_all\(self.*?\):.*?(?=\n\s*(?:async )?def|$)", content, re.DOTALL)
            if get_all_match:
                # Skip if already cached
                prev_lines = content[:get_all_match.start()].split("\n")
                if not any(f"@cached" in line for line in prev_lines[-2:]):
                    method_content = get_all_match.group(0)
                    method_start_pos = content.find(method_content)
                    
                    # Check if the method is properly indented
                    method_line = content[method_start_pos:content.find('\n', method_start_pos)]
                    if method_line.startswith('async') or method_line.startswith('def'):
                        # Method is not properly indented, fix it
                        new_content = content[:method_start_pos] + "    " + method_line + content[method_start_pos + len(method_line):]
                        content = new_content
                        method_start_pos = content.find(method_content)
                    
                    indent = "    "
                    
                    # Add decorator
                    content = content[:method_start_pos] + f"{indent}@cached(cache_factory.get_cache(), 'get_all', ttl=1800)\n" + content[method_start_pos:]
                    
                    modified_methods.append("get_all")
            
            # Then check for other methods
            for pattern in CACHEABLE_METHOD_PATTERNS:
                if pattern == r"^get_all$":  # Skip get_all as we already handled it
                    continue
                
                method_matches = re.finditer(pattern + r"\w*\s*\(self.*?\):.*?(?=\n\s*(?:async )?def|$)", content, re.DOTALL)
                for method_match in method_matches:
                    method_content = method_match.group(0)
                    method_name_match = re.match(r"(?:async\s+)?def\s+(\w+)\s*\(", method_content)
                    if not method_name_match:
                        continue
                    
                    method_name = method_name_match.group(1)
                    
                    # Skip if already cached
                    prev_lines = content[:method_match.start()].split("\n")
                    if any(f"@cached" in line for line in prev_lines[-2:]):
                        continue
                    
                    # Determine TTL
                    ttl = TTL_VALUES.get("default")
                    for key, value in TTL_VALUES.items():
                        if method_name.startswith(key):
                            ttl = value
                            break
                    
                    # Add caching decorator
                    method_start_pos = content.find(method_content)
                    is_async = "async " in method_content[:20]
                    indent = "    "
                    
                    # Generate cache key
                    cache_key = method_name.lower()
                    
                    # Add decorator
                    content = content[:method_start_pos] + f"{indent}@cached(cache_factory.get_cache(), '{cache_key}', ttl={ttl})\n" + content[method_start_pos:]
                    
                    modified_methods.append(method_name)
            
            # Write the updated content back to the file if changes were made
            if modified_methods:
                with open(repo_path, "w") as f:
                    f.write(content)
                
                logger.info(f"Applied caching to {repo_path}, modified methods: {modified_methods}")
                return True, modified_methods
            else:
                logger.info(f"No changes needed for {repo_path}")
                return False, []
        
        return False, []
    
    except Exception as e:
        logger.error(f"Error applying caching to {repo_path}: {e}")
        return False, []

def main():
    """
    Main function to apply caching to all services.
    """
    logger.info("Starting to apply caching to read repository methods")
    logger.info(f"Base directory: {BASE_DIR}")
    
    # Check if the base directory exists
    if not BASE_DIR.exists():
        logger.error(f"Base directory {BASE_DIR} does not exist")
        return
    
    # Check if the services exist
    for service_name in SERVICES:
        service_dir = BASE_DIR / service_name
        if not service_dir.exists():
            logger.warning(f"Service directory {service_dir} does not exist")
        else:
            logger.info(f"Service directory {service_dir} exists")
    
    total_repos = 0
    modified_repos = 0
    total_methods_modified = 0
    
    for service_name in SERVICES:
        service_dir = BASE_DIR / service_name
        
        if not service_dir.exists():
            logger.warning(f"Service directory {service_dir} does not exist")
            continue
        
        logger.info(f"Processing service: {service_name}")
        
        # Find read repositories
        read_repos = find_read_repositories(service_dir)
        
        if not read_repos:
            logger.warning(f"No read repositories found in {service_name}")
            continue
        
        logger.info(f"Found {len(read_repos)} read repositories in {service_name}")
        
        # Apply caching to each repository
        for repo_path in read_repos:
            total_repos += 1
            modified, methods = apply_caching_to_repository(repo_path, service_name)
            if modified:
                modified_repos += 1
                total_methods_modified += len(methods)
    
    logger.info(f"Applied caching to {modified_repos} out of {total_repos} read repositories")
    logger.info(f"Total methods modified: {total_methods_modified}")

if __name__ == "__main__":
    main()