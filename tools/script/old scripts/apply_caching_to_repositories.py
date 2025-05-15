
"""
Apply Caching to Repositories

This script applies caching to all read repositories in the forex trading platform.
"""
import os
import re
import sys
import logging
from pathlib import Path

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

# Define the services to apply caching to
SERVICES = [
    "causal-analysis-service",
    "backtesting-service",
    "market-analysis-service",
    "analysis-coordinator-service"
]

def find_read_repositories(service_dir):
    """
    Find all read repositories in a service directory.
    
    Args:
        service_dir: The service directory to search in
        
    Returns:
        A list of read repository file paths
    """
    read_repos = []
    
    # Walk through the service directory
    for root, dirs, files in os.walk(service_dir):
        # Skip __pycache__ directories
        if "__pycache__" in root:
            continue
            
        # Check each Python file
        for file in files:
            if not file.endswith(".py"):
                continue
                
            # Check if the file is a read repository
            if "read_repository" in file.lower() or "readrepository" in file.lower():
                read_repos.append(os.path.join(root, file))
    
    return read_repos

def apply_caching_to_repository(repo_path):
    """
    Apply caching to a read repository.
    
    Args:
        repo_path: The path to the read repository file
        
    Returns:
        True if caching was applied, False otherwise
    """
    logger.info(f"Applying caching to {repo_path}")
    
    # Read the repository file
    with open(repo_path, "r") as f:
        content = f.read()
    
    # Check if caching is already applied
    if "@cached" in content:
        logger.info(f"Caching already applied to {repo_path}")
        return False
    
    # Add caching imports
    if "from common_lib.caching.decorators import cached" not in content:
        # Find the last import statement
        import_match = re.search(r"^(from|import).*$", content, re.MULTILINE)
        if import_match:
            last_import_pos = content.rindex(import_match.group(0)) + len(import_match.group(0))
            content = content[:last_import_pos] + "
from common_lib.caching.decorators import cached" + content[last_import_pos:]
    
    # Extract service name from repo path
    repo_path_parts = Path(repo_path).parts
    for part in repo_path_parts:
        if part in SERVICES:
            service_name = part
            break
    else:
        logger.warning(f"Could not determine service name for {repo_path}")
        return False
    
    # Add cache factory import
    module_name = service_name.replace("-", "_")
    
    if f"from {module_name}.utils.cache_factory import cache_factory" not in content:
        # Find the last import statement
        import_match = re.search(r"^(from|import).*$", content, re.MULTILINE)
        if import_match:
            last_import_pos = content.rindex(import_match.group(0)) + len(import_match.group(0))
            content = content[:last_import_pos] + f"
from {module_name}.utils.cache_factory import cache_factory" + content[last_import_pos:]
    
    # Add cache initialization to __init__ method
    init_match = re.search(r"def __init__\(self.*?\):.*?(?=
\s*def|$)", content, re.DOTALL)
    if init_match:
        init_content = init_match.group(0)
        if "self.cache = cache_factory.get_cache()" not in init_content:
            # Find the end of the __init__ method
            init_end_pos = content.find(init_match.group(0)) + len(init_match.group(0))
            # Add cache initialization
            content = content[:init_end_pos] + "
        self.cache = cache_factory.get_cache()" + content[init_end_pos:]
    
    # Add caching to get_by_id method
    get_by_id_match = re.search(r"(async )?def get_by_id\(self.*?\):.*?(?=
\s*(async )?def|$)", content, re.DOTALL)
    if get_by_id_match:
        get_by_id_content = get_by_id_match.group(0)
        if "@cached" not in get_by_id_content:
            # Find the start of the get_by_id method
            get_by_id_start_pos = content.find(get_by_id_match.group(0))
            # Extract the entity name from the repository class name
            class_name_match = re.search(r"class (\w+)ReadRepository", content)
            if class_name_match:
                entity_name = class_name_match.group(1).lower()
                # Add cached decorator
                is_async = get_by_id_match.group(1) is not None
                indent = "    "
                if is_async:
                    content = content[:get_by_id_start_pos] + f"{indent}@cached(cache_factory.get_cache(), "{entity_name}", ttl=3600)
" + content[get_by_id_start_pos:]
                else:
                    content = content[:get_by_id_start_pos] + f"{indent}@cached(cache_factory.get_cache(), "{entity_name}", ttl=3600)
" + content[get_by_id_start_pos:]
    
    # Write the updated content back to the file
    with open(repo_path, "w") as f:
        f.write(content)
    
    logger.info(f"Caching applied to {repo_path}")
    return True

def main():
    """
    Main function to apply caching to all services.
    """
    logger.info("Starting to apply caching to repositories")
    
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
        for repo_path in read_repos:
            logger.info(f"  - {repo_path}")
            apply_caching_to_repository(repo_path)
    
    logger.info("Finished applying caching to repositories")

if __name__ == "__main__":
    main()
