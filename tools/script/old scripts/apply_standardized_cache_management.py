"""
Script to apply standardized cache management to all services.

This script applies the standardized cache management to all services in the codebase.
It updates imports and replaces service-specific cache implementations with the standardized ones.
"""

import os
import re
import shutil
import datetime
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

# Files to update
FILES_TO_UPDATE = [
    "utils/cache_manager.py",
    "utils/adaptive_cache_manager.py",
    "utils/predictive_cache_manager.py",
    "caching/cache_manager.py",
    "caching/enhanced_cache_manager.py"
]

# Import patterns to replace
IMPORT_PATTERNS = [
    (r"from .*\.utils\.cache_manager import .*", "from common_lib.caching import AdaptiveCacheManager, cached, get_cache_manager"),
    (r"from .*\.utils\.adaptive_cache_manager import .*", "from common_lib.caching import AdaptiveCacheManager, cached, get_cache_manager"),
    (r"from .*\.utils\.predictive_cache_manager import .*", "from common_lib.caching import PredictiveCacheManager, get_predictive_cache_manager"),
    (r"from .*\.caching\.cache_manager import .*", "from common_lib.caching import AdaptiveCacheManager, cached, get_cache_manager"),
    (r"from .*\.caching\.enhanced_cache_manager import .*", "from common_lib.caching import AdaptiveCacheManager, cached, get_cache_manager")
]

# Class instantiation patterns to replace
CLASS_PATTERNS = [
    (r"cache = CacheManager\(\)", "cache = get_cache_manager()"),
    (r"cache_manager = CacheManager\(\)", "cache_manager = get_cache_manager()"),
    (r"cache = AdaptiveCacheManager\(.*\)", "cache = get_cache_manager()"),
    (r"cache_manager = AdaptiveCacheManager\(.*\)", "cache_manager = get_cache_manager()"),
    (r"cache = PredictiveCacheManager\(.*\)", "cache = get_predictive_cache_manager()"),
    (r"cache_manager = PredictiveCacheManager\(.*\)", "cache_manager = get_predictive_cache_manager()"),
    (r"cache = EnhancedCacheManager\(.*\)", "cache = get_cache_manager()"),
    (r"cache_manager = EnhancedCacheManager\(.*\)", "cache_manager = get_cache_manager()")
]


def apply_standardized_cache_management():
    """
    Apply standardized cache management to all services.
    """
    # Process each service
    for service_dir in SERVICE_DIRS:
        service_name = service_dir.replace("-", "_").replace("_service", "")
        
        # Find Python files
        python_files = find_python_files(service_dir)
        
        # Update imports and class instantiations
        for file_path in python_files:
            update_file(file_path)
        
        # Check for service-specific cache implementations
        for file_pattern in FILES_TO_UPDATE:
            file_path = os.path.join(service_dir, file_pattern)
            if os.path.exists(file_path):
                backup_and_replace_file(file_path)
            
            # Check in service-specific directory
            service_file_path = os.path.join(service_dir, service_name, file_pattern)
            if os.path.exists(service_file_path):
                backup_and_replace_file(service_file_path)


def find_python_files(directory: str) -> List[str]:
    """
    Find all Python files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of Python file paths
    """
    python_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    return python_files


def update_file(file_path: str) -> None:
    """
    Update imports and class instantiations in a file.
    
    Args:
        file_path: Path to the file
    """
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if file contains cache-related imports or instantiations
        has_cache_imports = any(re.search(pattern, content) for pattern, _ in IMPORT_PATTERNS)
        has_cache_instantiations = any(re.search(pattern, content) for pattern, _ in CLASS_PATTERNS)
        
        if not has_cache_imports and not has_cache_instantiations:
            return
        
        # Create backup
        backup_file = f"{file_path}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2(file_path, backup_file)
        
        # Replace imports
        for pattern, replacement in IMPORT_PATTERNS:
            content = re.sub(pattern, replacement, content)
        
        # Replace class instantiations
        for pattern, replacement in CLASS_PATTERNS:
            content = re.sub(pattern, replacement, content)
        
        # Write updated content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"Updated cache imports and instantiations in: {file_path}")
    except Exception as e:
        print(f"Error updating file {file_path}: {e}")


def backup_and_replace_file(file_path: str) -> None:
    """
    Backup and replace a service-specific cache implementation file.
    
    Args:
        file_path: Path to the file
    """
    try:
        # Create backup
        backup_file = f"{file_path}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2(file_path, backup_file)
        
        # Create replacement content
        replacement_content = f"""\"\"\"
This file has been replaced by the standardized cache management implementation.

The original implementation has been backed up to:
{backup_file}

Please use the standardized cache management from common-lib instead:

from common_lib.caching import (
    AdaptiveCacheManager,
    PredictiveCacheManager,
    cached,
    get_cache_manager,
    get_predictive_cache_manager
)
\"\"\"

from common_lib.caching import (
    AdaptiveCacheManager,
    PredictiveCacheManager,
    cached,
    get_cache_manager,
    get_predictive_cache_manager
)

# For backward compatibility
CacheManager = AdaptiveCacheManager
EnhancedCacheManager = AdaptiveCacheManager
cache = get_cache_manager()
cache_manager = get_cache_manager()
"""
        
        # Write replacement content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(replacement_content)
        
        print(f"Replaced service-specific cache implementation: {file_path}")
    except Exception as e:
        print(f"Error replacing file {file_path}: {e}")


if __name__ == "__main__":
    apply_standardized_cache_management()
