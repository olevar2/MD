"""
Script to update configuration imports in all services.

This script updates imports in all services to use the standardized configuration management.
It replaces legacy configuration imports with standardized configuration imports.
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

# Import patterns to replace
IMPORT_PATTERNS = [
    # Legacy config imports
    (r"from .*\.config\.config import config_manager(?!.*as legacy_config_manager)", r"from \g<0> as legacy_config_manager"),
    (r"from .*\.config import config_manager(?!.*as legacy_config_manager)", r"from \g<0> as legacy_config_manager"),
    
    # Add standardized config imports
    (r"from (.*?)\.config\.config import (.*?)(?:\n|$)", r"from \1.config.config import \2\nfrom \1.config.standardized_config import config_manager, get_config\n"),
    (r"from (.*?)\.config import config_manager as legacy_config_manager(?:\n|$)", r"from \1.config import config_manager as legacy_config_manager\nfrom \1.config.standardized_config import config_manager, get_config\n"),
    
    # Replace direct config_manager usage with standardized version
    (r"config_manager\.get_service_specific_config\(\)", r"config_manager.settings"),
    (r"config_manager\.get_database_config\(\)", r"config_manager.settings"),
    (r"config_manager\.get_logging_config\(\)", r"config_manager.settings"),
    (r"config_manager\.get_service_config\(\)", r"config_manager.settings"),
    (r"config_manager\.get_service_clients_config\(\)", r"config_manager.settings"),
    
    # Replace config.get() with standardized version
    (r"config\.get\(['\"](.*?)['\"](.*?)\)", r"config_manager.get('\1'\2)"),
]

# Files to exclude from processing
EXCLUDE_FILES = [
    "config/config.py",
    "config/standardized_config.py",
    "config/__init__.py",
]


def update_configuration_imports():
    """
    Update configuration imports in all services.
    """
    # Process each service
    for service_dir in SERVICE_DIRS:
        service_name = service_dir.replace("-", "_").replace("_service", "")
        
        # Find Python files
        python_files = find_python_files(service_dir)
        
        # Update imports in each file
        for file_path in python_files:
            # Skip excluded files
            if any(exclude in file_path for exclude in EXCLUDE_FILES):
                continue
            
            update_file(file_path, service_name)


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


def update_file(file_path: str, service_name: str) -> None:
    """
    Update imports in a file.
    
    Args:
        file_path: Path to the file
        service_name: Service name
    """
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if file contains configuration imports
        has_config_imports = "config_manager" in content or "config.get" in content
        
        if not has_config_imports:
            return
        
        # Create backup
        backup_file = f"{file_path}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2(file_path, backup_file)
        
        # Replace imports
        original_content = content
        for pattern, replacement in IMPORT_PATTERNS:
            content = re.sub(pattern, replacement, content)
        
        # Check if content changed
        if content == original_content:
            # Remove backup if no changes
            os.remove(backup_file)
            return
        
        # Write updated content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"Updated configuration imports in: {file_path}")
    except Exception as e:
        print(f"Error updating file {file_path}: {e}")


if __name__ == "__main__":
    update_configuration_imports()
