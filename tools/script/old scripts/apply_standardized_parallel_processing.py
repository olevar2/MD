"""
Script to apply standardized parallel processing to all services.

This script applies the standardized parallel processing to all services in the codebase.
It updates imports and replaces service-specific parallel processing implementations with the standardized ones.
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
    "utils/optimized_parallel_processor.py",
    "parallel/parallel_processing_framework.py",
    "parallel/multi_instrument_processor.py",
    "parallel/multi_timeframe_processor.py",
    "parallel/batch_feature_processor.py"
]

# Import patterns to replace
IMPORT_PATTERNS = [
    (r"from .*\.utils\.optimized_parallel_processor import .*", "from common_lib.parallel import ParallelProcessor, get_parallel_processor"),
    (r"from .*\.parallel\.parallel_processing_framework import .*", "from common_lib.parallel import ParallelProcessor, ParallelizationMethod, ResourceManager, TaskDefinition, TaskPriority, TaskResult, get_parallel_processor"),
    (r"from .*\.parallel\.multi_instrument_processor import .*", "from common_lib.parallel import MultiInstrumentProcessor, get_multi_instrument_processor"),
    (r"from .*\.parallel\.multi_timeframe_processor import .*", "from common_lib.parallel import TimeframeHierarchy"),
    (r"from .*\.parallel\.batch_feature_processor import .*", "from common_lib.parallel import FeatureSpec")
]

# Class instantiation patterns to replace
CLASS_PATTERNS = [
    (r"processor = OptimizedParallelProcessor\(.*\)", "processor = get_parallel_processor()"),
    (r"parallel_processor = ParallelProcessor\(.*\)", "parallel_processor = get_parallel_processor()"),
    (r"multi_instrument_processor = MultiInstrumentProcessor\(.*\)", "multi_instrument_processor = get_multi_instrument_processor()")
]


def apply_standardized_parallel_processing():
    """
    Apply standardized parallel processing to all services.
    """
    # Process each service
    for service_dir in SERVICE_DIRS:
        service_name = service_dir.replace("-", "_").replace("_service", "")
        
        # Find Python files
        python_files = find_python_files(service_dir)
        
        # Update imports and class instantiations
        for file_path in python_files:
            update_file(file_path)
        
        # Check for service-specific parallel processing implementations
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
        
        # Check if file contains parallel processing-related imports or instantiations
        has_parallel_imports = any(re.search(pattern, content) for pattern, _ in IMPORT_PATTERNS)
        has_parallel_instantiations = any(re.search(pattern, content) for pattern, _ in CLASS_PATTERNS)
        
        if not has_parallel_imports and not has_parallel_instantiations:
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
        
        print(f"Updated parallel processing imports and instantiations in: {file_path}")
    except Exception as e:
        print(f"Error updating file {file_path}: {e}")


def backup_and_replace_file(file_path: str) -> None:
    """
    Backup and replace a service-specific parallel processing implementation file.
    
    Args:
        file_path: Path to the file
    """
    try:
        # Create backup
        backup_file = f"{file_path}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2(file_path, backup_file)
        
        # Create replacement content
        replacement_content = f"""\"\"\"
This file has been replaced by the standardized parallel processing implementation.

The original implementation has been backed up to:
{backup_file}

Please use the standardized parallel processing from common-lib instead:

from common_lib.parallel import (
    ParallelProcessor,
    ParallelizationMethod,
    ResourceManager,
    TaskDefinition,
    TaskPriority,
    TaskResult,
    get_parallel_processor,
    TimeframeHierarchy,
    FeatureSpec,
    MultiInstrumentProcessor,
    get_multi_instrument_processor
)
\"\"\"

from common_lib.parallel import (
    ParallelProcessor,
    ParallelizationMethod,
    ResourceManager,
    TaskDefinition,
    TaskPriority,
    TaskResult,
    get_parallel_processor,
    TimeframeHierarchy,
    FeatureSpec,
    MultiInstrumentProcessor,
    get_multi_instrument_processor
)

# For backward compatibility
OptimizedParallelProcessor = ParallelProcessor
parallel_processor = get_parallel_processor()
multi_instrument_processor = get_multi_instrument_processor()
"""
        
        # Write replacement content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(replacement_content)
        
        print(f"Replaced service-specific parallel processing implementation: {file_path}")
    except Exception as e:
        print(f"Error replacing file {file_path}: {e}")


if __name__ == "__main__":
    apply_standardized_parallel_processing()
