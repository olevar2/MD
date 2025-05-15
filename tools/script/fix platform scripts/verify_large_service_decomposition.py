"""
Verify Large Service Decomposition

This script verifies the implementation of the Large Service Decomposition component.
It checks that all required files and components exist and have the correct structure.
"""
import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Define the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Define the services to check
SERVICES = [
    "causal-analysis-service",
    "backtesting-service",
    "market-analysis-service",
    "analysis-coordinator-service"
]

# Define the required directories for each service
REQUIRED_DIRECTORIES = [
    "api",
    "api/v1",
    "core",
    "models",
    "repositories",
    "services",
    "utils",
    "tests"
]

# Define the required files for each service
REQUIRED_FILES = {
    "causal-analysis-service": [
        "main.py",
        "Dockerfile",
        "requirements.txt",
        "api/v1/causal_routes.py",
        "api/v1/health_routes.py",
        "core/algorithms/causal_discovery.py",
        "core/algorithms/base.py",
        "core/service_dependencies.py",
        "models/causal_models.py",
        "repositories/causal_repository.py",
        "services/causal_service.py",
        "utils/validation.py",
        "utils/correlation_id.py",
        "tests/test_causal_service.py"
    ],
    "backtesting-service": [
        "main.py",
        "Dockerfile",
        "requirements.txt",
        "app/core/engine/backtest_engine.py",
        "app/models/backtest_models.py",
        "app/repositories/backtest_repository.py"
    ],
    "market-analysis-service": [
        # Minimal requirements for now
        "api",
        "core",
        "models",
        "repositories",
        "services",
        "utils",
        "tests"
    ],
    "analysis-coordinator-service": [
        # Minimal requirements for now
        "api",
        "core",
        "models",
        "repositories",
        "services",
        "utils",
        "tests"
    ]
}

def check_directory_exists(directory: str) -> bool:
    """
    Check if a directory exists.
    
    Args:
        directory: Directory path
        
    Returns:
        bool: True if the directory exists, False otherwise
    """
    return os.path.isdir(directory)

def check_file_exists(file_path: str) -> bool:
    """
    Check if a file exists.
    
    Args:
        file_path: File path
        
    Returns:
        bool: True if the file exists, False otherwise
    """
    return os.path.isfile(file_path)

def check_service(service: str) -> Tuple[bool, List[str], List[str]]:
    """
    Check if a service has the required directories and files.
    
    Args:
        service: Service name
        
    Returns:
        Tuple[bool, List[str], List[str]]: (success, missing_directories, missing_files)
    """
    service_dir = os.path.join(ROOT_DIR, service)
    
    # Check if the service directory exists
    if not check_directory_exists(service_dir):
        logger.error(f"Service directory {service_dir} does not exist")
        return False, [service_dir], []
    
    # Check required directories
    missing_directories = []
    for directory in REQUIRED_DIRECTORIES:
        directory_path = os.path.join(service_dir, directory)
        if not check_directory_exists(directory_path):
            missing_directories.append(directory_path)
    
    # Check required files
    missing_files = []
    for file_path in REQUIRED_FILES[service]:
        # If the file path is a directory, check if it exists
        if not os.path.dirname(file_path) and check_directory_exists(os.path.join(service_dir, file_path)):
            continue
        
        # Otherwise, check if the file exists
        full_file_path = os.path.join(service_dir, file_path)
        if not check_file_exists(full_file_path):
            missing_files.append(full_file_path)
    
    # Return success if no missing directories or files
    success = len(missing_directories) == 0 and len(missing_files) == 0
    
    return success, missing_directories, missing_files

def check_all_services() -> Dict[str, Dict[str, Any]]:
    """
    Check all services.
    
    Returns:
        Dict[str, Dict[str, Any]]: Results for each service
    """
    results = {}
    
    for service in SERVICES:
        success, missing_directories, missing_files = check_service(service)
        
        results[service] = {
            "success": success,
            "missing_directories": missing_directories,
            "missing_files": missing_files
        }
    
    return results

def calculate_completion_percentage(results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate the completion percentage for each service.
    
    Args:
        results: Results for each service
        
    Returns:
        Dict[str, float]: Completion percentage for each service
    """
    completion_percentages = {}
    
    for service, result in results.items():
        # Count the number of required files and directories
        required_files_count = len(REQUIRED_FILES[service])
        
        # Count the number of missing files and directories
        missing_files_count = len(result["missing_files"])
        
        # Calculate the completion percentage
        if required_files_count > 0:
            completion_percentage = 100 * (required_files_count - missing_files_count) / required_files_count
        else:
            completion_percentage = 0
        
        completion_percentages[service] = completion_percentage
    
    return completion_percentages

def calculate_overall_completion_percentage(completion_percentages: Dict[str, float]) -> float:
    """
    Calculate the overall completion percentage.
    
    Args:
        completion_percentages: Completion percentage for each service
        
    Returns:
        float: Overall completion percentage
    """
    if not completion_percentages:
        return 0
    
    return sum(completion_percentages.values()) / len(completion_percentages)

def main():
    """
    Main function.
    """
    logger.info("Verifying Large Service Decomposition implementation")
    
    # Check all services
    results = check_all_services()
    
    # Calculate completion percentages
    completion_percentages = calculate_completion_percentage(results)
    
    # Calculate overall completion percentage
    overall_completion_percentage = calculate_overall_completion_percentage(completion_percentages)
    
    # Print results
    logger.info("Verification results:")
    for service, result in results.items():
        logger.info(f"Service: {service}")
        logger.info(f"  Success: {result['success']}")
        logger.info(f"  Completion: {completion_percentages[service]:.2f}%")
        
        if result["missing_directories"]:
            logger.info(f"  Missing directories: {len(result['missing_directories'])}")
            for directory in result["missing_directories"]:
                logger.info(f"    {directory}")
        
        if result["missing_files"]:
            logger.info(f"  Missing files: {len(result['missing_files'])}")
            for file_path in result["missing_files"]:
                logger.info(f"    {file_path}")
    
    logger.info(f"Overall completion: {overall_completion_percentage:.2f}%")
    
    # Save results to file
    output_dir = os.path.join(ROOT_DIR, "tools", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "large_service_decomposition_verification.json")
    
    with open(output_file, "w") as f:
        json.dump({
            "results": results,
            "completion_percentages": completion_percentages,
            "overall_completion_percentage": overall_completion_percentage
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Return success if all services are successful
    return all(result["success"] for result in results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)