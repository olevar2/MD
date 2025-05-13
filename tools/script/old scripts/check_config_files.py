"""
Script to check configuration files in each service.

This script checks the configuration files in each service to determine
if they use the legacy or standardized configuration system.
"""

import os
import json
from typing import Dict, List, Any

# Service directories to check
SERVICE_DIRS = [
    "analysis-engine-service",
    "data-pipeline-service",
    "trading-gateway-service",
    "ml-integration-service"
]

# Configuration file paths to check
CONFIG_FILE_PATHS = [
    "config/config.py",
    "config/settings.py",
    "{service_name}/config/config.py",
    "{service_name}/config/settings.py"
]

# Legacy configuration patterns
LEGACY_CONFIG_PATTERNS = [
    "from common_lib.config import Config",
    "from common_lib.config import ConfigManager",
    "from common_lib.config import ConfigLoader",
    "from common_lib.config import ServiceSpecificConfig",
    "class ServiceConfig(ServiceSpecificConfig)",
    "config_manager = ConfigManager(",
    "get_service_specific_config()",
    "get_database_config()",
    "get_service_config()",
    "get_logging_config()"
]

# Standardized configuration patterns
STANDARDIZED_CONFIG_PATTERNS = [
    "from common_lib.config import BaseAppSettings",
    "from common_lib.config import StandardizedConfigManager",
    "from common_lib.config import get_settings",
    "from common_lib.config import get_config_manager",
    "class ServiceSettings(BaseAppSettings)",
    "get_settings(",
    "get_config_manager("
]


def check_config_files() -> Dict[str, Dict[str, Any]]:
    """
    Check configuration files in each service.
    
    Returns:
        Dictionary mapping service names to configuration information
    """
    results = {}
    
    for service_dir in SERVICE_DIRS:
        service_name = service_dir.replace("-", "_").replace("_service", "")
        results[service_dir] = {
            "config_files": [],
            "legacy_config": False,
            "standardized_config": False
        }
        
        for config_file_path in CONFIG_FILE_PATHS:
            # Replace {service_name} with the actual service name
            config_file_path = config_file_path.format(service_name=service_name)
            
            # Check if the file exists
            file_path = os.path.join(service_dir, config_file_path)
            if not os.path.exists(file_path):
                continue
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check for legacy configuration patterns
            legacy_matches = []
            for pattern in LEGACY_CONFIG_PATTERNS:
                if pattern in content:
                    legacy_matches.append(pattern)
            
            # Check for standardized configuration patterns
            standardized_matches = []
            for pattern in STANDARDIZED_CONFIG_PATTERNS:
                if pattern in content:
                    standardized_matches.append(pattern)
            
            # Determine the configuration type
            config_type = "Unknown"
            if legacy_matches and standardized_matches:
                config_type = "Mixed"
                results[service_dir]["legacy_config"] = True
                results[service_dir]["standardized_config"] = True
            elif legacy_matches:
                config_type = "Legacy"
                results[service_dir]["legacy_config"] = True
            elif standardized_matches:
                config_type = "Standardized"
                results[service_dir]["standardized_config"] = True
            
            # Add the file to the results
            results[service_dir]["config_files"].append({
                "file": file_path,
                "type": config_type,
                "legacy_matches": legacy_matches,
                "standardized_matches": standardized_matches
            })
    
    return results


def main():
    """Main function."""
    results = check_config_files()
    
    # Print results
    print("Configuration Files Analysis")
    print("===========================")
    
    for service_dir, info in results.items():
        print(f"\n{service_dir}:")
        
        if not info["config_files"]:
            print("  No configuration files found")
            continue
        
        config_type = "Mixed" if info["legacy_config"] and info["standardized_config"] else "Legacy" if info["legacy_config"] else "Standardized" if info["standardized_config"] else "Unknown"
        print(f"  Configuration type: {config_type}")
        
        print("  Configuration files:")
        for file_info in info["config_files"]:
            print(f"    {file_info['file']}: {file_info['type']}")
            
            if file_info["legacy_matches"]:
                print(f"      Legacy matches: {', '.join(file_info['legacy_matches'])}")
            
            if file_info["standardized_matches"]:
                print(f"      Standardized matches: {', '.join(file_info['standardized_matches'])}")
    
    # Save results to file
    with open('config_files_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to config_files_analysis.json")


if __name__ == "__main__":
    main()
