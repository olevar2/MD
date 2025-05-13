"""
Script to migrate legacy configuration to standardized configuration.

This script migrates legacy configuration files to the standardized configuration system.
It creates a new standardized configuration file based on the legacy configuration file.
"""

import os
import re
import shutil
import datetime
from typing import Dict, Any, List, Optional, Set, Tuple

# Service directories to process
SERVICE_DIRS = [
    "trading-gateway-service",
    "analysis-engine-service"
]

# Legacy configuration file paths
LEGACY_CONFIG_FILE_PATHS = [
    "{service_dir}/config/config.py",
    "{service_dir}/{service_name}/config/config.py"
]

# Standardized configuration file paths
STANDARDIZED_CONFIG_FILE_PATHS = [
    "{service_dir}/config/settings.py",
    "{service_dir}/{service_name}/config/settings.py"
]

# Legacy configuration class pattern
LEGACY_CONFIG_CLASS_PATTERN = r"class\s+(\w+)\(ServiceSpecificConfig\):"

# Legacy configuration field pattern
LEGACY_CONFIG_FIELD_PATTERN = r"(\w+):\s*(\w+)\s*=\s*Field\(([^)]+)\)"

# Legacy configuration validator pattern
LEGACY_CONFIG_VALIDATOR_PATTERN = r"@validator\(\"(\w+)\"\)"

# Legacy configuration helper function pattern
LEGACY_CONFIG_HELPER_FUNCTION_PATTERN = r"def\s+get_(\w+)_config\(\)"


def migrate_legacy_config():
    """
    Migrate legacy configuration to standardized configuration.
    """
    for service_dir in SERVICE_DIRS:
        service_name = service_dir.replace("-", "_").replace("_service", "")
        
        # Find legacy configuration file
        legacy_config_file = None
        for path_template in LEGACY_CONFIG_FILE_PATHS:
            path = path_template.format(service_dir=service_dir, service_name=service_name)
            if os.path.exists(path):
                legacy_config_file = path
                break
        
        if not legacy_config_file:
            print(f"No legacy configuration file found for {service_dir}")
            continue
        
        # Check if standardized configuration file already exists
        standardized_config_file = None
        for path_template in STANDARDIZED_CONFIG_FILE_PATHS:
            path = path_template.format(service_dir=service_dir, service_name=service_name)
            if os.path.exists(path):
                standardized_config_file = path
                break
        
        if standardized_config_file:
            print(f"Standardized configuration file already exists for {service_dir}: {standardized_config_file}")
            continue
        
        # Create standardized configuration file
        standardized_config_file = STANDARDIZED_CONFIG_FILE_PATHS[0].format(service_dir=service_dir, service_name=service_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(standardized_config_file), exist_ok=True)
        
        # Read legacy configuration file
        with open(legacy_config_file, "r", encoding="utf-8") as f:
            legacy_config_content = f.read()
        
        # Create backup of legacy configuration file
        backup_file = f"{legacy_config_file}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2(legacy_config_file, backup_file)
        print(f"Created backup of legacy configuration file: {backup_file}")
        
        # Extract legacy configuration class name
        legacy_config_class_match = re.search(LEGACY_CONFIG_CLASS_PATTERN, legacy_config_content)
        if not legacy_config_class_match:
            print(f"No legacy configuration class found in {legacy_config_file}")
            continue
        
        legacy_config_class_name = legacy_config_class_match.group(1)
        
        # Extract legacy configuration fields
        legacy_config_fields = re.findall(LEGACY_CONFIG_FIELD_PATTERN, legacy_config_content)
        
        # Extract legacy configuration validators
        legacy_config_validators = re.findall(LEGACY_CONFIG_VALIDATOR_PATTERN, legacy_config_content)
        
        # Extract legacy configuration helper functions
        legacy_config_helper_functions = re.findall(LEGACY_CONFIG_HELPER_FUNCTION_PATTERN, legacy_config_content)
        
        # Create standardized configuration content
        standardized_config_content = create_standardized_config_content(
            service_dir,
            service_name,
            legacy_config_class_name,
            legacy_config_fields,
            legacy_config_validators,
            legacy_config_helper_functions
        )
        
        # Write standardized configuration file
        with open(standardized_config_file, "w", encoding="utf-8") as f:
            f.write(standardized_config_content)
        
        print(f"Created standardized configuration file: {standardized_config_file}")


def create_standardized_config_content(
    service_dir: str,
    service_name: str,
    legacy_config_class_name: str,
    legacy_config_fields: List[Tuple[str, str, str]],
    legacy_config_validators: List[str],
    legacy_config_helper_functions: List[str]
) -> str:
    """
    Create standardized configuration content.
    
    Args:
        service_dir: Service directory
        service_name: Service name
        legacy_config_class_name: Legacy configuration class name
        legacy_config_fields: Legacy configuration fields
        legacy_config_validators: Legacy configuration validators
        legacy_config_helper_functions: Legacy configuration helper functions
        
    Returns:
        Standardized configuration content
    """
    # Convert service name to title case
    service_title = " ".join(word.capitalize() for word in service_name.split("_"))
    
    # Create standardized configuration class name
    standardized_config_class_name = f"{service_title.replace(' ', '')}Settings"
    
    # Create imports
    imports = [
        "import os",
        "from functools import lru_cache",
        "from typing import Dict, Any, List, Optional, Union",
        "from pydantic import Field, field_validator, SecretStr",
        "from common_lib.config import BaseAppSettings, get_settings, get_config_manager"
    ]
    
    # Create class definition
    class_definition = [
        f"class {standardized_config_class_name}(BaseAppSettings):",
        f'    """',
        f"    {service_title} Service-specific settings.",
        f"    ",
        f"    This class extends the base application settings with service-specific configuration.",
        f'    """',
        f"    ",
        f'    # Override service name',
        f'    SERVICE_NAME: str = Field("{service_dir}", description="Name of the service")',
        f"    "
    ]
    
    # Convert legacy fields to standardized fields
    for field_name, field_type, field_args in legacy_config_fields:
        # Convert field name to uppercase
        standardized_field_name = field_name.upper()
        
        # Convert field type
        standardized_field_type = field_type
        
        # Extract field arguments
        field_args_dict = {}
        for arg in field_args.split(","):
            arg = arg.strip()
            if "=" in arg:
                key, value = arg.split("=", 1)
                field_args_dict[key.strip()] = value.strip()
        
        # Create standardized field
        field_line = f"    {standardized_field_name}: {standardized_field_type} = Field("
        
        # Add field arguments
        field_args_list = []
        if "default" in field_args_dict:
            field_args_list.append(f"default={field_args_dict['default']}")
        
        if "description" in field_args_dict:
            field_args_list.append(f"description={field_args_dict['description']}")
        
        field_line += ", ".join(field_args_list)
        field_line += ")"
        
        class_definition.append(field_line)
    
    # Add validators
    for field_name in legacy_config_validators:
        # Convert field name to uppercase
        standardized_field_name = field_name.upper()
        
        # Create standardized validator
        validator_definition = [
            f"    @field_validator('{standardized_field_name}')",
            f"    @classmethod",
            f"    def validate_{field_name}(cls, v):",
            f'        """',
            f"        Validate {field_name}.",
            f"        ",
            f"        Args:",
            f"            v: Value to validate",
            f"            ",
            f"        Returns:",
            f"            Validated value",
            f"            ",
            f"        Raises:",
            f"            ValueError: If the value is invalid",
            f'        """',
            f"        # Add validation logic here",
            f"        return v",
            f"    "
        ]
        
        class_definition.extend(validator_definition)
    
    # Create helper functions
    helper_functions = [
        f"@lru_cache()",
        f"def get_service_settings() -> {standardized_config_class_name}:
    """
    Get service settings.
    
    Returns:
        {standardized_config_class_name}: Description of return value
    
    """
",
        f'    """',
        f"    Get cached service settings.",
        f"    ",
        f"    Returns:",
        f"        Service settings",
        f'    """',
        f"    return get_settings(",
        f"        settings_class={standardized_config_class_name},",
        f'        env_file=os.environ.get("ENV_FILE", ".env"),',
        f'        config_file=os.environ.get("CONFIG_FILE", "config/config.yaml"),',
        f'        env_prefix=os.environ.get("ENV_PREFIX", "{service_name.upper()}_")',
        f"    )",
        f"",
        f"",
        f"@lru_cache()",
        f"def get_service_config_manager():
    """
    Get service config manager.
    
    """
",
        f'    """',
        f"    Get cached service configuration manager.",
        f"    ",
        f"    Returns:",
        f"        Service configuration manager",
        f'    """',
        f"    return get_config_manager(",
        f"        settings_class={standardized_config_class_name},",
        f'        env_file=os.environ.get("ENV_FILE", ".env"),',
        f'        config_file=os.environ.get("CONFIG_FILE", "config/config.yaml"),',
        f'        env_prefix=os.environ.get("ENV_PREFIX", "{service_name.upper()}_")',
        f"    )",
        f"",
        f"",
        f"# Create a settings instance for easy access",
        f"settings = get_service_settings()"
    ]
    
    # Add helper functions for legacy helper functions
    for helper_function in legacy_config_helper_functions:
        helper_function_definition = [
            f"",
            f"",
            f"def get_{helper_function}_settings() -> Dict[str, Any]:",
            f'    """',
            f"    Get {helper_function}-specific settings.",
            f"    ",
            f"    Returns:",
            f"        {helper_function.capitalize()} settings",
            f'    """',
            f"    # Add implementation here",
            f"    return {{}}"
        ]
        
        helper_functions.extend(helper_function_definition)
    
    # Combine all sections
    content = [
        f'"""',
        f"Standardized Configuration Module for {service_title} Service",
        f"",
        f"This module provides configuration management for the service using the standardized",
        f"configuration management system from common-lib.",
        f'"""',
        f"",
        f"{chr(10).join(imports)}",
        f"",
        f"",
        f"{chr(10).join(class_definition)}",
        f"",
        f"",
        f"{chr(10).join(helper_functions)}"
    ]
    
    return chr(10).join(content)


if __name__ == "__main__":
    migrate_legacy_config()
