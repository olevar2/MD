"""
Script to apply standardized exception handling to all services.

This script applies the standardized exception handling bridge to all services in the codebase.
It creates an exceptions_bridge.py file in each service based on the template in common-lib.
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

# Template file path
TEMPLATE_FILE_PATH = "common-lib/templates/service_template/error/exceptions_bridge.py"

# Target file paths
TARGET_FILE_PATHS = [
    "{service_dir}/error/exceptions_bridge.py",
    "{service_dir}/{service_name}/error/exceptions_bridge.py"
]

# Service name mapping
SERVICE_NAME_MAPPING = {
    "analysis-engine-service": "analysis-engine",
    "data-pipeline-service": "data-pipeline",
    "feature-store-service": "feature-store",
    "ml-integration-service": "ml-integration",
    "ml-workbench-service": "ml-workbench",
    "monitoring-alerting-service": "monitoring-alerting",
    "portfolio-management-service": "portfolio-management",
    "strategy-execution-engine": "strategy-execution",
    "trading-gateway-service": "trading-gateway",
    "ui-service": "ui"
}

# Service-specific error class mapping
SERVICE_ERROR_CLASS_MAPPING = {
    "analysis-engine-service": "AnalysisEngineError",
    "data-pipeline-service": "DataPipelineError",
    "feature-store-service": "FeatureStoreError",
    "ml-integration-service": "MLIntegrationError",
    "ml-workbench-service": "MLWorkbenchError",
    "monitoring-alerting-service": "MonitoringAlertingError",
    "portfolio-management-service": "PortfolioManagementError",
    "strategy-execution-engine": "StrategyExecutionError",
    "trading-gateway-service": "TradingGatewayError",
    "ui-service": "UIError"
}


def apply_standardized_exception_handling():
    """
    Apply standardized exception handling to all services.
    """
    # Check if template file exists
    if not os.path.exists(TEMPLATE_FILE_PATH):
        print(f"Template file not found: {TEMPLATE_FILE_PATH}")
        return
    
    # Read template file
    with open(TEMPLATE_FILE_PATH, "r", encoding="utf-8") as f:
        template_content = f.read()
    
    # Process each service
    for service_dir in SERVICE_DIRS:
        service_name = service_dir.replace("-", "_").replace("_service", "")
        
        # Find existing exceptions bridge file
        existing_file = None
        for path_template in TARGET_FILE_PATHS:
            path = path_template.format(service_dir=service_dir, service_name=service_name)
            if os.path.exists(path):
                existing_file = path
                break
        
        # Get service-specific values
        service_display_name = SERVICE_NAME_MAPPING.get(service_dir, service_dir)
        service_error_class = SERVICE_ERROR_CLASS_MAPPING.get(service_dir, "ServiceSpecificError")
        
        # Create customized content
        customized_content = template_content
        customized_content = customized_content.replace("SERVICE_NAME = \"service-template\"", f"SERVICE_NAME = \"{service_display_name}\"")
        customized_content = customized_content.replace("class ServiceSpecificError(ServiceError):
    """
    ServiceSpecificError class that inherits from ServiceError.
    
    Attributes:
        Add attributes here
    """
", f"class {service_error_class}(ServiceError):")
        customized_content = customized_content.replace("ServiceSpecificError", service_error_class)
        
        # Replace service-specific validation error class
        validation_error_class = service_error_class.replace("Error", "ValidationError")
        customized_content = customized_content.replace("class ServiceSpecificValidationError(ValidationError):
    """
    ServiceSpecificValidationError class that inherits from ValidationError.
    
    Attributes:
        Add attributes here
    """
", f"class {validation_error_class}(ValidationError):")
        customized_content = customized_content.replace("ServiceSpecificValidationError", validation_error_class)
        
        # Replace service-specific data error class
        data_error_class = service_error_class.replace("Error", "DataError")
        customized_content = customized_content.replace("class ServiceSpecificDataError(DataError):
    """
    ServiceSpecificDataError class that inherits from DataError.
    
    Attributes:
        Add attributes here
    """
", f"class {data_error_class}(DataError):")
        customized_content = customized_content.replace("ServiceSpecificDataError", data_error_class)
        
        # Replace service-specific business error class
        business_error_class = service_error_class.replace("Error", "BusinessError")
        customized_content = customized_content.replace("class ServiceSpecificBusinessError(BusinessError):
    """
    ServiceSpecificBusinessError class that inherits from BusinessError.
    
    Attributes:
        Add attributes here
    """
", f"class {business_error_class}(BusinessError):")
        customized_content = customized_content.replace("ServiceSpecificBusinessError", business_error_class)
        
        if existing_file:
            # Create backup of existing file
            backup_file = f"{existing_file}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            shutil.copy2(existing_file, backup_file)
            print(f"Created backup of existing exceptions bridge file: {backup_file}")
            
            # Replace existing file
            with open(existing_file, "w", encoding="utf-8") as f:
                f.write(customized_content)
            
            print(f"Updated exceptions bridge file: {existing_file}")
        else:
            # Create new file
            new_file = TARGET_FILE_PATHS[0].format(service_dir=service_dir, service_name=service_name)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            
            # Write new file
            with open(new_file, "w", encoding="utf-8") as f:
                f.write(customized_content)
            
            print(f"Created new exceptions bridge file: {new_file}")


if __name__ == "__main__":
    apply_standardized_exception_handling()
