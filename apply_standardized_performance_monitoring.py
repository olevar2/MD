"""
Script to apply standardized performance monitoring to all services.

This script applies the standardized performance monitoring to all services in the codebase.
It creates a performance_monitoring.py file in each service based on the template in common-lib.
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
TEMPLATE_FILE_PATH = "common-lib/templates/service_template/monitoring/performance_monitoring.py"

# Target file paths
TARGET_FILE_PATHS = [
    "{service_dir}/monitoring/performance_monitoring.py",
    "{service_dir}/{service_name}/monitoring/performance_monitoring.py"
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

# Service-specific component names
SERVICE_COMPONENT_NAMES = {
    "analysis-engine-service": {
        "analyzer": "Market Analyzers",
        "indicator": "Technical Indicators",
        "pattern": "Pattern Detection",
        "strategy": "Trading Strategies",
        "model": "ML Models",
        "data": "Market Data"
    },
    "data-pipeline-service": {
        "collector": "Data Collectors",
        "transformer": "Data Transformers",
        "validator": "Data Validators",
        "loader": "Data Loaders",
        "scheduler": "Pipeline Schedulers",
        "source": "Data Sources"
    },
    "feature-store-service": {
        "feature": "Feature Generators",
        "dataset": "Dataset Management",
        "transformation": "Feature Transformations",
        "registry": "Feature Registry",
        "versioning": "Feature Versioning",
        "serving": "Feature Serving"
    },
    "ml-integration-service": {
        "model": "Model Management",
        "inference": "Model Inference",
        "training": "Model Training",
        "evaluation": "Model Evaluation",
        "deployment": "Model Deployment",
        "monitoring": "Model Monitoring"
    },
    "ml-workbench-service": {
        "experiment": "Experiment Management",
        "notebook": "Notebook Management",
        "pipeline": "ML Pipelines",
        "artifact": "Artifact Management",
        "visualization": "Data Visualization",
        "collaboration": "Collaboration Tools"
    },
    "monitoring-alerting-service": {
        "collector": "Metric Collectors",
        "analyzer": "Alert Analyzers",
        "notifier": "Alert Notifiers",
        "dashboard": "Dashboard Management",
        "rule": "Alert Rules",
        "integration": "External Integrations"
    },
    "portfolio-management-service": {
        "portfolio": "Portfolio Management",
        "position": "Position Management",
        "risk": "Risk Management",
        "allocation": "Asset Allocation",
        "rebalancing": "Portfolio Rebalancing",
        "reporting": "Performance Reporting"
    },
    "strategy-execution-engine": {
        "executor": "Strategy Executors",
        "scheduler": "Execution Schedulers",
        "optimizer": "Strategy Optimizers",
        "backtest": "Backtest Engines",
        "signal": "Signal Generators",
        "order": "Order Management"
    },
    "trading-gateway-service": {
        "broker": "Broker Integrations",
        "order": "Order Management",
        "execution": "Order Execution",
        "market": "Market Data",
        "account": "Account Management",
        "risk": "Risk Management"
    },
    "ui-service": {
        "page": "Page Components",
        "widget": "UI Widgets",
        "chart": "Chart Components",
        "form": "Form Components",
        "auth": "Authentication Components",
        "layout": "Layout Components"
    }
}


def apply_standardized_performance_monitoring():
    """
    Apply standardized performance monitoring to all services.
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
        
        # Find existing performance monitoring file
        existing_file = None
        for path_template in TARGET_FILE_PATHS:
            path = path_template.format(service_dir=service_dir, service_name=service_name)
            if os.path.exists(path):
                existing_file = path
                break
        
        # Get service-specific values
        service_display_name = SERVICE_NAME_MAPPING.get(service_dir, service_dir)
        service_component_names = SERVICE_COMPONENT_NAMES.get(service_dir, {})
        
        # Create customized content
        customized_content = template_content
        customized_content = customized_content.replace("SERVICE_NAME = \"service-template\"", f"SERVICE_NAME = \"{service_display_name}\"")
        
        # Replace component names
        component_names_str = "COMPONENT_NAMES = {\n"
        for component, name in service_component_names.items():
            component_names_str += f"    \"{component}\": \"{name}\",\n"
        component_names_str += "    \"api\": \"API Endpoints\",\n"
        component_names_str += "    \"service\": \"Service Layer\",\n"
        component_names_str += "    \"repository\": \"Data Access\",\n"
        component_names_str += "    \"client\": \"External Clients\",\n"
        component_names_str += "    \"processor\": \"Data Processing\",\n"
        component_names_str += "    \"validation\": \"Data Validation\"\n"
        component_names_str += "}"
        
        customized_content = re.sub(
            r"COMPONENT_NAMES = \{[^}]+\}",
            component_names_str,
            customized_content
        )
        
        if existing_file:
            # Create backup of existing file
            backup_file = f"{existing_file}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            shutil.copy2(existing_file, backup_file)
            print(f"Created backup of existing performance monitoring file: {backup_file}")
            
            # Replace existing file
            with open(existing_file, "w", encoding="utf-8") as f:
                f.write(customized_content)
            
            print(f"Updated performance monitoring file: {existing_file}")
        else:
            # Create new file
            new_file = TARGET_FILE_PATHS[0].format(service_dir=service_dir, service_name=service_name)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            
            # Write new file
            with open(new_file, "w", encoding="utf-8") as f:
                f.write(customized_content)
            
            print(f"Created new performance monitoring file: {new_file}")


if __name__ == "__main__":
    apply_standardized_performance_monitoring()
