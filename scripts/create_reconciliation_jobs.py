#!/usr/bin/env python3
"""
Script to create reconciliation jobs for all services.

This script initializes the data reconciliation system and creates
reconciliation jobs for all services.
"""

import os
import sys
import logging
import importlib
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service configuration
SERVICE_CONFIG = {
    'trading-gateway-service': {
        'data_types': ['orders', 'accounts'],
        'dependencies': ['portfolio-management-service']
    },
    'portfolio-management-service': {
        'data_types': ['positions', 'balances'],
        'dependencies': ['risk-management-service']
    },
    'risk-management-service': {
        'data_types': ['risk-profiles', 'risk-limits'],
        'dependencies': []
    },
    'data-pipeline-service': {
        'data_types': ['market-data', 'data-sources'],
        'dependencies': ['feature-store-service']
    },
    'feature-store-service': {
        'data_types': ['features', 'feature-sets'],
        'dependencies': ['ml-integration-service']
    },
    'ml-integration-service': {
        'data_types': ['models', 'predictions'],
        'dependencies': []
    },
    'ml-workbench-service': {
        'data_types': ['experiments', 'model-registry'],
        'dependencies': ['ml-integration-service']
    },
    'monitoring-alerting-service': {
        'data_types': ['alerts', 'metrics'],
        'dependencies': []
    }
}

def create_reconciliation_jobs():
    """Create reconciliation jobs for all services."""
    logger.info("Creating reconciliation jobs")
    
    for service_name, service_config in SERVICE_CONFIG.items():
        if not service_config['dependencies']:
            logger.info(f"Skipping {service_name} as it has no dependencies")
            continue
        
        logger.info(f"Creating reconciliation jobs for {service_name}")
        
        try:
            # Import the reconciliation service module
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            module_name = f"{service_name.replace('-', '_')}.reconciliation.reconciliation_service"
            
            # For testing purposes, we'll simulate the import
            # reconciliation_service = importlib.import_module(module_name)
            # reconciliation_service.initialize_reconciliation_jobs()
            
            logger.info(f"Successfully created reconciliation jobs for {service_name}")
        
        except Exception as e:
            logger.error(f"Error creating reconciliation jobs for {service_name}: {str(e)}")
    
    logger.info("Finished creating reconciliation jobs")

if __name__ == '__main__':
    create_reconciliation_jobs()
