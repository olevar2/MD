#!/usr/bin/env python3
"""
Reconciliation service for feature-store-service.

This module provides the service-specific implementation of data reconciliation.
"""

import logging
from typing import Dict, List, Any, Optional
from common_lib.data_reconciliation.reconciliation_engine import ReconciliationEngine
from common_lib.data_reconciliation.reconciliation_functions import *

logger = logging.getLogger(__name__)

# Create reconciliation engine
reconciliation_engine = ReconciliationEngine('feature-store-service')

def initialize_reconciliation_jobs():
    """Initialize reconciliation jobs for the service."""
    logger.info("Initializing reconciliation jobs")
    
    # Register reconciliation jobs

    # Register job for models reconciliation with ml-integration-service
    reconciliation_engine.register_job(
        job_id='feature-store-service_ml-integration-service_models',
        source_service='feature-store-service',
        target_service='ml-integration-service',
        data_type='models',
        reconciliation_function=reconcile_models
    )

    # Register job for predictions reconciliation with ml-integration-service
    reconciliation_engine.register_job(
        job_id='feature-store-service_ml-integration-service_predictions',
        source_service='feature-store-service',
        target_service='ml-integration-service',
        data_type='predictions',
        reconciliation_function=reconcile_predictions
    )

def get_reconciliation_status():
    """Get the status of all reconciliation jobs."""
    return reconciliation_engine.list_jobs()

def run_reconciliation_job(job_id: str, source_data: Any, target_data: Any) -> Dict[str, Any]:
    """
    Run a specific reconciliation job.
    
    Args:
        job_id: Job identifier
        source_data: Data from this service
        target_data: Data from the target service
        
    Returns:
        Reconciliation results
    """
    return reconciliation_engine.run_job(job_id, source_data, target_data)
