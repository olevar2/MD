#!/usr/bin/env python3
"""
Reconciliation service for data-pipeline-service.

This module provides the service-specific implementation of data reconciliation.
"""

import logging
from typing import Dict, List, Any, Optional
from common_lib.data_reconciliation.reconciliation_engine import ReconciliationEngine
from common_lib.data_reconciliation.reconciliation_functions import *

logger = logging.getLogger(__name__)

# Create reconciliation engine
reconciliation_engine = ReconciliationEngine('data-pipeline-service')

def initialize_reconciliation_jobs():
    """Initialize reconciliation jobs for the service."""
    logger.info("Initializing reconciliation jobs")
    
    # Register reconciliation jobs

    # Register job for features reconciliation with feature-store-service
    reconciliation_engine.register_job(
        job_id='data-pipeline-service_feature-store-service_features',
        source_service='data-pipeline-service',
        target_service='feature-store-service',
        data_type='features',
        reconciliation_function=reconcile_features
    )

    # Register job for feature-sets reconciliation with feature-store-service
    reconciliation_engine.register_job(
        job_id='data-pipeline-service_feature-store-service_feature-sets',
        source_service='data-pipeline-service',
        target_service='feature-store-service',
        data_type='feature-sets',
        reconciliation_function=reconcile_feature_sets
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
