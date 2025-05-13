#!/usr/bin/env python3
"""
Reconciliation service for portfolio-management-service.

This module provides the service-specific implementation of data reconciliation.
"""

import logging
from typing import Dict, List, Any, Optional
from common_lib.data_reconciliation.reconciliation_engine import ReconciliationEngine
from common_lib.data_reconciliation.reconciliation_functions import *

logger = logging.getLogger(__name__)

# Create reconciliation engine
reconciliation_engine = ReconciliationEngine('portfolio-management-service')

def initialize_reconciliation_jobs():
    """Initialize reconciliation jobs for the service."""
    logger.info("Initializing reconciliation jobs")
    
    # Register reconciliation jobs

    # Register job for risk-profiles reconciliation with risk-management-service
    reconciliation_engine.register_job(
        job_id='portfolio-management-service_risk-management-service_risk-profiles',
        source_service='portfolio-management-service',
        target_service='risk-management-service',
        data_type='risk-profiles',
        reconciliation_function=reconcile_risk_profiles
    )

    # Register job for risk-limits reconciliation with risk-management-service
    reconciliation_engine.register_job(
        job_id='portfolio-management-service_risk-management-service_risk-limits',
        source_service='portfolio-management-service',
        target_service='risk-management-service',
        data_type='risk-limits',
        reconciliation_function=reconcile_risk_limits
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
