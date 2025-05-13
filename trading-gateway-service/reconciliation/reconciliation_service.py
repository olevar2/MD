#!/usr/bin/env python3
"""
Reconciliation service for trading-gateway-service.

This module provides the service-specific implementation of data reconciliation.
"""

import logging
from typing import Dict, List, Any, Optional
from common_lib.data_reconciliation.reconciliation_engine import ReconciliationEngine
from common_lib.data_reconciliation.reconciliation_functions import *

logger = logging.getLogger(__name__)

# Create reconciliation engine
reconciliation_engine = ReconciliationEngine('trading-gateway-service')

def initialize_reconciliation_jobs():
    """Initialize reconciliation jobs for the service."""
    logger.info("Initializing reconciliation jobs")
    
    # Register reconciliation jobs

    # Register job for positions reconciliation with portfolio-management-service
    reconciliation_engine.register_job(
        job_id='trading-gateway-service_portfolio-management-service_positions',
        source_service='trading-gateway-service',
        target_service='portfolio-management-service',
        data_type='positions',
        reconciliation_function=reconcile_positions
    )

    # Register job for balances reconciliation with portfolio-management-service
    reconciliation_engine.register_job(
        job_id='trading-gateway-service_portfolio-management-service_balances',
        source_service='trading-gateway-service',
        target_service='portfolio-management-service',
        data_type='balances',
        reconciliation_function=reconcile_balances
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
