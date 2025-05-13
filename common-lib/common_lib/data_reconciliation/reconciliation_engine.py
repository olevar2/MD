#!/usr/bin/env python3
"""
Reconciliation engine for data consistency checks.

This module provides the core functionality for data reconciliation
between different services in the Forex Trading Platform.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class ReconciliationEngine:
    """Engine for reconciling data between services."""
    
    def __init__(self, name: str):
        """
        Initialize the reconciliation engine.
        
        Args:
            name: Name of the reconciliation engine
        """
        self.name = name
        self.reconciliation_jobs = {}
        logger.info(f"Initialized reconciliation engine: {name}")
    
    def register_job(self, job_id: str, source_service: str, target_service: str,
                    data_type: str, reconciliation_function: Callable) -> None:
        """
        Register a reconciliation job.
        
        Args:
            job_id: Unique identifier for the job
            source_service: Source service name
            target_service: Target service name
            data_type: Type of data to reconcile
            reconciliation_function: Function to perform the reconciliation
        """
        self.reconciliation_jobs[job_id] = {
            'source_service': source_service,
            'target_service': target_service,
            'data_type': data_type,
            'reconciliation_function': reconciliation_function,
            'last_run': None,
            'status': 'registered'
        }
        logger.info(f"Registered reconciliation job: {job_id}")
    
    def run_job(self, job_id: str, source_data: Any, target_data: Any) -> Dict[str, Any]:
        """
        Run a reconciliation job.
        
        Args:
            job_id: Job identifier
            source_data: Data from the source service
            target_data: Data from the target service
            
        Returns:
            Reconciliation results
        """
        if job_id not in self.reconciliation_jobs:
            logger.error(f"Job not found: {job_id}")
            return {'status': 'error', 'message': f"Job not found: {job_id}"}
        
        job = self.reconciliation_jobs[job_id]
        
        try:
            result = job['reconciliation_function'](source_data, target_data)
            job['last_run'] = datetime.now()
            job['status'] = 'completed'
            logger.info(f"Successfully ran reconciliation job: {job_id}")
            return {
                'status': 'success',
                'job_id': job_id,
                'source_service': job['source_service'],
                'target_service': job['target_service'],
                'data_type': job['data_type'],
                'timestamp': job['last_run'].isoformat(),
                'result': result
            }
        
        except Exception as e:
            job['status'] = 'failed'
            logger.error(f"Error running reconciliation job {job_id}: {str(e)}")
            return {
                'status': 'error',
                'job_id': job_id,
                'message': str(e)
            }
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a reconciliation job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status
        """
        if job_id not in self.reconciliation_jobs:
            logger.error(f"Job not found: {job_id}")
            return {'status': 'error', 'message': f"Job not found: {job_id}"}
        
        job = self.reconciliation_jobs[job_id]
        return {
            'job_id': job_id,
            'source_service': job['source_service'],
            'target_service': job['target_service'],
            'data_type': job['data_type'],
            'last_run': job['last_run'].isoformat() if job['last_run'] else None,
            'status': job['status']
        }
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all reconciliation jobs.
        
        Returns:
            List of job statuses
        """
        return [
            {
                'job_id': job_id,
                'source_service': job['source_service'],
                'target_service': job['target_service'],
                'data_type': job['data_type'],
                'last_run': job['last_run'].isoformat() if job['last_run'] else None,
                'status': job['status']
            }
            for job_id, job in self.reconciliation_jobs.items()
        ]
