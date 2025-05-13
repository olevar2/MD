#!/usr/bin/env python3
"""
Reconciliation Job Scheduler for managing data reconciliation jobs.
"""

import logging
import datetime
import time
import threading
import schedule
from typing import Dict, List, Any, Optional, Callable, Tuple, Set, Union

from .reconciliation_engine import ReconciliationEngine, ReconciliationResult
from .storage import ReconciliationStorage

logger = logging.getLogger(__name__)

class ReconciliationJob:
    """A job for reconciling data between systems."""
    
    def __init__(
        self,
        name: str,
        entity_type: str,
        source_system: str,
        target_system: str,
        source_data_provider: Callable[[], List[Dict[str, Any]]],
        target_data_provider: Callable[[], List[Dict[str, Any]]],
        id_field: str = "id",
        comparison_fields: Optional[List[str]] = None,
        ignore_fields: Optional[List[str]] = None,
        max_records: int = 1000,
        schedule_interval: str = "1d",
        alert_on_failure: bool = True,
        alert_threshold: Optional[Dict[str, int]] = None,
        storage: Optional[ReconciliationStorage] = None
    ):
        """Initialize a reconciliation job.
        
        Args:
            name: Name of the job
            entity_type: Type of entity to reconcile
            source_system: Source system name
            target_system: Target system name
            source_data_provider: Function to get data from the source system
            target_data_provider: Function to get data from the target system
            id_field: Field to use as the unique identifier
            comparison_fields: Fields to compare (if None, compare all fields)
            ignore_fields: Fields to ignore in comparison
            max_records: Maximum number of records to process
            schedule_interval: Schedule interval (e.g., "1d", "1h", "30m")
            alert_on_failure: Whether to alert on failure
            alert_threshold: Thresholds for alerting (e.g., {"missing_in_target_count": 10})
            storage: Storage for reconciliation results
        """
        self.name = name
        self.entity_type = entity_type
        self.source_system = source_system
        self.target_system = target_system
        self.source_data_provider = source_data_provider
        self.target_data_provider = target_data_provider
        self.id_field = id_field
        self.comparison_fields = comparison_fields
        self.ignore_fields = ignore_fields
        self.max_records = max_records
        self.schedule_interval = schedule_interval
        self.alert_on_failure = alert_on_failure
        self.alert_threshold = alert_threshold or {}
        self.storage = storage
        
        self.engine = ReconciliationEngine(
            entity_type=entity_type,
            source_system=source_system,
            target_system=target_system,
            id_field=id_field,
            comparison_fields=comparison_fields,
            ignore_fields=ignore_fields,
            max_records=max_records
        )
        
        self.last_run_time = None
        self.last_result = None
    
    def run(self) -> ReconciliationResult:
        """Run the reconciliation job.
        
        Returns:
            Reconciliation result
        """
        logger.info(f"Running reconciliation job: {self.name}")
        
        try:
            # Get data from source and target systems
            source_data = self.source_data_provider()
            target_data = self.target_data_provider()
            
            # Run reconciliation
            result = self.engine.reconcile(source_data, target_data)
            
            # Update job state
            self.last_run_time = datetime.datetime.now()
            self.last_result = result
            
            # Store result
            if self.storage:
                self.storage.store_result(self.name, result)
            
            # Check for alerts
            if not result.success and self.alert_on_failure:
                self._check_alert_thresholds(result)
            
            logger.info(f"Reconciliation job completed: {self.name}")
            logger.info(str(result))
            
            return result
        
        except Exception as e:
            logger.exception(f"Error running reconciliation job {self.name}: {str(e)}")
            raise
    
    def _check_alert_thresholds(self, result: ReconciliationResult) -> None:
        """Check if alert thresholds are exceeded.
        
        Args:
            result: Reconciliation result
        """
        alerts = []
        
        # Check missing in target threshold
        if (
            "missing_in_target_count" in self.alert_threshold and
            result.missing_in_target_count > self.alert_threshold["missing_in_target_count"]
        ):
            alerts.append(
                f"Missing in target count ({result.missing_in_target_count}) "
                f"exceeds threshold ({self.alert_threshold['missing_in_target_count']})"
            )
        
        # Check missing in source threshold
        if (
            "missing_in_source_count" in self.alert_threshold and
            result.missing_in_source_count > self.alert_threshold["missing_in_source_count"]
        ):
            alerts.append(
                f"Missing in source count ({result.missing_in_source_count}) "
                f"exceeds threshold ({self.alert_threshold['missing_in_source_count']})"
            )
        
        # Check mismatched threshold
        if (
            "mismatched_count" in self.alert_threshold and
            result.mismatched_count > self.alert_threshold["mismatched_count"]
        ):
            alerts.append(
                f"Mismatched count ({result.mismatched_count}) "
                f"exceeds threshold ({self.alert_threshold['mismatched_count']})"
            )
        
        # Send alerts
        if alerts:
            alert_message = (
                f"Reconciliation job {self.name} failed with the following issues:\n"
                f"{chr(10).join(alerts)}"
            )
            logger.warning(alert_message)
            # TODO: Send alert to monitoring system


class ReconciliationJobScheduler:
    """Scheduler for reconciliation jobs."""
    
    def __init__(self):
        """Initialize the scheduler."""
        self.jobs = {}
        self.running = False
        self.thread = None
    
    def add_job(self, job: ReconciliationJob) -> None:
        """Add a job to the scheduler.
        
        Args:
            job: Reconciliation job to add
        """
        self.jobs[job.name] = job
        
        # Schedule the job
        interval = job.schedule_interval
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            schedule.every(minutes).minutes.do(job.run)
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            schedule.every(hours).hours.do(job.run)
        elif interval.endswith('d'):
            days = int(interval[:-1])
            schedule.every(days).days.do(job.run)
        else:
            raise ValueError(f"Invalid schedule interval: {interval}")
        
        logger.info(f"Added reconciliation job: {job.name} (schedule: {interval})")
    
    def remove_job(self, job_name: str) -> None:
        """Remove a job from the scheduler.
        
        Args:
            job_name: Name of the job to remove
        """
        if job_name in self.jobs:
            # Remove from scheduler
            schedule.clear(job_name)
            # Remove from jobs dict
            del self.jobs[job_name]
            logger.info(f"Removed reconciliation job: {job_name}")
    
    def start(self) -> None:
        """Start the scheduler."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Reconciliation job scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if not self.running:
            logger.warning("Scheduler is not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        
        logger.info("Reconciliation job scheduler stopped")
    
    def _run_scheduler(self) -> None:
        """Run the scheduler loop."""
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def run_job(self, job_name: str) -> Optional[ReconciliationResult]:
        """Run a job immediately.
        
        Args:
            job_name: Name of the job to run
            
        Returns:
            Reconciliation result or None if job not found
        """
        if job_name in self.jobs:
            job = self.jobs[job_name]
            return job.run()
        else:
            logger.warning(f"Job not found: {job_name}")
            return None
    
    def get_job(self, job_name: str) -> Optional[ReconciliationJob]:
        """Get a job by name.
        
        Args:
            job_name: Name of the job
            
        Returns:
            Reconciliation job or None if not found
        """
        return self.jobs.get(job_name)
