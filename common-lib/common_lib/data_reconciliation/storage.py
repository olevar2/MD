#!/usr/bin/env python3
"""
Storage for reconciliation results.
"""

import logging
import datetime
import json
import os
from typing import Dict, List, Any, Optional, Union
import sqlite3

from .reconciliation_engine import ReconciliationResult

logger = logging.getLogger(__name__)

class ReconciliationStorage:
    """Base class for reconciliation result storage."""
    
    def store_result(self, job_name: str, result: ReconciliationResult) -> None:
        """Store a reconciliation result.
        
        Args:
            job_name: Name of the job
            result: Reconciliation result
        """
        raise NotImplementedError("Subclasses must implement store_result")
    
    def get_result(self, job_name: str, result_id: str) -> Optional[ReconciliationResult]:
        """Get a reconciliation result.
        
        Args:
            job_name: Name of the job
            result_id: ID of the result
            
        Returns:
            Reconciliation result or None if not found
        """
        raise NotImplementedError("Subclasses must implement get_result")
    
    def get_latest_result(self, job_name: str) -> Optional[ReconciliationResult]:
        """Get the latest reconciliation result for a job.
        
        Args:
            job_name: Name of the job
            
        Returns:
            Latest reconciliation result or None if not found
        """
        raise NotImplementedError("Subclasses must implement get_latest_result")
    
    def get_results(
        self,
        job_name: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        limit: int = 100
    ) -> List[ReconciliationResult]:
        """Get reconciliation results for a job.
        
        Args:
            job_name: Name of the job
            start_time: Start time for filtering results
            end_time: End time for filtering results
            limit: Maximum number of results to return
            
        Returns:
            List of reconciliation results
        """
        raise NotImplementedError("Subclasses must implement get_results")


class FileReconciliationStorage(ReconciliationStorage):
    """File-based storage for reconciliation results."""
    
    def __init__(self, directory: str):
        """Initialize file-based storage.
        
        Args:
            directory: Directory to store results
        """
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
    
    def store_result(self, job_name: str, result: ReconciliationResult) -> None:
        """Store a reconciliation result.
        
        Args:
            job_name: Name of the job
            result: Reconciliation result
        """
        # Create job directory
        job_dir = os.path.join(self.directory, job_name)
        os.makedirs(job_dir, exist_ok=True)
        
        # Create result file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_id = f"{timestamp}_{result.entity_type}"
        result_file = os.path.join(job_dir, f"{result_id}.json")
        
        # Write result to file
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Stored reconciliation result: {result_file}")
    
    def get_result(self, job_name: str, result_id: str) -> Optional[ReconciliationResult]:
        """Get a reconciliation result.
        
        Args:
            job_name: Name of the job
            result_id: ID of the result
            
        Returns:
            Reconciliation result or None if not found
        """
        result_file = os.path.join(self.directory, job_name, f"{result_id}.json")
        
        if not os.path.exists(result_file):
            return None
        
        try:
            with open(result_file, "r") as f:
                result_dict = json.load(f)
            
            return self._dict_to_result(result_dict)
        except Exception as e:
            logger.error(f"Error loading reconciliation result {result_file}: {str(e)}")
            return None
    
    def get_latest_result(self, job_name: str) -> Optional[ReconciliationResult]:
        """Get the latest reconciliation result for a job.
        
        Args:
            job_name: Name of the job
            
        Returns:
            Latest reconciliation result or None if not found
        """
        job_dir = os.path.join(self.directory, job_name)
        
        if not os.path.exists(job_dir):
            return None
        
        # Get all result files
        result_files = [
            os.path.join(job_dir, f)
            for f in os.listdir(job_dir)
            if f.endswith(".json")
        ]
        
        if not result_files:
            return None
        
        # Sort by modification time (newest first)
        result_files.sort(key=os.path.getmtime, reverse=True)
        
        # Load the newest result
        try:
            with open(result_files[0], "r") as f:
                result_dict = json.load(f)
            
            return self._dict_to_result(result_dict)
        except Exception as e:
            logger.error(f"Error loading reconciliation result {result_files[0]}: {str(e)}")
            return None
    
    def get_results(
        self,
        job_name: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        limit: int = 100
    ) -> List[ReconciliationResult]:
        """Get reconciliation results for a job.
        
        Args:
            job_name: Name of the job
            start_time: Start time for filtering results
            end_time: End time for filtering results
            limit: Maximum number of results to return
            
        Returns:
            List of reconciliation results
        """
        job_dir = os.path.join(self.directory, job_name)
        
        if not os.path.exists(job_dir):
            return []
        
        # Get all result files
        result_files = [
            os.path.join(job_dir, f)
            for f in os.listdir(job_dir)
            if f.endswith(".json")
        ]
        
        if not result_files:
            return []
        
        # Sort by modification time (newest first)
        result_files.sort(key=os.path.getmtime, reverse=True)
        
        # Load results
        results = []
        for result_file in result_files[:limit]:
            try:
                with open(result_file, "r") as f:
                    result_dict = json.load(f)
                
                result = self._dict_to_result(result_dict)
                
                # Filter by time
                if start_time and result.start_time < start_time:
                    continue
                if end_time and result.end_time > end_time:
                    continue
                
                results.append(result)
            except Exception as e:
                logger.error(f"Error loading reconciliation result {result_file}: {str(e)}")
        
        return results
    
    def _dict_to_result(self, result_dict: Dict[str, Any]) -> ReconciliationResult:
        """Convert a dictionary to a ReconciliationResult.
        
        Args:
            result_dict: Dictionary representation of a result
            
        Returns:
            ReconciliationResult
        """
        # Convert ISO format strings to datetime objects
        start_time = datetime.datetime.fromisoformat(result_dict["start_time"])
        end_time = datetime.datetime.fromisoformat(result_dict["end_time"])
        
        return ReconciliationResult(
            entity_type=result_dict["entity_type"],
            source_system=result_dict["source_system"],
            target_system=result_dict["target_system"],
            matched_count=result_dict["matched_count"],
            missing_in_target_count=result_dict["missing_in_target_count"],
            missing_in_source_count=result_dict["missing_in_source_count"],
            mismatched_count=result_dict["mismatched_count"],
            missing_in_target=result_dict["missing_in_target"],
            missing_in_source=result_dict["missing_in_source"],
            mismatched=result_dict["mismatched"],
            start_time=start_time,
            end_time=end_time
        )
