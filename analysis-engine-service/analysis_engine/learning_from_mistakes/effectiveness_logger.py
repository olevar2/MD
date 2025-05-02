"""
Effectiveness Logger Module

This module provides logging capabilities to track the effectiveness of analysis
components and support the learning-from-mistakes framework.
"""
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class EffectivenessLogger:
    """
    Logger for tracking effectiveness of analyzers

    This class handles logging the accuracy of analyzer predictions compared to
    actual outcomes, facilitating continuous improvement of analysis algorithms.
    """
    
    def __init__(self, analyzer_name: str, storage_path: str = None):
        """
        Initialize the effectiveness logger
        
        Args:
            analyzer_name: Name of the analyzer being tracked
            storage_path: Path to store effectiveness logs (default to environment variable)
        """
        self.analyzer_name = analyzer_name
        self.storage_path = storage_path or os.environ.get(
            'EFFECTIVENESS_LOG_PATH', 
            './effectiveness_logs'
        )
        self._ensure_storage_path()
        
        self.logs = []
        self.max_in_memory_logs = 100
    
    def _ensure_storage_path(self):
        """Ensure the storage directory exists"""
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            logger.info(f"Effectiveness logs will be stored in: {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to create effectiveness log directory: {e}")
    
    def log(
        self,
        prediction: Dict[str, Any],
        actual: Dict[str, Any],
        accuracy: float,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None, 
        result_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log the effectiveness of a prediction
        
        Args:
            prediction: The prediction made by the analyzer
            actual: The actual outcome
            accuracy: Calculated accuracy score (0.0-1.0)
            timeframe: Optional timeframe of the analysis
            instrument: Optional instrument being analyzed
            result_id: Optional ID of the analysis result
            parameters: Optional parameters used for the analysis
            
        Returns:
            ID of the log entry
        """
        log_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "id": log_id,
            "timestamp": timestamp,
            "analyzer_name": self.analyzer_name,
            "prediction": prediction,
            "actual": actual,
            "accuracy": accuracy,
            "timeframe": timeframe,
            "instrument": instrument,
            "result_id": result_id,
            "parameters": parameters
        }
        
        # Store in memory (limited to max_in_memory_logs)
        self.logs.append(log_entry)
        if len(self.logs) > self.max_in_memory_logs:
            self.logs = self.logs[-self.max_in_memory_logs:]
        
        # Persist to storage
        self._persist_log(log_entry)
        
        return log_id
    
    def _persist_log(self, log_entry: Dict[str, Any]) -> None:
        """
        Persist log entry to storage
        
        Args:
            log_entry: Log entry to persist
        """
        try:
            # Create directory for analyzer if it doesn't exist
            analyzer_dir = os.path.join(self.storage_path, self.analyzer_name)
            os.makedirs(analyzer_dir, exist_ok=True)
            
            # File path for this log entry
            log_file = os.path.join(analyzer_dir, f"{log_entry['id']}.json")
            
            # Write to file
            with open(log_file, 'w') as f:
                json.dump(log_entry, f, indent=2)
            
            # Also update a summary file for this analyzer
            self._update_summary(log_entry)
            
        except Exception as e:
            logger.error(f"Failed to persist effectiveness log: {e}")
    
    def _update_summary(self, log_entry: Dict[str, Any]) -> None:
        """
        Update effectiveness summary for this analyzer
        
        Args:
            log_entry: New log entry to include in summary
        """
        try:
            summary_file = os.path.join(self.storage_path, f"{self.analyzer_name}_summary.json")
            
            # Load existing summary if available
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # Initialize new summary
                summary = {
                    "analyzer_name": self.analyzer_name,
                    "total_predictions": 0,
                    "average_accuracy": 0.0,
                    "by_timeframe": {},
                    "by_instrument": {},
                    "last_updated": None
                }
            
            # Update summary
            current_total = summary["total_predictions"]
            current_avg = summary["average_accuracy"]
            
            # Calculate new average
            summary["total_predictions"] += 1
            summary["average_accuracy"] = (
                (current_total * current_avg + log_entry["accuracy"]) / 
                summary["total_predictions"]
            )
            
            # Update timeframe stats
            timeframe = log_entry.get("timeframe")
            if timeframe:
                if timeframe not in summary["by_timeframe"]:
                    summary["by_timeframe"][timeframe] = {
                        "count": 0,
                        "average_accuracy": 0.0
                    }
                
                tf_stats = summary["by_timeframe"][timeframe]
                tf_stats["count"] += 1
                tf_stats["average_accuracy"] = (
                    (tf_stats["count"] - 1) * tf_stats["average_accuracy"] + log_entry["accuracy"]
                ) / tf_stats["count"]
            
            # Update instrument stats
            instrument = log_entry.get("instrument")
            if instrument:
                if instrument not in summary["by_instrument"]:
                    summary["by_instrument"][instrument] = {
                        "count": 0,
                        "average_accuracy": 0.0
                    }
                
                inst_stats = summary["by_instrument"][instrument]
                inst_stats["count"] += 1
                inst_stats["average_accuracy"] = (
                    (inst_stats["count"] - 1) * inst_stats["average_accuracy"] + log_entry["accuracy"]
                ) / inst_stats["count"]
            
            # Update timestamp
            summary["last_updated"] = datetime.now().isoformat()
            
            # Write updated summary
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update effectiveness summary: {e}")
    
    def get_logs(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get recent effectiveness logs
        
        Args:
            limit: Maximum number of logs to return
            offset: Offset from most recent
            
        Returns:
            List of log entries
        """
        # First try in-memory logs
        if offset < len(self.logs):
            return self.logs[-offset-limit:-offset] if offset > 0 else self.logs[-limit:]
        
        # Otherwise load from storage
        return self._load_logs_from_storage(limit, offset)
    
    def _load_logs_from_storage(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        """
        Load logs from persistent storage
        
        Args:
            limit: Maximum number of logs to return
            offset: Offset from most recent
            
        Returns:
            List of log entries
        """
        try:
            analyzer_dir = os.path.join(self.storage_path, self.analyzer_name)
            if not os.path.exists(analyzer_dir):
                return []
                
            # Get all log files
            log_files = [f for f in os.listdir(analyzer_dir) if f.endswith('.json')]
            log_files.sort(key=lambda f: os.path.getmtime(os.path.join(analyzer_dir, f)), reverse=True)
            
            # Apply offset and limit
            log_files = log_files[offset:offset+limit]
            
            # Load selected logs
            logs = []
            for file in log_files:
                try:
                    with open(os.path.join(analyzer_dir, file), 'r') as f:
                        logs.append(json.load(f))
                except Exception as e:
                    logger.warning(f"Failed to load log file {file}: {e}")
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to load logs from storage: {e}")
            return []
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get effectiveness summary for this analyzer
        
        Returns:
            Summary dictionary
        """
        try:
            summary_file = os.path.join(self.storage_path, f"{self.analyzer_name}_summary.json")
            
            with open(summary_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return empty summary if not available
            return {
                "analyzer_name": self.analyzer_name,
                "total_predictions": 0,
                "average_accuracy": 0.0,
                "by_timeframe": {},
                "by_instrument": {},
                "last_updated": None
            }
        except Exception as e:
            logger.error(f"Failed to load effectiveness summary: {e}")
            return {
                "analyzer_name": self.analyzer_name,
                "error": str(e)
            }
