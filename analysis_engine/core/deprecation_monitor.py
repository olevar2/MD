"""
Deprecation Monitoring Module

This module provides functionality to monitor usage of deprecated modules
and track migration progress.
"""

import logging
import inspect
import os
import time
from typing import Dict, Set, List, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import json
import atexit

logger = logging.getLogger(__name__)

@dataclass
class DeprecationUsage:
    """Record of a deprecated module usage."""
    module_name: str
    caller_file: str
    caller_line: int
    caller_function: str
    timestamp: float
    count: int = 1

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "module_name": self.module_name,
            "caller_file": self.caller_file,
            "caller_line": self.caller_line,
            "caller_function": self.caller_function,
            "timestamp": self.timestamp,
            "count": self.count,
            "last_seen": datetime.fromtimestamp(self.timestamp).isoformat()
        }


class DeprecationMonitor:
    """
    Monitor usage of deprecated modules and track migration progress.
    
    This class provides functionality to:
    1. Record usage of deprecated modules
    2. Generate reports on usage
    3. Save usage data to a file for analysis
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DeprecationMonitor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the deprecation monitor."""
        if self._initialized:
            return
            
        self._usages: Dict[str, DeprecationUsage] = {}
        self._lock = threading.Lock()
        self._report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "logs",
            "deprecation_report.json"
        )
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(self._report_path), exist_ok=True)
        
        # Load existing data if available
        self._load_data()
        
        # Register save on exit
        atexit.register(self.save_data)
        
        self._initialized = True
        logger.debug("Deprecation monitor initialized")
    
    def record_usage(self, module_name: str) -> None:
        """
        Record usage of a deprecated module.
        
        Args:
            module_name: Name of the deprecated module
        """
        # Get caller information
        frame = inspect.currentframe().f_back.f_back
        if not frame:
            return
            
        caller_file = frame.f_code.co_filename
        caller_line = frame.f_lineno
        caller_function = frame.f_code.co_name
        
        # Get relative path for better readability
        try:
            caller_file = os.path.relpath(caller_file)
        except ValueError:
            pass
        
        # Create a unique key for this usage
        key = f"{module_name}:{caller_file}:{caller_line}:{caller_function}"
        
        # Record usage
        with self._lock:
            if key in self._usages:
                self._usages[key].count += 1
                self._usages[key].timestamp = time.time()
            else:
                self._usages[key] = DeprecationUsage(
                    module_name=module_name,
                    caller_file=caller_file,
                    caller_line=caller_line,
                    caller_function=caller_function,
                    timestamp=time.time()
                )
        
        # Log usage
        logger.debug(f"Recorded usage of deprecated module {module_name} from {caller_file}:{caller_line}")
    
    def get_usage_report(self) -> Dict:
        """
        Generate a report on usage of deprecated modules.
        
        Returns:
            Dict: Report data
        """
        with self._lock:
            # Group by module
            modules: Dict[str, List[DeprecationUsage]] = {}
            for usage in self._usages.values():
                if usage.module_name not in modules:
                    modules[usage.module_name] = []
                modules[usage.module_name].append(usage)
            
            # Generate report
            report = {
                "generated_at": datetime.now().isoformat(),
                "total_usages": len(self._usages),
                "modules": {
                    module: {
                        "total_usages": sum(u.count for u in usages),
                        "unique_locations": len(usages),
                        "usages": [u.to_dict() for u in usages]
                    }
                    for module, usages in modules.items()
                }
            }
            
            return report
    
    def save_data(self) -> None:
        """Save usage data to a file."""
        try:
            report = self.get_usage_report()
            with open(self._report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.debug(f"Saved deprecation report to {self._report_path}")
        except Exception as e:
            logger.error(f"Failed to save deprecation report: {e}")
    
    def _load_data(self) -> None:
        """Load existing usage data if available."""
        if not os.path.exists(self._report_path):
            return
            
        try:
            with open(self._report_path, 'r') as f:
                report = json.load(f)
                
            # Convert back to usages
            for module_name, module_data in report.get("modules", {}).items():
                for usage_data in module_data.get("usages", []):
                    key = f"{usage_data['module_name']}:{usage_data['caller_file']}:{usage_data['caller_line']}:{usage_data['caller_function']}"
                    self._usages[key] = DeprecationUsage(
                        module_name=usage_data["module_name"],
                        caller_file=usage_data["caller_file"],
                        caller_line=usage_data["caller_line"],
                        caller_function=usage_data["caller_function"],
                        timestamp=usage_data["timestamp"],
                        count=usage_data["count"]
                    )
            
            logger.debug(f"Loaded {len(self._usages)} deprecation usages from {self._report_path}")
        except Exception as e:
            logger.error(f"Failed to load deprecation report: {e}")


# Global instance
_monitor = DeprecationMonitor()

def record_usage(module_name: str) -> None:
    """
    Record usage of a deprecated module.
    
    Args:
        module_name: Name of the deprecated module
    """
    _monitor.record_usage(module_name)

def get_usage_report() -> Dict:
    """
    Generate a report on usage of deprecated modules.
    
    Returns:
        Dict: Report data
    """
    return _monitor.get_usage_report()

def save_data() -> None:
    """Save usage data to a file."""
    _monitor.save_data()
