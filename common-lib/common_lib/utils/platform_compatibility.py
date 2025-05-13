"""
Platform Compatibility Module

This module provides utilities for platform-specific information and compatibility checks.
It helps ensure cross-platform compatibility and optimal resource utilization.
"""

import importlib.util
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple, Union


class PlatformInfo:
    """Information about the current platform."""
    
    @staticmethod
    def get_os() -> str:
        """
        Get the operating system name.
        
        Returns:
            Operating system name ("windows", "linux", "macos", or "unknown")
        """
        system = platform.system().lower()
        
        if system == "windows" or system.startswith("win"):
            return "windows"
        elif system == "linux":
            return "linux"
        elif system == "darwin":
            return "macos"
        else:
            return "unknown"
    
    @staticmethod
    def get_python_version() -> Tuple[int, int, int]:
        """
        Get the Python version.
        
        Returns:
            Tuple of (major, minor, micro) version numbers
        """
        return sys.version_info[:3]
    
    @staticmethod
    def is_64bit() -> bool:
        """
        Check if the platform is 64-bit.
        
        Returns:
            True if 64-bit, False otherwise
        """
        return sys.maxsize > 2**32
    
    @staticmethod
    def has_gpu() -> bool:
        """
        Check if a GPU is available.
        
        Returns:
            True if a GPU is available, False otherwise
        """
        # Try to detect CUDA
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        # Try to detect TensorFlow GPU
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            pass
        
        # Try to detect NVIDIA GPU using nvidia-smi
        try:
            import subprocess
            subprocess.check_output(['nvidia-smi'])
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return False
    
    @staticmethod
    def get_memory_info() -> Dict[str, Union[int, float]]:
        """
        Get memory information.
        
        Returns:
            Dictionary with memory information (total, available, used, percent)
        """
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            }
        except ImportError:
            return {
                "total": 0,
                "available": 0,
                "used": 0,
                "percent": 0
            }
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """
        Get CPU information.
        
        Returns:
            Dictionary with CPU information
        """
        try:
            import psutil
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "usage_percent": psutil.cpu_percent(interval=0.1)
            }
            return cpu_info
        except ImportError:
            return {
                "physical_cores": os.cpu_count(),
                "logical_cores": os.cpu_count(),
                "frequency": None,
                "usage_percent": None
            }
    
    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """
        Get GPU information.
        
        Returns:
            List of dictionaries with GPU information
        """
        gpus = []
        
        # Try to get GPU info using PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpus.append({
                        "name": torch.cuda.get_device_name(i),
                        "index": i,
                        "memory_total": torch.cuda.get_device_properties(i).total_memory,
                        "memory_reserved": torch.cuda.memory_reserved(i),
                        "memory_allocated": torch.cuda.memory_allocated(i)
                    })
                return gpus
        except ImportError:
            pass
        
        # Try to get GPU info using TensorFlow
        try:
            import tensorflow as tf
            gpu_devices = tf.config.list_physical_devices('GPU')
            for i, device in enumerate(gpu_devices):
                gpus.append({
                    "name": device.name,
                    "index": i,
                    "memory_total": None,  # TensorFlow doesn't provide this info directly
                    "memory_reserved": None,
                    "memory_allocated": None
                })
            if gpus:
                return gpus
        except ImportError:
            pass
        
        return gpus
    
    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """
        Get comprehensive platform information.
        
        Returns:
            Dictionary with platform information
        """
        return {
            "os": PlatformInfo.get_os(),
            "python_version": PlatformInfo.get_python_version(),
            "is_64bit": PlatformInfo.is_64bit(),
            "has_gpu": PlatformInfo.has_gpu(),
            "memory": PlatformInfo.get_memory_info(),
            "cpu": PlatformInfo.get_cpu_info(),
            "gpu": PlatformInfo.get_gpu_info()
        }


class PlatformCompatibility:
    """Utilities for ensuring cross-platform compatibility."""
    
    @staticmethod
    def is_module_available(module_name: str) -> bool:
        """
        Check if a module is available.
        
        Args:
            module_name: Name of the module
            
        Returns:
            True if the module is available, False otherwise
        """
        return importlib.util.find_spec(module_name) is not None
    
    @staticmethod
    def get_optimal_thread_count() -> int:
        """
        Get the optimal number of threads for parallel processing.
        
        Returns:
            Optimal number of threads
        """
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=True)
            
            # Use 75% of available cores
            return max(1, int(cpu_count * 0.75))
        except ImportError:
            # Fallback to os.cpu_count
            cpu_count = os.cpu_count() or 1
            return max(1, int(cpu_count * 0.75))
    
    @staticmethod
    def get_optimal_batch_size(item_size_bytes: int = 1000, memory_limit_mb: int = 100) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            item_size_bytes: Estimated size of each item in bytes
            memory_limit_mb: Memory limit in megabytes
            
        Returns:
            Optimal batch size
        """
        # Convert memory limit to bytes
        memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
        # Calculate batch size
        batch_size = memory_limit_bytes // item_size_bytes
        
        # Ensure batch size is at least 1
        return max(1, batch_size)
    
    @staticmethod
    def is_running_in_container() -> bool:
        """
        Check if the application is running in a container.
        
        Returns:
            True if running in a container, False otherwise
        """
        # Check for Docker
        if os.path.exists('/.dockerenv'):
            return True
        
        # Check for Kubernetes
        if os.path.exists('/var/run/secrets/kubernetes.io'):
            return True
        
        # Check cgroup
        try:
            with open('/proc/1/cgroup', 'r') as f:
                return 'docker' in f.read() or 'kubepods' in f.read()
        except (IOError, FileNotFoundError):
            pass
        
        return False
