"""
Platform Compatibility

This module provides utilities for ensuring cross-platform compatibility.
"""

import os
import sys
import platform
import logging
import importlib.util
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Configure logging
logger = logging.getLogger(__name__)

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
        # Try TensorFlow
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            return len(gpus) > 0
        except (ImportError, AttributeError):
            pass
        
        # Try PyTorch
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        # Try CUDA directly
        try:
            import cupy
            return cupy.cuda.runtime.getDeviceCount() > 0
        except ImportError:
            pass
        
        return False
    
    @staticmethod
    def get_memory_info() -> Dict[str, int]:
        """
        Get memory information.
        
        Returns:
            Dictionary with memory information (total, available)
        """
        try:
            import psutil
            vm = psutil.virtual_memory()
            return {
                "total": vm.total,
                "available": vm.available
            }
        except ImportError:
            return {
                "total": 0,
                "available": 0
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
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            
            return {
                "physical_cores": cpu_count,
                "logical_cores": cpu_count_logical,
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0
            }
        except ImportError:
            return {
                "physical_cores": os.cpu_count() or 0,
                "logical_cores": os.cpu_count() or 0,
                "frequency": 0
            }
    
    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """
        Get GPU information.
        
        Returns:
            List of dictionaries with GPU information
        """
        gpus = []
        
        # Try TensorFlow
        try:
            import tensorflow as tf
            tf_gpus = tf.config.list_physical_devices('GPU')
            
            for i, gpu in enumerate(tf_gpus):
                gpus.append({
                    "index": i,
                    "name": gpu.name,
                    "type": "tensorflow"
                })
            
            if gpus:
                return gpus
        except (ImportError, AttributeError):
            pass
        
        # Try PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpus.append({
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "type": "pytorch"
                    })
            
            if gpus:
                return gpus
        except ImportError:
            pass
        
        # Try CUDA directly
        try:
            import cupy
            num_gpus = cupy.cuda.runtime.getDeviceCount()
            
            for i in range(num_gpus):
                props = cupy.cuda.runtime.getDeviceProperties(i)
                gpus.append({
                    "index": i,
                    "name": props["name"].decode("utf-8"),
                    "type": "cuda",
                    "memory": props["totalGlobalMem"]
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
    def get_optimal_batch_size(gpu_memory_mb: Optional[int] = None) -> int:
        """
        Get the optimal batch size for GPU processing.
        
        Args:
            gpu_memory_mb: GPU memory in MB (if None, will be detected)
            
        Returns:
            Optimal batch size
        """
        if not PlatformInfo.has_gpu():
            # No GPU, use a small batch size
            return 32
        
        if gpu_memory_mb is None:
            # Try to detect GPU memory
            gpus = PlatformInfo.get_gpu_info()
            
            if gpus and "memory" in gpus[0]:
                gpu_memory_mb = gpus[0]["memory"] / (1024 * 1024)
            else:
                # Default to 4GB
                gpu_memory_mb = 4 * 1024
        
        # Heuristic: 1000 samples per GB of GPU memory
        return max(32, int(gpu_memory_mb / 1024 * 1000))
    
    @staticmethod
    def get_optimal_memory_limit() -> int:
        """
        Get the optimal memory limit for caching.
        
        Returns:
            Optimal memory limit in bytes
        """
        try:
            import psutil
            vm = psutil.virtual_memory()
            
            # Use 25% of available memory
            return int(vm.available * 0.25)
        except ImportError:
            # Default to 1GB
            return 1024 * 1024 * 1024
    
    @staticmethod
    def with_fallback(
        primary_func: Callable,
        fallback_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Call a function with a fallback.
        
        Args:
            primary_func: Primary function to call
            fallback_func: Fallback function to call if primary fails
            *args: Arguments to pass to the functions
            **kwargs: Keyword arguments to pass to the functions
            
        Returns:
            Result of the function call
        """
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed: {e}")
            return fallback_func(*args, **kwargs)
    
    @staticmethod
    def get_temp_dir() -> str:
        """
        Get a platform-appropriate temporary directory.
        
        Returns:
            Path to temporary directory
        """
        import tempfile
        return tempfile.gettempdir()
    
    @staticmethod
    def get_config_dir() -> str:
        """
        Get a platform-appropriate configuration directory.
        
        Returns:
            Path to configuration directory
        """
        os_name = PlatformInfo.get_os()
        
        if os_name == "windows":
            base_dir = os.environ.get("APPDATA", os.path.expanduser("~"))
            return os.path.join(base_dir, "ForexPlatform")
        elif os_name == "macos":
            return os.path.expanduser("~/Library/Application Support/ForexPlatform")
        else:  # linux or unknown
            xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
            if xdg_config_home:
                return os.path.join(xdg_config_home, "forex-platform")
            else:
                return os.path.expanduser("~/.config/forex-platform")
    
    @staticmethod
    def get_data_dir() -> str:
        """
        Get a platform-appropriate data directory.
        
        Returns:
            Path to data directory
        """
        os_name = PlatformInfo.get_os()
        
        if os_name == "windows":
            base_dir = os.environ.get("APPDATA", os.path.expanduser("~"))
            return os.path.join(base_dir, "ForexPlatform", "Data")
        elif os_name == "macos":
            return os.path.expanduser("~/Library/Application Support/ForexPlatform/Data")
        else:  # linux or unknown
            xdg_data_home = os.environ.get("XDG_DATA_HOME")
            if xdg_data_home:
                return os.path.join(xdg_data_home, "forex-platform")
            else:
                return os.path.expanduser("~/.local/share/forex-platform")
    
    @staticmethod
    def get_cache_dir() -> str:
        """
        Get a platform-appropriate cache directory.
        
        Returns:
            Path to cache directory
        """
        os_name = PlatformInfo.get_os()
        
        if os_name == "windows":
            base_dir = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            return os.path.join(base_dir, "ForexPlatform", "Cache")
        elif os_name == "macos":
            return os.path.expanduser("~/Library/Caches/ForexPlatform")
        else:  # linux or unknown
            xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
            if xdg_cache_home:
                return os.path.join(xdg_cache_home, "forex-platform")
            else:
                return os.path.expanduser("~/.cache/forex-platform")
    
    @staticmethod
    def ensure_dir_exists(path: str) -> str:
        """
        Ensure a directory exists.
        
        Args:
            path: Directory path
            
        Returns:
            Directory path
        """
        os.makedirs(path, exist_ok=True)
        return path
