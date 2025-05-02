"""
GPU Acceleration Module for Indicator Calculation

This module provides GPU-accelerated implementations of computation-intensive indicators
to improve performance and processing speed.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, List, Optional, Callable, Any
import logging
import importlib.util

# Check for GPU libraries
has_cupy = importlib.util.find_spec("cupy") is not None
has_torch = importlib.util.find_spec("torch") is not None
has_tensorflow = importlib.util.find_spec("tensorflow") is not None

logger = logging.getLogger(__name__)

# Import GPU libraries if available
if has_cupy:
    import cupy as cp
    logger.info("CuPy detected. GPU acceleration is available.")
elif has_torch:
    import torch
    logger.info("PyTorch detected. GPU acceleration is available.")
elif has_tensorflow:
    import tensorflow as tf
    logger.info("TensorFlow detected. GPU acceleration is available.")
else:
    logger.warning("No GPU acceleration libraries detected. Using CPU fallback.")


class GPUAccelerator:
    """
    Provides GPU acceleration for computation-intensive indicators.
    
    This class handles the transfer of data between CPU and GPU memory,
    performs accelerated calculations on the GPU, and manages memory efficiently.
    """
    
    def __init__(self, enable_gpu: bool = True, memory_limit: Optional[float] = None):
        """
        Initialize the GPU accelerator.
        
        Args:
            enable_gpu: Whether to enable GPU acceleration if available
            memory_limit: Memory limit in GB for GPU usage (None for no limit)
        """
        self.enable_gpu = enable_gpu and (has_cupy or has_torch or has_tensorflow)
        self.memory_limit = memory_limit
        
        # Select the GPU backend based on availability
        if self.enable_gpu:
            if has_cupy:
                self.backend = "cupy"
            elif has_torch:
                self.backend = "torch"
            elif has_tensorflow:
                self.backend = "tensorflow"
            else:
                self.backend = "numpy"
                self.enable_gpu = False
        else:
            self.backend = "numpy"
    
    def to_gpu(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Any:
        """
        Transfer data to GPU memory.
        
        Args:
            data: NumPy array, pandas DataFrame or Series to transfer
            
        Returns:
            GPU-compatible array
        """
        if not self.enable_gpu:
            return data
            
        try:
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                data_np = data.values if isinstance(data, pd.DataFrame) else data.to_numpy()
            else:
                data_np = data
                
            if self.backend == "cupy":
                return cp.asarray(data_np)
            elif self.backend == "torch":
                return torch.tensor(data_np, device="cuda")
            elif self.backend == "tensorflow":
                return tf.convert_to_tensor(data_np, device="/GPU:0")
            else:
                return data_np
        except Exception as e:
            logger.warning(f"Error transferring data to GPU: {e}. Using CPU fallback.")
            return data
    
    def to_cpu(self, data: Any) -> np.ndarray:
        """
        Transfer data back to CPU memory.
        
        Args:
            data: GPU array to transfer
            
        Returns:
            NumPy array
        """
        if not self.enable_gpu:
            return data
            
        try:
            if self.backend == "cupy":
                return cp.asnumpy(data)
            elif self.backend == "torch":
                return data.cpu().numpy()
            elif self.backend == "tensorflow":
                return data.numpy()
            else:
                return data
        except Exception as e:
            logger.warning(f"Error transferring data to CPU: {e}. Data may already be on CPU.")
            return data
    
    def compute_moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """
        Compute a moving average using GPU acceleration.
        
        Args:
            data: NumPy array of price data
            window: Window size for the moving average
            
        Returns:
            NumPy array with moving average values
        """
        if not self.enable_gpu:
            # CPU fallback implementation
            result = np.full_like(data, np.nan)
            for i in range(window - 1, len(data)):
                result[i] = np.mean(data[i - window + 1:i + 1])
            return result
        
        try:
            # Transfer data to GPU
            gpu_data = self.to_gpu(data)
            
            # Compute on GPU based on backend
            if self.backend == "cupy":
                result = cp.full_like(gpu_data, cp.nan)
                for i in range(window - 1, len(gpu_data)):
                    result[i] = cp.mean(gpu_data[i - window + 1:i + 1])
            elif self.backend == "torch":
                result = torch.full_like(gpu_data, float('nan'))
                for i in range(window - 1, len(gpu_data)):
                    result[i] = torch.mean(gpu_data[i - window + 1:i + 1])
            elif self.backend == "tensorflow":
                result = tf.constant(np.full_like(data, np.nan), dtype=tf.float32)
                for i in range(window - 1, len(gpu_data)):
                    result = tf.tensor_scatter_nd_update(
                        result, 
                        [[i]], 
                        [tf.reduce_mean(gpu_data[i - window + 1:i + 1])]
                    )
            
            # Transfer result back to CPU
            return self.to_cpu(result)
        except Exception as e:
            logger.warning(f"Error in GPU computation: {e}. Falling back to CPU implementation.")
            # CPU fallback implementation
            result = np.full_like(data, np.nan)
            for i in range(window - 1, len(data)):
                result[i] = np.mean(data[i - window + 1:i + 1])
            return result

    def compute_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Compute a correlation matrix using GPU acceleration.
        
        Args:
            data: NumPy array of price data with shape [n_samples, n_features]
            
        Returns:
            NumPy array with correlation matrix
        """
        if not self.enable_gpu:
            # CPU fallback implementation
            return np.corrcoef(data, rowvar=False)
        
        try:
            # Transfer data to GPU
            gpu_data = self.to_gpu(data)
            
            # Compute on GPU based on backend
            if self.backend == "cupy":
                result = cp.corrcoef(gpu_data, rowvar=False)
            elif self.backend == "torch":
                # Normalize the data
                mean = torch.mean(gpu_data, dim=0)
                std = torch.std(gpu_data, dim=0)
                normalized = (gpu_data - mean) / std
                # Compute correlation
                result = torch.matmul(normalized.T, normalized) / (gpu_data.shape[0] - 1)
            elif self.backend == "tensorflow":
                # Normalize the data
                mean = tf.reduce_mean(gpu_data, axis=0)
                std = tf.math.reduce_std(gpu_data, axis=0)
                normalized = (gpu_data - mean) / std
                # Compute correlation
                result = tf.matmul(tf.transpose(normalized), normalized) / (tf.shape(gpu_data)[0] - 1)
            
            # Transfer result back to CPU
            return self.to_cpu(result)
        except Exception as e:
            logger.warning(f"Error in GPU correlation computation: {e}. Falling back to CPU implementation.")
            # CPU fallback implementation
            return np.corrcoef(data, rowvar=False)
            
    def compute_volume_profile(self, high: np.ndarray, low: np.ndarray, volume: np.ndarray, 
                               num_bins: int = 100) -> tuple:
        """
        Compute volume profile using GPU acceleration.
        
        Args:
            high: NumPy array of high prices
            low: NumPy array of low prices
            volume: NumPy array of volume data
            num_bins: Number of price bins for the histogram
            
        Returns:
            Tuple of (bin_centers, volumes)
        """
        if not self.enable_gpu:
            # CPU fallback implementation
            min_price = np.min(low)
            max_price = np.max(high)
            bin_edges = np.linspace(min_price, max_price, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            volumes = np.zeros(num_bins)
            
            for i in range(len(high)):
                # Distribute volume across price range touched by this candle
                h = high[i]
                l = low[i]
                v = volume[i]
                
                # Find which bins this candle spans
                bin_indices = np.where((bin_centers >= l) & (bin_centers <= h))[0]
                if len(bin_indices) > 0:
                    # Distribute volume equally across the bins
                    volumes[bin_indices] += v / len(bin_indices)
            
            return bin_centers, volumes
        
        try:
            # Transfer data to GPU
            gpu_high = self.to_gpu(high)
            gpu_low = self.to_gpu(low)
            gpu_volume = self.to_gpu(volume)
            
            # Compute on GPU based on backend
            if self.backend == "cupy":
                min_price = cp.min(gpu_low)
                max_price = cp.max(gpu_high)
                bin_edges = cp.linspace(min_price, max_price, num_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                volumes = cp.zeros(num_bins)
                
                for i in range(len(gpu_high)):
                    # Distribute volume across price range touched by this candle
                    h = gpu_high[i]
                    l = gpu_low[i]
                    v = gpu_volume[i]
                    
                    # Find which bins this candle spans
                    bin_indices = cp.where((bin_centers >= l) & (bin_centers <= h))[0]
                    if len(bin_indices) > 0:
                        # Distribute volume equally across the bins
                        volumes[bin_indices] += v / len(bin_indices)
                
                bin_centers = self.to_cpu(bin_centers)
                volumes = self.to_cpu(volumes)
            elif self.backend == "torch":
                min_price = torch.min(gpu_low)
                max_price = torch.max(gpu_high)
                bin_edges = torch.linspace(min_price, max_price, num_bins + 1, device="cuda")
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                volumes = torch.zeros(num_bins, device="cuda")
                
                for i in range(len(gpu_high)):
                    # Distribute volume across price range touched by this candle
                    h = gpu_high[i]
                    l = gpu_low[i]
                    v = gpu_volume[i]
                    
                    # Find which bins this candle spans
                    bin_indices = torch.where((bin_centers >= l) & (bin_centers <= h))[0]
                    if len(bin_indices) > 0:
                        # Distribute volume equally across the bins
                        volumes[bin_indices] += v / len(bin_indices)
                
                bin_centers = self.to_cpu(bin_centers)
                volumes = self.to_cpu(volumes)
            elif self.backend == "tensorflow":
                min_price = tf.reduce_min(gpu_low)
                max_price = tf.reduce_max(gpu_high)
                bin_edges = tf.linspace(min_price, max_price, num_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                volumes = tf.zeros(num_bins)
                
                for i in range(len(gpu_high)):
                    # Distribute volume across price range touched by this candle
                    h = gpu_high[i]
                    l = gpu_low[i]
                    v = gpu_volume[i]
                    
                    # Find which bins this candle spans
                    bin_indices = tf.where((bin_centers >= l) & (bin_centers <= h))[:, 0]
                    if tf.size(bin_indices) > 0:
                        # Distribute volume equally across the bins
                        volume_per_bin = v / tf.cast(tf.size(bin_indices), tf.float32)
                        updates = tf.ones_like(bin_indices, dtype=tf.float32) * volume_per_bin
                        volumes = tf.tensor_scatter_nd_add(volumes, tf.expand_dims(bin_indices, 1), updates)
                
                bin_centers = self.to_cpu(bin_centers)
                volumes = self.to_cpu(volumes)
            
            return bin_centers, volumes
        except Exception as e:
            logger.warning(f"Error in GPU volume profile computation: {e}. Falling back to CPU implementation.")
            # CPU fallback implementation
            min_price = np.min(low)
            max_price = np.max(high)
            bin_edges = np.linspace(min_price, max_price, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            volumes = np.zeros(num_bins)
            
            for i in range(len(high)):
                # Distribute volume across price range touched by this candle
                h = high[i]
                l = low[i]
                v = volume[i]
                
                # Find which bins this candle spans
                bin_indices = np.where((bin_centers >= l) & (bin_centers <= h))[0]
                if len(bin_indices) > 0:
                    # Distribute volume equally across the bins
                    volumes[bin_indices] += v / len(bin_indices)
            
            return bin_centers, volumes


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.
    
    Returns:
        bool: True if GPU acceleration is available, False otherwise
    """
    return has_cupy or has_torch or has_tensorflow
