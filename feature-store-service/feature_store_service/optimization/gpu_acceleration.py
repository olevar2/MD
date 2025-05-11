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
            # Optimized CPU implementation using numpy's cumsum for better performance
            result = np.full_like(data, np.nan)
            cumsum = np.cumsum(np.insert(data, 0, 0))
            result[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
            return result
        
        try:
            # Transfer data to GPU
            gpu_data = self.to_gpu(data)
            
            # Compute on GPU based on backend using vectorized operations
            if self.backend == "cupy":
                # Use cumsum for efficient moving average calculation
                result = cp.full_like(gpu_data, cp.nan)
                cumsum = cp.cumsum(cp.insert(gpu_data, 0, 0))
                result[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
            elif self.backend == "torch":
                # Use cumsum for efficient moving average calculation
                result = torch.full_like(gpu_data, float('nan'))
                # Create a padded version with a zero at the beginning
                padded = torch.cat([torch.zeros(1, device="cuda"), gpu_data])
                cumsum = torch.cumsum(padded, dim=0)
                result[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
            elif self.backend == "tensorflow":
                # Use cumsum for efficient moving average calculation
                padded = tf.concat([tf.zeros(1, dtype=gpu_data.dtype), gpu_data], axis=0)
                cumsum = tf.cumsum(padded)
                # Create result array
                result = tf.concat([
                    tf.fill([window-1], tf.constant(np.nan, dtype=gpu_data.dtype)),
                    (cumsum[window:] - cumsum[:-window]) / window
                ], axis=0)
            
            # Transfer result back to CPU
            return self.to_cpu(result)
        except Exception as e:
            logger.warning(f"Error in GPU computation: {e}. Falling back to CPU implementation.")
            # Optimized CPU implementation
            result = np.full_like(data, np.nan)
            cumsum = np.cumsum(np.insert(data, 0, 0))
            result[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
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
            # Optimized CPU implementation using vectorized operations
            min_price = np.min(low)
            max_price = np.max(high)
            bin_edges = np.linspace(min_price, max_price, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Pre-allocate a matrix for bin contributions
            # Each row represents a candle, each column a price bin
            contributions = np.zeros((len(high), num_bins))
            
            # Create a mask for each candle-bin combination
            # This is much faster than looping through each candle
            for i in range(len(high)):
                # Create a mask for bins that this candle spans
                mask = (bin_centers >= low[i]) & (bin_centers <= high[i])
                # Count how many bins this candle spans
                bin_count = np.sum(mask)
                if bin_count > 0:
                    # Distribute volume equally across the bins
                    contributions[i, mask] = volume[i] / bin_count
            
            # Sum contributions across all candles
            volumes = np.sum(contributions, axis=0)
            
            return bin_centers, volumes
        
        try:
            # Transfer data to GPU
            gpu_high = self.to_gpu(high)
            gpu_low = self.to_gpu(low)
            gpu_volume = self.to_gpu(volume)
            
            # Compute on GPU based on backend using vectorized operations
            if self.backend == "cupy":
                min_price = cp.min(gpu_low)
                max_price = cp.max(gpu_high)
                bin_edges = cp.linspace(min_price, max_price, num_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Use a batched approach for better GPU utilization
                batch_size = 1000  # Process this many candles at once
                volumes = cp.zeros(num_bins)
                
                for batch_start in range(0, len(gpu_high), batch_size):
                    batch_end = min(batch_start + batch_size, len(gpu_high))
                    batch_high = gpu_high[batch_start:batch_end]
                    batch_low = gpu_low[batch_start:batch_end]
                    batch_volume = gpu_volume[batch_start:batch_end]
                    
                    # Create a mask matrix (candles x bins)
                    # Each row is a candle, each column is a bin
                    mask = (bin_centers.reshape(1, -1) >= batch_low.reshape(-1, 1)) & \
                           (bin_centers.reshape(1, -1) <= batch_high.reshape(-1, 1))
                    
                    # Count bins per candle
                    bin_counts = cp.sum(mask, axis=1, keepdims=True)
                    # Avoid division by zero
                    bin_counts = cp.maximum(bin_counts, 1)
                    
                    # Distribute volume
                    volume_per_bin = batch_volume.reshape(-1, 1) / bin_counts
                    contributions = mask * volume_per_bin
                    
                    # Sum contributions
                    volumes += cp.sum(contributions, axis=0)
                
                bin_centers = self.to_cpu(bin_centers)
                volumes = self.to_cpu(volumes)
                
            elif self.backend == "torch":
                min_price = torch.min(gpu_low)
                max_price = torch.max(gpu_high)
                bin_edges = torch.linspace(min_price, max_price, num_bins + 1, device="cuda")
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Use a batched approach for better GPU utilization
                batch_size = 1000  # Process this many candles at once
                volumes = torch.zeros(num_bins, device="cuda")
                
                for batch_start in range(0, len(gpu_high), batch_size):
                    batch_end = min(batch_start + batch_size, len(gpu_high))
                    batch_high = gpu_high[batch_start:batch_end]
                    batch_low = gpu_low[batch_start:batch_end]
                    batch_volume = gpu_volume[batch_start:batch_end]
                    
                    # Create a mask matrix (candles x bins)
                    mask = (bin_centers.unsqueeze(0) >= batch_low.unsqueeze(1)) & \
                           (bin_centers.unsqueeze(0) <= batch_high.unsqueeze(1))
                    
                    # Count bins per candle
                    bin_counts = torch.sum(mask, dim=1, keepdim=True)
                    # Avoid division by zero
                    bin_counts = torch.maximum(bin_counts, torch.ones_like(bin_counts))
                    
                    # Distribute volume
                    volume_per_bin = batch_volume.unsqueeze(1) / bin_counts
                    contributions = mask * volume_per_bin
                    
                    # Sum contributions
                    volumes += torch.sum(contributions, dim=0)
                
                bin_centers = self.to_cpu(bin_centers)
                volumes = self.to_cpu(volumes)
                
            elif self.backend == "tensorflow":
                min_price = tf.reduce_min(gpu_low)
                max_price = tf.reduce_max(gpu_high)
                bin_edges = tf.linspace(min_price, max_price, num_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Use a batched approach for better GPU utilization
                batch_size = 1000  # Process this many candles at once
                volumes = tf.zeros(num_bins)
                
                for batch_start in range(0, len(gpu_high), batch_size):
                    batch_end = min(batch_start + batch_size, len(gpu_high))
                    batch_high = gpu_high[batch_start:batch_end]
                    batch_low = gpu_low[batch_start:batch_end]
                    batch_volume = gpu_volume[batch_start:batch_end]
                    
                    # Create a mask matrix (candles x bins)
                    mask = tf.logical_and(
                        tf.expand_dims(bin_centers, 0) >= tf.expand_dims(batch_low, 1),
                        tf.expand_dims(bin_centers, 0) <= tf.expand_dims(batch_high, 1)
                    )
                    
                    # Count bins per candle
                    bin_counts = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1, keepdims=True)
                    # Avoid division by zero
                    bin_counts = tf.maximum(bin_counts, tf.ones_like(bin_counts))
                    
                    # Distribute volume
                    volume_per_bin = tf.expand_dims(batch_volume, 1) / bin_counts
                    contributions = tf.cast(mask, tf.float32) * volume_per_bin
                    
                    # Sum contributions
                    volumes += tf.reduce_sum(contributions, axis=0)
                
                bin_centers = self.to_cpu(bin_centers)
                volumes = self.to_cpu(volumes)
            
            return bin_centers, volumes
        except Exception as e:
            logger.warning(f"Error in GPU volume profile computation: {e}. Falling back to CPU implementation.")
            # Optimized CPU implementation using vectorized operations
            min_price = np.min(low)
            max_price = np.max(high)
            bin_edges = np.linspace(min_price, max_price, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Pre-allocate a matrix for bin contributions
            # Each row represents a candle, each column a price bin
            contributions = np.zeros((len(high), num_bins))
            
            # Create a mask for each candle-bin combination
            # This is much faster than looping through each candle
            for i in range(len(high)):
                # Create a mask for bins that this candle spans
                mask = (bin_centers >= low[i]) & (bin_centers <= high[i])
                # Count how many bins this candle spans
                bin_count = np.sum(mask)
                if bin_count > 0:
                    # Distribute volume equally across the bins
                    contributions[i, mask] = volume[i] / bin_count
            
            # Sum contributions across all candles
            volumes = np.sum(contributions, axis=0)
            
            return bin_centers, volumes


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.
    
    Returns:
        bool: True if GPU acceleration is available, False otherwise
    """
    return has_cupy or has_torch or has_tensorflow
