"""
GPU Accelerator Module

This module provides GPU acceleration for computationally intensive operations
in the forex trading platform, such as technical indicator calculation and
pattern recognition.

Features:
- Automatic detection of available GPU resources
- Fallback to CPU when GPU is not available
- Optimized implementations of common financial calculations
- Batched processing for efficient GPU utilization
"""
import logging
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import threading
logger = logging.getLogger(__name__)
try:
    import cupy as cp
    import numba
    from numba import cuda
    GPU_AVAILABLE = cuda.is_available()
except ImportError:
    cp = None
    cuda = None
    GPU_AVAILABLE = False
if GPU_AVAILABLE:
    logger.info(f'GPU acceleration available: {cuda.get_current_device().name}'
        )
else:
    logger.info('GPU acceleration not available, falling back to CPU')
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class GPUAccelerator:
    """
    GPU accelerator for computationally intensive operations.
    
    Features:
    - Automatic detection of available GPU resources
    - Fallback to CPU when GPU is not available
    - Optimized implementations of common financial calculations
    - Batched processing for efficient GPU utilization
    """

    @with_exception_handling
    def __init__(self, enable_gpu: bool=True, memory_limit_mb: Optional[int
        ]=None, batch_size: int=1000):
        """
        Initialize the GPU accelerator.
        
        Args:
            enable_gpu: Whether to enable GPU acceleration
            memory_limit_mb: Optional GPU memory limit in MB
            batch_size: Batch size for batched processing
        """
        self.enable_gpu = enable_gpu and GPU_AVAILABLE
        self.memory_limit_mb = memory_limit_mb
        self.batch_size = batch_size
        if self.enable_gpu:
            try:
                if memory_limit_mb:
                    cuda.set_memory_manager(cuda.MemoryManager(managed=True,
                        memory_limit=memory_limit_mb * 1024 * 1024))
                self._warm_up_gpu()
                logger.info(
                    f'GPU accelerator initialized with batch_size={batch_size}, memory_limit={memory_limit_mb}MB'
                    )
            except Exception as e:
                logger.error(f'Failed to initialize GPU: {e}', exc_info=True)
                self.enable_gpu = False
        self.context = threading.local()

    @with_exception_handling
    def _warm_up_gpu(self):
        """Warm up GPU with a simple calculation."""
        if not self.enable_gpu:
            return
        try:
            a = cp.random.rand(100, 100)
            b = cp.random.rand(100, 100)
            c = cp.matmul(a, b)
            cp.cuda.Stream.null.synchronize()
            logger.debug('GPU warm-up completed')
        except Exception as e:
            logger.warning(f'GPU warm-up failed: {e}')

    @with_exception_handling
    def to_gpu(self, data: np.ndarray) ->Union[np.ndarray, 'cp.ndarray']:
        """
        Transfer data to GPU if available.
        
        Args:
            data: NumPy array to transfer
            
        Returns:
            CuPy array if GPU is available, otherwise NumPy array
        """
        if not self.enable_gpu or cp is None:
            return data
        try:
            return cp.asarray(data)
        except Exception as e:
            logger.warning(f'Failed to transfer data to GPU: {e}')
            return data

    @with_exception_handling
    def to_cpu(self, data: Union[np.ndarray, 'cp.ndarray']) ->np.ndarray:
        """
        Transfer data from GPU to CPU if needed.
        
        Args:
            data: CuPy or NumPy array
            
        Returns:
            NumPy array
        """
        if not self.enable_gpu or cp is None or isinstance(data, np.ndarray):
            return data
        try:
            return cp.asnumpy(data)
        except Exception as e:
            logger.warning(f'Failed to transfer data from GPU: {e}')
            return data

    @with_analysis_resilience('calculate_technical_indicators')
    @with_exception_handling
    def calculate_technical_indicators(self, price_data: np.ndarray,
        indicators: List[str], parameters: Dict[str, Dict[str, Any]]=None
        ) ->Dict[str, np.ndarray]:
        """
        Calculate multiple technical indicators using GPU acceleration.
        
        Args:
            price_data: Price data array (OHLCV)
            indicators: List of indicator names to calculate
            parameters: Optional parameters for each indicator
            
        Returns:
            Dictionary mapping indicator names to result arrays
        """
        if not self.enable_gpu:
            return self._calculate_indicators_cpu(price_data, indicators,
                parameters)
        try:
            gpu_data = self.to_gpu(price_data)
            results = {}
            for indicator in indicators:
                if indicator.lower() == 'sma':
                    period = parameters.get('sma', {}).get('period', 14)
                    results[indicator] = self._calculate_sma_gpu(gpu_data,
                        period)
                elif indicator.lower() == 'ema':
                    period = parameters.get('ema', {}).get('period', 14)
                    results[indicator] = self._calculate_ema_gpu(gpu_data,
                        period)
                elif indicator.lower() == 'rsi':
                    period = parameters.get('rsi', {}).get('period', 14)
                    results[indicator] = self._calculate_rsi_gpu(gpu_data,
                        period)
                elif indicator.lower() == 'macd':
                    fast_period = parameters.get('macd', {}).get('fast_period',
                        12)
                    slow_period = parameters.get('macd', {}).get('slow_period',
                        26)
                    signal_period = parameters.get('macd', {}).get(
                        'signal_period', 9)
                    results[indicator] = self._calculate_macd_gpu(gpu_data,
                        fast_period, slow_period, signal_period)
                elif indicator.lower() == 'bollinger_bands':
                    period = parameters.get('bollinger_bands', {}).get('period'
                        , 20)
                    std_dev = parameters.get('bollinger_bands', {}).get(
                        'std_dev', 2.0)
                    results[indicator] = self._calculate_bollinger_bands_gpu(
                        gpu_data, period, std_dev)
                else:
                    logger.warning(
                        f'Unsupported indicator for GPU acceleration: {indicator}'
                        )
                    cpu_result = self._calculate_indicator_cpu(price_data,
                        indicator, parameters)
                    results[indicator] = self.to_gpu(cpu_result)
            cpu_results = {k: self.to_cpu(v) for k, v in results.items()}
            return cpu_results
        except Exception as e:
            logger.error(f'GPU calculation failed: {e}', exc_info=True)
            return self._calculate_indicators_cpu(price_data, indicators,
                parameters)

    def _calculate_indicators_cpu(self, price_data: np.ndarray, indicators:
        List[str], parameters: Dict[str, Dict[str, Any]]=None) ->Dict[str,
        np.ndarray]:
        """CPU fallback for technical indicator calculation."""
        results = {}
        for indicator in indicators:
            results[indicator] = self._calculate_indicator_cpu(price_data,
                indicator, parameters)
        return results

    def _calculate_indicator_cpu(self, price_data: np.ndarray, indicator:
        str, parameters: Dict[str, Dict[str, Any]]=None) ->np.ndarray:
        """Calculate a single technical indicator using CPU."""
        params = parameters.get(indicator.lower(), {}) if parameters else {}
        if indicator.lower() == 'sma':
            period = params.get('period', 14)
            return self._calculate_sma_cpu(price_data, period)
        elif indicator.lower() == 'ema':
            period = params.get('period', 14)
            return self._calculate_ema_cpu(price_data, period)
        elif indicator.lower() == 'rsi':
            period = params.get('period', 14)
            return self._calculate_rsi_cpu(price_data, period)
        elif indicator.lower() == 'macd':
            fast_period = params.get('fast_period', 12)
            slow_period = params.get('slow_period', 26)
            signal_period = params.get('signal_period', 9)
            return self._calculate_macd_cpu(price_data, fast_period,
                slow_period, signal_period)
        elif indicator.lower() == 'bollinger_bands':
            period = params.get('period', 20)
            std_dev = params.get('std_dev', 2.0)
            return self._calculate_bollinger_bands_cpu(price_data, period,
                std_dev)
        else:
            raise ValueError(f'Unsupported indicator: {indicator}')

    def _calculate_sma_gpu(self, price_data: 'cp.ndarray', period: int
        ) ->'cp.ndarray':
        """Calculate Simple Moving Average using GPU."""
        if len(price_data.shape) == 1:
            return cp.convolve(price_data, cp.ones(period) / period, mode=
                'valid')
        else:
            close_prices = price_data[:, 3]
            return cp.convolve(close_prices, cp.ones(period) / period, mode
                ='valid')

    def _calculate_ema_gpu(self, price_data: 'cp.ndarray', period: int
        ) ->'cp.ndarray':
        """
        Calculate Exponential Moving Average using GPU with optimized implementation.
        
        This implementation uses a more efficient algorithm that avoids the loop
        by using vectorized operations where possible.
        """
        if len(price_data.shape) == 1:
            prices = price_data
        else:
            prices = price_data[:, 3]
        if len(prices) < period:
            return cp.full_like(prices, cp.nan)
        alpha = 2.0 / (period + 1)
        ema = cp.empty_like(prices)
        ema[:period] = cp.nan
        ema[period - 1] = cp.mean(prices[:period])
        if len(prices) > period:
            decay_factors = cp.power(1 - alpha, cp.arange(len(prices) - period)
                )
            for i in range(period, len(prices)):
                window = prices[i - len(decay_factors):i + 1]
                window = window[::-1]
                if len(window) >= len(decay_factors):
                    weighted_sum = alpha * cp.sum(window[:len(decay_factors
                        )] * decay_factors)
                    weighted_sum += (1 - alpha) ** len(decay_factors) * ema[
                        period - 1]
                    ema[i] = weighted_sum
                else:
                    ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _calculate_rsi_gpu(self, price_data: 'cp.ndarray', period: int
        ) ->'cp.ndarray':
        """Calculate Relative Strength Index using GPU with optimized vectorized operations."""
        if len(price_data.shape) == 1:
            prices = price_data
        else:
            prices = price_data[:, 3]
        deltas = cp.diff(prices)
        gains = cp.maximum(deltas, 0)
        losses = cp.maximum(-deltas, 0)
        gains = cp.pad(gains, (1, 0), 'constant', constant_values=0)
        losses = cp.pad(losses, (1, 0), 'constant', constant_values=0)
        avg_gain = cp.zeros_like(prices)
        avg_loss = cp.zeros_like(prices)
        if len(gains) >= period:
            avg_gain[period - 1] = cp.sum(gains[1:period]) / period
            avg_loss[period - 1] = cp.sum(losses[1:period]) / period
            alpha = 1.0 / period
            avg_gains = cp.zeros_like(prices)
            avg_losses = cp.zeros_like(prices)
            avg_gains[period - 1] = avg_gain[period - 1]
            avg_losses[period - 1] = avg_loss[period - 1]
            for i in range(period, len(prices)):
                avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i - 1
                    ]
                avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[
                    i - 1]
            rs = cp.divide(avg_gains[period - 1:], avg_losses[period - 1:],
                out=cp.ones_like(avg_gains[period - 1:]), where=avg_losses[
                period - 1:] != 0)
            rsi = 100 - 100 / (1 + rs)
            result = cp.full_like(prices, cp.nan)
            result[period - 1:] = rsi
            return result
        else:
            return cp.full_like(prices, cp.nan)

    def _calculate_sma_cpu(self, price_data: np.ndarray, period: int
        ) ->np.ndarray:
        """Calculate Simple Moving Average using CPU."""
        if len(price_data.shape) == 1:
            return np.convolve(price_data, np.ones(period) / period, mode=
                'valid')
        else:
            close_prices = price_data[:, 3]
            return np.convolve(close_prices, np.ones(period) / period, mode
                ='valid')

    def _calculate_ema_cpu(self, price_data: np.ndarray, period: int
        ) ->np.ndarray:
        """Calculate Exponential Moving Average using CPU."""
        if len(price_data.shape) == 1:
            prices = price_data
        else:
            prices = price_data[:, 3]
        alpha = 2.0 / (period + 1)
        ema = np.empty_like(prices)
        ema[:period] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _calculate_rsi_cpu(self, price_data: np.ndarray, period: int
        ) ->np.ndarray:
        """Calculate Relative Strength Index using CPU."""
        if len(price_data.shape) == 1:
            prices = price_data
        else:
            prices = price_data[:, 3]
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]
                ) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]
                ) / period
        rs = np.divide(avg_gain[period:], avg_loss[period:], out=np.
            ones_like(avg_gain[period:]), where=avg_loss[period:] != 0)
        rsi = 100 - 100 / (1 + rs)
        result = np.full_like(prices, np.nan)
        result[period:] = rsi
        return result
