"""
Predictive Cache Manager

This module provides a predictive caching system that anticipates future requests
based on access patterns and precomputes likely-to-be-requested results.

Features:
- Access pattern analysis
- Predictive precomputation
- Adaptive cache sizing
- Priority-based eviction
- Background precomputation
"""

import logging
import time
import threading
import queue
import copy
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from collections import defaultdict, Counter
import concurrent.futures

from analysis_engine.utils.adaptive_cache_manager import AdaptiveCacheManager

logger = logging.getLogger(__name__)

class PredictiveCacheManager:
    """
    Predictive cache manager that anticipates future requests and precomputes results.
    
    Features:
    - Access pattern analysis
    - Predictive precomputation
    - Adaptive cache sizing
    - Priority-based eviction
    - Background precomputation
    """
    
    def __init__(
        self,
        default_ttl_seconds: int = 300,
        max_size: int = 1000,
        cleanup_interval_seconds: int = 60,
        prediction_threshold: float = 0.7,
        max_precompute_workers: int = 2,
        precomputation_interval_seconds: int = 10,
        pattern_history_size: int = 1000
    ):
        """
        Initialize the predictive cache manager.
        
        Args:
            default_ttl_seconds: Default time-to-live in seconds
            max_size: Maximum number of entries in the cache
            cleanup_interval_seconds: Interval for automatic cleanup
            prediction_threshold: Threshold for prediction confidence
            max_precompute_workers: Maximum number of precomputation workers
            precomputation_interval_seconds: Interval for precomputation
            pattern_history_size: Size of access pattern history
        """
        # Initialize underlying cache
        self.cache = AdaptiveCacheManager(
            default_ttl_seconds=default_ttl_seconds,
            max_size=max_size,
            cleanup_interval_seconds=cleanup_interval_seconds
        )
        
        self.prediction_threshold = prediction_threshold
        self.max_precompute_workers = max_precompute_workers
        self.precomputation_interval_seconds = precomputation_interval_seconds
        self.pattern_history_size = pattern_history_size
        
        # Access pattern tracking
        self.access_history = []
        self.access_patterns = defaultdict(Counter)
        self.precomputation_functions = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Precomputation queue and thread
        self.precompute_queue = queue.PriorityQueue()
        self.stop_precomputation = False
        self.precomputation_thread = threading.Thread(
            target=self._precomputation_loop,
            daemon=True
        )
        self.precomputation_thread.start()
        
        logger.debug(f"PredictiveCacheManager initialized with default_ttl={default_ttl_seconds}s, "
                    f"max_size={max_size}, prediction_threshold={prediction_threshold}")
    
    def get(self, key: Any) -> Tuple[bool, Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (hit, value) where hit is True if the key was found
        """
        # Get value from underlying cache
        hit, value = self.cache.get(key)
        
        # Update access patterns
        self._update_access_patterns(key, hit)
        
        # If cache hit, predict and precompute next likely keys
        if hit:
            self._predict_and_precompute(key)
        
        return hit, value
    
    def set(self, key: Any, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl_seconds: Optional custom TTL in seconds
        """
        # Set value in underlying cache
        self.cache.set(key, value, ttl_seconds)
    
    def register_precomputation_function(
        self,
        key_pattern: str,
        function: Callable,
        priority: int = 0
    ) -> None:
        """
        Register a function for precomputing cache values.
        
        Args:
            key_pattern: Pattern to match cache keys
            function: Function to call for precomputation
            priority: Priority for precomputation (higher is more important)
        """
        with self.lock:
            self.precomputation_functions[key_pattern] = (function, priority)
            logger.debug(f"Registered precomputation function for pattern '{key_pattern}'")
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self.lock:
            self.cache.clear()
            self.access_history.clear()
            self.access_patterns.clear()
            
            # Clear precomputation queue
            while not self.precompute_queue.empty():
                try:
                    self.precompute_queue.get_nowait()
                    self.precompute_queue.task_done()
                except queue.Empty:
                    break
            
            logger.debug("Predictive cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            cache_stats = self.cache.get_stats()
            
            stats = {
                **cache_stats,
                "access_history_size": len(self.access_history),
                "access_patterns_count": sum(len(patterns) for patterns in self.access_patterns.values()),
                "precomputation_queue_size": self.precompute_queue.qsize(),
                "precomputation_functions_count": len(self.precomputation_functions)
            }
            
            return stats
    
    def _update_access_patterns(self, key: Any, hit: bool) -> None:
        """
        Update access patterns based on a cache access.
        
        Args:
            key: Accessed cache key
            hit: Whether the access was a hit
        """
        with self.lock:
            # Add to access history
            self.access_history.append((key, time.time(), hit))
            
            # Limit history size
            if len(self.access_history) > self.pattern_history_size:
                self.access_history = self.access_history[-self.pattern_history_size:]
            
            # Update access patterns
            if len(self.access_history) >= 2:
                prev_key = self.access_history[-2][0]
                if prev_key != key:  # Don't count repeated accesses
                    self.access_patterns[prev_key][key] += 1
    
    def _predict_and_precompute(self, key: Any) -> None:
        """
        Predict and precompute next likely keys.
        
        Args:
            key: Current cache key
        """
        with self.lock:
            # Skip if no patterns for this key
            if key not in self.access_patterns:
                return
            
            # Get access patterns for this key
            patterns = self.access_patterns[key]
            
            # Skip if no patterns
            if not patterns:
                return
            
            # Calculate total accesses
            total_accesses = sum(patterns.values())
            
            # Find likely next keys
            for next_key, count in patterns.items():
                # Calculate probability
                probability = count / total_accesses
                
                # Skip if below threshold
                if probability < self.prediction_threshold:
                    continue
                
                # Check if already in cache
                hit, _ = self.cache.get(next_key)
                if hit:
                    continue
                
                # Find precomputation function
                precompute_func = self._find_precomputation_function(next_key)
                if not precompute_func:
                    continue
                
                # Add to precomputation queue
                priority = -probability  # Negative for priority queue (lower is higher priority)
                self.precompute_queue.put((priority, next_key, precompute_func))
                
                logger.debug(f"Queued precomputation for key '{next_key}' with probability {probability:.2f}")
    
    def _find_precomputation_function(self, key: Any) -> Optional[Callable]:
        """
        Find a precomputation function for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Precomputation function or None if not found
        """
        # Convert key to string for pattern matching
        key_str = str(key)
        
        # Find matching pattern
        for pattern, (func, _) in self.precomputation_functions.items():
            if pattern in key_str:
                return func
        
        return None
    
    def _precomputation_loop(self) -> None:
        """Background thread for precomputing cache values."""
        while not self.stop_precomputation:
            try:
                # Wait for precomputation interval
                time.sleep(self.precomputation_interval_seconds)
                
                # Skip if queue is empty
                if self.precompute_queue.empty():
                    continue
                
                # Process queue with thread pool
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_precompute_workers) as executor:
                    futures = []
                    
                    # Submit tasks to thread pool
                    while not self.precompute_queue.empty():
                        try:
                            priority, key, func = self.precompute_queue.get_nowait()
                            futures.append(executor.submit(self._precompute_value, key, func))
                            self.precompute_queue.task_done()
                        except queue.Empty:
                            break
                    
                    # Wait for all tasks to complete
                    concurrent.futures.wait(futures, timeout=self.precomputation_interval_seconds)
            except Exception as e:
                logger.error(f"Error in precomputation loop: {e}", exc_info=True)
    
    def _precompute_value(self, key: Any, func: Callable) -> None:
        """
        Precompute a cache value.
        
        Args:
            key: Cache key
            func: Precomputation function
        """
        try:
            # Check if already in cache
            hit, _ = self.cache.get(key)
            if hit:
                return
            
            # Call precomputation function
            start_time = time.time()
            value = func(key)
            execution_time = time.time() - start_time
            
            # Store in cache
            self.cache.set(key, value)
            
            logger.debug(f"Precomputed value for key '{key}' in {execution_time:.2f}s")
        except Exception as e:
            logger.error(f"Error precomputing value for key '{key}': {e}", exc_info=True)
    
    def shutdown(self) -> None:
        """Shutdown the cache manager and stop the precomputation thread."""
        self.stop_precomputation = True
        if self.precomputation_thread.is_alive():
            self.precomputation_thread.join(timeout=1.0)
        
        self.cache.shutdown()
        
        logger.debug("Predictive cache manager shutdown")
    
    def __del__(self) -> None:
        """Destructor to ensure threads are stopped."""
        self.shutdown()
