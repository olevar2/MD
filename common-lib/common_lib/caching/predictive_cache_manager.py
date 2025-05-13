"""
Predictive Cache Manager

This module provides a predictive caching system that anticipates future requests
and precomputes results. It extends the AdaptiveCacheManager with predictive capabilities.

Features:
- Access pattern analysis
- Predictive precomputation
- Adaptive cache sizing
- Priority-based eviction
- Background precomputation
"""

import threading
import time
import logging
import concurrent.futures
import re
from typing import Dict, Any, Optional, Callable, List, Tuple, Pattern, Union, Set
from collections import defaultdict, Counter
import heapq

from common_lib.caching.adaptive_cache_manager import (
    AdaptiveCacheManager,
    CacheEntry,
    cached
)

from common_lib.errors.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    BaseError,
    ServiceError
)

from common_lib.monitoring.performance_monitoring import (
    track_operation
)

# Create logger
logger = logging.getLogger(__name__)


class PrecomputationTask:
    """
    Task for precomputing a cache value.
    
    Attributes:
        key: Cache key
        function: Function to compute the value
        priority: Priority of the task (lower values are processed first)
        created_at: Creation timestamp
    """
    
    def __init__(
        self,
        key: Any,
        function: Callable[..., Any],
        priority: int = 0,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a precomputation task.
        
        Args:
            key: Cache key
            function: Function to compute the value
            priority: Priority of the task (lower values are processed first)
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
        """
        self.key = key
        self.function = function
        self.priority = priority
        self.args = args or []
        self.kwargs = kwargs or {}
        self.created_at = time.time()
    
    def __lt__(self, other: 'PrecomputationTask') -> bool:
        """
        Compare tasks by priority.
        
        Args:
            other: Other task
            
        Returns:
            True if this task has higher priority (lower value)
        """
        return self.priority < other.priority


class AccessPattern:
    """
    Access pattern for predictive caching.
    
    Attributes:
        source_key: Source key
        target_key: Target key
        count: Number of times this pattern has been observed
        last_observed: Last time this pattern was observed
    """
    
    def __init__(self, source_key: Any, target_key: Any):
        """
        Initialize an access pattern.
        
        Args:
            source_key: Source key
            target_key: Target key
        """
        self.source_key = source_key
        self.target_key = target_key
        self.count = 1
        self.last_observed = time.time()
    
    def observe(self) -> None:
        """Record an observation of this pattern."""
        self.count += 1
        self.last_observed = time.time()
    
    def get_score(self, current_time: Optional[float] = None) -> float:
        """
        Get the score of this pattern.
        
        Args:
            current_time: Current time (defaults to time.time())
            
        Returns:
            Pattern score
        """
        current = current_time or time.time()
        age_factor = max(0.1, min(1.0, 1.0 - (current - self.last_observed) / (24 * 3600)))
        count_factor = min(1.0, self.count / 10)
        return age_factor * count_factor


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
        pattern_history_size: int = 1000,
        redis_url: Optional[str] = None,
        service_name: str = "forex-service",
        enable_metrics: bool = True
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
            redis_url: Redis URL for distributed caching
            service_name: Service name for metrics
            enable_metrics: Whether to collect metrics
        """
        # Initialize adaptive cache manager
        self.cache = AdaptiveCacheManager(
            default_ttl_seconds=default_ttl_seconds,
            max_size=max_size,
            cleanup_interval_seconds=cleanup_interval_seconds,
            adaptive_ttl=True,
            redis_url=redis_url,
            service_name=service_name,
            enable_metrics=enable_metrics
        )
        
        # Initialize predictive caching parameters
        self.prediction_threshold = prediction_threshold
        self.max_precompute_workers = max_precompute_workers
        self.precomputation_interval_seconds = precomputation_interval_seconds
        self.pattern_history_size = pattern_history_size
        
        # Initialize access pattern tracking
        self.access_history: List[Any] = []
        self.access_patterns: Dict[Tuple[Any, Any], AccessPattern] = {}
        self.last_accessed_key: Optional[Any] = None
        self.access_lock = threading.RLock()
        
        # Initialize precomputation
        self.precomputation_functions: Dict[Pattern[str], Tuple[Callable[[Any], Any], int]] = {}
        self.precomputation_queue: List[PrecomputationTask] = []
        self.precomputation_lock = threading.RLock()
        self.precomputation_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_precompute_workers,
            thread_name_prefix="predictive-cache-"
        )
        self.precomputation_thread = None
        self.stop_precomputation = threading.Event()
        
        # Start precomputation thread
        self._start_precomputation_thread()
        
        logger.debug(
            f"PredictiveCacheManager initialized with prediction_threshold={prediction_threshold}, "
            f"max_precompute_workers={max_precompute_workers}, "
            f"precomputation_interval_seconds={precomputation_interval_seconds}, "
            f"pattern_history_size={pattern_history_size}"
        )
    
    def _start_precomputation_thread(self) -> None:
        """Start the precomputation thread."""
        self.precomputation_thread = threading.Thread(
            target=self._precomputation_loop,
            daemon=True,
            name="predictive-cache-precomputation"
        )
        self.precomputation_thread.start()
    
    def _precomputation_loop(self) -> None:
        """Precomputation loop."""
        while not self.stop_precomputation.is_set():
            try:
                # Process precomputation queue
                self._process_precomputation_queue()
                
                # Sleep for the precomputation interval
                time.sleep(self.precomputation_interval_seconds)
            except Exception as e:
                logger.error(f"Error in precomputation loop: {e}")
                time.sleep(self.precomputation_interval_seconds * 2)
    
    @with_exception_handling
    def _process_precomputation_queue(self) -> None:
        """Process the precomputation queue."""
        with self.precomputation_lock:
            # Get tasks to process
            tasks_to_process = []
            while self.precomputation_queue and len(tasks_to_process) < self.max_precompute_workers:
                tasks_to_process.append(heapq.heappop(self.precomputation_queue))
        
        # Process tasks
        for task in tasks_to_process:
            # Check if the key is already in the cache
            hit, _ = self.cache.get(task.key)
            if hit:
                continue
            
            # Submit task to executor
            self.precomputation_executor.submit(
                self._precompute_value,
                task
            )
    
    @with_exception_handling
    def _precompute_value(self, task: PrecomputationTask) -> None:
        """
        Precompute a value and store it in the cache.
        
        Args:
            task: Precomputation task
        """
        try:
            # Compute the value
            value = task.function(*task.args, **task.kwargs)
            
            # Store in cache
            self.cache.set(
                key=task.key,
                value=value,
                priority=task.priority
            )
            
            logger.debug(f"Precomputed value for key: {task.key}")
        except Exception as e:
            logger.error(f"Error precomputing value for key {task.key}: {e}")
    
    @track_operation("caching", "register_precomputation_function")
    def register_precomputation_function(
        self,
        key_pattern: str,
        function: Callable[[Any], Any],
        priority: int = 0
    ) -> None:
        """
        Register a function for precomputing values.
        
        Args:
            key_pattern: Regular expression pattern for matching keys
            function: Function to compute the value
            priority: Priority for precomputation (lower values are processed first)
        """
        pattern = re.compile(key_pattern)
        self.precomputation_functions[pattern] = (function, priority)
        logger.debug(f"Registered precomputation function for pattern: {key_pattern}")
    
    @track_operation("caching", "get")
    def get(self, key: Any) -> Tuple[bool, Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (hit, value) where hit is True if the key was found and not expired
        """
        # Record access for pattern analysis
        self._record_access(key)
        
        # Get from cache
        hit, value = self.cache.get(key)
        
        # Predict and precompute next accesses
        if hit:
            self._predict_next_accesses(key)
        
        return hit, value
    
    @track_operation("caching", "set")
    def set(
        self,
        key: Any,
        value: Any,
        ttl: Optional[int] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL in seconds
            priority: Priority for eviction (lower values are evicted first)
            metadata: Additional metadata
        """
        self.cache.set(key, value, ttl, priority, metadata)
    
    @track_operation("caching", "delete")
    def delete(self, key: Any) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        return self.cache.delete(key)
    
    @track_operation("caching", "clear")
    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
    
    @track_operation("caching", "get_or_set")
    def get_or_set(
        self,
        key: Any,
        value_func: Callable[[], Any],
        ttl: Optional[int] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get a value from the cache or compute and store it if not available.
        
        Args:
            key: Cache key
            value_func: Function to compute the value
            ttl: Optional custom TTL in seconds
            priority: Priority for eviction (lower values are evicted first)
            metadata: Additional metadata
            
        Returns:
            Cached or computed value
        """
        # Record access for pattern analysis
        self._record_access(key)
        
        # Get or compute value
        hit, value = self.cache.get(key)
        if hit:
            # Predict and precompute next accesses
            self._predict_next_accesses(key)
            return value
        
        value = value_func()
        self.cache.set(key, value, ttl, priority, metadata)
        return value
    
    def _record_access(self, key: Any) -> None:
        """
        Record an access for pattern analysis.
        
        Args:
            key: Accessed key
        """
        with self.access_lock:
            # Record access in history
            self.access_history.append(key)
            if len(self.access_history) > self.pattern_history_size:
                self.access_history.pop(0)
            
            # Record access pattern
            if self.last_accessed_key is not None and self.last_accessed_key != key:
                pattern_key = (self.last_accessed_key, key)
                if pattern_key in self.access_patterns:
                    self.access_patterns[pattern_key].observe()
                else:
                    self.access_patterns[pattern_key] = AccessPattern(
                        source_key=self.last_accessed_key,
                        target_key=key
                    )
            
            # Update last accessed key
            self.last_accessed_key = key
    
    def _predict_next_accesses(self, key: Any) -> None:
        """
        Predict and precompute next accesses.
        
        Args:
            key: Current key
        """
        with self.access_lock:
            # Find patterns with this key as source
            predictions = []
            current_time = time.time()
            
            for pattern_key, pattern in self.access_patterns.items():
                if pattern_key[0] == key:
                    score = pattern.get_score(current_time)
                    if score >= self.prediction_threshold:
                        predictions.append((pattern_key[1], score))
            
            # Sort predictions by score (descending)
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Precompute predicted values
            for predicted_key, score in predictions:
                self._schedule_precomputation(predicted_key)
    
    def _schedule_precomputation(self, key: Any) -> None:
        """
        Schedule precomputation for a key.
        
        Args:
            key: Key to precompute
        """
        # Check if the key is already in the cache
        hit, _ = self.cache.get(key)
        if hit:
            return
        
        # Find a matching precomputation function
        for pattern, (function, priority) in self.precomputation_functions.items():
            if isinstance(key, str) and pattern.match(key):
                # Create precomputation task
                task = PrecomputationTask(
                    key=key,
                    function=function,
                    priority=priority,
                    args=[key]
                )
                
                # Add to queue
                with self.precomputation_lock:
                    heapq.heappush(self.precomputation_queue, task)
                
                logger.debug(f"Scheduled precomputation for key: {key}")
                return
    
    @track_operation("caching", "get_stats")
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.cache.get_stats()
        
        with self.access_lock:
            stats.update({
                "access_history_size": len(self.access_history),
                "access_patterns_size": len(self.access_patterns),
                "precomputation_functions": len(self.precomputation_functions),
                "precomputation_queue_size": len(self.precomputation_queue)
            })
        
        return stats
    
    def __del__(self) -> None:
        """Destructor."""
        # Stop precomputation thread
        if hasattr(self, 'stop_precomputation'):
            self.stop_precomputation.set()
        
        # Shutdown executor
        if hasattr(self, 'precomputation_executor'):
            self.precomputation_executor.shutdown(wait=False)


# Create singleton instance
_default_predictive_cache_manager = PredictiveCacheManager()


def get_predictive_cache_manager() -> PredictiveCacheManager:
    """
    Get the default predictive cache manager.
    
    Returns:
        Default predictive cache manager
    """
    return _default_predictive_cache_manager
