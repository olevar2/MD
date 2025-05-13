"""
Feature Engineering Optimizer

This module provides tools for optimizing feature engineering pipelines through
techniques like caching, incremental computation, and parallel processing.

It includes:
- Feature computation caching
- Incremental feature computation
- Parallel feature processing
- Feature pipeline benchmarking
"""
import logging
import time
import os
import json
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import multiprocessing
from functools import partial
import concurrent.futures
logger = logging.getLogger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FeatureEngineeringOptimizer:
    """
    Optimizes feature engineering pipelines for performance.

    This class provides methods for:
    - Caching feature computation results
    - Implementing incremental feature computation
    - Parallelizing feature processing
    - Benchmarking feature pipeline performance
    """

    def __init__(self, cache_dir: str='./feature_cache', max_cache_size_mb:
        int=1024, cache_ttl_seconds: int=3600 * 24, n_jobs: int=-1):
        """
        Initialize the feature engineering optimizer.

        Args:
            cache_dir: Directory to cache computed features
            max_cache_size_mb: Maximum cache size in MB
            cache_ttl_seconds: Time-to-live for cache entries in seconds
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.max_cache_size_mb = max_cache_size_mb
        self.cache_ttl_seconds = cache_ttl_seconds
        if n_jobs < 0:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
        self.cache_metadata_path = self.cache_dir / 'cache_metadata.json'
        self._init_cache_metadata()
        self.baseline_metrics = {}
        self.optimized_metrics = {}

    @with_exception_handling
    def _init_cache_metadata(self):
        """Initialize or load cache metadata."""
        if self.cache_metadata_path.exists():
            try:
                with open(self.cache_metadata_path, 'r') as f:
                    self.cache_metadata = json.load(f)
            except Exception as e:
                logger.error(f'Error loading cache metadata: {str(e)}')
                self.cache_metadata = {'entries': {}, 'total_size_bytes': 0}
        else:
            self.cache_metadata = {'entries': {}, 'total_size_bytes': 0}

    @with_exception_handling
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.cache_metadata_path, 'w') as f:
                json.dump(self.cache_metadata, f)
        except Exception as e:
            logger.error(f'Error saving cache metadata: {str(e)}')

    def _compute_cache_key(self, data: Any, feature_func: Callable,
        func_args: Tuple=None, func_kwargs: Dict=None) ->str:
        """
        Compute a cache key for the given data and feature function.

        Args:
            data: Input data for feature computation
            feature_func: Feature computation function
            func_args: Additional positional arguments to feature_func
            func_kwargs: Additional keyword arguments to feature_func

        Returns:
            Cache key string
        """
        func_hash = hashlib.md5()
        func_hash.update(feature_func.__name__.encode())
        func_hash.update(feature_func.__module__.encode())
        if func_args:
            func_hash.update(str(func_args).encode())
        if func_kwargs:
            func_hash.update(str(sorted(func_kwargs.items())).encode())
        if isinstance(data, pd.DataFrame):
            columns_str = ','.join(data.columns)
            func_hash.update(columns_str.encode())
            sample_size = min(1000, len(data))
            if sample_size > 0:
                sample_indices = np.linspace(0, len(data) - 1, sample_size,
                    dtype=int)
                sample = data.iloc[sample_indices]
                data_hash = pd.util.hash_pandas_object(sample).sum()
                func_hash.update(str(data_hash).encode())
        elif isinstance(data, np.ndarray):
            func_hash.update(str(data.shape).encode())
            sample_size = min(1000, data.size)
            if sample_size > 0:
                sample_indices = np.linspace(0, data.size - 1, sample_size,
                    dtype=int).astype(int)
                sample = data.flatten()[sample_indices]
                func_hash.update(hashlib.md5(sample.tobytes()).hexdigest().
                    encode())
        else:
            func_hash.update(str(data).encode())
        return func_hash.hexdigest()

    @with_exception_handling
    def cached_feature_computation(self, data: Any, feature_func: Callable,
        func_args: Tuple=None, func_kwargs: Dict=None, force_recompute:
        bool=False) ->Tuple[Any, Dict[str, Any]]:
        """
        Compute features with caching.

        Args:
            data: Input data for feature computation
            feature_func: Feature computation function
            func_args: Additional positional arguments to feature_func
            func_kwargs: Additional keyword arguments to feature_func
            force_recompute: Whether to force recomputation even if cached

        Returns:
            Tuple of (computed features, computation metadata)
        """
        if func_args is None:
            func_args = ()
        if func_kwargs is None:
            func_kwargs = {}
        cache_key = self._compute_cache_key(data, feature_func, func_args,
            func_kwargs)
        cache_path = self.cache_dir / f'{cache_key}.pkl'
        is_cached = False
        if not force_recompute and cache_path.exists():
            if cache_key in self.cache_metadata['entries']:
                entry = self.cache_metadata['entries'][cache_key]
                if time.time() - entry['timestamp'] < self.cache_ttl_seconds:
                    is_cached = True
        if is_cached:
            try:
                start_time = time.time()
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                load_time = time.time() - start_time
                metadata = {'cache_hit': True, 'computation_time': 0,
                    'load_time': load_time, 'total_time': load_time,
                    'cache_key': cache_key, 'timestamp': datetime.now().
                    isoformat()}
                logger.info(
                    f'Loaded features from cache in {load_time:.4f} seconds')
                return result, metadata
            except Exception as e:
                logger.warning(
                    f'Error loading from cache: {str(e)}. Will recompute.')
                is_cached = False
        start_time = time.time()
        if func_args and func_kwargs:
            result = feature_func(data, *func_args, **func_kwargs)
        elif func_args:
            result = feature_func(data, *func_args)
        elif func_kwargs:
            result = feature_func(data, **func_kwargs)
        else:
            result = feature_func(data)
        computation_time = time.time() - start_time
        try:
            save_start = time.time()
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            save_time = time.time() - save_start
            file_size = os.path.getsize(cache_path)
            self.cache_metadata['entries'][cache_key] = {'path': str(
                cache_path), 'size_bytes': file_size, 'timestamp': time.time()}
            self.cache_metadata['total_size_bytes'] += file_size
            self._save_cache_metadata()
            if self.cache_metadata['total_size_bytes'
                ] > self.max_cache_size_mb * 1024 * 1024:
                self._clean_cache()
        except Exception as e:
            logger.error(f'Error saving to cache: {str(e)}')
            save_time = 0
        metadata = {'cache_hit': False, 'computation_time':
            computation_time, 'save_time': save_time, 'total_time': 
            computation_time + save_time, 'cache_key': cache_key,
            'timestamp': datetime.now().isoformat()}
        logger.info(f'Computed features in {computation_time:.4f} seconds')
        return result, metadata

    @with_exception_handling
    def _clean_cache(self):
        """Clean the cache by removing oldest entries until under size limit."""
        logger.info('Cleaning feature cache...')
        sorted_entries = sorted(self.cache_metadata['entries'].items(), key
            =lambda x: x[1]['timestamp'])
        removed_size = 0
        removed_keys = []
        target_size = self.max_cache_size_mb * 0.8 * 1024 * 1024
        for key, entry in sorted_entries:
            if self.cache_metadata['total_size_bytes'
                ] - removed_size <= target_size:
                break
            try:
                os.remove(entry['path'])
                removed_size += entry['size_bytes']
                removed_keys.append(key)
            except Exception as e:
                logger.error(
                    f"Error removing cache file {entry['path']}: {str(e)}")
        for key in removed_keys:
            self.cache_metadata['total_size_bytes'] -= self.cache_metadata[
                'entries'][key]['size_bytes']
            del self.cache_metadata['entries'][key]
        self._save_cache_metadata()
        logger.info(
            f'Removed {len(removed_keys)} cache entries ({removed_size / (1024 * 1024):.2f} MB)'
            )

    @with_exception_handling
    def parallel_feature_computation(self, data: pd.DataFrame,
        feature_funcs: List[Callable], func_args: List[Tuple]=None,
        func_kwargs: List[Dict]=None, use_cache: bool=True, force_recompute:
        bool=False) ->Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compute multiple features in parallel.

        Args:
            data: Input data for feature computation
            feature_funcs: List of feature computation functions
            func_args: List of additional positional arguments for each function
            func_kwargs: List of additional keyword arguments for each function
            use_cache: Whether to use caching
            force_recompute: Whether to force recomputation even if cached

        Returns:
            Tuple of (dictionary of computed features, computation metadata)
        """
        if func_args is None:
            func_args = [() for _ in feature_funcs]
        if func_kwargs is None:
            func_kwargs = [{} for _ in feature_funcs]
        if len(func_args) != len(feature_funcs) or len(func_kwargs) != len(
            feature_funcs):
            raise ValueError(
                'Length of func_args and func_kwargs must match length of feature_funcs'
                )
        func_names = [func.__name__ for func in feature_funcs]
        if len(set(func_names)) != len(func_names):
            func_names = [f'{func.__name__}_{i}' for i, func in enumerate(
                feature_funcs)]
        start_time = time.time()
        results = {}
        metadata = {'computation_times': {}, 'cache_hits': {}, 'total_time':
            0, 'timestamp': datetime.now().isoformat()}

        @with_exception_handling
        def compute_feature(idx, func, args, kwargs):
    """
    Compute feature.
    
    Args:
        idx: Description of idx
        func: Description of func
        args: Description of args
        kwargs: Description of kwargs
    
    """

            func_name = func_names[idx]
            feature_start = time.time()
            try:
                if use_cache:
                    result, feat_metadata = self.cached_feature_computation(
                        data, func, args, kwargs, force_recompute)
                    is_cached = feat_metadata['cache_hit']
                    compute_time = feat_metadata['total_time']
                else:
                    if args and kwargs:
                        result = func(data, *args, **kwargs)
                    elif args:
                        result = func(data, *args)
                    elif kwargs:
                        result = func(data, **kwargs)
                    else:
                        result = func(data)
                    is_cached = False
                    compute_time = time.time() - feature_start
                return func_name, result, compute_time, is_cached, None
            except Exception as e:
                logger.error(f'Error computing feature {func_name}: {str(e)}')
                return func_name, None, time.time(
                    ) - feature_start, False, str(e)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs
            ) as executor:
            futures = [executor.submit(compute_feature, i, func, args,
                kwargs) for i, (func, args, kwargs) in enumerate(zip(
                feature_funcs, func_args, func_kwargs))]
            for future in concurrent.futures.as_completed(futures):
                func_name, result, compute_time, is_cached, error = (future
                    .result())
                if error is None:
                    results[func_name] = result
                    metadata['computation_times'][func_name] = compute_time
                    metadata['cache_hits'][func_name] = is_cached
                else:
                    metadata['computation_times'][func_name] = compute_time
                    metadata['cache_hits'][func_name] = False
                    metadata[f'error_{func_name}'] = error
        metadata['total_time'] = time.time() - start_time
        metadata['success_rate'] = len(results) / len(feature_funcs
            ) if feature_funcs else 0
        logger.info(
            f"Computed {len(results)} features in {metadata['total_time']:.4f} seconds"
            )
        return results, metadata

    @with_exception_handling
    def incremental_feature_computation(self, previous_data: pd.DataFrame,
        previous_features: Dict[str, Any], new_data: pd.DataFrame,
        feature_funcs: List[Callable], func_args: List[Tuple]=None,
        func_kwargs: List[Dict]=None, incremental_funcs: List[Callable]=None
        ) ->Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compute features incrementally for new data.

        Args:
            previous_data: Previous input data
            previous_features: Previously computed features
            new_data: New data to compute features for
            feature_funcs: List of feature computation functions
            func_args: List of additional positional arguments for each function
            func_kwargs: List of additional keyword arguments for each function
            incremental_funcs: List of incremental computation functions (optional)

        Returns:
            Tuple of (dictionary of computed features, computation metadata)
        """
        if func_args is None:
            func_args = [() for _ in feature_funcs]
        if func_kwargs is None:
            func_kwargs = [{} for _ in feature_funcs]
        if len(func_args) != len(feature_funcs) or len(func_kwargs) != len(
            feature_funcs):
            raise ValueError(
                'Length of func_args and func_kwargs must match length of feature_funcs'
                )
        if incremental_funcs is None:
            incremental_funcs = feature_funcs
        elif len(incremental_funcs) != len(feature_funcs):
            raise ValueError(
                'Length of incremental_funcs must match length of feature_funcs'
                )
        func_names = [func.__name__ for func in feature_funcs]
        if len(set(func_names)) != len(func_names):
            func_names = [f'{func.__name__}_{i}' for i, func in enumerate(
                feature_funcs)]
        start_time = time.time()
        results = {}
        metadata = {'computation_times': {}, 'total_time': 0, 'timestamp':
            datetime.now().isoformat()}
        combined_data = pd.concat([previous_data, new_data], ignore_index=True)
        for i, (func_name, feature_func, inc_func, args, kwargs) in enumerate(
            zip(func_names, feature_funcs, incremental_funcs, func_args,
            func_kwargs)):
            feature_start = time.time()
            try:
                if func_name in previous_features:
                    if args and kwargs:
                        result = inc_func(previous_data, new_data,
                            previous_features[func_name], *args, **kwargs)
                    elif args:
                        result = inc_func(previous_data, new_data,
                            previous_features[func_name], *args)
                    elif kwargs:
                        result = inc_func(previous_data, new_data,
                            previous_features[func_name], **kwargs)
                    else:
                        result = inc_func(previous_data, new_data,
                            previous_features[func_name])
                elif args and kwargs:
                    result = feature_func(combined_data, *args, **kwargs)
                elif args:
                    result = feature_func(combined_data, *args)
                elif kwargs:
                    result = feature_func(combined_data, **kwargs)
                else:
                    result = feature_func(combined_data)
                results[func_name] = result
                metadata['computation_times'][func_name] = time.time(
                    ) - feature_start
            except Exception as e:
                logger.error(f'Error computing feature {func_name}: {str(e)}')
                metadata['computation_times'][func_name] = time.time(
                    ) - feature_start
                metadata[f'error_{func_name}'] = str(e)
        metadata['total_time'] = time.time() - start_time
        metadata['success_rate'] = len(results) / len(feature_funcs
            ) if feature_funcs else 0
        logger.info(
            f"Incrementally computed {len(results)} features in {metadata['total_time']:.4f} seconds"
            )
        return results, metadata

    @with_exception_handling
    def benchmark_feature_pipeline(self, data: pd.DataFrame, feature_funcs:
        List[Callable], func_args: List[Tuple]=None, func_kwargs: List[Dict
        ]=None, n_runs: int=5, use_cache: bool=False, use_parallel: bool=True
        ) ->Dict[str, Any]:
        """
        Benchmark a feature engineering pipeline.

        Args:
            data: Input data for feature computation
            feature_funcs: List of feature computation functions
            func_args: List of additional positional arguments for each function
            func_kwargs: List of additional keyword arguments for each function
            n_runs: Number of benchmark runs
            use_cache: Whether to use caching
            use_parallel: Whether to use parallel computation

        Returns:
            Dictionary with benchmark results
        """
        if func_args is None:
            func_args = [() for _ in feature_funcs]
        if func_kwargs is None:
            func_kwargs = [{} for _ in feature_funcs]
        func_names = [func.__name__ for func in feature_funcs]
        if len(set(func_names)) != len(func_names):
            func_names = [f'{func.__name__}_{i}' for i, func in enumerate(
                feature_funcs)]
        results = {'overall': {'total_times': [], 'avg_time': 0, 'min_time':
            0, 'max_time': 0}, 'per_feature': {name: {'times': []} for name in
            func_names}, 'timestamp': datetime.now().isoformat(), 'n_runs':
            n_runs, 'use_cache': use_cache, 'use_parallel': use_parallel,
            'n_features': len(feature_funcs), 'data_shape': data.shape}
        for run in range(n_runs):
            logger.info(f'Running benchmark iteration {run + 1}/{n_runs}')
            if not use_cache:
                self._clean_cache()
            if use_parallel:
                _, metadata = self.parallel_feature_computation(data,
                    feature_funcs, func_args, func_kwargs, use_cache,
                    force_recompute=run == 0)
            else:
                start_time = time.time()
                for i, (func_name, func, args, kwargs) in enumerate(zip(
                    func_names, feature_funcs, func_args, func_kwargs)):
                    feature_start = time.time()
                    try:
                        if use_cache:
                            _, feat_metadata = self.cached_feature_computation(
                                data, func, args, kwargs, force_recompute=
                                run == 0)
                            compute_time = feat_metadata['total_time']
                        else:
                            if args and kwargs:
                                _ = func(data, *args, **kwargs)
                            elif args:
                                _ = func(data, *args)
                            elif kwargs:
                                _ = func(data, **kwargs)
                            else:
                                _ = func(data)
                            compute_time = time.time() - feature_start
                        results['per_feature'][func_name]['times'].append(
                            compute_time)
                    except Exception as e:
                        logger.error(
                            f'Error computing feature {func_name}: {str(e)}')
                        results['per_feature'][func_name]['times'].append(float
                            ('nan'))
                total_time = time.time() - start_time
                metadata = {'total_time': total_time}
            results['overall']['total_times'].append(metadata['total_time'])
            if 'computation_times' in metadata:
                for func_name, time_taken in metadata['computation_times'
                    ].items():
                    results['per_feature'][func_name]['times'].append(
                        time_taken)
        results['overall']['avg_time'] = np.mean(results['overall'][
            'total_times'])
        results['overall']['min_time'] = np.min(results['overall'][
            'total_times'])
        results['overall']['max_time'] = np.max(results['overall'][
            'total_times'])
        results['overall']['std_time'] = np.std(results['overall'][
            'total_times'])
        for func_name in func_names:
            times = results['per_feature'][func_name]['times']
            if times:
                results['per_feature'][func_name]['avg_time'] = np.mean(times)
                results['per_feature'][func_name]['min_time'] = np.min(times)
                results['per_feature'][func_name]['max_time'] = np.max(times)
                results['per_feature'][func_name]['std_time'] = np.std(times)
                results['per_feature'][func_name]['pct_of_total'] = results[
                    'per_feature'][func_name]['avg_time'] / results['overall'][
                    'avg_time'] * 100 if results['overall']['avg_time'
                    ] > 0 else 0
        logger.info(
            f"Benchmark completed. Average time: {results['overall']['avg_time']:.4f} seconds"
            )
        return results
