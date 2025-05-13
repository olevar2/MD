"""
Feature Vector Caching Module for ML Integration Service.

This module provides caching for ML feature vectors with dependency-based invalidation.
"""

import logging
import time
import hashlib
import json
import pickle
from typing import Dict, Any, Optional, Union, List, Tuple, Set
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Import common library
from common_lib.caching.cache_service import get_cache_service, cache_result
from common_lib.caching.dependency_invalidation import DependencyInvalidationStrategy

# Configure logging
logger = logging.getLogger(__name__)

class FeatureVectorCache:
    """
    Cache for ML feature vectors with dependency-based invalidation.
    
    This class provides caching for ML feature vectors, with appropriate
    invalidation based on dependencies between features.
    """
    
    def __init__(
        self,
        service_name: str = "ml-integration-service",
        default_ttl: int = 3600,  # 1 hour
        enable_metrics: bool = True
    ):
        """
        Initialize the feature vector cache.
        
        Args:
            service_name: Name of the service
            default_ttl: Default time-to-live for cached items (in seconds)
            enable_metrics: Whether to enable metrics collection
        """
        self.service_name = service_name
        self.default_ttl = default_ttl
        self.enable_metrics = enable_metrics
        
        # Get cache service
        self.cache_service = get_cache_service(
            service_name=service_name,
            default_ttl=default_ttl
        )
        
        # Create invalidation strategy
        self.invalidation_strategy = DependencyInvalidationStrategy(
            cache_service=self.cache_service
        )
        
        # Feature dependencies
        self.feature_dependencies: Dict[str, Set[str]] = {}
        
        logger.info(f"Feature vector cache initialized with TTL {default_ttl}s")
    
    def get_feature_vector(
        self,
        model_name: str,
        symbol: str,
        timeframe: str,
        features: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get a cached feature vector.
        
        Args:
            model_name: Name of the model
            symbol: Symbol
            timeframe: Timeframe
            features: List of features
            start_time: Start time
            end_time: End time
            
        Returns:
            Cached feature vector or None if not found
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            model_name=model_name,
            symbol=symbol,
            timeframe=timeframe,
            features=features,
            start_time=start_time,
            end_time=end_time
        )
        
        # Get from cache
        cached_value = self.cache_service.get(cache_key)
        
        if cached_value is not None:
            # Return the cached value
            return cached_value.get("data")
        
        return None
    
    def set_feature_vector(
        self,
        model_name: str,
        symbol: str,
        timeframe: str,
        features: List[str],
        data: pd.DataFrame,
        dependencies: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a feature vector.
        
        Args:
            model_name: Name of the model
            symbol: Symbol
            timeframe: Timeframe
            features: List of features
            data: Feature vector data
            dependencies: List of dependencies
            start_time: Start time
            end_time: End time
            ttl: Time-to-live (in seconds)
            
        Returns:
            True if successful, False otherwise
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            model_name=model_name,
            symbol=symbol,
            timeframe=timeframe,
            features=features,
            start_time=start_time,
            end_time=end_time
        )
        
        # Create cache value
        cache_value = {
            "model_name": model_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "features": features,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Set in cache
        success = self.cache_service.set(cache_key, cache_value, ttl=ttl or self.default_ttl)
        
        # Register dependencies
        if success and dependencies:
            for dependency in dependencies:
                self.invalidation_strategy.register_dependency(cache_key, dependency)
                
                # Store feature dependencies
                if model_name not in self.feature_dependencies:
                    self.feature_dependencies[model_name] = set()
                self.feature_dependencies[model_name].add(dependency)
        
        return success
    
    def invalidate_feature_vector(
        self,
        model_name: str,
        symbol: str,
        timeframe: str,
        features: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """
        Invalidate a cached feature vector.
        
        Args:
            model_name: Name of the model
            symbol: Symbol
            timeframe: Timeframe
            features: List of features
            start_time: Start time
            end_time: End time
            
        Returns:
            True if successful, False otherwise
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            model_name=model_name,
            symbol=symbol,
            timeframe=timeframe,
            features=features,
            start_time=start_time,
            end_time=end_time
        )
        
        # Invalidate cache entry
        return self.invalidation_strategy.invalidate(cache_key)
    
    def invalidate_feature(self, feature: str) -> bool:
        """
        Invalidate all cached feature vectors that depend on a feature.
        
        Args:
            feature: Feature name
            
        Returns:
            True if successful, False otherwise
        """
        # Invalidate cache entry
        return self.invalidation_strategy.invalidate(feature)
    
    def invalidate_model(self, model_name: str) -> bool:
        """
        Invalidate all cached feature vectors for a model.
        
        Args:
            model_name: Model name
            
        Returns:
            True if successful, False otherwise
        """
        # Generate cache key pattern
        cache_key_pattern = f"feature_vector:{model_name}:*"
        
        # Clear from cache
        return self.cache_service.clear(cache_key_pattern)
    
    def _generate_cache_key(
        self,
        model_name: str,
        symbol: str,
        timeframe: str,
        features: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """
        Generate a cache key for a feature vector.
        
        Args:
            model_name: Name of the model
            symbol: Symbol
            timeframe: Timeframe
            features: List of features
            start_time: Start time
            end_time: End time
            
        Returns:
            Cache key
        """
        # Sort features
        sorted_features = sorted(features)
        
        # Convert features to a string
        features_str = json.dumps(sorted_features, sort_keys=True)
        
        # Convert start and end times to strings
        start_time_str = start_time.isoformat() if start_time else "none"
        end_time_str = end_time.isoformat() if end_time else "none"
        
        # Generate a hash
        key_str = f"feature_vector:{model_name}:{symbol}:{timeframe}:{features_str}:{start_time_str}:{end_time_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

# Create a decorator for caching feature vector calculations
def cache_feature_vector(
    ttl: Optional[int] = None,
    dependencies: Optional[List[str]] = None
):
    """
    Decorator for caching feature vector calculations.
    
    Args:
        ttl: Time-to-live for the cached item (in seconds)
        dependencies: List of dependencies
        
    Returns:
        Decorated function
    """
    def decorator(func):
    """
    Decorator.
    
    Args:
        func: Description of func
    
    """

        def wrapper(
            self,
            model_name: str,
            symbol: str,
            timeframe: str,
            features: List[str],
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            **kwargs
        ):
    """
    Wrapper.
    
    Args:
        model_name: Description of model_name
        symbol: Description of symbol
        timeframe: Description of timeframe
        features: Description of features
        start_time: Description of start_time
        end_time: Description of end_time
        kwargs: Description of kwargs
    
    """

            # Get feature vector cache
            feature_vector_cache = getattr(self, "_feature_vector_cache", None)
            if feature_vector_cache is None:
                feature_vector_cache = FeatureVectorCache()
                setattr(self, "_feature_vector_cache", feature_vector_cache)
            
            # Try to get from cache
            cached_result = feature_vector_cache.get_feature_vector(
                model_name=model_name,
                symbol=symbol,
                timeframe=timeframe,
                features=features,
                start_time=start_time,
                end_time=end_time
            )
            
            if cached_result is not None:
                return cached_result
            
            # Calculate feature vector
            result = func(
                self,
                model_name=model_name,
                symbol=symbol,
                timeframe=timeframe,
                features=features,
                start_time=start_time,
                end_time=end_time,
                **kwargs
            )
            
            # Cache result
            feature_vector_cache.set_feature_vector(
                model_name=model_name,
                symbol=symbol,
                timeframe=timeframe,
                features=features,
                data=result,
                dependencies=dependencies,
                start_time=start_time,
                end_time=end_time,
                ttl=ttl
            )
            
            return result
        
        return wrapper
    
    return decorator
