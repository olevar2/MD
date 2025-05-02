"""
Core Feature Store service coordinating feature calculations and storage with performance monitoring.
"""

import logging
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

from feature_store_service.monitoring.performance_monitoring import FeatureStoreMonitoring
from feature_store_service.optimization.resource_manager import AdaptiveResourceManager
from feature_store_service.validation.data_validator import DataValidationService
from feature_store_service.error.error_manager import IndicatorErrorManager
from feature_store_service.error.recovery_service import ErrorRecoveryService

logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Core Feature Store service coordinating feature calculations and storage.
    Integrates performance monitoring for critical operations.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        monitoring_dir: str = "monitoring/feature_store"
    ):
        self.config = config or {}
        
        # Initialize monitoring first
        self.monitoring = FeatureStoreMonitoring(base_dir=monitoring_dir)
        
        # Initialize core components
        self.resource_manager = AdaptiveResourceManager(
            cache_dir=self.config.get('cache_dir', 'cache')
        )
        self.validator = DataValidationService()
        self.error_manager = IndicatorErrorManager()
        self.recovery_service = ErrorRecoveryService()
        
        # Service state
        self.is_running = False
        self._health_check_task = None
        
    async def start(self) -> None:
        """Start the feature store service."""
        if self.is_running:
            return
            
        self.is_running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Feature Store service started")
        
    async def stop(self) -> None:
        """Stop the feature store service."""
        self.is_running = False
        if self._health_check_task:
            self._health_check_task.cancel()
        logger.info("Feature Store service stopped")
        
    async def calculate_features(
        self,
        symbol: str,
        features: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate features with performance tracking.
        
        Args:
            symbol: The market symbol
            features: List of features to calculate
            data: Market data for calculations
            
        Returns:
            Calculated feature values
        """
        @self.monitoring.track_feature_calculation
        async def _calculate():
            # Validate inputs
            validation_result = await self.validator.validate_calculation_inputs(
                symbol=symbol,
                features=features,
                data=data
            )
            
            if not validation_result.is_valid:
                raise ValueError(f"Invalid inputs: {validation_result.errors}")
            
            # Try to get from cache first
            cache_key = f"{symbol}_{','.join(sorted(features))}"
            cached_result = await self._get_cached_features(cache_key, data)
            if cached_result is not None:
                return cached_result
            
            # Calculate features
            try:
                result = await self.resource_manager.submit_calculation(
                    calc_id=f"features_{symbol}_{datetime.utcnow().timestamp()}",
                    calc_func=self._compute_features,
                    args=(symbol, features, data),
                    cache_key=cache_key
                )
                
                # Cache the results
                await self._cache_features(cache_key, result)
                
                return result
                
            except Exception as e:
                self.error_manager.handle_error(e)
                raise
                
        return await _calculate()
        
    async def retrieve_data(
        self,
        symbol: str,
        features: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Retrieve feature data with performance tracking.
        
        Args:
            symbol: The market symbol
            features: List of features to retrieve
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            
        Returns:
            Retrieved feature data
        """
        @self.monitoring.track_data_retrieval
        async def _retrieve():
            # Try cache first
            cache_key = f"{symbol}_{','.join(sorted(features))}_{start_time}_{end_time}"
            cached_data = await self._get_cached_features(cache_key, None)
            if cached_data is not None:
                return cached_data
            
            # Retrieve from storage
            try:
                data = await self.resource_manager.retrieve_data(
                    symbol=symbol,
                    features=features,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Cache the results
                await self._cache_features(cache_key, data)
                
                return data
                
            except Exception as e:
                self.error_manager.handle_error(e)
                raise
                
        return await _retrieve()
        
    async def _get_cached_features(
        self,
        cache_key: str,
        data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get features from cache with performance tracking."""
        @self.monitoring.track_cache_operation
        async def _get_cache():
            return await self.resource_manager.get_from_cache(cache_key)
            
        return await _get_cache()
        
    async def _cache_features(
        self,
        cache_key: str,
        data: Dict[str, Any]
    ) -> None:
        """Cache feature data with performance tracking."""
        @self.monitoring.track_cache_operation
        async def _cache():
            await self.resource_manager.store_in_cache(cache_key, data)
            
        await _cache()
        
    async def _compute_features(
        self,
        symbol: str,
        features: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal feature computation implementation."""
        # This would contain the actual feature calculation logic
        # For now, we'll just return a placeholder
        return {
            'symbol': symbol,
            'features': {f: [] for f in features},
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _health_check_loop(self) -> None:
        """Monitor service health including performance metrics."""
        while self.is_running:
            try:
                # Get current health status
                health_status = self.monitoring.get_health_status()
                
                # Log any issues
                if not health_status['healthy']:
                    for issue in health_status['issues']:
                        logger.warning(f"Health check issue: {issue}")
                        
                # Get performance metrics for monitoring
                metrics = self.monitoring.get_metrics()
                logger.info(f"Performance metrics: {metrics}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)  # Back off on error
