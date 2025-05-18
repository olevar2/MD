"""
Optimization read repository.

This module provides the read repository for strategy optimizations.
"""
import logging
from backtesting_service.utils.cache_factory import cache_factory
from common_lib.caching.decorators import cached
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from common_lib.cqrs.repositories import ReadRepository
from backtesting_service.models.backtest_models import OptimizationResult
logger = logging.getLogger(__name__)


class OptimizationReadRepository(ReadRepository):
    """
    Read repository for optimizations.
    """

    def __init__(self, storage_path: Optional[str]=None):
        """
        Initialize the repository with a storage path.
        
        Args:
            storage_path: Path to the storage directory
        """
        self.storage_path = storage_path or os.environ.get(
            'OPTIMIZATION_STORAGE_PATH', './data/optimizations')
        os.makedirs(self.storage_path, exist_ok=True)
        self.optimizations: Dict[str, OptimizationResult] = {}
        self.cache = cache_factory.get_cache()

    @cached(cache_factory.get_cache(), 'optimization', ttl=3600)
    async def get_by_id(self, id: str) ->Optional[OptimizationResult]:
        """
        Get an optimization by ID.
        
        Args:
            id: Optimization ID
            
        Returns:
            The optimization or None if not found
        """
        logger.info(f'Getting optimization with ID: {id}')
        if id in self.optimizations:
            logger.debug(f'Found optimization {id} in memory cache')
            return self.optimizations[id]
        file_path = os.path.join(self.storage_path, f'{id}.json')
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    optimization_data = json.load(f)
                optimization_result = OptimizationResult(**optimization_data)
                self.optimizations[id] = optimization_result
                logger.debug(f'Loaded optimization {id} from file storage')
                return optimization_result
            except Exception as e:
                logger.error(f'Error loading optimization {id} from file: {e}')
        logger.warning(f'Optimization {id} not found')
        return None

    @cached(cache_factory.get_cache(), 'optimization_all', ttl=3600)
    async def get_all(self) ->List[OptimizationResult]:
        """
        Get all optimizations.
        
        Returns:
            List of all optimizations
        """
        logger.info('Getting all optimizations')
        optimizations = []
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                try:
                    optimization_id = filename[:-5]
                    optimization = await self.get_by_id(optimization_id)
                    if optimization:
                        optimizations.append(optimization)
                except Exception as e:
                    logger.error(
                        f'Error loading optimization from {filename}: {e}')
        return optimizations

    @cached(cache_factory.get_cache(), 'optimization_criteria', ttl=3600)
    async def get_by_criteria(self, criteria: Dict[str, Any]) ->List[
        OptimizationResult]:
        """
        Get optimizations by criteria.
        
        Args:
            criteria: Criteria to filter by
            
        Returns:
            List of optimizations matching the criteria
        """
        logger.info(f'Getting optimizations with criteria: {criteria}')
        all_optimizations = await self.get_all()
        filtered_optimizations = []
        for optimization in all_optimizations:
            match = True
            for key, value in criteria.items():
                if hasattr(optimization, key):
                    if getattr(optimization, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            if match:
                filtered_optimizations.append(optimization)
        return filtered_optimizations
