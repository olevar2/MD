"""
Analysis read repository.

This module provides the read repository for market analysis results.
"""
import logging
from market_analysis_service.utils.cache_factory import cache_factory
from common_lib.caching.decorators import cached
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from common_lib.cqrs.repositories import ReadRepository
from market_analysis_service.models.market_analysis_models import (
    MarketAnalysisResponse,
    PatternRecognitionResponse,
    SupportResistanceResponse,
    MarketRegimeResponse,
    CorrelationAnalysisResponse,
    AnalysisType
)

logger = logging.getLogger(__name__)


class AnalysisReadRepository(ReadRepository):
    """
    Read repository for market analysis results.
    """
    
    def __init__(self, data_dir: str = "/data/market-analysis"):
        """
        Initialize the repository with a storage path.
        
        Args:
            data_dir: Path to the storage directory
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create subdirectories for different analysis types
        self.analysis_dir = os.path.join(self.data_dir, "analysis")
        self.pattern_dir = os.path.join(self.data_dir, "patterns")
        self.support_resistance_dir = os.path.join(self.data_dir, "support_resistance")
        self.market_regime_dir = os.path.join(self.data_dir, "market_regime")
        self.correlation_dir = os.path.join(self.data_dir, "correlation")
        
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.pattern_dir, exist_ok=True)
        os.makedirs(self.support_resistance_dir, exist_ok=True)
        os.makedirs(self.market_regime_dir, exist_ok=True)
        os.makedirs(self.correlation_dir, exist_ok=True)
        
        # In-memory cache
        self.analysis_cache: Dict[str, MarketAnalysisResponse] = {}
        self.pattern_cache: Dict[str, PatternRecognitionResponse] = {}
        self.support_resistance_cache: Dict[str, SupportResistanceResponse] = {}
        self.market_regime_cache: Dict[str, MarketRegimeResponse] = {}
        self.correlation_cache: Dict[str, CorrelationAnalysisResponse] = {}
        self.cache = cache_factory.get_cache()
    
    @cached(cache_factory.get_cache(), "analysis", ttl=3600)
    async def get_by_id(self, id: str) -> Optional[Any]:
        """
        Get an analysis result by ID.
        
        Args:
            id: Analysis result ID
            
        Returns:
            The analysis result or None if not found
        """
        logger.info(f"Getting analysis result with ID: {id}")
        
        # Try to find in all caches and directories
        result = await self._get_from_cache_or_file(id, self.analysis_cache, self.analysis_dir, MarketAnalysisResponse)
        if result:
            return result
        
        result = await self._get_from_cache_or_file(id, self.pattern_cache, self.pattern_dir, PatternRecognitionResponse)
        if result:
            return result
        
        result = await self._get_from_cache_or_file(id, self.support_resistance_cache, self.support_resistance_dir, SupportResistanceResponse)
        if result:
            return result
        
        result = await self._get_from_cache_or_file(id, self.market_regime_cache, self.market_regime_dir, MarketRegimeResponse)
        if result:
            return result
        
        result = await self._get_from_cache_or_file(id, self.correlation_cache, self.correlation_dir, CorrelationAnalysisResponse)
        if result:
            return result
        
        logger.warning(f"Analysis result {id} not found")
        return None
    
    async def get_all(self) -> List[Any]:
        """
        Get all analysis results.
        
        Returns:
            List of all analysis results
        """
        logger.info("Getting all analysis results")
        
        results = []
        
        # Get from all directories
        results.extend(await self._get_all_from_directory(self.analysis_dir, MarketAnalysisResponse))
        results.extend(await self._get_all_from_directory(self.pattern_dir, PatternRecognitionResponse))
        results.extend(await self._get_all_from_directory(self.support_resistance_dir, SupportResistanceResponse))
        results.extend(await self._get_all_from_directory(self.market_regime_dir, MarketRegimeResponse))
        results.extend(await self._get_all_from_directory(self.correlation_dir, CorrelationAnalysisResponse))
        
        return results
    
    async def get_by_criteria(self, criteria: Dict[str, Any]) -> List[Any]:
        """
        Get analysis results by criteria.
        
        Args:
            criteria: Criteria to filter by
            
        Returns:
            List of analysis results matching the criteria
        """
        logger.info(f"Getting analysis results with criteria: {criteria}")
        
        # Get all results
        all_results = await self.get_all()
        
        # Filter by criteria
        filtered_results = []
        for result in all_results:
            match = True
            for key, value in criteria.items():
                if hasattr(result, key):
                    if getattr(result, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                filtered_results.append(result)
        
        return filtered_results
    
    async def _get_from_cache_or_file(
        self,
        id: str,
        cache: Dict[str, Any],
        directory: str,
        model_class: Any
    ) -> Optional[Any]:
        """
        Get an analysis result from cache or file.
        
        Args:
            id: Analysis result ID
            cache: Cache dictionary
            directory: Directory to look in
            model_class: Model class to deserialize to
            
        Returns:
            The analysis result or None if not found
        """
        # Check cache first
        if id in cache:
            logger.debug(f"Found result {id} in cache")
            return cache[id]
        
        # Check file
        file_path = os.path.join(directory, f"{id}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                # Convert to model
                result = model_class(**data)
                
                # Cache result
                cache[id] = result
                
                logger.debug(f"Loaded result {id} from file")
                return result
            except Exception as e:
                logger.error(f"Error loading result {id} from file: {e}")
        
        return None
    
    async def _get_all_from_directory(
        self,
        directory: str,
        model_class: Any
    ) -> List[Any]:
        """
        Get all analysis results from a directory.
        
        Args:
            directory: Directory to look in
            model_class: Model class to deserialize to
            
        Returns:
            List of analysis results
        """
        results = []
        
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                try:
                    result_id = filename[:-5]  # Remove .json extension
                    result = await self._get_from_cache_or_file(result_id, {}, directory, model_class)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error loading result from {filename}: {e}")
        
        return results