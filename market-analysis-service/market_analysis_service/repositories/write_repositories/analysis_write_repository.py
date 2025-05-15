"""
Analysis write repository.

This module provides the write repository for market analysis results.
"""
import logging
import json
import os
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from common_lib.cqrs.repositories import WriteRepository
from market_analysis_service.models.market_analysis_models import (
    MarketAnalysisResponse,
    PatternRecognitionResponse,
    SupportResistanceResponse,
    MarketRegimeResponse,
    CorrelationAnalysisResponse,
    AnalysisType
)

logger = logging.getLogger(__name__)


class AnalysisWriteRepository(WriteRepository):
    """
    Write repository for market analysis results.
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
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    async def add(self, entity: Any) -> str:
        """
        Add an analysis result.
        
        Args:
            entity: Analysis result to add
            
        Returns:
            ID of the added analysis result
        """
        logger.info(f"Adding analysis result")
        
        async with self.lock:
            # Determine the type of entity and store accordingly
            if isinstance(entity, MarketAnalysisResponse):
                return await self._add_to_cache_and_file(entity, entity.request_id, self.analysis_cache, self.analysis_dir)
            elif isinstance(entity, PatternRecognitionResponse):
                return await self._add_to_cache_and_file(entity, entity.request_id, self.pattern_cache, self.pattern_dir)
            elif isinstance(entity, SupportResistanceResponse):
                return await self._add_to_cache_and_file(entity, entity.request_id, self.support_resistance_cache, self.support_resistance_dir)
            elif isinstance(entity, MarketRegimeResponse):
                return await self._add_to_cache_and_file(entity, entity.request_id, self.market_regime_cache, self.market_regime_dir)
            elif isinstance(entity, CorrelationAnalysisResponse):
                return await self._add_to_cache_and_file(entity, entity.request_id, self.correlation_cache, self.correlation_dir)
            else:
                raise ValueError(f"Unsupported entity type: {type(entity)}")
    
    async def update(self, entity: Any) -> None:
        """
        Update an analysis result.
        
        Args:
            entity: Analysis result to update
        """
        logger.info(f"Updating analysis result")
        
        # For simplicity, just add the entity again (overwrite)
        await self.add(entity)
    
    async def delete(self, id: str) -> None:
        """
        Delete an analysis result by ID.
        
        Args:
            id: ID of the analysis result to delete
        """
        logger.info(f"Deleting analysis result with ID: {id}")
        
        async with self.lock:
            # Try to delete from all caches and directories
            await self._delete_from_cache_and_file(id, self.analysis_cache, self.analysis_dir)
            await self._delete_from_cache_and_file(id, self.pattern_cache, self.pattern_dir)
            await self._delete_from_cache_and_file(id, self.support_resistance_cache, self.support_resistance_dir)
            await self._delete_from_cache_and_file(id, self.market_regime_cache, self.market_regime_dir)
            await self._delete_from_cache_and_file(id, self.correlation_cache, self.correlation_dir)
    
    async def add_batch(self, entities: List[Any]) -> List[str]:
        """
        Add multiple analysis results in a batch.
        
        Args:
            entities: Analysis results to add
            
        Returns:
            IDs of the added analysis results
        """
        logger.info(f"Adding {len(entities)} analysis results in batch")
        
        ids = []
        for entity in entities:
            id = await self.add(entity)
            ids.append(id)
        
        return ids
    
    async def _add_to_cache_and_file(
        self,
        entity: Any,
        id: str,
        cache: Dict[str, Any],
        directory: str
    ) -> str:
        """
        Add an analysis result to cache and file.
        
        Args:
            entity: Analysis result to add
            id: ID of the analysis result
            cache: Cache dictionary
            directory: Directory to store in
            
        Returns:
            ID of the added analysis result
        """
        # Add to cache
        cache[id] = entity
        
        # Add to file
        file_path = os.path.join(directory, f"{id}.json")
        try:
            with open(file_path, "w") as f:
                json.dump(entity.model_dump(), f, default=str)
            
            logger.debug(f"Saved result {id} to file")
        except Exception as e:
            logger.error(f"Error saving result to file: {e}")
        
        return id
    
    async def _delete_from_cache_and_file(
        self,
        id: str,
        cache: Dict[str, Any],
        directory: str
    ) -> None:
        """
        Delete an analysis result from cache and file.
        
        Args:
            id: ID of the analysis result
            cache: Cache dictionary
            directory: Directory to delete from
        """
        # Delete from cache
        if id in cache:
            del cache[id]
        
        # Delete from file
        file_path = os.path.join(directory, f"{id}.json")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Deleted result {id} from file")
            except Exception as e:
                logger.error(f"Error deleting result from file: {e}")