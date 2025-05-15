"""
Optimization write repository.

This module provides the write repository for strategy optimizations.
"""
import logging
import json
import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from common_lib.cqrs.repositories import WriteRepository
from backtesting_service.models.backtest_models import OptimizationResult

logger = logging.getLogger(__name__)


class OptimizationWriteRepository(WriteRepository):
    """
    Write repository for optimizations.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the repository with a storage path.
        
        Args:
            storage_path: Path to the storage directory
        """
        self.storage_path = storage_path or os.environ.get("OPTIMIZATION_STORAGE_PATH", "./data/optimizations")
        os.makedirs(self.storage_path, exist_ok=True)
        self.optimizations: Dict[str, OptimizationResult] = {}
        self.lock = asyncio.Lock()
    
    async def add(self, entity: OptimizationResult) -> str:
        """
        Add an optimization.
        
        Args:
            entity: Optimization to add
            
        Returns:
            ID of the added optimization
        """
        logger.info(f"Adding optimization with ID: {entity.optimization_id}")
        
        async with self.lock:
            # Store in memory
            self.optimizations[entity.optimization_id] = entity
            
            # Store in file
            file_path = os.path.join(self.storage_path, f"{entity.optimization_id}.json")
            try:
                with open(file_path, "w") as f:
                    json.dump(entity.model_dump(), f, default=str)
                
                logger.debug(f"Saved optimization {entity.optimization_id} to file storage")
            except Exception as e:
                logger.error(f"Error saving optimization to file: {e}")
        
        return entity.optimization_id
    
    async def update(self, entity: OptimizationResult) -> None:
        """
        Update an optimization.
        
        Args:
            entity: Optimization to update
        """
        logger.info(f"Updating optimization with ID: {entity.optimization_id}")
        
        async with self.lock:
            # Update in memory
            self.optimizations[entity.optimization_id] = entity
            
            # Update in file
            file_path = os.path.join(self.storage_path, f"{entity.optimization_id}.json")
            try:
                with open(file_path, "w") as f:
                    json.dump(entity.model_dump(), f, default=str)
                
                logger.debug(f"Updated optimization {entity.optimization_id} in file storage")
            except Exception as e:
                logger.error(f"Error updating optimization in file: {e}")
    
    async def delete(self, id: str) -> None:
        """
        Delete an optimization by ID.
        
        Args:
            id: ID of the optimization to delete
        """
        logger.info(f"Deleting optimization with ID: {id}")
        
        async with self.lock:
            # Remove from memory
            if id in self.optimizations:
                del self.optimizations[id]
            
            # Remove from file
            file_path = os.path.join(self.storage_path, f"{id}.json")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Deleted optimization {id} from file storage")
                except Exception as e:
                    logger.error(f"Error deleting optimization from file: {e}")
    
    async def add_batch(self, entities: List[OptimizationResult]) -> List[str]:
        """
        Add multiple optimizations in a batch.
        
        Args:
            entities: Optimizations to add
            
        Returns:
            IDs of the added optimizations
        """
        logger.info(f"Adding {len(entities)} optimizations in batch")
        
        ids = []
        for entity in entities:
            id = await self.add(entity)
            ids.append(id)
        
        return ids