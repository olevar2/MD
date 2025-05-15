"""
Backtest write repository.

This module provides the write repository for backtests.
"""
import logging
import json
import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from common_lib.cqrs.repositories import WriteRepository
from backtesting_service.models.backtest_models import BacktestResult, BacktestStatus

logger = logging.getLogger(__name__)


class BacktestWriteRepository(WriteRepository):
    """
    Write repository for backtests.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the repository with a storage path.
        
        Args:
            storage_path: Path to the storage directory
        """
        self.storage_path = storage_path or os.environ.get("BACKTEST_STORAGE_PATH", "./data/backtests")
        os.makedirs(self.storage_path, exist_ok=True)
        self.backtests: Dict[str, BacktestResult] = {}
        self.lock = asyncio.Lock()
    
    async def add(self, entity: BacktestResult) -> str:
        """
        Add a backtest.
        
        Args:
            entity: Backtest to add
            
        Returns:
            ID of the added backtest
        """
        logger.info(f"Adding backtest with ID: {entity.backtest_id}")
        
        async with self.lock:
            # Store in memory
            self.backtests[entity.backtest_id] = entity
            
            # Store in file
            file_path = os.path.join(self.storage_path, f"{entity.backtest_id}.json")
            try:
                with open(file_path, "w") as f:
                    json.dump(entity.model_dump(), f, default=str)
                
                logger.debug(f"Saved backtest {entity.backtest_id} to file storage")
            except Exception as e:
                logger.error(f"Error saving backtest to file: {e}")
        
        return entity.backtest_id
    
    async def update(self, entity: BacktestResult) -> None:
        """
        Update a backtest.
        
        Args:
            entity: Backtest to update
        """
        logger.info(f"Updating backtest with ID: {entity.backtest_id}")
        
        async with self.lock:
            # Update in memory
            self.backtests[entity.backtest_id] = entity
            
            # Update in file
            file_path = os.path.join(self.storage_path, f"{entity.backtest_id}.json")
            try:
                with open(file_path, "w") as f:
                    json.dump(entity.model_dump(), f, default=str)
                
                logger.debug(f"Updated backtest {entity.backtest_id} in file storage")
            except Exception as e:
                logger.error(f"Error updating backtest in file: {e}")
    
    async def delete(self, id: str) -> None:
        """
        Delete a backtest by ID.
        
        Args:
            id: ID of the backtest to delete
        """
        logger.info(f"Deleting backtest with ID: {id}")
        
        async with self.lock:
            # Remove from memory
            if id in self.backtests:
                del self.backtests[id]
            
            # Remove from file
            file_path = os.path.join(self.storage_path, f"{id}.json")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Deleted backtest {id} from file storage")
                except Exception as e:
                    logger.error(f"Error deleting backtest from file: {e}")
    
    async def add_batch(self, entities: List[BacktestResult]) -> List[str]:
        """
        Add multiple backtests in a batch.
        
        Args:
            entities: Backtests to add
            
        Returns:
            IDs of the added backtests
        """
        logger.info(f"Adding {len(entities)} backtests in batch")
        
        ids = []
        for entity in entities:
            id = await self.add(entity)
            ids.append(id)
        
        return ids