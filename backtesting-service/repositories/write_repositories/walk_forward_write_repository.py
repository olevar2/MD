"""
Walk-forward test write repository.

This module provides the write repository for walk-forward tests.
"""
import logging
import json
import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from common_lib.cqrs.repositories import WriteRepository
from backtesting_service.models.backtest_models import WalkForwardTestResult

logger = logging.getLogger(__name__)


class WalkForwardWriteRepository(WriteRepository):
    """
    Write repository for walk-forward tests.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the repository with a storage path.
        
        Args:
            storage_path: Path to the storage directory
        """
        self.storage_path = storage_path or os.environ.get("WALK_FORWARD_STORAGE_PATH", "./data/walk_forward_tests")
        os.makedirs(self.storage_path, exist_ok=True)
        self.tests: Dict[str, WalkForwardTestResult] = {}
        self.lock = asyncio.Lock()
    
    async def add(self, entity: WalkForwardTestResult) -> str:
        """
        Add a walk-forward test.
        
        Args:
            entity: Walk-forward test to add
            
        Returns:
            ID of the added walk-forward test
        """
        logger.info(f"Adding walk-forward test with ID: {entity.test_id}")
        
        async with self.lock:
            # Store in memory
            self.tests[entity.test_id] = entity
            
            # Store in file
            file_path = os.path.join(self.storage_path, f"{entity.test_id}.json")
            try:
                with open(file_path, "w") as f:
                    json.dump(entity.model_dump(), f, default=str)
                
                logger.debug(f"Saved walk-forward test {entity.test_id} to file storage")
            except Exception as e:
                logger.error(f"Error saving walk-forward test to file: {e}")
        
        return entity.test_id
    
    async def update(self, entity: WalkForwardTestResult) -> None:
        """
        Update a walk-forward test.
        
        Args:
            entity: Walk-forward test to update
        """
        logger.info(f"Updating walk-forward test with ID: {entity.test_id}")
        
        async with self.lock:
            # Update in memory
            self.tests[entity.test_id] = entity
            
            # Update in file
            file_path = os.path.join(self.storage_path, f"{entity.test_id}.json")
            try:
                with open(file_path, "w") as f:
                    json.dump(entity.model_dump(), f, default=str)
                
                logger.debug(f"Updated walk-forward test {entity.test_id} in file storage")
            except Exception as e:
                logger.error(f"Error updating walk-forward test in file: {e}")
    
    async def delete(self, id: str) -> None:
        """
        Delete a walk-forward test by ID.
        
        Args:
            id: ID of the walk-forward test to delete
        """
        logger.info(f"Deleting walk-forward test with ID: {id}")
        
        async with self.lock:
            # Remove from memory
            if id in self.tests:
                del self.tests[id]
            
            # Remove from file
            file_path = os.path.join(self.storage_path, f"{id}.json")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Deleted walk-forward test {id} from file storage")
                except Exception as e:
                    logger.error(f"Error deleting walk-forward test from file: {e}")
    
    async def add_batch(self, entities: List[WalkForwardTestResult]) -> List[str]:
        """
        Add multiple walk-forward tests in a batch.
        
        Args:
            entities: Walk-forward tests to add
            
        Returns:
            IDs of the added walk-forward tests
        """
        logger.info(f"Adding {len(entities)} walk-forward tests in batch")
        
        ids = []
        for entity in entities:
            id = await self.add(entity)
            ids.append(id)
        
        return ids