"""
Repository interfaces for CQRS pattern.

This module provides the base interfaces for read and write repositories
in the CQRS pattern.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Any, Dict

# Type variables for generic typing
TEntity = TypeVar('TEntity')
TId = TypeVar('TId')

class ReadRepository(Generic[TEntity, TId], ABC):
    """Base interface for read repositories."""
    
    @abstractmethod
    async def get_by_id(self, id: TId) -> Optional[TEntity]:
        """
        Get an entity by ID.
        
        Args:
            id: The ID of the entity
            
        Returns:
            The entity or None if not found
        """
        pass
    
    @abstractmethod
    async def get_all(self) -> List[TEntity]:
        """
        Get all entities.
        
        Returns:
            A list of all entities
        """
        pass
    
    @abstractmethod
    async def get_by_criteria(self, criteria: Dict[str, Any]) -> List[TEntity]:
        """
        Get entities by criteria.
        
        Args:
            criteria: The criteria to filter by
            
        Returns:
            A list of entities matching the criteria
        """
        pass


class WriteRepository(Generic[TEntity, TId], ABC):
    """Base interface for write repositories."""
    
    @abstractmethod
    async def add(self, entity: TEntity) -> TId:
        """
        Add an entity.
        
        Args:
            entity: The entity to add
            
        Returns:
            The ID of the added entity
        """
        pass
    
    @abstractmethod
    async def update(self, entity: TEntity) -> None:
        """
        Update an entity.
        
        Args:
            entity: The entity to update
        """
        pass
    
    @abstractmethod
    async def delete(self, id: TId) -> None:
        """
        Delete an entity by ID.
        
        Args:
            id: The ID of the entity to delete
        """
        pass
    
    @abstractmethod
    async def add_batch(self, entities: List[TEntity]) -> List[TId]:
        """
        Add multiple entities in a batch.
        
        Args:
            entities: The entities to add
            
        Returns:
            The IDs of the added entities
        """
        pass