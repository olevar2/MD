"""
Tests for repository caching integration.
"""
import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, List, Optional, Any

from common_lib.cqrs.repositories import ReadRepository
from common_lib.caching.decorators import cached
from common_lib.caching.cache import Cache


class TestEntity:
    """
    Test entity for repository tests.
    """
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name


class TestReadRepository(ReadRepository):
    """
    Test read repository for testing caching.
    """
    
    def __init__(self, cache: Cache):
        self.entities: Dict[str, TestEntity] = {}
        self.cache = cache
        self.get_by_id_called = 0
        self.get_all_called = 0
        self.get_by_criteria_called = 0
    
    # Don't use the decorator directly in the class definition
    # We'll apply it in the test methods
    async def get_by_id(self, id: str) -> Optional[TestEntity]:
        """
        Get an entity by ID.
        
        Args:
            id: The ID of the entity
            
        Returns:
            The entity or None if not found
        """
        self.get_by_id_called += 1
        return self.entities.get(id)
    
    async def get_all(self) -> List[TestEntity]:
        """
        Get all entities.
        
        Returns:
            A list of all entities
        """
        self.get_all_called += 1
        return list(self.entities.values())
    
    async def get_by_criteria(self, criteria: Dict[str, Any]) -> List[TestEntity]:
        """
        Get entities by criteria.
        
        Args:
            criteria: The criteria to filter by
            
        Returns:
            A list of entities matching the criteria
        """
        self.get_by_criteria_called += 1
        
        filtered_entities = []
        for entity in self.entities.values():
            match = True
            for key, value in criteria.items():
                if hasattr(entity, key):
                    if getattr(entity, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                filtered_entities.append(entity)
        
        return filtered_entities


class TestRepositoryCaching(unittest.TestCase):
    """
    Tests for repository caching integration.
    """
    
    def setUp(self):
        """Set up the test case."""
        self.cache = MagicMock(spec=Cache)
        # Configure the mock to return awaitable values
        self.cache.get = AsyncMock()
        self.cache.set = AsyncMock()
        self.cache.delete = AsyncMock()
        self.cache.exists = AsyncMock()
        
        self.repository = TestReadRepository(self.cache)
        
        # Add some test entities
        self.repository.entities = {
            "1": TestEntity("1", "Entity 1"),
            "2": TestEntity("2", "Entity 2"),
            "3": TestEntity("3", "Entity 3")
        }
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Tear down the test case."""
        self.loop.close()
    
    def test_get_by_id_cache_hit(self):
        """Test get_by_id with a cache hit."""
        # Set up the cache to return a value
        entity = TestEntity("1", "Entity 1")
        self.cache.get.return_value = entity
        
        # Apply the cached decorator to the method
        from common_lib.caching.decorators import cached
        self.repository.get_by_id = cached(self.cache, "test_entity", ttl=3600)(self.repository.get_by_id)
        
        # Call the method
        result = self.loop.run_until_complete(self.repository.get_by_id("1"))
        
        # Check that the cache was queried
        self.cache.get.assert_called_once()
        
        # Check that the method returned the cached value
        self.assertEqual(result, entity)
        
        # Check that the repository method was not called (since the value was in the cache)
        self.assertEqual(self.repository.get_by_id_called, 0)
    
    def test_get_by_id_cache_miss(self):
        """Test get_by_id with a cache miss."""
        # Set up the cache to return None (cache miss)
        self.cache.get.return_value = None
        
        # Apply the cached decorator to the method
        from common_lib.caching.decorators import cached
        self.repository.get_by_id = cached(self.cache, "test_entity", ttl=3600)(self.repository.get_by_id)
        
        # Call the method
        result = self.loop.run_until_complete(self.repository.get_by_id("1"))
        
        # Check that the cache was queried
        self.cache.get.assert_called_once()
        
        # Check that the method returned the correct value
        self.assertEqual(result.id, "1")
        self.assertEqual(result.name, "Entity 1")
        
        # Check that the repository method was called
        self.assertEqual(self.repository.get_by_id_called, 1)
        
        # Check that the result was cached
        self.cache.set.assert_called_once()
    
    def test_get_all_cache_hit(self):
        """Test get_all with a cache hit."""
        # Set up the cache to return a value
        entities = list(self.repository.entities.values())
        self.cache.get.return_value = entities
        
        # Apply the cached decorator to the method
        from common_lib.caching.decorators import cached
        self.repository.get_all = cached(self.cache, "get_all", ttl=1800)(self.repository.get_all)
        
        # Call the method
        result = self.loop.run_until_complete(self.repository.get_all())
        
        # Check that the cache was queried
        self.cache.get.assert_called_once()
        
        # Check that the method returned the cached value
        self.assertEqual(result, entities)
        
        # Check that the repository method was not called (since the value was in the cache)
        self.assertEqual(self.repository.get_all_called, 0)
    
    def test_get_all_cache_miss(self):
        """Test get_all with a cache miss."""
        # Set up the cache to return None (cache miss)
        self.cache.get.return_value = None
        
        # Apply the cached decorator to the method
        from common_lib.caching.decorators import cached
        self.repository.get_all = cached(self.cache, "get_all", ttl=1800)(self.repository.get_all)
        
        # Call the method
        result = self.loop.run_until_complete(self.repository.get_all())
        
        # Check that the cache was queried
        self.cache.get.assert_called_once()
        
        # Check that the method returned the correct value
        self.assertEqual(len(result), 3)
        
        # Check that the repository method was called
        self.assertEqual(self.repository.get_all_called, 1)
        
        # Check that the result was cached
        self.cache.set.assert_called_once()
    
    def test_get_by_criteria_cache_hit(self):
        """Test get_by_criteria with a cache hit."""
        # Set up the cache to return a value
        entities = [TestEntity("1", "Entity 1")]
        self.cache.get.return_value = entities
        
        # Apply the cached decorator to the method
        from common_lib.caching.decorators import cached
        self.repository.get_by_criteria = cached(self.cache, "get_by_criteria", ttl=1800)(self.repository.get_by_criteria)
        
        # Call the method
        result = self.loop.run_until_complete(self.repository.get_by_criteria({"name": "Entity 1"}))
        
        # Check that the cache was queried
        self.cache.get.assert_called_once()
        
        # Check that the method returned the cached value
        self.assertEqual(result, entities)
        
        # Check that the repository method was not called (since the value was in the cache)
        self.assertEqual(self.repository.get_by_criteria_called, 0)
    
    def test_get_by_criteria_cache_miss(self):
        """Test get_by_criteria with a cache miss."""
        # Set up the cache to return None (cache miss)
        self.cache.get.return_value = None
        
        # Apply the cached decorator to the method
        from common_lib.caching.decorators import cached
        self.repository.get_by_criteria = cached(self.cache, "get_by_criteria", ttl=1800)(self.repository.get_by_criteria)
        
        # Call the method
        result = self.loop.run_until_complete(self.repository.get_by_criteria({"name": "Entity 1"}))
        
        # Check that the cache was queried
        self.cache.get.assert_called_once()
        
        # Check that the method returned the correct value
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "1")
        self.assertEqual(result[0].name, "Entity 1")
        
        # Check that the repository method was called
        self.assertEqual(self.repository.get_by_criteria_called, 1)
        
        # Check that the result was cached
        self.cache.set.assert_called_once()


if __name__ == '__main__':
    unittest.main()