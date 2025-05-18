"""
Walk-forward test read repository.

This module provides the read repository for walk-forward tests.
"""
import logging
from backtesting_service.utils.cache_factory import cache_factory
from common_lib.caching.decorators import cached
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from common_lib.cqrs.repositories import ReadRepository
from backtesting_service.models.backtest_models import WalkForwardTestResult
logger = logging.getLogger(__name__)


class WalkForwardReadRepository(ReadRepository):
    """
    Read repository for walk-forward tests.
    """

    def __init__(self, storage_path: Optional[str]=None):
        """
        Initialize the repository with a storage path.
        
        Args:
            storage_path: Path to the storage directory
        """
        self.storage_path = storage_path or os.environ.get(
            'WALK_FORWARD_STORAGE_PATH', './data/walk_forward_tests')
        os.makedirs(self.storage_path, exist_ok=True)
        self.tests: Dict[str, WalkForwardTestResult] = {}
        self.cache = cache_factory.get_cache()

    @cached(cache_factory.get_cache(), 'walkforward', ttl=3600)
    async def get_by_id(self, id: str) ->Optional[WalkForwardTestResult]:
        """
        Get a walk-forward test by ID.
        
        Args:
            id: Test ID
            
        Returns:
            The walk-forward test or None if not found
        """
        logger.info(f'Getting walk-forward test with ID: {id}')
        if id in self.tests:
            logger.debug(f'Found walk-forward test {id} in memory cache')
            return self.tests[id]
        file_path = os.path.join(self.storage_path, f'{id}.json')
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    test_data = json.load(f)
                test_result = WalkForwardTestResult(**test_data)
                self.tests[id] = test_result
                logger.debug(f'Loaded walk-forward test {id} from file storage'
                    )
                return test_result
            except Exception as e:
                logger.error(
                    f'Error loading walk-forward test {id} from file: {e}')
        logger.warning(f'Walk-forward test {id} not found')
        return None

    @cached(cache_factory.get_cache(), 'walkforward_all', ttl=3600)
    async def get_all(self) ->List[WalkForwardTestResult]:
        """
        Get all walk-forward tests.
        
        Returns:
            List of all walk-forward tests
        """
        logger.info('Getting all walk-forward tests')
        tests = []
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                try:
                    test_id = filename[:-5]
                    test = await self.get_by_id(test_id)
                    if test:
                        tests.append(test)
                except Exception as e:
                    logger.error(
                        f'Error loading walk-forward test from {filename}: {e}'
                        )
        return tests

    @cached(cache_factory.get_cache(), 'walkforward_criteria', ttl=3600)
    async def get_by_criteria(self, criteria: Dict[str, Any]) ->List[
        WalkForwardTestResult]:
        """
        Get walk-forward tests by criteria.
        
        Args:
            criteria: Criteria to filter by
            
        Returns:
            List of walk-forward tests matching the criteria
        """
        logger.info(f'Getting walk-forward tests with criteria: {criteria}')
        all_tests = await self.get_all()
        filtered_tests = []
        for test in all_tests:
            match = True
            for key, value in criteria.items():
                if hasattr(test, key):
                    if getattr(test, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            if match:
                filtered_tests.append(test)
        return filtered_tests
