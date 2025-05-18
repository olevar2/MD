"""
Backtest read repository.

This module provides the read repository for backtests.
"""
import logging
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from common_lib.cqrs.repositories import ReadRepository
from backtesting_service.models.backtest_models import BacktestResult, BacktestStatus
from common_lib.caching.decorators import cached
from backtesting_service.utils.cache_factory import cache_factory
logger = logging.getLogger(__name__)


class BacktestReadRepository(ReadRepository):
    """
    Read repository for backtests.
    """

    def __init__(self, storage_path: Optional[str]=None):
        """
        Initialize the repository with a storage path.
        
        Args:
            storage_path: Path to the storage directory
        """
        self.storage_path = storage_path or os.environ.get(
            'BACKTEST_STORAGE_PATH', './data/backtests')
        os.makedirs(self.storage_path, exist_ok=True)
        self.backtests: Dict[str, BacktestResult] = {}
        self.cache = cache_factory.get_cache()

    @cached(cache_factory.get_cache(), 'backtest', ttl=3600)
    async def get_by_id(self, id: str) ->Optional[BacktestResult]:
        """
        Get a backtest by ID.
        
        Args:
            id: Backtest ID
            
        Returns:
            The backtest or None if not found
        """
        logger.info(f'Getting backtest with ID: {id}')
        if id in self.backtests:
            logger.debug(f'Found backtest {id} in memory cache')
            return self.backtests[id]
        file_path = os.path.join(self.storage_path, f'{id}.json')
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    backtest_data = json.load(f)
                backtest_result = BacktestResult(**backtest_data)
                self.backtests[id] = backtest_result
                logger.debug(f'Loaded backtest {id} from file storage')
                return backtest_result
            except Exception as e:
                logger.error(f'Error loading backtest {id} from file: {e}')
        logger.warning(f'Backtest {id} not found')
        return None

    @cached(cache_factory.get_cache(), 'backtest_all', ttl=3600)
    async def get_all(self) ->List[BacktestResult]:
        """
        Get all backtests.
        
        Returns:
            List of all backtests
        """
        logger.info('Getting all backtests')
        backtests = []
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                try:
                    backtest_id = filename[:-5]
                    backtest = await self.get_by_id(backtest_id)
                    if backtest:
                        backtests.append(backtest)
                except Exception as e:
                    logger.error(f'Error loading backtest from {filename}: {e}'
                        )
        return backtests

    @cached(cache_factory.get_cache(), 'backtest_criteria', ttl=3600)
    async def get_by_criteria(self, criteria: Dict[str, Any]) ->List[
        BacktestResult]:
        """
        Get backtests by criteria.
        
        Args:
            criteria: Criteria to filter by
            
        Returns:
            List of backtests matching the criteria
        """
        logger.info(f'Getting backtests with criteria: {criteria}')
        all_backtests = await self.get_all()
        filtered_backtests = []
        for backtest in all_backtests:
            match = True
            for key, value in criteria.items():
                if hasattr(backtest, key):
                    if getattr(backtest, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            if match:
                filtered_backtests.append(backtest)
        return filtered_backtests
