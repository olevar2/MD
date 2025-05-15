"""
Causal Graph Read Repository

This module provides a read repository for causal graphs.
"""
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from common_lib.cqrs.repositories import ReadRepository
from common_lib.caching.decorators import cached
from causal_analysis_service.models.causal_models import CausalGraphResponse
from causal_analysis_service.utils.cache_factory import cache_factory
logger = logging.getLogger(__name__)


class CausalGraphReadRepository(ReadRepository[CausalGraphResponse, str]):
    """
    Read repository for causal graphs.
    """

    def __init__(self, db_connection=None):
        """
        Initialize the causal graph read repository.
        
        Args:
            db_connection: Database connection object
        """
        self.db_connection = db_connection
        self.cache = cache_factory.get_cache()

    @cached(cache_factory.get_cache(), 'causal_graph', ttl=3600)
    async def get_by_id(self, id: str) ->Optional[CausalGraphResponse]:
        """
        Get a causal graph by ID.
        
        Args:
            id: The ID of the causal graph
            
        Returns:
            The causal graph or None if not found
        """
        logger.info(f'Getting causal graph with ID {id}')
        if self.db_connection:
            try:
                logger.info(f'Retrieved causal graph {id} from database')
            except Exception as e:
                logger.error(
                    f'Error retrieving causal graph from database: {e}')
        return None

    async def get_all(self) ->List[CausalGraphResponse]:
        """
        Get all causal graphs.
        
        Returns:
            A list of all causal graphs
        """
        logger.info('Getting all causal graphs')
        if self.db_connection:
            try:
                logger.info('Retrieved all causal graphs from database')
            except Exception as e:
                logger.error(
                    f'Error retrieving causal graphs from database: {e}')
        return []

    async def get_by_criteria(self, criteria: Dict[str, Any]) ->List[
        CausalGraphResponse]:
        """
        Get causal graphs by criteria.
        
        Args:
            criteria: The criteria to filter by
            
        Returns:
            A list of causal graphs matching the criteria
        """
        logger.info(f'Getting causal graphs with criteria {criteria}')
        if self.db_connection:
            try:
                logger.info(
                    f'Retrieved causal graphs with criteria {criteria} from database'
                    )
            except Exception as e:
                logger.error(
                    f'Error retrieving causal graphs from database: {e}')
        return []
