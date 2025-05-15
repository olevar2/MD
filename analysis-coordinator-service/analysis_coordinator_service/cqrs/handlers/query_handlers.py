"""
Query handlers for the Analysis Coordinator Service.

This module provides the query handlers for the Analysis Coordinator Service.
"""
import logging
from typing import Dict, List, Any, Optional

from common_lib.cqrs.queries import QueryHandler
from analysis_coordinator_service.cqrs.queries import (
    GetAnalysisTaskQuery,
    ListAnalysisTasksQuery,
    GetIntegratedAnalysisTaskQuery,
    ListIntegratedAnalysisTasksQuery,
    GetAnalysisTaskStatusQuery,
    GetAvailableServicesQuery
)
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisTaskStatus,
    AnalysisTaskResult,
    AnalysisServiceType
)
from analysis_coordinator_service.repositories.read_repositories import TaskReadRepository
from analysis_coordinator_service.services.coordinator_service import CoordinatorService

logger = logging.getLogger(__name__)


class GetAnalysisTaskQueryHandler(QueryHandler):
    """Handler for GetAnalysisTaskQuery."""
    
    def __init__(self, repository: TaskReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Task read repository
        """
        self.repository = repository
    
    async def handle(self, query: GetAnalysisTaskQuery) -> Optional[AnalysisTaskResult]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The analysis task or None if not found
        """
        logger.info(f"Handling GetAnalysisTaskQuery: {query}")
        
        return await self.repository.get_by_id(query.task_id)


class ListAnalysisTasksQueryHandler(QueryHandler):
    """Handler for ListAnalysisTasksQuery."""
    
    def __init__(self, repository: TaskReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Task read repository
        """
        self.repository = repository
    
    async def handle(self, query: ListAnalysisTasksQuery) -> List[AnalysisTaskResult]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            List of analysis tasks
        """
        logger.info(f"Handling ListAnalysisTasksQuery: {query}")
        
        # Build criteria
        criteria = {}
        if query.service_type:
            criteria["service_type"] = query.service_type
        if query.symbol:
            criteria["symbol"] = query.symbol
        if query.status:
            criteria["status"] = query.status
        
        # Get tasks by criteria
        tasks = await self.repository.get_by_criteria(criteria)
        
        # Apply pagination
        tasks = tasks[query.offset:query.offset + query.limit]
        
        return tasks


class GetIntegratedAnalysisTaskQueryHandler(QueryHandler):
    """Handler for GetIntegratedAnalysisTaskQuery."""
    
    def __init__(self, repository: TaskReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Task read repository
        """
        self.repository = repository
    
    async def handle(self, query: GetIntegratedAnalysisTaskQuery) -> Optional[AnalysisTaskResult]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The integrated analysis task or None if not found
        """
        logger.info(f"Handling GetIntegratedAnalysisTaskQuery: {query}")
        
        task = await self.repository.get_by_id(query.task_id)
        
        if task and task.service_type == "integrated":
            return task
        
        return None


class ListIntegratedAnalysisTasksQueryHandler(QueryHandler):
    """Handler for ListIntegratedAnalysisTasksQuery."""
    
    def __init__(self, repository: TaskReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Task read repository
        """
        self.repository = repository
    
    async def handle(self, query: ListIntegratedAnalysisTasksQuery) -> List[AnalysisTaskResult]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            List of integrated analysis tasks
        """
        logger.info(f"Handling ListIntegratedAnalysisTasksQuery: {query}")
        
        # Build criteria
        criteria = {"service_type": "integrated"}
        if query.symbol:
            criteria["symbol"] = query.symbol
        if query.status:
            criteria["status"] = query.status
        
        # Get tasks by criteria
        tasks = await self.repository.get_by_criteria(criteria)
        
        # Apply pagination
        tasks = tasks[query.offset:query.offset + query.limit]
        
        return tasks


class GetAnalysisTaskStatusQueryHandler(QueryHandler):
    """Handler for GetAnalysisTaskStatusQuery."""
    
    def __init__(self, repository: TaskReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Task read repository
        """
        self.repository = repository
    
    async def handle(self, query: GetAnalysisTaskStatusQuery) -> Optional[AnalysisTaskStatus]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The analysis task status or None if not found
        """
        logger.info(f"Handling GetAnalysisTaskStatusQuery: {query}")
        
        return await self.repository.get_task_status(query.task_id)


class GetAvailableServicesQueryHandler(QueryHandler):
    """Handler for GetAvailableServicesQuery."""
    
    def __init__(self, coordinator_service: CoordinatorService):
        """
        Initialize the handler.
        
        Args:
            coordinator_service: Coordinator service
        """
        self.coordinator_service = coordinator_service
    
    async def handle(self, query: GetAvailableServicesQuery) -> Dict[str, List[str]]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            Dictionary of available services
        """
        logger.info(f"Handling GetAvailableServicesQuery: {query}")
        
        return await self.coordinator_service.get_available_services()