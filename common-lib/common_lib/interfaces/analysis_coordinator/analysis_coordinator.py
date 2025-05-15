from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

class IAnalysisCoordinatorService(ABC):
    """
    Interface for analysis coordinator service.
    """
    
    @abstractmethod
    async def run_integrated_analysis(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        services: List[str] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run an integrated analysis across multiple analysis services.
        
        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            services: List of services to use for analysis
            parameters: Additional parameters for the analysis
            
        Returns:
            Integrated analysis task information
        """
        pass
        
    @abstractmethod
    async def create_analysis_task(
        self,
        service_type: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a new analysis task.
        
        Args:
            service_type: Type of service to use for analysis
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Additional parameters for the analysis
            
        Returns:
            Analysis task information
        """
        pass
        
    @abstractmethod
    async def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """
        Get the result of a previously created analysis task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task result
        """
        pass
        
    @abstractmethod
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a previously created analysis task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status
        """
        pass
        
    @abstractmethod
    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a previously created analysis task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            True if the task was deleted, False otherwise
        """
        pass
        
    @abstractmethod
    async def list_tasks(
        self,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all analysis tasks with optional filtering.
        
        Args:
            limit: Maximum number of tasks to return
            offset: Offset for pagination
            status: Filter by status
            
        Returns:
            List of tasks
        """
        pass
        
    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running analysis task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            True if the task was cancelled, False otherwise
        """
        pass
        
    @abstractmethod
    async def get_available_services(self) -> Dict[str, List[str]]:
        """
        Get available analysis services and their capabilities.
        
        Returns:
            Dictionary of available services and their capabilities
        """
        pass