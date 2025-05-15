"""
Interface for the Analysis Coordinator Service.

This module defines the interface for the Analysis Coordinator Service, which orchestrates
analysis across multiple analysis services, including causal analysis, backtesting,
and market analysis.
"""
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime


class IAnalysisCoordinatorService(ABC):
    """Interface for the Analysis Coordinator Service."""

    @abstractmethod
    async def run_integrated_analysis(self, 
                                     symbol: str,
                                     timeframe: str,
                                     data: Dict[str, Any],
                                     analysis_types: List[str],
                                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run integrated analysis across multiple analysis services.

        Args:
            symbol: The symbol to analyze
            timeframe: The timeframe to analyze
            data: The market data to analyze
            analysis_types: List of analysis types to perform
            config: Optional configuration parameters

        Returns:
            A dictionary containing the integrated analysis results
        """
        pass

    @abstractmethod
    async def create_analysis_task(self, 
                                  task_type: str,
                                  parameters: Dict[str, Any],
                                  priority: Optional[int] = 1,
                                  callback_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new analysis task to be executed asynchronously.

        Args:
            task_type: The type of analysis task to create
            parameters: The parameters for the analysis task
            priority: Optional priority for the task (higher number = higher priority)
            callback_url: Optional URL to call when the task is complete

        Returns:
            A dictionary containing the created task information
        """
        pass

    @abstractmethod
    async def get_task_status(self, 
                             task_id: str) -> Dict[str, Any]:
        """
        Get the status of an analysis task.

        Args:
            task_id: The ID of the task to check

        Returns:
            A dictionary containing the task status information
        """
        pass

    @abstractmethod
    async def get_task_result(self, 
                             task_id: str) -> Dict[str, Any]:
        """
        Get the result of a completed analysis task.

        Args:
            task_id: The ID of the task to get the result for

        Returns:
            A dictionary containing the task result
        """
        pass

    @abstractmethod
    async def cancel_task(self, 
                         task_id: str) -> Dict[str, Any]:
        """
        Cancel a running analysis task.

        Args:
            task_id: The ID of the task to cancel

        Returns:
            A dictionary containing the cancellation result
        """
        pass

    @abstractmethod
    async def list_tasks(self,
                        status: Optional[str] = None,
                        task_type: Optional[str] = None,
                        created_after: Optional[datetime] = None,
                        created_before: Optional[datetime] = None,
                        limit: int = 100,
                        offset: int = 0) -> Dict[str, Any]:
        """
        List analysis tasks with optional filtering.

        Args:
            status: Optional status filter
            task_type: Optional task type filter
            created_after: Optional creation date filter (after)
            created_before: Optional creation date filter (before)
            limit: Maximum number of results to return
            offset: Offset for pagination

        Returns:
            A dictionary containing the list of tasks and pagination information
        """
        pass

    @abstractmethod
    async def run_multi_timeframe_analysis(self,
                                          symbol: str,
                                          timeframes: List[str],
                                          data: Dict[str, Dict[str, Any]],
                                          analysis_types: List[str],
                                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run analysis across multiple timeframes and consolidate the results.

        Args:
            symbol: The symbol to analyze
            timeframes: List of timeframes to analyze
            data: Dictionary of market data for each timeframe
            analysis_types: List of analysis types to perform
            config: Optional configuration parameters

        Returns:
            A dictionary containing the multi-timeframe analysis results
        """
        pass

    @abstractmethod
    async def run_multi_symbol_analysis(self,
                                       symbols: List[str],
                                       timeframe: str,
                                       data: Dict[str, Dict[str, Any]],
                                       analysis_types: List[str],
                                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run analysis across multiple symbols and consolidate the results.

        Args:
            symbols: List of symbols to analyze
            timeframe: The timeframe to analyze
            data: Dictionary of market data for each symbol
            analysis_types: List of analysis types to perform
            config: Optional configuration parameters

        Returns:
            A dictionary containing the multi-symbol analysis results
        """
        pass