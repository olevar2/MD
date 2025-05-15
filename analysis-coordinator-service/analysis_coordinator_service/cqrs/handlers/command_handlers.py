"""
Command handlers for the Analysis Coordinator Service.

This module provides the command handlers for the Analysis Coordinator Service.
"""
import logging
import uuid
import asyncio
from datetime import datetime, timedelta, UTC
from typing import Dict, List, Any, Optional

from common_lib.cqrs.commands import CommandHandler
from analysis_coordinator_service.cqrs.commands import (
    RunIntegratedAnalysisCommand,
    CreateAnalysisTaskCommand,
    CancelAnalysisTaskCommand,
    DeleteAnalysisTaskCommand
)
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisTaskStatus,
    AnalysisTaskResult,
    AnalysisServiceType,
    IntegratedAnalysisResponse
)
from analysis_coordinator_service.repositories.write_repositories import TaskWriteRepository
from analysis_coordinator_service.services.coordinator_service import CoordinatorService

logger = logging.getLogger(__name__)


class RunIntegratedAnalysisCommandHandler(CommandHandler):
    """Handler for RunIntegratedAnalysisCommand."""
    
    def __init__(
        self,
        coordinator_service: CoordinatorService,
        repository: TaskWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            coordinator_service: Coordinator service
            repository: Task write repository
        """
        self.coordinator_service = coordinator_service
        self.repository = repository
    
    async def handle(self, command: RunIntegratedAnalysisCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the integrated analysis task
        """
        logger.info(f"Handling RunIntegratedAnalysisCommand: {command}")
        
        # Create a new task
        task_id = str(uuid.uuid4())
        estimated_completion_time = datetime.now(UTC) + timedelta(minutes=5)  # Estimate 5 minutes for completion
        
        # Create task in repository
        await self.repository.create_integrated_task(
            task_id=task_id,
            services=command.services,
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date,
            end_date=command.end_date,
            parameters=command.parameters
        )
        
        # Start the task asynchronously using asyncio.create_task
        asyncio.create_task(self.coordinator_service._execute_integrated_analysis(
            task_id=task_id,
            services=command.services,
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date,
            end_date=command.end_date,
            parameters=command.parameters
        ))
        
        return task_id


class CreateAnalysisTaskCommandHandler(CommandHandler):
    """Handler for CreateAnalysisTaskCommand."""
    
    def __init__(
        self,
        coordinator_service: CoordinatorService,
        repository: TaskWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            coordinator_service: Coordinator service
            repository: Task write repository
        """
        self.coordinator_service = coordinator_service
        self.repository = repository
    
    async def handle(self, command: CreateAnalysisTaskCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the analysis task
        """
        logger.info(f"Handling CreateAnalysisTaskCommand: {command}")
        
        # Create a new task
        task_id = str(uuid.uuid4())
        
        # Create task in repository
        await self.repository.create_task(
            task_id=task_id,
            service_type=command.service_type,
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date,
            end_date=command.end_date,
            parameters=command.parameters
        )
        
        # Start the task asynchronously using asyncio.create_task
        asyncio.create_task(self.coordinator_service._execute_analysis_task(
            task_id=task_id,
            service_type=command.service_type,
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date,
            end_date=command.end_date,
            parameters=command.parameters
        ))
        
        return task_id


class CancelAnalysisTaskCommandHandler(CommandHandler):
    """Handler for CancelAnalysisTaskCommand."""
    
    def __init__(
        self,
        repository: TaskWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            repository: Task write repository
        """
        self.repository = repository
    
    async def handle(self, command: CancelAnalysisTaskCommand) -> bool:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            True if the task was cancelled, False otherwise
        """
        logger.info(f"Handling CancelAnalysisTaskCommand: {command}")
        
        # Cancel the task
        return await self.repository.cancel_task(command.task_id)


class DeleteAnalysisTaskCommandHandler(CommandHandler):
    """Handler for DeleteAnalysisTaskCommand."""
    
    def __init__(
        self,
        repository: TaskWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            repository: Task write repository
        """
        self.repository = repository
    
    async def handle(self, command: DeleteAnalysisTaskCommand) -> None:
        """
        Handle the command.
        
        Args:
            command: The command
        """
        logger.info(f"Handling DeleteAnalysisTaskCommand: {command}")
        
        # Delete the task
        await self.repository.delete(command.task_id)