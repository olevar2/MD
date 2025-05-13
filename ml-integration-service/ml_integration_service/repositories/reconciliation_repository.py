"""
Repository for accessing reconciliation data in the database.

This module provides a repository for accessing reconciliation processes,
discrepancies, and resolutions in the database.
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete
from sqlalchemy.orm import selectinload
from ml_integration_service.models.reconciliation_models import ReconciliationProcess, Discrepancy, DataSourceConfig
from common_lib.data_reconciliation import ReconciliationStatus, ReconciliationSeverity, ReconciliationStrategy, ReconciliationResult, ReconciliationDiscrepancy, DataSource, DataSourceType
from common_lib.exceptions import DataFetchError, DataStorageError
logger = logging.getLogger(__name__)


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ReconciliationRepository:
    """Repository for accessing reconciliation data in the database."""

    def __init__(self, session: AsyncSession):
        """
        Initialize the repository.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    @async_with_exception_handling
    async def create_reconciliation_process(self, reconciliation_type: str,
        model_id: Optional[str]=None, version: Optional[str]=None, status:
        ReconciliationStatus=ReconciliationStatus.IN_PROGRESS, strategy:
        ReconciliationStrategy=ReconciliationStrategy.SOURCE_PRIORITY,
        tolerance: float=0.0001, auto_resolve: bool=True,
        notification_threshold: ReconciliationSeverity=
        ReconciliationSeverity.HIGH) ->ReconciliationProcess:
        """
        Create a new reconciliation process.
        
        Args:
            reconciliation_type: Type of reconciliation
            model_id: ID of the model
            version: Version of the model
            status: Status of the reconciliation process
            strategy: Strategy for resolving discrepancies
            tolerance: Tolerance for numerical differences
            auto_resolve: Whether to automatically resolve discrepancies
            notification_threshold: Minimum severity for notifications
            
        Returns:
            Created reconciliation process
            
        Raises:
            DataStorageError: If the process cannot be created
        """
        try:
            reconciliation_process = ReconciliationProcess(model_id=
                model_id, version=version, reconciliation_type=
                reconciliation_type, status=status, strategy=strategy,
                tolerance=tolerance, auto_resolve=auto_resolve,
                notification_threshold=notification_threshold, start_time=
                datetime.utcnow())
            self.session.add(reconciliation_process)
            await self.session.commit()
            return reconciliation_process
        except Exception as e:
            await self.session.rollback()
            logger.error(f'Error creating reconciliation process: {str(e)}')
            raise DataStorageError(
                f'Error creating reconciliation process: {str(e)}')

    @async_with_exception_handling
    async def update_reconciliation_process(self, reconciliation_id: str,
        status: Optional[ReconciliationStatus]=None, end_time: Optional[
        datetime]=None, duration_seconds: Optional[float]=None,
        discrepancy_count: Optional[int]=None, resolution_count: Optional[
        int]=None, resolution_rate: Optional[float]=None) ->Optional[
        ReconciliationProcess]:
        """
        Update a reconciliation process.
        
        Args:
            reconciliation_id: ID of the reconciliation process
            status: Status of the reconciliation process
            end_time: End time of the reconciliation process
            duration_seconds: Duration of the reconciliation process in seconds
            discrepancy_count: Number of discrepancies found
            resolution_count: Number of resolutions applied
            resolution_rate: Percentage of discrepancies resolved
            
        Returns:
            Updated reconciliation process, or None if not found
            
        Raises:
            DataStorageError: If the process cannot be updated
        """
        try:
            update_values = {}
            if status is not None:
                update_values['status'] = status
            if end_time is not None:
                update_values['end_time'] = end_time
            if duration_seconds is not None:
                update_values['duration_seconds'] = duration_seconds
            if discrepancy_count is not None:
                update_values['discrepancy_count'] = discrepancy_count
            if resolution_count is not None:
                update_values['resolution_count'] = resolution_count
            if resolution_rate is not None:
                update_values['resolution_rate'] = resolution_rate
            stmt = update(ReconciliationProcess).where(
                ReconciliationProcess.id == reconciliation_id).values(**
                update_values).execution_options(synchronize_session='fetch')
            await self.session.execute(stmt)
            await self.session.commit()
            return await self.get_reconciliation_process(reconciliation_id)
        except Exception as e:
            await self.session.rollback()
            logger.error(f'Error updating reconciliation process: {str(e)}')
            raise DataStorageError(
                f'Error updating reconciliation process: {str(e)}')

    @async_with_exception_handling
    async def get_reconciliation_process(self, reconciliation_id: str,
        include_discrepancies: bool=False) ->Optional[ReconciliationProcess]:
        """
        Get a reconciliation process by ID.
        
        Args:
            reconciliation_id: ID of the reconciliation process
            include_discrepancies: Whether to include discrepancies
            
        Returns:
            Reconciliation process, or None if not found
            
        Raises:
            DataFetchError: If the process cannot be fetched
        """
        try:
            if include_discrepancies:
                stmt = select(ReconciliationProcess).options(selectinload(
                    ReconciliationProcess.discrepancies)).where(
                    ReconciliationProcess.id == reconciliation_id)
            else:
                stmt = select(ReconciliationProcess).where(
                    ReconciliationProcess.id == reconciliation_id)
            result = await self.session.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            logger.error(f'Error getting reconciliation process: {str(e)}')
            raise DataFetchError(
                f'Error getting reconciliation process: {str(e)}')

    @async_with_exception_handling
    async def add_discrepancy(self, reconciliation_id: str, field_name: str,
        severity: ReconciliationSeverity, source_1_id: str, source_1_value:
        Any, source_2_id: str, source_2_value: Any, difference: Optional[
        float]=None, resolved: bool=False, resolution_strategy: Optional[
        ReconciliationStrategy]=None, resolved_value: Optional[Any]=None,
        metadata: Optional[Dict[str, Any]]=None) ->Discrepancy:
        """
        Add a discrepancy to a reconciliation process.
        
        Args:
            reconciliation_id: ID of the reconciliation process
            field_name: Name of the field with the discrepancy
            severity: Severity of the discrepancy
            source_1_id: ID of the first source
            source_1_value: Value from the first source
            source_2_id: ID of the second source
            source_2_value: Value from the second source
            difference: Numerical difference between the values
            resolved: Whether the discrepancy is resolved
            resolution_strategy: Strategy used for resolution
            resolved_value: Resolved value
            metadata: Additional metadata
            
        Returns:
            Created discrepancy
            
        Raises:
            DataStorageError: If the discrepancy cannot be created
        """
        try:
            source_1_value_str = str(source_1_value
                ) if source_1_value is not None else None
            source_2_value_str = str(source_2_value
                ) if source_2_value is not None else None
            resolved_value_str = str(resolved_value
                ) if resolved_value is not None else None
            discrepancy = Discrepancy(reconciliation_id=reconciliation_id,
                field_name=field_name, severity=severity, source_1_id=
                source_1_id, source_1_value=source_1_value_str, source_2_id
                =source_2_id, source_2_value=source_2_value_str, difference
                =difference, resolved=resolved, resolution_strategy=
                resolution_strategy, resolved_value=resolved_value_str,
                resolution_time=datetime.utcnow() if resolved else None,
                metadata=metadata)
            self.session.add(discrepancy)
            await self.session.commit()
            return discrepancy
        except Exception as e:
            await self.session.rollback()
            logger.error(f'Error adding discrepancy: {str(e)}')
            raise DataStorageError(f'Error adding discrepancy: {str(e)}')

    async def convert_to_result(self, reconciliation_process:
        ReconciliationProcess) ->ReconciliationResult:
        """
        Convert a reconciliation process to a result object.
        
        Args:
            reconciliation_process: Reconciliation process
            
        Returns:
            Reconciliation result
        """
        stmt = select(Discrepancy).where(Discrepancy.reconciliation_id ==
            reconciliation_process.id)
        result = await self.session.execute(stmt)
        discrepancies = result.scalars().all()
        discrepancy_objects = []
        for discrepancy in discrepancies:
            discrepancy_objects.append(ReconciliationDiscrepancy(field_name
                =discrepancy.field_name, severity=discrepancy.severity,
                source_1_id=discrepancy.source_1_id, source_1_value=
                discrepancy.source_1_value, source_2_id=discrepancy.
                source_2_id, source_2_value=discrepancy.source_2_value,
                difference=discrepancy.difference, resolved=discrepancy.
                resolved, resolution_strategy=discrepancy.
                resolution_strategy, resolved_value=discrepancy.resolved_value)
                )
        return ReconciliationResult(reconciliation_id=
            reconciliation_process.id, status=reconciliation_process.status,
            discrepancy_count=reconciliation_process.discrepancy_count,
            resolution_count=reconciliation_process.resolution_count,
            resolution_rate=reconciliation_process.resolution_rate,
            duration_seconds=reconciliation_process.duration_seconds,
            start_time=reconciliation_process.start_time, end_time=
            reconciliation_process.end_time, discrepancies=discrepancy_objects)
