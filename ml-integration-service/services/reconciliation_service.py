"""
Reconciliation Service for the ML Integration Service.

This service provides functionality for reconciling model data from different sources,
including training data and inference data.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import uuid
from config.enhanced_settings import enhanced_settings
from config.reconciliation_config import get_reconciliation_config
from models.model_data_reconciliation import TrainingDataReconciliation, InferenceDataReconciliation
from api.model_repository import ModelRepository
from repositories.reconciliation_repository import ReconciliationRepository
from services.feature_service import FeatureService
from core.data_validator import DataValidator
from common_lib.data_reconciliation import DataSource, DataSourceType, ReconciliationConfig, ReconciliationResult, ReconciliationStatus, ReconciliationSeverity, ReconciliationStrategy
from common_lib.exceptions import DataFetchError, DataValidationError, ReconciliationError
from common_lib.metrics.reconciliation_metrics import track_reconciliation, record_discrepancy, record_resolution
from ml_integration_service.tracing import trace_method
logger = logging.getLogger(__name__)


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ReconciliationService:
    """Service for reconciling model data from different sources."""

    def __init__(self, model_repository: ModelRepository, feature_service:
        FeatureService, data_validator: DataValidator,
        reconciliation_repository=None):
        """
        Initialize the reconciliation service.

        Args:
            model_repository: Repository for accessing model data
            feature_service: Service for accessing feature data
            data_validator: Validator for model data
            reconciliation_repository: Repository for accessing reconciliation data
        """
        self.model_repository = model_repository
        self.feature_service = feature_service
        self.data_validator = data_validator
        self.reconciliation_repository = reconciliation_repository
        self.reconciliation_results = {}

    @trace_method(name='reconcile_training_data')
    @track_reconciliation(service='ml_integration_service',
        reconciliation_type='training_data')
    @async_with_exception_handling
    async def reconcile_training_data(self, model_id: str, version:
        Optional[str]=None, start_time: Optional[datetime]=None, end_time:
        Optional[datetime]=None, strategy: Optional[ReconciliationStrategy]
        =None, tolerance: Optional[float]=None, auto_resolve: Optional[bool
        ]=None, notification_threshold: Optional[ReconciliationSeverity]=None
        ) ->ReconciliationResult:
        """
        Reconcile training data for a model.

        Args:
            model_id: ID of the model
            version: Version of the model
            start_time: Start time for data reconciliation
            end_time: End time for data reconciliation
            strategy: Strategy for resolving discrepancies
            tolerance: Tolerance for numerical differences
            auto_resolve: Whether to automatically resolve discrepancies
            notification_threshold: Minimum severity for notifications

        Returns:
            Results of the reconciliation process

        Raises:
            DataFetchError: If data fetching fails
            DataValidationError: If data validation fails
            ReconciliationError: If reconciliation fails
        """
        try:
            reconciliation_config = get_reconciliation_config()
            strategy = (strategy or reconciliation_config.defaults.
                default_strategy)
            tolerance = (tolerance if tolerance is not None else
                reconciliation_config.defaults.default_tolerance)
            auto_resolve = (auto_resolve if auto_resolve is not None else
                reconciliation_config.defaults.default_auto_resolve)
            notification_threshold = (notification_threshold or
                reconciliation_config.defaults.default_notification_threshold)
            database_source = DataSource(source_id='database', name=
                'Model Repository Database', source_type=DataSourceType.
                DATABASE, priority=1)
            cache_source = DataSource(source_id='cache', name=
                'Feature Service Cache', source_type=DataSourceType.CACHE,
                priority=2)
            config = ReconciliationConfig(sources=[database_source,
                cache_source], strategy=strategy, tolerance=tolerance,
                auto_resolve=auto_resolve, notification_threshold=
                notification_threshold)
            reconciliation = TrainingDataReconciliation(config=config,
                model_repository=self.model_repository, feature_service=
                self.feature_service, data_validator=self.data_validator)
            if self.reconciliation_repository:
                reconciliation_process = (await self.
                    reconciliation_repository.create_reconciliation_process
                    (reconciliation_type='training_data', model_id=model_id,
                    version=version, status=ReconciliationStatus.
                    IN_PROGRESS, strategy=strategy, tolerance=tolerance,
                    auto_resolve=auto_resolve, notification_threshold=
                    notification_threshold))
                reconciliation_id = reconciliation_process.id
            else:
                reconciliation_id = str(uuid.uuid4())
            logger.info(
                f'Starting training data reconciliation for model {model_id}, version {version}, reconciliation ID: {reconciliation_id}'
                )
            result = await reconciliation.reconcile(model_id=model_id,
                version=version, start_time=start_time, end_time=end_time)
            result.reconciliation_id = reconciliation_id
            logger.info(
                f'Completed training data reconciliation for model {model_id}, version {version}, reconciliation ID: {result.reconciliation_id}, status: {result.status.name}, discrepancies: {result.discrepancy_count}, resolutions: {result.resolution_count}'
                )
            for severity in ReconciliationSeverity:
                count = sum(1 for d in result.discrepancies if d.severity ==
                    severity)
                if count > 0:
                    record_discrepancy(service='ml_integration_service',
                        reconciliation_type='training_data', severity=
                        severity.name, count=count)
            record_resolution(service='ml_integration_service',
                reconciliation_type='training_data', strategy=strategy.name,
                count=result.resolution_count)
            if self.reconciliation_repository:
                await self.reconciliation_repository.update_reconciliation_process(
                    reconciliation_id=result.reconciliation_id, status=
                    result.status, end_time=result.end_time,
                    duration_seconds=result.duration_seconds,
                    discrepancy_count=result.discrepancy_count,
                    resolution_count=result.resolution_count,
                    resolution_rate=result.resolution_rate)
                for discrepancy in result.discrepancies:
                    await self.reconciliation_repository.add_discrepancy(
                        reconciliation_id=result.reconciliation_id,
                        field_name=discrepancy.field_name, severity=
                        discrepancy.severity, source_1_id=discrepancy.
                        source_1_id, source_1_value=discrepancy.
                        source_1_value, source_2_id=discrepancy.source_2_id,
                        source_2_value=discrepancy.source_2_value,
                        difference=discrepancy.difference, resolved=
                        discrepancy.resolved, resolution_strategy=
                        discrepancy.resolution_strategy, resolved_value=
                        discrepancy.resolved_value)
            self.reconciliation_results[result.reconciliation_id] = result
            return result
        except DataFetchError as e:
            logger.error(
                f'Data fetch error during training data reconciliation: {str(e)}'
                )
            raise
        except DataValidationError as e:
            logger.error(
                f'Data validation error during training data reconciliation: {str(e)}'
                )
            raise
        except ReconciliationError as e:
            logger.error(
                f'Reconciliation error during training data reconciliation: {str(e)}'
                )
            raise
        except Exception as e:
            logger.exception(
                f'Unexpected error during training data reconciliation: {str(e)}'
                )
            raise ReconciliationError(
                f'Unexpected error during training data reconciliation: {str(e)}'
                , details={'model_id': model_id, 'version': version})

    @trace_method(name='reconcile_inference_data')
    @track_reconciliation(service='ml_integration_service',
        reconciliation_type='inference_data')
    @async_with_exception_handling
    async def reconcile_inference_data(self, model_id: str, version:
        Optional[str]=None, start_time: Optional[datetime]=None, end_time:
        Optional[datetime]=None, strategy: Optional[ReconciliationStrategy]
        =None, tolerance: Optional[float]=None, auto_resolve: Optional[bool
        ]=None, notification_threshold: Optional[ReconciliationSeverity]=None
        ) ->ReconciliationResult:
        """
        Reconcile inference data for a model.

        Args:
            model_id: ID of the model
            version: Version of the model
            start_time: Start time for data reconciliation
            end_time: End time for data reconciliation
            strategy: Strategy for resolving discrepancies
            tolerance: Tolerance for numerical differences
            auto_resolve: Whether to automatically resolve discrepancies
            notification_threshold: Minimum severity for notifications

        Returns:
            Results of the reconciliation process

        Raises:
            DataFetchError: If data fetching fails
            DataValidationError: If data validation fails
            ReconciliationError: If reconciliation fails
        """
        try:
            reconciliation_config = get_reconciliation_config()
            strategy = (strategy or reconciliation_config.defaults.
                default_strategy)
            tolerance = (tolerance if tolerance is not None else
                reconciliation_config.defaults.default_tolerance)
            auto_resolve = (auto_resolve if auto_resolve is not None else
                reconciliation_config.defaults.default_auto_resolve)
            notification_threshold = (notification_threshold or
                reconciliation_config.defaults.default_notification_threshold)
            database_source = DataSource(source_id='database', name=
                'Model Repository Database', source_type=DataSourceType.
                DATABASE, priority=1)
            cache_source = DataSource(source_id='cache', name=
                'Feature Service Cache', source_type=DataSourceType.CACHE,
                priority=2)
            config = ReconciliationConfig(sources=[database_source,
                cache_source], strategy=strategy, tolerance=tolerance,
                auto_resolve=auto_resolve, notification_threshold=
                notification_threshold)
            reconciliation = InferenceDataReconciliation(config=config,
                model_repository=self.model_repository, feature_service=
                self.feature_service, data_validator=self.data_validator)
            if self.reconciliation_repository:
                reconciliation_process = (await self.
                    reconciliation_repository.create_reconciliation_process
                    (reconciliation_type='inference_data', model_id=
                    model_id, version=version, status=ReconciliationStatus.
                    IN_PROGRESS, strategy=strategy, tolerance=tolerance,
                    auto_resolve=auto_resolve, notification_threshold=
                    notification_threshold))
                reconciliation_id = reconciliation_process.id
            else:
                reconciliation_id = str(uuid.uuid4())
            logger.info(
                f'Starting inference data reconciliation for model {model_id}, version {version}, reconciliation ID: {reconciliation_id}'
                )
            result = await reconciliation.reconcile(model_id=model_id,
                version=version, start_time=start_time, end_time=end_time)
            result.reconciliation_id = reconciliation_id
            logger.info(
                f'Completed inference data reconciliation for model {model_id}, version {version}, reconciliation ID: {result.reconciliation_id}, status: {result.status.name}, discrepancies: {result.discrepancy_count}, resolutions: {result.resolution_count}'
                )
            for severity in ReconciliationSeverity:
                count = sum(1 for d in result.discrepancies if d.severity ==
                    severity)
                if count > 0:
                    record_discrepancy(service='ml_integration_service',
                        reconciliation_type='inference_data', severity=
                        severity.name, count=count)
            record_resolution(service='ml_integration_service',
                reconciliation_type='inference_data', strategy=strategy.
                name, count=result.resolution_count)
            if self.reconciliation_repository:
                await self.reconciliation_repository.update_reconciliation_process(
                    reconciliation_id=result.reconciliation_id, status=
                    result.status, end_time=result.end_time,
                    duration_seconds=result.duration_seconds,
                    discrepancy_count=result.discrepancy_count,
                    resolution_count=result.resolution_count,
                    resolution_rate=result.resolution_rate)
                for discrepancy in result.discrepancies:
                    await self.reconciliation_repository.add_discrepancy(
                        reconciliation_id=result.reconciliation_id,
                        field_name=discrepancy.field_name, severity=
                        discrepancy.severity, source_1_id=discrepancy.
                        source_1_id, source_1_value=discrepancy.
                        source_1_value, source_2_id=discrepancy.source_2_id,
                        source_2_value=discrepancy.source_2_value,
                        difference=discrepancy.difference, resolved=
                        discrepancy.resolved, resolution_strategy=
                        discrepancy.resolution_strategy, resolved_value=
                        discrepancy.resolved_value)
            self.reconciliation_results[result.reconciliation_id] = result
            return result
        except DataFetchError as e:
            logger.error(
                f'Data fetch error during inference data reconciliation: {str(e)}'
                )
            raise
        except DataValidationError as e:
            logger.error(
                f'Data validation error during inference data reconciliation: {str(e)}'
                )
            raise
        except ReconciliationError as e:
            logger.error(
                f'Reconciliation error during inference data reconciliation: {str(e)}'
                )
            raise
        except Exception as e:
            logger.exception(
                f'Unexpected error during inference data reconciliation: {str(e)}'
                )
            raise ReconciliationError(
                f'Unexpected error during inference data reconciliation: {str(e)}'
                , details={'model_id': model_id, 'version': version})

    @trace_method(name='get_reconciliation_status')
    @async_with_exception_handling
    async def get_reconciliation_status(self, reconciliation_id: str
        ) ->Optional[ReconciliationResult]:
        """
        Get the status of a reconciliation process.

        Args:
            reconciliation_id: ID of the reconciliation process

        Returns:
            Results of the reconciliation process, or None if not found
        """
        result = self.reconciliation_results.get(reconciliation_id)
        if not result and self.reconciliation_repository:
            try:
                reconciliation_process = (await self.
                    reconciliation_repository.get_reconciliation_process(
                    reconciliation_id=reconciliation_id,
                    include_discrepancies=True))
                if reconciliation_process:
                    result = (await self.reconciliation_repository.
                        convert_to_result(reconciliation_process))
                    self.reconciliation_results[reconciliation_id] = result
            except Exception as e:
                logger.error(
                    f'Error getting reconciliation status from database: {str(e)}'
                    )
        if result:
            logger.info(
                f'Retrieved reconciliation status for ID {reconciliation_id}, status: {result.status.name}'
                )
        else:
            logger.warning(f'Reconciliation ID {reconciliation_id} not found')
        return result
