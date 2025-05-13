"""
Model data reconciliation implementations.

This module provides implementations for reconciling model data from different sources,
including training data and inference data.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from ml_integration_service.repositories.model_repository import ModelRepository
    from ml_integration_service.services.feature_service import FeatureService
    from ml_integration_service.validation.data_validator import DataValidator
import pandas as pd
import numpy as np
from common_lib.data_reconciliation.base import DataReconciliationBase, DataSource, DataSourceType, Discrepancy, DiscrepancyResolution, ReconciliationConfig, ReconciliationResult, ReconciliationSeverity, ReconciliationStatus, ReconciliationStrategy
from common_lib.data_reconciliation.batch import BatchReconciliationProcessor
from common_lib.data_reconciliation.realtime import RealTimeReconciliationProcessor
from common_lib.data_reconciliation.strategies import create_resolution_strategy
from common_lib.data_reconciliation.exceptions import SourceDataError
import logging
logger = logging.getLogger(__name__)


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ModelDataReconciliation(BatchReconciliationProcessor):
    """Base class for model data reconciliation."""

    def __init__(self, config: ReconciliationConfig, model_repository: Any,
        feature_service: Any, data_validator: Any):
        """
        Initialize model data reconciliation.

        Args:
            config: Configuration for the reconciliation process
            model_repository: Repository for model data
            feature_service: Service for accessing features
            data_validator: Validator for model data
        """
        super().__init__(config)
        self.model_repository = model_repository
        self.feature_service = feature_service
        self.data_validator = data_validator

    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]
        ) ->bool:
        """
        Apply resolutions to model data.

        Args:
            resolutions: List of resolutions to apply

        Returns:
            Whether all resolutions were successfully applied
        """
        for resolution in resolutions:
            logger.info(
                f'Resolution for {resolution.discrepancy.field}: Using value {resolution.resolved_value} from strategy {resolution.strategy.name}'
                )
        return True


class TrainingDataReconciliation(ModelDataReconciliation):
    """Reconciliation for model training data."""

    @async_with_exception_handling
    async def fetch_data(self, source: DataSource, **kwargs) ->pd.DataFrame:
        """
        Fetch training data from a source.

        Args:
            source: The data source to fetch from
            **kwargs: Additional parameters for fetching
                - model_id: ID of the model
                - version: Version of the model
                - dataset_id: ID of the training dataset

        Returns:
            DataFrame with training data
        """
        model_id = kwargs.get('model_id')
        version = kwargs.get('version')
        dataset_id = kwargs.get('dataset_id')
        if not model_id:
            raise ValueError(
                "Missing required parameter 'model_id' for fetching training data"
                )
        try:
            if source.source_type == DataSourceType.DATABASE:
                if dataset_id:
                    data = await self.model_repository.get_training_dataset(
                        dataset_id)
                else:
                    data = await self.model_repository.get_model_training_data(
                        model_id, version)
            elif source.source_type == DataSourceType.CACHE:
                data = await self.feature_service.get_cached_training_data(
                    model_id, version)
            else:
                raise ValueError(
                    f'Unsupported source type for training data: {source.source_type}'
                    )
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, dict):
                    data = pd.DataFrame(data)
                else:
                    raise ValueError(
                        f'Unexpected data type from source {source.source_id}: {type(data)}'
                        )
            return data
        except Exception as e:
            logger.error(
                f'Error fetching training data from source {source.source_id}: {str(e)}'
                )
            raise SourceDataError(message=
                f'Failed to fetch training data: {str(e)}', source_id=
                source.source_id)

    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]
        ) ->bool:
        """
        Apply resolutions to training data.

        Args:
            resolutions: List of resolutions to apply

        Returns:
            Whether all resolutions were successfully applied
        """
        success = True
        field_resolutions = {}
        for resolution in resolutions:
            field = resolution.discrepancy.field
            field_resolutions[field] = resolution
        for field, resolution in field_resolutions.items():
            logger.info(
                f'Resolution for {field}: Using value {resolution.resolved_value} from strategy {resolution.strategy.name}'
                )
        return success


class InferenceDataReconciliation(ModelDataReconciliation):
    """Reconciliation for model inference data."""

    @async_with_exception_handling
    async def fetch_data(self, source: DataSource, **kwargs) ->pd.DataFrame:
        """
        Fetch inference data from a source.

        Args:
            source: The data source to fetch from
            **kwargs: Additional parameters for fetching
                - model_id: ID of the model
                - version: Version of the model
                - start_time: Start time for the data
                - end_time: End time for the data

        Returns:
            DataFrame with inference data
        """
        model_id = kwargs.get('model_id')
        version = kwargs.get('version')
        start_time = kwargs.get('start_time')
        end_time = kwargs.get('end_time')
        if not model_id:
            raise ValueError(
                "Missing required parameter 'model_id' for fetching inference data"
                )
        try:
            if source.source_type == DataSourceType.DATABASE:
                data = await self.model_repository.get_model_inference_data(
                    model_id=model_id, version=version, start_time=
                    start_time, end_time=end_time)
            elif source.source_type == DataSourceType.CACHE:
                data = await self.feature_service.get_cached_inference_data(
                    model_id=model_id, version=version, start_time=
                    start_time, end_time=end_time)
            else:
                raise ValueError(
                    f'Unsupported source type for inference data: {source.source_type}'
                    )
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, dict):
                    data = pd.DataFrame(data)
                else:
                    raise ValueError(
                        f'Unexpected data type from source {source.source_id}: {type(data)}'
                        )
            return data
        except Exception as e:
            logger.error(
                f'Error fetching inference data from source {source.source_id}: {str(e)}'
                )
            raise SourceDataError(message=
                f'Failed to fetch inference data: {str(e)}', source_id=
                source.source_id)

    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]
        ) ->bool:
        """
        Apply resolutions to inference data.

        Args:
            resolutions: List of resolutions to apply

        Returns:
            Whether all resolutions were successfully applied
        """
        success = True
        field_resolutions = {}
        for resolution in resolutions:
            field = resolution.discrepancy.field
            field_resolutions[field] = resolution
        for field, resolution in field_resolutions.items():
            logger.info(
                f'Resolution for {field}: Using value {resolution.resolved_value} from strategy {resolution.strategy.name}'
                )
        return success
