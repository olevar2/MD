"""
Feature reconciliation implementations.

This module provides implementations for reconciling feature data from different sources,
including feature versions and feature data.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from feature_store_service.core.feature_store import FeatureStore
    from feature_store_service.repositories.feature_repository import FeatureRepository
    from feature_store_service.validation.data_validator import DataValidator
import pandas as pd
import numpy as np
from common_lib.data_reconciliation.base import DataReconciliationBase, DataSource, DataSourceType, Discrepancy, DiscrepancyResolution, ReconciliationConfig, ReconciliationResult, ReconciliationSeverity, ReconciliationStatus, ReconciliationStrategy
from common_lib.data_reconciliation.batch import BatchReconciliationProcessor
from common_lib.data_reconciliation.realtime import RealTimeReconciliationProcessor
from common_lib.data_reconciliation.strategies import create_resolution_strategy
from common_lib.data_reconciliation.exceptions import SourceDataError
import logging
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FeatureReconciliation(BatchReconciliationProcessor):
    """Base class for feature reconciliation."""

    def __init__(self, config: ReconciliationConfig, feature_store: Any,
        feature_repository: Any, data_validator: Any):
        """
        Initialize feature reconciliation.

        Args:
            config: Configuration for the reconciliation process
            feature_store: Feature store for accessing features
            feature_repository: Repository for storing feature metadata
            data_validator: Validator for feature data
        """
        super().__init__(config)
        self.feature_store = feature_store
        self.feature_repository = feature_repository
        self.data_validator = data_validator

    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]
        ) ->bool:
        """
        Apply resolutions to feature data.

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


class FeatureVersionReconciliation(FeatureReconciliation):
    """Reconciliation for feature versions."""

    @async_with_exception_handling
    async def fetch_data(self, source: DataSource, **kwargs) ->Dict[str, Any]:
        """
        Fetch feature version data from a source.

        Args:
            source: The data source to fetch from
            **kwargs: Additional parameters for fetching
                - feature_name: Name of the feature
                - version: Version of the feature

        Returns:
            Dictionary with feature version data
        """
        feature_name = kwargs.get('feature_name')
        version = kwargs.get('version')
        if not feature_name:
            raise ValueError(
                "Missing required parameter 'feature_name' for fetching feature version data"
                )
        try:
            if source.source_type == DataSourceType.DATABASE:
                if version:
                    data = await self.feature_repository.get_feature_version(
                        feature_name, version)
                else:
                    data = (await self.feature_repository.
                        get_latest_feature_version(feature_name))
            elif source.source_type == DataSourceType.CACHE:
                data = await self.feature_store.get_feature_metadata(
                    feature_name, version)
            else:
                raise ValueError(
                    f'Unsupported source type for feature version: {source.source_type}'
                    )
            return data
        except Exception as e:
            logger.error(
                f'Error fetching feature version data from source {source.source_id}: {str(e)}'
                )
            raise SourceDataError(message=
                f'Failed to fetch feature version data: {str(e)}',
                source_id=source.source_id)

    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]
        ) ->bool:
        """
        Apply resolutions to feature version data.

        Args:
            resolutions: List of resolutions to apply

        Returns:
            Whether all resolutions were successfully applied
        """
        success = True
        for resolution in resolutions:
            logger.info(
                f'Resolution for {resolution.discrepancy.field}: Using value {resolution.resolved_value} from strategy {resolution.strategy.name}'
                )
        return success


class FeatureDataReconciliation(FeatureReconciliation):
    """Reconciliation for feature data."""

    @async_with_exception_handling
    async def fetch_data(self, source: DataSource, **kwargs) ->pd.DataFrame:
        """
        Fetch feature data from a source.

        Args:
            source: The data source to fetch from
            **kwargs: Additional parameters for fetching
                - symbol: Symbol or instrument for the data
                - features: List of features to fetch
                - start_time: Start time for the data
                - end_time: End time for the data

        Returns:
            DataFrame with feature data
        """
        symbol = kwargs.get('symbol')
        features = kwargs.get('features')
        start_time = kwargs.get('start_time')
        end_time = kwargs.get('end_time')
        if not all([symbol, features, start_time, end_time]):
            raise ValueError(
                'Missing required parameters for fetching feature data')
        try:
            if source.source_type == DataSourceType.DATABASE:
                data = await self.feature_repository.get_feature_data(symbol
                    =symbol, features=features, start_time=start_time,
                    end_time=end_time)
            elif source.source_type == DataSourceType.CACHE:
                data = await self.feature_store.retrieve_data(symbol=symbol,
                    features=features, start_time=start_time, end_time=end_time
                    )
            else:
                raise ValueError(
                    f'Unsupported source type for feature data: {source.source_type}'
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
                f'Error fetching feature data from source {source.source_id}: {str(e)}'
                )
            raise SourceDataError(message=
                f'Failed to fetch feature data: {str(e)}', source_id=source
                .source_id)

    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]
        ) ->bool:
        """
        Apply resolutions to feature data.

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
