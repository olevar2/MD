"""
Reconciliation Service for the Feature Store Service.

This service provides functionality for reconciling feature data from different sources,
including feature versions and feature data.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import uuid

from feature_store_service.config.settings import settings
from feature_store_service.reconciliation.feature_reconciliation import (
    FeatureVersionReconciliation,
    FeatureDataReconciliation,
)
from feature_store_service.core.feature_store import FeatureStore
from feature_store_service.repositories.feature_repository import FeatureRepository
from feature_store_service.validation.data_validator import DataValidator

from common_lib.data_reconciliation import (
    DataSource,
    DataSourceType,
    ReconciliationConfig,
    ReconciliationResult,
    ReconciliationStatus,
    ReconciliationSeverity,
    ReconciliationStrategy,
)
from common_lib.exceptions import (
    DataFetchError,
    DataValidationError,
    ReconciliationError,
)

logger = logging.getLogger(__name__)


class ReconciliationService:
    """Service for reconciling feature data from different sources."""

    def __init__(self):
        """Initialize the reconciliation service."""
        self.feature_store = FeatureStore()
        self.feature_repository = FeatureRepository()
        self.data_validator = DataValidator()
        self.reconciliation_results = {}  # Store reconciliation results for retrieval

    async def reconcile_feature_version(
        self,
        feature_name: str,
        version: Optional[str] = None,
        strategy: ReconciliationStrategy = ReconciliationStrategy.SOURCE_PRIORITY,
        tolerance: float = 0.0001,
        auto_resolve: bool = True,
        notification_threshold: ReconciliationSeverity = ReconciliationSeverity.HIGH
    ) -> ReconciliationResult:
        """
        Reconcile feature version data.

        Args:
            feature_name: Name of the feature
            version: Version of the feature
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
            # Define data sources
            database_source = DataSource(
                source_id="database",
                name="Feature Repository Database",
                source_type=DataSourceType.DATABASE,
                priority=1
            )

            cache_source = DataSource(
                source_id="cache",
                name="Feature Store Cache",
                source_type=DataSourceType.CACHE,
                priority=2
            )

            # Create configuration
            config = ReconciliationConfig(
                sources=[database_source, cache_source],
                strategy=strategy,
                tolerance=tolerance,
                auto_resolve=auto_resolve,
                notification_threshold=notification_threshold
            )

            # Create reconciliation processor
            reconciliation = FeatureVersionReconciliation(
                config=config,
                feature_store=self.feature_store,
                feature_repository=self.feature_repository,
                data_validator=self.data_validator
            )

            # Log reconciliation start
            reconciliation_id = str(uuid.uuid4())
            logger.info(
                f"Starting feature version reconciliation for feature {feature_name}, version {version}, "
                f"reconciliation ID: {reconciliation_id}"
            )

            # Perform reconciliation
            result = await reconciliation.reconcile(
                feature_name=feature_name,
                version=version
            )

            # Log reconciliation result
            logger.info(
                f"Completed feature version reconciliation for feature {feature_name}, version {version}, "
                f"reconciliation ID: {result.reconciliation_id}, "
                f"status: {result.status.name}, "
                f"discrepancies: {result.discrepancy_count}, "
                f"resolutions: {result.resolution_count}"
            )

            # Store the result for later retrieval
            self.reconciliation_results[result.reconciliation_id] = result

            return result

        except DataFetchError as e:
            logger.error(f"Data fetch error during feature version reconciliation: {str(e)}")
            raise
        except DataValidationError as e:
            logger.error(f"Data validation error during feature version reconciliation: {str(e)}")
            raise
        except ReconciliationError as e:
            logger.error(f"Reconciliation error during feature version reconciliation: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during feature version reconciliation: {str(e)}")
            raise ReconciliationError(
                f"Unexpected error during feature version reconciliation: {str(e)}",
                details={"feature_name": feature_name, "version": version}
            )

    async def reconcile_feature_data(
        self,
        symbol: str,
        features: List[str],
        start_time: datetime,
        end_time: datetime,
        strategy: ReconciliationStrategy = ReconciliationStrategy.SOURCE_PRIORITY,
        tolerance: float = 0.0001,
        auto_resolve: bool = True,
        notification_threshold: ReconciliationSeverity = ReconciliationSeverity.HIGH
    ) -> ReconciliationResult:
        """
        Reconcile feature data.

        Args:
            symbol: Symbol or instrument for the data
            features: List of features to reconcile
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
            # Define data sources
            database_source = DataSource(
                source_id="database",
                name="Feature Repository Database",
                source_type=DataSourceType.DATABASE,
                priority=1
            )

            cache_source = DataSource(
                source_id="cache",
                name="Feature Store Cache",
                source_type=DataSourceType.CACHE,
                priority=2
            )

            # Create configuration
            config = ReconciliationConfig(
                sources=[database_source, cache_source],
                strategy=strategy,
                tolerance=tolerance,
                auto_resolve=auto_resolve,
                notification_threshold=notification_threshold
            )

            # Create reconciliation processor
            reconciliation = FeatureDataReconciliation(
                config=config,
                feature_store=self.feature_store,
                feature_repository=self.feature_repository,
                data_validator=self.data_validator
            )

            # Log reconciliation start
            reconciliation_id = str(uuid.uuid4())
            logger.info(
                f"Starting feature data reconciliation for symbol {symbol}, features {features}, "
                f"reconciliation ID: {reconciliation_id}"
            )

            # Perform reconciliation
            result = await reconciliation.reconcile(
                symbol=symbol,
                features=features,
                start_time=start_time,
                end_time=end_time
            )

            # Log reconciliation result
            logger.info(
                f"Completed feature data reconciliation for symbol {symbol}, "
                f"reconciliation ID: {result.reconciliation_id}, "
                f"status: {result.status.name}, "
                f"discrepancies: {result.discrepancy_count}, "
                f"resolutions: {result.resolution_count}"
            )

            # Store the result for later retrieval
            self.reconciliation_results[result.reconciliation_id] = result

            return result

        except DataFetchError as e:
            logger.error(f"Data fetch error during feature data reconciliation: {str(e)}")
            raise
        except DataValidationError as e:
            logger.error(f"Data validation error during feature data reconciliation: {str(e)}")
            raise
        except ReconciliationError as e:
            logger.error(f"Reconciliation error during feature data reconciliation: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during feature data reconciliation: {str(e)}")
            raise ReconciliationError(
                f"Unexpected error during feature data reconciliation: {str(e)}",
                details={"symbol": symbol, "features": features}
            )

    async def get_reconciliation_status(self, reconciliation_id: str) -> Optional[ReconciliationResult]:
        """
        Get the status of a reconciliation process.

        Args:
            reconciliation_id: ID of the reconciliation process

        Returns:
            Results of the reconciliation process, or None if not found
        """
        result = self.reconciliation_results.get(reconciliation_id)

        if result:
            logger.info(f"Retrieved reconciliation status for ID {reconciliation_id}, status: {result.status.name}")
        else:
            logger.warning(f"Reconciliation ID {reconciliation_id} not found")

        return result
