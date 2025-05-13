"""
Reconciliation Service for the Data Pipeline Service.

This service provides functionality for reconciling market data from different sources,
including OHLCV data and tick data.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import uuid
from data_pipeline_service.config.settings import settings
from data_pipeline_service.reconciliation.market_data_reconciliation import OHLCVReconciliation, TickDataReconciliation
from data_pipeline_service.source_adapters.data_fetcher_manager import DataFetcherManager
from data_pipeline_service.repositories.ohlcv_repository import OHLCVRepository
from data_pipeline_service.repositories.tick_repository import TickRepository
from data_pipeline_service.validation.validation_engine import DataValidationEngine
from common_lib.data_reconciliation import DataSource, DataSourceType, ReconciliationConfig, ReconciliationResult, ReconciliationStatus, ReconciliationSeverity, ReconciliationStrategy
from common_lib.exceptions import DataFetchError, DataValidationError, ReconciliationError
logger = logging.getLogger(__name__)


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ReconciliationService:
    """Service for reconciling market data from different sources."""

    def __init__(self):
        """Initialize the reconciliation service."""
        self.data_fetcher_manager = DataFetcherManager()
        self.validation_engine = DataValidationEngine()
        self.ohlcv_repository = OHLCVRepository()
        self.tick_repository = TickRepository()
        self.reconciliation_results = {}

    @async_with_exception_handling
    async def reconcile_ohlcv_data(self, symbol: str, start_date: datetime,
        end_date: datetime, timeframe: str, strategy:
        ReconciliationStrategy=ReconciliationStrategy.SOURCE_PRIORITY,
        tolerance: float=0.0001, auto_resolve: bool=True,
        notification_threshold: ReconciliationSeverity=
        ReconciliationSeverity.HIGH) ->ReconciliationResult:
        """
        Reconcile OHLCV data.

        Args:
            symbol: Symbol or instrument for the data
            start_date: Start date for data reconciliation
            end_date: End date for data reconciliation
            timeframe: Timeframe for the data
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
            primary_source = DataSource(source_id='primary_provider', name=
                'Primary Data Provider', source_type=DataSourceType.API,
                priority=2)
            secondary_source = DataSource(source_id='secondary_provider',
                name='Secondary Data Provider', source_type=DataSourceType.
                API, priority=1)
            config = ReconciliationConfig(sources=[primary_source,
                secondary_source], strategy=strategy, tolerance=tolerance,
                auto_resolve=auto_resolve, notification_threshold=
                notification_threshold)
            reconciliation = OHLCVReconciliation(config=config,
                data_fetcher_manager=self.data_fetcher_manager,
                validation_engine=self.validation_engine, ohlcv_repository=
                self.ohlcv_repository)
            reconciliation_id = str(uuid.uuid4())
            logger.info(
                f'Starting OHLCV data reconciliation for symbol {symbol}, timeframe {timeframe}, from {start_date} to {end_date}, reconciliation ID: {reconciliation_id}'
                )
            result = await reconciliation.reconcile(symbol=symbol,
                start_date=start_date, end_date=end_date, timeframe=timeframe)
            logger.info(
                f'Completed OHLCV data reconciliation for symbol {symbol}, timeframe {timeframe}, reconciliation ID: {result.reconciliation_id}, status: {result.status.name}, discrepancies: {result.discrepancy_count}, resolutions: {result.resolution_count}'
                )
            self.reconciliation_results[result.reconciliation_id] = result
            return result
        except DataFetchError as e:
            logger.error(
                f'Data fetch error during OHLCV data reconciliation: {str(e)}')
            raise
        except DataValidationError as e:
            logger.error(
                f'Data validation error during OHLCV data reconciliation: {str(e)}'
                )
            raise
        except ReconciliationError as e:
            logger.error(
                f'Reconciliation error during OHLCV data reconciliation: {str(e)}'
                )
            raise
        except Exception as e:
            logger.exception(
                f'Unexpected error during OHLCV data reconciliation: {str(e)}')
            raise ReconciliationError(
                f'Unexpected error during OHLCV data reconciliation: {str(e)}',
                details={'symbol': symbol, 'timeframe': timeframe})

    @async_with_exception_handling
    async def reconcile_tick_data(self, symbol: str, start_date: datetime,
        end_date: datetime, strategy: ReconciliationStrategy=
        ReconciliationStrategy.SOURCE_PRIORITY, tolerance: float=0.0001,
        auto_resolve: bool=True, notification_threshold:
        ReconciliationSeverity=ReconciliationSeverity.HIGH
        ) ->ReconciliationResult:
        """
        Reconcile tick data.

        Args:
            symbol: Symbol or instrument for the data
            start_date: Start date for data reconciliation
            end_date: End date for data reconciliation
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
            primary_source = DataSource(source_id='primary_provider', name=
                'Primary Data Provider', source_type=DataSourceType.API,
                priority=2)
            secondary_source = DataSource(source_id='secondary_provider',
                name='Secondary Data Provider', source_type=DataSourceType.
                API, priority=1)
            config = ReconciliationConfig(sources=[primary_source,
                secondary_source], strategy=strategy, tolerance=tolerance,
                auto_resolve=auto_resolve, notification_threshold=
                notification_threshold)
            reconciliation = TickDataReconciliation(config=config,
                data_fetcher_manager=self.data_fetcher_manager,
                validation_engine=self.validation_engine, tick_repository=
                self.tick_repository)
            reconciliation_id = str(uuid.uuid4())
            logger.info(
                f'Starting tick data reconciliation for symbol {symbol}, from {start_date} to {end_date}, reconciliation ID: {reconciliation_id}'
                )
            result = await reconciliation.reconcile(symbol=symbol,
                start_date=start_date, end_date=end_date)
            logger.info(
                f'Completed tick data reconciliation for symbol {symbol}, reconciliation ID: {result.reconciliation_id}, status: {result.status.name}, discrepancies: {result.discrepancy_count}, resolutions: {result.resolution_count}'
                )
            self.reconciliation_results[result.reconciliation_id] = result
            return result
        except DataFetchError as e:
            logger.error(
                f'Data fetch error during tick data reconciliation: {str(e)}')
            raise
        except DataValidationError as e:
            logger.error(
                f'Data validation error during tick data reconciliation: {str(e)}'
                )
            raise
        except ReconciliationError as e:
            logger.error(
                f'Reconciliation error during tick data reconciliation: {str(e)}'
                )
            raise
        except Exception as e:
            logger.exception(
                f'Unexpected error during tick data reconciliation: {str(e)}')
            raise ReconciliationError(
                f'Unexpected error during tick data reconciliation: {str(e)}',
                details={'symbol': symbol})

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
        if result:
            logger.info(
                f'Retrieved reconciliation status for ID {reconciliation_id}, status: {result.status.name}'
                )
        else:
            logger.warning(f'Reconciliation ID {reconciliation_id} not found')
        return result
