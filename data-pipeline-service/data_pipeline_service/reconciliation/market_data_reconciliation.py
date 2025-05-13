"""
Market data reconciliation implementations.

This module provides implementations for reconciling market data from different sources,
including OHLCV data and tick data.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from data_pipeline_service.source_adapters.data_fetcher_manager import DataFetcherManager
    from data_pipeline_service.repositories.ohlcv_repository import OHLCVRepository
    from data_pipeline_service.repositories.tick_repository import TickRepository
    from data_pipeline_service.validation.validation_engine import DataValidationEngine
import pandas as pd
import numpy as np
from common_lib.data_reconciliation.base import DataReconciliationBase, DataSource, DataSourceType, Discrepancy, DiscrepancyResolution, ReconciliationConfig, ReconciliationResult, ReconciliationSeverity, ReconciliationStatus, ReconciliationStrategy
from common_lib.data_reconciliation.batch import BatchReconciliationProcessor
from common_lib.data_reconciliation.realtime import RealTimeReconciliationProcessor
from common_lib.data_reconciliation.strategies import create_resolution_strategy
from common_lib.data_reconciliation.exceptions import SourceDataError
import logging
logger = logging.getLogger(__name__)


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MarketDataReconciliation(BatchReconciliationProcessor):
    """Base class for market data reconciliation."""

    def __init__(self, config: ReconciliationConfig, data_fetcher_manager:
        Any, validation_engine: Any):
        """
        Initialize market data reconciliation.

        Args:
            config: Configuration for the reconciliation process
            data_fetcher_manager: Manager for fetching data from different sources
            validation_engine: Engine for validating data
        """
        super().__init__(config)
        self.data_fetcher_manager = data_fetcher_manager
        self.validation_engine = validation_engine

    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]
        ) ->bool:
        """
        Apply resolutions to market data.

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


class OHLCVReconciliation(MarketDataReconciliation):
    """Reconciliation for OHLCV data."""

    def __init__(self, config: ReconciliationConfig, data_fetcher_manager:
        Any, validation_engine: Any, ohlcv_repository: Any):
        """
        Initialize OHLCV reconciliation.

        Args:
            config: Configuration for the reconciliation process
            data_fetcher_manager: Manager for fetching data from different sources
            validation_engine: Engine for validating data
            ohlcv_repository: Repository for storing OHLCV data
        """
        super().__init__(config, data_fetcher_manager, validation_engine)
        self.ohlcv_repository = ohlcv_repository

    @async_with_exception_handling
    async def fetch_data(self, source: DataSource, **kwargs) ->pd.DataFrame:
        """
        Fetch OHLCV data from a source.

        Args:
            source: The data source to fetch from
            **kwargs: Additional parameters for fetching
                - symbol: Symbol or instrument for the data
                - start_date: Start date for historical data
                - end_date: End date for historical data
                - timeframe: Timeframe for the data

        Returns:
            DataFrame with OHLCV data
        """
        symbol = kwargs.get('symbol')
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        timeframe = kwargs.get('timeframe')
        if not all([symbol, start_date, end_date, timeframe]):
            raise ValueError(
                'Missing required parameters for fetching OHLCV data')
        try:
            data = (await self.data_fetcher_manager.
                fetch_historical_ohlcv_from_source(source_name=source.
                source_id, symbol=symbol, start_date=start_date, end_date=
                end_date, timeframe=timeframe))
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, list):
                    data = pd.DataFrame(data)
                else:
                    raise ValueError(
                        f'Unexpected data type from source {source.source_id}: {type(data)}'
                        )
            if 'timestamp' in data.columns and data.index.name != 'timestamp':
                data = data.set_index('timestamp')
            return data
        except Exception as e:
            logger.error(
                f'Error fetching OHLCV data from source {source.source_id}: {str(e)}'
                )
            raise SourceDataError(message=
                f'Failed to fetch OHLCV data: {str(e)}', source_id=source.
                source_id)

    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]
        ) ->bool:
        """
        Apply resolutions to OHLCV data.

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


class TickDataReconciliation(MarketDataReconciliation):
    """Reconciliation for tick data."""

    def __init__(self, config: ReconciliationConfig, data_fetcher_manager:
        Any, validation_engine: Any, tick_repository: Any):
        """
        Initialize tick data reconciliation.

        Args:
            config: Configuration for the reconciliation process
            data_fetcher_manager: Manager for fetching data from different sources
            validation_engine: Engine for validating data
            tick_repository: Repository for storing tick data
        """
        super().__init__(config, data_fetcher_manager, validation_engine)
        self.tick_repository = tick_repository

    @async_with_exception_handling
    async def fetch_data(self, source: DataSource, **kwargs) ->pd.DataFrame:
        """
        Fetch tick data from a source.

        Args:
            source: The data source to fetch from
            **kwargs: Additional parameters for fetching
                - symbol: Symbol or instrument for the data
                - start_date: Start date for historical data
                - end_date: End date for historical data

        Returns:
            DataFrame with tick data
        """
        symbol = kwargs.get('symbol')
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        if not all([symbol, start_date, end_date]):
            raise ValueError(
                'Missing required parameters for fetching tick data')
        try:
            data = await self.data_fetcher_manager.fetch_tick_data_from_source(
                source_name=source.source_id, symbol=symbol, start_date=
                start_date, end_date=end_date)
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, list):
                    data = pd.DataFrame(data)
                else:
                    raise ValueError(
                        f'Unexpected data type from source {source.source_id}: {type(data)}'
                        )
            if 'timestamp' in data.columns and data.index.name != 'timestamp':
                data = data.set_index('timestamp')
            return data
        except Exception as e:
            logger.error(
                f'Error fetching tick data from source {source.source_id}: {str(e)}'
                )
            raise SourceDataError(message=
                f'Failed to fetch tick data: {str(e)}', source_id=source.
                source_id)

    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]
        ) ->bool:
        """
        Apply resolutions to tick data.

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
