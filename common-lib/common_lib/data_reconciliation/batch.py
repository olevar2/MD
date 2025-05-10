"""
Batch reconciliation implementations.

This module provides implementations for batch reconciliation processes,
including historical data reconciliation and bulk data reconciliation.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import pandas as pd
import numpy as np

from common_lib.data_reconciliation.base import (
    DataReconciliationBase,
    DataSource,
    DataSourceType,
    Discrepancy,
    DiscrepancyResolution,
    ReconciliationConfig,
    ReconciliationResult,
    ReconciliationSeverity,
    ReconciliationStatus,
    ReconciliationStrategy,
)
from common_lib.data_reconciliation.strategies import create_resolution_strategy
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class BatchReconciliationProcessor(DataReconciliationBase):
    """Base class for batch reconciliation processes."""
    
    def __init__(
        self,
        config: ReconciliationConfig,
        data_fetchers: Dict[str, Callable] = None
    ):
        """
        Initialize batch reconciliation processor.
        
        Args:
            config: Configuration for the reconciliation process
            data_fetchers: Dictionary mapping source IDs to data fetcher functions
        """
        super().__init__(config)
        self.data_fetchers = data_fetchers or {}
        
    async def fetch_data(self, source: DataSource, **kwargs) -> Any:
        """
        Fetch data from a source.
        
        Args:
            source: The data source to fetch from
            **kwargs: Additional parameters for fetching
            
        Returns:
            The fetched data
        """
        if source.source_id in self.data_fetchers:
            fetcher = self.data_fetchers[source.source_id]
            return await fetcher(**kwargs)
        else:
            logger.warning(f"No data fetcher found for source {source.source_id}")
            return None
            
    async def compare_data(self, data_map: Dict[str, Any]) -> List[Discrepancy]:
        """
        Compare data from different sources to identify discrepancies.
        
        Args:
            data_map: Dictionary mapping source IDs to their data
            
        Returns:
            List of identified discrepancies
        """
        discrepancies = []
        
        # Filter out None values
        valid_data_map = {k: v for k, v in data_map.items() if v is not None}
        
        if len(valid_data_map) < 2:
            logger.warning("Not enough valid data sources for comparison")
            return discrepancies
            
        # Determine the type of data we're working with
        first_source_id = next(iter(valid_data_map))
        first_data = valid_data_map[first_source_id]
        
        if isinstance(first_data, pd.DataFrame):
            return await self._compare_dataframes(valid_data_map)
        elif isinstance(first_data, dict):
            return await self._compare_dicts(valid_data_map)
        elif isinstance(first_data, list):
            return await self._compare_lists(valid_data_map)
        else:
            logger.warning(f"Unsupported data type for comparison: {type(first_data)}")
            return discrepancies
            
    async def _compare_dataframes(self, data_map: Dict[str, pd.DataFrame]) -> List[Discrepancy]:
        """Compare DataFrame data from different sources."""
        discrepancies = []
        
        # Get common columns across all DataFrames
        common_columns = set()
        for source_id, df in data_map.items():
            if common_columns:
                common_columns &= set(df.columns)
            else:
                common_columns = set(df.columns)
                
        # Filter out ignored fields
        if self.config.fields_to_ignore:
            common_columns -= set(self.config.fields_to_ignore)
            
        # Filter to specific fields if configured
        if self.config.fields_to_reconcile:
            common_columns &= set(self.config.fields_to_reconcile)
            
        if not common_columns:
            logger.warning("No common columns found for comparison")
            return discrepancies
            
        # Get common index values
        common_indices = set()
        for source_id, df in data_map.items():
            if common_indices:
                common_indices &= set(df.index)
            else:
                common_indices = set(df.index)
                
        if not common_indices:
            logger.warning("No common indices found for comparison")
            return discrepancies
            
        # Compare values for each common column and index
        for column in common_columns:
            for idx in common_indices:
                values = {}
                for source_id, df in data_map.items():
                    values[source_id] = df.loc[idx, column]
                    
                # Check if there are discrepancies
                unique_values = set(values.values())
                if len(unique_values) > 1:
                    # For numeric values, check if within tolerance
                    if all(isinstance(v, (int, float)) for v in values.values()):
                        min_val = min(values.values())
                        max_val = max(values.values())
                        if max_val - min_val <= self.config.tolerance:
                            continue
                            
                    # Create discrepancy
                    discrepancy = Discrepancy(
                        field=f"{column}[{idx}]",
                        sources=values,
                        severity=self._determine_severity(values)
                    )
                    discrepancies.append(discrepancy)
                    
        return discrepancies
        
    async def _compare_dicts(self, data_map: Dict[str, Dict]) -> List[Discrepancy]:
        """Compare dictionary data from different sources."""
        discrepancies = []
        
        # Get common keys across all dictionaries
        common_keys = set()
        for source_id, data_dict in data_map.items():
            if common_keys:
                common_keys &= set(data_dict.keys())
            else:
                common_keys = set(data_dict.keys())
                
        # Filter out ignored fields
        if self.config.fields_to_ignore:
            common_keys -= set(self.config.fields_to_ignore)
            
        # Filter to specific fields if configured
        if self.config.fields_to_reconcile:
            common_keys &= set(self.config.fields_to_reconcile)
            
        if not common_keys:
            logger.warning("No common keys found for comparison")
            return discrepancies
            
        # Compare values for each common key
        for key in common_keys:
            values = {}
            for source_id, data_dict in data_map.items():
                values[source_id] = data_dict[key]
                
            # Check if there are discrepancies
            unique_values = set()
            for v in values.values():
                # Handle non-hashable types
                if isinstance(v, (list, dict)):
                    unique_values.add(str(v))
                else:
                    unique_values.add(v)
                    
            if len(unique_values) > 1:
                # For numeric values, check if within tolerance
                if all(isinstance(v, (int, float)) for v in values.values()):
                    min_val = min(values.values())
                    max_val = max(values.values())
                    if max_val - min_val <= self.config.tolerance:
                        continue
                        
                # Create discrepancy
                discrepancy = Discrepancy(
                    field=key,
                    sources=values,
                    severity=self._determine_severity(values)
                )
                discrepancies.append(discrepancy)
                
        return discrepancies
        
    async def _compare_lists(self, data_map: Dict[str, List]) -> List[Discrepancy]:
        """Compare list data from different sources."""
        discrepancies = []
        
        # Convert lists to dictionaries with indices as keys
        dict_data_map = {}
        for source_id, data_list in data_map.items():
            dict_data_map[source_id] = {i: item for i, item in enumerate(data_list)}
            
        # Use the dictionary comparison method
        return await self._compare_dicts(dict_data_map)
        
    def _determine_severity(self, values: Dict[str, Any]) -> ReconciliationSeverity:
        """Determine the severity of a discrepancy based on the values."""
        # For numeric values, use the percentage difference
        if all(isinstance(v, (int, float)) for v in values.values()):
            min_val = min(values.values())
            max_val = max(values.values())
            
            # Avoid division by zero
            if abs(min_val) < 1e-10:
                if abs(max_val) < 1e-10:
                    return ReconciliationSeverity.LOW
                return ReconciliationSeverity.HIGH
                
            pct_diff = (max_val - min_val) / abs(min_val) * 100
            
            if pct_diff > 10.0:
                return ReconciliationSeverity.CRITICAL
            elif pct_diff > 5.0:
                return ReconciliationSeverity.HIGH
            elif pct_diff > 1.0:
                return ReconciliationSeverity.MEDIUM
            else:
                return ReconciliationSeverity.LOW
        else:
            # For non-numeric values, use the number of unique values
            unique_values = set()
            for v in values.values():
                # Handle non-hashable types
                if isinstance(v, (list, dict)):
                    unique_values.add(str(v))
                else:
                    unique_values.add(v)
                    
            if len(unique_values) == len(values):
                return ReconciliationSeverity.HIGH
            else:
                return ReconciliationSeverity.MEDIUM
                
    async def resolve_discrepancies(self, discrepancies: List[Discrepancy]) -> List[DiscrepancyResolution]:
        """
        Resolve discrepancies using the configured strategy.
        
        Args:
            discrepancies: List of discrepancies to resolve
            
        Returns:
            List of discrepancy resolutions
        """
        resolutions = []
        
        # Create a dictionary of sources for easy lookup
        sources = {source.source_id: source for source in self.config.sources}
        
        # Create the resolution strategy
        strategy = create_resolution_strategy(self.config.strategy)
        
        # Resolve each discrepancy
        for discrepancy in discrepancies:
            resolution = await strategy.resolve(discrepancy, sources)
            resolutions.append(resolution)
            
        return resolutions
        
    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]) -> bool:
        """
        Apply resolutions to the data.
        
        Args:
            resolutions: List of resolutions to apply
            
        Returns:
            Whether all resolutions were successfully applied
        """
        # In the base implementation, we just log the resolutions
        # Subclasses should override this to actually apply the resolutions
        for resolution in resolutions:
            logger.info(
                f"Resolution for {resolution.discrepancy.field}: "
                f"Using value {resolution.resolved_value} from strategy {resolution.strategy.name}"
            )
            
        return True


class HistoricalDataReconciliation(BatchReconciliationProcessor):
    """Reconciliation for historical time series data."""
    
    async def fetch_data(self, source: DataSource, **kwargs) -> pd.DataFrame:
        """
        Fetch historical data from a source.
        
        Args:
            source: The data source to fetch from
            **kwargs: Additional parameters for fetching
                - start_date: Start date for historical data
                - end_date: End date for historical data
                - symbol: Symbol or instrument for the data
                - timeframe: Timeframe for the data
            
        Returns:
            DataFrame with historical data
        """
        if source.source_id in self.data_fetchers:
            fetcher = self.data_fetchers[source.source_id]
            return await fetcher(**kwargs)
        else:
            logger.warning(f"No data fetcher found for source {source.source_id}")
            return None
            
    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]) -> bool:
        """
        Apply resolutions to historical data.
        
        Args:
            resolutions: List of resolutions to apply
            
        Returns:
            Whether all resolutions were successfully applied
        """
        # Group resolutions by field
        field_resolutions = {}
        for resolution in resolutions:
            field = resolution.discrepancy.field
            field_resolutions[field] = resolution
            
        # Log the resolutions
        for field, resolution in field_resolutions.items():
            logger.info(
                f"Resolution for {field}: "
                f"Using value {resolution.resolved_value} from strategy {resolution.strategy.name}"
            )
            
        # In a real implementation, we would update the data store with the resolved values
        # For now, we just return True
        return True


class BulkDataReconciliation(BatchReconciliationProcessor):
    """Reconciliation for bulk data sets."""
    
    async def fetch_data(self, source: DataSource, **kwargs) -> Any:
        """
        Fetch bulk data from a source.
        
        Args:
            source: The data source to fetch from
            **kwargs: Additional parameters for fetching
                - data_type: Type of data to fetch
                - filters: Filters to apply to the data
            
        Returns:
            The fetched data
        """
        if source.source_id in self.data_fetchers:
            fetcher = self.data_fetchers[source.source_id]
            return await fetcher(**kwargs)
        else:
            logger.warning(f"No data fetcher found for source {source.source_id}")
            return None
            
    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]) -> bool:
        """
        Apply resolutions to bulk data.
        
        Args:
            resolutions: List of resolutions to apply
            
        Returns:
            Whether all resolutions were successfully applied
        """
        # Group resolutions by field
        field_resolutions = {}
        for resolution in resolutions:
            field = resolution.discrepancy.field
            field_resolutions[field] = resolution
            
        # Log the resolutions
        for field, resolution in field_resolutions.items():
            logger.info(
                f"Resolution for {field}: "
                f"Using value {resolution.resolved_value} from strategy {resolution.strategy.name}"
            )
            
        # In a real implementation, we would update the data store with the resolved values
        # For now, we just return True
        return True
