"""
Real-time reconciliation implementations.

This module provides implementations for real-time reconciliation processes,
including streaming data reconciliation and event-based reconciliation.
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


class RealTimeReconciliationProcessor(DataReconciliationBase):
    """Base class for real-time reconciliation processes."""
    
    def __init__(
        self,
        config: ReconciliationConfig,
        data_fetchers: Dict[str, Callable] = None,
        data_updaters: Dict[str, Callable] = None
    ):
        """
        Initialize real-time reconciliation processor.
        
        Args:
            config: Configuration for the reconciliation process
            data_fetchers: Dictionary mapping source IDs to data fetcher functions
            data_updaters: Dictionary mapping source IDs to data updater functions
        """
        super().__init__(config)
        self.data_fetchers = data_fetchers or {}
        self.data_updaters = data_updaters or {}
        self.latest_data = {}
        self.latest_timestamps = {}
        
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
            data = await fetcher(**kwargs)
            
            # Store the latest data and timestamp
            self.latest_data[source.source_id] = data
            self.latest_timestamps[source.source_id] = datetime.utcnow()
            
            return data
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
        
        if isinstance(first_data, dict):
            return await self._compare_dicts(valid_data_map)
        elif isinstance(first_data, list):
            return await self._compare_lists(valid_data_map)
        else:
            logger.warning(f"Unsupported data type for comparison: {type(first_data)}")
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
            # For most recent strategy, pass the timestamps
            if self.config.strategy == ReconciliationStrategy.MOST_RECENT:
                resolution = await strategy.resolve(
                    discrepancy,
                    sources,
                    timestamps=self.latest_timestamps
                )
            else:
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
        success = True
        
        for resolution in resolutions:
            # Log the resolution
            logger.info(
                f"Resolution for {resolution.discrepancy.field}: "
                f"Using value {resolution.resolved_value} from strategy {resolution.strategy.name}"
            )
            
            # Apply the resolution to the appropriate sources
            for source_id in resolution.discrepancy.sources.keys():
                if source_id in self.data_updaters:
                    try:
                        updater = self.data_updaters[source_id]
                        await updater(
                            field=resolution.discrepancy.field,
                            value=resolution.resolved_value
                        )
                    except Exception as e:
                        logger.error(f"Failed to update source {source_id}: {str(e)}")
                        success = False
                        
        return success


class StreamingDataReconciliation(RealTimeReconciliationProcessor):
    """Reconciliation for streaming data sources."""
    
    def __init__(
        self,
        config: ReconciliationConfig,
        data_fetchers: Dict[str, Callable] = None,
        data_updaters: Dict[str, Callable] = None,
        reconciliation_interval: float = 1.0
    ):
        """
        Initialize streaming data reconciliation.
        
        Args:
            config: Configuration for the reconciliation process
            data_fetchers: Dictionary mapping source IDs to data fetcher functions
            data_updaters: Dictionary mapping source IDs to data updater functions
            reconciliation_interval: Interval in seconds between reconciliations
        """
        super().__init__(config, data_fetchers, data_updaters)
        self.reconciliation_interval = reconciliation_interval
        self.running = False
        self.task = None
        
    async def start(self, **kwargs) -> None:
        """
        Start the streaming reconciliation process.
        
        Args:
            **kwargs: Additional parameters for reconciliation
        """
        if self.running:
            logger.warning("Streaming reconciliation already running")
            return
            
        self.running = True
        self.task = asyncio.create_task(self._reconciliation_loop(**kwargs))
        
    async def stop(self) -> None:
        """Stop the streaming reconciliation process."""
        if not self.running:
            logger.warning("Streaming reconciliation not running")
            return
            
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
            
    async def _reconciliation_loop(self, **kwargs) -> None:
        """Main reconciliation loop."""
        while self.running:
            try:
                # Perform reconciliation
                await self.reconcile(**kwargs)
                
                # Wait for the next interval
                await asyncio.sleep(self.reconciliation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in reconciliation loop: {str(e)}")
                # Wait a bit before retrying
                await asyncio.sleep(1.0)


class EventBasedReconciliation(RealTimeReconciliationProcessor):
    """Reconciliation triggered by events."""
    
    def __init__(
        self,
        config: ReconciliationConfig,
        data_fetchers: Dict[str, Callable] = None,
        data_updaters: Dict[str, Callable] = None,
        event_handlers: Dict[str, Callable] = None
    ):
        """
        Initialize event-based reconciliation.
        
        Args:
            config: Configuration for the reconciliation process
            data_fetchers: Dictionary mapping source IDs to data fetcher functions
            data_updaters: Dictionary mapping source IDs to data updater functions
            event_handlers: Dictionary mapping event types to handler functions
        """
        super().__init__(config, data_fetchers, data_updaters)
        self.event_handlers = event_handlers or {}
        
    async def handle_event(self, event_type: str, event_data: Any) -> Optional[ReconciliationResult]:
        """
        Handle an event that may trigger reconciliation.
        
        Args:
            event_type: Type of the event
            event_data: Data associated with the event
            
        Returns:
            Reconciliation result if reconciliation was performed, None otherwise
        """
        if event_type in self.event_handlers:
            handler = self.event_handlers[event_type]
            kwargs = await handler(event_data)
            
            if kwargs is not None:
                # Perform reconciliation with the parameters from the handler
                return await self.reconcile(**kwargs)
                
        return None
