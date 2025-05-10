"""
Resolution strategies for data reconciliation.

This module provides various strategies for resolving discrepancies
between data sources during reconciliation.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union

import numpy as np
import pandas as pd

from common_lib.data_reconciliation.base import (
    DataSource,
    Discrepancy,
    DiscrepancyResolution,
    ReconciliationStrategy,
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class ResolutionStrategyBase(ABC):
    """Base class for resolution strategies."""
    
    def __init__(self, strategy_type: ReconciliationStrategy):
        """
        Initialize resolution strategy.
        
        Args:
            strategy_type: Type of resolution strategy
        """
        self.strategy_type = strategy_type
        
    @abstractmethod
    async def resolve(
        self,
        discrepancy: Discrepancy,
        sources: Dict[str, DataSource],
        **kwargs
    ) -> DiscrepancyResolution:
        """
        Resolve a discrepancy.
        
        Args:
            discrepancy: The discrepancy to resolve
            sources: Dictionary mapping source IDs to DataSource objects
            **kwargs: Additional parameters for resolution
            
        Returns:
            Resolution for the discrepancy
        """
        pass


class SourcePriorityStrategy(ResolutionStrategyBase):
    """Resolve discrepancies by using data from the highest priority source."""
    
    def __init__(self):
        """Initialize source priority strategy."""
        super().__init__(ReconciliationStrategy.SOURCE_PRIORITY)
        
    async def resolve(
        self,
        discrepancy: Discrepancy,
        sources: Dict[str, DataSource],
        **kwargs
    ) -> DiscrepancyResolution:
        """
        Resolve a discrepancy using the highest priority source.
        
        Args:
            discrepancy: The discrepancy to resolve
            sources: Dictionary mapping source IDs to DataSource objects
            **kwargs: Additional parameters for resolution
            
        Returns:
            Resolution for the discrepancy
        """
        # Find the highest priority source that has data for this discrepancy
        highest_priority = -1
        highest_priority_source_id = None
        resolved_value = None
        
        for source_id, value in discrepancy.sources.items():
            if source_id in sources and sources[source_id].priority > highest_priority:
                highest_priority = sources[source_id].priority
                highest_priority_source_id = source_id
                resolved_value = value
                
        if highest_priority_source_id is None:
            logger.warning(f"No valid source found for discrepancy {discrepancy.discrepancy_id}")
            # Use the first available value as fallback
            source_id = next(iter(discrepancy.sources))
            resolved_value = discrepancy.sources[source_id]
            highest_priority_source_id = source_id
            
        return DiscrepancyResolution(
            discrepancy=discrepancy,
            resolved_value=resolved_value,
            strategy=self.strategy_type,
            resolution_source=highest_priority_source_id
        )


class MostRecentStrategy(ResolutionStrategyBase):
    """Resolve discrepancies by using the most recently updated data."""
    
    def __init__(self):
        """Initialize most recent strategy."""
        super().__init__(ReconciliationStrategy.MOST_RECENT)
        
    async def resolve(
        self,
        discrepancy: Discrepancy,
        sources: Dict[str, DataSource],
        **kwargs
    ) -> DiscrepancyResolution:
        """
        Resolve a discrepancy using the most recently updated data.
        
        Args:
            discrepancy: The discrepancy to resolve
            sources: Dictionary mapping source IDs to DataSource objects
            **kwargs: Additional parameters for resolution
                - timestamps: Dict mapping source_id to timestamp
            
        Returns:
            Resolution for the discrepancy
        """
        timestamps = kwargs.get("timestamps", {})
        if not timestamps:
            logger.warning("No timestamps provided for MostRecentStrategy, falling back to SourcePriorityStrategy")
            fallback = SourcePriorityStrategy()
            return await fallback.resolve(discrepancy, sources)
            
        # Find the most recent timestamp
        most_recent_time = datetime.min
        most_recent_source_id = None
        resolved_value = None
        
        for source_id, value in discrepancy.sources.items():
            if source_id in timestamps and timestamps[source_id] > most_recent_time:
                most_recent_time = timestamps[source_id]
                most_recent_source_id = source_id
                resolved_value = value
                
        if most_recent_source_id is None:
            logger.warning(f"No valid timestamp found for discrepancy {discrepancy.discrepancy_id}")
            # Use the first available value as fallback
            source_id = next(iter(discrepancy.sources))
            resolved_value = discrepancy.sources[source_id]
            most_recent_source_id = source_id
            
        return DiscrepancyResolution(
            discrepancy=discrepancy,
            resolved_value=resolved_value,
            strategy=self.strategy_type,
            resolution_source=most_recent_source_id,
            metadata={"timestamp": most_recent_time.isoformat() if isinstance(most_recent_time, datetime) else str(most_recent_time)}
        )


class AverageValuesStrategy(ResolutionStrategyBase):
    """Resolve discrepancies by using the average of all values."""
    
    def __init__(self):
        """Initialize average values strategy."""
        super().__init__(ReconciliationStrategy.AVERAGE)
        
    async def resolve(
        self,
        discrepancy: Discrepancy,
        sources: Dict[str, DataSource],
        **kwargs
    ) -> DiscrepancyResolution:
        """
        Resolve a discrepancy using the average of all values.
        
        Args:
            discrepancy: The discrepancy to resolve
            sources: Dictionary mapping source IDs to DataSource objects
            **kwargs: Additional parameters for resolution
            
        Returns:
            Resolution for the discrepancy
        """
        values = list(discrepancy.sources.values())
        
        # Check if values are numeric
        if not all(isinstance(v, (int, float)) for v in values):
            logger.warning(f"Non-numeric values found for discrepancy {discrepancy.discrepancy_id}, falling back to SourcePriorityStrategy")
            fallback = SourcePriorityStrategy()
            return await fallback.resolve(discrepancy, sources)
            
        # Calculate average
        resolved_value = sum(values) / len(values)
        
        return DiscrepancyResolution(
            discrepancy=discrepancy,
            resolved_value=resolved_value,
            strategy=self.strategy_type,
            metadata={"source_values": discrepancy.sources}
        )


class MedianValuesStrategy(ResolutionStrategyBase):
    """Resolve discrepancies by using the median of all values."""
    
    def __init__(self):
        """Initialize median values strategy."""
        super().__init__(ReconciliationStrategy.MEDIAN)
        
    async def resolve(
        self,
        discrepancy: Discrepancy,
        sources: Dict[str, DataSource],
        **kwargs
    ) -> DiscrepancyResolution:
        """
        Resolve a discrepancy using the median of all values.
        
        Args:
            discrepancy: The discrepancy to resolve
            sources: Dictionary mapping source IDs to DataSource objects
            **kwargs: Additional parameters for resolution
            
        Returns:
            Resolution for the discrepancy
        """
        values = list(discrepancy.sources.values())
        
        # Check if values are numeric
        if not all(isinstance(v, (int, float)) for v in values):
            logger.warning(f"Non-numeric values found for discrepancy {discrepancy.discrepancy_id}, falling back to SourcePriorityStrategy")
            fallback = SourcePriorityStrategy()
            return await fallback.resolve(discrepancy, sources)
            
        # Calculate median
        resolved_value = np.median(values)
        
        return DiscrepancyResolution(
            discrepancy=discrepancy,
            resolved_value=resolved_value,
            strategy=self.strategy_type,
            metadata={"source_values": discrepancy.sources}
        )


class CustomResolutionStrategy(ResolutionStrategyBase):
    """Resolve discrepancies using a custom resolution function."""
    
    def __init__(
        self,
        resolution_func: Callable[[Discrepancy, Dict[str, DataSource], Dict[str, Any]], Any]
    ):
        """
        Initialize custom resolution strategy.
        
        Args:
            resolution_func: Custom function for resolving discrepancies
        """
        super().__init__(ReconciliationStrategy.CUSTOM)
        self.resolution_func = resolution_func
        
    async def resolve(
        self,
        discrepancy: Discrepancy,
        sources: Dict[str, DataSource],
        **kwargs
    ) -> DiscrepancyResolution:
        """
        Resolve a discrepancy using the custom resolution function.
        
        Args:
            discrepancy: The discrepancy to resolve
            sources: Dictionary mapping source IDs to DataSource objects
            **kwargs: Additional parameters for resolution
            
        Returns:
            Resolution for the discrepancy
        """
        try:
            resolved_value = self.resolution_func(discrepancy, sources, kwargs)
            
            return DiscrepancyResolution(
                discrepancy=discrepancy,
                resolved_value=resolved_value,
                strategy=self.strategy_type,
                metadata={"custom_resolution": True}
            )
        except Exception as e:
            logger.error(f"Custom resolution failed: {str(e)}")
            fallback = SourcePriorityStrategy()
            return await fallback.resolve(discrepancy, sources)


class ThresholdBasedStrategy(ResolutionStrategyBase):
    """
    Resolve discrepancies based on thresholds.
    
    If the discrepancy is within the threshold, use one strategy;
    otherwise, use another strategy.
    """
    
    def __init__(
        self,
        threshold: float,
        within_threshold_strategy: ResolutionStrategyBase,
        outside_threshold_strategy: ResolutionStrategyBase
    ):
        """
        Initialize threshold-based strategy.
        
        Args:
            threshold: Threshold for determining which strategy to use
            within_threshold_strategy: Strategy to use when within threshold
            outside_threshold_strategy: Strategy to use when outside threshold
        """
        super().__init__(ReconciliationStrategy.THRESHOLD_BASED)
        self.threshold = threshold
        self.within_threshold_strategy = within_threshold_strategy
        self.outside_threshold_strategy = outside_threshold_strategy
        
    async def resolve(
        self,
        discrepancy: Discrepancy,
        sources: Dict[str, DataSource],
        **kwargs
    ) -> DiscrepancyResolution:
        """
        Resolve a discrepancy based on thresholds.
        
        Args:
            discrepancy: The discrepancy to resolve
            sources: Dictionary mapping source IDs to DataSource objects
            **kwargs: Additional parameters for resolution
            
        Returns:
            Resolution for the discrepancy
        """
        values = list(discrepancy.sources.values())
        
        # Check if values are numeric
        if not all(isinstance(v, (int, float)) for v in values):
            logger.warning(f"Non-numeric values found for discrepancy {discrepancy.discrepancy_id}, using outside threshold strategy")
            return await self.outside_threshold_strategy.resolve(discrepancy, sources, **kwargs)
            
        # Calculate range percentage
        if discrepancy.range_pct is not None and discrepancy.range_pct <= self.threshold:
            # Within threshold, use within_threshold_strategy
            resolution = await self.within_threshold_strategy.resolve(discrepancy, sources, **kwargs)
        else:
            # Outside threshold, use outside_threshold_strategy
            resolution = await self.outside_threshold_strategy.resolve(discrepancy, sources, **kwargs)
            
        # Update metadata to include threshold information
        resolution.metadata.update({
            "threshold": self.threshold,
            "range_pct": discrepancy.range_pct,
            "within_threshold": discrepancy.range_pct is not None and discrepancy.range_pct <= self.threshold
        })
        
        return resolution


def create_resolution_strategy(
    strategy_type: ReconciliationStrategy,
    **kwargs
) -> ResolutionStrategyBase:
    """
    Create a resolution strategy instance.
    
    Args:
        strategy_type: Type of resolution strategy to create
        **kwargs: Additional parameters for the strategy
        
    Returns:
        Resolution strategy instance
    """
    if strategy_type == ReconciliationStrategy.SOURCE_PRIORITY:
        return SourcePriorityStrategy()
    elif strategy_type == ReconciliationStrategy.MOST_RECENT:
        return MostRecentStrategy()
    elif strategy_type == ReconciliationStrategy.AVERAGE:
        return AverageValuesStrategy()
    elif strategy_type == ReconciliationStrategy.MEDIAN:
        return MedianValuesStrategy()
    elif strategy_type == ReconciliationStrategy.CUSTOM:
        if "resolution_func" not in kwargs:
            raise ValueError("resolution_func is required for CustomResolutionStrategy")
        return CustomResolutionStrategy(kwargs["resolution_func"])
    elif strategy_type == ReconciliationStrategy.THRESHOLD_BASED:
        if "threshold" not in kwargs:
            raise ValueError("threshold is required for ThresholdBasedStrategy")
        if "within_threshold_strategy" not in kwargs:
            kwargs["within_threshold_strategy"] = AverageValuesStrategy()
        if "outside_threshold_strategy" not in kwargs:
            kwargs["outside_threshold_strategy"] = SourcePriorityStrategy()
        return ThresholdBasedStrategy(
            kwargs["threshold"],
            kwargs["within_threshold_strategy"],
            kwargs["outside_threshold_strategy"]
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
