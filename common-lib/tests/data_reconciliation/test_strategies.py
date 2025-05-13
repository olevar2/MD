"""
Tests for the data reconciliation strategies module.
"""

import unittest
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from common_lib.data_reconciliation.base import (
    DataSource,
    DataSourceType,
    Discrepancy,
    DiscrepancyResolution,
    ReconciliationSeverity,
    ReconciliationStrategy,
)
from common_lib.data_reconciliation.strategies import (
    ResolutionStrategyBase,
    SourcePriorityStrategy,
    MostRecentStrategy,
    AverageValuesStrategy,
    MedianValuesStrategy,
    CustomResolutionStrategy,
    ThresholdBasedStrategy,
    create_resolution_strategy,
)


class TestResolutionStrategies(unittest.TestCase):
    """Tests for the resolution strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create data sources
        self.source1 = DataSource(
            source_id="source1",
            name="Source 1",
            source_type=DataSourceType.DATABASE,
            priority=1
        )
        
        self.source2 = DataSource(
            source_id="source2",
            name="Source 2",
            source_type=DataSourceType.API,
            priority=2
        )
        
        self.source3 = DataSource(
            source_id="source3",
            name="Source 3",
            source_type=DataSourceType.CACHE,
            priority=3
        )
        
        # Create sources dictionary
        self.sources = {
            "source1": self.source1,
            "source2": self.source2,
            "source3": self.source3,
        }
        
        # Create discrepancy
        self.discrepancy = Discrepancy(
            field="test_field",
            sources={
                "source1": 100,
                "source2": 110,
                "source3": 105,
            },
            severity=ReconciliationSeverity.MEDIUM
        )
        
        # Create non-numeric discrepancy
        self.non_numeric_discrepancy = Discrepancy(
            field="test_field",
            sources={
                "source1": "value1",
                "source2": "value2",
                "source3": "value3",
            },
            severity=ReconciliationSeverity.MEDIUM
        )
        
    def test_source_priority_strategy(self):
        """Test SourcePriorityStrategy."""
        strategy = SourcePriorityStrategy()
        
        # Run the strategy
        loop = asyncio.get_event_loop()
        resolution = loop.run_until_complete(strategy.resolve(self.discrepancy, self.sources))
        
        # Check the resolution
        self.assertEqual(resolution.discrepancy, self.discrepancy)
        self.assertEqual(resolution.resolved_value, 105)  # Value from source3 (highest priority)
        self.assertEqual(resolution.strategy, ReconciliationStrategy.SOURCE_PRIORITY)
        self.assertEqual(resolution.resolution_source, "source3")
        
    def test_most_recent_strategy(self):
        """Test MostRecentStrategy."""
        strategy = MostRecentStrategy()
        
        # Create timestamps
        now = datetime.utcnow()
        timestamps = {
            "source1": now - timedelta(minutes=10),
            "source2": now - timedelta(minutes=5),
            "source3": now - timedelta(minutes=15),
        }
        
        # Run the strategy
        loop = asyncio.get_event_loop()
        resolution = loop.run_until_complete(strategy.resolve(self.discrepancy, self.sources, timestamps=timestamps))
        
        # Check the resolution
        self.assertEqual(resolution.discrepancy, self.discrepancy)
        self.assertEqual(resolution.resolved_value, 110)  # Value from source2 (most recent)
        self.assertEqual(resolution.strategy, ReconciliationStrategy.MOST_RECENT)
        self.assertEqual(resolution.resolution_source, "source2")
        
        # Test fallback when no timestamps are provided
        resolution = loop.run_until_complete(strategy.resolve(self.discrepancy, self.sources))
        
        # Check that it falls back to SourcePriorityStrategy
        self.assertEqual(resolution.discrepancy, self.discrepancy)
        self.assertEqual(resolution.resolved_value, 105)  # Value from source3 (highest priority)
        self.assertEqual(resolution.strategy, ReconciliationStrategy.SOURCE_PRIORITY)
        self.assertEqual(resolution.resolution_source, "source3")
        
    def test_average_values_strategy(self):
        """Test AverageValuesStrategy."""
        strategy = AverageValuesStrategy()
        
        # Run the strategy
        loop = asyncio.get_event_loop()
        resolution = loop.run_until_complete(strategy.resolve(self.discrepancy, self.sources))
        
        # Check the resolution
        self.assertEqual(resolution.discrepancy, self.discrepancy)
        self.assertEqual(resolution.resolved_value, 105)  # Average of 100, 110, 105
        self.assertEqual(resolution.strategy, ReconciliationStrategy.AVERAGE)
        self.assertIsNone(resolution.resolution_source)
        
        # Test fallback for non-numeric values
        resolution = loop.run_until_complete(strategy.resolve(self.non_numeric_discrepancy, self.sources))
        
        # Check that it falls back to SourcePriorityStrategy
        self.assertEqual(resolution.discrepancy, self.non_numeric_discrepancy)
        self.assertEqual(resolution.resolved_value, "value3")  # Value from source3 (highest priority)
        self.assertEqual(resolution.strategy, ReconciliationStrategy.SOURCE_PRIORITY)
        self.assertEqual(resolution.resolution_source, "source3")
        
    def test_median_values_strategy(self):
        """Test MedianValuesStrategy."""
        strategy = MedianValuesStrategy()
        
        # Run the strategy
        loop = asyncio.get_event_loop()
        resolution = loop.run_until_complete(strategy.resolve(self.discrepancy, self.sources))
        
        # Check the resolution
        self.assertEqual(resolution.discrepancy, self.discrepancy)
        self.assertEqual(resolution.resolved_value, 105)  # Median of 100, 110, 105
        self.assertEqual(resolution.strategy, ReconciliationStrategy.MEDIAN)
        self.assertIsNone(resolution.resolution_source)
        
        # Test fallback for non-numeric values
        resolution = loop.run_until_complete(strategy.resolve(self.non_numeric_discrepancy, self.sources))
        
        # Check that it falls back to SourcePriorityStrategy
        self.assertEqual(resolution.discrepancy, self.non_numeric_discrepancy)
        self.assertEqual(resolution.resolved_value, "value3")  # Value from source3 (highest priority)
        self.assertEqual(resolution.strategy, ReconciliationStrategy.SOURCE_PRIORITY)
        self.assertEqual(resolution.resolution_source, "source3")
        
    def test_custom_resolution_strategy(self):
        """Test CustomResolutionStrategy."""
        # Define a custom resolution function
        def custom_resolver(discrepancy, sources, kwargs):
    """
    Custom resolver.
    
    Args:
        discrepancy: Description of discrepancy
        sources: Description of sources
        kwargs: Description of kwargs
    
    """

            # Use the minimum value
            return min(discrepancy.sources.values())
            
        strategy = CustomResolutionStrategy(custom_resolver)
        
        # Run the strategy
        loop = asyncio.get_event_loop()
        resolution = loop.run_until_complete(strategy.resolve(self.discrepancy, self.sources))
        
        # Check the resolution
        self.assertEqual(resolution.discrepancy, self.discrepancy)
        self.assertEqual(resolution.resolved_value, 100)  # Minimum value
        self.assertEqual(resolution.strategy, ReconciliationStrategy.CUSTOM)
        self.assertIsNone(resolution.resolution_source)
        
    def test_threshold_based_strategy(self):
        """Test ThresholdBasedStrategy."""
        # Create strategies for within and outside threshold
        within_strategy = AverageValuesStrategy()
        outside_strategy = SourcePriorityStrategy()
        
        # Create threshold-based strategy with 10% threshold
        strategy = ThresholdBasedStrategy(10.0, within_strategy, outside_strategy)
        
        # Run the strategy on discrepancy with 10% range
        loop = asyncio.get_event_loop()
        resolution = loop.run_until_complete(strategy.resolve(self.discrepancy, self.sources))
        
        # Check the resolution
        self.assertEqual(resolution.discrepancy, self.discrepancy)
        self.assertEqual(resolution.resolved_value, 105)  # Average (within threshold)
        self.assertEqual(resolution.strategy, ReconciliationStrategy.AVERAGE)
        self.assertIsNone(resolution.resolution_source)
        
        # Create discrepancy with larger range
        large_range_discrepancy = Discrepancy(
            field="test_field",
            sources={
                "source1": 100,
                "source2": 150,  # 50% difference
                "source3": 120,
            },
            severity=ReconciliationSeverity.MEDIUM
        )
        
        # Run the strategy on discrepancy with large range
        resolution = loop.run_until_complete(strategy.resolve(large_range_discrepancy, self.sources))
        
        # Check the resolution
        self.assertEqual(resolution.discrepancy, large_range_discrepancy)
        self.assertEqual(resolution.resolved_value, 120)  # Source priority (outside threshold)
        self.assertEqual(resolution.strategy, ReconciliationStrategy.SOURCE_PRIORITY)
        self.assertEqual(resolution.resolution_source, "source3")
        
    def test_create_resolution_strategy(self):
        """Test create_resolution_strategy function."""
        # Test creating each strategy type
        strategy = create_resolution_strategy(ReconciliationStrategy.SOURCE_PRIORITY)
        self.assertIsInstance(strategy, SourcePriorityStrategy)
        
        strategy = create_resolution_strategy(ReconciliationStrategy.MOST_RECENT)
        self.assertIsInstance(strategy, MostRecentStrategy)
        
        strategy = create_resolution_strategy(ReconciliationStrategy.AVERAGE)
        self.assertIsInstance(strategy, AverageValuesStrategy)
        
        strategy = create_resolution_strategy(ReconciliationStrategy.MEDIAN)
        self.assertIsInstance(strategy, MedianValuesStrategy)
        
        # Test creating custom strategy
        def custom_resolver(discrepancy, sources, kwargs):
    """
    Custom resolver.
    
    Args:
        discrepancy: Description of discrepancy
        sources: Description of sources
        kwargs: Description of kwargs
    
    """

            return min(discrepancy.sources.values())
            
        strategy = create_resolution_strategy(
            ReconciliationStrategy.CUSTOM,
            resolution_func=custom_resolver
        )
        self.assertIsInstance(strategy, CustomResolutionStrategy)
        
        # Test creating threshold-based strategy
        strategy = create_resolution_strategy(
            ReconciliationStrategy.THRESHOLD_BASED,
            threshold=10.0
        )
        self.assertIsInstance(strategy, ThresholdBasedStrategy)
        
        # Test error when missing required parameters
        with self.assertRaises(ValueError):
            create_resolution_strategy(ReconciliationStrategy.CUSTOM)
            
        with self.assertRaises(ValueError):
            create_resolution_strategy(ReconciliationStrategy.THRESHOLD_BASED)


if __name__ == "__main__":
    unittest.main()
