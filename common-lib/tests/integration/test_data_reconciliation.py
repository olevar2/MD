"""
Integration tests for the data reconciliation framework.

This module tests the integration between different services using the data reconciliation framework.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from common_lib.data_reconciliation import (
    DataSource,
    DataSourceType,
    ReconciliationConfig,
    ReconciliationStrategy,
    ReconciliationSeverity,
    BatchReconciliationProcessor,
)


class TestDataReconciliationIntegration:
    """Integration tests for data reconciliation."""
    
    @pytest.mark.asyncio
    async def test_batch_reconciliation_with_dataframes(self):
        """Test batch reconciliation with DataFrames."""
        # Create test data
        date_range = pd.date_range(start=datetime.utcnow() - timedelta(days=10), periods=100, freq="1H")
        
        # Create DataFrame 1
        df1 = pd.DataFrame(index=date_range)
        df1["value1"] = np.random.randn(len(date_range))
        df1["value2"] = np.random.randn(len(date_range))
        
        # Create DataFrame 2 with some discrepancies
        df2 = df1.copy()
        # Add some discrepancies
        for i in range(0, len(df2), 10):
            df2.iloc[i, 0] = df1.iloc[i, 0] * 1.1  # 10% difference
            
        # Create data sources
        source1 = DataSource(
            source_id="source1",
            name="Source 1",
            source_type=DataSourceType.DATABASE,
            priority=1
        )
        
        source2 = DataSource(
            source_id="source2",
            name="Source 2",
            source_type=DataSourceType.API,
            priority=2
        )
        
        # Create configuration
        config = ReconciliationConfig(
            sources=[source1, source2],
            strategy=ReconciliationStrategy.SOURCE_PRIORITY,
            tolerance=0.001,
            auto_resolve=True
        )
        
        # Create reconciliation processor
        reconciliation = BatchReconciliationProcessor(config)
        
        # Define data fetchers
        async def fetch_from_source1(**kwargs):
    """
    Fetch from source1.
    
    Args:
        kwargs: Description of kwargs
    
    """

            return df1
            
        async def fetch_from_source2(**kwargs):
    """
    Fetch from source2.
    
    Args:
        kwargs: Description of kwargs
    
    """

            return df2
            
        reconciliation.data_fetchers = {
            "source1": fetch_from_source1,
            "source2": fetch_from_source2,
        }
        
        # Perform reconciliation
        result = await reconciliation.reconcile()
        
        # Check the result
        assert result.status.name == "COMPLETED"
        assert result.discrepancy_count == 10  # One discrepancy every 10 rows
        assert result.resolution_count == 10
        assert result.resolution_rate == 100.0
        
    @pytest.mark.asyncio
    async def test_batch_reconciliation_with_dictionaries(self):
        """Test batch reconciliation with dictionaries."""
        # Create test data
        dict1 = {
            "key1": 100,
            "key2": 200,
            "key3": 300,
            "key4": 400,
            "key5": 500,
        }
        
        # Create dict2 with some discrepancies
        dict2 = dict1.copy()
        dict2["key1"] = 101  # Small discrepancy
        dict2["key3"] = 330  # Larger discrepancy
        
        # Create data sources
        source1 = DataSource(
            source_id="source1",
            name="Source 1",
            source_type=DataSourceType.DATABASE,
            priority=1
        )
        
        source2 = DataSource(
            source_id="source2",
            name="Source 2",
            source_type=DataSourceType.API,
            priority=2
        )
        
        # Create configuration
        config = ReconciliationConfig(
            sources=[source1, source2],
            strategy=ReconciliationStrategy.SOURCE_PRIORITY,
            tolerance=0.01,  # 1% tolerance
            auto_resolve=True
        )
        
        # Create reconciliation processor
        reconciliation = BatchReconciliationProcessor(config)
        
        # Define data fetchers
        async def fetch_from_source1(**kwargs):
    """
    Fetch from source1.
    
    Args:
        kwargs: Description of kwargs
    
    """

            return dict1
            
        async def fetch_from_source2(**kwargs):
    """
    Fetch from source2.
    
    Args:
        kwargs: Description of kwargs
    
    """

            return dict2
            
        reconciliation.data_fetchers = {
            "source1": fetch_from_source1,
            "source2": fetch_from_source2,
        }
        
        # Perform reconciliation
        result = await reconciliation.reconcile()
        
        # Check the result
        assert result.status.name == "COMPLETED"
        assert result.discrepancy_count == 2  # Two discrepancies
        assert result.resolution_count == 2
        assert result.resolution_rate == 100.0
        
        # Check that the discrepancies are for the expected keys
        discrepancy_fields = [d.field for d in result.discrepancies]
        assert "key1" in discrepancy_fields
        assert "key3" in discrepancy_fields
        
        # Check that the resolutions use the correct values (from source2 which has higher priority)
        for resolution in result.resolutions:
            if resolution.discrepancy.field == "key1":
                assert resolution.resolved_value == 101
            elif resolution.discrepancy.field == "key3":
                assert resolution.resolved_value == 330
