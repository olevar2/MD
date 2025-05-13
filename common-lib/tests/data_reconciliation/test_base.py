"""
Tests for the base data reconciliation module.
"""

import unittest
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

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


class MockDataReconciliation(DataReconciliationBase):
    """Mock implementation for testing."""

    def __init__(self, config: ReconciliationConfig):
        """Initialize mock reconciliation."""
        super().__init__(config)
        self.fetched_data = {}
        self.applied_resolutions = []

    async def fetch_data(self, source: DataSource, **kwargs) -> Any:
        """Mock implementation of fetch_data."""
        if source.source_id == "source1":
            return {"key1": 100, "key2": 200, "key3": 300}
        elif source.source_id == "source2":
            return {"key1": 101, "key2": 200, "key3": 305}
        else:
            return None

    async def compare_data(self, data_map: Dict[str, Any]) -> List[Discrepancy]:
        """Mock implementation of compare_data."""
        discrepancies = []

        # Compare values for each key
        for key in ["key1", "key2", "key3"]:
            values = {}
            for source_id, data in data_map.items():
                if key in data:
                    values[source_id] = data[key]

            # Check if there are discrepancies
            if len(set(values.values())) > 1:
                discrepancy = Discrepancy(
                    field=key,
                    sources=values,
                    severity=ReconciliationSeverity.MEDIUM
                )
                discrepancies.append(discrepancy)

        return discrepancies

    async def resolve_discrepancies(self, discrepancies: List[Discrepancy]) -> List[DiscrepancyResolution]:
        """Mock implementation of resolve_discrepancies."""
        resolutions = []

        for discrepancy in discrepancies:
            # Use the average value for resolution
            values = list(discrepancy.sources.values())
            resolved_value = sum(values) / len(values)

            resolution = DiscrepancyResolution(
                discrepancy=discrepancy,
                resolved_value=resolved_value,
                strategy=ReconciliationStrategy.AVERAGE
            )
            resolutions.append(resolution)

        return resolutions

    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]) -> bool:
        """Mock implementation of apply_resolutions."""
        self.applied_resolutions = resolutions
        return True


class TestDataReconciliationBase(unittest.TestCase):
    """Tests for the DataReconciliationBase class."""

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

        # Create configuration
        self.config = ReconciliationConfig(
            sources=[self.source1, self.source2],
            strategy=ReconciliationStrategy.AVERAGE,
            tolerance=0.01,
            auto_resolve=True
        )

        # Create reconciliation instance
        self.reconciliation = MockDataReconciliation(self.config)

    def test_data_source(self):
        """Test DataSource class."""
        source = DataSource(
            source_id="test",
            name="Test Source",
            source_type=DataSourceType.DATABASE,
            priority=1,
            metadata={"key": "value"}
        )

        self.assertEqual(source.source_id, "test")
        self.assertEqual(source.name, "Test Source")
        self.assertEqual(source.source_type, DataSourceType.DATABASE)
        self.assertEqual(source.priority, 1)
        self.assertEqual(source.metadata, {"key": "value"})

    def test_discrepancy(self):
        """Test Discrepancy class."""
        discrepancy = Discrepancy(
            field="test_field",
            sources={"source1": 100, "source2": 105},
            severity=ReconciliationSeverity.MEDIUM
        )

        self.assertEqual(discrepancy.field, "test_field")
        self.assertEqual(discrepancy.sources, {"source1": 100, "source2": 105})
        self.assertEqual(discrepancy.severity, ReconciliationSeverity.MEDIUM)
        self.assertEqual(discrepancy.min_value, 100)
        self.assertEqual(discrepancy.max_value, 105)
        self.assertEqual(discrepancy.mean_value, 102.5)
        self.assertEqual(discrepancy.median_value, 105)  # Only two values, so it's the second one
        self.assertAlmostEqual(discrepancy.range_pct, 5.0, delta=0.2)  # (105-100)/102.5 * 100

    def test_discrepancy_resolution(self):
        """Test DiscrepancyResolution class."""
        discrepancy = Discrepancy(
            field="test_field",
            sources={"source1": 100, "source2": 105},
            severity=ReconciliationSeverity.MEDIUM
        )

        resolution = DiscrepancyResolution(
            discrepancy=discrepancy,
            resolved_value=102.5,
            strategy=ReconciliationStrategy.AVERAGE,
            resolution_source=None
        )

        self.assertEqual(resolution.discrepancy, discrepancy)
        self.assertEqual(resolution.resolved_value, 102.5)
        self.assertEqual(resolution.strategy, ReconciliationStrategy.AVERAGE)
        self.assertIsNone(resolution.resolution_source)

    def test_reconciliation_config(self):
        """Test ReconciliationConfig class."""
        config = ReconciliationConfig(
            sources=[self.source1, self.source2],
            strategy=ReconciliationStrategy.AVERAGE,
            tolerance=0.01,
            fields_to_reconcile=["field1", "field2"],
            fields_to_ignore=["field3"],
            timeout_seconds=60.0,
            batch_size=1000,
            auto_resolve=True,
            notification_threshold=ReconciliationSeverity.HIGH
        )

        self.assertEqual(config.sources, [self.source1, self.source2])
        self.assertEqual(config.strategy, ReconciliationStrategy.AVERAGE)
        self.assertEqual(config.tolerance, 0.01)
        self.assertEqual(config.fields_to_reconcile, ["field1", "field2"])
        self.assertEqual(config.fields_to_ignore, ["field3"])
        self.assertEqual(config.timeout_seconds, 60.0)
        self.assertEqual(config.batch_size, 1000)
        self.assertTrue(config.auto_resolve)
        self.assertEqual(config.notification_threshold, ReconciliationSeverity.HIGH)

    def test_reconciliation_result(self):
        """Test ReconciliationResult class."""
        result = ReconciliationResult(
            reconciliation_id="test_id",
            config=self.config,
            status=ReconciliationStatus.COMPLETED,
            start_time=datetime.utcnow() - timedelta(seconds=10),
            end_time=datetime.utcnow(),
            discrepancies=[],
            resolutions=[]
        )

        self.assertEqual(result.reconciliation_id, "test_id")
        self.assertEqual(result.config, self.config)
        self.assertEqual(result.status, ReconciliationStatus.COMPLETED)
        self.assertIsNotNone(result.start_time)
        self.assertIsNotNone(result.end_time)
        self.assertEqual(result.discrepancies, [])
        self.assertEqual(result.resolutions, [])
        self.assertGreater(result.duration_seconds, 0)
        self.assertEqual(result.discrepancy_count, 0)
        self.assertEqual(result.resolution_count, 0)
        self.assertEqual(result.resolution_rate, 100.0)  # No discrepancies, so 100%

    def test_reconcile(self):
        """Test reconcile method."""
        # Run the reconciliation
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.reconciliation.reconcile())

        # Check the result
        self.assertEqual(result.status, ReconciliationStatus.COMPLETED)
        self.assertEqual(len(result.discrepancies), 2)  # key1 and key3 have discrepancies
        self.assertEqual(len(result.resolutions), 2)
        self.assertEqual(result.resolution_rate, 100.0)

        # Check the discrepancies
        discrepancy_fields = [d.field for d in result.discrepancies]
        self.assertIn("key1", discrepancy_fields)
        self.assertIn("key3", discrepancy_fields)

        # Check the resolutions
        for resolution in result.resolutions:
            if resolution.discrepancy.field == "key1":
                self.assertEqual(resolution.resolved_value, 100.5)  # Average of 100 and 101
            elif resolution.discrepancy.field == "key3":
                self.assertEqual(resolution.resolved_value, 302.5)  # Average of 300 and 305

        # Check that resolutions were applied
        self.assertEqual(len(self.reconciliation.applied_resolutions), 2)


if __name__ == "__main__":
    unittest.main()
