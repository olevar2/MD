"""
Tests for the Historical Data Management service.
"""

import asyncio
import datetime
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from models.models_1 import (
    DataSourceType,
    CorrectionType,
    HistoricalOHLCVRecord,
    HistoricalTickRecord,
    HistoricalAlternativeRecord,
    DataCorrectionRecord,
    DataQualityReport,
    MLDatasetConfig
)
from models.repository import HistoricalDataRepository
from services.service_1 import HistoricalDataService


class TestHistoricalDataService(unittest.TestCase):
    """Tests for the HistoricalDataService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.repository = AsyncMock(spec=HistoricalDataRepository)
        self.service = HistoricalDataService(self.repository)
    
    @pytest.mark.asyncio
    async def test_store_ohlcv_data(self):
        """Test storing OHLCV data."""
        # Arrange
        symbol = "EURUSD"
        timeframe = "1h"
        timestamp = datetime.datetime.utcnow()
        open_price = 1.1234
        high_price = 1.1256
        low_price = 1.1222
        close_price = 1.1245
        volume = 1000.0
        source_id = "provider1"
        metadata = {"exchange": "FXCM"}
        created_by = "test_user"
        
        self.repository.store_ohlcv_data.return_value = "123e4567-e89b-12d3-a456-426614174000"
        
        # Act
        result = await self.service.store_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
            source_id=source_id,
            metadata=metadata,
            created_by=created_by
        )
        
        # Assert
        self.assertEqual(result, "123e4567-e89b-12d3-a456-426614174000")
        self.repository.store_ohlcv_data.assert_called_once()
        
        # Check that the record was created correctly
        record = self.repository.store_ohlcv_data.call_args[0][0]
        self.assertIsInstance(record, HistoricalOHLCVRecord)
        self.assertEqual(record.symbol, symbol)
        self.assertEqual(record.timeframe, timeframe)
        self.assertEqual(record.timestamp, timestamp)
        self.assertEqual(record.data["open"], open_price)
        self.assertEqual(record.data["high"], high_price)
        self.assertEqual(record.data["low"], low_price)
        self.assertEqual(record.data["close"], close_price)
        self.assertEqual(record.data["volume"], volume)
        self.assertEqual(record.source_id, source_id)
        self.assertEqual(record.metadata, metadata)
        self.assertEqual(record.created_by, created_by)
    
    @pytest.mark.asyncio
    async def test_store_tick_data(self):
        """Test storing tick data."""
        # Arrange
        symbol = "EURUSD"
        timestamp = datetime.datetime.utcnow()
        bid = 1.1234
        ask = 1.1236
        bid_volume = 100.0
        ask_volume = 120.0
        source_id = "provider1"
        metadata = {"exchange": "FXCM"}
        created_by = "test_user"
        
        self.repository.store_tick_data.return_value = "123e4567-e89b-12d3-a456-426614174000"
        
        # Act
        result = await self.service.store_tick_data(
            symbol=symbol,
            timestamp=timestamp,
            bid=bid,
            ask=ask,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            source_id=source_id,
            metadata=metadata,
            created_by=created_by
        )
        
        # Assert
        self.assertEqual(result, "123e4567-e89b-12d3-a456-426614174000")
        self.repository.store_tick_data.assert_called_once()
        
        # Check that the record was created correctly
        record = self.repository.store_tick_data.call_args[0][0]
        self.assertIsInstance(record, HistoricalTickRecord)
        self.assertEqual(record.symbol, symbol)
        self.assertEqual(record.timestamp, timestamp)
        self.assertEqual(record.data["bid"], bid)
        self.assertEqual(record.data["ask"], ask)
        self.assertEqual(record.data["bid_volume"], bid_volume)
        self.assertEqual(record.data["ask_volume"], ask_volume)
        self.assertEqual(record.source_id, source_id)
        self.assertEqual(record.metadata, metadata)
        self.assertEqual(record.created_by, created_by)
    
    @pytest.mark.asyncio
    async def test_get_ohlcv_data(self):
        """Test retrieving OHLCV data."""
        # Arrange
        symbols = ["EURUSD", "GBPUSD"]
        timeframe = "1h"
        start_timestamp = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        end_timestamp = datetime.datetime.utcnow()
        
        # Mock repository response
        self.repository.get_ohlcv_data.return_value = [
            {
                "record_id": "123e4567-e89b-12d3-a456-426614174000",
                "symbol": "EURUSD",
                "timestamp": start_timestamp + datetime.timedelta(hours=1),
                "timeframe": "1h",
                "data": {
                    "open": 1.1234,
                    "high": 1.1256,
                    "low": 1.1222,
                    "close": 1.1245,
                    "volume": 1000.0
                },
                "version": 1,
                "is_correction": False
            },
            {
                "record_id": "223e4567-e89b-12d3-a456-426614174000",
                "symbol": "GBPUSD",
                "timestamp": start_timestamp + datetime.timedelta(hours=1),
                "timeframe": "1h",
                "data": {
                    "open": 1.3234,
                    "high": 1.3256,
                    "low": 1.3222,
                    "close": 1.3245,
                    "volume": 800.0
                },
                "version": 1,
                "is_correction": False
            }
        ]
        
        # Act
        result = await self.service.get_ohlcv_data(
            symbols=symbols,
            timeframe=timeframe,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp
        )
        
        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.repository.get_ohlcv_data.assert_called_once_with(
            symbols=symbols,
            timeframe=timeframe,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            version=None,
            point_in_time=None,
            include_corrections=True
        )
    
    @pytest.mark.asyncio
    async def test_create_correction(self):
        """Test creating a correction."""
        # Arrange
        original_record_id = "123e4567-e89b-12d3-a456-426614174000"
        correction_data = {"data": {"close": 1.1246}}
        correction_type = CorrectionType.PROVIDER_CORRECTION
        correction_reason = "Provider sent corrected data"
        corrected_by = "system"
        source_type = DataSourceType.OHLCV
        
        self.repository.create_correction.return_value = (
            "223e4567-e89b-12d3-a456-426614174000",  # corrected_record_id
            "323e4567-e89b-12d3-a456-426614174000"   # correction_id
        )
        
        # Act
        result = await self.service.create_correction(
            original_record_id=original_record_id,
            correction_data=correction_data,
            correction_type=correction_type,
            correction_reason=correction_reason,
            corrected_by=corrected_by,
            source_type=source_type
        )
        
        # Assert
        self.assertEqual(result, (
            "223e4567-e89b-12d3-a456-426614174000",
            "323e4567-e89b-12d3-a456-426614174000"
        ))
        self.repository.create_correction.assert_called_once_with(
            original_record_id=original_record_id,
            correction_data=correction_data,
            correction_type=correction_type,
            correction_reason=correction_reason,
            corrected_by=corrected_by,
            source_type=source_type
        )


if __name__ == "__main__":
    unittest.main()
