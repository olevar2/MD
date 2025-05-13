"""
Historical Data Management Service.

This module provides the service layer for the Historical Data Management system.
It implements the business logic for managing historical data, including storage,
retrieval, versioning, and corrections.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import uuid4

from models.models_1 import (
    HistoricalDataRecord,
    HistoricalOHLCVRecord,
    HistoricalTickRecord,
    HistoricalAlternativeRecord,
    DataCorrectionRecord,
    DataQualityReport,
    DataSourceType,
    CorrectionType,
    MLDatasetConfig,
    HistoricalDataQuery
)
from models.repository import HistoricalDataRepository

logger = logging.getLogger(__name__)


class HistoricalDataService:
    """Service for managing historical data."""

    def __init__(self, repository: HistoricalDataRepository):
        """
        Initialize the service.

        Args:
            repository: Repository for data storage and retrieval
        """
        self.repository = repository

    async def initialize(self):
        """Initialize the service."""
        await self.repository.initialize()

    async def store_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> str:
        """
        Store OHLCV data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            timestamp: Data timestamp
            open_price: Opening price
            high_price: Highest price
            low_price: Lowest price
            close_price: Closing price
            volume: Trading volume
            source_id: Data source identifier
            metadata: Additional metadata
            created_by: User who created the record

        Returns:
            Record ID
        """
        record = HistoricalOHLCVRecord(
            symbol=symbol,
            timestamp=timestamp,
            timeframe=timeframe,
            source_type=DataSourceType.OHLCV,
            source_id=source_id,
            data={
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            },
            metadata=metadata or {},
            created_by=created_by
        )

        return await self.repository.store_ohlcv_data(record)

    async def store_tick_data(
        self,
        symbol: str,
        timestamp: datetime,
        bid: float,
        ask: float,
        bid_volume: Optional[float] = None,
        ask_volume: Optional[float] = None,
        source_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> str:
        """
        Store tick data.

        Args:
            symbol: Trading symbol
            timestamp: Data timestamp
            bid: Bid price
            ask: Ask price
            bid_volume: Bid volume
            ask_volume: Ask volume
            source_id: Data source identifier
            metadata: Additional metadata
            created_by: User who created the record

        Returns:
            Record ID
        """
        data = {
            "bid": bid,
            "ask": ask
        }

        if bid_volume is not None:
            data["bid_volume"] = bid_volume

        if ask_volume is not None:
            data["ask_volume"] = ask_volume

        record = HistoricalTickRecord(
            symbol=symbol,
            timestamp=timestamp,
            source_type=DataSourceType.TICK,
            source_id=source_id,
            data=data,
            metadata=metadata or {},
            created_by=created_by
        )

        return await self.repository.store_tick_data(record)

    async def store_alternative_data(
        self,
        symbol: str,
        timestamp: datetime,
        data_type: str,
        data: Dict[str, Any],
        source_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> str:
        """
        Store alternative data.

        Args:
            symbol: Trading symbol
            timestamp: Data timestamp
            data_type: Type of alternative data
            data: Alternative data
            source_id: Data source identifier
            metadata: Additional metadata
            created_by: User who created the record

        Returns:
            Record ID
        """
        record = HistoricalAlternativeRecord(
            symbol=symbol,
            timestamp=timestamp,
            source_type=DataSourceType.ALTERNATIVE,
            source_id=source_id,
            data_type=data_type,
            data=data,
            metadata=metadata or {},
            created_by=created_by
        )

        return await self.repository.store_alternative_data(record)

    async def get_ohlcv_data(
        self,
        symbols: Union[str, List[str]],
        timeframe: str,
        start_timestamp: datetime,
        end_timestamp: datetime,
        version: Optional[int] = None,
        point_in_time: Optional[datetime] = None,
        include_corrections: bool = True
    ) -> pd.DataFrame:
        """
        Get OHLCV data as a DataFrame.

        Args:
            symbols: Trading symbol(s)
            timeframe: Timeframe
            start_timestamp: Start timestamp
            end_timestamp: End timestamp
            version: Specific version to retrieve (None for latest)
            point_in_time: Point-in-time for historical accuracy (None for latest)
            include_corrections: Whether to include corrections

        Returns:
            DataFrame with OHLCV data
        """
        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]

        # Get data from repository
        records = await self.repository.get_ohlcv_data(
            symbols=symbols,
            timeframe=timeframe,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            version=version,
            point_in_time=point_in_time,
            include_corrections=include_corrections
        )

        if not records:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                "symbol", "timestamp", "open", "high", "low", "close", "volume"
            ])

        # Convert to DataFrame
        df_data = []
        for record in records:
            df_data.append({
                "symbol": record["symbol"],
                "timestamp": record["timestamp"],
                "open": record["data"]["open"],
                "high": record["data"]["high"],
                "low": record["data"]["low"],
                "close": record["data"]["close"],
                "volume": record["data"]["volume"],
                "record_id": record["record_id"],
                "version": record["version"],
                "is_correction": record["is_correction"]
            })

        df = pd.DataFrame(df_data)

        # Set timestamp as index
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index(["symbol", "timestamp"], inplace=True)
            df.sort_index(inplace=True)

        return df

    async def get_tick_data(
        self,
        symbols: Union[str, List[str]],
        start_timestamp: datetime,
        end_timestamp: datetime,
        version: Optional[int] = None,
        point_in_time: Optional[datetime] = None,
        include_corrections: bool = True
    ) -> pd.DataFrame:
        """
        Get tick data as a DataFrame.

        Args:
            symbols: Trading symbol(s)
            start_timestamp: Start timestamp
            end_timestamp: End timestamp
            version: Specific version to retrieve (None for latest)
            point_in_time: Point-in-time for historical accuracy (None for latest)
            include_corrections: Whether to include corrections

        Returns:
            DataFrame with tick data
        """
        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]

        # Get data from repository
        records = await self.repository.get_tick_data(
            symbols=symbols,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            version=version,
            point_in_time=point_in_time,
            include_corrections=include_corrections
        )

        if not records:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                "symbol", "timestamp", "bid", "ask", "bid_volume", "ask_volume"
            ])

        # Convert to DataFrame
        df_data = []
        for record in records:
            row = {
                "symbol": record["symbol"],
                "timestamp": record["timestamp"],
                "bid": record["data"]["bid"],
                "ask": record["data"]["ask"],
                "record_id": record["record_id"],
                "version": record["version"],
                "is_correction": record["is_correction"]
            }

            # Add optional fields if present
            if "bid_volume" in record["data"]:
                row["bid_volume"] = record["data"]["bid_volume"]

            if "ask_volume" in record["data"]:
                row["ask_volume"] = record["data"]["ask_volume"]

            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Set timestamp as index
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index(["symbol", "timestamp"], inplace=True)
            df.sort_index(inplace=True)

        return df

    async def get_alternative_data(
        self,
        symbols: Union[str, List[str]],
        data_type: str,
        start_timestamp: datetime,
        end_timestamp: datetime,
        version: Optional[int] = None,
        point_in_time: Optional[datetime] = None,
        include_corrections: bool = True
    ) -> pd.DataFrame:
        """
        Get alternative data as a DataFrame.

        Args:
            symbols: Trading symbol(s)
            data_type: Type of alternative data
            start_timestamp: Start timestamp
            end_timestamp: End timestamp
            version: Specific version to retrieve (None for latest)
            point_in_time: Point-in-time for historical accuracy (None for latest)
            include_corrections: Whether to include corrections

        Returns:
            DataFrame with alternative data
        """
        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]

        # Get data from repository
        records = await self.repository.get_alternative_data(
            symbols=symbols,
            data_type=data_type,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            version=version,
            point_in_time=point_in_time,
            include_corrections=include_corrections
        )

        if not records:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=["symbol", "timestamp", "data_type"])

        # Convert to DataFrame
        df_data = []
        for record in records:
            row = {
                "symbol": record["symbol"],
                "timestamp": record["timestamp"],
                "data_type": record["data_type"],
                "record_id": record["record_id"],
                "version": record["version"],
                "is_correction": record["is_correction"]
            }

            # Add data fields
            for key, value in record["data"].items():
                row[key] = value

            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Set timestamp as index
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index(["symbol", "timestamp"], inplace=True)
            df.sort_index(inplace=True)

        return df

    async def create_correction(
        self,
        original_record_id: str,
        correction_data: Dict[str, Any],
        correction_type: CorrectionType,
        correction_reason: str,
        corrected_by: str,
        source_type: DataSourceType
    ) -> Tuple[str, str]:
        """
        Create a correction for an existing record.

        Args:
            original_record_id: ID of the original record
            correction_data: Corrected data
            correction_type: Type of correction
            correction_reason: Reason for correction
            corrected_by: User who made the correction
            source_type: Type of data source

        Returns:
            Tuple of (corrected_record_id, correction_id)
        """
        return await self.repository.create_correction(
            original_record_id=original_record_id,
            correction_data=correction_data,
            correction_type=correction_type,
            correction_reason=correction_reason,
            corrected_by=corrected_by,
            source_type=source_type
        )

    async def get_record_history(
        self,
        record_id: str,
        source_type: DataSourceType
    ) -> List[Dict[str, Any]]:
        """
        Get the history of a record, including all corrections.

        Args:
            record_id: ID of the record
            source_type: Type of data source

        Returns:
            List of records in chronological order
        """
        return await self.repository.get_record_history(
            record_id=record_id,
            source_type=source_type
        )

    async def generate_quality_report(
        self,
        symbol: str,
        source_type: DataSourceType,
        timeframe: Optional[str] = None,
        start_timestamp: datetime = None,
        end_timestamp: datetime = None
    ) -> str:
        """
        Generate a data quality report.

        Args:
            symbol: Trading symbol
            source_type: Type of data source
            timeframe: Timeframe (for OHLCV data)
            start_timestamp: Start timestamp
            end_timestamp: End timestamp

        Returns:
            Report ID
        """
        if start_timestamp is None:
            start_timestamp = datetime(2000, 1, 1)

        if end_timestamp is None:
            end_timestamp = datetime.utcnow()

        # Get data based on source type
        if source_type == DataSourceType.OHLCV:
            if timeframe is None:
                raise ValueError("Timeframe is required for OHLCV data")

            records = await self.repository.get_ohlcv_data(
                symbols=[symbol],
                timeframe=timeframe,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                include_corrections=True
            )
        elif source_type == DataSourceType.TICK:
            records = await self.repository.get_tick_data(
                symbols=[symbol],
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                include_corrections=True
            )
        elif source_type == DataSourceType.ALTERNATIVE:
            # For alternative data, we need to know the data type
            # For simplicity, we'll just count all alternative data records
            records = await self.repository.get_alternative_data(
                symbols=[symbol],
                data_type="*",  # All types
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                include_corrections=True
            )
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        # Count records
        total_records = len(records)
        corrected_records = sum(1 for r in records if r.get("is_correction", False))

        # Identify missing data (simplified approach)
        missing_records = 0
        quality_issues = []

        if source_type == DataSourceType.OHLCV and total_records > 0:
            # For OHLCV data, check for missing bars based on timeframe
            df = pd.DataFrame([{
                "timestamp": r["timestamp"],
                "record_id": r["record_id"]
            } for r in records])

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            # Resample based on timeframe
            timeframe_map = {
                "1m": "1min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "1h": "1H",
                "4h": "4H",
                "1d": "1D"
            }

            pandas_timeframe = timeframe_map.get(timeframe, "1D")
            expected_periods = pd.date_range(
                start=start_timestamp,
                end=end_timestamp,
                freq=pandas_timeframe
            )

            # Count missing periods
            missing_records = len(expected_periods) - len(df)

            # Identify gaps
            if len(df) > 0:
                df_resampled = df.resample(pandas_timeframe).count()
                gaps = df_resampled[df_resampled["record_id"] == 0]

                for timestamp, _ in gaps.iterrows():
                    quality_issues.append({
                        "type": "missing_data",
                        "timestamp": timestamp.isoformat(),
                        "description": f"Missing data at {timestamp}"
                    })

        # Create quality report
        report = DataQualityReport(
            symbol=symbol,
            source_type=source_type,
            timeframe=timeframe,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            total_records=total_records,
            missing_records=missing_records,
            corrected_records=corrected_records,
            quality_issues=quality_issues
        )

        # Store report
        return await self.repository.store_quality_report(report)

    async def create_ml_dataset(
        self,
        config: MLDatasetConfig
    ) -> pd.DataFrame:
        """
        Create a dataset for machine learning based on the provided configuration.

        Args:
            config: ML dataset configuration

        Returns:
            DataFrame with the dataset
        """
        # Validate configuration
        if not config.symbols:
            raise ValueError("At least one symbol is required")

        if not config.timeframes:
            raise ValueError("At least one timeframe is required")

        if not config.features:
            raise ValueError("At least one feature is required")

        # Get data for each symbol and timeframe
        all_data = []

        for symbol in config.symbols:
            for timeframe in config.timeframes:
                # Get OHLCV data
                ohlcv_data = await self.get_ohlcv_data(
                    symbols=symbol,
                    timeframe=timeframe,
                    start_timestamp=config.start_timestamp,
                    end_timestamp=config.end_timestamp,
                    point_in_time=None,  # Use latest data
                    include_corrections=True
                )

                if ohlcv_data.empty:
                    logger.warning(f"No OHLCV data found for {symbol} {timeframe}")
                    continue

                # Add timeframe column
                ohlcv_data["timeframe"] = timeframe

                # Reset index to make symbol and timestamp regular columns
                ohlcv_data = ohlcv_data.reset_index()

                all_data.append(ohlcv_data)

        if not all_data:
            logger.warning("No data found for any symbol/timeframe combination")
            return pd.DataFrame()

        # Combine all data
        df = pd.concat(all_data, ignore_index=True)

        # Apply transformations
        for transform in config.transformations:
            transform_type = transform.get("type")

            if transform_type == "add_technical_indicator":
                indicator = transform.get("indicator")
                params = transform.get("params", {})

                if indicator == "sma":
                    period = params.get("period", 14)
                    df[f"sma_{period}"] = df.groupby(["symbol", "timeframe"])["close"].transform(
                        lambda x: x.rolling(window=period).mean()
                    )

                elif indicator == "ema":
                    period = params.get("period", 14)
                    df[f"ema_{period}"] = df.groupby(["symbol", "timeframe"])["close"].transform(
                        lambda x: x.ewm(span=period, adjust=False).mean()
                    )

                elif indicator == "rsi":
                    period = params.get("period", 14)
                    # Simple RSI implementation
                    delta = df.groupby(["symbol", "timeframe"])["close"].transform("diff")
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.groupby(["symbol", "timeframe"]).transform(
                        lambda x: x.rolling(window=period).mean()
                    )
                    avg_loss = loss.groupby(["symbol", "timeframe"]).transform(
                        lambda x: x.rolling(window=period).mean()
                    )
                    rs = avg_gain / avg_loss
                    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

            elif transform_type == "add_returns":
                period = transform.get("period", 1)
                df[f"return_{period}"] = df.groupby(["symbol", "timeframe"])["close"].transform(
                    lambda x: x.pct_change(periods=period)
                )

            elif transform_type == "add_target":
                target_type = transform.get("target_type", "future_return")
                periods = transform.get("periods", 1)

                if target_type == "future_return":
                    df[f"target_return_{periods}"] = df.groupby(["symbol", "timeframe"])["close"].transform(
                        lambda x: x.pct_change(periods=periods).shift(-periods)
                    )

                elif target_type == "direction":
                    future_return = df.groupby(["symbol", "timeframe"])["close"].transform(
                        lambda x: x.pct_change(periods=periods).shift(-periods)
                    )
                    df[f"target_direction_{periods}"] = (future_return > 0).astype(int)

        # Apply filters
        for filter_config in config.filters:
            filter_type = filter_config.get("type")

            if filter_type == "remove_missing":
                df = df.dropna()

            elif filter_type == "min_date":
                min_date = filter_config.get("date")
                if min_date:
                    df = df[df["timestamp"] >= pd.to_datetime(min_date)]

            elif filter_type == "max_date":
                max_date = filter_config.get("date")
                if max_date:
                    df = df[df["timestamp"] <= pd.to_datetime(max_date)]

            elif filter_type == "symbol":
                symbols = filter_config.get("symbols", [])
                if symbols:
                    df = df[df["symbol"].isin(symbols)]

        # Select features and target
        columns_to_keep = ["symbol", "timestamp", "timeframe"]

        # Add features
        for feature in config.features:
            if feature in df.columns:
                columns_to_keep.append(feature)
            else:
                logger.warning(f"Feature {feature} not found in dataset")

        # Add target if specified
        if config.target and config.target in df.columns:
            columns_to_keep.append(config.target)

        # Filter columns
        df = df[columns_to_keep]

        # Split into train/validation/test if specified
        if config.validation_split or config.test_split:
            # Sort by timestamp
            df = df.sort_values("timestamp")

            # Calculate split indices
            n = len(df)
            test_idx = int(n * (1 - config.test_split)) if config.test_split else n
            val_idx = int(test_idx * (1 - config.validation_split)) if config.validation_split else test_idx

            # Add split column
            df["split"] = "train"
            if config.validation_split:
                df.loc[val_idx:test_idx-1, "split"] = "validation"
            if config.test_split:
                df.loc[test_idx:, "split"] = "test"

        return df

    async def get_point_in_time_data(
        self,
        query: HistoricalDataQuery
    ) -> pd.DataFrame:
        """
        Get point-in-time accurate data based on the query.

        Args:
            query: Historical data query

        Returns:
            DataFrame with point-in-time accurate data
        """
        if query.source_type == DataSourceType.OHLCV:
            if not query.timeframe:
                raise ValueError("Timeframe is required for OHLCV data")

            return await self.get_ohlcv_data(
                symbols=query.symbols,
                timeframe=query.timeframe,
                start_timestamp=query.start_timestamp,
                end_timestamp=query.end_timestamp,
                version=query.version,
                point_in_time=query.point_in_time,
                include_corrections=query.include_corrections
            )

        elif query.source_type == DataSourceType.TICK:
            return await self.get_tick_data(
                symbols=query.symbols,
                start_timestamp=query.start_timestamp,
                end_timestamp=query.end_timestamp,
                version=query.version,
                point_in_time=query.point_in_time,
                include_corrections=query.include_corrections
            )

        elif query.source_type == DataSourceType.ALTERNATIVE:
            # For alternative data, we need to know the data type
            # For simplicity, we'll use the timeframe field to store the data type
            if not query.timeframe:
                raise ValueError("Data type is required for alternative data (use timeframe field)")

            return await self.get_alternative_data(
                symbols=query.symbols,
                data_type=query.timeframe,  # Use timeframe field for data type
                start_timestamp=query.start_timestamp,
                end_timestamp=query.end_timestamp,
                version=query.version,
                point_in_time=query.point_in_time,
                include_corrections=query.include_corrections
            )

        else:
            raise ValueError(f"Unsupported source type: {query.source_type}")
