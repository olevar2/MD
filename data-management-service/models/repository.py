"""
Historical Data Repository.

This module provides repository classes for storing and retrieving historical data.
It implements the data access layer for the Historical Data Management system.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.dialects.postgresql import JSONB, insert

from models.models_1 import (
    HistoricalDataRecord,
    HistoricalOHLCVRecord,
    HistoricalTickRecord,
    HistoricalAlternativeRecord,
    DataCorrectionRecord,
    DataQualityReport,
    DataSourceType,
    CorrectionType
)

logger = logging.getLogger(__name__)


class HistoricalDataRepository:
    """Repository for historical data storage and retrieval."""

    def __init__(self, engine: AsyncEngine):
        """
        Initialize the repository.

        Args:
            engine: SQLAlchemy async engine
        """
        self.engine = engine
        self.metadata = sa.MetaData(schema="historical_data")

        # Define tables
        self.ohlcv_table = sa.Table(
            "ohlcv_data",
            self.metadata,
            sa.Column("record_id", sa.String, primary_key=True),
            sa.Column("symbol", sa.String, nullable=False, index=True),
            sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),
            sa.Column("timeframe", sa.String, nullable=False, index=True),
            sa.Column("source_type", sa.String, nullable=False),
            sa.Column("source_id", sa.String, nullable=False),
            sa.Column("data", JSONB, nullable=False),
            sa.Column("version", sa.Integer, nullable=False, default=1),
            sa.Column("is_correction", sa.Boolean, nullable=False, default=False),
            sa.Column("correction_of", sa.String, nullable=True),
            sa.Column("correction_type", sa.String, nullable=True),
            sa.Column("correction_reason", sa.String, nullable=True),
            sa.Column("metadata", JSONB, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("created_by", sa.String, nullable=True),
        )

        self.tick_table = sa.Table(
            "tick_data",
            self.metadata,
            sa.Column("record_id", sa.String, primary_key=True),
            sa.Column("symbol", sa.String, nullable=False, index=True),
            sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),
            sa.Column("source_type", sa.String, nullable=False),
            sa.Column("source_id", sa.String, nullable=False),
            sa.Column("data", JSONB, nullable=False),
            sa.Column("version", sa.Integer, nullable=False, default=1),
            sa.Column("is_correction", sa.Boolean, nullable=False, default=False),
            sa.Column("correction_of", sa.String, nullable=True),
            sa.Column("correction_type", sa.String, nullable=True),
            sa.Column("correction_reason", sa.String, nullable=True),
            sa.Column("metadata", JSONB, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("created_by", sa.String, nullable=True),
        )

        self.alternative_table = sa.Table(
            "alternative_data",
            self.metadata,
            sa.Column("record_id", sa.String, primary_key=True),
            sa.Column("symbol", sa.String, nullable=False, index=True),
            sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),
            sa.Column("source_type", sa.String, nullable=False),
            sa.Column("source_id", sa.String, nullable=False),
            sa.Column("data_type", sa.String, nullable=False, index=True),
            sa.Column("data", JSONB, nullable=False),
            sa.Column("version", sa.Integer, nullable=False, default=1),
            sa.Column("is_correction", sa.Boolean, nullable=False, default=False),
            sa.Column("correction_of", sa.String, nullable=True),
            sa.Column("correction_type", sa.String, nullable=True),
            sa.Column("correction_reason", sa.String, nullable=True),
            sa.Column("metadata", JSONB, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("created_by", sa.String, nullable=True),
        )

        self.corrections_table = sa.Table(
            "data_corrections",
            self.metadata,
            sa.Column("correction_id", sa.String, primary_key=True),
            sa.Column("original_record_id", sa.String, nullable=False, index=True),
            sa.Column("corrected_record_id", sa.String, nullable=False, index=True),
            sa.Column("correction_type", sa.String, nullable=False),
            sa.Column("correction_reason", sa.String, nullable=False),
            sa.Column("correction_timestamp", sa.DateTime(timezone=True), nullable=False),
            sa.Column("corrected_by", sa.String, nullable=False),
            sa.Column("approved_by", sa.String, nullable=True),
            sa.Column("approval_timestamp", sa.DateTime(timezone=True), nullable=True),
            sa.Column("metadata", JSONB, nullable=True),
        )

        self.quality_reports_table = sa.Table(
            "data_quality_reports",
            self.metadata,
            sa.Column("report_id", sa.String, primary_key=True),
            sa.Column("symbol", sa.String, nullable=False, index=True),
            sa.Column("source_type", sa.String, nullable=False),
            sa.Column("timeframe", sa.String, nullable=True),
            sa.Column("start_timestamp", sa.DateTime(timezone=True), nullable=False),
            sa.Column("end_timestamp", sa.DateTime(timezone=True), nullable=False),
            sa.Column("total_records", sa.Integer, nullable=False),
            sa.Column("missing_records", sa.Integer, nullable=False),
            sa.Column("corrected_records", sa.Integer, nullable=False),
            sa.Column("quality_issues", JSONB, nullable=False),
            sa.Column("report_timestamp", sa.DateTime(timezone=True), nullable=False),
            sa.Column("metadata", JSONB, nullable=True),
        )

    async def initialize(self):
        """Initialize the repository by creating tables if they don't exist."""
        async with self.engine.begin() as conn:
            # Create schema if it doesn't exist
            await conn.execute(sa.text("CREATE SCHEMA IF NOT EXISTS historical_data"))

            # Create tables
            await conn.run_sync(self.metadata.create_all)

            # Convert tables to TimescaleDB hypertables if TimescaleDB is available
            try:
                # Check if TimescaleDB extension is available
                result = await conn.execute(sa.text(
                    "SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'"
                ))
                has_timescale = result.scalar() > 0

                if has_timescale:
                    # Convert tables to hypertables
                    for table_name in ["ohlcv_data", "tick_data", "alternative_data"]:
                        await conn.execute(sa.text(
                            f"SELECT create_hypertable('historical_data.{table_name}', 'timestamp', "
                            f"if_not_exists => TRUE)"
                        ))
                    logger.info("Tables converted to TimescaleDB hypertables")
                else:
                    logger.warning("TimescaleDB extension not available, using regular tables")
            except Exception as e:
                logger.warning(f"Failed to convert tables to TimescaleDB hypertables: {e}")

    async def store_ohlcv_data(self, record: HistoricalOHLCVRecord) -> str:
        """
        Store OHLCV data record.

        Args:
            record: OHLCV data record

        Returns:
            Record ID
        """
        try:
            record_dict = record.dict()

            async with self.engine.begin() as conn:
                await conn.execute(
                    self.ohlcv_table.insert().values(**record_dict)
                )

                logger.info(f"Stored OHLCV record {record.record_id} for {record.symbol}")
                return record.record_id
        except Exception as e:
            logger.error(f"Failed to store OHLCV record: {e}")
            raise

    async def store_tick_data(self, record: HistoricalTickRecord) -> str:
        """
        Store tick data record.

        Args:
            record: Tick data record

        Returns:
            Record ID
        """
        try:
            record_dict = record.dict()

            async with self.engine.begin() as conn:
                await conn.execute(
                    self.tick_table.insert().values(**record_dict)
                )

                logger.info(f"Stored tick record {record.record_id} for {record.symbol}")
                return record.record_id
        except Exception as e:
            logger.error(f"Failed to store tick record: {e}")
            raise

    async def store_alternative_data(self, record: HistoricalAlternativeRecord) -> str:
        """
        Store alternative data record.

        Args:
            record: Alternative data record

        Returns:
            Record ID
        """
        try:
            record_dict = record.dict()

            async with self.engine.begin() as conn:
                await conn.execute(
                    self.alternative_table.insert().values(**record_dict)
                )

                logger.info(f"Stored alternative data record {record.record_id} for {record.symbol}")
                return record.record_id
        except Exception as e:
            logger.error(f"Failed to store alternative data record: {e}")
            raise

    async def store_correction(self, correction: DataCorrectionRecord) -> str:
        """
        Store data correction record.

        Args:
            correction: Data correction record

        Returns:
            Correction ID
        """
        try:
            correction_dict = correction.dict()

            async with self.engine.begin() as conn:
                await conn.execute(
                    self.corrections_table.insert().values(**correction_dict)
                )

                logger.info(f"Stored correction {correction.correction_id}")
                return correction.correction_id
        except Exception as e:
            logger.error(f"Failed to store correction: {e}")
            raise

    async def store_quality_report(self, report: DataQualityReport) -> str:
        """
        Store data quality report.

        Args:
            report: Data quality report

        Returns:
            Report ID
        """
        try:
            report_dict = report.dict()

            async with self.engine.begin() as conn:
                await conn.execute(
                    self.quality_reports_table.insert().values(**report_dict)
                )

                logger.info(f"Stored quality report {report.report_id}")
                return report.report_id
        except Exception as e:
            logger.error(f"Failed to store quality report: {e}")
            raise

    async def get_ohlcv_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_timestamp: datetime,
        end_timestamp: datetime,
        version: Optional[int] = None,
        point_in_time: Optional[datetime] = None,
        include_corrections: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get OHLCV data for the specified symbols and timeframe.

        Args:
            symbols: List of symbols
            timeframe: Timeframe
            start_timestamp: Start timestamp
            end_timestamp: End timestamp
            version: Specific version to retrieve (None for latest)
            point_in_time: Point-in-time for historical accuracy (None for latest)
            include_corrections: Whether to include corrections

        Returns:
            List of OHLCV data records
        """
        try:
            query = sa.select(self.ohlcv_table).where(
                self.ohlcv_table.c.symbol.in_(symbols),
                self.ohlcv_table.c.timeframe == timeframe,
                self.ohlcv_table.c.timestamp >= start_timestamp,
                self.ohlcv_table.c.timestamp <= end_timestamp
            )

            # Apply version filter if specified
            if version is not None:
                query = query.where(self.ohlcv_table.c.version == version)

            # Apply point-in-time filter if specified
            if point_in_time is not None:
                query = query.where(self.ohlcv_table.c.created_at <= point_in_time)

            # Apply corrections filter if needed
            if not include_corrections:
                query = query.where(self.ohlcv_table.c.is_correction == False)

            # Order by timestamp
            query = query.order_by(self.ohlcv_table.c.timestamp)

            async with self.engine.connect() as conn:
                result = await conn.execute(query)
                records = result.fetchall()

                # Convert to dictionaries
                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Failed to get OHLCV data: {e}")
            raise

    async def get_tick_data(
        self,
        symbols: List[str],
        start_timestamp: datetime,
        end_timestamp: datetime,
        version: Optional[int] = None,
        point_in_time: Optional[datetime] = None,
        include_corrections: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get tick data for the specified symbols.

        Args:
            symbols: List of symbols
            start_timestamp: Start timestamp
            end_timestamp: End timestamp
            version: Specific version to retrieve (None for latest)
            point_in_time: Point-in-time for historical accuracy (None for latest)
            include_corrections: Whether to include corrections

        Returns:
            List of tick data records
        """
        try:
            query = sa.select(self.tick_table).where(
                self.tick_table.c.symbol.in_(symbols),
                self.tick_table.c.timestamp >= start_timestamp,
                self.tick_table.c.timestamp <= end_timestamp
            )

            # Apply version filter if specified
            if version is not None:
                query = query.where(self.tick_table.c.version == version)

            # Apply point-in-time filter if specified
            if point_in_time is not None:
                query = query.where(self.tick_table.c.created_at <= point_in_time)

            # Apply corrections filter if needed
            if not include_corrections:
                query = query.where(self.tick_table.c.is_correction == False)

            # Order by timestamp
            query = query.order_by(self.tick_table.c.timestamp)

            async with self.engine.connect() as conn:
                result = await conn.execute(query)
                records = result.fetchall()

                # Convert to dictionaries
                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Failed to get tick data: {e}")
            raise

    async def get_alternative_data(
        self,
        symbols: List[str],
        data_type: str,
        start_timestamp: datetime,
        end_timestamp: datetime,
        version: Optional[int] = None,
        point_in_time: Optional[datetime] = None,
        include_corrections: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get alternative data for the specified symbols and type.

        Args:
            symbols: List of symbols
            data_type: Type of alternative data
            start_timestamp: Start timestamp
            end_timestamp: End timestamp
            version: Specific version to retrieve (None for latest)
            point_in_time: Point-in-time for historical accuracy (None for latest)
            include_corrections: Whether to include corrections

        Returns:
            List of alternative data records
        """
        try:
            query = sa.select(self.alternative_table).where(
                self.alternative_table.c.symbol.in_(symbols),
                self.alternative_table.c.data_type == data_type,
                self.alternative_table.c.timestamp >= start_timestamp,
                self.alternative_table.c.timestamp <= end_timestamp
            )

            # Apply version filter if specified
            if version is not None:
                query = query.where(self.alternative_table.c.version == version)

            # Apply point-in-time filter if specified
            if point_in_time is not None:
                query = query.where(self.alternative_table.c.created_at <= point_in_time)

            # Apply corrections filter if needed
            if not include_corrections:
                query = query.where(self.alternative_table.c.is_correction == False)

            # Order by timestamp
            query = query.order_by(self.alternative_table.c.timestamp)

            async with self.engine.connect() as conn:
                result = await conn.execute(query)
                records = result.fetchall()

                # Convert to dictionaries
                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Failed to get alternative data: {e}")
            raise

    async def get_corrections(
        self,
        original_record_id: Optional[str] = None,
        corrected_record_id: Optional[str] = None,
        correction_type: Optional[CorrectionType] = None,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get data corrections.

        Args:
            original_record_id: ID of the original record
            corrected_record_id: ID of the corrected record
            correction_type: Type of correction
            start_timestamp: Start timestamp for correction_timestamp
            end_timestamp: End timestamp for correction_timestamp

        Returns:
            List of correction records
        """
        try:
            query = sa.select(self.corrections_table)

            # Apply filters
            if original_record_id is not None:
                query = query.where(self.corrections_table.c.original_record_id == original_record_id)

            if corrected_record_id is not None:
                query = query.where(self.corrections_table.c.corrected_record_id == corrected_record_id)

            if correction_type is not None:
                query = query.where(self.corrections_table.c.correction_type == correction_type.value)

            if start_timestamp is not None:
                query = query.where(self.corrections_table.c.correction_timestamp >= start_timestamp)

            if end_timestamp is not None:
                query = query.where(self.corrections_table.c.correction_timestamp <= end_timestamp)

            # Order by timestamp
            query = query.order_by(self.corrections_table.c.correction_timestamp.desc())

            async with self.engine.connect() as conn:
                result = await conn.execute(query)
                records = result.fetchall()

                # Convert to dictionaries
                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Failed to get corrections: {e}")
            raise

    async def get_quality_reports(
        self,
        symbol: Optional[str] = None,
        source_type: Optional[DataSourceType] = None,
        timeframe: Optional[str] = None,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get data quality reports.

        Args:
            symbol: Symbol filter
            source_type: Source type filter
            timeframe: Timeframe filter
            start_timestamp: Start timestamp for report_timestamp
            end_timestamp: End timestamp for report_timestamp

        Returns:
            List of quality report records
        """
        try:
            query = sa.select(self.quality_reports_table)

            # Apply filters
            if symbol is not None:
                query = query.where(self.quality_reports_table.c.symbol == symbol)

            if source_type is not None:
                query = query.where(self.quality_reports_table.c.source_type == source_type.value)

            if timeframe is not None:
                query = query.where(self.quality_reports_table.c.timeframe == timeframe)

            if start_timestamp is not None:
                query = query.where(self.quality_reports_table.c.report_timestamp >= start_timestamp)

            if end_timestamp is not None:
                query = query.where(self.quality_reports_table.c.report_timestamp <= end_timestamp)

            # Order by timestamp
            query = query.order_by(self.quality_reports_table.c.report_timestamp.desc())

            async with self.engine.connect() as conn:
                result = await conn.execute(query)
                records = result.fetchall()

                # Convert to dictionaries
                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Failed to get quality reports: {e}")
            raise

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
        try:
            # Get the original record
            table_map = {
                DataSourceType.OHLCV: self.ohlcv_table,
                DataSourceType.TICK: self.tick_table,
                DataSourceType.ALTERNATIVE: self.alternative_table
            }

            table = table_map.get(source_type)
            if table is None:
                raise ValueError(f"Unsupported source type: {source_type}")

            async with self.engine.begin() as conn:
                # Get the original record
                query = sa.select(table).where(table.c.record_id == original_record_id)
                result = await conn.execute(query)
                original_record = result.fetchone()

                if original_record is None:
                    raise ValueError(f"Original record not found: {original_record_id}")

                # Create a new record with the correction
                original_dict = dict(original_record)

                # Update with correction data
                if "data" in original_dict and "data" in correction_data:
                    original_dict["data"].update(correction_data["data"])
                else:
                    original_dict.update(correction_data)

                # Set correction fields
                original_dict["record_id"] = str(uuid4())  # New record ID
                original_dict["is_correction"] = True
                original_dict["correction_of"] = original_record_id
                original_dict["correction_type"] = correction_type.value
                original_dict["correction_reason"] = correction_reason
                original_dict["created_by"] = corrected_by
                original_dict["created_at"] = datetime.utcnow()
                original_dict["version"] = original_dict.get("version", 1) + 1

                # Insert the corrected record
                await conn.execute(table.insert().values(**original_dict))

                # Create correction record
                correction = DataCorrectionRecord(
                    original_record_id=original_record_id,
                    corrected_record_id=original_dict["record_id"],
                    correction_type=correction_type,
                    correction_reason=correction_reason,
                    corrected_by=corrected_by
                )

                correction_dict = correction.dict()
                await conn.execute(self.corrections_table.insert().values(**correction_dict))

                logger.info(f"Created correction {correction.correction_id} for record {original_record_id}")
                return original_dict["record_id"], correction.correction_id
        except Exception as e:
            logger.error(f"Failed to create correction: {e}")
            raise

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
        try:
            # Get the table based on source type
            table_map = {
                DataSourceType.OHLCV: self.ohlcv_table,
                DataSourceType.TICK: self.tick_table,
                DataSourceType.ALTERNATIVE: self.alternative_table
            }

            table = table_map.get(source_type)
            if table is None:
                raise ValueError(f"Unsupported source type: {source_type}")

            async with self.engine.connect() as conn:
                # Get the original record
                query = sa.select(table).where(table.c.record_id == record_id)
                result = await conn.execute(query)
                record = result.fetchone()

                if record is None:
                    raise ValueError(f"Record not found: {record_id}")

                record_dict = dict(record)

                # Check if this is a correction
                if record_dict.get("is_correction") and record_dict.get("correction_of"):
                    # This is a correction, get the original record
                    original_id = record_dict["correction_of"]
                    return await self.get_record_history(original_id, source_type)

                # This is an original record, get all corrections
                query = sa.select(table).where(
                    table.c.correction_of == record_id,
                    table.c.is_correction == True
                ).order_by(table.c.version)

                result = await conn.execute(query)
                corrections = result.fetchall()

                # Combine original and corrections
                history = [record_dict]
                history.extend([dict(correction) for correction in corrections])

                return history
        except Exception as e:
            logger.error(f"Failed to get record history: {e}")
            raise
