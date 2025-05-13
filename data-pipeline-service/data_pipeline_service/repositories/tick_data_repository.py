"""
Repository for Tick data operations.
"""
from datetime import datetime
from typing import List, Tuple, Union
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from core_foundations.utils.logger import get_logger
from data_pipeline_service.models.schemas import TickData
from data_pipeline_service.repositories.database_models import tick_data_table
from data_pipeline_service.optimization import optimize_query
from data_pipeline_service.monitoring import track_query_performance
logger = get_logger('tick-data-repository')


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TickDataRepository:
    """Repository for handling Tick data database operations."""

    def __init__(self, session: AsyncSession):
        """
        Initialize the repository with a database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    @track_query_performance(query_type='select', table='tick_data')
    async def get_tick_data(self, symbol: str, from_time: datetime, to_time:
        datetime, limit: int=10000, offset: int=0, page_size: int=1000
        ) ->Tuple[List[TickData], int]:
        """
        Retrieve tick data from database.

        Args:
            symbol: Instrument symbol
            from_time: Start time
            to_time: End time
            limit: Maximum number of records to return
            offset: Offset for pagination
            page_size: Number of records per page

        Returns:
            Tuple of (list of tick data objects, total count of records)
        """
        query = sa.select(tick_data_table).where(tick_data_table.c.symbol ==
            symbol, tick_data_table.c.timestamp >= from_time, 
            tick_data_table.c.timestamp <= to_time).order_by(tick_data_table
            .c.timestamp).limit(min(limit, page_size)).offset(offset)
        query_str = str(query.compile(compile_kwargs={'literal_binds': True}))
        optimized_query_str, _ = optimize_query(query_str)
        result = await self.session.execute(sa.text(optimized_query_str))
        rows = result.all()
        count_query = sa.select(sa.func.count()).select_from(tick_data_table
            ).where(tick_data_table.c.symbol == symbol, tick_data_table.c.
            timestamp >= from_time, tick_data_table.c.timestamp <= to_time)
        count_query_str = str(count_query.compile(compile_kwargs={
            'literal_binds': True}))
        optimized_count_query_str, _ = optimize_query(count_query_str)
        count_result = await self.session.execute(sa.text(
            optimized_count_query_str))
        total_count = count_result.scalar() or 0
        tick_data = [TickData(symbol=row.symbol, timestamp=row.timestamp,
            bid=row.bid, ask=row.ask, bid_volume=row.bid_volume, ask_volume
            =row.ask_volume) for row in rows]
        return tick_data, min(total_count, limit)

    @track_query_performance(query_type='insert', table='tick_data')
    @async_with_exception_handling
    async def insert_tick_data(self, data: List[TickData]) ->int:
        """
        Insert tick data into database.

        Args:
            data: List of tick data objects

        Returns:
            Number of records inserted
        """
        if not data:
            return 0
        values = [{'symbol': item.symbol, 'timestamp': item.timestamp,
            'bid': item.bid, 'ask': item.ask, 'bid_volume': item.bid_volume,
            'ask_volume': item.ask_volume} for item in data]
        query = tick_data_table.insert().values(values).on_conflict_do_nothing(
            index_elements=['symbol', 'timestamp'])
        try:
            result = await self.session.execute(query)
            await self.session.commit()
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            logger.error(f'Error inserting tick data: {str(e)}')
            raise
