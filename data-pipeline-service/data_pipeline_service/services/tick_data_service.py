"""
Service for Tick data operations.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from core_foundations.exceptions.base_exceptions import DataValidationError
from core_foundations.utils.logger import get_logger
from data_pipeline_service.models.schemas import PaginatedResponse, TickData
from data_pipeline_service.repositories.tick_data_repository import TickDataRepository
from data_pipeline_service.validation import get_validation_engine

# Initialize logger
logger = get_logger("tick-data-service")


class TickDataService:
    """Service for handling tick data operations."""

    def __init__(self, repository: TickDataRepository):
        """
        Initialize the tick data service with repository.
        
        Args:
            repository: Repository for tick data access
        """
        self.repository = repository
        self.validation_engine = get_validation_engine()

    async def get_tick_data(
        self,
        symbol: str,
        from_time: datetime,
        to_time: datetime,
        limit: int = 10000,
        page: int = 1,
        page_size: int = 1000,
    ) -> PaginatedResponse:
        """
        Retrieve tick data for a specific instrument.
        
        Args:
            symbol: Trading instrument symbol
            from_time: Start time for data query
            to_time: End time for data query
            limit: Maximum number of ticks to return
            page: Page number for pagination
            page_size: Number of items per page
            
        Returns:
            PaginatedResponse of tick data
        """
        # Log the request
        logger.info(
            f"Fetching tick data for {symbol} "
            f"from {from_time} to {to_time}, limit {limit}"
        )
        
        # Fetch data from repository
        data, total_count = await self.repository.get_tick_data(
            symbol=symbol,
            from_time=from_time,
            to_time=to_time,
            limit=limit,
            offset=(page - 1) * page_size,
            page_size=page_size,
        )
        
        # Calculate total pages
        total_pages = (total_count + page_size - 1) // page_size
        
        # Format the response
        return PaginatedResponse(
            total=total_count,
            page=page,
            page_size=page_size,
            pages=total_pages,
            data=[item.model_dump() for item in data],
        )

    async def store_tick_data(self, data: List[TickData]) -> int:
        """
        Store a batch of tick data.
        
        Args:
            data: List of tick data objects to store
            
        Returns:
            Number of records stored
            
        Raises:
            DataValidationError: If data validation fails
        """
        logger.info(f"Storing {len(data)} tick records")
        
        # First validate the data using the validation engine
        validation_result = self.validate_tick_data(data)
        if not validation_result:
            msg = "Tick data validation failed"
            logger.warning(msg)
            raise DataValidationError(msg)
        
        # Store the data and get count of records stored
        count = await self.repository.insert_tick_data(data)
        
        logger.info(f"Successfully stored {count} tick records")
        return count

    def validate_tick_data(self, data: List[TickData]) -> bool:
        """
        Validate a batch of tick data using the validation engine.
        
        Args:
            data: List of tick data objects to validate
            
        Returns:
            True if all data is valid, False otherwise
        """
        if not data:
            logger.warning("Empty tick data batch")
            return False
        
        try:
            # Use the validation engine to validate the data
            return self.validation_engine.validate(data, "tick")
            
        except Exception as e:
            logger.error(f"Error validating tick data: {str(e)}")
            return False

    @staticmethod
    def get_time_window_delta(time_window: str) -> timedelta:
        """
        Calculate a timedelta representing a specific time window.
        
        Args:
            time_window: Time window string (e.g., '1h', '1d')
            
        Returns:
            Timedelta for the specified time window
        """
        # Map time window to minutes
        window_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,  # 24 hours
            "1w": 10080,  # 7 days
        }
        
        if time_window not in window_minutes:
            raise ValueError(f"Invalid time window: {time_window}")
        
        # Return as timedelta
        return timedelta(minutes=window_minutes[time_window])