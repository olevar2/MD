"""
Data Pipeline Adapter for Market Analysis Service.

This module provides an adapter for communicating with the Data Pipeline Service
to retrieve market data for analysis.
"""
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import httpx
from common_lib.resilience.decorators import (
    retry_with_backoff,
    circuit_breaker,
    timeout
)

logger = logging.getLogger(__name__)

class DataPipelineAdapter:
    """
    Adapter for communicating with the Data Pipeline Service.
    """

    def __init__(self, base_url: str = "http://data-pipeline-service:8000"):
        """
        Initialize the Data Pipeline Adapter.

        Args:
            base_url: Base URL of the Data Pipeline Service
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(timeout_seconds=10)
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        include_indicators: bool = False
    ) -> pd.DataFrame:
        """
        Get market data from the Data Pipeline Service.

        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe to get data for
            start_date: Start date for data
            end_date: End date for data
            include_indicators: Whether to include indicators in the data

        Returns:
            DataFrame with market data
        """
        try:
            request_id = str(uuid.uuid4())
            headers = {"X-Request-ID": request_id}

            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date.isoformat(),
                "include_indicators": str(include_indicators).lower()
            }

            if end_date:
                params["end_date"] = end_date.isoformat()

            url = f"{self.base_url}/api/v1/market-data"

            logger.info(f"Getting market data for {symbol} {timeframe} from {start_date} to {end_date}")
            response = await self.client.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()

            # Convert to DataFrame
            df = pd.DataFrame(data["data"])

            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            return df

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when getting market data: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error when getting market data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when getting market data: {e}")
            raise

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(timeout_seconds=10)
    async def get_symbols(self) -> List[Dict[str, Any]]:
        """
        Get available symbols from the Data Pipeline Service.

        Returns:
            List of available symbols
        """
        try:
            request_id = str(uuid.uuid4())
            headers = {"X-Request-ID": request_id}

            url = f"{self.base_url}/api/v1/symbols"

            logger.info("Getting available symbols")
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()

            return data["symbols"]

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when getting symbols: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error when getting symbols: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when getting symbols: {e}")
            raise

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(timeout_seconds=10)
    async def get_timeframes(self) -> List[Dict[str, Any]]:
        """
        Get available timeframes from the Data Pipeline Service.

        Returns:
            List of available timeframes
        """
        try:
            request_id = str(uuid.uuid4())
            headers = {"X-Request-ID": request_id}

            url = f"{self.base_url}/api/v1/timeframes"

            logger.info("Getting available timeframes")
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()

            return data["timeframes"]

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when getting timeframes: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error when getting timeframes: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when getting timeframes: {e}")
            raise