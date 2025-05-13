"""
Sentiment Data Adapter.

This module provides adapters for sentiment data sources.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import json
import aiohttp

import pandas as pd
import numpy as np

from common_lib.exceptions import DataFetchError, DataValidationError
from adapters.base_adapter import BaseAlternativeDataAdapter
from models.models import AlternativeDataType

logger = logging.getLogger(__name__)


class SentimentDataAdapter(BaseAlternativeDataAdapter):
    """Adapter for sentiment data sources."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this adapter.

        Returns:
            List of supported data types
        """
        return [AlternativeDataType.SENTIMENT]

    async def _fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch sentiment data from the source.

        Args:
            data_type: Type of alternative data
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters

        Returns:
            DataFrame containing the sentiment data
        """
        # Get API endpoint and key from config
        api_endpoint = self.config.get("api_endpoint")
        api_key = self.config.get("api_key")
        
        if not api_endpoint:
            raise ValueError("API endpoint not configured for sentiment data adapter")
        
        # Prepare request parameters
        params = {
            "symbols": ",".join(symbols),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "sources": kwargs.get("sources", ""),
            "limit": kwargs.get("limit", 100),
            "offset": kwargs.get("offset", 0)
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value
        
        # Prepare headers
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.get(api_endpoint, params=params, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise DataFetchError(f"Failed to fetch sentiment data: {response.status} - {error_text}")
                    
                    data = await response.json()
            
            # Convert to DataFrame
            if not data or "items" not in data:
                return pd.DataFrame()
            
            sentiment_items = data["items"]
            if not sentiment_items:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(sentiment_items)
            
            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Add source_id column
            df["source_id"] = self.source_id
            
            return df
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {str(e)}")
            raise DataFetchError(f"Failed to fetch sentiment data: {str(e)}")


class MockSentimentDataAdapter(BaseAlternativeDataAdapter):
    """Mock adapter for sentiment data (for testing and development)."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this adapter.

        Returns:
            List of supported data types
        """
        return [AlternativeDataType.SENTIMENT]

    async def _fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate mock sentiment data.

        Args:
            data_type: Type of alternative data
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters

        Returns:
            DataFrame containing mock sentiment data
        """
        # Define sentiment sources
        sources = kwargs.get("sources", ["news", "social_media", "analyst_ratings"])
        
        if isinstance(sources, str):
            sources = sources.split(",")
        
        # Generate mock sentiment data
        sentiment_items = []
        
        # Base sentiment values for each symbol
        base_sentiments = {
            "EUR/USD": 0.2,
            "GBP/USD": -0.1,
            "USD/JPY": 0.0,
            "AUD/USD": 0.3,
            "USD/CAD": -0.2,
            "EUR/GBP": 0.1,
            "USD/CHF": -0.1,
            "NZD/USD": 0.2,
            "EUR/JPY": 0.0,
            "GBP/JPY": -0.1
        }
        
        # Generate data for each day in the date range
        current_date = start_date
        while current_date <= end_date:
            # For each symbol
            for symbol in symbols:
                # Default sentiment if symbol not in base_sentiments
                base_sentiment = base_sentiments.get(symbol, 0.0)
                
                # For each source
                for source in sources:
                    # Add some random variation
                    # Use hash of date, symbol, and source for deterministic randomness
                    seed = hash(f"{current_date.isoformat()}_{symbol}_{source}") % 10000
                    np.random.seed(seed)
                    
                    # Different variation for different sources
                    if source == "news":
                        variation = np.random.normal(0, 0.2)
                    elif source == "social_media":
                        variation = np.random.normal(0, 0.3)  # More volatile
                    else:  # analyst_ratings
                        variation = np.random.normal(0, 0.1)  # More stable
                    
                    # Calculate sentiment score (-1 to 1)
                    sentiment_score = max(-1.0, min(1.0, base_sentiment + variation))
                    
                    # Calculate volume (amount of data points)
                    if source == "news":
                        volume = int(np.random.poisson(10))  # Average 10 news articles
                    elif source == "social_media":
                        volume = int(np.random.poisson(50))  # Average 50 social media posts
                    else:  # analyst_ratings
                        volume = int(np.random.poisson(5))   # Average 5 analyst ratings
                    
                    # Skip if volume is 0
                    if volume == 0:
                        continue
                    
                    # Create sentiment item
                    sentiment_item = {
                        "id": f"{symbol}_{source}_{current_date.strftime('%Y%m%d')}",
                        "symbol": symbol,
                        "source": source,
                        "timestamp": current_date.isoformat(),
                        "sentiment_score": round(sentiment_score, 2),
                        "volume": volume,
                        "positive_ratio": round(max(0, min(1, 0.5 + sentiment_score / 2)), 2),
                        "negative_ratio": round(max(0, min(1, 0.5 - sentiment_score / 2)), 2),
                        "neutral_ratio": round(max(0, min(1, 1 - abs(sentiment_score))), 2)
                    }
                    
                    sentiment_items.append(sentiment_item)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Create DataFrame
        df = pd.DataFrame(sentiment_items)
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Add source_id column
        df["source_id"] = self.source_id
        
        return df
