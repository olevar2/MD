"""
News Data Adapter.

This module provides adapters for news data sources.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import json
import aiohttp

import pandas as pd

from common_lib.exceptions import DataFetchError, DataValidationError
from adapters.base_adapter import BaseAlternativeDataAdapter
from models.models import AlternativeDataType

logger = logging.getLogger(__name__)


class NewsDataAdapter(BaseAlternativeDataAdapter):
    """Adapter for news data sources."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this adapter.

        Returns:
            List of supported data types
        """
        return [AlternativeDataType.NEWS]

    async def _fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch news data from the source.

        Args:
            data_type: Type of alternative data
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters

        Returns:
            DataFrame containing the news data
        """
        # Get API endpoint and key from config
        api_endpoint = self.config.get("api_endpoint")
        api_key = self.config.get("api_key")
        
        if not api_endpoint:
            raise ValueError("API endpoint not configured for news adapter")
        
        # Prepare request parameters
        params = {
            "symbols": ",".join(symbols),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
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
                        raise DataFetchError(f"Failed to fetch news data: {response.status} - {error_text}")
                    
                    data = await response.json()
            
            # Convert to DataFrame
            if not data or "items" not in data:
                return pd.DataFrame()
            
            news_items = data["items"]
            if not news_items:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(news_items)
            
            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Add source_id column
            df["source_id"] = self.source_id
            
            return df
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            raise DataFetchError(f"Failed to fetch news data: {str(e)}")


class MockNewsDataAdapter(BaseAlternativeDataAdapter):
    """Mock adapter for news data (for testing and development)."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this adapter.

        Returns:
            List of supported data types
        """
        return [AlternativeDataType.NEWS]

    async def _fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate mock news data.

        Args:
            data_type: Type of alternative data
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters

        Returns:
            DataFrame containing mock news data
        """
        # Generate mock news data
        news_items = []
        
        # Define some news templates
        news_templates = [
            {"title": "Central Bank Rate Decision", "impact": "high"},
            {"title": "GDP Report", "impact": "high"},
            {"title": "Inflation Data", "impact": "high"},
            {"title": "Employment Report", "impact": "high"},
            {"title": "Trade Balance", "impact": "medium"},
            {"title": "Retail Sales", "impact": "medium"},
            {"title": "Manufacturing PMI", "impact": "medium"},
            {"title": "Consumer Confidence", "impact": "medium"},
            {"title": "Housing Data", "impact": "low"},
            {"title": "Industrial Production", "impact": "low"}
        ]
        
        # Generate news items for each day in the date range
        current_date = start_date
        while current_date <= end_date:
            # Generate 1-3 news items per day
            num_items = min(3, max(1, hash(current_date.isoformat()) % 4))
            
            for i in range(num_items):
                # Select a symbol
                symbol = symbols[hash((current_date.isoformat(), i)) % len(symbols)]
                
                # Select a news template
                template = news_templates[hash((current_date.isoformat(), i, symbol)) % len(news_templates)]
                
                # Generate a news item
                news_item = {
                    "id": f"news_{hash((current_date.isoformat(), i, symbol)) % 10000}",
                    "title": f"{symbol}: {template['title']}",
                    "content": f"This is a mock news item about {symbol} regarding {template['title'].lower()}.",
                    "source": "Mock News Provider",
                    "timestamp": (current_date + timedelta(hours=hash((current_date.isoformat(), i)) % 24)).isoformat(),
                    "impact": template["impact"],
                    "symbols": [symbol],
                    "url": f"https://mocknews.example.com/{symbol}/{current_date.strftime('%Y-%m-%d')}/{i}"
                }
                
                news_items.append(news_item)
            
            current_date += timedelta(days=1)
        
        # Create DataFrame
        df = pd.DataFrame(news_items)
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Add source_id column
        df["source_id"] = self.source_id
        
        return df
