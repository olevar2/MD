"""
Social Media Data Adapter.

This module provides adapters for social media data sources.
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


class SocialMediaDataAdapter(BaseAlternativeDataAdapter):
    """Adapter for social media data sources."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this adapter.

        Returns:
            List of supported data types
        """
        return [AlternativeDataType.SOCIAL_MEDIA]

    async def _fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch social media data from the source.

        Args:
            data_type: Type of alternative data
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters

        Returns:
            DataFrame containing the social media data
        """
        # Get API endpoint and key from config
        api_endpoint = self.config.get("api_endpoint")
        api_key = self.config.get("api_key")
        
        if not api_endpoint:
            raise ValueError("API endpoint not configured for social media data adapter")
        
        # Prepare request parameters
        params = {
            "symbols": ",".join(symbols),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "platforms": kwargs.get("platforms", ""),
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
                        raise DataFetchError(f"Failed to fetch social media data: {response.status} - {error_text}")
                    
                    data = await response.json()
            
            # Convert to DataFrame
            if not data or "items" not in data:
                return pd.DataFrame()
            
            social_media_items = data["items"]
            if not social_media_items:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(social_media_items)
            
            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Add source_id column
            df["source_id"] = self.source_id
            
            return df
        except Exception as e:
            logger.error(f"Error fetching social media data: {str(e)}")
            raise DataFetchError(f"Failed to fetch social media data: {str(e)}")


class MockSocialMediaDataAdapter(BaseAlternativeDataAdapter):
    """Mock adapter for social media data (for testing and development)."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this adapter.

        Returns:
            List of supported data types
        """
        return [AlternativeDataType.SOCIAL_MEDIA]

    async def _fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate mock social media data.

        Args:
            data_type: Type of alternative data
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters

        Returns:
            DataFrame containing mock social media data
        """
        # Define social media platforms
        platforms = kwargs.get("platforms", ["twitter", "reddit", "stocktwits"])
        
        if isinstance(platforms, str):
            platforms = platforms.split(",")
        
        # Generate mock social media data
        social_media_items = []
        
        # Base metrics for each symbol
        base_metrics = {
            "EUR/USD": {"volume": 100, "sentiment": 0.2, "influence": 70},
            "GBP/USD": {"volume": 80, "sentiment": -0.1, "influence": 65},
            "USD/JPY": {"volume": 90, "sentiment": 0.0, "influence": 60},
            "AUD/USD": {"volume": 60, "sentiment": 0.3, "influence": 50},
            "USD/CAD": {"volume": 50, "sentiment": -0.2, "influence": 45},
            "EUR/GBP": {"volume": 40, "sentiment": 0.1, "influence": 40},
            "USD/CHF": {"volume": 30, "sentiment": -0.1, "influence": 35},
            "NZD/USD": {"volume": 20, "sentiment": 0.2, "influence": 30},
            "EUR/JPY": {"volume": 25, "sentiment": 0.0, "influence": 35},
            "GBP/JPY": {"volume": 15, "sentiment": -0.1, "influence": 25}
        }
        
        # Platform-specific multipliers
        platform_multipliers = {
            "twitter": {"volume": 1.5, "sentiment_var": 0.3, "influence": 1.2},
            "reddit": {"volume": 0.8, "sentiment_var": 0.2, "influence": 0.9},
            "stocktwits": {"volume": 1.0, "sentiment_var": 0.25, "influence": 1.0}
        }
        
        # Generate data for each day in the date range
        current_date = start_date
        while current_date <= end_date:
            # For each symbol
            for symbol in symbols:
                # Default metrics if symbol not in base_metrics
                base_metric = base_metrics.get(symbol, {"volume": 50, "sentiment": 0.0, "influence": 40})
                
                # For each platform
                for platform in platforms:
                    # Get platform multipliers
                    multiplier = platform_multipliers.get(platform, {"volume": 1.0, "sentiment_var": 0.2, "influence": 1.0})
                    
                    # Add some random variation
                    # Use hash of date, symbol, and platform for deterministic randomness
                    seed = hash(f"{current_date.isoformat()}_{symbol}_{platform}") % 10000
                    np.random.seed(seed)
                    
                    # Calculate metrics with variation
                    volume = int(base_metric["volume"] * multiplier["volume"] * np.random.normal(1, 0.3))
                    sentiment = max(-1.0, min(1.0, base_metric["sentiment"] + np.random.normal(0, multiplier["sentiment_var"])))
                    influence = int(base_metric["influence"] * multiplier["influence"] * np.random.normal(1, 0.2))
                    
                    # Skip if volume is 0
                    if volume <= 0:
                        continue
                    
                    # Create social media item
                    social_media_item = {
                        "id": f"{symbol}_{platform}_{current_date.strftime('%Y%m%d')}",
                        "symbol": symbol,
                        "platform": platform,
                        "timestamp": current_date.isoformat(),
                        "volume": volume,
                        "sentiment_score": round(sentiment, 2),
                        "influence_score": influence,
                        "positive_count": int(volume * max(0, min(1, 0.5 + sentiment / 2))),
                        "negative_count": int(volume * max(0, min(1, 0.5 - sentiment / 2))),
                        "neutral_count": int(volume * max(0, min(1, 1 - abs(sentiment)))),
                        "trending_score": int(volume * influence / 100)
                    }
                    
                    social_media_items.append(social_media_item)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Create DataFrame
        df = pd.DataFrame(social_media_items)
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Add source_id column
        df["source_id"] = self.source_id
        
        return df
