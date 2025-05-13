"""
Economic Data Adapter.

This module provides adapters for economic data sources.
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


class EconomicDataAdapter(BaseAlternativeDataAdapter):
    """Adapter for economic data sources."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this adapter.

        Returns:
            List of supported data types
        """
        return [AlternativeDataType.ECONOMIC]

    async def _fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch economic data from the source.

        Args:
            data_type: Type of alternative data
            symbols: List of symbols (currency codes or country codes)
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters

        Returns:
            DataFrame containing the economic data
        """
        # Get API endpoint and key from config
        api_endpoint = self.config.get("api_endpoint")
        api_key = self.config.get("api_key")
        
        if not api_endpoint:
            raise ValueError("API endpoint not configured for economic data adapter")
        
        # Prepare request parameters
        params = {
            "countries": ",".join(symbols),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "indicators": kwargs.get("indicators", ""),
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
                        raise DataFetchError(f"Failed to fetch economic data: {response.status} - {error_text}")
                    
                    data = await response.json()
            
            # Convert to DataFrame
            if not data or "items" not in data:
                return pd.DataFrame()
            
            economic_items = data["items"]
            if not economic_items:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(economic_items)
            
            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Add source_id column
            df["source_id"] = self.source_id
            
            return df
        except Exception as e:
            logger.error(f"Error fetching economic data: {str(e)}")
            raise DataFetchError(f"Failed to fetch economic data: {str(e)}")


class MockEconomicDataAdapter(BaseAlternativeDataAdapter):
    """Mock adapter for economic data (for testing and development)."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this adapter.

        Returns:
            List of supported data types
        """
        return [AlternativeDataType.ECONOMIC]

    async def _fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate mock economic data.

        Args:
            data_type: Type of alternative data
            symbols: List of symbols (currency codes or country codes)
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters

        Returns:
            DataFrame containing mock economic data
        """
        # Define economic indicators
        indicators = kwargs.get("indicators", [
            "gdp", "inflation", "unemployment", "interest_rate", 
            "trade_balance", "retail_sales", "industrial_production"
        ])
        
        if isinstance(indicators, str):
            indicators = indicators.split(",")
        
        # Generate mock economic data
        economic_items = []
        
        # Base values for each indicator by country
        base_values = {
            "USD": {"gdp": 21.4, "inflation": 2.1, "unemployment": 3.8, "interest_rate": 1.75, 
                   "trade_balance": -50.0, "retail_sales": 0.5, "industrial_production": 0.3},
            "EUR": {"gdp": 13.3, "inflation": 1.8, "unemployment": 7.5, "interest_rate": 0.0, 
                   "trade_balance": 20.0, "retail_sales": 0.3, "industrial_production": 0.2},
            "GBP": {"gdp": 2.8, "inflation": 2.0, "unemployment": 4.0, "interest_rate": 0.75, 
                   "trade_balance": -5.0, "retail_sales": 0.4, "industrial_production": 0.1},
            "JPY": {"gdp": 5.0, "inflation": 0.5, "unemployment": 2.5, "interest_rate": -0.1, 
                   "trade_balance": 10.0, "retail_sales": 0.2, "industrial_production": 0.4},
            "AUD": {"gdp": 1.4, "inflation": 1.6, "unemployment": 5.2, "interest_rate": 1.0, 
                   "trade_balance": 2.0, "retail_sales": 0.3, "industrial_production": 0.2},
            "CAD": {"gdp": 1.7, "inflation": 1.9, "unemployment": 5.7, "interest_rate": 1.75, 
                   "trade_balance": -1.0, "retail_sales": 0.4, "industrial_production": 0.3},
            "CHF": {"gdp": 0.7, "inflation": 0.4, "unemployment": 2.3, "interest_rate": -0.75, 
                   "trade_balance": 3.0, "retail_sales": 0.2, "industrial_production": 0.1},
            "CNY": {"gdp": 14.3, "inflation": 2.5, "unemployment": 3.6, "interest_rate": 4.35, 
                   "trade_balance": 40.0, "retail_sales": 0.8, "industrial_production": 0.6}
        }
        
        # Generate data for each month in the date range
        current_date = start_date.replace(day=1)  # Start at the beginning of the month
        while current_date <= end_date:
            # For each country/symbol
            for symbol in symbols:
                # Default to USD if symbol not in base_values
                country_base = base_values.get(symbol, base_values["USD"])
                
                # For each indicator
                for indicator in indicators:
                    if indicator in country_base:
                        # Get base value
                        base_value = country_base[indicator]
                        
                        # Add some random variation
                        # Use hash of date and symbol for deterministic randomness
                        seed = hash(f"{current_date.isoformat()}_{symbol}_{indicator}") % 10000
                        np.random.seed(seed)
                        
                        # Different variation for different indicators
                        if indicator == "gdp":
                            # Quarterly data with seasonal pattern
                            if current_date.month % 3 == 0:  # Only generate for end of quarter
                                variation = np.random.normal(0, 0.2)  # GDP growth rate variation
                                value = base_value * (1 + variation / 100)  # Convert to percentage
                                
                                economic_items.append({
                                    "id": f"{symbol}_{indicator}_{current_date.strftime('%Y%m%d')}",
                                    "country": symbol,
                                    "indicator": indicator,
                                    "timestamp": current_date.isoformat(),
                                    "value": round(value, 2),
                                    "previous": round(base_value, 2),
                                    "forecast": round(base_value * (1 + np.random.normal(0, 0.1) / 100), 2),
                                    "period": "quarterly"
                                })
                        else:
                            # Monthly data
                            variation_pct = {
                                "inflation": np.random.normal(0, 0.1),
                                "unemployment": np.random.normal(0, 0.2),
                                "interest_rate": np.random.normal(0, 0.05),
                                "trade_balance": np.random.normal(0, 2.0),
                                "retail_sales": np.random.normal(0, 0.3),
                                "industrial_production": np.random.normal(0, 0.2)
                            }.get(indicator, np.random.normal(0, 0.1))
                            
                            value = base_value + variation_pct
                            
                            economic_items.append({
                                "id": f"{symbol}_{indicator}_{current_date.strftime('%Y%m%d')}",
                                "country": symbol,
                                "indicator": indicator,
                                "timestamp": current_date.isoformat(),
                                "value": round(value, 2),
                                "previous": round(base_value, 2),
                                "forecast": round(base_value + np.random.normal(0, 0.05), 2),
                                "period": "monthly"
                            })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        # Create DataFrame
        df = pd.DataFrame(economic_items)
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Add source_id column
        df["source_id"] = self.source_id
        
        return df
