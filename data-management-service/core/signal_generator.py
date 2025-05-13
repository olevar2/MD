"""
Trading Signal Generator.

This module provides functionality for generating trading signals from alternative data.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

from common_lib.exceptions import DataProcessingError
from common_lib.interfaces.alternative_data import ITradingSignalGenerator
from models.models import AlternativeDataType

logger = logging.getLogger(__name__)


class BaseTradingSignalGenerator(ITradingSignalGenerator, ABC):
    """Base class for trading signal generators."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading signal generator.

        Args:
            config: Configuration for the signal generator
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.supported_data_types = self._get_supported_data_types()
        
        logger.info(f"Initialized {self.name} trading signal generator")

    @abstractmethod
    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this signal generator.

        Returns:
            List of supported data types
        """
        pass

    @abstractmethod
    async def _generate_signals_impl(
        self,
        alternative_data: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Implementation of signal generation for specific data types.

        Args:
            alternative_data: DataFrame containing the alternative data
            market_data: Optional DataFrame containing market data
            config: Optional configuration for signal generation
            **kwargs: Additional parameters specific to signal generation

        Returns:
            DataFrame containing the generated signals
        """
        pass

    async def generate_signals(
        self,
        alternative_data: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate trading signals from alternative data.

        Args:
            alternative_data: DataFrame containing the alternative data
            market_data: Optional DataFrame containing market data
            config: Optional configuration for signal generation
            **kwargs: Additional parameters specific to signal generation

        Returns:
            DataFrame containing the generated signals
        """
        # Get data type from kwargs or try to infer from data
        data_type = kwargs.get("data_type")
        if not data_type and "data_type" in alternative_data.columns:
            data_type = alternative_data["data_type"].iloc[0]
        
        if not data_type:
            raise ValueError("Data type not specified and could not be inferred from data")
        
        # Convert string to enum if needed
        if isinstance(data_type, str):
            try:
                data_type_enum = AlternativeDataType(data_type)
            except ValueError:
                raise ValueError(f"Unknown data type: {data_type}")
        else:
            data_type_enum = data_type
        
        # Check if data type is supported
        if data_type_enum.value not in self.supported_data_types:
            raise ValueError(f"Data type '{data_type_enum}' is not supported by this signal generator")
        
        try:
            # Generate signals
            signals = await self._generate_signals_impl(
                alternative_data=alternative_data,
                market_data=market_data,
                config=config or {},
                **kwargs
            )
            
            # Validate signals
            if signals is None or signals.empty:
                logger.warning(f"No signals generated for data type {data_type_enum}")
                return pd.DataFrame()
            
            return signals
        except Exception as e:
            logger.error(f"Error generating signals for data type {data_type_enum}: {str(e)}")
            raise DataProcessingError(f"Failed to generate signals: {str(e)}")


class NewsTradingSignalGenerator(BaseTradingSignalGenerator):
    """Trading signal generator for news data."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this signal generator.

        Returns:
            List of supported data types
        """
        return [AlternativeDataType.NEWS]

    async def _generate_signals_impl(
        self,
        alternative_data: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate trading signals from news data.

        Args:
            alternative_data: DataFrame containing the news data
            market_data: Optional DataFrame containing market data
            config: Optional configuration for signal generation
            **kwargs: Additional parameters specific to signal generation

        Returns:
            DataFrame containing the generated signals
        """
        if alternative_data.empty:
            return pd.DataFrame()
        
        # Get configuration
        config = config or {}
        sentiment_threshold = config.get("sentiment_threshold", 0.5)
        volume_threshold = config.get("volume_threshold", 5)
        signal_expiry_hours = config.get("signal_expiry_hours", 24)
        
        # Check required columns
        required_columns = ["timestamp", "symbols", "sentiment_score"]
        if not all(col in alternative_data.columns for col in required_columns):
            # Try to use news features instead
            if "symbols" in alternative_data.columns and "sentiment_score" in alternative_data.columns:
                # Already have the required columns
                pass
            elif "symbol" in alternative_data.columns and "sentiment_score" in alternative_data.columns:
                # Rename symbol to symbols
                alternative_data = alternative_data.rename(columns={"symbol": "symbols"})
            else:
                raise ValueError(f"Missing required columns in news data: {required_columns}")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(alternative_data["timestamp"]):
            alternative_data["timestamp"] = pd.to_datetime(alternative_data["timestamp"])
        
        # Generate signals based on sentiment and volume
        signals = []
        
        # Group by timestamp and symbol
        if isinstance(alternative_data["symbols"].iloc[0], list):
            # Explode symbols if it's a list
            data = alternative_data.explode("symbols")
        else:
            data = alternative_data
        
        for (timestamp, symbol), group in data.groupby(["timestamp", "symbols"]):
            # Calculate average sentiment
            avg_sentiment = group["sentiment_score"].mean()
            
            # Calculate volume if available
            volume = len(group)
            if "volume" in group.columns:
                volume = group["volume"].sum()
            elif "news_count" in group.columns:
                volume = group["news_count"].sum()
            
            # Generate signal if sentiment exceeds threshold and volume is sufficient
            if abs(avg_sentiment) >= sentiment_threshold and volume >= volume_threshold:
                signal_type = "buy" if avg_sentiment > 0 else "sell"
                
                # Calculate signal strength (0-100)
                signal_strength = min(100, max(0, int(abs(avg_sentiment) * 100)))
                
                # Calculate expiry time
                expiry_time = timestamp + pd.Timedelta(hours=signal_expiry_hours)
                
                # Create signal
                signal = {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "signal_type": signal_type,
                    "signal_source": "news",
                    "signal_strength": signal_strength,
                    "expiry_time": expiry_time,
                    "sentiment_score": avg_sentiment,
                    "volume": volume,
                    "confidence": min(1.0, abs(avg_sentiment) * (volume / volume_threshold))
                }
                
                signals.append(signal)
        
        # Create DataFrame
        if not signals:
            return pd.DataFrame()
        
        signals_df = pd.DataFrame(signals)
        
        return signals_df


class EconomicTradingSignalGenerator(BaseTradingSignalGenerator):
    """Trading signal generator for economic data."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this signal generator.

        Returns:
            List of supported data types
        """
        return [AlternativeDataType.ECONOMIC]

    async def _generate_signals_impl(
        self,
        alternative_data: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate trading signals from economic data.

        Args:
            alternative_data: DataFrame containing the economic data
            market_data: Optional DataFrame containing market data
            config: Optional configuration for signal generation
            **kwargs: Additional parameters specific to signal generation

        Returns:
            DataFrame containing the generated signals
        """
        if alternative_data.empty:
            return pd.DataFrame()
        
        # Get configuration
        config = config or {}
        surprise_threshold = config.get("surprise_threshold", 0.5)
        signal_expiry_hours = config.get("signal_expiry_hours", 48)
        
        # Check if we have surprise data
        has_surprise = any("surprise" in col for col in alternative_data.columns)
        
        if not has_surprise:
            logger.warning("No surprise data found in economic data, cannot generate signals")
            return pd.DataFrame()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(alternative_data["timestamp"]):
            alternative_data["timestamp"] = pd.to_datetime(alternative_data["timestamp"])
        
        # Generate signals based on economic surprises
        signals = []
        
        # Process each row
        for _, row in alternative_data.iterrows():
            timestamp = row["timestamp"]
            
            # Find surprise columns
            surprise_cols = [col for col in row.index if "surprise" in col]
            
            for col in surprise_cols:
                surprise_value = row[col]
                
                # Skip if surprise is not significant
                if abs(surprise_value) < surprise_threshold:
                    continue
                
                # Extract indicator and country from column name
                parts = col.split("_")
                if len(parts) < 3:
                    continue
                
                indicator = parts[0]
                country = parts[-1]
                
                # Determine affected symbols
                affected_symbols = []
                
                # Simple mapping of countries to currency pairs
                if country == "USD":
                    affected_symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
                elif country == "EUR":
                    affected_symbols = ["EUR/USD", "EUR/GBP", "EUR/JPY"]
                elif country == "GBP":
                    affected_symbols = ["GBP/USD", "EUR/GBP", "GBP/JPY"]
                elif country == "JPY":
                    affected_symbols = ["USD/JPY", "EUR/JPY", "GBP/JPY"]
                elif country == "AUD":
                    affected_symbols = ["AUD/USD", "AUD/JPY"]
                
                # Generate signal for each affected symbol
                for symbol in affected_symbols:
                    signal_type = "buy" if surprise_value > 0 else "sell"
                    
                    # For USD/XXX pairs, invert the signal if USD is the base currency
                    if country == "USD" and symbol.startswith("USD/"):
                        signal_type = "sell" if surprise_value > 0 else "buy"
                    
                    # Calculate signal strength (0-100)
                    signal_strength = min(100, max(0, int(abs(surprise_value) * 100)))
                    
                    # Calculate expiry time
                    expiry_time = timestamp + pd.Timedelta(hours=signal_expiry_hours)
                    
                    # Create signal
                    signal = {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "signal_type": signal_type,
                        "signal_source": f"economic_{indicator}",
                        "signal_strength": signal_strength,
                        "expiry_time": expiry_time,
                        "surprise_value": surprise_value,
                        "indicator": indicator,
                        "country": country,
                        "confidence": min(1.0, abs(surprise_value) / surprise_threshold)
                    }
                    
                    signals.append(signal)
        
        # Create DataFrame
        if not signals:
            return pd.DataFrame()
        
        signals_df = pd.DataFrame(signals)
        
        return signals_df
