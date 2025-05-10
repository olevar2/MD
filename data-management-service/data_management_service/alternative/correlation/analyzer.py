"""
Correlation Analyzer.

This module provides functionality for analyzing correlations between alternative data and market movements.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
from scipy import stats

from common_lib.exceptions import DataProcessingError
from common_lib.interfaces.alternative_data import ICorrelationAnalyzer
from data_management_service.alternative.models import (
    AlternativeDataType,
    CorrelationMetric,
    AlternativeDataCorrelation
)

logger = logging.getLogger(__name__)


class BaseCorrelationAnalyzer(ICorrelationAnalyzer, ABC):
    """Base class for correlation analyzers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the correlation analyzer.

        Args:
            config: Configuration for the analyzer
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.supported_data_types = self._get_supported_data_types()
        
        logger.info(f"Initialized {self.name} correlation analyzer")

    @abstractmethod
    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this analyzer.

        Returns:
            List of supported data types
        """
        pass

    @abstractmethod
    async def _analyze_correlation_impl(
        self,
        alternative_data: pd.DataFrame,
        market_data: pd.DataFrame,
        data_type: str,
        instruments: List[str],
        timeframes: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Implementation of correlation analysis for specific data types.

        Args:
            alternative_data: DataFrame containing the alternative data
            market_data: DataFrame containing market data
            data_type: Type of alternative data
            instruments: List of instruments to analyze
            timeframes: List of timeframes to analyze
            **kwargs: Additional parameters specific to correlation analysis

        Returns:
            Dictionary containing correlation analysis results
        """
        pass

    async def analyze_correlation(
        self,
        alternative_data: pd.DataFrame,
        market_data: pd.DataFrame,
        data_type: str,
        instruments: List[str],
        timeframes: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze correlations between alternative data and market movements.

        Args:
            alternative_data: DataFrame containing the alternative data
            market_data: DataFrame containing market data
            data_type: Type of alternative data
            instruments: List of instruments to analyze
            timeframes: List of timeframes to analyze
            **kwargs: Additional parameters specific to correlation analysis

        Returns:
            Dictionary containing correlation analysis results
        """
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
            raise ValueError(f"Data type '{data_type_enum}' is not supported by this analyzer")
        
        try:
            # Analyze correlation
            results = await self._analyze_correlation_impl(
                alternative_data=alternative_data,
                market_data=market_data,
                data_type=data_type_enum.value,
                instruments=instruments,
                timeframes=timeframes,
                **kwargs
            )
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing correlation for data type {data_type_enum}: {str(e)}")
            raise DataProcessingError(f"Failed to analyze correlation: {str(e)}")


class StandardCorrelationAnalyzer(BaseCorrelationAnalyzer):
    """Standard correlation analyzer for alternative data."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this analyzer.

        Returns:
            List of supported data types
        """
        return [
            AlternativeDataType.NEWS,
            AlternativeDataType.ECONOMIC,
            AlternativeDataType.SENTIMENT,
            AlternativeDataType.SOCIAL_MEDIA
        ]

    async def _analyze_correlation_impl(
        self,
        alternative_data: pd.DataFrame,
        market_data: pd.DataFrame,
        data_type: str,
        instruments: List[str],
        timeframes: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze correlation between alternative data and market movements.

        Args:
            alternative_data: DataFrame containing the alternative data
            market_data: DataFrame containing market data
            data_type: Type of alternative data
            instruments: List of instruments to analyze
            timeframes: List of timeframes to analyze
            **kwargs: Additional parameters specific to correlation analysis

        Returns:
            Dictionary containing correlation analysis results
        """
        if alternative_data.empty or market_data.empty:
            return {"metrics": [], "summary": "No data available for correlation analysis"}
        
        # Get configuration
        lag_periods = kwargs.get("lag_periods", [0, 1, 2, 3, 5, 10])
        correlation_method = kwargs.get("correlation_method", "pearson")
        min_periods = kwargs.get("min_periods", 10)
        
        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(alternative_data["timestamp"]):
            alternative_data["timestamp"] = pd.to_datetime(alternative_data["timestamp"])
        
        if not pd.api.types.is_datetime64_any_dtype(market_data["timestamp"]):
            market_data["timestamp"] = pd.to_datetime(market_data["timestamp"])
        
        # Set timestamp as index
        alt_data = alternative_data.set_index("timestamp")
        mkt_data = market_data.set_index("timestamp")
        
        # Get alternative data feature columns
        if data_type == AlternativeDataType.NEWS:
            feature_cols = ["sentiment_score", "news_count"]
        elif data_type == AlternativeDataType.ECONOMIC:
            feature_cols = [col for col in alt_data.columns if "surprise" in col or "change" in col]
        elif data_type == AlternativeDataType.SENTIMENT:
            feature_cols = ["sentiment_score", "volume"]
        elif data_type == AlternativeDataType.SOCIAL_MEDIA:
            feature_cols = ["sentiment_score", "volume", "influence_score"]
        else:
            feature_cols = [col for col in alt_data.columns if col not in ["symbol", "source", "id"]]
        
        # Filter to available columns
        feature_cols = [col for col in feature_cols if col in alt_data.columns]
        
        if not feature_cols:
            return {"metrics": [], "summary": "No suitable feature columns found in alternative data"}
        
        # Get market data columns
        market_cols = []
        for instrument in instruments:
            for timeframe in timeframes:
                # Look for price columns
                price_cols = [
                    f"{instrument}_{timeframe}_close",
                    f"{instrument}_{timeframe}_return",
                    f"{instrument}_{timeframe}_volatility"
                ]
                
                # Filter to available columns
                available_cols = [col for col in price_cols if col in mkt_data.columns]
                market_cols.extend(available_cols)
        
        if not market_cols:
            return {"metrics": [], "summary": "No suitable market data columns found"}
        
        # Calculate correlations
        metrics = []
        
        for feature_col in feature_cols:
            for market_col in market_cols:
                # Extract instrument and timeframe from market column
                parts = market_col.split("_")
                if len(parts) < 3:
                    continue
                
                instrument = parts[0]
                timeframe = parts[1]
                
                # Calculate correlation for different lag periods
                for lag in lag_periods:
                    # Shift market data by lag periods
                    if lag > 0:
                        shifted_market = mkt_data[market_col].shift(-lag)
                    else:
                        shifted_market = mkt_data[market_col]
                    
                    # Align data
                    aligned_data = pd.concat([alt_data[feature_col], shifted_market], axis=1).dropna()
                    
                    # Skip if not enough data
                    if len(aligned_data) < min_periods:
                        continue
                    
                    # Calculate correlation
                    if correlation_method == "pearson":
                        corr, p_value = stats.pearsonr(aligned_data[feature_col], aligned_data[market_col])
                    elif correlation_method == "spearman":
                        corr, p_value = stats.spearmanr(aligned_data[feature_col], aligned_data[market_col])
                    else:
                        corr, p_value = stats.kendalltau(aligned_data[feature_col], aligned_data[market_col])
                    
                    # Create correlation metric
                    metric = CorrelationMetric(
                        instrument=instrument,
                        timeframe=timeframe,
                        correlation_value=float(corr),
                        p_value=float(p_value),
                        sample_size=len(aligned_data),
                        start_date=aligned_data.index.min(),
                        end_date=aligned_data.index.max(),
                        methodology=f"{correlation_method}_lag{lag}",
                        notes=f"Correlation between {feature_col} and {market_col} with lag {lag}"
                    )
                    
                    metrics.append(metric.dict())
        
        # Create summary
        if metrics:
            # Find strongest correlations
            sorted_metrics = sorted(metrics, key=lambda x: abs(x["correlation_value"]), reverse=True)
            top_metrics = sorted_metrics[:5]
            
            summary = f"Analyzed {len(metrics)} correlations between {data_type} data and market movements. "
            summary += f"Strongest correlation: {top_metrics[0]['correlation_value']:.2f} "
            summary += f"({top_metrics[0]['instrument']} {top_metrics[0]['timeframe']}, {top_metrics[0]['methodology']})"
        else:
            summary = f"No significant correlations found between {data_type} data and market movements."
        
        return {
            "metrics": metrics,
            "summary": summary
        }
