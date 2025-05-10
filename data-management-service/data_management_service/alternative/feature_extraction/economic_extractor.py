"""
Economic Data Feature Extractor.

This module provides feature extractors for economic data.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

from common_lib.exceptions import DataProcessingError
from data_management_service.alternative.feature_extraction.base_extractor import BaseFeatureExtractor
from data_management_service.alternative.models import AlternativeDataType

logger = logging.getLogger(__name__)


class EconomicFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for economic data."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this feature extractor.

        Returns:
            List of supported data types
        """
        return [AlternativeDataType.ECONOMIC]

    async def _extract_features_impl(
        self,
        data: pd.DataFrame,
        data_type: str,
        feature_config: Dict[str, Any],
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract features from economic data.

        Args:
            data: DataFrame containing the economic data
            data_type: Type of alternative data
            feature_config: Configuration for feature extraction
            **kwargs: Additional parameters specific to the feature extraction

        Returns:
            DataFrame containing the extracted features
        """
        if data.empty:
            return pd.DataFrame()
        
        # Check required columns
        required_columns = ["timestamp", "country", "indicator", "value"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in economic data: {missing_columns}")
        
        # Get extraction parameters
        extraction_method = feature_config.get("extraction_method", "basic")
        
        if extraction_method == "basic":
            return await self._extract_basic_features(data, feature_config)
        elif extraction_method == "surprise":
            return await self._extract_surprise_features(data, feature_config)
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")

    async def _extract_basic_features(
        self,
        data: pd.DataFrame,
        feature_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Extract basic features from economic data.

        Args:
            data: DataFrame containing the economic data
            feature_config: Configuration for feature extraction

        Returns:
            DataFrame containing the extracted features
        """
        # Get parameters
        resample_freq = feature_config.get("resample_freq", "D")
        indicators = feature_config.get("indicators", None)
        
        # Filter indicators if specified
        if indicators:
            data = data[data["indicator"].isin(indicators)]
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])
        
        # Pivot the data to get indicators as columns
        pivot_data = data.pivot_table(
            index=["timestamp", "country"],
            columns="indicator",
            values="value",
            aggfunc="mean"
        ).reset_index()
        
        # Resample if needed
        if resample_freq:
            # Group by country and resample
            features_list = []
            for country, group in pivot_data.groupby("country"):
                resampled = group.set_index("timestamp").resample(resample_freq).mean()
                resampled["country"] = country
                features_list.append(resampled.reset_index())
            
            features = pd.concat(features_list, ignore_index=True)
        else:
            features = pivot_data
        
        # Fill NaN values with forward fill then backward fill
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features.groupby("country")[numeric_cols].transform(
            lambda x: x.fillna(method="ffill").fillna(method="bfill")
        )
        
        return features

    async def _extract_surprise_features(
        self,
        data: pd.DataFrame,
        feature_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Extract surprise-based features from economic data.

        Args:
            data: DataFrame containing the economic data
            feature_config: Configuration for feature extraction

        Returns:
            DataFrame containing the extracted features
        """
        # Get parameters
        resample_freq = feature_config.get("resample_freq", "D")
        indicators = feature_config.get("indicators", None)
        
        # Filter indicators if specified
        if indicators:
            data = data[data["indicator"].isin(indicators)]
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])
        
        # Check if forecast and previous values are available
        has_forecast = "forecast" in data.columns
        has_previous = "previous" in data.columns
        
        if not has_forecast and not has_previous:
            logger.warning("No forecast or previous values available for surprise calculation")
            return await self._extract_basic_features(data, feature_config)
        
        # Calculate surprise metrics
        if has_forecast:
            data["surprise"] = data["value"] - data["forecast"]
            data["surprise_pct"] = data["surprise"] / data["forecast"].abs()
        
        if has_previous:
            data["change"] = data["value"] - data["previous"]
            data["change_pct"] = data["change"] / data["previous"].abs()
        
        # Create feature columns for each indicator
        features_list = []
        
        for indicator, group in data.groupby("indicator"):
            # Pivot to get countries as columns
            pivot_cols = ["timestamp"]
            if has_forecast:
                pivot_cols.extend(["surprise", "surprise_pct"])
            if has_previous:
                pivot_cols.extend(["change", "change_pct"])
            
            for col in pivot_cols[1:]:  # Skip timestamp
                pivot = group.pivot_table(
                    index="timestamp",
                    columns="country",
                    values=col,
                    aggfunc="mean"
                ).reset_index()
                
                # Rename columns to include indicator and metric
                pivot.columns = [
                    f"{indicator}_{col}_{country}" if i > 0 else "timestamp"
                    for i, country in enumerate(pivot.columns)
                ]
                
                features_list.append(pivot)
        
        # Merge all features on timestamp
        features = features_list[0]
        for df in features_list[1:]:
            features = pd.merge(features, df, on="timestamp", how="outer")
        
        # Resample if needed
        if resample_freq:
            features = features.set_index("timestamp").resample(resample_freq).mean().reset_index()
        
        # Fill NaN values
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(method="ffill").fillna(method="bfill").fillna(0)
        
        return features
