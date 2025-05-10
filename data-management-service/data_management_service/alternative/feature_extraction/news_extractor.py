"""
News Feature Extractor.

This module provides feature extractors for news data.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import re

import pandas as pd
import numpy as np

from common_lib.exceptions import DataProcessingError
from data_management_service.alternative.feature_extraction.base_extractor import BaseFeatureExtractor
from data_management_service.alternative.models import AlternativeDataType

logger = logging.getLogger(__name__)


class NewsFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for news data."""

    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this feature extractor.

        Returns:
            List of supported data types
        """
        return [AlternativeDataType.NEWS]

    async def _extract_features_impl(
        self,
        data: pd.DataFrame,
        data_type: str,
        feature_config: Dict[str, Any],
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract features from news data.

        Args:
            data: DataFrame containing the news data
            data_type: Type of alternative data
            feature_config: Configuration for feature extraction
            **kwargs: Additional parameters specific to the feature extraction

        Returns:
            DataFrame containing the extracted features
        """
        if data.empty:
            return pd.DataFrame()
        
        # Check required columns
        required_columns = ["timestamp", "title", "content", "symbols"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in news data: {missing_columns}")
        
        # Get extraction parameters
        extraction_method = feature_config.get("extraction_method", "basic")
        
        if extraction_method == "basic":
            return await self._extract_basic_features(data, feature_config)
        elif extraction_method == "nlp":
            return await self._extract_nlp_features(data, feature_config)
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")

    async def _extract_basic_features(
        self,
        data: pd.DataFrame,
        feature_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Extract basic features from news data.

        Args:
            data: DataFrame containing the news data
            feature_config: Configuration for feature extraction

        Returns:
            DataFrame containing the extracted features
        """
        # Get parameters
        resample_freq = feature_config.get("resample_freq", "D")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])
        
        # Extract basic features
        
        # 1. Count news items by symbol and timestamp
        news_count = data.groupby([pd.Grouper(key="timestamp", freq=resample_freq), "symbols"]).size().reset_index(name="news_count")
        
        # Explode symbols if it's a list column
        if isinstance(data["symbols"].iloc[0], list):
            data = data.explode("symbols")
        
        # 2. Extract impact if available
        if "impact" in data.columns:
            # Convert impact to numeric if it's categorical
            if pd.api.types.is_string_dtype(data["impact"]):
                impact_map = {"low": 0.2, "medium": 0.5, "high": 1.0}
                data["impact_score"] = data["impact"].map(impact_map).fillna(0.5)
            else:
                data["impact_score"] = data["impact"]
            
            # Calculate average impact by symbol and timestamp
            impact = data.groupby([pd.Grouper(key="timestamp", freq=resample_freq), "symbols"])["impact_score"].mean().reset_index()
            
            # Merge with news count
            features = pd.merge(news_count, impact, on=["timestamp", "symbols"], how="outer")
        else:
            features = news_count
        
        # 3. Calculate sentiment if available
        if "sentiment" in data.columns:
            sentiment = data.groupby([pd.Grouper(key="timestamp", freq=resample_freq), "symbols"])["sentiment"].mean().reset_index()
            features = pd.merge(features, sentiment, on=["timestamp", "symbols"], how="outer")
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features

    async def _extract_nlp_features(
        self,
        data: pd.DataFrame,
        feature_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Extract NLP-based features from news data.

        Args:
            data: DataFrame containing the news data
            feature_config: Configuration for feature extraction

        Returns:
            DataFrame containing the extracted features
        """
        # Get parameters
        resample_freq = feature_config.get("resample_freq", "D")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])
        
        # Explode symbols if it's a list column
        if isinstance(data["symbols"].iloc[0], list):
            data = data.explode("symbols")
        
        # Extract NLP features
        
        # 1. Count news items by symbol and timestamp
        news_count = data.groupby([pd.Grouper(key="timestamp", freq=resample_freq), "symbols"]).size().reset_index(name="news_count")
        
        # 2. Extract sentiment using simple keyword-based approach
        # This is a simplified example - in a real implementation, use a proper NLP library
        positive_keywords = feature_config.get("positive_keywords", ["increase", "rise", "gain", "positive", "growth", "bullish", "up"])
        negative_keywords = feature_config.get("negative_keywords", ["decrease", "fall", "drop", "negative", "decline", "bearish", "down"])
        
        # Function to calculate simple sentiment score
        def calculate_sentiment(text):
            if not isinstance(text, str):
                return 0.0
            
            text = text.lower()
            positive_count = sum(1 for word in positive_keywords if word in text)
            negative_count = sum(1 for word in negative_keywords if word in text)
            
            if positive_count + negative_count == 0:
                return 0.0
            
            return (positive_count - negative_count) / (positive_count + negative_count)
        
        # Calculate sentiment for title and content
        data["title_sentiment"] = data["title"].apply(calculate_sentiment)
        data["content_sentiment"] = data["content"].apply(calculate_sentiment)
        
        # Combine title and content sentiment with more weight on title
        data["sentiment_score"] = data["title_sentiment"] * 0.7 + data["content_sentiment"] * 0.3
        
        # Calculate average sentiment by symbol and timestamp
        sentiment = data.groupby([pd.Grouper(key="timestamp", freq=resample_freq), "symbols"])["sentiment_score"].mean().reset_index()
        
        # 3. Calculate sentiment volatility
        sentiment_std = data.groupby([pd.Grouper(key="timestamp", freq=resample_freq), "symbols"])["sentiment_score"].std().reset_index(name="sentiment_volatility")
        
        # Merge features
        features = pd.merge(news_count, sentiment, on=["timestamp", "symbols"], how="outer")
        features = pd.merge(features, sentiment_std, on=["timestamp", "symbols"], how="outer")
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features
