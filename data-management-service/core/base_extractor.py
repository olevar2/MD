"""
Base Feature Extractor.

This module provides the base implementation for feature extractors from alternative data.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from common_lib.exceptions import DataProcessingError
from common_lib.interfaces.alternative_data import IFeatureExtractor
from models.models import AlternativeDataType, FeatureExtractionConfig

logger = logging.getLogger(__name__)


class BaseFeatureExtractor(IFeatureExtractor, ABC):
    """Base class for feature extractors from alternative data."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature extractor.

        Args:
            config: Configuration for the feature extractor
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.supported_data_types = self._get_supported_data_types()
        
        logger.info(f"Initialized {self.name} feature extractor")

    @abstractmethod
    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this feature extractor.

        Returns:
            List of supported data types
        """
        pass

    @abstractmethod
    async def _extract_features_impl(
        self,
        data: pd.DataFrame,
        data_type: str,
        feature_config: Dict[str, Any],
        **kwargs
    ) -> pd.DataFrame:
        """
        Implementation of feature extraction for specific data types.

        Args:
            data: DataFrame containing the alternative data
            data_type: Type of alternative data
            feature_config: Configuration for feature extraction
            **kwargs: Additional parameters specific to the feature extraction

        Returns:
            DataFrame containing the extracted features
        """
        pass

    async def extract_features(
        self,
        data: pd.DataFrame,
        data_type: str,
        feature_config: Dict[str, Any],
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract features from alternative data.

        Args:
            data: DataFrame containing the alternative data
            data_type: Type of alternative data
            feature_config: Configuration for feature extraction
            **kwargs: Additional parameters specific to the feature extraction

        Returns:
            DataFrame containing the extracted features
        """
        # Convert string to enum if needed
        if isinstance(data_type, str):
            try:
                data_type = AlternativeDataType(data_type)
            except ValueError:
                raise ValueError(f"Unknown data type: {data_type}")
        
        # Check if data type is supported
        if data_type.value not in self.supported_data_types:
            raise ValueError(f"Data type '{data_type}' is not supported by this feature extractor")
        
        try:
            # Extract features
            features = await self._extract_features_impl(data, data_type.value, feature_config, **kwargs)
            
            # Validate features
            if features is None or features.empty:
                logger.warning(f"No features extracted for data type {data_type}")
                return pd.DataFrame()
            
            return features
        except Exception as e:
            logger.error(f"Error extracting features for data type {data_type}: {str(e)}")
            raise DataProcessingError(f"Failed to extract features: {str(e)}")


class FeatureExtractorRegistry:
    """Registry for feature extractors."""

    def __init__(self):
        """Initialize the feature extractor registry."""
        self.extractors = {}

    def register(self, data_type: str, extractor_class: Type[BaseFeatureExtractor]) -> None:
        """
        Register a feature extractor for a data type.

        Args:
            data_type: Type of alternative data
            extractor_class: Feature extractor class
        """
        if data_type not in self.extractors:
            self.extractors[data_type] = []
        
        self.extractors[data_type].append(extractor_class)
        logger.info(f"Registered feature extractor {extractor_class.__name__} for data type {data_type}")

    def get_extractors(self, data_type: str) -> List[Type[BaseFeatureExtractor]]:
        """
        Get all feature extractors for a data type.

        Args:
            data_type: Type of alternative data

        Returns:
            List of feature extractor classes
        """
        # Convert string to enum if needed
        if isinstance(data_type, str):
            try:
                data_type = AlternativeDataType(data_type)
            except ValueError:
                raise ValueError(f"Unknown data type: {data_type}")
        
        return self.extractors.get(data_type.value, [])

    def create_extractor(
        self,
        data_type: str,
        config: Dict[str, Any]
    ) -> BaseFeatureExtractor:
        """
        Create a feature extractor for a data type.

        Args:
            data_type: Type of alternative data
            config: Configuration for the feature extractor

        Returns:
            Feature extractor instance
        """
        # Convert string to enum if needed
        if isinstance(data_type, str):
            try:
                data_type = AlternativeDataType(data_type)
            except ValueError:
                raise ValueError(f"Unknown data type: {data_type}")
        
        extractor_classes = self.get_extractors(data_type)
        if not extractor_classes:
            raise ValueError(f"No feature extractors registered for data type {data_type}")
        
        # Use the first extractor class by default
        extractor_class = extractor_classes[0]
        
        # If extractor_type is specified in config, use that one
        extractor_type = config.get("extractor_type")
        if extractor_type:
            for cls in extractor_classes:
                if cls.__name__ == extractor_type:
                    extractor_class = cls
                    break
        
        return extractor_class(config)
