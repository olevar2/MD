"""
Alternative Data Interfaces.

This module defines the interfaces for the Alternative Data Integration framework.
These interfaces are used to standardize the integration of various alternative data sources.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class IAlternativeDataProvider(ABC):
    """Interface for alternative data providers."""

    @abstractmethod
    async def get_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Retrieve alternative data for the specified parameters.

        Args:
            data_type: Type of alternative data to retrieve
            symbols: List of symbols to retrieve data for
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional parameters specific to the data type

        Returns:
            DataFrame containing the alternative data
        """
        pass

    @abstractmethod
    async def get_available_data_types(self) -> List[str]:
        """
        Get the list of available alternative data types from this provider.

        Returns:
            List of available data types
        """
        pass

    @abstractmethod
    async def get_metadata(self, data_type: str) -> Dict[str, Any]:
        """
        Get metadata about the specified data type.

        Args:
            data_type: Type of alternative data

        Returns:
            Dictionary containing metadata about the data type
        """
        pass


class IAlternativeDataTransformer(ABC):
    """Interface for alternative data transformers."""

    @abstractmethod
    async def transform(
        self,
        data: pd.DataFrame,
        data_type: str,
        transformation_rules: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Transform alternative data according to specified rules.

        Args:
            data: DataFrame containing the alternative data
            data_type: Type of alternative data
            transformation_rules: Optional list of transformation rules
            **kwargs: Additional parameters specific to the transformation

        Returns:
            Transformed DataFrame
        """
        pass


class IAlternativeDataValidator(ABC):
    """Interface for alternative data validators."""

    @abstractmethod
    async def validate(
        self,
        data: pd.DataFrame,
        data_type: str,
        validation_rules: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate alternative data according to specified rules.

        Args:
            data: DataFrame containing the alternative data
            data_type: Type of alternative data
            validation_rules: Optional list of validation rules
            **kwargs: Additional parameters specific to the validation

        Returns:
            Dictionary containing validation results
        """
        pass


class IFeatureExtractor(ABC):
    """Interface for feature extractors from alternative data."""

    @abstractmethod
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
        pass


class ITradingSignalGenerator(ABC):
    """Interface for generating trading signals from alternative data."""

    @abstractmethod
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
        pass


class ICorrelationAnalyzer(ABC):
    """Interface for analyzing correlations between alternative data and market movements."""

    @abstractmethod
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
        pass
