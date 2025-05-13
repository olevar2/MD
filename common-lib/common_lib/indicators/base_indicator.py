"""
Base Indicator Module

This module provides the base class for all indicators in the forex trading platform.
All indicators should inherit from the BaseIndicator class.

This is the standardized implementation that should be used across all services.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple, Type, ClassVar

import numpy as np
import pandas as pd
from common_lib.errors.base_exceptions import ValidationError


class IndicatorCategory(str, Enum):
    """Categories of technical indicators."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PATTERN = "pattern"
    OSCILLATOR = "oscillator"
    MOVING_AVERAGE = "moving_average"
    CUSTOM = "custom"


class BaseIndicator(ABC):
    """
    Base class for all indicators.

    This class defines the common interface and functionality for all indicators
    in the forex trading platform.

    Class Attributes:
        category: The category of the indicator
        name: The name of the indicator
        default_params: Default parameters for the indicator
        required_params: Required parameters and their types
    """

    # Class attributes to be defined by subclasses
    category: ClassVar[str] = "generic"
    name: ClassVar[str] = "indicator"
    default_params: ClassVar[Dict[str, Any]] = {}
    required_params: ClassVar[Dict[str, Type]] = {}

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        input_columns: Optional[List[str]] = None,
        output_columns: Optional[List[str]] = None
    ):
        """
        Initialize the indicator.

        Args:
            params: Parameters for the indicator
            input_columns: Input columns required by the indicator
            output_columns: Output columns produced by the indicator
        """
        # Initialize parameters with defaults and provided values
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)

        # Validate required parameters
        self._validate_params()

        # Set input and output columns
        self.input_columns = input_columns or ["open", "high", "low", "close", "volume"]
        self.output_columns = output_columns or [self.name]

        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Performance tracking
        self.last_calculation_time = 0.0
        self.last_update_time = None

    def _validate_params(self) -> None:
        """
        Validate that all required parameters are present and of the correct type.

        Raises:
            ValidationError: If a required parameter is missing or of the wrong type
        """
        for param_name, param_type in self.required_params.items():
            if param_name not in self.params:
                raise ValidationError(f"Required parameter '{param_name}' is missing")

            if not isinstance(self.params[param_name], param_type):
                raise ValidationError(
                    f"Parameter '{param_name}' should be of type {param_type.__name__}, "
                    f"got {type(self.params[param_name]).__name__}"
                )

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the indicator values.

        Args:
            data: Input data for the calculation

        Returns:
            DataFrame containing the indicator values
        """
        pass

    def calculate_with_timing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the indicator values with performance timing.

        Args:
            data: Input data for the calculation

        Returns:
            DataFrame containing the indicator values
        """
        start_time = time.time()
        result = self.calculate(data)
        self.last_calculation_time = time.time() - start_time
        self.last_update_time = datetime.now()
        return result

    def validate_input(self, data: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
        """
        Validate the input data.

        Args:
            data: Input data to validate
            required_columns: Specific columns to check for (defaults to self.input_columns)

        Returns:
            True if the input data is valid, False otherwise

        Raises:
            ValidationError: If the input data is invalid
        """
        columns_to_check = required_columns or self.input_columns

        # Check if the input data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            error_msg = "Input data is not a pandas DataFrame"
            self.logger.error(error_msg)
            raise ValidationError(error_msg)

        # Check if the input data is empty
        if data.empty:
            error_msg = "Input data is empty"
            self.logger.error(error_msg)
            raise ValidationError(error_msg)

        # Check if the input data has the required columns
        missing_columns = [col for col in columns_to_check if col not in data.columns]
        if missing_columns:
            error_msg = f"Input data is missing required columns: {missing_columns}"
            self.logger.error(error_msg)
            raise ValidationError(error_msg)

        return True

    def prepare_output(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the output data.

        Args:
            data: Input data
            result: Calculated indicator values

        Returns:
            DataFrame containing the input data and indicator values
        """
        # If the result is None or empty, return the input data
        if result is None or result.empty:
            self.logger.warning("Indicator calculation returned no results")
            return data

        # If the result has a different index, reindex it to match the input data
        if not result.index.equals(data.index):
            self.logger.warning("Indicator result has a different index than input data")
            result = result.reindex(data.index)

        # Merge the input data and indicator values
        output = data.copy()
        for col in result.columns:
            output[col] = result[col]

        return output

    def get_column_names(self) -> List[str]:
        """
        Get the names of columns added by this indicator.

        Returns:
            List of column names
        """
        return self.output_columns

    def get_params(self) -> Dict[str, Any]:
        """
        Get the indicator parameters.

        Returns:
            Dictionary containing the indicator parameters
        """
        return self.params.copy()

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set the indicator parameters.

        Args:
            params: New parameters for the indicator

        Raises:
            ValidationError: If the parameters are invalid
        """
        self.params.update(params)
        self._validate_params()

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the indicator.

        Returns:
            Dictionary containing information about the indicator
        """
        return {
            "name": self.name,
            "category": self.category,
            "class": self.__class__.__name__,
            "params": self.get_params(),
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "last_calculation_time": self.last_calculation_time,
            "last_update_time": self.last_update_time
        }

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """
        Get metadata about the indicator class.

        Returns:
            Dictionary containing metadata about the indicator class
        """
        return {
            "name": cls.name,
            "category": cls.category,
            "default_params": cls.default_params,
            "required_params": {k: v.__name__ for k, v in cls.required_params.items()},
            "class": cls.__name__
        }

    def __str__(self) -> str:
        """
        Get a string representation of the indicator.

        Returns:
            String representation of the indicator
        """
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"

    def __repr__(self) -> str:
        """
        Get a string representation of the indicator.

        Returns:
            String representation of the indicator
        """
        return self.__str__()