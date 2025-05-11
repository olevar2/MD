"""
Pattern Recognizer Adapter Module

This module implements the adapter pattern for the pattern service,
using the interfaces defined in common-lib.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd

from common_lib.interfaces.analysis_engine import IPatternRecognizer
from common_lib.errors.base_exceptions import (
    BaseError, ErrorCode, ValidationError, DataError, ServiceError
)

from analysis_engine.services.pattern_service import PatternService
from analysis_engine.config.settings import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class PatternRecognizerAdapter(IPatternRecognizer):
    """
    Adapter for the PatternService to implement the IPatternRecognizer interface.
    
    This adapter allows the PatternService to be used through the standardized
    IPatternRecognizer interface, enabling better service integration and
    reducing circular dependencies.
    """
    
    def __init__(self, pattern_service: Optional[PatternService] = None):
        """
        Initialize the PatternRecognizerAdapter.
        
        Args:
            pattern_service: Optional PatternService instance. If not provided,
                            a new instance will be created.
        """
        self._pattern_service = pattern_service or PatternService()
        self._settings = get_settings()
        logger.info("PatternRecognizerAdapter initialized")
    
    async def detect_pattern(
        self,
        pattern_name: str,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect a specific pattern in market data.
        
        Args:
            pattern_name: The name of the pattern to detect
            data: The market data to detect patterns in
            params: Optional parameters for pattern detection
            
        Returns:
            List of dictionaries containing information about detected pattern instances
            
        Raises:
            ValidationError: If the input parameters are invalid
            DataError: If there's an issue with the data
            ServiceError: If there's a service-related error
        """
        try:
            # Validate inputs
            if not pattern_name:
                raise ValidationError("Pattern name cannot be empty", field="pattern_name")
            
            if data is None or data.empty:
                raise ValidationError("Data cannot be empty", field="data")
            
            # Call the service method
            result = await self._pattern_service.detect_pattern(
                pattern_name=pattern_name,
                data=data,
                parameters=params or {}
            )
            
            return result
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Convert other exceptions to appropriate error types
            if "not found" in str(e).lower() or "unknown pattern" in str(e).lower():
                raise DataError(
                    f"Unknown pattern: {pattern_name}",
                    error_code=ErrorCode.DATA_MISSING_ERROR,
                    data_source="pattern_service",
                    data_type="pattern",
                    cause=e
                )
            elif "invalid data" in str(e).lower():
                raise DataError(
                    f"Invalid data for pattern {pattern_name}",
                    error_code=ErrorCode.DATA_VALIDATION_ERROR,
                    data_source="pattern_service",
                    data_type="market_data",
                    cause=e
                )
            else:
                raise ServiceError(
                    f"Error detecting pattern {pattern_name}: {str(e)}",
                    error_code=ErrorCode.SERVICE_UNAVAILABLE,
                    service_name="pattern_service",
                    operation="detect_pattern",
                    cause=e
                )
    
    async def detect_multiple_patterns(
        self,
        patterns: List[Dict[str, Any]],
        data: pd.DataFrame
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect multiple patterns in market data.
        
        Args:
            patterns: List of dictionaries containing pattern names and parameters
            data: The market data to detect patterns in
            
        Returns:
            Dictionary mapping pattern names to lists of detected pattern instances
            
        Raises:
            ValidationError: If the input parameters are invalid
            DataError: If there's an issue with the data
            ServiceError: If there's a service-related error
        """
        try:
            # Validate inputs
            if not patterns:
                raise ValidationError("Patterns list cannot be empty", field="patterns")
            
            if data is None or data.empty:
                raise ValidationError("Data cannot be empty", field="data")
            
            # Extract pattern names for the service method
            pattern_types = []
            pattern_params = {}
            
            for pattern_info in patterns:
                pattern_name = pattern_info.get("name")
                if not pattern_name:
                    raise ValidationError("Pattern name cannot be empty", field="name")
                
                pattern_types.append(pattern_name)
                pattern_params[pattern_name] = pattern_info.get("parameters", {})
            
            # Call the service method
            result = await self._pattern_service.recognize_patterns(
                data=data,
                pattern_types=pattern_types,
                parameters={"pattern_params": pattern_params}
            )
            
            return result
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Convert other exceptions to appropriate error types
            if "not found" in str(e).lower() or "unknown pattern" in str(e).lower():
                raise DataError(
                    f"Unknown pattern in the list",
                    error_code=ErrorCode.DATA_MISSING_ERROR,
                    data_source="pattern_service",
                    data_type="pattern",
                    cause=e
                )
            elif "invalid data" in str(e).lower():
                raise DataError(
                    f"Invalid data for patterns",
                    error_code=ErrorCode.DATA_VALIDATION_ERROR,
                    data_source="pattern_service",
                    data_type="market_data",
                    cause=e
                )
            else:
                raise ServiceError(
                    f"Error detecting multiple patterns: {str(e)}",
                    error_code=ErrorCode.SERVICE_UNAVAILABLE,
                    service_name="pattern_service",
                    operation="detect_multiple_patterns",
                    cause=e
                )
    
    async def get_pattern_info(self, pattern_name: str) -> Dict[str, Any]:
        """
        Get information about a specific pattern.
        
        Args:
            pattern_name: The name of the pattern
            
        Returns:
            Dictionary containing information about the pattern
            
        Raises:
            ValidationError: If the input parameters are invalid
            DataError: If there's an issue with the data
            ServiceError: If there's a service-related error
        """
        try:
            # Validate inputs
            if not pattern_name:
                raise ValidationError("Pattern name cannot be empty", field="pattern_name")
            
            # Call the service method
            result = await self._pattern_service.get_pattern_info(pattern_name=pattern_name)
            
            return result
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Convert other exceptions to appropriate error types
            if "not found" in str(e).lower() or "unknown pattern" in str(e).lower():
                raise DataError(
                    f"Unknown pattern: {pattern_name}",
                    error_code=ErrorCode.DATA_MISSING_ERROR,
                    data_source="pattern_service",
                    data_type="pattern",
                    cause=e
                )
            else:
                raise ServiceError(
                    f"Error getting pattern info for {pattern_name}: {str(e)}",
                    error_code=ErrorCode.SERVICE_UNAVAILABLE,
                    service_name="pattern_service",
                    operation="get_pattern_info",
                    cause=e
                )
    
    async def get_all_patterns_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all available patterns.
        
        Returns:
            List of dictionaries containing information about all available patterns
            
        Raises:
            ServiceError: If there's a service-related error
        """
        try:
            # Call the service method
            pattern_types = await self._pattern_service.get_pattern_types()
            
            # Convert to the expected format
            result = []
            for pattern_type in pattern_types:
                # Extract the pattern name from the pattern type info
                pattern_name = pattern_type.get("name")
                if pattern_name:
                    # Get detailed info for the pattern
                    pattern_info = await self._pattern_service.get_pattern_info(pattern_name=pattern_name)
                    result.append(pattern_info)
            
            return result
        except Exception as e:
            # Convert exceptions to appropriate error types
            raise ServiceError(
                f"Error getting all patterns info: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                service_name="pattern_service",
                operation="get_all_patterns_info",
                cause=e
            )
