\
# filepath: d:\\MD\\forex_trading_platform\\feature-store-service\\indicators\\base_indicator.py
\"\"\"
Base class for all technical indicators in the feature store service.
\"\"\"

import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseIndicator(ABC):
    \"\"\"
    Abstract base class for technical indicators.

    All indicators should inherit from this class and implement the 'calculate' method.
    \"\"\"

    # Class attributes to be potentially overridden by subclasses
    category: str = "Unknown"
    name: str = "BaseIndicator"
    required_params: Dict[str, type] = {} # Define required parameters and their types

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        \"\"\"
        Initialize the indicator with parameters.

        Args:
            params: Dictionary of parameters for the indicator calculation.
        \"\"\"
        self.params = params or {}
        self._validate_params()

    def _validate_params(self):
        \"\"\"
        Validate that all required parameters are provided and have the correct type.
        \"\"\"
        missing_params = [p for p in self.required_params if p not in self.params]
        if missing_params:
            raise ValueError(f"Missing required parameters for {self.name}: {', '.join(missing_params)}")

        for param_name, expected_type in self.required_params.items():
            if param_name in self.params:
                actual_value = self.params[param_name]
                if not isinstance(actual_value, expected_type):
                    # Allow int for float type for flexibility
                    if expected_type is float and isinstance(actual_value, int):
                        self.params[param_name] = float(actual_value) # Auto-convert
                    else:
                        raise TypeError(f"Parameter '{param_name}' for {self.name} expected type {expected_type.__name__}, got {type(actual_value).__name__}")

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Calculate the indicator values.

        Args:
            data: Input DataFrame, typically containing OHLCV data.

        Returns:
            DataFrame with the calculated indicator values added as new columns.
            The column names should be descriptive (e.g., 'SMA_10', 'RSI_14').
        \"\"\"
        pass

    def get_params(self) -> Dict[str, Any]:
        \"\"\"
        Get the parameters used by this indicator instance.
        \"\"\"
        return self.params.copy()

    def __repr__(self) -> str:
        param_str = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f"{self.__class__.__name__}({param_str})"

