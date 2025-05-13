"""
Utility functions for data validation.
"""
import pandas as pd
import numpy as np
from typing import Any, Optional, Union
import logging
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def is_data_valid(data: Any, context: str='data', min_length: int=1) ->bool:
    """Checks if the input data is valid for processing (not None, not empty DataFrame/Series).

    Args:
        data: The data to validate (DataFrame, Series, or other).
        context: A string describing the data source (e.g., 'feature input', 'indicator output') for logging.
        min_length: The minimum required length if data is a DataFrame or Series.

    Returns:
        True if the data is valid, False otherwise.
    """
    if data is None:
        logger.warning(f'Validation failed: {context} is None.')
        return False
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if data.empty:
            logger.warning(f'Validation failed: {context} is empty.')
            return False
        if len(data) < min_length:
            logger.warning(
                f'Validation failed: {context} has length {len(data)}, less than minimum {min_length}.'
                )
            return False
    return True


@with_exception_handling
def ensure_dataframe(X: Union[pd.DataFrame, np.ndarray, list], copy: bool=True
    ) ->Optional[pd.DataFrame]:
    """Ensures the input is a pandas DataFrame, converting if necessary.

    Args:
        X: The input data (DataFrame, NumPy array, list, etc.).
        copy: Whether to copy the DataFrame if it's already a DataFrame.'

    Returns:
        A pandas DataFrame, or None if conversion fails.
    """
    if isinstance(X, pd.DataFrame):
        return X.copy() if copy else X
    elif isinstance(X, np.ndarray):
        try:
            return pd.DataFrame(X)
        except Exception as e:
            logger.error(f'Failed to convert NumPy array to DataFrame: {e}')
            return None
    elif isinstance(X, list):
        try:
            return pd.DataFrame(X)
        except Exception as e:
            logger.error(f'Failed to convert list to DataFrame: {e}')
            return None
    else:
        logger.warning(
            f'Input type {type(X)} not directly convertible to DataFrame.')
        return None
