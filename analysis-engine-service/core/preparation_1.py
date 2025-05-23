"""
Causal Data Preparation Module

Provides tools for preparing financial time series data specifically for causal inference.
Includes preprocessing, feature engineering, and time-aware data splitting.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class BaseDataPreparation:
    """Base class for data preparation steps."""

    def __init__(self, parameters: Optional[Dict[str, Any]]=None):
    """
      init  .
    
    Args:
        parameters: Description of parameters
        Any]]: Description of Any]]
    
    """

        self.parameters = parameters or {}
        logger.info(
            f'Initializing {self.__class__.__name__} with parameters: {self.parameters}'
            )

    def transform(self, data: pd.DataFrame) ->pd.DataFrame:
        """Apply the transformation to the data."""
        raise NotImplementedError


class FinancialDataPreprocessor(BaseDataPreparation):
    """
    Handles preprocessing steps like stationarity, missing values, and scaling.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]]=None):
    """
      init  .
    
    Args:
        parameters: Description of parameters
        Any]]: Description of Any]]
    
    """

        defaults = {'stationarity_method': 'pct_change',
            'missing_value_method': 'ffill', 'scaling_method': 'standard'}
        defaults.update(parameters or {})
        super().__init__(defaults)
        self.scaler = None

    def transform(self, data: pd.DataFrame, target_columns: Optional[List[
        str]]=None) ->pd.DataFrame:
        """
        Apply preprocessing steps to the specified columns or all numeric columns.

        Args:
            data: Input DataFrame.
            target_columns: List of columns to preprocess. If None, applies to all numeric columns.

        Returns:
            Preprocessed DataFrame.
        """
        df = data.copy()
        if target_columns is None:
            target_columns = df.select_dtypes(include=np.number
                ).columns.tolist()
        logger.info(
            f"Applying preprocessing: Stationarity='{self.parameters['stationarity_method']}', Missing='{self.parameters['missing_value_method']}', Scaling='{self.parameters['scaling_method']}'"
            )
        for col in target_columns:
            if col not in df.columns:
                logger.warning(
                    f"Column '{col}' not found in data. Skipping preprocessing."
                    )
                continue
            if self.parameters['missing_value_method'] == 'ffill':
                df[col] = df[col].ffill()
            elif self.parameters['missing_value_method'] == 'bfill':
                df[col] = df[col].bfill()
            elif self.parameters['missing_value_method'] == 'interpolate':
                df[col] = df[col].interpolate(method='time')
            elif self.parameters['missing_value_method'] == 'drop':
                df = df.dropna(subset=[col])
            df = df.dropna(subset=[col])
            if df.empty:
                logger.warning(
                    f"DataFrame became empty after handling missing values for column '{col}'. Returning empty DataFrame."
                    )
                return df
            if self.parameters['stationarity_method'] == 'pct_change':
                df[col] = df[col].pct_change()
            elif self.parameters['stationarity_method'] == 'diff':
                df[col] = df[col].diff()
            df = df.dropna(subset=[col])
            if df.empty:
                logger.warning(
                    f"DataFrame became empty after applying stationarity method for column '{col}'. Returning empty DataFrame."
                    )
                return df
        numeric_cols_to_scale = [col for col in target_columns if col in df
            .columns and pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols_to_scale:
            logger.warning('No numeric columns found or left to scale.')
            return df
        if self.parameters['scaling_method'] == 'standard':
            self.scaler = StandardScaler()
            df[numeric_cols_to_scale] = self.scaler.fit_transform(df[
                numeric_cols_to_scale])
        elif self.parameters['scaling_method'] == 'minmax':
            self.scaler = MinMaxScaler()
            df[numeric_cols_to_scale] = self.scaler.fit_transform(df[
                numeric_cols_to_scale])
        elif self.parameters['scaling_method'] is not None:
            logger.warning(
                f"Unsupported scaling method: {self.parameters['scaling_method']}. No scaling applied."
                )
        logger.info('Preprocessing complete.')
        return df


class FinancialFeatureEngineering(BaseDataPreparation):
    """
    Adds financial-specific features like lagged variables and technical indicators.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]]=None):
    """
      init  .
    
    Args:
        parameters: Description of parameters
        Any]]: Description of Any]]
    
    """

        defaults = {'lags': [1, 2, 3, 5], 'include_volatility': True,
            'volatility_window': 14, 'include_indicators': []}
        defaults.update(parameters or {})
        super().__init__(defaults)

    @with_exception_handling
    def transform(self, data: pd.DataFrame, price_column: str='close',
        high_column: str='high', low_column: str='low', volume_column:
        Optional[str]='volume') ->pd.DataFrame:
        """
        Adds features to the DataFrame.

        Args:
            data: Input DataFrame with OHLCV data.
            price_column: Name of the column to use for lags and some indicators.
            high_column: Name of the high price column.
            low_column: Name of the low price column.
            volume_column: Name of the volume column (optional).

        Returns:
            DataFrame with added features.
        """
        df = data.copy()
        logger.info(
            f"Applying feature engineering: Lags={self.parameters['lags']}, Volatility={self.parameters['include_volatility']}, Indicators={self.parameters['include_indicators']}"
            )
        if price_column not in df.columns:
            logger.error(
                f"Price column '{price_column}' not found. Cannot create lags."
                )
            return df
        for lag in self.parameters['lags']:
            if lag > 0:
                df[f'{price_column}_lag_{lag}'] = df[price_column].shift(lag)
        if self.parameters['include_volatility']:
            if (high_column in df.columns and low_column in df.columns and 
                price_column in df.columns):
                try:
                    print(
                        f"Placeholder: ATR calculation logic executed with window {self.parameters['volatility_window']}."
                        )
                    tr1 = abs(df[high_column] - df[low_column])
                    tr2 = abs(df[high_column] - df[price_column].shift(1))
                    tr3 = abs(df[low_column] - df[price_column].shift(1))
                    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}
                        ).max(axis=1)
                    df['atr'] = tr.rolling(window=self.parameters[
                        'volatility_window']).mean()
                except Exception as e:
                    logger.error(f'Error calculating ATR: {e}')
            else:
                logger.warning(
                    'Required columns (high, low, close) not found for ATR calculation.'
                    )
        if self.parameters['include_indicators']:
            print(
                f"Placeholder: Technical indicator calculation logic executed for {self.parameters['include_indicators']}."
                )
        df = df.dropna()
        logger.info('Feature engineering complete.')
        return df


class FinancialDataSplitter:
    """
    Provides methods for splitting financial time series data while respecting temporal order.
    Useful for creating train/validation/test sets for causal model evaluation.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]]=None):
    """
      init  .
    
    Args:
        parameters: Description of parameters
        Any]]: Description of Any]]
    
    """

        defaults = {'n_splits': 5, 'test_size_ratio': 0.2, 'gap': 0}
        defaults.update(parameters or {})
        self.parameters = defaults
        logger.info(
            f'Initializing FinancialDataSplitter with parameters: {self.parameters}'
            )

    def time_series_split(self, data: pd.DataFrame) ->Tuple[pd.DataFrame,
        pd.DataFrame]:
        """
        Performs a single chronological train-test split.

        Args:
            data: Input DataFrame, assumed to be sorted by time.

        Returns:
            Tuple containing train_df and test_df.
        """
        split_index = int(len(data) * (1 - self.parameters['test_size_ratio']))
        train_df = data.iloc[:split_index]
        test_df = data.iloc[split_index:]
        logger.info(
            f'Performed single train-test split: Train size={len(train_df)}, Test size={len(test_df)}'
            )
        return train_df, test_df

    def time_series_cross_validation_split(self, data: pd.DataFrame) ->List[
        Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices for time series cross-validation using sklearn's TimeSeriesSplit.

        Args:
            data: Input DataFrame or array-like structure.

        Returns:
            List of tuples, where each tuple contains train indices and test indices for a split.
        """
        tscv = TimeSeriesSplit(n_splits=self.parameters['n_splits'], gap=
            self.parameters['gap'])
        splits = list(tscv.split(data))
        logger.info(
            f'Generated {len(splits)} time series cross-validation splits.')
        for i, (train_idx, test_idx) in enumerate(splits):
            logger.debug(
                f'Split {i + 1}: Train size={len(train_idx)}, Test size={len(test_idx)}, Train range=({train_idx.min()}-{train_idx.max()}), Test range=({test_idx.min()}-{test_idx.max()})'
                )
        return splits


if __name__ == '__main__':
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='B')
    data = pd.DataFrame(index=dates)
    data['open'] = 100 + np.random.randn(200).cumsum() * 0.5
    data['high'] = data['open'] + np.random.rand(200) * 2
    data['low'] = data['open'] - np.random.rand(200) * 2
    data['close'] = data['open'] + np.random.randn(200) * 1.5
    data['volume'] = np.random.randint(1000, 5000, 200)
    data.loc[data.sample(frac=0.05).index, 'close'] = np.nan
    print('--- Original Data --- ')
    print(data.head())
    print(f'NaNs before preprocessing:\n{data.isna().sum()}')
    print('\n--- Financial Data Preprocessor --- ')
    preprocessor = FinancialDataPreprocessor(parameters={
        'stationarity_method': 'pct_change', 'missing_value_method':
        'interpolate', 'scaling_method': 'standard'})
    preprocessed_data = preprocessor.transform(data, target_columns=['open',
        'high', 'low', 'close', 'volume'])
    print(preprocessed_data.head())
    print(f'NaNs after preprocessing:\n{preprocessed_data.isna().sum()}')
    print('\n--- Financial Feature Engineering --- ')
    feature_engineer = FinancialFeatureEngineering(parameters={'lags': [1, 
        3, 5], 'include_volatility': True, 'volatility_window': 10})
    featured_data = feature_engineer.transform(data.ffill().bfill())
    print(featured_data.head(10))
    print(f'NaNs after feature engineering:\n{featured_data.isna().sum()}')
    print('\n--- Financial Data Splitter --- ')
    splitter = FinancialDataSplitter(parameters={'n_splits': 3,
        'test_size_ratio': 0.25, 'gap': 1})
    train_df, test_df = splitter.time_series_split(featured_data)
    print(
        f'Single Split: Train shape={train_df.shape}, Test shape={test_df.shape}'
        )
    cv_splits = splitter.time_series_cross_validation_split(featured_data)
    print(f'Generated {len(cv_splits)} CV splits.')
    for i, (train_idx, test_idx) in enumerate(cv_splits):
        print(
            f'  Split {i + 1}: Train indices={len(train_idx)}, Test indices={len(test_idx)}'
            )
