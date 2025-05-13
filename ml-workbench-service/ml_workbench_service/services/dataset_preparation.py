"""
Dataset Preparation Module

This module provides functionality to prepare datasets for machine learning models,
including loading data from the feature store, transforming it, and creating
train/validation/test splits.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DatasetPreparation:
    """
    A class for preparing datasets for machine learning models.
    
    This class provides methods to load data from the feature store,
    preprocess it, create features, and split it into training and validation sets.
    
    Attributes:
        feature_client: Client for accessing the feature store
        logger: Logger instance
    """

    def __init__(self, feature_client=None):
        """
        Initialize the DatasetPreparation object.
        
        Args:
            feature_client: Client for accessing the feature store.
                           If None, a default client will be created.
        """
        from ..clients.feature_store_client import FeatureStoreClient
        self.feature_client = feature_client or FeatureStoreClient()
        self.logger = logging.getLogger(__name__)

    @with_exception_handling
    def load_ohlcv_data(self, symbol: str, start_date: Union[str, datetime],
        end_date: Union[str, datetime], timeframe: str='1h',
        include_indicators: Optional[List[str]]=None) ->pd.DataFrame:
        """
        Load OHLCV data and optional technical indicators from the feature store.
        
        Args:
            symbol: Trading symbol (e.g., 'EUR_USD')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for the data (e.g., '1m', '5m', '1h', '1d')
            include_indicators: Optional list of technical indicators to include
        
        Returns:
            pd.DataFrame: DataFrame containing OHLCV data and indicators
        """
        try:
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            self.logger.info(
                f'Loading OHLCV data for {symbol} from {start_date} to {end_date} ({timeframe})'
                )
            ohlcv_data = self.feature_client.get_ohlcv_data(symbol=symbol,
                start_date=start_date, end_date=end_date, timeframe=timeframe)
            if include_indicators and len(include_indicators) > 0:
                self.logger.info(f'Loading indicators: {include_indicators}')
                indicators_data = self.feature_client.get_indicators(symbol
                    =symbol, start_date=start_date, end_date=end_date,
                    timeframe=timeframe, indicators=include_indicators)
                if indicators_data is not None and not indicators_data.empty:
                    data = pd.merge(ohlcv_data, indicators_data, on=
                        'timestamp', how='left')
                else:
                    data = ohlcv_data
                    self.logger.warning(
                        f'No indicator data returned for {symbol}, returning OHLCV only'
                        )
            else:
                data = ohlcv_data
            data = data.sort_values('timestamp')
            return data
        except Exception as e:
            self.logger.error(f'Error loading data: {str(e)}')
            raise

    def create_features(self, data: pd.DataFrame, feature_config: Dict[str,
        Any]) ->pd.DataFrame:
        """
        Create features from raw OHLCV and indicator data.
        
        Args:
            data: DataFrame containing OHLCV data and potentially indicators
            feature_config: Configuration for feature creation
                            Keys can include:
                            - 'price_returns': Dict with lookback periods for returns
                            - 'moving_stats': Dict with windows and stats to calculate
                            - 'indicators': Dict with transformations to apply
        
        Returns:
            pd.DataFrame: DataFrame with created features
        """
        df = data.copy()
        if 'price_returns' in feature_config:
            periods = feature_config['price_returns'].get('periods', [1, 5, 10]
                )
            for period in periods:
                df[f'price_change_{period}'] = df['close'].diff(period)
                df[f'pct_change_{period}'] = df['close'].pct_change(period
                    ) * 100
                df[f'log_return_{period}'] = np.log(df['close'] / df[
                    'close'].shift(period))
        if 'moving_stats' in feature_config:
            windows = feature_config['moving_stats'].get('windows', [5, 10, 20]
                )
            for window in windows:
                df[f'close_ma_{window}'] = df['close'].rolling(window=window
                    ).mean()
                df[f'close_std_{window}'] = df['close'].rolling(window=window
                    ).std()
                df[f'close_min_{window}'] = df['close'].rolling(window=window
                    ).min()
                df[f'close_max_{window}'] = df['close'].rolling(window=window
                    ).max()
                if feature_config['moving_stats'].get('include_advanced', False
                    ):
                    df[f'close_median_{window}'] = df['close'].rolling(window
                        =window).median()
                    df[f'close_skew_{window}'] = df['close'].rolling(window
                        =window).skew()
                    df[f'close_q25_{window}'] = df['close'].rolling(window=
                        window).quantile(0.25)
                    df[f'close_q75_{window}'] = df['close'].rolling(window=
                        window).quantile(0.75)
        if ('indicators' in feature_config and 'transformations' in
            feature_config['indicators']):
            transformations = feature_config['indicators']['transformations']
            for transform_config in transformations:
                indicator = transform_config['indicator']
                operation = transform_config['operation']
                if indicator in df.columns:
                    if operation == 'diff':
                        period = transform_config_manager.get('period', 1)
                        df[f'{indicator}_diff_{period}'] = df[indicator].diff(
                            period)
                    elif operation == 'pct_change':
                        period = transform_config_manager.get('period', 1)
                        df[f'{indicator}_pct_{period}'] = df[indicator
                            ].pct_change(period) * 100
                    elif operation == 'cross_above':
                        other = transform_config_manager.get('other_indicator')
                        if other in df.columns:
                            df[f'{indicator}_above_{other}'] = ((df[
                                indicator] > df[other]) & (df[indicator].
                                shift(1) <= df[other].shift(1))).astype(int)
                    elif operation == 'cross_below':
                        other = transform_config_manager.get('other_indicator')
                        if other in df.columns:
                            df[f'{indicator}_below_{other}'] = ((df[
                                indicator] < df[other]) & (df[indicator].
                                shift(1) >= df[other].shift(1))).astype(int)
        if 'target' in feature_config:
            target_config = feature_config['target']
            horizon = target_config_manager.get('horizon', 1)
            if target_config_manager.get('type') == 'direction':
                df[f'target_direction_{horizon}'] = (df['close'].shift(-
                    horizon) > df['close']).astype(int)
            elif target_config_manager.get('type') == 'change':
                df[f'target_change_{horizon}'] = df['close'].shift(-horizon
                    ) - df['close']
            elif target_config_manager.get('type') == 'pct_change':
                df[f'target_pct_change_{horizon}'] = df['close'].pct_change(
                    -horizon) * 100
            elif target_config_manager.get('type') == 'log_return':
                df[f'target_log_return_{horizon}'] = np.log(df['close'].
                    shift(-horizon) / df['close'])
            elif target_config_manager.get('type') == 'movement_class':
                thresholds = target_config_manager.get('thresholds', [0.001, 0.005])
                future_pct_change = df['close'].pct_change(-horizon) * 100
                df[f'target_movement_{horizon}'] = 0
                df.loc[future_pct_change > thresholds[1],
                    f'target_movement_{horizon}'] = 2
                df.loc[(future_pct_change > thresholds[0]) & (
                    future_pct_change <= thresholds[1]),
                    f'target_movement_{horizon}'] = 1
                df.loc[(future_pct_change < -thresholds[0]) & (
                    future_pct_change >= -thresholds[1]),
                    f'target_movement_{horizon}'] = -1
                df.loc[future_pct_change < -thresholds[1],
                    f'target_movement_{horizon}'] = -2
        df = df.dropna()
        return df

    def preprocess_data(self, df: pd.DataFrame, preprocessing_config: Dict[
        str, Any], fit_transform: bool=True) ->Tuple[pd.DataFrame, Dict[str,
        Any]]:
        """
        Preprocess data for ML models.
        
        Args:
            df: DataFrame containing features and target(s)
            preprocessing_config: Configuration for preprocessing
                                Keys can include:
                                - 'scaling': Dict with scaling method and columns
                                - 'categorical_encoding': Dict with encoding methods
                                - 'feature_selection': Dict with selection methods
            fit_transform: If True, fit and transform data. If False, use
                          existing transformation parameters.
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: 
                Preprocessed data and preprocessing parameters
        """
        processed_df = df.copy()
        preprocessing_params = {}
        if 'scaling' in preprocessing_config:
            scaling_config = preprocessing_config['scaling']
            method = scaling_config_manager.get('method', 'standard')
            columns_to_scale = scaling_config_manager.get('columns', [])
            if not columns_to_scale:
                columns_to_scale = processed_df.select_dtypes(include=[np.
                    number]).columns.tolist()
                columns_to_scale = [col for col in columns_to_scale if not
                    col.startswith('target_')]
            if fit_transform:
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                scaled_data = scaler.fit_transform(processed_df[
                    columns_to_scale])
                preprocessing_params['scaler'] = scaler
                preprocessing_params['scaled_columns'] = columns_to_scale
            else:
                scaler = preprocessing_config_manager.get('scaler')
                columns_to_scale = preprocessing_config.get('scaled_columns',
                    columns_to_scale)
                scaled_data = scaler.transform(processed_df[columns_to_scale])
            processed_df[columns_to_scale] = scaled_data
        if 'categorical_encoding' in preprocessing_config and fit_transform:
            pass
        if 'feature_selection' in preprocessing_config and fit_transform:
            pass
        return processed_df, preprocessing_params

    def create_train_test_split(self, df: pd.DataFrame, target_column: str,
        test_size: float=0.2, validation_size: float=0.2, split_method: str
        ='time', random_state: int=42) ->Tuple[pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Create train, validation, and test splits from a dataset.
        
        Args:
            df: DataFrame containing features and target
            target_column: Column name of the target variable
            test_size: Proportion of data to use for testing
            validation_size: Proportion of non-test data to use for validation
            split_method: Method for splitting ('time' or 'random')
            random_state: Random state for reproducibility
        
        Returns:
            Tuple of train, validation, test DataFrames and their targets
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        if split_method == 'time':
            total_rows = len(df)
            test_rows = int(total_rows * test_size)
            X_train_val = X.iloc[:-test_rows]
            y_train_val = y.iloc[:-test_rows]
            X_test = X.iloc[-test_rows:]
            y_test = y.iloc[-test_rows:]
            val_rows = int(len(X_train_val) * validation_size)
            X_train = X_train_val.iloc[:-val_rows]
            y_train = y_train_val.iloc[:-val_rows]
            X_val = X_train_val.iloc[-val_rows:]
            y_val = y_train_val.iloc[-val_rows:]
        else:
            X_train_val, X_test, y_train_val, y_test = train_test_split(X,
                y, test_size=test_size, random_state=random_state)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                y_train_val, test_size=validation_size, random_state=
                random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_dataset_for_model(self, symbol: str, start_date: Union[str,
        datetime], end_date: Union[str, datetime], timeframe: str='1h',
        config: Dict[str, Any]=None) ->Dict[str, Any]:
        """
        Full pipeline to prepare a dataset for model training and evaluation.
        
        Args:
            symbol: Trading symbol (e.g., 'EUR_USD')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for the data
            config: Configuration for dataset preparation
        
        Returns:
            Dict containing train/val/test datasets and preprocessing parameters
        """
        default_config = {'indicators': ['sma_5', 'sma_10', 'sma_20',
            'ema_5', 'ema_10', 'ema_20', 'rsi_14', 'macd', 'macd_signal',
            'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower', 'atr_14'],
            'feature_config': {'price_returns': {'periods': [1, 5, 10]},
            'moving_stats': {'windows': [5, 10, 20], 'include_advanced': 
            True}, 'indicators': {'transformations': [{'indicator':
            'rsi_14', 'operation': 'diff', 'period': 1}, {'indicator':
            'macd', 'operation': 'cross_above', 'other_indicator':
            'macd_signal'}]}, 'target': {'type': 'direction', 'horizon': 5}
            }, 'preprocessing': {'scaling': {'method': 'standard'}},
            'split': {'test_size': 0.2, 'validation_size': 0.15, 'method':
            'time'}}
        if config:
            for key, value in config.items():
                if key in default_config and isinstance(value, dict
                    ) and isinstance(default_config[key], dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        config = default_config
        raw_data = self.load_ohlcv_data(symbol=symbol, start_date=
            start_date, end_date=end_date, timeframe=timeframe,
            include_indicators=config['indicators'])
        feature_data = self.create_features(data=raw_data, feature_config=
            config['feature_config'])
        target_type = config['feature_config']['target']['type']
        target_horizon = config['feature_config']['target']['horizon']
        target_column = f'target_{target_type}_{target_horizon}'
        processed_data, preprocessing_params = self.preprocess_data(df=
            feature_data, preprocessing_config=config['preprocessing'],
            fit_transform=True)
        X_train, X_val, X_test, y_train, y_val, y_test = (self.
            create_train_test_split(df=processed_data, target_column=
            target_column, test_size=config['split']['test_size'],
            validation_size=config['split']['validation_size'],
            split_method=config['split']['method']))
        return {'X_train': X_train, 'y_train': y_train, 'X_val': X_val,
            'y_val': y_val, 'X_test': X_test, 'y_test': y_test,
            'preprocessing_params': preprocessing_params, 'feature_config':
            config['feature_config'], 'raw_data_sample': raw_data.head(),
            'feature_data_sample': feature_data.head(), 'target_column':
            target_column}
