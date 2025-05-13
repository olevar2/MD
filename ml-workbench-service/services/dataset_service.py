"""
Dataset Service Module.

Provides business logic for managing ML datasets.
"""
import uuid
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.experiment_models import Dataset, DatasetCreate
from repositories.experiment_repository import ExperimentRepository
from adapters.feature_store_client import FeatureStoreClient


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DatasetService:
    """
    Service for managing ML datasets.
    
    This service provides methods for creating, retrieving, and processing
    datasets for machine learning experiments.
    """

    def __init__(self, repository: ExperimentRepository, feature_store:
        FeatureStoreClient):
        """
        Initialize the dataset service.
        
        Args:
            repository: Experiment repository for data access
            feature_store: Feature store client for data retrieval
        """
        self.repository = repository
        self.feature_store = feature_store
        self.dataset_dir = os.environ.get('DATASET_DIR', './datasets')
        os.makedirs(self.dataset_dir, exist_ok=True)

    async def create_dataset(self, dataset_data: DatasetCreate) ->Dataset:
        """
        Create a new dataset.
        
        Args:
            dataset_data: Data for the new dataset
            
        Returns:
            Created dataset
        """
        dataset_dict = dataset_data.dict()
        dataset_dict['id'] = str(uuid.uuid4())
        dataset_dict['created_at'] = datetime.utcnow()
        dataset_id = await self.repository.create_dataset(dataset_dict)
        return Dataset(**dataset_dict)

    async def get_dataset(self, dataset_id: str) ->Optional[Dataset]:
        """
        Get a dataset by ID.
        
        Args:
            dataset_id: ID of the dataset to retrieve
            
        Returns:
            Dataset or None if not found
        """
        dataset_data = await self.repository.get_dataset(dataset_id)
        if dataset_data:
            return Dataset(**dataset_data)
        return None

    async def list_datasets(self, skip: int=0, limit: int=100, symbol:
        Optional[str]=None, timeframe: Optional[str]=None, sort_by: str=
        'created_at', sort_order: str='desc') ->List[Dataset]:
        """
        List datasets with optional filtering.
        
        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            sort_by: Field to sort by
            sort_order: Sort order (asc or desc)
            
        Returns:
            List of datasets
        """
        datasets_data = await self.repository.list_datasets(skip=skip,
            limit=limit, symbol=symbol, timeframe=timeframe, sort_by=
            sort_by, sort_order=sort_order)
        return [Dataset(**ds) for ds in datasets_data]

    async def generate_dataset_from_features(self, dataset_create:
        DatasetCreate) ->Tuple[Dataset, str]:
        """
        Generate a new dataset from feature store data.
        
        This method retrieves the specified features from the feature store,
        creates a dataset suitable for ML training, and saves it to a file.
        
        Args:
            dataset_create: Dataset creation parameters
            
        Returns:
            Tuple of (created dataset object, path to dataset file)
        """
        df = await self.feature_store.create_dataset_from_features(symbol=
            dataset_create.symbol, timeframe=dataset_create.timeframe,
            feature_ids=dataset_create.features, target_feature=
            dataset_create.target, start_date=dataset_create.start_date,
            end_date=dataset_create.end_date, lookback_periods=
            dataset_create.preprocessing.get('lookback_periods', 0) if
            dataset_create.preprocessing else 0, forecast_periods=
            dataset_create.preprocessing.get('forecast_periods', 1) if
            dataset_create.preprocessing else 1, include_ohlcv=
            dataset_create.preprocessing.get('include_ohlcv', True) if
            dataset_create.preprocessing else True)
        if df.empty:
            raise ValueError('No data retrieved from feature store')
        dataset = await self.create_dataset(dataset_create)
        dataset_path = os.path.join(self.dataset_dir, f'{dataset.id}.parquet')
        df.to_parquet(dataset_path)
        return dataset, dataset_path

    async def load_dataset(self, dataset_id: str, as_train_test_split: bool
        =True) ->Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.
        DataFrame, pd.DataFrame]]:
        """
        Load a dataset from file.
        
        Args:
            dataset_id: ID of the dataset to load
            as_train_test_split: Whether to return split data for training
            
        Returns:
            Either the full DataFrame or a tuple of (X_train, X_test, y_train, y_test)
        """
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f'Dataset {dataset_id} not found')
        dataset_path = os.path.join(self.dataset_dir, f'{dataset_id}.parquet')
        if not os.path.exists(dataset_path):
            _, dataset_path = await self.generate_dataset_from_features(
                DatasetCreate(name=dataset.name, symbol=dataset.symbol,
                timeframe=dataset.timeframe, features=dataset.features,
                target=dataset.target, start_date=dataset.start_date,
                end_date=dataset.end_date, split_ratio=dataset.split_ratio,
                description=dataset.description, preprocessing=dataset.
                preprocessing))
        df = pd.read_parquet(dataset_path)
        if not as_train_test_split:
            return df
        target_col = None
        for col in df.columns:
            if col.endswith('_target') or col == dataset.target:
                target_col = col
                break
        if not target_col:
            raise ValueError(f'Target column not found in dataset')
        X = df.drop(columns=[target_col])
        y = df[target_col]
        train_ratio = dataset.split_ratio.get('train', 0.7)
        val_ratio = dataset.split_ratio.get('validation', 0.15)
        test_ratio = dataset.split_ratio.get('test', 0.15)
        total = train_ratio + val_ratio + test_ratio
        if total != 1.0:
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        test_val_ratio = (val_ratio + test_ratio) / (train_ratio +
            val_ratio + test_ratio)
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size
            =test_val_ratio, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
            test_size=val_test_ratio, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    async def get_feature_statistics(self, dataset_id: str) ->Dict[str,
        Dict[str, Any]]:
        """
        Calculate statistics for features in a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dictionary of feature statistics
        """
        df = await self.load_dataset(dataset_id, as_train_test_split=False)
        stats = {}
        for column in df.columns:
            col_stats = {'mean': float(df[column].mean()) if pd.api.types.
                is_numeric_dtype(df[column]) else None, 'std': float(df[
                column].std()) if pd.api.types.is_numeric_dtype(df[column])
                 else None, 'min': float(df[column].min()) if pd.api.types.
                is_numeric_dtype(df[column]) else None, 'max': float(df[
                column].max()) if pd.api.types.is_numeric_dtype(df[column])
                 else None, 'median': float(df[column].median()) if pd.api.
                types.is_numeric_dtype(df[column]) else None, 'missing':
                int(df[column].isna().sum()), 'count': int(df[column].count
                ()), 'type': str(df[column].dtype)}
            stats[column] = col_stats
        return stats

    @async_with_exception_handling
    async def get_available_features(self, symbol: Optional[str]=None,
        timeframe: Optional[str]=None) ->List[Dict[str, Any]]:
        """
        Get a list of available features from the feature store.
        
        Args:
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            
        Returns:
            List of available features with metadata
        """
        feature_ids = await self.feature_store.get_available_features(symbol
            =symbol, timeframe=timeframe)
        features = []
        for feature_id in feature_ids:
            try:
                metadata = await self.feature_store.get_feature_metadata(
                    feature_id)
                features.append(metadata)
            except Exception:
                pass
        return features
