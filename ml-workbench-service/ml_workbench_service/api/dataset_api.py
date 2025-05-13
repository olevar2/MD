"""
Dataset API Module.

Provides API endpoints for managing datasets for machine learning.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body, Path, BackgroundTasks
from fastapi.responses import FileResponse
from ml_workbench_service.models.experiment_models import Dataset, DatasetCreate
from ml_workbench_service.services.dataset_service import DatasetService
from ml_workbench_service.repositories.experiment_repository import ExperimentRepository
from ml_workbench_service.clients.feature_store_client import FeatureStoreClient
router = APIRouter()


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

async def get_dataset_service() ->DatasetService:
    """
    Dependency for getting the dataset service.

    Returns:
        Dataset service
    """
    mongo_url = 'mongodb://localhost:27017'
    feature_store_url = 'http://localhost:8001'
    repository = ExperimentRepository(mongo_url)
    feature_store = FeatureStoreClient(feature_store_url)
    return DatasetService(repository, feature_store)


@router.post('/', response_model=Dataset, summary='Create a new dataset',
    description='Create a new dataset for machine learning.')
@async_with_exception_handling
async def create_dataset(dataset_data: DatasetCreate, service:
    DatasetService=Depends(get_dataset_service)):
    """
    Create a new dataset.

    Args:
        dataset_data: Data for the new dataset
        service: Dataset service

    Returns:
        Created dataset
    """
    try:
        dataset = await service.create_dataset(dataset_data)
        return dataset
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/', response_model=List[Dataset], summary='List datasets',
    description='List datasets with optional filtering.')
@async_with_exception_handling
async def list_datasets(skip: int=Query(0, description=
    'Number of datasets to skip'), limit: int=Query(100, description=
    'Maximum number of datasets to return'), symbol: Optional[str]=Query(
    None, description='Filter by symbol'), timeframe: Optional[str]=Query(
    None, description='Filter by timeframe'), sort_by: str=Query(
    'created_at', description='Field to sort by'), sort_order: str=Query(
    'desc', description='Sort order (asc or desc)'), service:
    DatasetService=Depends(get_dataset_service)):
    """
    List datasets with optional filtering.

    Args:
        skip: Number of datasets to skip
        limit: Maximum number of datasets to return
        symbol: Optional filter by symbol
        timeframe: Optional filter by timeframe
        sort_by: Field to sort by
        sort_order: Sort order (asc or desc)
        service: Dataset service

    Returns:
        List of datasets
    """
    try:
        datasets = await service.list_datasets(skip=skip, limit=limit,
            symbol=symbol, timeframe=timeframe, sort_by=sort_by, sort_order
            =sort_order)
        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{dataset_id}', response_model=Dataset, summary='Get dataset',
    description='Get a specific dataset by ID.')
@async_with_exception_handling
async def get_dataset(dataset_id: str=Path(..., description=
    'ID of the dataset to retrieve'), service: DatasetService=Depends(
    get_dataset_service)):
    """
    Get a specific dataset by ID.

    Args:
        dataset_id: ID of the dataset to retrieve
        service: Dataset service

    Returns:
        Dataset
    """
    try:
        dataset = await service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=
                f'Dataset {dataset_id} not found')
        return dataset
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/generate', response_model=Dataset, summary=
    'Generate dataset from features', description=
    'Generate a new dataset from feature store data.')
@async_with_exception_handling
async def generate_dataset(dataset_data: DatasetCreate, service:
    DatasetService=Depends(get_dataset_service)):
    """
    Generate a new dataset from feature store data.

    Args:
        dataset_data: Dataset creation parameters
        service: Dataset service

    Returns:
        Created dataset
    """
    try:
        dataset, dataset_path = await service.generate_dataset_from_features(
            dataset_data)
        return dataset
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{dataset_id}/statistics', summary='Get dataset statistics',
    description='Calculate statistics for features in a dataset.')
@async_with_exception_handling
async def get_dataset_statistics(dataset_id: str=Path(..., description=
    'ID of the dataset'), service: DatasetService=Depends(get_dataset_service)
    ):
    """
    Get statistics for a dataset.

    Args:
        dataset_id: ID of the dataset
        service: Dataset service

    Returns:
        Dictionary of feature statistics
    """
    try:
        stats = await service.get_feature_statistics(dataset_id)
        return stats
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/available-features', summary='Get available features',
    description='Get a list of available features from the feature store.')
@async_with_exception_handling
async def get_available_features(symbol: Optional[str]=Query(None,
    description='Filter by symbol'), timeframe: Optional[str]=Query(None,
    description='Filter by timeframe'), service: DatasetService=Depends(
    get_dataset_service)):
    """
    Get available features from the feature store.

    Args:
        symbol: Optional symbol filter
        timeframe: Optional timeframe filter
        service: Dataset service

    Returns:
        List of available features with metadata
    """
    try:
        features = await service.get_available_features(symbol, timeframe)
        return features
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{dataset_id}/download', summary='Download dataset',
    description='Download a dataset file.')
@async_with_exception_handling
async def download_dataset(dataset_id: str=Path(..., description=
    'ID of the dataset to download'), format: str=Query('parquet',
    description='File format (parquet or csv)'), service: DatasetService=
    Depends(get_dataset_service)):
    """
    Download a dataset file.

    Args:
        dataset_id: ID of the dataset to download
        format: File format (parquet or csv)
        service: Dataset service

    Returns:
        Dataset file
    """
    try:
        dataset = await service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=
                f'Dataset {dataset_id} not found')
        import re
        if not re.match('^[a-zA-Z0-9_-]+$', dataset_id):
            raise HTTPException(status_code=400, detail=
                'Invalid dataset ID format')
        if format.lower() == 'csv':
            file_path = os.path.join(service.dataset_dir, f'{dataset_id}.csv')
            if not os.path.exists(file_path):
                parquet_path = os.path.join(service.dataset_dir,
                    f'{dataset_id}.parquet')
                if os.path.exists(parquet_path):
                    import pandas as pd
                    df = pd.read_parquet(parquet_path)
                    df.to_csv(file_path)
                else:
                    df = await service.load_dataset(dataset_id,
                        as_train_test_split=False)
                    df.to_csv(file_path)
            media_type = 'text/csv'
            safe_name = re.sub('[^\\w\\-\\.]', '_', dataset.name)
            safe_symbol = re.sub('[^\\w\\-\\.]', '_', dataset.symbol)
            safe_timeframe = re.sub('[^\\w\\-\\.]', '_', dataset.timeframe)
            filename = f'{safe_name}_{safe_symbol}_{safe_timeframe}.csv'
        else:
            file_path = os.path.join(service.dataset_dir,
                f'{dataset_id}.parquet')
            if not os.path.exists(file_path):
                _, file_path = await service.generate_dataset_from_features(
                    DatasetCreate(name=dataset.name, symbol=dataset.symbol,
                    timeframe=dataset.timeframe, features=dataset.features,
                    target=dataset.target, start_date=dataset.start_date,
                    end_date=dataset.end_date, split_ratio=dataset.
                    split_ratio, description=dataset.description,
                    preprocessing=dataset.preprocessing))
            media_type = 'application/octet-stream'
            safe_name = re.sub('[^\\w\\-\\.]', '_', dataset.name)
            safe_symbol = re.sub('[^\\w\\-\\.]', '_', dataset.symbol)
            safe_timeframe = re.sub('[^\\w\\-\\.]', '_', dataset.timeframe)
            filename = f'{safe_name}_{safe_symbol}_{safe_timeframe}.parquet'
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=
                f'Dataset file not found')
        return FileResponse(path=file_path, filename=filename, media_type=
            media_type)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
