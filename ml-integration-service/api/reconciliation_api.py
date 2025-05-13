"""
Reconciliation API endpoints for the ML Integration Service.

This module provides API endpoints for triggering and managing data reconciliation
processes for model data.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Body, status, Request
from pydantic import BaseModel, Field, validator
from core.container import get_reconciliation_service
from core.security import get_api_key
from core.rate_limiter import get_reconciliation_rate_limiter
from common_lib.data_reconciliation import ReconciliationStatus, ReconciliationSeverity, ReconciliationStrategy, DataSourceType
from common_lib.exceptions import DataFetchError, DataValidationError, ReconciliationError
logger = logging.getLogger(__name__)
router = APIRouter(prefix='/reconciliation', tags=['reconciliation'])


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ReconciliationRequest(BaseModel):
    """Request model for data reconciliation."""
    model_id: str = Field(..., description=
        'ID of the model to reconcile data for')
    version: Optional[str] = Field(None, description='Version of the model')
    start_time: Optional[datetime] = Field(None, description=
        'Start time for data reconciliation')
    end_time: Optional[datetime] = Field(None, description=
        'End time for data reconciliation')
    strategy: ReconciliationStrategy = Field(ReconciliationStrategy.
        SOURCE_PRIORITY, description='Strategy for resolving discrepancies')
    tolerance: float = Field(0.0001, description=
        'Tolerance for numerical differences')
    auto_resolve: bool = Field(True, description=
        'Whether to automatically resolve discrepancies')
    notification_threshold: ReconciliationSeverity = Field(
        ReconciliationSeverity.HIGH, description=
        'Minimum severity for notifications')

    @validator('tolerance')
    def validate_tolerance(cls, v):
        """Validate tolerance value."""
        if v < 0:
            raise ValueError('Tolerance must be non-negative')
        return v

    @validator('end_time')
    def validate_end_time(cls, v, values):
        """Validate end_time is after start_time."""
        if v and 'start_time' in values and values['start_time'
            ] and v < values['start_time']:
            raise ValueError('End time must be after start time')
        return v


class ReconciliationResponse(BaseModel):
    """Response model for data reconciliation."""
    reconciliation_id: str = Field(..., description=
        'ID of the reconciliation process')
    status: str = Field(..., description='Status of the reconciliation process'
        )
    discrepancy_count: int = Field(..., description=
        'Number of discrepancies found')
    resolution_count: int = Field(..., description=
        'Number of resolutions applied')
    resolution_rate: float = Field(..., description=
        'Percentage of discrepancies resolved')
    duration_seconds: Optional[float] = Field(None, description=
        'Duration of the reconciliation process in seconds')
    start_time: datetime = Field(..., description=
        'Start time of the reconciliation process')
    end_time: Optional[datetime] = Field(None, description=
        'End time of the reconciliation process')


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description='Error message')
    error_type: str = Field(..., description='Type of error')
    timestamp: datetime = Field(default_factory=datetime.utcnow,
        description='Timestamp of the error')
    correlation_id: Optional[str] = Field(None, description=
        'Correlation ID for tracking the request')
    additional_info: Optional[Dict[str, Any]] = Field(None, description=
        'Additional information about the error')


@router.post('/training-data', response_model=ReconciliationResponse,
    responses={(400): {'model': ErrorResponse, 'description': 'Bad Request'
    }, (401): {'model': ErrorResponse, 'description': 'Unauthorized'}, (429
    ): {'model': ErrorResponse, 'description': 'Too Many Requests'}, (500):
    {'model': ErrorResponse, 'description': 'Internal Server Error'}})
@async_with_exception_handling
async def reconcile_training_data(request: ReconciliationRequest=Body(...),
    api_key: str=Depends(get_api_key), req: Request=None, rate_limiter: Any
    =Depends(get_reconciliation_rate_limiter)) ->Dict[str, Any]:
    """
    Reconcile training data for a model.

    This endpoint triggers a reconciliation process for model training data,
    comparing data from different sources and resolving discrepancies.
    """
    await rate_limiter(req)
    reconciliation_service = get_reconciliation_service()
    try:
        logger.info(
            f'Received training data reconciliation request for model {request.model_id}, version {request.version}, strategy {request.strategy.name}'
            )
        result = await reconciliation_service.reconcile_training_data(model_id
            =request.model_id, version=request.version, start_time=request.
            start_time, end_time=request.end_time, strategy=request.
            strategy, tolerance=request.tolerance, auto_resolve=request.
            auto_resolve, notification_threshold=request.notification_threshold
            )
        logger.info(
            f'Completed training data reconciliation for model {request.model_id}, reconciliation ID: {result.reconciliation_id}, status: {result.status.name}, discrepancies: {result.discrepancy_count}, resolutions: {result.resolution_count}'
            )
        return {'reconciliation_id': result.reconciliation_id, 'status':
            result.status.name, 'discrepancy_count': result.
            discrepancy_count, 'resolution_count': result.resolution_count,
            'resolution_rate': result.resolution_rate, 'duration_seconds':
            result.duration_seconds, 'start_time': result.start_time,
            'end_time': result.end_time}
    except DataFetchError as e:
        logger.error(
            f'Data fetch error during training data reconciliation: {str(e)}')
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail
            ={'detail': f'Error fetching data: {str(e)}', 'error_type':
            'DataFetchError', 'additional_info': getattr(e, 'details', None)})
    except DataValidationError as e:
        logger.error(
            f'Data validation error during training data reconciliation: {str(e)}'
            )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail
            ={'detail': f'Data validation error: {str(e)}', 'error_type':
            'DataValidationError', 'additional_info': getattr(e, 'details',
            None)})
    except ReconciliationError as e:
        logger.error(
            f'Reconciliation error during training data reconciliation: {str(e)}'
            )
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail={'detail':
            f'Reconciliation error: {str(e)}', 'error_type':
            'ReconciliationError', 'additional_info': getattr(e, 'details',
            None)})
    except ValueError as e:
        logger.error(
            f'Validation error during training data reconciliation: {str(e)}')
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail
            ={'detail': f'Validation error: {str(e)}', 'error_type':
            'ValidationError'})
    except Exception as e:
        logger.exception(
            f'Unexpected error during training data reconciliation: {str(e)}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail={'detail':
            f'Unexpected error: {str(e)}', 'error_type': 'UnexpectedError'})


@router.post('/inference-data', response_model=ReconciliationResponse,
    responses={(400): {'model': ErrorResponse, 'description': 'Bad Request'
    }, (401): {'model': ErrorResponse, 'description': 'Unauthorized'}, (429
    ): {'model': ErrorResponse, 'description': 'Too Many Requests'}, (500):
    {'model': ErrorResponse, 'description': 'Internal Server Error'}})
@async_with_exception_handling
async def reconcile_inference_data(request: ReconciliationRequest=Body(...),
    api_key: str=Depends(get_api_key), req: Request=None, rate_limiter: Any
    =Depends(get_reconciliation_rate_limiter)) ->Dict[str, Any]:
    """
    Reconcile inference data for a model.

    This endpoint triggers a reconciliation process for model inference data,
    comparing data from different sources and resolving discrepancies.
    """
    await rate_limiter(req)
    reconciliation_service = get_reconciliation_service()
    try:
        logger.info(
            f'Received inference data reconciliation request for model {request.model_id}, version {request.version}, strategy {request.strategy.name}'
            )
        result = await reconciliation_service.reconcile_inference_data(model_id
            =request.model_id, version=request.version, start_time=request.
            start_time, end_time=request.end_time, strategy=request.
            strategy, tolerance=request.tolerance, auto_resolve=request.
            auto_resolve, notification_threshold=request.notification_threshold
            )
        logger.info(
            f'Completed inference data reconciliation for model {request.model_id}, reconciliation ID: {result.reconciliation_id}, status: {result.status.name}, discrepancies: {result.discrepancy_count}, resolutions: {result.resolution_count}'
            )
        return {'reconciliation_id': result.reconciliation_id, 'status':
            result.status.name, 'discrepancy_count': result.
            discrepancy_count, 'resolution_count': result.resolution_count,
            'resolution_rate': result.resolution_rate, 'duration_seconds':
            result.duration_seconds, 'start_time': result.start_time,
            'end_time': result.end_time}
    except DataFetchError as e:
        logger.error(
            f'Data fetch error during inference data reconciliation: {str(e)}')
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail
            ={'detail': f'Error fetching data: {str(e)}', 'error_type':
            'DataFetchError', 'additional_info': getattr(e, 'details', None)})
    except DataValidationError as e:
        logger.error(
            f'Data validation error during inference data reconciliation: {str(e)}'
            )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail
            ={'detail': f'Data validation error: {str(e)}', 'error_type':
            'DataValidationError', 'additional_info': getattr(e, 'details',
            None)})
    except ReconciliationError as e:
        logger.error(
            f'Reconciliation error during inference data reconciliation: {str(e)}'
            )
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail={'detail':
            f'Reconciliation error: {str(e)}', 'error_type':
            'ReconciliationError', 'additional_info': getattr(e, 'details',
            None)})
    except ValueError as e:
        logger.error(
            f'Validation error during inference data reconciliation: {str(e)}')
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail
            ={'detail': f'Validation error: {str(e)}', 'error_type':
            'ValidationError'})
    except Exception as e:
        logger.exception(
            f'Unexpected error during inference data reconciliation: {str(e)}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail={'detail':
            f'Unexpected error: {str(e)}', 'error_type': 'UnexpectedError'})


@router.get('/{reconciliation_id}', response_model=ReconciliationResponse,
    responses={(400): {'model': ErrorResponse, 'description': 'Bad Request'
    }, (401): {'model': ErrorResponse, 'description': 'Unauthorized'}, (404
    ): {'model': ErrorResponse, 'description': 'Not Found'}, (429): {
    'model': ErrorResponse, 'description': 'Too Many Requests'}, (500): {
    'model': ErrorResponse, 'description': 'Internal Server Error'}})
@async_with_exception_handling
async def get_reconciliation_status(reconciliation_id: str, api_key: str=
    Depends(get_api_key), req: Request=None, rate_limiter: Any=Depends(
    get_reconciliation_rate_limiter)) ->Dict[str, Any]:
    """
    Get the status of a reconciliation process.

    This endpoint retrieves the current status and results of a reconciliation process.
    """
    await rate_limiter(req)
    reconciliation_service = get_reconciliation_service()
    try:
        logger.info(
            f'Received reconciliation status request for ID {reconciliation_id}'
            )
        result = await reconciliation_service.get_reconciliation_status(
            reconciliation_id)
        if not result:
            logger.warning(
                f'Reconciliation process with ID {reconciliation_id} not found'
                )
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                detail={'detail':
                f'Reconciliation process with ID {reconciliation_id} not found'
                , 'error_type': 'NotFoundError'})
        logger.info(
            f'Retrieved reconciliation status for ID {reconciliation_id}, status: {result.status.name}'
            )
        return {'reconciliation_id': result.reconciliation_id, 'status':
            result.status.name, 'discrepancy_count': result.
            discrepancy_count, 'resolution_count': result.resolution_count,
            'resolution_rate': result.resolution_rate, 'duration_seconds':
            result.duration_seconds, 'start_time': result.start_time,
            'end_time': result.end_time}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f'Unexpected error getting reconciliation status: {str(e)}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail={'detail':
            f'Unexpected error: {str(e)}', 'error_type': 'UnexpectedError'})
