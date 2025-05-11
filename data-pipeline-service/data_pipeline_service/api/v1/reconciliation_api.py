"""
Reconciliation API endpoints for the Data Pipeline Service.

This module provides API endpoints for triggering and managing data reconciliation
processes for market data.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from pydantic import BaseModel, Field, validator

from data_pipeline_service.services.reconciliation_service import ReconciliationService
from data_pipeline_service.api.auth import get_api_key
from common_lib.data_reconciliation import (
    ReconciliationStatus,
    ReconciliationSeverity,
    ReconciliationStrategy,
    DataSourceType,
)
from common_lib.exceptions import (
    DataFetchError,
    DataValidationError,
    ReconciliationError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reconciliation", tags=["reconciliation"])


class OHLCVReconciliationRequest(BaseModel):
    """Request model for OHLCV data reconciliation."""

    symbol: str = Field(..., description="Symbol or instrument for the data")
    start_date: datetime = Field(..., description="Start date for data reconciliation")
    end_date: datetime = Field(..., description="End date for data reconciliation")
    timeframe: str = Field(..., description="Timeframe for the data (e.g., '1h', '1d')")
    strategy: ReconciliationStrategy = Field(
        ReconciliationStrategy.SOURCE_PRIORITY,
        description="Strategy for resolving discrepancies"
    )
    tolerance: float = Field(0.0001, description="Tolerance for numerical differences")
    auto_resolve: bool = Field(True, description="Whether to automatically resolve discrepancies")
    notification_threshold: ReconciliationSeverity = Field(
        ReconciliationSeverity.HIGH,
        description="Minimum severity for notifications"
    )

    @validator('tolerance')
    def validate_tolerance(cls, v):
        """Validate tolerance value."""
        if v < 0:
            raise ValueError("Tolerance must be non-negative")
        return v

    @validator('end_date')
    def validate_end_date(cls, v, values):
        """Validate end_date is after start_date."""
        if v and 'start_date' in values and values['start_date'] and v < values['start_date']:
            raise ValueError("End date must be after start date")
        return v

    @validator('timeframe')
    def validate_timeframe(cls, v):
        """Validate timeframe format."""
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
        if v not in valid_timeframes:
            raise ValueError(f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}")
        return v


class TickDataReconciliationRequest(BaseModel):
    """Request model for tick data reconciliation."""

    symbol: str = Field(..., description="Symbol or instrument for the data")
    start_date: datetime = Field(..., description="Start date for data reconciliation")
    end_date: datetime = Field(..., description="End date for data reconciliation")
    strategy: ReconciliationStrategy = Field(
        ReconciliationStrategy.SOURCE_PRIORITY,
        description="Strategy for resolving discrepancies"
    )
    tolerance: float = Field(0.0001, description="Tolerance for numerical differences")
    auto_resolve: bool = Field(True, description="Whether to automatically resolve discrepancies")
    notification_threshold: ReconciliationSeverity = Field(
        ReconciliationSeverity.HIGH,
        description="Minimum severity for notifications"
    )

    @validator('tolerance')
    def validate_tolerance(cls, v):
        """Validate tolerance value."""
        if v < 0:
            raise ValueError("Tolerance must be non-negative")
        return v

    @validator('end_date')
    def validate_end_date(cls, v, values):
        """Validate end_date is after start_date."""
        if v and 'start_date' in values and values['start_date'] and v < values['start_date']:
            raise ValueError("End date must be after start date")
        return v


class ReconciliationResponse(BaseModel):
    """Response model for data reconciliation."""

    reconciliation_id: str = Field(..., description="ID of the reconciliation process")
    status: str = Field(..., description="Status of the reconciliation process")
    discrepancy_count: int = Field(..., description="Number of discrepancies found")
    resolution_count: int = Field(..., description="Number of resolutions applied")
    resolution_rate: float = Field(..., description="Percentage of discrepancies resolved")
    duration_seconds: Optional[float] = Field(None, description="Duration of the reconciliation process in seconds")
    start_time: datetime = Field(..., description="Start time of the reconciliation process")
    end_time: Optional[datetime] = Field(None, description="End time of the reconciliation process")


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the error")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking the request")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Additional information about the error")


@router.post("/ohlcv", response_model=ReconciliationResponse, responses={
    400: {"model": ErrorResponse, "description": "Bad Request"},
    401: {"model": ErrorResponse, "description": "Unauthorized"},
    500: {"model": ErrorResponse, "description": "Internal Server Error"}
})
async def reconcile_ohlcv_data(
    request: OHLCVReconciliationRequest = Body(...),
    api_key: str = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Reconcile OHLCV data.

    This endpoint triggers a reconciliation process for OHLCV data,
    comparing data from different sources and resolving discrepancies.
    """
    reconciliation_service = ReconciliationService()

    try:
        # Log the reconciliation request
        logger.info(
            f"Received OHLCV data reconciliation request for symbol {request.symbol}, "
            f"timeframe {request.timeframe}, strategy {request.strategy.name}"
        )

        result = await reconciliation_service.reconcile_ohlcv_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
            strategy=request.strategy,
            tolerance=request.tolerance,
            auto_resolve=request.auto_resolve,
            notification_threshold=request.notification_threshold
        )

        # Log the reconciliation result
        logger.info(
            f"Completed OHLCV data reconciliation for symbol {request.symbol}, "
            f"reconciliation ID: {result.reconciliation_id}, "
            f"status: {result.status.name}, "
            f"discrepancies: {result.discrepancy_count}, "
            f"resolutions: {result.resolution_count}"
        )

        return {
            "reconciliation_id": result.reconciliation_id,
            "status": result.status.name,
            "discrepancy_count": result.discrepancy_count,
            "resolution_count": result.resolution_count,
            "resolution_rate": result.resolution_rate,
            "duration_seconds": result.duration_seconds,
            "start_time": result.start_time,
            "end_time": result.end_time
        }
    except DataFetchError as e:
        logger.error(f"Data fetch error during OHLCV data reconciliation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": f"Error fetching data: {str(e)}",
                "error_type": "DataFetchError",
                "additional_info": getattr(e, "details", None)
            }
        )
    except DataValidationError as e:
        logger.error(f"Data validation error during OHLCV data reconciliation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": f"Data validation error: {str(e)}",
                "error_type": "DataValidationError",
                "additional_info": getattr(e, "details", None)
            }
        )
    except ReconciliationError as e:
        logger.error(f"Reconciliation error during OHLCV data reconciliation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": f"Reconciliation error: {str(e)}",
                "error_type": "ReconciliationError",
                "additional_info": getattr(e, "details", None)
            }
        )
    except ValueError as e:
        logger.error(f"Validation error during OHLCV data reconciliation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": f"Validation error: {str(e)}",
                "error_type": "ValidationError"
            }
        )
    except Exception as e:
        logger.exception(f"Unexpected error during OHLCV data reconciliation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": f"Unexpected error: {str(e)}",
                "error_type": "UnexpectedError"
            }
        )


@router.post("/tick-data", response_model=ReconciliationResponse, responses={
    400: {"model": ErrorResponse, "description": "Bad Request"},
    401: {"model": ErrorResponse, "description": "Unauthorized"},
    500: {"model": ErrorResponse, "description": "Internal Server Error"}
})
async def reconcile_tick_data(
    request: TickDataReconciliationRequest = Body(...),
    api_key: str = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Reconcile tick data.

    This endpoint triggers a reconciliation process for tick data,
    comparing data from different sources and resolving discrepancies.
    """
    reconciliation_service = ReconciliationService()

    try:
        # Log the reconciliation request
        logger.info(
            f"Received tick data reconciliation request for symbol {request.symbol}, "
            f"strategy {request.strategy.name}"
        )

        result = await reconciliation_service.reconcile_tick_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            strategy=request.strategy,
            tolerance=request.tolerance,
            auto_resolve=request.auto_resolve,
            notification_threshold=request.notification_threshold
        )

        # Log the reconciliation result
        logger.info(
            f"Completed tick data reconciliation for symbol {request.symbol}, "
            f"reconciliation ID: {result.reconciliation_id}, "
            f"status: {result.status.name}, "
            f"discrepancies: {result.discrepancy_count}, "
            f"resolutions: {result.resolution_count}"
        )

        return {
            "reconciliation_id": result.reconciliation_id,
            "status": result.status.name,
            "discrepancy_count": result.discrepancy_count,
            "resolution_count": result.resolution_count,
            "resolution_rate": result.resolution_rate,
            "duration_seconds": result.duration_seconds,
            "start_time": result.start_time,
            "end_time": result.end_time
        }
    except DataFetchError as e:
        logger.error(f"Data fetch error during tick data reconciliation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": f"Error fetching data: {str(e)}",
                "error_type": "DataFetchError",
                "additional_info": getattr(e, "details", None)
            }
        )
    except DataValidationError as e:
        logger.error(f"Data validation error during tick data reconciliation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": f"Data validation error: {str(e)}",
                "error_type": "DataValidationError",
                "additional_info": getattr(e, "details", None)
            }
        )
    except ReconciliationError as e:
        logger.error(f"Reconciliation error during tick data reconciliation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": f"Reconciliation error: {str(e)}",
                "error_type": "ReconciliationError",
                "additional_info": getattr(e, "details", None)
            }
        )
    except ValueError as e:
        logger.error(f"Validation error during tick data reconciliation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": f"Validation error: {str(e)}",
                "error_type": "ValidationError"
            }
        )
    except Exception as e:
        logger.exception(f"Unexpected error during tick data reconciliation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": f"Unexpected error: {str(e)}",
                "error_type": "UnexpectedError"
            }
        )


@router.get("/{reconciliation_id}", response_model=ReconciliationResponse, responses={
    400: {"model": ErrorResponse, "description": "Bad Request"},
    401: {"model": ErrorResponse, "description": "Unauthorized"},
    404: {"model": ErrorResponse, "description": "Not Found"},
    500: {"model": ErrorResponse, "description": "Internal Server Error"}
})
async def get_reconciliation_status(
    reconciliation_id: str,
    api_key: str = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Get the status of a reconciliation process.

    This endpoint retrieves the current status and results of a reconciliation process.
    """
    reconciliation_service = ReconciliationService()

    try:
        # Log the status request
        logger.info(f"Received reconciliation status request for ID {reconciliation_id}")

        result = await reconciliation_service.get_reconciliation_status(reconciliation_id)

        if not result:
            logger.warning(f"Reconciliation process with ID {reconciliation_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "detail": f"Reconciliation process with ID {reconciliation_id} not found",
                    "error_type": "NotFoundError"
                }
            )

        # Log the status result
        logger.info(
            f"Retrieved reconciliation status for ID {reconciliation_id}, "
            f"status: {result.status.name}"
        )

        return {
            "reconciliation_id": result.reconciliation_id,
            "status": result.status.name,
            "discrepancy_count": result.discrepancy_count,
            "resolution_count": result.resolution_count,
            "resolution_rate": result.resolution_rate,
            "duration_seconds": result.duration_seconds,
            "start_time": result.start_time,
            "end_time": result.end_time
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error getting reconciliation status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": f"Unexpected error: {str(e)}",
                "error_type": "UnexpectedError"
            }
        )
