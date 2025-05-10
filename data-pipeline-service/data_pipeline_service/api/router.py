"""
Main API router for data pipeline service.

This module includes all API routers from different API versions
and components of the data pipeline service.
"""
from fastapi import APIRouter

from data_pipeline_service.api.v1 import adapters, cleaning, instruments, ohlcv, tick_data, data_access, monitoring

# Create main API router
api_router = APIRouter()

# Include all routers from v1 API
api_router.include_router(adapters.router, prefix="/v1/adapters", tags=["adapters"])
api_router.include_router(cleaning.router, prefix="/v1/cleaning", tags=["cleaning"])
api_router.include_router(instruments.router, prefix="/v1/instruments", tags=["instruments"])
api_router.include_router(ohlcv.router, prefix="/v1/ohlcv", tags=["ohlcv"])
api_router.include_router(tick_data.router, prefix="/v1/tick-data", tags=["tick-data"])
api_router.include_router(data_access.router, prefix="/v1/data-access", tags=["data-access"])
api_router.include_router(monitoring.router, prefix="/v1/monitoring", tags=["monitoring"])