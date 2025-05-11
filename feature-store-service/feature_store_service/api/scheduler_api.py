"""
Scheduler API module.

This module provides API endpoints for scheduling and managing recurring indicator computations.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field, validator
import uuid
from core_foundations.utils.logger import get_logger
from feature_store_service.scheduling.scheduler import ComputationScheduler
logger = get_logger('feature-store-service.scheduler-api')
scheduler_router = APIRouter(prefix='/api/v1/scheduler', tags=['scheduler'], responses={404: {'description': 'Not found'}})
scheduler: Optional[ComputationScheduler] = None

class ScheduleInterval(BaseModel):
    """Schedule interval for recurring computations."""
    unit: str = Field(..., description='Time unit (minutes, hours, days)')
    value: int = Field(..., description='Number of time units')

    @validator('unit')
    def validate_unit(cls, v):
        """Validate the time unit."""
        if v not in ['minutes', 'hours', 'days', 'weeks']:
            raise ValueError('Unit must be one of: minutes, hours, days, weeks')
        return v

    @validator('value')
    def validate_value(cls, v, values):
        """Validate the value based on the unit."""
        unit = values.get('unit')
        if unit == 'minutes' and (v < 5 or v > 1440):
            raise ValueError('Minutes must be between 5 and 1440')
        elif unit == 'hours' and (v < 1 or v > 168):
            raise ValueError('Hours must be between 1 and 168')
        elif unit == 'days' and (v < 1 or v > 90):
            raise ValueError('Days must be between 1 and 90')
        elif unit == 'weeks' and (v < 1 or v > 52):
            raise ValueError('Weeks must be between 1 and 52')
        return v

class IndicatorParams(BaseModel):
    """Parameters for an indicator."""
    indicator_id: str
    params: Optional[Dict[str, Any]] = None

class ScheduleRequest(BaseModel):
    """Request model for scheduling recurring computations."""
    name: str = Field(..., description='Name of the scheduled job')
    description: Optional[str] = None
    symbols: List[str] = Field(..., description='List of symbols to compute indicators for')
    timeframes: List[str] = Field(..., description='List of timeframes to compute indicators for')
    indicators: List[Union[str, IndicatorParams]] = Field(..., description='Indicators to compute')
    lookback_days: int = Field(30, description='Number of days of historical data to compute initially')
    interval: ScheduleInterval
    enabled: bool = True

class ScheduleInfo(BaseModel):
    """Information about a scheduled job."""
    id: str
    name: str
    description: Optional[str] = None
    symbols: List[str]
    timeframes: List[str]
    indicators: List[Union[str, Dict[str, Any]]]
    lookback_days: int
    interval: Dict[str, Any]
    enabled: bool
    created_at: datetime
    updated_at: datetime
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: str

@scheduler_router.post('/jobs', response_model=ScheduleInfo)
async def create_schedule(request: ScheduleRequest):
    """
    Create a new scheduled computation job.
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail='Scheduler is not initialized')
    try:
        indicator_specs = []
        for item in request.indicators:
            if isinstance(item, str):
                indicator_specs.append({'id': item})
            else:
                indicator_specs.append({'id': item.indicator_id, 'params': item.params})
        job_id = str(uuid.uuid4())
        interval_seconds = _get_interval_seconds(request.interval)
        job_info = await scheduler.add_job(job_id=job_id, name=request.name, description=request.description, symbols=request.symbols, timeframes=request.timeframes, indicators=indicator_specs, lookback_days=request.lookback_days, interval_seconds=interval_seconds, enabled=request.enabled)
        return job_info
    except Exception as e:
        logger.error(f'Error creating scheduled job: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@scheduler_router.get('/jobs', response_model=List[ScheduleInfo])
async def get_all_schedules():
    """
    Get all scheduled computation jobs.
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail='Scheduler is not initialized')
    try:
        return await scheduler.get_all_jobs()
    except Exception as e:
        logger.error(f'Error getting scheduled jobs: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@scheduler_router.get('/jobs/{job_id}', response_model=ScheduleInfo)
async def get_schedule(job_id: str):
    """
    Get details of a specific scheduled computation job.
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail='Scheduler is not initialized')
    try:
        job_info = await scheduler.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail=f'Job with ID {job_id} not found')
        return job_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting scheduled job: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@scheduler_router.put('/jobs/{job_id}', response_model=ScheduleInfo)
async def update_schedule(job_id: str, request: ScheduleRequest):
    """
    Update a scheduled computation job.
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail='Scheduler is not initialized')
    try:
        indicator_specs = []
        for item in request.indicators:
            if isinstance(item, str):
                indicator_specs.append({'id': item})
            else:
                indicator_specs.append({'id': item.indicator_id, 'params': item.params})
        interval_seconds = _get_interval_seconds(request.interval)
        job_info = await scheduler.update_job(job_id=job_id, name=request.name, description=request.description, symbols=request.symbols, timeframes=request.timeframes, indicators=indicator_specs, lookback_days=request.lookback_days, interval_seconds=interval_seconds, enabled=request.enabled)
        if not job_info:
            raise HTTPException(status_code=404, detail=f'Job with ID {job_id} not found')
        return job_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error updating scheduled job: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@scheduler_router.delete('/jobs/{job_id}')
async def delete_schedule(job_id: str):
    """
    Delete a scheduled computation job.
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail='Scheduler is not initialized')
    try:
        success = await scheduler.delete_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f'Job with ID {job_id} not found')
        return {'message': f'Job {job_id} deleted successfully'}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error deleting scheduled job: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@scheduler_router.post('/jobs/{job_id}/enable')
async def enable_schedule(job_id: str):
    """
    Enable a scheduled computation job.
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail='Scheduler is not initialized')
    try:
        success = await scheduler.enable_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f'Job with ID {job_id} not found')
        return {'message': f'Job {job_id} enabled successfully'}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error enabling scheduled job: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@scheduler_router.post('/jobs/{job_id}/disable')
async def disable_schedule(job_id: str):
    """
    Disable a scheduled computation job.
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail='Scheduler is not initialized')
    try:
        success = await scheduler.disable_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f'Job with ID {job_id} not found')
        return {'message': f'Job {job_id} disabled successfully'}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error disabling scheduled job: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@scheduler_router.post('/jobs/{job_id}/run')
async def run_schedule_now(job_id: str, background_tasks: BackgroundTasks):
    """
    Manually trigger a scheduled computation job to run immediately.
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail='Scheduler is not initialized')
    try:
        job_info = await scheduler.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail=f'Job with ID {job_id} not found')
        background_tasks.add_task(scheduler.run_job_now, job_id)
        return {'message': f'Job {job_id} triggered to run', 'job_name': job_info['name']}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error running scheduled job: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

def get_interval_seconds(interval: ScheduleInterval) -> int:
    """Convert interval to seconds."""
    if interval.unit == 'minutes':
        return interval.value * 60
    elif interval.unit == 'hours':
        return interval.value * 60 * 60
    elif interval.unit == 'days':
        return interval.value * 24 * 60 * 60
    elif interval.unit == 'weeks':
        return interval.value * 7 * 24 * 60 * 60
    else:
        raise ValueError(f'Unsupported interval unit: {interval.unit}')

def set_scheduler(scheduler_instance: ComputationScheduler):
    """Set the scheduler instance."""
    global scheduler
    scheduler = scheduler_instance