"""
Feature Computation Scheduler Module.

Provides scheduling capabilities for periodic feature computation.
"""
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Union, Any, Callable
import uuid
import aiohttp
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from core_foundations.utils.logger import get_logger
logger = get_logger('feature-scheduler')


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FeatureComputationScheduler:
    """
    Scheduler for periodic feature computation.
    
    This class manages periodic jobs to compute features, ensuring that
    the feature store is kept up-to-date with the latest data.
    """

    def __init__(self, api_base_url: str, schedule_config_path: Optional[
        str]=None):
        """
        Initialize the feature computation scheduler.
        
        Args:
            api_base_url: Base URL for the feature-store API
            schedule_config_path: Path to the schedule configuration file (optional)
        """
        self.api_base_url = api_base_url
        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_jobstore(MemoryJobStore(), 'default')
        self.schedule_config_path = schedule_config_path
        self.schedules = {}

    @async_with_exception_handling
    async def initialize(self) ->None:
        """
        Initialize the scheduler.
        
        This loads any saved schedules from the configuration file and sets up
        the required jobs.
        """
        if self.schedule_config_path and os.path.exists(self.
            schedule_config_path):
            try:
                with open(self.schedule_config_path, 'r') as f:
                    self.schedules = json.load(f)
                for schedule_id, schedule_info in self.schedules.items():
                    await self._create_job_from_config(schedule_id,
                        schedule_info)
                logger.info(
                    f'Loaded {len(self.schedules)} schedules from configuration'
                    )
            except Exception as e:
                logger.error(
                    f'Error loading schedules from {self.schedule_config_path}: {str(e)}'
                    )

    async def start(self) ->None:
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info('Feature computation scheduler started')

    async def stop(self) ->None:
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info('Feature computation scheduler stopped')

    @async_with_exception_handling
    async def save_schedules(self) ->None:
        """Save the current schedules to the configuration file."""
        if self.schedule_config_path:
            try:
                with open(self.schedule_config_path, 'w') as f:
                    json.dump(self.schedules, f, indent=2, default=str)
                logger.info(
                    f'Saved {len(self.schedules)} schedules to {self.schedule_config_path}'
                    )
            except Exception as e:
                logger.error(
                    f'Error saving schedules to {self.schedule_config_path}: {str(e)}'
                    )

    async def create_schedule(self, name: str, symbols: List[str],
        timeframes: List[str], features: List[Union[str, Dict[str, Any]]],
        schedule_type: str, schedule_params: Dict[str, Any], lookback_days:
        int=7, store_results: bool=True, compute_transformations: bool=
        False, enabled: bool=True) ->str:
        """
        Create a new schedule for feature computation.
        
        Args:
            name: Human-readable name for the schedule
            symbols: List of symbols to compute features for
            timeframes: List of timeframes to compute features for
            features: List of features to compute
            schedule_type: Type of schedule (interval, cron, or fixed_time)
            schedule_params: Parameters for the schedule
            lookback_days: Number of days to look back for data
            store_results: Whether to store results in the feature repository
            compute_transformations: Whether to compute time series transformations
            enabled: Whether the schedule is enabled
            
        Returns:
            ID of the created schedule
        """
        schedule_id = str(uuid.uuid4())
        schedule_config = {'id': schedule_id, 'name': name, 'symbols':
            symbols, 'timeframes': timeframes, 'features': features,
            'schedule_type': schedule_type, 'schedule_params':
            schedule_params, 'lookback_days': lookback_days,
            'store_results': store_results, 'compute_transformations':
            compute_transformations, 'enabled': enabled, 'created_at':
            datetime.now(timezone.utc).isoformat(), 'last_run': None,
            'next_run': None, 'status': 'created'}
        self.schedules[schedule_id] = schedule_config
        if enabled:
            await self._create_job_from_config(schedule_id, schedule_config)
        await self.save_schedules()
        return schedule_id

    async def update_schedule(self, schedule_id: str, **kwargs) ->None:
        """
        Update an existing schedule.
        
        Args:
            schedule_id: ID of the schedule to update
            **kwargs: Fields to update
        
        Raises:
            ValueError: If the schedule doesn't exist
        """
        if schedule_id not in self.schedules:
            raise ValueError(f'Schedule {schedule_id} not found')
        for key, value in kwargs.items():
            if key in self.schedules[schedule_id]:
                self.schedules[schedule_id][key] = value
        if self.schedules[schedule_id]['enabled']:
            self.scheduler.remove_job(schedule_id)
            await self._create_job_from_config(schedule_id, self.schedules[
                schedule_id])
        await self.save_schedules()

    @async_with_exception_handling
    async def delete_schedule(self, schedule_id: str) ->None:
        """
        Delete a schedule.
        
        Args:
            schedule_id: ID of the schedule to delete
        
        Raises:
            ValueError: If the schedule doesn't exist
        """
        if schedule_id not in self.schedules:
            raise ValueError(f'Schedule {schedule_id} not found')
        try:
            self.scheduler.remove_job(schedule_id)
        except:
            pass
        del self.schedules[schedule_id]
        await self.save_schedules()

    async def get_schedule(self, schedule_id: str) ->Dict[str, Any]:
        """
        Get a schedule by ID.
        
        Args:
            schedule_id: ID of the schedule to retrieve
            
        Returns:
            Schedule configuration
            
        Raises:
            ValueError: If the schedule doesn't exist
        """
        if schedule_id not in self.schedules:
            raise ValueError(f'Schedule {schedule_id} not found')
        return self.schedules[schedule_id]

    async def list_schedules(self) ->List[Dict[str, Any]]:
        """
        List all schedules.
        
        Returns:
            List of schedule configurations
        """
        return list(self.schedules.values())

    async def run_schedule_now(self, schedule_id: str) ->str:
        """
        Manually trigger a schedule to run immediately.
        
        Args:
            schedule_id: ID of the schedule to run
            
        Returns:
            ID of the triggered job
            
        Raises:
            ValueError: If the schedule doesn't exist
        """
        if schedule_id not in self.schedules:
            raise ValueError(f'Schedule {schedule_id} not found')
        schedule_config = self.schedules[schedule_id]
        job_id = await self._trigger_feature_computation(schedule_config)
        self.schedules[schedule_id]['last_run'] = datetime.now(timezone.utc
            ).isoformat()
        self.schedules[schedule_id]['status'] = 'running'
        await self.save_schedules()
        return job_id

    async def _create_job_from_config(self, schedule_id: str,
        schedule_config: Dict[str, Any]) ->None:
        """
        Create a scheduled job from a configuration.
        
        Args:
            schedule_id: ID of the schedule
            schedule_config: Schedule configuration
        """
        schedule_type = schedule_config['schedule_type']
        params = schedule_config['schedule_params']
        if schedule_type == 'interval':
            trigger = IntervalTrigger(**params)
        elif schedule_type == 'cron':
            trigger = CronTrigger(**params)
        elif schedule_type == 'fixed_time':
            run_time = params.get('run_time', '00:00')
            hour, minute = run_time.split(':')
            trigger = CronTrigger(hour=hour, minute=minute)
        else:
            logger.error(f'Unknown schedule type: {schedule_type}')
            return
        next_run = self.scheduler.add_job(func=self._scheduled_job, trigger
            =trigger, args=[schedule_id], id=schedule_id)
        if next_run:
            self.schedules[schedule_id]['next_run'
                ] = next_run.next_run_time.isoformat()

    @async_with_exception_handling
    async def _scheduled_job(self, schedule_id: str) ->None:
        """
        Function called by the scheduler when a job is triggered.
        
        Args:
            schedule_id: ID of the schedule being run
        """
        if schedule_id not in self.schedules:
            logger.error(f'Schedule {schedule_id} not found')
            return
        schedule_config = self.schedules[schedule_id]
        try:
            self.schedules[schedule_id]['status'] = 'running'
            self.schedules[schedule_id]['last_run'] = datetime.now(timezone.utc
                ).isoformat()
            await self._trigger_feature_computation(schedule_config)
            self.schedules[schedule_id]['status'] = 'completed'
            next_run = self.scheduler.get_job(schedule_id).next_run_time
            if next_run:
                self.schedules[schedule_id]['next_run'] = next_run.isoformat()
            await self.save_schedules()
        except Exception as e:
            logger.error(f'Error running scheduled job {schedule_id}: {str(e)}'
                )
            self.schedules[schedule_id]['status'] = 'error'
            self.schedules[schedule_id]['last_error'] = str(e)
            self.schedules[schedule_id]['last_error_time'] = datetime.now(
                timezone.utc).isoformat()
            await self.save_schedules()

    @async_with_exception_handling
    async def _trigger_feature_computation(self, schedule_config: Dict[str,
        Any]) ->str:
        """
        Trigger a feature computation job via the API.
        
        Args:
            schedule_config: Schedule configuration
            
        Returns:
            ID of the triggered job
        """
        try:
            to_time = datetime.now(timezone.utc)
            lookback_days = schedule_config_manager.get('lookback_days', 7)
            from_time = to_time - timedelta(days=lookback_days)
            request_data = {'symbols': schedule_config['symbols'],
                'timeframes': schedule_config['timeframes'], 'features':
                schedule_config['features'], 'from_time': from_time.
                isoformat(), 'to_time': to_time.isoformat(),
                'store_results': schedule_config_manager.get('store_results', True),
                'compute_transformations': schedule_config.get(
                'compute_transformations', False)}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{self.api_base_url}/api/v1/feature-computation/compute-batch'
                    , json=request_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        job_id = result.get('job_id')
                        logger.info(
                            f"Triggered computation job {job_id} for schedule {schedule_config_manager.get('name')}"
                            )
                        return job_id
                    else:
                        error_text = await response.text()
                        raise ValueError(
                            f'Error triggering computation job: {response.status} - {error_text}'
                            )
        except Exception as e:
            logger.error(f'Error triggering feature computation: {str(e)}')
            raise
