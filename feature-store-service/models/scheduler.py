"""
Computation Scheduler Module.

This module provides functionality for scheduling and executing recurring indicator
computation jobs.
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from core_foundations.utils.logger import get_logger
from core.feature_computation_engine import FeatureComputationEngine
logger = get_logger('feature-store-service.scheduler')


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ComputationScheduler:
    """
    Manages scheduled computation jobs for technical indicators.
    
    This class provides functionality for creating, updating, and executing
    recurring computation jobs for technical indicators.
    """

    def __init__(self, engine: AsyncEngine, computation_engine:
        FeatureComputationEngine, poll_interval_seconds: int=60):
        """
        Initialize the computation scheduler.
        
        Args:
            engine: SQLAlchemy async engine for database access
            computation_engine: Engine for computing technical indicators
            poll_interval_seconds: How often to check for jobs that need to run
        """
        self.engine = engine
        self.computation_engine = computation_engine
        self.poll_interval_seconds = poll_interval_seconds
        self.running = False
        self.jobs_task: Optional[asyncio.Task] = None

    async def initialize(self) ->None:
        """
        Initialize the scheduler tables and start the job runner.
        """
        await self._ensure_tables_exist()

    @async_with_exception_handling
    async def _ensure_tables_exist(self) ->None:
        """
        Ensure that the necessary tables exist in the database.
        """
        create_tables_sql = """
        -- Create the scheduler schema if it doesn't exist
        CREATE SCHEMA IF NOT EXISTS scheduler;
        
        -- Create the jobs table if it doesn't exist
        CREATE TABLE IF NOT EXISTS scheduler.jobs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            config JSONB NOT NULL,
            interval_seconds INTEGER NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_run TIMESTAMPTZ,
            next_run TIMESTAMPTZ,
            status TEXT NOT NULL DEFAULT 'pending'
        );
        
        -- Create the job history table
        CREATE TABLE IF NOT EXISTS scheduler.job_history (
            id SERIAL PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES scheduler.jobs(id) ON DELETE CASCADE,
            start_time TIMESTAMPTZ NOT NULL,
            end_time TIMESTAMPTZ,
            status TEXT NOT NULL,
            message TEXT,
            details JSONB,
            CONSTRAINT job_history_job_id_fk FOREIGN KEY (job_id) REFERENCES scheduler.jobs(id)
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_jobs_next_run ON scheduler.jobs(next_run) WHERE enabled = TRUE;
        CREATE INDEX IF NOT EXISTS idx_job_history_job_id ON scheduler.job_history(job_id);
        """
        try:
            async with self.engine.begin() as conn:
                await conn.execute(sa.text(create_tables_sql))
            logger.info('Scheduler tables initialized')
        except Exception as e:
            logger.error(f'Failed to create scheduler tables: {str(e)}')
            raise

    async def start(self) ->None:
        """
        Start the scheduler.
        """
        if self.running:
            logger.warning('Scheduler is already running')
            return
        self.running = True
        self.jobs_task = asyncio.create_task(self._run_scheduler_loop())
        logger.info('Scheduler started')

    @async_with_exception_handling
    async def stop(self) ->None:
        """
        Stop the scheduler.
        """
        if not self.running:
            logger.warning('Scheduler is not running')
            return
        self.running = False
        if self.jobs_task:
            self.jobs_task.cancel()
            try:
                await self.jobs_task
            except asyncio.CancelledError:
                pass
            self.jobs_task = None
        logger.info('Scheduler stopped')

    @async_with_exception_handling
    async def _run_scheduler_loop(self) ->None:
        """
        Main scheduler loop that checks for and runs due jobs.
        """
        while self.running:
            try:
                due_jobs = await self._get_due_jobs()
                for job_id, job_info in due_jobs:
                    asyncio.create_task(self._execute_job(job_id, job_info))
                await asyncio.sleep(self.poll_interval_seconds)
            except asyncio.CancelledError:
                logger.info('Scheduler loop cancelled')
                break
            except Exception as e:
                logger.error(f'Error in scheduler loop: {str(e)}')
                await asyncio.sleep(self.poll_interval_seconds)

    @async_with_exception_handling
    async def _get_due_jobs(self) ->List[Tuple[str, Dict[str, Any]]]:
        """
        Get jobs that are due to run.
        
        Returns:
            List of tuples with job ID and job info
        """
        query = """
        SELECT id, name, description, config, interval_seconds, created_at, updated_at, last_run, next_run, status
        FROM scheduler.jobs
        WHERE enabled = TRUE
        AND (next_run IS NULL OR next_run <= NOW())
        ORDER BY next_run NULLS FIRST
        LIMIT 10  -- Process in batches to avoid overwhelming the system
        """
        try:
            result = []
            async with self.engine.begin() as conn:
                rows = await conn.execute(sa.text(query))
                for row in rows:
                    job_id = row.id
                    job_info = {'id': job_id, 'name': row.name,
                        'description': row.description, 'config': row.
                        config, 'interval_seconds': row.interval_seconds,
                        'created_at': row.created_at, 'updated_at': row.
                        updated_at, 'last_run': row.last_run, 'next_run':
                        row.next_run, 'status': row.status}
                    result.append((job_id, job_info))
            return result
        except Exception as e:
            logger.error(f'Error getting due jobs: {str(e)}')
            return []

    @async_with_exception_handling
    async def _execute_job(self, job_id: str, job_info: Dict[str, Any]) ->None:
        """
        Execute a scheduled job.
        
        Args:
            job_id: ID of the job to execute
            job_info: Information about the job
        """
        semaphore_key = f'job_semaphore_{job_id}'
        if not hasattr(self, semaphore_key):
            setattr(self, semaphore_key, asyncio.Semaphore(1))
        job_semaphore = getattr(self, semaphore_key)
        if not job_semaphore.locked():
            async with job_semaphore:
                await self._update_job_status(job_id, 'running')
                history_id = await self._create_history_record(job_id)
                try:
                    config = job_info['config']
                    symbols = config_manager.get('symbols', [])
                    timeframes = config_manager.get('timeframes', [])
                    indicators = config_manager.get('indicators', [])
                    lookback_days = config_manager.get('lookback_days', 30)
                    end_date = datetime.utcnow()
                    start_date = end_date - timedelta(days=lookback_days)
                    indicator_ids = []
                    params_map = {}
                    for indicator in indicators:
                        if isinstance(indicator, dict):
                            indicator_id = indicator.get('id')
                            if not indicator_id:
                                continue
                            indicator_ids.append(indicator_id)
                            if 'params' in indicator:
                                params_map[indicator_id] = indicator['params']
                        else:
                            indicator_ids.append(indicator)
                    pairs = []
                    for symbol in symbols:
                        for timeframe in timeframes:
                            pairs.append((symbol, timeframe))
                    if pairs and indicator_ids:
                        logger.info(
                            f"Running job {job_id} ({job_info['name']}) for {len(pairs)} pairs and {len(indicator_ids)} indicators"
                            )
                        compute_semaphore = asyncio.Semaphore(5)
                        tasks = []
                        for symbol, timeframe in pairs:
                            task = self._compute_features_for_pair(
                                compute_semaphore, symbol, timeframe,
                                start_date, end_date, indicator_ids, params_map
                                )
                            tasks.append(task)
                        await asyncio.gather(*tasks)
                    next_run = datetime.utcnow() + timedelta(seconds=
                        job_info['interval_seconds'])
                    await self._update_job_after_run(job_id, next_run)
                    await self._update_history_record(history_id, True,
                        'completed', 'Job completed successfully')
                    logger.info(
                        f"Job {job_id} ({job_info['name']}) completed successfully"
                        )
                except Exception as e:
                    error_message = f'Error executing job: {str(e)}'
                    logger.error(error_message)
                    await self._update_job_status(job_id, 'error')
                    await self._update_history_record(history_id, False,
                        'failed', error_message)
        else:
            logger.warning(
                f'Job {job_id} is already running, skipping this execution')

    @async_with_exception_handling
    async def _compute_features_for_pair(self, semaphore: asyncio.Semaphore,
        symbol: str, timeframe: str, start_date: datetime, end_date:
        datetime, indicators: List[str], params_map: Dict[str, Any]) ->None:
        """
        Compute features for a symbol-timeframe pair with semaphore control.
        
        Args:
            semaphore: Semaphore to limit concurrent computations
            symbol: Symbol to compute features for
            timeframe: Timeframe to compute features for
            start_date: Start date for data
            end_date: End date for data
            indicators: List of indicator IDs
            params_map: Map of indicator IDs to parameters
        """
        async with semaphore:
            try:
                await self.computation_engine.compute_features_for_symbol(
                    symbol=symbol, timeframe=timeframe, start_date=
                    start_date, end_date=end_date, indicators=indicators,
                    params_map=params_map)
                logger.info(f'Computed features for {symbol} {timeframe}')
            except Exception as e:
                logger.error(
                    f'Error computing features for {symbol} {timeframe}: {str(e)}'
                    )

    @async_with_exception_handling
    async def _update_job_status(self, job_id: str, status: str) ->None:
        """
        Update a job's status.
        
        Args:
            job_id: ID of the job to update
            status: New status for the job
        """
        query = """
        UPDATE scheduler.jobs
        SET status = :status, updated_at = NOW()
        WHERE id = :job_id
        """
        try:
            async with self.engine.begin() as conn:
                await conn.execute(sa.text(query), {'job_id': job_id,
                    'status': status})
        except Exception as e:
            logger.error(f'Error updating job status: {str(e)}')

    @async_with_exception_handling
    async def _update_job_after_run(self, job_id: str, next_run: datetime
        ) ->None:
        """
        Update a job after it has been run.
        
        Args:
            job_id: ID of the job to update
            next_run: Next scheduled run time
        """
        query = """
        UPDATE scheduler.jobs
        SET status = 'pending', updated_at = NOW(), last_run = NOW(), next_run = :next_run
        WHERE id = :job_id
        """
        try:
            async with self.engine.begin() as conn:
                await conn.execute(sa.text(query), {'job_id': job_id,
                    'next_run': next_run})
        except Exception as e:
            logger.error(f'Error updating job after run: {str(e)}')

    @async_with_exception_handling
    async def _create_history_record(self, job_id: str) ->int:
        """
        Create a job history record.
        
        Args:
            job_id: ID of the job
            
        Returns:
            ID of the created history record
        """
        query = """
        INSERT INTO scheduler.job_history (job_id, start_time, status)
        VALUES (:job_id, NOW(), 'running')
        RETURNING id
        """
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(sa.text(query), {'job_id': job_id})
                row = result.fetchone()
                return row.id if row else 0
        except Exception as e:
            logger.error(f'Error creating history record: {str(e)}')
            return 0

    @async_with_exception_handling
    async def _update_history_record(self, history_id: int, success: bool,
        status: str, message: Optional[str]=None, details: Optional[Dict[
        str, Any]]=None) ->None:
        """
        Update a job history record.
        
        Args:
            history_id: ID of the history record
            success: Whether the job succeeded
            status: Status of the job
            message: Optional message
            details: Optional details
        """
        if history_id <= 0:
            return
        query = """
        UPDATE scheduler.job_history
        SET end_time = NOW(), status = :status, message = :message, details = :details
        WHERE id = :history_id
        """
        try:
            async with self.engine.begin() as conn:
                await conn.execute(sa.text(query), {'history_id':
                    history_id, 'status': status, 'message': message,
                    'details': json.dumps(details) if details else None})
        except Exception as e:
            logger.error(f'Error updating history record: {str(e)}')

    @async_with_exception_handling
    async def add_job(self, job_id: str, name: str, description: Optional[
        str], symbols: List[str], timeframes: List[str], indicators: List[
        Dict[str, Any]], lookback_days: int, interval_seconds: int, enabled:
        bool=True) ->Dict[str, Any]:
        """
        Add a new scheduled job.
        
        Args:
            job_id: Unique identifier for the job
            name: Name of the job
            description: Optional description
            symbols: List of symbols to compute indicators for
            timeframes: List of timeframes to compute indicators for
            indicators: List of indicators to compute
            lookback_days: Number of days of historical data to compute
            interval_seconds: Interval between job runs in seconds
            enabled: Whether the job is enabled
            
        Returns:
            Dictionary with job information
        """
        next_run = datetime.utcnow()
        config = {'symbols': symbols, 'timeframes': timeframes,
            'indicators': indicators, 'lookback_days': lookback_days}
        query = """
        INSERT INTO scheduler.jobs (
            id, name, description, config, interval_seconds, enabled, created_at, updated_at, next_run, status
        )
        VALUES (
            :job_id, :name, :description, :config, :interval_seconds, :enabled, NOW(), NOW(), :next_run, 'pending'
        )
        RETURNING id, name, description, config, interval_seconds, enabled, created_at, updated_at, last_run, next_run, status
        """
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(sa.text(query), {'job_id':
                    job_id, 'name': name, 'description': description,
                    'config': json.dumps(config), 'interval_seconds':
                    interval_seconds, 'enabled': enabled, 'next_run': next_run}
                    )
                row = result.fetchone()
                job_info = {'id': row.id, 'name': row.name, 'description':
                    row.description, 'symbols': config['symbols'],
                    'timeframes': config['timeframes'], 'indicators':
                    config['indicators'], 'lookback_days': config[
                    'lookback_days'], 'interval': {'seconds': row.
                    interval_seconds, 'formatted': self._format_interval(
                    row.interval_seconds)}, 'enabled': row.enabled,
                    'created_at': row.created_at, 'updated_at': row.
                    updated_at, 'last_run': row.last_run, 'next_run': row.
                    next_run, 'status': row.status}
                return job_info
        except Exception as e:
            logger.error(f'Error adding job: {str(e)}')
            raise

    @async_with_exception_handling
    async def update_job(self, job_id: str, name: str, description:
        Optional[str], symbols: List[str], timeframes: List[str],
        indicators: List[Dict[str, Any]], lookback_days: int,
        interval_seconds: int, enabled: bool=True) ->Optional[Dict[str, Any]]:
        """
        Update an existing scheduled job.
        
        Args:
            job_id: Unique identifier for the job
            name: Name of the job
            description: Optional description
            symbols: List of symbols to compute indicators for
            timeframes: List of timeframes to compute indicators for
            indicators: List of indicators to compute
            lookback_days: Number of days of historical data to compute
            interval_seconds: Interval between job runs in seconds
            enabled: Whether the job is enabled
            
        Returns:
            Dictionary with updated job information, or None if the job doesn't exist
        """
        config = {'symbols': symbols, 'timeframes': timeframes,
            'indicators': indicators, 'lookback_days': lookback_days}
        query = """
        UPDATE scheduler.jobs
        SET name = :name,
            description = :description,
            config = :config,
            interval_seconds = :interval_seconds,
            enabled = :enabled,
            updated_at = NOW()
        WHERE id = :job_id
        RETURNING id, name, description, config, interval_seconds, enabled, created_at, updated_at, last_run, next_run, status
        """
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(sa.text(query), {'job_id':
                    job_id, 'name': name, 'description': description,
                    'config': json.dumps(config), 'interval_seconds':
                    interval_seconds, 'enabled': enabled})
                row = result.fetchone()
                if not row:
                    return None
                job_config = json.loads(row.config) if isinstance(row.
                    config, str) else row.config
                job_info = {'id': row.id, 'name': row.name, 'description':
                    row.description, 'symbols': job_config.get('symbols', [
                    ]), 'timeframes': job_config_manager.get('timeframes', []),
                    'indicators': job_config_manager.get('indicators', []),
                    'lookback_days': job_config_manager.get('lookback_days', 30),
                    'interval': {'seconds': row.interval_seconds,
                    'formatted': self._format_interval(row.interval_seconds
                    )}, 'enabled': row.enabled, 'created_at': row.
                    created_at, 'updated_at': row.updated_at, 'last_run':
                    row.last_run, 'next_run': row.next_run, 'status': row.
                    status}
                return job_info
        except Exception as e:
            logger.error(f'Error updating job: {str(e)}')
            raise

    @async_with_exception_handling
    async def delete_job(self, job_id: str) ->bool:
        """
        Delete a scheduled job.
        
        Args:
            job_id: ID of the job to delete
            
        Returns:
            True if the job was deleted, False if it doesn't exist
        """
        query = 'DELETE FROM scheduler.jobs WHERE id = :job_id'
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(sa.text(query), {'job_id': job_id})
                return result.rowcount > 0
        except Exception as e:
            logger.error(f'Error deleting job: {str(e)}')
            return False

    @async_with_exception_handling
    async def enable_job(self, job_id: str) ->bool:
        """
        Enable a scheduled job.
        
        Args:
            job_id: ID of the job to enable
            
        Returns:
            True if the job was enabled, False if it doesn't exist
        """
        next_run = datetime.utcnow()
        query = """
        UPDATE scheduler.jobs
        SET enabled = TRUE, updated_at = NOW(), next_run = :next_run
        WHERE id = :job_id
        """
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(sa.text(query), {'job_id':
                    job_id, 'next_run': next_run})
                return result.rowcount > 0
        except Exception as e:
            logger.error(f'Error enabling job: {str(e)}')
            return False

    @async_with_exception_handling
    async def disable_job(self, job_id: str) ->bool:
        """
        Disable a scheduled job.
        
        Args:
            job_id: ID of the job to disable
            
        Returns:
            True if the job was disabled, False if it doesn't exist
        """
        query = """
        UPDATE scheduler.jobs
        SET enabled = FALSE, updated_at = NOW()
        WHERE id = :job_id
        """
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(sa.text(query), {'job_id': job_id})
                return result.rowcount > 0
        except Exception as e:
            logger.error(f'Error disabling job: {str(e)}')
            return False

    @async_with_exception_handling
    async def get_job(self, job_id: str) ->Optional[Dict[str, Any]]:
        """
        Get information about a specific job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Dictionary with job information, or None if the job doesn't exist
        """
        query = """
        SELECT id, name, description, config, interval_seconds, enabled, created_at, updated_at, last_run, next_run, status
        FROM scheduler.jobs
        WHERE id = :job_id
        """
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(sa.text(query), {'job_id': job_id})
                row = result.fetchone()
                if not row:
                    return None
                job_config = json.loads(row.config) if isinstance(row.
                    config, str) else row.config
                job_info = {'id': row.id, 'name': row.name, 'description':
                    row.description, 'symbols': job_config.get('symbols', [
                    ]), 'timeframes': job_config_manager.get('timeframes', []),
                    'indicators': job_config_manager.get('indicators', []),
                    'lookback_days': job_config_manager.get('lookback_days', 30),
                    'interval': {'seconds': row.interval_seconds,
                    'formatted': self._format_interval(row.interval_seconds
                    )}, 'enabled': row.enabled, 'created_at': row.
                    created_at, 'updated_at': row.updated_at, 'last_run':
                    row.last_run, 'next_run': row.next_run, 'status': row.
                    status}
                return job_info
        except Exception as e:
            logger.error(f'Error getting job: {str(e)}')
            return None

    @async_with_exception_handling
    async def get_all_jobs(self) ->List[Dict[str, Any]]:
        """
        Get information about all jobs.
        
        Returns:
            List of dictionaries with job information
        """
        query = """
        SELECT id, name, description, config, interval_seconds, enabled, created_at, updated_at, last_run, next_run, status
        FROM scheduler.jobs
        ORDER BY created_at DESC
        """
        try:
            result = []
            async with self.engine.begin() as conn:
                rows = await conn.execute(sa.text(query))
                for row in rows:
                    job_config = json.loads(row.config) if isinstance(row.
                        config, str) else row.config
                    job_info = {'id': row.id, 'name': row.name,
                        'description': row.description, 'symbols':
                        job_config_manager.get('symbols', []), 'timeframes':
                        job_config_manager.get('timeframes', []), 'indicators':
                        job_config_manager.get('indicators', []), 'lookback_days':
                        job_config_manager.get('lookback_days', 30), 'interval': {
                        'seconds': row.interval_seconds, 'formatted': self.
                        _format_interval(row.interval_seconds)}, 'enabled':
                        row.enabled, 'created_at': row.created_at,
                        'updated_at': row.updated_at, 'last_run': row.
                        last_run, 'next_run': row.next_run, 'status': row.
                        status}
                    result.append(job_info)
            return result
        except Exception as e:
            logger.error(f'Error getting all jobs: {str(e)}')
            return []

    @async_with_exception_handling
    async def get_job_history(self, job_id: str, limit: int=10) ->List[Dict
        [str, Any]]:
        """
        Get the execution history for a specific job.
        
        Args:
            job_id: ID of the job
            limit: Maximum number of history records to return
            
        Returns:
            List of dictionaries with job history information
        """
        query = """
        SELECT id, job_id, start_time, end_time, status, message, details
        FROM scheduler.job_history
        WHERE job_id = :job_id
        ORDER BY start_time DESC
        LIMIT :limit
        """
        try:
            result = []
            async with self.engine.begin() as conn:
                rows = await conn.execute(sa.text(query), {'job_id': job_id,
                    'limit': limit})
                for row in rows:
                    details = json.loads(row.details) if row.details else None
                    history_info = {'id': row.id, 'job_id': row.job_id,
                        'start_time': row.start_time, 'end_time': row.
                        end_time, 'status': row.status, 'message': row.
                        message, 'details': details, 'duration_seconds': (
                        row.end_time - row.start_time).total_seconds() if
                        row.end_time else None}
                    result.append(history_info)
            return result
        except Exception as e:
            logger.error(f'Error getting job history: {str(e)}')
            return []

    @async_with_exception_handling
    async def run_job_now(self, job_id: str) ->bool:
        """
        Run a job immediately.
        
        Args:
            job_id: ID of the job to run
            
        Returns:
            True if the job was triggered, False if it doesn't exist
        """
        try:
            job_info = await self.get_job(job_id)
            if not job_info:
                logger.warning(f'Job {job_id} not found')
                return False
            await self._execute_job(job_id, job_info)
            return True
        except Exception as e:
            logger.error(f'Error running job {job_id}: {str(e)}')
            return False

    def _format_interval(self, seconds: int) ->str:
        """
        Format an interval in seconds to a human-readable string.
        
        Args:
            seconds: Interval in seconds
            
        Returns:
            Human-readable interval string
        """
        if seconds < 60:
            return f"{seconds} second{'s' if seconds != 1 else ''}"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        elif seconds < 86400:
            hours = seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''}"
        else:
            days = seconds // 86400
            return f"{days} day{'s' if days != 1 else ''}"
