"""
Standalone test script for async patterns.

This script tests the async patterns without depending on pytest or the full application.
"""
import asyncio
import logging
from datetime import datetime, timedelta
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MockScheduler:
    """Mock scheduler for testing async patterns."""

    def __init__(self):
        self.running = False
        self.task = None
        self.scheduled_tasks = {}
        self.logger = logging.getLogger(__name__)

    async def start(self):
        """Start the scheduler."""
        if self.task and not self.task.done():
            self.logger.warning('Scheduler already running')
            return False
        self.running = True
        self.scheduled_tasks = {'hourly': {'interval': 60 * 60, 'next_run':
            self._next_hour(), 'func': self.hourly_task}, 'daily': {
            'interval': 24 * 60 * 60, 'next_run': self._next_time(0, 30),
            'func': self.daily_task}}
        self.task = asyncio.create_task(self._run_scheduler())
        self.logger.info('Scheduler started')
        return True

    @async_with_exception_handling
    async def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.logger.info('Scheduler stopped')

    @async_with_exception_handling
    async def _run_scheduler(self):
        """Run the scheduler loop."""
        try:
            while self.running:
                now = datetime.now().timestamp()
                for task_name, task_info in self.scheduled_tasks.items():
                    if now >= task_info['next_run']:
                        self.logger.info(f'Running scheduled task: {task_name}'
                            )
                        asyncio.create_task(task_info['func']())
                        task_info['next_run'] = now + task_info['interval']
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            self.logger.info('Scheduler task cancelled')
            raise
        except Exception as e:
            self.logger.error(f'Error in scheduler loop: {e}')

    def _next_hour(self):
        """Get timestamp for the start of the next hour."""
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(
            hours=1)
        return next_hour.timestamp()

    def _next_time(self, hour, minute):
        """Get timestamp for the next occurrence of a specific time."""
        now = datetime.now()
        target_time = now.replace(hour=hour, minute=minute, second=0,
            microsecond=0)
        if target_time <= now:
            target_time += timedelta(days=1)
        return target_time.timestamp()

    async def hourly_task(self):
        """Mock hourly task."""
        self.logger.info('Running hourly task')
        await asyncio.sleep(0.1)
        self.logger.info('Hourly task completed')

    async def daily_task(self):
        """Mock daily task."""
        self.logger.info('Running daily task')
        await asyncio.sleep(0.1)
        self.logger.info('Daily task completed')


class MockService:
    """Mock service for testing async patterns."""

    def __init__(self, name):
        self.name = name
        self.initialized = False
        self.cleaned_up = False
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the service."""
        self.logger.info(f'Initializing service: {self.name}')
        await asyncio.sleep(0.1)
        self.initialized = True
        self.logger.info(f'Service initialized: {self.name}')

    async def cleanup(self):
        """Clean up the service."""
        self.logger.info(f'Cleaning up service: {self.name}')
        await asyncio.sleep(0.1)
        self.cleaned_up = True
        self.logger.info(f'Service cleaned up: {self.name}')


class MockContainer:
    """Mock container for testing async patterns."""

    def __init__(self):
        self.services = {}
        self.logger = logging.getLogger(__name__)

    def register_service(self, name, service):
        """Register a service."""
        self.services[name] = service
        self.logger.info(f'Registered service: {name}')

    @with_resilience('get_service')
    def get_service(self, name):
        """Get a service by name."""
        return self.services.get(name)

    async def initialize(self):
        """Initialize all services."""
        self.logger.info('Initializing container')
        for name, service in self.services.items():
            if hasattr(service, 'initialize'):
                await service.initialize()
        self.logger.info('Container initialized')

    async def cleanup(self):
        """Clean up all services."""
        self.logger.info('Cleaning up container')
        for name, service in self.services.items():
            if hasattr(service, 'cleanup'):
                await service.cleanup()
        self.logger.info('Container cleaned up')


async def test_scheduler():
    """Test the scheduler."""
    logger.info('Testing scheduler')
    scheduler = MockScheduler()
    await scheduler.start()
    assert scheduler.running
    assert scheduler.task is not None
    logger.info('Waiting for tasks to run')
    await asyncio.sleep(0.5)
    await scheduler.stop()
    assert not scheduler.running
    assert scheduler.task.done()
    logger.info('Scheduler test completed')


async def test_container():
    """Test the container."""
    logger.info('Testing container')
    container = MockContainer()
    service1 = MockService('service1')
    service2 = MockService('service2')
    container.register_service('service1', service1)
    container.register_service('service2', service2)
    await container.initialize()
    assert service1.initialized
    assert service2.initialized
    await container.cleanup()
    assert service1.cleaned_up
    assert service2.cleaned_up
    logger.info('Container test completed')


async def test_scheduler_container_integration():
    """Test the integration of scheduler and container."""
    logger.info('Testing scheduler-container integration')
    container = MockContainer()
    scheduler = MockScheduler()
    container.register_service('scheduler', scheduler)
    await container.initialize()
    await scheduler.start()
    logger.info('Waiting for tasks to run')
    await asyncio.sleep(0.5)
    await scheduler.stop()
    await container.cleanup()
    logger.info('Scheduler-container integration test completed')


async def main():
    """Run all tests."""
    logger.info('Starting tests')
    await test_scheduler()
    await test_container()
    await test_scheduler_container_integration()
    logger.info('All tests completed successfully')


if __name__ == '__main__':
    asyncio.run(main())
