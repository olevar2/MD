"""
Feedback Loop Connector

This module implements the bidirectional connection between the strategy execution engine
and the adaptive layer, ensuring feedback flows smoothly in both directions.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx
from core_foundations.utils.logger import get_logger
from core_foundations.models.feedback import TradeFeedback, FeedbackStatus
from core_foundations.events.event_publisher import EventPublisher
from analysis_engine.adaptive_layer.feedback_loop import FeedbackLoop
from analysis_engine.adaptive_layer.trading_feedback_collector import TradingFeedbackCollector
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class FeedbackLoopConnector:
    """
    Implements bidirectional connection between strategy execution and adaptive layer.
    
    This connector ensures that:
    1. Feedback from strategy execution is properly collected and processed
    2. Adaptations from the adaptive layer are properly applied to strategy execution
    3. The loop is properly monitored and maintained
    """

    def __init__(self, feedback_loop: FeedbackLoop,
        trading_feedback_collector: TradingFeedbackCollector,
        strategy_execution_api_url: str, event_publisher: Optional[
        EventPublisher]=None, config: Dict[str, Any]=None):
        """
        Initialize the feedback loop connector.
        
        Args:
            feedback_loop: The feedback loop instance
            trading_feedback_collector: The trading feedback collector instance
            strategy_execution_api_url: URL to the strategy execution API
            event_publisher: Optional event publisher for monitoring events
            config: Configuration dictionary
        """
        self.feedback_loop = feedback_loop
        self.trading_feedback_collector = trading_feedback_collector
        self.strategy_execution_api_url = strategy_execution_api_url
        self.event_publisher = event_publisher
        self.config = config or {}
        self.http_client = None
        self.last_feedback_time = None
        self.last_adaptation_time = None
        self.feedback_count = 0
        self.adaptation_count = 0
        self.connection_errors = 0
        self._monitoring_task = None
        self._is_running = False
        logger.info(
            'FeedbackLoopConnector initialized with strategy execution API at %s'
            , strategy_execution_api_url)

    async def start(self):
        """Start the connector operations."""
        if self._is_running:
            logger.warning('FeedbackLoopConnector already running')
            return
        self._is_running = True
        self.http_client = httpx.AsyncClient(timeout=self.config.get(
            'http_timeout', 10.0), limits=httpx.Limits(
            max_keepalive_connections=self.config.get('max_connections', 10
            ), max_connections=self.config_manager.get('max_connections', 10)))
        if self.config_manager.get('enable_monitoring', True):
            self._monitoring_task = asyncio.create_task(self.
                _monitor_loop_health())
        logger.info('FeedbackLoopConnector started')

    @async_with_exception_handling
    async def stop(self):
        """Stop the connector operations."""
        if not self._is_running:
            return
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        logger.info('FeedbackLoopConnector stopped')

    @with_resilience('send_adaptation_to_strategy_execution')
    @async_with_exception_handling
    async def send_adaptation_to_strategy_execution(self, adaptation: Dict[
        str, Any]) ->bool:
        """
        Send an adaptation from the adaptive layer to strategy execution.
        
        Args:
            adaptation: The adaptation data to send
            
        Returns:
            bool: Success status
        """
        if not self._is_running or not self.http_client:
            logger.error('Cannot send adaptation - connector not running')
            return False
        strategy_id = adaptation.get('strategy_id')
        if not strategy_id:
            logger.error('Cannot send adaptation - missing strategy ID')
            return False
        try:
            endpoint = (
                f'{self.strategy_execution_api_url}/api/v1/strategies/{strategy_id}/adapt'
                )
            response = await self.http_client.post(endpoint, json=adaptation)
            response.raise_for_status()
            self.last_adaptation_time = datetime.utcnow()
            self.adaptation_count += 1
            if self.event_publisher:
                await self.event_publisher.publish('feedback.adaptation.sent',
                    {'strategy_id': strategy_id, 'adaptation_id':
                    adaptation.get('adaptation_id'), 'timestamp': self.
                    last_adaptation_time.isoformat(), 'success': True})
            logger.debug('Successfully sent adaptation to strategy %s',
                strategy_id)
            return True
        except Exception as e:
            self.connection_errors += 1
            logger.error('Failed to send adaptation to strategy %s: %s',
                strategy_id, str(e))
            if self.event_publisher:
                await self.event_publisher.publish('feedback.adaptation.error',
                    {'strategy_id': strategy_id, 'adaptation_id':
                    adaptation.get('adaptation_id'), 'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()})
            return False

    @with_database_resilience('process_execution_feedback')
    @async_with_exception_handling
    async def process_execution_feedback(self, feedback_data: Dict[str, Any]
        ) ->str:
        """
        Process feedback from strategy execution.
        
        Args:
            feedback_data: The feedback data from strategy execution
            
        Returns:
            str: Feedback ID if successful, empty string otherwise
        """
        try:
            feedback = TradeFeedback(id=feedback_data.get('feedback_id') or
                '', strategy_id=feedback_data.get('strategy_id'), model_id=
                feedback_data.get('model_id'), instrument=feedback_data.get
                ('instrument'), timeframe=feedback_data.get('timeframe'),
                source=feedback_data.get('source'), category=feedback_data.
                get('category'), status=FeedbackStatus.NEW, outcome_metrics
                =feedback_data.get('metrics') or {}, metadata=feedback_data
                .get('metadata') or {}, timestamp=feedback_data.get(
                'timestamp') or datetime.utcnow().isoformat())
            feedback_id = (await self.trading_feedback_collector.
                collect_feedback(feedback))
            self.last_feedback_time = datetime.utcnow()
            self.feedback_count += 1
            return feedback_id
        except Exception as e:
            logger.error('Error processing execution feedback: %s', str(e))
            return ''

    @with_resilience('get_loop_health')
    async def get_loop_health(self) ->Dict[str, Any]:
        """
        Get health metrics for the feedback loop.
        
        Returns:
            Dict containing health metrics
        """
        now = datetime.utcnow()
        time_since_feedback = None
        if self.last_feedback_time:
            time_since_feedback = (now - self.last_feedback_time
                ).total_seconds()
        time_since_adaptation = None
        if self.last_adaptation_time:
            time_since_adaptation = (now - self.last_adaptation_time
                ).total_seconds()
        return {'is_running': self._is_running, 'feedback_count': self.
            feedback_count, 'adaptation_count': self.adaptation_count,
            'connection_errors': self.connection_errors,
            'last_feedback_time': self.last_feedback_time.isoformat() if
            self.last_feedback_time else None, 'last_adaptation_time': self
            .last_adaptation_time.isoformat() if self.last_adaptation_time else
            None, 'seconds_since_last_feedback': time_since_feedback,
            'seconds_since_last_adaptation': time_since_adaptation,
            'strategy_execution_api': self.strategy_execution_api_url}

    @async_with_exception_handling
    async def _monitor_loop_health(self):
        """Background task to monitor the health of the feedback loop."""
        check_interval = self.config_manager.get('monitoring_interval_seconds', 300)
        alert_threshold = self.config.get('feedback_alert_threshold_seconds',
            3600)
        while self._is_running:
            try:
                now = datetime.utcnow()
                if self.last_feedback_time and (now - self.last_feedback_time
                    ).total_seconds() > alert_threshold:
                    logger.warning('No feedback received in %d seconds', (
                        now - self.last_feedback_time).total_seconds())
                    if self.event_publisher:
                        await self.event_publisher.publish(
                            'feedback.loop.alert', {'alert_type':
                            'no_recent_feedback', 'seconds_since_last': (
                            now - self.last_feedback_time).total_seconds(),
                            'threshold': alert_threshold, 'timestamp': now.
                            isoformat()})
                if self.event_publisher and self.config.get(
                    'publish_health_metrics', True):
                    await self.event_publisher.publish('feedback.loop.health',
                        {'feedback_count': self.feedback_count,
                        'adaptation_count': self.adaptation_count,
                        'connection_errors': self.connection_errors,
                        'is_healthy': self._is_running and self.
                        connection_errors < self.config.get(
                        'max_connection_errors', 10), 'timestamp': now.
                        isoformat()})
            except Exception as e:
                logger.error('Error in feedback loop health monitor: %s',
                    str(e))
            await asyncio.sleep(check_interval)
