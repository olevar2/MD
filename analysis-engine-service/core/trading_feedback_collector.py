"""
Trading Feedback Collector

This module implements the TradingFeedbackCollector class that captures trading
execution outcomes and feeds them into the feedback loop system.
"""
from typing import Dict, List, Any, Optional, Set, Union
import logging
from datetime import datetime, timedelta, timezone
import asyncio
import uuid
import httpx
import random
from core_foundations.models.feedback import TradeFeedback, FeedbackSource, FeedbackCategory, FeedbackStatus
from core_foundations.utils.logger import get_logger
from core_foundations.events.event_publisher import EventPublisher
from core_foundations.events.event_schema import Event, EventType
from core_foundations.exceptions.feedback_exceptions import FeedbackCollectionError
from core_foundations.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerState
from core_foundations.resilience.service_registry import ServiceRegistryClient
from common_lib.ml.model_feedback_interfaces import ModelFeedback, IModelFeedbackProcessor, IModelTrainingFeedbackIntegrator
from analysis_engine.adapters.model_feedback_adapter import ModelTrainingFeedbackAdapter
logger = get_logger(__name__)
CONFIG_HTTP_TIMEOUT = 'http_timeout'
CONFIG_CB_FAILURE_THRESHOLD = 'circuit_breaker_failure_threshold'
CONFIG_CB_RECOVERY_TIMEOUT = 'circuit_breaker_recovery_timeout'
CONFIG_CB_SUCCESS_THRESHOLD = 'circuit_breaker_success_threshold'
CONFIG_ENABLE_BATCH = 'enable_batch_processing'
CONFIG_BATCH_INTERVAL = 'batch_processing_interval'
CONFIG_RECENT_LIMIT = 'recent_feedback_limit'
CONFIG_ORCH_RETRY_ATTEMPTS = 'orchestration_retry_attempts'
CONFIG_ORCH_RETRY_DELAY = 'orchestration_retry_delay'
CONFIG_ENABLE_ORCHESTRATION = 'enable_orchestration'
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class TradingFeedbackCollector:
    """
    The TradingFeedbackCollector captures and processes trading execution results
    to feed into the feedback loop system.

    Key capabilities:
    - Collect trade execution feedback
    - Store and retrieve historical feedback
    - Aggregate related feedback items
    - Support batch processing for efficient analysis
    - Provide statistics and metrics on collected feedback
    - Integrate with the orchestration service for system-wide feedback processing
    """

    def __init__(self, feedback_loop: Any=None, event_publisher: Optional[
        EventPublisher]=None, config: Dict[str, Any]=None,
        orchestration_service_url: Optional[str]=None,
        service_registry_client: Optional[ServiceRegistryClient]=None,
        model_training_feedback: Optional[ModelTrainingFeedbackAdapter]=None):
        """
        Initialize the TradingFeedbackCollector.

        Args:
            feedback_loop: The feedback loop to send processed feedback to
            event_publisher: Event publisher for broadcasting feedback events
            config: Configuration parameters
            orchestration_service_url: URL to connect to the orchestration service
            service_registry_client: Client for service discovery and registration
        """
        self.feedback_loop = feedback_loop
        self.event_publisher = event_publisher
        self.config = config or {}
        self.orchestration_service_url = orchestration_service_url
        self.service_registry_client = service_registry_client
        self.model_training_feedback = model_training_feedback
        self.http_client = httpx.AsyncClient(timeout=self.config.get(
            CONFIG_HTTP_TIMEOUT, 10.0))
        self._set_default_config()
        self.recent_feedback: Dict[str, TradeFeedback] = {}
        self.stats = {'total_collected': 0, 'by_source': {}, 'by_category':
            {}, 'by_instrument': {}, 'by_strategy': {}, 'by_model': {}}
        self._batch_task = None
        self._is_running = False
        self.orchestration_circuit_breaker = CircuitBreaker(name=
            'orchestration_service', failure_threshold=self.config.get(
            CONFIG_CB_FAILURE_THRESHOLD, 5), recovery_timeout=self.config.
            get(CONFIG_CB_RECOVERY_TIMEOUT, 30),
            half_open_success_threshold=self.config.get(
            CONFIG_CB_SUCCESS_THRESHOLD, 3))
        logger.info('TradingFeedbackCollector initialized')

    def _set_default_config(self):
        """Set default configuration parameters."""
        defaults = {CONFIG_ENABLE_BATCH: True, CONFIG_BATCH_INTERVAL: 3600,
            CONFIG_RECENT_LIMIT: 1000, 'model_error_threshold': 0.05,
            CONFIG_ORCH_RETRY_ATTEMPTS: 3, CONFIG_ORCH_RETRY_DELAY: 1.0,
            'orchestration_timeout': 10.0, CONFIG_CB_FAILURE_THRESHOLD: 5,
            CONFIG_CB_RECOVERY_TIMEOUT: 30, CONFIG_CB_SUCCESS_THRESHOLD: 3,
            CONFIG_ENABLE_ORCHESTRATION: True}
        self.config.update({k: v for k, v in defaults.items() if k not in
            self.config})

    async def start(self):
        """Start the feedback collector service."""
        self._is_running = True
        if self.config[CONFIG_ENABLE_BATCH]:
            self._batch_task = asyncio.create_task(self.
                _batch_processing_loop())
            logger.info('Batch processing started')
        if self.config[CONFIG_ENABLE_ORCHESTRATION]:
            await self._try_register_with_orchestration()

    @async_with_exception_handling
    async def _try_register_with_orchestration(self):
        """Attempt to register with the orchestration service."""
        try:
            await self._register_with_orchestration()
            logger.info('Successfully registered with orchestration service')
        except Exception as e:
            logger.warning(
                f'Failed to register with orchestration service: {e}. Operating standalone.'
                )

    async def stop(self):
        """Stop the feedback collector service."""
        self._is_running = False
        await self._cancel_batch_task()
        if self.config[CONFIG_ENABLE_ORCHESTRATION]:
            await self._try_unregister_from_orchestration()
        await self.http_client.aclose()
        logger.info('TradingFeedbackCollector stopped')

    @async_with_exception_handling
    async def _cancel_batch_task(self):
        """Cancel the background batch processing task if running."""
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                logger.info('Batch processing task cancelled.')
            except Exception as e:
                logger.error(f'Error during batch task cancellation: {e}',
                    exc_info=True)

    @async_with_exception_handling
    async def _try_unregister_from_orchestration(self):
        """Attempt to unregister from the orchestration service."""
        try:
            await self._unregister_from_orchestration()
            logger.info('Successfully unregistered from orchestration service')
        except Exception as e:
            logger.warning(
                f'Failed to unregister from orchestration service: {e}')

    @async_with_exception_handling
    async def collect_feedback(self, feedback: TradeFeedback) ->str:
        """Collect, store, publish, and process/queue feedback."""
        try:
            self._ensure_feedback_metadata(feedback)
            self._store_recent_feedback(feedback)
            self._update_statistics(feedback)
            await self._publish_feedback_collected_event(feedback)
            await self._route_or_queue_feedback(feedback)
            return feedback.id
        except Exception as e:
            logger.error(f'Error collecting feedback: {e}', exc_info=True)
            raise FeedbackCollectionError(f'Failed to collect feedback: {e}')

    def _ensure_feedback_metadata(self, feedback: TradeFeedback):
        """Ensure feedback has an ID and timestamp."""
        if not feedback.id:
            feedback.id = str(uuid.uuid4())
        if not feedback.timestamp:
            feedback.timestamp = datetime.now(timezone.utc).isoformat()

    def _store_recent_feedback(self, feedback: TradeFeedback):
        """Store feedback in memory and trim if necessary."""
        self.recent_feedback[feedback.id] = feedback
        if len(self.recent_feedback) > self.config[CONFIG_RECENT_LIMIT]:
            self._trim_recent_feedback()

    def _trim_recent_feedback(self):
        """Remove the oldest feedback items to stay within the limit."""
        limit = self.config[CONFIG_RECENT_LIMIT]
        if len(self.recent_feedback) > limit:
            sorted_keys = sorted(self.recent_feedback, key=lambda k: self.
                recent_feedback[k].timestamp)
            keys_to_remove = sorted_keys[:-limit]
            for key in keys_to_remove:
                if key in self.recent_feedback:
                    del self.recent_feedback[key]
            logger.debug(
                f'Trimmed recent feedback cache. Removed {len(keys_to_remove)} items.'
                )

    @async_with_exception_handling
    async def _publish_feedback_collected_event(self, feedback: TradeFeedback):
        """Publish a FEEDBACK_COLLECTED event."""
        if not self.event_publisher:
            return
        try:
            event_payload = self._create_event_payload(feedback)
            await self.event_publisher.publish(EventType.FEEDBACK_COLLECTED,
                event_payload)
            logger.debug(
                f'Published FEEDBACK_COLLECTED event for {feedback.id}')
        except Exception as e:
            logger.error(
                f'Failed to publish FEEDBACK_COLLECTED event for {feedback.id}: {e}'
                , exc_info=True)

    def _create_event_payload(self, feedback: TradeFeedback) ->Dict[str, Any]:
        """Create the payload for the feedback collected event."""
        return {'feedback_id': feedback.id, 'source': feedback.source.value,
            'category': feedback.category.value, 'timestamp': feedback.
            timestamp, 'instrument': getattr(feedback, 'instrument', None),
            'strategy_id': getattr(feedback, 'strategy_id', None),
            'model_id': getattr(feedback, 'model_id', None)}

    @with_exception_handling
    def _convert_to_model_feedback(self, feedback: TradeFeedback) ->Optional[
        ModelFeedback]:
        """
        Convert TradeFeedback to ModelFeedback for the model training feedback adapter.

        Args:
            feedback: TradeFeedback object

        Returns:
            ModelFeedback object or None if conversion fails
        """
        try:
            source_mapping = {FeedbackSource.MODEL_PREDICTION:
                'MODEL_PREDICTION', FeedbackSource.STRATEGY_EXECUTION:
                'TRADING', FeedbackSource.MARKET_CONDITION:
                'MARKET_CONDITION', FeedbackSource.TECHNICAL_ERROR:
                'SYSTEM', FeedbackSource.USER_FEEDBACK: 'MANUAL'}
            category_mapping = {FeedbackCategory.SUCCESS: 'ACCURACY',
                FeedbackCategory.PARTIAL_SUCCESS: 'ACCURACY',
                FeedbackCategory.FAILURE: 'ACCURACY', FeedbackCategory.
                MARKET_CONDITION: 'REGIME_CHANGE', FeedbackCategory.
                TECHNICAL_ERROR: 'LATENCY'}
            severity_mapping = {FeedbackCategory.SUCCESS: 'LOW',
                FeedbackCategory.PARTIAL_SUCCESS: 'MEDIUM',
                FeedbackCategory.FAILURE: 'HIGH', FeedbackCategory.
                MARKET_CONDITION: 'MEDIUM', FeedbackCategory.
                TECHNICAL_ERROR: 'HIGH'}
            source = source_mapping.get(feedback.source, 'TRADING')
            category = category_mapping.get(feedback.category, 'ACCURACY')
            severity = severity_mapping.get(feedback.category, 'MEDIUM')
            model_id = getattr(feedback, 'model_id', None)
            if not model_id:
                return None
            metrics = {}
            if hasattr(feedback, 'outcome_metrics'
                ) and feedback.outcome_metrics:
                metrics = feedback.outcome_metrics
            context = {'feedback_id': feedback.id, 'source': feedback.
                source.value, 'category': feedback.category.value}
            if hasattr(feedback, 'instrument') and feedback.instrument:
                context['instrument'] = feedback.instrument
            if hasattr(feedback, 'timeframe') and feedback.timeframe:
                context['timeframe'] = feedback.timeframe
            if hasattr(feedback, 'strategy_id') and feedback.strategy_id:
                context['strategy_id'] = feedback.strategy_id
            if hasattr(feedback, 'metadata') and feedback.metadata:
                context.update(feedback.metadata)
            from common_lib.ml.model_feedback_interfaces import ModelFeedback
            model_feedback = ModelFeedback(model_id=model_id, timestamp=
                datetime.now(), source=source, category=category, severity=
                severity, description=
                f'Feedback from trading: {feedback.category.value}',
                metrics=metrics, context=context, feedback_id=feedback.id)
            return model_feedback
        except Exception as e:
            logger.error(
                f'Error converting TradeFeedback to ModelFeedback: {e}')
            return None

    async def _route_or_queue_feedback(self, feedback: TradeFeedback):
        """Process feedback immediately or queue it for batch processing."""
        if not self.config[CONFIG_ENABLE_BATCH]:
            await self._process_feedback_realtime(feedback)
        else:
            feedback.status = FeedbackStatus.PENDING
            logger.debug(f'Feedback {feedback.id} queued for batch processing.'
                )

    @async_with_exception_handling
    async def _process_feedback_realtime(self, feedback: TradeFeedback):
        """Process feedback immediately (real-time)."""
        logger.debug(f'Real-time processing for feedback {feedback.id}')
        processed_locally = await self._try_send_to_local_loop(feedback)
        if hasattr(feedback, 'model_id'
            ) and feedback.model_id and self.model_training_feedback:
            try:
                model_feedback = self._convert_to_model_feedback(feedback)
                if model_feedback:
                    await self.model_training_feedback.process_trading_feedback(
                        [model_feedback])
                    logger.debug(
                        f'Forwarded feedback {feedback.id} to model training feedback adapter'
                        )
            except Exception as e:
                logger.error(
                    f'Error forwarding feedback {feedback.id} to model training: {e}'
                    )
        sent_to_orchestration = await self._try_send_to_orchestration(feedback)
        self._update_feedback_status_after_processing(feedback,
            processed_locally, sent_to_orchestration)

    @async_with_exception_handling
    async def _try_send_to_local_loop(self, feedback: TradeFeedback) ->bool:
        """Attempt to send feedback to the local feedback loop."""
        if not self.feedback_loop:
            return False
        try:
            await self.feedback_loop.add_feedback(feedback)
            feedback.status = FeedbackStatus.PROCESSING
            logger.debug(f'Sent feedback {feedback.id} to local FeedbackLoop')
            return True
        except Exception as e:
            logger.error(
                f'Error sending feedback {feedback.id} to local FeedbackLoop: {e}'
                , exc_info=True)
            feedback.status = FeedbackStatus.ERROR
            return False

    @async_with_exception_handling
    async def _try_send_to_orchestration(self, feedback: TradeFeedback) ->bool:
        """Attempt to send feedback to the orchestration service."""
        if not self.config[CONFIG_ENABLE_ORCHESTRATION]:
            return False
        try:
            sent = await self._send_feedback_to_orchestration(feedback)
            if sent:
                logger.debug(
                    f'Successfully sent feedback {feedback.id} to orchestration service'
                    )
            return sent
        except Exception as e:
            logger.error(
                f'Error sending feedback {feedback.id} to orchestration: {e}',
                exc_info=True)
            return False

    def _update_feedback_status_after_processing(self, feedback:
        TradeFeedback, processed_locally: bool, sent_to_orchestration: bool):
        """Update feedback status based on processing outcomes."""
        if feedback.status == FeedbackStatus.ERROR:
            return
        if processed_locally or sent_to_orchestration:
            feedback.status = FeedbackStatus.PROCESSED
        else:
            feedback.status = FeedbackStatus.PENDING
            logger.warning(
                f'Feedback {feedback.id} failed both local and orchestration sending.'
                )

    @async_with_exception_handling
    async def _batch_processing_loop(self):
        """Background task for batch processing of feedback."""
        interval = self.config[CONFIG_BATCH_INTERVAL]
        try:
            while self._is_running:
                await asyncio.sleep(interval)
                await self._process_feedback_batch()
        except asyncio.CancelledError:
            logger.info('Batch processing loop canceled')
        except Exception as e:
            logger.error(f'Error in batch processing loop: {e}', exc_info=True)

    async def _process_feedback_batch(self):
        """Process a batch of accumulated pending feedback."""
        if not self._should_process_batch():
            return
        pending_items = self._get_pending_feedback()
        if not pending_items:
            logger.debug('No pending feedback items in batch.')
            return
        logger.info(
            f'Processing batch of {len(pending_items)} pending feedback items')
        grouped = self._group_feedback_by_source(pending_items)
        processed_count, failed_count = await self._process_batch_groups(
            grouped)
        logger.info(
            f'Batch processing finished. Processed: {processed_count}, Failed/Pending Retry: {failed_count}'
            )
        await self._publish_batch_processed_event(processed_count,
            failed_count, grouped)

    def _should_process_batch(self) ->bool:
        """Check if conditions are met to process a batch."""
        return (self.feedback_loop or self.config[CONFIG_ENABLE_ORCHESTRATION]
            ) and self.recent_feedback

    def _get_pending_feedback(self) ->List[TradeFeedback]:
        """Retrieve feedback items marked as PENDING."""
        return [fb for fb in self.recent_feedback.values() if fb.status ==
            FeedbackStatus.PENDING]

    def _group_feedback_by_source(self, feedback_items: List[TradeFeedback]
        ) ->Dict[str, List[TradeFeedback]]:
        """Group feedback items by source."""
        grouped = {}
        for fb in feedback_items:
            source = fb.source.value
            grouped.setdefault(source, []).append(fb)
        return grouped

    async def _process_batch_groups(self, grouped_feedback: Dict[str, List[
        TradeFeedback]]) ->Tuple[int, int]:
        """Process the batch, group by group, returning counts."""
        total_processed = 0
        total_failed = 0
        for source, items in grouped_feedback.items():
            processed_in_group, failed_in_group = (await self.
                _process_single_group(source, items))
            total_processed += processed_in_group
            total_failed += failed_in_group
        return total_processed, total_failed

    @async_with_exception_handling
    async def _process_single_group(self, source: str, items: List[
        TradeFeedback]) ->Tuple[int, int]:
        """Process a single group of feedback items for a specific source."""
        processed_ids = set()
        failed_ids = set()
        await self._send_batch_to_local_loop(items, failed_ids)
        if self.model_training_feedback:
            model_feedback_list = []
            for fb in items:
                if hasattr(fb, 'model_id'
                    ) and fb.model_id and fb.id not in failed_ids:
                    model_feedback = self._convert_to_model_feedback(fb)
                    if model_feedback:
                        model_feedback_list.append(model_feedback)
            if model_feedback_list:
                try:
                    await self.model_training_feedback.process_trading_feedback(
                        model_feedback_list)
                    logger.debug(
                        f'Batch: Forwarded {len(model_feedback_list)} model feedback items to model training feedback adapter'
                        )
                except Exception as e:
                    logger.error(
                        f'Batch: Error forwarding model feedback to model training: {e}'
                        )
        if self.config[CONFIG_ENABLE_ORCHESTRATION]:
            items_for_orchestration = [fb for fb in items if fb.id not in
                failed_ids]
            if items_for_orchestration:
                sent_to_orch = await self._try_send_batch_to_orchestration(
                    source, items_for_orchestration)
                if sent_to_orch:
                    for fb in items_for_orchestration:
                        fb.status = FeedbackStatus.PROCESSED
                        processed_ids.add(fb.id)
                else:
                    for fb in items_for_orchestration:
                        fb.status = FeedbackStatus.PENDING
                        failed_ids.add(fb.id)
            else:
                logger.debug(
                    f'Batch: No items left to send to orchestration for source {source} after local processing.'
                    )
        processed_count = len(processed_ids)
        failed_count = len(failed_ids)
        return processed_count, failed_count

    @async_with_exception_handling
    async def _send_batch_to_local_loop(self, items: List[TradeFeedback],
        failed_ids: Set[str]):
        """Send a batch of feedback items to the local feedback loop."""
        if not self.feedback_loop:
            return
        for fb in items:
            if fb.status == FeedbackStatus.ERROR:
                failed_ids.add(fb.id)
                continue
            try:
                await self.feedback_loop.add_feedback(fb)
                fb.status = FeedbackStatus.PROCESSING
                logger.debug(
                    f'Batch: Sent feedback {fb.id} to local FeedbackLoop')
            except Exception as e:
                logger.error(
                    f'Batch: Error sending feedback {fb.id} to local FeedbackLoop: {e}'
                    , exc_info=True)
                fb.status = FeedbackStatus.ERROR
                failed_ids.add(fb.id)

    @async_with_exception_handling
    async def _try_send_batch_to_orchestration(self, source: str, items:
        List[TradeFeedback]) ->bool:
        """Attempt to send a batch to orchestration, handling circuit breaker."""
        if (self.orchestration_circuit_breaker.state == CircuitBreakerState
            .OPEN):
            logger.warning(
                f'Circuit breaker for orchestration is OPEN. Skipping batch for source {source}'
                )
            return False
        try:
            sent = await self._send_batch_to_orchestration(source, items)
            if sent:
                logger.info(
                    f'Batch: Successfully sent {len(items)} items for source {source} to orchestration.'
                    )
                self.orchestration_circuit_breaker.record_success()
            else:
                logger.error(
                    f'Batch: Failed to send batch for source {source} to orchestration.'
                    )
                self.orchestration_circuit_breaker.record_failure()
            return sent
        except Exception as e:
            logger.error(
                f'Batch: Unexpected error sending batch for source {source} to orchestration: {e}'
                , exc_info=True)
            self.orchestration_circuit_breaker.record_failure()
            return False

    @async_with_exception_handling
    async def _publish_batch_processed_event(self, processed_count: int,
        failed_count: int, grouped_data: Dict[str, List]):
        """Publish the FEEDBACK_BATCH_PROCESSED event."""
        if not self.event_publisher:
            return
        try:
            payload = {'total_processed_in_batch': processed_count,
                'total_failed_or_pending': failed_count, 'by_source': {s:
                len(items) for s, items in grouped_data.items()},
                'timestamp': datetime.now(timezone.utc).isoformat()}
            await self.event_publisher.publish(EventType.
                FEEDBACK_BATCH_PROCESSED, payload)
        except Exception as e:
            logger.error(
                f'Failed to publish FEEDBACK_BATCH_PROCESSED event: {e}',
                exc_info=True)

    @async_with_exception_handling
    async def _get_orchestration_endpoint(self, path: str) ->Optional[str]:
        """Discover orchestration service endpoint dynamically if possible."""
        base_url = self.orchestration_service_url
        if self.service_registry_client:
            try:
                service = await self.service_registry_client.get_service(
                    'orchestration-service')
                if service and service.host and service.port:
                    base_url = f'http://{service.host}:{service.port}'
                    logger.debug(
                        f'Discovered orchestration service at {base_url}')
                else:
                    logger.warning(
                        'Orchestration service not found in registry, using configured URL.'
                        )
            except Exception as e:
                logger.warning(
                    f'Failed to discover orchestration service via registry: {e}. Using configured URL.'
                    )
        if not base_url:
            logger.warning('Orchestration service base URL is not configured.')
            return None
        if not path.startswith('/'):
            path = '/' + path
        if base_url.endswith('/'):
            base_url = base_url[:-1]
        return f'{base_url}{path}'

    @async_with_exception_handling
    async def _send_feedback_to_orchestration(self, feedback: TradeFeedback
        ) ->bool:
        """
        Send individual feedback to orchestration service using HTTP client.
        Handles retries and circuit breaker.
        """
        if (self.orchestration_circuit_breaker.state == CircuitBreakerState
            .OPEN):
            logger.warning(
                f'Circuit breaker for orchestration is OPEN. Skipping feedback {feedback.id}'
                )
            return False
        endpoint = await self._get_orchestration_endpoint('/api/v1/feedback')
        if not endpoint:
            logger.error(
                'Cannot send feedback: Orchestration endpoint not available.')
            return False
        retry_count = 0
        last_error = None
        feedback_payload = feedback.dict()
        max_attempts = self.config[CONFIG_ORCH_RETRY_ATTEMPTS]
        while retry_count < max_attempts:
            try:
                logger.debug(
                    f'Attempt {retry_count + 1}/{max_attempts}: Sending feedback {feedback.id} to {endpoint}'
                    )
                response = await self.http_client.post(endpoint, json=
                    feedback_payload)
                response.raise_for_status()
                logger.info(
                    f'Successfully sent feedback {feedback.id} to orchestration. Status: {response.status_code}'
                    )
                return True
            except httpx.HTTPStatusError as e:
                last_error = e
                retry_count += 1
                logger.warning(
                    f'Attempt {retry_count} failed for feedback {feedback.id}: {e}'
                    )
                if 400 <= e.response.status_code < 500:
                    logger.error(
                        f'Client error {e.response.status_code} sending feedback {feedback.id}. No retry.'
                        )
                    break
                if retry_count < max_attempts:
                    await self._wait_before_retry(retry_count)
            except httpx.RequestError as e:
                last_error = e
                retry_count += 1
                logger.warning(
                    f'Attempt {retry_count} failed (RequestError) for feedback {feedback.id}: {e}'
                    )
                if retry_count < max_attempts:
                    await self._wait_before_retry(retry_count)
            except Exception as e:
                last_error = e
                logger.error(
                    f'Unexpected error sending feedback {feedback.id} on attempt {retry_count + 1}: {e}'
                    , exc_info=True)
                break
        logger.error(
            f'All {max_attempts} attempts failed to send feedback {feedback.id} to orchestration. Last error: {last_error}'
            )
        return False

    async def _wait_before_retry(self, attempt: int):
        """Calculate and sleep for exponential backoff duration with jitter."""
        base_delay = self.config[CONFIG_ORCH_RETRY_DELAY]
        delay = base_delay * 2 ** (attempt - 1)
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        actual_delay = max(0.1, delay + jitter)
        logger.debug(f'Retrying in {actual_delay:.2f} seconds')
        await asyncio.sleep(actual_delay)

    @async_with_exception_handling
    async def _send_batch_to_orchestration(self, source: str,
        feedback_items: List[TradeFeedback]) ->bool:
        """
        Send a batch of feedback to the orchestration service using HTTP client.
        No retries implemented at the batch level for simplicity, relies on individual retries if needed.
        """
        endpoint = await self._get_orchestration_endpoint(
            '/api/v1/feedback/batch')
        if not endpoint:
            logger.error(
                'Cannot send batch: Orchestration batch endpoint not available.'
                )
            return False
        batch_payload = {'source': source, 'feedback_items': [fb.dict() for
            fb in feedback_items]}
        try:
            logger.debug(
                f'Sending batch of {len(feedback_items)} items for source {source} to {endpoint}'
                )
            response = await self.http_client.post(endpoint, json=batch_payload
                )
            response.raise_for_status()
            return True
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            raise

    @async_with_exception_handling
    async def _register_with_orchestration(self):
        """Register this collector instance with the orchestration service."""
        if not self.config[CONFIG_ENABLE_ORCHESTRATION]:
            return True
        endpoint = await self._get_orchestration_endpoint(
            '/api/v1/registry/register')
        if not endpoint:
            logger.error(
                'Cannot register: Orchestration registration endpoint not available.'
                )
            return False
        try:
            instance_id = str(uuid.uuid4())
            registration_payload = {'service_type': 'feedback_collector',
                'instance_id': instance_id, 'capabilities': [
                'trade_feedback', 'model_feedback', 'strategy_feedback'],
                'supported_sources': list(FeedbackSource.__members__.keys()
                ), 'supported_categories': list(FeedbackCategory.
                __members__.keys()), 'health_check_endpoint': '/health'}
            logger.debug(
                f'Registering feedback collector instance {instance_id} with orchestration at {endpoint}'
                )
            response = await self.http_client.post(endpoint, json=
                registration_payload)
            response.raise_for_status()
            logger.info(
                f'Successfully registered with orchestration service. Instance ID: {instance_id}'
                )
            self.instance_id = instance_id
            return True
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f'Failed to register with orchestration service: {e}')
            return False
        except Exception as e:
            logger.error(f'Unexpected error during registration: {e}',
                exc_info=True)
            return False

    @async_with_exception_handling
    async def _unregister_from_orchestration(self):
        """Unregister from the orchestration service."""
        if not self.config[CONFIG_ENABLE_ORCHESTRATION] or not hasattr(self,
            'instance_id'):
            return True
        endpoint = await self._get_orchestration_endpoint(
            '/api/v1/registry/unregister')
        if not endpoint:
            logger.warning(
                'Cannot unregister: Orchestration unregistration endpoint not available.'
                )
            return False
        try:
            payload = {'instance_id': self.instance_id}
            logger.debug(
                f'Unregistering feedback collector instance {self.instance_id} from orchestration at {endpoint}'
                )
            response = await self.http_client.post(endpoint, json=payload)
            response.raise_for_status()
            logger.info(
                f'Successfully unregistered instance {self.instance_id} from orchestration service.'
                )
            del self.instance_id
            return True
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(
                f'Failed to unregister from orchestration service (instance: {self.instance_id}): {e}'
                )
            return False
        except Exception as e:
            logger.error(f'Unexpected error during unregistration: {e}',
                exc_info=True)
            return False

    @with_resilience('get_statistics')
    async def get_statistics(self, strategy_id: Optional[str]=None,
        model_id: Optional[str]=None, instrument: Optional[str]=None,
        start_time: Optional[datetime]=None, end_time: Optional[datetime]=None
        ) ->Dict[str, Any]:
        """
        Get feedback collection statistics.

        Args:
            strategy_id: Filter by strategy ID
            model_id: Filter by model ID
            instrument: Filter by instrument
            start_time: Start time for filtering
            end_time: End time for filtering

        Returns:
            Dict[str, Any]: Feedback statistics
        """
        result = {'total_collected': self.stats['total_collected'],
            'by_source': dict(self.stats['by_source']), 'by_category': dict
            (self.stats['by_category'])}
        filtered_feedback = self.recent_feedback.values()
        if strategy_id:
            filtered_feedback = [fb for fb in filtered_feedback if hasattr(
                fb, 'strategy_id') and fb.strategy_id == strategy_id]
            result['by_strategy'] = {strategy_id: self.stats['by_strategy']
                .get(strategy_id, 0)}
        if model_id:
            filtered_feedback = [fb for fb in filtered_feedback if hasattr(
                fb, 'model_id') and fb.model_id == model_id]
            result['by_model'] = {model_id: self.stats['by_model'].get(
                model_id, 0)}
        if instrument:
            filtered_feedback = [fb for fb in filtered_feedback if hasattr(
                fb, 'instrument') and fb.instrument == instrument]
            result['by_instrument'] = {instrument: self.stats[
                'by_instrument'].get(instrument, 0)}
        if start_time:
            filtered_feedback = [fb for fb in filtered_feedback if datetime
                .fromisoformat(fb.timestamp) >= start_time]
        if end_time:
            filtered_feedback = [fb for fb in filtered_feedback if datetime
                .fromisoformat(fb.timestamp) <= end_time]
        result['recent_items'] = len(filtered_feedback)
        result['recent_by_category'] = {}
        result['recent_by_source'] = {}
        for fb in filtered_feedback:
            source = fb.source.value
            category = fb.category.value
            result['recent_by_source'][source] = result['recent_by_source'
                ].get(source, 0) + 1
            result['recent_by_category'][category] = result[
                'recent_by_category'].get(category, 0) + 1
        if self.config['enable_orchestration']:
            result['orchestration_status'] = {'circuit_breaker_state': self
                .orchestration_circuit_breaker.state.value, 'failures':
                self.orchestration_circuit_breaker.failure_count,
                'successes': self.orchestration_circuit_breaker.success_count}
        return result

    @with_database_resilience('get_feedback_by_id')
    async def get_feedback_by_id(self, feedback_id: str) ->Optional[
        TradeFeedback]:
        """
        Get feedback by ID.

        Args:
            feedback_id: ID of the feedback to retrieve

        Returns:
            Optional[TradeFeedback]: The feedback object if found, None otherwise
        """
        return self.recent_feedback.get(feedback_id)

    @with_database_resilience('get_feedback_by_filter')
    async def get_feedback_by_filter(self, source: Optional[FeedbackSource]
        =None, category: Optional[FeedbackCategory]=None, strategy_id:
        Optional[str]=None, model_id: Optional[str]=None, instrument:
        Optional[str]=None, start_time: Optional[datetime]=None, end_time:
        Optional[datetime]=None, limit: int=100) ->List[TradeFeedback]:
        """
        Get feedback by filter criteria.

        Args:
            source: Filter by source
            category: Filter by category
            strategy_id: Filter by strategy ID
            model_id: Filter by model ID
            instrument: Filter by instrument
            start_time: Start time for filtering
            end_time: End time for filtering
            limit: Maximum number of items to return

        Returns:
            List[TradeFeedback]: Filtered feedback items
        """
        filtered_feedback = list(self.recent_feedback.values())
        if source:
            filtered_feedback = [fb for fb in filtered_feedback if fb.
                source == source]
        if category:
            filtered_feedback = [fb for fb in filtered_feedback if fb.
                category == category]
        if strategy_id:
            filtered_feedback = [fb for fb in filtered_feedback if hasattr(
                fb, 'strategy_id') and fb.strategy_id == strategy_id]
        if model_id:
            filtered_feedback = [fb for fb in filtered_feedback if hasattr(
                fb, 'model_id') and fb.model_id == model_id]
        if instrument:
            filtered_feedback = [fb for fb in filtered_feedback if hasattr(
                fb, 'instrument') and fb.instrument == instrument]
        if start_time:
            filtered_feedback = [fb for fb in filtered_feedback if datetime
                .fromisoformat(fb.timestamp) >= start_time]
        if end_time:
            filtered_feedback = [fb for fb in filtered_feedback if datetime
                .fromisoformat(fb.timestamp) <= end_time]
        sorted_feedback = sorted(filtered_feedback, key=lambda fb: fb.
            timestamp, reverse=True)[:limit]
        return sorted_feedback

    @with_resilience('get_orchestration_health')
    @async_with_exception_handling
    async def get_orchestration_health(self) ->Dict[str, Any]:
        """Get health status of the orchestration service integration."""
        if not self.config['enable_orchestration']:
            return {'enabled': False, 'message':
                'Orchestration integration is disabled'}
        health_info = {'enabled': True, 'circuit_breaker_state': self.
            orchestration_circuit_breaker.state.value, 'failure_count':
            self.orchestration_circuit_breaker.failure_count,
            'success_count': self.orchestration_circuit_breaker.
            success_count, 'last_failure_time': self.
            orchestration_circuit_breaker.last_failure_time.isoformat() if
            self.orchestration_circuit_breaker.last_failure_time else None,
            'last_success_time': self.orchestration_circuit_breaker.
            last_success_time.isoformat() if self.
            orchestration_circuit_breaker.last_success_time else None,
            'connectivity': 'UNKNOWN'}
        health_endpoint = await self._get_orchestration_endpoint('/health')
        if not health_endpoint:
            health_info['connectivity'] = 'ERROR'
            health_info['error'
                ] = 'Health endpoint not configured or discoverable'
            return health_info
        if (self.orchestration_circuit_breaker.state == CircuitBreakerState
            .OPEN):
            health_info['connectivity'] = 'FAILED'
            health_info['error'] = 'Circuit breaker is open'
            return health_info
        try:
            start_time = asyncio.get_event_loop().time()
            response = await self.http_client.get(health_endpoint, timeout=5.0)
            end_time = asyncio.get_event_loop().time()
            response.raise_for_status()
            health_info['connectivity'] = 'OK'
            health_info['latency_ms'] = round((end_time - start_time) * 1000, 2
                )
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            health_info['connectivity'] = 'FAILED'
            health_info['error'] = str(e)
        except Exception as e:
            health_info['connectivity'] = 'ERROR'
            health_info['error'
                ] = f'Unexpected error during health check: {str(e)}'
        return health_info
