"""
EnhancedFeedbackKafkaHandler

This module provides a robust Kafka integration handler for feedback events,
incorporating error handling, reconnection logic, event serialization/deserialization,
and routing to appropriate feedback system components. It merges functionality
from the previous feedback_kafka_integration.py.
"""
import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from core_foundations.utils.logger import get_logger
from core_foundations.events.kafka_event_bus import KafkaEventBus
from core_foundations.events.event_schema import Event, EventType, create_event
from core_foundations.models.feedback import TradeFeedback, FeedbackSource, FeedbackCategory, FeedbackStatus
from analysis_engine.adaptive_layer.feedback_loop import FeedbackLoop
from analysis_engine.adaptive_layer.parameter_tracking_service import ParameterTrackingService
from analysis_engine.adaptive_layer.strategy_mutation_service import StrategyMutationService
from analysis_engine.adaptive_layer.feedback_router import FeedbackRouter
import uuid
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class EnhancedFeedbackKafkaHandler:
    """
    Enhanced Kafka handler for feedback events with robust error handling,
    reconnection logic, proper serialization/deserialization, and internal routing.
    """

    def __init__(self, event_bus: KafkaEventBus, service_name: str,
        feedback_loop: FeedbackLoop, feedback_router: FeedbackRouter,
        parameter_tracking: Optional[ParameterTrackingService]=None,
        strategy_mutation: Optional[StrategyMutationService]=None, config:
        Dict[str, Any]=None):
        """
        Initialize the enhanced Kafka handler.

        Args:
            event_bus: Kafka event bus for communication
            service_name: Service identifier for event source
            feedback_loop: Feedback loop component for processing
            feedback_router: Router for directing categorized feedback
            parameter_tracking: Parameter tracking service (optional)
            strategy_mutation: Strategy mutation service (optional)
            config: Configuration dictionary
        """
        self.event_bus = event_bus
        self.service_name = service_name
        self.config = config or {}
        self.feedback_loop = feedback_loop
        self.feedback_router = feedback_router
        self.parameter_tracking = parameter_tracking
        self.strategy_mutation = strategy_mutation
        self.max_retries = self.config_manager.get('max_retries', 5)
        self.retry_delay = self.config_manager.get('retry_delay_ms', 500) / 1000
        self.max_retry_delay = self.config.get('max_retry_delay_ms', 30000
            ) / 1000
        self.subscribed_topics = set()
        self.subscription_tasks = {}
        self.events_published = 0
        self.events_received = 0
        self.events_failed = 0
        self.last_reconnect_time = None
        self.reconnect_attempts = 0
        self.last_event_time = None
        logger.info('EnhancedFeedbackKafkaHandler initialized for service %s',
            service_name)

    async def start(self):
        """Start the Kafka handler and subscribe to configured topics."""
        topics_config = self.config_manager.get('kafka_topics', {})
        topics_to_subscribe = [topics_config.get('trading_outcome',
            'feedback.trading.outcome'), topics_config.get(
            'parameter_performance', 'feedback.parameter.performance'),
            topics_config.get('strategy_effectiveness',
            'feedback.strategy.effectiveness'), topics_config.get(
            'model_prediction', 'feedback.model.prediction'), topics_config
            .get('execution_quality', 'feedback.execution.quality')]
        topics_to_subscribe.extend(self.config_manager.get('subscribe_topics', []))
        topics_to_subscribe = list(set(topics_to_subscribe))
        for topic in topics_to_subscribe:
            await self.subscribe_to_topic(topic)
        logger.info(
            'EnhancedFeedbackKafkaHandler started and subscribed to %d topics: %s'
            , len(topics_to_subscribe), ', '.join(topics_to_subscribe))
        self._setup_parameter_tracking_hooks()
        self._setup_strategy_mutation_hooks()

    @async_with_exception_handling
    async def stop(self):
        """Stop the Kafka handler and unsubscribe from all topics."""
        for task in self.subscription_tasks.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.subscription_tasks.clear()
        self.subscribed_topics.clear()
        logger.info('EnhancedFeedbackKafkaHandler stopped')

    @async_with_exception_handling
    async def subscribe_to_topic(self, topic: str) ->bool:
        """
        Subscribe to a Kafka topic.

        Args:
            topic: Topic name to subscribe to

        Returns:
            bool: Success status
        """
        if topic in self.subscribed_topics:
            logger.debug('Already subscribed to topic %s', topic)
            return True
        try:
            task = asyncio.create_task(self._subscription_task(topic))
            self.subscription_tasks[topic] = task
            self.subscribed_topics.add(topic)
            logger.info('Subscribed to topic %s', topic)
            return True
        except Exception as e:
            logger.error('Failed to subscribe to topic %s: %s', topic, str(e))
            return False

    @async_with_exception_handling
    async def publish_event(self, event_type: Union[str, EventType], data:
        Dict[str, Any], topic: Optional[str]=None, key: Optional[str]=None
        ) ->bool:
        """
        Publish an event to Kafka with retry logic.

        Args:
            event_type: Type of event (string or EventType enum)
            data: Event data
            topic: Optional topic override
            key: Optional message key

        Returns:
            bool: Success status
        """
        if isinstance(event_type, str):
            try:
                event_type_enum = EventType(event_type)
            except ValueError:
                logger.warning(
                    f"Invalid event type string '{event_type}'. Using as is.")
                event_type_enum = event_type
        else:
            event_type_enum = event_type
        if not topic:
            topic_map = self.config_manager.get('topic_map', {})
            event_type_key = event_type_enum.value if isinstance(
                event_type_enum, EventType) else event_type_enum
            topic = topic_map.get(event_type_key, self.config.get(
                'default_publish_topic', 'feedback'))
        event = create_event(event_type=event_type_enum, source_service=
            self.service_name, data=data)
        return await self._publish_with_retry(event, topic, key)

    async def publish_feedback_event(self, feedback: TradeFeedback, topic:
        Optional[str]=None) ->bool:
        """
        Publish a feedback event to Kafka.

        Args:
            feedback: Feedback object to publish
            topic: Optional topic override

        Returns:
            bool: Success status
        """
        feedback_data = self._feedback_to_dict(feedback)
        event_type = f'feedback.{feedback.source.value.lower()}'
        if feedback.category:
            event_type += f'.{feedback.category.value.lower()}'
        key = feedback.id
        success = await self.publish_event(event_type, feedback_data, topic,
            key)
        if success:
            logger.debug('Published feedback event %s with ID %s',
                event_type, feedback.id)
        else:
            logger.warning('Failed to publish feedback event %s with ID %s',
                event_type, feedback.id)
        return success

    async def publish_parameter_change(self, strategy_id: str,
        parameter_name: str, old_value: Any, new_value: Any, change_reason:
        str, source_component: str, parameter_id: Optional[str]=None,
        confidence_level: float=0.5, metadata: Optional[Dict[str, Any]]=None
        ) ->str:
        """Publish a parameter change event."""
        event_data = {'parameter_id': parameter_id or str(uuid.uuid4()),
            'strategy_id': strategy_id, 'parameter_name': parameter_name,
            'old_value': self._prepare_value_for_serialization(old_value),
            'new_value': self._prepare_value_for_serialization(new_value),
            'change_reason': change_reason, 'source_component':
            source_component, 'confidence_level': confidence_level,
            'timestamp': datetime.utcnow().isoformat(), 'metadata': 
            metadata or {}}
        event_type = EventType('feedback.parameter_change')
        topic = self.config_manager.get('kafka_topics', {}).get('parameter_change',
            'feedback.parameter.change')
        success = await self.publish_event(event_type, event_data, topic=
            topic, key=strategy_id)
        event_id = event_data['parameter_id']
        if success:
            logger.debug(
                f'Published parameter change event {event_id} for {parameter_name} in {strategy_id}'
                )
        return event_id if success else ''

    async def publish_strategy_adaptation(self, strategy_id: str,
        adaptation_id: str, market_regime: str, changes: List[Dict[str, Any
        ]], adaptation_type: str, expected_improvement: float, metadata:
        Optional[Dict[str, Any]]=None) ->str:
        """Publish a strategy adaptation event."""
        event_data = {'adaptation_id': adaptation_id, 'strategy_id':
            strategy_id, 'market_regime': market_regime, 'changes': changes,
            'adaptation_type': adaptation_type, 'expected_improvement':
            expected_improvement, 'timestamp': datetime.utcnow().isoformat(
            ), 'metadata': metadata or {}}
        event_type = EventType('feedback.strategy_adaptation')
        topic = self.config_manager.get('kafka_topics', {}).get('strategy_adaptation',
            'feedback.strategy.adaptation')
        success = await self.publish_event(event_type, event_data, topic=
            topic, key=strategy_id)
        if success:
            logger.debug(
                f'Published strategy adaptation event {adaptation_id} for {strategy_id}'
                )
        return adaptation_id if success else ''

    async def publish_strategy_mutation(self, strategy_id: str, mutation_id:
        str, parent_version: str, new_version: str, mutation_type: str,
        changes: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]]=None
        ) ->str:
        """Publish a strategy mutation event."""
        event_data = {'mutation_id': mutation_id, 'strategy_id':
            strategy_id, 'parent_version': parent_version, 'new_version':
            new_version, 'mutation_type': mutation_type, 'changes': changes,
            'timestamp': datetime.utcnow().isoformat(), 'metadata': 
            metadata or {}}
        event_type = EventType('feedback.strategy_mutation')
        topic = self.config_manager.get('kafka_topics', {}).get('strategy_mutation',
            'feedback.strategy.mutation')
        success = await self.publish_event(event_type, event_data, topic=
            topic, key=strategy_id)
        if success:
            logger.debug(
                f'Published strategy mutation event {mutation_id} for {strategy_id}'
                )
        return mutation_id if success else ''

    async def publish_feedback_insight(self, strategy_id: str, insight_type:
        str, message: str, data: Dict[str, Any], priority: str='medium',
        metadata: Optional[Dict[str, Any]]=None) ->str:
        """Publish a feedback insight event."""
        insight_id = str(uuid.uuid4())
        event_data = {'insight_id': insight_id, 'strategy_id': strategy_id,
            'insight_type': insight_type, 'message': message, 'data': data,
            'priority': priority, 'timestamp': datetime.utcnow().isoformat(
            ), 'metadata': metadata or {}}
        event_type = EventType('feedback.insight')
        topic = self.config_manager.get('kafka_topics', {}).get('feedback_insight',
            'feedback.insight')
        success = await self.publish_event(event_type, event_data, topic=
            topic, key=strategy_id)
        if success:
            logger.debug(
                f'Published feedback insight event {insight_id} for {strategy_id}'
                )
        return insight_id if success else ''

    @async_with_exception_handling
    async def _subscription_task(self, topic: str):
        """
        Background task that handles subscription to a Kafka topic.
        Routes events to _process_event.
        """
        while True:
            try:
                async for event in self.event_bus.subscribe(topic):
                    await self._process_event(event)
            except asyncio.CancelledError:
                logger.debug('Subscription task for topic %s cancelled', topic)
                break
            except Exception as e:
                logger.error('Error in Kafka subscription for topic %s: %s',
                    topic, str(e))
                self.events_failed += 1
                self.reconnect_attempts += 1
                self.last_reconnect_time = datetime.utcnow()
                delay = min(self.retry_delay * 2 ** (self.
                    reconnect_attempts - 1), self.max_retry_delay)
                logger.info('Reconnecting to topic %s in %.1f seconds',
                    topic, delay)
                await asyncio.sleep(delay)

    @async_with_exception_handling
    async def _process_event(self, event: Event):
        """
        Process a received Kafka event and route it to the appropriate handler.
        (Replaces the old _handle_feedback_event and routing logic).
        """
        try:
            self.events_received += 1
            self.last_event_time = datetime.utcnow()
            event_type = event.event_type
            event_data = event.data
            topics_config = self.config_manager.get('kafka_topics', {})
            logger.debug(
                f'Received event {event.event_id} of type {event_type}')
            if event_type == EventType(topics_config.get('trading_outcome',
                'feedback.trading.outcome')):
                await self._handle_trading_outcome(event)
            elif event_type == EventType(topics_config.get(
                'parameter_performance', 'feedback.parameter.performance')):
                await self._handle_parameter_performance(event_data)
            elif event_type == EventType(topics_config.get(
                'strategy_effectiveness', 'feedback.strategy.effectiveness')):
                await self._handle_strategy_effectiveness(event_data)
            elif event_type == EventType(topics_config.get(
                'model_prediction', 'feedback.model.prediction')):
                await self._handle_model_prediction(event_data)
            elif event_type == EventType(topics_config.get(
                'execution_quality', 'feedback.execution.quality')):
                await self._handle_execution_quality(event_data)
            else:
                logger.debug(
                    f'No specific handler for event type {event_type}. Ignoring.'
                    )
        except Exception as e:
            self.events_failed += 1
            error_msg = (
                f'Error processing event {event.event_id} ({event_type}): {str(e)}'
                )
            logger.error(error_msg, exc_info=True)
            await self._publish_to_dlq(event, error_msg)

    @async_with_exception_handling
    async def _handle_trading_outcome(self, event: Event):
        """Handle trading outcome feedback."""
        feedback: Optional[TradeFeedback] = None
        try:
            feedback = self._dict_to_feedback(event.data)
            if not feedback:
                logger.warning(
                    f'Could not create TradeFeedback from event {event.event_id}. Skipping.'
                    )
                self.events_failed += 1
                await self._publish_to_dlq(event,
                    'Failed to parse TradeFeedback')
                return
            if feedback.source is None:
                feedback.source = FeedbackSource.TRADING_OUTCOME
            if feedback.category is None:
                feedback.category = FeedbackCategory.UNCATEGORIZED
            feedback.status = FeedbackStatus.RECEIVED
            if self.feedback_loop:
                await self.feedback_loop.process_incoming_feedback(feedback)
                logger.info(
                    f'Passed feedback {feedback.id} from event {event.event_id} to FeedbackLoop.'
                    )
            else:
                logger.warning(
                    f'FeedbackLoop not configured. Cannot process feedback {feedback.id}.'
                    )
                self.events_failed += 1
                await self._publish_to_dlq(event, 'FeedbackLoop not configured'
                    )
        except Exception as e:
            self.events_failed += 1
            error_msg = (
                f'Error processing trading outcome event {event.event_id}: {str(e)}'
                )
            logger.error(error_msg, exc_info=True)
            await self._publish_to_dlq(event, error_msg)

    @async_with_exception_handling
    async def _handle_parameter_performance(self, data: Dict[str, Any]):
        """Handle parameter performance feedback."""
        if not self.parameter_tracking:
            logger.debug(
                'ParameterTrackingService not configured. Skipping parameter performance event.'
                )
            return
        try:
            parameter_id = data.get('parameter_id')
            strategy_id = data.get('strategy_id')
            parameter_name = data.get('parameter_name')
            metrics = data.get('metrics', {})
            timestamp = data.get('timestamp')
            if not parameter_id or not strategy_id or not parameter_name:
                logger.warning(
                    'Parameter performance feedback missing required fields. Data: %s'
                    , data)
                self.events_failed += 1
                return
            await self.parameter_tracking.record_parameter_performance(
                parameter_id=parameter_id, strategy_id=strategy_id,
                parameter_name=parameter_name, performance_metrics=metrics,
                timestamp=timestamp)
            logger.debug(
                f'Recorded performance for parameter {parameter_name} in strategy {strategy_id}'
                )
        except Exception as e:
            self.events_failed += 1
            error_msg = (
                f'Error processing parameter performance feedback: {str(e)}. Data: {data}'
                )
            logger.error(error_msg, exc_info=True)

    @async_with_exception_handling
    async def _handle_strategy_effectiveness(self, data: Dict[str, Any]):
        """Handle strategy effectiveness feedback."""
        try:
            strategy_id = data.get('strategy_id')
            if not strategy_id:
                logger.warning(
                    'Strategy effectiveness feedback missing strategy ID. Data: %s'
                    , data)
                self.events_failed += 1
                return
            if self.strategy_mutation and data.get('evaluate_mutation', False):
                market_regime = data.get('market_regime')
                await self.strategy_mutation.evaluate_and_select_best_version(
                    strategy_id=strategy_id, market_regime=market_regime)
                logger.debug(f'Evaluated strategy versions for {strategy_id}')
            if self.feedback_loop and data.get('generate_insights', False):
                insights = await self.feedback_loop.generate_insights(
                    strategy_id)
                logger.debug(
                    f'Generated {len(insights)} insights for strategy {strategy_id}'
                    )
        except Exception as e:
            self.events_failed += 1
            error_msg = (
                f'Error processing strategy effectiveness feedback: {str(e)}. Data: {data}'
                )
            logger.error(error_msg, exc_info=True)

    @async_with_exception_handling
    async def _handle_model_prediction(self, data: Dict[str, Any]):
        """Handle model prediction feedback."""
        try:
            feedback = self._dict_to_feedback(data)
            if not feedback:
                logger.warning(
                    'Could not parse model prediction feedback. Data: %s', data
                    )
                self.events_failed += 1
                return
            feedback.source = FeedbackSource.MODEL_PREDICTION
            feedback.category = FeedbackCategory.MODEL_PERFORMANCE
            if self.feedback_router:
                await self.feedback_router.route_feedback(feedback)
                logger.debug(f'Routed model prediction feedback {feedback.id}')
            else:
                logger.warning(
                    'FeedbackRouter not configured. Cannot route model prediction feedback.'
                    )
                self.events_failed += 1
        except Exception as e:
            self.events_failed += 1
            error_msg = (
                f'Error processing model prediction feedback: {str(e)}. Data: {data}'
                )
            logger.error(error_msg, exc_info=True)

    @async_with_exception_handling
    async def _handle_execution_quality(self, data: Dict[str, Any]):
        """Handle execution quality feedback."""
        try:
            feedback = self._dict_to_feedback(data)
            if not feedback:
                logger.warning(
                    'Could not parse execution quality feedback. Data: %s',
                    data)
                self.events_failed += 1
                return
            feedback.source = FeedbackSource.EXECUTION_QUALITY
            feedback.category = FeedbackCategory.EXECUTION_PERFORMANCE
            if self.feedback_router:
                await self.feedback_router.route_feedback(feedback)
                logger.debug(f'Routed execution quality feedback {feedback.id}'
                    )
            else:
                logger.warning(
                    'FeedbackRouter not configured. Cannot route execution quality feedback.'
                    )
                self.events_failed += 1
        except Exception as e:
            self.events_failed += 1
            error_msg = (
                f'Error processing execution quality feedback: {str(e)}. Data: {data}'
                )
            logger.error(error_msg, exc_info=True)

    def _setup_parameter_tracking_hooks(self):
        """Set up hooks to automatically publish parameter tracking events."""
        if not self.parameter_tracking:
            return
        original_record = self.parameter_tracking.record_parameter_change
        publisher = self

        async def patched_record_parameter_change(*args, **kwargs):
    """
    Patched record parameter change.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            param_id = await original_record(*args, **kwargs)
            await publisher.publish_parameter_change(parameter_id=param_id,
                strategy_id=kwargs.get('strategy_id'), parameter_name=
                kwargs.get('parameter_name'), old_value=kwargs.get(
                'old_value'), new_value=kwargs.get('new_value'),
                change_reason=kwargs.get('change_reason'), source_component
                =kwargs.get('source_component'), confidence_level=kwargs.
                get('confidence_level', 0.5), metadata=kwargs.get(
                'market_conditions', {}))
            return param_id
        self.parameter_tracking.record_parameter_change = (
            patched_record_parameter_change)
        logger.debug(
            'Parameter tracking hooks set up in EnhancedFeedbackKafkaHandler')

    def _setup_strategy_mutation_hooks(self):
        """Set up hooks to automatically publish strategy mutation events."""
        if not self.strategy_mutation:
            return
        original_mutate = self.strategy_mutation.mutate_strategy
        publisher = self

        async def patched_mutate_strategy(*args, **kwargs):
    """
    Patched mutate strategy.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            new_version_id = await original_mutate(*args, **kwargs)
            if new_version_id:
                strategy_id = kwargs.get('strategy_id') or (args[0] if args
                     else None)
                mutation = None
                if hasattr(self.strategy_mutation, 'get_mutation_details'):
                    mutation_details = (await self.strategy_mutation.
                        get_mutation_details(new_version_id))
                    if mutation_details and strategy_id:
                        await publisher.publish_strategy_mutation(strategy_id
                            =strategy_id, mutation_id=mutation_details.get(
                            'mutation_id'), parent_version=mutation_details
                            .get('parent_version'), new_version=
                            new_version_id, mutation_type=mutation_details.
                            get('mutation_type'), changes=mutation_details.
                            get('changes', []), metadata=mutation_details.
                            get('metadata', {}))
                else:
                    logger.warning(
                        'Cannot publish strategy mutation event: StrategyMutationService lacks details or method.'
                        )
            return new_version_id
        self.strategy_mutation.mutate_strategy = patched_mutate_strategy
        logger.debug(
            'Strategy mutation hooks set up in EnhancedFeedbackKafkaHandler')

    @async_with_exception_handling
    async def _publish_with_retry(self, event: Event, topic: str, key:
        Optional[str]=None) ->bool:
        """Publish an event to Kafka with retry logic."""
        retries = 0
        last_error = None
        while retries <= self.max_retries:
            try:
                await self.event_bus.publish(event, topic=topic, key=key)
                self.events_published += 1
                if retries > 0:
                    logger.info('Successfully published event after %d retries'
                        , retries)
                return True
            except Exception as e:
                last_error = e
                retries += 1
                if retries == 1:
                    logger.warning('Error publishing to topic %s: %s',
                        topic, str(e))
                if retries > self.max_retries:
                    break
                delay = self.retry_delay * 2 ** (retries - 1)
                await asyncio.sleep(delay)
        self.events_failed += 1
        logger.error('Failed to publish event to topic %s after %d retries: %s'
            , topic, self.max_retries, str(last_error))
        return False

    def _feedback_to_dict(self, feedback: TradeFeedback) ->Dict[str, Any]:
        """Convert a feedback object to a serializable dictionary."""
        result = {'id': feedback.id, 'timestamp': feedback.timestamp}
        for field, attr in [('strategy_id', 'strategy_id'), ('model_id',
            'model_id'), ('instrument', 'instrument'), ('timeframe',
            'timeframe')]:
            if hasattr(feedback, attr) and getattr(feedback, attr) is not None:
                result[field] = getattr(feedback, attr)
        if hasattr(feedback, 'source') and feedback.source is not None:
            result['source'] = feedback.source.value
        if hasattr(feedback, 'category') and feedback.category is not None:
            result['category'] = feedback.category.value
        if hasattr(feedback, 'status') and feedback.status is not None:
            result['status'] = feedback.status.value
        for field, attr in [('outcome_metrics', 'outcome_metrics'), (
            'metadata', 'metadata'), ('content', 'content')]:
            if hasattr(feedback, attr) and getattr(feedback, attr) is not None:
                result[field] = getattr(feedback, attr)
        return result

    @with_exception_handling
    def _dict_to_feedback(self, data: Dict[str, Any]) ->Optional[TradeFeedback
        ]:
        """Convert a dictionary to a feedback object."""
        try:
            source_val = data.get('source')
            source = FeedbackSource(source_val
                ) if source_val else FeedbackSource.UNKNOWN
            category_val = data.get('category')
            category = FeedbackCategory(category_val
                ) if category_val else FeedbackCategory.UNKNOWN
            status_val = data.get('status')
            status = FeedbackStatus(status_val
                ) if status_val else FeedbackStatus.UNKNOWN
            feedback = TradeFeedback(id=data.get('id') or data.get(
                'feedback_id') or str(uuid.uuid4()), strategy_id=data.get(
                'strategy_id'), model_id=data.get('model_id'), instrument=
                data.get('instrument') or data.get('symbol'), timeframe=
                data.get('timeframe'), source=source, category=category,
                status=status, outcome_metrics=data.get('outcome_metrics') or
                data.get('metrics') or {}, metadata=data.get('metadata') or
                {}, timestamp=data.get('timestamp') or datetime.utcnow().
                isoformat(), content=data.get('content'))
            return feedback
        except Exception as e:
            logger.error('Error converting dict to feedback: %s. Data: %s',
                str(e), data)
            return None

    def _prepare_value_for_serialization(self, value: Any) ->Any:
        """Prepare a value for JSON serialization."""
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [self._prepare_value_for_serialization(v) for v in value]
        if isinstance(value, dict):
            return {k: self._prepare_value_for_serialization(v) for k, v in
                value.items()}
        return str(value)

    @async_with_exception_handling
    async def _publish_to_dlq(self, original_event: Event, error_reason: str):
        """Publish a failed event to the Dead Letter Queue topic."""
        try:
            dlq_topic = self.config_manager.get('kafka_topics', {}).get('feedback_dlq',
                'feedback.dead_letter_queue')
            if not dlq_topic:
                logger.warning(
                    'DLQ topic not configured. Cannot publish failed event.')
                return
            dlq_event_data = {'original_event_id': original_event.event_id,
                'original_event_type': original_event.event_type.value if
                original_event.event_type else None, 'original_payload':
                original_event.data, 'error_reason': error_reason,
                'failed_at': datetime.utcnow().isoformat(),
                'consumer_service': self.service_name, 'consumer_group':
                self.config_manager.get('consumer_config', {}).get('group.id',
                'unknown_group')}
            dlq_event = create_event(event_type=EventType(
                'system.processing_failure'), source_service=self.
                service_name, data=dlq_event_data)
            success = await self._publish_with_retry(dlq_event, dlq_topic)
            if success:
                logger.info(
                    f'Published failed event {original_event.event_id} to DLQ topic {dlq_topic}'
                    )
            else:
                logger.error(
                    f'Failed to publish event {original_event.event_id} to DLQ after retries.'
                    )
        except Exception as pub_exc:
            logger.error(
                f'Unexpected error publishing event {original_event.event_id} to DLQ: {str(pub_exc)}'
                , exc_info=True)

    @with_resilience('get_health_metrics')
    async def get_health_metrics(self) ->Dict[str, Any]:
        """Get health metrics for the Kafka handler."""
        return {'events_published': self.events_published,
            'events_received': self.events_received, 'events_failed': self.
            events_failed, 'subscribed_topics': list(self.subscribed_topics
            ), 'reconnect_attempts': self.reconnect_attempts,
            'last_reconnect_time': self.last_reconnect_time.isoformat() if
            self.last_reconnect_time else None, 'last_event_time': self.
            last_event_time.isoformat() if self.last_event_time else None}
    get_stats = get_health_metrics
