"""
Feedback Integration Service

This module implements the integration service that connects all feedback components
and coordinates the bidirectional feedback loop between trading, analysis, and ML components.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from core_foundations.utils.logger import get_logger
from core_foundations.config.configuration import ConfigurationManager
from core_foundations.events.event_publisher import EventPublisher
from core_foundations.events.event_subscriber import EventSubscriber
from core_foundations.events.event_schema import EventType
from analysis_engine.adaptive_layer.adaptation_engine import AdaptationEngine
from analysis_engine.adaptive_layer.feedback_loop import FeedbackLoop
from analysis_engine.adapters import ModelTrainingFeedbackAdapter
logger = get_logger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FeedbackIntegrationService:
    """
    Service that integrates all feedback components and coordinates the bidirectional
    feedback loop between trading, analysis, and ML components.

    This service is responsible for:
    - Initializing and connecting all feedback components
    - Coordinating the flow of feedback between components
    - Managing event subscriptions for feedback-related events
    - Providing a unified interface for feedback operations
    """

    def __init__(self, config_manager: ConfigurationManager,
        event_publisher: EventPublisher, event_subscriber: EventSubscriber,
        adaptation_engine: AdaptationEngine, feedback_loop: FeedbackLoop,
        ml_client=None, model_training_feedback: Optional[
        ModelTrainingFeedbackAdapter]=None):
        """
        Initialize the feedback integration service.

        Args:
            config_manager: Configuration manager for service configuration
            event_publisher: Event publisher for broadcasting events
            event_subscriber: Event subscriber for receiving events
            adaptation_engine: Adaptation engine for strategy adaptation
            feedback_loop: Feedback loop for processing trading feedback
            ml_client: Optional ML client for model integration
            model_training_feedback: Optional model training feedback adapter
        """
        self.config_manager = config_manager
        self.event_publisher = event_publisher
        self.event_subscriber = event_subscriber
        self.adaptation_engine = adaptation_engine
        self.feedback_loop = feedback_loop
        self.ml_client = ml_client
        self.model_training_feedback = model_training_feedback
        self.config = config_manager.get_configuration('feedback_integration',
            {})
        self.active_tasks = {}
        self.is_running = False
        logger.info('FeedbackIntegrationService initialized')

    async def start(self):
        """Start the feedback integration service."""
        if self.is_running:
            logger.warning('FeedbackIntegrationService is already running')
            return
        logger.info('Starting FeedbackIntegrationService')
        await self._initialize_event_subscriptions()
        await self.feedback_loop.initialize_event_subscriptions(self.
            event_subscriber)
        self._start_background_tasks()
        self.is_running = True
        logger.info('FeedbackIntegrationService started successfully')

    @async_with_exception_handling
    async def stop(self):
        """Stop the feedback integration service."""
        if not self.is_running:
            logger.warning('FeedbackIntegrationService is not running')
            return
        logger.info('Stopping FeedbackIntegrationService')
        for task_name, task in self.active_tasks.items():
            if not task.done():
                logger.info(f'Cancelling task: {task_name}')
                task.cancel()
        for task_name, task in self.active_tasks.items():
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f'Task {task_name} cancelled')
            except Exception as e:
                logger.error(f'Error while stopping task {task_name}: {str(e)}'
                    )
        self.active_tasks = {}
        self.is_running = False
        logger.info('FeedbackIntegrationService stopped successfully')

    async def _initialize_event_subscriptions(self):
        """Initialize event subscriptions for the service."""
        logger.info('Initializing event subscriptions')
        await self.event_subscriber.subscribe(EventType.TRADING_OUTCOME,
            self._handle_trading_outcome)
        await self.event_subscriber.subscribe(EventType.MODEL_PREDICTION,
            self._handle_model_prediction)
        await self.event_subscriber.subscribe(EventType.STRATEGY_EXECUTION,
            self._handle_strategy_execution)
        await self.event_subscriber.subscribe(EventType.
            MARKET_REGIME_CHANGE, self._handle_market_regime_change)
        logger.info('Event subscriptions initialized successfully')

    def _start_background_tasks(self):
        """Start background tasks for the service."""
        logger.info('Starting background tasks')
        self.active_tasks['periodic_feedback_processing'
            ] = asyncio.create_task(self._periodic_feedback_processing())
        if self.model_training_feedback:
            self.active_tasks['periodic_model_evaluation'
                ] = asyncio.create_task(self._periodic_model_evaluation())
        logger.info('Background tasks started successfully')

    @async_with_exception_handling
    async def _periodic_feedback_processing(self):
        """Periodically process accumulated feedback."""
        logger.info('Starting periodic feedback processing task')
        interval = self.config_manager.get('feedback_processing_interval_seconds', 300)
        while True:
            try:
                await self._process_accumulated_feedback()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logger.info('Periodic feedback processing task cancelled')
                break
            except Exception as e:
                logger.error(f'Error in periodic feedback processing: {str(e)}'
                    )
                await asyncio.sleep(60)

    @async_with_exception_handling
    async def _periodic_model_evaluation(self):
        """Periodically evaluate model performance."""
        logger.info('Starting periodic model evaluation task')
        interval = self.config_manager.get('model_evaluation_interval_seconds', 3600)
        while True:
            try:
                await self._evaluate_model_performance()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logger.info('Periodic model evaluation task cancelled')
                break
            except Exception as e:
                logger.error(f'Error in periodic model evaluation: {str(e)}')
                await asyncio.sleep(60)

    async def _process_accumulated_feedback(self):
        """Process accumulated feedback from various sources."""
        logger.debug('Processing accumulated feedback')
        await self.feedback_loop.process_accumulated_feedback()
        if self.model_training_feedback:
            pass

    @async_with_exception_handling
    async def _evaluate_model_performance(self):
        """Evaluate model performance and trigger retraining if needed."""
        logger.debug('Evaluating model performance')
        if not self.model_training_feedback:
            logger.warning('Model training feedback adapter not available')
            return
        active_models = self.config_manager.get('active_models', [])
        for model_id in active_models:
            try:
                metrics = (await self.model_training_feedback.
                    get_model_performance_metrics(model_id=model_id,
                    start_date=datetime.now().replace(hour=0, minute=0,
                    second=0, microsecond=0)))
                if self._should_retrain_model(model_id, metrics):
                    logger.info(f'Triggering retraining for model {model_id}')
                    await self.model_training_feedback.trigger_model_update(
                        model_id=model_id, reason='performance_degradation',
                        context={'metrics': metrics})
            except Exception as e:
                logger.error(
                    f'Error evaluating performance for model {model_id}: {str(e)}'
                    )

    def _should_retrain_model(self, model_id: str, metrics: Dict[str, Any]
        ) ->bool:
        """
        Determine if a model should be retrained based on performance metrics.

        Args:
            model_id: ID of the model
            metrics: Performance metrics for the model

        Returns:
            True if the model should be retrained, False otherwise
        """
        if metrics.get('is_fallback', False):
            return False
        thresholds = self.config_manager.get('retraining_thresholds', {})
        accuracy_threshold = thresholds.get('accuracy', 0.7)
        if metrics.get('metrics', {}).get('accuracy', 1.0
            ) < accuracy_threshold:
            return True
        precision_threshold = thresholds.get('precision', 0.7)
        if metrics.get('metrics', {}).get('precision', 1.0
            ) < precision_threshold:
            return True
        recall_threshold = thresholds.get('recall', 0.7)
        if metrics.get('metrics', {}).get('recall', 1.0) < recall_threshold:
            return True
        f1_threshold = thresholds.get('f1_score', 0.7)
        if metrics.get('metrics', {}).get('f1_score', 1.0) < f1_threshold:
            return True
        return False

    async def _handle_trading_outcome(self, event: Dict[str, Any]):
        """
        Handle trading outcome events.

        Args:
            event: Trading outcome event data
        """
        logger.debug(f"Handling trading outcome event: {event.get('id')}")
        await self.feedback_loop.process_trading_outcome(event)
        if self.model_training_feedback and 'model_id' in event:
            pass

    async def _handle_model_prediction(self, event: Dict[str, Any]):
        """
        Handle model prediction events.

        Args:
            event: Model prediction event data
        """
        logger.debug(f"Handling model prediction event: {event.get('id')}")
        pass

    async def _handle_strategy_execution(self, event: Dict[str, Any]):
        """
        Handle strategy execution events.

        Args:
            event: Strategy execution event data
        """
        logger.debug(f"Handling strategy execution event: {event.get('id')}")
        await self.adaptation_engine.process_strategy_execution(event)
        await self.feedback_loop.process_strategy_execution(event)

    async def _handle_market_regime_change(self, event: Dict[str, Any]):
        """
        Handle market regime change events.

        Args:
            event: Market regime change event data
        """
        logger.debug(f"Handling market regime change event: {event.get('id')}")
        await self.adaptation_engine.process_market_regime_change(event)
