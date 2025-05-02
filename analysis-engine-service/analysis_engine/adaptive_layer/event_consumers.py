"""
Specialized Kafka event consumers for the feedback system.
Implements dedicated consumers for different feedback event types, batch processing, fault tolerance, retry, and metrics.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone  # Added timezone
from typing import Any, Dict, List, Optional

from core_foundations.events.kafka_event_bus import KafkaEventBus
from core_foundations.events.event_schema import Event, EventType
from core_foundations.models.feedback import TradeFeedback

from analysis_engine.adaptive_layer.feedback_router import FeedbackRouter

logger = logging.getLogger(__name__)


class BaseFeedbackConsumer:
    """
    Base class for Kafka feedback event consumers with retry and metrics.
    """
    def __init__(
        self,
        event_bus: KafkaEventBus,
        router: FeedbackRouter,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.event_bus = event_bus
        self.router = router
        self.config = config or {}
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.retry_backoff = self.config.get("retry_backoff_seconds", 5)
        self.processed_count = 0
        self.failed_count = 0

    def start(self):
        """Subscribe and start consumption."""
        self.event_bus.subscribe(
            event_types=self.event_types,
            handler=self._handle_event
        )
        self.event_bus.start_consuming(blocking=False)
        logger.info(f"{self.__class__.__name__} started subscribing to {self.event_types}")

    def stop(self):
        """Stop consumption."""
        self.event_bus.stop_consuming()
        logger.info(f"{self.__class__.__name__} stopped consuming")

    async def _handle_event(self, event: Event):
        """Internal handler with retry logic."""
        attempt = 0
        while attempt < self.retry_attempts:
            try:
                await self.process_event(event)
                self.processed_count += 1
                return
            except Exception as e:
                attempt += 1
                self.failed_count += 1
                logger.error(
                    f"{self.__class__.__name__} error processing {event.event_type.value}: {e} (attempt {attempt})",
                    exc_info=True
                )
                await asyncio.sleep(self.retry_backoff * attempt)

    async def process_event(self, event: Event):
        """
        Override this method to implement event processing logic.
        Should create or map to TradeFeedback and call router.route_feedback.
        """
        raise NotImplementedError

    def get_metrics(self) -> Dict[str, Any]:
        """Return consumer health metrics."""
        return {
            "consumer": self.__class__.__name__,
            "processed": self.processed_count,
            "failed": self.failed_count,
        }


class TradingOutcomeConsumer(BaseFeedbackConsumer):
    """Consumer for trading outcome feedback events."""
    event_types = [EventType("feedback.trading_outcome")]

    async def process_event(self, event: Event):
        # Convert to TradeFeedback if possible
        try:
            feedback = self.router._create_feedback_from_event(event)  # reuse creation logic
        except Exception:
            feedback = None
        # For critical outcomes, route immediately
        if feedback:
            await self.router.route_feedback(feedback)
        else:
            logger.warning("TradingOutcomeConsumer: invalid feedback format for event %s", event)


class ParameterPerformanceConsumer(BaseFeedbackConsumer):
    """Consumer for parameter performance feedback events."""
    event_types = [EventType("feedback.parameter_performance")]

    async def process_event(self, event: Event):
        data = event.data
        # wrap in TradeFeedback model for routing
        feedback = TradeFeedback(
            feedback_id=data.get("parameter_id") or event.event_id,
            strategy_id=data.get("strategy_id"),
            model_id=None,
            instrument=None,
            timeframe=None,
            source=None,
            category=None,
            outcome_metrics=data.get("metrics", {}),
            metadata={"parameter_name": data.get("parameter_name")},
            timestamp=data.get("timestamp") or datetime.now(timezone.utc).isoformat()  # Changed utcnow()
        )
        await self.router.route_feedback(feedback)


class StrategyEffectivenessConsumer(BaseFeedbackConsumer):
    """Consumer for strategy effectiveness feedback events."""
    event_types = [EventType("feedback.strategy_effectiveness")]

    async def process_event(self, event: Event):
        data = event.data
        # direct routing
        feedback = TradeFeedback(
            feedback_id=event.event_id,
            strategy_id=data.get("strategy_id"),
            model_id=None,
            instrument=None,
            timeframe=None,
            source=None,
            category=None,
            outcome_metrics=data.get("metrics", {}),
            metadata=data,
            timestamp=data.get("timestamp") or datetime.now(timezone.utc).isoformat()  # Changed utcnow()
        )
        await self.router.route_feedback(feedback)


class ModelPredictionConsumer(BaseFeedbackConsumer):
    """Consumer for model prediction feedback events with batch processing."""
    event_types = [EventType("feedback.model_prediction")]

    def __init__(self, event_bus: KafkaEventBus, router: FeedbackRouter, config: Optional[Dict[str, Any]] = None):
        super().__init__(event_bus, router, config)
        self.batch: List[Event] = []
        self.batch_size = self.config.get("batch_size", 50)
        self.batch_timeout = self.config.get("batch_timeout_seconds", 30)
        self._batch_task = None

    def start(self):
        super().start()
        # schedule periodic batch flush
        loop = asyncio.get_event_loop()
        self._batch_task = loop.create_task(self._flush_batches_periodically())

    async def _flush_batches_periodically(self):
        while True:
            await asyncio.sleep(self.batch_timeout)
            if self.batch:
                await self._process_batch()

    async def process_event(self, event: Event):
        self.batch.append(event)
        if len(self.batch) >= self.batch_size:
            await self._process_batch()

    async def _process_batch(self):
        events, self.batch = self.batch, []
        feedbacks: List[TradeFeedback] = []
        for evt in events:
            try:
                fb = self.router._create_feedback_from_event(evt)
                if fb:
                    feedbacks.append(fb)
            except Exception:
                logger.warning("ModelPredictionConsumer: failed to map event %s", evt)
        # route all feedbacks
        for fb in feedbacks:
            await self.router.route_feedback(fb)


class ExecutionQualityConsumer(BaseFeedbackConsumer):
    """Consumer for execution quality feedback events with backpressure handling."""
    event_types = [EventType("feedback.execution_quality")]

    def __init__(self, event_bus: KafkaEventBus, router: FeedbackRouter, config: Optional[Dict[str, Any]] = None):
        super().__init__(event_bus, router, config)
        self.max_pending = self.config.get("max_pending_tasks", 100)
        self._pending = 0
        self._lock = asyncio.Lock()

    async def process_event(self, event: Event):
        # backpressure: wait if too many pending
        async with self._lock:
            while self._pending >= self.max_pending:
                await asyncio.sleep(1)
            self._pending += 1
        try:
            fb = self.router._create_feedback_from_event(event)
            if fb:
                await self.router.route_feedback(fb)
            else:
                logger.warning("ExecutionQualityConsumer: invalid feedback for %s", event)
        finally:
            async with self._lock:
                self._pending -= 1

