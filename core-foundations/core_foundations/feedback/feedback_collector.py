"""
Enhanced feedback collection system with structured categorization and validation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from uuid import uuid4

from core_foundations.events.kafka_event_bus import KafkaEventBus
from core_foundations.events.event_topics import EventTopics
from core_foundations.models.feedback import TradeFeedback, FeedbackSource, FeedbackCategory
from core_foundations.utils.validation import validate_feedback

logger = logging.getLogger(__name__)

@dataclass
class FeedbackValidationResult:
    """
    FeedbackValidationResult class.
    
    Attributes:
        Add attributes here
    """

    is_valid: bool
    error_message: Optional[str] = None
    feedback_id: Optional[str] = None

class FeedbackCollector:
    """
    FeedbackCollector class.
    
    Attributes:
        Add attributes here
    """

    def __init__(
        self,
        event_bus: KafkaEventBus,
        batch_size: int = 100,
        batch_timeout_seconds: int = 30,
        validation_rules: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        event_bus: Description of event_bus
        batch_size: Description of batch_size
        batch_timeout_seconds: Description of batch_timeout_seconds
        validation_rules: Description of validation_rules
        Any]]: Description of Any]]
    
    """

        self.event_bus = event_bus
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_seconds
        self.validation_rules = validation_rules or {}
        self.pending_batch: List[TradeFeedback] = []
        self._batch_lock = asyncio.Lock()
        self._flush_task = None
        
    async def start(self):
        """Start the feedback collection service."""
        self._flush_task = asyncio.create_task(self._periodic_batch_flush())
        logger.info("FeedbackCollector started with batch size %d and timeout %d seconds",
                   self.batch_size, self.batch_timeout)
        
    async def stop(self):
        """Stop the feedback collection service."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush_batch()  # Flush any remaining items
        
    async def collect_feedback(self, feedback: TradeFeedback) -> FeedbackValidationResult:
        """
        Collect and validate trading feedback.
        
        Args:
            feedback: The feedback to collect
            
        Returns:
            FeedbackValidationResult containing validation status
        """
        # Ensure basic fields are present
        if not feedback.feedback_id:
            feedback.feedback_id = str(uuid4())
        if not feedback.timestamp:
            feedback.timestamp = datetime.utcnow().isoformat()
            
        # Validate feedback
        validation_result = validate_feedback(feedback, self.validation_rules)
        if not validation_result.is_valid:
            logger.warning("Invalid feedback received: %s", validation_result.error_message)
            return validation_result
            
        # Add to batch or process immediately based on priority
        if feedback.priority in ('HIGH', 'CRITICAL'):
            await self._process_feedback(feedback)
        else:
            async with self._batch_lock:
                self.pending_batch.append(feedback)
                if len(self.pending_batch) >= self.batch_size:
                    await self._flush_batch()
                    
        return FeedbackValidationResult(
            is_valid=True,
            feedback_id=feedback.feedback_id
        )
        
    async def _periodic_batch_flush(self):
        """Periodically flush pending feedback batch."""
        while True:
            try:
                await asyncio.sleep(self.batch_timeout)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in periodic batch flush: %s", str(e))
                
    async def _flush_batch(self):
        """Flush the current batch of feedback."""
        async with self._batch_lock:
            if not self.pending_batch:
                return
                
            batch = self.pending_batch
            self.pending_batch = []
            
        try:
            # Group feedback by category for efficient processing
            categorized_batch = self._categorize_batch(batch)
            
            # Process each category
            for category, items in categorized_batch.items():
                await self._publish_batch(category, items)
                
        except Exception as e:
            logger.error("Error processing feedback batch: %s", str(e))
            # Return items to pending batch
            async with self._batch_lock:
                self.pending_batch.extend(batch)
                
    async def _publish_batch(self, category: str, items: List[TradeFeedback]):
        """Publish a batch of feedback for a specific category."""
        try:
            event_data = {
                "batch_id": str(uuid4()),
                "category": category,
                "timestamp": datetime.utcnow().isoformat(),
                "items": [item.dict() for item in items]
            }
            
            await self.event_bus.publish(
                EventTopics.FEEDBACK_BATCH,
                event_data,
                key=category  # Use category as key for partitioning
            )
            
        except Exception as e:
            logger.error("Failed to publish feedback batch: %s", str(e))
            raise
            
    async def _process_feedback(self, feedback: TradeFeedback):
        """Process a single feedback item immediately."""
        try:
            await self.event_bus.publish(
                EventTopics.FEEDBACK_COLLECTION,
                feedback.dict(),
                key=str(feedback.feedback_id)
            )
        except Exception as e:
            logger.error("Failed to process feedback: %s", str(e))
            raise
            
    def _categorize_batch(self, batch: List[TradeFeedback]) -> Dict[str, List[TradeFeedback]]:
        """Group feedback items by category."""
        categorized = {}
        for item in batch:
            category = item.category.value if item.category else 'UNCATEGORIZED'
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(item)
        return categorized
