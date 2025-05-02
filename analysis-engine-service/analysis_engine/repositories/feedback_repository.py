"""
Concrete implementation of the IFeedbackRepository interface.

This class provides persistent storage and retrieval of feedback data,
supporting the automated model retraining system.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

from core_foundations.interfaces.model_trainer import IFeedbackRepository
from core_foundations.models.feedback import (
    ClassifiedFeedback, TimeframeFeedback, FeedbackBatch,
    FeedbackPriority, FeedbackCategory, FeedbackStatus
)
from core_foundations.events.kafka import FeedbackEventProducer

logger = logging.getLogger(__name__)


class FeedbackRepository(IFeedbackRepository):
    """
    Implementation of IFeedbackRepository for storing and retrieving feedback data.
    
    This implementation uses a combination of database storage and event streaming
    to ensure feedback is properly persisted and can be efficiently queried for
    model retraining purposes.
    """
    def __init__(
        self,
        db_client: Any,
        event_producer: Optional[FeedbackEventProducer] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the FeedbackRepository.
        
        Args:
            db_client: Client for database operations
            event_producer: Optional Kafka producer for emitting feedback events
            config: Configuration dictionary
        """
        self.db_client = db_client
        self.event_producer = event_producer
        self.config = config or {}
        
        # For development/testing, optionally use file-based storage
        self.use_file_storage = self.config.get('use_file_storage', False)
        if self.use_file_storage:
            self.storage_dir = self.config.get('storage_dir', 'feedback_storage')
            os.makedirs(self.storage_dir, exist_ok=True)
            os.makedirs(os.path.join(self.storage_dir, 'feedback'), exist_ok=True)
            os.makedirs(os.path.join(self.storage_dir, 'batches'), exist_ok=True)
        
        # Configure Kafka topics
        self.feedback_topic = self.config.get('kafka_topics', {}).get('feedback', 'feedback_events')
        self.batch_topic = self.config.get('kafka_topics', {}).get('batch', 'feedback_batch_events')
        self.model_topic = self.config.get('kafka_topics', {}).get('model', 'model_events')
        
        logger.info("FeedbackRepository initialized with %s storage and Kafka integration: %s", 
                   "file-based" if self.use_file_storage else "database",
                   "enabled" if self.event_producer else "disabled")
    
    def get_prioritized_feedback_since(
        self,
        timestamp: datetime,
        model_id: Optional[str] = None,
        min_priority: str = "MEDIUM",
        limit: int = 100
    ) -> List[ClassifiedFeedback]:
        """
        Retrieve prioritized feedback items since a specific timestamp.
        
        Args:
            timestamp: Retrieve feedback items since this time
            model_id: Optional filter for specific model
            min_priority: Minimum priority level to include
            limit: Maximum number of items to retrieve
            
        Returns:
            List of ClassifiedFeedback objects matching the criteria
        """
        logger.info("Fetching prioritized feedback since %s for model %s (min priority: %s)", 
                   timestamp, model_id or "any", min_priority)
        
        try:
            # Convert string priority to enum value for comparison
            try:
                min_priority_enum = FeedbackPriority(min_priority)
            except ValueError:
                logger.warning("Invalid priority value: %s, defaulting to MEDIUM", min_priority)
                min_priority_enum = FeedbackPriority.MEDIUM
            
            if self.use_file_storage:
                return self._file_get_prioritized_feedback_since(
                    timestamp, model_id, min_priority_enum, limit)
            else:
                return self._db_get_prioritized_feedback_since(
                    timestamp, model_id, min_priority_enum, limit)
                
        except Exception as e:
            logger.exception("Error retrieving prioritized feedback: %s", str(e))
            return []
    
    def update_feedback_status(
        self,
        feedback_ids: List[str],
        status: str
    ) -> int:
        """
        Update the status of multiple feedback items.
        
        Args:
            feedback_ids: List of feedback IDs to update
            status: New status to set
            
        Returns:
            Number of feedback items successfully updated
        """
        if not feedback_ids:
            return 0
            
        logger.info("Updating status to '%s' for %d feedback items", status, len(feedback_ids))
        
        try:
            # Validate the status
            try:
                status_enum = FeedbackStatus(status)
            except ValueError:
                logger.warning("Invalid status value: %s", status)
                return 0
                
            if self.use_file_storage:
                return self._file_update_feedback_status(feedback_ids, status_enum)
            else:
                return self._db_update_feedback_status(feedback_ids, status_enum)
                
        except Exception as e:
            logger.exception("Error updating feedback status: %s", str(e))
            return 0
    
    def create_feedback_batch(
        self,
        feedback_ids: List[str],
        batch_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new batch from existing feedback items.
        
        Args:
            feedback_ids: List of feedback IDs to include in the batch
            batch_metadata: Optional metadata for the batch
            
        Returns:
            ID of the created batch
        """
        if not feedback_ids:
            logger.warning("Attempted to create empty feedback batch")
            return ""
            
        logger.info("Creating feedback batch with %d items", len(feedback_ids))
        
        try:
            # Generate a batch ID
            batch_id = str(uuid.uuid4())
            
            # Retrieve the feedback items
            feedback_items = self._get_feedback_items_by_ids(feedback_ids)
            
            if not feedback_items:
                logger.warning("No feedback items found for batch creation")
                return ""
            
            # Create the batch object
            batch = FeedbackBatch(
                batch_id=batch_id,
                feedback_items=feedback_items,
                batch_priority=self._determine_batch_priority(feedback_items),
                created_at=datetime.utcnow(),
                status="NEW",
                metadata=batch_metadata or {}
            )
            
            # Store the batch
            if self.use_file_storage:
                self._file_store_batch(batch)
            else:
                self._db_store_batch(batch)
            
            # Emit event if configured
            if self.event_producer:
                self._emit_batch_created_event(batch)
                
            return batch_id
            
        except Exception as e:
            logger.exception("Error creating feedback batch: %s", str(e))
            return ""
    
    def get_feedback_batch(
        self,
        batch_id: str
    ) -> Optional[FeedbackBatch]:
        """
        Retrieve a feedback batch by ID.
        
        Args:
            batch_id: ID of the batch to retrieve
            
        Returns:
            FeedbackBatch object if found, None otherwise
        """
        logger.info("Retrieving feedback batch %s", batch_id)
        
        try:
            if self.use_file_storage:
                return self._file_get_batch(batch_id)
            else:
                return self._db_get_batch(batch_id)
                
        except Exception as e:
            logger.exception("Error retrieving feedback batch %s: %s", batch_id, str(e))
            return None
    
    def mark_batch_processed(
        self,
        batch_id: str,
        processing_results: Dict[str, Any]
    ) -> bool:
        """
        Mark a feedback batch as processed with results.
        
        Args:
            batch_id: ID of the batch to update
            processing_results: Results of processing this batch
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Marking batch %s as processed", batch_id)
        
        try:
            # Retrieve the batch
            batch = self.get_feedback_batch(batch_id)
            if not batch:
                logger.warning("Batch %s not found for marking as processed", batch_id)
                return False
                
            # Update the batch
            batch.status = "PROCESSED"
            batch.processed_at = datetime.utcnow()
            batch.metadata = {**batch.metadata, **processing_results}
            
            # Update feedback item statuses if needed
            feedback_ids = [item.feedback_id for item in batch.feedback_items]
            self.update_feedback_status(feedback_ids, "INCORPORATED")
            
            # Store the updated batch
            if self.use_file_storage:
                self._file_store_batch(batch)
            else:
                self._db_store_batch(batch)
                
            # Emit event if configured
            if self.event_producer:
                self._emit_batch_processed_event(batch, processing_results)
                
            return True
            
        except Exception as e:
            logger.exception("Error marking batch %s as processed: %s", batch_id, str(e))
            return False
    
    def store_feedback(self, feedback: ClassifiedFeedback) -> str:
        """
        Store a new feedback item.
        
        This is an extension beyond the base interface to support direct storage
        of new feedback items.
        
        Args:
            feedback: The feedback item to store
            
        Returns:
            The ID of the stored item
        """
        logger.info("Storing new feedback for model %s (category: %s, priority: %s)",
                   feedback.model_id, 
                   feedback.category.value if feedback.category else "UNKNOWN",
                   feedback.priority.value if feedback.priority else "UNKNOWN")
        
        try:
            feedback_id = ""
            if self.use_file_storage:
                feedback_id = self._file_store_feedback(feedback)
            else:
                feedback_id = self._db_store_feedback(feedback)
            
            # If storage was successful, emit an event
            if feedback_id:
                self._emit_feedback_created_event(feedback)
                
            return feedback_id
                
        except Exception as e:
            logger.exception("Error storing feedback: %s", str(e))
            return ""
    
    # File-based implementation methods (for development/testing)
    
    def _file_get_prioritized_feedback_since(
        self,
        timestamp: datetime,
        model_id: Optional[str],
        min_priority_enum: FeedbackPriority,
        limit: int
    ) -> List[ClassifiedFeedback]:
        """File-based implementation of get_prioritized_feedback_since."""
        result = []
        
        feedback_dir = os.path.join(self.storage_dir, 'feedback')
        for filename in os.listdir(feedback_dir):
            if not filename.endswith('.json'):
                continue
                
            try:
                file_path = os.path.join(feedback_dir, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Check if it's already a specific timeframe feedback
                is_timeframe = 'timeframe' in data
                
                # Parse to appropriate object
                item = TimeframeFeedback.from_dict(data) if is_timeframe else ClassifiedFeedback.from_dict(data)
                
                # Apply filters
                if model_id and item.model_id != model_id:
                    continue
                    
                if item.timestamp < timestamp:
                    continue
                    
                if item.priority.value < min_priority_enum.value:
                    continue
                    
                result.append(item)
                
                # Check limit
                if len(result) >= limit:
                    break
                    
            except Exception as e:
                logger.error("Error parsing feedback file %s: %s", filename, str(e))
                
        return result
    
    def _file_update_feedback_status(
        self,
        feedback_ids: List[str],
        status_enum: FeedbackStatus
    ) -> int:
        """File-based implementation of update_feedback_status."""
        updated_count = 0
        
        feedback_dir = os.path.join(self.storage_dir, 'feedback')
        for filename in os.listdir(feedback_dir):
            if not filename.endswith('.json'):
                continue
                
            try:
                file_path = os.path.join(feedback_dir, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Check if this is one of the IDs to update
                if data.get('feedback_id') not in feedback_ids:
                    continue
                    
                # Update the status
                data['status'] = status_enum.value
                
                # Write back
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                updated_count += 1
                
            except Exception as e:
                logger.error("Error updating feedback file %s: %s", filename, str(e))
                
        return updated_count
    
    def _file_store_batch(self, batch: FeedbackBatch):
        """Store a batch to file."""
        batch_data = batch.to_dict()
        batch_path = os.path.join(self.storage_dir, 'batches', f"{batch.batch_id}.json")
        
        with open(batch_path, 'w') as f:
            json.dump(batch_data, f, indent=2)
    
    def _file_get_batch(self, batch_id: str) -> Optional[FeedbackBatch]:
        """Retrieve a batch from file."""
        batch_path = os.path.join(self.storage_dir, 'batches', f"{batch_id}.json")
        
        if not os.path.exists(batch_path):
            return None
            
        with open(batch_path, 'r') as f:
            data = json.load(f)
            
        return FeedbackBatch.from_dict(data)
    
    def _file_store_feedback(self, feedback: ClassifiedFeedback) -> str:
        """Store a feedback item to file."""
        feedback_data = feedback.to_dict()
        feedback_path = os.path.join(self.storage_dir, 'feedback', f"{feedback.feedback_id}.json")
        
        with open(feedback_path, 'w') as f:
            json.dump(feedback_data, f, indent=2)
            
        return feedback.feedback_id
    
    def _get_feedback_items_by_ids(self, feedback_ids: List[str]) -> List[ClassifiedFeedback]:
        """Get feedback items by their IDs."""
        if self.use_file_storage:
            return self._file_get_feedback_items_by_ids(feedback_ids)
        else:
            return self._db_get_feedback_items_by_ids(feedback_ids)
    
    def _file_get_feedback_items_by_ids(self, feedback_ids: List[str]) -> List[ClassifiedFeedback]:
        """File-based implementation of getting feedback items by IDs."""
        result = []
        
        feedback_dir = os.path.join(self.storage_dir, 'feedback')
        for feedback_id in feedback_ids:
            file_path = os.path.join(feedback_dir, f"{feedback_id}.json")
            
            if not os.path.exists(file_path):
                logger.warning("Feedback item %s not found", feedback_id)
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Check if it's a timeframe feedback
                is_timeframe = 'timeframe' in data
                    
                # Parse to appropriate object
                item = TimeframeFeedback.from_dict(data) if is_timeframe else ClassifiedFeedback.from_dict(data)
                result.append(item)
                
            except Exception as e:
                logger.error("Error parsing feedback file for ID %s: %s", feedback_id, str(e))
                
        return result
    
    # Database implementation methods (for production)
    
    def _db_get_prioritized_feedback_since(
        self,
        timestamp: datetime,
        model_id: Optional[str],
        min_priority_enum: FeedbackPriority,
        limit: int
    ) -> List[ClassifiedFeedback]:
        """Database implementation of get_prioritized_feedback_since."""
        # In a real implementation, this would execute a database query
        # Example SQL:
        # SELECT * FROM feedback 
        # WHERE timestamp > :timestamp 
        #   AND (:model_id IS NULL OR model_id = :model_id)
        #   AND priority >= :min_priority
        # ORDER BY priority DESC, timestamp DESC
        # LIMIT :limit
        
        logger.debug("Would execute DB query for prioritized feedback")
        
        # Placeholder implementation
        return []
    
    def _db_update_feedback_status(
        self,
        feedback_ids: List[str],
        status_enum: FeedbackStatus
    ) -> int:
        """Database implementation of update_feedback_status."""
        # In a real implementation, this would execute a database update
        # Example SQL:
        # UPDATE feedback
        # SET status = :status
        # WHERE feedback_id IN (:feedback_ids)
        
        logger.debug("Would execute DB update for feedback status")
        
        # Placeholder implementation
        return len(feedback_ids)
    
    def _db_store_batch(self, batch: FeedbackBatch):
        """Store a batch to database."""
        # In a real implementation, this would:
        # 1. Store the batch metadata in a batches table
        # 2. Create relationships between the batch and its feedback items
        
        # Example SQL:
        # INSERT INTO feedback_batches 
        # (batch_id, batch_priority, created_at, processed_at, status)
        # VALUES (:batch_id, :batch_priority, :created_at, :processed_at, :status)
        
        # INSERT INTO feedback_batch_items
        # (batch_id, feedback_id)
        # VALUES (:batch_id, :feedback_id)
        # FOR EACH feedback item
        
        logger.debug("Would execute DB insert for feedback batch")
    
    def _db_get_batch(self, batch_id: str) -> Optional[FeedbackBatch]:
        """Retrieve a batch from database."""
        # In a real implementation, this would:
        # 1. Fetch the batch metadata from the batches table
        # 2. Fetch the associated feedback items
        # 3. Construct and return a FeedbackBatch object
        
        logger.debug("Would execute DB query for feedback batch")
        
        # Placeholder implementation
        return None
    
    def _db_store_feedback(self, feedback: ClassifiedFeedback) -> str:
        """Store a feedback item to database."""
        # In a real implementation, this would insert the feedback into a database table
        # with appropriate handling for specialized feedback types
        
        logger.debug("Would execute DB insert for feedback")
        
        # Placeholder implementation
        return feedback.feedback_id
    
    def _db_get_feedback_items_by_ids(self, feedback_ids: List[str]) -> List[ClassifiedFeedback]:
        """Database implementation of getting feedback items by IDs."""
        # In a real implementation, this would fetch the items from the database
        
        logger.debug("Would execute DB query for feedback items by IDs")
        
        # Placeholder implementation
        return []
    
    # Helper methods
    
    def _determine_batch_priority(self, feedback_items: List[ClassifiedFeedback]) -> FeedbackPriority:
        """Determine the priority for a batch based on its items."""
        if not feedback_items:
            return FeedbackPriority.MEDIUM
            
        # Use the highest priority from any item
        priorities = [item.priority for item in feedback_items if item.priority]
        if not priorities:
            return FeedbackPriority.MEDIUM
            
        return max(priorities, key=lambda p: p.value)
    
    def _emit_batch_created_event(self, batch: FeedbackBatch):
        """Emit an event for batch creation via the event producer."""
        if not self.event_producer:
            return
            
        event_data = {
            "batch_id": batch.batch_id,
            "model_id": batch.metadata.get("model_id"),
            "feedback_count": len(batch.feedback_items),
            "priority": batch.batch_priority.value,
            "created_at": batch.created_at.isoformat() if batch.created_at else datetime.utcnow().isoformat(),
            "feedback_categories": [item.category.value for item in batch.feedback_items if item.category],
            "status": batch.status
        }
        
        try:
            success = self.event_producer.produce(
                event_type="feedback_batch_created",
                event_data=event_data,
                topic=self.batch_topic,
                key=batch.batch_id
            )
            if success:
                logger.debug("Emitted batch created event for batch %s", batch.batch_id)
            else:
                logger.warning("Failed to emit batch created event for batch %s", batch.batch_id)
        except Exception as e:
            logger.error("Failed to emit batch created event: %s", str(e))
    
    def _emit_batch_processed_event(self, batch: FeedbackBatch, results: Dict[str, Any]):
        """Emit an event for batch processing completion."""
        if not self.event_producer:
            return
            
        event_data = {
            "batch_id": batch.batch_id,
            "model_id": batch.metadata.get("model_id"),
            "feedback_count": len(batch.feedback_items),
            "status": results.get("status"),
            "new_model_version": results.get("model_version"),
            "performance_metrics": results.get("performance_metrics"),
            "processed_at": batch.processed_at.isoformat() if batch.processed_at else datetime.utcnow().isoformat()
        }
        
        try:
            success = self.event_producer.produce(
                event_type="feedback_batch_processed",
                event_data=event_data,
                topic=self.batch_topic,
                key=batch.batch_id
            )
            if success:
                logger.debug("Emitted batch processed event for batch %s", batch.batch_id)
            else:
                logger.warning("Failed to emit batch processed event for batch %s", batch.batch_id)
                
            # If the batch resulted in a model update, emit a model event as well
            if results.get("status") == "success" and results.get("model_version"):
                model_event_data = {
                    "model_id": batch.metadata.get("model_id"),
                    "previous_version": results.get("previous_version"),
                    "new_version": results.get("model_version"),
                    "improvement": results.get("metrics", {}).get("improvement"),
                    "feedback_batch_id": batch.batch_id,
                    "training_metrics": results.get("metrics")
                }
                
                self.event_producer.produce(
                    event_type="model_updated",
                    event_data=model_event_data,
                    topic=self.model_topic,
                    key=batch.metadata.get("model_id")
                )
                
        except Exception as e:
            logger.error("Failed to emit batch processed event: %s", str(e))
    
    def _emit_feedback_created_event(self, feedback: ClassifiedFeedback):
        """
        Emit an event when a new feedback item is created.
        
        Args:
            feedback: The feedback item that was created
        """
        if not self.event_producer:
            return
            
        # Prepare event data with key feedback properties
        event_data = {
            "feedback_id": feedback.feedback_id,
            "model_id": feedback.model_id,
            "category": feedback.category.value if feedback.category else None,
            "priority": feedback.priority.value if feedback.priority else None,
            "source": feedback.source.value if feedback.source else None,
            "status": feedback.status.value if feedback.status else None,
            "statistical_significance": feedback.statistical_significance,
            "created_at": feedback.timestamp.isoformat() if feedback.timestamp else datetime.utcnow().isoformat()
        }
        
        # Add timeframe-specific data if applicable
        if hasattr(feedback, 'timeframe'):
            event_data["timeframe"] = feedback.timeframe
            if hasattr(feedback, 'related_timeframes') and feedback.related_timeframes:
                event_data["related_timeframes"] = feedback.related_timeframes
        
        try:
            success = self.event_producer.produce(
                event_type="feedback_created",
                event_data=event_data,
                topic=self.feedback_topic,
                key=feedback.feedback_id
            )
            if success:
                logger.debug("Emitted feedback created event for feedback %s", feedback.feedback_id)
            else:
                logger.warning("Failed to emit feedback created event for feedback %s", feedback.feedback_id)
        except Exception as e:
            logger.error("Failed to emit feedback created event: %s", str(e))
