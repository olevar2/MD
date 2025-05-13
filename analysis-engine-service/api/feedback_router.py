"""
Feedback Routing System

This module implements a structured feedback routing system that directs feedback
to appropriate services based on its type, content, and significance.
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable
import logging
import asyncio
from datetime import datetime
from core_foundations.models.feedback import TradeFeedback, FeedbackSource, FeedbackCategory, FeedbackStatus
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class FeedbackRouter:
    """
    The FeedbackRouter directs feedback to appropriate downstream systems
    and services based on feedback type, content, and significance.
    
    Key capabilities:
    - Route feedback to appropriate services based on configurable rules
    - Batch similar feedback for efficient processing
    - Apply priority-based routing for high-impact feedback
    - Track routing history and outcomes
    """

    def __init__(self, event_bus: Any=None, config: Dict[str, Any]=None):
        """
        Initialize the FeedbackRouter.
        
        Args:
            event_bus: The event bus for publishing routing events
            config: Configuration parameters
        """
        self.event_bus = event_bus
        self.config = config or {}
        self._set_default_config()
        self.route_handlers = {}
        self.batch_queues = {}
        self.route_history = []
        self.max_history_size = self.config_manager.get('max_history_size', 1000)
        logger.info('FeedbackRouter initialized')

    def _set_default_config(self):
        """Set default configuration parameters."""
        defaults = {'batch_size': 10, 'batch_timeout': 60,
            'priority_threshold': 0.8, 'max_history_size': 1000,
            'route_timeout': 5.0}
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    def register_route_handler(self, source: Optional[str]=None, category:
        Optional[str]=None, handler: Callable[[TradeFeedback], Awaitable[
        bool]]=None):
        """
        Register a handler function for a specific feedback source and category.
        
        Args:
            source: The feedback source to handle (None for any)
            category: The feedback category to handle (None for any)
            handler: Async function that processes the feedback and returns success status
        """
        if handler is None:
            logger.error('Cannot register None as a route handler')
            return
        route_key = source, category
        if route_key not in self.route_handlers:
            self.route_handlers[route_key] = []
        self.route_handlers[route_key].append(handler)
        logger.info('Registered route handler for source=%s, category=%s', 
            source if source else 'any', category if category else 'any')

    async def route_feedback(self, feedback: TradeFeedback) ->bool:
        """
        Route a feedback item to appropriate handlers based on its characteristics.
        
        Args:
            feedback: The feedback to route
            
        Returns:
            bool: True if routing was successful, False otherwise
        """
        priority_score = self._calculate_priority(feedback)
        feedback.status = FeedbackStatus.PROCESSING
        if priority_score >= self.config['priority_threshold']:
            success = await self._process_single_feedback(feedback)
            self._record_route(feedback, success, priority_score, batched=False
                )
            return success
        batch_key = self._get_batch_key(feedback)
        if batch_key not in self.batch_queues:
            self.batch_queues[batch_key] = {'items': [], 'last_updated':
                datetime.utcnow()}
        self.batch_queues[batch_key]['items'].append(feedback)
        self.batch_queues[batch_key]['last_updated'] = datetime.utcnow()
        if len(self.batch_queues[batch_key]['items']) >= self.config[
            'batch_size']:
            await self._process_batch(batch_key)
        return True

    @with_resilience('process_pending_batches')
    async def process_pending_batches(self):
        """
        Process any pending batches that have reached timeout.
        This should be called periodically to ensure batches are processed.
        """
        now = datetime.utcnow()
        timeout = self.config['batch_timeout']
        expired_batches = [key for key, data in self.batch_queues.items() if
            (now - data['last_updated']).total_seconds() >= timeout and len
            (data['items']) > 0]
        for batch_key in expired_batches:
            await self._process_batch(batch_key)

    @async_with_exception_handling
    async def _process_batch(self, batch_key):
        """
        Process a batch of feedback items.
        
        Args:
            batch_key: The key identifying the batch
        """
        if batch_key not in self.batch_queues:
            return
        batch = self.batch_queues[batch_key]['items']
        if not batch:
            return
        source, category = batch_key
        handlers = self._get_handlers(source, category)
        if not handlers:
            handlers = self._get_handlers(None, None)
        if not handlers:
            logger.warning('No handlers found for batch %s', batch_key)
            return
        results = []
        for handler in handlers:
            try:
                try:
                    batch_result = await asyncio.wait_for(handler(batch),
                        timeout=self.config['route_timeout'])
                    results.append(batch_result)
                except TypeError:
                    batch_results = []
                    for item in batch:
                        item_result = await asyncio.wait_for(handler(item),
                            timeout=self.config['route_timeout'])
                        batch_results.append(item_result)
                    results.append(all(batch_results))
            except asyncio.TimeoutError:
                logger.error('Handler timed out processing batch %s', batch_key
                    )
                results.append(False)
            except Exception as e:
                logger.error('Error processing batch %s: %s', batch_key, str(e)
                    )
                results.append(False)
        success = all(results)
        for item in batch:
            item.status = (FeedbackStatus.ROUTED if success else
                FeedbackStatus.ERROR)
        avg_priority = sum(self._calculate_priority(item) for item in batch
            ) / len(batch)
        self._record_route(batch, success, avg_priority, batched=True)
        if self.event_bus:
            try:
                await self.event_bus.publish('feedback.batch_routed', {
                    'batch_key': str(batch_key), 'size': len(batch),
                    'success': success, 'timestamp': datetime.utcnow().
                    isoformat()})
            except Exception as e:
                logger.error(f'Failed to publish batch routing event: {str(e)}'
                    )
        self.batch_queues[batch_key]['items'] = []
        self.batch_queues[batch_key]['last_updated'] = datetime.utcnow()
        logger.info('Processed batch of %d items for %s', len(batch), batch_key
            )

    @async_with_exception_handling
    async def _process_single_feedback(self, feedback: TradeFeedback) ->bool:
        """
        Process a single high-priority feedback item.
        
        Args:
            feedback: The feedback to process
            
        Returns:
            bool: True if processing was successful
        """
        source = feedback.source.value if hasattr(feedback.source, 'value'
            ) else feedback.source
        category = feedback.category.value if hasattr(feedback.category,
            'value') else feedback.category
        handlers = self._get_handlers(source, category)
        if not handlers:
            handlers = self._get_handlers(None, None)
        if not handlers:
            logger.warning(
                'No handlers found for feedback source=%s, category=%s',
                source, category)
            return False
        results = []
        for handler in handlers:
            try:
                result = await asyncio.wait_for(handler(feedback), timeout=
                    self.config['route_timeout'])
                results.append(result)
            except asyncio.TimeoutError:
                logger.error('Handler timed out processing feedback %s',
                    feedback.feedback_id)
                results.append(False)
            except Exception as e:
                logger.error('Error processing feedback %s: %s', feedback.
                    feedback_id, str(e))
                results.append(False)
        success = all(results)
        feedback.status = (FeedbackStatus.ROUTED if success else
            FeedbackStatus.ERROR)
        if self.event_bus:
            try:
                await self.event_bus.publish('feedback.routed', {
                    'feedback_id': feedback.feedback_id, 'source': source,
                    'category': category, 'success': success, 'timestamp':
                    datetime.utcnow().isoformat()})
            except Exception as e:
                logger.error(
                    f'Failed to publish feedback routing event: {str(e)}')
        return success

    def _calculate_priority(self, feedback: TradeFeedback) ->float:
        """
        Calculate priority score for a feedback item.
        
        Args:
            feedback: The feedback to calculate priority for
            
        Returns:
            float: Priority score between 0.0 and 1.0
        """
        priority = 0.5
        if hasattr(feedback, 'tags'):
            for tag in feedback.tags:
                tag_value = tag.value if hasattr(tag, 'value') else tag
                if tag_value == 'high_impact':
                    priority += 0.3
                elif tag_value == 'requires_attention':
                    priority += 0.2
                elif tag_value == 'anomaly':
                    priority += 0.15
                elif tag_value == 'trending':
                    priority += 0.1
        if feedback.source:
            source = feedback.source.value if hasattr(feedback.source, 'value'
                ) else feedback.source
            if source == 'risk_management':
                priority += 0.15
        if feedback.category:
            category = feedback.category.value if hasattr(feedback.category,
                'value') else feedback.category
            if category in ('warning', 'error'):
                priority += 0.2
        return min(priority, 1.0)

    def _get_batch_key(self, feedback: TradeFeedback) ->tuple:
        """
        Get the key for batching feedback.
        
        Args:
            feedback: The feedback to get key for
            
        Returns:
            tuple: Batch key (source, category)
        """
        source = feedback.source.value if hasattr(feedback.source, 'value'
            ) else feedback.source
        category = feedback.category.value if hasattr(feedback.category,
            'value') else feedback.category
        return source, category

    def _get_handlers(self, source: Optional[str], category: Optional[str]
        ) ->List[Callable]:
        """
        Get handlers for a specific source and category.
        
        Args:
            source: The feedback source
            category: The feedback category
            
        Returns:
            List[Callable]: List of handler functions
        """
        handlers = []
        if (source, category) in self.route_handlers:
            handlers.extend(self.route_handlers[source, category])
        if (source, None) in self.route_handlers:
            handlers.extend(self.route_handlers[source, None])
        if (None, category) in self.route_handlers:
            handlers.extend(self.route_handlers[None, category])
        return handlers

    def _record_route(self, feedback: (TradeFeedback or List[TradeFeedback]
        ), success: bool, priority: float, batched: bool):
        """
        Record a routing operation in the history.
        
        Args:
            feedback: The feedback(s) that was routed
            success: Whether routing was successful
            priority: The priority score of the feedback
            batched: Whether this was a batch operation
        """
        record = {'timestamp': datetime.utcnow(), 'success': success,
            'priority': priority, 'batched': batched}
        if batched:
            record['batch_size'] = len(feedback)
            record['sources'] = list(set(f.source.value if hasattr(f.source,
                'value') else f.source for f in feedback))
            record['categories'] = list(set(f.category.value if hasattr(f.
                category, 'value') else f.category for f in feedback))
            record['feedback_ids'] = [f.feedback_id for f in feedback]
        else:
            record['feedback_id'] = feedback.feedback_id
            record['source'] = feedback.source.value if hasattr(feedback.
                source, 'value') else feedback.source
            record['category'] = feedback.category.value if hasattr(feedback
                .category, 'value') else feedback.category
        self.route_history.append(record)
        if len(self.route_history) > self.max_history_size:
            self.route_history = self.route_history[-self.max_history_size:]

    @with_resilience('get_route_stats')
    def get_route_stats(self) ->Dict[str, Any]:
        """
        Get statistics about routing operations.
        
        Returns:
            Dict[str, Any]: Routing statistics
        """
        if not self.route_history:
            return {'total_count': 0, 'success_rate': 0.0, 'avg_priority': 
                0.0, 'batch_percentage': 0.0, 'by_source': {},
                'by_category': {}}
        total = len(self.route_history)
        successful = sum(1 for r in self.route_history if r['success'])
        batched = sum(1 for r in self.route_history if r['batched'])
        source_stats = {}
        category_stats = {}
        for record in self.route_history:
            if 'sources' in record:
                for source in record['sources']:
                    if source not in source_stats:
                        source_stats[source] = {'count': 0, 'successful': 0}
                    source_stats[source]['count'] += 1
                    if record['success']:
                        source_stats[source]['successful'] += 1
            elif 'source' in record:
                source = record['source']
                if source not in source_stats:
                    source_stats[source] = {'count': 0, 'successful': 0}
                source_stats[source]['count'] += 1
                if record['success']:
                    source_stats[source]['successful'] += 1
            if 'categories' in record:
                for category in record['categories']:
                    if category not in category_stats:
                        category_stats[category] = {'count': 0, 'successful': 0
                            }
                    category_stats[category]['count'] += 1
                    if record['success']:
                        category_stats[category]['successful'] += 1
            elif 'category' in record:
                category = record['category']
                if category not in category_stats:
                    category_stats[category] = {'count': 0, 'successful': 0}
                category_stats[category]['count'] += 1
                if record['success']:
                    category_stats[category]['successful'] += 1
        for stats in source_stats.values():
            stats['success_rate'] = stats['successful'] / stats['count'
                ] if stats['count'] > 0 else 0.0
        for stats in category_stats.values():
            stats['success_rate'] = stats['successful'] / stats['count'
                ] if stats['count'] > 0 else 0.0
        return {'total_count': total, 'success_rate': successful / total if
            total > 0 else 0.0, 'avg_priority': sum(r['priority'] for r in
            self.route_history) / total if total > 0 else 0.0,
            'batch_percentage': batched / total * 100 if total > 0 else 0.0,
            'by_source': source_stats, 'by_category': category_stats}

    def clear_history(self):
        """Clear the routing history."""
        self.route_history = []
        logger.info('Routing history cleared')
