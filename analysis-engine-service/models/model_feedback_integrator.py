"""
Model Feedback Integration

This module implements the integration between trading feedback and model training,
enabling automated model updates based on trading outcomes and creating the
bidirectional loop between strategy execution and the ML components.
"""
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime, timedelta
import asyncio
import uuid
from core_foundations.models.feedback import TradeFeedback, FeedbackSource, FeedbackCategory
from core_foundations.utils.logger import get_logger
from core_foundations.events.event_publisher import EventPublisher
from core_foundations.events.event_schema import EventType
from core_foundations.exceptions.feedback_exceptions import ModelRetrainingError
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ModelFeedbackIntegrator:
    """
    The ModelFeedbackIntegrator enables the bidirectional loop between trading outcomes
    and model training by collecting, aggregating, and applying trading feedback
    to improve model performance.
    
    Key capabilities:
    - Collect model performance feedback from trading outcomes
    - Aggregate feedback to identify model weaknesses
    - Schedule and trigger model retraining when needed
    - Track model performance improvements over time
    - Manage the feedback-training cycle
    """

    def __init__(self, model_service: Any=None, event_publisher: Optional[
        EventPublisher]=None, database: Any=None, config: Dict[str, Any]=None):
        """
        Initialize the ModelFeedbackIntegrator.
        
        Args:
            model_service: Service for interacting with ML models (e.g., triggering training, getting metrics).
            event_publisher: Event publisher for broadcasting model lifecycle events.
            database: Database interface for storing retraining history (optional).
            config: Configuration parameters.
        """
        self.model_service = model_service
        self.event_publisher = event_publisher
        self.database = database
        self.config = config or {}
        self._set_default_config()
        self.feedback_by_model = {}
        self.retraining_history = []
        self.performance_improvements = {}
        self.active_retraining_tasks = {}
        logger.info('ModelFeedbackIntegrator initialized')

    def _set_default_config(self):
        """Set default configuration parameters."""
        defaults = {'min_feedback_samples': 20, 'retraining_cooldown': 
            86400, 'error_threshold': 0.02, 'error_improvement_threshold': 
            0.01, 'max_retrain_attempts': 3, 'performance_window': 30,
            'batch_size': 50}
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    async def collect_model_feedback(self, feedback: TradeFeedback) ->bool:
        """
        Collect and process feedback for a specific model.
        
        Args:
            feedback: Feedback to process
            
        Returns:
            bool: Whether feedback was successfully collected
        """
        is_model_feedback = (feedback.source == FeedbackSource.
            MODEL_PREDICTION or feedback.source == FeedbackSource.
            STRATEGY_EXECUTION and feedback.model_id)
        if not is_model_feedback or not feedback.model_id:
            logger.debug('Feedback not related to model training, skipping')
            return False
        model_id = feedback.model_id
        if model_id not in self.feedback_by_model:
            self.feedback_by_model[model_id] = {'recent_feedback': [],
                'error_trends': [], 'last_retrained': None,
                'retrain_attempts': 0, 'instrument': feedback.instrument,
                'timeframe': feedback.timeframe}
        self.feedback_by_model[model_id]['recent_feedback'].append(feedback)
        max_recent = self.config_manager.get('max_recent_feedback', 1000)
        if len(self.feedback_by_model[model_id]['recent_feedback']
            ) > max_recent:
            self.feedback_by_model[model_id]['recent_feedback'
                ] = self.feedback_by_model[model_id]['recent_feedback'][-
                max_recent:]
        if (feedback.source == FeedbackSource.MODEL_PREDICTION and 'error' in
            feedback.outcome_metrics):
            error = feedback.outcome_metrics['error']
            self.feedback_by_model[model_id]['error_trends'].append({
                'timestamp': feedback.timestamp, 'error': error})
            max_trends = self.config_manager.get('max_error_trends', 1000)
            if len(self.feedback_by_model[model_id]['error_trends']
                ) > max_trends:
                self.feedback_by_model[model_id]['error_trends'
                    ] = self.feedback_by_model[model_id]['error_trends'][-
                    max_trends:]
        should_retrain = await self._check_retraining_needed(model_id)
        if should_retrain:
            if model_id not in self.active_retraining_tasks:
                logger.info(f'Scheduling retraining task for model {model_id}')
                task = asyncio.create_task(self._retrain_model_workflow(
                    model_id))
                self.active_retraining_tasks[model_id] = task
                task.add_done_callback(lambda t, mid=model_id: self.
                    _handle_retraining_completion(mid, t))
            else:
                logger.debug(
                    f'Retraining task already active for model {model_id}, skipping new schedule.'
                    )
        return True

    @with_exception_handling
    def _handle_retraining_completion(self, model_id: str, task: asyncio.Task):
        """Callback executed when a retraining task finishes."""
        logger.debug(
            f'Retraining task finished for model {model_id}. Removing from active tasks.'
            )
        self.active_retraining_tasks.pop(model_id, None)
        try:
            result = task.result()
            logger.info(
                f'Retraining task for model {model_id} completed with result: {result}'
                )
        except asyncio.CancelledError:
            logger.warning(
                f'Retraining task for model {model_id} was cancelled.')
        except Exception as e:
            logger.error(
                f'Retraining task for model {model_id} failed with exception: {e}'
                , exc_info=True)

    @async_with_exception_handling
    async def _check_retraining_needed(self, model_id: str) ->bool:
        """
        Check if a model needs to be retrained based on collected feedback.
        
        Args:
            model_id: The model to check
            
        Returns:
            bool: Whether retraining is needed
        """
        if model_id not in self.feedback_by_model:
            return False
        model_data = self.feedback_by_model[model_id]
        if len(model_data['recent_feedback']) < self.config[
            'min_feedback_samples']:
            return False
        if model_data['last_retrained']:
            time_since_last = datetime.utcnow() - model_data['last_retrained']
            if time_since_last.total_seconds() < self.config[
                'retraining_cooldown']:
                return False
        if model_data['retrain_attempts'] >= self.config['max_retrain_attempts'
            ]:
            logger.warning(
                f"Model {model_id} reached max retrain attempts ({model_data['retrain_attempts']}), requires human review."
                )
            if self.event_publisher:
                try:
                    await self.event_publisher.publish(EventType.
                        MODEL_NEEDS_REVIEW, {'model_id': model_id,
                        'attempts': model_data['retrain_attempts'],
                        'reason': 'max_retrains_reached', 'timestamp':
                        datetime.utcnow().isoformat()})
                except Exception as e:
                    logger.error(
                        f'Failed to publish MODEL_NEEDS_REVIEW event for {model_id}: {e}'
                        )
            return False
        if not model_data['error_trends']:
            return False
        recent_period = datetime.utcnow() - timedelta(days=self.config.get(
            'trend_window_days', 3))
        recent_errors = [e['error'] for e in model_data['error_trends'] if 
            e['timestamp'] > recent_period]
        if not recent_errors:
            return False
        avg_recent_error = sum(recent_errors) / len(recent_errors)
        if avg_recent_error > self.config['error_threshold']:
            logger.info(
                f'Model {model_id} error {avg_recent_error:.4f} exceeds threshold, retraining needed'
                )
            return True
        if len(recent_errors) >= 10:
            half = len(recent_errors) // 2
            first_half = recent_errors[:half]
            second_half = recent_errors[half:]
            first_avg = sum(first_half) / len(first_half) if first_half else 0
            second_avg = sum(second_half) / len(second_half
                ) if second_half else 0
            if second_avg > first_avg * 1.25:
                logger.info(
                    f'Model {model_id} shows increasing error trend, retraining needed'
                    )
                return True
        return False

    @async_with_exception_handling
    async def _retrain_model_workflow(self, model_id: str) ->bool:
        """
        Orchestrates the model retraining process, including data prep, training,
        evaluation, and potential deployment/reversion.
        
        Args:
            model_id: The model to retrain.
            
        Returns:
            bool: True if the retrained model was successfully deployed, False otherwise.
        """
        if model_id not in self.feedback_by_model or not self.model_service:
            logger.error(
                f'Cannot retrain model {model_id}: Missing data or model service.'
                )
            return False
        model_data = self.feedback_by_model[model_id]
        model_data['retrain_attempts'] += 1
        model_data['last_retrained'] = datetime.utcnow()
        current_attempt = model_data['retrain_attempts']
        logger.info(
            f'Starting retraining workflow for model {model_id} (Attempt {current_attempt})'
            )
        session_id = str(uuid.uuid4())
        session = {'session_id': session_id, 'model_id': model_id,
            'start_time': datetime.utcnow(), 'status': 'started', 'attempt':
            current_attempt}
        self.retraining_history.append(session)
        await self._publish_event(EventType.MODEL_RETRAINING_STARTED, {
            'session_id': session_id, 'model_id': model_id, 'attempt':
            current_attempt, 'timestamp': session['start_time'].isoformat()})
        try:
            logger.debug(
                f'[{session_id}] Getting baseline metrics for model {model_id}'
                )
            baseline_metrics = await self._get_model_metrics(model_id,
                version='active')
            session['baseline_metrics'] = baseline_metrics
            logger.info(
                f'[{session_id}] Baseline metrics for {model_id}: {baseline_metrics}'
                )
            logger.debug(
                f'[{session_id}] Preparing training data for model {model_id}')
            training_data = await self._prepare_training_data(model_id)
            if not training_data:
                raise ModelRetrainingError(
                    f'Failed to prepare training data for model {model_id}')
            session['samples_used'] = len(training_data)
            logger.info(
                f'[{session_id}] Prepared {len(training_data)} training samples for {model_id}'
                )
            logger.debug(
                f'[{session_id}] Triggering retraining via model service for {model_id}'
                )
            session['status'] = 'training'
            retraining_result = await self.model_service.retrain_model(model_id
                =model_id, training_data=training_data, session_id=
                session_id, instrument=model_data.get('instrument'),
                timeframe=model_data.get('timeframe'))
            if not retraining_result or not retraining_result.get('success'):
                error_msg = retraining_result.get('error',
                    'Unknown error from model service')
                raise ModelRetrainingError(
                    f'Model service failed retraining for {model_id}: {error_msg}'
                    )
            new_model_version = retraining_result.get('version_id')
            if not new_model_version:
                raise ModelRetrainingError(
                    f'Model service did not return a new version ID for model {model_id}'
                    )
            session['new_model_version'] = new_model_version
            logger.info(
                f'[{session_id}] Model service completed training for {model_id}. New version: {new_model_version}'
                )
            logger.debug(
                f'[{session_id}] Evaluating new model version {new_model_version} for {model_id}'
                )
            session['status'] = 'evaluating'
            new_metrics = await self._get_model_metrics(model_id, version=
                new_model_version)
            if not new_metrics:
                raise ModelRetrainingError(
                    f'Failed to get metrics for new model version {new_model_version}'
                    )
            session['new_metrics'] = new_metrics
            logger.info(
                f'[{session_id}] Metrics for new version {new_model_version}: {new_metrics}'
                )
            improvement = self._calculate_metrics_improvement(baseline_metrics,
                new_metrics)
            session['improvement'] = improvement
            logger.info(
                f'[{session_id}] Calculated improvement for {model_id}: {improvement}'
                )
            min_improvement = self.config.get('min_error_reduction_improvement'
                , self.config['error_improvement_threshold'])
            if improvement.get('error_reduction', -1) >= min_improvement:
                logger.info(
                    f'[{session_id}] Improvement sufficient for {model_id}. Deploying version {new_model_version}.'
                    )
                session['status'] = 'deploying'
                deploy_success = await self.model_service.deploy_model_version(
                    model_id, new_model_version)
                if deploy_success:
                    session['status'] = 'completed'
                    model_data['retrain_attempts'] = 0
                    self._update_performance_tracking(model_id, improvement)
                    logger.info(
                        f'[{session_id}] Successfully deployed new version {new_model_version} for model {model_id}'
                        )
                    await self._publish_event(EventType.
                        MODEL_RETRAINING_COMPLETED, {'session_id':
                        session_id, 'model_id': model_id, 'status':
                        'completed', 'new_version': new_model_version,
                        'improvement': improvement, 'timestamp': datetime.
                        utcnow().isoformat()})
                    return True
                else:
                    raise ModelRetrainingError(
                        f'Failed to deploy new model version {new_model_version} for {model_id}'
                        )
            else:
                logger.warning(
                    f"[{session_id}] Insufficient improvement for {model_id} (Required reduction: {min_improvement}, Got: {improvement.get('error_reduction', 'N/A')}). Discarding version {new_model_version}."
                    )
                session['status'] = 'reverted'
                session['revert_reason'] = 'insufficient_improvement'
                await self.model_service.discard_model_version(model_id,
                    new_model_version)
                logger.info(
                    f'[{session_id}] Discarded new version {new_model_version} for model {model_id}'
                    )
                await self._publish_event(EventType.
                    MODEL_RETRAINING_COMPLETED, {'session_id': session_id,
                    'model_id': model_id, 'status': 'reverted', 'reason':
                    'insufficient_improvement', 'improvement': improvement,
                    'timestamp': datetime.utcnow().isoformat()})
                return False
        except ModelRetrainingError as e:
            logger.error(
                f'[{session_id}] Model retraining workflow failed for {model_id}: {e}'
                )
            session['status'] = 'failed'
            session['failure_reason'] = str(e)
            await self._publish_event(EventType.MODEL_RETRAINING_FAILED, {
                'session_id': session_id, 'model_id': model_id, 'error':
                str(e), 'timestamp': datetime.utcnow().isoformat()})
            return False
        except Exception as e:
            logger.error(
                f'[{session_id}] Unexpected error during model retraining workflow for {model_id}: {e}'
                , exc_info=True)
            session['status'] = 'failed'
            session['failure_reason'] = f'Unexpected error: {str(e)}'
            await self._publish_event(EventType.MODEL_RETRAINING_FAILED, {
                'session_id': session_id, 'model_id': model_id, 'error':
                session['failure_reason'], 'timestamp': datetime.utcnow().
                isoformat()})
            return False
        finally:
            session['end_time'] = datetime.utcnow()
            if self.database:
                try:
                    await self.database.save_retraining_session(session)
                except Exception as db_e:
                    logger.error(
                        f'[{session_id}] Failed to save retraining session details to database: {db_e}'
                        )

    @async_with_exception_handling
    async def _publish_event(self, event_type: EventType, payload: Dict[str,
        Any]):
        """Helper method to publish events with error handling."""
        if not self.event_publisher:
            return
        try:
            await self.event_publisher.publish(event_type, payload)
            logger.debug(
                f'Published event {event_type.name} with payload: {payload}')
        except Exception as e:
            logger.error(f'Failed to publish event {event_type.name}: {e}',
                exc_info=True)

    async def _prepare_training_data(self, model_id: str) ->Optional[List[
        Dict[str, Any]]]:
        """
        Prepare training data from collected feedback.
        This might involve fetching additional features based on feedback context.
        """
        if model_id not in self.feedback_by_model:
            return []
        model_data = self.feedback_by_model[model_id]
        feedback_items = model_data['recent_feedback']
        training_data = []
        for feedback in feedback_items:
            if feedback.source == FeedbackSource.MODEL_PREDICTION:
                if ('prediction_value' in feedback.outcome_metrics and 
                    'actual_value' in feedback.outcome_metrics):
                    entry = {'timestamp': feedback.timestamp, 'instrument':
                        feedback.instrument, 'timeframe': feedback.
                        timeframe, 'target': feedback.outcome_metrics[
                        'actual_value'], 'metadata': feedback.metadata}
                    if feedback.metadata and 'features' in feedback.metadata:
                        entry['features'] = feedback.metadata['features']
                        training_data.append(entry)
            elif feedback.source == FeedbackSource.STRATEGY_EXECUTION:
                if feedback.metadata and 'signal_data' in feedback.metadata:
                    entry = {'timestamp': feedback.timestamp, 'instrument':
                        feedback.instrument, 'timeframe': feedback.
                        timeframe, 'target': feedback.outcome_metrics.get(
                        'profit_loss', feedback.outcome_metrics.get(
                        'profit', 0)), 'signal_data': feedback.metadata[
                        'signal_data'], 'metadata': feedback.metadata}
                    if 'features' in feedback.metadata:
                        entry['features'] = feedback.metadata['features']
                        training_data.append(entry)
        if not training_data:
            logger.warning(
                f'No suitable training data found in recent feedback for model {model_id}'
                )
            return None
        logger.info(
            f'Prepared {len(training_data)} data points for retraining model {model_id}'
            )
        return training_data

    @async_with_exception_handling
    async def _get_model_metrics(self, model_id: str, version: str='active'
        ) ->Dict[str, Any]:
        """
        Get performance metrics for a specific model version.
        """
        if not self.model_service:
            logger.warning('Model service not available to get metrics.')
            return {}
        try:
            logger.debug(
                f'Requesting metrics for model {model_id}, version {version}')
            metrics = await self.model_service.get_model_metrics(model_id,
                version=version)
            if metrics is None:
                logger.warning(
                    f'Received null metrics for model {model_id}, version {version}'
                    )
                return {}
            logger.debug(
                f'Received metrics for model {model_id}, version {version}: {metrics}'
                )
            return metrics
        except Exception as e:
            logger.error(
                f'Error getting model metrics for {model_id}, version {version}: {e}'
                , exc_info=True)
            return {}

    def _calculate_metrics_improvement(self, baseline_metrics: Dict[str,
        Any], new_metrics: Dict[str, Any]) ->Dict[str, Any]:
        """
        Calculate improvement between baseline and new metrics.
        
        Args:
            baseline_metrics: Baseline model metrics
            new_metrics: New model metrics
            
        Returns:
            Dict[str, Any]: Improvement metrics
        """
        improvement = {}
        if 'mean_error' in baseline_metrics and 'mean_error' in new_metrics:
            baseline_error = baseline_metrics['mean_error']
            new_error = new_metrics['mean_error']
            if baseline_error > 0:
                error_reduction = (baseline_error - new_error) / baseline_error
                improvement['error_reduction'] = error_reduction
        if 'accuracy' in baseline_metrics and 'accuracy' in new_metrics:
            baseline_accuracy = baseline_metrics['accuracy']
            new_accuracy = new_metrics['accuracy']
            accuracy_improvement = new_accuracy - baseline_accuracy
            improvement['accuracy_improvement'] = accuracy_improvement
        if 'precision' in baseline_metrics and 'precision' in new_metrics:
            improvement['precision_improvement'] = new_metrics['precision'
                ] - baseline_metrics['precision']
        if 'recall' in baseline_metrics and 'recall' in new_metrics:
            improvement['recall_improvement'] = new_metrics['recall'
                ] - baseline_metrics['recall']
        if 'f1' in baseline_metrics and 'f1' in new_metrics:
            improvement['f1_improvement'] = new_metrics['f1'
                ] - baseline_metrics['f1']
        if 'model_score' in baseline_metrics and 'model_score' in new_metrics:
            if baseline_metrics['model_score'] > 0:
                score_improvement = (new_metrics['model_score'] -
                    baseline_metrics['model_score']) / baseline_metrics[
                    'model_score']
                improvement['score_improvement'] = score_improvement
        return improvement

    def _update_performance_tracking(self, model_id: str, improvement: Dict
        [str, Any]):
        """
        Update performance tracking for a model.
        
        Args:
            model_id: The model ID
            improvement: Improvement metrics
        """
        if model_id not in self.performance_improvements:
            self.performance_improvements[model_id] = {'improvements': [],
                'cumulative_improvement': {}, 'last_updated': datetime.utcnow()
                }
        self.performance_improvements[model_id]['improvements'].append({
            'timestamp': datetime.utcnow(), 'metrics': improvement})
        cumulative = self.performance_improvements[model_id][
            'cumulative_improvement']
        for key, value in improvement.items():
            if key not in cumulative:
                cumulative[key] = 0.0
            cumulative[key] += value
        self.performance_improvements[model_id]['last_updated'
            ] = datetime.utcnow()
        window_days = self.config_manager.get('performance_window', 30)
        cutoff_date = datetime.utcnow() - timedelta(days=window_days)
        self.performance_improvements[model_id]['improvements'] = [imp for
            imp in self.performance_improvements[model_id]['improvements'] if
            imp['timestamp'] > cutoff_date]

    @with_resilience('get_model_performance_summary')
    def get_model_performance_summary(self, model_id: str) ->Dict[str, Any]:
        """
        Get a summary of model performance improvements.
        
        Args:
            model_id: The model to get summary for
            
        Returns:
            Dict[str, Any]: Performance summary
        """
        if model_id not in self.performance_improvements:
            return {'model_id': model_id, 'improvements_count': 0,
                'cumulative_improvement': {}, 'last_updated': None}
        data = self.performance_improvements[model_id]
        return {'model_id': model_id, 'improvements_count': len(data[
            'improvements']), 'cumulative_improvement': data[
            'cumulative_improvement'], 'last_updated': data['last_updated'],
            'recent_improvements': data['improvements'][-5:] if data[
            'improvements'] else []}

    @with_resilience('get_retraining_history')
    def get_retraining_history(self, model_id: Optional[str]=None) ->List[Dict
        [str, Any]]:
        """
        Get the history of retraining sessions.
        
        Args:
            model_id: Optional filter by model ID
            
        Returns:
            List[Dict[str, Any]]: Retraining history
        """
        if model_id:
            return [session for session in self.retraining_history if 
                session['model_id'] == model_id]
        return self.retraining_history
