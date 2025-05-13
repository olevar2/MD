"""
Analysis Engine Metrics Exporter

This module exports metrics from the Analysis Engine Service to the monitoring system,
focusing on adaptation effectiveness, feedback processing, and model training metrics.
"""
import time
import logging
from typing import Dict, Any, Optional, List, Deque
from datetime import datetime, timedelta
import asyncio
from collections import deque
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client.exposition import start_http_server
from core_foundations.utils.logger import get_logger
from core_foundations.config.configuration import ConfigurationManager
from analysis_engine.adaptive_layer.feedback_loop import FeedbackLoop
from analysis_engine.adaptive_layer.adaptation_engine import AdaptationEngine
from analysis_engine.adaptive_layer.model_retraining_service import TrainingPipelineIntegrator
logger = get_logger(__name__)
ADAPTATIONS_COUNT = Counter('forex_adaptations_count',
    'Count of strategy adaptations performed', ['strategy_id',
    'adaptation_type', 'market_regime'])
ADAPTATIONS_SUCCESS_RATE = Gauge('forex_adaptations_success_rate',
    'Success rate of strategy adaptations', ['strategy_id', 'adaptation_type'])
ADAPTATION_IMPACT = Gauge('forex_adaptation_impact',
    'Impact of strategy adaptations on performance', ['strategy_id',
    'adaptation_type', 'metric'])
FEEDBACK_PROCESSED = Counter('forex_feedback_processed_count',
    'Count of feedback items processed', ['source', 'category', 'status'])
FEEDBACK_PROCESSING_TIME = Histogram('forex_feedback_processing_time_seconds',
    'Time taken to process feedback items', ['source', 'category'], buckets
    =(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0))
PARAMETER_ADAPTATION_SUCCESS_RATE = Gauge(
    'forex_parameter_adaptation_success_rate',
    'Success rate of parameter adaptations', ['strategy_id', 'parameter_name'])
POST_ADAPTATION_PROFIT = Gauge('forex_post_adaptation_profit',
    'Profit after adaptation over time', ['strategy_id', 'adaptation_id'])
MODEL_RETRAINING_EVENTS = Counter('forex_model_retraining_events',
    'Count of model retraining events', ['model_id', 'event_type'])


from monitoring_alerting_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AnalysisEngineMetricsExporter:
    """
    Exports metrics from the Analysis Engine Service to the monitoring system.
    """

    def __init__(self, feedback_loop: FeedbackLoop, adaptation_engine:
        AdaptationEngine, training_integrator: Optional[
        TrainingPipelineIntegrator]=None, config_manager: Optional[
        ConfigurationManager]=None):
        """
        Initialize the Analysis Engine metrics exporter.
        
        Args:
            feedback_loop: Feedback loop system
            adaptation_engine: Adaptation engine
            training_integrator: Training pipeline integrator
            config_manager: Configuration manager
        """
        self.feedback_loop = feedback_loop
        self.adaptation_engine = adaptation_engine
        self.training_integrator = training_integrator
        self.config_manager = config_manager
        self.config = self._load_config()
        self.export_interval = self.config_manager.get('export_interval_seconds', 60)
        self.exporter_port = self.config_manager.get('exporter_port', 9103)
        self.is_running = False
        self.exporter_task = None
        self.feedback_timestamps: Dict[str, datetime] = {}
        self.recent_adaptations: Deque[Dict[str, Any]] = deque(maxlen=100)
        logger.info(
            f'Analysis Engine Metrics Exporter initialized with port {self.exporter_port} and interval {self.export_interval}s'
            )

    @with_exception_handling
    def _load_config(self) ->Dict[str, Any]:
        """
        Load configuration from the configuration manager or use defaults.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        default_config = {'export_interval_seconds': 60, 'exporter_port': 
            9103, 'metrics_enabled': True, 'adaptation_lookback_days': 30,
            'post_adaptation_window_days': 7}
        if self.config_manager:
            try:
                metrics_config = self.config_manager.get_config(
                    'metrics_exporter')
                if metrics_config:
                    return {**default_config, **metrics_config}
            except Exception as e:
                logger.warning(f'Failed to load metrics exporter config: {e}')
        return default_config

    @with_exception_handling
    def start(self, port: Optional[int]=None) ->None:
        """
        Start the metrics exporter HTTP server and metrics collection task.
        
        Args:
            port: Optional override for the exporter port
        """
        if self.is_running:
            logger.warning('Metrics exporter is already running')
            return
        exporter_port = port or self.exporter_port
        try:
            start_http_server(exporter_port)
            logger.info(f'Started metrics HTTP server on port {exporter_port}')
            self._setup_feedback_hooks()
            self.exporter_task = asyncio.create_task(self.
                _collect_metrics_periodically())
            self.is_running = True
            logger.info('Analysis Engine metrics exporter started successfully'
                )
        except Exception as e:
            logger.error(f'Failed to start metrics exporter: {e}')
            raise

    def _setup_feedback_hooks(self) ->None:
        """Set up hooks to capture metrics from the feedback loop in real-time."""
        original_add_feedback = self.feedback_loop.add_feedback

        async def wrapped_add_feedback(feedback):
    """
    Wrapped add feedback.
    
    Args:
        feedback: Description of feedback
    
    """

            feedback_id = getattr(feedback, 'id', None) or getattr(feedback,
                'feedback_id', None)
            if feedback_id:
                self.feedback_timestamps[feedback_id] = datetime.utcnow()
            result = await original_add_feedback(feedback)
            if feedback_id and feedback_id in self.feedback_timestamps:
                start_time = self.feedback_timestamps.pop(feedback_id)
                processing_time = (datetime.utcnow() - start_time
                    ).total_seconds()
                source = getattr(feedback, 'source', 'unknown')
                category = getattr(feedback, 'category', 'unknown')
                if hasattr(source, 'value'):
                    source = source.value
                if hasattr(category, 'value'):
                    category = category.value
                FEEDBACK_PROCESSING_TIME.labels(source=source, category=
                    category).observe(processing_time)
                status = getattr(feedback, 'status', 'unknown')
                if hasattr(status, 'value'):
                    status = status.value
                FEEDBACK_PROCESSED.labels(source=source, category=category,
                    status=status).inc()
            return result
        self.feedback_loop.add_feedback = wrapped_add_feedback
        original_evaluate_and_adapt = self.adaptation_engine.evaluate_and_adapt

        async def wrapped_evaluate_and_adapt(context):
    """
    Wrapped evaluate and adapt.
    
    Args:
        context: Description of context
    
    """

            result = await original_evaluate_and_adapt(context)
            if result and isinstance(result, dict):
                strategy_id = result.get('strategy_id') or context.get(
                    'strategy_id')
                adaptation_type = result.get('action', 'unknown')
                market_regime = context.get('market_regime', 'unknown')
                if strategy_id:
                    ADAPTATIONS_COUNT.labels(strategy_id=strategy_id,
                        adaptation_type=adaptation_type, market_regime=
                        market_regime).inc()
                    adaptation_record = {'strategy_id': strategy_id,
                        'adaptation_type': adaptation_type, 'market_regime':
                        market_regime, 'timestamp': datetime.utcnow(),
                        'adaptation_id': result.get('adaptation_id', str(
                        uuid.uuid4())), 'parameters': result.get(
                        'parameters', {})}
                    self.recent_adaptations.append(adaptation_record)
            return result
        self.adaptation_engine.evaluate_and_adapt = wrapped_evaluate_and_adapt

    @async_with_exception_handling
    async def stop(self) ->None:
        """Stop the metrics collection task."""
        if not self.is_running:
            return
        if self.exporter_task and not self.exporter_task.done():
            self.exporter_task.cancel()
            try:
                await self.exporter_task
            except asyncio.CancelledError:
                pass
        self.is_running = False
        logger.info('Analysis Engine metrics exporter stopped')

    @async_with_exception_handling
    async def _collect_metrics_periodically(self) ->None:
        """Collect metrics at regular intervals."""
        try:
            while True:
                await self._collect_and_export_metrics()
                await asyncio.sleep(self.export_interval)
        except asyncio.CancelledError:
            logger.info('Metrics collection task cancelled')
            raise
        except Exception as e:
            logger.error(f'Error in metrics collection: {e}')
            raise

    @async_with_exception_handling
    async def _collect_and_export_metrics(self) ->None:
        """Collect current metrics and update exporters."""
        try:
            effectiveness = self.feedback_loop.get_adaptation_effectiveness()
            for strategy_type, strategy_data in effectiveness.get(
                'strategy_type', {}).items():
                if not isinstance(strategy_data, dict):
                    continue
                success_rate = strategy_data.get('success_rate', 0)
                ADAPTATIONS_SUCCESS_RATE.labels(strategy_id=strategy_type,
                    adaptation_type='overall').set(success_rate)
            for parameter_type, param_data in effectiveness.get(
                'parameter_type', {}).items():
                if not isinstance(param_data, dict):
                    continue
                for strategy_id, success_rate in param_data.items():
                    if isinstance(success_rate, (int, float)):
                        PARAMETER_ADAPTATION_SUCCESS_RATE.labels(strategy_id
                            =strategy_id, parameter_name=parameter_type).set(
                            success_rate)
            if hasattr(self.feedback_loop, 'adaptation_outcomes'):
                adaptation_outcomes = list(self.feedback_loop.
                    adaptation_outcomes)
                recent_cutoff = datetime.utcnow() - timedelta(days=self.
                    config_manager.get('post_adaptation_window_days', 7))
                grouped_outcomes = {}
                for outcome in adaptation_outcomes:
                    timestamp_str = outcome.get('timestamp')
                    if not timestamp_str:
                        continue
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str
                            ) if isinstance(timestamp_str, str
                            ) else timestamp_str
                    except (ValueError, TypeError):
                        continue
                    if timestamp < recent_cutoff:
                        continue
                    strategy_id = outcome.get('strategy_id')
                    adaptation_id = outcome.get('adaptation_id')
                    if not strategy_id or not adaptation_id:
                        continue
                    key = f'{strategy_id}:{adaptation_id}'
                    if key not in grouped_outcomes:
                        grouped_outcomes[key] = []
                    grouped_outcomes[key].append(outcome)
                for key, outcomes in grouped_outcomes.items():
                    if not outcomes:
                        continue
                    strategy_id, adaptation_id = key.split(':', 1)
                    total_profit = sum(outcome.get('metrics', {}).get(
                        'profit', 0) for outcome in outcomes)
                    POST_ADAPTATION_PROFIT.labels(strategy_id=strategy_id,
                        adaptation_id=adaptation_id).set(total_profit)
                    adaptation_type = 'unknown'
                    for adaptation in self.recent_adaptations:
                        if adaptation.get('adaptation_id') == adaptation_id:
                            adaptation_type = adaptation.get('adaptation_type',
                                'unknown')
                            break
                    ADAPTATION_IMPACT.labels(strategy_id=strategy_id,
                        adaptation_type=adaptation_type, metric='profit').set(
                        total_profit)
            if hasattr(self.feedback_loop, 'adaptation_effectiveness'):
                for key, model_data in self.feedback_loop.adaptation_effectiveness.items(
                    ):
                    if not key.startswith('model_') or not isinstance(
                        model_data, dict):
                        continue
                    model_id = key[6:]
                    training_history = model_data.get('training_history', [])
                    for event in training_history:
                        if 'timestamp' not in event:
                            continue
                        timestamp_str = event.get('timestamp')
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str
                                ) if isinstance(timestamp_str, str
                                ) else timestamp_str
                            if datetime.utcnow() - timestamp > timedelta(hours
                                =24):
                                continue
                        except (ValueError, TypeError):
                            continue
                        status = event.get('status', 'unknown')
                        MODEL_RETRAINING_EVENTS.labels(model_id=model_id,
                            event_type=status).inc()
        except Exception as e:
            logger.error(f'Error collecting Analysis Engine metrics: {e}',
                exc_info=True)


def create_metrics_exporter(feedback_loop: FeedbackLoop, adaptation_engine:
    AdaptationEngine, training_integrator: Optional[
    TrainingPipelineIntegrator]=None, config_manager: Optional[
    ConfigurationManager]=None) ->AnalysisEngineMetricsExporter:
    """
    Create and initialize the Analysis Engine metrics exporter.
    
    Args:
        feedback_loop: Feedback loop system
        adaptation_engine: Adaptation engine
        training_integrator: Training pipeline integrator
        config_manager: Configuration manager
        
    Returns:
        AnalysisEngineMetricsExporter: Initialized metrics exporter
    """
    exporter = AnalysisEngineMetricsExporter(feedback_loop=feedback_loop,
        adaptation_engine=adaptation_engine, training_integrator=
        training_integrator, config_manager=config_manager)
    return exporter
