"""
Feedback Loop

This module implements the feedback loop system that connects strategy execution outcomes
with the adaptation engine, enabling continuous improvement of parameter adjustments.
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from collections import deque
from core_foundations.models.feedback import TradeFeedback, FeedbackSource, FeedbackCategory, FeedbackStatus
from core_foundations.events.event_publisher import EventPublisher
from core_foundations.events.event_schema import EventType
from .multi_timeframe_feedback import MultiTimeframeFeedbackProcessor
from .parameter_feedback import ParameterFeedbackAnalyzer
from .strategy_mutation import StrategyMutator
from analysis_engine.adaptive_layer.adaptation_engine import AdaptationEngine
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class FeedbackLoop:
    """
    The FeedbackLoop system captures the outcomes of strategy execution and feeds them
    back to the adaptation engine, enabling continuous learning and improvement of
    parameter adjustments.

    Key capabilities:
    - Track the performance of parameter adjustments
    - Analyze the effectiveness of adaptation decisions
    - Adjust adaptation strategies based on outcome data
    - Provide insights for strategy optimization
    """

    def __init__(self, adaptation_engine: AdaptationEngine, event_publisher:
        Optional[EventPublisher]=None, config: Dict[str, Any]=None,
        multi_timeframe_processor: Optional[MultiTimeframeFeedbackProcessor
        ]=None, parameter_analyzer: Optional[ParameterFeedbackAnalyzer]=
        None, strategy_mutator: Optional[StrategyMutator]=None):
        """
        Initialize the FeedbackLoop system.

        Args:
            adaptation_engine: The adaptation engine to integrate with.
            event_publisher: Optional event publisher for feedback loop events.
            config: Configuration parameters for the feedback loop.
            multi_timeframe_processor: Processor for multi-timeframe analysis.
            parameter_analyzer: Analyzer for parameter-specific feedback.
            strategy_mutator: Framework for mutating strategies.
        """
        self.adaptation_engine = adaptation_engine
        self.event_publisher = event_publisher
        self.config = config or {}
        self.multi_timeframe_processor = (multi_timeframe_processor or
            MultiTimeframeFeedbackProcessor(config=self.config.get(
            'multi_timeframe_config')))
        self.parameter_analyzer = (parameter_analyzer or
            ParameterFeedbackAnalyzer(config=self.config.get(
            'parameter_analyzer_config')))
        self.strategy_mutator = strategy_mutator or StrategyMutator(config=
            self.config_manager.get('strategy_mutator_config'))
        self.recent_feedback = deque(maxlen=self.config.get(
            'recent_feedback_buffer_size', 1000))
        self.adaptation_outcomes = deque(maxlen=self.config.get(
            'adaptation_outcome_buffer_size', 500))
        self.performance_by_regime = {}
        self.adaptation_effectiveness = {'strategy_type': {}, 'regime_type':
            {}, 'parameter_type': {}}
        logger.info('FeedbackLoop initialized')

    @async_with_exception_handling
    async def add_feedback(self, feedback: TradeFeedback) ->None:
        """
        Processes incoming feedback, updates internal state, and potentially triggers adaptation.
        This is the entry point called by TradingFeedbackCollector.

        Args:
            feedback: The feedback object received.
        """
        logger.debug(
            f'FeedbackLoop received feedback: {feedback.id} (Source: {feedback.source}, Category: {feedback.category})'
            )
        self.recent_feedback.append(feedback)
        if (feedback.source == FeedbackSource.STRATEGY_EXECUTION and 
            feedback.category == FeedbackCategory.PERFORMANCE):
            if hasattr(feedback, 'strategy_id') and hasattr(feedback,
                'instrument') and hasattr(feedback, 'timeframe') and hasattr(
                feedback, 'adaptation_id') and hasattr(feedback,
                'outcome_metrics') and hasattr(feedback, 'market_regime'):
                self.record_strategy_outcome(strategy_id=feedback.
                    strategy_id, instrument=feedback.instrument, timeframe=
                    feedback.timeframe, adaptation_id=feedback.
                    adaptation_id, outcome_metrics=feedback.outcome_metrics,
                    market_regime=feedback.market_regime, timestamp=
                    datetime.fromisoformat(feedback.timestamp) if feedback.
                    timestamp else datetime.utcnow())
            else:
                logger.warning(
                    f'Skipping outcome recording for feedback {feedback.id}: Missing required attributes.'
                    )
        multi_timeframe_insights = None
        parameter_insights = None
        if self.multi_timeframe_processor:
            try:
                multi_timeframe_insights = (await self.
                    multi_timeframe_processor.process(feedback))
                if multi_timeframe_insights:
                    logger.info(
                        f'Multi-timeframe insights generated for feedback {feedback.id}: {multi_timeframe_insights}'
                        )
            except Exception as e:
                logger.error(
                    f'Error processing multi-timeframe feedback for {feedback.id}: {e}'
                    , exc_info=True)
        if self.parameter_analyzer:
            try:
                parameter_insights = await self.parameter_analyzer.analyze(
                    feedback)
                if parameter_insights:
                    logger.info(
                        f'Parameter insights generated for feedback {feedback.id}: {parameter_insights}'
                        )
            except Exception as e:
                logger.error(
                    f'Error processing parameter feedback for {feedback.id}: {e}'
                    , exc_info=True)
        adaptation_context = {'triggering_feedback_id': feedback.id,
            'category': feedback.category.value, 'details': feedback.
            details, 'strategy_id': getattr(feedback, 'strategy_id', None),
            'instrument': getattr(feedback, 'instrument', None),
            'timeframe': getattr(feedback, 'timeframe', None),
            'multi_timeframe_insights': multi_timeframe_insights,
            'parameter_insights': parameter_insights}
        adaptation_decision = await self.adaptation_engine.evaluate_and_adapt(
            adaptation_context)
        if adaptation_decision and adaptation_decision.get('action'
            ) == 'mutate_strategy' and self.strategy_mutator:
            strategy_id_to_mutate = adaptation_decision.get('strategy_id')
            if strategy_id_to_mutate:
                current_strategy_config = {'id': strategy_id_to_mutate,
                    'parameters': {'param1': 10, 'param2': 0.5}, 'version': 1.0
                    }
                if current_strategy_config:
                    logger.info(
                        f'Triggering strategy mutation for {strategy_id_to_mutate} based on adaptation decision.'
                        )
                    try:
                        mutated_config = (await self.strategy_mutator.
                            mutate_strategy(current_strategy_config,
                            feedback_context=[feedback]))
                        if mutated_config:
                            logger.info(
                                f'Strategy {strategy_id_to_mutate} mutated. New config: {mutated_config}'
                                )
                    except Exception as e:
                        logger.error(
                            f'Error during strategy mutation for {strategy_id_to_mutate}: {e}'
                            , exc_info=True)
        if self.event_publisher:
            try:
                await self.event_publisher.publish(EventType.
                    FEEDBACK_PROCESSED, {'feedback_id': feedback.id,
                    'status': feedback.status.value, 'timestamp': datetime.
                    utcnow().isoformat()})
            except Exception as e:
                logger.error(
                    f'Failed to publish FEEDBACK_PROCESSED event for {feedback.id}: {e}'
                    , exc_info=True)
        logger.debug(
            f'FeedbackLoop finished processing feedback: {feedback.id}')

    def record_strategy_outcome(self, strategy_id: str, instrument: str,
        timeframe: str, adaptation_id: str, outcome_metrics: Dict[str, Any],
        market_regime: str, timestamp: datetime=None) ->None:
        """
        Record the outcome of a strategy execution with adapted parameters.
        (Now uses deque for storage)
        """
        timestamp = timestamp or datetime.utcnow()
        outcome_record = {'timestamp': timestamp, 'strategy_id':
            strategy_id, 'instrument': instrument, 'timeframe': timeframe,
            'adaptation_id': adaptation_id, 'market_regime': market_regime,
            'metrics': outcome_metrics}
        self.adaptation_outcomes.append(outcome_record)
        self._update_performance_tracking(outcome_record)
        self._update_adaptation_effectiveness(outcome_record)
        logger.info(
            f'Recorded strategy outcome for {strategy_id} (Adaptation: {adaptation_id}, Regime: {market_regime})'
            )

    def _update_performance_tracking(self, outcome_record: Dict[str, Any]
        ) ->None:
        """
        Update the performance tracking data.

        Args:
            outcome_record: The outcome record to process
        """
        strategy_id = outcome_record['strategy_id']
        market_regime = outcome_record['market_regime']
        metrics = outcome_record['metrics']
        if strategy_id not in self.performance_by_regime:
            self.performance_by_regime[strategy_id] = {}
        if market_regime not in self.performance_by_regime[strategy_id]:
            self.performance_by_regime[strategy_id][market_regime] = {'count':
                0, 'win_count': 0, 'loss_count': 0, 'profit_sum': 0.0,
                'loss_sum': 0.0, 'avg_profit_factor': 0.0, 'avg_win_rate': 0.0}
        perf = self.performance_by_regime[strategy_id][market_regime]
        perf['count'] += 1
        if metrics.get('profit', 0) > 0:
            perf['win_count'] += 1
            perf['profit_sum'] += metrics.get('profit', 0)
        else:
            perf['loss_count'] += 1
            perf['loss_sum'] += abs(metrics.get('profit', 0))
        if perf['count'] > 0:
            perf['avg_win_rate'] = perf['win_count'] / perf['count']
        if perf['loss_sum'] > 0:
            perf['avg_profit_factor'] = perf['profit_sum'] / perf['loss_sum']
        else:
            perf['avg_profit_factor'] = perf['profit_sum'] if perf['profit_sum'
                ] > 0 else 0

    def _update_adaptation_effectiveness(self, outcome_record: Dict[str, Any]
        ) ->None:
        """
        Update the effectiveness metrics for adaptation strategies.

        Args:
            outcome_record: The outcome record to process
        """
        strategy_id = outcome_record['strategy_id']
        market_regime = outcome_record['market_regime']
        metrics = outcome_record['metrics']
        if strategy_id not in self.adaptation_effectiveness['strategy_type']:
            self.adaptation_effectiveness['strategy_type'][strategy_id] = {
                'count': 0, 'success_count': 0, 'failure_count': 0,
                'success_rate': 0.0}
        if market_regime not in self.adaptation_effectiveness['regime_type']:
            self.adaptation_effectiveness['regime_type'][market_regime] = {
                'count': 0, 'success_count': 0, 'failure_count': 0,
                'success_rate': 0.0}
        strat = self.adaptation_effectiveness['strategy_type'][strategy_id]
        strat['count'] += 1
        is_success = metrics.get('profit', 0) > 0
        if is_success:
            strat['success_count'] += 1
        else:
            strat['failure_count'] += 1
        strat['success_rate'] = strat['success_count'] / strat['count'
            ] if strat['count'] > 0 else 0.0
        regime = self.adaptation_effectiveness['regime_type'][market_regime]
        regime['count'] += 1
        if is_success:
            regime['success_count'] += 1
        else:
            regime['failure_count'] += 1
        regime['success_rate'] = regime['success_count'] / regime['count'
            ] if regime['count'] > 0 else 0.0

    @with_resilience('get_adaptation_effectiveness')
    def get_adaptation_effectiveness(self, strategy_id: Optional[str]=None,
        market_regime: Optional[str]=None) ->Dict[str, Any]:
        """
        Get the effectiveness metrics for adaptation strategies.

        Args:
            strategy_id: Optional filter for a specific strategy
            market_regime: Optional filter for a specific market regime

        Returns:
            Dict[str, Any]: Adaptation effectiveness metrics
        """
        if strategy_id:
            return self.adaptation_effectiveness['strategy_type'].get(
                strategy_id, {})
        if market_regime:
            return self.adaptation_effectiveness['regime_type'].get(
                market_regime, {})
        return self.adaptation_effectiveness

    @with_resilience('get_performance_by_regime')
    def get_performance_by_regime(self, strategy_id: str, market_regime:
        Optional[str]=None) ->Dict[str, Any]:
        """
        Get performance metrics by market regime for a specific strategy.

        Args:
            strategy_id: Identifier for the strategy
            market_regime: Optional filter for a specific market regime

        Returns:
            Dict[str, Any]: Performance metrics
        """
        if strategy_id not in self.performance_by_regime:
            return {}
        if market_regime:
            return self.performance_by_regime[strategy_id].get(market_regime,
                {})
        return self.performance_by_regime[strategy_id]

    @with_resilience('update_adaptation_strategy')
    def update_adaptation_strategy(self, strategy_id: str) ->bool:
        """
        Update the adaptation strategies based on feedback data.

        Args:
            strategy_id: The strategy to update adaptation strategies for

        Returns:
            bool: True if strategies were updated, False otherwise
        """
        logger.info('Adaptation strategy update not yet implemented for %s',
            strategy_id)
        return False

    def generate_insights(self, strategy_id: str) ->List[Dict[str, Any]]:
        """
        Generate insights from feedback data to optimize strategy.

        Args:
            strategy_id: The strategy to generate insights for

        Returns:
            List[Dict[str, Any]]: List of insights
        """
        insights = []
        if strategy_id not in self.performance_by_regime:
            return insights
        performance = self.performance_by_regime[strategy_id]
        best_regime = None
        best_rate = 0.0
        for regime, metrics in performance.items():
            if metrics['count'] >= self.config_manager.get('min_samples', 10):
                if metrics['avg_win_rate'] > best_rate:
                    best_rate = metrics['avg_win_rate']
                    best_regime = regime
        if best_regime:
            insights.append({'type': 'best_regime', 'message':
                f'Strategy {strategy_id} performs best in {best_regime} regime with {best_rate:.1%} win rate'
                , 'data': {'regime': best_regime, 'win_rate': best_rate}})
        worst_regime = None
        worst_rate = 1.0
        for regime, metrics in performance.items():
            if metrics['count'] >= self.config_manager.get('min_samples', 10):
                if metrics['avg_win_rate'] < worst_rate:
                    worst_rate = metrics['avg_win_rate']
                    worst_regime = regime
        if worst_regime:
            insights.append({'type': 'worst_regime', 'message':
                f'Strategy {strategy_id} performs poorly in {worst_regime} regime with only {worst_rate:.1%} win rate'
                , 'data': {'regime': worst_regime, 'win_rate': worst_rate}})
        for regime, metrics in performance.items():
            if metrics['count'] >= self.config_manager.get('min_samples', 10):
                profit_factor = metrics['avg_profit_factor']
                win_rate = metrics['avg_win_rate']
                if win_rate > 0.5 and profit_factor < 1.0:
                    insights.append({'type': 'inconsistent_performance',
                        'message':
                        f'Strategy {strategy_id} has positive win rate {win_rate:.1%} but negative profit factor {profit_factor:.2f} in {regime} regime'
                        , 'data': {'regime': regime, 'win_rate': win_rate,
                        'profit_factor': profit_factor}})
        return insights

    @with_database_resilience('process_incoming_feedback')
    @async_with_exception_handling
    async def process_incoming_feedback(self, feedback: TradeFeedback) ->str:
        """
        Process incoming feedback received from Kafka or other event sources.

        This method orchestrates the flow through the feedback components:
        1. Stores the feedback
        2. Categorizes it
        3. Routes it to appropriate handlers
        4. Optionally publishes events about the feedback processing

        Args:
            feedback: The feedback object to process

        Returns:
            str: The feedback ID
        """
        logger.info(
            f'Processing incoming feedback: {feedback.feedback_id}, source: {feedback.source}, category: {feedback.category}'
            )
        try:
            self.record_feedback(feedback)
            if feedback.category == FeedbackCategory.UNCATEGORIZED:
                categorized = await self._categorize_feedback(feedback)
                if not categorized:
                    logger.warning(
                        f'Could not categorize feedback {feedback.feedback_id}'
                        )
            if feedback.source == FeedbackSource.MODEL_PREDICTION:
                await self._handle_model_feedback(feedback)
            elif feedback.source == FeedbackSource.TRADING_OUTCOME:
                await self._handle_trading_outcome_feedback(feedback)
            elif feedback.source == FeedbackSource.PARAMETER_ADJUSTMENT:
                await self._handle_parameter_feedback(feedback)
            else:
                await self._handle_general_feedback(feedback)
            if self.event_publisher:
                event_data = {'feedback_id': feedback.feedback_id, 'source':
                    feedback.source.value if hasattr(feedback.source,
                    'value') else str(feedback.source), 'category': 
                    feedback.category.value if hasattr(feedback.category,
                    'value') else str(feedback.category), 'status':
                    FeedbackStatus.PROCESSED.value, 'processed_at':
                    datetime.utcnow().isoformat()}
                self.event_publisher.publish(event_type=EventType(
                    'feedback.processed'), data=event_data)
            feedback.status = FeedbackStatus.PROCESSED
            return feedback.feedback_id
        except Exception as e:
            logger.error(
                f'Error processing feedback {feedback.feedback_id}: {str(e)}',
                exc_info=True)
            feedback.status = FeedbackStatus.ERROR
            if self.event_publisher:
                error_event = {'feedback_id': feedback.feedback_id, 'error':
                    str(e), 'timestamp': datetime.utcnow().isoformat()}
                try:
                    self.event_publisher.publish(event_type=EventType(
                        'feedback.processing_error'), data=error_event)
                except Exception as pub_e:
                    logger.error(f'Failed to publish error event: {str(pub_e)}'
                        )
            return feedback.feedback_id

    @async_with_exception_handling
    async def _categorize_feedback(self, feedback: TradeFeedback) ->bool:
        """
        Categorize uncategorized feedback based on its content

        Args:
            feedback: The feedback to categorize

        Returns:
            bool: True if categorization was successful
        """
        try:
            metrics = feedback.outcome_metrics or {}
            if 'profit_loss' in metrics:
                pl = metrics['profit_loss']
                if isinstance(pl, (int, float)):
                    if pl > 0:
                        feedback.category = FeedbackCategory.SUCCESS
                    else:
                        feedback.category = FeedbackCategory.FAILURE
                    return True
            if 'prediction_error' in metrics:
                error = metrics['prediction_error']
                if isinstance(error, (int, float)):
                    if error < 0.1:
                        feedback.category = FeedbackCategory.SUCCESS
                    else:
                        feedback.category = FeedbackCategory.FAILURE
                    return True
            if feedback.category == FeedbackCategory.UNCATEGORIZED:
                feedback.category = FeedbackCategory.OTHER
            return False
        except Exception as e:
            logger.error(f'Error categorizing feedback: {str(e)}')
            return False

    async def _handle_model_feedback(self, feedback: TradeFeedback):
        """Handle feedback related to model predictions"""
        logger.info(f'Processing model feedback: {feedback.feedback_id}')

    async def _handle_trading_outcome_feedback(self, feedback: TradeFeedback):
        """Handle feedback from trading outcomes"""
        strategy_id = feedback.strategy_id
        instrument = feedback.instrument
        metrics = feedback.outcome_metrics or {}
        logger.info(
            f'Processing trading outcome feedback for {strategy_id} on {instrument}'
            )
        if strategy_id and self.adaptation_engine:
            market_regime = feedback.metadata.get('market_regime', 'unknown')
            await self.adaptation_engine.record_strategy_outcome(strategy_id
                =strategy_id, instrument=instrument, outcome_metrics=
                metrics, market_regime=market_regime, timeframe=feedback.
                timeframe)

    async def _handle_parameter_feedback(self, feedback: TradeFeedback):
        """Handle feedback about parameter adjustments"""
        parameter_name = feedback.metadata.get('parameter_name')
        if not parameter_name:
            logger.warning(
                f'Parameter feedback missing parameter_name: {feedback.feedback_id}'
                )
            return
        logger.info(f'Processing parameter feedback for {parameter_name}')

    async def _handle_general_feedback(self, feedback: TradeFeedback):
        """Handle general feedback types"""
        logger.info(
            f'Processing general feedback: {feedback.feedback_id}, category: {feedback.category}'
            )

    async def handle_model_training_completed(self, event: Event) ->None:
        """
        Handles model training completion events, updating adaptation outcomes
        and model performance metrics.

        Args:
            event: Model training completed event
        """
        event_data = event.data
        if not event_data:
            logger.warning(
                'Received model_training_completed event with no data')
            return
        model_id = event_data.get('model_id')
        job_id = event_data.get('job_id')
        status = event_data.get('status')
        model_metrics = event_data.get('model_metrics', {})
        if not model_id:
            logger.warning(
                f'Received model_training_completed event without model_id: {event_data}'
                )
            return
        logger.info(
            f'Processing model training completion for model {model_id}, job {job_id}, status: {status}'
            )
        outcome = {'type': 'model_training', 'model_id': model_id, 'job_id':
            job_id, 'status': status, 'timestamp': event_data.get(
            'completion_time') or datetime.utcnow().isoformat(), 'metrics':
            model_metrics, 'feedback_ids': event_data.get('feedback_ids', [])}
        self.adaptation_outcomes.append(outcome)
        model_key = f'model_{model_id}'
        if model_key not in self.adaptation_effectiveness:
            self.adaptation_effectiveness[model_key] = {'training_history':
                deque(maxlen=10), 'current_metrics': {}, 'training_count': 
                0, 'success_count': 0, 'last_updated': None}
        model_tracking = self.adaptation_effectiveness[model_key]
        model_tracking['training_history'].append(outcome)
        model_tracking['training_count'] += 1
        if status == 'success':
            model_tracking['success_count'] += 1
            model_tracking['current_metrics'] = model_metrics
            model_tracking['last_updated'] = datetime.utcnow().isoformat()
        if model_tracking['training_count'] > 0:
            model_tracking['success_rate'] = model_tracking['success_count'
                ] / model_tracking['training_count']
        if model_metrics:
            metrics_str = ', '.join(f'{k}: {v}' for k, v in model_metrics.
                items() if k in ['accuracy', 'precision', 'recall',
                'f1_score', 'mae', 'rmse'])
            if metrics_str:
                logger.info(
                    f'Updated metrics for model {model_id}: {metrics_str}')
        logger.info(
            f'Successfully processed model training completion for model {model_id}'
            )

    async def handle_model_training_failed(self, event: Event) ->None:
        """
        Handles model training failure events, updating adaptation outcomes.

        Args:
            event: Model training failed event
        """
        event_data = event.data
        if not event_data:
            logger.warning('Received model_training_failed event with no data')
            return
        model_id = event_data.get('model_id')
        job_id = event_data.get('job_id')
        error = event_data.get('error', 'Unknown error')
        if not model_id:
            logger.warning(
                f'Received model_training_failed event without model_id: {event_data}'
                )
            return
        logger.warning(
            f'Processing model training failure for model {model_id}, job {job_id}: {error}'
            )
        outcome = {'type': 'model_training', 'model_id': model_id, 'job_id':
            job_id, 'status': 'failed', 'error': error, 'timestamp': 
            event_data.get('failure_time') or datetime.utcnow().isoformat(),
            'feedback_ids': event_data.get('feedback_ids', [])}
        self.adaptation_outcomes.append(outcome)
        model_key = f'model_{model_id}'
        if model_key not in self.adaptation_effectiveness:
            self.adaptation_effectiveness[model_key] = {'training_history':
                deque(maxlen=10), 'current_metrics': {}, 'training_count': 
                0, 'success_count': 0, 'last_updated': None}
        model_tracking = self.adaptation_effectiveness[model_key]
        model_tracking['training_history'].append(outcome)
        model_tracking['training_count'] += 1
        if model_tracking['training_count'] > 0:
            model_tracking['success_rate'] = model_tracking['success_count'
                ] / model_tracking['training_count']
        logger.info(f'Recorded model training failure for model {model_id}')

    @async_with_exception_handling
    async def initialize_event_subscriptions(self, event_subscriber):
        """
        Initialize event subscriptions for the feedback loop.
        This should be called after the feedback loop is created.

        Args:
            event_subscriber: Event subscriber to use for subscribing to events
        """
        if not event_subscriber:
            logger.warning(
                'No event subscriber provided, cannot register event handlers')
            return
        try:
            await event_subscriber.subscribe(EventType.
                MODEL_TRAINING_COMPLETED, self.handle_model_training_completed)
            await event_subscriber.subscribe(EventType.
                MODEL_TRAINING_FAILED, self.handle_model_training_failed)
            logger.info('Successfully registered feedback loop event handlers')
        except Exception as e:
            logger.error(f'Failed to register event handlers: {e}',
                exc_info=True)
