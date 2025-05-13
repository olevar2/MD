"""
Integration Module for Adaptive Layer and Learning from Past Mistakes Components

from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

This module provides seamless data flow and integration between the Adaptive Layer and 
the Learning from Past Mistakes Module to enable continuous strategy improvement.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import json
from pydantic import BaseModel, Field
from core_foundations.utils.logger import get_logger
from analysis_engine.learning_from_mistakes.error_pattern_recognition import ErrorPatternRecognitionSystem, ErrorPattern
from analysis_engine.learning_from_mistakes.risk_adjustment import RiskAdjustmentManager
from analysis_engine.learning_from_mistakes.predictive_failure_modeling import PredictiveFailureModel
from analysis_engine.adaptive_layer.adaptation_engine import AdaptationEngine
from analysis_engine.adaptive_layer.parameter_adjustment_service import ParameterAdjustmentService
from analysis_engine.adaptive_layer.feedback_loop import FeedbackLoop
logger = get_logger(__name__)


class AdaptiveRiskAdjustmentIntegration(BaseModel):
    """
    Configuration model for integration between Adaptive Layer and Risk Adjustment
    """
    enable_risk_based_adaptation: bool = True
    risk_weight_in_adaptation: float = 0.3
    adaptation_threshold_multiplier: float = 1.0
    update_interval_minutes: int = 60


class LearningAdaptiveIntegration:
    """
    Integration service connecting the Learning from Past Mistakes Module with the Adaptive Layer.
    
    This service facilitates bidirectional data flow between error pattern recognition, 
    predictive failure modeling, and adaptive strategy adjustments.
    """

    def __init__(self, error_pattern_system: Optional[
        ErrorPatternRecognitionSystem]=None, risk_adjustment_manager:
        Optional[RiskAdjustmentManager]=None, predictive_failure_model:
        Optional[PredictiveFailureModel]=None, adaptation_engine: Optional[
        AdaptationEngine]=None, parameter_adjustment: Optional[
        ParameterAdjustmentService]=None, feedback_loop: Optional[
        FeedbackLoop]=None, config: Optional[Dict[str, Any]]=None):
        """
        Initialize the integration service
        
        Args:
            error_pattern_system: Error pattern recognition system instance
            risk_adjustment_manager: Risk adjustment manager instance
            predictive_failure_model: Predictive failure model instance
            adaptation_engine: Adaptation engine instance
            parameter_adjustment: Parameter adjustment service instance
            feedback_loop: Feedback loop instance
            config: Configuration dictionary
        """
        self.error_pattern_system = (error_pattern_system or
            ErrorPatternRecognitionSystem())
        self.risk_adjustment_manager = (risk_adjustment_manager or
            RiskAdjustmentManager())
        self.predictive_failure_model = (predictive_failure_model or
            PredictiveFailureModel())
        self.adaptation_engine = adaptation_engine or AdaptationEngine()
        self.parameter_adjustment = (parameter_adjustment or
            ParameterAdjustmentService())
        self.feedback_loop = feedback_loop or FeedbackLoop()
        self.config = config or {}
        self.default_config = {'enable_risk_based_adaptation': True,
            'risk_weight_in_adaptation': 0.3,
            'adaptation_threshold_multiplier': 1.0,
            'update_interval_minutes': 60, 'enable_predictive_adjustments':
            True, 'predictive_horizon_hours': 24,
            'max_pattern_history_days': 30, 'enable_logging': True}
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        self.last_sync_time = None
        self._running = False
        self._sync_task = None
        logger.info(
            f'Learning-Adaptive Integration service initialized with config: {json.dumps(self.config, indent=2)}'
            )

    async def start(self):
        """
        Start the integration service with periodic synchronization
        """
        if self._running:
            logger.warning(
                'Learning-Adaptive Integration service is already running')
            return
        self._running = True
        self._sync_task = asyncio.create_task(self._periodic_sync())
        logger.info('Learning-Adaptive Integration service started')

    @async_with_exception_handling
    async def stop(self):
        """
        Stop the integration service
        """
        if not self._running:
            logger.warning(
                'Learning-Adaptive Integration service is not running')
            return
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        logger.info('Learning-Adaptive Integration service stopped')

    @async_with_exception_handling
    async def _periodic_sync(self):
        """
        Periodically synchronize data between components
        """
        update_interval = self.config['update_interval_minutes']
        try:
            while self._running:
                logger.info(
                    f'Running periodic sync between Learning and Adaptive components'
                    )
                await self.sync_error_patterns_to_adaptation()
                await self.sync_predictive_models_to_adaptation()
                await self.sync_adaptation_feedback_to_learning()
                self.last_sync_time = datetime.now()
                logger.info(
                    f'Periodic sync completed. Next sync in {update_interval} minutes'
                    )
                await asyncio.sleep(update_interval * 60)
        except asyncio.CancelledError:
            logger.info('Periodic sync task cancelled')
            raise
        except Exception as e:
            logger.error(f'Error in periodic sync: {str(e)}', exc_info=True)
            if self._running:
                self._sync_task = asyncio.create_task(self._periodic_sync())

    @async_with_exception_handling
    async def sync_error_patterns_to_adaptation(self) ->Dict[str, Any]:
        """
        Sync error patterns from Learning module to Adaptive Layer for parameter adjustments
        
        Returns:
            Dictionary with synchronization results
        """
        try:
            max_days = self.config['max_pattern_history_days']
            from_date = datetime.now() - timedelta(days=max_days)
            error_patterns = (await self.error_pattern_system.
                get_recent_patterns(from_date=from_date))
            logger.info(
                f'Retrieved {len(error_patterns)} error patterns for adaptation'
                )
            grouped_patterns = {}
            for pattern in error_patterns:
                strategy_id = pattern.strategy_id
                if strategy_id not in grouped_patterns:
                    grouped_patterns[strategy_id] = {}
                pattern_type = pattern.pattern_type
                if pattern_type not in grouped_patterns[strategy_id]:
                    grouped_patterns[strategy_id][pattern_type] = []
                grouped_patterns[strategy_id][pattern_type].append(pattern)
            risk_adjustments = {}
            for strategy_id in grouped_patterns.keys():
                strategy_risk = (await self.risk_adjustment_manager.
                    get_strategy_risk_adjustment(strategy_id=strategy_id))
                risk_adjustments[strategy_id] = strategy_risk
            adaptation_updates = {}
            for strategy_id, patterns_by_type in grouped_patterns.items():
                adjustment_params = {}
                if 'entry_timing' in patterns_by_type:
                    entry_timing_patterns = patterns_by_type['entry_timing']
                    adjustment_params['entry_delay'
                        ] = self._calculate_entry_delay_adjustment(
                        entry_timing_patterns)
                if 'exit_timing' in patterns_by_type:
                    exit_timing_patterns = patterns_by_type['exit_timing']
                    adjustment_params['exit_speed'
                        ] = self._calculate_exit_speed_adjustment(
                        exit_timing_patterns)
                if 'stop_loss' in patterns_by_type:
                    stop_loss_patterns = patterns_by_type['stop_loss']
                    adjustment_params['stop_loss_multiplier'
                        ] = self._calculate_stop_loss_adjustment(
                        stop_loss_patterns)
                if self.config['enable_risk_based_adaptation'
                    ] and strategy_id in risk_adjustments:
                    risk_level = risk_adjustments[strategy_id].get('risk_level'
                        , 1.0)
                    risk_weight = self.config['risk_weight_in_adaptation']
                    for key in adjustment_params:
                        original_value = adjustment_params[key]
                        risk_modified_value = original_value * (1 + (
                            risk_level - 1) * risk_weight)
                        adjustment_params[key] = risk_modified_value
                if adjustment_params:
                    context = {'source': 'learning_from_mistakes',
                        'error_pattern_count': sum(len(patterns) for
                        patterns in patterns_by_type.values()), 'timestamp':
                        datetime.now().isoformat()}
                    await self.adaptation_engine.update_adaptive_parameters(
                        strategy_id=strategy_id, parameters=
                        adjustment_params, context=context)
                    await self.parameter_adjustment.record_adjustment(
                        strategy_id=strategy_id, parameters=
                        adjustment_params, reason='Error pattern detection',
                        metadata=context)
                    adaptation_updates[strategy_id] = {'adjusted_parameters':
                        adjustment_params, 'error_patterns': {k: len(v) for
                        k, v in patterns_by_type.items()},
                        'applied_risk_level': risk_adjustments.get(
                        strategy_id, {}).get('risk_level', 1.0)}
            logger.info(
                f'Applied error pattern-based adaptations to {len(adaptation_updates)} strategies'
                )
            return {'success': True, 'strategies_updated': len(
                adaptation_updates), 'adaptations': adaptation_updates}
        except Exception as e:
            logger.error(
                f'Error syncing error patterns to adaptation: {str(e)}',
                exc_info=True)
            return {'success': False, 'error': str(e)}

    @async_with_exception_handling
    async def sync_predictive_models_to_adaptation(self) ->Dict[str, Any]:
        """
        Sync predictive failure models to Adaptive Layer for proactive adjustments
        
        Returns:
            Dictionary with synchronization results
        """
        if not self.config['enable_predictive_adjustments']:
            return {'success': True, 'skipped':
                'Predictive adjustments disabled'}
        try:
            horizon_hours = self.config['predictive_horizon_hours']
            failure_predictions = (await self.predictive_failure_model.
                predict_failures(horizon_hours=horizon_hours))
            logger.info(
                f'Generated {len(failure_predictions)} failure predictions for next {horizon_hours} hours'
                )
            predictions_by_strategy = {}
            for prediction in failure_predictions:
                strategy_id = prediction['strategy_id']
                if strategy_id not in predictions_by_strategy:
                    predictions_by_strategy[strategy_id] = []
                predictions_by_strategy[strategy_id].append(prediction)
            adaptation_updates = {}
            for strategy_id, predictions in predictions_by_strategy.items():
                adjustment_params = {}
                sorted_predictions = sorted(predictions, key=lambda p: p[
                    'probability'], reverse=True)
                failure_types = {}
                for prediction in sorted_predictions:
                    failure_type = prediction['failure_type']
                    if failure_type not in failure_types and prediction[
                        'probability'] > 0.5:
                        failure_types[failure_type] = prediction
                if 'trend_reversal' in failure_types:
                    prediction = failure_types['trend_reversal']
                    prob = prediction['probability']
                    adjustment_params['trend_sensitivity'] = 1.0 + (prob - 0.5
                        ) * 0.5
                if 'stop_loss_hit' in failure_types:
                    prediction = failure_types['stop_loss_hit']
                    prob = prediction['probability']
                    adjustment_params['stop_loss_multiplier'] = 1.0 + (prob -
                        0.5) * 0.2
                if 'volatility_spike' in failure_types:
                    prediction = failure_types['volatility_spike']
                    prob = prediction['probability']
                    adjustment_params['position_size_multiplier'] = 1.0 - (prob
                         - 0.5) * 0.4
                if adjustment_params:
                    context = {'source': 'predictive_failure_model',
                        'prediction_horizon_hours': horizon_hours,
                        'highest_probability': max(p['probability'] for p in
                        predictions), 'predicted_failure_types': list(
                        failure_types.keys()), 'timestamp': datetime.now().
                        isoformat()}
                    await self.adaptation_engine.update_adaptive_parameters(
                        strategy_id=strategy_id, parameters=
                        adjustment_params, context=context)
                    await self.parameter_adjustment.record_adjustment(
                        strategy_id=strategy_id, parameters=
                        adjustment_params, reason=
                        'Predictive failure prevention', metadata=context)
                    adaptation_updates[strategy_id] = {'adjusted_parameters':
                        adjustment_params, 'failure_predictions': [{'type':
                        k, 'probability': v['probability']} for k, v in
                        failure_types.items()]}
            logger.info(
                f'Applied predictive model-based adaptations to {len(adaptation_updates)} strategies'
                )
            return {'success': True, 'strategies_updated': len(
                adaptation_updates), 'adaptations': adaptation_updates}
        except Exception as e:
            logger.error(
                f'Error syncing predictive models to adaptation: {str(e)}',
                exc_info=True)
            return {'success': False, 'error': str(e)}

    @async_with_exception_handling
    async def sync_adaptation_feedback_to_learning(self) ->Dict[str, Any]:
        """
        Sync feedback from Adaptive Layer to Learning module for improved future predictions
        
        Returns:
            Dictionary with synchronization results
        """
        try:
            feedback_entries = await self.feedback_loop.get_recent_feedback(
                hours=24)
            logger.info(
                f'Retrieved {len(feedback_entries)} adaptation feedback entries for learning'
                )
            outcomes = []
            for entry in feedback_entries:
                if ('outcome' in entry and 'parameters' in entry and 
                    'strategy_id' in entry):
                    outcome = {'strategy_id': entry['strategy_id'],
                        'timestamp': entry['timestamp'], 'parameters':
                        entry['parameters'], 'outcome_type': entry[
                        'outcome']['type'], 'outcome_success': entry[
                        'outcome']['success'], 'metrics': entry['outcome'].
                        get('metrics', {})}
                    outcomes.append(outcome)
            await self.error_pattern_system.process_adaptation_outcomes(
                outcomes)
            if outcomes:
                await self.predictive_failure_model.update_with_outcomes(
                    outcomes)
            logger.info(
                f'Synced {len(outcomes)} adaptation outcomes to learning modules'
                )
            return {'success': True, 'adaptation_outcomes_synced': len(
                outcomes)}
        except Exception as e:
            logger.error(
                f'Error syncing adaptation feedback to learning: {str(e)}',
                exc_info=True)
            return {'success': False, 'error': str(e)}

    def _calculate_entry_delay_adjustment(self, entry_timing_patterns: List
        [ErrorPattern]) ->float:
        """
        Calculate entry delay adjustment based on entry timing error patterns
        
        Args:
            entry_timing_patterns: List of entry timing error patterns
            
        Returns:
            Entry delay adjustment factor
        """
        if not entry_timing_patterns:
            return 1.0
        early_entries = sum(1 for p in entry_timing_patterns if p.
            error_subtype == 'early_entry')
        late_entries = sum(1 for p in entry_timing_patterns if p.
            error_subtype == 'late_entry')
        total_patterns = len(entry_timing_patterns)
        if total_patterns == 0:
            return 1.0
        early_ratio = early_entries / total_patterns
        late_ratio = late_entries / total_patterns
        if early_ratio > late_ratio:
            return 1.0 + early_ratio * 0.2
        elif late_ratio > early_ratio:
            return 1.0 - late_ratio * 0.2
        else:
            return 1.0

    def _calculate_exit_speed_adjustment(self, exit_timing_patterns: List[
        ErrorPattern]) ->float:
        """
        Calculate exit speed adjustment based on exit timing error patterns
        
        Args:
            exit_timing_patterns: List of exit timing error patterns
            
        Returns:
            Exit speed adjustment factor
        """
        if not exit_timing_patterns:
            return 1.0
        early_exits = sum(1 for p in exit_timing_patterns if p.
            error_subtype == 'early_exit')
        late_exits = sum(1 for p in exit_timing_patterns if p.error_subtype ==
            'late_exit')
        total_patterns = len(exit_timing_patterns)
        if total_patterns == 0:
            return 1.0
        early_ratio = early_exits / total_patterns
        late_ratio = late_exits / total_patterns
        if early_ratio > late_ratio:
            return 1.0 - early_ratio * 0.3
        elif late_ratio > early_ratio:
            return 1.0 + late_ratio * 0.3
        else:
            return 1.0

    def _calculate_stop_loss_adjustment(self, stop_loss_patterns: List[
        ErrorPattern]) ->float:
        """
        Calculate stop loss adjustment based on stop loss error patterns
        
        Args:
            stop_loss_patterns: List of stop loss error patterns
            
        Returns:
            Stop loss adjustment factor
        """
        if not stop_loss_patterns:
            return 1.0
        premature_stops = sum(1 for p in stop_loss_patterns if p.
            error_subtype == 'premature_stop')
        missed_stops = sum(1 for p in stop_loss_patterns if p.error_subtype ==
            'missed_stop')
        total_patterns = len(stop_loss_patterns)
        if total_patterns == 0:
            return 1.0
        premature_ratio = premature_stops / total_patterns
        missed_ratio = missed_stops / total_patterns
        if premature_ratio > missed_ratio:
            return 1.0 + premature_ratio * 0.25
        elif missed_ratio > premature_ratio:
            return 1.0 - missed_ratio * 0.25
        else:
            return 1.0
