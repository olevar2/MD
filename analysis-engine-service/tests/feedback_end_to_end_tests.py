"""
Feedback Loop End-to-End Tests

This module provides comprehensive end-to-end testing for the entire feedback loop pipeline,
ensuring that the system correctly processes feedback, performs statistical validation,
and adapts strategies based on validated insights.

Tests cover:
- Feedback collection and routing
- Parameter tracking and effectiveness measurement
- Statistical validation of adaptation decisions
- Integration with adaptation engine
- Regime-specific adaptation
- Cross-component event propagation
"""
import pytest
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import logging
import os
import json
from analysis_engine.adaptive_layer.feedback_loop import FeedbackLoop
from analysis_engine.adaptive_layer.feedback_loop_validator import FeedbackLoopValidator
from analysis_engine.adaptive_layer.parameter_tracking_service import ParameterTrackingService
from analysis_engine.adaptive_layer.parameter_statistical_analyzer import ParameterStatisticalAnalyzer
from analysis_engine.adaptive_layer.feedback_router import FeedbackRouter
from analysis_engine.adaptive_layer.model_feedback_integrator import ModelFeedbackIntegrator
from analysis_engine.adaptive_layer.adaptation_engine import AdaptationEngine
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from core_foundations.models.feedback import TradeFeedback, FeedbackSource, FeedbackCategory, FeedbackStatus
from core_foundations.events.event_bus import EventBus
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MockEventBus(EventBus):
    """Mock implementation of EventBus for testing."""

    def __init__(self):
        self.published_events = []
        self.event_handlers = {}

    async def publish(self, event_type: str, data: Any) ->None:
        """Record published events."""
        self.published_events.append({'event_type': event_type, 'data':
            data, 'timestamp': datetime.utcnow().isoformat()})

    async def subscribe(self, event_type: str, handler) ->None:
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    @with_resilience('process_published_events')
    async def process_published_events(self):
        """Process all published events by calling registered handlers."""
        for event in self.published_events:
            event_type = event['event_type']
            data = event['data']
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    await handler(data)


class MockToolEffectivenessRepository(ToolEffectivenessRepository):
    """Mock repository for testing."""

    def __init__(self):
    """
      init  .
    
    """

        self.strategy_outcomes = {}
        self.parameter_outcomes = {}
        self.adaptation_outcomes = {}
        self.market_regimes = {}

    @with_resilience('get_parameter_variation_outcomes')
    def get_parameter_variation_outcomes(self, strategy_id: str,
        parameter_name: str, adaptation_id: Optional[str]=None) ->List[Dict
        [str, Any]]:
        """Get parameter outcomes."""
        key = f'{strategy_id}:{parameter_name}'
        return self.parameter_outcomes.get(key, [])

    @with_resilience('get_strategy_outcomes')
    def get_strategy_outcomes(self, strategy_id: str, market_regime:
        Optional[str]=None) ->List[Dict[str, Any]]:
        """Get strategy outcomes."""
        outcomes = self.strategy_outcomes.get(strategy_id, [])
        if market_regime:
            return [o for o in outcomes if o.get('market_regime') ==
                market_regime]
        return outcomes

    @with_resilience('get_adaptation_outcomes')
    def get_adaptation_outcomes(self, adaptation_id: str) ->List[Dict[str, Any]
        ]:
        """Get adaptation outcomes."""
        return self.adaptation_outcomes.get(adaptation_id, [])

    @with_resilience('get_adaptation_details')
    def get_adaptation_details(self, adaptation_id: str) ->Dict[str, Any]:
        """Get adaptation details."""
        for adaptations in self.adaptation_outcomes.values():
            for adaptation in adaptations:
                if adaptation.get('adaptation_id') == adaptation_id:
                    return adaptation
        return {}

    @with_resilience('get_outcomes_before_date')
    @with_exception_handling
    def get_outcomes_before_date(self, strategy_id: str, date: str, days: int
        ) ->List[Dict[str, Any]]:
        """Get outcomes before a date."""
        outcomes = self.strategy_outcomes.get(strategy_id, [])
        try:
            target_date = datetime.fromisoformat(date)
            start_date = target_date - timedelta(days=days)
            return [o for o in outcomes if start_date <= datetime.
                fromisoformat(o.get('timestamp', '')) < target_date]
        except (ValueError, TypeError):
            return []

    @with_resilience('get_regimes_for_strategy')
    def get_regimes_for_strategy(self, strategy_id: str) ->List[str]:
        """Get market regimes for a strategy."""
        return self.market_regimes.get(strategy_id, [])

    def add_parameter_outcomes(self, strategy_id: str, parameter_name: str,
        outcomes: List[Dict[str, Any]]) ->None:
        """Add parameter outcomes for testing."""
        key = f'{strategy_id}:{parameter_name}'
        if key not in self.parameter_outcomes:
            self.parameter_outcomes[key] = []
        self.parameter_outcomes[key].extend(outcomes)

    def add_strategy_outcomes(self, strategy_id: str, outcomes: List[Dict[
        str, Any]]) ->None:
        """Add strategy outcomes for testing."""
        if strategy_id not in self.strategy_outcomes:
            self.strategy_outcomes[strategy_id] = []
        self.strategy_outcomes[strategy_id].extend(outcomes)

    def add_adaptation_outcomes(self, adaptation_id: str, outcomes: List[
        Dict[str, Any]]) ->None:
        """Add adaptation outcomes for testing."""
        if adaptation_id not in self.adaptation_outcomes:
            self.adaptation_outcomes[adaptation_id] = []
        self.adaptation_outcomes[adaptation_id].extend(outcomes)

    def set_market_regimes(self, strategy_id: str, regimes: List[str]) ->None:
        """Set market regimes for testing."""
        self.market_regimes[strategy_id] = regimes


class MockAdaptationEngine:
    """Mock adaptation engine for testing."""

    def __init__(self):
    """
      init  .
    
    """

        self.adaptations = []
        self.strategies = {}

    async def adapt_strategy(self, strategy_id: str, market_conditions:
        Dict[str, Any], feedback_data: Optional[Dict[str, Any]]=None) ->Dict[
        str, Any]:
        """Mock adaptation."""
        adaptation_id = str(uuid.uuid4())
        adaptation = {'adaptation_id': adaptation_id, 'strategy_id':
            strategy_id, 'timestamp': datetime.utcnow().isoformat(),
            'market_conditions': market_conditions, 'feedback_data':
            feedback_data, 'parameters_adjusted': {}}
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            for param, config in strategy.get('parameters', {}).items():
                current = config_manager.get('current', 0)
                min_val = config_manager.get('min', current - 1)
                max_val = config_manager.get('max', current + 1)
                new_value = current + (max_val - min_val) * (np.random.
                    random() - 0.5) * 0.1
                new_value = max(min_val, min(max_val, new_value))
                adaptation['parameters_adjusted'][param] = {'old_value':
                    current, 'new_value': new_value, 'confidence': np.
                    random.random()}
                self.strategies[strategy_id]['parameters'][param]['current'
                    ] = new_value
        self.adaptations.append(adaptation)
        return adaptation

    def add_strategy(self, strategy_id: str, parameters: Dict[str, Dict[str,
        Any]]) ->None:
        """Add strategy for testing."""
        self.strategies[strategy_id] = {'strategy_id': strategy_id,
            'parameters': parameters}


@pytest.fixture
def event_bus():
    """Create mock event bus."""
    return MockEventBus()


@pytest.fixture
def repository():
    """Create mock repository."""
    repo = MockToolEffectivenessRepository()
    repo.set_market_regimes('test_strategy', ['trend', 'range', 'volatile'])
    generate_test_data(repo)
    return repo


@pytest.fixture
def adaptation_engine():
    """Create mock adaptation engine."""
    engine = MockAdaptationEngine()
    engine.add_strategy('test_strategy', {'stop_loss': {'current': 20,
        'min': 10, 'max': 50}, 'take_profit': {'current': 40, 'min': 20,
        'max': 100}, 'entry_threshold': {'current': 0.7, 'min': 0.5, 'max':
        0.9}})
    return engine


@pytest.fixture
def feedback_loop(adaptation_engine):
    """Create feedback loop."""
    return FeedbackLoop(adaptation_engine)


@pytest.fixture
def validator(feedback_loop, repository):
    """Create validator."""
    return FeedbackLoopValidator(feedback_loop, repository)


@pytest.fixture
def parameter_tracking(event_bus):
    """Create parameter tracking service."""
    return ParameterTrackingService(event_bus)


@pytest.fixture
def statistical_analyzer(validator, parameter_tracking, repository):
    """Create statistical analyzer."""
    return ParameterStatisticalAnalyzer(validator, parameter_tracking,
        repository)


@pytest.fixture
def feedback_router(event_bus):
    """Create feedback router."""
    return FeedbackRouter(event_bus)


@pytest.fixture
def model_feedback_integrator(event_bus):
    """Create model feedback integrator."""
    return ModelFeedbackIntegrator(event_bus=event_bus)


def generate_test_data(repo: MockToolEffectivenessRepository) ->None:
    """Generate test data for the repository."""
    strategy_id = 'test_strategy'
    param_name = 'stop_loss'
    param_values = [15, 20, 25, 30]
    regimes = ['trend', 'range', 'volatile']
    for regime in regimes:
        for value in param_values:
            if regime == 'trend':
                base_profit = value * 0.5
                variance = 10
            elif regime == 'range':
                base_profit = 20 - abs(value - 22.5)
                variance = 8
            else:
                base_profit = 30 - value * 0.8
                variance = 15
            outcomes = []
            for i in range(40):
                profit = base_profit + np.random.normal(0, variance)
                outcomes.append({'strategy_id': strategy_id,
                    'parameter_name': param_name, 'parameter_value': value,
                    'market_regime': regime, 'profit': profit, 'win': 
                    profit > 0, 'timestamp': (datetime.now() - timedelta(
                    days=i % 30)).isoformat()})
            repo.add_parameter_outcomes(strategy_id, param_name, outcomes)
    adaptation_id = 'test_adaptation'
    adaptation_outcomes = []
    control_base_profit = 5
    for i in range(50):
        profit = control_base_profit + np.random.normal(0, 10)
        adaptation_outcomes.append({'adaptation_id': adaptation_id,
            'strategy_id': strategy_id, 'profit': profit, 'win': profit > 0,
            'timestamp': (datetime.now() - timedelta(days=i + 30)).isoformat()}
            )
    adapted_base_profit = 10
    for i in range(50):
        profit = adapted_base_profit + np.random.normal(0, 10)
        adaptation_outcomes.append({'adaptation_id': adaptation_id,
            'strategy_id': strategy_id, 'profit': profit, 'win': profit > 0,
            'timestamp': (datetime.now() - timedelta(days=i)).isoformat()})
    repo.add_adaptation_outcomes(adaptation_id, adaptation_outcomes)
    repo.add_strategy_outcomes(strategy_id, adaptation_outcomes)


def generate_trade_feedback(strategy_id: str, parameter_name: Optional[str]
    =None, parameter_value: Any=None, profit: Optional[float]=None,
    market_regime: Optional[str]=None, source: FeedbackSource=
    FeedbackSource.STRATEGY_EXECUTION, category: FeedbackCategory=
    FeedbackCategory.SUCCESS) ->TradeFeedback:
    """Generate mock trade feedback."""
    if profit is None:
        profit = np.random.normal(5, 10)
    if market_regime is None:
        market_regime = np.random.choice(['trend', 'range', 'volatile'])
    feedback_id = str(uuid.uuid4())
    outcome_metrics = {'profit': profit, 'win': profit > 0}
    if parameter_name and parameter_value is not None:
        outcome_metrics['parameter_name'] = parameter_name
        outcome_metrics['parameter_value'] = parameter_value
    return TradeFeedback(feedback_id=feedback_id, strategy_id=strategy_id,
        source=source, category=category, status=FeedbackStatus.PENDING,
        timestamp=datetime.utcnow(), instrument='EURUSD', timeframe='H1',
        market_regime=market_regime, outcome_metrics=outcome_metrics)


class TestFeedbackLoopEndToEnd:
    """Comprehensive end-to-end tests for the feedback loop pipeline."""

    @pytest.mark.asyncio
    async def test_feedback_collection_and_routing(self, feedback_router,
        event_bus):
        """Test that feedback collection and routing works correctly."""
        received_feedback = []

        async def test_handler(feedback):
            received_feedback.append(feedback)
            return True
        feedback_router.add_route(source=FeedbackSource.STRATEGY_EXECUTION,
            category=FeedbackCategory.SUCCESS, handler=test_handler)
        feedback = generate_trade_feedback('test_strategy')
        result = await feedback_router.route_feedback(feedback)
        assert result is True
        assert len(received_feedback) == 1
        assert received_feedback[0].feedback_id == feedback.feedback_id
        assert feedback.status == FeedbackStatus.ROUTED
        published_events = [e for e in event_bus.published_events if e[
            'event_type'] == 'feedback.routed']
        assert len(published_events) == 1
        assert published_events[0]['data']['feedback_id'
            ] == feedback.feedback_id

    @pytest.mark.asyncio
    async def test_parameter_tracking_and_effectiveness(self,
        parameter_tracking):
        """Test parameter tracking and effectiveness calculation."""
        strategy_id = 'test_strategy'
        parameter_name = 'stop_loss'
        old_value = 20
        new_value = 25
        param_id = await parameter_tracking.record_parameter_change(strategy_id
            =strategy_id, parameter_name=parameter_name, old_value=
            old_value, new_value=new_value, change_reason='Test change',
            source_component='test', confidence_level=0.8)
        history = parameter_tracking.get_parameter_history(strategy_id,
            parameter_name)
        assert len(history) == 1
        assert history[0]['parameter_id'] == param_id
        assert history[0]['old_value'] == old_value
        assert history[0]['new_value'] == new_value
        success = await parameter_tracking.record_parameter_performance(
            parameter_id=param_id, strategy_id=strategy_id, parameter_name=
            parameter_name, performance_metrics={'win_rate': 0.65,
            'profit_factor': 1.5, 'profit': 100})
        assert success is True
        effectiveness = (await parameter_tracking.
            calculate_parameter_effectiveness(strategy_id=strategy_id,
            parameter_name=parameter_name))
        assert effectiveness['parameter'] == parameter_name
        assert effectiveness['strategy_id'] == strategy_id
        assert effectiveness['sample_size'] == 1
        assert effectiveness['confidence'] > 0

    @pytest.mark.asyncio
    async def test_statistical_validation(self, validator, repository):
        """Test statistical validation of parameter effectiveness."""
        strategy_id = 'test_strategy'
        parameter_name = 'stop_loss'
        result = validator.validate_parameter_effectiveness(strategy_id=
            strategy_id, parameter_name=parameter_name)
        assert result['status'] == 'completed'
        assert result['parameter'] == parameter_name
        assert result['strategy_id'] == strategy_id
        assert 'performance_metrics' in result
        assert 'tests' in result
        assert 'optimal_value' in result
        result = validator.validate_regime_effectiveness_differences(
            strategy_id=strategy_id, metric_name='profit')
        assert result['status'] == 'completed'
        assert result['strategy_id'] == strategy_id
        assert result['metric'] == 'profit'
        assert 'regimes' in result
        assert 'means' in result
        assert 'anova_result' in result
        assert 'pairwise_comparisons' in result

    @pytest.mark.asyncio
    async def test_advanced_parameter_analysis(self, statistical_analyzer):
        """Test advanced parameter statistical analysis."""
        strategy_id = 'test_strategy'
        parameter_name = 'stop_loss'
        result = await statistical_analyzer.analyze_parameter_cross_regime(
            strategy_id=strategy_id, parameter_name=parameter_name)
        assert result['status'] == 'completed'
        assert result['parameter'] == parameter_name
        assert 'regimes_analyzed' in result
        assert 'optimal_by_regime' in result
        assert 'sensitivity_by_regime' in result
        result = await statistical_analyzer.analyze_parameter_sensitivity(
            strategy_id=strategy_id, parameter_name=parameter_name)
        assert result['status'] == 'completed'
        assert result['parameter'] == parameter_name
        assert 'sensitivity_score' in result
        assert 'optimal_value' in result
        result = await statistical_analyzer.generate_parameter_insights(
            strategy_id=strategy_id, parameter_name=parameter_name)
        assert result['status'] == 'completed'
        assert result['parameter'] == parameter_name
        assert 'insights' in result
        assert 'recommendations' in result

    @pytest.mark.asyncio
    async def test_feedback_loop_integration(self, feedback_loop, repository):
        """Test feedback loop integration with outcomes."""
        strategy_id = 'test_strategy'
        adaptation_id = 'test_adaptation_1'
        feedback_loop.record_strategy_outcome(strategy_id=strategy_id,
            instrument='EURUSD', timeframe='H1', adaptation_id=
            adaptation_id, outcome_metrics={'profit': 50, 'win': True},
            market_regime='trend')
        performance = feedback_loop.get_performance_by_regime(strategy_id)
        assert 'trend' in performance
        assert performance['trend']['count'] == 1
        assert performance['trend']['win_count'] == 1
        effectiveness = feedback_loop.get_adaptation_effectiveness()
        assert 'strategy_type' in effectiveness
        assert 'regime_type' in effectiveness
        assert 'trend' in effectiveness['regime_type']
        insights = feedback_loop.generate_insights(strategy_id)
        assert isinstance(insights, list)

    @pytest.mark.asyncio
    async def test_adaptation_engine_integration(self, adaptation_engine,
        feedback_loop, validator, event_bus):
        """Test integration with adaptation engine."""
        strategy_id = 'test_strategy'
        adaptation = await adaptation_engine.adapt_strategy(strategy_id=
            strategy_id, market_conditions={'regime': 'trend', 'volatility':
            'high'})
        assert 'adaptation_id' in adaptation
        assert 'parameters_adjusted' in adaptation
        feedback_loop.record_strategy_outcome(strategy_id=strategy_id,
            instrument='EURUSD', timeframe='H1', adaptation_id=adaptation[
            'adaptation_id'], outcome_metrics={'profit': 50, 'win': True},
            market_regime='trend')
        result = validator.validate_adaptation_significance(adaptation_id=
            adaptation['adaptation_id'])
        assert 'status' in result

    @pytest.mark.asyncio
    async def test_model_feedback_integration(self,
        model_feedback_integrator, event_bus):
        """Test integration with model feedback."""
        model_id = 'test_model'
        feedback = generate_trade_feedback(strategy_id='test_strategy',
            source=FeedbackSource.MODEL_PREDICTION, profit=10)
        feedback.model_id = model_id
        feedback.outcome_metrics['error'] = 0.05
        feedback.outcome_metrics['prediction_value'] = 0.65
        feedback.outcome_metrics['actual_value'] = 0.7
        result = await model_feedback_integrator.collect_model_feedback(
            feedback)
        assert result is True
        assert model_id in model_feedback_integrator.feedback_by_model
        assert len(model_feedback_integrator.feedback_by_model[model_id][
            'recent_feedback']) == 1
        assert len(model_feedback_integrator.feedback_by_model[model_id][
            'error_trends']) == 1

    @pytest.mark.asyncio
    async def test_end_to_end_feedback_flow(self, feedback_router,
        parameter_tracking, feedback_loop, adaptation_engine, event_bus):
        """Test complete end-to-end feedback flow."""
        strategy_id = 'test_strategy'
        parameter_name = 'stop_loss'

        async def parameter_update_handler(feedback):
    """
    Parameter update handler.
    
    Args:
        feedback: Description of feedback
    
    """

            if feedback.outcome_metrics.get('profit', 0) > 0:
                param_id = str(uuid.uuid4())
                await parameter_tracking.record_parameter_performance(
                    parameter_id=param_id, strategy_id=feedback.strategy_id,
                    parameter_name=parameter_name, performance_metrics={
                    'win_rate': 1.0, 'profit_factor': 2.0, 'profit':
                    feedback.outcome_metrics.get('profit', 0)})
            return True
        feedback_router.add_route(source=FeedbackSource.STRATEGY_EXECUTION,
            category=FeedbackCategory.SUCCESS, handler=parameter_update_handler
            )

        async def feedback_event_handler(data):
    """
    Feedback event handler.
    
    Args:
        data: Description of data
    
    """

            strategy_id = data.get('strategy_id')
            if strategy_id:
                feedback_loop.record_strategy_outcome(strategy_id=
                    strategy_id, instrument=data.get('instrument', 'EURUSD'
                    ), timeframe=data.get('timeframe', 'H1'), adaptation_id
                    =data.get('adaptation_id', 'test_adaptation'),
                    outcome_metrics={'profit': data.get('profit', 0)},
                    market_regime=data.get('market_regime', 'trend'))
        await event_bus.subscribe('feedback.routed', feedback_event_handler)
        feedback = generate_trade_feedback(strategy_id=strategy_id,
            parameter_name=parameter_name, parameter_value=25, profit=100,
            market_regime='trend')
        await feedback_router.route_feedback(feedback)
        await event_bus.process_published_events()
        performance = feedback_loop.get_performance_by_regime(strategy_id)
        assert 'trend' in performance
        assert performance['trend']['count'] >= 1
        adaptation = await adaptation_engine.adapt_strategy(strategy_id=
            strategy_id, market_conditions={'regime': 'trend'},
            feedback_data={'recent_performance': 0.8})
        assert 'adaptation_id' in adaptation
        assert 'parameters_adjusted' in adaptation
        for param_name, param_data in adaptation['parameters_adjusted'].items(
            ):
            param_id = await parameter_tracking.record_parameter_change(
                strategy_id=strategy_id, parameter_name=param_name,
                old_value=param_data['old_value'], new_value=param_data[
                'new_value'], change_reason='Adaptation', source_component=
                'AdaptationEngine', confidence_level=param_data['confidence'])
            assert param_id is not None
        assert True


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
