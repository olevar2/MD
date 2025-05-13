"""
End-to-end tests for the adaptive feedback loop.
"""
import pytest
import unittest.mock as mock
import time
import json
from analysis_engine.adaptive_layer.strategy_adaptation_service import StrategyAdaptationService
from analysis_engine.adaptive_layer.parameter_statistical_analyzer import ParameterStatisticalAnalyzer
from analysis_engine.adaptive_layer.feedback_loop_validator import FeedbackLoopValidator


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@pytest.fixture
@with_exception_handling
def adaptation_service():
    """Fixture to provide an instance of the StrategyAdaptationService with mocked dependencies."""
    mock_kafka_consumer = mock.Mock()
    mock_db_client = mock.Mock()
    mock_analyzer = mock.Mock(spec=ParameterStatisticalAnalyzer)
    mock_validator = mock.Mock(spec=FeedbackLoopValidator)
    try:
        service = StrategyAdaptationService(kafka_consumer=
            mock_kafka_consumer, db_client=mock_db_client,
            statistical_analyzer=mock_analyzer, feedback_validator=
            mock_validator)
    except TypeError:
        service = StrategyAdaptationService()
        service.kafka_consumer = mock_kafka_consumer
        service.db_client = mock_db_client
        service.statistical_analyzer = mock_analyzer
        service.feedback_validator = mock_validator
    return (service, mock_kafka_consumer, mock_db_client, mock_analyzer,
        mock_validator)


def test_successful_feedback_processing(adaptation_service):
    """
    Test a typical successful feedback event processing flow.
    """
    service, mock_kafka, mock_db, mock_analyzer, mock_validator = (
        adaptation_service)
    feedback_event = {'event_id': 'evt_123', 'trade_id': 'trade_abc',
        'strategy_id': 'strat_xyz', 'outcome': {'pnl': 100.5, 'slippage': 
        0.0002}, 'parameters_used': {'param1': 10, 'param2': 0.5},
        'timestamp': time.time()}
    mock_validator.validate_feedback_event.return_value = True
    mock_analyzer.analyze_parameter_impact.return_value = {'suggested_change':
        {'param1': 11}}
    mock_db.get_strategy_state.return_value = {'parameters': {'param1': 10,
        'param2': 0.5}}
    mock_db.update_strategy_parameters.return_value = True
    service.process_feedback_event(feedback_event)
    mock_validator.validate_feedback_event.assert_called_once_with(
        feedback_event)
    mock_analyzer.analyze_parameter_impact.assert_called_once_with(
        feedback_event)
    mock_db.get_strategy_state.assert_called_once_with('strat_xyz')
    mock_db.update_strategy_parameters.assert_called_once_with('strat_xyz',
        {'param1': 11, 'param2': 0.5})


def test_invalid_feedback_event(adaptation_service):
    """
    Test handling of an invalid feedback event.
    """
    service, _, mock_db, mock_analyzer, mock_validator = adaptation_service
    feedback_event = {'event_id': 'evt_456', 'trade_id': 'trade_def'}
    mock_validator.validate_feedback_event.return_value = False
    service.process_feedback_event(feedback_event)
    mock_validator.validate_feedback_event.assert_called_once_with(
        feedback_event)
    mock_analyzer.analyze_parameter_impact.assert_not_called()
    mock_db.update_strategy_parameters.assert_not_called()
    mock_db.get_strategy_state.assert_not_called()


def test_ab_test_scenario(adaptation_service):
    """
    Test feedback processing when an A/B test is active.
    (Implementation depends heavily on how A/B testing state is managed and used)
    """
    pytest.skip('Test implementation depends on A/B test logic details')


@with_exception_handling
def test_error_handling_in_analyzer(adaptation_service):
    """
    Test how the service handles errors raised by the statistical analyzer.
    """
    service, _, mock_db, mock_analyzer, mock_validator = adaptation_service
    feedback_event = {'event_id': 'evt_err_1', 'trade_id': 'trade_err',
        'strategy_id': 'strat_err', 'outcome': {'pnl': -20},
        'parameters_used': {'param1': 5}, 'timestamp': time.time()}
    mock_validator.validate_feedback_event.return_value = True
    mock_analyzer.analyze_parameter_impact.side_effect = ValueError(
        'Statistical error during analysis')
    mock_db.get_strategy_state.return_value = {'parameters': {'param1': 5}}
    with mock.patch('logging.error') as mock_log_error:
        try:
            service.process_feedback_event(feedback_event)
        except Exception as e:
            pytest.fail(
                f'Service should handle internal errors gracefully, but raised {e}'
                )
    mock_validator.validate_feedback_event.assert_called_once_with(
        feedback_event)
    mock_db.get_strategy_state.assert_called_once_with('strat_err')
    mock_analyzer.analyze_parameter_impact.assert_called_once_with(
        feedback_event)
    mock_log_error.assert_called_once()
    assert 'Statistical error during analysis' in mock_log_error.call_args[0][0
        ]
    mock_db.update_strategy_parameters.assert_not_called()


def test_no_change_suggested(adaptation_service):
    """Test scenario where the analyzer suggests no parameter changes."""
    service, _, mock_db, mock_analyzer, mock_validator = adaptation_service
    feedback_event = {'event_id': 'evt_nochange', 'strategy_id':
        'strat_stable', 'outcome': {'pnl': 10}, 'parameters_used': {
        'param1': 20}, 'timestamp': time.time()}
    mock_validator.validate_feedback_event.return_value = True
    mock_analyzer.analyze_parameter_impact.return_value = {}
    mock_db.get_strategy_state.return_value = {'parameters': {'param1': 20}}
    service.process_feedback_event(feedback_event)
    mock_validator.validate_feedback_event.assert_called_once_with(
        feedback_event)
    mock_analyzer.analyze_parameter_impact.assert_called_once_with(
        feedback_event)
    mock_db.get_strategy_state.assert_called_once_with('strat_stable')
    mock_db.update_strategy_parameters.assert_not_called()


def test_feedback_for_loss(adaptation_service):
    """Test feedback processing for a losing trade."""
    service, _, mock_db, mock_analyzer, mock_validator = adaptation_service
    feedback_event = {'event_id': 'evt_loss_1', 'strategy_id': 'strat_loss',
        'outcome': {'pnl': -50.25, 'slippage': 0.0003}, 'parameters_used':
        {'param_risk': 0.02, 'param_entry': 1.1}, 'timestamp': time.time()}
    mock_validator.validate_feedback_event.return_value = True
    mock_analyzer.analyze_parameter_impact.return_value = {'suggested_change':
        {'param_risk': 0.015}}
    mock_db.get_strategy_state.return_value = {'parameters': {'param_risk':
        0.02, 'param_entry': 1.1}}
    mock_db.update_strategy_parameters.return_value = True
    service.process_feedback_event(feedback_event)
    mock_validator.validate_feedback_event.assert_called_once_with(
        feedback_event)
    mock_analyzer.analyze_parameter_impact.assert_called_once_with(
        feedback_event)
    mock_db.get_strategy_state.assert_called_once_with('strat_loss')
    mock_db.update_strategy_parameters.assert_called_once_with('strat_loss',
        {'param_risk': 0.015, 'param_entry': 1.1})
