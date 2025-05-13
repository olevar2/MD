"""
Feedback Router for the Forex Trading Platform.

This module handles the structured routing of trading feedback to appropriate services
based on the type of feedback and its intended target.
"""
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
import uuid
from enum import Enum
import httpx
from strategy_execution_engine.trading.feedback_collector import FeedbackCollector
from strategy_execution_engine.adaptive_layer.strategy_mutator import StrategyMutator
from analysis_engine.adaptive_layer.timeframe_feedback_service import TimeframeFeedbackService
from analysis_engine.adaptive_layer.statistical_validator import StatisticalValidator
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FeedbackTarget(Enum):
    """Enum defining different targets for feedback routing."""
    STRATEGY_PARAMS = 'strategy_params'
    MODEL_RETRAINING = 'model_retraining'
    TIMEFRAME_PREDICTION = 'timeframe_prediction'
    MARKET_REGIME = 'market_regime'
    RISK_MANAGEMENT = 'risk_management'
    EXECUTION_QUALITY = 'execution_quality'


class FeedbackRouter:
    """
    Routes trading feedback to appropriate service endpoints based on feedback type.
    This class serves as the central hub for the bidirectional feedback loop.
    """

    def __init__(self, feedback_collector: Optional[FeedbackCollector]=None,
        timeframe_feedback_service: Optional[TimeframeFeedbackService]=None,
        statistical_validator: Optional[StatisticalValidator]=None,
        strategy_mutator: Optional[StrategyMutator]=None, ml_service_url:
        str='http://ml-integration-service/retrain', risk_service_url: str=
        'http://risk-management-service/feedback',
        market_regime_service_url: str=
        'http://analysis-engine-service/market-regime/feedback'):
        """
        Initialize the feedback router with required services and target URLs.

        Args:
            feedback_collector: The FeedbackCollector instance
            timeframe_feedback_service: The TimeframeFeedbackService instance
            statistical_validator: The StatisticalValidator instance
            strategy_mutator: The StrategyMutator instance
            ml_service_url: URL for the ML model retraining endpoint
            risk_service_url: URL for the risk management feedback endpoint
            market_regime_service_url: URL for the market regime feedback endpoint
        """
        self.feedback_collector = feedback_collector or FeedbackCollector()
        self.timeframe_feedback_service = timeframe_feedback_service
        self.statistical_validator = (statistical_validator or
            StatisticalValidator())
        self.strategy_mutator = strategy_mutator
        self.ml_service_url = ml_service_url
        self.risk_service_url = risk_service_url
        self.market_regime_service_url = market_regime_service_url
        self.routers = {}
        self._initialize_routing_table()
        self.logger = logger
        self.http_client = httpx.AsyncClient()

    def _initialize_routing_table(self) ->None:
        """Initialize the routing table mapping feedback types to handling functions."""
        self.routers = {FeedbackTarget.STRATEGY_PARAMS: self.
            _route_strategy_params_feedback, FeedbackTarget.
            MODEL_RETRAINING: self._route_model_retraining_feedback,
            FeedbackTarget.TIMEFRAME_PREDICTION: self.
            _route_timeframe_prediction_feedback, FeedbackTarget.
            MARKET_REGIME: self._route_market_regime_feedback,
            FeedbackTarget.RISK_MANAGEMENT: self.
            _route_risk_management_feedback, FeedbackTarget.
            EXECUTION_QUALITY: self._route_execution_quality_feedback}

    @with_exception_handling
    def _convert_targets_to_enum(self, targets: List[Union[FeedbackTarget,
        str]]) ->List[FeedbackTarget]:
        """Convert a list of string or enum targets to a list of enums, logging warnings for invalid ones."""
        enum_targets = []
        for target in targets:
            if isinstance(target, str):
                try:
                    enum_targets.append(FeedbackTarget(target))
                except ValueError:
                    self.logger.warning(
                        f'Invalid feedback target string: {target}')
            elif isinstance(target, FeedbackTarget):
                enum_targets.append(target)
            else:
                self.logger.warning(f'Invalid target type: {type(target)}')
        return enum_targets

    def _prepare_feedback_metadata(self, feedback: Dict[str, Any]) ->Dict[
        str, Any]:
        """Add required metadata to feedback if not present."""
        if 'routing_id' not in feedback:
            feedback['routing_id'] = str(uuid.uuid4())
        if 'timestamp' not in feedback:
            feedback['timestamp'] = datetime.now(timezone.utc).isoformat()
        return feedback

    async def _process_routing_results(self, enum_targets: List[
        FeedbackTarget], routing_results: List[Any]) ->Dict[str, Any]:
        """Process results from routing tasks."""
        results = {}
        for i, target in enumerate(enum_targets):
            if target in self.routers:
                result = routing_results[i]
                if isinstance(result, Exception):
                    self.logger.error(
                        f'Error during routing feedback to {target.value}: {result}'
                        )
                    results[target.value] = {'error': str(result)}
                else:
                    results[target.value] = result
        return results

    async def route_feedback(self, feedback: Dict[str, Any], targets: List[
        Union[FeedbackTarget, str]]=None) ->Dict[str, Any]:
        """
        Route feedback to appropriate service endpoints.
        
        Args:
            feedback: The feedback data to route
            targets: List of targets to route the feedback to (if None, determine from feedback content)
            
        Returns:
            Dictionary with routing results
        """
        if not feedback:
            return {'error': 'Empty feedback provided'}
        feedback = self._prepare_feedback_metadata(feedback)
        enum_targets = self._convert_targets_to_enum(targets or self.
            _determine_targets(feedback))
        if not enum_targets:
            return {'error': 'No valid targets determined for feedback routing'
                }
        results = {}
        routing_tasks = []
        for target in enum_targets:
            if target in self.routers:
                routing_tasks.append(self._execute_route(target, feedback))
            else:
                self.logger.warning(
                    f'No router defined for target: {target.value}')
                results[target.value] = {'error':
                    'No router defined for this target'}
        if routing_tasks:
            routing_results = await asyncio.gather(*routing_tasks,
                return_exceptions=True)
            results.update(await self._process_routing_results([t for t in
                enum_targets if t in self.routers], routing_results))
        return {'routing_id': feedback['routing_id'], 'targets_processed':
            len(results), 'results': results}

    @async_with_exception_handling
    async def _execute_route(self, target: FeedbackTarget, feedback: Dict[
        str, Any]) ->Dict[str, Any]:
        """Helper coroutine to execute a single route and handle exceptions."""
        try:
            return await self.routers[target](feedback)
        except Exception as e:
            self.logger.error(f'Exception in router for {target.value}: {e}',
                exc_info=True)
            raise

    def _determine_targets(self, feedback: Dict[str, Any]) ->List[
        FeedbackTarget]:
        """
        Determine appropriate targets based on feedback content.
        Removed unused 'category' variable.
        
        Args:
            feedback: The feedback data
            
        Returns:
            List of determined feedback targets
        """
        targets = []
        has_prediction = ('prediction' in feedback or 'prediction_error' in
            feedback)
        has_strategy_params = 'strategy_params' in feedback
        has_execution_metrics = 'execution_metrics' in feedback
        has_market_regime = 'market_regime' in feedback
        has_timeframe = 'timeframe' in feedback
        if has_prediction or 'model_id' in feedback:
            targets.append(FeedbackTarget.MODEL_RETRAINING)
        if has_strategy_params or 'strategy_id' in feedback:
            targets.append(FeedbackTarget.STRATEGY_PARAMS)
        if has_timeframe and has_prediction:
            targets.append(FeedbackTarget.TIMEFRAME_PREDICTION)
        if has_market_regime:
            targets.append(FeedbackTarget.MARKET_REGIME)
        if 'risk_metrics' in feedback or 'drawdown' in feedback:
            targets.append(FeedbackTarget.RISK_MANAGEMENT)
        if has_execution_metrics or 'slippage' in feedback:
            targets.append(FeedbackTarget.EXECUTION_QUALITY)
        return targets

    @async_with_exception_handling
    async def _route_strategy_params_feedback(self, feedback: Dict[str, Any]
        ) ->Dict[str, Any]:
        """
        Route feedback for strategy parameter optimization.
        
        Args:
            feedback: The feedback data
            
        Returns:
            Dictionary with routing results
        """
        self.logger.info(
            f"Routing strategy parameter feedback: {feedback.get('routing_id')}"
            )
        if not self.strategy_mutator:
            return {'status': 'error', 'reason':
                'StrategyMutator not initialized', 'timestamp': datetime.
                now(timezone.utc).isoformat()}
        if 'trade_result' in feedback:
            try:
                trade_result = feedback['trade_result']
                self.strategy_mutator.register_trade_result(trade_result)
                return {'status': 'success', 'message':
                    'Trade result registered for parameter adaptation',
                    'timestamp': datetime.now(timezone.utc).isoformat()}
            except Exception as e:
                self.logger.error(f'Error registering trade result: {str(e)}')
                return {'status': 'error', 'reason':
                    f'Error registering trade result: {str(e)}',
                    'timestamp': datetime.now(timezone.utc).isoformat()}
        elif 'parameter_feedback' in feedback:
            try:
                parameter_feedback = feedback['parameter_feedback']
                result = self.strategy_mutator.register_feedback(
                    parameter_feedback)
                return {'status': 'success', 'result': result, 'timestamp':
                    datetime.now(timezone.utc).isoformat()}
            except Exception as e:
                self.logger.error(
                    f'Error registering parameter feedback: {str(e)}')
                return {'status': 'error', 'reason':
                    f'Error registering parameter feedback: {str(e)}',
                    'timestamp': datetime.now(timezone.utc).isoformat()}
        return {'status': 'acknowledged', 'timestamp': datetime.now(
            timezone.utc).isoformat()}

    @async_with_exception_handling
    async def _route_model_retraining_feedback(self, feedback: Dict[str, Any]
        ) ->Dict[str, Any]:
        """
        Route feedback for model retraining by sending it to the ML integration service.

        Args:
            feedback: The feedback data

        Returns:
            Dictionary with routing results
        """
        self.logger.info(
            f"Routing model retraining feedback: {feedback.get('routing_id')}")
        try:
            response = await self.http_client.post(self.ml_service_url,
                json=feedback, timeout=10.0)
            response.raise_for_status()
            service_response = response.json()
            self.logger.info(
                f"Successfully routed feedback {feedback.get('routing_id')} to ML service. Response: {service_response}"
                )
            return {'status': 'routed_successfully', 'target_service':
                'ml_integration', 'service_response': service_response,
                'timestamp': datetime.now(timezone.utc).isoformat()}
        except httpx.RequestError as e:
            self.logger.error(
                f"HTTP error routing feedback {feedback.get('routing_id')} to ML service {self.ml_service_url}: {e}"
                )
            return {'status': 'error', 'reason':
                f'HTTP request failed: {str(e)}', 'timestamp': datetime.now
                (timezone.utc).isoformat()}
        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"HTTP status error routing feedback {feedback.get('routing_id')} to ML service {self.ml_service_url}: {e.response.status_code} - {e.response.text}"
                )
            return {'status': 'error', 'reason':
                f'HTTP status error: {e.response.status_code}', 'details':
                e.response.text, 'timestamp': datetime.now(timezone.utc).
                isoformat()}
        except Exception as e:
            self.logger.error(
                f"Unexpected error routing feedback {feedback.get('routing_id')} to ML service: {e}"
                )
            return {'status': 'error', 'reason':
                f'Unexpected error: {str(e)}', 'timestamp': datetime.now(
                timezone.utc).isoformat()}

    async def _route_timeframe_prediction_feedback(self, feedback: Dict[str,
        Any]) ->Dict[str, Any]:
        """
        Route feedback regarding timeframe predictions.
        
        Args:
            feedback: The feedback data
            
        Returns:
            Dictionary with routing results
        """
        if not self.timeframe_feedback_service:
            return {'error': 'TimeframeFeedbackService not initialized'}
        self.logger.info(
            f"Routing timeframe prediction feedback: {feedback.get('routing_id')}"
            )
        instrument = feedback.get('instrument_id', '').replace('_', '/')
        timeframe = feedback.get('timeframe', '')
        prediction_error = feedback.get('prediction_error')
        prediction_direction = feedback.get('prediction_direction')
        actual_direction = feedback.get('actual_direction')
        result = (await self.timeframe_feedback_service.
            register_prediction_feedback(instrument=instrument, timeframe=
            timeframe, error=prediction_error, predicted_direction=
            prediction_direction, actual_direction=actual_direction,
            metadata=feedback.get('metadata', {})))
        return {'status': 'processed', 'result': result, 'timestamp':
            datetime.now(timezone.utc).isoformat()}

    @async_with_exception_handling
    async def _route_market_regime_feedback(self, feedback: Dict[str, Any]
        ) ->Dict[str, Any]:
        """
        Route feedback regarding market regime detection.
        
        Args:
            feedback: The feedback data
            
        Returns:
            Dictionary with routing results
        """
        self.logger.info(
            f"Routing market regime feedback: {feedback.get('routing_id')}")
        try:
            instrument = feedback.get('instrument', '')
            regime_type = feedback.get('market_regime', {}).get('type',
                'unknown')
            confidence = feedback.get('market_regime', {}).get('confidence',
                0.0)
            self.logger.debug(
                f'Sending market regime feedback for {instrument}: {regime_type} (conf: {confidence})'
                )
            response = await self.http_client.post(self.
                market_regime_service_url, json={'instrument': instrument,
                'regime_type': regime_type, 'confidence': confidence,
                'metadata': feedback.get('metadata', {}), 'source_id':
                feedback.get('routing_id'), 'timestamp': feedback.get(
                'timestamp', datetime.now(timezone.utc).isoformat())},
                timeout=10.0)
            response.raise_for_status()
            service_response = response.json()
            return {'status': 'routed_successfully', 'target_service':
                'market_regime', 'service_response': service_response,
                'timestamp': datetime.now(timezone.utc).isoformat()}
        except httpx.RequestError as e:
            self.logger.error(
                f'HTTP request error routing market regime feedback: {e}')
            return {'status': 'error', 'reason':
                f'HTTP request failed: {str(e)}', 'timestamp': datetime.now
                (timezone.utc).isoformat()}
        except httpx.HTTPStatusError as e:
            self.logger.error(
                f'HTTP status error from market regime service: {e.response.status_code}'
                )
            return {'status': 'error', 'reason':
                f'HTTP status error: {e.response.status_code}', 'details': 
                e.response.text if hasattr(e, 'response') and hasattr(e.
                response, 'text') else 'Unknown error', 'timestamp':
                datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            self.logger.error(
                f'Unexpected error routing market regime feedback: {str(e)}')
            return {'status': 'error', 'reason':
                f'Unexpected error: {str(e)}', 'timestamp': datetime.now(
                timezone.utc).isoformat()}

    @async_with_exception_handling
    async def _route_risk_management_feedback(self, feedback: Dict[str, Any]
        ) ->Dict[str, Any]:
        """
        Route feedback regarding risk management.
        
        Args:
            feedback: The feedback data
            
        Returns:
            Dictionary with routing results
        """
        self.logger.info(
            f"Routing risk management feedback: {feedback.get('routing_id')}")
        try:
            risk_metrics = feedback.get('risk_metrics', {})
            drawdown = feedback.get('drawdown')
            strategy_id = feedback.get('strategy_id', '')
            if not (risk_metrics or drawdown):
                return {'status': 'error', 'reason':
                    'Missing risk metrics or drawdown in feedback',
                    'timestamp': datetime.now(timezone.utc).isoformat()}
            self.logger.debug(
                f'Sending risk management feedback for strategy {strategy_id}')
            payload = {'strategy_id': strategy_id, 'risk_metrics':
                risk_metrics, 'source_id': feedback.get('routing_id'),
                'timestamp': feedback.get('timestamp', datetime.now(
                timezone.utc).isoformat())}
            if drawdown is not None:
                payload['drawdown'] = drawdown
            response = await self.http_client.post(self.risk_service_url,
                json=payload, timeout=10.0)
            response.raise_for_status()
            service_response = response.json()
            return {'status': 'routed_successfully', 'target_service':
                'risk_management', 'service_response': service_response,
                'timestamp': datetime.now(timezone.utc).isoformat()}
        except httpx.RequestError as e:
            self.logger.error(
                f'HTTP request error routing risk management feedback: {e}')
            return {'status': 'error', 'reason':
                f'HTTP request failed: {str(e)}', 'timestamp': datetime.now
                (timezone.utc).isoformat()}
        except httpx.HTTPStatusError as e:
            self.logger.error(
                f'HTTP status error from risk management service: {e.response.status_code}'
                )
            return {'status': 'error', 'reason':
                f'HTTP status error: {e.response.status_code}', 'details': 
                e.response.text if hasattr(e, 'response') and hasattr(e.
                response, 'text') else 'Unknown error', 'timestamp':
                datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            self.logger.error(
                f'Unexpected error routing risk management feedback: {str(e)}')
            return {'status': 'error', 'reason':
                f'Unexpected error: {str(e)}', 'timestamp': datetime.now(
                timezone.utc).isoformat()}

    @async_with_exception_handling
    async def _route_execution_quality_feedback(self, feedback: Dict[str, Any]
        ) ->Dict[str, Any]:
        """
        Route feedback regarding execution quality.
        
        Args:
            feedback: The feedback data
            
        Returns:
            Dictionary with routing results
        """
        self.logger.info(
            f"Routing execution quality feedback: {feedback.get('routing_id')}"
            )
        try:
            execution_metrics = feedback.get('execution_metrics', {})
            instrument = feedback.get('instrument', '')
            strategy_id = feedback.get('strategy_id', '')
            if not execution_metrics:
                return {'status': 'error', 'reason':
                    'Missing execution metrics in feedback', 'timestamp':
                    datetime.now(timezone.utc).isoformat()}
            if 'slippage_pips' in execution_metrics:
                slippage_pips = execution_metrics['slippage_pips']
                price_improvement = execution_metrics.get('price_improvement',
                    False)
                self.logger.info(
                    f'Execution quality metrics for {instrument} (strategy: {strategy_id}): slippage_pips={slippage_pips}, price_improvement={price_improvement}'
                    )
                await self.feedback_collector.store_execution_quality_metrics(
                    instrument=instrument, strategy_id=strategy_id, metrics
                    =execution_metrics, feedback_id=feedback.get('routing_id'))
            if 'execution_time_ms' in execution_metrics:
                execution_time_ms = execution_metrics['execution_time_ms']
                self.logger.info(
                    f'Execution time for {instrument}: {execution_time_ms}ms')
            if self.strategy_mutator and hasattr(self.strategy_mutator,
                'process_execution_quality'):
                await self.strategy_mutator.process_execution_quality(
                    strategy_id=strategy_id, instrument=instrument,
                    execution_metrics=execution_metrics, market_conditions=
                    feedback.get('market_conditions', {}))
            return {'status': 'processed', 'instrument': instrument,
                'strategy_id': strategy_id, 'metrics_processed': list(
                execution_metrics.keys()), 'timestamp': datetime.now(
                timezone.utc).isoformat()}
        except Exception as e:
            self.logger.error(
                f'Error processing execution quality feedback: {str(e)}')
            return {'status': 'error', 'reason': str(e), 'timestamp':
                datetime.now(timezone.utc).isoformat()}

    async def close_client(self):
    """
    Close client.
    
    """

        await self.http_client.aclose()
