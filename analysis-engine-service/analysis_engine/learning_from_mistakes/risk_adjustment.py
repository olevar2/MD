"""
Risk Adjustment Module

This module implements the integration between the Learning from Past Mistakes Module
and the Risk Management Service, enabling situation-specific risk adjustments based on
historical error patterns.
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json
from analysis_engine.learning_from_mistakes.error_pattern_recognition import ErrorPatternRecognitionSystem, ErrorPattern
from common_lib.risk import RiskManagementClient, RiskParameters
from common_lib.service_client.base_client import ServiceClientConfig
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class RiskAdjustmentManager:
    """
    The RiskAdjustmentManager integrates error pattern recognition with the Risk Management Service
    to implement situation-specific risk adjustments based on historical outcomes.

    Key capabilities:
    - Detect potential high-risk situations based on historical error patterns
    - Generate appropriate risk adjustment recommendations
    - Communicate with Risk Management Service to implement adjustments
    - Track effectiveness of risk adjustments over time
    """

    def __init__(self, error_pattern_system: ErrorPatternRecognitionSystem,
        risk_management_service_url: str, config: Dict[str, Any]=None):
        """
        Initialize the RiskAdjustmentManager.

        Args:
            error_pattern_system: The error pattern recognition system
            risk_management_service_url: URL for the Risk Management Service API
            config: Configuration parameters for the manager
        """
        self.error_pattern_system = error_pattern_system
        self.risk_management_service_url = risk_management_service_url
        self.config = config or {}
        client_config = ServiceClientConfig(base_url=
            risk_management_service_url, timeout=30, retry_config={
            'max_retries': 3, 'backoff_factor': 0.5, 'retry_statuses': [408,
            429, 500, 502, 503, 504]})
        self.risk_client = RiskManagementClient(config=client_config)
        self.risk_confidence_threshold = self.config.get(
            'risk_confidence_threshold', 0.7)
        self.max_risk_reduction_pct = self.config.get('max_risk_reduction_pct',
            0.5)
        self.pattern_weights = self.config.get('pattern_weights', {
            'trend_reversal': 1.0, 'stop_hunt': 1.2, 'news_impact': 1.5,
            'over_leveraged': 1.3, 'correlation_breakdown': 1.1,
            'volatility_spike': 1.4, 'signal_false_positive': 0.9})
        self.risk_adjustment_history = []
        logger.info(
            'RiskAdjustmentManager initialized with Risk Management Client')

    @with_resilience('check_for_risk_patterns')
    async def check_for_risk_patterns(self, strategy_id: str, instrument:
        str, timeframe: str, market_conditions: Dict[str, Any]) ->List[Dict
        [str, Any]]:
        """
        Check for risk patterns that match current market conditions.

        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument (e.g., 'EUR_USD')
            timeframe: The timeframe for analysis (e.g., '1H', '4H', 'D')
            market_conditions: Current market conditions

        Returns:
            List[Dict[str, Any]]: Identified risk patterns with adjustment recommendations
        """
        matching_patterns = self.error_pattern_system.find_matching_patterns(
            instrument=instrument, timeframe=timeframe, current_conditions=
            market_conditions)
        risk_patterns = []
        for pattern in matching_patterns:
            if pattern.confidence < self.risk_confidence_threshold:
                continue
            pattern_weight = self.pattern_weights.get(pattern.pattern_type, 1.0
                )
            adjustment_factor = pattern.confidence * pattern_weight
            adjustment_factor = min(adjustment_factor, self.
                max_risk_reduction_pct)
            risk_pattern = {'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type, 'description':
                pattern.description, 'confidence': pattern.confidence,
                'risk_adjustment_factor': adjustment_factor,
                'mitigation_strategies': pattern.mitigation_strategies}
            risk_patterns.append(risk_pattern)
        if risk_patterns:
            logger.info('Identified %d risk patterns for %s on %s (%s)',
                len(risk_patterns), strategy_id, instrument, timeframe)
        return risk_patterns

    async def apply_risk_adjustments(self, strategy_id: str, instrument:
        str, timeframe: str, risk_patterns: List[Dict[str, Any]],
        current_risk_params: Dict[str, Any]) ->Dict[str, Any]:
        """
        Apply risk adjustments based on identified patterns.

        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument
            timeframe: The timeframe for analysis
            risk_patterns: Identified risk patterns
            current_risk_params: Current risk parameters

        Returns:
            Dict[str, Any]: Adjusted risk parameters
        """
        if not risk_patterns:
            return current_risk_params
        max_adjustment = max(p['risk_adjustment_factor'] for p in risk_patterns
            )
        adjusted_params = current_risk_params.copy()
        if 'position_size' in adjusted_params:
            adjusted_params['position_size'] = adjusted_params['position_size'
                ] * (1 - max_adjustment)
        if 'stop_loss_pips' in adjusted_params:
            adjusted_params['stop_loss_pips'] = adjusted_params[
                'stop_loss_pips'] * (1 + max_adjustment / 2)
        if 'max_drawdown_pct' in adjusted_params:
            adjusted_params['max_drawdown_pct'] = adjusted_params[
                'max_drawdown_pct'] * (1 - max_adjustment / 3)
        pattern_ids = [p['pattern_id'] for p in risk_patterns]
        pattern_types = [p['pattern_type'] for p in risk_patterns]
        adjustment_record = {'timestamp': datetime.utcnow(), 'strategy_id':
            strategy_id, 'instrument': instrument, 'timeframe': timeframe,
            'original_params': current_risk_params, 'adjusted_params':
            adjusted_params, 'adjustment_factor': max_adjustment,
            'pattern_ids': pattern_ids, 'pattern_types': pattern_types}
        self.risk_adjustment_history.append(adjustment_record)
        await self._send_adjustment_to_risk_service(strategy_id=strategy_id,
            instrument=instrument, timeframe=timeframe, adjusted_params=
            adjusted_params, pattern_ids=pattern_ids)
        logger.info(
            'Applied risk adjustment for %s on %s (%s): Factor %.2f based on patterns %s'
            , strategy_id, instrument, timeframe, max_adjustment, ', '.join
            (pattern_types))
        return adjusted_params

    @async_with_exception_handling
    async def _send_adjustment_to_risk_service(self, strategy_id: str,
        instrument: str, timeframe: str, adjusted_params: Dict[str, Any],
        pattern_ids: List[str]) ->bool:
        """
        Send risk adjustment to the Risk Management Service.

        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument
            timeframe: The timeframe for analysis
            adjusted_params: Adjusted risk parameters
            pattern_ids: IDs of the patterns triggering adjustment

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            risk_params = RiskParameters(position_size_method=
                adjusted_params.get('position_size_method', 'fixed_percent'
                ), position_size_value=adjusted_params.get('position_size',
                1.0), max_position_size=adjusted_params.get(
                'max_position_size', 5.0), stop_loss_atr_multiplier=
                adjusted_params.get('stop_loss_atr_multiplier', 2.0),
                take_profit_atr_multiplier=adjusted_params.get(
                'take_profit_atr_multiplier', 3.0), max_risk_per_trade_pct=
                adjusted_params.get('max_risk_per_trade_pct', 1.0),
                max_correlation_allowed=adjusted_params.get(
                'max_correlation_allowed', 0.7), max_portfolio_heat=
                adjusted_params.get('max_portfolio_heat', 20.0),
                volatility_scaling_enabled=adjusted_params.get(
                'volatility_scaling_enabled', True), news_sensitivity=
                adjusted_params.get('news_sensitivity', 0.5),
                regime_adaptation_level=adjusted_params.get(
                'regime_adaptation_level', 0.5))
            payload = {'strategy_id': strategy_id, 'instrument': instrument,
                'timeframe': timeframe, 'adjusted_parameters': risk_params.
                to_dict(), 'reason': 'historical_error_pattern',
                'pattern_ids': pattern_ids, 'source':
                'learning_from_mistakes_module'}
            response = await self.risk_client.client.post('/risk/adjustments',
                json=payload)
            logger.info(
                'Successfully sent risk adjustment to Risk Management Service for %s'
                , strategy_id)
            return True
        except Exception as e:
            logger.error('Error communicating with Risk Management Service: %s'
                , str(e))
            return False

    @with_resilience('get_adjustment_history')
    def get_adjustment_history(self, strategy_id: Optional[str]=None,
        instrument: Optional[str]=None, limit: int=100) ->List[Dict[str, Any]]:
        """
        Get history of risk adjustments with optional filtering.

        Args:
            strategy_id: Optional filter by strategy ID
            instrument: Optional filter by instrument
            limit: Maximum number of records to return

        Returns:
            List[Dict[str, Any]]: Filtered adjustment history
        """
        filtered_history = self.risk_adjustment_history
        if strategy_id:
            filtered_history = [record for record in filtered_history if 
                record['strategy_id'] == strategy_id]
        if instrument:
            filtered_history = [record for record in filtered_history if 
                record['instrument'] == instrument]
        filtered_history.sort(key=lambda x: x['timestamp'], reverse=True)
        return filtered_history[:limit]

    @with_analysis_resilience('calculate_adjustment_effectiveness')
    def calculate_adjustment_effectiveness(self, strategy_id: str,
        timeframe_days: int=30) ->Dict[str, Any]:
        """
        Calculate effectiveness metrics for risk adjustments.

        Args:
            strategy_id: Strategy to analyze
            timeframe_days: Analysis period in days

        Returns:
            Dict[str, Any]: Effectiveness metrics
        """
        return {'strategy_id': strategy_id, 'effectiveness_score': 0.0,
            'trades_analyzed': 0, 'status': 'not_implemented'}
