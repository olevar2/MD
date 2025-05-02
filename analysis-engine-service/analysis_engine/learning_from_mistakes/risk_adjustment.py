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
import httpx

from analysis_engine.learning_from_mistakes.error_pattern_recognition import ErrorPatternRecognitionSystem, ErrorPattern
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)

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
    
    def __init__(
        self,
        error_pattern_system: ErrorPatternRecognitionSystem,
        risk_management_service_url: str,
        config: Dict[str, Any] = None
    ):
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
        
        # Risk adjustment settings
        self.risk_confidence_threshold = self.config.get('risk_confidence_threshold', 0.7)
        self.max_risk_reduction_pct = self.config.get('max_risk_reduction_pct', 0.5)
        self.pattern_weights = self.config.get('pattern_weights', {
            'trend_reversal': 1.0,
            'stop_hunt': 1.2,
            'news_impact': 1.5,
            'over_leveraged': 1.3,
            'correlation_breakdown': 1.1,
            'volatility_spike': 1.4,
            'signal_false_positive': 0.9
        })
        
        # Tracking risk adjustments
        self.risk_adjustment_history = []
        
        logger.info("RiskAdjustmentManager initialized")

    async def check_for_risk_patterns(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        market_conditions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
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
        # Find matching patterns from error pattern system
        matching_patterns = self.error_pattern_system.find_matching_patterns(
            instrument=instrument,
            timeframe=timeframe,
            current_conditions=market_conditions
        )
        
        risk_patterns = []
        
        for pattern in matching_patterns:
            # Only consider patterns with sufficient confidence
            if pattern.confidence < self.risk_confidence_threshold:
                continue
                
            # Calculate risk adjustment factor based on pattern type and confidence
            pattern_weight = self.pattern_weights.get(pattern.pattern_type, 1.0)
            adjustment_factor = pattern.confidence * pattern_weight
            
            # Cap the adjustment factor
            adjustment_factor = min(adjustment_factor, self.max_risk_reduction_pct)
            
            risk_pattern = {
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type,
                'description': pattern.description,
                'confidence': pattern.confidence,
                'risk_adjustment_factor': adjustment_factor,
                'mitigation_strategies': pattern.mitigation_strategies
            }
            
            risk_patterns.append(risk_pattern)
            
        if risk_patterns:
            logger.info(
                "Identified %d risk patterns for %s on %s (%s)",
                len(risk_patterns), strategy_id, instrument, timeframe
            )
            
        return risk_patterns
        
    async def apply_risk_adjustments(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        risk_patterns: List[Dict[str, Any]],
        current_risk_params: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            
        # Calculate overall risk adjustment factor - use the strongest adjustment
        max_adjustment = max(p['risk_adjustment_factor'] for p in risk_patterns)
        
        # Create adjusted risk parameters
        adjusted_params = current_risk_params.copy()
        
        # Adjust position size
        if 'position_size' in adjusted_params:
            adjusted_params['position_size'] = adjusted_params['position_size'] * (1 - max_adjustment)
            
        # Adjust stop loss distance
        if 'stop_loss_pips' in adjusted_params:
            # For stop loss, we make it wider (safer) during high risk situations
            adjusted_params['stop_loss_pips'] = adjusted_params['stop_loss_pips'] * (1 + max_adjustment / 2)
            
        # Adjust max drawdown tolerance
        if 'max_drawdown_pct' in adjusted_params:
            adjusted_params['max_drawdown_pct'] = adjusted_params['max_drawdown_pct'] * (1 - max_adjustment / 3)
            
        # Add pattern information to adjustment record
        pattern_ids = [p['pattern_id'] for p in risk_patterns]
        pattern_types = [p['pattern_type'] for p in risk_patterns]
        
        # Record the adjustment for tracking
        adjustment_record = {
            'timestamp': datetime.utcnow(),
            'strategy_id': strategy_id,
            'instrument': instrument,
            'timeframe': timeframe,
            'original_params': current_risk_params,
            'adjusted_params': adjusted_params,
            'adjustment_factor': max_adjustment,
            'pattern_ids': pattern_ids,
            'pattern_types': pattern_types
        }
        
        self.risk_adjustment_history.append(adjustment_record)
        
        # Try to send the adjustment to the Risk Management Service
        await self._send_adjustment_to_risk_service(
            strategy_id=strategy_id,
            instrument=instrument,
            timeframe=timeframe,
            adjusted_params=adjusted_params,
            pattern_ids=pattern_ids
        )
        
        logger.info(
            "Applied risk adjustment for %s on %s (%s): Factor %.2f based on patterns %s",
            strategy_id, instrument, timeframe, max_adjustment, ', '.join(pattern_types)
        )
        
        return adjusted_params
        
    async def _send_adjustment_to_risk_service(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        adjusted_params: Dict[str, Any],
        pattern_ids: List[str]
    ) -> bool:
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
            endpoint = f"{self.risk_management_service_url}/api/v1/risk/adjustments"
            
            payload = {
                'strategy_id': strategy_id,
                'instrument': instrument,
                'timeframe': timeframe,
                'adjusted_parameters': adjusted_params,
                'reason': 'historical_error_pattern',
                'pattern_ids': pattern_ids,
                'source': 'learning_from_mistakes_module'
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(endpoint, json=payload)
                
                if response.status_code == 200:
                    logger.info(
                        "Successfully sent risk adjustment to Risk Management Service for %s", 
                        strategy_id
                    )
                    return True
                else:
                    logger.error(
                        "Failed to send risk adjustment to Risk Management Service: %s - %s",
                        response.status_code, response.text
                    )
                    return False
                    
        except Exception as e:
            logger.error(
                "Error communicating with Risk Management Service: %s", 
                str(e)
            )
            return False
            
    def get_adjustment_history(
        self,
        strategy_id: Optional[str] = None,
        instrument: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
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
            filtered_history = [
                record for record in filtered_history 
                if record['strategy_id'] == strategy_id
            ]
            
        if instrument:
            filtered_history = [
                record for record in filtered_history 
                if record['instrument'] == instrument
            ]
            
        # Sort by timestamp (newest first) and limit
        filtered_history.sort(key=lambda x: x['timestamp'], reverse=True)
        return filtered_history[:limit]
        
    def calculate_adjustment_effectiveness(
        self,
        strategy_id: str,
        timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate effectiveness metrics for risk adjustments.
        
        Args:
            strategy_id: Strategy to analyze
            timeframe_days: Analysis period in days
            
        Returns:
            Dict[str, Any]: Effectiveness metrics
        """
        # This would analyze outcomes of trades where risk adjustments were applied
        # compared to similar situations without adjustments
        
        # For future implementation - requires integration with trade outcomes
        return {
            'strategy_id': strategy_id,
            'effectiveness_score': 0.0,
            'trades_analyzed': 0,
            'status': 'not_implemented'
        }
