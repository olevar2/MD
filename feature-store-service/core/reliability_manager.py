"""
Reliability Manager for integrating verification and recovery systems.
"""
import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from ..verification.multi_level_verifier import MultiLevelVerifier, VerificationLevel
from ..verification.signal_filter import SignalFilter, SignalType
from ..recovery.integrated_recovery import IntegratedRecoverySystem, RecoveryStrategy
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ReliabilityManager:
    """
    Manages system reliability by coordinating verification and recovery components.
    Acts as a facade for the reliability subsystem.
    """

    def __init__(self, config_path: Optional[str]=None):
        """
        Initialize the reliability manager.
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self.verifier = MultiLevelVerifier()
        self.signal_filter = SignalFilter()
        self.recovery_system = IntegratedRecoverySystem(state_dir=self.
            config['recovery']['storage']['state_dir'])
        logger.info('Reliability Manager initialized')

    def _load_config(self, config_path: Optional[str]=None) ->Dict[str, Any]:
        """Load configuration from file"""
        if not config_path:
            config_path = Path(__file__
                ).parent.parent.parent / 'config' / 'reliability.json'
        with open(config_path, 'r') as f:
            return json.load(f)

    def verify_input_data(self, data: Any, data_type: str) ->bool:
        """
        Verify input data integrity.
        Args:
            data: Data to verify
            data_type: Type of data (e.g., 'price_data', 'indicator_data')
        Returns:
            bool: Whether verification passed
        """
        required_columns = self.config['verification']['input_validation'][
            'required_columns'].get(data_type, [])
        result = self.verifier.verify(VerificationLevel.INPUT, data,
            required_columns=required_columns)
        if not result.is_valid:
            logger.error(f'Input verification failed: {result.message}')
        return result.is_valid

    def verify_risk_compliance(self, risk_metrics: Dict[str, float]) ->bool:
        """
        Verify risk limits compliance.
        Args:
            risk_metrics: Current risk metrics
        Returns:
            bool: Whether verification passed
        """
        result = self.verifier.verify(VerificationLevel.RISK, risk_metrics,
            risk_limits=self.config['verification']['risk_limits'])
        if not result.is_valid:
            logger.warning(
                f'Risk compliance verification failed: {result.message}')
        return result.is_valid

    def verify_decision(self, decision: Any, context: Dict[str, Any],
        historical_decisions: List[Dict[str, Any]]) ->bool:
        """
        Verify decision consistency.
        Args:
            decision: Decision to verify
            context: Current market context
            historical_decisions: Previous decisions for comparison
        Returns:
            bool: Whether verification passed
        """
        result = self.verifier.verify(VerificationLevel.DECISION, {
            'decision': decision, 'context': context}, historical_decisions
            =historical_decisions)
        if not result.is_valid:
            logger.warning(f'Decision verification failed: {result.message}')
        return result.is_valid

    @with_exception_handling
    def filter_signal(self, signal_type: str, value: Any, context: Optional
        [Dict[str, Any]]=None) ->Any:
        """
        Filter a signal value.
        Args:
            signal_type: Type of signal
            value: Signal value
            context: Additional context
        Returns:
            Filtered signal value
        """
        try:
            result = self.signal_filter.filter_signal(signal_type, value,
                context)
            return result.filtered_value
        except Exception as e:
            logger.error(f'Signal filtering failed: {str(e)}')
            return None

    @with_exception_handling
    def handle_error(self, component: str, error: Exception, context:
        Optional[Dict[str, Any]]=None) ->bool:
        """
        Handle system error through recovery.
        Args:
            component: Affected component
            error: The error that occurred
            context: Additional context
        Returns:
            bool: Whether recovery was successful
        """
        try:
            result = self.recovery_system.attempt_recovery(component=
                component, error=error, context=context)
            if not result.success:
                logger.error(
                    f'Recovery failed for {component}: {result.message}')
            return result.success
        except Exception as e:
            logger.error(f'Error during recovery: {str(e)}')
            return False

    def get_system_health(self) ->Dict[str, Any]:
        """Get overall system health status"""
        verification_summary = self.verifier.get_verification_summary()
        filter_summary = self.signal_filter.get_filter_summary()
        recovery_summary = self.recovery_system.get_recovery_summary()
        return {'verification': verification_summary, 'signal_filtering':
            filter_summary, 'recovery': recovery_summary, 'overall_status':
            self._calculate_overall_status(verification_summary,
            filter_summary, recovery_summary), 'timestamp': datetime.utcnow
            ().isoformat()}

    def _calculate_overall_status(self, verification_summary: Dict[str, Any
        ], filter_summary: Dict[str, Any], recovery_summary: Dict[str, Any]
        ) ->str:
        """Calculate overall system status"""
        verification_rate = verification_summary.get('success_rate', 0)
        filter_quality = self._calculate_filter_quality(filter_summary)
        recovery_rate = recovery_summary.get('success_rate', 0)
        overall_score = (verification_rate * 0.4 + filter_quality * 0.3 + 
            recovery_rate * 0.3)
        if overall_score >= 0.95:
            return 'HEALTHY'
        elif overall_score >= 0.8:
            return 'WARNING'
        else:
            return 'CRITICAL'

    def _calculate_filter_quality(self, filter_summary: Dict[str, Any]
        ) ->float:
        """Calculate signal filter quality score"""
        if not filter_summary.get('total_signals', 0):
            return 0.0
        confidence_scores = filter_summary.get('by_confidence', {})
        weighted_sum = 0
        total_signals = filter_summary['total_signals']
        weights = {'very_high': 1.0, 'high': 0.8, 'medium': 0.6, 'low': 0.4,
            'very_low': 0.2, 'invalid': 0.0}
        for confidence, count in confidence_scores.items():
            weight = weights.get(confidence, 0)
            weighted_sum += count * weight
        return weighted_sum / total_signals
