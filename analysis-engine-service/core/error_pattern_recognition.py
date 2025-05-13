"""
Error Pattern Recognition System

This module implements pattern recognition capabilities for identifying recurring
error patterns in trading strategy performance.
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

@dataclass
class ErrorPattern:
    """
    Data class representing an identified error pattern.
    """
    pattern_id: str
    pattern_type: str
    description: str
    confidence: float
    frequency: int
    avg_loss: float
    first_occurrence: datetime
    last_occurrence: datetime
    market_conditions: Dict[str, Any]
    root_causes: List[str]
    mitigation_strategies: List[str]


class ErrorPatternRecognitionSystem:
    """
    System for identifying recurring patterns in trading strategy errors and failures.
    
    Key capabilities:
    - Analyze historical trading errors to identify patterns
    - Classify error types and determine root causes
    - Calculate pattern frequency and severity metrics
    - Generate mitigation strategies for recognized patterns
    - Provide pattern matching for current market conditions
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the ErrorPatternRecognitionSystem.
        
        Args:
            config: Configuration parameters for the system
        """
        self.config = config or {}
        self.error_records = []
        self.identified_patterns = []
        self.min_pattern_occurrences = self.config.get(
            'min_pattern_occurrences', 3)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8
            )
        self.lookback_period_days = self.config_manager.get('lookback_period_days', 90)
        self.pattern_categories = {'trend_reversal': self.
            _analyze_trend_reversal_errors, 'stop_hunt': self.
            _analyze_stop_hunt_patterns, 'news_impact': self.
            _analyze_news_impact_errors, 'over_leveraged': self.
            _analyze_over_leveraged_trades, 'correlation_breakdown': self.
            _analyze_correlation_breakdowns, 'volatility_spike': self.
            _analyze_volatility_spike_errors, 'signal_false_positive': self
            ._analyze_signal_false_positives}
        logger.info(
            'ErrorPatternRecognitionSystem initialized with %d pattern categories'
            , len(self.pattern_categories))

    def record_error(self, strategy_id: str, instrument: str, timeframe:
        str, timestamp: datetime, error_type: str, loss_amount: float,
        market_conditions: Dict[str, Any], trade_data: Dict[str, Any],
        signals_used: List[Dict[str, Any]]) ->None:
        """
        Record a trading error for pattern analysis.
        
        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument (e.g., 'EUR_USD')
            timeframe: The timeframe for analysis (e.g., '1H', '4H', 'D')
            timestamp: When the error occurred
            error_type: Classification of the error
            loss_amount: Financial loss from the error
            market_conditions: Market context when error occurred
            trade_data: Details about the trade that failed
            signals_used: Trading signals that led to the decision
        """
        error_record = {'id':
            f"err_{strategy_id}_{instrument}_{timestamp.strftime('%Y%m%d%H%M%S')}"
            , 'strategy_id': strategy_id, 'instrument': instrument,
            'timeframe': timeframe, 'timestamp': timestamp, 'error_type':
            error_type, 'loss_amount': loss_amount, 'market_conditions':
            market_conditions, 'trade_data': trade_data, 'signals_used':
            signals_used}
        self.error_records.append(error_record)
        if len(self.error_records) % self.config.get('analysis_frequency', 10
            ) == 0:
            self.analyze_patterns()
        logger.info('Error recorded for %s on %s (%s): %s with loss %.2f',
            strategy_id, instrument, timeframe, error_type, loss_amount)

    @with_analysis_resilience('analyze_patterns')
    def analyze_patterns(self) ->List[ErrorPattern]:
        """
        Analyze recorded errors to identify recurring patterns.
        
        Returns:
            List[ErrorPattern]: Newly identified patterns
        """
        previous_patterns = self.identified_patterns.copy()
        self.identified_patterns = []
        if len(self.error_records) < self.min_pattern_occurrences:
            logger.info(
                'Insufficient error records for pattern analysis: %d (need %d)'
                , len(self.error_records), self.min_pattern_occurrences)
            return []
        cutoff_date = datetime.utcnow() - timedelta(days=self.
            lookback_period_days)
        recent_errors = [err for err in self.error_records if err[
            'timestamp'] > cutoff_date]
        logger.info(
            'Analyzing patterns from %d recent errors (out of %d total)',
            len(recent_errors), len(self.error_records))
        for pattern_type, analyzer_func in self.pattern_categories.items():
            patterns = analyzer_func(recent_errors)
            if patterns:
                self.identified_patterns.extend(patterns)
                logger.info("Found %d patterns of type '%s'", len(patterns),
                    pattern_type)
        new_patterns = [p for p in self.identified_patterns if not any(p.
            pattern_id == old_p.pattern_id for old_p in previous_patterns)]
        if new_patterns:
            logger.info('Identified %d new error patterns', len(new_patterns))
        return new_patterns

    def find_matching_patterns(self, instrument: str, timeframe: str,
        current_conditions: Dict[str, Any]) ->List[ErrorPattern]:
        """
        Find patterns that match current market conditions.
        
        Args:
            instrument: The trading instrument
            timeframe: The timeframe for analysis
            current_conditions: Current market conditions to match against
            
        Returns:
            List[ErrorPattern]: Matching error patterns
        """
        matching_patterns = []
        for pattern in self.identified_patterns:
            similarity_score = self._calculate_condition_similarity(
                current_conditions, pattern.market_conditions)
            if (similarity_score >= self.similarity_threshold and 
                instrument in pattern.market_conditions.get('instrument',
                '') and timeframe in pattern.market_conditions.get(
                'timeframe', '')):
                matching_patterns.append(pattern)
        matching_patterns.sort(key=lambda p: (p.confidence, p.frequency),
            reverse=True)
        return matching_patterns

    @with_resilience('get_mitigation_strategies')
    def get_mitigation_strategies(self, pattern_ids: Optional[List[str]]=None
        ) ->Dict[str, List[str]]:
        """
        Get mitigation strategies for specific patterns or all patterns.
        
        Args:
            pattern_ids: Optional list of specific pattern IDs to get strategies for
            
        Returns:
            Dict[str, List[str]]: Mapping of pattern IDs to mitigation strategies
        """
        strategies = {}
        for pattern in self.identified_patterns:
            if pattern_ids is None or pattern.pattern_id in pattern_ids:
                strategies[pattern.pattern_id] = pattern.mitigation_strategies
        return strategies

    @with_resilience('get_identified_patterns')
    def get_identified_patterns(self, pattern_type: Optional[str]=None,
        min_confidence: float=0.0, min_frequency: int=0) ->List[ErrorPattern]:
        """
        Get identified error patterns with optional filtering.
        
        Args:
            pattern_type: Optional filter by pattern type
            min_confidence: Minimum confidence threshold
            min_frequency: Minimum frequency threshold
            
        Returns:
            List[ErrorPattern]: Filtered error patterns
        """
        filtered_patterns = []
        for pattern in self.identified_patterns:
            if ((pattern_type is None or pattern.pattern_type ==
                pattern_type) and pattern.confidence >= min_confidence and 
                pattern.frequency >= min_frequency):
                filtered_patterns.append(pattern)
        return filtered_patterns

    def _analyze_trend_reversal_errors(self, errors: List[Dict[str, Any]]
        ) ->List[ErrorPattern]:
        """
        Identify patterns related to missed trend reversals.
        
        Args:
            errors: List of error records to analyze
            
        Returns:
            List[ErrorPattern]: Identified trend reversal error patterns
        """
        patterns = []
        errors_by_instrument = {}
        for error in errors:
            if error['error_type'] == 'trend_reversal' or 'reversal' in error[
                'error_type'].lower():
                instr = error['instrument']
                if instr not in errors_by_instrument:
                    errors_by_instrument[instr] = []
                errors_by_instrument[instr].append(error)
        for instrument, instr_errors in errors_by_instrument.items():
            if len(instr_errors) >= self.min_pattern_occurrences:
                common_conditions = self._extract_common_conditions(
                    instr_errors)
                if common_conditions:
                    first_occurrence = min(err['timestamp'] for err in
                        instr_errors)
                    last_occurrence = max(err['timestamp'] for err in
                        instr_errors)
                    avg_loss = sum(err['loss_amount'] for err in instr_errors
                        ) / len(instr_errors)
                    confidence = self._calculate_pattern_confidence(
                        instr_errors, common_conditions)
                    pattern = ErrorPattern(pattern_id=
                        f"trend_reversal_{instrument}_{first_occurrence.strftime('%Y%m%d')}"
                        , pattern_type='trend_reversal', description=
                        f'Trend reversal errors on {instrument} related to {self._summarize_conditions(common_conditions)}'
                        , confidence=confidence, frequency=len(instr_errors
                        ), avg_loss=avg_loss, first_occurrence=
                        first_occurrence, last_occurrence=last_occurrence,
                        market_conditions=common_conditions, root_causes=
                        self._identify_trend_reversal_causes(instr_errors),
                        mitigation_strategies=self.
                        _generate_trend_reversal_strategies(instr_errors,
                        common_conditions))
                    patterns.append(pattern)
        return patterns

    def _analyze_stop_hunt_patterns(self, errors: List[Dict[str, Any]]) ->List[
        ErrorPattern]:
        """
        Identify patterns related to stop hunting market behavior.
        
        Args:
            errors: List of error records to analyze
            
        Returns:
            List[ErrorPattern]: Identified stop hunt error patterns
        """
        patterns = []
        stop_hunt_errors = [err for err in errors if 'stop_hunt' in err[
            'error_type'].lower() or err['trade_data'].get('stopped_out', 
            False) and err['trade_data'].get('price_return_after_stop', 0) > 0]
        if len(stop_hunt_errors) < self.min_pattern_occurrences:
            return patterns
        errors_by_hour = {}
        for error in stop_hunt_errors:
            hour = error['timestamp'].hour
            if hour not in errors_by_hour:
                errors_by_hour[hour] = []
            errors_by_hour[hour].append(error)
        for hour, hour_errors in errors_by_hour.items():
            if len(hour_errors) >= self.min_pattern_occurrences:
                errors_by_instrument = {}
                for err in hour_errors:
                    instr = err['instrument']
                    if instr not in errors_by_instrument:
                        errors_by_instrument[instr] = []
                    errors_by_instrument[instr].append(err)
                for instr, instr_errors in errors_by_instrument.items():
                    if len(instr_errors) >= self.min_pattern_occurrences:
                        first_occurrence = min(err['timestamp'] for err in
                            instr_errors)
                        last_occurrence = max(err['timestamp'] for err in
                            instr_errors)
                        avg_loss = sum(err['loss_amount'] for err in
                            instr_errors) / len(instr_errors)
                        common_conditions = self._extract_common_conditions(
                            instr_errors)
                        confidence = self._calculate_pattern_confidence(
                            instr_errors, common_conditions)
                        pattern = ErrorPattern(pattern_id=
                            f"stop_hunt_{instr}_{hour:02d}h_{first_occurrence.strftime('%Y%m%d')}"
                            , pattern_type='stop_hunt', description=
                            f'Stop hunting pattern on {instr} around {hour:02d}:00 UTC'
                            , confidence=confidence, frequency=len(
                            instr_errors), avg_loss=avg_loss,
                            first_occurrence=first_occurrence,
                            last_occurrence=last_occurrence,
                            market_conditions=common_conditions,
                            root_causes=self._identify_stop_hunt_causes(
                            instr_errors), mitigation_strategies=self.
                            _generate_stop_hunt_strategies(instr_errors, hour))
                        patterns.append(pattern)
        return patterns

    def _analyze_news_impact_errors(self, errors: List[Dict[str, Any]]) ->List[
        ErrorPattern]:
        """
        Identify patterns related to news events impacting trades.
        
        Args:
            errors: List of error records to analyze
            
        Returns:
            List[ErrorPattern]: Identified news impact error patterns
        """
        return []

    def _analyze_over_leveraged_trades(self, errors: List[Dict[str, Any]]
        ) ->List[ErrorPattern]:
        """
        Identify patterns related to excessive position sizing.
        
        Args:
            errors: List of error records to analyze
            
        Returns:
            List[ErrorPattern]: Identified over-leveraging patterns
        """
        return []

    def _analyze_correlation_breakdowns(self, errors: List[Dict[str, Any]]
        ) ->List[ErrorPattern]:
        """
        Identify patterns where expected correlations between instruments broke down.
        
        Args:
            errors: List of error records to analyze
            
        Returns:
            List[ErrorPattern]: Identified correlation breakdown patterns
        """
        return []

    def _analyze_volatility_spike_errors(self, errors: List[Dict[str, Any]]
        ) ->List[ErrorPattern]:
        """
        Identify patterns related to sudden volatility spikes causing losses.
        
        Args:
            errors: List of error records to analyze
            
        Returns:
            List[ErrorPattern]: Identified volatility spike patterns
        """
        patterns = []
        volatility_errors = [err for err in errors if 'volatility' in err[
            'error_type'].lower() or err['market_conditions'].get(
            'volatility_state', '') == 'spike']
        if len(volatility_errors) < self.min_pattern_occurrences:
            return patterns
        errors_by_instrument = {}
        for error in volatility_errors:
            instr = error['instrument']
            if instr not in errors_by_instrument:
                errors_by_instrument[instr] = []
            errors_by_instrument[instr].append(error)
        for instr, instr_errors in errors_by_instrument.items():
            if len(instr_errors) >= self.min_pattern_occurrences:
                common_conditions = self._extract_common_conditions(
                    instr_errors)
                if common_conditions:
                    first_occurrence = min(err['timestamp'] for err in
                        instr_errors)
                    last_occurrence = max(err['timestamp'] for err in
                        instr_errors)
                    avg_loss = sum(err['loss_amount'] for err in instr_errors
                        ) / len(instr_errors)
                    confidence = self._calculate_pattern_confidence(
                        instr_errors, common_conditions)
                    pattern = ErrorPattern(pattern_id=
                        f"volatility_spike_{instr}_{first_occurrence.strftime('%Y%m%d')}"
                        , pattern_type='volatility_spike', description=
                        f'Volatility spike pattern on {instr} preceded by {self._summarize_conditions(common_conditions)}'
                        , confidence=confidence, frequency=len(instr_errors
                        ), avg_loss=avg_loss, first_occurrence=
                        first_occurrence, last_occurrence=last_occurrence,
                        market_conditions=common_conditions, root_causes=
                        self._identify_volatility_causes(instr_errors),
                        mitigation_strategies=self.
                        _generate_volatility_strategies(instr_errors,
                        common_conditions))
                    patterns.append(pattern)
        return patterns

    def _analyze_signal_false_positives(self, errors: List[Dict[str, Any]]
        ) ->List[ErrorPattern]:
        """
        Identify patterns where specific signals consistently generate false positives.
        
        Args:
            errors: List of error records to analyze
            
        Returns:
            List[ErrorPattern]: Identified false positive signal patterns
        """
        patterns = []
        signal_counts = {}
        for error in errors:
            for signal in error.get('signals_used', []):
                sig_id = signal.get('signal_id', 'unknown')
                if sig_id not in signal_counts:
                    signal_counts[sig_id] = {'count': 0, 'errors': []}
                signal_counts[sig_id]['count'] += 1
                signal_counts[sig_id]['errors'].append(error)
        for sig_id, data in signal_counts.items():
            if data['count'] >= self.min_pattern_occurrences:
                sig_errors = data['errors']
                errors_by_instrument = {}
                for err in sig_errors:
                    instr = err['instrument']
                    if instr not in errors_by_instrument:
                        errors_by_instrument[instr] = []
                    errors_by_instrument[instr].append(err)
                for instr, instr_errors in errors_by_instrument.items():
                    if len(instr_errors) >= self.min_pattern_occurrences:
                        common_conditions = self._extract_common_conditions(
                            instr_errors)
                        if common_conditions:
                            first_occurrence = min(err['timestamp'] for err in
                                instr_errors)
                            last_occurrence = max(err['timestamp'] for err in
                                instr_errors)
                            avg_loss = sum(err['loss_amount'] for err in
                                instr_errors) / len(instr_errors)
                            confidence = self._calculate_pattern_confidence(
                                instr_errors, common_conditions)
                            pattern = ErrorPattern(pattern_id=
                                f"false_positive_{sig_id}_{instr}_{first_occurrence.strftime('%Y%m%d')}"
                                , pattern_type='signal_false_positive',
                                description=
                                f'False positive pattern for signal {sig_id} on {instr} during {self._summarize_conditions(common_conditions)}'
                                , confidence=confidence, frequency=len(
                                instr_errors), avg_loss=avg_loss,
                                first_occurrence=first_occurrence,
                                last_occurrence=last_occurrence,
                                market_conditions=common_conditions,
                                root_causes=self.
                                _identify_false_positive_causes(
                                instr_errors, sig_id),
                                mitigation_strategies=self.
                                _generate_false_positive_strategies(
                                instr_errors, sig_id))
                            patterns.append(pattern)
        return patterns

    def _extract_common_conditions(self, errors: List[Dict[str, Any]]) ->Dict[
        str, Any]:
        """
        Extract common market conditions across a set of errors.
        
        Args:
            errors: List of error records to analyze
            
        Returns:
            Dict[str, Any]: Common market conditions
        """
        common_conditions = {}
        if not errors:
            return common_conditions
        first_error = errors[0]
        condition_keys = first_error.get('market_conditions', {}).keys()
        for key in condition_keys:
            values = [error.get('market_conditions', {}).get(key) for error in
                errors if key in error.get('market_conditions', {})]
            if not values:
                continue
            if isinstance(values[0], str):
                value_counts = {}
                for value in values:
                    if value not in value_counts:
                        value_counts[value] = 0
                    value_counts[value] += 1
                most_common_value = max(value_counts.items(), key=lambda x:
                    x[1])
                most_common_count = most_common_value[1]
                if most_common_count / len(errors) >= 0.6:
                    common_conditions[key] = most_common_value[0]
            elif isinstance(values[0], (int, float)):
                min_val = min(values)
                max_val = max(values)
                mean_val = sum(values) / len(values)
                if max_val == 0:
                    if min_val == 0:
                        common_conditions[key] = 0
                else:
                    range_ratio = (max_val - min_val) / abs(mean_val
                        ) if mean_val != 0 else float('inf')
                    if range_ratio < 0.5:
                        common_conditions[key] = mean_val
        common_conditions['instrument'] = errors[0]['instrument']
        common_conditions['timeframe'] = errors[0]['timeframe']
        return common_conditions

    def _calculate_condition_similarity(self, conditions1: Dict[str, Any],
        conditions2: Dict[str, Any]) ->float:
        """
        Calculate similarity score between two sets of market conditions.
        
        Args:
            conditions1: First set of conditions
            conditions2: Second set of conditions
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        common_keys = set(conditions1.keys()) & set(conditions2.keys())
        if not common_keys:
            return 0.0
        similarities = []
        for key in common_keys:
            val1 = conditions1[key]
            val2 = conditions2[key]
            if isinstance(val1, str) and isinstance(val2, str):
                similarities.append(1.0 if val1 == val2 else 0.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int,
                float)):
                if val1 == val2:
                    similarities.append(1.0)
                else:
                    max_val = max(abs(val1), abs(val2))
                    if max_val == 0:
                        similarities.append(1.0)
                    else:
                        diff_ratio = abs(val1 - val2) / max_val
                        similarities.append(max(0.0, 1.0 - diff_ratio))
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_pattern_confidence(self, errors: List[Dict[str, Any]],
        common_conditions: Dict[str, Any]) ->float:
        """
        Calculate confidence level for a pattern.
        
        Args:
            errors: List of error records in the pattern
            common_conditions: Extracted common conditions
            
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        occurrence_factor = min(1.0, len(errors) / (2 * self.
            min_pattern_occurrences))
        consistency_scores = []
        for error in errors:
            similarity = self._calculate_condition_similarity(error.get(
                'market_conditions', {}), common_conditions)
            consistency_scores.append(similarity)
        condition_factor = sum(consistency_scores) / len(consistency_scores
            ) if consistency_scores else 0.0
        current_time = datetime.utcnow()
        recency_scores = []
        for error in errors:
            days_ago = (current_time - error['timestamp']).days
            max_days = self.lookback_period_days
            recency = max(0.0, 1.0 - days_ago / max_days)
            recency_scores.append(recency)
        recency_factor = sum(recency_scores) / len(recency_scores
            ) if recency_scores else 0.0
        confidence = (0.4 * occurrence_factor + 0.4 * condition_factor + 
            0.2 * recency_factor)
        return confidence

    def _summarize_conditions(self, conditions: Dict[str, Any]) ->str:
        """
        Create a human-readable summary of market conditions.
        
        Args:
            conditions: Market conditions to summarize
            
        Returns:
            str: Summary description
        """
        summary_parts = []
        if 'market_regime' in conditions:
            summary_parts.append(f"{conditions['market_regime']} market")
        if 'volatility_state' in conditions:
            summary_parts.append(f"{conditions['volatility_state']} volatility"
                )
        if 'trend_strength' in conditions:
            strength = conditions['trend_strength']
            if isinstance(strength, (int, float)):
                if strength > 0.7:
                    summary_parts.append('strong trend')
                elif strength > 0.3:
                    summary_parts.append('moderate trend')
                else:
                    summary_parts.append('weak trend')
            else:
                summary_parts.append(f'{strength} trend')
        if 'liquidity' in conditions:
            liquidity = conditions['liquidity']
            if isinstance(liquidity, (int, float)):
                if liquidity < 0.3:
                    summary_parts.append('low liquidity')
                elif liquidity > 0.7:
                    summary_parts.append('high liquidity')
            else:
                summary_parts.append(f'{liquidity} liquidity')
        if 'hour_of_day' in conditions:
            hour = conditions['hour_of_day']
            summary_parts.append(f'around {hour:02d}:00 UTC')
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
            'Saturday', 'Sunday']
        if 'day_of_week' in conditions:
            day_idx = conditions['day_of_week']
            if 0 <= day_idx < len(days):
                summary_parts.append(f'on {days[day_idx]}')
        return ', '.join(summary_parts
            ) if summary_parts else 'various market conditions'

    def _identify_trend_reversal_causes(self, errors: List[Dict[str, Any]]
        ) ->List[str]:
        """Identify root causes for trend reversal errors."""
        causes = []
        if any('late_recognition' in str(err.get('error_type', '')) for err in
            errors):
            causes.append('Late recognition of reversal signals')
        if any('false_breakout' in str(err.get('error_type', '')) for err in
            errors):
            causes.append('False breakouts preceding reversals')
        if not causes:
            causes.append('Failure to recognize reversal patterns')
            causes.append('Over-reliance on trend-following indicators')
        return causes

    def _identify_stop_hunt_causes(self, errors: List[Dict[str, Any]]) ->List[
        str]:
        """Identify root causes for stop hunt patterns."""
        causes = []
        causes.append('Stop levels placed at predictable technical levels')
        causes.append(
            'Trading during low liquidity periods prone to stop hunting')
        return causes

    def _identify_volatility_causes(self, errors: List[Dict[str, Any]]) ->List[
        str]:
        """Identify root causes for volatility spike errors."""
        causes = []
        causes.append('Position sizing not adjusted for volatility conditions')
        causes.append('Stop levels too tight for prevailing volatility')
        return causes

    def _identify_false_positive_causes(self, errors: List[Dict[str, Any]],
        signal_id: str) ->List[str]:
        """Identify root causes for false positive signals."""
        causes = []
        causes.append(
            f'Signal {signal_id} not reliable in current market context')
        causes.append('Insufficient confirmation from complementary signals')
        return causes

    def _generate_trend_reversal_strategies(self, errors: List[Dict[str,
        Any]], conditions: Dict[str, Any]) ->List[str]:
        """Generate mitigation strategies for trend reversal errors."""
        strategies = []
        strategies.append(
            'Implement confirmation requirements before trading against established trends'
            )
        strategies.append(
            'Add momentum divergence detection to identify potential reversals earlier'
            )
        strategies.append(
            'Reduce position size when trading against the primary trend')
        return strategies

    def _generate_stop_hunt_strategies(self, errors: List[Dict[str, Any]],
        hour: int) ->List[str]:
        """Generate mitigation strategies for stop hunt patterns."""
        strategies = []
        strategies.append(
            f'Avoid trading during the {hour:02d}:00 UTC hour which shows frequent stop hunting activity'
            )
        strategies.append(
            'Place stops at less obvious levels beyond technical support/resistance'
            )
        strategies.append(
            'Consider wider stops with proportionally smaller position sizes')
        return strategies

    def _generate_volatility_strategies(self, errors: List[Dict[str, Any]],
        conditions: Dict[str, Any]) ->List[str]:
        """Generate mitigation strategies for volatility spike errors."""
        strategies = []
        strategies.append(
            'Implement dynamic position sizing based on current volatility metrics'
            )
        strategies.append(
            'Adjust stop distances proportionally to ATR or other volatility measures'
            )
        strategies.append(
            'Reduce overall exposure during high-volatility market conditions')
        return strategies

    def _generate_false_positive_strategies(self, errors: List[Dict[str,
        Any]], signal_id: str) ->List[str]:
        """Generate mitigation strategies for false positive signals."""
        strategies = []
        strategies.append(
            f'Require additional confirmation when signal {signal_id} is triggered'
            )
        strategies.append('Downweight this signal in strategy decision making')
        strategies.append(
            'Analyze specific conditions where this signal fails and add filters'
            )
        return strategies
