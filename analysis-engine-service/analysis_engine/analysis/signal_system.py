"""
Central Signal System Module

This module provides a comprehensive system for managing indicator signals with:
- Signal aggregation with quality rating
- Conflict detection and resolution between signals
- Concordance analysis to determine agreement between indicators
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum
import logging
from dataclasses import dataclass
from datetime import datetime
from analysis_engine.config import settings
from analysis_engine.events.publisher import EventPublisher
from analysis_engine.events.schemas import SignalGeneratedEvent, SignalGeneratedPayload
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    event_publisher = EventPublisher()
except Exception as e:
    logger.error(f'Failed to initialize EventPublisher in SignalSystem: {e}',
        exc_info=True)
    event_publisher = None
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class SignalType(Enum):
    """Types of signals that can be generated."""
    BUY = 1
    SELL = -1
    NEUTRAL = 0
    STRONG_BUY = 2
    STRONG_SELL = -2


@dataclass
class Signal:
    """Represents a trading signal with metadata."""
    timestamp: datetime
    indicator_name: str
    signal_type: SignalType
    strength: float
    price: float
    metadata: Dict = None

    def __post_init__(self):
        """Validate signal after initialization."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError('Signal strength must be between 0.0 and 1.0')
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_buy(self) ->bool:
        """Check if this is a buy signal."""
        return self.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]

    @property
    def is_sell(self) ->bool:
        """Check if this is a sell signal."""
        return self.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]

    @property
    def is_neutral(self) ->bool:
        """Check if this is a neutral signal."""
        return self.signal_type == SignalType.NEUTRAL

    def __str__(self) ->str:
        return (
            f'{self.indicator_name} {self.signal_type.name} ({self.strength:.2f}) @ {self.timestamp}'
            )


@dataclass
class AggregatedSignal:
    """Represents a signal aggregated from multiple indicators."""
    timestamp: datetime
    signal_type: SignalType
    strength: float
    confidence: float
    price: float
    contributing_signals: List[Signal]

    @property
    def num_contributors(self) ->int:
        """Get the number of contributing signals."""
        return len(self.contributing_signals)

    @property
    def majority_vote(self) ->SignalType:
        """Get the majority vote from contributing signals."""
        buy_count = sum(1 for s in self.contributing_signals if s.is_buy)
        sell_count = sum(1 for s in self.contributing_signals if s.is_sell)
        if buy_count > sell_count:
            return SignalType.BUY
        elif sell_count > buy_count:
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL

    @property
    def weighted_strength(self) ->float:
        """Get strength weighted by indicator confidence."""
        if not self.contributing_signals:
            return 0.0
        total_strength = sum(s.strength for s in self.contributing_signals)
        return total_strength / len(self.contributing_signals)

    def __str__(self) ->str:
    """
      str  .
    
    Returns:
        str: Description of return value
    
    """

        return (
            f'{self.signal_type.name} ({self.strength:.2f}) @ {self.timestamp} [Confidence: {self.confidence:.2f}, Contributors: {self.num_contributors}]'
            )


class SignalConflict:
    """Represents a conflict between signals."""

    def __init__(self, timestamp: datetime, conflicting_signals: List[Signal]):
    """
      init  .
    
    Args:
        timestamp: Description of timestamp
        conflicting_signals: Description of conflicting_signals
    
    """

        self.timestamp = timestamp
        self.conflicting_signals = conflicting_signals
        self.resolution_applied = None
        self.resolved_signal = None

    @property
    def conflict_severity(self) ->float:
        """Calculate the severity of the conflict based on signal strengths."""
        if len(self.conflicting_signals) <= 1:
            return 0.0
        buy_strength = sum(s.strength for s in self.conflicting_signals if
            s.is_buy)
        sell_strength = sum(s.strength for s in self.conflicting_signals if
            s.is_sell)
        return abs(buy_strength - sell_strength) / (buy_strength +
            sell_strength)

    def __str__(self) ->str:
    """
      str  .
    
    Returns:
        str: Description of return value
    
    """

        return (
            f'Conflict @ {self.timestamp} [Severity: {self.conflict_severity:.2f}, Signals: {len(self.conflicting_signals)}]'
            )


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving signal conflicts."""
    MAJORITY_VOTE = 'majority_vote'
    WEIGHTED_MAJORITY = 'weighted_majority'
    STRONGEST_SIGNAL = 'strongest_signal'
    TIME_PRIORITY = 'time_priority'
    MOST_RELIABLE = 'most_reliable'
    CUSTOM = 'custom'


class SignalConcordance:
    """Analyzes agreement between different indicators."""

    def __init__(self, indicators: List[str]):
    """
      init  .
    
    Args:
        indicators: Description of indicators
    
    """

        self.indicators = indicators
        self.agreement_matrix = pd.DataFrame(np.identity(len(indicators)),
            index=indicators, columns=indicators)
        self.samples_count = pd.DataFrame(np.zeros((len(indicators), len(
            indicators))), index=indicators, columns=indicators)

    def update(self, signals_by_indicator: Dict[str, List[Signal]]) ->None:
        """
        Update concordance based on new signals.
        
        Args:
            signals_by_indicator: Dictionary mapping indicator names to signal lists
        """
        signals_by_time = {}
        for indicator, signals in signals_by_indicator.items():
            for signal in signals:
                if signal.timestamp not in signals_by_time:
                    signals_by_time[signal.timestamp] = {}
                signals_by_time[signal.timestamp][indicator] = signal
        for timestamp, indicators_signals in signals_by_time.items():
            for ind1 in indicators_signals:
                for ind2 in indicators_signals:
                    if ind1 == ind2:
                        continue
                    signal1 = indicators_signals[ind1]
                    signal2 = indicators_signals[ind2]
                    self.samples_count.loc[ind1, ind2] += 1
                    agreement = self._calculate_signal_agreement(signal1,
                        signal2)
                    alpha = 0.1
                    old_agreement = self.agreement_matrix.loc[ind1, ind2]
                    new_agreement = (1 - alpha
                        ) * old_agreement + alpha * agreement
                    self.agreement_matrix.loc[ind1, ind2] = new_agreement

    def _calculate_signal_agreement(self, signal1: Signal, signal2: Signal
        ) ->float:
        """Calculate agreement between two signals from -1.0 to 1.0."""
        if signal1.signal_type == signal2.signal_type:
            return 1.0
        if (signal1.is_buy and signal2.is_sell or signal1.is_sell and
            signal2.is_buy):
            return -1.0
        if signal1.is_neutral or signal2.is_neutral:
            return 0.0
        return 0.0

    @with_resilience('get_most_concordant_indicators')
    def get_most_concordant_indicators(self, n: int=3) ->List[Tuple[str,
        str, float]]:
        """Get top N most agreeing indicator pairs."""
        flat_matrix = []
        for i, ind1 in enumerate(self.indicators):
            for j, ind2 in enumerate(self.indicators):
                if i < j:
                    agreement = self.agreement_matrix.loc[ind1, ind2]
                    samples = self.samples_count.loc[ind1, ind2]
                    if samples > 5:
                        flat_matrix.append((ind1, ind2, agreement))
        flat_matrix.sort(key=lambda x: x[2], reverse=True)
        return flat_matrix[:n]

    @with_resilience('get_most_discordant_indicators')
    def get_most_discordant_indicators(self, n: int=3) ->List[Tuple[str,
        str, float]]:
        """Get top N most disagreeing indicator pairs."""
        flat_matrix = []
        for i, ind1 in enumerate(self.indicators):
            for j, ind2 in enumerate(self.indicators):
                if i < j:
                    agreement = self.agreement_matrix.loc[ind1, ind2]
                    samples = self.samples_count.loc[ind1, ind2]
                    if samples > 5:
                        flat_matrix.append((ind1, ind2, agreement))
        flat_matrix.sort(key=lambda x: x[2])
        return flat_matrix[:n]

    @with_resilience('get_agreement_matrix')
    def get_agreement_matrix(self) ->pd.DataFrame:
        """Get the current agreement matrix."""
        return self.agreement_matrix.copy()


class SignalSystem:
    """Central system for managing indicator signals."""

    def __init__(self):
        """Initialize the signal system."""
        self.signals: Dict[str, List[Signal]] = {}
        self.aggregated_signals: List[AggregatedSignal] = []
        self.conflicts: List[SignalConflict] = []
        self.concordance = None
        self.indicator_reliability: Dict[str, float] = {}
        self._conflict_resolution_strategy = (ConflictResolutionStrategy.
            WEIGHTED_MAJORITY)
        self._custom_resolution_func = None

    @with_exception_handling
    def add_signal(self, signal: Signal) ->None:
        """
        Add a new signal to the system.

        Args:
            signal: The signal to add
        """
        if signal.indicator_name not in self.signals:
            self.signals[signal.indicator_name] = []
        self.signals[signal.indicator_name].append(signal)
        logger.debug(f'Added signal: {signal}')
        if event_publisher:
            try:
                symbol = signal.metadata.get('symbol', 'UNKNOWN')
                timeframe = signal.metadata.get('timeframe', 'UNKNOWN')
                payload = SignalGeneratedPayload(signal_id=
                    f'{signal.indicator_name}_{signal.timestamp.isoformat()}',
                    symbol=symbol, timeframe=timeframe, signal_type=signal.
                    signal_type.name, strength=signal.strength, price=
                    signal.price, indicator_name=signal.indicator_name,
                    timestamp=signal.timestamp, metadata=signal.metadata)
                event = SignalGeneratedEvent(payload=payload)
                event_publisher.publish(topic=settings.KAFKA_SIGNAL_TOPIC,
                    event=event)
            except Exception as pub_exc:
                logger.error(
                    f'Failed to publish SignalGeneratedEvent for {signal}: {pub_exc}'
                    , exc_info=True)

    def add_signals(self, signals: List[Signal]) ->None:
        """
        Add multiple signals to the system.
        
        Args:
            signals: List of signals to add
        """
        for signal in signals:
            self.add_signal(signal)

    def set_indicator_reliability(self, indicator_name: str, reliability: float
        ) ->None:
        """
        Set historical reliability score for an indicator.
        
        Args:
            indicator_name: Name of the indicator
            reliability: Reliability score (0.0 to 1.0)
        """
        if not 0.0 <= reliability <= 1.0:
            raise ValueError('Reliability must be between 0.0 and 1.0')
        self.indicator_reliability[indicator_name] = reliability
        logger.debug(f'Set reliability for {indicator_name}: {reliability:.2f}'
            )

    def set_conflict_resolution_strategy(self, strategy:
        ConflictResolutionStrategy, custom_func: Callable=None) ->None:
        """
        Set the strategy for resolving signal conflicts.
        
        Args:
            strategy: The conflict resolution strategy to use
            custom_func: Custom resolution function (required if strategy is CUSTOM)
        """
        self._conflict_resolution_strategy = strategy
        if strategy == ConflictResolutionStrategy.CUSTOM:
            if custom_func is None:
                raise ValueError(
                    'Custom resolution function must be provided with CUSTOM strategy'
                    )
            self._custom_resolution_func = custom_func

    def _detect_conflicts(self, timestamp: datetime, signals: List[Signal],
        threshold: float=0.3) ->Optional[SignalConflict]:
        """
        Detect conflicts among signals at a specific timestamp.
        
        Args:
            timestamp: The timestamp to check for conflicts
            signals: List of signals at this timestamp
            threshold: Threshold for considering signals in conflict
            
        Returns:
            SignalConflict object if conflict is detected, None otherwise
        """
        if len(signals) <= 1:
            return None
        buy_signals = [s for s in signals if s.is_buy]
        sell_signals = [s for s in signals if s.is_sell]
        if buy_signals and sell_signals:
            conflicting = buy_signals + sell_signals
            return SignalConflict(timestamp, conflicting)
        return None

    @with_exception_handling
    def _resolve_conflict(self, conflict: SignalConflict) ->Optional[Signal]:
        """
        Resolve a signal conflict using the current strategy.
        
        Args:
            conflict: The conflict to resolve
            
        Returns:
            Resolved signal, or None if resolution fails
        """
        if not conflict.conflicting_signals:
            return None
        strategy = self._conflict_resolution_strategy
        signals = conflict.conflicting_signals
        timestamp = conflict.timestamp
        strongest = max(signals, key=lambda s: s.strength)
        result = None
        if strategy == ConflictResolutionStrategy.STRONGEST_SIGNAL:
            result = strongest
            conflict.resolution_applied = 'strongest_signal'
        elif strategy == ConflictResolutionStrategy.MAJORITY_VOTE:
            buy_count = sum(1 for s in signals if s.is_buy)
            sell_count = sum(1 for s in signals if s.is_sell)
            if buy_count > sell_count:
                result = Signal(timestamp=timestamp, indicator_name=
                    'MAJORITY_VOTE', signal_type=SignalType.BUY, strength=
                    sum(s.strength for s in signals if s.is_buy) /
                    buy_count, price=signals[0].price, metadata={
                    'contributing_signals': [s.indicator_name for s in
                    signals if s.is_buy]})
            elif sell_count > buy_count:
                result = Signal(timestamp=timestamp, indicator_name=
                    'MAJORITY_VOTE', signal_type=SignalType.SELL, strength=
                    sum(s.strength for s in signals if s.is_sell) /
                    sell_count, price=signals[0].price, metadata={
                    'contributing_signals': [s.indicator_name for s in
                    signals if s.is_sell]})
            else:
                result = strongest
            conflict.resolution_applied = 'majority_vote'
        elif strategy == ConflictResolutionStrategy.WEIGHTED_MAJORITY:
            buy_weight = 0.0
            sell_weight = 0.0
            for s in signals:
                reliability = self.indicator_reliability.get(s.
                    indicator_name, 0.5)
                weight = s.strength * reliability
                if s.is_buy:
                    buy_weight += weight
                elif s.is_sell:
                    sell_weight += weight
            if buy_weight > sell_weight:
                result = Signal(timestamp=timestamp, indicator_name=
                    'WEIGHTED_MAJORITY', signal_type=SignalType.BUY,
                    strength=buy_weight / (buy_weight + sell_weight), price
                    =signals[0].price, metadata={'buy_weight': buy_weight,
                    'sell_weight': sell_weight})
            elif sell_weight > buy_weight:
                result = Signal(timestamp=timestamp, indicator_name=
                    'WEIGHTED_MAJORITY', signal_type=SignalType.SELL,
                    strength=sell_weight / (buy_weight + sell_weight),
                    price=signals[0].price, metadata={'buy_weight':
                    buy_weight, 'sell_weight': sell_weight})
            else:
                result = strongest
            conflict.resolution_applied = 'weighted_majority'
        elif strategy == ConflictResolutionStrategy.TIME_PRIORITY:
            result = max(signals, key=lambda s: s.timestamp)
            conflict.resolution_applied = 'time_priority'
        elif strategy == ConflictResolutionStrategy.MOST_RELIABLE:
            if self.indicator_reliability:
                result = max(signals, key=lambda s: self.
                    indicator_reliability.get(s.indicator_name, 0.0))
            else:
                result = strongest
            conflict.resolution_applied = 'most_reliable'
        elif strategy == ConflictResolutionStrategy.CUSTOM and self._custom_resolution_func:
            try:
                result = self._custom_resolution_func(conflict)
                conflict.resolution_applied = 'custom'
            except Exception as e:
                logger.error(f'Error in custom conflict resolution: {str(e)}')
                result = strongest
                conflict.resolution_applied = 'custom_failed'
        conflict.resolved_signal = result
        return result

    def _aggregate_signals(self, timestamp: datetime, signals: List[Signal]
        ) ->AggregatedSignal:
        """
        Aggregate multiple signals at a timestamp into one signal.
        
        Args:
            timestamp: The timestamp for aggregation
            signals: List of signals to aggregate
            
        Returns:
            An aggregated signal
        """
        if not signals:
            return None
        conflict = self._detect_conflicts(timestamp, signals)
        if conflict:
            self.conflicts.append(conflict)
            resolved = self._resolve_conflict(conflict)
            if resolved:
                signal_type = resolved.signal_type
                strength = resolved.strength
            else:
                signal_type = SignalType.NEUTRAL
                strength = 0.0
        else:
            buy_signals = [s for s in signals if s.is_buy]
            sell_signals = [s for s in signals if s.is_sell]
            if buy_signals and not sell_signals:
                signal_type = SignalType.BUY
                strength = sum(s.strength for s in buy_signals) / len(
                    buy_signals)
            elif sell_signals and not buy_signals:
                signal_type = SignalType.SELL
                strength = sum(s.strength for s in sell_signals) / len(
                    sell_signals)
            else:
                signal_type = SignalType.NEUTRAL
                strength = 0.0
        if len(signals) > 1:
            total_agreement = 0.0
            pairs = 0
            for i in range(len(signals)):
                for j in range(i + 1, len(signals)):
                    s1, s2 = signals[i], signals[j]
                    if s1.signal_type == s2.signal_type:
                        total_agreement += 1.0
                    elif s1.is_buy and s2.is_sell or s1.is_sell and s2.is_buy:
                        total_agreement += 0.0
                    else:
                        total_agreement += 0.5
                    pairs += 1
            confidence = total_agreement / pairs if pairs > 0 else 0.5
        else:
            confidence = signals[0].strength
        return AggregatedSignal(timestamp=timestamp, signal_type=
            signal_type, strength=strength, confidence=confidence, price=
            signals[0].price, contributing_signals=signals)

    @with_resilience('process_signals')
    def process_signals(self) ->List[AggregatedSignal]:
        """
        Process all signals to create aggregated signals.
        
        Returns:
            List of aggregated signals
        """
        self.aggregated_signals = []
        signals_by_timestamp = {}
        for indicator_name, indicator_signals in self.signals.items():
            for signal in indicator_signals:
                if signal.timestamp not in signals_by_timestamp:
                    signals_by_timestamp[signal.timestamp] = []
                signals_by_timestamp[signal.timestamp].append(signal)
        for timestamp, timestamp_signals in sorted(signals_by_timestamp.items()
            ):
            aggregated = self._aggregate_signals(timestamp, timestamp_signals)
            if aggregated:
                self.aggregated_signals.append(aggregated)
        if not self.concordance:
            all_indicators = list(self.signals.keys())
            self.concordance = SignalConcordance(all_indicators)
        self.concordance.update(self.signals)
        return self.aggregated_signals

    @with_resilience('get_indicators_by_reliability')
    def get_indicators_by_reliability(self) ->List[Tuple[str, float]]:
        """Get indicators sorted by reliability."""
        return sorted(self.indicator_reliability.items(), key=lambda x: x[1
            ], reverse=True)

    @with_resilience('get_conflicts')
    def get_conflicts(self) ->List[SignalConflict]:
        """Get all detected conflicts."""
        return self.conflicts

    @with_resilience('get_signals_for_period')
    def get_signals_for_period(self, start_time: datetime, end_time: datetime
        ) ->List[AggregatedSignal]:
        """
        Get aggregated signals for a specific time period.
        
        Args:
            start_time: Start of the period
            end_time: End of the period
            
        Returns:
            List of signals within the time period
        """
        return [s for s in self.aggregated_signals if start_time <= s.
            timestamp <= end_time]

    def clear_signals(self, older_than: Optional[datetime]=None) ->None:
        """
        Clear signals from the system.
        
        Args:
            older_than: If provided, only clear signals older than this timestamp
        """
        if older_than is None:
            self.signals = {}
            self.aggregated_signals = []
            self.conflicts = []
            logger.info('Cleared all signals')
        else:
            for indicator, signals in self.signals.items():
                self.signals[indicator] = [s for s in signals if s.
                    timestamp >= older_than]
            self.aggregated_signals = [s for s in self.aggregated_signals if
                s.timestamp >= older_than]
            self.conflicts = [c for c in self.conflicts if c.timestamp >=
                older_than]
            logger.info(f'Cleared signals older than {older_than}')


signal_system = SignalSystem()
