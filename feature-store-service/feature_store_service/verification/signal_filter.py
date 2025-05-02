"""
Signal Filtering System for detecting and filtering false signals and errors.
Implements sophisticated filtering mechanisms for improving signal quality.
"""
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class SignalConfidence(Enum):
    """Confidence levels for signals"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
    INVALID = "invalid"

class SignalType(Enum):
    """Types of signals that can be filtered"""
    PRICE = "price"
    VOLUME = "volume"
    INDICATOR = "indicator"
    PATTERN = "pattern"
    MARKET = "market"
    COMPOSITE = "composite"

class FilteredSignal:
    """Represents a filtered signal with metadata"""
    def __init__(self, 
                 signal_type: SignalType,
                 original_value: Any,
                 filtered_value: Any,
                 confidence: SignalConfidence,
                 metadata: Optional[Dict[str, Any]] = None):
        self.signal_type = signal_type
        self.original_value = original_value
        self.filtered_value = filtered_value
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()

class SignalFilter:
    """
    Implements comprehensive signal filtering capabilities.
    Provides methods for detecting and filtering false signals and errors.
    """
    def __init__(self):
        self.filter_history: List[FilteredSignal] = []
        self.signal_patterns: Dict[SignalType, List[Dict[str, Any]]] = {
            signal_type: [] for signal_type in SignalType
        }

    def filter_signal(self, 
                     signal_type: Union[str, SignalType],
                     value: Any,
                     context: Optional[Dict[str, Any]] = None,
                     **kwargs) -> FilteredSignal:
        """
        Filter a signal and assess its reliability.
        Args:
            signal_type: Type of signal to filter
            value: Signal value to filter
            context: Additional context for filtering
            **kwargs: Additional filtering parameters
        Returns:
            FilteredSignal containing filtered value and metadata
        """
        try:
            if isinstance(signal_type, str):
                try:
                    signal_type = SignalType(signal_type.upper())
                except ValueError:
                    logger.error(f"Invalid signal type: {signal_type}")
                    return self._create_invalid_signal(signal_type, value)

            # Apply appropriate filtering strategy
            if signal_type == SignalType.PRICE:
                filtered = self._filter_price_signal(value, context, **kwargs)
            elif signal_type == SignalType.VOLUME:
                filtered = self._filter_volume_signal(value, context, **kwargs)
            elif signal_type == SignalType.INDICATOR:
                filtered = self._filter_indicator_signal(value, context, **kwargs)
            elif signal_type == SignalType.PATTERN:
                filtered = self._filter_pattern_signal(value, context, **kwargs)
            elif signal_type == SignalType.MARKET:
                filtered = self._filter_market_signal(value, context, **kwargs)
            elif signal_type == SignalType.COMPOSITE:
                filtered = self._filter_composite_signal(value, context, **kwargs)
            else:
                return self._create_invalid_signal(signal_type, value)

            # Update signal history and patterns
            self._update_signal_patterns(signal_type, filtered)
            self.filter_history.append(filtered)
            
            return filtered
        except Exception as e:
            logger.error(f"Signal filtering error: {str(e)}")
            return self._create_invalid_signal(signal_type, value)

    def _filter_price_signal(self, 
                           value: Union[float, pd.Series],
                           context: Optional[Dict[str, Any]] = None,
                           **kwargs) -> FilteredSignal:
        """Filter price-related signals"""
        if isinstance(value, pd.Series):
            filtered_value = self._filter_price_series(value, **kwargs)
            confidence = self._assess_price_confidence(value, filtered_value, context)
        else:
            filtered_value = self._filter_single_price(value, context, **kwargs)
            confidence = self._assess_single_price_confidence(value, filtered_value, context)

        return FilteredSignal(
            signal_type=SignalType.PRICE,
            original_value=value,
            filtered_value=filtered_value,
            confidence=confidence,
            metadata={'context': context}
        )

    def _filter_volume_signal(self,
                            value: Union[float, pd.Series],
                            context: Optional[Dict[str, Any]] = None,
                            **kwargs) -> FilteredSignal:
        """Filter volume-related signals"""
        if isinstance(value, pd.Series):
            filtered_value = self._filter_volume_series(value, **kwargs)
            confidence = self._assess_volume_confidence(value, filtered_value, context)
        else:
            filtered_value = self._filter_single_volume(value, context, **kwargs)
            confidence = self._assess_single_volume_confidence(value, filtered_value, context)

        return FilteredSignal(
            signal_type=SignalType.VOLUME,
            original_value=value,
            filtered_value=filtered_value,
            confidence=confidence,
            metadata={'context': context}
        )

    def _filter_pattern_signal(self,
                             value: Dict[str, Any],
                             context: Optional[Dict[str, Any]] = None,
                             **kwargs) -> FilteredSignal:
        """Filter pattern recognition signals"""
        pattern_type = value.get('type', 'unknown')
        confidence_score = value.get('confidence', 0.5)
        
        # Validate pattern characteristics
        completion = value.get('completion', 1.0)
        quality = value.get('quality', 0.5)
        
        # Apply quality thresholds
        if completion < 0.7 or quality < 0.3:
            confidence = SignalConfidence.LOW
        elif completion > 0.9 and quality > 0.7:
            confidence = SignalConfidence.HIGH
        else:
            confidence = SignalConfidence.MEDIUM
            
        # Adjust confidence based on historical pattern reliability
        if pattern_type in self.pattern_reliability:
            historical_reliability = self.pattern_reliability[pattern_type]
            confidence = self._adjust_confidence_by_history(confidence, historical_reliability)

        return FilteredSignal(
            signal_type=SignalType.PATTERN,
            original_value=value,
            filtered_value=value,  # Patterns are typically not modified, just validated
            confidence=confidence,
            metadata={
                'pattern_type': pattern_type,
                'completion': completion,
                'quality': quality,
                'context': context
            }
        )

    def _filter_indicator_signal(self,
                               value: Any,
                               context: Optional[Dict[str, Any]] = None,
                               **kwargs) -> FilteredSignal:
        """Filter technical indicator signals"""
        indicator_type = kwargs.get('indicator_type', 'generic')
        
        if isinstance(value, pd.Series):
            filtered_value = self._filter_indicator_series(value, **kwargs)
            confidence = self._assess_indicator_confidence(value, filtered_value, context)
        else:
            filtered_value = self._filter_single_indicator(value, context, **kwargs)
            confidence = self._assess_single_indicator_confidence(value, filtered_value, context)

        return FilteredSignal(
            signal_type=SignalType.INDICATOR,
            original_value=value,
            filtered_value=filtered_value,
            confidence=confidence,
            metadata={'indicator_type': indicator_type, 'context': context}
        )

    def _filter_market_signal(self,
                            value: Any,
                            context: Optional[Dict[str, Any]] = None,
                            **kwargs) -> FilteredSignal:
        """Filter market condition signals"""
        if isinstance(value, dict):
            filtered_value = self._filter_market_conditions(value, **kwargs)
            confidence = self._assess_market_confidence(value, filtered_value, context)
        else:
            filtered_value = self._filter_single_market_condition(value, context, **kwargs)
            confidence = self._assess_single_market_confidence(value, filtered_value, context)

        return FilteredSignal(
            signal_type=SignalType.MARKET,
            original_value=value,
            filtered_value=filtered_value,
            confidence=confidence,
            metadata={'context': context}
        )

    def _filter_composite_signal(self,
                               value: Dict[str, Any],
                               context: Optional[Dict[str, Any]] = None,
                               **kwargs) -> FilteredSignal:
        """Filter composite signals combining multiple signal types"""
        filtered_components = {}
        component_confidences = []

        for component, comp_value in value.items():
            if component in SignalType.__members__:
                signal_type = SignalType[component]
                filtered = self.filter_signal(signal_type, comp_value, context, **kwargs)
                filtered_components[component] = filtered.filtered_value
                component_confidences.append(filtered.confidence)

        # Determine overall confidence
        confidence = self._aggregate_confidences(component_confidences)

        return FilteredSignal(
            signal_type=SignalType.COMPOSITE,
            original_value=value,
            filtered_value=filtered_components,
            confidence=confidence,
            metadata={'components': list(filtered_components.keys()), 'context': context}
        )

    def _filter_price_series(self, series: pd.Series, **kwargs) -> pd.Series:
        """Apply filtering to price time series"""
        std_threshold = kwargs.get('std_threshold', 3.0)
        window = kwargs.get('window', 20)
        
        # Calculate rolling statistics
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        # Identify outliers
        upper_bound = rolling_mean + (std_threshold * rolling_std)
        lower_bound = rolling_mean - (std_threshold * rolling_std)
        
        # Replace outliers with rolling mean
        mask = (series > upper_bound) | (series < lower_bound)
        filtered = series.copy()
        filtered[mask] = rolling_mean[mask]
        
        return filtered

    def _filter_volume_series(self, series: pd.Series, **kwargs) -> pd.Series:
        """Apply filtering to volume time series"""
        std_threshold = kwargs.get('std_threshold', 4.0)
        min_volume = kwargs.get('min_volume', 0)
        
        # Remove negative volumes
        series = series.clip(lower=min_volume)
        
        # Filter extreme volumes
        mean_volume = series.mean()
        std_volume = series.std()
        
        upper_bound = mean_volume + (std_threshold * std_volume)
        mask = series > upper_bound
        
        filtered = series.copy()
        filtered[mask] = upper_bound
        
        return filtered

    def _assess_price_confidence(self,
                               original: pd.Series,
                               filtered: pd.Series,
                               context: Optional[Dict[str, Any]] = None) -> SignalConfidence:
        """Assess confidence in price signal"""
        if context is None:
            context = {}
            
        # Calculate percentage of outliers
        total_points = len(original)
        outliers = (original != filtered).sum()
        outlier_ratio = outliers / total_points
        
        # Consider market volatility if available
        volatility = context.get('volatility', 0.0)
        
        if outlier_ratio > 0.2 or volatility > 0.5:
            return SignalConfidence.LOW
        elif outlier_ratio > 0.1 or volatility > 0.3:
            return SignalConfidence.MEDIUM
        else:
            return SignalConfidence.HIGH

    def _update_signal_patterns(self, 
                              signal_type: SignalType,
                              filtered_signal: FilteredSignal) -> None:
        """Update known signal patterns"""
        pattern = {
            'timestamp': filtered_signal.timestamp,
            'confidence': filtered_signal.confidence,
            'metadata': filtered_signal.metadata
        }
        
        self.signal_patterns[signal_type].append(pattern)
        
        # Keep only recent patterns
        max_patterns = 1000
        if len(self.signal_patterns[signal_type]) > max_patterns:
            self.signal_patterns[signal_type] = self.signal_patterns[signal_type][-max_patterns:]

    def _create_invalid_signal(self, 
                             signal_type: Union[str, SignalType],
                             value: Any) -> FilteredSignal:
        """Create an invalid signal response"""
        return FilteredSignal(
            signal_type=signal_type if isinstance(signal_type, SignalType) else SignalType.COMPOSITE,
            original_value=value,
            filtered_value=None,
            confidence=SignalConfidence.INVALID,
            metadata={'error': 'Invalid signal type or processing error'}
        )

    def _aggregate_confidences(self, 
                             confidences: List[SignalConfidence]) -> SignalConfidence:
        """Aggregate multiple confidence levels"""
        if not confidences:
            return SignalConfidence.INVALID
            
        if SignalConfidence.INVALID in confidences:
            return SignalConfidence.INVALID
            
        confidence_weights = {
            SignalConfidence.VERY_HIGH: 4,
            SignalConfidence.HIGH: 3,
            SignalConfidence.MEDIUM: 2,
            SignalConfidence.LOW: 1,
            SignalConfidence.VERY_LOW: 0
        }
        
        total_weight = sum(confidence_weights[conf] for conf in confidences)
        max_possible = len(confidences) * confidence_weights[SignalConfidence.VERY_HIGH]
        
        ratio = total_weight / max_possible
        
        if ratio >= 0.8:
            return SignalConfidence.VERY_HIGH
        elif ratio >= 0.6:
            return SignalConfidence.HIGH
        elif ratio >= 0.4:
            return SignalConfidence.MEDIUM
        elif ratio >= 0.2:
            return SignalConfidence.LOW
        else:
            return SignalConfidence.VERY_LOW

    def get_filter_summary(self) -> Dict[str, Any]:
        """Get summary of filtering results"""
        summary = {
            'total_signals': len(self.filter_history),
            'by_type': {},
            'by_confidence': {
                confidence.value: 0 for confidence in SignalConfidence
            }
        }

        for signal_type in SignalType:
            type_signals = [s for s in self.filter_history if s.signal_type == signal_type]
            if type_signals:
                summary['by_type'][signal_type.value] = {
                    'total': len(type_signals),
                    'confidence_distribution': {
                        confidence.value: len([s for s in type_signals if s.confidence == confidence])
                        for confidence in SignalConfidence
                    }
                }

        for signal in self.filter_history:
            summary['by_confidence'][signal.confidence.value] += 1

        return summary
