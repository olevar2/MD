"""
Business Domain Metrics for Analysis Engine Service.

This module provides Prometheus metrics for business-specific aspects of the
forex trading analysis domain, including signal quality, prediction accuracy,
market regime detection, and pattern recognition performance.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary
from typing import Dict, Any, Optional, List, Union
import time

# Signal Quality Metrics
SIGNAL_QUALITY_SCORE = Gauge(
    'analysis_engine_signal_quality_score',
    'Quality score of trading signals (0-100)',
    ['signal_type', 'symbol', 'timeframe', 'strategy']
)

SIGNAL_STRENGTH = Gauge(
    'analysis_engine_signal_strength',
    'Strength of trading signals (-100 to 100, negative for bearish)',
    ['signal_type', 'symbol', 'timeframe', 'strategy']
)

SIGNAL_CONFIDENCE = Gauge(
    'analysis_engine_signal_confidence',
    'Confidence level in trading signals (0-100)',
    ['signal_type', 'symbol', 'timeframe', 'strategy']
)

# Prediction Accuracy Metrics
PREDICTION_ACCURACY = Gauge(
    'analysis_engine_prediction_accuracy',
    'Accuracy of price movement predictions (0-100)',
    ['prediction_type', 'symbol', 'timeframe', 'horizon']
)

PREDICTION_ERROR = Gauge(
    'analysis_engine_prediction_error',
    'Error in price movement predictions (absolute value)',
    ['prediction_type', 'symbol', 'timeframe', 'horizon']
)

PREDICTION_BIAS = Gauge(
    'analysis_engine_prediction_bias',
    'Bias in price movement predictions (positive for upward bias)',
    ['prediction_type', 'symbol', 'timeframe', 'horizon']
)

# Market Regime Metrics
MARKET_REGIME = Gauge(
    'analysis_engine_market_regime',
    'Current market regime classification',
    ['symbol', 'timeframe', 'regime_type']
)

REGIME_TRANSITION_PROBABILITY = Gauge(
    'analysis_engine_regime_transition_probability',
    'Probability of transition to a different market regime (0-100)',
    ['symbol', 'timeframe', 'from_regime', 'to_regime']
)

REGIME_STABILITY = Gauge(
    'analysis_engine_regime_stability',
    'Stability of current market regime (0-100)',
    ['symbol', 'timeframe', 'regime_type']
)

# Pattern Recognition Metrics
PATTERN_RECOGNITION_TOTAL = Counter(
    'analysis_engine_pattern_recognition_total',
    'Total number of patterns recognized',
    ['pattern_type', 'symbol', 'timeframe']
)

PATTERN_STRENGTH = Gauge(
    'analysis_engine_pattern_strength',
    'Strength of recognized patterns (0-100)',
    ['pattern_type', 'symbol', 'timeframe']
)

PATTERN_COMPLETION = Gauge(
    'analysis_engine_pattern_completion',
    'Completion percentage of recognized patterns (0-100)',
    ['pattern_type', 'symbol', 'timeframe']
)

# Correlation Metrics
ASSET_CORRELATION = Gauge(
    'analysis_engine_asset_correlation',
    'Correlation between assets (-100 to 100)',
    ['base_symbol', 'correlated_symbol', 'timeframe']
)

CORRELATION_CHANGE_RATE = Gauge(
    'analysis_engine_correlation_change_rate',
    'Rate of change in correlation between assets',
    ['base_symbol', 'correlated_symbol', 'timeframe']
)

# Volatility Metrics
VOLATILITY_INDEX = Gauge(
    'analysis_engine_volatility_index',
    'Volatility index for assets (0-100)',
    ['symbol', 'timeframe', 'calculation_method']
)

VOLATILITY_FORECAST = Gauge(
    'analysis_engine_volatility_forecast',
    'Forecasted volatility for assets (0-100)',
    ['symbol', 'timeframe', 'horizon']
)

# Sentiment Analysis Metrics
SENTIMENT_SCORE = Gauge(
    'analysis_engine_sentiment_score',
    'Sentiment score for assets (-100 to 100, negative for bearish)',
    ['symbol', 'source', 'timeframe']
)

SENTIMENT_VOLUME = Gauge(
    'analysis_engine_sentiment_volume',
    'Volume of sentiment data for assets',
    ['symbol', 'source', 'timeframe']
)

# Strategy Performance Metrics
STRATEGY_PERFORMANCE_SCORE = Gauge(
    'analysis_engine_strategy_performance_score',
    'Performance score of trading strategies (0-100)',
    ['strategy', 'symbol', 'timeframe']
)

STRATEGY_SHARPE_RATIO = Gauge(
    'analysis_engine_strategy_sharpe_ratio',
    'Sharpe ratio of trading strategies',
    ['strategy', 'symbol', 'timeframe']
)

STRATEGY_DRAWDOWN = Gauge(
    'analysis_engine_strategy_drawdown',
    'Maximum drawdown of trading strategies (percentage)',
    ['strategy', 'symbol', 'timeframe']
)

# Utility class for recording business metrics
class BusinessMetricsRecorder:
    """Utility class for recording business-specific metrics."""
    
    @staticmethod
    def record_signal_quality(
        signal_type: str,
        symbol: str,
        timeframe: str,
        strategy: str,
        quality: float
    ) -> None:
        """
        Record signal quality score.
        
        Args:
            signal_type: Type of signal (e.g., 'buy', 'sell', 'hold')
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            strategy: Strategy name
            quality: Quality score (0-100)
        """
        SIGNAL_QUALITY_SCORE.labels(
            signal_type=signal_type,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy
        ).set(quality)
    
    @staticmethod
    def record_signal_strength(
        signal_type: str,
        symbol: str,
        timeframe: str,
        strategy: str,
        strength: float
    ) -> None:
        """
        Record signal strength.
        
        Args:
            signal_type: Type of signal (e.g., 'buy', 'sell', 'hold')
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            strategy: Strategy name
            strength: Signal strength (-100 to 100, negative for bearish)
        """
        SIGNAL_STRENGTH.labels(
            signal_type=signal_type,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy
        ).set(strength)
    
    @staticmethod
    def record_signal_confidence(
        signal_type: str,
        symbol: str,
        timeframe: str,
        strategy: str,
        confidence: float
    ) -> None:
        """
        Record signal confidence.
        
        Args:
            signal_type: Type of signal (e.g., 'buy', 'sell', 'hold')
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            strategy: Strategy name
            confidence: Confidence level (0-100)
        """
        SIGNAL_CONFIDENCE.labels(
            signal_type=signal_type,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy
        ).set(confidence)
    
    @staticmethod
    def record_prediction_accuracy(
        prediction_type: str,
        symbol: str,
        timeframe: str,
        horizon: str,
        accuracy: float
    ) -> None:
        """
        Record prediction accuracy.
        
        Args:
            prediction_type: Type of prediction (e.g., 'price', 'direction', 'volatility')
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            horizon: Prediction horizon (e.g., '1h', '1d', '1w')
            accuracy: Accuracy score (0-100)
        """
        PREDICTION_ACCURACY.labels(
            prediction_type=prediction_type,
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon
        ).set(accuracy)
    
    @staticmethod
    def record_prediction_error(
        prediction_type: str,
        symbol: str,
        timeframe: str,
        horizon: str,
        error: float
    ) -> None:
        """
        Record prediction error.
        
        Args:
            prediction_type: Type of prediction (e.g., 'price', 'direction', 'volatility')
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            horizon: Prediction horizon (e.g., '1h', '1d', '1w')
            error: Error value (absolute)
        """
        PREDICTION_ERROR.labels(
            prediction_type=prediction_type,
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon
        ).set(error)
    
    @staticmethod
    def record_market_regime(
        symbol: str,
        timeframe: str,
        regime_type: str,
        value: float = 1.0
    ) -> None:
        """
        Record market regime classification.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            regime_type: Type of market regime (e.g., 'trending', 'ranging', 'volatile')
            value: Value to set (typically 1.0 for active regime, 0.0 for inactive)
        """
        MARKET_REGIME.labels(
            symbol=symbol,
            timeframe=timeframe,
            regime_type=regime_type
        ).set(value)
    
    @staticmethod
    def record_regime_stability(
        symbol: str,
        timeframe: str,
        regime_type: str,
        stability: float
    ) -> None:
        """
        Record market regime stability.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            regime_type: Type of market regime (e.g., 'trending', 'ranging', 'volatile')
            stability: Stability score (0-100)
        """
        REGIME_STABILITY.labels(
            symbol=symbol,
            timeframe=timeframe,
            regime_type=regime_type
        ).set(stability)
    
    @staticmethod
    def record_pattern_recognition(
        pattern_type: str,
        symbol: str,
        timeframe: str
    ) -> None:
        """
        Record pattern recognition.
        
        Args:
            pattern_type: Type of pattern (e.g., 'head_and_shoulders', 'double_top')
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
        """
        PATTERN_RECOGNITION_TOTAL.labels(
            pattern_type=pattern_type,
            symbol=symbol,
            timeframe=timeframe
        ).inc()
    
    @staticmethod
    def record_pattern_strength(
        pattern_type: str,
        symbol: str,
        timeframe: str,
        strength: float
    ) -> None:
        """
        Record pattern strength.
        
        Args:
            pattern_type: Type of pattern (e.g., 'head_and_shoulders', 'double_top')
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            strength: Strength score (0-100)
        """
        PATTERN_STRENGTH.labels(
            pattern_type=pattern_type,
            symbol=symbol,
            timeframe=timeframe
        ).set(strength)
    
    @staticmethod
    def record_asset_correlation(
        base_symbol: str,
        correlated_symbol: str,
        timeframe: str,
        correlation: float
    ) -> None:
        """
        Record asset correlation.
        
        Args:
            base_symbol: Base trading symbol (e.g., 'EURUSD')
            correlated_symbol: Correlated trading symbol (e.g., 'GBPUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            correlation: Correlation value (-100 to 100)
        """
        ASSET_CORRELATION.labels(
            base_symbol=base_symbol,
            correlated_symbol=correlated_symbol,
            timeframe=timeframe
        ).set(correlation)
    
    @staticmethod
    def record_volatility_index(
        symbol: str,
        timeframe: str,
        calculation_method: str,
        volatility: float
    ) -> None:
        """
        Record volatility index.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            calculation_method: Method used to calculate volatility (e.g., 'atr', 'std_dev')
            volatility: Volatility index (0-100)
        """
        VOLATILITY_INDEX.labels(
            symbol=symbol,
            timeframe=timeframe,
            calculation_method=calculation_method
        ).set(volatility)
    
    @staticmethod
    def record_sentiment_score(
        symbol: str,
        source: str,
        timeframe: str,
        score: float
    ) -> None:
        """
        Record sentiment score.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            source: Sentiment data source (e.g., 'news', 'social_media')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            score: Sentiment score (-100 to 100, negative for bearish)
        """
        SENTIMENT_SCORE.labels(
            symbol=symbol,
            source=source,
            timeframe=timeframe
        ).set(score)
    
    @staticmethod
    def record_strategy_performance(
        strategy: str,
        symbol: str,
        timeframe: str,
        performance: float
    ) -> None:
        """
        Record strategy performance score.
        
        Args:
            strategy: Strategy name
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            performance: Performance score (0-100)
        """
        STRATEGY_PERFORMANCE_SCORE.labels(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe
        ).set(performance)
    
    @staticmethod
    def record_strategy_sharpe_ratio(
        strategy: str,
        symbol: str,
        timeframe: str,
        sharpe_ratio: float
    ) -> None:
        """
        Record strategy Sharpe ratio.
        
        Args:
            strategy: Strategy name
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            sharpe_ratio: Sharpe ratio value
        """
        STRATEGY_SHARPE_RATIO.labels(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe
        ).set(sharpe_ratio)
    
    @staticmethod
    def record_strategy_drawdown(
        strategy: str,
        symbol: str,
        timeframe: str,
        drawdown: float
    ) -> None:
        """
        Record strategy maximum drawdown.
        
        Args:
            strategy: Strategy name
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Analysis timeframe (e.g., '1h', '4h', '1d')
            drawdown: Maximum drawdown percentage
        """
        STRATEGY_DRAWDOWN.labels(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe
        ).set(drawdown)
