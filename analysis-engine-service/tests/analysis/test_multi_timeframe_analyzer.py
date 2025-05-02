"""
Unit tests for the MultiTimeframeAnalyzer component.

This module provides comprehensive tests for the MultiTimeframeAnalyzer, which analyzes
market data across multiple timeframes to identify stronger signals and trends.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from analysis_engine.analysis.multi_timeframe.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from analysis_engine.core.errors import AnalysisError

@pytest.fixture
def sample_multi_timeframe_data():
    """Create sample multi-timeframe market data for testing."""
    # Create base data for M15 timeframe
    base_dates = [datetime.now() - timedelta(minutes=15*i) for i in range(200, 0, -1)]
    
    # Create a price series with an uptrend
    base_close = [1.1000]
    for i in range(1, 200):
        # Add trend
        trend = 0.0001 * i
        # Add some noise
        noise = np.random.normal(0, 0.0002)
        base_close.append(base_close[-1] + trend + noise)
    
    # Create high and low prices
    base_high = [price + np.random.uniform(0.0005, 0.0015) for price in base_close]
    base_low = [price - np.random.uniform(0.0005, 0.0015) for price in base_close]
    base_open = [prev_close + np.random.normal(0, 0.0003) for prev_close in base_close[:-1]]
    base_open.insert(0, base_close[0] - 0.0005)
    
    # Create volume data
    base_volume = [int(1000 * (1 + np.random.uniform(-0.3, 0.3))) for _ in range(200)]
    
    # Create M15 DataFrame
    df_m15 = pd.DataFrame({
        'timestamp': base_dates,
        'open': base_open,
        'high': base_high,
        'low': base_low,
        'close': base_close,
        'volume': base_volume
    })
    
    # Create H1 data by resampling M15 data
    df_h1 = df_m15.copy()
    df_h1 = df_h1.set_index('timestamp')
    df_h1 = df_h1.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    # Create H4 data by resampling H1 data
    df_h4 = df_h1.copy()
    df_h4 = df_h4.set_index('timestamp')
    df_h4 = df_h4.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    # Convert to dictionary format expected by the analyzer
    market_data = {
        "M15": {
            'timestamp': df_m15['timestamp'].tolist(),
            'open': df_m15['open'].tolist(),
            'high': df_m15['high'].tolist(),
            'low': df_m15['low'].tolist(),
            'close': df_m15['close'].tolist(),
            'volume': df_m15['volume'].tolist()
        },
        "H1": {
            'timestamp': df_h1['timestamp'].tolist(),
            'open': df_h1['open'].tolist(),
            'high': df_h1['high'].tolist(),
            'low': df_h1['low'].tolist(),
            'close': df_h1['close'].tolist(),
            'volume': df_h1['volume'].tolist()
        },
        "H4": {
            'timestamp': df_h4['timestamp'].tolist(),
            'open': df_h4['open'].tolist(),
            'high': df_h4['high'].tolist(),
            'low': df_h4['low'].tolist(),
            'close': df_h4['close'].tolist(),
            'volume': df_h4['volume'].tolist()
        }
    }
    
    return market_data

@pytest.fixture
def analyzer():
    """Create a MultiTimeframeAnalyzer instance for testing."""
    return MultiTimeframeAnalyzer()

def test_multi_timeframe_analyzer_initialization():
    """Test that the MultiTimeframeAnalyzer initializes with correct default parameters."""
    analyzer = MultiTimeframeAnalyzer()
    
    assert analyzer.name == "multi_timeframe_analyzer"
    assert isinstance(analyzer.parameters, dict)
    assert "correlation_threshold" in analyzer.parameters
    assert "min_timeframes" in analyzer.parameters

def test_analyze_with_valid_data(analyzer, sample_multi_timeframe_data):
    """Test the analyze method with valid market data."""
    data = {
        "symbol": "EURUSD",
        "timeframes": ["M15", "H1", "H4"],
        "market_data": sample_multi_timeframe_data
    }
    
    result = analyzer.analyze(data)
    
    # Verify result structure
    assert result.analyzer_name == "multi_timeframe_analyzer"
    assert result.is_valid is True
    assert result.error is None
    assert isinstance(result.result, dict)
    
    # Verify result content
    assert "symbol" in result.result
    assert result.result["symbol"] == "EURUSD"
    assert "timeframe_analysis" in result.result
    assert "correlation_matrix" in result.result
    assert "aligned_signals" in result.result
    assert "dominant_timeframe" in result.result
    
    # Verify timeframe analysis
    for tf in ["M15", "H1", "H4"]:
        assert tf in result.result["timeframe_analysis"]
        tf_analysis = result.result["timeframe_analysis"][tf]
        assert "trend" in tf_analysis
        assert "strength" in tf_analysis
        assert "key_levels" in tf_analysis
        assert "volume_profile" in tf_analysis
    
    # Verify correlation matrix
    assert isinstance(result.result["correlation_matrix"], dict)
    for tf1 in ["M15", "H1", "H4"]:
        assert tf1 in result.result["correlation_matrix"]
        for tf2 in ["M15", "H1", "H4"]:
            assert tf2 in result.result["correlation_matrix"][tf1]
            assert -1 <= result.result["correlation_matrix"][tf1][tf2] <= 1
    
    # Verify metadata
    assert "timeframe_count" in result.metadata
    assert result.metadata["timeframe_count"] == 3
    assert "analysis_duration_ms" in result.metadata

def test_analyze_with_missing_data(analyzer):
    """Test the analyze method with missing required data."""
    # Missing market_data
    data = {
        "symbol": "EURUSD",
        "timeframes": ["M15", "H1", "H4"]
    }
    
    result = analyzer.analyze(data)
    
    assert result.is_valid is False
    assert result.error is not None
    assert "market_data" in str(result.error)
    
    # Missing symbol
    data = {
        "timeframes": ["M15", "H1", "H4"],
        "market_data": {}
    }
    
    result = analyzer.analyze(data)
    
    assert result.is_valid is False
    assert result.error is not None
    assert "symbol" in str(result.error)
    
    # Missing timeframes
    data = {
        "symbol": "EURUSD",
        "market_data": {}
    }
    
    result = analyzer.analyze(data)
    
    assert result.is_valid is False
    assert result.error is not None
    assert "timeframes" in str(result.error)

def test_analyze_timeframe(analyzer, sample_multi_timeframe_data):
    """Test the _analyze_timeframe method."""
    tf_data = sample_multi_timeframe_data["H1"]
    
    result = analyzer._analyze_timeframe("H1", tf_data)
    
    assert isinstance(result, dict)
    assert "trend" in result
    assert "strength" in result
    assert "key_levels" in result
    assert "volume_profile" in result
    
    # Verify trend
    assert result["trend"] in ["bullish", "bearish", "neutral"]
    
    # Verify strength
    assert 0 <= result["strength"] <= 1
    
    # Verify key levels
    assert isinstance(result["key_levels"], list)
    if result["key_levels"]:
        level = result["key_levels"][0]
        assert "price" in level
        assert "type" in level
        assert level["type"] in ["support", "resistance"]
    
    # Verify volume profile
    assert isinstance(result["volume_profile"], dict)
    assert "high_volume_nodes" in result["volume_profile"]
    assert "low_volume_nodes" in result["volume_profile"]

def test_calculate_correlation_matrix(analyzer, sample_multi_timeframe_data):
    """Test the _calculate_correlation_matrix method."""
    # First analyze each timeframe
    timeframe_analysis = {}
    for tf, tf_data in sample_multi_timeframe_data.items():
        timeframe_analysis[tf] = analyzer._analyze_timeframe(tf, tf_data)
    
    # Then calculate correlation matrix
    correlation_matrix = analyzer._calculate_correlation_matrix(timeframe_analysis)
    
    assert isinstance(correlation_matrix, dict)
    
    # Check correlation values
    for tf1 in sample_multi_timeframe_data.keys():
        assert tf1 in correlation_matrix
        for tf2 in sample_multi_timeframe_data.keys():
            assert tf2 in correlation_matrix[tf1]
            assert -1 <= correlation_matrix[tf1][tf2] <= 1
            # Correlation with self should be 1
            if tf1 == tf2:
                assert correlation_matrix[tf1][tf2] == 1

def test_identify_aligned_signals(analyzer, sample_multi_timeframe_data):
    """Test the _identify_aligned_signals method."""
    # First analyze each timeframe
    timeframe_analysis = {}
    for tf, tf_data in sample_multi_timeframe_data.items():
        timeframe_analysis[tf] = analyzer._analyze_timeframe(tf, tf_data)
    
    # Then calculate correlation matrix
    correlation_matrix = analyzer._calculate_correlation_matrix(timeframe_analysis)
    
    # Then identify aligned signals
    aligned_signals = analyzer._identify_aligned_signals(timeframe_analysis, correlation_matrix)
    
    assert isinstance(aligned_signals, dict)
    assert "trend" in aligned_signals
    assert "key_levels" in aligned_signals
    assert "volume_nodes" in aligned_signals
    
    # Check trend alignment
    assert "direction" in aligned_signals["trend"]
    assert "strength" in aligned_signals["trend"]
    assert "timeframes" in aligned_signals["trend"]
    assert aligned_signals["trend"]["direction"] in ["bullish", "bearish", "neutral"]
    assert 0 <= aligned_signals["trend"]["strength"] <= 1
    assert isinstance(aligned_signals["trend"]["timeframes"], list)

def test_determine_dominant_timeframe(analyzer, sample_multi_timeframe_data):
    """Test the _determine_dominant_timeframe method."""
    # First analyze each timeframe
    timeframe_analysis = {}
    for tf, tf_data in sample_multi_timeframe_data.items():
        timeframe_analysis[tf] = analyzer._analyze_timeframe(tf, tf_data)
    
    # Then calculate correlation matrix
    correlation_matrix = analyzer._calculate_correlation_matrix(timeframe_analysis)
    
    # Then determine dominant timeframe
    dominant_tf = analyzer._determine_dominant_timeframe(timeframe_analysis, correlation_matrix)
    
    assert dominant_tf in sample_multi_timeframe_data.keys()

def test_analyze_with_single_timeframe(analyzer, sample_multi_timeframe_data):
    """Test the analyze method with a single timeframe."""
    data = {
        "symbol": "EURUSD",
        "timeframes": ["H1"],
        "market_data": {
            "H1": sample_multi_timeframe_data["H1"]
        }
    }
    
    result = analyzer.analyze(data)
    
    assert result.is_valid is True
    assert "timeframe_analysis" in result.result
    assert "H1" in result.result["timeframe_analysis"]
    assert result.result["dominant_timeframe"] == "H1"
    assert result.metadata["timeframe_count"] == 1

def test_analyze_with_insufficient_timeframes(analyzer, sample_multi_timeframe_data):
    """Test the analyze method with insufficient timeframes."""
    # Set min_timeframes to 3 but only provide 2
    analyzer.parameters["min_timeframes"] = 3
    
    data = {
        "symbol": "EURUSD",
        "timeframes": ["M15", "H1"],
        "market_data": {
            "M15": sample_multi_timeframe_data["M15"],
            "H1": sample_multi_timeframe_data["H1"]
        }
    }
    
    result = analyzer.analyze(data)
    
    assert result.is_valid is False
    assert result.error is not None
    assert "insufficient timeframes" in str(result.error).lower()

def test_analyze_with_empty_timeframe_data(analyzer, sample_multi_timeframe_data):
    """Test the analyze method with empty timeframe data."""
    data = {
        "symbol": "EURUSD",
        "timeframes": ["M15", "H1", "H4"],
        "market_data": {
            "M15": {
                'timestamp': [],
                'open': [],
                'high': [],
                'low': [],
                'close': [],
                'volume': []
            },
            "H1": sample_multi_timeframe_data["H1"],
            "H4": sample_multi_timeframe_data["H4"]
        }
    }
    
    result = analyzer.analyze(data)
    
    assert result.is_valid is False
    assert result.error is not None
    assert "insufficient data" in str(result.error).lower()

def test_analyze_with_invalid_timeframe_data_format(analyzer, sample_multi_timeframe_data):
    """Test the analyze method with invalid timeframe data format."""
    data = {
        "symbol": "EURUSD",
        "timeframes": ["M15", "H1", "H4"],
        "market_data": {
            "M15": {
                # Missing required fields
                'timestamp': [datetime.now()],
                'close': [1.1000]
            },
            "H1": sample_multi_timeframe_data["H1"],
            "H4": sample_multi_timeframe_data["H4"]
        }
    }
    
    result = analyzer.analyze(data)
    
    assert result.is_valid is False
    assert result.error is not None
