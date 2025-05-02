"""
Tests for the MultiTimeframeAnalyzer.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List

from analysis_engine.analysis.multi_timeframe.multi_timeframe_analyzer import MultiTimeframeAnalyzer

@pytest.fixture
def market_data() -> Dict[str, Dict[str, List]]:
    """Create sample market data for testing"""
    now = datetime.now()
    timestamps = [
        (now - timedelta(minutes=i)).isoformat()
        for i in range(100)
    ]
    
    base_data = {
        "open": [1.09000 + (i * 0.0001) for i in range(100)],
        "high": [1.09100 + (i * 0.0001) for i in range(100)],
        "low": [1.08900 + (i * 0.0001) for i in range(100)],
        "close": [1.09050 + (i * 0.0001) for i in range(100)],
        "volume": [1000 + (i * 10) for i in range(100)],
        "timestamp": timestamps
    }
    
    return {
        "M15": base_data,
        "H1": base_data
    }

@pytest.fixture
def analyzer() -> MultiTimeframeAnalyzer:
    """Create a MultiTimeframeAnalyzer instance for testing"""
    return MultiTimeframeAnalyzer()

def test_multi_timeframe_analyzer_initialization():
    """Test MultiTimeframeAnalyzer initialization"""
    analyzer = MultiTimeframeAnalyzer()
    
    assert analyzer.name == "multi_timeframe_analyzer"
    assert analyzer.parameters["correlation_threshold"] == 0.7
    assert analyzer.parameters["min_timeframes"] == 2
    assert analyzer.parameters["trend_strength_threshold"] == 0.6
    assert analyzer.parameters["volume_threshold"] == 1.5

def test_multi_timeframe_analyzer_custom_parameters():
    """Test MultiTimeframeAnalyzer with custom parameters"""
    custom_params = {
        "correlation_threshold": 0.8,
        "min_timeframes": 3,
        "trend_strength_threshold": 0.7,
        "volume_threshold": 2.0
    }
    
    analyzer = MultiTimeframeAnalyzer(parameters=custom_params)
    
    assert analyzer.parameters["correlation_threshold"] == 0.8
    assert analyzer.parameters["min_timeframes"] == 3
    assert analyzer.parameters["trend_strength_threshold"] == 0.7
    assert analyzer.parameters["volume_threshold"] == 2.0

def test_multi_timeframe_analyzer_analysis(analyzer: MultiTimeframeAnalyzer, market_data: Dict[str, Dict[str, List]]):
    """Test multi-timeframe analysis"""
    data = {
        "symbol": "EURUSD",
        "timeframes": ["M15", "H1"],
        "market_data": market_data
    }
    
    result = analyzer.analyze(data)
    
    assert result.analyzer_name == "multi_timeframe_analyzer"
    assert result.is_valid is True
    assert result.error is None
    
    # Check result structure
    assert "timestamp" in result.result
    assert result.result["symbol"] == "EURUSD"
    assert "timeframe_analysis" in result.result
    assert "correlation_matrix" in result.result
    assert "overall_assessment" in result.result
    
    # Check metadata
    assert result.metadata["timeframes"] == ["M15", "H1"]
    assert "analysis_count" in result.metadata

def test_multi_timeframe_analyzer_error_handling(analyzer: MultiTimeframeAnalyzer):
    """Test error handling in multi-timeframe analysis"""
    # Test with invalid data
    invalid_data = {
        "symbol": "EURUSD",
        "timeframes": ["M15", "H1"]
        # Missing market_data
    }
    
    result = analyzer.analyze(invalid_data)
    
    assert result.is_valid is False
    assert result.error is not None
    assert result.result == {}

def test_timeframe_analysis(analyzer: MultiTimeframeAnalyzer, market_data: Dict[str, Dict[str, List]]):
    """Test single timeframe analysis"""
    analysis = analyzer._analyze_timeframe("M15", market_data["M15"])
    
    assert "trend" in analysis
    assert "strength" in analysis
    assert "key_levels" in analysis
    assert "volume_profile" in analysis
    
    assert analysis["trend"] in ["bullish", "bearish"]
    assert 0 <= analysis["strength"] <= 1
    assert isinstance(analysis["key_levels"], list)
    assert isinstance(analysis["volume_profile"], dict)

def test_trend_calculation(analyzer: MultiTimeframeAnalyzer, market_data: Dict[str, Dict[str, List]]):
    """Test trend calculation"""
    trend, strength = analyzer._calculate_trend(market_data["M15"])
    
    assert trend in ["bullish", "bearish"]
    assert 0 <= strength <= 1

def test_key_levels_identification(analyzer: MultiTimeframeAnalyzer, market_data: Dict[str, Dict[str, List]]):
    """Test key levels identification"""
    levels = analyzer._identify_key_levels(market_data["M15"])
    
    assert isinstance(levels, list)
    if levels:  # If any levels were identified
        level = levels[0]
        assert "price" in level
        assert "type" in level
        assert "strength" in level
        assert level["type"] in ["support", "resistance"]
        assert 0 <= level["strength"] <= 1

def test_volume_profile_calculation(analyzer: MultiTimeframeAnalyzer, market_data: Dict[str, Dict[str, List]]):
    """Test volume profile calculation"""
    profile = analyzer._calculate_volume_profile(market_data["M15"])
    
    assert isinstance(profile, dict)
    assert "volume_trend" in profile
    assert "relative_volume" in profile
    assert "volume_climax" in profile
    assert profile["volume_trend"] in ["increasing", "decreasing", "stable"]
    assert isinstance(profile["relative_volume"], float)
    assert isinstance(profile["volume_climax"], bool)

def test_correlation_matrix_calculation(analyzer: MultiTimeframeAnalyzer, market_data: Dict[str, Dict[str, List]]):
    """Test correlation matrix calculation"""
    timeframe_analysis = {
        "M15": analyzer._analyze_timeframe("M15", market_data["M15"]),
        "H1": analyzer._analyze_timeframe("H1", market_data["H1"])
    }
    
    matrix = analyzer._calculate_correlation_matrix(timeframe_analysis)
    
    assert isinstance(matrix, dict)
    assert "M15_H1" in matrix
    assert 0 <= matrix["M15_H1"] <= 1

def test_overall_assessment(analyzer: MultiTimeframeAnalyzer, market_data: Dict[str, Dict[str, List]]):
    """Test overall assessment calculation"""
    timeframe_analysis = {
        "M15": analyzer._analyze_timeframe("M15", market_data["M15"]),
        "H1": analyzer._analyze_timeframe("H1", market_data["H1"])
    }
    correlation_matrix = analyzer._calculate_correlation_matrix(timeframe_analysis)
    
    assessment = analyzer._determine_overall_assessment(
        timeframe_analysis,
        correlation_matrix
    )
    
    assert isinstance(assessment, dict)
    assert "trend" in assessment
    assert "strength" in assessment
    assert "confidence" in assessment
    assert "key_levels" in assessment
    
    assert assessment["trend"] in ["bullish", "bearish"]
    assert 0 <= assessment["strength"] <= 1
    assert 0 <= assessment["confidence"] <= 1
    assert isinstance(assessment["key_levels"], list) 