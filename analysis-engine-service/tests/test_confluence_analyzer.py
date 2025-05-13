"""
Tests for the ConfluenceAnalyzer.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List

from analysis_engine.analysis.confluence.confluence_analyzer import ConfluenceAnalyzer

@pytest.fixture
def market_data() -> Dict[str, List]:
    """Create sample market data for testing"""
    now = datetime.now()
    timestamps = [
        (now - timedelta(minutes=i)).isoformat()
        for i in range(100)
    ]
    
    return {
        "open": [1.09000 + (i * 0.0001) for i in range(100)],
        "high": [1.09100 + (i * 0.0001) for i in range(100)],
        "low": [1.08900 + (i * 0.0001) for i in range(100)],
        "close": [1.09050 + (i * 0.0001) for i in range(100)],
        "volume": [1000 + (i * 10) for i in range(100)],
        "timestamp": timestamps
    }

@pytest.fixture
def analyzer() -> ConfluenceAnalyzer:
    """Create a ConfluenceAnalyzer instance for testing"""
    return ConfluenceAnalyzer()

def test_confluence_analyzer_initialization():
    """Test ConfluenceAnalyzer initialization"""
    analyzer = ConfluenceAnalyzer()
    
    assert analyzer.name == "confluence_analyzer"
    assert analyzer.parameters["min_tools_for_confluence"] == 2
    assert analyzer.parameters["effectiveness_threshold"] == 0.5
    assert analyzer.parameters["sr_proximity_threshold"] == 0.0015
    assert analyzer.parameters["zone_width_pips"] == 20

def test_confluence_analyzer_custom_parameters():
    """Test ConfluenceAnalyzer with custom parameters"""
    custom_params = {
        "min_tools_for_confluence": 3,
        "effectiveness_threshold": 0.7,
        "sr_proximity_threshold": 0.0020,
        "zone_width_pips": 30
    }
    
    analyzer = ConfluenceAnalyzer(parameters=custom_params)
    
    assert analyzer.parameters["min_tools_for_confluence"] == 3
    assert analyzer.parameters["effectiveness_threshold"] == 0.7
    assert analyzer.parameters["sr_proximity_threshold"] == 0.0020
    assert analyzer.parameters["zone_width_pips"] == 30

def test_confluence_analyzer_analysis(analyzer: ConfluenceAnalyzer, market_data: Dict[str, List]):
    """Test confluence analysis"""
    data = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": market_data
    }
    
    result = analyzer.analyze(data)
    
    assert result.analyzer_name == "confluence_analyzer"
    assert result.is_valid is True
    assert result.error is None
    
    # Check result structure
    assert "timestamp" in result.result
    assert result.result["symbol"] == "EURUSD"
    assert "current_price" in result.result
    assert "confluence_zones" in result.result
    assert "market_regime" in result.result
    assert "effective_tools" in result.result
    
    # Check metadata
    assert result.metadata["timeframe"] == "H1"
    assert "zone_count" in result.metadata

def test_confluence_analyzer_error_handling(analyzer: ConfluenceAnalyzer):
    """Test error handling in confluence analysis"""
    # Test with invalid data
    invalid_data = {
        "symbol": "EURUSD",
        "timeframe": "H1"
        # Missing market_data
    }
    
    result = analyzer.analyze(invalid_data)
    
    assert result.is_valid is False
    assert result.error is not None
    assert result.result == {}

def test_confluence_zone_identification(analyzer: ConfluenceAnalyzer, market_data: Dict[str, List]):
    """Test confluence zone identification"""
    zones = analyzer._identify_confluence_zones(market_data, 1.09200)
    
    assert isinstance(zones, list)
    if zones:  # If any zones were identified
        zone = zones[0]
        assert "price_level" in zone
        assert "zone_width" in zone
        assert "upper_bound" in zone
        assert "lower_bound" in zone
        assert "confluence_types" in zone
        assert "strength" in zone
        assert "strength_name" in zone
        assert "contributing_tools" in zone
        assert "timeframes" in zone
        assert "direction" in zone
        assert "expected_reaction" in zone

def test_market_regime_determination(analyzer: ConfluenceAnalyzer, market_data: Dict[str, List]):
    """Test market regime determination"""
    regime = analyzer._determine_market_regime(market_data)
    
    assert isinstance(regime, str)
    assert regime in ["TRENDING", "RANGING", "VOLATILE"]

def test_tool_effectiveness_calculation(analyzer: ConfluenceAnalyzer, market_data: Dict[str, List]):
    """Test tool effectiveness calculation"""
    zones = analyzer._identify_confluence_zones(market_data, 1.09200)
    effectiveness = analyzer._calculate_tool_effectiveness(market_data, zones)
    
    assert isinstance(effectiveness, dict)
    for tool, score in effectiveness.items():
        assert isinstance(tool, str)
        assert isinstance(score, float)
        assert 0 <= score <= 1

def test_strength_name_determination(analyzer: ConfluenceAnalyzer):
    """Test strength name determination"""
    assert analyzer._get_strength_name(4) == "VERY_STRONG"
    assert analyzer._get_strength_name(3) == "STRONG"
    assert analyzer._get_strength_name(2) == "MODERATE"
    assert analyzer._get_strength_name(1) == "WEAK"

def test_zone_direction_determination(analyzer: ConfluenceAnalyzer):
    """Test zone direction determination"""
    assert analyzer._determine_zone_direction(1.09300, 1.09200) == "bullish"
    assert analyzer._determine_zone_direction(1.09100, 1.09200) == "bearish"

def test_expected_reaction_determination(analyzer: ConfluenceAnalyzer, market_data: Dict[str, List]):
    """Test expected reaction determination"""
    reaction = analyzer._determine_expected_reaction(1.09200, 1.09100, market_data)
    
    assert isinstance(reaction, str)
    assert reaction in ["bounce", "break", "consolidate"] 