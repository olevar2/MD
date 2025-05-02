"""
Unit tests for the ConfluenceAnalyzer component.

This module provides comprehensive tests for the ConfluenceAnalyzer, which identifies
confluence zones where multiple technical analysis factors align to provide stronger
trading signals.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from analysis_engine.analysis.confluence_analyzer import ConfluenceAnalyzer
from analysis_engine.core.errors import AnalysisError

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    # Create 100 data points with a clear trend and some support/resistance levels
    dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
    
    # Create a price series with an uptrend and some clear support/resistance levels
    close_prices = [1.1000]
    for i in range(1, 100):
        # Add trend
        trend = 0.0001 * i
        # Add some noise
        noise = np.random.normal(0, 0.0002)
        # Add some support/resistance levels
        if 20 <= i < 30 or 60 <= i < 70:
            # Create resistance level
            level = 1.1050 + trend
            close_prices.append(min(level, close_prices[-1] + trend + noise))
        elif 40 <= i < 50 or 80 <= i < 90:
            # Create support level
            level = 1.0950 + trend
            close_prices.append(max(level, close_prices[-1] + trend + noise))
        else:
            close_prices.append(close_prices[-1] + trend + noise)
    
    # Create high and low prices
    high_prices = [price + np.random.uniform(0.0005, 0.0015) for price in close_prices]
    low_prices = [price - np.random.uniform(0.0005, 0.0015) for price in close_prices]
    open_prices = [prev_close + np.random.normal(0, 0.0003) for prev_close in close_prices[:-1]]
    open_prices.insert(0, close_prices[0] - 0.0005)
    
    # Create volume data
    volumes = [int(1000 * (1 + np.random.uniform(-0.3, 0.3))) for _ in range(100)]
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # Convert to dictionary format expected by the analyzer
    market_data = {
        'timestamp': df['timestamp'].tolist(),
        'open': df['open'].tolist(),
        'high': df['high'].tolist(),
        'low': df['low'].tolist(),
        'close': df['close'].tolist(),
        'volume': df['volume'].tolist()
    }
    
    return market_data

@pytest.fixture
def analyzer():
    """Create a ConfluenceAnalyzer instance for testing."""
    return ConfluenceAnalyzer()

def test_confluence_analyzer_initialization():
    """Test that the ConfluenceAnalyzer initializes with correct default parameters."""
    analyzer = ConfluenceAnalyzer()
    
    assert analyzer.name == "confluence_analyzer"
    assert isinstance(analyzer.parameters, dict)
    assert "min_tools_for_confluence" in analyzer.parameters
    assert "effectiveness_threshold" in analyzer.parameters
    assert "sr_proximity_threshold" in analyzer.parameters
    assert "zone_width_pips" in analyzer.parameters

def test_analyze_with_valid_data(analyzer, sample_market_data):
    """Test the analyze method with valid market data."""
    data = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": sample_market_data,
        "market_regime": "TRENDING"
    }
    
    result = analyzer.analyze(data)
    
    # Verify result structure
    assert result.analyzer_name == "confluence_analyzer"
    assert result.is_valid is True
    assert result.error is None
    assert isinstance(result.result, dict)
    
    # Verify result content
    assert "timestamp" in result.result
    assert result.result["symbol"] == "EURUSD"
    assert "current_price" in result.result
    assert "confluence_zones" in result.result
    assert "market_regime" in result.result
    assert isinstance(result.result["confluence_zones"], list)
    
    # Verify metadata
    assert result.metadata["timeframe"] == "H1"
    assert "zone_count" in result.metadata
    assert "analysis_duration_ms" in result.metadata

def test_analyze_with_missing_data(analyzer):
    """Test the analyze method with missing required data."""
    # Missing market_data
    data = {
        "symbol": "EURUSD",
        "timeframe": "H1"
    }
    
    result = analyzer.analyze(data)
    
    assert result.is_valid is False
    assert result.error is not None
    assert "market_data" in str(result.error)
    
    # Missing symbol
    data = {
        "timeframe": "H1",
        "market_data": {}
    }
    
    result = analyzer.analyze(data)
    
    assert result.is_valid is False
    assert result.error is not None
    assert "symbol" in str(result.error)

def test_identify_confluence_zones(analyzer, sample_market_data):
    """Test the _identify_confluence_zones method."""
    # Get the current price from the sample data
    current_price = sample_market_data["close"][-1]
    
    zones = analyzer._identify_confluence_zones(sample_market_data, current_price)
    
    # Verify zones structure
    assert isinstance(zones, list)
    
    if zones:  # If zones were identified
        zone = zones[0]
        assert "price_level" in zone
        assert "zone_width" in zone
        assert "upper_bound" in zone
        assert "lower_bound" in zone
        assert "confluence_types" in zone
        assert "strength" in zone
        assert "strength_name" in zone
        assert "contributing_tools" in zone
        assert "direction" in zone
        
        # Verify zone values
        assert zone["price_level"] > 0
        assert zone["upper_bound"] > zone["price_level"]
        assert zone["lower_bound"] < zone["price_level"]
        assert isinstance(zone["confluence_types"], list)
        assert 0 <= zone["strength"] <= 1
        assert zone["direction"] in ["support", "resistance", "neutral"]

def test_determine_market_regime(analyzer, sample_market_data):
    """Test the _determine_market_regime method."""
    regime = analyzer._determine_market_regime(sample_market_data)
    
    assert regime in ["TRENDING", "RANGING", "VOLATILE"]

def test_calculate_tool_effectiveness(analyzer, sample_market_data):
    """Test the _calculate_tool_effectiveness method."""
    # Get the current price from the sample data
    current_price = sample_market_data["close"][-1]
    
    # First identify some zones
    zones = analyzer._identify_confluence_zones(sample_market_data, current_price)
    
    # Then calculate effectiveness
    effectiveness = analyzer._calculate_tool_effectiveness(sample_market_data, zones)
    
    assert isinstance(effectiveness, dict)
    
    # Check that effectiveness scores are between 0 and 1
    for tool, score in effectiveness.items():
        assert isinstance(tool, str)
        assert 0 <= score <= 1

def test_collect_all_levels(analyzer, sample_market_data):
    """Test the _collect_all_levels method."""
    # Get the current price from the sample data
    current_price = sample_market_data["close"][-1]
    
    # Convert to DataFrame for the method
    df = pd.DataFrame(sample_market_data)
    
    levels = analyzer._collect_all_levels(df, current_price)
    
    assert isinstance(levels, list)
    
    if levels:  # If levels were identified
        level = levels[0]
        assert "price" in level
        assert "type" in level
        assert "source" in level
        assert level["price"] > 0
        assert level["type"] in ["support", "resistance"]
        assert isinstance(level["source"], str)

def test_group_levels_into_zones(analyzer, sample_market_data):
    """Test the _group_levels_into_zones method."""
    # Get the current price from the sample data
    current_price = sample_market_data["close"][-1]
    
    # First collect all levels
    df = pd.DataFrame(sample_market_data)
    levels = analyzer._collect_all_levels(df, current_price)
    
    # Then group into zones
    zones = analyzer._group_levels_into_zones(levels, current_price, {})
    
    assert isinstance(zones, list)
    
    if zones:  # If zones were identified
        zone = zones[0]
        assert "price_level" in zone
        assert "zone_width" in zone
        assert "upper_bound" in zone
        assert "lower_bound" in zone
        assert "confluence_types" in zone
        assert "strength" in zone
        assert "strength_name" in zone
        assert "contributing_tools" in zone
        assert "direction" in zone

def test_analyze_with_different_market_regimes(analyzer, sample_market_data):
    """Test the analyze method with different market regimes."""
    regimes = ["TRENDING", "RANGING", "VOLATILE"]
    
    for regime in regimes:
        data = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "market_data": sample_market_data,
            "market_regime": regime
        }
        
        result = analyzer.analyze(data)
        
        assert result.is_valid is True
        assert result.result["market_regime"] == regime

def test_analyze_with_empty_market_data(analyzer):
    """Test the analyze method with empty market data."""
    data = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": {
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        }
    }
    
    result = analyzer.analyze(data)
    
    assert result.is_valid is False
    assert result.error is not None
    assert "insufficient data" in str(result.error).lower()

def test_analyze_with_invalid_market_data_format(analyzer):
    """Test the analyze method with invalid market data format."""
    data = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": {
            # Missing required fields
            "timestamp": [datetime.now()],
            "close": [1.1000]
        }
    }
    
    result = analyzer.analyze(data)
    
    assert result.is_valid is False
    assert result.error is not None
