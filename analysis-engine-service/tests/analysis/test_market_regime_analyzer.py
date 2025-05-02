"""
Unit tests for the MarketRegimeAnalyzer component.

This module provides comprehensive tests for the MarketRegimeAnalyzer, which identifies
market regimes (trending, ranging, volatile) to adapt analysis strategies accordingly.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from analysis_engine.analysis.advanced_ta.market_regime import MarketRegimeAnalyzer
from analysis_engine.core.errors import AnalysisError

@pytest.fixture
def trending_market_data():
    """Create sample trending market data for testing."""
    # Create 100 data points with a clear uptrend
    dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
    
    # Create a price series with a strong uptrend
    close_prices = [1.1000]
    for i in range(1, 100):
        # Add strong trend
        trend = 0.0002 * i
        # Add minimal noise
        noise = np.random.normal(0, 0.0001)
        close_prices.append(close_prices[-1] + trend + noise)
    
    # Create high and low prices
    high_prices = [price + np.random.uniform(0.0005, 0.0010) for price in close_prices]
    low_prices = [price - np.random.uniform(0.0005, 0.0010) for price in close_prices]
    open_prices = [prev_close + np.random.normal(0, 0.0002) for prev_close in close_prices[:-1]]
    open_prices.insert(0, close_prices[0] - 0.0005)
    
    # Create volume data with increasing trend
    volumes = [int(1000 * (1 + 0.01 * i + np.random.uniform(-0.2, 0.2))) for i in range(100)]
    
    # Convert to dictionary format expected by the analyzer
    market_data = {
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }
    
    return market_data

@pytest.fixture
def ranging_market_data():
    """Create sample ranging market data for testing."""
    # Create 100 data points with a sideways movement
    dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
    
    # Create a price series with a range
    base_price = 1.1000
    close_prices = []
    for i in range(100):
        # Add oscillation
        oscillation = 0.0030 * np.sin(i / 10)
        # Add some noise
        noise = np.random.normal(0, 0.0002)
        close_prices.append(base_price + oscillation + noise)
    
    # Create high and low prices
    high_prices = [price + np.random.uniform(0.0005, 0.0010) for price in close_prices]
    low_prices = [price - np.random.uniform(0.0005, 0.0010) for price in close_prices]
    open_prices = [prev_close + np.random.normal(0, 0.0002) for prev_close in close_prices[:-1]]
    open_prices.insert(0, close_prices[0] - 0.0005)
    
    # Create volume data with no trend
    volumes = [int(1000 * (1 + np.random.uniform(-0.3, 0.3))) for _ in range(100)]
    
    # Convert to dictionary format expected by the analyzer
    market_data = {
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }
    
    return market_data

@pytest.fixture
def volatile_market_data():
    """Create sample volatile market data for testing."""
    # Create 100 data points with high volatility
    dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
    
    # Create a price series with high volatility
    close_prices = [1.1000]
    for i in range(1, 100):
        # Add high volatility
        volatility = np.random.normal(0, 0.0020)
        close_prices.append(close_prices[-1] + volatility)
    
    # Create high and low prices with wide ranges
    high_prices = [price + np.random.uniform(0.0015, 0.0030) for price in close_prices]
    low_prices = [price - np.random.uniform(0.0015, 0.0030) for price in close_prices]
    open_prices = [prev_close + np.random.normal(0, 0.0010) for prev_close in close_prices[:-1]]
    open_prices.insert(0, close_prices[0] - 0.0015)
    
    # Create volume data with spikes
    base_volumes = [int(1000 * (1 + np.random.uniform(-0.3, 0.3))) for _ in range(100)]
    # Add volume spikes
    for i in range(10, 100, 10):
        base_volumes[i] = base_volumes[i] * 3
    
    # Convert to dictionary format expected by the analyzer
    market_data = {
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': base_volumes
    }
    
    return market_data

@pytest.fixture
def analyzer():
    """Create a MarketRegimeAnalyzer instance for testing."""
    return MarketRegimeAnalyzer()

def test_market_regime_analyzer_initialization():
    """Test that the MarketRegimeAnalyzer initializes with correct default parameters."""
    analyzer = MarketRegimeAnalyzer()
    
    assert analyzer.name == "market_regime_analyzer"
    assert isinstance(analyzer.parameters, dict)
    assert "atr_period" in analyzer.parameters
    assert "adx_period" in analyzer.parameters
    assert "ma_fast_period" in analyzer.parameters
    assert "ma_slow_period" in analyzer.parameters
    assert "volatility_threshold_low" in analyzer.parameters
    assert "volatility_threshold_high" in analyzer.parameters
    assert "adx_threshold_trend" in analyzer.parameters
    assert "ma_slope_threshold" in analyzer.parameters

def test_analyze_trending_market(analyzer, trending_market_data):
    """Test the analyze method with trending market data."""
    data = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": trending_market_data
    }
    
    result = analyzer.analyze(data)
    
    # Verify result structure
    assert result.analyzer_name == "market_regime_analyzer"
    assert result.is_valid is True
    assert result.error is None
    assert isinstance(result.result, dict)
    
    # Verify result content
    assert "regime" in result.result
    assert "direction" in result.result
    assert "volatility" in result.result
    assert "strength" in result.result
    assert "metrics" in result.result
    
    # Verify trending regime
    assert result.result["regime"] == "TRENDING"
    assert result.result["direction"] in ["BULLISH", "BEARISH"]
    assert result.result["volatility"] in ["LOW", "MEDIUM", "HIGH"]
    assert 0 <= result.result["strength"] <= 1
    
    # Verify metrics
    assert "adx" in result.result["metrics"]
    assert "atr" in result.result["metrics"]
    assert "ma_slope" in result.result["metrics"]

def test_analyze_ranging_market(analyzer, ranging_market_data):
    """Test the analyze method with ranging market data."""
    data = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": ranging_market_data
    }
    
    result = analyzer.analyze(data)
    
    # Verify result structure
    assert result.is_valid is True
    
    # Verify ranging regime
    assert result.result["regime"] == "RANGING"
    assert result.result["direction"] in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert result.result["volatility"] in ["LOW", "MEDIUM", "HIGH"]

def test_analyze_volatile_market(analyzer, volatile_market_data):
    """Test the analyze method with volatile market data."""
    data = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": volatile_market_data
    }
    
    result = analyzer.analyze(data)
    
    # Verify result structure
    assert result.is_valid is True
    
    # Verify volatile regime
    assert result.result["regime"] == "VOLATILE"
    assert result.result["direction"] in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert result.result["volatility"] in ["MEDIUM", "HIGH"]

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

def test_calculate_atr(analyzer, volatile_market_data):
    """Test the _calculate_atr method."""
    # Convert to DataFrame
    df = pd.DataFrame(volatile_market_data)
    
    atr = analyzer._calculate_atr(df, analyzer.parameters["atr_period"])
    
    assert isinstance(atr, float)
    assert atr > 0

def test_calculate_adx(analyzer, trending_market_data):
    """Test the _calculate_adx method."""
    # Convert to DataFrame
    df = pd.DataFrame(trending_market_data)
    
    adx = analyzer._calculate_adx(df, analyzer.parameters["adx_period"])
    
    assert isinstance(adx, float)
    assert 0 <= adx <= 100

def test_calculate_ma_slope(analyzer, trending_market_data):
    """Test the _calculate_ma_slope method."""
    # Convert to DataFrame
    df = pd.DataFrame(trending_market_data)
    
    slope = analyzer._calculate_ma_slope(df, analyzer.parameters["ma_fast_period"])
    
    assert isinstance(slope, float)

def test_determine_regime(analyzer):
    """Test the _determine_regime method."""
    # Test trending regime
    regime = analyzer._determine_regime(30, 0.0005, 0.002)
    assert regime == "TRENDING"
    
    # Test ranging regime
    regime = analyzer._determine_regime(15, 0.0001, 0.001)
    assert regime == "RANGING"
    
    # Test volatile regime
    regime = analyzer._determine_regime(15, 0.0001, 0.005)
    assert regime == "VOLATILE"

def test_determine_direction(analyzer):
    """Test the _determine_direction method."""
    # Test bullish direction
    direction = analyzer._determine_direction(0.0005, 10, 5)
    assert direction == "BULLISH"
    
    # Test bearish direction
    direction = analyzer._determine_direction(-0.0005, -10, -5)
    assert direction == "BEARISH"
    
    # Test neutral direction
    direction = analyzer._determine_direction(0.0001, 2, 1)
    assert direction == "NEUTRAL"

def test_determine_volatility(analyzer):
    """Test the _determine_volatility method."""
    # Test low volatility
    volatility = analyzer._determine_volatility(0.001, 0.002, 0.005)
    assert volatility == "LOW"
    
    # Test medium volatility
    volatility = analyzer._determine_volatility(0.003, 0.002, 0.005)
    assert volatility == "MEDIUM"
    
    # Test high volatility
    volatility = analyzer._determine_volatility(0.006, 0.002, 0.005)
    assert volatility == "HIGH"

def test_calculate_regime_strength(analyzer):
    """Test the _calculate_regime_strength method."""
    # Test trending strength
    strength = analyzer._calculate_regime_strength("TRENDING", 40, 0.0008, 0.002)
    assert 0 <= strength <= 1
    
    # Test ranging strength
    strength = analyzer._calculate_regime_strength("RANGING", 15, 0.0001, 0.001)
    assert 0 <= strength <= 1
    
    # Test volatile strength
    strength = analyzer._calculate_regime_strength("VOLATILE", 15, 0.0001, 0.006)
    assert 0 <= strength <= 1

def test_analyze_with_custom_parameters(trending_market_data):
    """Test the analyze method with custom parameters."""
    # Create analyzer with custom parameters
    custom_params = {
        "atr_period": 10,
        "adx_period": 10,
        "ma_fast_period": 10,
        "ma_slow_period": 30,
        "volatility_threshold_low": 0.3,
        "volatility_threshold_high": 1.0,
        "adx_threshold_trend": 20,
        "ma_slope_threshold": 0.0002
    }
    analyzer = MarketRegimeAnalyzer(parameters=custom_params)
    
    data = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": trending_market_data
    }
    
    result = analyzer.analyze(data)
    
    assert result.is_valid is True
    assert "regime" in result.result

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
