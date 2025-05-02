# Import necessary modules and classes
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Conditional import for Analysis Engine components
try:
    # Attempt to import directly from the source if in the same project structure or installed
    # from ml_integration_service.analysis_engine_client import AnalysisEngineFeatureClient # REMOVED
    from analysis_engine.analysis.feature_extraction import (
        Feature as AEFeature, FeatureType as AEFeatureType, FeatureScope as AEFeatureScope,
        default_feature_extractor as analysis_engine_actual_extractor
    )
    from analysis_engine.registry.indicator_registry import indicator_registry
    from analysis_engine.indicators.base_indicator import BaseIndicatorResult
    # Import FeatureDefinition from ml_integration_service (still needed for legacy)
    from ml_integration_service.feature_extraction import FeatureDefinition, FeatureType as MLFeatureType
    # Import Legacy Extractor
    from ml_integration_service.feature_extraction import FeatureExtractor as LegacyFeatureExtractor

    ANALYSIS_ENGINE_AVAILABLE = True
    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Integration test setup warning: {e}")
    ANALYSIS_ENGINE_AVAILABLE = False
    LEGACY_COMPONENTS_AVAILABLE = False
    # Define dummy classes if imports fail, tests requiring them will be skipped
    # class AnalysisEngineFeatureClient: # REMOVED
    #     def __init__(self, *args, **kwargs): pass
    #     def extract_features(self, *args, **kwargs): return pd.DataFrame()
    class FeatureDefinition:
        def __init__(self, *args, **kwargs): pass
    class MLFeatureType: MOMENTUM = 'momentum'; TREND = 'trend'
    class LegacyFeatureExtractor:
        def extract_features(self, *args, **kwargs): return pd.DataFrame({'legacy_sma': [1,2,3]})

# --- Fixtures ---

@pytest.fixture(scope="module")
def sample_ohlcv_data_integration():
    """Provides sample OHLCV data for integration tests."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    data = {
        'open': np.random.rand(100) * 10 + 1.1,
        'high': lambda df: df['open'] + np.random.rand(100) * 0.05,
        'low': lambda df: df['open'] - np.random.rand(100) * 0.05,
        'close': lambda df: df['open'] + (np.random.rand(100) - 0.5) * 0.02,
        'volume': np.random.randint(100, 1000, size=100)
    }
    df = pd.DataFrame(data, index=dates)
    df['high'] = df.apply(lambda row: max(row['high'], row['close'], row['open']), axis=1)
    df['low'] = df.apply(lambda row: min(row['low'], row['close'], row['open']), axis=1)
    # Ensure close is within high/low bounds after calculation
    df['close'] = df.apply(lambda row: np.clip(row['close'], row['low'], row['high']), axis=1)
    return df

@pytest.fixture
def feature_definitions_integration():
    \"\"\"Provides a list of FeatureDefinition objects for tests.\"\"\"
    # Use FeatureDefinition from ml_integration_service
    return [
        FeatureDefinition(
            name="RSI_14",
            indicator_name="rsi", # Assuming AE uses lowercase names
            output_column="rsi", # Assuming AE outputs 'rsi' column
            feature_type=MLFeatureType.MOMENTUM,
            parameters={"window": 14, "column": "close"},
            description="14-period Relative Strength Index"
        ),
        FeatureDefinition(
            name="SMA_50",
            indicator_name="sma",
            output_column="sma",
            feature_type=MLFeatureType.TREND,
            parameters={"window": 50, "column": "close"},
            description="50-period Simple Moving Average"
        )
    ]

@pytest.fixture
def setup_analysis_engine_indicators():
    """Ensures required indicators are registered in the actual AE extractor."""
    if not ANALYSIS_ENGINE_AVAILABLE: yield; return # Skip if AE not available

    # Register indicators needed for the tests if not already present
    # This assumes direct access to the extractor instance
    if not analysis_engine_actual_extractor.has_indicator('sma'):
        analysis_engine_actual_extractor.register_indicator(SMAIndicator())
    if not analysis_engine_actual_extractor.has_indicator('rsi'):
        analysis_engine_actual_extractor.register_indicator(RSIIndicator())
    yield # Let the test run
    # Optional: Clean up / unregister indicators after test if needed

# --- Integration Tests ---

# REMOVED: test_ml_to_analysis_engine_success
# REMOVED: test_ml_to_analysis_engine_fallback_to_legacy
# REMOVED: test_ml_to_analysis_engine_ae_fails_no_fallback
# REMOVED: test_ml_to_analysis_engine_with_target_timeframe
# REMOVED: test_ml_to_analysis_engine_error_handling_invalid_feature

# --- Performance Benchmarking Considerations ---
# The file testing/performance_benchmark.py or testing/indicator_performance_benchmarks.py
# seem like appropriate places to add benchmarks comparing legacy vs AE client.
# Example structure (conceptual):
#
# import time
# from .ml_analysis_integration_test import sample_ohlcv_data_integration, feature_definitions_integration # Reuse fixtures
# # from ml_integration_service.analysis_engine_client import AnalysisEngineFeatureClient # REMOVED
# from ml_integration_service.feature_extraction import FeatureExtractor as LegacyFeatureExtractor
#
# def benchmark_legacy_extraction(benchmark):
#     data = sample_ohlcv_data_integration()
#     defs = feature_definitions_integration()
#     legacy_extractor = LegacyFeatureExtractor()
#     benchmark(legacy_extractor.extract_features, data=data, feature_definitions=defs)
#
# # @pytest.mark.skipif(not ANALYSIS_ENGINE_AVAILABLE, reason="Analysis Engine components not available") # REMOVED
# # def benchmark_analysis_engine_extraction(benchmark): # REMOVED
# #     data = sample_ohlcv_data_integration() # REMOVED
# #     defs = feature_definitions_integration() # REMOVED
# #     client = AnalysisEngineFeatureClient(use_fallback=False) # REMOVED
# #     benchmark(client.extract_features, data=data, feature_definitions=defs) # REMOVED

# --- Test Legacy Extractor Directly (Example) ---
# You might want tests that specifically target the legacy extractor now

@pytest.mark.skipif(not LEGACY_COMPONENTS_AVAILABLE, reason="Legacy components not available")
def test_legacy_extractor_direct(sample_ohlcv_data_integration, feature_definitions_integration):
    \"\"\"Test the legacy feature extractor directly.\"\"\"
    legacy_extractor = LegacyFeatureExtractor()
    features = legacy_extractor.extract_features(
        data=sample_ohlcv_data_integration,
        feature_definitions=feature_definitions_integration
    )
    assert isinstance(features, pd.DataFrame)
    assert "RSI_14" in features.columns
    assert "SMA_50" in features.columns
    assert features.index.equals(sample_ohlcv_data_integration.index)
    # Add more specific assertions about legacy behavior if needed
