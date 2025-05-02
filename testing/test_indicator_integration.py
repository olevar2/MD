"""
Indicator Pipeline Integration Tests

This module provides comprehensive integration tests for the entire indicator pipeline,
ensuring that all components work together seamlessly. It validates the following:

1. Cross-component integration between different indicator types
2. Multi-timeframe analysis with indicator combinations
3. End-to-end testing of the indicator pipeline with realistic data
4. Integration with the adaptive layer and ML components

These tests complete Phase 7 of the implementation plan by validating that all
the components implemented in previous phases work together correctly.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
import pytest

# Import core components
from core_foundations.utils.logger import get_logger
from core_foundations.config.settings import get_settings

# Import feature store components for indicators
from feature_store_service.feature_store_service.indicators.advanced_moving_averages import (
    TripleExponentialMovingAverage, DoubleExponentialMovingAverage, HullMovingAverage,
    KaufmanAdaptiveMovingAverage, ZeroLagExponentialMovingAverage, ArnaudLegouxMovingAverage
)
from feature_store_service.feature_store_service.indicators.advanced_oscillators import (
    AwesomeOscillator, AcceleratorOscillator, UltimateOscillator, DeMarker,
    TRIX, KnowSureThing, ElderForceIndex, RelativeVigorIndex,
    FisherTransform, CoppockCurve, ChandeMomentumOscillator
)
from feature_store_service.feature_store_service.indicators.volume_analysis import (
    VolumeProfile, VWAPBands, MarketFacilitationIndex, VolumeZoneOscillator,
    EaseOfMovement, NVIAndPVI, DemandIndex, RelativeVolume, VolumeDelta
)

# Import analysis engine components
from analysis_engine.analysis.advanced_ta.chart_patterns import (
    ChartPatternRecognizer, HarmonicPatternFinder, CandlestickPatterns
)
from analysis_engine.analysis.advanced_ta.gann_tools import (
    GannAngles, GannSquare9, GannFan
)
from analysis_engine.analysis.advanced_ta.fractal_indicators import (
    FractalIndicator, AlligatorIndicator, ElliottWaveAnalyzer, HurstExponent
)

# Import multi-timeframe components
from analysis_engine.analysis.multi_timeframe import (
    MultiTimeframeIndicator, TimeframeComparison, TimeframeConfluenceScanner
)

# Import intelligent indicator selection components
from analysis_engine.adaptive_layer.indicator_selection import (
    MarketRegimeClassifier, IndicatorPerformanceMetrics, IndicatorSelectionEngine
)

# Import ML integration components
from ml_integration_service.ml_integration_service.feature_extraction import (
    FeatureExtractor, FeatureSelector
)

logger = get_logger(__name__)
settings = get_settings()


class TestIndicatorPipelineIntegration:
    """Integration tests for the complete indicator pipeline."""
    
    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Generate sample market data for testing."""
        # Create a realistic OHLCV dataset with 1000 rows
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
        
        # Start with a base price and add random walks with some trend
        base_price = 100.0
        volatility = 0.5
        drift = 0.01
        
        # Generate random returns with drift
        returns = np.random.normal(drift, volatility, size=len(dates)) / 100
        
        # Calculate prices using cumulative returns
        price_path = base_price * (1 + np.cumsum(returns))
        
        # Generate OHLCV data
        high_vals = price_path * (1 + np.random.uniform(0, 0.01, size=len(dates)))
        low_vals = price_path * (1 - np.random.uniform(0, 0.01, size=len(dates)))
        
        # Make sure high is always >= open/close, and low is always <= open/close
        close_vals = price_path
        open_vals = price_path * (1 + np.random.normal(0, 0.003, size=len(dates)))
        
        # Ensure proper OHLC relationships
        for i in range(len(dates)):
            high_vals[i] = max(high_vals[i], open_vals[i], close_vals[i])
            low_vals[i] = min(low_vals[i], open_vals[i], close_vals[i])
        
        # Generate volume data - higher volume on bigger price moves
        volume = np.abs(np.diff(np.append(0, price_path))) * 1000000 + 100000
        volume = volume * np.random.uniform(0.8, 1.2, size=len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            'datetime': dates,
            'open': open_vals,
            'high': high_vals,
            'low': low_vals,
            'close': close_vals,
            'volume': volume
        })
        
        df.set_index('datetime', inplace=True)
        return df
    
    @pytest.fixture
    def multi_timeframe_data(self, sample_market_data) -> Dict[str, pd.DataFrame]:
        """Generate multi-timeframe data from the sample data."""
        # Create 1h, 4h, and 1d timeframes from the 1h data
        data_1h = sample_market_data.copy()
        
        # Resample to 4h
        data_4h = data_1h.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Resample to 1d
        data_1d = data_1h.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return {
            '1h': data_1h,
            '4h': data_4h,
            '1d': data_1d
        }
    
    def test_advanced_moving_averages_integration(self, sample_market_data):
        """Test that all advanced moving averages work together coherently."""
        # Create and calculate multiple moving averages
        tema = TripleExponentialMovingAverage(period=20)
        dema = DoubleExponentialMovingAverage(period=20)
        hull = HullMovingAverage(period=20)
        kama = KaufmanAdaptiveMovingAverage(period=20, fast_period=2, slow_period=30)
        zlema = ZeroLagExponentialMovingAverage(period=20)
        alma = ArnaudLegouxMovingAverage(period=20, sigma=6.0, offset=0.85)
        
        # Calculate all indicators on the same dataset
        data = sample_market_data.copy()
        
        # Apply each indicator
        tema_result = tema.calculate(data)
        dema_result = dema.calculate(data)
        hull_result = hull.calculate(data)
        kama_result = kama.calculate(data)
        zlema_result = zlema.calculate(data)
        alma_result = alma.calculate(data)
        
        # Verify results have the expected columns
        assert 'tema_20' in tema_result.columns
        assert 'dema_20' in dema_result.columns
        assert 'hull_20' in hull_result.columns
        assert 'kama_20_2_30' in kama_result.columns
        assert 'zlema_20' in zlema_result.columns
        assert 'alma_20' in alma_result.columns
        
        # Check for NaN values after the initial window
        warmup_period = 30  # Allow some warmup periods
        for result in [tema_result, dema_result, hull_result, kama_result, zlema_result, alma_result]:
            indicator_column = result.columns[-1]
            assert not result.iloc[warmup_period:][indicator_column].isna().any()
        
        # Verify relative positions of indicators (slower vs. faster)
        # In trending periods, Hull MA should react faster than DEMA and TEMA
        # Get a sample trending period from the latter part of the data
        trending_period = data.iloc[-200:-100]
        trend_direction = 1 if trending_period['close'].iloc[-1] > trending_period['close'].iloc[0] else -1
        
        if trend_direction > 0:
            # In an uptrend, faster MAs should generally be above slower ones
            assert hull_result['hull_20'].iloc[-150] >= dema_result['dema_20'].iloc[-150]
        else:
            # In a downtrend, faster MAs should generally be below slower ones
            assert hull_result['hull_20'].iloc[-150] <= dema_result['dema_20'].iloc[-150]
            
        # Test crossover detection between fast and slow indicators
        combined = pd.DataFrame({
            'close': data['close'],
            'fast_ma': hull_result['hull_20'],
            'slow_ma': tema_result['tema_20']
        })
        combined['crossover'] = ((combined['fast_ma'] > combined['slow_ma']) & 
                                 (combined['fast_ma'].shift(1) <= combined['slow_ma'].shift(1)))
        combined['crossunder'] = ((combined['fast_ma'] < combined['slow_ma']) & 
                                  (combined['fast_ma'].shift(1) >= combined['slow_ma'].shift(1)))
        
        # Ensure we detect some crossovers in the test data
        assert combined['crossover'].sum() > 0
        assert combined['crossunder'].sum() > 0
        
        logger.info(f"Advanced Moving Averages integration test passed with "
                    f"{combined['crossover'].sum()} crossovers and "
                    f"{combined['crossunder'].sum()} crossunders detected")
    
    def test_oscillators_integration(self, sample_market_data):
        """Test that advanced oscillators work together and provide coherent signals."""
        # Create the oscillators
        awesome_osc = AwesomeOscillator()
        acc_osc = AcceleratorOscillator()
        ult_osc = UltimateOscillator()
        demarker = DeMarker()
        trix_ind = TRIX()
        kst = KnowSureThing()
        rvi = RelativeVigorIndex()
        
        # Calculate all indicators
        data = sample_market_data.copy()
        
        # Apply each indicator
        ao_result = awesome_osc.calculate(data)
        acc_result = acc_osc.calculate(data)
        ult_result = ult_osc.calculate(data)
        demarker_result = demarker.calculate(data)
        trix_result = trix_ind.calculate(data)
        kst_result = kst.calculate(data)
        rvi_result = rvi.calculate(data)
        
        # Verify results have the expected columns
        assert 'ao' in ao_result.columns
        assert 'acc' in acc_result.columns
        assert 'ult_osc' in ult_result.columns
        assert 'demarker_14' in demarker_result.columns
        assert 'trix_15' in trix_result.columns
        assert 'kst' in kst_result.columns
        assert 'rvi_10' in rvi_result.columns
        
        # Combine results for comparison
        combined = pd.DataFrame({
            'close': data['close'],
            'ao': ao_result['ao'],
            'acc': acc_result['acc'],
            'ult_osc': ult_result['ult_osc'],
            'demarker': demarker_result['demarker_14'],
            'trix': trix_result['trix_15'],
            'kst': kst_result['kst'],
            'rvi': rvi_result['rvi_10']
        })
        
        # Check for correlation between oscillators during trending periods
        warmup_period = 50
        correlations = combined.iloc[warmup_period:].corr()
        
        # In a well-designed oscillator set, we expect significant correlation between certain pairs
        assert correlations.loc['ao', 'acc'] > 0.5, "AO and ACC should be positively correlated"
        assert abs(correlations.loc['ult_osc', 'demarker']) > 0.3, "UO and DeMarker should show correlation"
        
        # Check for overbought/oversold conditions
        combined['rvi_overbought'] = combined['rvi'] > 0.7
        combined['rvi_oversold'] = combined['rvi'] < 0.3
        combined['demarker_overbought'] = combined['demarker'] > 0.7
        combined['demarker_oversold'] = combined['demarker'] < 0.3
        
        # Check agreement between oscillators for overbought/oversold conditions
        combined['oscillator_agreement_overbought'] = (
            combined['rvi_overbought'] & combined['demarker_overbought']
        )
        combined['oscillator_agreement_oversold'] = (
            combined['rvi_oversold'] & combined['demarker_oversold']
        )
        
        # Verify that we have some periods of agreement
        assert combined['oscillator_agreement_overbought'].sum() > 0
        assert combined['oscillator_agreement_oversold'].sum() > 0
        
        logger.info(f"Advanced Oscillators integration test passed with "
                    f"{combined['oscillator_agreement_overbought'].sum()} agreed overbought signals and "
                    f"{combined['oscillator_agreement_oversold'].sum()} agreed oversold signals")
    
    def test_volume_indicators_integration(self, sample_market_data):
        """Test that volume indicators work together and provide coherent signals."""
        # Create volume indicators
        volume_zone = VolumeZoneOscillator()
        eom = EaseOfMovement()
        nvi_pvi = NVIAndPVI()
        relative_volume = RelativeVolume()
        volume_delta = VolumeDelta()
        
        # Calculate all indicators
        data = sample_market_data.copy()
        
        # Apply each indicator
        vzo_result = volume_zone.calculate(data)
        eom_result = eom.calculate(data)
        nvi_pvi_result = nvi_pvi.calculate(data)
        rel_vol_result = relative_volume.calculate(data)
        vol_delta_result = volume_delta.calculate(data)
        
        # Verify results have the expected columns
        assert 'vzo_14' in vzo_result.columns
        assert 'eom_14' in eom_result.columns
        assert 'nvi' in nvi_pvi_result.columns
        assert 'pvi' in nvi_pvi_result.columns
        assert 'relative_volume_10' in rel_vol_result.columns
        assert 'volume_delta' in vol_delta_result.columns
        
        # Combine results for comparison
        combined = pd.DataFrame({
            'close': data['close'],
            'volume': data['volume'],
            'vzo': vzo_result['vzo_14'],
            'eom': eom_result['eom_14'],
            'nvi': nvi_pvi_result['nvi'],
            'pvi': nvi_pvi_result['pvi'],
            'rel_volume': rel_vol_result['relative_volume_10'],
            'vol_delta': vol_delta_result['volume_delta']
        })
        
        # Check for volume spikes and corresponding price movements
        warmup_period = 30
        combined = combined.iloc[warmup_period:]
        combined['volume_spike'] = combined['rel_volume'] > 2.0
        
        # Check if price moves following volume spikes (either up or down)
        combined['next_day_abs_move'] = combined['close'].pct_change().shift(-1).abs()
        volume_spike_moves = combined.loc[combined['volume_spike'], 'next_day_abs_move']
        normal_moves = combined.loc[~combined['volume_spike'], 'next_day_abs_move']
        
        if not volume_spike_moves.empty and not normal_moves.empty:
            # Check if price moves tend to be larger after volume spikes
            assert volume_spike_moves.mean() > normal_moves.mean(), "Volume spikes should lead to larger price moves"
        
        # Check correlation between volume delta and price changes
        price_changes = combined['close'].pct_change()
        volume_delta_corr = price_changes.corr(combined['vol_delta'])
        
        # There should be some non-trivial correlation between volume delta and price changes
        assert abs(volume_delta_corr) > 0.1, "Volume delta should show some correlation with price changes"
        
        logger.info(f"Volume indicators integration test passed with volume spike correlation: {volume_delta_corr:.3f}")
    
    def test_chart_pattern_recognition_integration(self, sample_market_data):
        """Test that chart pattern recognition components work together."""
        # Create chart pattern components
        pattern_recognizer = ChartPatternRecognizer()
        harmonic_finder = HarmonicPatternFinder()
        candlestick_patterns = CandlestickPatterns()
        
        # Apply pattern recognition
        data = sample_market_data.copy()
        
        # Detect chart patterns (just testing that they run without errors)
        chart_patterns = pattern_recognizer.find_patterns(data)
        harmonic_patterns = harmonic_finder.find_patterns(data)
        candle_patterns = candlestick_patterns.find_patterns(data)
        
        # Simple validation of results
        assert isinstance(chart_patterns, dict)
        assert isinstance(harmonic_patterns, dict)
        assert isinstance(candle_patterns, dict)
        
        # Count detected patterns
        chart_pattern_count = sum(len(patterns) for patterns in chart_patterns.values())
        harmonic_pattern_count = sum(len(patterns) for patterns in harmonic_patterns.values())
        candle_pattern_count = sum(len(patterns) for patterns in candle_patterns.values())
        
        logger.info(f"Chart pattern recognition test passed with {chart_pattern_count} chart patterns, "
                    f"{harmonic_pattern_count} harmonic patterns, and {candle_pattern_count} candlestick patterns")
        
        # Convert pattern results to signals
        signals = []
        
        # Process chart patterns
        for pattern_type, instances in chart_patterns.items():
            for instance in instances:
                signals.append({
                    'type': 'chart',
                    'pattern': pattern_type,
                    'start_idx': instance.get('start_idx'),
                    'end_idx': instance.get('end_idx'),
                    'direction': instance.get('direction', 'neutral'),
                    'strength': instance.get('strength', 0.5),
                })
        
        # Process harmonic patterns
        for pattern_type, instances in harmonic_patterns.items():
            for instance in instances:
                signals.append({
                    'type': 'harmonic',
                    'pattern': pattern_type,
                    'target_idx': instance.get('target_idx'),
                    'direction': instance.get('direction', 'neutral'),
                    'completion': instance.get('completion_percentage', 100),
                })
        
        # Process candlestick patterns
        for pattern_type, instances in candle_patterns.items():
            for instance in instances:
                signals.append({
                    'type': 'candlestick',
                    'pattern': pattern_type,
                    'idx': instance.get('idx'),
                    'direction': instance.get('direction', 'neutral'),
                })
        
        # Verify that we have some signals
        assert len(signals) > 0, "Pattern recognition should produce some signals"
    
    def test_gann_tools_integration(self, sample_market_data):
        """Test that Gann tools work correctly together."""
        # Create Gann tools
        gann_angles = GannAngles()
        gann_square = GannSquare9()
        gann_fan = GannFan()
        
        # Apply Gann tools
        data = sample_market_data.copy()
        
        # Find a significant pivot for Gann analysis
        # For testing, we'll use a simple approach to find a major high or low
        rolling_max = data['high'].rolling(window=50).max()
        rolling_min = data['low'].rolling(window=50).min()
        data['is_pivot_high'] = (data['high'] == rolling_max)
        data['is_pivot_low'] = (data['low'] == rolling_min)
        
        # Find the most recent significant pivot
        pivot_highs = data[data['is_pivot_high']].index.tolist()
        pivot_lows = data[data['is_pivot_low']].index.tolist()
        
        all_pivots = pivot_highs + pivot_lows
        if all_pivots:
            # Sort pivots by date
            all_pivots.sort()
            # Take a pivot from the middle of the data for testing
            pivot_idx = len(all_pivots) // 2
            pivot_date = all_pivots[pivot_idx]
            pivot_price = data.loc[pivot_date, 'high'] if pivot_date in pivot_highs else data.loc[pivot_date, 'low']
            pivot_type = 'high' if pivot_date in pivot_highs else 'low'
            
            # Calculate Gann angles from the pivot
            angles_result = gann_angles.calculate_angles(
                pivot_date=pivot_date,
                pivot_price=pivot_price,
                pivot_type=pivot_type,
                price_data=data
            )
            
            # Calculate Square of 9 levels
            square9_result = gann_square.calculate_square9_levels(
                base_price=pivot_price,
                levels=5
            )
            
            # Calculate Gann fan
            fan_result = gann_fan.calculate_fan(
                pivot_date=pivot_date,
                pivot_price=pivot_price,
                pivot_type=pivot_type,
                price_data=data
            )
            
            # Verify results
            assert isinstance(angles_result, dict) and 'angles' in angles_result
            assert isinstance(square9_result, dict) and 'levels' in square9_result
            assert isinstance(fan_result, dict) and 'fan_lines' in fan_result
            
            # Check if some key prices from Square of 9 correspond to support/resistance in the data
            if pivot_type == 'high':
                # After a high, check if price found support at Square of 9 levels
                square9_levels = [level['price'] for level in square9_result['levels']]
                subsequent_data = data.loc[data.index > pivot_date]
                
                # Check if price came within 0.5% of any Square of 9 level
                min_distances = []
                for _, row in subsequent_data.iterrows():
                    low_price = row['low']
                    distances = [abs(low_price - level) / level * 100 for level in square9_levels]
                    min_distances.append(min(distances) if distances else float('inf'))
                
                # Check if price approached Square of 9 levels at any point
                assert any(dist < 0.5 for dist in min_distances), "Price should approach Square of 9 levels"
            
            logger.info(f"Gann tools integration test passed with {len(angles_result['angles'])} Gann angles and "
                        f"{len(square9_result['levels'])} Square of 9 levels")
        else:
            pytest.skip("No significant pivots found in the test data")
    
    def test_fractal_indicators_integration(self, sample_market_data):
        """Test that fractal geometry indicators work together."""
        # Create fractal indicators
        fractal_indicator = FractalIndicator()
        alligator = AlligatorIndicator()
        elliott_wave = ElliottWaveAnalyzer()
        hurst = HurstExponent()
        
        # Apply fractal indicators
        data = sample_market_data.copy()
        
        # Calculate fractal patterns
        fractal_result = fractal_indicator.find_fractals(data)
        alligator_result = alligator.calculate(data)
        elliott_result = elliott_wave.analyze(data)
        hurst_result = hurst.calculate(data, min_window=10, max_window=100)
        
        # Verify results
        assert 'bullish_fractals' in fractal_result
        assert 'bearish_fractals' in fractal_result
        assert 'jaw' in alligator_result.columns
        assert 'teeth' in alligator_result.columns
        assert 'lips' in alligator_result.columns
        assert isinstance(elliott_result, dict)
        assert 'hurst_exponent' in hurst_result
        
        # Check if Hurst exponent is in a reasonable range
        assert 0 < hurst_result['hurst_exponent'] < 1, "Hurst exponent should be between 0 and 1"
        
        # Combine fractal indicators with alligator for analysis
        combined = alligator_result.copy()
        
        # Add bullish and bearish fractals to the dataframe
        combined['bullish_fractal'] = False
        combined['bearish_fractal'] = False
        
        for idx in fractal_result['bullish_fractals']:
            if idx in combined.index:
                combined.at[idx, 'bullish_fractal'] = True
        
        for idx in fractal_result['bearish_fractals']:
            if idx in combined.index:
                combined.at[idx, 'bearish_fractal'] = True
        
        # Check alligator state
        combined['alligator_eating'] = (
            (combined['jaw'] < combined['teeth']) & 
            (combined['teeth'] < combined['lips'])
        )
        combined['alligator_sleeping'] = (
            abs(combined['jaw'] - combined['teeth']) < 
            0.1 * combined['close']
        )
        
        # Check for fractal signals above/below alligator mouth
        combined['valid_bullish_signal'] = (
            combined['bullish_fractal'] & 
            (combined['low'] > combined['jaw']) &
            (combined['alligator_eating'])
        )
        
        combined['valid_bearish_signal'] = (
            combined['bearish_fractal'] & 
            (combined['high'] < combined['jaw']) &
            (combined['alligator_eating'])
        )
        
        # Verify that we have some valid signals
        assert combined['valid_bullish_signal'].sum() + combined['valid_bearish_signal'].sum() > 0, \
            "Should have some valid fractal signals with alligator confirmation"
        
        logger.info(f"Fractal indicators integration test passed with "
                    f"{combined['valid_bullish_signal'].sum()} bullish signals and "
                    f"{combined['valid_bearish_signal'].sum()} bearish signals")
        
        # Check if Hurst exponent aligns with Elliott wave analysis
        if 'wave_degree' in elliott_result:
            hurst_value = hurst_result['hurst_exponent']
            wave_degree = elliott_result['wave_degree']
            
            # Higher Hurst values (> 0.5) indicate trending markets (suitable for impulse waves)
            # Lower Hurst values (< 0.5) indicate mean-reverting markets (suitable for corrective waves)
            if hurst_value > 0.5:
                logger.info(f"Hurst exponent {hurst_value:.2f} indicates trending market")
            else:
                logger.info(f"Hurst exponent {hurst_value:.2f} indicates mean-reverting market")
    
    def test_multi_timeframe_integration(self, multi_timeframe_data):
        """Test multi-timeframe analysis integration."""
        # Create some indicators for multi-timeframe analysis
        tema = TripleExponentialMovingAverage(period=20)
        rsi = RelativeVigorIndex(period=14)
        macd_lines = {'fast': 12, 'slow': 26, 'signal': 9}
        
        # Get the data
        tf_data = multi_timeframe_data
        
        # Create multi-timeframe wrapper
        mtf_tema = MultiTimeframeIndicator(
            indicator=tema,
            timeframes=['1h', '4h', '1d']
        )
        
        mtf_rsi = MultiTimeframeIndicator(
            indicator=rsi,
            timeframes=['1h', '4h', '1d']
        )
        
        # Calculate indicators on multiple timeframes
        mtf_tema_result = mtf_tema.calculate(tf_data)
        mtf_rsi_result = mtf_rsi.calculate(tf_data)
        
        # Create timeframe comparison tool
        tf_comparison = TimeframeComparison()
        
        # Analyze trend alignment across timeframes
        trend_alignment = tf_comparison.analyze_trend_alignment(
            mtf_tema_result,
            timeframes=['1h', '4h', '1d'],
            indicator_column='tema_20'
        )
        
        # Analyze momentum alignment across timeframes
        momentum_alignment = tf_comparison.analyze_momentum_alignment(
            mtf_rsi_result,
            timeframes=['1h', '4h', '1d'],
            indicator_column='rvi_14',
            overbought_threshold=0.7,
            oversold_threshold=0.3
        )
        
        # Create timeframe confluence scanner
        confluence_scanner = TimeframeConfluenceScanner(
            timeframes=['1h', '4h', '1d'],
            indicators=['tema_20', 'rvi_14']
        )
        
        # Combine the results
        combined_data = {}
        for tf in ['1h', '4h', '1d']:
            combined_data[tf] = pd.DataFrame({
                'close': tf_data[tf]['close'],
                'tema_20': mtf_tema_result[tf]['tema_20'],
                'rvi_14': mtf_rsi_result[tf]['rvi_14']
            })
        
        # Scan for confluence signals
        confluence_signals = confluence_scanner.scan_for_confluence(combined_data)
        
        # Verify results
        assert 'aligned_timeframes' in trend_alignment
        assert 'trend_direction' in trend_alignment
        assert 'aligned_timeframes' in momentum_alignment
        assert 'momentum_state' in momentum_alignment
        assert 'bullish_signals' in confluence_signals
        assert 'bearish_signals' in confluence_signals
        
        # There should be some confluence signals
        all_signals = len(confluence_signals['bullish_signals']) + len(confluence_signals['bearish_signals'])
        assert all_signals > 0, "Multi-timeframe analysis should produce some confluence signals"
        
        logger.info(f"Multi-timeframe integration test passed with {all_signals} confluence signals")
    
    def test_indicator_selection_integration(self, sample_market_data):
        """Test that the intelligent indicator selection system works."""
        # Create market regime classifier
        regime_classifier = MarketRegimeClassifier()
        
        # Create indicator performance metrics
        performance_metrics = IndicatorPerformanceMetrics()
        
        # Create indicator selection engine
        selection_engine = IndicatorSelectionEngine(
            regime_classifier=regime_classifier,
            performance_metrics=performance_metrics
        )
        
        # Apply regime classification
        data = sample_market_data.copy()
        regime_result = regime_classifier.classify_regimes(data)
        
        # Define a set of indicators with performance history
        indicator_performance = {
            'tema_20': {
                'trending': 0.75,
                'ranging': 0.45,
                'volatile': 0.30,
                'breakout': 0.65
            },
            'rsi_14': {
                'trending': 0.40,
                'ranging': 0.80,
                'volatile': 0.55,
                'breakout': 0.35
            },
            'bb_20_2': {
                'trending': 0.30,
                'ranging': 0.85,
                'volatile': 0.65,
                'breakout': 0.40
            },
            'macd': {
                'trending': 0.70,
                'ranging': 0.35,
                'volatile': 0.45,
                'breakout': 0.75
            }
        }
        
        # Register indicators with their performance metrics
        for indicator, performance in indicator_performance.items():
            for regime, score in performance.items():
                performance_metrics.register_indicator_performance(
                    indicator_name=indicator,
                    market_regime=regime,
                    accuracy=score
                )
        
        # Register indicator categories
        selection_engine.register_indicator_category('trend', ['tema_20', 'macd'])
        selection_engine.register_indicator_category('momentum', ['rsi_14'])
        selection_engine.register_indicator_category('volatility', ['bb_20_2'])
        
        # Get the current market regime from the classified data
        # For simplicity, use the most recent regime
        current_regime = regime_result['regime'].iloc[-1]
        
        # Select indicators for the current market regime
        selected_indicators = selection_engine.select_indicators(current_regime)
        
        # Verify results
        assert isinstance(selected_indicators, dict)
        assert 'trend' in selected_indicators
        assert 'momentum' in selected_indicators
        assert 'volatility' in selected_indicators
        
        # The selection should prioritize higher-performing indicators for the current regime
        if current_regime == 'trending':
            assert 'tema_20' in selected_indicators['trend']
        elif current_regime == 'ranging':
            assert 'bb_20_2' in selected_indicators['volatility']
        
        logger.info(f"Indicator selection test passed for {current_regime} regime with "
                    f"{len(selected_indicators)} indicator categories")
    
    def test_ml_integration(self, sample_market_data, multi_timeframe_data):
        """Test integration with ML components."""
        # Create feature extraction components
        feature_extractor = FeatureExtractor()
        feature_selector = FeatureSelector()
        
        # Create some indicators
        tema = TripleExponentialMovingAverage(period=20)
        hull = HullMovingAverage(period=10)
        rvi = RelativeVigorIndex(period=14)
        
        # Calculate indicators
        data = sample_market_data.copy()
        tema_result = tema.calculate(data)
        hull_result = hull.calculate(data)
        rvi_result = rvi.calculate(data)
        
        # Combine indicators
        combined = pd.DataFrame({
            'close': data['close'],
            'tema_20': tema_result['tema_20'],
            'hull_10': hull_result['hull_10'],
            'rvi_14': rvi_result['rvi_14']
        })
        
        # Create target variable (next day return)
        combined['next_day_return'] = combined['close'].pct_change(1).shift(-1)
        
        # Drop NaN values
        combined = combined.dropna()
        
        # Extract features using the feature extractor
        feature_data = feature_extractor.extract_features(
            data=combined,
            feature_configs=[
                # Price trend features
                {'type': 'trend', 'source': 'close', 'window': 10},
                # Indicator crossover features
                {'type': 'crossover', 'fast': 'hull_10', 'slow': 'tema_20'},
                # Oscillator state features
                {'type': 'oscillator_state', 'source': 'rvi_14', 'overbought': 0.7, 'oversold': 0.3},
                # Indicator divergence features
                {'type': 'divergence', 'price': 'close', 'indicator': 'rvi_14', 'window': 10}
            ]
        )
        
        # Add the target variable
        X = feature_data
        y = (combined['next_day_return'] > 0).astype(int)  # Binary classification: up or down
        
        # Select features
        selected_features = feature_selector.select_features(
            X=X,
            y=y,
            method='importance',
            threshold=0.01
        )
        
        # Verify results
        assert len(selected_features) > 0, "Feature selection should return some features"
        assert isinstance(selected_features, list)
        
        # Check that selected features exist in the feature data
        for feature in selected_features:
            assert feature in X.columns, f"Selected feature {feature} should exist in the feature data"
        
        logger.info(f"ML integration test passed with {len(selected_features)} selected features "
                    f"out of {X.shape[1]} total features")
    
    def test_end_to_end_indicator_pipeline(self, sample_market_data, multi_timeframe_data):
        """Test the entire indicator pipeline from raw data to final signals."""
        # This test simulates a complete workflow through the entire indicator pipeline
        
        # STEP 1: Market Regime Classification
        regime_classifier = MarketRegimeClassifier()
        regime_result = regime_classifier.classify_regimes(sample_market_data)
        current_regime = regime_result['regime'].iloc[-1]
        
        logger.info(f"Current market regime identified as: {current_regime}")
        
        # STEP 2: Indicator Selection based on market regime
        performance_metrics = IndicatorPerformanceMetrics()
        selection_engine = IndicatorSelectionEngine(
            regime_classifier=regime_classifier,
            performance_metrics=performance_metrics
        )
        
        # Configure some sample indicator performance data
        indicators_by_regime = {
            'trending': ['tema_20', 'macd', 'ao'],
            'ranging': ['bb_20_2', 'rvi_14', 'demarker_14'],
            'volatile': ['atr_14', 'kama_20_2_30', 'nvi_pvi'],
            'breakout': ['macd', 'vzo_14', 'hull_10']
        }
        
        # Register indicators with performance metrics
        for regime, indicators in indicators_by_regime.items():
            for indicator in indicators:
                # Assign high score for indicators matching the regime
                score = 0.85 if indicator in indicators_by_regime[regime] else 0.4
                performance_metrics.register_indicator_performance(
                    indicator_name=indicator,
                    market_regime=regime,
                    accuracy=score
                )
        
        # Register indicator categories
        selection_engine.register_indicator_category('trend', ['tema_20', 'hull_10', 'kama_20_2_30'])
        selection_engine.register_indicator_category('momentum', ['rvi_14', 'macd', 'ao'])
        selection_engine.register_indicator_category('volatility', ['bb_20_2', 'atr_14'])
        selection_engine.register_indicator_category('volume', ['vzo_14', 'nvi_pvi'])
        
        # Select indicators for current market regime
        selected_indicators = selection_engine.select_indicators(current_regime)
        
        logger.info(f"Selected indicators for {current_regime} regime: {selected_indicators}")
        
        # STEP 3: Calculate selected indicators
        data = sample_market_data.copy()
        indicator_results = {}
        
        # Calculate all indicators
        if 'tema_20' in selected_indicators.get('trend', []):
            tema = TripleExponentialMovingAverage(period=20)
            indicator_results['tema_20'] = tema.calculate(data)['tema_20']
            
        if 'hull_10' in selected_indicators.get('trend', []):
            hull = HullMovingAverage(period=10)
            indicator_results['hull_10'] = hull.calculate(data)['hull_10']
            
        if 'rvi_14' in selected_indicators.get('momentum', []):
            rvi = RelativeVigorIndex(period=14)
            indicator_results['rvi_14'] = rvi.calculate(data)['rvi_14']
            
        if 'ao' in selected_indicators.get('momentum', []):
            ao = AwesomeOscillator()
            indicator_results['ao'] = ao.calculate(data)['ao']
            
        if 'bb_20_2' in selected_indicators.get('volatility', []):
            # Simulating Bollinger Bands calculation
            indicator_results['bb_upper'] = data['close'].rolling(20).mean() + 2 * data['close'].rolling(20).std()
            indicator_results['bb_lower'] = data['close'].rolling(20).mean() - 2 * data['close'].rolling(20).std()
            
        if 'vzo_14' in selected_indicators.get('volume', []):
            vzo = VolumeZoneOscillator(period=14)
            indicator_results['vzo_14'] = vzo.calculate(data)['vzo_14']
        
        # STEP 4: Multi-timeframe analysis
        if len(indicator_results) >= 2:
            # Choose two indicators for multi-timeframe analysis
            indicator_keys = list(indicator_results.keys())
            indicator1 = indicator_keys[0]
            indicator2 = indicator_keys[1]
            
            # Create multi-timeframe wrapper for these indicators
            # For this test, we'll just simulate multi-timeframe analysis
            mtf_results = {}
            for tf in ['1h', '4h', '1d']:
                mtf_results[tf] = {}
                
                # For testing, create a simulated version of the indicator at different timeframes
                if tf == '1h':
                    mtf_results[tf][indicator1] = indicator_results[indicator1]
                    mtf_results[tf][indicator2] = indicator_results[indicator2]
                else:
                    # Resample the indicator (simplified method for testing)
                    mtf_results[tf][indicator1] = indicator_results[indicator1].resample(
                        '4H' if tf == '4h' else '1D').mean()
                    mtf_results[tf][indicator2] = indicator_results[indicator2].resample(
                        '4H' if tf == '4h' else '1D').mean()
            
            # Create timeframe confluence scanner
            confluence_scanner = TimeframeConfluenceScanner(
                timeframes=['1h', '4h', '1d'],
                indicators=[indicator1, indicator2]
            )
            
            # Scan for confluence signals
            confluence_data = {}
            for tf in ['1h', '4h', '1d']:
                confluence_data[tf] = pd.DataFrame({
                    'close': multi_timeframe_data[tf]['close'],
                    indicator1: mtf_results[tf][indicator1],
                    indicator2: mtf_results[tf][indicator2]
                })
            
            confluence_signals = confluence_scanner.scan_for_confluence(confluence_data)
            
            logger.info(f"Multi-timeframe confluence analysis found "
                        f"{len(confluence_signals['bullish_signals'])} bullish signals and "
                        f"{len(confluence_signals['bearish_signals'])} bearish signals")
        
        # STEP 5: Feature extraction for ML
        feature_extractor = FeatureExtractor()
        
        # Combine indicators into a single dataframe
        combined = pd.DataFrame({'close': data['close']})
        for name, series in indicator_results.items():
            combined[name] = series
        
        # Drop NaN values
        combined = combined.dropna()
        
        # Extract features
        feature_configs = []
        
        # Add trend features for each indicator
        for name in indicator_results.keys():
            feature_configs.append({
                'type': 'trend', 
                'source': name, 
                'window': 5
            })
            
        # Add crossover features if we have multiple indicators
        if len(indicator_results) >= 2:
            keys = list(indicator_results.keys())
            feature_configs.append({
                'type': 'crossover',
                'fast': keys[0],
                'slow': keys[1]
            })
        
        # Extract features
        feature_data = feature_extractor.extract_features(
            data=combined,
            feature_configs=feature_configs
        )
        
        logger.info(f"Extracted {feature_data.shape[1]} features for ML from {len(indicator_results)} indicators")
        
        # STEP 6: Generate final trading signals
        # For testing, we'll use a simple combination of indicators and ML features
        
        # Create a simple signal generator
        signals = pd.DataFrame({
            'date': combined.index,
            'close': combined['close']
        })
        
        # Add signals based on the regime
        if current_regime == 'trending':
            # In trending regimes, use moving average crossovers
            if 'hull_10' in combined.columns and 'tema_20' in combined.columns:
                signals['trend_signal'] = np.where(
                    combined['hull_10'] > combined['tema_20'], 1, -1
                )
        elif current_regime == 'ranging':
            # In ranging regimes, use oscillator overbought/oversold
            if 'rvi_14' in combined.columns:
                signals['range_signal'] = np.where(
                    combined['rvi_14'] > 0.7, -1,  # Overbought
                    np.where(
                        combined['rvi_14'] < 0.3, 1,  # Oversold
                        0  # Neither
                    )
                )
        elif current_regime == 'volatile':
            # In volatile regimes, use ATR for position sizing
            # (simulated for this test)
            signals['volatility'] = combined['close'].rolling(14).std()
            signals['position_size'] = 1 / signals['volatility']
        elif current_regime == 'breakout':
            # In breakout regimes, use volume confirmation
            if 'vzo_14' in combined.columns:
                signals['breakout_confirmed'] = np.where(
                    abs(combined['close'].pct_change()) > 0.01,  # Price move of >1%
                    np.where(
                        combined['vzo_14'] > 30,  # Volume confirmation
                        np.sign(combined['close'].pct_change()),  # Direction of breakout
                        0  # No confirmation
                    ),
                    0  # No significant price move
                )
        
        # Final signal combines available indicators
        signal_columns = [col for col in signals.columns if col not in ['date', 'close']]
        if signal_columns:
            signals['final_signal'] = signals[signal_columns].sum(axis=1)
            signals['final_signal'] = np.sign(signals['final_signal'])  # Convert to -1, 0, 1
            
            # Count the number of signals
            bullish_count = (signals['final_signal'] > 0).sum()
            bearish_count = (signals['final_signal'] < 0).sum()
            neutral_count = (signals['final_signal'] == 0).sum()
            
            logger.info(f"Generated {bullish_count} bullish, {bearish_count} bearish, "
                        f"and {neutral_count} neutral signals for the {current_regime} regime")
            
            # Test passes if we generated some signals
            assert bullish_count + bearish_count > 0, "Signal generation should produce some trading signals"
        
        # If we reached this point, the pipeline works end-to-end
        logger.info("End-to-end indicator pipeline integration test passed successfully")


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
