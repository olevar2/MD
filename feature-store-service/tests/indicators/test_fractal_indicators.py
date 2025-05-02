"""
Unit tests for fractal geometry indicators.
"""

import unittest
import pandas as pd
import numpy as np
from feature_store_service.indicators.fractal_indicators import (
    FractalIndicator,
    AlligatorIndicator,
    AwesomeOscillatorFractal,
    ElliottWaveAnalyzer,
    HurstExponent
)


class TestFractalIndicator(unittest.TestCase):
    """Test suite for Bill Williams' Fractal indicator."""

    def setUp(self):
        """Set up test data with clear fractal patterns."""
        np.random.seed(42)
        n_samples = 200
        
        # Generate price data with clear fractal patterns
        date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # Create base price with some trends
        price = np.cumsum(np.random.normal(0, 1, n_samples)) + 100
        
        # Insert bullish fractal pattern (low point with higher lows on both sides)
        # Center at index 50
        price[48:53] = [105, 103, 100, 103, 106]
        
        # Insert bearish fractal pattern (high point with lower highs on both sides)
        # Center at index 100
        price[98:103] = [115, 117, 120, 118, 115]
        
        self.data = pd.DataFrame({
            'open': price * (1 + 0.003 * np.random.randn(n_samples)),
            'high': price * (1 + 0.006 * np.random.randn(n_samples)),
            'low': price * (1 - 0.006 * np.random.randn(n_samples)),
            'close': price,
            'volume': 1000000 * (1 + 0.1 * np.random.randn(n_samples))
        }, index=date_range)
        
        # Ensure high is highest and low is lowest
        self.data['high'] = np.maximum(
            np.maximum(self.data['high'], self.data['open']), 
            self.data['close']
        )
        self.data['low'] = np.minimum(
            np.minimum(self.data['low'], self.data['open']), 
            self.data['close']
        )
        
        # Initialize fractal indicator
        self.fractal_indicator = FractalIndicator()
    
    def test_fractal_detection(self):
        """Test fractal pattern detection."""
        # Calculate fractals
        result = self.fractal_indicator.calculate(self.data)
        
        # Should have columns for bullish and bearish fractals
        self.assertIn('fractal_bullish', result.columns)
        self.assertIn('fractal_bearish', result.columns)
        
        # Check if the inserted bullish fractal was detected (at index 50)
        self.assertEqual(result['fractal_bullish'].iloc[50], 1)
        
        # Check if the inserted bearish fractal was detected (at index 100)
        self.assertEqual(result['fractal_bearish'].iloc[100], 1)
    
    def test_fractal_properties(self):
        """Test properties of detected fractals."""
        # Calculate fractals
        result = self.fractal_indicator.calculate(self.data)
        
        # Get all detected fractals
        bullish_fractals = result[result['fractal_bullish'] == 1]
        bearish_fractals = result[result['fractal_bearish'] == 1]
        
        # Should have detected at least a few of each
        self.assertGreater(len(bullish_fractals), 0)
        self.assertGreater(len(bearish_fractals), 0)
        
        # Verify that bearish fractals are local highs
        for idx in bearish_fractals.index:
            # Can't check the very start or end of the data
            if idx < 2 or idx > len(self.data) - 3:
                continue
                
            # Central point should be higher than two points on each side
            central_high = self.data['high'].loc[idx]
            left_high1 = self.data['high'].loc[idx-1]
            left_high2 = self.data['high'].loc[idx-2]
            right_high1 = self.data['high'].loc[idx+1]
            right_high2 = self.data['high'].loc[idx+2]
            
            self.assertGreaterEqual(central_high, left_high1)
            self.assertGreaterEqual(central_high, left_high2)
            self.assertGreaterEqual(central_high, right_high1)
            self.assertGreaterEqual(central_high, right_high2)
        
        # Verify that bullish fractals are local lows
        for idx in bullish_fractals.index:
            # Can't check the very start or end of the data
            if idx < 2 or idx > len(self.data) - 3:
                continue
                
            # Central point should be lower than two points on each side
            central_low = self.data['low'].loc[idx]
            left_low1 = self.data['low'].loc[idx-1]
            left_low2 = self.data['low'].loc[idx-2]
            right_low1 = self.data['low'].loc[idx+1]
            right_low2 = self.data['low'].loc[idx+2]
            
            self.assertLessEqual(central_low, left_low1)
            self.assertLessEqual(central_low, left_low2)
            self.assertLessEqual(central_low, right_low1)
            self.assertLessEqual(central_low, right_low2)


class TestAlligatorIndicator(unittest.TestCase):
    """Test suite for Bill Williams' Alligator indicator."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 200
        
        # Generate price data with trends
        date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # Create base price with a trend change
        trend1 = np.linspace(0, 10, 100)
        trend2 = np.linspace(10, 0, 100)
        trend = np.concatenate([trend1, trend2])
        
        noise = np.random.normal(0, 1, n_samples)
        price = 100 + trend + noise
        
        self.data = pd.DataFrame({
            'open': price * (1 + 0.003 * np.random.randn(n_samples)),
            'high': price * (1 + 0.006 * np.random.randn(n_samples)),
            'low': price * (1 - 0.006 * np.random.randn(n_samples)),
            'close': price,
            'volume': 1000000 * (1 + 0.1 * np.random.randn(n_samples))
        }, index=date_range)
        
        # Initialize Alligator indicator
        self.alligator = AlligatorIndicator()
    
    def test_alligator_line_calculation(self):
        """Test calculation of Alligator indicator lines."""
        # Calculate Alligator lines
        result = self.alligator.calculate(self.data)
        
        # Should have columns for jaw, teeth, and lips
        self.assertIn('alligator_jaw', result.columns)
        self.assertIn('alligator_teeth', result.columns)
        self.assertIn('alligator_lips', result.columns)
        
        # All three lines should be present
        self.assertGreater(result['alligator_jaw'].dropna().shape[0], 0)
        self.assertGreater(result['alligator_teeth'].dropna().shape[0], 0)
        self.assertGreater(result['alligator_lips'].dropna().shape[0], 0)
    
    def test_alligator_trend_identification(self):
        """Test Alligator's trend identification capabilities."""
        # Calculate Alligator lines
        result = self.alligator.calculate(self.data)
        
        # During an uptrend, lines should be ordered: lips > teeth > jaw
        # During a downtrend, lines should be ordered: jaw > teeth > lips
        
        # First half of data should be in uptrend
        uptrend_idx = 80  # Choose a point well into the uptrend
        self.assertGreaterEqual(
            result['alligator_lips'].iloc[uptrend_idx],
            result['alligator_teeth'].iloc[uptrend_idx]
        )
        self.assertGreaterEqual(
            result['alligator_teeth'].iloc[uptrend_idx],
            result['alligator_jaw'].iloc[uptrend_idx]
        )
        
        # Second half of data should be in downtrend
        downtrend_idx = 180  # Choose a point well into the downtrend
        self.assertLessEqual(
            result['alligator_lips'].iloc[downtrend_idx],
            result['alligator_teeth'].iloc[downtrend_idx]
        )
        self.assertLessEqual(
            result['alligator_teeth'].iloc[downtrend_idx],
            result['alligator_jaw'].iloc[downtrend_idx]
        )


class TestAwesomeOscillatorFractal(unittest.TestCase):
    """Test suite for enhanced Awesome Oscillator with fractal signals."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 200
        
        # Generate price data with trends and reversals
        date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # Create base price with trends
        trend1 = np.linspace(0, 20, 80)
        flat = np.ones(20) * 20
        trend2 = np.linspace(20, 0, 80)
        trend = np.concatenate([trend1, flat, trend2])
        
        noise = np.random.normal(0, 1, n_samples)
        
        # Base price
        price = 100 + trend + noise
        
        self.data = pd.DataFrame({
            'open': price * (1 + 0.003 * np.random.randn(n_samples)),
            'high': price * (1 + 0.006 * np.random.randn(n_samples)),
            'low': price * (1 - 0.006 * np.random.randn(n_samples)),
            'close': price,
            'volume': 1000000 * (1 + 0.1 * np.random.randn(n_samples))
        }, index=date_range)
        
        # Initialize AwesomeOscillator
        self.ao = AwesomeOscillatorFractal()
    
    def test_awesome_oscillator_calculation(self):
        """Test calculation of Awesome Oscillator."""
        # Calculate AO
        result = self.ao.calculate(self.data)
        
        # Should have AO column
        self.assertIn('awesome_oscillator', result.columns)
        
        # AO should be calculated for most of the data
        self.assertGreater(result['awesome_oscillator'].dropna().shape[0], 0)
        
        # AO should be positive during uptrend and negative during downtrend
        # Choose points well into the trends
        uptrend_idx = 50
        downtrend_idx = 150
        
        self.assertGreater(result['awesome_oscillator'].iloc[uptrend_idx], 0)
        self.assertLess(result['awesome_oscillator'].iloc[downtrend_idx], 0)
    
    def test_ao_fractal_signals(self):
        """Test fractal-based signals from Awesome Oscillator."""
        # Calculate AO with signals
        result = self.ao.calculate(self.data, include_signals=True)
        
        # Should have signal columns
        self.assertIn('ao_saucer_signal', result.columns)
        self.assertIn('ao_twin_peaks_signal', result.columns)
        
        # Check if any signals were generated
        signals_present = (
            (result['ao_saucer_signal'] != 0).any() or
            (result['ao_twin_peaks_signal'] != 0).any()
        )
        
        self.assertTrue(signals_present, "No AO signals were generated")


class TestElliottWaveAnalyzer(unittest.TestCase):
    """Test suite for Elliott Wave analysis."""

    def setUp(self):
        """Set up test data with a simulated Elliott Wave pattern."""
        np.random.seed(42)
        n_samples = 300
        
        # Generate date range
        date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # Create a price series with embedded Elliott Wave pattern
        # 5-wave impulse followed by 3-wave correction
        price = np.ones(n_samples) * 100
        
        # Impulse wave 1 (up)
        price[10:30] = np.linspace(100, 110, 20)
        
        # Corrective wave 2 (down)
        price[30:50] = np.linspace(110, 105, 20)
        
        # Impulse wave 3 (up, extended)
        price[50:90] = np.linspace(105, 125, 40)
        
        # Corrective wave 4 (down)
        price[90:110] = np.linspace(125, 120, 20)
        
        # Impulse wave 5 (final up)
        price[110:140] = np.linspace(120, 130, 30)
        
        # Corrective wave A (down)
        price[140:170] = np.linspace(130, 115, 30)
        
        # Corrective wave B (up)
        price[170:190] = np.linspace(115, 123, 20)
        
        # Corrective wave C (down)
        price[190:220] = np.linspace(123, 105, 30)
        
        # Add some noise
        noise = np.random.normal(0, 0.5, n_samples)
        price += noise
        
        self.data = pd.DataFrame({
            'open': price * (1 + 0.002 * np.random.randn(n_samples)),
            'high': price * (1 + 0.004 * np.random.randn(n_samples)),
            'low': price * (1 - 0.004 * np.random.randn(n_samples)),
            'close': price,
            'volume': 1000000 * (1 + 0.1 * np.random.randn(n_samples))
        }, index=date_range)
        
        # Initialize Elliott Wave Analyzer
        self.elliott_wave = ElliottWaveAnalyzer()
    
    def test_elliott_wave_detection(self):
        """Test detection of Elliott Wave patterns."""
        # Analyze Elliott Wave patterns
        result = self.elliott_wave.analyze(self.data)
        
        # Should have identified wave patterns
        self.assertIn('waves', result)
        self.assertGreater(len(result['waves']), 0)
        
        # Should have at least one impulse and one corrective pattern
        wave_types = [wave['type'] for wave in result['waves']]
        self.assertIn('impulse', wave_types)
        self.assertIn('corrective', wave_types)
    
    def test_wave_degree_classification(self):
        """Test classification of wave degrees (Grand Supercycle, Supercycle, etc.)."""
        # Analyze Elliott Waves with degree classification
        result = self.elliott_wave.analyze(self.data, include_degrees=True)
        
        # Check if degrees were assigned
        for wave in result['waves']:
            self.assertIn('degree', wave)
            self.assertIsNotNone(wave['degree'])
    
    def test_fibonacci_targets(self):
        """Test calculation of Fibonacci targets from waves."""
        # Analyze Elliott Waves with Fibonacci projections
        result = self.elliott_wave.analyze(self.data, include_targets=True)
        
        # Check if targets were calculated
        self.assertIn('targets', result)
        self.assertGreater(len(result['targets']), 0)
        
        # Each target should have a price level and confidence
        for target in result['targets']:
            self.assertIn('price', target)
            self.assertIn('confidence', target)
            self.assertIn('type', target)


class TestHurstExponent(unittest.TestCase):
    """Test suite for Hurst exponent calculation."""

    def setUp(self):
        """Set up test data with different types of price series."""
        np.random.seed(42)
        n_samples = 500  # Need sufficient data for Hurst estimation
        
        # Generate 3 types of price series:
        # 1. Persistent (trending) series
        # 2. Random walk
        # 3. Mean-reverting series
        
        # Date range
        date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # 1. Trending series (Hurst > 0.5)
        trend = np.linspace(0, 50, n_samples)  # Strong trend
        persistent = 100 + trend + np.cumsum(np.random.normal(0, 1, n_samples) * 0.5)
        
        # 2. Random walk (Hurst ≈ 0.5)
        random_walk = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
        
        # 3. Mean-reverting series (Hurst < 0.5)
        mean_level = 100
        mean_reversion_rate = 0.2
        mean_reverting = np.zeros(n_samples)
        mean_reverting[0] = mean_level
        
        for i in range(1, n_samples):
            # Mean-reverting process with noise
            deviation = mean_reverting[i-1] - mean_level
            mean_reverting[i] = (
                mean_reverting[i-1] - mean_reversion_rate * deviation
                + np.random.normal(0, 1)
            )
        
        # Create separate DataFrames for each type
        self.trending_data = pd.DataFrame({
            'close': persistent,
            'open': persistent * (1 + 0.002 * np.random.randn(n_samples)),
            'high': persistent * (1 + 0.004 * np.random.randn(n_samples)),
            'low': persistent * (1 - 0.004 * np.random.randn(n_samples)),
            'volume': 1000000 * (1 + 0.1 * np.random.randn(n_samples))
        }, index=date_range)
        
        self.random_data = pd.DataFrame({
            'close': random_walk,
            'open': random_walk * (1 + 0.002 * np.random.randn(n_samples)),
            'high': random_walk * (1 + 0.004 * np.random.randn(n_samples)),
            'low': random_walk * (1 - 0.004 * np.random.randn(n_samples)),
            'volume': 1000000 * (1 + 0.1 * np.random.randn(n_samples))
        }, index=date_range)
        
        self.mean_reverting_data = pd.DataFrame({
            'close': mean_reverting,
            'open': mean_reverting * (1 + 0.002 * np.random.randn(n_samples)),
            'high': mean_reverting * (1 + 0.004 * np.random.randn(n_samples)),
            'low': mean_reverting * (1 - 0.004 * np.random.randn(n_samples)),
            'volume': 1000000 * (1 + 0.1 * np.random.randn(n_samples))
        }, index=date_range)
        
        # Initialize Hurst Exponent calculator
        self.hurst = HurstExponent()
    
    def test_hurst_calculation(self):
        """Test calculation of Hurst exponent on different price series."""
        # Calculate Hurst exponent for trending data
        trending_result = self.hurst.calculate(self.trending_data)
        
        # Calculate Hurst exponent for random walk data
        random_result = self.hurst.calculate(self.random_data)
        
        # Calculate Hurst exponent for mean-reverting data
        revert_result = self.hurst.calculate(self.mean_reverting_data)
        
        # Check that the indicator column was added
        self.assertIn('hurst_exponent', trending_result.columns)
        self.assertIn('hurst_exponent', random_result.columns)
        self.assertIn('hurst_exponent', revert_result.columns)
        
        # Get the final Hurst value for each series
        trending_hurst = trending_result['hurst_exponent'].iloc[-1]
        random_hurst = random_result['hurst_exponent'].iloc[-1]
        revert_hurst = revert_result['hurst_exponent'].iloc[-1]
        
        # Check that values are within appropriate ranges
        # Trending data should have Hurst > 0.5
        self.assertGreater(trending_hurst, 0.5)
        
        # Random walk should have Hurst ≈ 0.5
        self.assertAlmostEqual(random_hurst, 0.5, delta=0.15)
        
        # Mean-reverting should have Hurst < 0.5
        self.assertLess(revert_hurst, 0.5)
    
    def test_hurst_interpretation(self):
        """Test interpretation of Hurst exponent values."""
        # Calculate Hurst with interpretation
        result = self.hurst.calculate(self.trending_data, include_interpretation=True)
        
        # Should have interpretation column
        self.assertIn('hurst_interpretation', result.columns)
        
        # Check that interpretations are correct
        for i, h_value in enumerate(result['hurst_exponent']):
            if pd.isna(h_value):
                continue
                
            interp = result['hurst_interpretation'].iloc[i]
            
            if h_value > 0.5:
                self.assertEqual(interp, 'trending')
            elif h_value < 0.5:
                self.assertEqual(interp, 'mean_reverting')
            else:
                self.assertEqual(interp, 'random_walk')


if __name__ == '__main__':
    unittest.main()
