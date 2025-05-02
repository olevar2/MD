"""
Integration tests for the entire indicator pipeline.
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import json
from datetime import datetime, timedelta

from feature_store_service.indicators.advanced_moving_averages import TEMA, DEMA, HullMA
from feature_store_service.indicators.chart_patterns import ChartPatternRecognizer, HarmonicPatternFinder
from feature_store_service.indicators.gann_tools import GannAngles, GannSquare9
from feature_store_service.indicators.fractal_indicators import (
    FractalIndicator, 
    AlligatorIndicator, 
    ElliottWaveAnalyzer
)
from feature_store_service.indicators.multi_timeframe import (
    MultiTimeframeIndicator,
    TimeframeComparison,
    TimeframeConfluenceScanner
)
from feature_store_service.indicators.indicator_selection import (
    MarketRegimeClassifier,
    IndicatorPerformanceMetrics,
    IndicatorSelectionEngine,
    MarketRegime
)
from feature_store_service.caching import IndicatorCache
from feature_store_service.indicators.base_indicator import BaseIndicator, IndicatorRegistry

from ml_integration_service.feature_extraction import FeatureExtractor
from ml_integration_service.feature_selection import FeatureSelector, ImportanceBasedSelector
from ml_integration_service.indicator_feedback import IndicatorMLFeedback
from ml_integration_service.model_preparation import ModelInputPreparation, TabularInputPreparator


class TestIndicatorPipeline(unittest.TestCase):
    """Test suite for the entire indicator pipeline integration."""

    @classmethod
    def setUpClass(cls):
        """Set up data and components for all tests."""
        # Generate realistic market data for different market regimes
        cls.trending_data = cls._generate_trending_data()
        cls.ranging_data = cls._generate_ranging_data()
        cls.volatile_data = cls._generate_volatile_data()
        cls.multi_timeframe_data = cls._generate_multi_timeframe_data()
        
        # Create temporary directory for cache and storage
        cls.temp_dir = tempfile.mkdtemp()
        
        # Initialize cache
        cls.cache = IndicatorCache(
            cache_dir=os.path.join(cls.temp_dir, 'indicator_cache'),
            max_memory_mb=200,
            expire_after_days=7
        )
        
        # Initialize indicator registry
        cls.registry = IndicatorRegistry()
        cls._register_all_indicators(cls.registry)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(cls.temp_dir)
        except (OSError, IOError):
            pass
    
    @classmethod
    def _generate_trending_data(cls, n_samples=500):
        """Generate data with a clear trend."""
        np.random.seed(42)
        date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # Strong uptrend with minor pullbacks
        trend = np.linspace(0, 20, n_samples)
        noise = np.random.normal(0, 1, n_samples)
        cycles = 2 * np.sin(np.linspace(0, 6*np.pi, n_samples))
        
        price = 100 + trend + noise + cycles
        
        return pd.DataFrame({
            'open': price * (1 + 0.003 * np.random.randn(n_samples)),
            'high': price * (1 + 0.006 * np.random.randn(n_samples)),
            'low': price * (1 - 0.006 * np.random.randn(n_samples)),
            'close': price,
            'volume': 1000000 * (1 + 0.2 * np.random.randn(n_samples))
        }, index=date_range)
    
    @classmethod
    def _generate_ranging_data(cls, n_samples=500):
        """Generate data in a trading range."""
        np.random.seed(43)
        date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # Oscillating price in a range
        base = 100
        amplitude = 10
        cycles = amplitude * np.sin(np.linspace(0, 10*np.pi, n_samples))
        noise = np.random.normal(0, 1, n_samples)
        
        price = base + cycles + noise
        
        return pd.DataFrame({
            'open': price * (1 + 0.003 * np.random.randn(n_samples)),
            'high': price * (1 + 0.006 * np.random.randn(n_samples)),
            'low': price * (1 - 0.006 * np.random.randn(n_samples)),
            'close': price,
            'volume': 1000000 * (1 + 0.2 * np.random.randn(n_samples))
        }, index=date_range)
    
    @classmethod
    def _generate_volatile_data(cls, n_samples=500):
        """Generate data with high volatility."""
        np.random.seed(44)
        date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # Base price with increasing volatility
        base = np.cumsum(np.random.normal(0, 0.3, n_samples))
        
        # Add volatility clusters
        volatility = np.ones(n_samples)
        volatility[100:150] = 3.0  # High volatility period
        volatility[300:350] = 4.0  # Another high volatility period
        
        noise = np.random.normal(0, 1, n_samples) * volatility
        price = 100 + base + noise
        
        return pd.DataFrame({
            'open': price * (1 + 0.005 * np.random.randn(n_samples) * volatility),
            'high': price * (1 + 0.01 * np.random.randn(n_samples) * volatility),
            'low': price * (1 - 0.01 * np.random.randn(n_samples) * volatility),
            'close': price,
            'volume': 1000000 * (1 + 0.3 * np.random.randn(n_samples) * volatility)
        }, index=date_range)
    
    @classmethod
    def _generate_multi_timeframe_data(cls):
        """Generate data for multiple timeframes."""
        # Start with 1-minute data for a week
        np.random.seed(45)
        minutes_per_day = 1440
        days = 7
        n_samples = minutes_per_day * days
        
        start_date = datetime(2023, 1, 1)
        date_range_1min = [start_date + timedelta(minutes=i) for i in range(n_samples)]
        
        # Generate price with intraday patterns
        price = 100
        prices_1min = [price]
        
        for i in range(1, n_samples):
            # Add time-of-day effect (U-shape volatility)
            minute_of_day = i % minutes_per_day
            hour_of_day = minute_of_day // 60
            
            # Higher volatility at market open and close
            if hour_of_day < 2 or hour_of_day > 21:
                volatility = 0.02
            else:
                volatility = 0.005
                
            # Add some price drift based on time of day
            if 9 <= hour_of_day < 16:  # Trading hours
                drift = 0.001
            else:
                drift = -0.0002
                
            # Random price change
            price += np.random.normal(drift, volatility)
            prices_1min.append(price)
        
        # Create 1-minute DataFrame
        data_1min = pd.DataFrame({
            'open': [p - np.random.uniform(0, 0.01) for p in prices_1min],
            'high': [p + np.random.uniform(0, 0.02) for p in prices_1min],
            'low': [p - np.random.uniform(0, 0.02) for p in prices_1min],
            'close': prices_1min,
            'volume': [np.random.randint(100, 1000) for _ in range(n_samples)]
        }, index=date_range_1min)
        
        # Create higher timeframe data by resampling
        data_5min = cls._resample_ohlc(data_1min, '5T')
        data_15min = cls._resample_ohlc(data_1min, '15T')
        data_1hour = cls._resample_ohlc(data_1min, '1H')
        data_4hour = cls._resample_ohlc(data_1min, '4H')
        data_1day = cls._resample_ohlc(data_1min, '1D')
        
        return {
            '1T': data_1min,
            '5T': data_5min,
            '15T': data_15min,
            '1H': data_1hour,
            '4H': data_4hour,
            '1D': data_1day
        }
    
    @classmethod
    def _resample_ohlc(cls, df, timeframe):
        """Resample OHLCV data to a higher timeframe."""
        resampled = pd.DataFrame()
        resampled['open'] = df['open'].resample(timeframe).first()
        resampled['high'] = df['high'].resample(timeframe).max()
        resampled['low'] = df['low'].resample(timeframe).min()
        resampled['close'] = df['close'].resample(timeframe).last()
        resampled['volume'] = df['volume'].resample(timeframe).sum()
        return resampled
    
    @classmethod
    def _register_all_indicators(cls, registry):
        """Register all indicators with the registry."""
        # Moving averages
        registry.register('tema', TEMA)
        registry.register('dema', DEMA)
        registry.register('hullma', HullMA)
        
        # Chart patterns
        registry.register('pattern_recognizer', ChartPatternRecognizer)
        registry.register('harmonic_patterns', HarmonicPatternFinder)
        
        # Gann tools
        registry.register('gann_angles', GannAngles)
        registry.register('gann_square9', GannSquare9)
        
        # Fractal indicators
        registry.register('fractal', FractalIndicator)
        registry.register('alligator', AlligatorIndicator)
        registry.register('elliott_wave', ElliottWaveAnalyzer)
        
        # Return the populated registry
        return registry
    
    def test_end_to_end_indicator_pipeline(self):
        """Test the complete indicator pipeline from data to features."""
        print("\nRunning end-to-end indicator pipeline test...")
        
        # 1. Market regime classification
        regime_classifier = MarketRegimeClassifier()
        trending_regime = regime_classifier.classify(self.trending_data)
        ranging_regime = regime_classifier.classify(self.ranging_data)
        
        # Verify regime classification
        self.assertEqual(trending_regime, MarketRegime.TRENDING_BULLISH)
        self.assertEqual(ranging_regime, MarketRegime.RANGING_TIGHT)
        
        # 2. Indicator selection based on market regime
        selection_engine = IndicatorSelectionEngine(
            indicators_registry=self.registry.get_all_indicators(),
            performance_history_path=os.path.join(self.temp_dir, 'performance.json'),
            config_path=os.path.join(self.temp_dir, 'config.json')
        )
        
        # Select indicators for trending regime
        trending_indicators = selection_engine.select_indicators(self.trending_data)
        
        # Verify that we selected some indicators
        self.assertGreater(len(trending_indicators), 0)
        print(f"Selected {len(trending_indicators)} indicators for trending regime")
        
        # 3. Calculate indicators
        results = self.trending_data.copy()
        calculated_indicators = []
        
        for name, indicator in trending_indicators.items():
            try:
                # Calculate and time each indicator
                import time
                start_time = time.time()
                
                results = indicator.calculate(results)
                
                calc_time = time.time() - start_time
                calculated_indicators.append(name)
                
                print(f"Calculated {name} in {calc_time:.4f} seconds")
            except Exception as e:
                print(f"Failed to calculate {name}: {str(e)}")
        
        # Verify that indicators were calculated
        self.assertGreater(len(calculated_indicators), 0)
        
        # 4. Extract features for ML
        feature_extractor = FeatureExtractor(normalize=True)
        features = feature_extractor.extract_features(results)
        
        # Verify features were created
        self.assertGreater(len(features.columns), len(self.trending_data.columns))
        print(f"Extracted {len(features.columns)} features from indicators")
        
        # 5. Create ML inputs
        # Create a synthetic target for testing
        target = pd.Series(
            np.sign(self.trending_data['close'].pct_change(5).shift(-5)),
            index=self.trending_data.index
        )
        
        # Prepare inputs for ML model
        input_prep = TabularInputPreparator()
        X, y = input_prep.prepare(features, target)
        
        # Verify inputs were prepared
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        print(f"Prepared ML inputs with shape {X.shape}")
        
        # Verify the complete pipeline ran successfully
        print("End-to-end indicator pipeline test completed successfully")
    
    def test_multi_timeframe_integration(self):
        """Test integration of multi-timeframe analysis components."""
        print("\nRunning multi-timeframe integration test...")
        
        # Get data for multiple timeframes
        data_1hour = self.multi_timeframe_data['1H']
        data_4hour = self.multi_timeframe_data['4H']
        data_1day = self.multi_timeframe_data['1D']
        
        # 1. Create multi-timeframe indicators
        multi_tema = MultiTimeframeIndicator(
            indicator=TEMA(period=20),
            timeframes=['1H', '4H', '1D']
        )
        
        # 2. Calculate indicators across timeframes
        results = multi_tema.calculate(data_1hour)
        
        # Verify results contain indicators for each timeframe
        self.assertIn('tema_20_1H', results.columns)
        self.assertIn('tema_20_4H', results.columns)
        self.assertIn('tema_20_1D', results.columns)
        
        # 3. Test timeframe comparison
        tf_comparison = TimeframeComparison(
            indicator=TEMA(period=20),
            timeframes=['1H', '4H', '1D']
        )
        
        comparison_results = tf_comparison.calculate_trend_alignment(data_1hour)
        
        # Verify comparison results
        self.assertIn('trend_alignment', comparison_results.columns)
        
        # 4. Test confluence scanning
        confluence_scanner = TimeframeConfluenceScanner(
            indicators=[TEMA(period=20), DEMA(period=20)],
            timeframes=['1H', '4H', '1D']
        )
        
        confluence_results = confluence_scanner.scan_for_confluence(data_1hour)
        
        # Verify confluence scanning results
        self.assertIn('bullish_confluence', confluence_results.columns)
        self.assertIn('bearish_confluence', confluence_results.columns)
        
        # 5. Generate trading signals based on confluence
        signal_results = confluence_scanner.generate_signals(data_1hour)
        
        # Verify signal generation
        self.assertIn('confluence_signal', signal_results.columns)
        
        # Check that signals have expected values (-1, 0, 1)
        unique_signals = signal_results['confluence_signal'].dropna().unique()
        for signal in unique_signals:
            self.assertIn(signal, [-1, 0, 1])
        
        print("Multi-timeframe integration test completed successfully")
    
    def test_caching_integration(self):
        """Test integration of caching system with indicators."""
        print("\nRunning caching integration test...")
        
        # Create cacheable indicator
        class CacheableTEMA(TEMA):
            """TEMA with caching capability."""
            
            def calculate(self, data):
                """Calculate with caching support."""
                cache_key = f"tema_{self.period}_{hash(tuple(data.index))}"
                
                # Try to get from cache
                cached_result = TestIndicatorPipeline.cache.get_memory(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Calculate if not cached
                result = super().calculate(data)
                
                # Store in cache
                TestIndicatorPipeline.cache.set_memory(cache_key, result)
                
                return result
        
        # Create indicator instances
        regular_tema = TEMA(period=20)
        cacheable_tema = CacheableTEMA(period=20)
        
        # First calculation - should be similar timing
        start_time = time.time()
        regular_result = regular_tema.calculate(self.trending_data)
        regular_time = time.time() - start_time
        
        start_time = time.time()
        cacheable_result1 = cacheable_tema.calculate(self.trending_data)
        cacheable_time1 = time.time() - start_time
        
        # Verify results are the same
        pd.testing.assert_frame_equal(regular_result, cacheable_result1)
        
        # Second calculation - cacheable should be faster
        start_time = time.time()
        regular_result = regular_tema.calculate(self.trending_data)
        regular_time_repeat = time.time() - start_time
        
        start_time = time.time()
        cacheable_result2 = cacheable_tema.calculate(self.trending_data)
        cacheable_time2 = time.time() - start_time
        
        # Verify second calculation is faster due to caching
        self.assertLess(cacheable_time2, regular_time_repeat)
        self.assertLess(cacheable_time2, cacheable_time1)
        
        # Verify cached results are the same
        pd.testing.assert_frame_equal(cacheable_result1, cacheable_result2)
        
        print(f"First calculation: Regular={regular_time:.6f}s, Cacheable={cacheable_time1:.6f}s")
        print(f"Second calculation: Regular={regular_time_repeat:.6f}s, Cacheable={cacheable_time2:.6f}s")
        print(f"Caching speedup: {regular_time_repeat/cacheable_time2:.2f}x")
        
        print("Caching integration test completed successfully")
    
    def test_ml_feedback_integration(self):
        """Test integration of ML feedback mechanism with indicators."""
        print("\nRunning ML feedback integration test...")
        
        # 1. Calculate various indicators
        data = self.trending_data.copy()
        
        tema = TEMA(period=20)
        dema = DEMA(period=20)
        hull = HullMA(period=20)
        fractal = FractalIndicator()
        
        data = tema.calculate(data)
        data = dema.calculate(data)
        data = hull.calculate(data)
        data = fractal.calculate(data)
        
        # 2. Extract features
        feature_extractor = FeatureExtractor(normalize=True)
        features = feature_extractor.extract_features(data)
        
        # Create a synthetic target for testing
        target = pd.Series(
            np.sign(data['close'].pct_change(5).shift(-5)),
            index=data.index
        )
        
        # Remove NaN values
        features_clean = features.dropna()
        target_clean = target.loc[features_clean.index].dropna()
        
        # Align indices
        common_idx = features_clean.index.intersection(target_clean.index)
        features_clean = features_clean.loc[common_idx]
        target_clean = target_clean.loc[common_idx]
        
        # 3. Select features
        feature_selector = FeatureSelector(methods=[
            ('importance', ImportanceBasedSelector(n_features=10))
        ])
        
        selected_features = feature_selector.select(features_clean, target_clean)
        
        # 4. Train a simple model (just for testing the feedback loop)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(selected_features, target_clean)
        
        # 5. Initialize feedback mechanism
        feedback = IndicatorMLFeedback(
            storage_path=os.path.join(self.temp_dir, 'feedback.pkl')
        )
        
        # Register indicators
        indicators_info = {
            'tema_20': {'class': 'TEMA', 'params': {'period': 20}, 'category': 'trend'},
            'dema_20': {'class': 'DEMA', 'params': {'period': 20}, 'category': 'trend'},
            'hullma_20': {'class': 'HullMA', 'params': {'period': 20}, 'category': 'trend'},
            'fractal': {'class': 'FractalIndicator', 'params': {}, 'category': 'pattern'}
        }
        
        for name, info in indicators_info.items():
            feedback.register_indicator(name, info)
        
        # Get feature importances from model
        feature_importances = dict(zip(
            selected_features.columns,
            model.feature_importances_
        ))
        
        # 6. Update feedback system with model performance
        feedback.update_model_performance({
            'accuracy': 0.65,
            'precision': 0.67,
            'recall': 0.68
        })
        
        feedback.update_feature_importances(feature_importances)
        
        # 7. Get indicator evaluations and recommendations
        evaluations = feedback.evaluate_indicators()
        recommendations = feedback.get_recommendations()
        
        # Verify that evaluations and recommendations were generated
        self.assertGreater(len(evaluations), 0)
        self.assertGreater(len(recommendations), 0)
        
        print(f"Generated evaluations for {len(evaluations)} indicators")
        print(f"Generated {len(recommendations)} optimization recommendations")
        
        # 8. Optimize indicator parameters based on feedback
        # This would typically update indicator parameters for the next iteration
        # We'll just verify that recommendations are in the expected format
        for rec in recommendations:
            self.assertIn('indicator', rec)
            self.assertIn('action', rec)
        
        print("ML feedback integration test completed successfully")
    
    def test_cross_component_integration(self):
        """Test integration between different types of indicators and components."""
        print("\nRunning cross-component integration test...")
        
        # 1. Detect market regime
        regime_classifier = MarketRegimeClassifier()
        current_regime = regime_classifier.classify(self.trending_data)
        
        # 2. Select appropriate indicators based on regime
        if current_regime == MarketRegime.TRENDING_BULLISH:
            # Use trend-following indicators
            indicators = [
                TEMA(period=20),
                HullMA(period=9),
                FractalIndicator()
            ]
        elif current_regime in [MarketRegime.RANGING_TIGHT, MarketRegime.RANGING_VOLATILE]:
            # Use oscillators and pattern recognition
            indicators = [
                HarmonicPatternFinder(),
                GannAngles(),
                ElliottWaveAnalyzer()
            ]
        else:
            # Default set
            indicators = [
                TEMA(period=20),
                FractalIndicator(),
                HarmonicPatternFinder()
            ]
        
        # 3. Apply indicators in sequence
        data = self.trending_data.copy()
        
        for indicator in indicators:
            try:
                data = indicator.calculate(data)
            except Exception as e:
                print(f"Failed to calculate {indicator.__class__.__name__}: {str(e)}")
        
        # 4. Extract features from calculated indicators
        feature_extractor = FeatureExtractor(normalize=True)
        features = feature_extractor.extract_features(data)
        
        # Verify features were created from multiple indicator types
        self.assertGreater(len(features.columns), len(self.trending_data.columns))
        
        # 5. Apply multi-timeframe analysis on a subset of indicators
        multi_tf_indicator = MultiTimeframeIndicator(
            indicator=TEMA(period=20),
            timeframes=['1D']
        )
        
        # Calculate on 1-day data
        multi_tf_result = multi_tf_indicator.calculate(self.trending_data)
        
        # Merge multi-timeframe results
        for col in multi_tf_result.columns:
            if col not in data.columns:
                data[col] = multi_tf_result[col]
        
        # Verify multi-timeframe columns were added
        self.assertIn('tema_20_1D', data.columns)
        
        # 6. Update feature extraction with new indicators
        updated_features = feature_extractor.extract_features(data)
        
        # Verify new features were added
        self.assertGreater(len(updated_features.columns), len(features.columns))
        
        print(f"Cross-component integration test successfully processed {len(data.columns)} columns")
        print(f"Generated {len(updated_features.columns)} features from multiple indicator types")
    
    def test_end_to_end_with_cached_pipeline(self):
        """Test entire pipeline with caching for performance optimization."""
        print("\nRunning end-to-end pipeline with caching test...")
        
        # Define a complete cached indicator pipeline
        class CachedPipeline:
            """A complete indicator pipeline with caching."""
            
            def __init__(self, cache):
                self.cache = cache
                self.indicators = {
                    'tema': TEMA(period=20),
                    'dema': DEMA(period=20),
                    'hullma': HullMA(period=20),
                    'fractal': FractalIndicator(),
                    'alligator': AlligatorIndicator()
                }
                self.feature_extractor = FeatureExtractor(normalize=True)
            
            def process(self, data):
                """Process data through the complete pipeline."""
                # Generate a cache key based on the data
                cache_key = f"pipeline_{hash(tuple(data.index))}"
                
                # Try to get complete result from cache
                cached_result = self.cache.get_memory(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # If not cached, run the full pipeline
                result = data.copy()
                
                # Apply each indicator
                for name, indicator in self.indicators.items():
                    # Try to get indicator result from cache
                    ind_key = f"{name}_{hash(tuple(data.index))}"
                    ind_result = self.cache.get_memory(ind_key)
                    
                    if ind_result is not None:
                        # Merge cached indicator result
                        for col in ind_result.columns:
                            if col not in result.columns:
                                result[col] = ind_result[col]
                    else:
                        # Calculate indicator and cache
                        ind_result = indicator.calculate(data)
                        self.cache.set_memory(ind_key, ind_result)
                        
                        # Merge result
                        for col in ind_result.columns:
                            if col not in result.columns:
                                result[col] = ind_result[col]
                
                # Extract features
                features = self.feature_extractor.extract_features(result)
                
                # Store final result in cache
                pipeline_result = {
                    'indicators': result,
                    'features': features
                }
                self.cache.set_memory(cache_key, pipeline_result)
                
                return pipeline_result
        
        # Create cached pipeline
        pipeline = CachedPipeline(self.cache)
        
        # First run - no cache
        start_time = time.time()
        first_result = pipeline.process(self.trending_data)
        first_run_time = time.time() - start_time
        
        # Second run - should use cache
        start_time = time.time()
        second_result = pipeline.process(self.trending_data)
        second_run_time = time.time() - start_time
        
        # Verify results match
        pd.testing.assert_frame_equal(
            first_result['indicators'], 
            second_result['indicators']
        )
        
        pd.testing.assert_frame_equal(
            first_result['features'], 
            second_result['features']
        )
        
        # Verify caching improved performance
        self.assertLess(second_run_time, first_run_time)
        
        print(f"First run (no cache): {first_run_time:.6f} seconds")
        print(f"Second run (cached): {second_run_time:.6f} seconds")
        print(f"Caching speedup: {first_run_time/second_run_time:.2f}x")
        
        print("End-to-end cached pipeline test completed successfully")


if __name__ == '__main__':
    unittest.main()
