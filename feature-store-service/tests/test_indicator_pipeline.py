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
from core.advanced_moving_averages import TEMA, DEMA, HullMA
from core.chart_patterns import ChartPatternRecognizer, HarmonicPatternFinder
from feature_store_service.indicators.gann import GannAngles, GannSquare9
from core.fractal_indicators import FractalIndicator, AlligatorIndicator, ElliottWaveAnalyzer
from core.multi_timeframe import MultiTimeframeIndicator, TimeframeComparison, TimeframeConfluenceScanner
from core.indicator_selection import MarketRegimeClassifier, IndicatorPerformanceMetrics, IndicatorSelectionEngine, MarketRegime
from feature_store_service.caching import IndicatorCache
from core.base_indicator import BaseIndicator, IndicatorRegistry
from adapters.ml_integration_adapter import MLFeatureConsumerAdapter
from adapters.advanced_indicator_adapter import AdvancedIndicatorAdapter, FibonacciAnalyzerAdapter
from common_lib.ml.feature_interfaces import FeatureType, FeatureScope, SelectionMethod
from common_lib.indicators.indicator_interfaces import IBaseIndicator, IAdvancedIndicator, IFibonacciAnalyzer, IndicatorCategory, IIndicatorAdapter


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TestIndicatorPipeline(unittest.TestCase):
    """Test suite for the entire indicator pipeline integration."""

    @classmethod
    def setUpClass(cls):
        """Set up data and components for all tests."""
        cls.trending_data = cls._generate_trending_data()
        cls.ranging_data = cls._generate_ranging_data()
        cls.volatile_data = cls._generate_volatile_data()
        cls.multi_timeframe_data = cls._generate_multi_timeframe_data()
        cls.temp_dir = tempfile.mkdtemp()
        cls.cache = IndicatorCache(cache_dir=os.path.join(cls.temp_dir,
            'indicator_cache'), max_memory_mb=200, expire_after_days=7)
        cls.registry = IndicatorRegistry()
        cls._register_all_indicators(cls.registry)

    @classmethod
    @with_exception_handling
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
        date_range = pd.date_range(start='2023-01-01', periods=n_samples,
            freq='D')
        trend = np.linspace(0, 20, n_samples)
        noise = np.random.normal(0, 1, n_samples)
        cycles = 2 * np.sin(np.linspace(0, 6 * np.pi, n_samples))
        price = 100 + trend + noise + cycles
        return pd.DataFrame({'open': price * (1 + 0.003 * np.random.randn(
            n_samples)), 'high': price * (1 + 0.006 * np.random.randn(
            n_samples)), 'low': price * (1 - 0.006 * np.random.randn(
            n_samples)), 'close': price, 'volume': 1000000 * (1 + 0.2 * np.
            random.randn(n_samples))}, index=date_range)

    @classmethod
    def _generate_ranging_data(cls, n_samples=500):
        """Generate data in a trading range."""
        np.random.seed(43)
        date_range = pd.date_range(start='2023-01-01', periods=n_samples,
            freq='D')
        base = 100
        amplitude = 10
        cycles = amplitude * np.sin(np.linspace(0, 10 * np.pi, n_samples))
        noise = np.random.normal(0, 1, n_samples)
        price = base + cycles + noise
        return pd.DataFrame({'open': price * (1 + 0.003 * np.random.randn(
            n_samples)), 'high': price * (1 + 0.006 * np.random.randn(
            n_samples)), 'low': price * (1 - 0.006 * np.random.randn(
            n_samples)), 'close': price, 'volume': 1000000 * (1 + 0.2 * np.
            random.randn(n_samples))}, index=date_range)

    @classmethod
    def _generate_volatile_data(cls, n_samples=500):
        """Generate data with high volatility."""
        np.random.seed(44)
        date_range = pd.date_range(start='2023-01-01', periods=n_samples,
            freq='D')
        base = np.cumsum(np.random.normal(0, 0.3, n_samples))
        volatility = np.ones(n_samples)
        volatility[100:150] = 3.0
        volatility[300:350] = 4.0
        noise = np.random.normal(0, 1, n_samples) * volatility
        price = 100 + base + noise
        return pd.DataFrame({'open': price * (1 + 0.005 * np.random.randn(
            n_samples) * volatility), 'high': price * (1 + 0.01 * np.random
            .randn(n_samples) * volatility), 'low': price * (1 - 0.01 * np.
            random.randn(n_samples) * volatility), 'close': price, 'volume':
            1000000 * (1 + 0.3 * np.random.randn(n_samples) * volatility)},
            index=date_range)

    @classmethod
    def _generate_multi_timeframe_data(cls):
        """Generate data for multiple timeframes."""
        np.random.seed(45)
        minutes_per_day = 1440
        days = 7
        n_samples = minutes_per_day * days
        start_date = datetime(2023, 1, 1)
        date_range_1min = [(start_date + timedelta(minutes=i)) for i in
            range(n_samples)]
        price = 100
        prices_1min = [price]
        for i in range(1, n_samples):
            minute_of_day = i % minutes_per_day
            hour_of_day = minute_of_day // 60
            if hour_of_day < 2 or hour_of_day > 21:
                volatility = 0.02
            else:
                volatility = 0.005
            if 9 <= hour_of_day < 16:
                drift = 0.001
            else:
                drift = -0.0002
            price += np.random.normal(drift, volatility)
            prices_1min.append(price)
        data_1min = pd.DataFrame({'open': [(p - np.random.uniform(0, 0.01)) for
            p in prices_1min], 'high': [(p + np.random.uniform(0, 0.02)) for
            p in prices_1min], 'low': [(p - np.random.uniform(0, 0.02)) for
            p in prices_1min], 'close': prices_1min, 'volume': [np.random.
            randint(100, 1000) for _ in range(n_samples)]}, index=
            date_range_1min)
        data_5min = cls._resample_ohlc(data_1min, '5T')
        data_15min = cls._resample_ohlc(data_1min, '15T')
        data_1hour = cls._resample_ohlc(data_1min, '1H')
        data_4hour = cls._resample_ohlc(data_1min, '4H')
        data_1day = cls._resample_ohlc(data_1min, '1D')
        return {'1T': data_1min, '5T': data_5min, '15T': data_15min, '1H':
            data_1hour, '4H': data_4hour, '1D': data_1day}

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
        registry.register('tema', TEMA)
        registry.register('dema', DEMA)
        registry.register('hullma', HullMA)
        registry.register('pattern_recognizer', ChartPatternRecognizer)
        registry.register('harmonic_patterns', HarmonicPatternFinder)
        registry.register('gann_angles', GannAngles)
        registry.register('gann_square9', GannSquare9)
        registry.register('fractal', FractalIndicator)
        registry.register('alligator', AlligatorIndicator)
        registry.register('elliott_wave', ElliottWaveAnalyzer)
        return registry

    @with_exception_handling
    def test_end_to_end_indicator_pipeline(self):
        """Test the complete indicator pipeline from data to features."""
        print('\nRunning end-to-end indicator pipeline test...')
        regime_classifier = MarketRegimeClassifier()
        trending_regime = regime_classifier.classify(self.trending_data)
        ranging_regime = regime_classifier.classify(self.ranging_data)
        self.assertEqual(trending_regime, MarketRegime.TRENDING_BULLISH)
        self.assertEqual(ranging_regime, MarketRegime.RANGING_TIGHT)
        selection_engine = IndicatorSelectionEngine(indicators_registry=
            self.registry.get_all_indicators(), performance_history_path=os
            .path.join(self.temp_dir, 'performance.json'), config_path=os.
            path.join(self.temp_dir, 'config.json'))
        trending_indicators = selection_engine.select_indicators(self.
            trending_data)
        self.assertGreater(len(trending_indicators), 0)
        print(
            f'Selected {len(trending_indicators)} indicators for trending regime'
            )
        results = self.trending_data.copy()
        calculated_indicators = []
        for name, indicator in trending_indicators.items():
            try:
                import time
                start_time = time.time()
                results = indicator.calculate(results)
                calc_time = time.time() - start_time
                calculated_indicators.append(name)
                print(f'Calculated {name} in {calc_time:.4f} seconds')
            except Exception as e:
                print(f'Failed to calculate {name}: {str(e)}')
        self.assertGreater(len(calculated_indicators), 0)
        ml_feature_consumer = MLFeatureConsumerAdapter()
        feature_definitions = ml_feature_consumer.get_required_features()
        features = results.copy()
        self.assertGreater(len(features.columns), len(self.trending_data.
            columns))
        print(f'Using {len(features.columns)} features from indicators')
        target = pd.Series(np.sign(self.trending_data['close'].pct_change(5
            ).shift(-5)), index=self.trending_data.index)
        X, y = ml_feature_consumer.prepare_model_inputs(features=features,
            target_column=None, model_id='test_model')
        y = target.loc[X.index]
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        print(f'Prepared ML inputs with shape {X.shape}')
        print('End-to-end indicator pipeline test completed successfully')

    def test_multi_timeframe_integration(self):
        """Test integration of multi-timeframe analysis components."""
        print('\nRunning multi-timeframe integration test...')
        data_1hour = self.multi_timeframe_data['1H']
        data_4hour = self.multi_timeframe_data['4H']
        data_1day = self.multi_timeframe_data['1D']
        multi_tema = MultiTimeframeIndicator(indicator=TEMA(period=20),
            timeframes=['1H', '4H', '1D'])
        results = multi_tema.calculate(data_1hour)
        self.assertIn('tema_20_1H', results.columns)
        self.assertIn('tema_20_4H', results.columns)
        self.assertIn('tema_20_1D', results.columns)
        tf_comparison = TimeframeComparison(indicator=TEMA(period=20),
            timeframes=['1H', '4H', '1D'])
        comparison_results = tf_comparison.calculate_trend_alignment(data_1hour
            )
        self.assertIn('trend_alignment', comparison_results.columns)
        confluence_scanner = TimeframeConfluenceScanner(indicators=[TEMA(
            period=20), DEMA(period=20)], timeframes=['1H', '4H', '1D'])
        confluence_results = confluence_scanner.scan_for_confluence(data_1hour)
        self.assertIn('bullish_confluence', confluence_results.columns)
        self.assertIn('bearish_confluence', confluence_results.columns)
        signal_results = confluence_scanner.generate_signals(data_1hour)
        self.assertIn('confluence_signal', signal_results.columns)
        unique_signals = signal_results['confluence_signal'].dropna().unique()
        for signal in unique_signals:
            self.assertIn(signal, [-1, 0, 1])
        print('Multi-timeframe integration test completed successfully')

    def test_caching_integration(self):
        """Test integration of caching system with indicators."""
        print('\nRunning caching integration test...')


        class CacheableTEMA(TEMA):
            """TEMA with caching capability."""

            def calculate(self, data):
                """Calculate with caching support."""
                cache_key = f'tema_{self.period}_{hash(tuple(data.index))}'
                cached_result = TestIndicatorPipeline.cache.get_memory(
                    cache_key)
                if cached_result is not None:
                    return cached_result
                result = super().calculate(data)
                TestIndicatorPipeline.cache.set_memory(cache_key, result)
                return result
        regular_tema = TEMA(period=20)
        cacheable_tema = CacheableTEMA(period=20)
        start_time = time.time()
        regular_result = regular_tema.calculate(self.trending_data)
        regular_time = time.time() - start_time
        start_time = time.time()
        cacheable_result1 = cacheable_tema.calculate(self.trending_data)
        cacheable_time1 = time.time() - start_time
        pd.testing.assert_frame_equal(regular_result, cacheable_result1)
        start_time = time.time()
        regular_result = regular_tema.calculate(self.trending_data)
        regular_time_repeat = time.time() - start_time
        start_time = time.time()
        cacheable_result2 = cacheable_tema.calculate(self.trending_data)
        cacheable_time2 = time.time() - start_time
        self.assertLess(cacheable_time2, regular_time_repeat)
        self.assertLess(cacheable_time2, cacheable_time1)
        pd.testing.assert_frame_equal(cacheable_result1, cacheable_result2)
        print(
            f'First calculation: Regular={regular_time:.6f}s, Cacheable={cacheable_time1:.6f}s'
            )
        print(
            f'Second calculation: Regular={regular_time_repeat:.6f}s, Cacheable={cacheable_time2:.6f}s'
            )
        print(f'Caching speedup: {regular_time_repeat / cacheable_time2:.2f}x')
        print('Caching integration test completed successfully')

    def test_ml_feedback_integration(self):
        """Test integration of ML feedback mechanism with indicators."""
        print('\nRunning ML feedback integration test...')
        data = self.trending_data.copy()
        tema = TEMA(period=20)
        dema = DEMA(period=20)
        hull = HullMA(period=20)
        fractal = FractalIndicator()
        data = tema.calculate(data)
        data = dema.calculate(data)
        data = hull.calculate(data)
        data = fractal.calculate(data)
        ml_feature_consumer = MLFeatureConsumerAdapter()
        features = data.copy()
        target = pd.Series(np.sign(data['close'].pct_change(5).shift(-5)),
            index=data.index)
        features_clean = features.dropna()
        target_clean = target.loc[features_clean.index].dropna()
        common_idx = features_clean.index.intersection(target_clean.index)
        features_clean = features_clean.loc[common_idx]
        target_clean = target_clean.loc[common_idx]
        X, _ = ml_feature_consumer.prepare_model_inputs(features=
            features_clean, model_id='test_model')
        feature_importances = (ml_feature_consumer.
            get_feature_importance_feedback(model_id='test_model', features
            =list(X.columns)))
        sorted_features = sorted(feature_importances.items(), key=lambda x:
            x[1], reverse=True)[:10]
        selected_feature_names = [f[0] for f in sorted_features]
        selected_features = X[selected_feature_names]
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(selected_features, target_clean)
        feedback = IndicatorMLFeedback(storage_path=os.path.join(self.
            temp_dir, 'feedback.pkl'))
        indicators_info = {'tema_20': {'class': 'TEMA', 'params': {'period':
            20}, 'category': 'trend'}, 'dema_20': {'class': 'DEMA',
            'params': {'period': 20}, 'category': 'trend'}, 'hullma_20': {
            'class': 'HullMA', 'params': {'period': 20}, 'category':
            'trend'}, 'fractal': {'class': 'FractalIndicator', 'params': {},
            'category': 'pattern'}}
        for name, info in indicators_info.items():
            feedback.register_indicator(name, info)
        feature_importances = dict(zip(selected_features.columns, model.
            feature_importances_))
        feedback.update_model_performance({'accuracy': 0.65, 'precision': 
            0.67, 'recall': 0.68})
        feedback.update_feature_importances(feature_importances)
        evaluations = feedback.evaluate_indicators()
        recommendations = feedback.get_recommendations()
        self.assertGreater(len(evaluations), 0)
        self.assertGreater(len(recommendations), 0)
        print(f'Generated evaluations for {len(evaluations)} indicators')
        print(f'Generated {len(recommendations)} optimization recommendations')
        for rec in recommendations:
            self.assertIn('indicator', rec)
            self.assertIn('action', rec)
        print('ML feedback integration test completed successfully')

    @with_exception_handling
    def test_cross_component_integration(self):
        """Test integration between different types of indicators and components."""
        print('\nRunning cross-component integration test...')
        regime_classifier = MarketRegimeClassifier()
        current_regime = regime_classifier.classify(self.trending_data)
        if current_regime == MarketRegime.TRENDING_BULLISH:
            indicators = [TEMA(period=20), HullMA(period=9), FractalIndicator()
                ]
        elif current_regime in [MarketRegime.RANGING_TIGHT, MarketRegime.
            RANGING_VOLATILE]:
            indicators = [HarmonicPatternFinder(), GannAngles(),
                ElliottWaveAnalyzer()]
        else:
            indicators = [TEMA(period=20), FractalIndicator(),
                HarmonicPatternFinder()]
        data = self.trending_data.copy()
        for indicator in indicators:
            try:
                data = indicator.calculate(data)
            except Exception as e:
                print(
                    f'Failed to calculate {indicator.__class__.__name__}: {str(e)}'
                    )
        feature_extractor = FeatureExtractor(normalize=True)
        features = feature_extractor.extract_features(data)
        self.assertGreater(len(features.columns), len(self.trending_data.
            columns))
        multi_tf_indicator = MultiTimeframeIndicator(indicator=TEMA(period=
            20), timeframes=['1D'])
        multi_tf_result = multi_tf_indicator.calculate(self.trending_data)
        for col in multi_tf_result.columns:
            if col not in data.columns:
                data[col] = multi_tf_result[col]
        self.assertIn('tema_20_1D', data.columns)
        updated_features = feature_extractor.extract_features(data)
        self.assertGreater(len(updated_features.columns), len(features.columns)
            )
        print(
            f'Cross-component integration test successfully processed {len(data.columns)} columns'
            )
        print(
            f'Generated {len(updated_features.columns)} features from multiple indicator types'
            )

    def test_end_to_end_with_cached_pipeline(self):
        """Test entire pipeline with caching for performance optimization."""
        print('\nRunning end-to-end pipeline with caching test...')


        class CachedPipeline:
            """A complete indicator pipeline with caching."""

            def __init__(self, cache):
    """
      init  .
    
    Args:
        cache: Description of cache
    
    """

                self.cache = cache
                self.indicators = {'tema': TEMA(period=20), 'dema': DEMA(
                    period=20), 'hullma': HullMA(period=20), 'fractal':
                    FractalIndicator(), 'alligator': AlligatorIndicator()}
                self.feature_extractor = FeatureExtractor(normalize=True)

            def process(self, data):
                """Process data through the complete pipeline."""
                cache_key = f'pipeline_{hash(tuple(data.index))}'
                cached_result = self.cache.get_memory(cache_key)
                if cached_result is not None:
                    return cached_result
                result = data.copy()
                for name, indicator in self.indicators.items():
                    ind_key = f'{name}_{hash(tuple(data.index))}'
                    ind_result = self.cache.get_memory(ind_key)
                    if ind_result is not None:
                        for col in ind_result.columns:
                            if col not in result.columns:
                                result[col] = ind_result[col]
                    else:
                        ind_result = indicator.calculate(data)
                        self.cache.set_memory(ind_key, ind_result)
                        for col in ind_result.columns:
                            if col not in result.columns:
                                result[col] = ind_result[col]
                features = self.feature_extractor.extract_features(result)
                pipeline_result = {'indicators': result, 'features': features}
                self.cache.set_memory(cache_key, pipeline_result)
                return pipeline_result
        pipeline = CachedPipeline(self.cache)
        start_time = time.time()
        first_result = pipeline.process(self.trending_data)
        first_run_time = time.time() - start_time
        start_time = time.time()
        second_result = pipeline.process(self.trending_data)
        second_run_time = time.time() - start_time
        pd.testing.assert_frame_equal(first_result['indicators'],
            second_result['indicators'])
        pd.testing.assert_frame_equal(first_result['features'],
            second_result['features'])
        self.assertLess(second_run_time, first_run_time)
        print(f'First run (no cache): {first_run_time:.6f} seconds')
        print(f'Second run (cached): {second_run_time:.6f} seconds')
        print(f'Caching speedup: {first_run_time / second_run_time:.2f}x')
        print('End-to-end cached pipeline test completed successfully')


if __name__ == '__main__':
    unittest.main()
