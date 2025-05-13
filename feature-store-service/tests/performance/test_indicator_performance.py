"""
Performance benchmarking tests for technical indicators.
"""
import unittest
import pandas as pd
import numpy as np
import time
import os
import psutil
import gc
import cProfile
import pstats
from io import StringIO
import tempfile
import pickle
from memory_profiler import profile as memory_profile
from functools import wraps
import matplotlib.pyplot as plt
from feature_store_service.indicators.advanced_moving_averages import TEMA, DEMA, HullMA
from feature_store_service.indicators.chart_patterns import ChartPatternRecognizer
from feature_store_service.indicators.gann import GannAngles
from feature_store_service.indicators.fractal_indicators import FractalIndicator
from feature_store_service.indicators.multi_timeframe import MultiTimeframeIndicator
from feature_store_service.indicators.indicator_selection import MarketRegimeClassifier
from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.caching import IndicatorCache


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def timeit(func):
    """Decorator to measure execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'{func.__name__} executed in {execution_time:.6f} seconds')
        return result, execution_time
    return wrapper


def measure_memory(func):
    """Decorator to measure memory usage of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        gc.collect()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024
        result = func(*args, **kwargs)
        gc.collect()
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_used = end_memory - start_memory
        print(f'{func.__name__} used {memory_used:.2f} MB of memory')
        return result, memory_used
    return wrapper


class TestIndicatorPerformance(unittest.TestCase):
    """Base test class for indicator performance benchmarking."""

    def setUp(self):
        """Set up test data with varying sizes."""
        np.random.seed(42)
        self.small_data = self._generate_test_data(500)
        self.medium_data = self._generate_test_data(5000)
        self.large_data = self._generate_test_data(50000)
        self.test_dir = tempfile.mkdtemp()
        self.indicators = {'tema': TEMA(period=20), 'dema': DEMA(period=20),
            'hullma': HullMA(period=20), 'fractal': FractalIndicator(),
            'pattern': ChartPatternRecognizer(), 'gann': GannAngles()}

    @with_exception_handling
    def tearDown(self):
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(self.test_dir)
        except (OSError, IOError):
            pass

    def _generate_test_data(self, n_samples):
        """Generate test OHLCV data with specified size."""
        base = np.cumsum(np.random.normal(0, 1, n_samples))
        noise = np.random.normal(0, 0.5, n_samples)
        trend = np.linspace(0, 5, n_samples)
        price = 100 + 5 * (base + noise + trend)
        date_range = pd.date_range(start='2020-01-01', periods=n_samples,
            freq=pd.Timedelta(minutes=5))
        data = pd.DataFrame({'open': price * (1 + 0.005 * np.random.randn(
            n_samples)), 'high': price * (1 + 0.01 * np.random.randn(
            n_samples)), 'low': price * (1 - 0.01 * np.random.randn(
            n_samples)), 'close': price, 'volume': 1000000 * (1 + 0.2 * np.
            random.randn(n_samples))}, index=date_range)
        data['high'] = np.maximum(np.maximum(data['high'], data['open']),
            data['close'])
        data['low'] = np.minimum(np.minimum(data['low'], data['open']),
            data['close'])
        return data

    def test_indicator_execution_time_scaling(self):
        """Test how indicator calculation time scales with data size."""
        results = {}
        for name, indicator in self.indicators.items():
            print(f'\nBenchmarking {name} indicator:')
            if name == 'pattern' and hasattr(self, 'large_data'):
                continue
            _, small_time = timeit(indicator.calculate)(self.small_data)
            _, medium_time = timeit(indicator.calculate)(self.medium_data)
            if name not in ['pattern', 'gann'] and hasattr(self, 'large_data'):
                _, large_time = timeit(indicator.calculate)(self.large_data)
            else:
                large_time = None
            results[name] = {'small': small_time, 'medium': medium_time,
                'large': large_time}
            if large_time:
                scaling_factor = large_time / small_time
                expected_linear_factor = len(self.large_data) / len(self.
                    small_data)
                print(f'Scaling factor: {scaling_factor:.2f}x')
                print(f'Data size increase: {expected_linear_factor:.2f}x')
                self.assertLess(scaling_factor, expected_linear_factor ** 2,
                    f'{name} scaling is worse than O(n²)')
        print('\nExecution time summary (seconds):')
        for name, times in results.items():
            print(
                f"{name}: small={times['small']:.6f}, medium={times['medium']:.6f}, large={'N/A' if times['large'] is None else times['large']:.6f}"
                )

    def test_indicator_memory_usage(self):
        """Test memory usage of different indicators."""
        results = {}
        for name, indicator in self.indicators.items():
            print(f'\nBenchmarking memory usage for {name} indicator:')
            if name == 'pattern':
                datasets = [self.small_data, self.medium_data]
            else:
                datasets = [self.small_data, self.medium_data, self.large_data]
            dataset_memory = []
            for i, dataset in enumerate(['small', 'medium', 'large']):
                if i < len(datasets):
                    _, memory = measure_memory(indicator.calculate)(datasets[i]
                        )
                    dataset_memory.append(memory)
                else:
                    dataset_memory.append(None)
            results[name] = {'small': dataset_memory[0], 'medium':
                dataset_memory[1], 'large': dataset_memory[2] if len(
                dataset_memory) > 2 else None}
        print('\nMemory usage summary (MB):')
        for name, memory in results.items():
            print(
                f"{name}: small={memory['small']:.2f}, medium={memory['medium']:.2f}, large={'N/A' if memory['large'] is None else memory['large']:.2f}"
                )

    def test_profile_indicator_calculation(self):
        """Profile indicator calculation to identify bottlenecks."""
        for name in ['tema', 'hullma', 'fractal']:
            indicator = self.indicators[name]
            print(f'\nProfiling {name} indicator calculation:')
            profiler = cProfile.Profile()
            profiler.enable()
            indicator.calculate(self.medium_data)
            profiler.disable()
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)
            print(s.getvalue())


class TestIndicatorCaching(unittest.TestCase):
    """Test suite for indicator caching strategies."""

    def setUp(self):
        """Set up test data and caching."""
        np.random.seed(42)
        n_samples = 5000
        base = np.cumsum(np.random.normal(0, 1, n_samples))
        noise = np.random.normal(0, 0.5, n_samples)
        trend = np.linspace(0, 5, n_samples)
        price = 100 + 5 * (base + noise + trend)
        date_range = pd.date_range(start='2020-01-01', periods=n_samples,
            freq='D')
        self.data = pd.DataFrame({'open': price * (1 + 0.005 * np.random.
            randn(n_samples)), 'high': price * (1 + 0.01 * np.random.randn(
            n_samples)), 'low': price * (1 - 0.01 * np.random.randn(
            n_samples)), 'close': price, 'volume': 1000000 * (1 + 0.2 * np.
            random.randn(n_samples))}, index=date_range)
        self.cache_dir = tempfile.mkdtemp()
        self.tema = TEMA(period=20)
        self.tema_cached = TEMA(period=20)
        self.cache = IndicatorCache(cache_dir=self.cache_dir, max_memory_mb
            =100, expire_after_days=30)

    @with_exception_handling
    def tearDown(self):
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
        except (OSError, IOError):
            pass

    def test_memory_caching(self):
        """Test in-memory caching performance."""
        print('\nTesting in-memory caching:')
        start_time = time.time()
        result1 = self.tema.calculate(self.data)
        first_run_time = time.time() - start_time
        print(f'First run (no cache): {first_run_time:.6f} seconds')
        cache_key = f'tema_{self.tema.period}_{hash(tuple(self.data.index))}'
        self.cache.set_memory(cache_key, result1)
        start_time = time.time()
        cached_result = self.cache.get_memory(cache_key)
        memory_cache_time = time.time() - start_time
        print(f'Memory cache retrieval: {memory_cache_time:.6f} seconds')
        pd.testing.assert_frame_equal(result1, cached_result)
        self.assertLess(memory_cache_time, first_run_time)
        print(
            f'Memory cache speedup: {first_run_time / memory_cache_time:.2f}x')

    def test_disk_caching(self):
        """Test disk-based caching performance."""
        print('\nTesting disk-based caching:')
        start_time = time.time()
        result1 = self.tema.calculate(self.data)
        first_run_time = time.time() - start_time
        print(f'First run (no cache): {first_run_time:.6f} seconds')
        cache_key = f'tema_{self.tema.period}_{hash(tuple(self.data.index))}'
        cache_path = os.path.join(self.cache_dir, f'{cache_key}.pkl')
        start_time = time.time()
        with open(cache_path, 'wb') as f:
            pickle.dump(result1, f)
        disk_write_time = time.time() - start_time
        print(f'Disk cache write: {disk_write_time:.6f} seconds')
        start_time = time.time()
        with open(cache_path, 'rb') as f:
            disk_result = pickle.load(f)
        disk_read_time = time.time() - start_time
        print(f'Disk cache read: {disk_read_time:.6f} seconds')
        pd.testing.assert_frame_equal(result1, disk_result)
        total_cached_time = disk_write_time + disk_read_time
        print(f'Total disk cache overhead: {total_cached_time:.6f} seconds')
        self.assertLess(disk_read_time, first_run_time)
        print(f'Disk read speedup: {first_run_time / disk_read_time:.2f}x')

    def test_incremental_calculation(self):
        """Test performance of incremental calculation vs. full calculation."""
        print('\nTesting incremental calculation:')
        start_time = time.time()
        full_result = self.tema.calculate(self.data)
        full_calc_time = time.time() - start_time
        print(f'Full calculation: {full_calc_time:.6f} seconds')
        split_point = int(len(self.data) * 0.95)
        existing_data = self.data.iloc[:split_point]
        new_data = self.data.iloc[split_point:]
        existing_result = self.tema.calculate(existing_data)


        class IncrementalTEMA(TEMA):
            """TEMA with incremental calculation capability."""

            def calculate_incremental(self, existing_data, existing_result,
                new_data):
                """Calculate incrementally using existing results and new data."""
                combined = pd.concat([existing_data.iloc[-self.period * 3:],
                    new_data])
                temp_result = super().calculate(combined)
                new_result = temp_result.iloc[self.period * 3:]
                final_result = pd.concat([existing_result, new_result])
                return final_result
        inc_tema = IncrementalTEMA(period=20)
        start_time = time.time()
        inc_result = inc_tema.calculate_incremental(existing_data,
            existing_result, new_data)
        inc_calc_time = time.time() - start_time
        print(f'Incremental calculation: {inc_calc_time:.6f} seconds')
        pd.testing.assert_frame_equal(full_result, inc_result)
        self.assertLess(inc_calc_time, full_calc_time)
        print(f'Incremental speedup: {full_calc_time / inc_calc_time:.2f}x')

    def test_cache_invalidation(self):
        """Test cache invalidation strategies."""
        result_p20 = self.tema.calculate(self.data)
        cache_key = f'tema_20_{hash(tuple(self.data.index))}'
        self.cache.set_memory(cache_key, result_p20)
        tema_p30 = TEMA(period=30)
        result_p30 = tema_p30.calculate(self.data)
        new_cache_key = f'tema_30_{hash(tuple(self.data.index))}'
        self.cache.set_memory(new_cache_key, result_p30)
        cached_p20 = self.cache.get_memory(cache_key)
        pd.testing.assert_frame_equal(result_p20, cached_p20)
        cached_p30 = self.cache.get_memory(new_cache_key)
        pd.testing.assert_frame_equal(result_p30, cached_p30)
        modified_data = self.data.copy()
        modified_data.iloc[-1, modified_data.columns.get_loc('close')] += 1.0
        mod_cache_key = f'tema_20_{hash(tuple(modified_data.index))}'
        cached_mod = self.cache.get_memory(mod_cache_key)
        self.assertIsNone(cached_mod)


class TestMultiIndicatorPipeline(unittest.TestCase):
    """Test performance of complete indicator pipeline."""

    def setUp(self):
        """Set up test data for pipeline testing."""
        np.random.seed(42)
        n_samples = 5000
        base = np.cumsum(np.random.normal(0, 1, n_samples))
        noise = np.random.normal(0, 0.5, n_samples)
        trend = np.linspace(0, 5, n_samples)
        price = 100 + 5 * (base + noise + trend)
        date_range = pd.date_range(start='2020-01-01', periods=n_samples,
            freq='D')
        self.data = pd.DataFrame({'open': price * (1 + 0.005 * np.random.
            randn(n_samples)), 'high': price * (1 + 0.01 * np.random.randn(
            n_samples)), 'low': price * (1 - 0.01 * np.random.randn(
            n_samples)), 'close': price, 'volume': 1000000 * (1 + 0.2 * np.
            random.randn(n_samples))}, index=date_range)
        self.tema = TEMA(period=20)
        self.dema = DEMA(period=20)
        self.hullma = HullMA(period=20)
        self.fractal = FractalIndicator()
        self.cache_dir = tempfile.mkdtemp()
        self.cache = IndicatorCache(cache_dir=self.cache_dir, max_memory_mb
            =200, expire_after_days=30)

    @with_exception_handling
    def tearDown(self):
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
        except (OSError, IOError):
            pass

    def test_pipeline_execution_strategies(self):
        """Test different strategies for executing indicator pipelines."""
        print('\nTesting indicator pipeline execution strategies:')
        start_time = time.time()
        result1 = self.data.copy()
        result1 = self.tema.calculate(result1)
        result1 = self.dema.calculate(result1)
        result1 = self.hullma.calculate(result1)
        result1 = self.fractal.calculate(result1)
        sequential_time = time.time() - start_time
        print(f'Sequential execution: {sequential_time:.6f} seconds')
        for indicator in [self.tema, self.dema, self.hullma, self.fractal]:
            name = indicator.__class__.__name__.lower()
            result = indicator.calculate(self.data)
            cache_key = f'{name}_{hash(tuple(self.data.index))}'
            self.cache.set_memory(cache_key, result)
        start_time = time.time()
        result2 = self.data.copy()
        for indicator in [self.tema, self.dema, self.hullma, self.fractal]:
            name = indicator.__class__.__name__.lower()
            cache_key = f'{name}_{hash(tuple(self.data.index))}'
            cached_result = self.cache.get_memory(cache_key)
            for col in cached_result.columns:
                if col not in result2.columns:
                    result2[col] = cached_result[col]
        cached_time = time.time() - start_time
        print(f'Cached execution: {cached_time:.6f} seconds')
        for col in result1.columns:
            if col not in self.data.columns:
                pd.testing.assert_series_equal(result1[col].dropna(),
                    result2[col].dropna())
        speedup = sequential_time / cached_time
        print(f'Cache speedup factor: {speedup:.2f}x')
        self.assertGreater(speedup, 1.0)

    def test_multi_timeframe_pipeline(self):
        """Test performance of multi-timeframe indicator pipeline."""
        print('\nTesting multi-timeframe pipeline performance:')
        multi_tema = MultiTimeframeIndicator(indicator=self.tema,
            timeframes=['1D', '5D', '20D'])
        start_time = time.time()
        multi_result = multi_tema.calculate(self.data)
        multi_time = time.time() - start_time
        print(f'Multi-timeframe calculation: {multi_time:.6f} seconds')
        start_time = time.time()
        result_1d = self.tema.calculate(self.data)
        data_5d = self.data.resample('5D').agg({'open': 'first', 'high':
            'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        result_5d = self.tema.calculate(data_5d)
        data_20d = self.data.resample('20D').agg({'open': 'first', 'high':
            'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        result_20d = self.tema.calculate(data_20d)
        sequential_time = time.time() - start_time
        print(
            f'Sequential timeframe calculation: {sequential_time:.6f} seconds')
        self.assertLess(multi_time, sequential_time)
        print(
            f'Multi-timeframe optimization factor: {sequential_time / multi_time:.2f}x'
            )

    @with_exception_handling
    def test_parallel_execution(self):
        """Test parallel execution of indicators."""
        print('\nTesting parallel indicator execution:')
        try:
            import multiprocessing
        except ImportError:
            print(
                'Multiprocessing not available, skipping parallel execution test'
                )
            return

        def calculate_parallel(indicators, data):
            """Calculate indicators in parallel using multiprocessing."""
            with multiprocessing.Pool(processes=min(len(indicators),
                multiprocessing.cpu_count())) as pool:
                tasks = [(indicator, data) for indicator in indicators]
                results = pool.starmap(lambda ind, d: ind.calculate(d), tasks)
                combined = data.copy()
                for result in results:
                    for col in result.columns:
                        if col not in combined.columns:
                            combined[col] = result[col]
                return combined
        indicators = [self.tema, self.dema, self.hullma, self.fractal]
        start_time = time.time()
        sequential_result = self.data.copy()
        for indicator in indicators:
            sequential_result = indicator.calculate(sequential_result)
        sequential_time = time.time() - start_time
        print(f'Sequential execution: {sequential_time:.6f} seconds')
        start_time = time.time()
        parallel_result = calculate_parallel(indicators, self.data)
        parallel_time = time.time() - start_time
        print(f'Parallel execution: {parallel_time:.6f} seconds')
        for col in sequential_result.columns:
            if col not in self.data.columns:
                pd.testing.assert_series_equal(sequential_result[col].
                    dropna(), parallel_result[col].dropna())
        print(
            f'Parallel vs. Sequential: {sequential_time / parallel_time:.2f}x')
        if multiprocessing.cpu_count() > 1 and len(self.data) > 1000:
            self.assertLess(parallel_time, sequential_time * 1.1)

    def test_memory_optimization_strategies(self):
        """Test memory optimization strategies for indicator pipelines."""
        print('\nTesting memory optimization strategies:')

        def full_memory_strategy():
    """
    Full memory strategy.
    
    """

            result = self.data.copy()
            result = self.tema.calculate(result)
            result = self.dema.calculate(result)
            result = self.hullma.calculate(result)
            result = self.fractal.calculate(result)
            return result

        def optimized_memory_strategy():
    """
    Optimized memory strategy.
    
    """

            result = self.data[['close']].copy()
            tema_result = self.tema.calculate(self.data)
            result['tema_20'] = tema_result['tema_20']
            dema_result = self.dema.calculate(self.data)
            result['dema_20'] = dema_result['dema_20']
            hull_result = self.hullma.calculate(self.data)
            result['hullma_20'] = hull_result['hullma_20']
            fractal_data = self.data[['open', 'high', 'low', 'close']].copy()
            fractal_result = self.fractal.calculate(fractal_data)
            result['fractal_bullish'] = fractal_result['fractal_bullish']
            result['fractal_bearish'] = fractal_result['fractal_bearish']
            return result
        _, full_memory = measure_memory(full_memory_strategy)()
        print(f'Full memory strategy usage: {full_memory:.2f} MB')
        _, optimized_memory = measure_memory(optimized_memory_strategy)()
        print(f'Optimized memory strategy usage: {optimized_memory:.2f} MB')
        self.assertLess(optimized_memory, full_memory)
        print(f'Memory reduction factor: {full_memory / optimized_memory:.2f}x'
            )


if __name__ == '__main__':
    unittest.main()
