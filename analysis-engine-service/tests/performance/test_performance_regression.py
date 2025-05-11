"""
Performance Regression Tests

This module provides automated performance regression tests for the Analysis Engine.
It compares the performance of the current implementation with baseline measurements
to detect performance regressions.
"""
import os
import sys
import json
import time
import logging
import unittest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.utils.test_data_generator import generate_test_data
from tests.utils.test_server import TestServer
from tests.utils.test_client import TestClient
from analysis_engine.multi_asset.optimized_confluence_detector import OptimizedConfluenceDetector
from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
from analysis_engine.utils.adaptive_cache_manager import AdaptiveCacheManager
from analysis_engine.utils.optimized_parallel_processor import OptimizedParallelProcessor
from analysis_engine.utils.memory_optimized_dataframe import MemoryOptimizedDataFrame
from analysis_engine.utils.distributed_tracing import DistributedTracer
from analysis_engine.utils.gpu_accelerator import GPUAccelerator
from analysis_engine.utils.predictive_cache_manager import PredictiveCacheManager
from analysis_engine.ml.pattern_recognition_model import PatternRecognitionModel
from analysis_engine.ml.price_prediction_model import PricePredictionModel
from analysis_engine.ml.ml_confluence_detector import MLConfluenceDetector
from analysis_engine.ml.model_manager import ModelManager

class PerformanceRegressionTests(unittest.TestCase):
    """Performance regression tests for the Analysis Engine."""

    @classmethod
    def set_up_class(cls):
        """Set up the test environment."""
        cls.test_data = generate_test_data(symbols=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'], timeframes=['H1', 'H4', 'D1'], days=30)
        cls.baseline = cls._load_baseline()
        os.makedirs('performance_results', exist_ok=True)

    @classmethod
    def _load_baseline(cls) -> Dict[str, Any]:
        """
        Load baseline measurements.
        
        Returns:
            Baseline measurements
        """
        baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline.json')
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f'Baseline file not found: {baseline_path}')
            return {}

    def _save_baseline(self, baseline: Dict[str, Any]):
        """
        Save baseline measurements.
        
        Args:
            baseline: Baseline measurements
        """
        baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline.json')
        with open(baseline_path, 'w') as f:
            json.dump(baseline, f, indent=2)

    def _compare_with_baseline(self, component: str, operation: str, current: Dict[str, Any], threshold: float=0.1) -> Tuple[bool, Dict[str, Any]]:
        """
        Compare current measurements with baseline.
        
        Args:
            component: Component name
            operation: Operation name
            current: Current measurements
            threshold: Threshold for regression detection
            
        Returns:
            Tuple of (regression detected, comparison results)
        """
        if not self.baseline or component not in self.baseline or operation not in self.baseline[component]:
            logger.warning(f'No baseline found for {component}.{operation}')
            return (False, {})
        baseline = self.baseline[component][operation]
        comparison = {}
        regression_detected = False
        for metric, value in current.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                if isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
                    if baseline_value == 0:
                        diff_pct = float('inf') if value > 0 else 0
                    else:
                        diff_pct = (value - baseline_value) / baseline_value
                    comparison[metric] = {'current': value, 'baseline': baseline_value, 'diff': value - baseline_value, 'diff_pct': diff_pct}
                    if metric.endswith('_time') and diff_pct > threshold:
                        regression_detected = True
                        logger.warning(f'Performance regression detected in {component}.{operation}.{metric}: {diff_pct:.2%}')
            else:
                comparison[metric] = {'current': value, 'baseline': None, 'diff': None, 'diff_pct': None}
        return (regression_detected, comparison)

    def _update_baseline(self, component: str, operation: str, measurements: Dict[str, Any]):
        """
        Update baseline measurements.
        
        Args:
            component: Component name
            operation: Operation name
            measurements: Measurements to update
        """
        if component not in self.baseline:
            self.baseline[component] = {}
        self.baseline[component][operation] = measurements
        self._save_baseline(self.baseline)
        logger.info(f'Baseline updated for {component}.{operation}')

    def _plot_comparison(self, component: str, operation: str, comparison: Dict[str, Dict[str, Any]]):
        """
        Plot comparison results.
        
        Args:
            component: Component name
            operation: Operation name
            comparison: Comparison results
        """
        time_metrics = {k: v for k, v in comparison.items() if k.endswith('_time') and v['baseline'] is not None}
        if not time_metrics:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = list(time_metrics.keys())
        current_values = [time_metrics[m]['current'] for m in metrics]
        baseline_values = [time_metrics[m]['baseline'] for m in metrics]
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width / 2, baseline_values, width, label='Baseline')
        ax.bar(x + width / 2, current_values, width, label='Current')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'{component}.{operation} Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        for i, v in enumerate(baseline_values):
            ax.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center')
        for i, v in enumerate(current_values):
            ax.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center')
        plt.tight_layout()
        plt.savefig(f'performance_results/{component}_{operation}_comparison.png')
        plt.close()

    def test_optimized_confluence_detector(self):
        """Test OptimizedConfluenceDetector performance."""
        logger.info('Testing OptimizedConfluenceDetector performance...')
        correlation_service = MockCorrelationService()
        currency_strength_analyzer = CurrencyStrengthAnalyzer()
        detector = OptimizedConfluenceDetector(correlation_service=correlation_service, currency_strength_analyzer=currency_strength_analyzer, correlation_threshold=0.7, lookback_periods=20, cache_ttl_minutes=60, max_workers=4)
        symbol = 'EURUSD'
        timeframe = 'H1'
        signal_type = 'trend'
        signal_direction = 'bullish'
        price_data = {symbol: self.test_data[symbol][timeframe]}
        related_pairs = asyncio.run(detector.find_related_pairs(symbol))
        for pair in related_pairs.keys():
            if pair in self.test_data and timeframe in self.test_data[pair]:
                price_data[pair] = self.test_data[pair][timeframe]
        detector.cache_manager.clear()
        start_time = time.time()
        result = detector.detect_confluence_optimized(symbol=symbol, price_data=price_data, signal_type=signal_type, signal_direction=signal_direction, related_pairs=related_pairs)
        cold_time = time.time() - start_time
        start_time = time.time()
        result = detector.detect_confluence_optimized(symbol=symbol, price_data=price_data, signal_type=signal_type, signal_direction=signal_direction, related_pairs=related_pairs)
        warm_time = time.time() - start_time
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024
        measurements = {'cold_time': cold_time, 'warm_time': warm_time, 'memory_usage': memory_usage, 'confirmation_count': result['confirmation_count'], 'contradiction_count': result['contradiction_count'], 'confluence_score': result['confluence_score']}
        regression_detected, comparison = self._compare_with_baseline(component='OptimizedConfluenceDetector', operation='detect_confluence_optimized', current=measurements)
        self._plot_comparison(component='OptimizedConfluenceDetector', operation='detect_confluence_optimized', comparison=comparison)
        if os.environ.get('UPDATE_BASELINE') == '1':
            self._update_baseline(component='OptimizedConfluenceDetector', operation='detect_confluence_optimized', measurements=measurements)
        self.assertFalse(regression_detected, 'Performance regression detected')

    def test_ml_confluence_detector(self):
        """Test MLConfluenceDetector performance."""
        logger.info('Testing MLConfluenceDetector performance...')
        correlation_service = MockCorrelationService()
        currency_strength_analyzer = CurrencyStrengthAnalyzer()
        model_manager = ModelManager(model_dir='models', use_gpu=False, correlation_service=correlation_service, currency_strength_analyzer=currency_strength_analyzer)
        detector = model_manager.load_ml_confluence_detector()
        symbol = 'EURUSD'
        timeframe = 'H1'
        signal_type = 'trend'
        signal_direction = 'bullish'
        price_data = {symbol: self.test_data[symbol][timeframe]}
        related_pairs = asyncio.run(detector.find_related_pairs(symbol))
        for pair in related_pairs.keys():
            if pair in self.test_data and timeframe in self.test_data[pair]:
                price_data[pair] = self.test_data[pair][timeframe]
        detector.cache_manager.clear()
        start_time = time.time()
        result = detector.detect_confluence_ml(symbol=symbol, price_data=price_data, signal_type=signal_type, signal_direction=signal_direction, related_pairs=related_pairs)
        cold_time = time.time() - start_time
        start_time = time.time()
        result = detector.detect_confluence_ml(symbol=symbol, price_data=price_data, signal_type=signal_type, signal_direction=signal_direction, related_pairs=related_pairs)
        warm_time = time.time() - start_time
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024
        measurements = {'cold_time': cold_time, 'warm_time': warm_time, 'memory_usage': memory_usage, 'confirmation_count': result['confirmation_count'], 'contradiction_count': result['contradiction_count'], 'confluence_score': result['confluence_score'], 'pattern_score': result['pattern_score'], 'prediction_score': result['prediction_score']}
        regression_detected, comparison = self._compare_with_baseline(component='MLConfluenceDetector', operation='detect_confluence_ml', current=measurements)
        self._plot_comparison(component='MLConfluenceDetector', operation='detect_confluence_ml', comparison=comparison)
        if os.environ.get('UPDATE_BASELINE') == '1':
            self._update_baseline(component='MLConfluenceDetector', operation='detect_confluence_ml', measurements=measurements)
        self.assertFalse(regression_detected, 'Performance regression detected')

class MockCorrelationService:
    """Mock correlation service for testing."""

    async def get_all_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Get all correlations between pairs.
        
        Returns:
            Dictionary mapping pairs to dictionaries of correlations
        """
        return {'EURUSD': {'GBPUSD': 0.85, 'AUDUSD': 0.75, 'USDCAD': -0.65, 'USDJPY': -0.55, 'EURGBP': 0.62, 'EURJPY': 0.78}, 'GBPUSD': {'EURUSD': 0.85, 'AUDUSD': 0.7, 'USDCAD': -0.6, 'USDJPY': -0.5, 'EURGBP': -0.58, 'GBPJPY': 0.75}, 'USDJPY': {'EURUSD': -0.55, 'GBPUSD': -0.5, 'AUDUSD': -0.45, 'USDCAD': 0.4, 'EURJPY': 0.65, 'GBPJPY': 0.7}, 'AUDUSD': {'EURUSD': 0.75, 'GBPUSD': 0.7, 'USDCAD': -0.55, 'USDJPY': -0.45}}
if __name__ == '__main__':
    unittest.main()