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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import test utilities
from tests.utils.test_data_generator import generate_test_data
from tests.utils.test_server import TestServer
from tests.utils.test_client import TestClient

# Import components for direct testing
from analysis_engine.multi_asset.optimized_confluence_detector import OptimizedConfluenceDetector
from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
from analysis_engine.utils.adaptive_cache_manager import AdaptiveCacheManager
from analysis_engine.utils.optimized_parallel_processor import OptimizedParallelProcessor
from analysis_engine.utils.memory_optimized_dataframe import MemoryOptimizedDataFrame
from analysis_engine.utils.distributed_tracing import DistributedTracer
from analysis_engine.utils.gpu_accelerator import GPUAccelerator
from analysis_engine.utils.predictive_cache_manager import PredictiveCacheManager

# Import ML components
from analysis_engine.ml.pattern_recognition_model import PatternRecognitionModel
from analysis_engine.ml.price_prediction_model import PricePredictionModel
from analysis_engine.ml.ml_confluence_detector import MLConfluenceDetector
from analysis_engine.ml.model_manager import ModelManager

class PerformanceRegressionTests(unittest.TestCase):
    """Performance regression tests for the Analysis Engine."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Generate test data
        cls.test_data = generate_test_data(
            symbols=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
            timeframes=["H1", "H4", "D1"],
            days=30
        )
        
        # Load baseline measurements
        cls.baseline = cls._load_baseline()
        
        # Create output directory
        os.makedirs("performance_results", exist_ok=True)
    
    @classmethod
    def _load_baseline(cls) -> Dict[str, Any]:
        """
        Load baseline measurements.
        
        Returns:
            Baseline measurements
        """
        baseline_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "baseline.json"
        )
        
        if os.path.exists(baseline_path):
            with open(baseline_path, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"Baseline file not found: {baseline_path}")
            return {}
    
    def _save_baseline(self, baseline: Dict[str, Any]):
        """
        Save baseline measurements.
        
        Args:
            baseline: Baseline measurements
        """
        baseline_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "baseline.json"
        )
        
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
    
    def _compare_with_baseline(
        self,
        component: str,
        operation: str,
        current: Dict[str, Any],
        threshold: float = 0.1
    ) -> Tuple[bool, Dict[str, Any]]:
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
        # Check if baseline exists
        if not self.baseline or component not in self.baseline or operation not in self.baseline[component]:
            logger.warning(f"No baseline found for {component}.{operation}")
            return False, {}
        
        # Get baseline
        baseline = self.baseline[component][operation]
        
        # Compare measurements
        comparison = {}
        regression_detected = False
        
        for metric, value in current.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                
                # Calculate difference
                if isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
                    if baseline_value == 0:
                        diff_pct = float("inf") if value > 0 else 0
                    else:
                        diff_pct = (value - baseline_value) / baseline_value
                    
                    comparison[metric] = {
                        "current": value,
                        "baseline": baseline_value,
                        "diff": value - baseline_value,
                        "diff_pct": diff_pct
                    }
                    
                    # Check for regression
                    if metric.endswith("_time") and diff_pct > threshold:
                        regression_detected = True
                        logger.warning(f"Performance regression detected in {component}.{operation}.{metric}: {diff_pct:.2%}")
            else:
                comparison[metric] = {
                    "current": value,
                    "baseline": None,
                    "diff": None,
                    "diff_pct": None
                }
        
        return regression_detected, comparison
    
    def _update_baseline(
        self,
        component: str,
        operation: str,
        measurements: Dict[str, Any]
    ):
        """
        Update baseline measurements.
        
        Args:
            component: Component name
            operation: Operation name
            measurements: Measurements to update
        """
        # Initialize component if needed
        if component not in self.baseline:
            self.baseline[component] = {}
        
        # Update operation
        self.baseline[component][operation] = measurements
        
        # Save baseline
        self._save_baseline(self.baseline)
        
        logger.info(f"Baseline updated for {component}.{operation}")
    
    def _plot_comparison(
        self,
        component: str,
        operation: str,
        comparison: Dict[str, Dict[str, Any]]
    ):
        """
        Plot comparison results.
        
        Args:
            component: Component name
            operation: Operation name
            comparison: Comparison results
        """
        # Filter metrics for plotting
        time_metrics = {k: v for k, v in comparison.items() if k.endswith("_time") and v["baseline"] is not None}
        
        if not time_metrics:
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up data
        metrics = list(time_metrics.keys())
        current_values = [time_metrics[m]["current"] for m in metrics]
        baseline_values = [time_metrics[m]["baseline"] for m in metrics]
        
        # Set up bar positions
        x = np.arange(len(metrics))
        width = 0.35
        
        # Create bars
        ax.bar(x - width/2, baseline_values, width, label="Baseline")
        ax.bar(x + width/2, current_values, width, label="Current")
        
        # Add labels and title
        ax.set_xlabel("Metric")
        ax.set_ylabel("Time (seconds)")
        ax.set_title(f"{component}.{operation} Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add value labels
        for i, v in enumerate(baseline_values):
            ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center")
        
        for i, v in enumerate(current_values):
            ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center")
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"performance_results/{component}_{operation}_comparison.png")
        plt.close()
    
    def test_optimized_confluence_detector(self):
        """Test OptimizedConfluenceDetector performance."""
        logger.info("Testing OptimizedConfluenceDetector performance...")
        
        # Create components
        correlation_service = MockCorrelationService()
        currency_strength_analyzer = CurrencyStrengthAnalyzer()
        
        detector = OptimizedConfluenceDetector(
            correlation_service=correlation_service,
            currency_strength_analyzer=currency_strength_analyzer,
            correlation_threshold=0.7,
            lookback_periods=20,
            cache_ttl_minutes=60,
            max_workers=4
        )
        
        # Prepare test data
        symbol = "EURUSD"
        timeframe = "H1"
        signal_type = "trend"
        signal_direction = "bullish"
        
        # Get price data
        price_data = {
            symbol: self.test_data[symbol][timeframe]
        }
        
        # Get related pairs
        related_pairs = asyncio.run(detector.find_related_pairs(symbol))
        
        # Add related pairs to price data
        for pair in related_pairs.keys():
            if pair in self.test_data and timeframe in self.test_data[pair]:
                price_data[pair] = self.test_data[pair][timeframe]
        
        # Measure cold cache performance
        detector.cache_manager.clear()
        
        start_time = time.time()
        result = detector.detect_confluence_optimized(
            symbol=symbol,
            price_data=price_data,
            signal_type=signal_type,
            signal_direction=signal_direction,
            related_pairs=related_pairs
        )
        cold_time = time.time() - start_time
        
        # Measure warm cache performance
        start_time = time.time()
        result = detector.detect_confluence_optimized(
            symbol=symbol,
            price_data=price_data,
            signal_type=signal_type,
            signal_direction=signal_direction,
            related_pairs=related_pairs
        )
        warm_time = time.time() - start_time
        
        # Measure memory usage
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # Collect measurements
        measurements = {
            "cold_time": cold_time,
            "warm_time": warm_time,
            "memory_usage": memory_usage,
            "confirmation_count": result["confirmation_count"],
            "contradiction_count": result["contradiction_count"],
            "confluence_score": result["confluence_score"]
        }
        
        # Compare with baseline
        regression_detected, comparison = self._compare_with_baseline(
            component="OptimizedConfluenceDetector",
            operation="detect_confluence_optimized",
            current=measurements
        )
        
        # Plot comparison
        self._plot_comparison(
            component="OptimizedConfluenceDetector",
            operation="detect_confluence_optimized",
            comparison=comparison
        )
        
        # Update baseline if requested
        if os.environ.get("UPDATE_BASELINE") == "1":
            self._update_baseline(
                component="OptimizedConfluenceDetector",
                operation="detect_confluence_optimized",
                measurements=measurements
            )
        
        # Assert no regression
        self.assertFalse(regression_detected, "Performance regression detected")
    
    def test_ml_confluence_detector(self):
        """Test MLConfluenceDetector performance."""
        logger.info("Testing MLConfluenceDetector performance...")
        
        # Create components
        correlation_service = MockCorrelationService()
        currency_strength_analyzer = CurrencyStrengthAnalyzer()
        
        model_manager = ModelManager(
            model_dir="models",
            use_gpu=False,
            correlation_service=correlation_service,
            currency_strength_analyzer=currency_strength_analyzer
        )
        
        detector = model_manager.load_ml_confluence_detector()
        
        # Prepare test data
        symbol = "EURUSD"
        timeframe = "H1"
        signal_type = "trend"
        signal_direction = "bullish"
        
        # Get price data
        price_data = {
            symbol: self.test_data[symbol][timeframe]
        }
        
        # Get related pairs
        related_pairs = asyncio.run(detector.find_related_pairs(symbol))
        
        # Add related pairs to price data
        for pair in related_pairs.keys():
            if pair in self.test_data and timeframe in self.test_data[pair]:
                price_data[pair] = self.test_data[pair][timeframe]
        
        # Measure cold cache performance
        detector.cache_manager.clear()
        
        start_time = time.time()
        result = detector.detect_confluence_ml(
            symbol=symbol,
            price_data=price_data,
            signal_type=signal_type,
            signal_direction=signal_direction,
            related_pairs=related_pairs
        )
        cold_time = time.time() - start_time
        
        # Measure warm cache performance
        start_time = time.time()
        result = detector.detect_confluence_ml(
            symbol=symbol,
            price_data=price_data,
            signal_type=signal_type,
            signal_direction=signal_direction,
            related_pairs=related_pairs
        )
        warm_time = time.time() - start_time
        
        # Measure memory usage
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # Collect measurements
        measurements = {
            "cold_time": cold_time,
            "warm_time": warm_time,
            "memory_usage": memory_usage,
            "confirmation_count": result["confirmation_count"],
            "contradiction_count": result["contradiction_count"],
            "confluence_score": result["confluence_score"],
            "pattern_score": result["pattern_score"],
            "prediction_score": result["prediction_score"]
        }
        
        # Compare with baseline
        regression_detected, comparison = self._compare_with_baseline(
            component="MLConfluenceDetector",
            operation="detect_confluence_ml",
            current=measurements
        )
        
        # Plot comparison
        self._plot_comparison(
            component="MLConfluenceDetector",
            operation="detect_confluence_ml",
            comparison=comparison
        )
        
        # Update baseline if requested
        if os.environ.get("UPDATE_BASELINE") == "1":
            self._update_baseline(
                component="MLConfluenceDetector",
                operation="detect_confluence_ml",
                measurements=measurements
            )
        
        # Assert no regression
        self.assertFalse(regression_detected, "Performance regression detected")

# Mock correlation service for testing
class MockCorrelationService:
    """Mock correlation service for testing."""
    
    async def get_all_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Get all correlations between pairs.
        
        Returns:
            Dictionary mapping pairs to dictionaries of correlations
        """
        return {
            "EURUSD": {
                "GBPUSD": 0.85,
                "AUDUSD": 0.75,
                "USDCAD": -0.65,
                "USDJPY": -0.55,
                "EURGBP": 0.62,
                "EURJPY": 0.78
            },
            "GBPUSD": {
                "EURUSD": 0.85,
                "AUDUSD": 0.70,
                "USDCAD": -0.60,
                "USDJPY": -0.50,
                "EURGBP": -0.58,
                "GBPJPY": 0.75
            },
            "USDJPY": {
                "EURUSD": -0.55,
                "GBPUSD": -0.50,
                "AUDUSD": -0.45,
                "USDCAD": 0.40,
                "EURJPY": 0.65,
                "GBPJPY": 0.70
            },
            "AUDUSD": {
                "EURUSD": 0.75,
                "GBPUSD": 0.70,
                "USDCAD": -0.55,
                "USDJPY": -0.45
            }
        }

if __name__ == "__main__":
    unittest.main()
