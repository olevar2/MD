"""
Performance Testing Framework for Phase 4

This module implements comprehensive performance testing for the adaptive strategies
and confluence analysis components introduced in Phase 4, with a focus on testing
under high load conditions.
"""

import logging
import time
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import concurrent.futures
from dataclasses import dataclass

from analysis_engine.services.market_regime import MarketRegimeService
from analysis_engine.analysis.confluence_analyzer import ConfluenceAnalyzer
from strategy_execution_engine.signal_aggregation.signal_aggregator import SignalAggregator
from strategy_execution_engine.adaptive_layer.adaptive_layer_service import AdaptiveLayerService
from data_pipeline_service.services.historical_data_service import HistoricalDataService
from strategy_execution_engine.strategies.strategy_loader import StrategyLoader
from ml_integration_service.strategy_filters.ml_confirmation_filter import MLConfirmationFilter


@dataclass
class PerformanceTestConfig:
    """Configuration for the performance test."""
    symbols: List[str]
    timeframes: List[str]
    test_duration_seconds: int
    parallel_requests: int
    strategy_types: List[str]
    include_ml_confirmation: bool
    test_market_regimes: List[str]
    backtest_days: int
    load_historical_data: bool
    max_signals_per_batch: int
    measure_memory_usage: bool


class PerformanceTestingFramework:
    """
    Framework for testing the performance of Phase 4 components under load.
    Simulates real-world conditions with varying levels of concurrent requests
    and measures latency, throughput, and resource utilization.
    """
    
    def __init__(self, config: Optional[PerformanceTestConfig] = None):
        """Initialize the performance testing framework."""
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        default_config = PerformanceTestConfig(
            symbols=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
            timeframes=["1h", "4h", "1d"],
            test_duration_seconds=300,  # 5 minutes
            parallel_requests=10,
            strategy_types=["adaptive_ma", "harmonic_pattern", "elliott_wave", "advanced_breakout"],
            include_ml_confirmation=True,
            test_market_regimes=["trending", "ranging", "volatile", "breakout"],
            backtest_days=30,
            load_historical_data=True,
            max_signals_per_batch=100,
            measure_memory_usage=True
        )
        
        # Use provided config or default
        self.config = config or default_config
        
        # Initialize components to test
        self.historical_data_service = HistoricalDataService()
        self.market_regime_service = MarketRegimeService()
        self.confluence_analyzer = ConfluenceAnalyzer()
        self.signal_aggregator = SignalAggregator()
        self.adaptive_layer_service = AdaptiveLayerService()
        self.strategy_loader = StrategyLoader()
        self.ml_filter = MLConfirmationFilter() if self.config.include_ml_confirmation else None
        
        # Metrics storage
        self.metrics = {
            "latency_ms": [],
            "throughput_signals_per_second": [],
            "resource_utilization": {
                "cpu_percent": [],
                "memory_mb": []
            },
            "error_rate": 0.0,
            "successful_requests": 0,
            "failed_requests": 0,
            "component_latency": {
                "data_fetch": [],
                "confluence_analysis": [],
                "signal_generation": [],
                "signal_aggregation": [],
                "ml_confirmation": []
            },
            "regime_metrics": {},
            "strategy_metrics": {}
        }
        
        # Initialize metrics for each market regime
        for regime in self.config.test_market_regimes:
            self.metrics["regime_metrics"][regime] = {
                "latency_ms": [],
                "signal_count": 0
            }
        
        # Initialize metrics for each strategy type
        for strategy in self.config.strategy_types:
            self.metrics["strategy_metrics"][strategy] = {
                "latency_ms": [],
                "signal_count": 0,
                "error_count": 0
            }
            
        self.logger.info("Performance Testing Framework initialized")
        
    async def run_performance_test(self) -> Dict[str, Any]:
        """
        Run a comprehensive performance test of Phase 4 components under load.
        
        Returns:
            Dictionary with test results and metrics
        """
        self.logger.info(f"Starting performance test for {self.config.test_duration_seconds} seconds")
        self.logger.info(f"Testing with {self.config.parallel_requests} parallel requests")
        
        start_time = time.time()
        end_time = start_time + self.config.test_duration_seconds
        
        # Preload historical data if configured
        if self.config.load_historical_data:
            await self._preload_historical_data()
        
        # Set up tasks for parallel execution
        tasks = []
        for _ in range(self.config.parallel_requests):
            tasks.append(self._run_continuous_test_worker(end_time))
        
        # Run tasks
        await asyncio.gather(*tasks)
        
        # Calculate final metrics
        test_duration = time.time() - start_time
        total_requests = self.metrics["successful_requests"] + self.metrics["failed_requests"]
        
        if total_requests > 0:
            self.metrics["error_rate"] = self.metrics["failed_requests"] / total_requests
            
        self.metrics["test_duration_seconds"] = test_duration
        self.metrics["requests_per_second"] = total_requests / test_duration
        
        if self.metrics["latency_ms"]:
            self.metrics["avg_latency_ms"] = sum(self.metrics["latency_ms"]) / len(self.metrics["latency_ms"])
            self.metrics["min_latency_ms"] = min(self.metrics["latency_ms"])
            self.metrics["max_latency_ms"] = max(self.metrics["latency_ms"])
            
            # Calculate percentiles
            latency_array = np.array(self.metrics["latency_ms"])
            self.metrics["p50_latency_ms"] = np.percentile(latency_array, 50)
            self.metrics["p90_latency_ms"] = np.percentile(latency_array, 90)
            self.metrics["p95_latency_ms"] = np.percentile(latency_array, 95)
            self.metrics["p99_latency_ms"] = np.percentile(latency_array, 99)
        
        # Calculate component-specific metrics
        for component, latencies in self.metrics["component_latency"].items():
            if latencies:
                self.metrics[f"{component}_avg_latency_ms"] = sum(latencies) / len(latencies)
        
        # Calculate metrics for each market regime
        for regime, regime_metrics in self.metrics["regime_metrics"].items():
            if regime_metrics["latency_ms"]:
                regime_metrics["avg_latency_ms"] = sum(regime_metrics["latency_ms"]) / len(regime_metrics["latency_ms"])
                
        # Calculate metrics for each strategy
        for strategy, strategy_metrics in self.metrics["strategy_metrics"].items():
            if strategy_metrics["latency_ms"]:
                strategy_metrics["avg_latency_ms"] = sum(strategy_metrics["latency_ms"]) / len(strategy_metrics["latency_ms"])
                if strategy_metrics["signal_count"] > 0:
                    strategy_metrics["error_rate"] = strategy_metrics["error_count"] / strategy_metrics["signal_count"]
        
        self.logger.info(f"Performance test completed. Processed {total_requests} requests "
                       f"with avg latency of {self.metrics.get('avg_latency_ms', 0):.2f}ms "
                       f"and error rate of {self.metrics['error_rate'] * 100:.2f}%")
                       
        return self.metrics
    
    async def _run_continuous_test_worker(self, end_time: float) -> None:
        """Run continuous test iterations until the end time is reached."""
        while time.time() < end_time:
            try:
                # Select random symbol, timeframe, and strategy for this iteration
                symbol = np.random.choice(self.config.symbols)
                timeframe = np.random.choice(self.config.timeframes)
                strategy_type = np.random.choice(self.config.strategy_types)
                
                # Run a single test iteration
                await self._run_test_iteration(symbol, timeframe, strategy_type)
                
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in test worker: {str(e)}", exc_info=True)
                self.metrics["failed_requests"] += 1
    
    async def _run_test_iteration(self, symbol: str, timeframe: str, strategy_type: str) -> None:
        """Run a single test iteration with the specified parameters."""
        start_time = time.time()
        
        try:
            # Measure time for data fetch
            data_fetch_start = time.time()
            price_data = await self._get_historical_data(symbol, timeframe)
            data_fetch_time = time.time() - data_fetch_start
            self.metrics["component_latency"]["data_fetch"].append(data_fetch_time * 1000)
            
            # Detect market regime
            regime_result = self.market_regime_service.detect_regime(price_data)
            regime = regime_result.get("regime", "trending")
            
            # Record which regime we're testing
            regime_name = regime.name.lower() if hasattr(regime, "name") else str(regime).lower()
            
            # Measure time for confluence analysis
            confluence_start = time.time()
            confluence_results = self.confluence_analyzer.analyze(
                symbol=symbol,
                price_data={timeframe: price_data},
                timeframes=[timeframe]
            )
            confluence_time = time.time() - confluence_start
            self.metrics["component_latency"]["confluence_analysis"].append(confluence_time * 1000)
            
            # Load strategy
            strategy = self.strategy_loader.get_strategy(strategy_type)
            if not strategy:
                raise ValueError(f"Strategy {strategy_type} not found")
            
            # Apply adaptive parameters via the adaptive layer
            self.adaptive_layer_service.adapt_strategy(strategy, regime, symbol)
            
            # Generate signals with the strategy
            signal_start = time.time()
            signals = strategy.generate_signals(
                symbol=symbol,
                price_data={timeframe: price_data},
                confluence_results=confluence_results
            )
            signal_time = time.time() - signal_start
            self.metrics["component_latency"]["signal_generation"].append(signal_time * 1000)
            
            # Apply signal aggregation if multiple signals
            if signals and len(signals) > 1:
                agg_start = time.time()
                aggregated_signals = self.signal_aggregator.aggregate_signals(signals, symbol)
                agg_time = time.time() - agg_start
                self.metrics["component_latency"]["signal_aggregation"].append(agg_time * 1000)
            else:
                aggregated_signals = signals
            
            # Apply ML confirmation if configured
            if self.config.include_ml_confirmation and self.ml_filter and aggregated_signals:
                ml_start = time.time()
                for i, signal in enumerate(aggregated_signals):
                    features = self._extract_features_from_price_data(price_data)
                    aggregated_signals[i] = self.ml_filter.filter_signal(
                        signal=signal,
                        features=features,
                        price_data=price_data
                    )
                ml_time = time.time() - ml_start
                self.metrics["component_latency"]["ml_confirmation"].append(ml_time * 1000)
            
            # Record total latency
            total_time = time.time() - start_time
            latency_ms = total_time * 1000
            self.metrics["latency_ms"].append(latency_ms)
            
            # Update metrics for this regime
            if regime_name in self.metrics["regime_metrics"]:
                self.metrics["regime_metrics"][regime_name]["latency_ms"].append(latency_ms)
                self.metrics["regime_metrics"][regime_name]["signal_count"] += len(aggregated_signals) if aggregated_signals else 0
            
            # Update metrics for this strategy
            strategy_key = strategy_type.replace("_strategy", "")
            if strategy_key in self.metrics["strategy_metrics"]:
                self.metrics["strategy_metrics"][strategy_key]["latency_ms"].append(latency_ms)
                self.metrics["strategy_metrics"][strategy_key]["signal_count"] += len(aggregated_signals) if aggregated_signals else 0
            
            # Record success
            self.metrics["successful_requests"] += 1
            
        except Exception as e:
            self.logger.error(f"Error in test iteration: {str(e)}", exc_info=True)
            self.metrics["failed_requests"] += 1
            
            # Update error count for this strategy
            strategy_key = strategy_type.replace("_strategy", "")
            if strategy_key in self.metrics["strategy_metrics"]:
                self.metrics["strategy_metrics"][strategy_key]["error_count"] += 1
    
    async def _preload_historical_data(self) -> None:
        """Preload historical data for all symbols and timeframes to speed up testing."""
        self.logger.info("Preloading historical data for all symbols and timeframes")
        
        preload_tasks = []
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                preload_tasks.append(self._get_historical_data(symbol, timeframe, preload=True))
        
        await asyncio.gather(*preload_tasks)
        self.logger.info("Completed preloading historical data")
    
    async def _get_historical_data(self, symbol: str, timeframe: str, preload: bool = False) -> pd.DataFrame:
        """Get historical data for the specified symbol and timeframe."""
        end_date = datetime.now()
        
        try:
            data = await self.historical_data_service.get_historical_data_async(
                symbol=symbol,
                timeframe=timeframe,
                days=self.config.backtest_days,
                end_date=end_date
            )
            
            if preload:
                # For preloading, we don't need to do anything with the data
                return data
            
            # For real test, return the data
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol} {timeframe}: {str(e)}")
            raise
    
    def _extract_features_from_price_data(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract features from price data for ML model inputs."""
        # This would normally extract complex features, but for testing we'll use simplified ones
        features = {
            "rsi_14": None,
            "macd": None,
            "bollinger_bands": None,
            "atr": None
        }
        
        try:
            # Last 14 periods close prices
            close_prices = price_data["close"].values[-14:]
            
            # Simple RSI calculation (very simplified, just for testing)
            diff = np.diff(close_prices)
            gains = diff[diff > 0].mean() if len(diff[diff > 0]) > 0 else 0
            losses = -diff[diff < 0].mean() if len(diff[diff < 0]) > 0 else 0
            
            if losses == 0:
                rsi = 100
            else:
                rs = gains / losses if losses != 0 else float('inf')
                rsi = 100 - (100 / (1 + rs))
                
            features["rsi_14"] = rsi
            
            # ATR (very simplified)
            high_low = price_data["high"].values[-14:] - price_data["low"].values[-14:]
            features["atr"] = np.mean(high_low)
            
            # Return extracted features
            return features
            
        except Exception as e:
            self.logger.warning(f"Error extracting features: {str(e)}")
            return features
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance test report.
        
        Returns:
            Dictionary with formatted test results for reporting
        """
        report = {
            "summary": {
                "test_duration_seconds": self.metrics.get("test_duration_seconds", 0),
                "total_requests": self.metrics["successful_requests"] + self.metrics["failed_requests"],
                "successful_requests": self.metrics["successful_requests"],
                "failed_requests": self.metrics["failed_requests"],
                "error_rate_percent": self.metrics["error_rate"] * 100,
                "avg_latency_ms": self.metrics.get("avg_latency_ms", 0),
                "p95_latency_ms": self.metrics.get("p95_latency_ms", 0),
                "p99_latency_ms": self.metrics.get("p99_latency_ms", 0),
                "requests_per_second": self.metrics.get("requests_per_second", 0)
            },
            "latency_breakdown": {},
            "component_performance": {},
            "market_regime_performance": {},
            "strategy_performance": {}
        }
        
        # Add component performance
        for component, latencies in self.metrics["component_latency"].items():
            if latencies:
                report["component_performance"][component] = {
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "percent_of_total": sum(latencies) / sum(self.metrics["latency_ms"]) * 100 if self.metrics["latency_ms"] else 0,
                    "call_count": len(latencies)
                }
        
        # Add market regime performance
        for regime, regime_metrics in self.metrics["regime_metrics"].items():
            if regime_metrics.get("latency_ms"):
                report["market_regime_performance"][regime] = {
                    "avg_latency_ms": regime_metrics.get("avg_latency_ms", 0),
                    "signal_count": regime_metrics.get("signal_count", 0),
                    "request_count": len(regime_metrics["latency_ms"])
                }
        
        # Add strategy performance
        for strategy, strategy_metrics in self.metrics["strategy_metrics"].items():
            if strategy_metrics.get("latency_ms"):
                report["strategy_performance"][strategy] = {
                    "avg_latency_ms": strategy_metrics.get("avg_latency_ms", 0),
                    "signal_count": strategy_metrics.get("signal_count", 0),
                    "error_count": strategy_metrics.get("error_count", 0),
                    "error_rate_percent": strategy_metrics.get("error_rate", 0) * 100
                }
        
        return report


def run_load_testing():
    """Run the performance test framework with a default configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create instance of the test framework
    config = PerformanceTestConfig(
        symbols=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
        timeframes=["1h", "4h"],
        test_duration_seconds=60,  # 1 minute for quick test
        parallel_requests=5,
        strategy_types=["adaptive_ma", "elliott_wave"],
        include_ml_confirmation=True,
        test_market_regimes=["trending", "ranging"],
        backtest_days=10,
        load_historical_data=True,
        max_signals_per_batch=50,
        measure_memory_usage=True
    )
    
    test_framework = PerformanceTestingFramework(config)
    
    # Run the test
    logger.info("Starting performance test")
    try:
        import asyncio
        results = asyncio.run(test_framework.run_performance_test())
        
        # Generate and log report
        report = test_framework.generate_report()
        logger.info(f"Performance test completed. Summary: {report['summary']}")
        
        return report
        
    except Exception as e:
        logger.error(f"Error running performance test: {str(e)}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    run_load_testing()
