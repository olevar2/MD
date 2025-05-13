#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated Stress Test Runner

This module integrates the market scenario generator and user load generator
to create comprehensive stress tests for the forex trading platform. It coordinates
simultaneous execution of extreme market scenarios and high user loads to test
the platform's resilience under combined stress conditions.

The integrated test framework allows for various combinations of market events and
user load patterns to simulate real-world crisis scenarios.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .market_scenario_generator import MarketScenarioGenerator 

from .user_load_generator import (
    UserActivityType,
    UserType,
    LoadProfile,
    AsyncUserLoadGenerator,
    create_default_load_profile,
    create_default_endpoints,
)
from .environment_config import ResourceConstraintType, ResourceConstraint, StressLevel 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegratedScenarioType(Enum):
    """Types of integrated stress test scenarios."""
    FLASH_CRASH_WITH_HIGH_LOAD = "flash_crash_with_high_load"
    VOLATILITY_SPIKE_WITH_CYCLIC_LOAD = "volatility_spike_with_cyclic_load"
    BLACK_SWAN_WITH_LOAD_SPIKE = "black_swan_with_load_spike"
    LIQUIDITY_GAP_WITH_RANDOM_LOAD = "liquidity_gap_with_random_load"
    TRENDING_MARKET_WITH_CONSTANT_LOAD = "trending_market_with_constant_load"
    MULTI_ASSET_CORRELATION_BREAKDOWN = "multi_asset_correlation_breakdown"
    CENTRAL_BANK_INTERVENTION_WITH_NEWS_SPIKE = "central_bank_intervention_with_news_spike"
    OVERNIGHT_GAP_WITH_ASIA_EUROPE_HANDOVER = "overnight_gap_with_asia_europe_handover"
    CHOPPY_MARKET_WITH_ALGO_DOMINANCE = "choppy_market_with_algo_dominance"
    CUSTOM = "custom"  # For custom scenario configurations


@dataclass
class IntegratedScenarioConfig:
    """Configuration for an integrated stress test scenario."""
    name: str
    description: str
    # Change MarketEventType to string to match MarketScenarioGenerator's set_scenario method
    market_scenarios: List[str] # Was: market_events: List[MarketEventType]
    currency_pairs: List[str] # Was: CurrencyPair, assuming it was just a type hint for str
    user_load_profile: LoadProfile
    duration_seconds: float
    resource_constraints: List[ResourceConstraint] = field(default_factory=list)
    geographic_focus: Optional[List[str]] = None
    correlation_matrix: Optional[pd.DataFrame] = None
    time_compression_factor: float = 1.0  # How much faster than real-time
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            # Update key name and value format
            "market_scenarios": self.market_scenarios, # Was: market_events: [event.value for event in self.market_events]
            "currency_pairs": self.currency_pairs,
            "user_load_profile": self.user_load_profile.to_dict(),
            "duration_seconds": self.duration_seconds,
            "resource_constraints": [rc.to_dict() for rc in self.resource_constraints],
            "geographic_focus": self.geographic_focus,
            "time_compression_factor": self.time_compression_factor,
            "tags": self.tags
        }


@dataclass
class IntegratedTestResults:
    """Results collected from an integrated stress test."""
    scenario_config: IntegratedScenarioConfig
    market_data: Dict[str, pd.DataFrame]  # Currency pair -> price time series
    user_stats: Dict[str, Any]  # Statistics from user load generator
    system_metrics: Dict[str, List[float]]  # Metric name -> time series
    bottlenecks: List[str]  # Identified system bottlenecks
    errors: List[Dict[str, Any]]  # Error events detected
    correlation_breakdown: Dict[Tuple[str, str], float]  # Pair of assets -> correlation change
    start_time: datetime
    end_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "scenario_config": self.scenario_config.to_dict(),
            "user_stats": self.user_stats,
            "system_metrics_summary": {
                k: {
                    "min": min(v),
                    "max": max(v),
                    "mean": sum(v) / len(v),
                    "last": v[-1]
                } for k, v in self.system_metrics.items()
            },
            "bottlenecks": self.bottlenecks,
            "error_count": len(self.errors),
            "error_types": self._categorize_errors(),
            "test_duration_seconds": (self.end_time - self.start_time).total_seconds(),
        }
    
    def _categorize_errors(self) -> Dict[str, int]:
        """Categorize errors by type and count occurrences."""
        error_categories = {}
        for error in self.errors:
            error_type = error.get("type", "unknown")
            error_categories[error_type] = error_categories.get(error_type, 0) + 1
        return error_categories
    
    def plot_market_data(self, output_dir: Optional[str] = None) -> None:
        """Plot market data for visual inspection."""
        fig, axes = plt.subplots(len(self.market_data), 1, figsize=(12, 4 * len(self.market_data)))
        if len(self.market_data) == 1:
            axes = [axes]
        
        for i, (pair, data) in enumerate(self.market_data.items()):
            axes[i].plot(data.index, data['price'], label=f"{pair} Price")
            axes[i].set_title(f"{pair} Price Movement")
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Price")
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        if output_dir:
            fig.savefig(f"{output_dir}/market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        else:
            plt.show()
    
    def plot_user_load(self, output_dir: Optional[str] = None) -> None:
        """Plot user load metrics."""
        if "active_sessions_time_series" not in self.user_stats:
            logger.warning("No time series data for user activity available")
            return
            
        times = self.user_stats["active_sessions_time_series"]["timestamps"]
        sessions = self.user_stats["active_sessions_time_series"]["values"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, sessions, label="Active Sessions")
        ax.set_title("User Load Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Active Sessions")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        if output_dir:
            fig.savefig(f"{output_dir}/user_load_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        else:
            plt.show()


class IntegratedStressTest:
    """Coordinates market scenario generation and user load generation for integrated stress testing."""
    
    def __init__(self, 
                api_base_url: str = "http://localhost:8080/api",
                results_dir: Optional[str] = None):
        """
        Initialize the integrated stress test coordinator.
        
        Args:
            api_base_url: Base URL for the forex platform API
            results_dir: Directory to save test results, if None uses ./stress_test_results
        """
        self.api_base_url = api_base_url
        self.results_dir = results_dir or Path("./stress_test_results")
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        self._market_generator = None
        self._user_generator = None
        self._system_metrics_collector = None
        self._stop_event = asyncio.Event()
        self._market_data = {}
        self._system_metrics = {}
        self._errors = []
    
    async def run_scenario(self, config: IntegratedScenarioConfig) -> IntegratedTestResults:
        """
        Run an integrated stress test with the specified configuration.
        
        Args:
            config: Configuration for the integrated test scenario
            
        Returns:
            IntegratedTestResults with complete test data
        """
        logger.info(f"Starting integrated stress test: {config.name}")
        logger.info(f"Description: {config.description}")
        
        start_time = datetime.now()
        
        # Reset state
        self._stop_event.clear()
        self._market_data = {}
        self._system_metrics = {}
        self._errors = []
        
        # Initialize components
        endpoints = self._create_endpoints()
        
        # Start collecting system metrics in the background
        metrics_task = asyncio.create_task(self._collect_system_metrics(config.duration_seconds))
        
        # Create and start market scenario generator
        # (This assumes a specific interface for your market scenario generator)
        market_scenario_task = asyncio.create_task(
            self._generate_market_scenarios(config)
        )
        
        # Create and start user load generator
        user_load_task = asyncio.create_task(
            self._generate_user_load(config, endpoints)
        )
        
        # Wait for the configured test duration
        await asyncio.sleep(config.duration_seconds)
        
        # Signal components to stop
        self._stop_event.set()
        
        # Wait for tasks to complete
        await asyncio.gather(market_scenario_task, user_load_task, metrics_task)
        
        # Collect results
        end_time = datetime.now()
        
        # Get user stats
        user_stats = await self._user_generator.get_stats() if self._user_generator else {}
        
        # Create and return test results
        results = IntegratedTestResults(
            scenario_config=config,
            market_data=self._market_data,
            user_stats=user_stats,
            system_metrics=self._system_metrics,
            bottlenecks=self._identify_bottlenecks(),
            errors=self._errors,
            correlation_breakdown=self._calculate_correlation_changes(),
            start_time=start_time,
            end_time=end_time
        )
        
        # Save results to file
        self._save_results(results)
        
        logger.info(f"Integrated stress test completed: {config.name}")
        return results
    
    async def _generate_market_scenarios(self, config: IntegratedScenarioConfig) -> None:
        """Generate market scenarios based on the configuration."""
        try:
            # Initialize the generator with its required config structure
            # Assuming the generator needs a config dict like the one in its __main__ example
            market_gen_config = {
                'market_scenario': {
                    'symbols': config.currency_pairs,
                    'base_prices': {pair: 1.0 for pair in config.currency_pairs}, # Example base prices
                    # Add scenario parameters if needed, based on MarketScenarioGenerator's example
                }
            }
            self._market_generator = MarketScenarioGenerator(market_gen_config)
            
            # Initialize market data collectors for each pair
            for pair in config.currency_pairs:
                # Ensure timestamp is the index for potential resampling later
                self._market_data[pair] = pd.DataFrame(columns=['price', 'bid', 'ask', 'volume']).set_index(pd.to_datetime([]))
            
            # Loop to generate data based on the specified scenarios
            start_time = time.time()
            while time.time() - start_time < config.duration_seconds and not self._stop_event.is_set():
                # Cycle through configured scenarios or apply logic based on config.market_scenarios
                # For simplicity, let's just pick one scenario or cycle through them
                current_scenario_name = random.choice(config.market_scenarios) if config.market_scenarios else 'normal'
                self._market_generator.set_scenario(current_scenario_name)

                # Generate an update
                update = self._market_generator.generate_market_data_update()
                if update:
                    timestamp = pd.to_datetime(update['timestamp'], unit='s')
                    pair = update['symbol']
                    if pair in self._market_data:
                         # Use loc for index-based assignment
                        self._market_data[pair].loc[timestamp] = {
                            'price': (update['bid'] + update['ask']) / 2, # Calculate mid-price
                            'bid': update['bid'],
                            'ask': update['ask'],
                            'volume': random.randint(100, 10000) # Add dummy volume if not present
                        }
                
                # Generate an order (optional, depending on test goals)
                # order = self._market_generator.generate_order_request()
                # if order:
                #     # Process or log the order
                #     pass

                await asyncio.sleep(0.05) # Control update frequency

            logger.info("Market scenario generation finished.")

        except Exception as e:
            logger.error(f"Error generating market scenarios: {str(e)}", exc_info=True)
            self._errors.append({
                "type": "market_scenario_error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    async def _generate_user_load(self, config: IntegratedScenarioConfig, endpoints: Dict[str, str]) -> None:
        """Generate user load based on the configuration."""
        try:
            # Create user load generator
            self._user_generator = AsyncUserLoadGenerator(
                load_profile=config.user_load_profile,
                endpoints=endpoints,
                stress_level=StressLevel.EXTREME  # Use highest stress level for integration tests
            )
            
            # Start the user load generator
            await self._user_generator.start()
            
            # Wait until signaled to stop
            await self._stop_event.wait()
            
            # Stop the user load generator
            await self._user_generator.stop()
            
        except Exception as e:
            logger.error(f"Error generating user load: {str(e)}", exc_info=True)
            self._errors.append({
                "type": "user_load_error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    async def _collect_system_metrics(self, duration_seconds: float) -> None:
        """Collect system metrics during the test."""
        try:
            # Initialize metrics collection
            self._system_metrics = {
                "cpu_usage": [],
                "memory_usage": [],
                "network_throughput": [],
                "database_connections": [],
                "api_latency": [],
                "order_processing_time": []
            }
            
            # Collect metrics at regular intervals
            interval = 1.0  # 1 second between measurements
            iterations = int(duration_seconds / interval)
            
            for _ in range(iterations):
                if self._stop_event.is_set():
                    break
                    
                # Collect CPU usage (example - replace with actual implementation)
                self._system_metrics["cpu_usage"].append(random.uniform(10, 90))  # Placeholder
                
                # Collect memory usage (example - replace with actual implementation)
                self._system_metrics["memory_usage"].append(random.uniform(20, 80))  # Placeholder
                
                # Collect network throughput (example - replace with actual implementation)
                self._system_metrics["network_throughput"].append(random.uniform(100, 5000))  # Placeholder
                
                # Collect database connections (example - replace with actual implementation)
                self._system_metrics["database_connections"].append(random.uniform(5, 50))  # Placeholder
                
                # Collect API latency (example - replace with actual implementation)
                self._system_metrics["api_latency"].append(random.uniform(50, 500))  # Placeholder
                
                # Collect order processing time (example - replace with actual implementation)
                self._system_metrics["order_processing_time"].append(random.uniform(10, 200))  # Placeholder
                
                await asyncio.sleep(interval)
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}", exc_info=True)
            self._errors.append({
                "type": "metrics_collection_error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify system bottlenecks based on collected metrics."""
        bottlenecks = []
        
        # Example logic - replace with actual analysis
        if self._system_metrics.get("cpu_usage") and max(self._system_metrics["cpu_usage"]) > 80:
            bottlenecks.append("CPU usage exceeded 80%")
            
        if self._system_metrics.get("memory_usage") and max(self._system_metrics["memory_usage"]) > 85:
            bottlenecks.append("Memory usage exceeded 85%")
            
        if self._system_metrics.get("api_latency") and max(self._system_metrics["api_latency"]) > 300:
            bottlenecks.append("API latency exceeded 300ms")
            
        if self._system_metrics.get("order_processing_time") and max(self._system_metrics["order_processing_time"]) > 100:
            bottlenecks.append("Order processing time exceeded 100ms")
            
        # Check user stats for error rates
        if hasattr(self, "_user_generator") and self._user_generator:
            user_stats = asyncio.run(self._user_generator.get_stats())
            if user_stats.get("error_rate", 0) > 0.05:
                bottlenecks.append(f"User request error rate exceeded 5% ({user_stats.get('error_rate', 0)*100:.1f}%)")
        
        return bottlenecks
    
    def _calculate_correlation_changes(self) -> Dict[Tuple[str, str], float]:
        """Calculate correlation changes between currency pairs."""
        correlations = {}
        pairs = list(self._market_data.keys())
        
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                pair1 = pairs[i]
                pair2 = pairs[j]
                
                df1 = self._market_data.get(pair1)
                df2 = self._market_data.get(pair2)
                
                if df1 is None or df2 is None or df1.empty or df2.empty:
                    logger.warning(f"Skipping correlation for {pair1}/{pair2} due to missing/empty data.")
                    continue
                
                try:
                    # Ensure dataframes have datetime index
                    if not isinstance(df1.index, pd.DatetimeIndex):
                        df1.index = pd.to_datetime(df1.index)
                    if not isinstance(df2.index, pd.DatetimeIndex):
                        df2.index = pd.to_datetime(df2.index)

                    # Resample to a common frequency (e.g., 1 second) and forward fill missing values
                    resample_freq = '1S'
                    df1_resampled = df1['price'].resample(resample_freq).ffill()
                    df2_resampled = df2['price'].resample(resample_freq).ffill()
                    
                    # Align dataframes on the common index
                    aligned_df1, aligned_df2 = df1_resampled.align(df2_resampled, join='inner')

                    if len(aligned_df1) > 1: # Need at least 2 data points for correlation
                        corr = aligned_df1.corr(aligned_df2)
                        if pd.notna(corr): # Check if correlation is not NaN
                             correlations[(pair1, pair2)] = corr
                        else:
                            logger.warning(f"Correlation calculation resulted in NaN for {pair1}/{pair2}.")
                    else:
                        logger.warning(f"Not enough overlapping data points to calculate correlation for {pair1}/{pair2}.")

                except Exception as e:
                    logger.error(f"Error calculating correlation between {pair1} and {pair2}: {str(e)}", exc_info=True)
        
        return correlations
    
    def _create_endpoints(self) -> Dict[str, str]:
        """Create endpoints dictionary for the user load generator."""
        base_url = self.api_base_url
        return {
            "auth": f"{base_url}/auth",
            "market_data": f"{base_url}/market-data",
            "orders": f"{base_url}/orders",
            "portfolio": f"{base_url}/portfolio",
            "analysis": f"{base_url}/analysis",
            "reports": f"{base_url}/reports",
            "alerts": f"{base_url}/alerts",
            "charts": f"{base_url}/charts",
            "strategies": f"{base_url}/strategies"
        }
    
    def _save_results(self, results: IntegratedTestResults) -> None:
        """Save test results to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scenario_name = results.scenario_config.name.lower().replace(" ", "_")
            
            # Save summary as JSON
            summary_path = Path(self.results_dir) / f"{scenario_name}_{timestamp}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
            logger.info(f"Test summary saved to {summary_path}")
            
            # Save market data as CSV files
            market_data_dir = Path(self.results_dir) / "market_data" / f"{scenario_name}_{timestamp}"
            market_data_dir.mkdir(parents=True, exist_ok=True)
            
            for pair, data in results.market_data.items():
                pair_file = market_data_dir / f"{pair.replace('/', '_')}.csv"
                data.to_csv(pair_file, index=False)
            
            # Save system metrics as CSV
            metrics_path = Path(self.results_dir) / f"{scenario_name}_{timestamp}_metrics.csv"
            metrics_df = pd.DataFrame(results.system_metrics)
            metrics_df.to_csv(metrics_path, index=False)
            
            # Save plots
            plots_dir = Path(self.results_dir) / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            results.plot_market_data(str(plots_dir))
            results.plot_user_load(str(plots_dir))
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}", exc_info=True)


def create_flash_crash_high_load_scenario() -> IntegratedScenarioConfig:
    """Create a predefined scenario with flash crash and high user load."""
    load_profile = LoadProfile(
        concurrent_users=200,  # High number of users
        ramp_up_time=30.0,
        steady_state_time=180.0,  # 3 minutes of steady state
        ramp_down_time=30.0,
        user_distribution={
            UserType.RETAIL: 0.6,
            UserType.INSTITUTIONAL: 0.2,
            UserType.ALGORITHMIC: 0.1,
            UserType.HFT: 0.05,
            UserType.API_CONSUMER: 0.05
        },
        connection_pattern="spike",  # Use spike pattern to simulate panic
        geographic_distribution={
            "us-east": 0.3,
            "us-west": 0.2,
            "europe": 0.3,
            "asia": 0.2
        }
    )
    
    return IntegratedScenarioConfig(
        name="Flash Crash with High User Load",
        description="Simulates a market flash crash with panic leading to high user load",
        # Use scenario names as strings
        market_scenarios=["flash_crash_test", "liquidity_gap"], # Was: market_events=[MarketEventType.FLASH_CRASH, MarketEventType.LIQUIDITY_GAP]
        currency_pairs=["EUR/USD", "GBP/USD", "USD/JPY"],
        user_load_profile=load_profile,
        duration_seconds=240.0,  # 4 minutes total
        resource_constraints=[
            ResourceConstraint(
                type=ResourceConstraintType.DATABASE_CONNECTIONS,
                limit_value=50,
                unit="connections"
            )
        ],
        time_compression_factor=5.0,  # 5x faster than real-time
        tags=["flash_crash", "high_load", "critical_scenario"]
    )


def create_volatility_spike_cyclic_load_scenario() -> IntegratedScenarioConfig:
    """Create a predefined scenario with volatility spike and cyclic user load."""
    load_profile = LoadProfile(
        concurrent_users=150,
        ramp_up_time=60.0,
        steady_state_time=300.0,  # 5 minutes of steady state
        ramp_down_time=60.0,
        user_distribution={
            UserType.RETAIL: 0.5,
            UserType.INSTITUTIONAL: 0.2,
            UserType.ALGORITHMIC: 0.2,
            UserType.HFT: 0.05,
            UserType.API_CONSUMER: 0.05
        },
        connection_pattern="cyclic",  # Cyclic pattern to simulate waves of user activity
        geographic_distribution={
            "us-east": 0.25,
            "us-west": 0.25,
            "europe": 0.3,
            "asia": 0.2
        }
    )
    
    return IntegratedScenarioConfig(
        name="Volatility Spike with Cyclic User Load",
        description="Simulates high market volatility with waves of user activity",
         # Use scenario names as strings
        market_scenarios=["volatility_spike", "choppy_market"], # Was: market_events=[MarketEventType.VOLATILITY_SPIKE, MarketEventType.CHOPPY_MARKET]
        currency_pairs=["EUR/USD", "USD/CHF", "AUD/USD", "USD/CAD"],
        user_load_profile=load_profile,
        duration_seconds=420.0,  # 7 minutes total
        resource_constraints=[
            ResourceConstraint(
                type=ResourceConstraintType.CPU,
                limit_value=80,
                unit="percent"
            ),
            ResourceConstraint(
                type=ResourceConstraintType.MEMORY,
                limit_value=70,
                unit="percent"
            )
        ],
        time_compression_factor=3.0,  # 3x faster than real-time
        tags=["volatility", "cyclic_load", "high_frequency"]
    )


async def run_sample_integrated_test() -> None:
    """Run a sample integrated stress test."""
    # Create test coordinator
    test_coordinator = IntegratedStressTest(
        api_base_url="http://localhost:8080/api",
        results_dir="./stress_test_results"
    )
    
    # Create scenario
    scenario = create_flash_crash_high_load_scenario()
    
    # Run the test
    logger.info("Starting integrated stress test")
    results = await test_coordinator.run_scenario(scenario)
    
    # Log results summary
    logger.info(f"Test completed: {len(results.market_data)} currency pairs monitored")
    logger.info(f"User statistics: {results.user_stats.get('total_requests', 0)} total requests, "
                f"{results.user_stats.get('error_rate', 0)*100:.1f}% error rate")
    
    if results.bottlenecks:
        logger.warning(f"Identified bottlenecks: {', '.join(results.bottlenecks)}")
    else:
        logger.info("No system bottlenecks identified")
    
    logger.info(f"Test duration: {(results.end_time - results.start_time).total_seconds():.1f} seconds")


if __name__ == '__main__':
    # Example: run standalone stress test environment
    # This part seems to use EnvironmentConfig and StressTestEnvironment which might be defined elsewhere
    # Keeping it as is, but it might need adjustments depending on those classes.
    # from .environment import StressTestEnvironment, EnvironmentConfig # Assuming these exist
    
    # # Load configuration from YAML
    # try:
    #     env_conf = EnvironmentConfig() # Assumes default config path is correct
    #     config = env_conf.config

    #     profile = config.get('environment_name', 'default_stress_profile')
    #     stress_env = StressTestEnvironment(profile_name=profile, config=config)

    #     # Execute the stress test lifecycle
    #     report = stress_env.run() # Assuming run is synchronous or handled internally

    #     # Print results
    #     import json
    #     print(json.dumps(report, indent=2, default=str))
    # except FileNotFoundError:
    #      logger.error("Stress test configuration file not found. Cannot run __main__ example.")
    # except ImportError:
    #      logger.error("Could not import StressTestEnvironment or EnvironmentConfig. Cannot run __main__ example.")
    # except Exception as e:
    #      logger.error(f"Error running __main__ example: {e}", exc_info=True)
    
    # Let's run the async sample test instead if the environment part fails
    async def run_main_async():
    """
    Run main async.
    
    """

        logging.basicConfig(level=logging.INFO)
        await run_sample_integrated_test()

    asyncio.run(run_main_async())
