"""
Environment module.

This module provides functionality for...
"""

import logging
import time
import threading
import psutil # type: ignore # For system metrics monitoring
from typing import Dict, Any, Optional, List, Tuple
import asyncio
from testing.stress_testing.load_generator import LoadGenerator, MarketDataGenerator, ApiRequestGenerator
from testing.stress_testing.environment_config import EnvironmentConfig

# Placeholder for potential integration with central monitoring service client
# from monitoring_alerting_service.client import MonitoringClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

class StressTestEnvironment:
    """
    Manages the environment setup, monitoring, and evaluation for stress tests.
    """

    def __init__(self, profile_name: str, config: Dict[str, Any]):
        """
        Initializes the stress test environment.

        Args:
            profile_name: Name of the stress test profile (e.g., 'high_load_1hr').
            config: Configuration dictionary containing:
                - duration_seconds (int): How long the stress test should run.
                - load_profile (Dict): Parameters for the load generator (e.g., users, ramp-up).
                - monitored_components (List[str]): List of system components/services to monitor.
                - performance_thresholds (Dict[str, Dict[str, float]]): Pass/fail criteria, e.g.,
                    {
                        'cpu_utilization_percent': {'max': 80.0},
                        'memory_utilization_percent': {'max': 75.0},
                        'api_latency_p95_ms': {'max': 500.0},
                        'error_rate_percent': {'max': 1.0}
                    }
                - deployment_config (Optional[Dict]): Info needed to deploy/configure services.
        """
        self.profile_name = profile_name
        self.config = config
        self.duration_seconds = config.get('duration_seconds', 60)
        self.performance_thresholds = config.get('performance_thresholds', {})
        self.monitored_components = config.get('monitored_components', [])
        self.deployment_config = config.get('deployment_config', {}) # Placeholder

        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._collected_metrics: Dict[str, List[Tuple[float, float]]] = { # {metric_name: [(timestamp, value)]}
            'cpu_utilization_percent': [],
            'memory_utilization_percent': [],
            # Add application-specific metrics here (e.g., latency, error rate)
        }
        self.test_start_time: Optional[float] = None
        self.test_end_time: Optional[float] = None
        self.results: Dict[str, Any] = {}

        # Initialize load generators and threads
        self.load_generators: List[LoadGenerator] = []
        self._load_generator_threads: List[threading.Thread] = []
        # Initialize environment config
        config_path = self.config.get('config_path') if isinstance(self.config.get('config_path'), str) else None
        self.env_config = EnvironmentConfig(config_path=config_path) if config_path else EnvironmentConfig()
        # Distributed workers from config
        self.distributed_workers: List[str] = self.env_config.get_distributed_workers()

        logging.info(f"Initialized StressTestEnvironment for profile: {self.profile_name}")

    def setup(self):
        """
        Sets up the testing environment.
        - Deploys/configures necessary services based on deployment_config.
        - Initializes connections to monitoring systems.
        """
        logging.info("Setting up stress test environment...")
        # Placeholder: Implement service deployment/configuration logic here
        # e.g., using docker-compose, kubectl, or custom scripts based on self.deployment_config
        logging.info("Simulating service deployment/configuration...")
        time.sleep(2) # Simulate setup time

        # Placeholder: Initialize connections to monitoring endpoints or agents
        # self.monitoring_client.connect()

        logging.info("Stress test environment setup complete.")

    def _monitor_resources(self):
        """Background task to collect system metrics."""
        logging.info("Monitoring thread started.")
        while self._monitoring_active:
            try:
                timestamp = time.time()
                cpu_percent = psutil.cpu_percent(interval=None) # Use interval=None for non-blocking
                mem_info = psutil.virtual_memory()
                mem_percent = mem_info.percent

                self._collected_metrics['cpu_utilization_percent'].append((timestamp, cpu_percent))
                self._collected_metrics['memory_utilization_percent'].append((timestamp, mem_percent))

                # Placeholder: Query application-specific metrics from monitoring system
                # latency = self.monitoring_client.get_metric('api_latency_p95_ms')
                # error_rate = self.monitoring_client.get_metric('error_rate_percent')
                # if latency is not None: self._collected_metrics['api_latency_p95_ms'].append((timestamp, latency))
                # if error_rate is not None: self._collected_metrics['error_rate_percent'].append((timestamp, error_rate))

                logging.debug(f"Metrics collected: CPU={cpu_percent:.1f}%, Mem={mem_percent:.1f}%")

            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")

            # Adjust sleep interval as needed
            time.sleep(5) # Collect metrics every 5 seconds
        logging.info("Monitoring thread stopped.")

    def start_monitoring(self):
        """Starts the background resource monitoring."""
        if self._monitoring_active:
            logging.warning("Monitoring is already active.")
            return

        logging.info("Starting resource monitoring...")
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitor_resources, name="ResourceMonitor", daemon=True)
        self._monitoring_thread.start()
        self.test_start_time = time.time()

    def stop_monitoring(self):
        """Stops the background resource monitoring."""
        if not self._monitoring_active:
            logging.warning("Monitoring is not active.")
            return

        logging.info("Stopping resource monitoring...")
        self.test_end_time = time.time()
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10) # Wait for thread to finish
            if self._monitoring_thread.is_alive():
                logging.warning("Monitoring thread did not stop gracefully.")
        self._monitoring_thread = None
        logging.info("Resource monitoring stopped.")

    def run(self) -> Dict[str, Any]:
        """
        Executes the full stress test lifecycle: setup, monitor, generate load, evaluate, report, teardown.
        """
        try:
            self.setup()
            self.initialize_load_generators()
            self.start_monitoring()
            self.start_load_generation()

            logging.info(f"Stress test running for {self.duration_seconds} seconds...")
            time.sleep(self.duration_seconds)

            self.stop_load_generation()
            self.stop_monitoring()
            self.evaluate()
            report = self.report()
            return report
        except Exception as e:
            logging.exception(f"Error during stress test run for profile '{self.profile_name}': {e}")
            self.results['overall_status'] = 'ERROR'
            self.results['error_message'] = str(e)
            return self.results
        finally:
            self.teardown()

    # New methods for load generation and analysis
    def initialize_load_generators(self):
        """
        Instantiate load generators based on the load_profile and distributed worker config.
        """
        load_profile = self.config.get('load_profile', {})
        gen_type = load_profile.get('type')
        if gen_type in ('market_data', 'tick', 'bar'):
            gen = MarketDataGenerator(load_profile)
            self.load_generators.append(gen)
        elif gen_type in ('api', 'api_request'):
            gen = ApiRequestGenerator(load_profile)
            self.load_generators.append(gen)
        else:
            logging.warning(f"Unsupported load generator type '{gen_type}'")

        if self.distributed_workers:
            logging.info(f"Distributed mode enabled for workers: {self.distributed_workers}")
            for worker in self.distributed_workers:
                # TODO: Dispatch load generator to remote worker via SSH or RPC
                logging.info(f"Dispatching load generator to remote worker: {worker}")

    def start_load_generation(self):
        """
        Starts all configured load generators in separate threads.
        """
        logging.info("Starting load generation...")
        for gen in self.load_generators:
            thread = threading.Thread(
                target=lambda g=gen: asyncio.run(g.start()),
                name=f"LoadGen-{gen.__class__.__name__}", daemon=True
            )
            thread.start()
            self._load_generator_threads.append(thread)

    def stop_load_generation(self):
        """
        Stops all running load generators and joins threads.
        """
        logging.info("Stopping load generation...")
        for gen in self.load_generators:
            gen.stop()
        for thread in self._load_generator_threads:
            thread.join(timeout=5)
        self._load_generator_threads.clear()

    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """
        Analyze collected metrics to identify performance bottlenecks.
        """
        analysis: Dict[str, Any] = {}
        for metric, data in self._collected_metrics.items():
            thresholds = self.performance_thresholds.get(metric, {})
            max_thresh = thresholds.get('max')
            if max_thresh and data:
                exceed_count = sum(1 for _, v in data if v > max_thresh)
                analysis[metric] = {'exceed_count': exceed_count, 'threshold': max_thresh}
        logging.info(f"Bottleneck analysis: {analysis}")
        self.results['bottleneck_analysis'] = analysis
        return analysis

    def evaluate(self) -> bool:
        """
        Evaluates the collected metrics against the defined performance thresholds.

        Returns:
            bool: True if all thresholds passed, False otherwise.
        """
        if not self.test_start_time or not self.test_end_time:
            logging.error("Cannot evaluate: Test start/end times not recorded.")
            return False
        if not self._collected_metrics['cpu_utilization_percent']: # Check if any metrics were collected
             logging.error("Cannot evaluate: No metrics collected.")
             return False

        logging.info("Evaluating performance metrics against thresholds...")
        passed = True
        evaluation_summary: Dict[str, Dict[str, Any]] = {}

        for metric_name, thresholds in self.performance_thresholds.items():
            if metric_name not in self._collected_metrics or not self._collected_metrics[metric_name]:
                logging.warning(f"Metric '{metric_name}' defined in thresholds but not collected.")
                evaluation_summary[metric_name] = {'status': 'MISSING_DATA', 'message': 'Metric not collected'}
                passed = False # Consider missing data a failure? Or just warn? Depends on requirements.
                continue

            metric_values = [value for timestamp, value in self._collected_metrics[metric_name]]
            # Calculate relevant aggregate (e.g., max, average, p95)
            # For simplicity, using max here. Real implementation might need more complex aggregation.
            actual_max = max(metric_values) if metric_values else 0
            # actual_avg = sum(metric_values) / len(metric_values) if metric_values else 0
            # actual_p95 = ... # Calculate percentile if needed

            status = 'PASS'
            message = f"Max value {actual_max:.2f} within threshold."

            if 'max' in thresholds and actual_max > thresholds['max']:
                status = 'FAIL'
                message = f"Max value {actual_max:.2f} EXCEEDED threshold ({thresholds['max']})"
                passed = False
            elif 'min' in thresholds and actual_max < thresholds['min']: # Assuming max is the value to check against min too
                 status = 'FAIL'
                 message = f"Max value {actual_max:.2f} BELOW threshold ({thresholds['min']})"
                 passed = False

            evaluation_summary[metric_name] = {
                'status': status,
                'threshold_max': thresholds.get('max'),
                'threshold_min': thresholds.get('min'),
                'actual_max': actual_max,
                'message': message
            }
            logging.info(f" - {metric_name}: {status} ({message})")

        self.results['evaluation'] = evaluation_summary
        self.results['overall_status'] = 'PASS' if passed else 'FAIL'
        logging.info(f"Overall evaluation result: {self.results['overall_status']}")
        return passed

    def report(self) -> Dict[str, Any]:
        """
        Generates a summary report of the stress test.

        Returns:
            Dict[str, Any]: A dictionary containing the test results and summary.
        """
        logging.info("Generating stress test report...")
        if not self.results:
             self.evaluate() # Ensure evaluation happened

        report_data = {
            'profile_name': self.profile_name,
            'config': self.config,
            'test_start_time': self.test_start_time,
            'test_end_time': self.test_end_time,
            'duration_seconds': self.test_end_time - self.test_start_time if self.test_start_time and self.test_end_time else None,
            'overall_status': self.results.get('overall_status', 'UNKNOWN'),
            'evaluation_summary': self.results.get('evaluation', {}),
            'collected_metrics_summary': { # Provide aggregate summaries
                metric: {
                    'count': len(values),
                    'min': min(v for t, v in values) if values else None,
                    'max': max(v for t, v in values) if values else None,
                    'avg': sum(v for t, v in values) / len(values) if values else None,
                }
                for metric, values in self._collected_metrics.items() if values # Only report metrics with data
            },
            'load_generator_metrics': {
                gen.__class__.__name__: gen.get_metrics() for gen in self.load_generators
            }
        }
        # Invoke bottleneck analysis if not already present
        if 'bottleneck_analysis' not in self.results:
            self.analyze_bottlenecks()
        logging.info(f"Report generated for profile '{self.profile_name}'. Overall Status: {report_data['overall_status']}")
        return report_data


    def teardown(self):
        """
        Cleans up the testing environment.
        - Stops monitoring.
        - Stops/removes deployed services.
        - Releases resources.
        """
        logging.info("Tearing down stress test environment...")
        if self._monitoring_active:
            self.stop_monitoring()

        # Placeholder: Implement service teardown logic here
        # e.g., docker-compose down, kubectl delete, etc.
        logging.info("Simulating service teardown...")
        time.sleep(1)

        # Placeholder: Disconnect from monitoring systems
        # self.monitoring_client.disconnect()

        logging.info("Stress test environment teardown complete.")

    def run(self) -> Dict[str, Any]:
        """
        Executes the full stress test lifecycle: setup, monitor, generate load, evaluate, report, teardown.
        """
        try:
            self.setup()
            self.initialize_load_generators()
            self.start_monitoring()
            self.start_load_generation()

            logging.info(f"Stress test running for {self.duration_seconds} seconds...")
            time.sleep(self.duration_seconds)

            self.stop_load_generation()
            self.stop_monitoring()
            self.evaluate()
            report = self.report()
            return report
        except Exception as e:
            logging.exception(f"Error during stress test run for profile '{self.profile_name}': {e}")
            self.results['overall_status'] = 'ERROR'
            self.results['error_message'] = str(e)
            return self.results # Return partial results/error status
        finally:
            self.teardown()


# Example Usage (typically called from a main stress test runner script)
if __name__ == '__main__':
    test_config = {
        'duration_seconds': 15, # Short duration for example
        'load_profile': {
            'type': 'constant_users',
            'users': 10,
            'spawn_rate': 2
        },
        'monitored_components': ['api_gateway', 'analysis_engine'],
        'performance_thresholds': {
            'cpu_utilization_percent': {'max': 95.0}, # High threshold for example
            'memory_utilization_percent': {'max': 90.0},
            # 'api_latency_p95_ms': {'max': 1000.0}, # Requires app metric collection
            # 'error_rate_percent': {'max': 5.0}     # Requires app metric collection
        },
        'deployment_config': {
            'strategy': 'mock' # Indicates mock deployment for this example
        }
    }

    stress_env = StressTestEnvironment(profile_name='example_short_test', config=test_config)

    # --- Manual Lifecycle Control ---
    # stress_env.setup()
    # stress_env.start_monitoring()
    # print("Simulating external load generation for 15 seconds...")
    # time.sleep(15) # Simulate the load test running
    # print("Load generation finished.")
    # stress_env.stop_monitoring()
    # stress_env.evaluate()
    # final_report = stress_env.report()
    # stress_env.teardown()

    # --- Using the run() method ---
    print("\n--- Running test via run() method ---")
    final_report = stress_env.run()
    print("\n--- Final Report ---")
    import json
    print(json.dumps(final_report, indent=2, default=str)) # Use default=str for non-serializable types if any

    print("\nExample finished.")

