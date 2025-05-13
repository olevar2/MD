"""
Runner module.

This module provides functionality for...
"""

import os
import time
import json
import argparse
import logging
import yaml

from testing.stress_testing.environment import StressTestEnvironment
from testing.stress_testing.environment_config import EnvironmentConfig
from testing.stress_testing.analysis.performance_report import PerformanceReport

class StressTestRunner:
    """
    Orchestrates stress tests:
      - Executes tests with optional ramp-up and stable phases
      - Collects system and application metrics
      - Generates performance reports and visualizations
    """
    def __init__(self, config_path: str):
    """
      init  .
    
    Args:
        config_path: Description of config_path
    
    """

        # Load environment configuration
        self.env_config = EnvironmentConfig(config_path)
        self.config = self.env_config.config
        # Reporting settings (e.g., format, endpoint, directory)
        self.reporting_config = self.env_config.get_reporting_config()

    def run_test(self) -> dict:
    """
    Run test.
    
    Returns:
        dict: Description of return value
    
    """

        profile_name = self.config.get('environment_name', 'stress_test')
        test_config = self.config
        env = StressTestEnvironment(profile_name, test_config)

        # Setup and initialize
        env.setup()
        env.initialize_load_generators()
        env.start_monitoring()

        # Ramp-up phase
        load_profile = test_config.get('load_profile', {})
        target_rate = load_profile.get('rate_per_second', None)
        ramp_up = load_profile.get('ramp_up_seconds', 0)
        total_duration = test_config.get('duration_seconds', 0)
        stable_duration = max(total_duration - ramp_up, 0)

        if ramp_up and target_rate:
            steps = 10
            step_duration = ramp_up / steps
            logging.info(f"Gradual ramp-up to {target_rate} over {ramp_up}s in {steps} steps.")
            for i in range(1, steps + 1):
                current_rate = target_rate * (i / steps)
                for gen in env.load_generators:
                    if hasattr(gen, 'rate_per_second'):
                        setattr(gen, 'rate_per_second', current_rate)
                env.start_load_generation()
                logging.info(f"Ramp step {i}/{steps}: rate={current_rate:.1f} for {step_duration:.1f}s")
                time.sleep(step_duration)
                env.stop_load_generation()

        # Stable load phase
        if target_rate and stable_duration > 0:
            for gen in env.load_generators:
                if hasattr(gen, 'rate_per_second'):
                    setattr(gen, 'rate_per_second', target_rate)
            logging.info(f"Running stable load at {target_rate} for {stable_duration}s.")
            env.start_load_generation()
            time.sleep(stable_duration)
            env.stop_load_generation()

        # Monitoring and evaluation
        env.stop_monitoring()
        env.evaluate()
        summary = env.report()

        # Persist raw metrics
        report_dir = self.reporting_config.get('report_directory', 'stress-test-reports')
        os.makedirs(report_dir, exist_ok=True)
        metrics_file = os.path.join(report_dir, f"{profile_name}_metrics.jsonl")
        with open(metrics_file, 'w') as mf:
            for metric, series in env._collected_metrics.items():
                for ts, value in series:
                    mf.write(json.dumps({'metric': metric, 'timestamp': ts, 'value': value}) + '\n')
        logging.info(f"Raw metrics saved to: {metrics_file}")

        # Generate detailed performance report
        perf_report = PerformanceReport(self.config, metrics_file)
        perf_report.generate_report(report_format=self.reporting_config.get('format', 'json'))

        # Cleanup
        env.teardown()
        return summary


def main():
    """
    Main.
    
    """

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Stress Test Runner for Forex Trading Platform')
    parser.add_argument('-c', '--config', help='Path to stress test YAML configuration')
    parser.add_argument('-f', '--format', choices=['json', 'html'], help='Report format override')
    args = parser.parse_args()

    # Determine configuration file
    config_path = args.config or EnvironmentConfig().config_path

    runner = StressTestRunner(config_path)
    if args.format:
        runner.reporting_config['format'] = args.format

    summary = runner.run_test()
    logging.info('Stress test completed. Summary:')
    logging.info(json.dumps(summary, indent=2, default=str))

if __name__ == '__main__':
    main()
