"""
Configuration for the stress testing environment.
"""
import os
import yaml
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "stress_test_config.yaml") # Default path relative to environment_config.py

class EnvironmentConfig:
    """
    Loads and provides access to stress testing environment configurations.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
    """
      init  .
    
    Args:
        config_path: Description of config_path
    
    """

        self.config_path = config_path
        self.config = self._load_config()
        logger.info(f"Stress test configuration loaded from: {self.config_path}")

    def _load_config(self) -> dict:
        """
        Loads the YAML configuration file.
        """
        try:
            # TODO: Implement robust path handling (e.g., relative to project root)
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found at {self.config_path}. Using default empty config.")
                # TODO: Define default configuration structure
                return {
                    "environment_name": "default_stress_env",
                    "target_endpoints": {},
                    "load_profile": {},
                    "reporting": {},
                    "resource_limits": {},
                    "distributed_workers": []
                }

            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                # TODO: Add validation for config structure
                return config_data
        except Exception as e:
            logger.error(f"Error loading stress test config from {self.config_path}: {e}", exc_info=True)
            raise

    def get_target_endpoint(self, service_name: str) -> str | None:
        """Gets the target endpoint URL for a specific service."""
        return self.config.get('target_endpoints', {}).get(service_name)

    def get_load_profile(self) -> dict:
        """Gets the load profile configuration (e.g., RPS, duration)."""
        return self.config.get('load_profile', {})

    def get_reporting_config(self) -> dict:
        """Gets the configuration for reporting results."""
        return self.config.get('reporting', {})

    def get_resource_limits(self) -> dict:
        """Gets resource limits for the test environment (if applicable)."""
        return self.config.get('resource_limits', {})

    def get_distributed_workers(self) -> list:
        """Gets the list of worker nodes for distributed testing."""
        return self.config.get('distributed_workers', [])

    def is_distributed_mode(self) -> bool:
        """Checks if distributed testing is configured."""
        return bool(self.get_distributed_workers())

# Example usage:
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     # Assume config/stress_test_config.yaml exists
#     env_config = EnvironmentConfig()
#     print(f"Target Order Service: {env_config.get_target_endpoint('order_service')}")
#     print(f"Load Profile: {env_config.get_load_profile()}")
#     print(f"Distributed Mode: {env_config.is_distributed_mode()}")

# TODO: Create a sample stress_test_config.yaml file
# Example config/stress_test_config.yaml:
# ```yaml
# environment_name: high_volume_forex_stress
# target_endpoints:
#   order_service: http://order-service.stress-test-ns.svc.cluster.local:8080/api/v1/orders
#   market_data_service: ws://market-data-service.stress-test-ns.svc.cluster.local:9090/stream
#   # ... other services
# load_profile:
#   type: rps # requests per second
#   target_rps: 5000
#   duration_seconds: 600
#   ramp_up_seconds: 60
# reporting:
#   format: prometheus # or influxdb, json_file
#   endpoint: http://prometheus-pushgateway.monitoring.svc.cluster.local:9091
# resource_limits:
#   load_generator_cpu: "2"
#   load_generator_memory: "4Gi"
# distributed_workers: # Optional: for distributed load generation
#   # - worker-1.stress-test-ns.svc.cluster.local
#   # - worker-2.stress-test-ns.svc.cluster.local
# ```
