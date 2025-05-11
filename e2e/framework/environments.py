\
\
\

Manages isolated test environments for E2E testing.::

This module defines the TestEnvironment class responsible for setting up,
managing, and tearing down the necessary infrastructure (services, databases, etc.)
for running end-to-end tests. It supports different configurations, including
mocked services and data seeding.
\"\"\"

import logging
import os
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Dict, Optional, List, Generator, AsyncGenerator
import docker
from docker.models.containers import Container
from docker.errors import NotFound

# Assuming utils classes are reusable or will be refactored/moved here later
# If these imports fail, the classes might need to be moved or adjusted.
try:
    from e2e.utils.test_environment import ServiceManager, TestDatabase, TestEnvironmentConfig, EnvironmentMode
except ImportError:
    # Provide basic stubs if the import fails, indicating dependency
    logging.warning("Could not import from e2e.utils.test_environment. Using basic stubs.")

    class EnvironmentMode(str, Enum):
        SIMULATED = "simulated"
        HYBRID = "hybrid"
        PRODUCTION = "production"

    @dataclass
    class TestEnvironmentConfig:
        mode: EnvironmentMode = EnvironmentMode.SIMULATED
        use_live_market_data: bool = False
        use_persistent_storage: bool = False
        enable_service_logs: bool = True
        browser_headless: bool = True

    class ServiceManager:
        def __init__(self, config: TestEnvironmentConfig):
            self.config = config
            self.client = docker.from_env()
            self.containers: Dict[str, Container] = {}
            self.service_urls: Dict[str, str] = {}
            self.network = None
            logging.info("Initialized STUB ServiceManager")

        def create_network(self, name: str):
             try:
                 self.network = self.client.networks.create(name, driver="bridge")
                 logging.info(f"Created Docker network: {name}")
             except docker.errors.APIError as e:
                 # Handle cases where network might already exist (e.g., from previous unclean shutdown)
                 if "already exists" in str(e):
                     logging.warning(f"Docker network '{name}' already exists. Attempting to use it.")
                     self.network = self.client.networks.get(name)
                 else:
                     raise

        def start_service(self, service_name: str, image: str,
                          environment: Optional[Dict[str, str]] = None,
                          ports: Optional[Dict[str, int]] = None,
                          network_name: Optional[str] = None,
                          depends_on: Optional[List[str]] = None, # Placeholder for dependency mgmt
                          healthcheck_url: Optional[str] = None,
                          **kwargs) -> str:
            logging.info(f"STUB: Starting service {service_name}...")
            # Simplified: Assume service starts instantly and is available at localhost:port
            port_key = list(ports.keys())[0] if ports else None
            host_port = ports[port_key] if port_key else None
            url = f"http://localhost:{host_port}" if host_port else f"http://{service_name}:8000"
            self.service_urls[service_name] = url
            # In a real implementation, this would involve docker run, waiting, health checks
            time.sleep(1) # Simulate startup time
            logging.info(f"STUB: Service {service_name} presumed started at {url}")
            return url

        def stop_all_services(self):
            logging.info("STUB: Stopping all services...")
            self.containers = {}
            self.service_urls = {}
            if self.network:
                try:
                    logging.info(f"Removing Docker network: {self.network.name}")
                    self.network.remove()
                except (docker.errors.APIError, docker.errors.NotFound) as e:
                     logging.warning(f"Could not remove network {self.network.name}: {e}")
                self.network = None


        def get_service_url(self, service_name: str) -> Optional[str]:
            return self.service_urls.get(service_name)

    class TestDatabase:
        def __init__(self, config: TestEnvironmentConfig):
            self.config = config
            self.databases = set()
            logging.info("Initialized STUB TestDatabase")

        def create_test_database(self, db_name: str):
            logging.info(f"STUB: Creating test database {db_name}...")
            self.databases.add(db_name)

        def load_test_data(self, db_name: str, data_fixture_path: str):
            logging.info(f"STUB: Loading data from {data_fixture_path} into {db_name}...")
            if not os.path.exists(data_fixture_path):
                 logging.warning(f"Data fixture path not found: {data_fixture_path}")
                 return
            # In real implementation: connect to DB and execute SQL/load data

        def cleanup_all_databases(self):
            logging.info("STUB: Cleaning up all test databases...")
            self.databases = set()


logger = logging.getLogger(__name__)

DEFAULT_SERVICE_CONFIG = {
    # Define default images, ports, env vars for services
    # Example:
    "api-gateway": {
        "image": "forex-trading-platform/api-gateway:latest",
        "ports": {"8000/tcp": 8000},
        "environment": {"LOG_LEVEL": "INFO"},
        "healthcheck_url": "/health",
    },
    "data-pipeline-service": {
        "image": "forex-trading-platform/data-pipeline-service:latest",
        "ports": {"8001/tcp": 8001},
        "environment": {"KAFKA_BOOTSTRAP_SERVERS": "kafka:9092"},
         "healthcheck_url": "/health/ready",
    },
     "trading-gateway-service": {
        "image": "forex-trading-platform/trading-gateway-service:latest",
        "ports": {"8008/tcp": 8008},
        "environment": {
            "EXECUTION_MODE": "SIMULATED",
            "KAFKA_BOOTSTRAP_SERVERS": "kafka:9092",
            "DATABASE_URL": "postgresql+psycopg2://user:password@postgres:5432/trading_db"
            },
         "healthcheck_url": "/health/ready",
    },
    # Add other services: feature-store, analysis-engine, portfolio, risk, etc.
    "kafka": {
        "image": "confluentinc/cp-kafka:latest",
        "ports": {"9092/tcp": 9092},
        "environment": {
            "KAFKA_BROKER_ID": 1,
            "KAFKA_ZOOKEEPER_CONNECT": "zookeeper:2181",
            "KAFKA_ADVERTISED_LISTENERS": "PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:29092", # Allow external connection if needed
            "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP": "PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT",
            "KAFKA_INTER_BROKER_LISTENER_NAME": "PLAINTEXT",
            "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR": 1,
        },
        "depends_on": ["zookeeper"]
    },
    "zookeeper": {
        "image": "confluentinc/cp-zookeeper:latest",
        "ports": {"2181/tcp": 2181},
        "environment": {
            "ZOOKEEPER_CLIENT_PORT": 2181,
            "ZOOKEEPER_TICK_TIME": 2000,
        }
    },
     "postgres": {
        "image": "postgres:14-alpine",
        "ports": {"5432/tcp": 5432},
        "environment": {
            "POSTGRES_USER": "user",
            "POSTGRES_PASSWORD": "password",
            "POSTGRES_DB": "trading_db",
        },
        "volumes": {"postgres_data": "/var/lib/postgresql/data"} # Optional persistence
    },
    # Mock service example
     "mock-broker": {
        "image": "forex-trading-platform/mock-broker:latest", # Assuming a mock image exists
        "ports": {"9000/tcp": 9000},
    }
}


class TestEnvironment:
    \"\"\"
    Manages the setup and teardown of an isolated E2E test environment.

    Orchestrates Docker containers for services, databases, and potentially
    other dependencies like message queues. Provides methods for data seeding
    and accessing service endpoints.
    \"\"\"

    def __init__(self, config: Optional[TestEnvironmentConfig] = None, network_name: str = "forex_e2e_test_net"):
        \"\"\"
        Initialize the test environment.

        Args:
            config: Configuration for the environment mode and settings.
            network_name: Name for the Docker network to connect services.
        \"\"\"
        self.config = config or TestEnvironmentConfig()
        self.service_manager = ServiceManager(self.config)
        self.database_manager = TestDatabase(self.config)
        self.network_name = f"{network_name}_{int(time.time())}" # Unique network name per instance
        self._services_to_start: Dict[str, Dict] = {}
        self._databases_to_seed: Dict[str, str] = {}
        self._is_setup = False
        self._volumes_to_create = set() # Track named volumes needed

        logger.info(f"Initializing TestEnvironment in {self.config.mode.value} mode.")

    def add_service(self, service_name: str, use_mock: bool = False, **override_config):
        \"\"\"
        Declare a service required for the test environment.

        Args:
            service_name: The logical name of the service (e.g., 'trading-gateway').
            use_mock: If True, try to use a mock version of the service if available.
            **override_config: Specific configurations to override the defaults
                               (e.g., image, ports, environment).
        \"\"\"
        if self._is_setup:
            raise RuntimeError("Cannot add services after the environment is set up.")

        effective_service_name = service_name
        if use_mock:
            mock_name = f"mock-{service_name}"
            if mock_name in DEFAULT_SERVICE_CONFIG:
                effective_service_name = mock_name
                logger.info(f"Using mock service '{effective_service_name}' for '{service_name}'.")
            else:
                logger.warning(f"Mock service '{mock_name}' not defined; using real service '{service_name}'.")

        if effective_service_name not in DEFAULT_SERVICE_CONFIG:
            raise ValueError(f"Service '{effective_service_name}' is not defined in DEFAULT_SERVICE_CONFIG.")

        base_config = DEFAULT_SERVICE_CONFIG[effective_service_name].copy()

        # Deep merge environment variables if overridden
        if 'environment' in override_config and 'environment' in base_config:
            merged_env = base_config['environment'].copy()
            merged_env.update(override_config['environment'])
            override_config['environment'] = merged_env

        # Merge volumes - collect volume names
        if 'volumes' in base_config:
             for vol_name in base_config['volumes']:
                 self._volumes_to_create.add(vol_name)
        if 'volumes' in override_config:
             for vol_name in override_config['volumes']:
                 self._volumes_to_create.add(vol_name)
             if 'volumes' in base_config:
                 merged_vols = base_config['volumes'].copy()
                 merged_vols.update(override_config['volumes'])
                 override_config['volumes'] = merged_vols


        merged_config = {**base_config, **override_config}
        self._services_to_start[service_name] = merged_config # Store under logical name
        logger.debug(f"Added service '{service_name}' (using config for '{effective_service_name}') with config: {merged_config}")


    def seed_database(self, db_service_name: str, fixture_path: str):
        \"\"\"
        Declare data seeding for a database associated with a service.

        Args:
            db_service_name: The name of the database service (e.g., 'postgres').
                             Needs to be added via add_service first.
            fixture_path: Path to the SQL or data file in e2e/fixtures/.
        \"\"\"
        if self._is_setup:
            raise RuntimeError("Cannot seed database after the environment is set up.")

        # Assume db_service_name corresponds to a logical DB name used by TestDatabase
        # This might need refinement based on how TestDatabase identifies DBs
        db_identifier = f"{db_service_name}_db" # Example identifier
        full_fixture_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fixtures', fixture_path))
        self._databases_to_seed[db_identifier] = full_fixture_path
        logger.debug(f"Added database seeding for '{db_identifier}' using fixture: {full_fixture_path}")


    def setup(self):
        \"\"\"Creates the network, starts declared services, and seeds databases.\"\"\"
        if self._is_setup:
            logger.warning("Setup already called.")
            return

        logger.info(f"Setting up test environment (Network: {self.network_name})...")
        try:
            # 1. Create Docker Network
            self.service_manager.create_network(self.network_name)

            # 2. Create necessary Docker Volumes
            for vol_name in self._volumes_to_create:
                 try:
                     self.service_manager.client.volumes.create(name=vol_name)
                     logger.info(f"Created Docker volume: {vol_name}")
                 except docker.errors.APIError as e:
                     if "already exists" in str(e):
                         logger.warning(f"Docker volume '{vol_name}' already exists.")
                     else:
                         raise


            # 3. Start Services (handle dependencies if necessary - basic example)
            started_services = set()
            services_to_start_ordered = list(self._services_to_start.items()) # Basic order, needs topological sort for complex deps

            # Simple dependency handling: start services without deps first
            services_to_start_ordered.sort(key=lambda item: len(item[1].get('depends_on', [])))

            for service_name, config in services_to_start_ordered:
                 if service_name in started_services:
                     continue

                 # Check dependencies (basic check)
                 dependencies = config.get('depends_on', [])
                 missing_deps = [dep for dep in dependencies if dep not in started_services]
                 if missing_deps:
                      # This indicates an issue with the simple sort or circular deps
                      logger.error(f"Cannot start service '{service_name}' due to missing dependencies: {missing_deps}. Check order/definitions.")
                      raise RuntimeError(f"Dependency error for service {service_name}")


                 logger.info(f"Starting service '{service_name}'...")
                 self.service_manager.start_service(
                     service_name=service_name, # Use logical name
                     image=config['image'],
                     environment=config.get('environment'),
                     ports=config.get('ports'),
                     network_name=self.network_name, # Connect to the test network
                     volumes=config.get('volumes'), # Pass volume mappings
                     # Add healthcheck logic here based on config['healthcheck_url']
                     **config.get('extra_docker_opts', {}) # Pass other docker options if needed
                 )
                 started_services.add(service_name)


            # 4. Seed Databases
            if not self.config.use_persistent_storage:
                for db_identifier, fixture_path in self._databases_to_seed.items():
                    # Assuming db_identifier maps to a DB name TestDatabase understands
                    # This might require waiting for the DB container to be ready
                    logger.info(f"Waiting briefly for DB {db_identifier} to initialize...")
                    time.sleep(5) # Simple wait, replace with proper health check
                    self.database_manager.create_test_database(db_identifier)
                    self.database_manager.load_test_data(db_identifier, fixture_path)
            else:
                 logger.info("Persistent storage enabled, skipping database seeding.")


            self._is_setup = True
            logger.info("Test environment setup complete.")

        except Exception as e:
            logger.error(f"Error during test environment setup: {e}", exc_info=True)
            # Attempt cleanup even if setup failed
            self.teardown()
            raise

    def teardown(self):
        \"\"\"Stops services, cleans up databases, and removes the network.\"\"\"
        if not self._is_setup and not self.service_manager.network: # Avoid teardown if setup never started properly unless network exists
             logger.info("Teardown skipped - environment was not fully set up.")
             return

        logger.info("Tearing down test environment...")
        # 1. Stop Services
        self.service_manager.stop_all_services() # This should also remove the network

        # 2. Clean Databases (if not persistent)
        if not self.config.use_persistent_storage:
            self.database_manager.cleanup_all_databases()
        else:
            logger.info("Persistent storage enabled, skipping database cleanup.")

        # 3. Remove Volumes (Optional - depends on whether they should persist)
        # Be cautious with removing volumes if persistence across runs is desired
        # for vol_name in self._volumes_to_create:
        #     try:
        #         volume = self.service_manager.client.volumes.get(vol_name)
        #         volume.remove(force=True) # Force remove if containers used it
        #         logger.info(f"Removed Docker volume: {vol_name}")
        #     except docker.errors.NotFound:
        #         logger.warning(f"Volume '{vol_name}' not found during teardown.")
        #     except docker.errors.APIError as e:
        #         logger.error(f"Failed to remove volume '{vol_name}': {e}")


        self._is_setup = False
        logger.info("Test environment teardown complete.")

    def get_service_url(self, service_name: str) -> Optional[str]:
        \"\"\"Get the base URL for an active service.\"\"\"
        if not self._is_setup:
            logger.warning("Attempted to get service URL before environment setup.")
            return None
        return self.service_manager.get_service_url(service_name)

    # --- Context Manager Protocol ---

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()

    # --- Async Context Manager Protocol (Optional) ---
    # If setup/teardown become async

    # async def __aenter__(self):
    #     await self.setup() # Assuming setup becomes async
    #     return self

    # async def __aexit__(self, exc_type, exc_val, exc_tb):
    #     await self.teardown() # Assuming teardown becomes async


# --- Helper Functions / Fixtures (Example for Pytest) ---

@contextmanager
def create_environment(services: List[str],
                       use_mocks: Optional[Dict[str, bool]] = None,
                       seed_data: Optional[Dict[str, str]] = None,
                       config: Optional[TestEnvironmentConfig] = None) -> Generator['TestEnvironment', None, None]:
    \"\"\"
    Context manager to create and manage a TestEnvironment for specific tests.

    Example Usage:
        services_needed = ["api-gateway", "trading-gateway-service", "postgres"]
        mocks = {"trading-gateway-service": True}
        data = {"postgres": "trading_setup.sql"}

        with create_environment(services_needed, use_mocks=mocks, seed_data=data) as env:
            api_gw_url = env.get_service_url("api-gateway")
            # ... run tests ...

    Args:
        services: List of logical service names required.
        use_mocks: Dictionary mapping service names to True if a mock should be used.
        seed_data: Dictionary mapping database service name to fixture file path.
        config: Optional TestEnvironmentConfig.

    Yields:
        The configured and set up TestEnvironment instance.
    \"\"\"
    env = TestEnvironment(config=config)
    use_mocks = use_mocks or {}
    seed_data = seed_data or {}

    # Add required services
    for service_name in services:
        env.add_service(service_name, use_mock=use_mocks.get(service_name, False))

        # Automatically add dependencies (basic example)
        service_config = DEFAULT_SERVICE_CONFIG.get(f"mock-{service_name}" if use_mocks.get(service_name) else service_name)
        if service_config:
             for dep in service_config.get('depends_on', []):
                 if dep not in services: # Avoid adding if already explicitly requested
                     env.add_service(dep, use_mock=use_mocks.get(dep, False)) # Add dependency


    # Add database seeding
    for db_service, fixture in seed_data.items():
        if db_service not in services:
             logger.warning(f"Database service '{db_service}' for seeding not added. Add it to the 'services' list.")
             # Optionally add the db service automatically if found in defaults
             if db_service in DEFAULT_SERVICE_CONFIG:
                 logger.info(f"Automatically adding DB service '{db_service}' for seeding.")
                 env.add_service(db_service)
             else:
                 continue # Skip seeding if DB service is unknown

        env.seed_database(db_service, fixture)


    try:
        env.setup()
        yield env
    finally:
        env.teardown()


# Example Pytest fixture (can be adapted)
# @pytest.fixture(scope="function") # or "session"
# def live_trading_env():
#     services = ["api-gateway", "trading-gateway-service", "data-pipeline-service", "kafka", "zookeeper", "postgres"]
#     data = {"postgres": "initial_portfolio.sql"}
#     with create_environment(services, seed_data=data) as env:
#         yield env

