"""
Utility for seeding test data in the E2E test environment.
Handles loading test data from files and seeding into services.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests

logger = logging.getLogger(__name__)


class DataSeeder:
    """
    Seeds test data into the Forex trading platform services.
    Supports different data profiles for various testing scenarios.
    """
    
    def __init__(self):
        """Initialize the data seeder."""
        self.seeded_data = {}  # Track what data has been seeded for cleanup
        
    def seed_all_services(
        self, 
        service_endpoints: Dict[str, str], 
        profile: str = "default",
        test_data_path: Optional[str] = None
    ) -> None:
        """
        Seed all services with test data.
        
        Args:
            service_endpoints: Dictionary of service name to endpoint URL
            profile: Data profile to use (e.g., 'default', 'highvolume')
            test_data_path: Path to test data directory
        """
        logger.info(f"Seeding all services with profile: {profile}")
        
        # Use default test data path if not provided
        if not test_data_path:
            test_data_path = str(Path(__file__).parent.parent / "fixtures" / "test_data")
            
        # Seed each service in the correct order to respect dependencies
        for service_name in self._get_service_seeding_order():
            if service_name in service_endpoints:
                self._seed_service(
                    service_name=service_name,
                    endpoint=service_endpoints[service_name],
                    profile=profile,
                    test_data_path=test_data_path
                )
                
        logger.info("All services seeded successfully")
                
    def reset_all_services(
        self,
        service_endpoints: Dict[str, str],
        profile: str = "default",
        test_data_path: Optional[str] = None
    ) -> None:
        """
        Reset all services to a clean state and reseed them.
        
        Args:
            service_endpoints: Dictionary of service name to endpoint URL
            profile: Data profile to use (e.g., 'default', 'highvolume')
            test_data_path: Path to test data directory
        """
        logger.info("Resetting all services")
        
        # Clear data from each service in reverse order
        for service_name in reversed(self._get_service_seeding_order()):
            if service_name in service_endpoints:
                self._clear_service_data(
                    service_name=service_name,
                    endpoint=service_endpoints[service_name]
                )
                
        # Reseed all services
        self.seed_all_services(
            service_endpoints=service_endpoints,
            profile=profile,
            test_data_path=test_data_path
        )
        
    def _seed_service(
        self,
        service_name: str,
        endpoint: str,
        profile: str,
        test_data_path: str
    ) -> None:
        """
        Seed a specific service with test data.
        
        Args:
            service_name: Name of the service to seed
            endpoint: Service endpoint URL
            profile: Data profile to use
            test_data_path: Path to test data directory
        """
        logger.info(f"Seeding service: {service_name}")
        
        # Skip non-HTTP services
        if not endpoint.startswith("http"):
            logger.info(f"Skipping non-HTTP service: {service_name}")
            return
            
        # Build path to test data file
        data_file = os.path.join(
            test_data_path,
            profile,
            f"{service_name}.json"
        )
        
        # Check if data file exists
        if not os.path.exists(data_file):
            logger.info(f"No test data file found for {service_name} with profile {profile}")
            return
            
        # Load test data
        try:
            with open(data_file, 'r') as f:
                test_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load test data for {service_name}: {e}")
            return
            
        # Seed each entity type
        for entity_type, entities in test_data.items():
            self._seed_entity_type(
                service_name=service_name,
                endpoint=endpoint,
                entity_type=entity_type,
                entities=entities
            )
            
    def _seed_entity_type(
        self,
        service_name: str,
        endpoint: str,
        entity_type: str,
        entities: List[Dict[str, Any]]
    ) -> None:
        """
        Seed a specific entity type in a service.
        
        Args:
            service_name: Name of the service to seed
            endpoint: Service endpoint URL
            entity_type: Type of entity to seed (e.g., 'users', 'accounts')
            entities: List of entity data to seed
        """
        logger.info(f"Seeding {len(entities)} {entity_type} in {service_name}")
        
        # Skip if no entities to seed
        if not entities:
            return
            
        # Determine API endpoint for this entity type
        api_endpoint = f"{endpoint}/api/{entity_type}"
        
        # Track seeded data for cleanup
        if service_name not in self.seeded_data:
            self.seeded_data[service_name] = {}
            
        if entity_type not in self.seeded_data[service_name]:
            self.seeded_data[service_name][entity_type] = []
        
        # Seed each entity
        for entity in entities:
            try:
                response = requests.post(
                    api_endpoint,
                    json=entity,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code in (200, 201):
                    # Store ID of created entity if available
                    entity_id = None
                    try:
                        resp_data = response.json()
                        entity_id = resp_data.get("id") or resp_data.get("_id")
                    except Exception:
                        pass
                        
                    if entity_id:
                        self.seeded_data[service_name][entity_type].append(entity_id)
                else:
                    logger.warning(
                        f"Failed to seed entity in {service_name}/{entity_type}: "
                        f"Status {response.status_code}"
                    )
            except Exception as e:
                logger.error(f"Error seeding entity in {service_name}/{entity_type}: {e}")
                
    def _clear_service_data(self, service_name: str, endpoint: str) -> None:
        """
        Clear data from a service.
        
        Args:
            service_name: Name of the service to clear
            endpoint: Service endpoint URL
        """
        logger.info(f"Clearing data from service: {service_name}")
        
        # Skip non-HTTP services
        if not endpoint.startswith("http"):
            return
            
        # Skip if no data was seeded for this service
        if service_name not in self.seeded_data:
            return
            
        # Clear each entity type
        for entity_type, entity_ids in self.seeded_data[service_name].items():
            for entity_id in entity_ids:
                try:
                    api_endpoint = f"{endpoint}/api/{entity_type}/{entity_id}"
                    requests.delete(api_endpoint, timeout=5)
                except Exception as e:
                    logger.warning(f"Error clearing entity {entity_id} from {service_name}: {e}")
                    
        # Clear tracking data
        self.seeded_data[service_name] = {}
        
    @staticmethod
    def _get_service_seeding_order() -> List[str]:
        """
        Get the order in which services should be seeded.
        Order is important to ensure dependencies are met.
        
        Returns:
            List of service names in the order they should be seeded
        """
        return [
            "auth_service",             # Authentication must be first
            "user_service",             # Users need to exist before other entities
            "market_data_provider",     # Market data is needed for many other services
            "exchange_connector",       # Exchange connection info needs to be set up
            "portfolio_service",        # Portfolio needs to exist for positions
            "strategy_execution_engine", # Strategies need to be defined
            "risk_management_service",  # Risk rules need to be defined
            "analysis_engine_service",  # Analysis engines need to be configured
            "order_service",            # Orders depend on users and portfolios
            "position_service",         # Positions depend on orders
            "notification_service",     # Notifications depend on many other entities
            "ml_integration_service",   # ML models depend on historical data
            "feature_store_service",    # Features depend on market data
            "monitoring_alerting_service", # Monitoring needs services to be running
            "ui_service"                # UI configuration comes last
        ]
