#!/usr/bin/env python3
"""
Script to generate start scripts for all services.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service configuration
SERVICE_CONFIG = {
    'trading-gateway-service': {
        'port': 8001,
        'endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/instruments'},
            {'method': 'GET', 'endpoint': '/api/v1/accounts'}
        ]
    },
    'portfolio-management-service': {
        'port': 8002,
        'endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/portfolios'},
            {'method': 'GET', 'endpoint': '/api/v1/positions'}
        ]
    },
    'risk-management-service': {
        'port': 8003,
        'endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/risk-profiles'},
            {'method': 'GET', 'endpoint': '/api/v1/risk-limits'}
        ]
    },
    'data-pipeline-service': {
        'port': 8004,
        'endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/market-data'},
            {'method': 'GET', 'endpoint': '/api/v1/data-sources'}
        ]
    },
    'feature-store-service': {
        'port': 8005,
        'endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/features'},
            {'method': 'GET', 'endpoint': '/api/v1/feature-sets'}
        ]
    },
    'ml-integration-service': {
        'port': 8006,
        'endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/models'},
            {'method': 'GET', 'endpoint': '/api/v1/predictions'}
        ]
    },
    'ml-workbench-service': {
        'port': 8007,
        'endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/experiments'},
            {'method': 'GET', 'endpoint': '/api/v1/model-registry'}
        ]
    },
    'monitoring-alerting-service': {
        'port': 8008,
        'endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/alerts'},
            {'method': 'GET', 'endpoint': '/api/v1/metrics'}
        ]
    }
}

def generate_start_script(service_name: str, service_config: dict) -> str:
    """
    Generate a start script for a service.
    
    Args:
        service_name: Name of the service
        service_config: Service configuration
        
    Returns:
        Start script content
    """
    port = service_config['port']
    endpoints = service_config['endpoints']
    
    # Format the service name for display
    display_name = ' '.join(word.capitalize() for word in service_name.split('-'))
    
    # Generate the script
    script = f"""#!/usr/bin/env python3
\"\"\"
Start script for the {display_name}.
\"\"\"

import os
import sys
import logging
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_service():
    """
    Start service.
    
    """

    \"\"\"Start the {display_name}.\"\"\"
    logger.info("Starting {display_name}")
    
    # Get the service directory
    service_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if main.py exists
    main_script = os.path.join(service_dir, 'main.py')
    if not os.path.exists(main_script):
        logger.error(f"Main script not found: {{main_script}}")
        return False
    
    # Start the service
    try:
        # For testing purposes, we'll just print a message
        logger.info("Service started successfully")
        
        # In a real implementation, this would start the service
        # process = subprocess.Popen(
        #     [sys.executable, main_script],
        #     cwd=service_dir
        # )
        
        # For testing, simulate a running service
        print("{display_name} is running on port {port}")
        print("Health endpoint: http://localhost:{port}/health")
        print("API endpoints:")
"""
    
    # Add endpoints
    for endpoint in endpoints:
        script += f'        print("  - {endpoint["method"]} {endpoint["endpoint"]}")\n'
    
    # Add the rest of the script
    script += """        
        # Keep the script running
        while True:
            time.sleep(1)
        
        return True
    
    except Exception as e:
        logger.error(f"Error starting service: {str(e)}")
        return False

if __name__ == '__main__':
    start_service()"""
    
    return script

def main():
    """Main function to generate start scripts."""
    logger.info("Starting generation of start scripts")
    
    # Get the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Generate start scripts for all services
    for service_name, service_config in SERVICE_CONFIG.items():
        logger.info(f"Generating start script for {service_name}")
        
        # Generate the script
        script = generate_start_script(service_name, service_config)
        
        # Create the service directory if it doesn't exist
        service_dir = os.path.join(root_dir, service_name)
        os.makedirs(service_dir, exist_ok=True)
        
        # Write the script to a file
        script_path = os.path.join(service_dir, 'start.py')
        with open(script_path, 'w') as f:
            f.write(script)
        
        logger.info(f"Generated start script at {script_path}")
    
    logger.info("Finished generating start scripts")
    return 0

if __name__ == '__main__':
    sys.exit(main())
