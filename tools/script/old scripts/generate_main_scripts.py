#!/usr/bin/env python3
"""
Script to generate main.py files for all services.
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

def generate_main_script(service_name: str, service_config: dict) -> str:
    """
    Generate a main.py file for a service.
    
    Args:
        service_name: Name of the service
        service_config: Service configuration
        
    Returns:
        Main script content
    """
    port = service_config['port']
    endpoints = service_config['endpoints']
    
    # Format the service name for display
    display_name = ' '.join(word.capitalize() for word in service_name.split('-'))
    
    # Generate the script
    script = f"""#!/usr/bin/env python3
\"\"\"
Main module for the {display_name}.
\"\"\"

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="{display_name}",
    description="API for the {display_name}",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check.
    
    Returns:
        Dict[str, str]: Description of return value
    
    """

    \"\"\"
    Health check endpoint.
    
    Returns:
        Health status
    \"\"\"
    return {{"status": "healthy"}}

# Liveness probe
@app.get("/health/liveness")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check.
    
    Returns:
        Dict[str, str]: Description of return value
    
    """

    \"\"\"
    Liveness probe for Kubernetes.
    
    Returns:
        Simple status response
    \"\"\"
    return {{"status": "alive"}}

# Readiness probe
@app.get("/health/readiness")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check.
    
    Returns:
        Dict[str, str]: Description of return value
    
    """

    \"\"\"
    Readiness probe for Kubernetes.
    
    Returns:
        Simple status response
    \"\"\"
    return {{"status": "ready"}}

# Metrics endpoint
@app.get("/metrics")
async def metrics() -> Response:
    """
    Metrics.
    
    Returns:
        Response: Description of return value
    
    """

    \"\"\"
    Prometheus metrics endpoint.
    
    Returns:
        Metrics in Prometheus format
    \"\"\"
    return Response(
        content="# Metrics will be provided by the Prometheus client",
        media_type="text/plain"
    )
"""
    
    # Add API endpoints
    for endpoint in endpoints:
        method = endpoint['method'].lower()
        path = endpoint['endpoint']
        name = path.split('/')[-1].replace('-', '_')
        
        script += f"""
@app.{method}("{path}")
async def {name}() -> Dict[str, Any]:
    \"\"\"
    {name.capitalize()} endpoint.
    
    Returns:
        {name.capitalize()} data
    \"\"\"
    # This is a mock implementation
    return {{"status": "success", "data": []}}
"""
    
    # Add test interaction endpoints
    script += """
# Test interaction endpoints
@app.get("/api/v1/test/portfolio-interaction")
async def test_portfolio_interaction() -> Dict[str, Any]:
    """
    script += f'    """\n    Test interaction with the Portfolio Management Service.\n    \n    Returns:\n        Test result\n    """\n    # This is a mock implementation\n    return {{"status": "success", "message": "Interaction with Portfolio Management Service successful"}}\n'
    
    script += """
@app.get("/api/v1/test/risk-interaction")
async def test_risk_interaction() -> Dict[str, Any]:
    """
    script += f'    """\n    Test interaction with the Risk Management Service.\n    \n    Returns:\n        Test result\n    """\n    # This is a mock implementation\n    return {{"status": "success", "message": "Interaction with Risk Management Service successful"}}\n'
    
    script += """
@app.get("/api/v1/test/feature-interaction")
async def test_feature_interaction() -> Dict[str, Any]:
    """
    script += f'    """\n    Test interaction with the Feature Store Service.\n    \n    Returns:\n        Test result\n    """\n    # This is a mock implementation\n    return {{"status": "success", "message": "Interaction with Feature Store Service successful"}}\n'
    
    script += """
@app.get("/api/v1/test/ml-interaction")
async def test_ml_interaction() -> Dict[str, Any]:
    """
    script += f'    """\n    Test interaction with the ML Integration Service.\n    \n    Returns:\n        Test result\n    """\n    # This is a mock implementation\n    return {{"status": "success", "message": "Interaction with ML Integration Service successful"}}\n'
    
    # Add main function
    script += f"""
def main():
    """
    Main.
    
    """

    \"\"\"Main function to run the service.\"\"\"
    import uvicorn
    
    logger.info("Starting {display_name}")
    uvicorn.run(app, host="0.0.0.0", port={port})

if __name__ == "__main__":
    main()
"""
    
    return script

def main():
    """Main function to generate main.py files."""
    logger.info("Starting generation of main.py files")
    
    # Get the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Generate main.py files for all services
    for service_name, service_config in SERVICE_CONFIG.items():
        logger.info(f"Generating main.py for {service_name}")
        
        # Generate the script
        script = generate_main_script(service_name, service_config)
        
        # Create the service directory if it doesn't exist
        service_dir = os.path.join(root_dir, service_name)
        os.makedirs(service_dir, exist_ok=True)
        
        # Write the script to a file
        script_path = os.path.join(service_dir, 'main.py')
        with open(script_path, 'w') as f:
            f.write(script)
        
        logger.info(f"Generated main.py at {script_path}")
    
    logger.info("Finished generating main.py files")
    return 0

if __name__ == '__main__':
    sys.exit(main())
