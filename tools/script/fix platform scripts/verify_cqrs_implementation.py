#!/usr/bin/env python
"""
Verify CQRS implementation in the Forex Trading Platform.

This script verifies the CQRS implementation in the Forex Trading Platform services.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Services to verify
SERVICES = [
    "causal-analysis-service",
    "backtesting-service",
    "market-analysis-service",
    "analysis-coordinator-service",
]

def verify_cqrs_implementation(service_name=None):
    """
    Verify CQRS implementation in the specified service or all services.
    
    Args:
        service_name: Name of the service to verify, or None to verify all services
    
    Returns:
        Dictionary with verification results
    """
    logger.info("Verifying CQRS implementation")
    
    services_to_verify = [service_name] if service_name else SERVICES
    results = {}
    
    for service in services_to_verify:
        logger.info(f"Verifying CQRS implementation in {service}")
        
        service_path = project_root / service
        
        # Check if the service exists
        if not service_path.exists():
            logger.warning(f"Service {service} not found at {service_path}")
            results[service] = {
                "success": False,
                "completion": 0.0,
                "message": f"Service not found at {service_path}",
            }
            continue
        
        # Check for CQRS components
        cqrs_dir = service_path / "cqrs"
        if not cqrs_dir.exists():
            cqrs_dir = service_path / service.replace("-", "_") / "cqrs"
            if not cqrs_dir.exists():
                logger.warning(f"CQRS directory not found in {service}")
                results[service] = {
                    "success": False,
                    "completion": 0.0,
                    "message": "CQRS directory not found",
                }
                continue
        
        # Check for command and query models
        commands_file = cqrs_dir / "commands.py"
        queries_file = cqrs_dir / "queries.py"
        
        # Check for command and query handlers
        handlers_dir = cqrs_dir / "handlers"
        command_handlers_file = handlers_dir / "command_handlers.py"
        query_handlers_file = handlers_dir / "query_handlers.py"
        
        # Check for read and write repositories
        read_repos_dir = None
        write_repos_dir = None
        
        if service_path.joinpath(service.replace("-", "_"), "repositories", "read_repositories").exists():
            read_repos_dir = service_path.joinpath(service.replace("-", "_"), "repositories", "read_repositories")
        elif service_path.joinpath("repositories", "read_repositories").exists():
            read_repos_dir = service_path.joinpath("repositories", "read_repositories")
        
        if service_path.joinpath(service.replace("-", "_"), "repositories", "write_repositories").exists():
            write_repos_dir = service_path.joinpath(service.replace("-", "_"), "repositories", "write_repositories")
        elif service_path.joinpath("repositories", "write_repositories").exists():
            write_repos_dir = service_path.joinpath("repositories", "write_repositories")
        
        # Calculate completion percentage
        components = [
            commands_file.exists(),
            queries_file.exists(),
            handlers_dir.exists(),
            command_handlers_file.exists() if handlers_dir.exists() else False,
            query_handlers_file.exists() if handlers_dir.exists() else False,
            read_repos_dir is not None,
            write_repos_dir is not None,
        ]
        
        completion = sum(components) / len(components) * 100.0
        
        # Determine success
        success = completion >= 90.0
        
        results[service] = {
            "success": success,
            "completion": completion,
            "components": {
                "commands": commands_file.exists(),
                "queries": queries_file.exists(),
                "handlers_dir": handlers_dir.exists(),
                "command_handlers": command_handlers_file.exists() if handlers_dir.exists() else False,
                "query_handlers": query_handlers_file.exists() if handlers_dir.exists() else False,
                "read_repositories": read_repos_dir is not None,
                "write_repositories": write_repos_dir is not None,
            },
        }
    
    # Calculate overall completion
    overall_completion = sum(result["completion"] for result in results.values()) / len(results) if results else 0.0
    
    return {
        "services": results,
        "overall_completion": overall_completion,
    }

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Verify CQRS implementation in the Forex Trading Platform")
    parser.add_argument("--service", help="Name of the service to verify")
    parser.add_argument("--output", help="Path to output file")
    args = parser.parse_args()
    
    results = verify_cqrs_implementation(args.service)
    
    # Print results
    logger.info("Verification results:")
    for service, result in results["services"].items():
        logger.info(f"Service: {service}")
        logger.info(f"  Success: {result['success']}")
        logger.info(f"  Completion: {result['completion']:.2f}%")
        if "components" in result:
            logger.info("  Components:")
            for component, exists in result["components"].items():
                logger.info(f"    {component}: {exists}")
    
    logger.info(f"Overall completion: {results['overall_completion']:.2f}%")
    
    # Save results to file
    output_path = args.output or os.path.join(project_root, "tools", "output", "cqrs_implementation_verification.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())