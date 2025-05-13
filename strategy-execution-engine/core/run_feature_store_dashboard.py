"""
Run Feature Store Dashboard

This script runs the feature store monitoring dashboard.
"""
import argparse
import logging
import sys

from core.feature_store_dashboard import run_dashboard


def main():
    """Run the feature store dashboard."""
    parser = argparse.ArgumentParser(description="Run the feature store monitoring dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Run the dashboard
    print(f"Starting feature store dashboard on port {args.port}")
    run_dashboard(debug=args.debug)


if __name__ == "__main__":
    main()
