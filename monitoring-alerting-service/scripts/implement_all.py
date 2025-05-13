"""
Complete Monitoring and Alerting Implementation Script

This script runs all the implementation steps for the monitoring and alerting infrastructure.
"""
import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('monitoring-alerting-implementation')


from monitoring_alerting_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@with_exception_handling
def run_script(script_name, args=None):
    """Run a script with the given arguments."""
    logger.info(f'Running {script_name}')
    cmd = [sys.executable, f'monitoring-alerting-service/scripts/{script_name}'
        ]
    if args:
        cmd.extend(args)
    try:
        subprocess.run(cmd, check=True)
        logger.info(f'{script_name} completed successfully')
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f'{script_name} failed: {e}')
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description=
        'Implement all monitoring and alerting components')
    parser.add_argument('--skip-tracing', action='store_true', help=
        'Skip distributed tracing implementation')
    parser.add_argument('--skip-baselines', action='store_true', help=
        'Skip performance baseline establishment')
    parser.add_argument('--skip-testing', action='store_true', help=
        'Skip regular performance testing setup')
    parser.add_argument('--skip-slos', action='store_true', help=
        'Skip SLO and SLI implementation')
    args = parser.parse_args()
    success = True
    if not args.skip_tracing:
        if not run_script('implement_distributed_tracing.py', ['--all']):
            logger.warning(
                'Distributed tracing implementation failed, continuing with other steps'
                )
            success = False
    if not args.skip_baselines:
        if not run_script('establish_performance_baselines.py', ['--all',
            '--skip-tests']):
            logger.warning(
                'Performance baseline establishment failed, continuing with other steps'
                )
            success = False
    if not args.skip_testing:
        if not run_script('setup_regular_performance_testing.py'):
            logger.warning(
                'Regular performance testing setup failed, continuing with other steps'
                )
            success = False
    if not args.skip_slos:
        if not run_script('implement_slos_slis.py'):
            logger.warning('SLO and SLI implementation failed')
            success = False
    if success:
        logger.info(
            'All monitoring and alerting components implemented successfully')
    else:
        logger.warning(
            'Some monitoring and alerting components failed to implement')
        logger.info(
            'Please check the logs for details and run the failed steps individually'
            )


if __name__ == '__main__':
    main()
