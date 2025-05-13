"""
Run Feature Store Monitoring

This script runs the feature store monitoring dashboard and alerts.
"""
import argparse
import logging
import sys
import asyncio
import threading
from typing import Dict, Any
from strategy_execution_engine.monitoring.feature_store_dashboard import run_dashboard
from strategy_execution_engine.monitoring.feature_store_alerts import monitor_feature_store_metrics, feature_store_alerts


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def run_dashboard_thread(port: int, debug: bool) ->None:
    """
    Run the feature store dashboard in a separate thread.
    
    Args:
        port: Port to run the dashboard on
        debug: Whether to run in debug mode
    """
    print(f'Starting feature store dashboard on port {port}')
    run_dashboard(port=port, debug=debug)


async def main_async(config: Dict[str, Any]) ->None:
    """
    Run the feature store monitoring.
    
    Args:
        config: Configuration dictionary
    """
    feature_store_alerts.alert_config.update(config_manager.get('alerts', {}))
    await monitor_feature_store_metrics(interval=config.get(
        'monitoring_interval', 60))


@with_exception_handling
def main() ->None:
    """Run the feature store monitoring."""
    parser = argparse.ArgumentParser(description=
        'Run the feature store monitoring')
    parser.add_argument('--port', type=int, default=8050, help=
        'Port to run the dashboard on')
    parser.add_argument('--debug', action='store_true', help=
        'Run in debug mode')
    parser.add_argument('--monitoring-interval', type=int, default=60, help
        ='Monitoring interval in seconds')
    parser.add_argument('--log-alerts', action='store_true', help=
        'Enable logging alerts')
    parser.add_argument('--email-alerts', action='store_true', help=
        'Enable email alerts')
    parser.add_argument('--slack-alerts', action='store_true', help=
        'Enable Slack alerts')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[
        logging.StreamHandler(sys.stdout)])
    config = {'monitoring_interval': args.monitoring_interval, 'alerts': {
        'enabled': True, 'log_alerts': args.log_alerts, 'email_alerts':
        args.email_alerts, 'slack_alerts': args.slack_alerts}}
    dashboard_thread = threading.Thread(target=run_dashboard_thread, args=(
        args.port, args.debug), daemon=True)
    dashboard_thread.start()
    try:
        asyncio.run(main_async(config))
    except KeyboardInterrupt:
        print('Monitoring stopped by user')


if __name__ == '__main__':
    main()
