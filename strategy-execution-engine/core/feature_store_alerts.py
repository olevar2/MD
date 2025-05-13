"""
Feature Store Alerts Module

This module provides alerting for the feature store client.
"""
import time
import logging
import json
import os
import smtplib
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from core_foundations.utils.logger import get_logger
from core.feature_store_metrics import feature_store_metrics
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    has_slack = True
except ImportError:
    has_slack = False


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AlertLevel:
    """Alert levels for feature store alerts."""
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'


class FeatureStoreAlerts:
    """
    Alerts manager for feature store client.
    
    This class provides alerting for the feature store client,
    including email, Slack, and logging alerts.
    
    Attributes:
        logger: Logger instance
        alert_config: Dictionary with alert configuration
        alert_history: List of recent alerts
        alert_handlers: Dictionary of alert handlers
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one alerts manager exists."""
        if cls._instance is None:
            cls._instance = super(FeatureStoreAlerts, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @with_exception_handling
    def __init__(self, config: Optional[Dict[str, Any]]=None):
        """
        Initialize the alerts manager.
        
        Args:
            config: Dictionary with alert configuration
        """
        if self._initialized:
            return
        self.logger = get_logger('feature_store_alerts')
        default_config = {'enabled': True, 'log_alerts': True,
            'email_alerts': False, 'slack_alerts': False, 'alert_levels': [
            AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL],
            'email_config': {'smtp_server': 'smtp.example.com', 'smtp_port':
            587, 'smtp_username': '', 'smtp_password': '', 'from_address':
            'alerts@example.com', 'to_addresses': ['admin@example.com']},
            'slack_config': {'token': '', 'channel': '#alerts'},
            'thresholds': {'error_rate': {'warning': 0.05, 'error': 0.1,
            'critical': 0.2}, 'response_time': {'warning': 500, 'error': 
            1000, 'critical': 2000}, 'cache_hit_rate': {'warning': 0.5,
            'error': 0.3, 'critical': 0.1}, 'fallback_rate': {'warning': 
            0.05, 'error': 0.1, 'critical': 0.2}}, 'alert_cooldown': {
            AlertLevel.INFO: 3600, AlertLevel.WARNING: 1800, AlertLevel.
            ERROR: 600, AlertLevel.CRITICAL: 300}}
        self.alert_config = {**default_config, **config or {}}
        self.alert_history = []
        self.alert_handlers = {'log': self._log_alert, 'email': self.
            _email_alert, 'slack': self._slack_alert}
        self.slack_client = None
        if self.alert_config['slack_alerts'] and has_slack:
            try:
                self.slack_client = WebClient(token=self.alert_config[
                    'slack_config']['token'])
                self.logger.info('Slack client initialized')
            except Exception as e:
                self.logger.warning(
                    f'Failed to initialize Slack client: {str(e)}')
        self._initialized = True
        self.logger.info('Feature store alerts initialized')

    @with_exception_handling
    def check_metrics(self) ->None:
        """
        Check metrics and trigger alerts if thresholds are exceeded.
        """
        if not self.alert_config['enabled']:
            return
        try:
            metrics = feature_store_metrics.get_metrics()
            total_api_calls = metrics['api_calls']['total']
            if total_api_calls > 0:
                error_rate = metrics['errors']['total'] / total_api_calls
                self._check_threshold('error_rate', error_rate,
                    'Error Rate', f'{error_rate:.2%}')
            avg_response_time = metrics['performance']['avg_response_time_ms']
            self._check_threshold('response_time', avg_response_time,
                'Response Time', f'{avg_response_time:.2f}ms')
            total_cache_requests = metrics['cache']['hits'] + metrics['cache'][
                'misses']
            if total_cache_requests > 0:
                cache_hit_rate = metrics['cache']['hits'
                    ] / total_cache_requests
                self._check_threshold('cache_hit_rate', cache_hit_rate,
                    'Cache Hit Rate', f'{cache_hit_rate:.2%}')
            if total_api_calls > 0:
                fallback_rate = metrics['fallbacks']['total'] / total_api_calls
                self._check_threshold('fallback_rate', fallback_rate,
                    'Fallback Rate', f'{fallback_rate:.2%}')
        except Exception as e:
            self.logger.error(f'Error checking metrics: {str(e)}')

    def _check_threshold(self, metric_name: str, value: float, display_name:
        str, formatted_value: str) ->None:
        """
        Check if a metric exceeds thresholds and trigger alerts.
        
        Args:
            metric_name: Name of the metric
            value: Current value of the metric
            display_name: Display name for the metric
            formatted_value: Formatted value for display
        """
        thresholds = self.alert_config['thresholds'].get(metric_name)
        if not thresholds:
            return
        if metric_name in ['error_rate', 'fallback_rate']:
            if value >= thresholds.get('critical', float('inf')):
                self.trigger_alert(AlertLevel.CRITICAL,
                    f'{display_name} Critical',
                    f"{display_name} is {formatted_value}, exceeding critical threshold of {thresholds['critical']:.2%}"
                    , {'metric': metric_name, 'value': value, 'threshold':
                    thresholds['critical']})
            elif value >= thresholds.get('error', float('inf')):
                self.trigger_alert(AlertLevel.ERROR,
                    f'{display_name} Error',
                    f"{display_name} is {formatted_value}, exceeding error threshold of {thresholds['error']:.2%}"
                    , {'metric': metric_name, 'value': value, 'threshold':
                    thresholds['error']})
            elif value >= thresholds.get('warning', float('inf')):
                self.trigger_alert(AlertLevel.WARNING,
                    f'{display_name} Warning',
                    f"{display_name} is {formatted_value}, exceeding warning threshold of {thresholds['warning']:.2%}"
                    , {'metric': metric_name, 'value': value, 'threshold':
                    thresholds['warning']})
        elif metric_name == 'response_time':
            if value >= thresholds.get('critical', float('inf')):
                self.trigger_alert(AlertLevel.CRITICAL,
                    f'{display_name} Critical',
                    f"{display_name} is {formatted_value}, exceeding critical threshold of {thresholds['critical']}ms"
                    , {'metric': metric_name, 'value': value, 'threshold':
                    thresholds['critical']})
            elif value >= thresholds.get('error', float('inf')):
                self.trigger_alert(AlertLevel.ERROR,
                    f'{display_name} Error',
                    f"{display_name} is {formatted_value}, exceeding error threshold of {thresholds['error']}ms"
                    , {'metric': metric_name, 'value': value, 'threshold':
                    thresholds['error']})
            elif value >= thresholds.get('warning', float('inf')):
                self.trigger_alert(AlertLevel.WARNING,
                    f'{display_name} Warning',
                    f"{display_name} is {formatted_value}, exceeding warning threshold of {thresholds['warning']}ms"
                    , {'metric': metric_name, 'value': value, 'threshold':
                    thresholds['warning']})
        elif metric_name == 'cache_hit_rate':
            if value <= thresholds.get('critical', 0):
                self.trigger_alert(AlertLevel.CRITICAL,
                    f'{display_name} Critical',
                    f"{display_name} is {formatted_value}, below critical threshold of {thresholds['critical']:.2%}"
                    , {'metric': metric_name, 'value': value, 'threshold':
                    thresholds['critical']})
            elif value <= thresholds.get('error', 0):
                self.trigger_alert(AlertLevel.ERROR,
                    f'{display_name} Error',
                    f"{display_name} is {formatted_value}, below error threshold of {thresholds['error']:.2%}"
                    , {'metric': metric_name, 'value': value, 'threshold':
                    thresholds['error']})
            elif value <= thresholds.get('warning', 0):
                self.trigger_alert(AlertLevel.WARNING,
                    f'{display_name} Warning',
                    f"{display_name} is {formatted_value}, below warning threshold of {thresholds['warning']:.2%}"
                    , {'metric': metric_name, 'value': value, 'threshold':
                    thresholds['warning']})

    def trigger_alert(self, level: str, title: str, message: str, data:
        Optional[Dict[str, Any]]=None) ->None:
        """
        Trigger an alert.
        
        Args:
            level: Alert level
            title: Alert title
            message: Alert message
            data: Additional data for the alert
        """
        if not self.alert_config['enabled'] or level not in self.alert_config[
            'alert_levels']:
            return
        cooldown = self.alert_config['alert_cooldown'].get(level, 0)
        if self._is_in_cooldown(level, title, cooldown):
            return
        alert = {'level': level, 'title': title, 'message': message,
            'timestamp': datetime.now().isoformat(), 'data': data or {}}
        self.alert_history.append(alert)
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        if self.alert_config['log_alerts']:
            self.alert_handlers['log'](alert)
        if self.alert_config['email_alerts']:
            self.alert_handlers['email'](alert)
        if self.alert_config['slack_alerts'] and self.slack_client:
            self.alert_handlers['slack'](alert)

    def _is_in_cooldown(self, level: str, title: str, cooldown: int) ->bool:
        """
        Check if a similar alert is in cooldown.
        
        Args:
            level: Alert level
            title: Alert title
            cooldown: Cooldown period in seconds
            
        Returns:
            True if a similar alert is in cooldown, False otherwise
        """
        now = datetime.now()
        for alert in reversed(self.alert_history):
            if alert['level'] == level and alert['title'] == title:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if (now - alert_time).total_seconds() < cooldown:
                    return True
        return False

    def _log_alert(self, alert: Dict[str, Any]) ->None:
        """
        Log an alert.
        
        Args:
            alert: Alert to log
        """
        level = alert['level']
        message = f"{alert['title']}: {alert['message']}"
        if level == AlertLevel.INFO:
            self.logger.info(message)
        elif level == AlertLevel.WARNING:
            self.logger.warning(message)
        elif level == AlertLevel.ERROR:
            self.logger.error(message)
        elif level == AlertLevel.CRITICAL:
            self.logger.critical(message)

    @with_exception_handling
    def _email_alert(self, alert: Dict[str, Any]) ->None:
        """
        Send an email alert.
        
        Args:
            alert: Alert to send
        """
        try:
            config = self.alert_config['email_config']
            msg = MIMEMultipart()
            msg['From'] = config['from_address']
            msg['To'] = ', '.join(config['to_addresses'])
            msg['Subject'] = f"[{alert['level']}] {alert['title']}"
            body = f"""
            <html>
            <body>
                <h2>{alert['title']}</h2>
                <p><strong>Level:</strong> {alert['level']}</p>
                <p><strong>Time:</strong> {alert['timestamp']}</p>
                <p><strong>Message:</strong> {alert['message']}</p>
                <hr>
                <h3>Additional Data:</h3>
                <pre>{json.dumps(alert['data'], indent=2)}</pre>
            </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']
                ) as server:
                server.starttls()
                if config['smtp_username'] and config['smtp_password']:
                    server.login(config['smtp_username'], config[
                        'smtp_password'])
                server.send_message(msg)
            self.logger.info(f"Email alert sent: {alert['title']}")
        except Exception as e:
            self.logger.error(f'Error sending email alert: {str(e)}')

    @with_exception_handling
    def _slack_alert(self, alert: Dict[str, Any]) ->None:
        """
        Send a Slack alert.
        
        Args:
            alert: Alert to send
        """
        if not has_slack or not self.slack_client:
            return
        try:
            config = self.alert_config['slack_config']
            blocks = [{'type': 'header', 'text': {'type': 'plain_text',
                'text': f"{alert['title']}"}}, {'type': 'section', 'fields':
                [{'type': 'mrkdwn', 'text':
                f"""*Level:*
{alert['level']}"""}, {'type': 'mrkdwn',
                'text': f"""*Time:*
{alert['timestamp']}"""}]}, {'type':
                'section', 'text': {'type': 'mrkdwn', 'text':
                f"""*Message:*
{alert['message']}"""}}]
            if alert['data']:
                blocks.append({'type': 'section', 'text': {'type': 'mrkdwn',
                    'text':
                    f"""*Additional Data:*
```{json.dumps(alert['data'], indent=2)}```"""
                    }})
            color = '#36a64f'
            if alert['level'] == AlertLevel.WARNING:
                color = '#ffcc00'
            elif alert['level'] == AlertLevel.ERROR:
                color = '#ff9900'
            elif alert['level'] == AlertLevel.CRITICAL:
                color = '#ff0000'
            self.slack_client.chat_postMessage(channel=config['channel'],
                blocks=blocks, attachments=[{'color': color}])
            self.logger.info(f"Slack alert sent: {alert['title']}")
        except Exception as e:
            self.logger.error(f'Error sending Slack alert: {str(e)}')


feature_store_alerts = FeatureStoreAlerts()


@async_with_exception_handling
async def monitor_feature_store_metrics(interval: int=60) ->None:
    """
    Monitor feature store metrics and trigger alerts.
    
    Args:
        interval: Monitoring interval in seconds
    """
    while True:
        try:
            feature_store_alerts.check_metrics()
        except Exception as e:
            logging.error(f'Error monitoring feature store metrics: {str(e)}')
        await asyncio.sleep(interval)


if __name__ == '__main__':
    asyncio.run(monitor_feature_store_metrics())
