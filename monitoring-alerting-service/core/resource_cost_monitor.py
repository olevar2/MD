"""
Resource and Cost Monitoring Collector

This module collects resource utilization metrics and cloud cost data,
analyzing efficiency and providing optimization recommendations.
"""
import logging
import time
from typing import Dict, List, Any
from prometheus_client import Counter, Gauge, Summary
import os
from datetime import datetime, timedelta
import requests
import boto3
import azure.mgmt.costmanagement
RESOURCE_USAGE = Gauge('resource_usage_percentage',
    'Resource utilization percentage', ['service', 'resource', 'instance'])
SERVICE_COST = Gauge('service_cost_hourly', 'Hourly cost by service', [
    'service', 'resource_type', 'region'])
SCALING_EFFICIENCY = Gauge('scaling_efficiency_score',
    'Resource scaling efficiency score (0-1)', ['service', 'resource_type'])
COST_ANOMALY = Gauge('service_cost_zscore',
    'Z-score of service cost vs historical average', ['service'])


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ResourceCostMonitor:
    """
    ResourceCostMonitor class.
    
    Attributes:
        Add attributes here
    """


    def __init__(self, cloud_provider: str):
    """
      init  .
    
    Args:
        cloud_provider: Description of cloud_provider
    
    """

        self.cloud_provider = cloud_provider
        self.historical_costs = {}
        self.usage_thresholds = {'cpu': 80, 'memory': 80, 'disk': 85,
            'network': 70}
        if cloud_provider == 'aws':
            self.cost_explorer = boto3.client('ce')
        elif cloud_provider == 'azure':
            self.cost_client = azure.mgmt.costmanagement.CostManagementClient()

    @with_exception_handling
    def collect_resource_metrics(self) ->None:
        """Collect resource utilization metrics."""
        try:
            pods = self._get_pod_metrics()
            for pod in pods:
                service = pod['labels'].get('app', 'unknown')
                instance = pod['name']
                cpu_usage = self._calculate_cpu_usage(pod)
                RESOURCE_USAGE.labels(service=service, resource='cpu',
                    instance=instance).set(cpu_usage)
                memory_usage = self._calculate_memory_usage(pod)
                RESOURCE_USAGE.labels(service=service, resource='memory',
                    instance=instance).set(memory_usage)
                self._update_scaling_efficiency(service, cpu_usage,
                    memory_usage)
        except Exception as e:
            logging.error(f'Failed to collect resource metrics: {str(e)}')

    @with_exception_handling
    def collect_cost_metrics(self) ->None:
        """Collect and analyze cost metrics."""
        try:
            current_costs = self._get_current_costs()
            for service, cost_data in current_costs.items():
                SERVICE_COST.labels(service=service, resource_type=
                    cost_data['resource_type'], region=cost_data['region']
                    ).set(cost_data['hourly_cost'])
                if service in self.historical_costs:
                    zscore = self._calculate_cost_zscore(service, cost_data
                        ['hourly_cost'])
                    COST_ANOMALY.labels(service=service).set(zscore)
                self.historical_costs[service] = self.historical_costs.get(
                    service, []) + [cost_data['hourly_cost']]
                self.historical_costs[service] = self.historical_costs[service
                    ][-168:]
        except Exception as e:
            logging.error(f'Failed to collect cost metrics: {str(e)}')

    def generate_optimization_recommendations(self) ->List[Dict[str, Any]]:
        """Generate cost and resource optimization recommendations."""
        recommendations = []
        for service, metrics in self._get_service_metrics().items():
            avg_cpu = metrics.get('cpu_usage', 0)
            avg_memory = metrics.get('memory_usage', 0)
            if avg_cpu < 20 and avg_memory < 20:
                recommendations.append({'service': service, 'type':
                    'downsizing', 'message':
                    f'Consider downsizing {service} resources. Average utilization: CPU {avg_cpu}%, Memory {avg_memory}%'
                    , 'estimated_savings': self.
                    _calculate_downsizing_savings(service)})
        for service, costs in self.historical_costs.items():
            if len(costs) >= 24:
                current_cost = costs[-1]
                avg_cost = sum(costs) / len(costs)
                if current_cost > avg_cost * 1.5:
                    recommendations.append({'service': service, 'type':
                        'cost_anomaly', 'message':
                        f'Investigate high costs in {service}. Current cost is 50% above average.'
                        , 'current_cost': current_cost, 'average_cost':
                        avg_cost})
        return recommendations

    def _calculate_cpu_usage(self, pod: Dict[str, Any]) ->float:
        """Calculate CPU usage percentage for a pod."""
        return pod['usage']['cpu'] / pod['limits']['cpu'] * 100

    def _calculate_memory_usage(self, pod: Dict[str, Any]) ->float:
        """Calculate memory usage percentage for a pod."""
        return pod['usage']['memory'] / pod['limits']['memory'] * 100

    def _update_scaling_efficiency(self, service: str, cpu_usage: float,
        memory_usage: float) ->None:
        """Update scaling efficiency score for a service."""
        cpu_efficiency = 1 - abs(cpu_usage - 75) / 75
        memory_efficiency = 1 - abs(memory_usage - 75) / 75
        efficiency_score = (cpu_efficiency + memory_efficiency) / 2
        SCALING_EFFICIENCY.labels(service=service, resource_type='compute'
            ).set(efficiency_score)

    def _get_current_costs(self) ->Dict[str, Any]:
        """Get current cost data from cloud provider."""
        if self.cloud_provider == 'aws':
            return self._get_aws_costs()
        elif self.cloud_provider == 'azure':
            return self._get_azure_costs()
        return {}

    def _calculate_cost_zscore(self, service: str, current_cost: float
        ) ->float:
        """Calculate z-score for current cost vs historical data."""
        historical = self.historical_costs[service]
        if len(historical) < 2:
            return 0
        mean = sum(historical) / len(historical)
        std_dev = (sum((x - mean) ** 2 for x in historical) / len(historical)
            ) ** 0.5
        if std_dev == 0:
            return 0
        return (current_cost - mean) / std_dev

    def _calculate_downsizing_savings(self, service: str) ->float:
        """Estimate potential savings from downsizing resources."""
        current_costs = self._get_current_costs()
        if service not in current_costs:
            return 0
        return current_costs[service]['hourly_cost'] * 0.3 * 24 * 30
