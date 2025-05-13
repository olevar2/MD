"""
Resource and Cost Monitoring Exporter

This module collects and exports metrics related to resource utilization
and cloud costs for the Forex trading platform services.
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from prometheus_client import Gauge, start_http_server
import boto3
from azure.mgmt.costmanagement import CostManagementClient


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ResourceCostExporter:
    """
    ResourceCostExporter class.
    
    Attributes:
        Add attributes here
    """


    def __init__(self, port: int=9090, cloud_provider: str='aws',
        refresh_interval: int=300):
    """
      init  .
    
    Args:
        port: Description of port
        cloud_provider: Description of cloud_provider
        refresh_interval: Description of refresh_interval
    
    """

        self.cloud_provider = cloud_provider
        self.refresh_interval = refresh_interval
        self.cpu_utilization = Gauge('resource_cpu_utilization',
            'CPU utilization percentage by service', ['service', 'instance'])
        self.memory_utilization = Gauge('resource_memory_utilization',
            'Memory utilization percentage by service', ['service', 'instance']
            )
        self.storage_utilization = Gauge('resource_storage_utilization',
            'Storage utilization percentage by service', ['service',
            'storage_type'])
        self.hourly_cost = Gauge('service_hourly_cost',
            'Hourly cost by service in USD', ['service', 'resource_type'])
        self.daily_cost = Gauge('service_daily_cost',
            'Daily cost by service in USD', ['service', 'resource_type'])
        self.cost_efficiency = Gauge('service_cost_efficiency',
            'Cost efficiency score (0-1)', ['service'])
        self._init_cloud_clients()

    def _init_cloud_clients(self):
        """Initialize cloud provider specific clients."""
        if self.cloud_provider == 'aws':
            self.cost_client = boto3.client('ce')
            self.ec2_client = boto3.client('ec2')
            self.cloudwatch = boto3.client('cloudwatch')
        elif self.cloud_provider == 'azure':
            pass

    def collect_resource_metrics(self):
        """Collect current resource utilization metrics."""
        if self.cloud_provider == 'aws':
            self._collect_aws_metrics()
        elif self.cloud_provider == 'azure':
            self._collect_azure_metrics()

    def _collect_aws_metrics(self):
        """Collect AWS specific metrics."""
        instances = self.ec2_client.describe_instances()
        for reservation in instances['Reservations']:
            for instance in reservation['Instances']:
                instance_id = instance['InstanceId']
                tags = {tag['Key']: tag['Value'] for tag in instance.get(
                    'Tags', [])}
                service = tags.get('Service', 'unknown')
                cpu_stats = self.cloudwatch.get_metric_statistics(Namespace
                    ='AWS/EC2', MetricName='CPUUtilization', Dimensions=[{
                    'Name': 'InstanceId', 'Value': instance_id}], StartTime
                    =datetime.utcnow() - timedelta(minutes=5), EndTime=
                    datetime.utcnow(), Period=300, Statistics=['Average'])
                if cpu_stats['Datapoints']:
                    self.cpu_utilization.labels(service=service, instance=
                        instance_id).set(cpu_stats['Datapoints'][0]['Average'])

    def collect_cost_metrics(self):
        """Collect cost metrics from cloud provider."""
        if self.cloud_provider == 'aws':
            self._collect_aws_costs()
        elif self.cloud_provider == 'azure':
            self._collect_azure_costs()

    def _collect_aws_costs(self):
        """Collect AWS specific cost metrics."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        cost_response = self.cost_client.get_cost_and_usage(TimePeriod={
            'Start': start_date.strftime('%Y-%m-%d'), 'End': end_date.
            strftime('%Y-%m-%d')}, Granularity='HOURLY', Metrics=[
            'UnblendedCost'], GroupBy=[{'Type': 'TAG', 'Key': 'Service'}, {
            'Type': 'DIMENSION', 'Key': 'SERVICE'}])
        for result in cost_response['ResultsByTime']:
            for group in result['Groups']:
                service = group['Keys'][0].split('$')[-1]
                resource_type = group['Keys'][1]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                self.hourly_cost.labels(service=service, resource_type=
                    resource_type).set(cost)

    def calculate_cost_efficiency(self):
        """Calculate cost efficiency scores based on resource utilization and costs."""
        for service in self._get_services():
            cpu_util = float(self.cpu_utilization.labels(service=service))
            memory_util = float(self.memory_utilization.labels(service=service)
                )
            efficiency = (cpu_util + memory_util) / (2 * 100)
            self.cost_efficiency.labels(service=service).set(efficiency)

    def _get_services(self) ->List[str]:
        """Get list of all services being monitored."""
        if self.cloud_provider == 'aws':
            response = self.ec2_client.describe_instances()
            services = set()
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    for tag in instance.get('Tags', []):
                        if tag['Key'] == 'Service':
                            services.add(tag['Value'])
            return list(services)
        return []

    @with_exception_handling
    def run(self):
        """Start the exporter and collect metrics periodically."""
        start_http_server(self.port)
        while True:
            try:
                self.collect_resource_metrics()
                self.collect_cost_metrics()
                self.calculate_cost_efficiency()
            except Exception as e:
                print(f'Error collecting metrics: {str(e)}')
            time.sleep(self.refresh_interval)


if __name__ == '__main__':
    exporter = ResourceCostExporter()
    exporter.run()
