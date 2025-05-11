#!/usr/bin/env python3
"""
Runtime Health Analyzer

This script analyzes runtime health of the forex trading platform:
1. Error rates in production
2. Service uptime percentages
3. Resource utilization
4. Crash reports

Output is a JSON file with comprehensive runtime health metrics.
"""

import os
import sys
import json
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import glob
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output"
DEFAULT_LOG_DIR = r"D:\MD\forex_trading_platform\logs"

class RuntimeHealthAnalyzer:
    """Analyzes runtime health of the forex trading platform."""
    
    def __init__(self, project_root: Path, log_dir: Path):
        """
        Initialize the analyzer.
        
        Args:
            project_root: Root directory of the project
            log_dir: Directory containing log files
        """
        self.project_root = project_root
        self.log_dir = log_dir
        
    def analyze_error_rates(self) -> Dict[str, Any]:
        """
        Analyze error rates in production.
        
        Returns:
            Dictionary with error rate metrics
        """
        logger.info("Analyzing error rates...")
        
        error_metrics = {
            'total_errors': 0,
            'errors_by_service': {},
            'errors_by_type': {},
            'errors_by_day': {},
            'error_trend': [],
            'top_errors': []
        }
        
        # Check if log directory exists
        if not os.path.exists(self.log_dir):
            logger.warning(f"Log directory {self.log_dir} does not exist")
            return error_metrics
        
        # Find log files
        log_files = []
        for root, _, files in os.walk(self.log_dir):
            for file in files:
                if file.endswith('.log'):
                    log_files.append(os.path.join(root, file))
        
        # Error patterns to look for
        error_patterns = [
            r'ERROR',
            r'CRITICAL',
            r'FATAL',
            r'Exception',
            r'Error:',
            r'Failed',
            r'Failure'
        ]
        
        # Extract errors from logs
        errors = []
        errors_by_service = defaultdict(int)
        errors_by_type = defaultdict(int)
        errors_by_day = defaultdict(int)
        
        for log_file in log_files:
            try:
                # Determine service from log file path
                service = 'unknown'
                log_path = Path(log_file)
                if 'services' in log_path.parts:
                    service_idx = log_path.parts.index('services') + 1
                    if service_idx < len(log_path.parts):
                        service = log_path.parts[service_idx]
                elif any(s in log_path.stem for s in ['service', 'api', 'server']):
                    service = log_path.stem
                
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Check if line contains an error
                        if any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns):
                            # Extract timestamp if available
                            timestamp = None
                            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', line)
                            if timestamp_match:
                                try:
                                    timestamp_str = timestamp_match.group(1).replace(' ', 'T')
                                    timestamp = datetime.fromisoformat(timestamp_str)
                                except ValueError:
                                    pass
                            
                            # Extract error type
                            error_type = 'Unknown'
                            for pattern in error_patterns:
                                if re.search(pattern, line, re.IGNORECASE):
                                    error_type = pattern.strip('r\'')
                                    break
                            
                            # Extract error message
                            message = line.strip()
                            
                            # Add to errors list
                            errors.append({
                                'timestamp': timestamp.isoformat() if timestamp else None,
                                'service': service,
                                'type': error_type,
                                'message': message
                            })
                            
                            # Update counts
                            errors_by_service[service] += 1
                            errors_by_type[error_type] += 1
                            
                            if timestamp:
                                day = timestamp.date().isoformat()
                                errors_by_day[day] += 1
            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {e}")
        
        # Sort errors by timestamp
        errors.sort(key=lambda e: e['timestamp'] if e['timestamp'] else '')
        
        # Calculate error trend (last 30 days)
        today = datetime.now().date()
        trend_days = []
        for i in range(30):
            day = (today - timedelta(days=i)).isoformat()
            trend_days.append(day)
        
        trend_days.reverse()  # Oldest to newest
        
        error_trend = []
        for day in trend_days:
            error_trend.append({
                'date': day,
                'count': errors_by_day.get(day, 0)
            })
        
        # Find top errors
        error_messages = defaultdict(int)
        for error in errors:
            # Simplify message by removing timestamps, IDs, etc.
            simplified_message = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '', error['message'])
            simplified_message = re.sub(r'0x[0-9a-f]+', 'MEMORY_ADDRESS', simplified_message)
            simplified_message = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 'UUID', simplified_message)
            
            error_messages[simplified_message] += 1
        
        top_errors = []
        for message, count in sorted(error_messages.items(), key=lambda x: x[1], reverse=True)[:10]:
            top_errors.append({
                'message': message,
                'count': count
            })
        
        # Update metrics
        error_metrics['total_errors'] = len(errors)
        error_metrics['errors_by_service'] = dict(errors_by_service)
        error_metrics['errors_by_type'] = dict(errors_by_type)
        error_metrics['errors_by_day'] = dict(errors_by_day)
        error_metrics['error_trend'] = error_trend
        error_metrics['top_errors'] = top_errors
        
        return error_metrics
    
    def analyze_service_uptime(self) -> Dict[str, Any]:
        """
        Analyze service uptime percentages.
        
        Returns:
            Dictionary with uptime metrics
        """
        logger.info("Analyzing service uptime...")
        
        uptime_metrics = {
            'services': {},
            'overall_uptime': None
        }
        
        # Check if log directory exists
        if not os.path.exists(self.log_dir):
            logger.warning(f"Log directory {self.log_dir} does not exist")
            return uptime_metrics
        
        # Find uptime logs
        uptime_logs = []
        for pattern in ['uptime*.log', 'health*.log', 'monitoring*.log']:
            uptime_logs.extend(glob.glob(os.path.join(self.log_dir, '**', pattern), recursive=True))
        
        # Extract uptime information
        service_uptimes = defaultdict(list)
        
        for log_file in uptime_logs:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Look for uptime percentage
                        uptime_match = re.search(r'Service\s+([^\s]+)\s+uptime:\s+(\d+\.?\d*)%', line)
                        if uptime_match:
                            service = uptime_match.group(1)
                            uptime = float(uptime_match.group(2))
                            service_uptimes[service].append(uptime)
            except Exception as e:
                logger.error(f"Error processing uptime log {log_file}: {e}")
        
        # Calculate average uptime for each service
        services = {}
        overall_uptimes = []
        
        for service, uptimes in service_uptimes.items():
            avg_uptime = sum(uptimes) / len(uptimes) if uptimes else None
            services[service] = {
                'uptime_percentage': avg_uptime,
                'measurements': len(uptimes)
            }
            if avg_uptime is not None:
                overall_uptimes.append(avg_uptime)
        
        # Calculate overall uptime
        overall_uptime = sum(overall_uptimes) / len(overall_uptimes) if overall_uptimes else None
        
        # Update metrics
        uptime_metrics['services'] = services
        uptime_metrics['overall_uptime'] = overall_uptime
        
        return uptime_metrics
    
    def analyze_resource_utilization(self) -> Dict[str, Any]:
        """
        Analyze resource utilization.
        
        Returns:
            Dictionary with resource utilization metrics
        """
        logger.info("Analyzing resource utilization...")
        
        resource_metrics = {
            'cpu': {},
            'memory': {},
            'disk': {},
            'network': {}
        }
        
        # Check if log directory exists
        if not os.path.exists(self.log_dir):
            logger.warning(f"Log directory {self.log_dir} does not exist")
            return resource_metrics
        
        # Find resource logs
        resource_logs = []
        for pattern in ['resource*.log', 'metrics*.log', 'monitoring*.log', 'performance*.log']:
            resource_logs.extend(glob.glob(os.path.join(self.log_dir, '**', pattern), recursive=True))
        
        # Extract resource utilization information
        cpu_by_service = defaultdict(list)
        memory_by_service = defaultdict(list)
        disk_by_service = defaultdict(list)
        network_by_service = defaultdict(list)
        
        for log_file in resource_logs:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Look for CPU usage
                        cpu_match = re.search(r'Service\s+([^\s]+)\s+CPU\s+usage:\s+(\d+\.?\d*)%', line)
                        if cpu_match:
                            service = cpu_match.group(1)
                            cpu = float(cpu_match.group(2))
                            cpu_by_service[service].append(cpu)
                        
                        # Look for memory usage
                        memory_match = re.search(r'Service\s+([^\s]+)\s+memory\s+usage:\s+(\d+\.?\d*)\s*([KMG]B)', line)
                        if memory_match:
                            service = memory_match.group(1)
                            memory = float(memory_match.group(2))
                            unit = memory_match.group(3)
                            
                            # Convert to MB for consistency
                            if unit == 'KB':
                                memory /= 1024
                            elif unit == 'GB':
                                memory *= 1024
                            
                            memory_by_service[service].append(memory)
                        
                        # Look for disk usage
                        disk_match = re.search(r'Service\s+([^\s]+)\s+disk\s+usage:\s+(\d+\.?\d*)%', line)
                        if disk_match:
                            service = disk_match.group(1)
                            disk = float(disk_match.group(2))
                            disk_by_service[service].append(disk)
                        
                        # Look for network usage
                        network_match = re.search(r'Service\s+([^\s]+)\s+network\s+usage:\s+(\d+\.?\d*)\s*([KMG]B/s)', line)
                        if network_match:
                            service = network_match.group(1)
                            network = float(network_match.group(2))
                            unit = network_match.group(3)
                            
                            # Convert to MB/s for consistency
                            if unit == 'KB/s':
                                network /= 1024
                            elif unit == 'GB/s':
                                network *= 1024
                            
                            network_by_service[service].append(network)
            except Exception as e:
                logger.error(f"Error processing resource log {log_file}: {e}")
        
        # Calculate average resource utilization for each service
        for service, values in cpu_by_service.items():
            resource_metrics['cpu'][service] = {
                'average': sum(values) / len(values) if values else None,
                'max': max(values) if values else None,
                'measurements': len(values)
            }
        
        for service, values in memory_by_service.items():
            resource_metrics['memory'][service] = {
                'average_mb': sum(values) / len(values) if values else None,
                'max_mb': max(values) if values else None,
                'measurements': len(values)
            }
        
        for service, values in disk_by_service.items():
            resource_metrics['disk'][service] = {
                'average': sum(values) / len(values) if values else None,
                'max': max(values) if values else None,
                'measurements': len(values)
            }
        
        for service, values in network_by_service.items():
            resource_metrics['network'][service] = {
                'average_mbps': sum(values) / len(values) if values else None,
                'max_mbps': max(values) if values else None,
                'measurements': len(values)
            }
        
        return resource_metrics
    
    def analyze_crash_reports(self) -> Dict[str, Any]:
        """
        Analyze crash reports.
        
        Returns:
            Dictionary with crash report metrics
        """
        logger.info("Analyzing crash reports...")
        
        crash_metrics = {
            'total_crashes': 0,
            'crashes_by_service': {},
            'crashes_by_day': {},
            'crash_trend': [],
            'top_crash_reasons': []
        }
        
        # Check if log directory exists
        if not os.path.exists(self.log_dir):
            logger.warning(f"Log directory {self.log_dir} does not exist")
            return crash_metrics
        
        # Find crash logs
        crash_logs = []
        for pattern in ['crash*.log', 'error*.log', 'fatal*.log']:
            crash_logs.extend(glob.glob(os.path.join(self.log_dir, '**', pattern), recursive=True))
        
        # Extract crash information
        crashes = []
        crashes_by_service = defaultdict(int)
        crashes_by_day = defaultdict(int)
        crash_reasons = defaultdict(int)
        
        for log_file in crash_logs:
            try:
                # Determine service from log file path
                service = 'unknown'
                log_path = Path(log_file)
                if 'services' in log_path.parts:
                    service_idx = log_path.parts.index('services') + 1
                    if service_idx < len(log_path.parts):
                        service = log_path.parts[service_idx]
                elif any(s in log_path.stem for s in ['service', 'api', 'server']):
                    service = log_path.stem
                
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if this is a crash log
                if re.search(r'crash|fatal|killed|terminated|segmentation fault|core dump', content, re.IGNORECASE):
                    # Extract timestamp if available
                    timestamp = None
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', content)
                    if timestamp_match:
                        try:
                            timestamp_str = timestamp_match.group(1).replace(' ', 'T')
                            timestamp = datetime.fromisoformat(timestamp_str)
                        except ValueError:
                            pass
                    
                    # Extract reason
                    reason = 'Unknown'
                    reason_patterns = [
                        (r'segmentation fault', 'Segmentation Fault'),
                        (r'out of memory', 'Out of Memory'),
                        (r'killed', 'Process Killed'),
                        (r'core dump', 'Core Dump'),
                        (r'stack overflow', 'Stack Overflow'),
                        (r'unhandled exception', 'Unhandled Exception')
                    ]
                    
                    for pattern, label in reason_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            reason = label
                            break
                    
                    # Add to crashes list
                    crashes.append({
                        'timestamp': timestamp.isoformat() if timestamp else None,
                        'service': service,
                        'reason': reason
                    })
                    
                    # Update counts
                    crashes_by_service[service] += 1
                    crash_reasons[reason] += 1
                    
                    if timestamp:
                        day = timestamp.date().isoformat()
                        crashes_by_day[day] += 1
            except Exception as e:
                logger.error(f"Error processing crash log {log_file}: {e}")
        
        # Calculate crash trend (last 30 days)
        today = datetime.now().date()
        trend_days = []
        for i in range(30):
            day = (today - timedelta(days=i)).isoformat()
            trend_days.append(day)
        
        trend_days.reverse()  # Oldest to newest
        
        crash_trend = []
        for day in trend_days:
            crash_trend.append({
                'date': day,
                'count': crashes_by_day.get(day, 0)
            })
        
        # Find top crash reasons
        top_crash_reasons = []
        for reason, count in sorted(crash_reasons.items(), key=lambda x: x[1], reverse=True):
            top_crash_reasons.append({
                'reason': reason,
                'count': count
            })
        
        # Update metrics
        crash_metrics['total_crashes'] = len(crashes)
        crash_metrics['crashes_by_service'] = dict(crashes_by_service)
        crash_metrics['crashes_by_day'] = dict(crashes_by_day)
        crash_metrics['crash_trend'] = crash_trend
        crash_metrics['top_crash_reasons'] = top_crash_reasons
        
        return crash_metrics
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze runtime health.
        
        Returns:
            Dictionary with runtime health metrics
        """
        # Analyze error rates
        error_metrics = self.analyze_error_rates()
        
        # Analyze service uptime
        uptime_metrics = self.analyze_service_uptime()
        
        # Analyze resource utilization
        resource_metrics = self.analyze_resource_utilization()
        
        # Analyze crash reports
        crash_metrics = self.analyze_crash_reports()
        
        # Generate summary
        summary = {
            'total_errors': error_metrics['total_errors'],
            'overall_uptime': uptime_metrics['overall_uptime'],
            'total_crashes': crash_metrics['total_crashes'],
            'services_count': len(uptime_metrics['services']),
            'high_cpu_services': sum(1 for service, metrics in resource_metrics['cpu'].items() if metrics['average'] and metrics['average'] > 80),
            'high_memory_services': sum(1 for service, metrics in resource_metrics['memory'].items() if metrics['average_mb'] and metrics['average_mb'] > 1024)  # >1GB
        }
        
        # Combine results
        return {
            'error_rates': error_metrics,
            'service_uptime': uptime_metrics,
            'resource_utilization': resource_metrics,
            'crash_reports': crash_metrics,
            'summary': summary
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze runtime health")
    parser.add_argument(
        "--project-root", 
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--output-dir", 
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--log-dir", 
        default=DEFAULT_LOG_DIR,
        help="Directory containing log files"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze runtime health
    analyzer = RuntimeHealthAnalyzer(Path(args.project_root), Path(args.log_dir))
    results = analyzer.analyze()
    
    # Save results
    output_path = os.path.join(args.output_dir, "runtime_health.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Runtime health metrics saved to {output_path}")

if __name__ == "__main__":
    main()
