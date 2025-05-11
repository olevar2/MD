#!/usr/bin/env python3
"""
Deployment Status Analyzer

This script analyzes deployment status of the forex trading platform:
1. Current deployed version
2. Deployment frequency
3. Rollback frequency
4. Deployment pipeline health

Output is a JSON file with comprehensive deployment status metrics.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output"
DEFAULT_DEPLOYMENT_LOG_DIR = r"D:\MD\forex_trading_platform\logs\deployments"

class DeploymentStatusAnalyzer:
    """Analyzes deployment status of the forex trading platform."""
    
    def __init__(self, project_root: Path, deployment_log_dir: Path):
        """
        Initialize the analyzer.
        
        Args:
            project_root: Root directory of the project
            deployment_log_dir: Directory containing deployment logs
        """
        self.project_root = project_root
        self.deployment_log_dir = deployment_log_dir
        
    def get_current_version(self) -> Dict[str, Any]:
        """
        Get the current deployed version.
        
        Returns:
            Dictionary with current version information
        """
        logger.info("Getting current version...")
        
        version_info = {
            'version': None,
            'timestamp': None,
            'environment': None,
            'services': {}
        }
        
        # Look for version files
        version_files = [
            os.path.join(self.project_root, 'VERSION'),
            os.path.join(self.project_root, 'version.txt'),
            os.path.join(self.project_root, '.version')
        ]
        
        for version_file in version_files:
            if os.path.exists(version_file):
                try:
                    with open(version_file, 'r', encoding='utf-8') as f:
                        version_info['version'] = f.read().strip()
                    break
                except Exception as e:
                    logger.error(f"Error reading version file {version_file}: {e}")
        
        # Look for service-specific version files
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.lower() in ('version', 'version.txt', '.version'):
                    service_dir = os.path.basename(root)
                    if service_dir.endswith('-service') or service_dir.endswith('_service'):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                version_info['services'][service_dir] = f.read().strip()
                        except Exception as e:
                            logger.error(f"Error reading service version file {os.path.join(root, file)}: {e}")
        
        # Look for environment information
        env_files = [
            os.path.join(self.project_root, '.env'),
            os.path.join(self.project_root, '.env.local'),
            os.path.join(self.project_root, 'environment.json')
        ]
        
        for env_file in env_files:
            if os.path.exists(env_file):
                try:
                    if env_file.endswith('.json'):
                        with open(env_file, 'r', encoding='utf-8') as f:
                            env_data = json.load(f)
                            version_info['environment'] = env_data.get('environment') or env_data.get('ENV') or env_data.get('NODE_ENV')
                    else:
                        with open(env_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.startswith('ENVIRONMENT=') or line.startswith('ENV=') or line.startswith('NODE_ENV='):
                                    version_info['environment'] = line.split('=', 1)[1].strip().strip('"\'')
                                    break
                    if version_info['environment']:
                        break
                except Exception as e:
                    logger.error(f"Error reading environment file {env_file}: {e}")
        
        # Look for timestamp information
        if os.path.exists(self.deployment_log_dir):
            try:
                # Find the most recent deployment log
                deployment_logs = glob.glob(os.path.join(self.deployment_log_dir, '*.log'))
                if deployment_logs:
                    latest_log = max(deployment_logs, key=os.path.getmtime)
                    version_info['timestamp'] = datetime.fromtimestamp(os.path.getmtime(latest_log)).isoformat()
            except Exception as e:
                logger.error(f"Error getting deployment timestamp: {e}")
        
        return version_info
    
    def analyze_deployment_frequency(self) -> Dict[str, Any]:
        """
        Analyze deployment frequency.
        
        Returns:
            Dictionary with deployment frequency metrics
        """
        logger.info("Analyzing deployment frequency...")
        
        frequency_metrics = {
            'total_deployments': 0,
            'deployments_last_7_days': 0,
            'deployments_last_30_days': 0,
            'deployments_last_90_days': 0,
            'average_deployments_per_week': 0,
            'average_deployments_per_month': 0,
            'deployment_dates': [],
            'deployment_by_day_of_week': {
                'Monday': 0,
                'Tuesday': 0,
                'Wednesday': 0,
                'Thursday': 0,
                'Friday': 0,
                'Saturday': 0,
                'Sunday': 0
            },
            'deployment_by_hour': {str(i): 0 for i in range(24)}
        }
        
        # Check if deployment log directory exists
        if not os.path.exists(self.deployment_log_dir):
            logger.warning(f"Deployment log directory {self.deployment_log_dir} does not exist")
            return frequency_metrics
        
        # Find deployment logs
        deployment_logs = glob.glob(os.path.join(self.deployment_log_dir, '*.log'))
        
        # Extract deployment dates
        deployment_dates = []
        for log_file in deployment_logs:
            try:
                # Get deployment date from file modification time
                mtime = os.path.getmtime(log_file)
                deployment_date = datetime.fromtimestamp(mtime)
                deployment_dates.append(deployment_date)
                
                # Update day of week and hour counts
                day_of_week = deployment_date.strftime('%A')
                hour = str(deployment_date.hour)
                
                frequency_metrics['deployment_by_day_of_week'][day_of_week] += 1
                frequency_metrics['deployment_by_hour'][hour] += 1
            except Exception as e:
                logger.error(f"Error processing deployment log {log_file}: {e}")
        
        # Sort deployment dates
        deployment_dates.sort()
        
        # Update metrics
        frequency_metrics['total_deployments'] = len(deployment_dates)
        frequency_metrics['deployment_dates'] = [d.isoformat() for d in deployment_dates]
        
        # Calculate recent deployment counts
        now = datetime.now()
        frequency_metrics['deployments_last_7_days'] = sum(1 for d in deployment_dates if (now - d).days <= 7)
        frequency_metrics['deployments_last_30_days'] = sum(1 for d in deployment_dates if (now - d).days <= 30)
        frequency_metrics['deployments_last_90_days'] = sum(1 for d in deployment_dates if (now - d).days <= 90)
        
        # Calculate averages
        if deployment_dates:
            # Calculate weeks and months between first and last deployment
            first_deployment = deployment_dates[0]
            last_deployment = deployment_dates[-1]
            days_between = (last_deployment - first_deployment).days
            
            if days_between > 0:
                weeks_between = days_between / 7
                months_between = days_between / 30
                
                if weeks_between > 0:
                    frequency_metrics['average_deployments_per_week'] = round(len(deployment_dates) / weeks_between, 2)
                
                if months_between > 0:
                    frequency_metrics['average_deployments_per_month'] = round(len(deployment_dates) / months_between, 2)
        
        return frequency_metrics
    
    def analyze_rollback_frequency(self) -> Dict[str, Any]:
        """
        Analyze rollback frequency.
        
        Returns:
            Dictionary with rollback frequency metrics
        """
        logger.info("Analyzing rollback frequency...")
        
        rollback_metrics = {
            'total_rollbacks': 0,
            'rollbacks_last_30_days': 0,
            'rollback_percentage': 0,
            'rollback_dates': [],
            'rollback_reasons': {}
        }
        
        # Check if deployment log directory exists
        if not os.path.exists(self.deployment_log_dir):
            logger.warning(f"Deployment log directory {self.deployment_log_dir} does not exist")
            return rollback_metrics
        
        # Find deployment logs
        deployment_logs = glob.glob(os.path.join(self.deployment_log_dir, '*.log'))
        
        # Extract rollback information
        rollback_dates = []
        rollback_reasons = {}
        
        for log_file in deployment_logs:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if this is a rollback
                is_rollback = False
                reason = None
                
                if 'rollback' in content.lower():
                    is_rollback = True
                    
                    # Try to extract reason
                    reason_match = re.search(r'reason:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
                    if reason_match:
                        reason = reason_match.group(1).strip()
                    else:
                        reason = "Unknown"
                
                if is_rollback:
                    # Get rollback date from file modification time
                    mtime = os.path.getmtime(log_file)
                    rollback_date = datetime.fromtimestamp(mtime)
                    rollback_dates.append(rollback_date)
                    
                    # Update reason counts
                    if reason not in rollback_reasons:
                        rollback_reasons[reason] = 0
                    rollback_reasons[reason] += 1
            except Exception as e:
                logger.error(f"Error processing deployment log {log_file}: {e}")
        
        # Sort rollback dates
        rollback_dates.sort()
        
        # Update metrics
        rollback_metrics['total_rollbacks'] = len(rollback_dates)
        rollback_metrics['rollback_dates'] = [d.isoformat() for d in rollback_dates]
        rollback_metrics['rollback_reasons'] = rollback_reasons
        
        # Calculate recent rollback counts
        now = datetime.now()
        rollback_metrics['rollbacks_last_30_days'] = sum(1 for d in rollback_dates if (now - d).days <= 30)
        
        # Calculate rollback percentage
        if deployment_logs:
            rollback_metrics['rollback_percentage'] = round((len(rollback_dates) / len(deployment_logs)) * 100, 2)
        
        return rollback_metrics
    
    def analyze_pipeline_health(self) -> Dict[str, Any]:
        """
        Analyze deployment pipeline health.
        
        Returns:
            Dictionary with pipeline health metrics
        """
        logger.info("Analyzing pipeline health...")
        
        pipeline_metrics = {
            'pipeline_exists': False,
            'pipeline_type': None,
            'success_rate': None,
            'average_duration': None,
            'last_run_status': None,
            'last_run_timestamp': None
        }
        
        # Check for CI/CD configuration files
        ci_files = [
            ('.github/workflows', 'GitHub Actions'),
            ('.gitlab-ci.yml', 'GitLab CI'),
            ('Jenkinsfile', 'Jenkins'),
            ('azure-pipelines.yml', 'Azure Pipelines'),
            ('bitbucket-pipelines.yml', 'Bitbucket Pipelines'),
            ('buildspec.yml', 'AWS CodeBuild'),
            ('cloudbuild.yaml', 'Google Cloud Build')
        ]
        
        for file_path, pipeline_type in ci_files:
            full_path = os.path.join(self.project_root, file_path)
            if os.path.exists(full_path):
                pipeline_metrics['pipeline_exists'] = True
                pipeline_metrics['pipeline_type'] = pipeline_type
                break
        
        # Look for pipeline logs
        pipeline_logs = []
        pipeline_log_dirs = [
            os.path.join(self.project_root, 'logs', 'ci'),
            os.path.join(self.project_root, 'logs', 'pipeline'),
            os.path.join(self.project_root, 'logs', 'builds'),
            self.deployment_log_dir
        ]
        
        for log_dir in pipeline_log_dirs:
            if os.path.exists(log_dir):
                pipeline_logs.extend(glob.glob(os.path.join(log_dir, '*.log')))
        
        # Analyze pipeline logs
        if pipeline_logs:
            success_count = 0
            durations = []
            last_run = None
            last_status = None
            
            for log_file in pipeline_logs:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if successful
                    is_success = 'success' in content.lower() and not ('fail' in content.lower() or 'error' in content.lower())
                    if is_success:
                        success_count += 1
                    
                    # Extract duration if available
                    duration_match = re.search(r'duration:\s*(\d+)', content, re.IGNORECASE)
                    if duration_match:
                        durations.append(int(duration_match.group(1)))
                    
                    # Check if this is the most recent log
                    mtime = os.path.getmtime(log_file)
                    log_date = datetime.fromtimestamp(mtime)
                    
                    if last_run is None or log_date > last_run:
                        last_run = log_date
                        last_status = 'success' if is_success else 'failure'
                except Exception as e:
                    logger.error(f"Error processing pipeline log {log_file}: {e}")
            
            # Update metrics
            if pipeline_logs:
                pipeline_metrics['success_rate'] = round((success_count / len(pipeline_logs)) * 100, 2)
            
            if durations:
                pipeline_metrics['average_duration'] = sum(durations) / len(durations)
            
            if last_run:
                pipeline_metrics['last_run_timestamp'] = last_run.isoformat()
                pipeline_metrics['last_run_status'] = last_status
        
        return pipeline_metrics
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze deployment status.
        
        Returns:
            Dictionary with deployment status metrics
        """
        # Get current version
        version_info = self.get_current_version()
        
        # Analyze deployment frequency
        frequency_metrics = self.analyze_deployment_frequency()
        
        # Analyze rollback frequency
        rollback_metrics = self.analyze_rollback_frequency()
        
        # Analyze pipeline health
        pipeline_metrics = self.analyze_pipeline_health()
        
        # Generate summary
        summary = {
            'current_version': version_info['version'],
            'environment': version_info['environment'],
            'last_deployment': version_info['timestamp'],
            'total_deployments': frequency_metrics['total_deployments'],
            'deployments_last_30_days': frequency_metrics['deployments_last_30_days'],
            'average_deployments_per_week': frequency_metrics['average_deployments_per_week'],
            'total_rollbacks': rollback_metrics['total_rollbacks'],
            'rollback_percentage': rollback_metrics['rollback_percentage'],
            'pipeline_exists': pipeline_metrics['pipeline_exists'],
            'pipeline_type': pipeline_metrics['pipeline_type'],
            'pipeline_success_rate': pipeline_metrics['success_rate']
        }
        
        # Combine results
        return {
            'version': version_info,
            'deployment_frequency': frequency_metrics,
            'rollback_frequency': rollback_metrics,
            'pipeline_health': pipeline_metrics,
            'summary': summary
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze deployment status")
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
        "--deployment-log-dir", 
        default=DEFAULT_DEPLOYMENT_LOG_DIR,
        help="Directory containing deployment logs"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze deployment status
    analyzer = DeploymentStatusAnalyzer(Path(args.project_root), Path(args.deployment_log_dir))
    results = analyzer.analyze()
    
    # Save results
    output_path = os.path.join(args.output_dir, "deployment_status.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Deployment status metrics saved to {output_path}")

if __name__ == "__main__":
    main()
