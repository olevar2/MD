#!/usr/bin/env python
"""
Reconciliation Monitor.

This script monitors reconciliation tasks and results and sends alerts when issues are detected.
It is designed to be run as a cron job or a scheduled task.
"""

import asyncio
import datetime
import logging
import json
import time
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("reconciliation_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API URL
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Email configuration
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.example.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "user@example.com")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "password")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "reconciliation@example.com")
EMAIL_TO = os.environ.get("EMAIL_TO", "alerts@example.com").split(",")

# Alert thresholds
ALERT_THRESHOLD_ERROR = int(os.environ.get("ALERT_THRESHOLD_ERROR", "10"))
ALERT_THRESHOLD_WARNING = int(os.environ.get("ALERT_THRESHOLD_WARNING", "100"))
ALERT_THRESHOLD_PERCENTAGE = float(os.environ.get("ALERT_THRESHOLD_PERCENTAGE", "95"))


async def get_results(
    config_id: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Get reconciliation results.
    
    Args:
        config_id: Filter by config ID
        status: Filter by status
        start_date: Filter by start time (start)
        end_date: Filter by start time (end)
        limit: Maximum number of records to return
        offset: Offset for pagination
        
    Returns:
        List of reconciliation results
    """
    params = {
        "limit": limit,
        "offset": offset
    }
    
    if config_id:
        params["config_id"] = config_id
    
    if status:
        params["status"] = status
    
    if start_date:
        params["start_date"] = start_date.isoformat()
    
    if end_date:
        params["end_date"] = end_date.isoformat()
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/reconciliation/results", params=params)
        response.raise_for_status()
        
        return response.json()


async def get_configs(enabled: bool = True) -> List[Dict[str, Any]]:
    """
    Get reconciliation configurations.
    
    Args:
        enabled: Filter by enabled status
        
    Returns:
        List of reconciliation configurations
    """
    params = {
        "enabled": enabled
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/reconciliation/configs", params=params)
        response.raise_for_status()
        
        return response.json()


def send_email(subject: str, body: str, recipients: List[str]) -> None:
    """
    Send an email.
    
    Args:
        subject: Email subject
        body: Email body
        recipients: List of recipients
    """
    # Create message
    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    
    # Add body
    msg.attach(MIMEText(body, "html"))
    
    try:
        # Connect to SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        # Send email
        server.sendmail(EMAIL_FROM, recipients, msg.as_string())
        
        # Disconnect
        server.quit()
        
        logger.info(f"Sent email to {recipients}: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


def generate_alert_email(
    config: Dict[str, Any],
    result: Dict[str, Any],
    alert_type: str
) -> str:
    """
    Generate an alert email.
    
    Args:
        config: Reconciliation configuration
        result: Reconciliation result
        alert_type: Alert type
        
    Returns:
        Email body
    """
    # Get result details
    result_id = result.get("result_id", "")
    status = result.get("status", "")
    start_time = result.get("start_time", "")
    end_time = result.get("end_time", "")
    total_records = result.get("total_records", 0)
    matched_records = result.get("matched_records", 0)
    match_percentage = matched_records / total_records * 100 if total_records > 0 else 0
    issues = result.get("issues", [])
    
    # Count issues by severity
    error_issues = [issue for issue in issues if issue.get("severity") == "ERROR"]
    warning_issues = [issue for issue in issues if issue.get("severity") == "WARNING"]
    info_issues = [issue for issue in issues if issue.get("severity") == "INFO"]
    
    # Generate email body
    body = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .error {{
                color: red;
            }}
            .warning {{
                color: orange;
            }}
            .info {{
                color: blue;
            }}
        </style>
    </head>
    <body>
        <h1>Reconciliation Alert: {alert_type}</h1>
        <p>A reconciliation alert has been triggered for the following configuration:</p>
        
        <h2>Configuration</h2>
        <table>
            <tr>
                <th>Name</th>
                <td>{config.get("name", "")}</td>
            </tr>
            <tr>
                <th>ID</th>
                <td>{config.get("config_id", "")}</td>
            </tr>
            <tr>
                <th>Type</th>
                <td>{config.get("reconciliation_type", "")}</td>
            </tr>
            <tr>
                <th>Description</th>
                <td>{config.get("description", "")}</td>
            </tr>
        </table>
        
        <h2>Result</h2>
        <table>
            <tr>
                <th>ID</th>
                <td>{result_id}</td>
            </tr>
            <tr>
                <th>Status</th>
                <td>{status}</td>
            </tr>
            <tr>
                <th>Start Time</th>
                <td>{start_time}</td>
            </tr>
            <tr>
                <th>End Time</th>
                <td>{end_time}</td>
            </tr>
            <tr>
                <th>Total Records</th>
                <td>{total_records}</td>
            </tr>
            <tr>
                <th>Matched Records</th>
                <td>{matched_records}</td>
            </tr>
            <tr>
                <th>Match Percentage</th>
                <td>{match_percentage:.2f}%</td>
            </tr>
            <tr>
                <th>Total Issues</th>
                <td>{len(issues)}</td>
            </tr>
            <tr>
                <th>Error Issues</th>
                <td class="error">{len(error_issues)}</td>
            </tr>
            <tr>
                <th>Warning Issues</th>
                <td class="warning">{len(warning_issues)}</td>
            </tr>
            <tr>
                <th>Info Issues</th>
                <td class="info">{len(info_issues)}</td>
            </tr>
        </table>
    """
    
    # Add issues
    if issues:
        body += """
        <h2>Issues</h2>
        <table>
            <tr>
                <th>Field</th>
                <th>Severity</th>
                <th>Description</th>
            </tr>
        """
        
        for issue in issues[:20]:  # Limit to 20 issues
            severity = issue.get("severity", "")
            severity_class = "error" if severity == "ERROR" else "warning" if severity == "WARNING" else "info"
            
            body += f"""
            <tr>
                <td>{issue.get("field", "")}</td>
                <td class="{severity_class}">{severity}</td>
                <td>{issue.get("description", "")}</td>
            </tr>
            """
        
        if len(issues) > 20:
            body += f"""
            <tr>
                <td colspan="3">... and {len(issues) - 20} more issues</td>
            </tr>
            """
        
        body += "</table>"
    
    body += """
    </body>
    </html>
    """
    
    return body


async def check_results() -> None:
    """Check reconciliation results and send alerts."""
    # Get enabled configurations
    logger.info("Getting enabled configurations")
    configs = await get_configs(enabled=True)
    
    logger.info(f"Found {len(configs)} enabled configurations")
    
    # Current time
    now = datetime.datetime.utcnow()
    
    # Get results from the last 24 hours
    start_date = now - datetime.timedelta(hours=24)
    
    logger.info(f"Getting results from {start_date} to {now}")
    results = await get_results(
        start_date=start_date,
        end_date=now,
        status="completed"
    )
    
    logger.info(f"Found {len(results)} results")
    
    # Group results by configuration
    results_by_config = {}
    for result in results:
        config_id = result.get("config_id")
        if config_id not in results_by_config:
            results_by_config[config_id] = []
        
        results_by_config[config_id].append(result)
    
    # Check each configuration
    for config in configs:
        config_id = config.get("config_id")
        name = config.get("name")
        
        # Get results for this configuration
        config_results = results_by_config.get(config_id, [])
        
        if not config_results:
            logger.info(f"No results found for configuration {name} ({config_id})")
            continue
        
        logger.info(f"Checking {len(config_results)} results for configuration {name} ({config_id})")
        
        # Check each result
        for result in config_results:
            # Get result details
            result_id = result.get("result_id")
            total_records = result.get("total_records", 0)
            matched_records = result.get("matched_records", 0)
            match_percentage = matched_records / total_records * 100 if total_records > 0 else 0
            issues = result.get("issues", [])
            
            # Count issues by severity
            error_issues = [issue for issue in issues if issue.get("severity") == "ERROR"]
            warning_issues = [issue for issue in issues if issue.get("severity") == "WARNING"]
            
            # Check for alerts
            alerts = []
            
            # Check for error threshold
            if len(error_issues) >= ALERT_THRESHOLD_ERROR:
                alerts.append(f"Error threshold exceeded: {len(error_issues)} errors (threshold: {ALERT_THRESHOLD_ERROR})")
            
            # Check for warning threshold
            if len(warning_issues) >= ALERT_THRESHOLD_WARNING:
                alerts.append(f"Warning threshold exceeded: {len(warning_issues)} warnings (threshold: {ALERT_THRESHOLD_WARNING})")
            
            # Check for match percentage threshold
            if match_percentage < ALERT_THRESHOLD_PERCENTAGE:
                alerts.append(f"Match percentage below threshold: {match_percentage:.2f}% (threshold: {ALERT_THRESHOLD_PERCENTAGE}%)")
            
            # Send alerts
            if alerts:
                logger.info(f"Sending alerts for result {result_id}: {', '.join(alerts)}")
                
                # Generate email subject
                subject = f"Reconciliation Alert: {name} - {', '.join(alerts)}"
                
                # Generate email body
                body = generate_alert_email(config, result, ", ".join(alerts))
                
                # Send email
                send_email(subject, body, EMAIL_TO)


async def main() -> None:
    """Main entry point."""
    logger.info("Starting reconciliation monitor")
    
    while True:
        try:
            # Check results
            await check_results()
            
            # Wait for next hour
            now = datetime.datetime.utcnow()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
            sleep_seconds = (next_hour - now).total_seconds()
            
            logger.info(f"Waiting {sleep_seconds:.2f} seconds until next hour")
            await asyncio.sleep(sleep_seconds)
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(3600)  # Wait an hour before retrying


if __name__ == "__main__":
    asyncio.run(main())
