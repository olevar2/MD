#!/usr/bin/env python
"""
Scheduled Deprecation Report Generator

This script generates deprecation reports on a schedule and sends them to the team.
It can be run as a cron job or scheduled task.

Usage:
    python scheduled_deprecation_report.py [--email] [--slack] [--jira]

Options:
    --email       Send report via email
    --slack       Send report to Slack
    --jira        Create or update Jira ticket with report
"""

import os
import sys
import argparse
import datetime
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import deprecation monitor and report generator
from analysis_engine.core.deprecation_monitor import get_usage_report
from tools.deprecation_report import format_html, format_text


def generate_report() -> Dict:
    """Generate the deprecation report."""
    return get_usage_report()


def send_email(report: Dict, recipients: List[str]) -> bool:
    """Send report via email."""
    # Email settings
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.example.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_username = os.environ.get("SMTP_USERNAME", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("EMAIL_SENDER", "noreply@example.com")
    
    # Create message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Configuration Migration Progress Report - {datetime.date.today().isoformat()}"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    
    # Add text and HTML parts
    text_part = MIMEText(format_text(report), "plain")
    html_part = MIMEText(format_html(report), "html")
    msg.attach(text_part)
    msg.attach(html_part)
    
    try:
        # Connect to server and send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            if smtp_username and smtp_password:
                server.login(smtp_username, smtp_password)
            server.sendmail(sender, recipients, msg.as_string())
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


def send_slack(report: Dict, webhook_url: str) -> bool:
    """Send report to Slack."""
    # Generate summary text
    summary = []
    summary.append(f"*Configuration Migration Progress Report - {datetime.date.today().isoformat()}*")
    summary.append(f"Total usages of deprecated modules: {report['total_usages']}")
    
    # Add module summary
    for module_name, module_data in report.get("modules", {}).items():
        summary.append(f"*{module_name}*: {module_data['total_usages']} usages in {module_data['unique_locations']} locations")
    
    # Add link to full report
    summary.append("\n<https://confluence.example.com/display/DEV/Configuration+Migration+Progress|View full report>")
    
    # Create message payload
    payload = {
        "text": "\n".join(summary),
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "\n".join(summary)
                }
            }
        ]
    }
    
    try:
        # Send to Slack
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending to Slack: {e}")
        return False


def update_jira(report: Dict, jira_config: Dict) -> bool:
    """Create or update Jira ticket with report."""
    # Jira settings
    jira_url = jira_config.get("url", "https://jira.example.com")
    jira_token = jira_config.get("token", "")
    jira_project = jira_config.get("project", "PLATFORM")
    jira_issue_type = jira_config.get("issue_type", "Task")
    jira_ticket = jira_config.get("ticket", "")
    
    # Generate summary text
    summary = []
    summary.append(f"h2. Configuration Migration Progress Report - {datetime.date.today().isoformat()}")
    summary.append(f"Total usages of deprecated modules: {report['total_usages']}")
    
    # Add module summary
    for module_name, module_data in report.get("modules", {}).items():
        summary.append(f"h3. {module_name}")
        summary.append(f"* Total usages: {module_data['total_usages']}")
        summary.append(f"* Unique locations: {module_data['unique_locations']}")
        
        # Add top 5 usage locations
        usages = sorted(module_data["usages"], key=lambda u: u["count"], reverse=True)[:5]
        if usages:
            summary.append("h4. Top usage locations:")
            for usage in usages:
                summary.append(f"* {usage['caller_file']}:{usage['caller_line']} ({usage['count']} usages)")
    
    # Add link to full report
    summary.append("\n[View full report|https://confluence.example.com/display/DEV/Configuration+Migration+Progress]")
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jira_token}"
        }
        
        if jira_ticket:
            # Update existing ticket
            update_url = f"{jira_url}/rest/api/2/issue/{jira_ticket}"
            payload = {
                "fields": {
                    "description": "\n".join(summary)
                }
            }
            response = requests.put(update_url, headers=headers, json=payload)
        else:
            # Create new ticket
            create_url = f"{jira_url}/rest/api/2/issue"
            payload = {
                "fields": {
                    "project": {
                        "key": jira_project
                    },
                    "summary": f"Configuration Migration Progress - {datetime.date.today().isoformat()}",
                    "description": "\n".join(summary),
                    "issuetype": {
                        "name": jira_issue_type
                    }
                }
            }
            response = requests.post(create_url, headers=headers, json=payload)
        
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error updating Jira: {e}")
        return False


def save_report_to_file(report: Dict) -> str:
    """Save report to file and return the file path."""
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "reports"
    )
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate file path
    date_str = datetime.date.today().isoformat()
    file_path = os.path.join(reports_dir, f"deprecation_report_{date_str}.html")
    
    # Write report to file
    with open(file_path, "w") as f:
        f.write(format_html(report))
    
    return file_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate scheduled deprecation report")
    parser.add_argument("--email", action="store_true", help="Send report via email")
    parser.add_argument("--slack", action="store_true", help="Send report to Slack")
    parser.add_argument("--jira", action="store_true", help="Create or update Jira ticket with report")
    args = parser.parse_args()
    
    # Generate report
    report = generate_report()
    
    # Save report to file
    file_path = save_report_to_file(report)
    print(f"Report saved to {file_path}")
    
    # Send via email if requested
    if args.email:
        recipients = os.environ.get("EMAIL_RECIPIENTS", "").split(",")
        if recipients and recipients[0]:
            if send_email(report, recipients):
                print(f"Report sent via email to {', '.join(recipients)}")
            else:
                print("Failed to send report via email")
        else:
            print("No email recipients specified")
    
    # Send to Slack if requested
    if args.slack:
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
        if webhook_url:
            if send_slack(report, webhook_url):
                print("Report sent to Slack")
            else:
                print("Failed to send report to Slack")
        else:
            print("No Slack webhook URL specified")
    
    # Update Jira if requested
    if args.jira:
        jira_config = {
            "url": os.environ.get("JIRA_URL", ""),
            "token": os.environ.get("JIRA_TOKEN", ""),
            "project": os.environ.get("JIRA_PROJECT", ""),
            "issue_type": os.environ.get("JIRA_ISSUE_TYPE", "Task"),
            "ticket": os.environ.get("JIRA_TICKET", "")
        }
        if jira_config["url"] and jira_config["token"] and jira_config["project"]:
            if update_jira(report, jira_config):
                print("Jira ticket updated")
            else:
                print("Failed to update Jira ticket")
        else:
            print("Incomplete Jira configuration")


if __name__ == "__main__":
    main()
