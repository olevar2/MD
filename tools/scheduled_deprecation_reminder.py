#!/usr/bin/env python
"""
Scheduled Deprecation Reminder

This script sends reminders about deprecated modules that need to be migrated.
It can be scheduled to run periodically (e.g., weekly) to keep the team informed
about migration progress and upcoming deadlines.

Usage:
    python scheduled_deprecation_reminder.py [--email] [--slack] [--jira]

Options:
    --email    Send reminders via email
    --slack    Post reminders to Slack
    --jira     Create or update Jira tickets for migration tasks
"""

import os
import sys
import json
import argparse
import datetime
from typing import Dict, List, Any, Optional
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import the deprecation monitor
try:
    from analysis_engine.core.deprecation_monitor import get_usage_report
    CAN_GET_LIVE_REPORT = True
except ImportError:
    CAN_GET_LIVE_REPORT = False


def load_report() -> Dict[str, Any]:
    """
    Load the deprecation report.
    
    Returns:
        Dict containing the report data
    """
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "logs",
        "deprecation_report.json"
    )
    
    if not os.path.exists(report_path):
        if CAN_GET_LIVE_REPORT:
            print(f"Report file not found at {report_path}, generating live report")
            return get_usage_report()
        else:
            print(f"Report file not found at {report_path} and cannot generate live report")
            return {
                "generated_at": datetime.datetime.now().isoformat(),
                "total_usages": 0,
                "modules": {}
            }
    
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading report: {e}")
        return {
            "generated_at": datetime.datetime.now().isoformat(),
            "total_usages": 0,
            "modules": {}
        }


def send_email_reminder(report: Dict[str, Any]) -> None:
    """
    Send a reminder email about deprecated modules.
    
    Args:
        report: The deprecation report
    """
    # Email configuration
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.example.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    sender_email = os.environ.get("SENDER_EMAIL", "noreply@example.com")
    recipient_emails = os.environ.get("RECIPIENT_EMAILS", "").split(",")
    
    if not recipient_emails or recipient_emails[0] == "":
        print("No recipient emails configured, skipping email reminder")
        return
    
    # Calculate days until removal
    removal_date = datetime.date(2023, 12, 31)
    days_until_removal = (removal_date - datetime.date.today()).days
    days_message = f"{days_until_removal} days" if days_until_removal > 0 else "PAST DUE"
    
    # Determine urgency level
    urgency = "Low"
    if days_until_removal <= 30:
        urgency = "Critical"
    elif days_until_removal <= 60:
        urgency = "High"
    elif days_until_removal <= 90:
        urgency = "Medium"
    
    # Create message
    msg = MIMEMultipart()
    msg["Subject"] = f"[{urgency}] Deprecated Module Migration Reminder - {days_message} until removal"
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipient_emails)
    
    # Email body
    body = f"""
    <html>
    <body>
        <h1>Deprecated Module Migration Reminder</h1>
        <p>This is a reminder about deprecated modules that need to be migrated before the removal date.</p>
        
        <h2>Timeline</h2>
        <p><strong>Removal Date:</strong> December 31, 2023</p>
        <p><strong>Days Remaining:</strong> {days_message}</p>
        <p><strong>Urgency Level:</strong> {urgency}</p>
        
        <h2>Modules to Migrate</h2>
        <ul>
    """
    
    for module, data in report.get("modules", {}).items():
        body += f"""
            <li>
                <strong>{module}</strong>
                <ul>
                    <li>Unique Locations: {data.get('unique_locations', 0)}</li>
                    <li>Total Usages: {data.get('total_usages', 0)}</li>
                </ul>
            </li>
        """
    
    body += f"""
        </ul>
        
        <h2>Migration Resources</h2>
        <ul>
            <li><a href="https://confluence.example.com/display/DEV/Configuration+Migration+Guide">Configuration Migration Guide</a></li>
            <li><a href="https://confluence.example.com/display/DEV/API+Router+Migration+Guide">API Router Migration Guide</a></li>
            <li>Migration Tools:
                <ul>
                    <li><code>python tools/migrate_config_imports.py</code> - Migrate configuration imports</li>
                    <li><code>python tools/migrate_router_imports.py</code> - Migrate router imports</li>
                </ul>
            </li>
        </ul>
        
        <h2>Need Help?</h2>
        <p>If you need assistance with migration, please contact the Analysis Engine team or open a support ticket.</p>
    </body>
    </html>
    """
    
    msg.attach(MIMEText(body, "html"))
    
    # Send email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        if smtp_user and smtp_password:
            server.login(smtp_user, smtp_password)
        server.sendmail(sender_email, recipient_emails, msg.as_string())
        server.quit()
        print(f"Email reminder sent to {', '.join(recipient_emails)}")
    except Exception as e:
        print(f"Error sending email: {e}")


def post_slack_reminder(report: Dict[str, Any]) -> None:
    """
    Post a reminder to Slack about deprecated modules.
    
    Args:
        report: The deprecation report
    """
    # Slack configuration
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    
    if not webhook_url:
        print("No Slack webhook URL configured, skipping Slack reminder")
        return
    
    # Calculate days until removal
    removal_date = datetime.date(2023, 12, 31)
    days_until_removal = (removal_date - datetime.date.today()).days
    days_message = f"{days_until_removal} days" if days_until_removal > 0 else "PAST DUE"
    
    # Determine urgency level and emoji
    urgency = "Low"
    emoji = ":information_source:"
    if days_until_removal <= 30:
        urgency = "Critical"
        emoji = ":rotating_light:"
    elif days_until_removal <= 60:
        urgency = "High"
        emoji = ":warning:"
    elif days_until_removal <= 90:
        urgency = "Medium"
        emoji = ":warning:"
    
    # Create message
    message = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Deprecated Module Migration Reminder"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"This is a reminder about deprecated modules that need to be migrated before the removal date."
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Removal Date:*\nDecember 31, 2023"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Days Remaining:*\n{days_message}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Urgency Level:*\n{urgency}"
                    }
                ]
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Modules to Migrate:*"
                }
            }
        ]
    }
    
    # Add module details
    for module, data in report.get("modules", {}).items():
        message["blocks"].append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{module}*\nLocations: {data.get('unique_locations', 0)} | Usages: {data.get('total_usages', 0)}"
            }
        })
    
    # Add migration resources
    message["blocks"].append({
        "type": "divider"
    })
    message["blocks"].append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*Migration Resources:*"
        }
    })
    message["blocks"].append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "• <https://confluence.example.com/display/DEV/Configuration+Migration+Guide|Configuration Migration Guide>\n• <https://confluence.example.com/display/DEV/API+Router+Migration+Guide|API Router Migration Guide>\n• Migration Tools:\n  - `python tools/migrate_config_imports.py`\n  - `python tools/migrate_router_imports.py`"
        }
    })
    
    # Add help section
    message["blocks"].append({
        "type": "divider"
    })
    message["blocks"].append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*Need Help?*\nIf you need assistance with migration, please contact the Analysis Engine team or open a support ticket."
        }
    })
    
    # Send message
    try:
        response = requests.post(webhook_url, json=message)
        if response.status_code == 200:
            print("Slack reminder posted successfully")
        else:
            print(f"Error posting to Slack: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Error posting to Slack: {e}")


def create_jira_tickets(report: Dict[str, Any]) -> None:
    """
    Create or update Jira tickets for migration tasks.
    
    Args:
        report: The deprecation report
    """
    # Jira configuration
    jira_url = os.environ.get("JIRA_URL", "")
    jira_user = os.environ.get("JIRA_USER", "")
    jira_token = os.environ.get("JIRA_TOKEN", "")
    jira_project = os.environ.get("JIRA_PROJECT", "")
    
    if not jira_url or not jira_user or not jira_token or not jira_project:
        print("Jira configuration incomplete, skipping Jira ticket creation")
        return
    
    # Calculate days until removal
    removal_date = datetime.date(2023, 12, 31)
    days_until_removal = (removal_date - datetime.date.today()).days
    days_message = f"{days_until_removal} days" if days_until_removal > 0 else "PAST DUE"
    
    # Determine priority
    priority = "Medium"
    if days_until_removal <= 30:
        priority = "Highest"
    elif days_until_removal <= 60:
        priority = "High"
    elif days_until_removal <= 90:
        priority = "Medium"
    
    # Create tickets for each module
    for module, data in report.get("modules", {}).items():
        # Skip if no usages
        if data.get("total_usages", 0) == 0:
            continue
        
        # Create ticket summary and description
        summary = f"Migrate from deprecated module: {module}"
        description = f"""
        This ticket tracks the migration from the deprecated module {module}.
        
        h2. Timeline
        * Removal Date: December 31, 2023
        * Days Remaining: {days_message}
        
        h2. Migration Details
        * Unique Locations: {data.get('unique_locations', 0)}
        * Total Usages: {data.get('total_usages', 0)}
        
        h2. Migration Resources
        * [Configuration Migration Guide|https://confluence.example.com/display/DEV/Configuration+Migration+Guide]
        * [API Router Migration Guide|https://confluence.example.com/display/DEV/API+Router+Migration+Guide]
        * Migration Tools:
        ** {{python tools/migrate_config_imports.py}}
        ** {{python tools/migrate_router_imports.py}}
        
        h2. Usage Locations
        """
        
        # Add usage locations
        for usage in data.get("usages", []):
            description += f"* {usage.get('caller_file', '')}:{usage.get('caller_line', '')} in {usage.get('caller_function', '')}\n"
        
        # Create ticket data
        ticket_data = {
            "fields": {
                "project": {
                    "key": jira_project
                },
                "summary": summary,
                "description": description,
                "issuetype": {
                    "name": "Task"
                },
                "priority": {
                    "name": priority
                },
                "labels": ["deprecated-module-migration", module.replace(".", "-")]
            }
        }
        
        # Check if ticket already exists
        search_url = f"{jira_url}/rest/api/2/search"
        search_params = {
            "jql": f'project = {jira_project} AND summary ~ "{summary}" AND resolution = Unresolved',
            "fields": "id,key"
        }
        
        try:
            response = requests.get(
                search_url,
                params=search_params,
                auth=(jira_user, jira_token),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                issues = response.json().get("issues", [])
                
                if issues:
                    # Update existing ticket
                    issue_key = issues[0]["key"]
                    update_url = f"{jira_url}/rest/api/2/issue/{issue_key}"
                    
                    update_data = {
                        "fields": {
                            "description": description,
                            "priority": {
                                "name": priority
                            }
                        }
                    }
                    
                    update_response = requests.put(
                        update_url,
                        json=update_data,
                        auth=(jira_user, jira_token),
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if update_response.status_code == 204:
                        print(f"Updated Jira ticket {issue_key} for {module}")
                    else:
                        print(f"Error updating Jira ticket for {module}: {update_response.status_code} {update_response.text}")
                else:
                    # Create new ticket
                    create_url = f"{jira_url}/rest/api/2/issue"
                    
                    create_response = requests.post(
                        create_url,
                        json=ticket_data,
                        auth=(jira_user, jira_token),
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if create_response.status_code == 201:
                        issue_key = create_response.json()["key"]
                        print(f"Created Jira ticket {issue_key} for {module}")
                    else:
                        print(f"Error creating Jira ticket for {module}: {create_response.status_code} {create_response.text}")
            else:
                print(f"Error searching for Jira tickets: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Error interacting with Jira: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Send deprecation reminders")
    parser.add_argument("--email", action="store_true", help="Send reminders via email")
    parser.add_argument("--slack", action="store_true", help="Post reminders to Slack")
    parser.add_argument("--jira", action="store_true", help="Create or update Jira tickets")
    args = parser.parse_args()
    
    # Load report
    report = load_report()
    
    # Send email reminder
    if args.email:
        send_email_reminder(report)
    
    # Post Slack reminder
    if args.slack:
        post_slack_reminder(report)
    
    # Create Jira tickets
    if args.jira:
        create_jira_tickets(report)
    
    # If no options specified, show help
    if not (args.email or args.slack or args.jira):
        parser.print_help()


if __name__ == "__main__":
    main()
