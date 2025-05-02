# Configuration Migration Tools

This directory contains tools to help with the migration from deprecated configuration modules to the new consolidated module.

## Overview

The Analysis Engine Service has consolidated its configuration system into a single module. These tools help with the migration process:

1. **Deprecation Report Generator**: Generates reports on usage of deprecated modules
2. **Migration Tool**: Helps automate the migration of imports
3. **Scheduled Report Generator**: Generates and distributes reports on a schedule

## Deprecation Report Generator

This tool generates reports on usage of deprecated modules. It helps track migration progress and identify areas that need attention.

### Usage

```bash
python tools/deprecation_report.py [--format {text,json,html,csv}] [--output FILE]
```

### Options

- `--format FORMAT`: Output format (text, json, html, csv) [default: text]
- `--output FILE`: Output file [default: stdout]

### Example

```bash
# Generate HTML report
python tools/deprecation_report.py --format html --output deprecation_report.html

# Generate CSV report
python tools/deprecation_report.py --format csv --output deprecation_report.csv
```

## Migration Tool

This tool helps automate the migration of imports from deprecated configuration modules to the new consolidated module.

### Usage

```bash
python tools/migrate_config_imports.py [--path PATH] [--dry-run] [--verbose]
```

### Options

- `--path PATH`: Path to search for Python files [default: .]
- `--dry-run`: Show changes without applying them
- `--verbose`: Show detailed information

### Example

```bash
# Dry run to see what would be changed
python tools/migrate_config_imports.py --dry-run --verbose

# Apply changes to a specific directory
python tools/migrate_config_imports.py --path analysis_engine/api
```

## Scheduled Report Generator

This tool generates deprecation reports on a schedule and sends them to the team. It can be run as a cron job or scheduled task.

### Usage

```bash
python tools/scheduled_deprecation_report.py [--email] [--slack] [--jira]
```

### Options

- `--email`: Send report via email
- `--slack`: Send report to Slack
- `--jira`: Create or update Jira ticket with report

### Environment Variables

The following environment variables are used by the scheduled report generator:

#### Email

- `SMTP_SERVER`: SMTP server hostname [default: smtp.example.com]
- `SMTP_PORT`: SMTP server port [default: 587]
- `SMTP_USERNAME`: SMTP username
- `SMTP_PASSWORD`: SMTP password
- `EMAIL_SENDER`: Sender email address [default: noreply@example.com]
- `EMAIL_RECIPIENTS`: Comma-separated list of recipient email addresses

#### Slack

- `SLACK_WEBHOOK_URL`: Slack webhook URL

#### Jira

- `JIRA_URL`: Jira URL
- `JIRA_TOKEN`: Jira API token
- `JIRA_PROJECT`: Jira project key
- `JIRA_ISSUE_TYPE`: Jira issue type [default: Task]
- `JIRA_TICKET`: Existing Jira ticket ID (if updating)

### Example

```bash
# Send report via email
export EMAIL_RECIPIENTS="team@example.com"
python tools/scheduled_deprecation_report.py --email

# Send report to Slack
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
python tools/scheduled_deprecation_report.py --slack

# Update Jira ticket
export JIRA_URL="https://jira.example.com"
export JIRA_TOKEN="your-token"
export JIRA_PROJECT="PLATFORM"
export JIRA_TICKET="PLATFORM-123"
python tools/scheduled_deprecation_report.py --jira
```

### Scheduling

To run the report generator on a schedule, you can use cron (Linux/macOS) or Task Scheduler (Windows).

#### Cron Example (Linux/macOS)

```bash
# Run every Monday at 9:00 AM
0 9 * * 1 cd /path/to/project && python tools/scheduled_deprecation_report.py --email --slack
```

#### Task Scheduler Example (Windows)

Create a batch file (e.g., `generate_report.bat`):

```batch
@echo off
cd /d D:\path\to\project
python tools\scheduled_deprecation_report.py --email --slack
```

Then schedule this batch file to run weekly using Task Scheduler.
