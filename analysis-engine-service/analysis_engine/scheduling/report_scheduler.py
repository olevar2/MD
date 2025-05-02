"""
Report Scheduler

This module provides functionality for scheduling regular report generation
and distributing reports to subscribers.
"""

import asyncio
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from analysis_engine.batch.metric_calculator import MetricBatchCalculator
from analysis_engine.services.tool_effectiveness_service import ToolEffectivenessService


logger = logging.getLogger(__name__)


class ReportScheduler:
    """
    Scheduler for automatic report generation and distribution
    """

    def __init__(self, db_factory):
        """
        Initialize with a factory function that creates database sessions
        The factory is used to ensure fresh connections for long-running processes
        """
        self.db_factory = db_factory
        self._scheduler_task = None
        self._running = False
        self._subscribers = {}  # {report_type: [subscriber_info]}
        self.scheduled_tasks = {}

    async def start(self):
        """Start the scheduler as an async task"""
        if self._scheduler_task and not self._scheduler_task.done():
            logger.warning("Scheduler is already running")
            return

        self._running = True

        # Setup schedules
        self._setup_schedules()

        # Start scheduler as an asyncio task
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        logger.info("Report scheduler started")

    async def stop(self):
        """Stop the scheduler task"""
        self._running = False
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Report scheduler stopped")

    def _setup_schedules(self):
        """Setup recurring schedules for different report types"""
        # Daily reports - run at 1:00 AM
        self.scheduled_tasks["daily"] = {
            "interval": 24 * 60 * 60,  # 24 hours in seconds
            "next_run": self._next_time(1, 0),  # 1:00 AM
            "func": self._generate_daily_reports
        }

        # Weekly reports - run on Monday at 2:00 AM
        self.scheduled_tasks["weekly"] = {
            "interval": 7 * 24 * 60 * 60,  # 7 days in seconds
            "next_run": self._next_monday(2, 0),  # 2:00 AM
            "func": self._generate_weekly_reports
        }

        # Monthly reports - run on the 1st of each month at 3:00 AM
        self.scheduled_tasks["monthly"] = {
            "interval": 30 * 24 * 60 * 60,  # Approximately 30 days in seconds
            "next_run": self._next_month_day(1, 3, 0),  # 3:00 AM on the 1st
            "func": self._generate_monthly_reports
        }

        logger.info("Report schedules created")

    async def _run_scheduler(self):
        """Run the scheduler loop"""
        try:
            while self._running:
                now = datetime.now().timestamp()

                # Check each scheduled task
                for task_name, task_info in self.scheduled_tasks.items():
                    if now >= task_info["next_run"]:
                        # Run the task
                        logger.info(f"Running scheduled report task: {task_name}")
                        asyncio.create_task(task_info["func"]())

                        # Update next run time
                        task_info["next_run"] = now + task_info["interval"]

                # Sleep for a minute before checking again
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            logger.info("Scheduler task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
            if self._running:
                # Restart the scheduler task
                self._scheduler_task = asyncio.create_task(self._run_scheduler())

    def _next_time(self, hour, minute):
        """Get timestamp for the next occurrence of a specific time"""
        now = datetime.now()
        target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target_time <= now:
            target_time += timedelta(days=1)
        return target_time.timestamp()

    def _next_monday(self, hour, minute):
        """Get timestamp for the next Monday at a specific time"""
        now = datetime.now()
        days_ahead = 7 - now.weekday()
        if days_ahead == 0 and now.hour >= hour and now.minute >= minute:
            days_ahead = 7
        next_monday = now.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=days_ahead)
        return next_monday.timestamp()

    def _next_month_day(self, day, hour, minute):
        """Get timestamp for the specified day of the next month"""
        now = datetime.now()
        if now.day < day:
            # Target day is in the current month
            target_date = now.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
        else:
            # Target day is in the next month
            if now.month == 12:
                target_date = now.replace(year=now.year+1, month=1, day=day, hour=hour, minute=minute, second=0, microsecond=0)
            else:
                target_date = now.replace(month=now.month+1, day=day, hour=hour, minute=minute, second=0, microsecond=0)
        return target_date.timestamp()

    async def _generate_daily_reports(self):
        """Generate daily reports for all tools"""
        logger.info("Generating daily reports")
        try:
            with self.db_factory() as db:
                batch_calculator = MetricBatchCalculator(db)
                results = await batch_calculator.recalculate_all_metrics(days_back=7)
                logger.info(f"Daily metrics recalculation complete: {results['tools_processed']} tools processed")

                # Generate daily reports
                report_results = await batch_calculator.generate_periodic_reports(report_type="daily")
                logger.info(f"Daily reports generation complete: {report_results['reports_generated']} reports generated")

                # Distribute reports to subscribers
                await self._distribute_reports("daily", report_results)

            return True
        except Exception as e:
            logger.error(f"Error generating daily reports: {e}")
            return False

    async def _generate_weekly_reports(self):
        """Generate weekly reports for all tools"""
        logger.info("Generating weekly reports")
        try:
            with self.db_factory() as db:
                batch_calculator = MetricBatchCalculator(db)
                results = await batch_calculator.recalculate_all_metrics(days_back=30)
                logger.info(f"Weekly metrics recalculation complete: {results['tools_processed']} tools processed")

                # Generate weekly reports
                report_results = await batch_calculator.generate_periodic_reports(report_type="weekly")
                logger.info(f"Weekly reports generation complete: {report_results['reports_generated']} reports generated")

                # Distribute reports to subscribers
                await self._distribute_reports("weekly", report_results)

            return True
        except Exception as e:
            logger.error(f"Error generating weekly reports: {e}")
            return False

    async def _generate_monthly_reports(self):
        """Generate monthly reports for all tools"""
        logger.info("Generating monthly reports")
        try:
            with self.db_factory() as db:
                batch_calculator = MetricBatchCalculator(db)
                results = await batch_calculator.recalculate_all_metrics(days_back=90)
                logger.info(f"Monthly metrics recalculation complete: {results['tools_processed']} tools processed")

                # Generate monthly reports
                report_results = await batch_calculator.generate_periodic_reports(report_type="monthly")
                logger.info(f"Monthly reports generation complete: {report_results['reports_generated']} reports generated")

                # Check if it's time for quarterly reports (every 3 months)
                today = datetime.now()
                if today.month in (3, 6, 9, 12):
                    logger.info("Generating quarterly reports")
                    quarterly_results = await batch_calculator.generate_periodic_reports(report_type="quarterly")
                    logger.info(f"Quarterly reports generation complete: {quarterly_results['reports_generated']} reports generated")
                    await self._distribute_reports("quarterly", quarterly_results)

                # Distribute reports to subscribers
                await self._distribute_reports("monthly", report_results)

            return True
        except Exception as e:
            logger.error(f"Error generating monthly reports: {e}")
            return False

    async def _distribute_reports(self, report_type: str, report_results: Dict[str, Any]):
        """Distribute reports to subscribers"""
        if report_type not in self._subscribers or not self._subscribers[report_type]:
            logger.info(f"No subscribers for {report_type} reports")
            return

        logger.info(f"Distributing {report_type} reports to {len(self._subscribers[report_type])} subscribers")

        # In a real implementation, this would send emails, notifications, etc. asynchronously
        # For now, we just log the distribution
        for subscriber in self._subscribers[report_type]:
            logger.info(f"Would send {report_type} report to {subscriber['email'] if 'email' in subscriber else 'unknown'}")
            # Simulate async sending
            await asyncio.sleep(0.1)

    def add_subscriber(self, report_type: str, subscriber_info: Dict[str, Any]):
        """Add a subscriber for a specific report type"""
        if report_type not in self._subscribers:
            self._subscribers[report_type] = []

        self._subscribers[report_type].append(subscriber_info)
        logger.info(f"Added subscriber for {report_type} reports: {subscriber_info.get('email', 'unknown')}")

    def remove_subscriber(self, report_type: str, email: str):
        """Remove a subscriber for a specific report type"""
        if report_type in self._subscribers:
            self._subscribers[report_type] = [
                s for s in self._subscribers[report_type]
                if s.get('email') != email
            ]
            logger.info(f"Removed subscriber for {report_type} reports: {email}")
