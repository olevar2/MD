"""
Tool Effectiveness Scheduler

This module provides scheduled tasks for calculating and storing tool effectiveness metrics
on a regular basis, ensuring metrics are up-to-date for dashboard visualization and analysis.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services.enhanced_tool_effectiveness import EnhancedToolEffectivenessTracker
from analysis_engine.db.connection import SessionLocal
from analysis_engine.services.tool_effectiveness import MarketRegime, TimeFrame

class ToolEffectivenessScheduler:
    """Schedules regular calculation of tool effectiveness metrics"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scheduler_task = None
        self.running = False
        self.scheduled_tasks = {}

    async def start(self):
        """Start the scheduler as an async task"""
        if self.scheduler_task and not self.scheduler_task.done():
            self.logger.warning("Scheduler already running")
            return False

        self.running = True

        # Schedule tasks with their intervals and execution times
        self.scheduled_tasks = {
            "hourly": {"interval": 60 * 60, "next_run": self._next_hour(), "func": self.calculate_hourly_metrics},
            "daily": {"interval": 24 * 60 * 60, "next_run": self._next_time(0, 30), "func": self.calculate_daily_metrics},
            "weekly": {"interval": 7 * 24 * 60 * 60, "next_run": self._next_monday(1, 0), "func": self.calculate_weekly_metrics},
            "monthly": {"interval": 30 * 24 * 60 * 60, "next_run": self._next_month_day(1, 2, 0), "func": self.calculate_monthly_metrics}
        }

        # Start scheduler as an asyncio task
        self.scheduler_task = asyncio.create_task(self._run_scheduler())

        self.logger.info("Tool effectiveness scheduler started")
        return True

    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_task and not self.scheduler_task.done():
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Tool effectiveness scheduler stopped")

    async def _run_scheduler(self):
        """Run the scheduler loop"""
        try:
            while self.running:
                now = datetime.now().timestamp()

                # Check each scheduled task
                for task_name, task_info in self.scheduled_tasks.items():
                    if now >= task_info["next_run"]:
                        # Run the task
                        self.logger.info(f"Running scheduled task: {task_name}")
                        asyncio.create_task(task_info["func"]())

                        # Update next run time
                        task_info["next_run"] = now + task_info["interval"]

                # Sleep for a minute before checking again
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.logger.info("Scheduler task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in scheduler loop: {e}")
            if self.running:
                # Restart the scheduler task
                self.scheduler_task = asyncio.create_task(self._run_scheduler())

    def _next_hour(self):
        """Get timestamp for the start of the next hour"""
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return next_hour.timestamp()

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

    async def calculate_hourly_metrics(self):
        """Calculate metrics for the past hour"""
        self.logger.info("Calculating hourly effectiveness metrics")

        try:
            # Use a new session for this task
            db = SessionLocal()
            await self._calculate_metrics(
                db=db,
                lookback_hours=1,
                save_to_db=True,
                metric_types=["win_rate", "profit_factor", "expected_payoff"]
            )
            db.close()
            self.logger.info("Hourly effectiveness metrics calculation completed")
            return True

        except Exception as e:
            self.logger.error(f"Error calculating hourly metrics: {str(e)}")
            return False

    async def calculate_daily_metrics(self):
        """Calculate metrics for the past day"""
        self.logger.info("Calculating daily effectiveness metrics")

        try:
            # Use a new session for this task
            db = SessionLocal()
            await self._calculate_metrics(
                db=db,
                lookback_hours=24,
                save_to_db=True,
                metric_types=["win_rate", "profit_factor", "expected_payoff", "reliability_by_regime"]
            )
            db.close()
            self.logger.info("Daily effectiveness metrics calculation completed")
            return True

        except Exception as e:
            self.logger.error(f"Error calculating daily metrics: {str(e)}")
            return False

    async def calculate_weekly_metrics(self):
        """Calculate metrics for the past week"""
        self.logger.info("Calculating weekly effectiveness metrics")

        try:
            # Use a new session for this task
            db = SessionLocal()
            await self._calculate_metrics(
                db=db,
                lookback_hours=168,  # 7 days
                save_to_db=True,
                metric_types=["win_rate", "profit_factor", "expected_payoff", "reliability_by_regime"],
                include_regime_specific=True
            )
            db.close()
            self.logger.info("Weekly effectiveness metrics calculation completed")
            return True

        except Exception as e:
            self.logger.error(f"Error calculating weekly metrics: {str(e)}")
            return False

    async def calculate_monthly_metrics(self):
        """Calculate metrics for the past month"""
        self.logger.info("Calculating monthly effectiveness metrics")

        try:
            # Use a new session for this task
            db = SessionLocal()
            await self._calculate_metrics(
                db=db,
                lookback_hours=720,  # 30 days
                save_to_db=True,
                metric_types=["win_rate", "profit_factor", "expected_payoff", "reliability_by_regime"],
                include_regime_specific=True,
                include_signal_quality=True
            )
            db.close()
            self.logger.info("Monthly effectiveness metrics calculation completed")
            return True

        except Exception as e:
            self.logger.error(f"Error calculating monthly metrics: {str(e)}")
            return False

    async def _calculate_metrics(
        self,
        db: Session,
        lookback_hours: int,
        save_to_db: bool = True,
        metric_types: Optional[List[str]] = None,
        include_regime_specific: bool = False,
        include_signal_quality: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate metrics for the specified lookback period

        Args:
            db: Database session
            lookback_hours: Hours to look back for data
            save_to_db: Whether to save results to the database
            metric_types: Types of metrics to calculate
            include_regime_specific: Whether to include regime-specific metrics
            include_signal_quality: Whether to include signal quality metrics

        Returns:
            Dictionary with calculated metrics
        """
        # Create repository and tracker
        repository = ToolEffectivenessRepository(db)
        tracker = EnhancedToolEffectivenessTracker()

        # Calculate time range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=lookback_hours)

        # Get all tool IDs with signals in this period
        tool_ids = repository.get_tool_ids(start_date=start_date, end_date=end_date)
        results = {}

        for tool_id in tool_ids:
            # Get outcomes for this tool
            outcomes = repository.get_outcomes(
                tool_id=tool_id,
                start_date=start_date,
                end_date=end_date
            )

            if not outcomes:
                continue

            # Convert to domain models
            domain_outcomes = []
            for outcome in outcomes:
                signal = outcome.signal

                signal_event = {
                    "id": signal.signal_id,
                    "tool_name": signal.tool_id,
                    "signal_type": signal.signal_type,
                    "direction": signal.additional_data.get("direction", "unknown") if signal.additional_data else "unknown",
                    "strength": signal.confidence,
                    "timestamp": signal.timestamp,
                    "symbol": signal.instrument,
                    "timeframe": signal.timeframe,
                    "price_at_signal": signal.additional_data.get("price_at_signal") if signal.additional_data else None,
                    "market_context": {"regime": signal.market_regime}
                }

                domain_outcome = {
                    "signal_event": signal_event,
                    "outcome": "success" if outcome.success else "failure",
                    "exit_price": outcome.additional_data.get("exit_price") if outcome.additional_data else None,
                    "exit_timestamp": outcome.exit_timestamp,
                    "profit_loss": outcome.realized_profit,
                    "max_favorable_price": outcome.additional_data.get("max_favorable_price") if outcome.additional_data else None,
                    "max_adverse_price": outcome.additional_data.get("max_adverse_price") if outcome.additional_data else None
                }

                domain_outcomes.append(domain_outcome)

            # Calculate metrics
            metrics = tracker.calculate_metrics(
                outcomes=domain_outcomes,
                tool_name=tool_id,
                start_date=start_date,
                end_date=end_date
            )

            results[tool_id] = metrics

            # Save to database if requested
            if save_to_db:
                self._save_metrics_to_db(repository, tool_id, metrics, lookback_hours)

            # Calculate regime-specific metrics if requested
            if include_regime_specific:
                for regime in MarketRegime:
                    regime_metrics = tracker.calculate_metrics(
                        outcomes=domain_outcomes,
                        tool_name=tool_id,
                        market_regime=regime,
                        start_date=start_date,
                        end_date=end_date
                    )

                    # Save regime-specific metrics
                    if save_to_db and regime_metrics.get("filtered_sample_size", 0) > 0:
                        self._save_regime_metrics_to_db(repository, tool_id, regime, regime_metrics, lookback_hours)

        return results

    def _save_metrics_to_db(
        self,
        repository: ToolEffectivenessRepository,
        tool_id: str,
        metrics: Dict[str, Any],
        lookback_hours: int
    ) -> None:
        """Save metrics to the database"""
        try:
            # Extract core metrics
            core_metrics = metrics.get("core_metrics", {})
            for metric_name, metric_data in core_metrics.items():
                if metric_name == "reliability_by_regime":
                    # Skip this as it's a complex metric with its own structure
                    continue

                if metric_data.get("value") is not None:
                    # Create effectiveness metric record
                    repository.create_metric({
                        'tool_id': tool_id,
                        'metric_type': metric_name,
                        'value': metric_data["value"],
                        'sample_size': metrics.get("filtered_sample_size", 0),
                        'lookback_hours': lookback_hours,
                        'created_at': datetime.utcnow()
                    })

            # Save composite score if available
            composite_score = metrics.get("composite_score")
            if composite_score is not None:
                repository.create_metric({
                    'tool_id': tool_id,
                    'metric_type': 'composite_score',
                    'value': composite_score,
                    'sample_size': metrics.get("filtered_sample_size", 0),
                    'lookback_hours': lookback_hours,
                    'created_at': datetime.utcnow()
                })

            # Save signal quality metrics if available
            signal_quality = metrics.get("signal_quality_metrics", {})
            quality_metrics = signal_quality.get("metrics", {})

            for metric_name, quality_data in quality_metrics.items():
                if quality_data.get("value") is not None:
                    repository.create_metric({
                        'tool_id': tool_id,
                        'metric_type': f"quality_{metric_name}",
                        'value': quality_data["value"],
                        'sample_size': quality_data.get("sample_size", 0),
                        'lookback_hours': lookback_hours,
                        'created_at': datetime.utcnow()
                    })

        except Exception as e:
            self.logger.error(f"Error saving metrics to database: {str(e)}")

    def _save_regime_metrics_to_db(
        self,
        repository: ToolEffectivenessRepository,
        tool_id: str,
        regime: MarketRegime,
        metrics: Dict[str, Any],
        lookback_hours: int
    ) -> None:
        """Save regime-specific metrics to the database"""
        try:
            # Extract core metrics
            core_metrics = metrics.get("core_metrics", {})
            for metric_name, metric_data in core_metrics.items():
                if metric_name == "reliability_by_regime":
                    continue

                if metric_data.get("value") is not None:
                    # Create effectiveness metric record with regime
                    repository.create_metric({
                        'tool_id': tool_id,
                        'metric_type': f"{regime.value}_{metric_name}",
                        'value': metric_data["value"],
                        'sample_size': metrics.get("filtered_sample_size", 0),
                        'market_regime': regime.value,
                        'lookback_hours': lookback_hours,
                        'created_at': datetime.utcnow()
                    })

        except Exception as e:
            self.logger.error(f"Error saving regime metrics to database: {str(e)}")


# Singleton instance
effectiveness_scheduler = ToolEffectivenessScheduler()
