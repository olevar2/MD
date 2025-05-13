"""
Data Reconciliation Repository.

This module provides repository classes for storing and retrieving reconciliation data.
It implements the data access layer for the Data Reconciliation system.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.dialects.postgresql import JSONB, insert

from models.models_2 import (
    ReconciliationConfig,
    ReconciliationTask,
    ReconciliationResult,
    ReconciliationIssue,
    ReconciliationStatus,
    ReconciliationSeverity,
    ReconciliationType
)

logger = logging.getLogger(__name__)


class ReconciliationRepository:
    """Repository for data reconciliation storage and retrieval."""

    def __init__(self, engine: AsyncEngine):
        """
        Initialize the repository.

        Args:
            engine: SQLAlchemy async engine
        """
        self.engine = engine
        self.metadata = sa.MetaData(schema="reconciliation")

        # Define tables
        self.configs_table = sa.Table(
            "reconciliation_configs",
            self.metadata,
            sa.Column("config_id", sa.String, primary_key=True),
            sa.Column("name", sa.String, nullable=False),
            sa.Column("description", sa.String, nullable=True),
            sa.Column("reconciliation_type", sa.String, nullable=False),
            sa.Column("primary_source", JSONB, nullable=False),
            sa.Column("secondary_source", JSONB, nullable=True),
            sa.Column("rules", JSONB, nullable=False),
            sa.Column("schedule", sa.String, nullable=True),
            sa.Column("enabled", sa.Boolean, nullable=False, default=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("created_by", sa.String, nullable=True),
            sa.Column("metadata", JSONB, nullable=True),
        )

        self.tasks_table = sa.Table(
            "reconciliation_tasks",
            self.metadata,
            sa.Column("task_id", sa.String, primary_key=True),
            sa.Column("config_id", sa.String, nullable=False),
            sa.Column("status", sa.String, nullable=False),
            sa.Column("scheduled_time", sa.DateTime(timezone=True), nullable=False),
            sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
            sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
            sa.Column("result_id", sa.String, nullable=True),
            sa.Column("metadata", JSONB, nullable=True),
        )

        self.results_table = sa.Table(
            "reconciliation_results",
            self.metadata,
            sa.Column("result_id", sa.String, primary_key=True),
            sa.Column("config_id", sa.String, nullable=False),
            sa.Column("status", sa.String, nullable=False),
            sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
            sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
            sa.Column("total_records", sa.Integer, nullable=False, default=0),
            sa.Column("matched_records", sa.Integer, nullable=False, default=0),
            sa.Column("summary", JSONB, nullable=True),
            sa.Column("metadata", JSONB, nullable=True),
        )

        self.issues_table = sa.Table(
            "reconciliation_issues",
            self.metadata,
            sa.Column("issue_id", sa.String, primary_key=True),
            sa.Column("result_id", sa.String, nullable=False),
            sa.Column("rule_id", sa.String, nullable=False),
            sa.Column("field", sa.String, nullable=False),
            sa.Column("primary_value", JSONB, nullable=True),
            sa.Column("secondary_value", JSONB, nullable=True),
            sa.Column("difference", sa.Float, nullable=True),
            sa.Column("severity", sa.String, nullable=False),
            sa.Column("description", sa.String, nullable=False),
            sa.Column("metadata", JSONB, nullable=True),
        )

    async def initialize(self):
        """Initialize the repository by creating tables if they don't exist."""
        async with self.engine.begin() as conn:
            # Create schema if it doesn't exist
            await conn.execute(sa.text("CREATE SCHEMA IF NOT EXISTS reconciliation"))

            # Create tables
            await conn.run_sync(self.metadata.create_all)

    async def store_config(self, config: ReconciliationConfig) -> str:
        """
        Store reconciliation configuration.

        Args:
            config: Reconciliation configuration

        Returns:
            Config ID
        """
        try:
            config_dict = config.dict()

            async with self.engine.begin() as conn:
                await conn.execute(
                    self.configs_table.insert().values(**config_dict)
                )

                logger.info(f"Stored reconciliation config {config.config_id}")
                return config.config_id
        except Exception as e:
            logger.error(f"Failed to store reconciliation config: {e}")
            raise

    async def store_task(self, task: ReconciliationTask) -> str:
        """
        Store reconciliation task.

        Args:
            task: Reconciliation task

        Returns:
            Task ID
        """
        try:
            task_dict = task.dict()

            async with self.engine.begin() as conn:
                await conn.execute(
                    self.tasks_table.insert().values(**task_dict)
                )

                logger.info(f"Stored reconciliation task {task.task_id}")
                return task.task_id
        except Exception as e:
            logger.error(f"Failed to store reconciliation task: {e}")
            raise

    async def store_result(self, result: ReconciliationResult) -> str:
        """
        Store reconciliation result.

        Args:
            result: Reconciliation result

        Returns:
            Result ID
        """
        try:
            # Extract issues for separate storage
            issues = result.issues
            result_dict = result.dict(exclude={"issues"})

            async with self.engine.begin() as conn:
                # Store result
                await conn.execute(
                    self.results_table.insert().values(**result_dict)
                )

                # Store issues
                for issue in issues:
                    issue_dict = issue.dict()
                    issue_dict["result_id"] = result.result_id
                    await conn.execute(
                        self.issues_table.insert().values(**issue_dict)
                    )

                logger.info(f"Stored reconciliation result {result.result_id} with {len(issues)} issues")
                return result.result_id
        except Exception as e:
            logger.error(f"Failed to store reconciliation result: {e}")
            raise

    async def update_task_status(
        self,
        task_id: str,
        status: ReconciliationStatus,
        result_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """
        Update the status of a reconciliation task.

        Args:
            task_id: Task ID
            status: New status
            result_id: Result ID (if completed)
            start_time: Start time (if started)
            end_time: End time (if completed or failed)

        Returns:
            Success status
        """
        try:
            update_values = {"status": status.value}

            if result_id is not None:
                update_values["result_id"] = result_id

            if start_time is not None:
                update_values["start_time"] = start_time

            if end_time is not None:
                update_values["end_time"] = end_time

            async with self.engine.begin() as conn:
                result = await conn.execute(
                    self.tasks_table.update()
                    .where(self.tasks_table.c.task_id == task_id)
                    .values(**update_values)
                )

                if result.rowcount == 0:
                    logger.warning(f"No task found with ID {task_id}")
                    return False

                logger.info(f"Updated task {task_id} status to {status.value}")
                return True
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
            raise

    async def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """
        Get reconciliation configuration.

        Args:
            config_id: Config ID

        Returns:
            Reconciliation configuration or None if not found
        """
        try:
            query = sa.select(self.configs_table).where(
                self.configs_table.c.config_id == config_id
            )

            async with self.engine.connect() as conn:
                result = await conn.execute(query)
                record = result.fetchone()

                if record is None:
                    return None

                return dict(record)
        except Exception as e:
            logger.error(f"Failed to get reconciliation config: {e}")
            raise

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get reconciliation task.

        Args:
            task_id: Task ID

        Returns:
            Reconciliation task or None if not found
        """
        try:
            query = sa.select(self.tasks_table).where(
                self.tasks_table.c.task_id == task_id
            )

            async with self.engine.connect() as conn:
                result = await conn.execute(query)
                record = result.fetchone()

                if record is None:
                    return None

                return dict(record)
        except Exception as e:
            logger.error(f"Failed to get reconciliation task: {e}")
            raise

    async def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Get reconciliation result.

        Args:
            result_id: Result ID

        Returns:
            Reconciliation result or None if not found
        """
        try:
            # Get result
            query = sa.select(self.results_table).where(
                self.results_table.c.result_id == result_id
            )

            async with self.engine.connect() as conn:
                result = await conn.execute(query)
                record = result.fetchone()

                if record is None:
                    return None

                result_dict = dict(record)

                # Get issues
                issues_query = sa.select(self.issues_table).where(
                    self.issues_table.c.result_id == result_id
                )

                issues_result = await conn.execute(issues_query)
                issues = [dict(issue) for issue in issues_result.fetchall()]

                result_dict["issues"] = issues

                return result_dict
        except Exception as e:
            logger.error(f"Failed to get reconciliation result: {e}")
            raise

    async def get_configs(
        self,
        enabled: Optional[bool] = None,
        reconciliation_type: Optional[ReconciliationType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get reconciliation configurations.

        Args:
            enabled: Filter by enabled status
            reconciliation_type: Filter by reconciliation type
            limit: Maximum number of records to return
            offset: Offset for pagination

        Returns:
            List of reconciliation configurations
        """
        try:
            query = sa.select(self.configs_table)

            if enabled is not None:
                query = query.where(self.configs_table.c.enabled == enabled)

            if reconciliation_type is not None:
                query = query.where(self.configs_table.c.reconciliation_type == reconciliation_type.value)

            query = query.order_by(self.configs_table.c.created_at.desc())
            query = query.limit(limit).offset(offset)

            async with self.engine.connect() as conn:
                result = await conn.execute(query)
                records = result.fetchall()

                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Failed to get reconciliation configs: {e}")
            raise

    async def get_tasks(
        self,
        config_id: Optional[str] = None,
        status: Optional[ReconciliationStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get reconciliation tasks.

        Args:
            config_id: Filter by config ID
            status: Filter by status
            start_date: Filter by scheduled time (start)
            end_date: Filter by scheduled time (end)
            limit: Maximum number of records to return
            offset: Offset for pagination

        Returns:
            List of reconciliation tasks
        """
        try:
            query = sa.select(self.tasks_table)

            if config_id is not None:
                query = query.where(self.tasks_table.c.config_id == config_id)

            if status is not None:
                query = query.where(self.tasks_table.c.status == status.value)

            if start_date is not None:
                query = query.where(self.tasks_table.c.scheduled_time >= start_date)

            if end_date is not None:
                query = query.where(self.tasks_table.c.scheduled_time <= end_date)

            query = query.order_by(self.tasks_table.c.scheduled_time.desc())
            query = query.limit(limit).offset(offset)

            async with self.engine.connect() as conn:
                result = await conn.execute(query)
                records = result.fetchall()

                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Failed to get reconciliation tasks: {e}")
            raise

    async def get_results(
        self,
        config_id: Optional[str] = None,
        status: Optional[ReconciliationStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
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
        try:
            query = sa.select(self.results_table)

            if config_id is not None:
                query = query.where(self.results_table.c.config_id == config_id)

            if status is not None:
                query = query.where(self.results_table.c.status == status.value)

            if start_date is not None:
                query = query.where(self.results_table.c.start_time >= start_date)

            if end_date is not None:
                query = query.where(self.results_table.c.start_time <= end_date)

            query = query.order_by(self.results_table.c.start_time.desc())
            query = query.limit(limit).offset(offset)

            async with self.engine.connect() as conn:
                result = await conn.execute(query)
                records = result.fetchall()

                results = []
                for record in records:
                    result_dict = dict(record)
                    result_id = result_dict["result_id"]

                    # Get issues
                    issues_query = sa.select(self.issues_table).where(
                        self.issues_table.c.result_id == result_id
                    )

                    issues_result = await conn.execute(issues_query)
                    issues = [dict(issue) for issue in issues_result.fetchall()]

                    result_dict["issues"] = issues
                    results.append(result_dict)

                return results
        except Exception as e:
            logger.error(f"Failed to get reconciliation results: {e}")
            raise
