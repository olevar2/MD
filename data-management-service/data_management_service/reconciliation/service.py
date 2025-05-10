"""
Data Reconciliation Service.

This module provides the service layer for the Data Reconciliation system.
It implements the business logic for reconciling data from different sources.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from uuid import uuid4

from data_management_service.reconciliation.models import (
    ReconciliationConfig,
    ReconciliationTask,
    ReconciliationResult,
    ReconciliationIssue,
    ReconciliationStatus,
    ReconciliationSeverity,
    ReconciliationType,
    ReconciliationRule,
    DataSourceConfig
)
from data_management_service.reconciliation.repository import ReconciliationRepository
from data_management_service.historical.service import HistoricalDataService
from data_management_service.reconciliation.executor import ReconciliationExecutor

logger = logging.getLogger(__name__)


class ReconciliationService:
    """Service for data reconciliation."""

    def __init__(
        self,
        repository: ReconciliationRepository,
        historical_service: HistoricalDataService
    ):
        """
        Initialize the service.

        Args:
            repository: Repository for data storage and retrieval
            historical_service: Service for historical data
        """
        self.repository = repository
        self.historical_service = historical_service
        self.executor = ReconciliationExecutor(historical_service)

    async def initialize(self):
        """Initialize the service."""
        await self.repository.initialize()

    async def create_config(
        self,
        name: str,
        reconciliation_type: ReconciliationType,
        primary_source: DataSourceConfig,
        secondary_source: Optional[DataSourceConfig] = None,
        rules: List[ReconciliationRule] = None,
        description: Optional[str] = None,
        schedule: Optional[str] = None,
        enabled: bool = True,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a reconciliation configuration.

        Args:
            name: Configuration name
            reconciliation_type: Type of reconciliation
            primary_source: Primary data source configuration
            secondary_source: Secondary data source configuration
            rules: Reconciliation rules
            description: Configuration description
            schedule: Cron expression for scheduling
            enabled: Whether the configuration is enabled
            created_by: User who created the configuration
            metadata: Additional metadata

        Returns:
            Config ID
        """
        config = ReconciliationConfig(
            name=name,
            reconciliation_type=reconciliation_type,
            primary_source=primary_source,
            secondary_source=secondary_source,
            rules=rules or [],
            description=description,
            schedule=schedule,
            enabled=enabled,
            created_by=created_by,
            metadata=metadata or {}
        )

        return await self.repository.store_config(config)

    async def schedule_task(
        self,
        config_id: str,
        scheduled_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Schedule a reconciliation task.

        Args:
            config_id: Configuration ID
            scheduled_time: Scheduled time (default: now)
            metadata: Additional metadata

        Returns:
            Task ID
        """
        # Get configuration
        config = await self.repository.get_config(config_id)
        if config is None:
            raise ValueError(f"Configuration not found: {config_id}")

        # Create task
        task = ReconciliationTask(
            config_id=config_id,
            scheduled_time=scheduled_time or datetime.utcnow(),
            metadata=metadata or {}
        )

        return await self.repository.store_task(task)

    async def run_task(self, task_id: str) -> str:
        """
        Run a reconciliation task.

        Args:
            task_id: Task ID

        Returns:
            Result ID
        """
        # Get task
        task_dict = await self.repository.get_task(task_id)
        if task_dict is None:
            raise ValueError(f"Task not found: {task_id}")

        # Convert to model
        task = ReconciliationTask(**task_dict)

        # Check if task is already running or completed
        if task.status in [ReconciliationStatus.RUNNING, ReconciliationStatus.COMPLETED]:
            logger.warning(f"Task {task_id} is already {task.status}")
            return task.result_id

        # Get configuration
        config_dict = await self.repository.get_config(task.config_id)
        if config_dict is None:
            raise ValueError(f"Configuration not found: {task.config_id}")

        # Convert to model
        config = ReconciliationConfig(**config_dict)

        # Update task status to running
        start_time = datetime.utcnow()
        await self.repository.update_task_status(
            task_id=task_id,
            status=ReconciliationStatus.RUNNING,
            start_time=start_time
        )

        try:
            # Create initial result
            result = ReconciliationResult(
                result_id=str(uuid4()),
                task_id=task_id,
                config_id=task.config_id,
                status=ReconciliationStatus.RUNNING,
                start_time=start_time,
                end_time=None,
                total_records=0,
                matched_records=0,
                issues=[],
                summary={},
                metadata=task.metadata.copy() if task.metadata else {}
            )

            # Store initial result
            result_id = await self.repository.store_result(result)

            # Execute task using the executor
            updated_result = await self.executor.execute_task(task, config)

            # Update result ID
            updated_result.result_id = result_id

            # Store updated result
            await self.repository.store_result(updated_result)

            # Update task status
            await self.repository.update_task_status(
                task_id=task_id,
                status=updated_result.status,
                result_id=result_id,
                end_time=updated_result.end_time
            )

            return result_id
        except Exception as e:
            logger.error(f"Failed to run reconciliation task: {e}")

            # Update task status to failed
            end_time = datetime.utcnow()
            await self.repository.update_task_status(
                task_id=task_id,
                status=ReconciliationStatus.FAILED,
                end_time=end_time
            )

            # Update result status to failed
            if "result_id" in locals():
                result = ReconciliationResult(
                    result_id=result_id,
                    task_id=task_id,
                    config_id=task.config_id,
                    status=ReconciliationStatus.FAILED,
                    start_time=start_time,
                    end_time=end_time,
                    total_records=0,
                    matched_records=0,
                    issues=[],
                    summary={"error": str(e)},
                    metadata=task.metadata.copy() if task.metadata else {}
                )

                await self.repository.store_result(result)

            raise

    async def _run_cross_source_reconciliation(
        self,
        config: Dict[str, Any],
        result: ReconciliationResult
    ) -> ReconciliationResult:
        """
        Run cross-source reconciliation.

        Args:
            config: Reconciliation configuration
            result: Reconciliation result

        Returns:
            Updated reconciliation result
        """
        # Get data sources
        primary_source = config["primary_source"]
        secondary_source = config["secondary_source"]

        if secondary_source is None:
            raise ValueError("Secondary source is required for cross-source reconciliation")

        # Get data from primary source
        primary_data = await self._get_data_from_source(primary_source)

        # Get data from secondary source
        secondary_data = await self._get_data_from_source(secondary_source)

        # Reconcile data
        return await self._reconcile_dataframes(
            config=config,
            result=result,
            primary_df=primary_data,
            secondary_df=secondary_data,
            primary_name=primary_source["source_id"],
            secondary_name=secondary_source["source_id"]
        )

    async def _run_temporal_reconciliation(
        self,
        config: Dict[str, Any],
        result: ReconciliationResult
    ) -> ReconciliationResult:
        """
        Run temporal reconciliation.

        Args:
            config: Reconciliation configuration
            result: Reconciliation result

        Returns:
            Updated reconciliation result
        """
        # Get data source
        primary_source = config["primary_source"]

        # Get current data
        current_data = await self._get_data_from_source(primary_source)

        # Get historical data (from a day ago)
        historical_source = dict(primary_source)

        # Modify query params for historical data
        if "query_params" in historical_source:
            query_params = dict(historical_source["query_params"])

            # If point_in_time is specified, use a day before that
            if "point_in_time" in query_params:
                point_in_time = datetime.fromisoformat(query_params["point_in_time"])
                query_params["point_in_time"] = (point_in_time - timedelta(days=1)).isoformat()
            else:
                # Otherwise, use a day ago
                query_params["point_in_time"] = (datetime.utcnow() - timedelta(days=1)).isoformat()

            historical_source["query_params"] = query_params

        historical_data = await self._get_data_from_source(historical_source)

        # Reconcile data
        return await self._reconcile_dataframes(
            config=config,
            result=result,
            primary_df=current_data,
            secondary_df=historical_data,
            primary_name="current",
            secondary_name="historical"
        )

    async def _run_derived_reconciliation(
        self,
        config: Dict[str, Any],
        result: ReconciliationResult
    ) -> ReconciliationResult:
        """
        Run derived reconciliation.

        Args:
            config: Reconciliation configuration
            result: Reconciliation result

        Returns:
            Updated reconciliation result
        """
        # Get data sources
        primary_source = config["primary_source"]
        secondary_source = config["secondary_source"]

        if secondary_source is None:
            raise ValueError("Secondary source is required for derived reconciliation")

        # Get data from primary source (derived data)
        primary_data = await self._get_data_from_source(primary_source)

        # Get data from secondary source (source data)
        secondary_data = await self._get_data_from_source(secondary_source)

        # Apply transformations to secondary data to derive comparable data
        if "transformations" in secondary_source:
            for transformation in secondary_source["transformations"]:
                secondary_data = self._apply_transformation(secondary_data, transformation)

        # Reconcile data
        return await self._reconcile_dataframes(
            config=config,
            result=result,
            primary_df=primary_data,
            secondary_df=secondary_data,
            primary_name="derived",
            secondary_name="source"
        )

    async def _run_custom_reconciliation(
        self,
        config: Dict[str, Any],
        result: ReconciliationResult
    ) -> ReconciliationResult:
        """
        Run custom reconciliation.

        Args:
            config: Reconciliation configuration
            result: Reconciliation result

        Returns:
            Updated reconciliation result
        """
        # Custom reconciliation requires a custom implementation
        # This is a placeholder for custom reconciliation logic
        raise NotImplementedError("Custom reconciliation is not implemented")

    async def _get_data_from_source(
        self,
        source_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Get data from a source.

        Args:
            source_config: Source configuration

        Returns:
            DataFrame with data
        """
        source_id = source_config["source_id"]
        source_type = source_config["source_type"]
        query_params = source_config.get("query_params", {})
        filters = source_config.get("filters", {})

        logger.info(f"Getting data from source {source_id} of type {source_type}")

        # Get data based on source type
        if source_type == "ohlcv":
            df = await self.historical_service.get_ohlcv_data(
                symbols=query_params.get("symbols", []),
                timeframe=query_params.get("timeframe", "1h"),
                start_timestamp=datetime.fromisoformat(query_params.get("start_timestamp", "2000-01-01T00:00:00")),
                end_timestamp=datetime.fromisoformat(query_params.get("end_timestamp", datetime.utcnow().isoformat())),
                version=query_params.get("version"),
                point_in_time=query_params.get("point_in_time"),
                include_corrections=query_params.get("include_corrections", True)
            )
        elif source_type == "tick":
            df = await self.historical_service.get_tick_data(
                symbols=query_params.get("symbols", []),
                start_timestamp=datetime.fromisoformat(query_params.get("start_timestamp", "2000-01-01T00:00:00")),
                end_timestamp=datetime.fromisoformat(query_params.get("end_timestamp", datetime.utcnow().isoformat())),
                version=query_params.get("version"),
                point_in_time=query_params.get("point_in_time"),
                include_corrections=query_params.get("include_corrections", True)
            )
        elif source_type == "alternative":
            df = await self.historical_service.get_alternative_data(
                symbols=query_params.get("symbols", []),
                data_type=query_params.get("data_type", ""),
                start_timestamp=datetime.fromisoformat(query_params.get("start_timestamp", "2000-01-01T00:00:00")),
                end_timestamp=datetime.fromisoformat(query_params.get("end_timestamp", datetime.utcnow().isoformat())),
                version=query_params.get("version"),
                point_in_time=query_params.get("point_in_time"),
                include_corrections=query_params.get("include_corrections", True)
            )
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        # Apply filters
        if filters and not df.empty:
            for field, value in filters.items():
                if field in df.columns:
                    df = df[df[field] == value]

        # Apply transformations
        if "transformations" in source_config and not df.empty:
            for transformation in source_config["transformations"]:
                df = self._apply_transformation(df, transformation)

        return df

    def _apply_transformation(
        self,
        df: pd.DataFrame,
        transformation: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply a transformation to a DataFrame.

        Args:
            df: DataFrame to transform
            transformation: Transformation configuration

        Returns:
            Transformed DataFrame
        """
        if df.empty:
            return df

        transform_type = transformation.get("type")

        if transform_type == "rename":
            # Rename columns
            columns = transformation.get("columns", {})
            df = df.rename(columns=columns)

        elif transform_type == "select":
            # Select columns
            columns = transformation.get("columns", [])
            if columns:
                df = df[columns]

        elif transform_type == "filter":
            # Filter rows
            field = transformation.get("field")
            operator = transformation.get("operator")
            value = transformation.get("value")

            if field and operator and value is not None:
                if operator == "eq":
                    df = df[df[field] == value]
                elif operator == "ne":
                    df = df[df[field] != value]
                elif operator == "gt":
                    df = df[df[field] > value]
                elif operator == "lt":
                    df = df[df[field] < value]
                elif operator == "ge":
                    df = df[df[field] >= value]
                elif operator == "le":
                    df = df[df[field] <= value]
                elif operator == "in":
                    df = df[df[field].isin(value)]
                elif operator == "not_in":
                    df = df[~df[field].isin(value)]

        elif transform_type == "add_column":
            # Add a new column
            field = transformation.get("field")
            expression = transformation.get("expression")

            if field and expression:
                # Simple expressions
                if expression == "timestamp":
                    df[field] = pd.to_datetime(df.index.get_level_values("timestamp") if "timestamp" in df.index.names else df["timestamp"])
                elif expression.startswith("column:"):
                    column = expression.split(":", 1)[1]
                    if column in df.columns:
                        df[field] = df[column]
                elif expression.startswith("constant:"):
                    value = expression.split(":", 1)[1]
                    df[field] = value

        elif transform_type == "aggregate":
            # Aggregate data
            group_by = transformation.get("group_by", [])
            aggregations = transformation.get("aggregations", {})

            if group_by and aggregations:
                df = df.groupby(group_by).agg(aggregations).reset_index()

        elif transform_type == "join":
            # Join with another DataFrame
            # This is a placeholder for joining with another DataFrame
            # In a real implementation, this would require access to the other DataFrame
            pass

        return df

    async def _reconcile_dataframes(
        self,
        config: Dict[str, Any],
        result: ReconciliationResult,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        primary_name: str,
        secondary_name: str
    ) -> ReconciliationResult:
        """
        Reconcile two DataFrames.

        Args:
            config: Reconciliation configuration
            result: Reconciliation result
            primary_df: Primary DataFrame
            secondary_df: Secondary DataFrame
            primary_name: Name of the primary source
            secondary_name: Name of the secondary source

        Returns:
            Updated reconciliation result
        """
        if primary_df.empty or secondary_df.empty:
            logger.warning("One or both DataFrames are empty")
            result.total_records = 0
            result.matched_records = 0
            result.issues = []
            result.summary = {
                "primary_records": len(primary_df),
                "secondary_records": len(secondary_df),
                "matched_records": 0,
                "issues_count": 0
            }
            return result

        # Reset index to make joining easier
        if not primary_df.index.empty:
            primary_df = primary_df.reset_index()

        if not secondary_df.index.empty:
            secondary_df = secondary_df.reset_index()

        # Get rules
        rules = config.get("rules", [])

        # Get join keys
        join_keys = []
        for rule in rules:
            if rule.get("comparison_type") == "join_key":
                join_keys.append(rule.get("field"))

        if not join_keys:
            # If no join keys are specified, try to use common columns
            common_columns = set(primary_df.columns) & set(secondary_df.columns)
            if "symbol" in common_columns and "timestamp" in common_columns:
                join_keys = ["symbol", "timestamp"]
            elif "timestamp" in common_columns:
                join_keys = ["timestamp"]
            else:
                # Use the first common column as a fallback
                join_keys = list(common_columns)[:1]

        if not join_keys:
            logger.error("No join keys found")
            result.total_records = 0
            result.matched_records = 0
            result.issues = []
            result.summary = {
                "primary_records": len(primary_df),
                "secondary_records": len(secondary_df),
                "issues_count": 0,
                "error": "No join keys found"
            }
            return result

        # Merge DataFrames
        merged_df = pd.merge(
            primary_df,
            secondary_df,
            on=join_keys,
            how="outer",
            suffixes=(f"_{primary_name}", f"_{secondary_name}")
        )

        # Count records
        total_records = len(merged_df)
        matched_records = len(merged_df.dropna(subset=[f"{col}_{primary_name}" for col in primary_df.columns if col not in join_keys] + [f"{col}_{secondary_name}" for col in secondary_df.columns if col not in join_keys]))

        # Apply rules
        issues = []

        for rule in rules:
            if rule.get("comparison_type") == "join_key":
                # Skip join keys
                continue

            field = rule.get("field")
            comparison_type = rule.get("comparison_type")
            parameters = rule.get("parameters", {})
            severity = ReconciliationSeverity(rule.get("severity", ReconciliationSeverity.ERROR.value))

            # Check if field exists in both DataFrames
            primary_field = f"{field}_{primary_name}"
            secondary_field = f"{field}_{secondary_name}"

            if primary_field not in merged_df.columns or secondary_field not in merged_df.columns:
                logger.warning(f"Field {field} not found in both DataFrames")
                continue

            # Apply comparison
            if comparison_type == "exact":
                # Exact match
                mask = merged_df[primary_field] != merged_df[secondary_field]
                mask = mask & ~(merged_df[primary_field].isna() & merged_df[secondary_field].isna())

                for idx in merged_df[mask].index:
                    row = merged_df.loc[idx]

                    # Create issue
                    issue = ReconciliationIssue(
                        rule_id=rule.get("rule_id"),
                        field=field,
                        primary_value=row[primary_field],
                        secondary_value=row[secondary_field],
                        severity=severity,
                        description=f"Values do not match: {row[primary_field]} != {row[secondary_field]}"
                    )

                    issues.append(issue)

            elif comparison_type == "tolerance":
                # Tolerance match
                tolerance = parameters.get("tolerance", 0.0001)

                # Convert to numeric
                primary_values = pd.to_numeric(merged_df[primary_field], errors="coerce")
                secondary_values = pd.to_numeric(merged_df[secondary_field], errors="coerce")

                # Calculate difference
                diff = (primary_values - secondary_values).abs()

                # Check if difference exceeds tolerance
                mask = diff > tolerance
                mask = mask & ~(primary_values.isna() & secondary_values.isna())

                for idx in merged_df[mask].index:
                    row = merged_df.loc[idx]

                    # Create issue
                    issue = ReconciliationIssue(
                        rule_id=rule.get("rule_id"),
                        field=field,
                        primary_value=row[primary_field],
                        secondary_value=row[secondary_field],
                        difference=diff[idx],
                        severity=severity,
                        description=f"Difference exceeds tolerance: {diff[idx]} > {tolerance}"
                    )

                    issues.append(issue)

            elif comparison_type == "custom":
                # Custom comparison
                # This is a placeholder for custom comparison logic
                pass

        # Update result
        result.total_records = total_records
        result.matched_records = matched_records
        result.issues = issues
        result.summary = {
            "primary_records": len(primary_df),
            "secondary_records": len(secondary_df),
            "matched_records": matched_records,
            "issues_count": len(issues),
            "issues_by_severity": {
                severity.value: len([i for i in issues if i.severity == severity])
                for severity in ReconciliationSeverity
            },
            "issues_by_field": {
                field: len([i for i in issues if i.field == field])
                for field in set(i.field for i in issues)
            }
        }

        return result

    async def get_config(self, config_id: str) -> Optional[ReconciliationConfig]:
        """
        Get a reconciliation configuration.

        Args:
            config_id: Configuration ID

        Returns:
            Reconciliation configuration or None if not found
        """
        config_dict = await self.repository.get_config(config_id)

        if config_dict is None:
            return None

        return ReconciliationConfig(**config_dict)

    async def get_task(self, task_id: str) -> Optional[ReconciliationTask]:
        """
        Get a reconciliation task.

        Args:
            task_id: Task ID

        Returns:
            Reconciliation task or None if not found
        """
        task_dict = await self.repository.get_task(task_id)

        if task_dict is None:
            return None

        return ReconciliationTask(**task_dict)

    async def get_result(self, result_id: str) -> Optional[ReconciliationResult]:
        """
        Get a reconciliation result.

        Args:
            result_id: Result ID

        Returns:
            Reconciliation result or None if not found
        """
        result_dict = await self.repository.get_result(result_id)

        if result_dict is None:
            return None

        # Convert issues
        issues = []
        for issue_dict in result_dict.pop("issues", []):
            issues.append(ReconciliationIssue(**issue_dict))

        result = ReconciliationResult(**result_dict)
        result.issues = issues

        return result

    async def get_configs(
        self,
        enabled: Optional[bool] = None,
        reconciliation_type: Optional[ReconciliationType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ReconciliationConfig]:
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
        config_dicts = await self.repository.get_configs(
            enabled=enabled,
            reconciliation_type=reconciliation_type,
            limit=limit,
            offset=offset
        )

        return [ReconciliationConfig(**config_dict) for config_dict in config_dicts]

    async def get_tasks(
        self,
        config_id: Optional[str] = None,
        status: Optional[ReconciliationStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ReconciliationTask]:
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
        task_dicts = await self.repository.get_tasks(
            config_id=config_id,
            status=status,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )

        return [ReconciliationTask(**task_dict) for task_dict in task_dicts]

    async def get_results(
        self,
        config_id: Optional[str] = None,
        status: Optional[ReconciliationStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ReconciliationResult]:
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
        result_dicts = await self.repository.get_results(
            config_id=config_id,
            status=status,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )

        results = []
        for result_dict in result_dicts:
            # Convert issues
            issues = []
            for issue_dict in result_dict.pop("issues", []):
                issues.append(ReconciliationIssue(**issue_dict))

            result = ReconciliationResult(**result_dict)
            result.issues = issues

            results.append(result)

        return results