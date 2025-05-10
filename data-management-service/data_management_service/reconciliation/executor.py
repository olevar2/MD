"""
Data Reconciliation Executor.

This module provides the executor for running reconciliation tasks.
It handles data loading, transformation, comparison, and result generation.
"""

import logging
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

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
from data_management_service.reconciliation.rule_engine import (
    RuleContext,
    RuleFactory
)
from data_management_service.historical.service import HistoricalDataService

logger = logging.getLogger(__name__)


class ReconciliationExecutor:
    """Executor for reconciliation tasks."""
    
    def __init__(self, historical_service: HistoricalDataService):
        """
        Initialize the executor.
        
        Args:
            historical_service: Historical data service
        """
        self.historical_service = historical_service
    
    async def execute_task(
        self,
        task: ReconciliationTask,
        config: ReconciliationConfig
    ) -> ReconciliationResult:
        """
        Execute a reconciliation task.
        
        Args:
            task: Reconciliation task
            config: Reconciliation configuration
            
        Returns:
            Reconciliation result
        """
        logger.info(f"Executing reconciliation task: {task.task_id}")
        
        try:
            # Create result
            result = ReconciliationResult(
                result_id=str(uuid.uuid4()),
                task_id=task.task_id,
                config_id=config.config_id,
                status=ReconciliationStatus.RUNNING,
                start_time=datetime.utcnow(),
                end_time=None,
                total_records=0,
                matched_records=0,
                issues=[],
                summary={},
                metadata=task.metadata.copy() if task.metadata else {}
            )
            
            # Load data
            primary_data, secondary_data = await self._load_data(config)
            
            # Execute reconciliation based on type
            if config.reconciliation_type == ReconciliationType.CROSS_SOURCE:
                result = await self._execute_cross_source(
                    result=result,
                    config=config,
                    primary_data=primary_data,
                    secondary_data=secondary_data
                )
            elif config.reconciliation_type == ReconciliationType.TEMPORAL:
                result = await self._execute_temporal(
                    result=result,
                    config=config,
                    primary_data=primary_data
                )
            elif config.reconciliation_type == ReconciliationType.DERIVED:
                result = await self._execute_derived(
                    result=result,
                    config=config,
                    primary_data=primary_data,
                    secondary_data=secondary_data
                )
            elif config.reconciliation_type == ReconciliationType.CUSTOM:
                result = await self._execute_custom(
                    result=result,
                    config=config,
                    primary_data=primary_data,
                    secondary_data=secondary_data
                )
            else:
                raise ValueError(f"Unsupported reconciliation type: {config.reconciliation_type}")
            
            # Update result
            result.status = ReconciliationStatus.COMPLETED
            result.end_time = datetime.utcnow()
            
            # Generate summary
            result.summary = self._generate_summary(result)
            
            logger.info(f"Completed reconciliation task: {task.task_id}")
            
            return result
        except Exception as e:
            logger.error(f"Failed to execute reconciliation task: {e}")
            
            # Create error result
            result = ReconciliationResult(
                result_id=str(uuid.uuid4()),
                task_id=task.task_id,
                config_id=config.config_id,
                status=ReconciliationStatus.FAILED,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                total_records=0,
                matched_records=0,
                issues=[],
                summary={"error": str(e)},
                metadata=task.metadata.copy() if task.metadata else {}
            )
            
            return result
    
    async def _load_data(
        self,
        config: ReconciliationConfig
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load data for reconciliation.
        
        Args:
            config: Reconciliation configuration
            
        Returns:
            Tuple of primary and secondary data
        """
        # Load primary data
        primary_data = await self._load_source_data(config.primary_source)
        
        # Load secondary data if available
        secondary_data = None
        if config.secondary_source:
            secondary_data = await self._load_source_data(config.secondary_source)
        
        return primary_data, secondary_data
    
    async def _load_source_data(
        self,
        source: DataSourceConfig
    ) -> pd.DataFrame:
        """
        Load data from a source.
        
        Args:
            source: Data source configuration
            
        Returns:
            DataFrame with source data
        """
        # Get data based on source type
        if source.source_type == "ohlcv":
            data = await self.historical_service.get_ohlcv_data(
                symbols=source.query_params.get("symbols", []),
                timeframe=source.query_params.get("timeframe", "1h"),
                start_timestamp=source.query_params.get("start_timestamp"),
                end_timestamp=source.query_params.get("end_timestamp")
            )
        elif source.source_type == "tick":
            data = await self.historical_service.get_tick_data(
                symbols=source.query_params.get("symbols", []),
                start_timestamp=source.query_params.get("start_timestamp"),
                end_timestamp=source.query_params.get("end_timestamp")
            )
        elif source.source_type == "alternative":
            data = await self.historical_service.get_alternative_data(
                data_type=source.query_params.get("data_type", ""),
                symbols=source.query_params.get("symbols", []),
                start_timestamp=source.query_params.get("start_timestamp"),
                end_timestamp=source.query_params.get("end_timestamp")
            )
        else:
            raise ValueError(f"Unsupported source type: {source.source_type}")
        
        # Apply transformations if any
        if source.transformations:
            data = self._apply_transformations(data, source.transformations)
        
        return data
    
    def _apply_transformations(
        self,
        data: pd.DataFrame,
        transformations: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Apply transformations to data.
        
        Args:
            data: Data to transform
            transformations: List of transformations
            
        Returns:
            Transformed data
        """
        for transformation in transformations:
            transformation_type = transformation.get("type")
            
            if transformation_type == "add_column":
                field = transformation.get("field")
                expression = transformation.get("expression")
                
                if field and expression:
                    # Simple column reference
                    if expression.startswith("column:"):
                        column_name = expression.split(":", 1)[1]
                        data[field] = data[column_name]
                    # Python expression
                    else:
                        data[field] = data.eval(expression)
            
            elif transformation_type == "filter":
                condition = transformation.get("condition")
                
                if condition:
                    data = data.query(condition)
            
            elif transformation_type == "aggregate":
                group_by = transformation.get("group_by", [])
                aggregations = transformation.get("aggregations", {})
                
                if group_by and aggregations:
                    agg_dict = {}
                    
                    for field, agg in aggregations.items():
                        column = agg.get("column")
                        function = agg.get("function")
                        
                        if column and function:
                            agg_dict[field] = (column, function)
                    
                    data = data.groupby(group_by).agg(**agg_dict).reset_index()
        
        return data
    
    async def _execute_cross_source(
        self,
        result: ReconciliationResult,
        config: ReconciliationConfig,
        primary_data: pd.DataFrame,
        secondary_data: Optional[pd.DataFrame]
    ) -> ReconciliationResult:
        """
        Execute cross-source reconciliation.
        
        Args:
            result: Reconciliation result
            config: Reconciliation configuration
            primary_data: Primary data
            secondary_data: Secondary data
            
        Returns:
            Updated reconciliation result
        """
        if secondary_data is None:
            raise ValueError("Secondary data is required for cross-source reconciliation")
        
        # Create rule context
        context = RuleContext(
            primary_data=primary_data,
            secondary_data=secondary_data,
            metadata=config.metadata
        )
        
        # Create and evaluate rules
        for rule_config in config.rules:
            rule = RuleFactory.create_rule(
                name=rule_config.name,
                field=rule_config.field,
                comparison_type=rule_config.comparison_type,
                parameters=rule_config.parameters,
                severity=rule_config.severity
            )
            
            rule.evaluate(context)
        
        # Update result
        result.total_records = len(primary_data)
        result.matched_records = result.total_records - len(context.issues)
        result.issues = context.issues
        
        return result
    
    async def _execute_temporal(
        self,
        result: ReconciliationResult,
        config: ReconciliationConfig,
        primary_data: pd.DataFrame
    ) -> ReconciliationResult:
        """
        Execute temporal reconciliation.
        
        Args:
            result: Reconciliation result
            config: Reconciliation configuration
            primary_data: Primary data
            
        Returns:
            Updated reconciliation result
        """
        # Get historical data
        historical_data = await self.historical_service.get_historical_data(
            data_type=config.primary_source.source_type,
            symbols=config.primary_source.query_params.get("symbols", []),
            timeframe=config.primary_source.query_params.get("timeframe") if config.primary_source.source_type == "ohlcv" else None,
            start_timestamp=None,  # Get from historical repository
            end_timestamp=None,  # Get from historical repository
            limit=1000  # Limit to recent data
        )
        
        # Create rule context
        context = RuleContext(
            primary_data=primary_data,
            secondary_data=historical_data,
            metadata=config.metadata
        )
        
        # Create and evaluate rules
        for rule_config in config.rules:
            rule = RuleFactory.create_rule(
                name=rule_config.name,
                field=rule_config.field,
                comparison_type=rule_config.comparison_type,
                parameters=rule_config.parameters,
                severity=rule_config.severity
            )
            
            rule.evaluate(context)
        
        # Update result
        result.total_records = len(primary_data)
        result.matched_records = result.total_records - len(context.issues)
        result.issues = context.issues
        
        return result
    
    async def _execute_derived(
        self,
        result: ReconciliationResult,
        config: ReconciliationConfig,
        primary_data: pd.DataFrame,
        secondary_data: Optional[pd.DataFrame]
    ) -> ReconciliationResult:
        """
        Execute derived data reconciliation.
        
        Args:
            result: Reconciliation result
            config: Reconciliation configuration
            primary_data: Primary data
            secondary_data: Secondary data
            
        Returns:
            Updated reconciliation result
        """
        if secondary_data is None:
            raise ValueError("Secondary data is required for derived reconciliation")
        
        # Create rule context
        context = RuleContext(
            primary_data=primary_data,
            secondary_data=secondary_data,
            metadata=config.metadata
        )
        
        # Create and evaluate rules
        for rule_config in config.rules:
            rule = RuleFactory.create_rule(
                name=rule_config.name,
                field=rule_config.field,
                comparison_type=rule_config.comparison_type,
                parameters=rule_config.parameters,
                severity=rule_config.severity
            )
            
            rule.evaluate(context)
        
        # Update result
        result.total_records = len(primary_data)
        result.matched_records = result.total_records - len(context.issues)
        result.issues = context.issues
        
        return result
    
    async def _execute_custom(
        self,
        result: ReconciliationResult,
        config: ReconciliationConfig,
        primary_data: pd.DataFrame,
        secondary_data: Optional[pd.DataFrame]
    ) -> ReconciliationResult:
        """
        Execute custom reconciliation.
        
        Args:
            result: Reconciliation result
            config: Reconciliation configuration
            primary_data: Primary data
            secondary_data: Secondary data
            
        Returns:
            Updated reconciliation result
        """
        # Create rule context
        context = RuleContext(
            primary_data=primary_data,
            secondary_data=secondary_data,
            metadata=config.metadata
        )
        
        # Create and evaluate rules
        for rule_config in config.rules:
            rule = RuleFactory.create_rule(
                name=rule_config.name,
                field=rule_config.field,
                comparison_type=rule_config.comparison_type,
                parameters=rule_config.parameters,
                severity=rule_config.severity
            )
            
            rule.evaluate(context)
        
        # Update result
        result.total_records = len(primary_data)
        result.matched_records = result.total_records - len(context.issues)
        result.issues = context.issues
        
        return result
    
    def _generate_summary(self, result: ReconciliationResult) -> Dict[str, Any]:
        """
        Generate summary for a reconciliation result.
        
        Args:
            result: Reconciliation result
            
        Returns:
            Summary dictionary
        """
        # Count issues by severity
        issues_by_severity = {}
        for issue in result.issues:
            severity = issue.severity
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
        
        # Count issues by field
        issues_by_field = {}
        for issue in result.issues:
            field = issue.field
            issues_by_field[field] = issues_by_field.get(field, 0) + 1
        
        # Create summary
        summary = {
            "total_records": result.total_records,
            "matched_records": result.matched_records,
            "match_percentage": result.matched_records / result.total_records * 100 if result.total_records > 0 else 0,
            "total_issues": len(result.issues),
            "issues_by_severity": issues_by_severity,
            "issues_by_field": issues_by_field
        }
        
        return summary
