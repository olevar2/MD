"""
Advanced Rule Engine for Data Reconciliation.

This module provides a flexible rule engine for data reconciliation.
It supports complex validation rules beyond simple comparisons.
"""

import logging
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from models.models_2 import (
    ReconciliationIssue,
    ReconciliationSeverity
)

logger = logging.getLogger(__name__)


class RuleContext:
    """Context for rule evaluation."""
    
    def __init__(
        self,
        primary_data: pd.DataFrame,
        secondary_data: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the context.
        
        Args:
            primary_data: Primary data
            secondary_data: Secondary data
            metadata: Additional metadata
        """
        self.primary_data = primary_data
        self.secondary_data = secondary_data
        self.metadata = metadata or {}
        self.issues = []
    
    def add_issue(
        self,
        field: str,
        description: str,
        severity: ReconciliationSeverity,
        primary_value: Any = None,
        secondary_value: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an issue to the context.
        
        Args:
            field: Field name
            description: Issue description
            severity: Issue severity
            primary_value: Value from primary data
            secondary_value: Value from secondary data
            metadata: Additional metadata
        """
        issue = ReconciliationIssue(
            issue_id=f"ISSUE-{len(self.issues) + 1}",
            field=field,
            description=description,
            severity=severity,
            primary_value=primary_value,
            secondary_value=secondary_value,
            metadata=metadata or {}
        )
        
        self.issues.append(issue)


class Rule:
    """Base class for reconciliation rules."""
    
    def __init__(
        self,
        name: str,
        field: str,
        severity: ReconciliationSeverity,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the rule.
        
        Args:
            name: Rule name
            field: Field to check
            severity: Issue severity
            parameters: Rule parameters
        """
        self.name = name
        self.field = field
        self.severity = severity
        self.parameters = parameters or {}
    
    def evaluate(self, context: RuleContext) -> None:
        """
        Evaluate the rule.
        
        Args:
            context: Rule context
        """
        raise NotImplementedError("Subclasses must implement evaluate()")


class ToleranceRule(Rule):
    """Rule for checking numerical values within a tolerance."""
    
    def evaluate(self, context: RuleContext) -> None:
        """
        Evaluate the rule.
        
        Args:
            context: Rule context
        """
        if context.secondary_data is None:
            logger.warning(f"ToleranceRule requires secondary data: {self.name}")
            return
        
        # Get tolerance
        tolerance = self.parameters.get("tolerance", 0.0001)
        relative = self.parameters.get("relative", True)
        
        # Join data on index
        joined = context.primary_data.join(
            context.secondary_data[self.field].rename(f"{self.field}_secondary"),
            how="inner"
        )
        
        # Check tolerance
        for idx, row in joined.iterrows():
            primary_value = row[self.field]
            secondary_value = row[f"{self.field}_secondary"]
            
            # Skip if either value is null
            if pd.isna(primary_value) or pd.isna(secondary_value):
                continue
            
            # Calculate difference
            if relative and secondary_value != 0:
                diff = abs((primary_value - secondary_value) / secondary_value)
            else:
                diff = abs(primary_value - secondary_value)
            
            # Check if difference exceeds tolerance
            if diff > tolerance:
                context.add_issue(
                    field=self.field,
                    description=f"Value {primary_value} differs from {secondary_value} by {diff:.6f} (tolerance: {tolerance})",
                    severity=self.severity,
                    primary_value=primary_value,
                    secondary_value=secondary_value,
                    metadata={"difference": diff, "tolerance": tolerance}
                )


class RangeRule(Rule):
    """Rule for checking values within a range."""
    
    def evaluate(self, context: RuleContext) -> None:
        """
        Evaluate the rule.
        
        Args:
            context: Rule context
        """
        # Get range parameters
        min_value = self.parameters.get("min_value")
        max_value = self.parameters.get("max_value")
        
        # Check range
        for idx, row in context.primary_data.iterrows():
            value = row[self.field]
            
            # Skip if value is null
            if pd.isna(value):
                continue
            
            # Check minimum
            if min_value is not None and value < min_value:
                context.add_issue(
                    field=self.field,
                    description=f"Value {value} is below minimum {min_value}",
                    severity=self.severity,
                    primary_value=value,
                    metadata={"min_value": min_value, "max_value": max_value}
                )
            
            # Check maximum
            if max_value is not None and value > max_value:
                context.add_issue(
                    field=self.field,
                    description=f"Value {value} is above maximum {max_value}",
                    severity=self.severity,
                    primary_value=value,
                    metadata={"min_value": min_value, "max_value": max_value}
                )


class NullRule(Rule):
    """Rule for checking null values."""
    
    def evaluate(self, context: RuleContext) -> None:
        """
        Evaluate the rule.
        
        Args:
            context: Rule context
        """
        # Get parameters
        allow_null = self.parameters.get("allow_null", False)
        
        # Check null values
        null_count = context.primary_data[self.field].isna().sum()
        
        if not allow_null and null_count > 0:
            context.add_issue(
                field=self.field,
                description=f"Found {null_count} null values (null not allowed)",
                severity=self.severity,
                metadata={"null_count": null_count}
            )


class UniqueRule(Rule):
    """Rule for checking unique values."""
    
    def evaluate(self, context: RuleContext) -> None:
        """
        Evaluate the rule.
        
        Args:
            context: Rule context
        """
        # Check uniqueness
        duplicate_count = context.primary_data[self.field].duplicated().sum()
        
        if duplicate_count > 0:
            context.add_issue(
                field=self.field,
                description=f"Found {duplicate_count} duplicate values (must be unique)",
                severity=self.severity,
                metadata={"duplicate_count": duplicate_count}
            )


class PatternRule(Rule):
    """Rule for checking values against a pattern."""
    
    def evaluate(self, context: RuleContext) -> None:
        """
        Evaluate the rule.
        
        Args:
            context: Rule context
        """
        # Get pattern
        pattern = self.parameters.get("pattern")
        
        if not pattern:
            logger.warning(f"PatternRule requires a pattern: {self.name}")
            return
        
        # Compile regex
        regex = re.compile(pattern)
        
        # Check pattern
        for idx, row in context.primary_data.iterrows():
            value = row[self.field]
            
            # Skip if value is null
            if pd.isna(value):
                continue
            
            # Convert to string
            value_str = str(value)
            
            # Check pattern
            if not regex.match(value_str):
                context.add_issue(
                    field=self.field,
                    description=f"Value '{value_str}' does not match pattern '{pattern}'",
                    severity=self.severity,
                    primary_value=value,
                    metadata={"pattern": pattern}
                )


class CrossFieldRule(Rule):
    """Rule for checking relationships between fields."""
    
    def evaluate(self, context: RuleContext) -> None:
        """
        Evaluate the rule.
        
        Args:
            context: Rule context
        """
        # Get parameters
        other_field = self.parameters.get("other_field")
        relation = self.parameters.get("relation", "equal")
        
        if not other_field:
            logger.warning(f"CrossFieldRule requires other_field: {self.name}")
            return
        
        # Check relation
        for idx, row in context.primary_data.iterrows():
            value = row[self.field]
            other_value = row[other_field]
            
            # Skip if either value is null
            if pd.isna(value) or pd.isna(other_value):
                continue
            
            # Check relation
            if relation == "equal" and value != other_value:
                context.add_issue(
                    field=self.field,
                    description=f"Value {value} is not equal to {other_field} value {other_value}",
                    severity=self.severity,
                    primary_value=value,
                    secondary_value=other_value,
                    metadata={"other_field": other_field, "relation": relation}
                )
            elif relation == "greater_than" and value <= other_value:
                context.add_issue(
                    field=self.field,
                    description=f"Value {value} is not greater than {other_field} value {other_value}",
                    severity=self.severity,
                    primary_value=value,
                    secondary_value=other_value,
                    metadata={"other_field": other_field, "relation": relation}
                )
            elif relation == "less_than" and value >= other_value:
                context.add_issue(
                    field=self.field,
                    description=f"Value {value} is not less than {other_field} value {other_value}",
                    severity=self.severity,
                    primary_value=value,
                    secondary_value=other_value,
                    metadata={"other_field": other_field, "relation": relation}
                )


class DistributionRule(Rule):
    """Rule for checking data distributions."""
    
    def evaluate(self, context: RuleContext) -> None:
        """
        Evaluate the rule.
        
        Args:
            context: Rule context
        """
        if context.secondary_data is None:
            logger.warning(f"DistributionRule requires secondary data: {self.name}")
            return
        
        # Get parameters
        method = self.parameters.get("method", "ks_test")
        threshold = self.parameters.get("threshold", 0.05)
        
        # Get data
        primary_values = context.primary_data[self.field].dropna()
        secondary_values = context.secondary_data[self.field].dropna()
        
        # Skip if not enough data
        if len(primary_values) < 10 or len(secondary_values) < 10:
            logger.warning(f"Not enough data for distribution test: {self.name}")
            return
        
        # Perform test
        if method == "ks_test":
            from scipy import stats
            statistic, p_value = stats.ks_2samp(primary_values, secondary_values)
            
            if p_value < threshold:
                context.add_issue(
                    field=self.field,
                    description=f"Distribution differs significantly (KS test p-value: {p_value:.6f}, threshold: {threshold})",
                    severity=self.severity,
                    metadata={"method": method, "p_value": p_value, "statistic": statistic, "threshold": threshold}
                )
        else:
            logger.warning(f"Unsupported distribution test method: {method}")


class CustomRule(Rule):
    """Rule for custom validation logic."""
    
    def evaluate(self, context: RuleContext) -> None:
        """
        Evaluate the rule.
        
        Args:
            context: Rule context
        """
        # Get custom function
        custom_func = self.parameters.get("function")
        
        if not custom_func or not callable(custom_func):
            logger.warning(f"CustomRule requires a callable function: {self.name}")
            return
        
        # Call custom function
        try:
            custom_func(self, context)
        except Exception as e:
            logger.error(f"Error in custom rule function: {e}")
            context.add_issue(
                field=self.field,
                description=f"Error in custom rule: {str(e)}",
                severity=self.severity,
                metadata={"error": str(e)}
            )


class RuleFactory:
    """Factory for creating rules."""
    
    @staticmethod
    def create_rule(
        name: str,
        field: str,
        comparison_type: str,
        parameters: Dict[str, Any],
        severity: ReconciliationSeverity
    ) -> Rule:
        """
        Create a rule.
        
        Args:
            name: Rule name
            field: Field to check
            comparison_type: Type of comparison
            parameters: Rule parameters
            severity: Issue severity
            
        Returns:
            Rule instance
        """
        if comparison_type == "tolerance":
            return ToleranceRule(name, field, severity, parameters)
        elif comparison_type == "range":
            return RangeRule(name, field, severity, parameters)
        elif comparison_type == "null":
            return NullRule(name, field, severity, parameters)
        elif comparison_type == "unique":
            return UniqueRule(name, field, severity, parameters)
        elif comparison_type == "pattern":
            return PatternRule(name, field, severity, parameters)
        elif comparison_type == "cross_field":
            return CrossFieldRule(name, field, severity, parameters)
        elif comparison_type == "distribution":
            return DistributionRule(name, field, severity, parameters)
        elif comparison_type == "custom":
            return CustomRule(name, field, severity, parameters)
        else:
            logger.warning(f"Unsupported comparison type: {comparison_type}")
            return Rule(name, field, severity, parameters)
