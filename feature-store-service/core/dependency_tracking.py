"""
Dependency Tracking Module.

This module tracks dependencies between indicators and analysis components
to prevent redundant calculations and ensure proper update ordering.
"""
from typing import Dict, Set, List, Optional
from datetime import datetime
import logging
import json
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DependencyTracker:
    """
    Tracks dependencies between indicators and analysis components.
    Helps prevent redundant calculations and ensures proper update ordering.
    """

    def __init__(self):
        """Initialize dependency tracker"""
        self._dependencies: Dict[str, Set[str]] = {}
        self._reverse_deps: Dict[str, Set[str]] = {}
        self._last_updated: Dict[str, datetime] = {}
        self._calculation_order: List[str] = []

    def add_dependency(self, dependent: str, prerequisite: str) ->None:
        """
        Add a dependency relationship between indicators
        
        Args:
            dependent: The indicator that depends on another
            prerequisite: The indicator that must be calculated first
        """
        if dependent not in self._dependencies:
            self._dependencies[dependent] = set()
        self._dependencies[dependent].add(prerequisite)
        if prerequisite not in self._reverse_deps:
            self._reverse_deps[prerequisite] = set()
        self._reverse_deps[prerequisite].add(dependent)
        self._update_calculation_order()

    def get_prerequisites(self, indicator: str) ->Set[str]:
        """
        Get all prerequisites for an indicator
        
        Args:
            indicator: The indicator to check
            
        Returns:
            Set of prerequisite indicators
        """
        return self._dependencies.get(indicator, set())

    def get_dependents(self, indicator: str) ->Set[str]:
        """
        Get all indicators that depend on this one
        
        Args:
            indicator: The indicator to check
            
        Returns:
            Set of dependent indicators
        """
        return self._reverse_deps.get(indicator, set())

    @with_exception_handling
    def _update_calculation_order(self) ->None:
        """Update the topological sort of indicators for calculation order"""
        visited = set()
        temp = set()
        order = []

        def visit(indicator: str) ->None:
    """
    Visit.
    
    Args:
        indicator: Description of indicator
    
    """

            if indicator in temp:
                raise ValueError(
                    f'Circular dependency detected involving {indicator}')
            if indicator in visited:
                return
            temp.add(indicator)
            for prereq in self._dependencies.get(indicator, set()):
                visit(prereq)
            temp.remove(indicator)
            visited.add(indicator)
            order.append(indicator)
        try:
            for indicator in self._dependencies:
                if indicator not in visited:
                    visit(indicator)
            self._calculation_order = order
        except ValueError as e:
            logger.error(f'Error updating calculation order: {e}')
            raise

    def get_calculation_order(self) ->List[str]:
        """
        Get the correct order for calculating indicators
        
        Returns:
            List of indicators in calculation order
        """
        return self._calculation_order

    def record_update(self, indicator: str) ->None:
        """
        Record that an indicator has been updated
        
        Args:
            indicator: The indicator that was updated
        """
        self._last_updated[indicator] = datetime.now()

    def get_last_updated(self, indicator: str) ->Optional[datetime]:
        """
        Get when an indicator was last updated
        
        Args:
            indicator: The indicator to check
            
        Returns:
            Datetime of last update or None if never updated
        """
        return self._last_updated.get(indicator)

    def export_dependencies(self) ->Dict:
        """
        Export dependency information for monitoring/debugging
        
        Returns:
            Dict containing dependency information
        """
        return {'dependencies': {k: list(v) for k, v in self._dependencies.
            items()}, 'reverse_deps': {k: list(v) for k, v in self.
            _reverse_deps.items()}, 'calculation_order': self.
            _calculation_order, 'last_updated': {k: v.isoformat() for k, v in
            self._last_updated.items()}}
