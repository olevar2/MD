"""
Indicator Registry Module.

This module provides a registry for managing technical indicators.
"""

from typing import Dict, List, Optional, Type, Any

from core_foundations.utils.logger import get_logger
from feature_store_service.indicators.base_indicator import BaseIndicator

logger = get_logger("feature-store-service.indicator-registry")


class IndicatorRegistry:
    """
    Registry for managing technical indicators.
    
    This class keeps track of all available indicators and provides methods
    to access them by ID, category, etc.
    """
    
    def __init__(self):
        """
        Initialize an empty indicator registry.
        """
        # Dictionary mapping indicator IDs to indicator classes
        self._indicators: Dict[str, Type[BaseIndicator]] = {}
        
    def register_indicator(self, indicator_class: Type[BaseIndicator]) -> None:
        """
        Register an indicator with the registry.
        
        Args:
            indicator_class: Class derived from BaseIndicator to register
        """
        indicator_id = indicator_class.__name__
        
        if indicator_id in self._indicators:
            logger.warning(f"Indicator '{indicator_id}' already registered, overwriting")
            
        self._indicators[indicator_id] = indicator_class
        logger.debug(f"Registered indicator '{indicator_id}'")
        
    def register_indicators(self, indicator_classes: List[Type[BaseIndicator]]) -> None:
        """
        Register multiple indicators at once.
        
        Args:
            indicator_classes: List of indicator classes to register
        """
        for indicator_class in indicator_classes:
            self.register_indicator(indicator_class)
            
    def get_indicator(self, indicator_id: str) -> Optional[Type[BaseIndicator]]:
        """
        Get an indicator by its ID.
        
        Args:
            indicator_id: ID of the indicator to get
            
        Returns:
            Indicator class, or None if not found
        """
        return self._indicators.get(indicator_id)
        
    def get_all_indicators(self) -> Dict[str, Type[BaseIndicator]]:
        """
        Get all registered indicators.
        
        Returns:
            Dictionary mapping indicator IDs to indicator classes
        """
        return self._indicators.copy()
        
    def get_indicators_by_category(self, category: str) -> Dict[str, Type[BaseIndicator]]:
        """
        Get all indicators in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            Dictionary mapping indicator IDs to indicator classes in the category
        """
        return {
            indicator_id: indicator_class
            for indicator_id, indicator_class in self._indicators.items()
            if indicator_class.category == category
        }
        
    def get_categories(self) -> List[str]:
        """
        Get a list of all indicator categories.
        
        Returns:
            List of category names
        """
        categories = set()
        for indicator_class in self._indicators.values():
            categories.add(indicator_class.category)
        return sorted(list(categories))
        
    def create_indicator(self, indicator_id: str, **kwargs) -> Optional[BaseIndicator]:
        """
        Create an indicator instance by ID with the specified parameters.
        
        Args:
            indicator_id: ID of the indicator to create
            **kwargs: Parameters to pass to the indicator constructor
            
        Returns:
            Indicator instance, or None if the indicator ID is not found
        """
        indicator_class = self.get_indicator(indicator_id)
        if indicator_class:
            try:
                return indicator_class(**kwargs)
            except Exception as e:
                logger.error(f"Error creating indicator '{indicator_id}': {str(e)}")
        return None
        
    def validate_parameters(self, indicator_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize parameters for an indicator.
        
        This checks that the parameters are valid for the indicator and applies
        defaults for any missing parameters.
        
        Args:
            indicator_id: ID of the indicator
            params: Parameters to validate
            
        Returns:
            Validated and normalized parameters
            
        Raises:
            ValueError: If any parameter is invalid
        """
        indicator_class = self.get_indicator(indicator_id)
        if not indicator_class:
            raise ValueError(f"Indicator '{indicator_id}' not found")
            
        result = {}
        
        # Get parameter specifications
        param_specs = indicator_class.params
        
        # Apply defaults for missing parameters
        for param_name, param_spec in param_specs.items():
            if param_name in params:
                # Parameter is provided, validate it
                value = params[param_name]
                
                # Check type
                param_type = param_spec.get("type")
                if param_type == "int" and not isinstance(value, int):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        raise ValueError(f"Parameter '{param_name}' must be an integer")
                elif param_type == "float" and not isinstance(value, float):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        raise ValueError(f"Parameter '{param_name}' must be a float")
                elif param_type == "str" and not isinstance(value, str):
                    value = str(value)
                    
                # Check min/max constraints
                if param_type in ("int", "float"):
                    min_value = param_spec.get("min")
                    if min_value is not None and value < min_value:
                        raise ValueError(f"Parameter '{param_name}' must be at least {min_value}")
                        
                    max_value = param_spec.get("max")
                    if max_value is not None and value > max_value:
                        raise ValueError(f"Parameter '{param_name}' must be at most {max_value}")
                        
                # Check options constraint
                options = param_spec.get("options")
                if options and value not in options:
                    raise ValueError(f"Parameter '{param_name}' must be one of: {options}")
                    
                result[param_name] = value
            else:
                # Parameter is missing, use default
                default = param_spec.get("default")
                if default is not None:
                    result[param_name] = default
                    
        return result