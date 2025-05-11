"""
Advanced Indicators Registry.

This module registers all advanced indicators implemented for Phase 5
and provides a central access point for indicator discovery and usage.
"""

from typing import Dict, List, Type, Any, Optional

from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.indicators.advanced.fourier_analysis import FourierAnalysisIndicator
from feature_store_service.indicators.advanced.seasonal_analysis import SeasonalAnalysisIndicator
from feature_store_service.indicators.advanced.enhanced_obv import EnhancedOBVIndicator
from feature_store_service.indicators.advanced.volume_profile import VolumeProfileIndicator
from feature_store_service.indicators.advanced.volume_zone_oscillator import AdvancedVZOIndicator


class AdvancedIndicatorRegistry:
    """
    Registry for advanced technical indicators implemented in Phase 5.
    
    This class provides a central access point for discovering and creating
    advanced indicator instances.
    """
    
    # Dictionary mapping indicator names to their classes
    _indicators: Dict[str, Type[BaseIndicator]] = {
        # Seasonal/Cyclical Indicators
        "fourier_analysis": FourierAnalysisIndicator,
        "seasonal_analysis": SeasonalAnalysisIndicator,
        
        # Volume-Based Indicators
        "enhanced_obv": EnhancedOBVIndicator,
        "volume_profile": VolumeProfileIndicator,
        "volume_zone_oscillator": AdvancedVZOIndicator,
    }
    
    @classmethod
    def get_available_indicators(cls) -> List[str]:
        """
        Get a list of all available advanced indicator names.
        
        Returns:
            List of indicator names
        """
        return sorted(list(cls._indicators.keys()))
    
    @classmethod
    def get_indicators_by_category(cls) -> Dict[str, List[str]]:
        """
        Get indicators grouped by category.
        
        Returns:
            Dictionary mapping category names to lists of indicator names
        """
        categories: Dict[str, List[str]] = {}
        
        for name, indicator_class in cls._indicators.items():
            # Create temporary instance to get category
            category = getattr(indicator_class, "category", "other")
            
            if category not in categories:
                categories[category] = []
                
            categories[category].append(name)
            
        # Sort indicators within each category
        for category in categories:
            categories[category].sort()
            
        return categories
    
    @classmethod
    def create_indicator(cls, name: str, **params) -> Optional[BaseIndicator]:
        """
        Create an instance of the named indicator with the provided parameters.
        
        Args:
            name: Name of the indicator to create
            **params: Parameters to pass to the indicator constructor
            
        Returns:
            Instance of the indicator, or None if not found
        """
        if name not in cls._indicators:
            return None
            
        indicator_class = cls._indicators[name]
        return indicator_class(**params)
    
    @classmethod
    def get_indicator_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific indicator.
        
        Args:
            name: Name of the indicator
            
        Returns:
            Dictionary with indicator information, or None if not found
        """
        if name not in cls._indicators:
            return None
            
        indicator_class = cls._indicators[name]
        
        # Get docstring for description
        description = indicator_class.__doc__ or ""
        description = description.strip()
        
        # Create temporary instance to get category
        category = getattr(indicator_class, "category", "other")
        
        return {
            "name": name,
            "category": category,
            "description": description,
            "class": indicator_class.__name__
        }


# Convenience function for getting all advanced indicators
def get_all_advanced_indicators() -> Dict[str, Type[BaseIndicator]]:
    """
    Get all advanced indicators as a dictionary.
    
    Returns:
        Dictionary mapping indicator names to their classes
    """
    return AdvancedIndicatorRegistry._indicators.copy()
""""""
