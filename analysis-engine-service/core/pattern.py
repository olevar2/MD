"""
Elliott Wave Pattern Module.

This module provides the ElliottWavePattern class for representing detected patterns.
"""

from typing import Dict, Any
from analysis_engine.analysis.advanced_ta.base import PatternResult
from analysis_engine.analysis.advanced_ta.elliott_wave.models import WaveType, WaveDegree


class ElliottWavePattern(PatternResult):
    """
    Represents a detected Elliott Wave pattern
    
    Extends PatternResult with Elliott Wave specific attributes.
    """
    
    def __init__(self, **kwargs):
    """
      init  .
    
    Args:
        kwargs: Description of kwargs
    
    """

        super().__init__(**kwargs)
        self.wave_type = kwargs.get("wave_type", WaveType.UNKNOWN)
        self.wave_degree = kwargs.get("wave_degree", WaveDegree.INTERMEDIATE)
        self.waves = kwargs.get("waves", {})  # Dict mapping wave positions to (time, price) tuples
        self.sub_waves = kwargs.get("sub_waves", {})  # Sub-waves if identified
        self.fibonacci_levels = kwargs.get("fibonacci_levels", {})  # Fibonacci projections
        self.completion_percentage = kwargs.get("completion_percentage", 100.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        pattern_dict = super().to_dict()
        pattern_dict.update({
            "wave_type": self.wave_type.value,
            "wave_degree": self.wave_degree.value,
            "waves": {str(k.value): {"time": v[0].isoformat(), "price": v[1]} 
                    for k, v in self.waves.items()},
            "completion_percentage": self.completion_percentage
        })
        
        # Add fibonacci levels if present
        if self.fibonacci_levels:
            pattern_dict["fibonacci_levels"] = {
                k: {"price": v} for k, v in self.fibonacci_levels.items()
            }
            
        return pattern_dict