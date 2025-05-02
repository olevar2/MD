"""
Model Metrics

This module defines the ModelMetrics class for tracking model performance metrics.
"""

class ModelMetrics:
    """
    Class representing performance metrics for a model version.
    
    This dynamic class can store any metrics as attributes.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize with provided metrics.
        
        Args:
            **kwargs: Arbitrary keyword arguments representing metrics (e.g., accuracy=0.95)
        """
        # Set all metrics as attributes
        for name, value in kwargs.items():
            setattr(self, name, value)
    
    def __repr__(self) -> str:
        """String representation showing all metrics."""
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                              for k, v in vars(self).items())
        return f"ModelMetrics({metrics_str})"
    
    def as_dict(self) -> dict:
        """Convert metrics to a dictionary."""
        return vars(self)
    
    def add_metric(self, name: str, value):
        """
        Add a new metric or update an existing one.
        
        Args:
            name: Name of the metric
            value: Value of the metric
        """
        setattr(self, name, value)
        
    def get_metric(self, name: str, default=None):
        """
        Get a specific metric value.
        
        Args:
            name: Name of the metric
            default: Default value if metric doesn't exist
            
        Returns:
            The metric value or default if not found
        """
        return getattr(self, name, default)
