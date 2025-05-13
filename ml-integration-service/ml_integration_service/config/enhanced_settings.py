"""
Configuration settings for enhanced ML Integration Service features.

This module provides configuration settings for visualization,
optimization, and stress testing components.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseSettings, Field


class VisualizationSettings(BaseSettings):
    """Settings for visualization components."""
    
    DEFAULT_PLOT_HEIGHT: int = Field(600, description="Default height for plots")
    DEFAULT_PLOT_WIDTH: int = Field(800, description="Default width for plots")
    MAX_POINTS_INTERACTIVE: int = Field(10000, description="Maximum points for interactive plots")
    CHART_THEME: str = Field("plotly_white", description="Default chart theme")
    CONFIDENCE_COLORS: Dict[str, str] = Field(
        default={
            "high": "rgba(0,255,0,0.2)",
            "medium": "rgba(255,255,0,0.2)",
            "low": "rgba(255,0,0,0.2)"
        }
    )


class OptimizationSettings(BaseSettings):
    """Settings for optimization algorithms."""
    
    MAX_OPTIMIZATION_ITERATIONS: int = Field(1000, description="Maximum optimization iterations")
    CONVERGENCE_THRESHOLD: float = Field(1e-6, description="Convergence threshold")
    DEFAULT_POPULATION_SIZE: int = Field(100, description="Default population size for evolutionary algorithms")
    MAX_PARALLEL_TRIALS: int = Field(10, description="Maximum parallel optimization trials")
    
    # Regime-aware optimization
    REGIME_HISTORY_WINDOW: int = Field(
        90,
        description="Days of history to consider for regime analysis"
    )
    REGIME_WEIGHT_DECAY: float = Field(
        0.95,
        description="Weight decay factor for historical regimes"
    )
    
    # Multi-objective optimization
    DEFAULT_OBJECTIVE_WEIGHTS: Dict[str, float] = Field(
        default={
            "return": 0.4,
            "risk": 0.3,
            "robustness": 0.3
        }
    )
    
    # Online learning
    DEFAULT_LEARNING_RATE: float = Field(0.01, description="Default learning rate")
    DEFAULT_MOMENTUM: float = Field(0.9, description="Default momentum factor")
    ADAPTATION_THRESHOLD: float = Field(
        0.05,
        description="Minimum performance change to trigger adaptation"
    )


class StressTestingSettings(BaseSettings):
    """Settings for stress testing components."""
    
    # Robustness testing
    DEFAULT_SCENARIO_COUNT: int = Field(1000, description="Default number of test scenarios")
    EXTREME_EVENT_PROBABILITY: float = Field(0.1, description="Probability of extreme events")
    MAX_VOLATILITY_MULTIPLIER: float = Field(5.0, description="Maximum volatility multiplication")
    
    # Sensitivity analysis
    PARAMETER_TEST_POINTS: int = Field(20, description="Points to test per parameter")
    SENSITIVITY_THRESHOLD: float = Field(0.1, description="Threshold for sensitivity flagging")
    
    # Load testing
    DEFAULT_TEST_DURATION: int = Field(300, description="Default load test duration in seconds")
    MAX_TEST_DURATION: int = Field(3600, description="Maximum allowed test duration")
    DEFAULT_TARGET_RPS: int = Field(100, description="Default requests per second")
    MAX_TARGET_RPS: int = Field(1000, description="Maximum allowed RPS")
    RESPONSE_TIME_THRESHOLD: float = Field(0.1, description="Response time threshold in seconds")


class EnhancedServiceSettings(BaseSettings):
    """Main settings for enhanced service features."""
    
    # Integration settings
    ML_WORKBENCH_API_URL: str = Field(
        "http://ml_workbench:8000",
        description="ML Workbench service URL"
    )
    STRATEGY_ENGINE_API_URL: str = Field(
        "http://strategy-engine:8000",
        description="Strategy execution engine URL"
    )
    DATA_PIPELINE_API_URL: str = Field(
        "http://data-pipeline:8000",
        description="Data pipeline service URL"
    )
    
    # Feature toggles
    ENABLE_INTERACTIVE_PLOTS: bool = Field(True, description="Enable interactive plotting")
    ENABLE_ADVANCED_OPTIMIZATION: bool = Field(True, description="Enable advanced optimization")
    ENABLE_STRESS_TESTING: bool = Field(True, description="Enable stress testing")
    
    # Performance settings
    CACHE_DURATION: int = Field(3600, description="Cache duration in seconds")
    MAX_CACHE_ITEMS: int = Field(1000, description="Maximum items in cache")
    BATCH_SIZE: int = Field(100, description="Default batch size for operations")
    
    # Component settings
    visualization: VisualizationSettings = VisualizationSettings()
    optimization: OptimizationSettings = OptimizationSettings()
    stress_testing: StressTestingSettings = StressTestingSettings()
    
    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        env_prefix = "ML_INTEGRATION_"
        
enhanced_settings = EnhancedServiceSettings()
