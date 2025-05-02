"""
Stress Testing Module

This module provides functionality for stress testing trading strategies under
extreme market conditions, historical events, and custom scenarios to evaluate
robustness and worst-case performance.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import uuid
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from pydantic import BaseModel, Field

from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class ScenarioType(str, Enum):
    """Types of stress test scenarios."""
    HISTORICAL_EVENT = "historical_event"
    SYNTHETIC_EVENT = "synthetic_event"
    MONTE_CARLO = "monte_carlo"
    CUSTOM = "custom"


class MarketCondition(str, Enum):
    """Market conditions for stress testing."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    HIGHLY_VOLATILE = "highly_volatile"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    CRASH = "crash"
    RALLY = "rally"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    BLACK_SWAN = "black_swan"


class HistoricalEvent(str, Enum):
    """Known historical market events for stress testing."""
    GFC_2008 = "global_financial_crisis_2008"
    FLASH_CRASH_2010 = "flash_crash_2010"
    EURO_CRISIS_2012 = "european_debt_crisis_2012"
    CHINA_CRASH_2015 = "china_stock_market_crash_2015"
    BREXIT_2016 = "brexit_vote_2016"
    COVID_2020 = "covid_19_crash_2020"
    SWISS_FRANC_2015 = "swiss_franc_unpegging_2015"  # Specific to forex
    INFLATION_SPIKE_2022 = "inflation_spike_2022"
    US_DEBT_CEILING_2023 = "us_debt_ceiling_2023"


class ScenarioParameter(BaseModel):
    """Parameter for a stress test scenario."""
    name: str
    value: Any
    description: Optional[str] = None


class StressTestScenario(BaseModel):
    """Configuration for a stress test scenario."""
    scenario_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    type: ScenarioType
    market_condition: Optional[MarketCondition] = None
    historical_event: Optional[HistoricalEvent] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    parameters: List[ScenarioParameter] = Field(default_factory=list)
    data_transformations: List[Dict[str, Any]] = Field(default_factory=list)
    instruments: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScenarioDataTransformer:
    """
    Transform market data according to a stress test scenario.
    """
    
    def __init__(self):
        """Initialize the data transformer."""
        # Register transformation functions
        self.transformations = {
            "price_shock": self._apply_price_shock,
            "volatility_increase": self._apply_volatility_increase,
            "trend_change": self._apply_trend_change,
            "liquidity_reduction": self._apply_liquidity_reduction,
            "correlation_change": self._apply_correlation_change,
            "gap_event": self._apply_gap_event
        }
        
    def transform_data(
        self, 
        data: pd.DataFrame, 
        transformations: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Apply a series of transformations to market data.
        
        Args:
            data: Market data DataFrame (OHLCV format expected)
            transformations: List of transformation specifications
            
        Returns:
            DataFrame: Transformed market data
        """
        transformed_data = data.copy()
        
        for transform in transformations:
            transform_type = transform.get("type")
            if transform_type in self.transformations:
                params = transform.get("parameters", {})
                transformed_data = self.transformations[transform_type](transformed_data, **params)
            else:
                logger.warning(f"Unknown transformation type: {transform_type}")
                
        return transformed_data
        
    def _apply_price_shock(
        self, data: pd.DataFrame, 
        magnitude: float, 
        direction: str = "down", 
        start_index: Optional[int] = None,
        duration_bars: Optional[int] = None,
        recovery_rate: Optional[float] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Apply a sudden price shock to the data.
        
        Args:
            data: Market data DataFrame
            magnitude: Size of the shock as a decimal percentage (e.g., 0.05 for 5%)
            direction: "up" or "down"
            start_index: Index position to start the shock (default: middle of series)
            duration_bars: Number of bars the shock should last
            recovery_rate: Rate at which price recovers after shock
        """
        result = data.copy()
        
        # Default to middle of the series if not specified
        if start_index is None:
            start_index = len(data) // 2
            
        # Make sure start index is valid
        start_index = max(0, min(start_index, len(data) - 1))
        
        # Apply multiplier based on direction
        multiplier = 1 - magnitude if direction == "down" else 1 + magnitude
        
        # Apply the shock to OHLC
        for col in ["open", "high", "low", "close"]:
            if col in result.columns:
                result.loc[start_index:, col] *= multiplier
                
        # If duration specified, implement recovery
        if duration_bars is not None and recovery_rate is not None:
            end_index = start_index + duration_bars
            end_index = min(end_index, len(data) - 1)
            
            # Calculate recovery factors
            recovery_indices = range(end_index + 1, len(data))
            if recovery_indices:
                recovery_steps = len(recovery_indices)
                
                for col in ["open", "high", "low", "close"]:
                    if col in result.columns:
                        # Get the shocked price at end_index
                        shocked_price = result.loc[end_index, col]
                        # Get the original price that would have been at end_index
                        original_end_price = data.loc[end_index, col]
                        
                        # Calculate recovery increment
                        price_diff = original_end_price - shocked_price
                        increment = price_diff * recovery_rate
                        
                        # Apply gradual recovery
                        for i, idx in enumerate(recovery_indices, 1):
                            recovery_factor = min(1.0, i / (recovery_steps / recovery_rate))
                            result.loc[idx, col] = shocked_price + (price_diff * recovery_factor)
        
        return result
        
    def _apply_volatility_increase(
        self, data: pd.DataFrame, 
        factor: float, 
        start_index: Optional[int] = None,
        duration_bars: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Increase volatility by widening the range between high and low.
        
        Args:
            data: Market data DataFrame
            factor: Volatility multiplier (e.g., 2.0 doubles volatility)
            start_index: Index position to start the volatility increase
            duration_bars: Number of bars the increased volatility should last
        """
        result = data.copy()
        
        # Default to middle of the series if not specified
        if start_index is None:
            start_index = len(data) // 2
            
        # Make sure start index is valid
        start_index = max(0, min(start_index, len(data) - 1))
        
        # Calculate end index if duration specified
        end_index = len(data) - 1
        if duration_bars is not None:
            end_index = start_index + duration_bars
            end_index = min(end_index, len(data) - 1)
        
        for i in range(start_index, end_index + 1):
            if "close" in result.columns and i > 0:
                # Get mid price
                mid_price = result.loc[i, "close"]
                
                # Recalculate high and low
                if "high" in result.columns and "low" in result.columns:
                    original_range = data.loc[i, "high"] - data.loc[i, "low"]
                    new_range = original_range * factor
                    
                    result.loc[i, "high"] = mid_price + (new_range / 2)
                    result.loc[i, "low"] = mid_price - (new_range / 2)
                    
                    # Ensure open stays within the new range
                    if "open" in result.columns:
                        result.loc[i, "open"] = np.clip(
                            result.loc[i, "open"],
                            result.loc[i, "low"],
                            result.loc[i, "high"]
                        )
                        
        return result
        
    def _apply_trend_change(
        self, data: pd.DataFrame, 
        new_trend: str,
        strength: float = 0.01,
        start_index: Optional[int] = None,
        duration_bars: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Apply a trend change to the price data.
        
        Args:
            data: Market data DataFrame
            new_trend: "up", "down", or "sideways"
            strength: Daily percentage change for the trend
            start_index: Index position to start the trend change
            duration_bars: Number of bars the trend should last
        """
        result = data.copy()
        
        # Default to middle of the series if not specified
        if start_index is None:
            start_index = len(data) // 2
            
        # Make sure start index is valid
        start_index = max(0, min(start_index, len(data) - 1))
        
        # Calculate end index if duration specified
        end_index = len(data) - 1
        if duration_bars is not None:
            end_index = start_index + duration_bars
            end_index = min(end_index, len(data) - 1)
            
        # Starting price
        base_price = result.loc[start_index, "close"] if "close" in result.columns else 1.0
        
        # Calculate trend multiplier
        if new_trend == "up":
            daily_factor = 1 + strength
        elif new_trend == "down":
            daily_factor = 1 - strength
        else:  # sideways
            daily_factor = 1.0
            
        # Apply trend
        for i in range(start_index + 1, end_index + 1):
            # Calculate cumulative factor
            days = i - start_index
            cumulative_factor = daily_factor ** days
            
            # Apply to OHLC
            for col in ["open", "high", "low", "close"]:
                if col in result.columns:
                    result.loc[i, col] = data.loc[i, col] * cumulative_factor
                    
        return result
        
    def _apply_liquidity_reduction(
        self, data: pd.DataFrame, 
        reduction_factor: float = 0.5,
        spread_increase: float = 2.0,
        start_index: Optional[int] = None,
        duration_bars: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Simulate reduced liquidity by increasing spreads and reducing volume.
        
        Args:
            data: Market data DataFrame
            reduction_factor: Factor to reduce volume by (0.5 = half volume)
            spread_increase: Factor to increase spreads by
            start_index: Index position to start the liquidity reduction
            duration_bars: Number of bars the reduced liquidity should last
        """
        result = data.copy()
        
        # Default to middle of the series
        if start_index is None:
            start_index = len(data) // 2
            
        # Calculate end index if duration specified
        end_index = len(data) - 1
        if duration_bars is not None:
            end_index = start_index + duration_bars
            end_index = min(end_index, len(data) - 1)
            
        # Reduce volume
        if "volume" in result.columns:
            result.loc[start_index:end_index, "volume"] *= reduction_factor
            
        # Increase spread by adjusting high and low
        if "high" in result.columns and "low" in result.columns:
            for i in range(start_index, end_index + 1):
                mid_price = (data.loc[i, "high"] + data.loc[i, "low"]) / 2
                original_spread = data.loc[i, "high"] - data.loc[i, "low"]
                new_spread = original_spread * spread_increase
                
                result.loc[i, "high"] = mid_price + (new_spread / 2)
                result.loc[i, "low"] = mid_price - (new_spread / 2)
                
        return result
        
    def _apply_correlation_change(
        self, data: pd.DataFrame, 
        correlation_target: float,
        reference_column: str = "external_reference",
        target_columns: Optional[List[str]] = None,
        start_index: Optional[int] = None,
        duration_bars: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Modify data to change correlation with a reference series.
        Note: This is a simplified implementation and requires an external reference series.
        
        Args:
            data: Market data DataFrame
            correlation_target: Target correlation coefficient (-1 to 1)
            reference_column: Column name of reference series
            target_columns: Columns to modify correlation for
            start_index: Index position to start the correlation change
            duration_bars: Number of bars the changed correlation should last
        """
        # This is a simplified placeholder that would require a more complex implementation
        # For real correlation modification, you'd need more sophisticated statistical methods
        logger.warning("Correlation change transformation is a simplified placeholder")
        return data
        
    def _apply_gap_event(
        self, data: pd.DataFrame, 
        gap_size: float,
        direction: str = "down",
        index: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Create a price gap event (e.g., weekend gap, overnight gap).
        
        Args:
            data: Market data DataFrame
            gap_size: Size of gap as decimal percentage
            direction: "up" or "down"
            index: Index position to place the gap
        """
        result = data.copy()
        
        # Default index if not provided
        if index is None:
            index = len(data) // 2
            
        # Ensure index is valid
        index = max(1, min(index, len(data) - 1))
        
        # Calculate multiplier
        multiplier = 1 - gap_size if direction == "down" else 1 + gap_size
        
        # Apply gap to all prices after the specified index
        for col in ["open", "high", "low", "close"]:
            if col in result.columns:
                result.loc[index:, col] *= multiplier
                
        return result


class HistoricalEventScenarios:
    """
    Predefined scenarios based on historical market events.
    """
    
    @staticmethod
    def get_scenario(event: HistoricalEvent) -> StressTestScenario:
        """Get a predefined scenario for a historical event."""
        scenarios = {
            HistoricalEvent.GFC_2008: HistoricalEventScenarios._global_financial_crisis_2008(),
            HistoricalEvent.FLASH_CRASH_2010: HistoricalEventScenarios._flash_crash_2010(),
            HistoricalEvent.EURO_CRISIS_2012: HistoricalEventScenarios._european_debt_crisis_2012(),
            HistoricalEvent.BREXIT_2016: HistoricalEventScenarios._brexit_2016(),
            HistoricalEvent.COVID_2020: HistoricalEventScenarios._covid_crash_2020(),
            HistoricalEvent.SWISS_FRANC_2015: HistoricalEventScenarios._swiss_franc_2015(),
        }
        
        if event in scenarios:
            return scenarios[event]
        else:
            # Return a default scenario
            return StressTestScenario(
                name=f"{event.value} scenario",
                description=f"Generic scenario for {event.value}",
                type=ScenarioType.HISTORICAL_EVENT,
                historical_event=event,
                market_condition=MarketCondition.HIGHLY_VOLATILE
            )
    
    @staticmethod
    def _global_financial_crisis_2008() -> StressTestScenario:
        """2008 Global Financial Crisis scenario."""
        return StressTestScenario(
            name="Global Financial Crisis 2008",
            description="Simulates market conditions during the 2008 financial crisis",
            type=ScenarioType.HISTORICAL_EVENT,
            historical_event=HistoricalEvent.GFC_2008,
            market_condition=MarketCondition.CRASH,
            start_date=datetime(2008, 9, 1),
            end_date=datetime(2009, 3, 31),
            parameters=[
                ScenarioParameter(
                    name="drawdown_magnitude",
                    value=0.50,
                    description="Maximum drawdown during crisis period"
                ),
                ScenarioParameter(
                    name="volatility_increase",
                    value=3.0,
                    description="Volatility multiplication factor"
                ),
                ScenarioParameter(
                    name="liquidity_reduction",
                    value=0.7,
                    description="Reduction in market liquidity"
                )
            ],
            data_transformations=[
                {
                    "type": "price_shock",
                    "parameters": {
                        "magnitude": 0.20,
                        "direction": "down",
                        "duration_bars": 5,
                        "recovery_rate": 0.1
                    }
                },
                {
                    "type": "volatility_increase",
                    "parameters": {
                        "factor": 3.0,
                        "duration_bars": 120
                    }
                },
                {
                    "type": "trend_change",
                    "parameters": {
                        "new_trend": "down",
                        "strength": 0.01,
                        "duration_bars": 90
                    }
                },
                {
                    "type": "liquidity_reduction",
                    "parameters": {
                        "reduction_factor": 0.3,
                        "spread_increase": 3.0,
                        "duration_bars": 60
                    }
                }
            ]
        )
        
    @staticmethod
    def _flash_crash_2010() -> StressTestScenario:
        """2010 Flash Crash scenario."""
        return StressTestScenario(
            name="Flash Crash 2010",
            description="Simulates the extreme but short-lived market drop of May 6, 2010",
            type=ScenarioType.HISTORICAL_EVENT,
            historical_event=HistoricalEvent.FLASH_CRASH_2010,
            market_condition=MarketCondition.BLACK_SWAN,
            start_date=datetime(2010, 5, 6, 14, 0),  # 2:00 PM
            end_date=datetime(2010, 5, 6, 16, 0),    # 4:00 PM
            parameters=[
                ScenarioParameter(
                    name="crash_magnitude",
                    value=0.10,
                    description="Magnitude of the crash in decimal percentage"
                ),
                ScenarioParameter(
                    name="crash_duration_minutes",
                    value=30,
                    description="Duration of the crash in minutes"
                ),
                ScenarioParameter(
                    name="recovery_rate",
                    value=0.8,
                    description="Rate of recovery after the crash (0-1)"
                )
            ],
            data_transformations=[
                {
                    "type": "price_shock",
                    "parameters": {
                        "magnitude": 0.10,
                        "direction": "down",
                        "duration_bars": 6,  # Assuming 5-min bars
                        "recovery_rate": 0.8
                    }
                },
                {
                    "type": "volatility_increase",
                    "parameters": {
                        "factor": 5.0,
                        "duration_bars": 12
                    }
                },
                {
                    "type": "liquidity_reduction",
                    "parameters": {
                        "reduction_factor": 0.1,
                        "spread_increase": 10.0,
                        "duration_bars": 6
                    }
                }
            ]
        )
        
    @staticmethod
    def _european_debt_crisis_2012() -> StressTestScenario:
        """European Debt Crisis scenario."""
        return StressTestScenario(
            name="European Debt Crisis 2012",
            description="Simulates market conditions during the peak of the European sovereign debt crisis",
            type=ScenarioType.HISTORICAL_EVENT,
            historical_event=HistoricalEvent.EURO_CRISIS_2012,
            market_condition=MarketCondition.HIGHLY_VOLATILE,
            start_date=datetime(2012, 5, 1),
            end_date=datetime(2012, 8, 31),
            parameters=[
                ScenarioParameter(
                    name="euro_weakness",
                    value=0.15,
                    description="EUR currency weakness factor"
                ),
                ScenarioParameter(
                    name="market_volatility",
                    value=2.0,
                    description="Volatility multiplication factor"
                )
            ],
            data_transformations=[
                {
                    "type": "trend_change",
                    "parameters": {
                        "new_trend": "down",
                        "strength": 0.005,
                        "duration_bars": 60
                    }
                },
                {
                    "type": "volatility_increase",
                    "parameters": {
                        "factor": 2.0,
                        "duration_bars": 80
                    }
                },
                {
                    "type": "gap_event",
                    "parameters": {
                        "gap_size": 0.03,
                        "direction": "down",
                        "index": 15
                    }
                }
            ]
        )
        
    @staticmethod
    def _brexit_2016() -> StressTestScenario:
        """Brexit Vote 2016 scenario."""
        return StressTestScenario(
            name="Brexit Vote 2016",
            description="Simulates market reaction to the UK Brexit referendum",
            type=ScenarioType.HISTORICAL_EVENT,
            historical_event=HistoricalEvent.BREXIT_2016,
            market_condition=MarketCondition.HIGHLY_VOLATILE,
            start_date=datetime(2016, 6, 23),
            end_date=datetime(2016, 6, 27),
            parameters=[
                ScenarioParameter(
                    name="gbp_drop",
                    value=0.10,
                    description="GBP drop against USD after vote"
                )
            ],
            data_transformations=[
                {
                    "type": "gap_event",
                    "parameters": {
                        "gap_size": 0.10,
                        "direction": "down",
                        "index": 1
                    }
                },
                {
                    "type": "volatility_increase",
                    "parameters": {
                        "factor": 3.0,
                        "duration_bars": 15
                    }
                }
            ]
        )
        
    @staticmethod
    def _covid_crash_2020() -> StressTestScenario:
        """COVID-19 Market Crash scenario."""
        return StressTestScenario(
            name="COVID-19 Market Crash 2020",
            description="Simulates the rapid market decline during the initial COVID-19 outbreak",
            type=ScenarioType.HISTORICAL_EVENT,
            historical_event=HistoricalEvent.COVID_2020,
            market_condition=MarketCondition.CRASH,
            start_date=datetime(2020, 2, 19),
            end_date=datetime(2020, 3, 23),
            parameters=[
                ScenarioParameter(
                    name="drawdown_magnitude",
                    value=0.35,
                    description="Maximum drawdown during crash"
                ),
                ScenarioParameter(
                    name="vix_spike",
                    value=4.0,
                    description="VIX increase factor"
                )
            ],
            data_transformations=[
                {
                    "type": "trend_change",
                    "parameters": {
                        "new_trend": "down",
                        "strength": 0.02,
                        "duration_bars": 25
                    }
                },
                {
                    "type": "volatility_increase",
                    "parameters": {
                        "factor": 4.0,
                        "duration_bars": 25
                    }
                },
                {
                    "type": "gap_event",
                    "parameters": {
                        "gap_size": 0.05,
                        "direction": "down",
                        "index": 5
                    }
                },
                {
                    "type": "gap_event",
                    "parameters": {
                        "gap_size": 0.07,
                        "direction": "down",
                        "index": 15
                    }
                }
            ]
        )
        
    @staticmethod
    def _swiss_franc_2015() -> StressTestScenario:
        """Swiss Franc Unpegging 2015 scenario."""
        return StressTestScenario(
            name="Swiss Franc Unpegging 2015",
            description="Simulates the extreme move in CHF when SNB removed the EUR/CHF floor",
            type=ScenarioType.HISTORICAL_EVENT,
            historical_event=HistoricalEvent.SWISS_FRANC_2015,
            market_condition=MarketCondition.BLACK_SWAN,
            start_date=datetime(2015, 1, 15),
            end_date=datetime(2015, 1, 16),
            parameters=[
                ScenarioParameter(
                    name="price_shock",
                    value=0.20,
                    description="Magnitude of price shock in CHF pairs"
                )
            ],
            data_transformations=[
                {
                    "type": "price_shock",
                    "parameters": {
                        "magnitude": 0.20,
                        "direction": "down", # For EUR/CHF, USD/CHF
                        "duration_bars": 1,
                        "recovery_rate": 0.1
                    }
                },
                {
                    "type": "liquidity_reduction",
                    "parameters": {
                        "reduction_factor": 0.05,
                        "spread_increase": 50.0,
                        "duration_bars": 5
                    }
                },
                {
                    "type": "volatility_increase",
                    "parameters": {
                        "factor": 10.0,
                        "duration_bars": 10
                    }
                }
            ]
        )


class StressTestResult(BaseModel):
    """Results of a stress test."""
    test_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario: StressTestScenario
    strategy_id: str
    execution_time: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: float
    baseline_metrics: Dict[str, float]
    stress_metrics: Dict[str, float]
    impact_percentage: Dict[str, float]
    passed: bool
    failure_reasons: List[str] = Field(default_factory=list)
    plots: Optional[Dict[str, str]] = None  # Base64 encoded plots
    detailed_results: Optional[Dict[str, Any]] = None


class StressTestEngine:
    """
    Engine for performing stress tests on trading strategies.
    """
    
    def __init__(self, data_provider: Callable, strategy_evaluator: Callable):
        """
        Initialize the stress test engine.
        
        Args:
            data_provider: Function that provides market data
            strategy_evaluator: Function that evaluates a strategy on data
        """
        self.data_provider = data_provider
        self.evaluate_strategy = strategy_evaluator
        self.transformer = ScenarioDataTransformer()
        self.failure_thresholds = {
            'max_drawdown': 0.25,    # 25%
            'sharpe_ratio': 0.5,     # Minimum acceptable Sharpe ratio
            'profit_factor': 1.0,    # Minimum 1:1 profit factor
            'win_rate': 0.4,         # Minimum 40% win rate
            'avg_profit_loss': 0.7   # Average profit must be at least 70% of baseline
        }
        
    def run_stress_test(
        self, 
        scenario: StressTestScenario,
        strategy_id: str,
        parameters: Dict[str, Any] = None,
        custom_thresholds: Dict[str, float] = None
    ) -> StressTestResult:
        """
        Run a stress test with the given scenario against a strategy.
        
        Args:
            scenario: The stress test scenario to apply
            strategy_id: ID of the strategy to test
            parameters: Additional parameters for strategy
            custom_thresholds: Custom failure thresholds
            
        Returns:
            StressTestResult: Results of the stress test
        """
        start_time = datetime.utcnow()
        
        try:
            # Apply thresholds
            if custom_thresholds:
                test_thresholds = {**self.failure_thresholds, **custom_thresholds}
            else:
                test_thresholds = self.failure_thresholds
                
            # Get baseline data
            if scenario.historical_event and scenario.start_date and scenario.end_date:
                # Use actual historical data if available
                baseline_data = self._get_historical_data(
                    scenario.historical_event,
                    scenario.start_date,
                    scenario.end_date,
                    scenario.instruments
                )
            else:
                # Otherwise get normal sample data
                baseline_data = self._get_baseline_data(scenario)
            
            # Get transformed data according to scenario
            stress_data = self._apply_scenario(baseline_data, scenario)
            
            # Evaluate strategy on baseline data
            baseline_metrics = self.evaluate_strategy(
                strategy_id=strategy_id, 
                data=baseline_data, 
                parameters=parameters
            )
            
            # Evaluate strategy on stress data
            stress_metrics = self.evaluate_strategy(
                strategy_id=strategy_id, 
                data=stress_data, 
                parameters=parameters
            )
            
            # Calculate impact
            impact = self._calculate_impact(baseline_metrics, stress_metrics)
            
            # Determine pass/fail
            passed, reasons = self._evaluate_results(stress_metrics, baseline_metrics, test_thresholds)
            
            # Create visualization
            plots = self._create_visualizations(baseline_data, stress_data, baseline_metrics, stress_metrics)
            
            # Create result object
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            result = StressTestResult(
                scenario=scenario,
                strategy_id=strategy_id,
                duration_seconds=duration,
                baseline_metrics=baseline_metrics,
                stress_metrics=stress_metrics,
                impact_percentage=impact,
                passed=passed,
                failure_reasons=reasons,
                plots=plots
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during stress test execution: {str(e)}")
            # Return a failed test result
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            return StressTestResult(
                scenario=scenario,
                strategy_id=strategy_id,
                duration_seconds=duration,
                baseline_metrics={},
                stress_metrics={},
                impact_percentage={},
                passed=False,
                failure_reasons=[f"Test error: {str(e)}"]
            )
            
    def run_multiple_scenarios(
        self, 
        scenarios: List[StressTestScenario],
        strategy_id: str,
        parameters: Dict[str, Any] = None
    ) -> List[StressTestResult]:
        """
        Run multiple stress test scenarios against a strategy.
        
        Args:
            scenarios: List of stress test scenarios to apply
            strategy_id: ID of the strategy to test
            parameters: Additional parameters for strategy
            
        Returns:
            List[StressTestResult]: Results of the stress tests
        """
        results = []
        for scenario in scenarios:
            result = self.run_stress_test(scenario, strategy_id, parameters)
            results.append(result)
            
        return results
        
    def run_all_historical_scenarios(
        self,
        strategy_id: str,
        parameters: Dict[str, Any] = None
    ) -> Dict[HistoricalEvent, StressTestResult]:
        """
        Run stress tests for all available historical scenarios.
        
        Args:
            strategy_id: ID of the strategy to test
            parameters: Additional parameters for strategy
            
        Returns:
            Dict[HistoricalEvent, StressTestResult]: Results by historical event
        """
        results = {}
        
        for event in HistoricalEvent:
            scenario = HistoricalEventScenarios.get_scenario(event)
            result = self.run_stress_test(scenario, strategy_id, parameters)
            results[event] = result
            
        return results
    
    def _get_historical_data(
        self, 
        event: HistoricalEvent,
        start_date: datetime,
        end_date: datetime,
        instruments: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Get actual historical data for a time period."""
        try:
            data = {}
            for instrument in instruments:
                # Call data provider
                instrument_data = self.data_provider(
                    instrument=instrument,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True  # Use cached data when possible
                )
                
                if instrument_data is not None and not instrument_data.empty:
                    data[instrument] = instrument_data
                    
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return {}
    
    def _get_baseline_data(self, scenario: StressTestScenario) -> Dict[str, pd.DataFrame]:
        """Get baseline data for a scenario."""
        # Implementation would fetch appropriate baseline data
        # This is a placeholder
        return {}
        
    def _apply_scenario(
        self, 
        baseline_data: Dict[str, pd.DataFrame], 
        scenario: StressTestScenario
    ) -> Dict[str, pd.DataFrame]:
        """Apply scenario transformations to baseline data."""
        transformed_data = {}
        
        for instrument, data in baseline_data.items():
            transformed_data[instrument] = self.transformer.transform_data(
                data, 
                scenario.data_transformations
            )
            
        return transformed_data
        
    def _calculate_impact(
        self, 
        baseline_metrics: Dict[str, float],
        stress_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate the impact of the stress scenario on metrics."""
        impact = {}
        
        for key in baseline_metrics:
            if key in stress_metrics and baseline_metrics[key] != 0:
                impact[key] = ((stress_metrics[key] - baseline_metrics[key]) / 
                              abs(baseline_metrics[key])) * 100
                
        return impact
        
    def _evaluate_results(
        self,
        stress_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """Evaluate if stress test passed based on failure thresholds."""
        passed = True
        failure_reasons = []
        
        # Check absolute thresholds
        if 'max_drawdown' in stress_metrics and thresholds.get('max_drawdown'):
            if stress_metrics['max_drawdown'] > thresholds['max_drawdown']:
                passed = False
                failure_reasons.append(
                    f"Max drawdown {stress_metrics['max_drawdown']:.2%} exceeds threshold {thresholds['max_drawdown']:.2%}"
                )
                
        if 'sharpe_ratio' in stress_metrics and thresholds.get('sharpe_ratio'):
            if stress_metrics['sharpe_ratio'] < thresholds['sharpe_ratio']:
                passed = False
                failure_reasons.append(
                    f"Sharpe ratio {stress_metrics['sharpe_ratio']:.2f} below threshold {thresholds['sharpe_ratio']:.2f}"
                )
                
        if 'profit_factor' in stress_metrics and thresholds.get('profit_factor'):
            if stress_metrics['profit_factor'] < thresholds['profit_factor']:
                passed = False
                failure_reasons.append(
                    f"Profit factor {stress_metrics['profit_factor']:.2f} below threshold {thresholds['profit_factor']:.2f}"
                )
                
        if 'win_rate' in stress_metrics and thresholds.get('win_rate'):
            if stress_metrics['win_rate'] < thresholds['win_rate']:
                passed = False
                failure_reasons.append(
                    f"Win rate {stress_metrics['win_rate']:.2%} below threshold {thresholds['win_rate']:.2%}"
                )
                
        # Check relative decline from baseline
        if 'net_profit' in stress_metrics and 'net_profit' in baseline_metrics:
            if baseline_metrics['net_profit'] > 0 and stress_metrics['net_profit'] < 0:
                passed = False
                failure_reasons.append("Strategy became unprofitable under stress conditions")
                
        if 'avg_profit_loss' in thresholds:
            threshold = thresholds['avg_profit_loss']
            for metric in ['avg_profit', 'average_profit', 'mean_return']:
                if (metric in stress_metrics and metric in baseline_metrics and 
                    baseline_metrics[metric] > 0):
                    ratio = stress_metrics[metric] / baseline_metrics[metric]
                    if ratio < threshold:
                        passed = False
                        failure_reasons.append(
                            f"Average profit/return declined by {(1-ratio):.2%}, exceeding threshold {(1-threshold):.2%}"
                        )
                        break
                        
        return passed, failure_reasons
        
    def _create_visualizations(
        self,
        baseline_data: Dict[str, pd.DataFrame],
        stress_data: Dict[str, pd.DataFrame],
        baseline_metrics: Dict[str, float],
        stress_metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """Create visualizations comparing baseline and stress test results."""
        plots = {}
        
        try:
            # Example: Price comparison chart for first instrument
            if baseline_data and stress_data:
                instrument = next(iter(baseline_data.keys()))
                if instrument in stress_data:
                    plt.figure(figsize=(12, 6))
                    
                    # Plot baseline price
                    if 'close' in baseline_data[instrument].columns:
                        plt.plot(
                            baseline_data[instrument].index, 
                            baseline_data[instrument]['close'], 
                            label='Baseline', 
                            color='blue'
                        )
                    
                    # Plot stressed price
                    if 'close' in stress_data[instrument].columns:
                        plt.plot(
                            stress_data[instrument].index, 
                            stress_data[instrument]['close'], 
                            label='Stress Scenario', 
                            color='red'
                        )
                        
                    plt.title(f'Price Comparison - {instrument}')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Save to base64
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plots['price_comparison'] = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()
                    
            # Example: Metrics comparison
            plt.figure(figsize=(10, 6))
            
            # Select metrics to display
            display_metrics = [
                'net_profit', 'max_drawdown', 'sharpe_ratio', 
                'win_rate', 'profit_factor'
            ]
            
            # Filter to metrics that exist in both
            metrics_to_plot = [
                m for m in display_metrics 
                if m in baseline_metrics and m in stress_metrics
            ]
            
            if metrics_to_plot:
                x = np.arange(len(metrics_to_plot))
                width = 0.35
                
                baseline_values = [baseline_metrics[m] for m in metrics_to_plot]
                stress_values = [stress_metrics[m] for m in metrics_to_plot]
                
                plt.bar(x - width/2, baseline_values, width, label='Baseline')
                plt.bar(x + width/2, stress_values, width, label='Stress Test')
                
                plt.xlabel('Metric')
                plt.ylabel('Value')
                plt.title('Performance Metrics Comparison')
                plt.xticks(x, metrics_to_plot)
                plt.legend()
                
                # Save to base64
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plots['metrics_comparison'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                
            return plots
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            return {}


class CustomScenarioBuilder:
    """
    Utility for building custom stress test scenarios.
    """
    
    @staticmethod
    def create_price_shock_scenario(
        name: str,
        description: str,
        magnitude: float,
        direction: str = "down",
        duration_bars: int = 10,
        recovery_rate: float = 0.2,
        instruments: List[str] = None
    ) -> StressTestScenario:
        """Create a price shock scenario."""
        return StressTestScenario(
            name=name,
            description=description,
            type=ScenarioType.CUSTOM,
            market_condition=MarketCondition.HIGHLY_VOLATILE,
            instruments=instruments or [],
            parameters=[
                ScenarioParameter(
                    name="magnitude",
                    value=magnitude,
                    description="Magnitude of price shock"
                ),
                ScenarioParameter(
                    name="direction",
                    value=direction,
                    description="Direction of price movement"
                ),
                ScenarioParameter(
                    name="duration_bars",
                    value=duration_bars,
                    description="Duration of shock in bars"
                ),
                ScenarioParameter(
                    name="recovery_rate",
                    value=recovery_rate,
                    description="Rate of recovery after shock"
                )
            ],
            data_transformations=[
                {
                    "type": "price_shock",
                    "parameters": {
                        "magnitude": magnitude,
                        "direction": direction,
                        "duration_bars": duration_bars,
                        "recovery_rate": recovery_rate
                    }
                }
            ]
        )
        
    @staticmethod
    def create_volatility_scenario(
        name: str,
        description: str,
        volatility_factor: float,
        duration_bars: int = 20,
        instruments: List[str] = None
    ) -> StressTestScenario:
        """Create a volatility increase scenario."""
        return StressTestScenario(
            name=name,
            description=description,
            type=ScenarioType.CUSTOM,
            market_condition=MarketCondition.VOLATILE,
            instruments=instruments or [],
            parameters=[
                ScenarioParameter(
                    name="volatility_factor",
                    value=volatility_factor,
                    description="Factor to increase volatility"
                ),
                ScenarioParameter(
                    name="duration_bars",
                    value=duration_bars,
                    description="Duration in bars"
                )
            ],
            data_transformations=[
                {
                    "type": "volatility_increase",
                    "parameters": {
                        "factor": volatility_factor,
                        "duration_bars": duration_bars
                    }
                }
            ]
        )
        
    @staticmethod
    def create_trend_change_scenario(
        name: str,
        description: str,
        new_trend: str,
        strength: float,
        duration_bars: int = 20,
        instruments: List[str] = None
    ) -> StressTestScenario:
        """Create a trend change scenario."""
        return StressTestScenario(
            name=name,
            description=description,
            type=ScenarioType.CUSTOM,
            market_condition=(
                MarketCondition.TRENDING_UP if new_trend == "up" else
                MarketCondition.TRENDING_DOWN if new_trend == "down" else
                MarketCondition.SIDEWAYS
            ),
            instruments=instruments or [],
            parameters=[
                ScenarioParameter(
                    name="new_trend",
                    value=new_trend,
                    description="Direction of new trend"
                ),
                ScenarioParameter(
                    name="strength",
                    value=strength,
                    description="Strength of trend"
                ),
                ScenarioParameter(
                    name="duration_bars",
                    value=duration_bars,
                    description="Duration in bars"
                )
            ],
            data_transformations=[
                {
                    "type": "trend_change",
                    "parameters": {
                        "new_trend": new_trend,
                        "strength": strength,
                        "duration_bars": duration_bars
                    }
                }
            ]
        )
        
    @staticmethod
    def create_liquidity_crisis_scenario(
        name: str,
        description: str,
        volume_reduction: float = 0.8,
        spread_multiplier: float = 5.0,
        duration_bars: int = 15,
        instruments: List[str] = None
    ) -> StressTestScenario:
        """Create a liquidity crisis scenario."""
        return StressTestScenario(
            name=name,
            description=description,
            type=ScenarioType.CUSTOM,
            market_condition=MarketCondition.LIQUIDITY_CRISIS,
            instruments=instruments or [],
            parameters=[
                ScenarioParameter(
                    name="volume_reduction",
                    value=volume_reduction,
                    description="Factor to reduce volume"
                ),
                ScenarioParameter(
                    name="spread_multiplier",
                    value=spread_multiplier,
                    description="Factor to increase spreads"
                ),
                ScenarioParameter(
                    name="duration_bars",
                    value=duration_bars,
                    description="Duration in bars"
                )
            ],
            data_transformations=[
                {
                    "type": "liquidity_reduction",
                    "parameters": {
                        "reduction_factor": 1.0 - volume_reduction,
                        "spread_increase": spread_multiplier,
                        "duration_bars": duration_bars
                    }
                }
            ]
        )
        
    @staticmethod
    def create_combined_scenario(
        name: str,
        description: str,
        transformations: List[Dict[str, Any]],
        market_condition: MarketCondition,
        instruments: List[str] = None
    ) -> StressTestScenario:
        """Create a custom scenario with multiple transformations."""
        return StressTestScenario(
            name=name,
            description=description,
            type=ScenarioType.CUSTOM,
            market_condition=market_condition,
            instruments=instruments or [],
            data_transformations=transformations
        )
