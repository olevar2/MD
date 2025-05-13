"""
Stress Testing Module.

This module provides capabilities for stress testing trading strategies under
extreme market conditions and historical events.
"""
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import uuid
import numpy as np
import pandas as pd
from enum import Enum
from pydantic import BaseModel, Field
from analysis_engine.analysis.backtesting.core import BacktestResult, BacktestConfiguration, BacktestEngine
from analysis_engine.utils.logger import get_logger
logger = get_logger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class StressEventType(str, Enum):
    """Types of stress events for testing."""
    HISTORICAL_CRASH = 'historical_crash'
    VOLATILITY_SPIKE = 'volatility_spike'
    LIQUIDITY_CRISIS = 'liquidity_crisis'
    FLASH_CRASH = 'flash_crash'
    CORRELATION_BREAKDOWN = 'correlation_breakdown'
    TREND_REVERSAL = 'trend_reversal'
    CUSTOM = 'custom'


class StressEvent(BaseModel):
    """Definition of a market stress event."""
    event_id: str = Field(default_factory=lambda : str(uuid.uuid4()))
    name: str
    type: StressEventType
    description: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    affected_instruments: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    transformation_function: Optional[str] = None
    historical_data_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StressTestConfiguration(BaseModel):
    """Configuration for a stress test."""
    strategy_name: str
    strategy_parameters: Dict[str, Any] = Field(default_factory=dict)
    events: List[StressEvent]
    instruments: List[str]
    initial_capital: float = 100000.0
    base_period_start: Optional[datetime] = None
    base_period_end: Optional[datetime] = None
    data_timeframe: str = '1H'
    scenario_combinations: bool = False
    include_base_scenario: bool = True
    position_sizing: Optional[str] = 'fixed'
    position_sizing_settings: Dict[str, Any] = Field(default_factory=dict)
    slippage_model: Optional[str] = 'fixed'
    slippage_settings: Dict[str, Any] = Field(default_factory=dict)
    commission_model: Optional[str] = 'fixed'
    commission_settings: Dict[str, Any] = Field(default_factory=dict)
    risk_management_settings: Dict[str, Any] = Field(default_factory=dict)
    data_settings: Dict[str, Any] = Field(default_factory=dict)


class StressTestScenario(BaseModel):
    """A scenario for stress testing."""
    scenario_id: str = Field(default_factory=lambda : str(uuid.uuid4()))
    name: str
    description: str
    events: List[StressEvent] = Field(default_factory=list)
    is_base_scenario: bool = False
    parameters: Dict[str, Any] = Field(default_factory=dict)


class StressTestResult(BaseModel):
    """Result of a stress test."""
    test_id: str = Field(default_factory=lambda : str(uuid.uuid4()))
    strategy_name: str
    configuration: Dict[str, Any]
    scenario_results: Dict[str, BacktestResult] = Field(default_factory=dict)
    comparative_metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StressTester:
    """
    Stress tester for evaluating trading strategies under extreme conditions.
    
    This class supports:
    1. Historical event replay
    2. Synthetic scenario generation
    3. Parameter stress testing
    4. Comparison with base scenario
    """

    def __init__(self, config: StressTestConfiguration, strategy_factory:
        Callable[[BacktestConfiguration], BacktestEngine], data_provider:
        Callable[[List[str], datetime, datetime, str], Dict[str, pd.DataFrame]]
        ):
        """
        Initialize stress tester.
        
        Args:
            config: Configuration for the stress test
            strategy_factory: Function that creates a strategy instance
            data_provider: Function that provides market data
        """
        self.config = config
        self.strategy_factory = strategy_factory
        self.data_provider = data_provider
        self.scenarios = self._generate_scenarios()

    @with_exception_handling
    def run_stress_test(self) ->StressTestResult:
        """
        Run a comprehensive stress test across all scenarios.
        
        Returns:
            StressTestResult: Results of the stress test
        """
        logger.info(f'Starting stress test for {self.config.strategy_name}')
        logger.info(f'Running {len(self.scenarios)} scenarios')
        scenario_results = {}
        for scenario in self.scenarios:
            logger.info(f'Running scenario: {scenario.name}')
            try:
                scenario_data = self._prepare_scenario_data(scenario)
                backtest_config = self._create_backtest_config(scenario)
                strategy = self.strategy_factory(backtest_config)
                backtest_result = strategy.run_backtest(scenario_data)
                scenario_results[scenario.scenario_id] = backtest_result
                logger.info(f'Completed scenario: {scenario.name}')
            except Exception as e:
                logger.error(f'Error running scenario {scenario.name}: {e}')
        comparative_metrics = self._calculate_comparative_metrics(
            scenario_results)
        result = StressTestResult(strategy_name=self.config.strategy_name,
            configuration=self.config.dict(), scenario_results=
            scenario_results, comparative_metrics=comparative_metrics)
        return result

    def _generate_scenarios(self) ->List[StressTestScenario]:
        """Generate scenarios based on the configuration."""
        scenarios = []
        if self.config.include_base_scenario:
            base_scenario = StressTestScenario(name='Base Scenario',
                description='Baseline scenario without stress events',
                is_base_scenario=True)
            scenarios.append(base_scenario)
        if not self.config.scenario_combinations:
            for event in self.config.events:
                scenario = StressTestScenario(name=
                    f'Scenario: {event.name}', description=event.
                    description, events=[event])
                scenarios.append(scenario)
        else:
            for event in self.config.events:
                scenario = StressTestScenario(name=
                    f'Scenario: {event.name}', description=event.
                    description, events=[event])
                scenarios.append(scenario)
            if len(self.config.events) > 1:
                from itertools import combinations
                for i in range(2, min(len(self.config.events) + 1, 4)):
                    for combo in combinations(self.config.events, i):
                        event_names = [event.name for event in combo]
                        scenario = StressTestScenario(name=
                            f"Combined: {' + '.join(event_names)}",
                            description=
                            f'Combined stress scenario with multiple events',
                            events=list(combo))
                        scenarios.append(scenario)
        return scenarios

    def _prepare_scenario_data(self, scenario: StressTestScenario) ->Dict[
        str, pd.DataFrame]:
        """
        Prepare data for a specific scenario, applying stress events.
        
        Args:
            scenario: The scenario to prepare data for
            
        Returns:
            dict: Dictionary of DataFrames with scenario data
        """
        if scenario.is_base_scenario:
            return self._get_base_period_data()
        data = self._get_base_period_data()
        for event in scenario.events:
            data = self._apply_stress_event(data, event)
        return data

    def _get_base_period_data(self) ->Dict[str, pd.DataFrame]:
        """Get data for the base period."""
        start_date = self.config.base_period_start
        end_date = self.config.base_period_end
        if start_date is None or end_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
        data = self.data_provider(self.config.instruments, start_date,
            end_date, self.config.data_timeframe)
        return data

    def _apply_stress_event(self, data: Dict[str, pd.DataFrame], event:
        StressEvent) ->Dict[str, pd.DataFrame]:
        """
        Apply a stress event to market data.
        
        Args:
            data: Dictionary of DataFrames with market data
            event: The stress event to apply
            
        Returns:
            dict: Transformed market data
        """
        transformed_data = {k: df.copy() for k, df in data.items()}
        if event.type == StressEventType.HISTORICAL_CRASH:
            transformed_data = self._apply_historical_crash(transformed_data,
                event)
        elif event.type == StressEventType.VOLATILITY_SPIKE:
            transformed_data = self._apply_volatility_spike(transformed_data,
                event)
        elif event.type == StressEventType.LIQUIDITY_CRISIS:
            transformed_data = self._apply_liquidity_crisis(transformed_data,
                event)
        elif event.type == StressEventType.FLASH_CRASH:
            transformed_data = self._apply_flash_crash(transformed_data, event)
        elif event.type == StressEventType.CORRELATION_BREAKDOWN:
            transformed_data = self._apply_correlation_breakdown(
                transformed_data, event)
        elif event.type == StressEventType.TREND_REVERSAL:
            transformed_data = self._apply_trend_reversal(transformed_data,
                event)
        elif event.type == StressEventType.CUSTOM:
            transformed_data = self._apply_custom_transformation(
                transformed_data, event)
        else:
            logger.warning(f'Unknown event type: {event.type}, skipping')
        return transformed_data

    @with_exception_handling
    def _apply_historical_crash(self, data: Dict[str, pd.DataFrame], event:
        StressEvent) ->Dict[str, pd.DataFrame]:
        """Apply a historical crash pattern to data."""
        transformed_data = data.copy()
        affected_instruments = event.affected_instruments or list(data.keys())
        for instrument in affected_instruments:
            if instrument not in data:
                logger.warning(f'Instrument {instrument} not in data, skipping'
                    )
                continue
            df = data[instrument].copy()
            if event.historical_data_path:
                try:
                    historical_pattern = pd.read_csv(event.historical_data_path
                        )
                    pattern_pct_changes = historical_pattern['close'
                        ].pct_change().dropna()
                    pattern_length = len(pattern_pct_changes)
                    crash_start_idx = len(df) // 2
                    crash_end_idx = min(crash_start_idx + pattern_length,
                        len(df))
                    for i, pct_change in enumerate(pattern_pct_changes[:
                        crash_end_idx - crash_start_idx]):
                        idx = crash_start_idx + i
                        if idx < len(df):
                            df.loc[df.index[idx], 'close'] = df.loc[df.
                                index[idx - 1], 'close'] * (1 + pct_change)
                            if 'high' in df.columns and df.loc[df.index[idx
                                ], 'close'] > df.loc[df.index[idx], 'high']:
                                df.loc[df.index[idx], 'high'] = df.loc[df.
                                    index[idx], 'close']
                            if 'low' in df.columns and df.loc[df.index[idx],
                                'close'] < df.loc[df.index[idx], 'low']:
                                df.loc[df.index[idx], 'low'] = df.loc[df.
                                    index[idx], 'close']
                except Exception as e:
                    logger.error(f'Error loading historical pattern: {e}')
            else:
                crash_magnitude = event.parameters.get('magnitude', 0.25)
                crash_duration = event.parameters.get('duration', 10)
                crash_recovery = event.parameters.get('recovery', 20)
                crash_start_idx = len(df) // 2
                if crash_start_idx + crash_duration + crash_recovery < len(df):
                    pre_crash_price = df.loc[df.index[crash_start_idx - 1],
                        'close']
                    crash_pattern = np.linspace(0, -crash_magnitude,
                        crash_duration)
                    recovery_pattern = np.linspace(-crash_magnitude, 0,
                        crash_recovery)
                    for i, pct in enumerate(crash_pattern):
                        idx = crash_start_idx + i
                        if idx < len(df):
                            df.loc[df.index[idx], 'close'
                                ] = pre_crash_price * (1 + pct)
                            if 'high' in df.columns and df.loc[df.index[idx
                                ], 'close'] > df.loc[df.index[idx], 'high']:
                                df.loc[df.index[idx], 'high'] = df.loc[df.
                                    index[idx], 'close']
                            if 'low' in df.columns and df.loc[df.index[idx],
                                'close'] < df.loc[df.index[idx], 'low']:
                                df.loc[df.index[idx], 'low'] = df.loc[df.
                                    index[idx], 'close']
                    for i, pct in enumerate(recovery_pattern):
                        idx = crash_start_idx + crash_duration + i
                        if idx < len(df):
                            df.loc[df.index[idx], 'close'
                                ] = pre_crash_price * (1 + pct)
                            if 'high' in df.columns and df.loc[df.index[idx
                                ], 'close'] > df.loc[df.index[idx], 'high']:
                                df.loc[df.index[idx], 'high'] = df.loc[df.
                                    index[idx], 'close']
                            if 'low' in df.columns and df.loc[df.index[idx],
                                'close'] < df.loc[df.index[idx], 'low']:
                                df.loc[df.index[idx], 'low'] = df.loc[df.
                                    index[idx], 'close']
            transformed_data[instrument] = df
        return transformed_data

    def _apply_volatility_spike(self, data: Dict[str, pd.DataFrame], event:
        StressEvent) ->Dict[str, pd.DataFrame]:
        """Apply a volatility spike to data."""
        transformed_data = data.copy()
        affected_instruments = event.affected_instruments or list(data.keys())
        volatility_multiplier = event.parameters.get('volatility_multiplier',
            3.0)
        spike_duration = event.parameters.get('duration', 15)
        for instrument in affected_instruments:
            if instrument not in data:
                logger.warning(f'Instrument {instrument} not in data, skipping'
                    )
                continue
            df = data[instrument].copy()
            spike_start_idx = len(df) // 2
            if spike_start_idx + spike_duration < len(df):
                returns = df['close'].pct_change().dropna()
                current_volatility = returns.std()
                target_volatility = current_volatility * volatility_multiplier
                original_returns = returns.iloc[spike_start_idx:
                    spike_start_idx + spike_duration].values
                if np.std(original_returns) > 0:
                    scaled_returns = original_returns * (target_volatility /
                        np.std(original_returns))
                else:
                    scaled_returns = np.random.normal(np.mean(
                        original_returns), target_volatility, size=len(
                        original_returns))
                for i, ret in enumerate(scaled_returns):
                    idx = spike_start_idx + i + 1
                    if idx < len(df):
                        prev_price = df.loc[df.index[idx - 1], 'close']
                        new_price = prev_price * (1 + ret)
                        df.loc[df.index[idx], 'close'] = new_price
                        if 'high' in df.columns:
                            volatility_factor = volatility_multiplier
                            original_range = df.loc[df.index[idx], 'high'
                                ] - df.loc[df.index[idx], 'low']
                            new_range = original_range * volatility_factor
                            df.loc[df.index[idx], 'high'
                                ] = new_price + new_range / 2
                            df.loc[df.index[idx], 'low'
                                ] = new_price - new_range / 2
            transformed_data[instrument] = df
        return transformed_data

    def _apply_liquidity_crisis(self, data: Dict[str, pd.DataFrame], event:
        StressEvent) ->Dict[str, pd.DataFrame]:
        """Apply a liquidity crisis to data."""
        transformed_data = data.copy()
        affected_instruments = event.affected_instruments or list(data.keys())
        spread_multiplier = event.parameters.get('spread_multiplier', 5.0)
        slippage_multiplier = event.parameters.get('slippage_multiplier', 10.0)
        crisis_duration = event.parameters.get('duration', 20)
        for instrument in affected_instruments:
            if instrument not in data:
                logger.warning(f'Instrument {instrument} not in data, skipping'
                    )
                continue
            df = data[instrument].copy()
            crisis_start_idx = len(df) // 2
            if crisis_start_idx + crisis_duration < len(df):
                if 'spread' not in df.columns:
                    df['spread'] = df['close'] * 0.0002
                if 'slippage' not in df.columns:
                    df['slippage'] = df['spread'] * 0.5
                for i in range(crisis_duration):
                    idx = crisis_start_idx + i
                    if idx < len(df):
                        df.loc[df.index[idx], 'spread'] *= spread_multiplier
                        df.loc[df.index[idx], 'slippage'
                            ] *= slippage_multiplier
                        if i > 0:
                            volatility_factor = 1.5 + np.random.random() * 1.5
                            if idx > 0:
                                pct_change = df.loc[df.index[idx], 'close'
                                    ] / df.loc[df.index[idx - 1], 'close'] - 1
                                amplified_change = (pct_change *
                                    volatility_factor)
                                df.loc[df.index[idx], 'close'] = df.loc[df.
                                    index[idx - 1], 'close'] * (1 +
                                    amplified_change)
            transformed_data[instrument] = df
        return transformed_data

    def _apply_flash_crash(self, data: Dict[str, pd.DataFrame], event:
        StressEvent) ->Dict[str, pd.DataFrame]:
        """Apply a flash crash to data."""
        transformed_data = data.copy()
        affected_instruments = event.affected_instruments or list(data.keys())
        crash_magnitude = event.parameters.get('magnitude', 0.1)
        crash_duration = event.parameters.get('crash_duration', 3)
        recovery_duration = event.parameters.get('recovery_duration', 5)
        for instrument in affected_instruments:
            if instrument not in data:
                logger.warning(f'Instrument {instrument} not in data, skipping'
                    )
                continue
            df = data[instrument].copy()
            crash_start_idx = len(df) // 2
            if crash_start_idx + crash_duration + recovery_duration < len(df):
                pre_crash_price = df.loc[df.index[crash_start_idx - 1], 'close'
                    ]
                crash_points = np.linspace(0, 1, crash_duration)
                crash_pattern = -crash_magnitude * (np.exp(crash_points) /
                    np.exp(1))
                recovery_points = np.linspace(0, 1, recovery_duration)
                recovery_pattern = (-crash_magnitude + crash_magnitude *
                    recovery_points)
                for i, pct in enumerate(crash_pattern):
                    idx = crash_start_idx + i
                    if idx < len(df):
                        df.loc[df.index[idx], 'close'] = pre_crash_price * (
                            1 + pct)
                        if 'high' in df.columns:
                            if i == 0:
                                df.loc[df.index[idx], 'high'] = max(df.loc[
                                    df.index[idx], 'high'], pre_crash_price)
                            else:
                                range_factor = 3.0
                                mid_price = df.loc[df.index[idx], 'close']
                                original_range = df.loc[df.index[idx], 'high'
                                    ] - df.loc[df.index[idx], 'low']
                                new_range = original_range * range_factor
                                df.loc[df.index[idx], 'high'
                                    ] = mid_price + new_range * 0.3
                                df.loc[df.index[idx], 'low'
                                    ] = mid_price - new_range * 0.7
                crash_end_price = df.loc[df.index[crash_start_idx +
                    crash_duration - 1], 'close']
                for i, pct in enumerate(recovery_pattern):
                    idx = crash_start_idx + crash_duration + i
                    if idx < len(df):
                        df.loc[df.index[idx], 'close'] = pre_crash_price * (
                            1 + pct)
                        if 'high' in df.columns:
                            range_factor = 2.0
                            mid_price = df.loc[df.index[idx], 'close']
                            original_range = df.loc[df.index[idx], 'high'
                                ] - df.loc[df.index[idx], 'low']
                            new_range = original_range * range_factor
                            df.loc[df.index[idx], 'high'
                                ] = mid_price + new_range * 0.6
                            df.loc[df.index[idx], 'low'
                                ] = mid_price - new_range * 0.4
            transformed_data[instrument] = df
        return transformed_data

    @with_exception_handling
    def _apply_correlation_breakdown(self, data: Dict[str, pd.DataFrame],
        event: StressEvent) ->Dict[str, pd.DataFrame]:
        """Apply correlation breakdown to data."""
        transformed_data = data.copy()
        breakdown_duration = event.parameters.get('duration', 30)
        correlation_shift = event.parameters.get('correlation_shift', 0.8)
        if len(data) < 2:
            logger.warning(
                'Need at least 2 instruments for correlation breakdown, skipping'
                )
            return transformed_data
        min_length = min(len(df) for df in data.values())
        breakdown_start_idx = min_length // 2
        if breakdown_start_idx + breakdown_duration < min_length:
            returns_dict = {}
            for instrument, df in data.items():
                returns_dict[instrument] = df['close'].pct_change().fillna(0)
            returns_df = pd.DataFrame(returns_dict)
            pre_corr = returns_df.iloc[:breakdown_start_idx].corr()
            target_corr = pre_corr.copy()
            for i in range(len(pre_corr)):
                for j in range(i + 1, len(pre_corr)):
                    target_corr.iloc[i, j] = -pre_corr.iloc[i, j
                        ] * correlation_shift
                    target_corr.iloc[j, i] = target_corr.iloc[i, j]
            np.fill_diagonal(target_corr.values, 1.0)
            try:
                eigvals = np.linalg.eigvals(target_corr)
                if np.any(eigvals < 0):
                    logger.warning(
                        'Target correlation matrix is not positive semi-definite'
                        )
                    min_eigval = min(eigvals)
                    if min_eigval < 0:
                        np.fill_diagonal(target_corr.values, 1.0 -
                            min_eigval + 1e-06)
            except:
                logger.warning(
                    'Error validating correlation matrix, using simplified approach'
                    )
            instruments = list(data.keys())
            for i in range(breakdown_duration):
                idx = breakdown_start_idx + i
                if idx < min_length:
                    for j, instrument in enumerate(instruments):
                        orig_return = returns_dict[instrument].iloc[idx]
                        breakdown_progress = min(i / (breakdown_duration * 
                            0.7), 1.0)
                        df = data[instrument]
                        hist_volatility = returns_dict[instrument].iloc[:
                            breakdown_start_idx].std()
                        random_return = np.random.normal(0, hist_volatility *
                            1.5)
                        new_return = (1 - breakdown_progress
                            ) * orig_return + breakdown_progress * random_return
                        if idx > 0:
                            prev_price = df.loc[df.index[idx - 1], 'close']
                            df.loc[df.index[idx], 'close'] = prev_price * (
                                1 + new_return)
                        transformed_data[instrument] = df
        return transformed_data

    def _apply_trend_reversal(self, data: Dict[str, pd.DataFrame], event:
        StressEvent) ->Dict[str, pd.DataFrame]:
        """Apply a trend reversal to data."""
        transformed_data = data.copy()
        affected_instruments = event.affected_instruments or list(data.keys())
        reversal_factor = event.parameters.get('reversal_factor', -1.5)
        pre_period = event.parameters.get('pre_period', 30)
        post_period = event.parameters.get('post_period', 50)
        for instrument in affected_instruments:
            if instrument not in data:
                logger.warning(f'Instrument {instrument} not in data, skipping'
                    )
                continue
            df = data[instrument].copy()
            reversal_idx = len(df) // 2
            if (reversal_idx - pre_period >= 0 and reversal_idx +
                post_period < len(df)):
                pre_trend_prices = df['close'].iloc[reversal_idx -
                    pre_period:reversal_idx]
                pre_trend = pre_trend_prices.iloc[-1] / pre_trend_prices.iloc[0
                    ] - 1
                daily_trend = (1 + pre_trend) ** (1 / pre_period) - 1
                reversed_daily_trend = daily_trend * reversal_factor
                start_price = df.loc[df.index[reversal_idx], 'close']
                cumulative_factor = 1.0
                for i in range(post_period):
                    idx = reversal_idx + i + 1
                    if idx < len(df):
                        cumulative_factor *= 1 + reversed_daily_trend
                        noise_factor = 1 + (np.random.random() * 0.01 - 0.005)
                        new_price = (start_price * cumulative_factor *
                            noise_factor)
                        df.loc[df.index[idx], 'close'] = new_price
                        if 'high' in df.columns:
                            range_factor = 1.0 + abs(reversed_daily_trend) * 5
                            original_range = df.loc[df.index[idx], 'high'
                                ] - df.loc[df.index[idx], 'low']
                            new_range = original_range * range_factor
                            if reversed_daily_trend > 0:
                                df.loc[df.index[idx], 'high'
                                    ] = new_price + new_range * 0.6
                                df.loc[df.index[idx], 'low'
                                    ] = new_price - new_range * 0.4
                            else:
                                df.loc[df.index[idx], 'high'
                                    ] = new_price + new_range * 0.4
                                df.loc[df.index[idx], 'low'
                                    ] = new_price - new_range * 0.6
            transformed_data[instrument] = df
        return transformed_data

    @with_exception_handling
    def _apply_custom_transformation(self, data: Dict[str, pd.DataFrame],
        event: StressEvent) ->Dict[str, pd.DataFrame]:
        """Apply a custom transformation to data using the specified function."""
        transformed_data = data.copy()
        if not event.transformation_function:
            logger.warning(
                'No transformation function specified for custom event')
            return transformed_data
        try:
            from importlib import import_module
            module_path, function_name = event.transformation_function.split(
                ':')
            module = import_module(module_path)
            transform_func = getattr(module, function_name)
            result = transform_func(transformed_data, event.parameters)
            if isinstance(result, dict) and all(isinstance(v, pd.DataFrame) for
                v in result.values()):
                transformed_data = result
            else:
                logger.error(
                    f'Invalid result from transformation function: {type(result)}'
                    )
        except Exception as e:
            logger.error(f'Error applying custom transformation: {e}')
        return transformed_data

    def _create_backtest_config(self, scenario: StressTestScenario
        ) ->BacktestConfiguration:
        """Create a backtest configuration for the scenario."""
        start_date = self.config.base_period_start
        end_date = self.config.base_period_end
        if start_date is None or end_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
        backtest_config = BacktestConfiguration(strategy_name=self.config.
            strategy_name, strategy_parameters=self.config.
            strategy_parameters, start_date=start_date, end_date=end_date,
            initial_capital=self.config.initial_capital, instruments=self.
            config.instruments, data_timeframe=self.config.data_timeframe,
            slippage_model=self.config.slippage_model, slippage_settings=
            self.config.slippage_settings, commission_model=self.config.
            commission_model, commission_settings=self.config.
            commission_settings, position_sizing=self.config.
            position_sizing, position_sizing_settings=self.config.
            position_sizing_settings, risk_management_settings=self.config.
            risk_management_settings, data_settings=self.config.data_settings)
        return backtest_config

    def _calculate_comparative_metrics(self, scenario_results: Dict[str,
        BacktestResult]) ->Dict[str, Any]:
        """
        Calculate comparative metrics across scenarios.
        
        Args:
            scenario_results: Results of backtests for each scenario
            
        Returns:
            dict: Comparative metrics
        """
        if not scenario_results:
            return {}
        base_result = None
        for scenario in self.scenarios:
            if (scenario.is_base_scenario and scenario.scenario_id in
                scenario_results):
                base_result = scenario_results[scenario.scenario_id]
                break
        metrics = {}
        table_data = []
        for scenario in self.scenarios:
            if scenario.scenario_id not in scenario_results:
                continue
            result = scenario_results[scenario.scenario_id]
            row = {'scenario_id': scenario.scenario_id, 'name': scenario.
                name, 'is_base': scenario.is_base_scenario, 'final_value':
                result.final_portfolio_value, 'total_return': result.
                total_return, 'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown, 'win_rate': result.
                win_rate, 'trades': result.trades_total}
            if base_result and not scenario.is_base_scenario:
                row['return_delta'
                    ] = result.total_return - base_result.total_return
                row['return_delta_pct'] = ((1 + result.total_return) / (1 +
                    base_result.total_return) - 1) * 100
                row['drawdown_delta'
                    ] = result.max_drawdown - base_result.max_drawdown
            table_data.append(row)
        metrics['comparison_table'] = table_data
        if table_data:
            worst_return = min(row['total_return'] for row in table_data)
            worst_scenario = next(row['name'] for row in table_data if row[
                'total_return'] == worst_return)
            metrics['worst_case'] = {'return': worst_return, 'scenario':
                worst_scenario}
            if base_result:
                metrics['worst_case']['delta_from_base'
                    ] = worst_return - base_result.total_return
                metrics['worst_case']['delta_pct_from_base'] = ((1 +
                    worst_return) / (1 + base_result.total_return) - 1) * 100
        if base_result and len(table_data) > 1:
            non_base_returns = [row['total_return'] for row in table_data if
                not row.get('is_base', False)]
            avg_stress_return = sum(non_base_returns) / len(non_base_returns
                ) if non_base_returns else 0
            if base_result.total_return > 0:
                robustness = max(0, min(100, avg_stress_return /
                    base_result.total_return * 100))
            elif avg_stress_return >= base_result.total_return:
                robustness = 100
            else:
                robustness = max(0, 100 * (1 + avg_stress_return) / (1 +
                    base_result.total_return))
            metrics['robustness_score'] = robustness
            worst_case_impact = abs(metrics['worst_case'][
                'delta_pct_from_base']) if 'delta_pct_from_base' in metrics[
                'worst_case'] else 0
            if worst_case_impact < 10:
                risk_rating = 1
            elif worst_case_impact < 25:
                risk_rating = 2
            elif worst_case_impact < 50:
                risk_rating = 3
            elif worst_case_impact < 75:
                risk_rating = 4
            else:
                risk_rating = 5
            metrics['risk_rating'] = risk_rating
        return metrics
