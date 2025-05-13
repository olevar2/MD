"""
Reinforcement Learning Agent Benchmarking System

This module provides comprehensive benchmarking capabilities for evaluating
and comparing reinforcement learning agents across different market conditions,
time periods, and trading scenarios.
"""
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from dataclasses import dataclass, field
import logging
import uuid
from trading_gateway_service.simulation.reinforcement_learning.environment_generator import ForexTradingEnvironment, EnvConfiguration, EnvironmentGenerator
from trading_gateway_service.simulation.advanced_market_regime_simulator import AdvancedMarketRegimeSimulator, MarketCondition, SimulationScenario
from trading_gateway_service.simulation.enhanced_market_condition_generator import EnhancedMarketConditionGenerator
from trading_gateway_service.simulation.forex_broker_simulator import ForexBrokerSimulator
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)
from trading_gateway_service.error.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class BenchmarkCategory(str, Enum):
    """Categories of benchmark tests."""
    STANDARD = 'standard'
    TRENDING = 'trending'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    NEWS_DRIVEN = 'news_driven'
    LOW_LIQUIDITY = 'low_liquidity'
    FLASH_EVENTS = 'flash_events'
    MIXED = 'mixed'
    CUSTOM = 'custom'


@dataclass
class BenchmarkScenario:
    """A scenario for benchmarking."""
    id: str
    name: str
    description: str
    category: BenchmarkCategory
    simulation_scenario: SimulationScenario
    duration_days: int = 1
    repetitions: int = 1
    metrics: Dict[str, float] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    benchmark_id: str
    agent_id: str
    scenario_id: str
    run_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    success_rate: float
    duration_seconds: float
    notes: Optional[str] = None


class RLAgentBenchmark:
    """
    System for benchmarking reinforcement learning agents against standardized
    market scenarios and comparing performance across different conditions.
    """

    def __init__(self, market_simulator: Optional[
        AdvancedMarketRegimeSimulator]=None, market_generator: Optional[
        EnhancedMarketConditionGenerator]=None, output_dir: Optional[str]=
        None, random_seed: Optional[int]=None):
        """
        Initialize the benchmark system.
        
        Args:
            market_simulator: Advanced market regime simulator
            market_generator: Enhanced market condition generator
            output_dir: Directory to save benchmark results
            random_seed: Optional seed for reproducibility
        """
        self.broker_simulator = ForexBrokerSimulator()
        if market_simulator is None:
            self.market_simulator = AdvancedMarketRegimeSimulator(
                broker_simulator=self.broker_simulator)
        else:
            self.market_simulator = market_simulator
        if market_generator is None:
            self.market_generator = EnhancedMarketConditionGenerator(
                broker_simulator=self.broker_simulator)
        else:
            self.market_generator = market_generator
        self.output_dir = output_dir or os.path.join(os.getcwd(),
            'benchmark_results')
        os.makedirs(self.output_dir, exist_ok=True)
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.scenarios = {}
        self._initialize_standard_scenarios()
        self.results = {}

    def _initialize_standard_scenarios(self):
        """Initialize a standard set of benchmark scenarios."""
        self._add_trending_scenarios()
        self._add_ranging_scenarios()
        self._add_volatile_scenarios()
        self._add_news_driven_scenarios()
        self._add_low_liquidity_scenarios()
        self._add_flash_event_scenarios()
        self._add_mixed_scenarios()

    def _add_trending_scenarios(self):
        """Add trending market scenarios."""
        scenario = self.market_generator.generate_market_scenario(symbol=
            'EUR/USD', condition=MarketCondition.TRENDING_BULLISH, duration
            =timedelta(days=2))
        self.add_scenario(name='Strong Bullish Trend', description=
            'Steady bullish trend with clear directional movement',
            category=BenchmarkCategory.TRENDING, simulation_scenario=
            scenario, duration_days=2)
        scenario = self.market_generator.generate_market_scenario(symbol=
            'EUR/USD', condition=MarketCondition.TRENDING_BEARISH, duration
            =timedelta(days=2))
        self.add_scenario(name='Strong Bearish Trend', description=
            'Steady bearish trend with clear directional movement',
            category=BenchmarkCategory.TRENDING, simulation_scenario=
            scenario, duration_days=2)
        scenario = self.market_generator.generate_market_scenario(symbol=
            'EUR/USD', condition=MarketCondition.TRENDING_BULLISH, duration
            =timedelta(days=3), pattern=None, anomalies=None)
        self.add_scenario(name='Trend with Pullbacks', description=
            'Bullish trend with periodic pullbacks', category=
            BenchmarkCategory.TRENDING, simulation_scenario=scenario,
            duration_days=3)

    def _add_ranging_scenarios(self):
        """Add ranging market scenarios."""
        scenario = self.market_generator.generate_market_scenario(symbol=
            'EUR/USD', condition=MarketCondition.RANGING_NARROW, duration=
            timedelta(days=2))
        self.add_scenario(name='Narrow Range', description=
            'Tight consolidation with minimal volatility', category=
            BenchmarkCategory.RANGING, simulation_scenario=scenario,
            duration_days=2)
        scenario = self.market_generator.generate_market_scenario(symbol=
            'EUR/USD', condition=MarketCondition.RANGING_WIDE, duration=
            timedelta(days=2))
        self.add_scenario(name='Wide Range', description=
            'Ranging market with higher volatility and wider boundaries',
            category=BenchmarkCategory.RANGING, simulation_scenario=
            scenario, duration_days=2)

    def _add_volatile_scenarios(self):
        """Add volatile market scenarios."""
        scenario = self.market_generator.generate_market_scenario(symbol=
            'EUR/USD', condition=MarketCondition.HIGH_VOLATILITY, duration=
            timedelta(days=1))
        self.add_scenario(name='High Volatility', description=
            'Highly volatile market with erratic price movement', category=
            BenchmarkCategory.VOLATILE, simulation_scenario=scenario,
            duration_days=1)
        scenario = self.market_generator.generate_market_scenario(symbol=
            'EUR/USD', condition=MarketCondition.BREAKOUT_BULLISH, duration
            =timedelta(days=1))
        self.add_scenario(name='Bullish Breakout', description=
            'Bullish breakout from consolidation with increased volatility',
            category=BenchmarkCategory.VOLATILE, simulation_scenario=
            scenario, duration_days=1)

    def _add_news_driven_scenarios(self):
        """Add news-driven market scenarios."""
        scenario = self.market_generator.generate_market_scenario(symbol=
            'EUR/USD', condition=MarketCondition.NEWS_REACTION, duration=
            timedelta(days=1))
        self.add_scenario(name='Major News Event', description=
            'Market reaction to high-impact economic news', category=
            BenchmarkCategory.NEWS_DRIVEN, simulation_scenario=scenario,
            duration_days=1)

    def _add_low_liquidity_scenarios(self):
        """Add low liquidity market scenarios."""
        scenario = self.market_generator.generate_market_scenario(symbol=
            'EUR/USD', condition=MarketCondition.LIQUIDITY_GAP, duration=
            timedelta(days=1))
        self.add_scenario(name='Low Liquidity', description=
            'Thin market with potential gaps and slippage', category=
            BenchmarkCategory.LOW_LIQUIDITY, simulation_scenario=scenario,
            duration_days=1)

    def _add_flash_event_scenarios(self):
        """Add flash crash/spike scenarios."""
        scenario = self.market_generator.generate_market_scenario(symbol=
            'EUR/USD', condition=MarketCondition.FLASH_CRASH, duration=
            timedelta(days=1))
        self.add_scenario(name='Flash Crash', description=
            'Sudden severe price drop with quick partial recovery',
            category=BenchmarkCategory.FLASH_EVENTS, simulation_scenario=
            scenario, duration_days=1)
        scenario = self.market_generator.generate_market_scenario(symbol=
            'EUR/USD', condition=MarketCondition.FLASH_SPIKE, duration=
            timedelta(days=1))
        self.add_scenario(name='Flash Spike', description=
            'Sudden severe price spike with quick partial retracement',
            category=BenchmarkCategory.FLASH_EVENTS, simulation_scenario=
            scenario, duration_days=1)

    def _add_mixed_scenarios(self):
        """Add mixed market scenarios."""
        scenario = self.market_generator.generate_market_scenario(symbol=
            'EUR/USD', condition=MarketCondition.NORMAL, duration=timedelta
            (days=5))
        self.add_scenario(name='Mixed Market Conditions', description=
            'Sequence of different market conditions in a single scenario',
            category=BenchmarkCategory.MIXED, simulation_scenario=scenario,
            duration_days=5)

    def add_scenario(self, name: str, description: str, category:
        BenchmarkCategory, simulation_scenario: SimulationScenario,
        duration_days: int=1, repetitions: int=1, weight: float=1.0) ->str:
        """
        Add a new benchmark scenario.
        
        Args:
            name: Name of the scenario
            description: Description of the scenario
            category: Category of the benchmark
            simulation_scenario: SimulationScenario object
            duration_days: Duration in days
            repetitions: Number of times to repeat
            weight: Weight in overall scoring
            
        Returns:
            ID of the created scenario
        """
        scenario_id = str(uuid.uuid4())
        scenario = BenchmarkScenario(id=scenario_id, name=name, description
            =description, category=category, simulation_scenario=
            simulation_scenario, duration_days=duration_days, repetitions=
            repetitions, weight=weight)
        self.scenarios[scenario_id] = scenario
        return scenario_id

    @with_broker_api_resilience('create_custom_scenario')
    def create_custom_scenario(self, name: str, symbol: str, condition:
        MarketCondition, duration_days: int=1, volatility_factor: float=1.0,
        description: Optional[str]=None) ->str:
        """
        Create and add a custom benchmark scenario.
        
        Args:
            name: Name of the scenario
            symbol: Trading symbol
            condition: Market condition
            duration_days: Duration in days
            volatility_factor: Volatility scaling factor
            description: Optional description
            
        Returns:
            ID of the created scenario
        """
        simulation_scenario = self.market_generator.generate_market_scenario(
            symbol=symbol, condition=condition, duration=timedelta(days=
            duration_days))
        simulation_scenario.volatility_factor *= volatility_factor
        if description is None:
            description = (
                f'Custom {condition.value} scenario for {symbol} with {volatility_factor}x volatility'
                )
        return self.add_scenario(name=name, description=description,
            category=BenchmarkCategory.CUSTOM, simulation_scenario=
            simulation_scenario, duration_days=duration_days)

    def remove_scenario(self, scenario_id: str) ->bool:
        """
        Remove a scenario from the benchmark system.
        
        Args:
            scenario_id: ID of the scenario to remove
            
        Returns:
            True if removal was successful
        """
        if scenario_id in self.scenarios:
            del self.scenarios[scenario_id]
            return True
        return False

    @with_broker_api_resilience('get_scenarios_by_category')
    def get_scenarios_by_category(self, category: BenchmarkCategory) ->List[
        BenchmarkScenario]:
        """
        Get all scenarios in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of scenarios in the category
        """
        return [scenario for scenario in self.scenarios.values() if 
            scenario.category == category]

    @with_broker_api_resilience('create_environment_from_scenario')
    def create_environment_from_scenario(self, scenario: BenchmarkScenario,
        config: Optional[EnvConfiguration]=None) ->ForexTradingEnvironment:
        """
        Create a ForexTradingEnvironment from a benchmark scenario.
        
        Args:
            scenario: Benchmark scenario
            config: Optional environment configuration
            
        Returns:
            Configured environment
        """
        self.market_simulator.apply_scenario(scenario.simulation_scenario, 
            datetime.now() - timedelta(days=30))
        if config is None:
            config = EnvConfiguration(symbols=[scenario.simulation_scenario
                .symbol], timeframe='1h', max_episode_steps=24 * scenario.
                duration_days)
        env = EnvironmentGenerator.create_environment(market_simulator=self
            .market_simulator, broker_simulator=ForexBrokerSimulator(),
            config=config)
        return env

    def run_benchmark(self, agent, scenario_id: Optional[str]=None,
        category: Optional[BenchmarkCategory]=None, num_episodes: int=1
        ) ->Dict[str, List[BenchmarkResult]]:
        """
        Run benchmark tests for an agent on specified scenarios.
        
        Args:
            agent: The agent to benchmark (must have predict method)
            scenario_id: Optional specific scenario ID to run
            category: Optional category of scenarios to run
            num_episodes: Number of episodes per scenario
            
        Returns:
            Dictionary of scenario IDs to lists of benchmark results
        """
        benchmark_id = str(uuid.uuid4())
        agent_id = getattr(agent, 'id', str(uuid.uuid4())[:8])
        scenarios_to_run = []
        if scenario_id is not None:
            if scenario_id in self.scenarios:
                scenarios_to_run.append(self.scenarios[scenario_id])
        elif category is not None:
            scenarios_to_run.extend(self.get_scenarios_by_category(category))
        else:
            scenarios_to_run.extend(self.scenarios.values())
        if not scenarios_to_run:
            logger.warning('No scenarios to run in benchmark')
            return {}
        results = {}
        for scenario in scenarios_to_run:
            scenario_results = []
            env = self.create_environment_from_scenario(scenario)
            for episode in range(num_episodes):
                logger.info(
                    f'Running benchmark {benchmark_id} - Scenario: {scenario.name} ({episode + 1}/{num_episodes})'
                    )
                run_id = f'{benchmark_id}_{scenario.id}_{episode}'
                result = self._run_single_episode(agent, env, run_id,
                    scenario.id, agent_id)
                scenario_results.append(result)
            results[scenario.id] = scenario_results
            env.close()
        self.results[benchmark_id] = results
        self._save_benchmark_results(benchmark_id, agent_id)
        return results

    def _run_single_episode(self, agent, env: ForexTradingEnvironment,
        run_id: str, scenario_id: str, agent_id: str) ->BenchmarkResult:
        """
        Run a single benchmark episode.
        
        Args:
            agent: The agent to benchmark
            env: The environment to run in
            run_id: Unique ID for this run
            scenario_id: ID of the scenario
            agent_id: ID of the agent
            
        Returns:
            BenchmarkResult for this episode
        """
        start_time = datetime.now()
        observation = env.reset()
        done = False
        equity_curve = []
        while not done:
            action = agent.predict(observation)
            observation, reward, done, info = env.step(action)
            if 'equity' in info:
                equity_curve.append(info['equity'])
        duration_seconds = (datetime.now() - start_time).total_seconds()
        metrics = env.get_performance_summary()
        trades = env.trades
        success_rate = 0.0
        if metrics['returns_pct'] > 0 and metrics['sharpe_ratio'] > 0:
            success_rate = min(1.0, metrics['returns_pct'] / 5.0)
        result = BenchmarkResult(benchmark_id=run_id.split('_')[0],
            agent_id=agent_id, scenario_id=scenario_id, run_id=run_id,
            timestamp=datetime.now(), metrics=metrics, trades=trades,
            equity_curve=equity_curve, success_rate=success_rate,
            duration_seconds=duration_seconds)
        return result

    def _save_benchmark_results(self, benchmark_id: str, agent_id: str):
        """
        Save benchmark results to disk.
        
        Args:
            benchmark_id: ID of the benchmark
            agent_id: ID of the agent
        """
        if benchmark_id not in self.results:
            return
        output_dir = os.path.join(self.output_dir, benchmark_id)
        os.makedirs(output_dir, exist_ok=True)
        summary = self.get_benchmark_summary(benchmark_id)
        with open(os.path.join(output_dir, f'{agent_id}_summary.json'), 'w'
            ) as f:
            serializable_summary = {'agent_id': agent_id, 'benchmark_id':
                benchmark_id, 'timestamp': datetime.now().isoformat(),
                'overall_score': summary.get('overall_score', 0.0),
                'category_scores': summary.get('category_scores', {}),
                'scenario_results': {str(k): {'name': v.get('name', ''),
                'category': v.get('category', ''), 'avg_returns': v.get(
                'avg_returns', 0.0), 'avg_sharpe': v.get('avg_sharpe', 0.0),
                'success_rate': v.get('success_rate', 0.0)} for k, v in
                summary.get('scenario_results', {}).items()}}
            json.dump(serializable_summary, f, indent=2)
        self._generate_benchmark_visualizations(benchmark_id, agent_id,
            output_dir)

    def _generate_benchmark_visualizations(self, benchmark_id: str,
        agent_id: str, output_dir: str):
        """
        Generate and save visualizations of benchmark results.
        
        Args:
            benchmark_id: ID of the benchmark
            agent_id: ID of the agent
            output_dir: Directory to save visualizations
        """
        if benchmark_id not in self.results:
            return
        plt.style.use('seaborn-v0_8-darkgrid')
        summary = self.get_benchmark_summary(benchmark_id)
        category_scores = summary.get('category_scores', {})
        if category_scores:
            categories = list(category_scores.keys())
            scores = [category_scores[c] for c in categories]
            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, scores, color='skyblue')
            for bar in bars:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(
                    ) + 0.02, f'{bar.get_height():.2f}', ha='center', va=
                    'bottom')
            plt.xlabel('Market Condition Category')
            plt.ylabel('Score (0-1)')
            plt.title(f'Agent Performance by Market Category')
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,
                f'{agent_id}_category_performance.png'))
            plt.close()
        for scenario_id, results_list in self.results[benchmark_id].items():
            scenario = self.scenarios.get(scenario_id)
            if not scenario or not results_list:
                continue
            plt.figure(figsize=(12, 6))
            for i, result in enumerate(results_list):
                if not result.equity_curve:
                    continue
                norm_equity = np.array(result.equity_curve
                    ) / result.equity_curve[0] * 100
                plt.plot(norm_equity, label=f'Episode {i + 1}')
            plt.xlabel('Steps')
            plt.ylabel('Equity (%)')
            plt.title(f'Equity Curves for {scenario.name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,
                f'{agent_id}_{scenario_id}_equity.png'))
            plt.close()

    @with_broker_api_resilience('get_benchmark_summary')
    def get_benchmark_summary(self, benchmark_id: str) ->Dict[str, Any]:
        """
        Get a summary of benchmark results.
        
        Args:
            benchmark_id: ID of the benchmark run
            
        Returns:
            Dictionary with benchmark summary
        """
        if benchmark_id not in self.results:
            return {}
        benchmark_results = self.results[benchmark_id]
        scenario_summaries = {}
        category_results = {}
        for scenario_id, results_list in benchmark_results.items():
            if scenario_id not in self.scenarios or not results_list:
                continue
            scenario = self.scenarios[scenario_id]
            category = scenario.category
            avg_returns = np.mean([r.metrics.get('returns_pct', 0.0) for r in
                results_list])
            avg_sharpe = np.mean([r.metrics.get('sharpe_ratio', 0.0) for r in
                results_list])
            avg_drawdown = np.mean([r.metrics.get('max_drawdown_pct', 0.0) for
                r in results_list])
            avg_success = np.mean([r.success_rate for r in results_list])
            scenario_summaries[scenario_id] = {'name': scenario.name,
                'category': scenario.category.value, 'avg_returns':
                avg_returns, 'avg_sharpe': avg_sharpe, 'avg_drawdown':
                avg_drawdown, 'success_rate': avg_success, 'weight':
                scenario.weight}
            if category not in category_results:
                category_results[category] = {'total_weight': 0.0,
                    'weighted_score': 0.0}
            scenario_score = (max(0, min(1, avg_returns / 10.0)) + max(0,
                min(1, avg_sharpe / 1.5))) / 2
            category_results[category]['weighted_score'
                ] += scenario_score * scenario.weight
            category_results[category]['total_weight'] += scenario.weight
        category_scores = {}
        for category, data in category_results.items():
            if data['total_weight'] > 0:
                category_scores[category.value] = data['weighted_score'
                    ] / data['total_weight']
            else:
                category_scores[category.value] = 0.0
        category_weights = {BenchmarkCategory.STANDARD: 1.0,
            BenchmarkCategory.TRENDING: 1.0, BenchmarkCategory.RANGING: 1.0,
            BenchmarkCategory.VOLATILE: 1.2, BenchmarkCategory.NEWS_DRIVEN:
            1.1, BenchmarkCategory.LOW_LIQUIDITY: 1.3, BenchmarkCategory.
            FLASH_EVENTS: 1.4, BenchmarkCategory.MIXED: 1.5,
            BenchmarkCategory.CUSTOM: 1.0}
        weighted_sum = 0.0
        total_weights = 0.0
        for category, score in category_scores.items():
            weight = category_weights.get(BenchmarkCategory(category), 1.0)
            weighted_sum += score * weight
            total_weights += weight
        overall_score = (weighted_sum / total_weights if total_weights > 0 else
            0.0)
        return {'benchmark_id': benchmark_id, 'overall_score':
            overall_score, 'category_scores': category_scores,
            'scenario_results': scenario_summaries}

    def compare_agents(self, agents_dict: Dict[str, Any], category:
        Optional[BenchmarkCategory]=None, num_episodes: int=3) ->Dict[str, Any
        ]:
        """
        Run benchmarks to compare multiple agents.
        
        Args:
            agents_dict: Dictionary of agent_id -> agent
            category: Optional specific category to benchmark
            num_episodes: Number of episodes per scenario
            
        Returns:
            Dictionary with comparison results
        """
        comparison_id = str(uuid.uuid4())
        comparison_results = {}
        for agent_id, agent in agents_dict.items():
            logger.info(f'Benchmarking agent: {agent_id}')
            results = self.run_benchmark(agent=agent, category=category,
                num_episodes=num_episodes)
            benchmark_id = list(self.results.keys())[-1]
            summary = self.get_benchmark_summary(benchmark_id)
            comparison_results[agent_id] = {'benchmark_id': benchmark_id,
                'overall_score': summary.get('overall_score', 0.0),
                'category_scores': summary.get('category_scores', {})}
        output_dir = os.path.join(self.output_dir,
            f'comparison_{comparison_id}')
        os.makedirs(output_dir, exist_ok=True)
        self._generate_comparison_visualizations(comparison_results, output_dir
            )
        with open(os.path.join(output_dir, 'comparison_results.json'), 'w'
            ) as f:
            json.dump({'comparison_id': comparison_id, 'timestamp':
                datetime.now().isoformat(), 'num_episodes': num_episodes,
                'category': category.value if category else 'all',
                'agent_results': comparison_results}, f, indent=2)
        return comparison_results

    def _generate_comparison_visualizations(self, comparison_results: Dict[
        str, Dict[str, Any]], output_dir: str):
        """
        Generate visualizations comparing multiple agents.
        
        Args:
            comparison_results: Results from compare_agents
            output_dir: Directory to save visualizations
        """
        if not comparison_results:
            return
        plt.style.use('seaborn-v0_8-darkgrid')
        agent_ids = list(comparison_results.keys())
        overall_scores = [results.get('overall_score', 0.0) for results in
            comparison_results.values()]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(agent_ids, overall_scores, color='skyblue')
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 
                0.02, f'{bar.get_height():.2f}', ha='center', va='bottom')
        plt.xlabel('Agent')
        plt.ylabel('Overall Score (0-1)')
        plt.title('Agent Performance Comparison')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_comparison.png'))
        plt.close()
        all_categories = set()
        for results in comparison_results.values():
            all_categories.update(results.get('category_scores', {}).keys())
        if all_categories:
            all_categories = sorted(all_categories)
            data = []
            for agent_id, results in comparison_results.items():
                category_scores = results.get('category_scores', {})
                agent_data = [category_scores.get(category, 0.0) for
                    category in all_categories]
                data.append(agent_data)
            df = pd.DataFrame(data, columns=all_categories, index=agent_ids)
            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(df, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
            plt.xlabel('Market Category')
            plt.ylabel('Agent')
            plt.title('Agent Performance by Market Category')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'category_comparison.png'))
            plt.close()

    @with_broker_api_resilience('get_performance_report')
    def get_performance_report(self, benchmark_id: str) ->str:
        """
        Generate a detailed performance report for a benchmark run.
        
        Args:
            benchmark_id: ID of the benchmark
            
        Returns:
            Markdown-formatted report
        """
        if benchmark_id not in self.results:
            return 'Benchmark ID not found.'
        summary = self.get_benchmark_summary(benchmark_id)
        report = '# RL Agent Benchmark Report\n\n'
        report += f'**Benchmark ID:** {benchmark_id}  \n'
        report += (
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        report += (
            f"**Overall Score:** {summary.get('overall_score', 0.0):.2f}/1.0  \n\n"
            )
        report += '## Performance by Market Category\n\n'
        report += '| Category | Score |\n'
        report += '| --- | --- |\n'
        category_scores = summary.get('category_scores', {})
        for category, score in category_scores.items():
            report += f'| {category} | {score:.2f} |\n'
        report += '\n'
        report += '## Performance by Scenario\n\n'
        report += """| Scenario | Category | Returns (%) | Sharpe Ratio | Max Drawdown (%) | Success Rate |
"""
        report += '| --- | --- | --- | --- | --- | --- |\n'
        scenario_results = summary.get('scenario_results', {})
        for scenario_id, results in scenario_results.items():
            report += f"| {results.get('name', '')} | "
            report += f"{results.get('category', '')} | "
            report += f"{results.get('avg_returns', 0.0):.2f}% | "
            report += f"{results.get('avg_sharpe', 0.0):.2f} | "
            report += f"{results.get('avg_drawdown', 0.0):.2f}% | "
            report += f"{results.get('success_rate', 0.0):.2f} |\n"
        return report

    @with_exception_handling
    def export_scenarios(self, file_path: str) ->bool:
        """
        Export benchmark scenarios to a JSON file.
        
        Args:
            file_path: Path to save the file
            
        Returns:
            True if export was successful
        """
        try:
            scenarios_data = {}
            for scenario_id, scenario in self.scenarios.items():
                sim_scenario = scenario.simulation_scenario
                sim_scenario_dict = {'name': sim_scenario.name, 'symbol':
                    sim_scenario.symbol, 'duration_seconds': sim_scenario.
                    duration.total_seconds(), 'market_condition':
                    sim_scenario.market_condition.value,
                    'liquidity_profile': sim_scenario.liquidity_profile.
                    value, 'volatility_factor': sim_scenario.
                    volatility_factor, 'spread_factor': sim_scenario.
                    spread_factor, 'trend_strength': sim_scenario.
                    trend_strength, 'mean_reversion_strength': sim_scenario
                    .mean_reversion_strength, 'price_jump_probability':
                    sim_scenario.price_jump_probability,
                    'price_jump_magnitude': sim_scenario.
                    price_jump_magnitude, 'special_events': sim_scenario.
                    special_events, 'description': sim_scenario.description}
                scenarios_data[scenario_id] = {'name': scenario.name,
                    'description': scenario.description, 'category':
                    scenario.category.value, 'duration_days': scenario.
                    duration_days, 'repetitions': scenario.repetitions,
                    'weight': scenario.weight, 'simulation_scenario':
                    sim_scenario_dict}
            with open(file_path, 'w') as f:
                json.dump(scenarios_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f'Error exporting scenarios: {e}')
            return False

    @with_exception_handling
    def import_scenarios(self, file_path: str) ->bool:
        """
        Import benchmark scenarios from a JSON file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if import was successful
        """
        try:
            with open(file_path, 'r') as f:
                scenarios_data = json.load(f)
            self.scenarios = {}
            for scenario_id, data in scenarios_data.items():
                sim_data = data.get('simulation_scenario', {})
                market_condition = MarketCondition(sim_data.get(
                    'market_condition', 'normal'))
                liquidity_profile_value = sim_data.get('liquidity_profile',
                    'medium')
                duration_seconds = sim_data.get('duration_seconds', 86400)
                simulation_scenario = (self.market_generator.
                    generate_market_scenario(symbol=sim_data.get('symbol',
                    'EUR/USD'), condition=market_condition, duration=
                    timedelta(seconds=duration_seconds)))
                self.add_scenario(name=data.get('name', ''), description=
                    data.get('description', ''), category=BenchmarkCategory
                    (data.get('category', 'custom')), simulation_scenario=
                    simulation_scenario, duration_days=data.get(
                    'duration_days', 1), repetitions=data.get('repetitions',
                    1), weight=data.get('weight', 1.0))
            return True
        except Exception as e:
            logger.error(f'Error importing scenarios: {e}')
            return False
