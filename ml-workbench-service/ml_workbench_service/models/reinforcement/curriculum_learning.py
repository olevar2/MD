"""
Curriculum Learning Framework for Reinforcement Learning.

This module provides tools for progressive training of RL agents using
increasingly difficult market conditions and scenarios.
"""
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import json
import logging
import matplotlib.pyplot as plt
from enum import Enum
from ml_workbench_service.rl_model_factory import RLModelFactory, RLAlgorithm
from ml_workbench_service.models.reinforcement.enhanced_rl_env import EnhancedForexTradingEnv
from trading_gateway_service.simulation.forex_broker_simulator import ForexBrokerSimulator
from trading_gateway_service.simulation.news_sentiment_simulator import NewsAndSentimentSimulator
from trading_gateway_service.simulation.advanced_market_regime_simulator import AdvancedMarketRegimeSimulator, SimulationScenario
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TrainingStage(str, Enum):
    """Training stages for curriculum learning."""
    BASIC = 'basic'
    INTERMEDIATE = 'intermediate'
    ADVANCED = 'advanced'
    EXPERT = 'expert'
    MASTER = 'master'


class CurriculumProgressionCriteria(str, Enum):
    """Criteria for progressing to the next curriculum level."""
    EPISODES_COMPLETED = 'episodes_completed'
    PERFORMANCE_THRESHOLD = 'performance_threshold'
    MANUAL = 'manual'
    AUTO_ADAPTIVE = 'auto_adaptive'


class CurriculumLearningFramework:
    """
    Framework for progressive training of RL agents with increasing difficulty.
    
    This framework helps agents learn gradually by starting with simple scenarios
    and progressing to more complex and realistic market conditions as they improve.
    It includes:
    - Multi-stage training curriculum
    - Performance-based progression
    - Agent evaluation on standardized scenarios
    - Stage-specific environments and reward functions
    """

    def __init__(self, rl_model_factory: RLModelFactory, broker_simulator:
        ForexBrokerSimulator, market_simulator:
        AdvancedMarketRegimeSimulator, news_simulator: Optional[
        NewsAndSentimentSimulator]=None, base_symbol: str='EUR/USD',
        save_dir: str='./models/curriculum/', tensorboard_log_dir: str=
        './logs/curriculum/', progression_criteria:
        CurriculumProgressionCriteria=CurriculumProgressionCriteria.
        PERFORMANCE_THRESHOLD, random_seed: Optional[int]=None):
        """
        Initialize the curriculum learning framework.
        
        Args:
            rl_model_factory: Factory for creating RL models
            broker_simulator: Forex broker simulator
            market_simulator: Advanced market regime simulator
            news_simulator: News and sentiment simulator
            base_symbol: Primary symbol for training
            save_dir: Directory to save models
            tensorboard_log_dir: Directory for tensorboard logs
            progression_criteria: Criteria for advancing curriculum
            random_seed: Random seed for reproducibility
        """
        self.rl_model_factory = rl_model_factory
        self.broker_simulator = broker_simulator
        self.market_simulator = market_simulator
        self.news_simulator = news_simulator
        self.base_symbol = base_symbol
        self.save_dir = save_dir
        self.tensorboard_log_dir = tensorboard_log_dir
        self.progression_criteria = progression_criteria
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.current_stage = TrainingStage.BASIC
        self.current_level = 1
        self.episodes_completed = 0
        self.stage_episodes = 0
        self.current_model = None
        self.training_history = []
        self.stage_configs = self._create_stage_configs()
        self.curriculum = self._create_curriculum()

    def _create_stage_configs(self) ->Dict[TrainingStage, Dict[str, Any]]:
        """
        Create configuration for each training stage.
        
        Returns:
            Dictionary of stage configurations
        """
        return {TrainingStage.BASIC: {'description':
            'Basic training with simple market conditions', 'min_level': 1,
            'max_level': 2, 'min_episodes': 100, 'performance_threshold': 
            0.3, 'reward_scale': 1.0, 'include_news': False,
            'observation_features': ['price', 'volume', 'basic_indicators'],
            'env_params': {'lookback_periods': 20, 'reward_mode': 'pnl',
            'trading_fee_percent': 0.001, 'curriculum_level': 1,
            'include_news_sentiment': False}}, TrainingStage.INTERMEDIATE:
            {'description':
            'Intermediate training with varied market conditions',
            'min_level': 2, 'max_level': 3, 'min_episodes': 200,
            'performance_threshold': 0.5, 'reward_scale': 1.0,
            'include_news': True, 'observation_features': ['price',
            'volume', 'basic_indicators', 'advanced_indicators'],
            'env_params': {'lookback_periods': 30, 'reward_mode':
            'risk_adjusted', 'trading_fee_percent': 0.0015,
            'curriculum_level': 2, 'include_news_sentiment': True}},
            TrainingStage.ADVANCED: {'description':
            'Advanced training with complex scenarios', 'min_level': 3,
            'max_level': 4, 'min_episodes': 300, 'performance_threshold': 
            0.7, 'reward_scale': 1.0, 'include_news': True,
            'observation_features': ['price', 'volume', 'basic_indicators',
            'advanced_indicators', 'order_book'], 'env_params': {
            'lookback_periods': 40, 'reward_mode': 'risk_adjusted',
            'trading_fee_percent': 0.002, 'curriculum_level': 3,
            'include_news_sentiment': True, 'include_order_book': True}},
            TrainingStage.EXPERT: {'description':
            'Expert training with extreme market events', 'min_level': 4,
            'max_level': 5, 'min_episodes': 400, 'performance_threshold': 
            0.8, 'reward_scale': 1.0, 'include_news': True,
            'observation_features': ['price', 'volume', 'basic_indicators',
            'advanced_indicators', 'order_book', 'regime'], 'env_params': {
            'lookback_periods': 50, 'reward_mode': 'risk_adjusted',
            'trading_fee_percent': 0.0025, 'curriculum_level': 4,
            'include_news_sentiment': True, 'include_order_book': True,
            'include_broker_state': True}}, TrainingStage.MASTER: {
            'description': 'Master level with full market complexity',
            'min_level': 5, 'max_level': 5, 'min_episodes': 500,
            'performance_threshold': 0.9, 'reward_scale': 1.0,
            'include_news': True, 'observation_features': ['price',
            'volume', 'basic_indicators', 'advanced_indicators',
            'order_book', 'regime', 'liquidity'], 'env_params': {
            'lookback_periods': 60, 'reward_mode': 'custom',
            'trading_fee_percent': 0.003, 'curriculum_level': 5,
            'include_news_sentiment': True, 'include_order_book': True,
            'include_broker_state': True}}}

    def _create_curriculum(self) ->Dict[int, List[Dict[str, Any]]]:
        """
        Create curriculum with progressive difficulty levels.
        
        Returns:
            Dictionary of difficulty levels and their scenarios
        """
        scenario_curriculum = {}
        raw_curriculum = self.market_simulator.create_curriculum(symbol=
            self.base_symbol, difficulty_levels=5, scenario_duration=
            timedelta(hours=8))
        for level, scenarios in raw_curriculum.items():
            scenario_curriculum[level] = []
            for scenario in scenarios:
                scenario_curriculum[level].append({'scenario': scenario,
                    'training_episodes': 50 * level, 'evaluation_episodes':
                    10, 'description': scenario.description})
        return scenario_curriculum

    def create_environment(self, level: int, stage: TrainingStage, scenario:
        Optional[SimulationScenario]=None) ->EnhancedForexTradingEnv:
        """
        Create an environment with appropriate difficulty for the current stage.
        
        Args:
            level: Difficulty level
            stage: Training stage
            scenario: Optional specific scenario to use
            
        Returns:
            Configured environment
        """
        stage_config = self.stage_configs[stage]
        if scenario:
            self.market_simulator.apply_scenario(scenario=scenario,
                start_time=datetime.now())
        env_params = stage_config['env_params'].copy()
        env_params['curriculum_level'] = level
        env = EnhancedForexTradingEnv(broker_simulator=self.
            broker_simulator, symbol=self.base_symbol,
            news_sentiment_simulator=self.news_simulator if stage_config[
            'include_news'] else None, include_news_sentiment=stage_config[
            'include_news'], **env_params)
        return env

    def train_curriculum(self, algorithm: Union[str, RLAlgorithm]='PPO',
        total_episodes: int=1000, eval_frequency: int=50, save_frequency:
        int=100, starting_stage: Optional[TrainingStage]=None,
        starting_level: Optional[int]=None) ->Dict[str, Any]:
        """
        Train an agent through the full curriculum.
        
        Args:
            algorithm: RL algorithm to use
            total_episodes: Total training episodes
            eval_frequency: Episodes between evaluations
            save_frequency: Episodes between model saving
            starting_stage: Optional stage to start from
            starting_level: Optional level to start from
            
        Returns:
            Training results and metrics
        """
        if starting_stage:
            self.current_stage = starting_stage
        if starting_level:
            self.current_level = starting_level
        stage_config = self.stage_configs[self.current_stage]
        env = self.create_environment(level=self.current_level, stage=self.
            current_stage)
        self.current_model = self.rl_model_factory.create_model(algorithm=
            algorithm, environment=env, tensorboard_log=os.path.join(self.
            tensorboard_log_dir,
            f'{self.current_stage.value}_level{self.current_level}'))
        logger.info(
            f'Beginning curriculum training: {total_episodes} total episodes')
        logger.info(
            f'Starting with {self.current_stage.value} stage at level {self.current_level}'
            )
        self.episodes_completed = 0
        self.stage_episodes = 0
        evaluation_results = []
        while self.episodes_completed < total_episodes:
            scenarios = self._get_scenarios_for_current_level()
            for scenario_config in scenarios:
                scenario = scenario_config['scenario']
                scenario_episodes = min(scenario_config['training_episodes'
                    ], total_episodes - self.episodes_completed)
                if scenario_episodes <= 0:
                    break
                logger.info(
                    f'Training on scenario: {scenario.name} for {scenario_episodes} episodes'
                    )
                scenario_env = self.create_environment(level=self.
                    current_level, stage=self.current_stage, scenario=scenario)
                self.current_model.set_env(scenario_env)
                for episode in range(scenario_episodes):
                    self.current_model.learn(total_timesteps=scenario_env.
                        max_episode_steps, reset_num_timesteps=False)
                    self.episodes_completed += 1
                    self.stage_episodes += 1
                    if self.episodes_completed % eval_frequency == 0:
                        eval_result = self.evaluate_agent()
                        evaluation_results.append(eval_result)
                        self._check_progression(eval_result)
                    if self.episodes_completed % save_frequency == 0:
                        self.save_model()
                    if episode % 10 == 0:
                        logger.info(
                            f'Episode {self.episodes_completed}/{total_episodes} completed. Stage: {self.current_stage.value}, Level: {self.current_level}'
                            )
                    if self.episodes_completed >= total_episodes:
                        break
                if self.episodes_completed >= total_episodes:
                    break
            if self.episodes_completed >= total_episodes:
                break
        final_eval = self.evaluate_agent()
        evaluation_results.append(final_eval)
        self.save_model()
        summary = {'total_episodes': self.episodes_completed, 'final_stage':
            self.current_stage.value, 'final_level': self.current_level,
            'evaluation_results': evaluation_results, 'final_performance':
            final_eval}
        return summary

    def _get_scenarios_for_current_level(self) ->List[Dict[str, Any]]:
        """Get scenarios for the current difficulty level."""
        if self.current_level in self.curriculum:
            return self.curriculum[self.current_level]
        else:
            scenario = self.market_simulator.get_random_scenario(self.
                base_symbol)
            return [{'scenario': scenario, 'training_episodes': 50,
                'evaluation_episodes': 10, 'description':
                f'Random scenario (level {self.current_level} not defined)'}]

    def evaluate_agent(self) ->Dict[str, Any]:
        """
        Evaluate the current agent on standardized scenarios.
        
        Returns:
            Evaluation metrics
        """
        original_env = self.current_model.get_env()
        eval_env = self.create_environment(level=self.current_level, stage=
            self.current_stage)
        returns = []
        sharpe_ratios = []
        win_rates = []
        drawdowns = []
        if self.current_level in self.curriculum:
            eval_scenarios = self.curriculum[self.current_level]
        else:
            scenario = self.market_simulator.get_random_scenario(self.
                base_symbol)
            eval_scenarios = [{'scenario': scenario, 'evaluation_episodes':
                5, 'description': 'Default evaluation scenario'}]
        scenario_results = []
        for scenario_config in eval_scenarios:
            scenario = scenario_config['scenario']
            eval_episodes = scenario_config['evaluation_episodes']
            self.market_simulator.apply_scenario(scenario=scenario,
                start_time=datetime.now())
            eval_env = self.create_environment(level=self.current_level,
                stage=self.current_stage, scenario=scenario)
            scenario_returns = []
            scenario_win_rate = []
            scenario_drawdown = []
            for _ in range(eval_episodes):
                obs = eval_env.reset()
                done = False
                episode_return = 0
                max_equity = 0
                min_drawdown = 0
                trades_won = 0
                trades_total = 0
                while not done:
                    action, _ = self.current_model.predict(obs,
                        deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    episode_return += reward
                    equity = info.get('equity', 0)
                    if equity > max_equity:
                        max_equity = equity
                    current_drawdown = (max_equity - equity
                        ) / max_equity if max_equity > 0 else 0
                    min_drawdown = max(min_drawdown, current_drawdown)
                    if info.get('trade_closed', False):
                        trades_total += 1
                        if info.get('trade_profit', 0) > 0:
                            trades_won += 1
                scenario_returns.append(episode_return)
                win_rate = trades_won / trades_total if trades_total > 0 else 0
                scenario_win_rate.append(win_rate)
                scenario_drawdown.append(min_drawdown)
            if len(scenario_returns) > 1:
                mean_return = np.mean(scenario_returns)
                std_return = np.std(scenario_returns) if np.std(
                    scenario_returns) > 0 else 1e-06
                sharpe = mean_return / std_return
            else:
                sharpe = 0
            returns.extend(scenario_returns)
            sharpe_ratios.append(sharpe)
            win_rates.extend(scenario_win_rate)
            drawdowns.extend(scenario_drawdown)
            scenario_results.append({'scenario': scenario.name,
                'description': scenario.description, 'mean_return': np.mean
                (scenario_returns), 'sharpe_ratio': sharpe, 'mean_win_rate':
                np.mean(scenario_win_rate), 'mean_drawdown': np.mean(
                scenario_drawdown), 'episodes': eval_episodes})
        mean_return = np.mean(returns)
        std_return = np.std(returns) if np.std(returns) > 0 else 1e-06
        overall_sharpe = mean_return / std_return
        overall_win_rate = np.mean(win_rates)
        overall_drawdown = np.mean(drawdowns)
        self.current_model.set_env(original_env)
        evaluation = {'timestamp': datetime.now().isoformat(),
            'episodes_completed': self.episodes_completed, 'stage': self.
            current_stage.value, 'level': self.current_level, 'overall': {
            'mean_return': float(mean_return), 'sharpe_ratio': float(
            overall_sharpe), 'win_rate': float(overall_win_rate),
            'drawdown': float(overall_drawdown)}, 'scenarios': scenario_results
            }
        logger.info(
            f'Evaluation at episode {self.episodes_completed}: Sharpe={overall_sharpe:.2f}, Win Rate={overall_win_rate:.2%}'
            )
        return evaluation

    def _check_progression(self, eval_result: Dict[str, Any]) ->bool:
        """
        Check if agent should progress to next level or stage.
        
        Args:
            eval_result: Latest evaluation result
            
        Returns:
            True if progression occurred
        """
        stage_config = self.stage_configs[self.current_stage]
        current_performance = eval_result['overall']['sharpe_ratio']
        if (self.progression_criteria == CurriculumProgressionCriteria.
            EPISODES_COMPLETED):
            should_progress = self.stage_episodes >= stage_config[
                'min_episodes']
        elif self.progression_criteria == CurriculumProgressionCriteria.PERFORMANCE_THRESHOLD:
            should_progress = current_performance >= stage_config[
                'performance_threshold'
                ] and self.stage_episodes >= stage_config['min_episodes']
        elif self.progression_criteria == CurriculumProgressionCriteria.AUTO_ADAPTIVE:
            min_episodes = stage_config['min_episodes'] // 2
            if len(self.training_history
                ) >= 3 and self.stage_episodes >= min_episodes:
                recent_perf = [r['overall']['sharpe_ratio'] for r in self.
                    training_history[-3:]]
                should_progress = max(recent_perf) - min(recent_perf
                    ) < 0.1 and current_performance >= stage_config[
                    'performance_threshold'] * 0.8
            else:
                should_progress = False
        else:
            should_progress = False
        if should_progress:
            if self.current_level < stage_config['max_level']:
                self.current_level += 1
                self.stage_episodes = 0
                logger.info(
                    f'Advancing to level {self.current_level} in {self.current_stage.value} stage'
                    )
                return True
            elif self.current_stage != TrainingStage.MASTER:
                stages = list(TrainingStage)
                current_idx = stages.index(self.current_stage)
                if current_idx < len(stages) - 1:
                    self.current_stage = stages[current_idx + 1]
                    self.current_level = self.stage_configs[self.current_stage
                        ]['min_level']
                    self.stage_episodes = 0
                    logger.info(
                        f'Advancing to {self.current_stage.value} stage at level {self.current_level}'
                        )
                    return True
        return False

    def save_model(self) ->str:
        """
        Save the current model.
        
        Returns:
            Path to the saved model
        """
        save_path = os.path.join(self.save_dir,
            f'{self.current_stage.value}_level{self.current_level}_ep{self.episodes_completed}'
            )
        self.current_model.save(save_path)
        state_path = f'{save_path}_state.json'
        training_state = {'episodes_completed': self.episodes_completed,
            'stage_episodes': self.stage_episodes, 'current_stage': self.
            current_stage.value, 'current_level': self.current_level,
            'timestamp': datetime.now().isoformat()}
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
        logger.info(f'Model and training state saved to {save_path}')
        return save_path

    @with_exception_handling
    def load_model(self, path: str, load_state: bool=True) ->bool:
        """
        Load a previously saved model and optionally its training state.
        
        Args:
            path: Path to the saved model
            load_state: Whether to load the training state
            
        Returns:
            Success status
        """
        try:
            if 'PPO' in path:
                algorithm = 'PPO'
            elif 'A2C' in path:
                algorithm = 'A2C'
            elif 'SAC' in path:
                algorithm = 'SAC'
            elif 'TD3' in path:
                algorithm = 'TD3'
            elif 'DQN' in path:
                algorithm = 'DQN'
            else:
                algorithm = 'PPO'
            env = self.create_environment(level=self.current_level, stage=
                self.current_stage)
            self.current_model = self.rl_model_factory.load_model(algorithm
                =algorithm, model_path=path, environment=env)
            if load_state:
                state_path = f'{path}_state.json'
                if os.path.exists(state_path):
                    with open(state_path, 'r') as f:
                        state = json.load(f)
                    self.episodes_completed = state.get('episodes_completed', 0
                        )
                    self.stage_episodes = state.get('stage_episodes', 0)
                    self.current_stage = TrainingStage(state.get(
                        'current_stage', TrainingStage.BASIC))
                    self.current_level = state.get('current_level', 1)
            logger.info(f'Model loaded from {path}')
            if load_state:
                logger.info(
                    f'Resumed at {self.current_stage.value} stage, level {self.current_level}, episode {self.episodes_completed}'
                    )
            return True
        except Exception as e:
            logger.error(f'Failed to load model from {path}: {e}')
            return False

    def generate_curriculum_report(self, output_path: str) ->Dict[str, Any]:
        """
        Generate a detailed report on curriculum training progress.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Report data
        """
        if not self.training_history:
            logger.warning('No training history available for report')
            return {}
        episodes = [r['episodes_completed'] for r in self.training_history]
        returns = [r['overall']['mean_return'] for r in self.training_history]
        sharpes = [r['overall']['sharpe_ratio'] for r in self.training_history]
        win_rates = [r['overall']['win_rate'] for r in self.training_history]
        drawdowns = [r['overall']['drawdown'] for r in self.training_history]
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(episodes, returns, 'b-')
        plt.xlabel('Episodes')
        plt.ylabel('Mean Return')
        plt.title('Returns over Training')
        plt.grid(True)
        plt.subplot(2, 2, 2)
        plt.plot(episodes, sharpes, 'g-')
        plt.xlabel('Episodes')
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe Ratio over Training')
        plt.grid(True)
        plt.subplot(2, 2, 3)
        plt.plot(episodes, win_rates, 'r-')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate')
        plt.title('Win Rate over Training')
        plt.grid(True)
        plt.subplot(2, 2, 4)
        plt.plot(episodes, drawdowns, 'k-')
        plt.xlabel('Episodes')
        plt.ylabel('Mean Drawdown')
        plt.title('Drawdown over Training')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f'Curriculum training report saved to {output_path}')
        report = {'total_episodes': self.episodes_completed,
            'highest_stage': self.current_stage.value, 'highest_level':
            self.current_level, 'final_performance': self.training_history[
            -1]['overall'] if self.training_history else None,
            'training_progress': {'episodes': episodes, 'returns': returns,
            'sharpes': sharpes, 'win_rates': win_rates, 'drawdowns': drawdowns}
            }
        return report


class DistributedCurriculumTrainer:
    """
    Distributed training for RL agents using the curriculum framework.
    
    This class handles parallelized training across multiple environments,
    scenarios, and difficulty levels for faster and more robust training.
    
    Features:
    - Vectorized environments for parallel training
    - Synchronized curriculum progression across environments
    - Performance-based difficulty adjustment
    - Distributed experience collection with centralized updates
    """

    def __init__(self, curriculum_framework: CurriculumLearningFramework,
        num_parallel_envs: int=4, save_dir: str='./models/distributed/',
        tensorboard_log_dir: str='./logs/distributed/', device: str='auto',
        random_seed: Optional[int]=None):
        """
        Initialize the distributed trainer.
        
        Args:
            curriculum_framework: Curriculum learning framework
            num_parallel_envs: Number of parallel environments
            save_dir: Directory to save models
            tensorboard_log_dir: Directory for tensorboard logs
            device: Device for training ('cpu', 'cuda', 'auto')
            random_seed: Random seed for reproducibility
        """
        self.curriculum = curriculum_framework
        self.num_parallel_envs = num_parallel_envs
        self.save_dir = save_dir
        self.tensorboard_log_dir = tensorboard_log_dir
        self.device = device
        self.random_seed = random_seed
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
                torch.backends.cudnn.deterministic = True
        self.training_metrics = {'timesteps': [], 'rewards': [],
            'eval_returns': [], 'sharpe_ratios': [], 'win_rates': [],
            'levels_completed': []}
        self.current_model = None

    def _make_env(self, rank: int, level: int, stage: TrainingStage,
        scenario: Optional[SimulationScenario]=None) ->Callable:
        """
        Create an environment factory function for vectorized environments.
        
        Args:
            rank: Environment rank for seeding
            level: Curriculum difficulty level
            stage: Training stage
            scenario: Optional scenario configuration
            
        Returns:
            Environment factory function
        """

        def _init() ->EnhancedForexTradingEnv:
    """
     init.
    
    Returns:
        EnhancedForexTradingEnv: Description of return value
    
    """

            env = self.curriculum.create_environment(level, stage, scenario)
            if self.random_seed is not None:
                env.seed(self.random_seed + rank)
            env = Monitor(env, os.path.join(self.tensorboard_log_dir,
                f'env_{rank}'))
            return env
        return _init

    def _create_vectorized_env(self, level: int, stage: TrainingStage,
        scenarios: Optional[List[SimulationScenario]]=None) ->VecEnv:
        """
        Create a vectorized environment for parallel training.
        
        Args:
            level: Curriculum difficulty level
            stage: Training stage
            scenarios: Optional list of scenarios (one per env)
            
        Returns:
            Vectorized environment
        """
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
        env_factories = []
        for i in range(self.num_parallel_envs):
            scenario = None
            if scenarios and len(scenarios) > 0:
                scenario = scenarios[i % len(scenarios)]
            env_factories.append(self._make_env(i, level, stage, scenario))
        if self.num_parallel_envs > 1:
            vec_env = SubprocVecEnv(env_factories)
        else:
            vec_env = DummyVecEnv(env_factories)
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0)
        return vec_env

    def train_distributed(self, algorithm: Union[str, RLAlgorithm]='PPO',
        total_timesteps: int=1000000, eval_frequency: int=10000,
        save_frequency: int=50000, n_eval_episodes: int=5, callback_classes:
        Optional[List[type]]=None) ->Dict[str, Any]:
        """
        Train an agent using distributed environments.
        
        Args:
            algorithm: RL algorithm to use
            total_timesteps: Total training timesteps
            eval_frequency: Timesteps between evaluations
            save_frequency: Timesteps between model saving
            n_eval_episodes: Number of episodes for evaluation
            callback_classes: Optional list of callback classes
            
        Returns:
            Training results
        """
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
        logger.info(
            f'Starting distributed training with {self.num_parallel_envs} parallel environments'
            )
        level = self.curriculum.current_level
        stage = self.curriculum.current_stage
        scenarios = self.curriculum._get_scenarios_for_current_level()
        scenario_configs = [config['scenario'] for config in scenarios]
        vec_env = self._create_vectorized_env(level, stage, scenario_configs)
        eval_env = self.curriculum.create_environment(level, stage)
        if isinstance(algorithm, str):
            if algorithm == 'PPO':
                model_cls = PPO
            elif algorithm == 'A2C':
                model_cls = A2C
            elif algorithm == 'SAC':
                model_cls = SAC
            elif algorithm == 'TD3':
                model_cls = TD3
            elif algorithm == 'DQN':
                model_cls = DQN
            else:
                raise ValueError(f'Unsupported algorithm: {algorithm}')
        else:
            model_cls = algorithm
        model = model_cls('MlpPolicy', vec_env, verbose=1, tensorboard_log=
            self.tensorboard_log_dir, device=self.device)
        self.current_model = model
        callbacks = []
        eval_callback = EvalCallback(eval_env, best_model_save_path=os.path
            .join(self.save_dir, 'best_model'), log_path=os.path.join(self.
            tensorboard_log_dir, 'eval'), eval_freq=eval_frequency,
            n_eval_episodes=n_eval_episodes, deterministic=True, render=False)
        callbacks.append(eval_callback)
        checkpoint_callback = CheckpointCallback(save_freq=save_frequency,
            save_path=os.path.join(self.save_dir, 'checkpoints'),
            name_prefix='curriculum_model')
        callbacks.append(checkpoint_callback)


        class CurriculumCallback(BaseCallback):
    """
    CurriculumCallback class that inherits from BaseCallback.
    
    Attributes:
        Add attributes here
    """


            def __init__(self, trainer, verbose=0):
    """
      init  .
    
    Args:
        trainer: Description of trainer
        verbose: Description of verbose
    
    """

                super().__init__(verbose)
                self.trainer = trainer
                self.evaluation_results = []
                self.last_eval_timestep = 0

            def _on_step(self):
    """
     on step.
    
    """

                if (self.num_timesteps - self.last_eval_timestep >=
                    eval_frequency):
                    mean_reward, std_reward = evaluate_policy(self.model,
                        self.trainer.curriculum.create_environment(self.
                        trainer.curriculum.current_level, self.trainer.
                        curriculum.current_stage), n_eval_episodes=
                        n_eval_episodes)
                    metrics = self.trainer.curriculum.evaluate_agent()
                    eval_result = {'timesteps': self.num_timesteps,
                        'mean_reward': mean_reward, 'std_reward':
                        std_reward, 'level': self.trainer.curriculum.
                        current_level, 'stage': self.trainer.curriculum.
                        current_stage.value, 'sharpe_ratio': metrics.get(
                        'overall', {}).get('sharpe_ratio', 0), 'win_rate':
                        metrics.get('overall', {}).get('win_rate', 0)}
                    self.evaluation_results.append(eval_result)
                    self.logger.record('curriculum/level', self.trainer.
                        curriculum.current_level)
                    self.logger.record('curriculum/stage', self.trainer.
                        curriculum.current_stage.value)
                    self.logger.record('curriculum/sharpe_ratio', metrics.
                        get('overall', {}).get('sharpe_ratio', 0))
                    self.logger.record('curriculum/win_rate', metrics.get(
                        'overall', {}).get('win_rate', 0))
                    old_level = self.trainer.curriculum.current_level
                    old_stage = self.trainer.curriculum.current_stage
                    self.trainer.curriculum._check_progression(metrics)
                    if (old_level != self.trainer.curriculum.current_level or
                        old_stage != self.trainer.curriculum.current_stage):
                        logger.info(
                            f'Advancing curriculum to level {self.trainer.curriculum.current_level}, stage {self.trainer.curriculum.current_stage.value}'
                            )
                        new_scenarios = (self.trainer.curriculum.
                            _get_scenarios_for_current_level())
                        new_scenario_configs = [config['scenario'] for
                            config in new_scenarios]
                        new_vec_env = self.trainer._create_vectorized_env(self
                            .trainer.curriculum.current_level, self.trainer
                            .curriculum.current_stage, new_scenario_configs)
                        self.model.set_env(new_vec_env)
                    self.last_eval_timestep = self.num_timesteps
                return True
        curriculum_callback = CurriculumCallback(self)
        callbacks.append(curriculum_callback)
        if callback_classes:
            for callback_cls in callback_classes:
                callbacks.append(callback_cls())
        from stable_baselines3.common.callbacks import CallbackList
        callback = CallbackList(callbacks)
        logger.info(
            f'Starting distributed training for {total_timesteps} timesteps')
        model.learn(total_timesteps=total_timesteps, callback=callback,
            tb_log_name=f'{algorithm}_distributed')
        final_model_path = os.path.join(self.save_dir, 'final_model')
        model.save(final_model_path)
        logger.info(f'Saved final model to {final_model_path}')
        final_eval = self.curriculum.evaluate_agent()
        return {'status': 'completed', 'total_timesteps': total_timesteps,
            'final_level': self.curriculum.current_level, 'final_stage':
            self.curriculum.current_stage.value, 'evaluation_history':
            curriculum_callback.evaluation_results, 'final_evaluation':
            final_eval, 'model_paths': {'final': final_model_path, 'best':
            os.path.join(self.save_dir, 'best_model', 'best_model.zip')}}

    def create_parallel_scenarios(self, base_scenario: SimulationScenario,
        variations: int) ->List[SimulationScenario]:
        """
        Create variations of a scenario for parallel training environments.
        
        Args:
            base_scenario: Base scenario configuration
            variations: Number of variations to create
        
        Returns:
            List of scenario variations
        """
        variations_list = []
        for i in range(variations):
            variation = copy.deepcopy(base_scenario)
            variation.volatility_factor *= random.uniform(0.8, 1.2)
            variation.spread_factor *= random.uniform(0.9, 1.1)
            if variation.trend_strength != 0:
                variation.trend_strength *= random.uniform(0.7, 1.3)
            if variation.special_events:
                for event in variation.special_events:
                    if 'time_offset_minutes' in event:
                        event['time_offset_minutes'] = int(event[
                            'time_offset_minutes'] * random.uniform(0.8, 1.2))
            variation.name = f'{base_scenario.name}_var{i + 1}'
            variations_list.append(variation)
        return variations_list

    def visualize_training_progress(self, output_path: str) ->None:
        """
        Generate visualizations of the training progress.
        
        Args:
            output_path: Path to save visualizations
        """
        if not self.training_metrics['timesteps']:
            logger.warning('No training metrics available for visualization')
            return
        os.makedirs(output_path, exist_ok=True)
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self.training_metrics['timesteps'], self.training_metrics[
            'rewards'])
        plt.xlabel('Timesteps')
        plt.ylabel('Average Reward')
        plt.title('Training Reward')
        plt.grid(True)
        plt.subplot(2, 2, 2)
        plt.plot(self.training_metrics['timesteps'], self.training_metrics[
            'eval_returns'])
        plt.xlabel('Timesteps')
        plt.ylabel('Evaluation Return')
        plt.title('Evaluation Performance')
        plt.grid(True)
        plt.subplot(2, 2, 3)
        plt.plot(self.training_metrics['timesteps'], self.training_metrics[
            'sharpe_ratios'])
        plt.xlabel('Timesteps')
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe Ratio Progression')
        plt.grid(True)
        plt.subplot(2, 2, 4)
        plt.plot(self.training_metrics['timesteps'], self.training_metrics[
            'win_rates'])
        plt.xlabel('Timesteps')
        plt.ylabel('Win Rate')
        plt.title('Win Rate Progression')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path,
            'distributed_training_metrics.png'))
        plt.close()
        with open(os.path.join(output_path, 'training_metrics.json'), 'w'
            ) as f:
            json.dump(self.training_metrics, f, indent=2)
