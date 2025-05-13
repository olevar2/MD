"""
Reinforcement Learning Training Module for Forex Trading

This module provides sample implementations of reinforcement learning agents
and training procedures for forex trading, demonstrating how to use the
curriculum learning framework and trading environment.
"""
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import random
import uuid
import matplotlib.pyplot as plt
import logging
from collections import deque
from pathlib import Path
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
from core.curriculum_learning_framework import CurriculumLearningFramework, DifficultyLevel
from core.environment_generator import ForexTradingEnvironment, EnvConfiguration, EnvironmentGenerator, ActionType
from core.agent_benchmarking import RLAgentBenchmark, BenchmarkCategory
from core.advanced_market_regime_simulator import AdvancedMarketRegimeSimulator, MarketCondition, SimulationScenario
from core.enhanced_market_condition_generator import EnhancedMarketConditionGenerator
from core.forex_broker_simulator import ForexBrokerSimulator
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)
from core.exceptions_bridge_1 import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class ReplayBuffer:
    """Simple experience replay buffer for RL agents."""

    def __init__(self, capacity: int=10000):
        """
        Initialize replay buffer with fixed capacity.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) ->Tuple:
        """Sample a batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for
            i in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype
            =np.float32), np.array(next_states), np.array(dones, dtype=np.
            float32)

    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:


    class DQNNetwork(nn.Module):
        """Deep Q-Network for discrete action spaces."""

        def __init__(self, input_shape: Tuple[int, int], num_actions: int):
            """
            Initialize the DQN network.
            
            Args:
                input_shape: Shape of input observations (window_size, features)
                num_actions: Number of possible actions
            """
            super(DQNNetwork, self).__init__()
            flattened_size = input_shape[0] * input_shape[1]
            self.feature_extractor = nn.Sequential(nn.Linear(flattened_size,
                512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
            self.value_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(
                ), nn.Linear(128, 1))
            self.advantage_stream = nn.Sequential(nn.Linear(256, 128), nn.
                ReLU(), nn.Linear(128, num_actions))

        def forward(self, x):
            """Forward pass through the network."""
            x = x.view(x.size(0), -1)
            features = self.feature_extractor(x)
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            q_values = value + (advantages - advantages.mean(dim=1, keepdim
                =True))
            return q_values


class DQNAgent:
    """
    Deep Q-Network agent for forex trading.
    
    This agent uses a dueling DQN architecture with experience replay
    and target network for stable learning.
    """

    def __init__(self, input_shape: Tuple[int, int], num_actions: int,
        learning_rate: float=0.001, gamma: float=0.99, epsilon_start: float
        =1.0, epsilon_end: float=0.1, epsilon_decay: float=0.995,
        buffer_size: int=10000, batch_size: int=64, target_update: int=10,
        device: str='cpu', random_seed: Optional[int]=None):
        """
        Initialize the DQN agent.
        
        Args:
            input_shape: Shape of input observations (window_size, features)
            num_actions: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of epsilon decay
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update: Steps between target network updates
            device: Device to use for tensor operations
            random_seed: Random seed for reproducibility
        """
        if not TORCH_AVAILABLE:
            raise ImportError('PyTorch is required for DQNAgent')
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device(device)
        self.policy_net = DQNNetwork(input_shape, num_actions).to(self.device)
        self.target_net = DQNNetwork(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=
            learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps = 0
        self.training_rewards = []
        self.loss_history = []
        self.epsilon_history = []
        self.id = str(uuid.uuid4())[:8]

    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current observation
            
        Returns:
            Selected action
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device
                )
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def predict(self, state):
        """
        Predict best action (no exploration).
        
        Args:
            state: Current observation
            
        Returns:
            Best action according to policy
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device
                )
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    @with_broker_api_resilience('update_epsilon')
    def update_epsilon(self):
        """Update exploration rate."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        self.epsilon_history.append(self.epsilon)

    def train(self):
        """Train the agent on a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = (self.replay_buffer.
            sample(self.batch_size))
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        current_q_values = self.policy_net(states_tensor).gather(1,
            actions_tensor.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_net(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values * (
                1 - dones_tensor)
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss_value

    def learn(self, env, num_episodes: int=1000, max_steps: int=1000):
        """
        Train the agent on the environment.
        
        Args:
            env: Training environment
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary of training statistics
        """
        all_rewards = []
        episode_lengths = []
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            episode_reward = 0.0
            episode_loss = 0.0
            step = 0
            for step in range(1, max_steps + 1):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                if len(self.replay_buffer) >= self.batch_size:
                    loss = self.train()
                    episode_loss += loss
                state = next_state
                episode_reward += reward
                if done:
                    break
            self.update_epsilon()
            all_rewards.append(episode_reward)
            episode_lengths.append(step)
            self.training_rewards.append(episode_reward)
            avg_reward = np.mean(all_rewards[-100:])
            avg_loss = episode_loss / step if step > 0 else 0
            if episode % 10 == 0:
                logger.info(
                    f'Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {self.epsilon:.2f} | Steps: {step}'
                    )
        return {'rewards': all_rewards, 'episode_lengths': episode_lengths,
            'losses': self.loss_history, 'epsilons': self.epsilon_history}

    def save(self, path: str):
        """Save agent to disk."""
        save_dict = {'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), 'epsilon':
            self.epsilon, 'steps': self.steps, 'input_shape': self.
            input_shape, 'num_actions': self.num_actions}
        torch.save(save_dict, path)

    def load(self, path: str):
        """Load agent from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint['input_shape'] != self.input_shape or checkpoint[
            'num_actions'] != self.num_actions:
            logger.warning(
                'Model architecture mismatch. Creating new networks.')
            self.policy_net = DQNNetwork(checkpoint['input_shape'],
                checkpoint['num_actions']).to(self.device)
            self.target_net = DQNNetwork(checkpoint['input_shape'],
                checkpoint['num_actions']).to(self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


class StableBaselineAgent:
    """
    Wrapper for Stable-Baselines3 RL algorithms.
    """

    def __init__(self, env, algorithm: str='PPO', policy: str='MlpPolicy',
        learning_rate: float=0.0003, custom_policy=None, custom_hyperparams:
        Dict=None, verbose: int=1, random_seed: Optional[int]=None):
        """
        Initialize the agent.
        
        Args:
            env: Training environment
            algorithm: Algorithm name ('PPO', 'A2C', 'DQN')
            policy: Policy network architecture
            learning_rate: Learning rate
            custom_policy: Optional custom policy network
            custom_hyperparams: Optional custom hyperparameters
            verbose: Verbosity level
            random_seed: Optional random seed
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                'stable-baselines3 is required for StableBaselineAgent')
        self.env = env
        self.algorithm_name = algorithm
        self.vec_env = DummyVecEnv([lambda : env])
        self.hyperparams = {'learning_rate': learning_rate, 'seed': random_seed
            }
        if custom_hyperparams:
            self.hyperparams.update(custom_hyperparams)
        if algorithm == 'PPO':
            self.model = PPO(policy=custom_policy or policy, env=self.
                vec_env, verbose=verbose, **self.hyperparams)
        elif algorithm == 'A2C':
            self.model = A2C(policy=custom_policy or policy, env=self.
                vec_env, verbose=verbose, **self.hyperparams)
        elif algorithm == 'DQN':
            self.model = DQN(policy=custom_policy or policy, env=self.
                vec_env, verbose=verbose, **self.hyperparams)
        else:
            raise ValueError(f'Unsupported algorithm: {algorithm}')
        self.id = f'{algorithm}_{str(uuid.uuid4())[:6]}'

    def learn(self, total_timesteps: int, callback=None):
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training steps
            callback: Optional callback for training
            
        Returns:
            Trained model
        """
        return self.model.learn(total_timesteps=total_timesteps, callback=
            callback)

    def predict(self, state, deterministic: bool=True):
        """
        Predict action for a state.
        
        Args:
            state: Environment state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Predicted action
        """
        action, _ = self.model.predict(state, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save the model."""
        self.model.save(path)

    def load(self, path: str):
        """Load the model."""
        if self.algorithm_name == 'PPO':
            self.model = PPO.load(path, env=self.vec_env)
        elif self.algorithm_name == 'A2C':
            self.model = A2C.load(path, env=self.vec_env)
        elif self.algorithm_name == 'DQN':
            self.model = DQN.load(path, env=self.vec_env)


class CurriculumTrainer:
    """
    Training system for reinforcement learning agents using curriculum learning.
    
    This system integrates the curriculum learning framework with RL agents
    to progressively train agents on increasingly difficult market scenarios.
    """

    def __init__(self, framework: CurriculumLearningFramework,
        agent_factory: Callable, output_dir: Optional[str]=None,
        random_seed: Optional[int]=None):
        """
        Initialize the curriculum trainer.
        
        Args:
            framework: Curriculum learning framework
            agent_factory: Function to create new agent instances
            output_dir: Output directory for saving results
            random_seed: Optional random seed
        """
        self.framework = framework
        self.agent_factory = agent_factory
        self.output_dir = output_dir or os.path.join(os.getcwd(),
            'training_results')
        os.makedirs(self.output_dir, exist_ok=True)
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.agent = None
        self.create_new_agent()
        self.training_history = []

    @with_broker_api_resilience('create_new_agent')
    def create_new_agent(self) ->Any:
        """
        Create a new agent instance using the factory.
        
        Returns:
            Newly created agent
        """
        self.agent = self.agent_factory()
        return self.agent

    def evaluate_agent(self, env: ForexTradingEnvironment, num_episodes: int=3
        ) ->Dict[str, float]:
        """
        Evaluate agent performance on an environment.
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of evaluation metrics
        """
        all_returns = []
        all_sharpe = []
        all_drawdowns = []
        all_trade_counts = []
        all_win_rates = []
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.agent.predict(state)
                state, _, done, _ = env.step(action)
            metrics = env.get_performance_summary()
            all_returns.append(metrics['returns_pct'])
            all_sharpe.append(metrics['sharpe_ratio'])
            all_drawdowns.append(metrics['max_drawdown_pct'])
            all_trade_counts.append(metrics['num_trades'])
            all_win_rates.append(metrics['win_rate'])
        return {'returns_pct': np.mean(all_returns), 'sharpe_ratio': np.
            mean(all_sharpe), 'max_drawdown_pct': np.mean(all_drawdowns),
            'num_trades': np.mean(all_trade_counts), 'win_rate': np.mean(
            all_win_rates), 'profit_factor': metrics.get('profit_factor', 0.0)}

    @with_exception_handling
    def train_on_scenario(self, scenario: SimulationScenario, env_config:
        EnvConfiguration, num_episodes: int=100, steps_per_episode: int=
        1000, eval_frequency: int=10) ->Dict[str, Any]:
        """
        Train agent on a specific scenario.
        
        Args:
            scenario: Simulation scenario to train on
            env_config: Environment configuration
            num_episodes: Number of training episodes
            steps_per_episode: Maximum steps per episode
            eval_frequency: Evaluation frequency in episodes
            
        Returns:
            Dictionary of training results
        """
        env = EnvironmentGenerator.create_environment(market_simulator=self
            .framework.market_generator.market_simulator, broker_simulator=
            ForexBrokerSimulator(), config=env_config)
        self.framework.market_generator.market_simulator.apply_scenario(
            scenario, datetime.now() - timedelta(days=30))
        training_metrics = {'episode_rewards': [], 'evaluations': [],
            'scenario_name': scenario.name, 'market_condition': scenario.
            market_condition}
        if hasattr(self.agent, 'learn'):
            try:
                train_results = self.agent.learn(env, num_episodes,
                    steps_per_episode)
                if isinstance(train_results, dict
                    ) and 'rewards' in train_results:
                    training_metrics['episode_rewards'] = train_results[
                        'rewards']
            except Exception as e:
                logger.error(f'Error during agent training: {e}')
        else:
            for episode in range(1, num_episodes + 1):
                state = env.reset()
                episode_reward = 0
                for step in range(steps_per_episode):
                    action = self.agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    if hasattr(self.agent, 'replay_buffer'):
                        self.agent.replay_buffer.add(state, action, reward,
                            next_state, done)
                    if hasattr(self.agent, 'train'):
                        self.agent.train()
                    state = next_state
                    episode_reward += reward
                    if done:
                        break
                if hasattr(self.agent, 'update_epsilon'):
                    self.agent.update_epsilon()
                training_metrics['episode_rewards'].append(episode_reward)
                if episode % eval_frequency == 0:
                    eval_metrics = self.evaluate_agent(env)
                    eval_metrics['episode'] = episode
                    training_metrics['evaluations'].append(eval_metrics)
                    logger.info(
                        f"Episode {episode}/{num_episodes} | Return: {eval_metrics['returns_pct']:.2f}% | Sharpe: {eval_metrics['sharpe_ratio']:.2f}"
                        )
        final_metrics = self.evaluate_agent(env, num_episodes=3)
        training_metrics['final_metrics'] = final_metrics
        env.close()
        return training_metrics

    def train_current_level(self, num_scenarios: int=3,
        episodes_per_scenario: int=100) ->Dict[str, Any]:
        """
        Train agent on the current curriculum level.
        
        Args:
            num_scenarios: Number of scenarios to train on
            episodes_per_scenario: Episodes to train per scenario
            
        Returns:
            Dictionary of training results
        """
        level = self.framework.current_level
        level_info = self.framework.difficulty_levels[level]
        logger.info(f'Training on level {level}: {level_info.name}')
        logger.info(f'Description: {level_info.description}')
        scenarios = self.framework.get_current_level_scenarios()
        if len(scenarios) > num_scenarios:
            scenarios = random.sample(scenarios, num_scenarios)
        results = {}
        for i, scenario in enumerate(scenarios):
            logger.info(
                f'Training on scenario {i + 1}/{len(scenarios)}: {scenario.market_condition} condition'
                )
            env_config = EnvConfiguration(symbols=[scenario.symbol],
                timeframe='1h', max_episode_steps=int(scenario.duration.
                total_seconds() / 3600))
            scenario_results = self.train_on_scenario(scenario=scenario,
                env_config=env_config, num_episodes=episodes_per_scenario)
            self.framework.report_training_results(scenario_index=i,
                performance_metrics=scenario_results['final_metrics'])
            results[f'scenario_{i}'] = scenario_results
        results['summary'] = {'level': level, 'level_name': level_info.name,
            'consecutive_successes': self.framework.consecutive_successes,
            'required_successes': self.framework.consecutive_successes_required
            }
        return results

    def run_curriculum_training(self, max_levels: Optional[int]=None,
        scenarios_per_level: int=3, episodes_per_scenario: int=100,
        save_checkpoints: bool=True) ->Dict[str, Any]:
        """
        Run full curriculum training.
        
        Args:
            max_levels: Maximum number of levels to train on (None = all levels)
            scenarios_per_level: Number of scenarios to train on per level
            episodes_per_scenario: Episodes to train per scenario
            save_checkpoints: Whether to save agent checkpoints
            
        Returns:
            Dictionary of training results
        """
        if max_levels is None:
            max_levels = self.framework.num_levels
        max_levels = min(max_levels, self.framework.num_levels)
        self.framework.reset_progress()
        results = {'levels': {}, 'progress': {}}
        checkpoint_paths = []
        while self.framework.current_level <= max_levels:
            level = self.framework.current_level
            level_results = self.train_current_level(num_scenarios=
                scenarios_per_level, episodes_per_scenario=
                episodes_per_scenario)
            results['levels'][f'level_{level}'] = level_results
            if save_checkpoints and hasattr(self.agent, 'save'):
                checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir,
                    f'agent_level_{level}.pt')
                self.agent.save(checkpoint_path)
                checkpoint_paths.append(checkpoint_path)
                logger.info(
                    f'Saved agent checkpoint for level {level} to {checkpoint_path}'
                    )
            previous_level = level
            current_level = self.framework.current_level
            progress = self.framework.get_progress_summary()
            results['progress'][f'level_{level}'] = progress
            if previous_level == current_level and level < max_levels:
                logger.info(f'Continuing training on level {level}')
                continue
        results['final_progress'] = self.framework.get_progress_summary()
        results['checkpoint_paths'] = checkpoint_paths
        self._generate_training_report(results)
        return results

    def _generate_training_report(self, results: Dict[str, Any]) ->None:
        """
        Generate a training report with visualizations.
        
        Args:
            results: Results from run_curriculum_training
        """
        report_dir = os.path.join(self.output_dir, 'reports',
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(report_dir, exist_ok=True)
        levels = sorted(results['levels'].keys())
        returns_by_level = []
        sharpe_by_level = []
        drawdown_by_level = []
        win_rate_by_level = []
        for level in levels:
            level_data = results['levels'][level]
            metrics = level_data.get('summary', {})
            all_returns = []
            all_sharpes = []
            all_drawdowns = []
            all_win_rates = []
            for scenario, scenario_data in level_data.items():
                if scenario == 'summary':
                    continue
                if 'final_metrics' in scenario_data:
                    final_metrics = scenario_data['final_metrics']
                    all_returns.append(final_metrics.get('returns_pct', 0.0))
                    all_sharpes.append(final_metrics.get('sharpe_ratio', 0.0))
                    all_drawdowns.append(final_metrics.get(
                        'max_drawdown_pct', 0.0))
                    all_win_rates.append(final_metrics.get('win_rate', 0.0))
            if all_returns:
                returns_by_level.append(np.mean(all_returns))
                sharpe_by_level.append(np.mean(all_sharpes))
                drawdown_by_level.append(np.mean(all_drawdowns))
                win_rate_by_level.append(np.mean(all_win_rates))
        if levels and returns_by_level:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.bar(range(1, len(returns_by_level) + 1), returns_by_level)
            plt.xlabel('Curriculum Level')
            plt.ylabel('Average Return (%)')
            plt.title('Returns by Curriculum Level')
            plt.subplot(2, 2, 2)
            plt.bar(range(1, len(sharpe_by_level) + 1), sharpe_by_level)
            plt.xlabel('Curriculum Level')
            plt.ylabel('Average Sharpe Ratio')
            plt.title('Sharpe Ratio by Curriculum Level')
            plt.subplot(2, 2, 3)
            plt.bar(range(1, len(drawdown_by_level) + 1), drawdown_by_level)
            plt.xlabel('Curriculum Level')
            plt.ylabel('Average Max Drawdown (%)')
            plt.title('Max Drawdown by Curriculum Level')
            plt.subplot(2, 2, 4)
            plt.bar(range(1, len(win_rate_by_level) + 1), win_rate_by_level)
            plt.xlabel('Curriculum Level')
            plt.ylabel('Average Win Rate')
            plt.title('Win Rate by Curriculum Level')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'performance_by_level.png'))
            plt.close()
            plt.figure(figsize=(10, 6))
            for level in levels:
                level_data = results['levels'][level]
                for scenario, scenario_data in level_data.items():
                    if scenario == 'summary':
                        continue
                    if 'episode_rewards' in scenario_data and scenario_data[
                        'episode_rewards']:
                        plt.plot(scenario_data['episode_rewards'], label=
                            f'Level {level[-1]} - {scenario}')
                        break
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards')
            plt.legend()
            plt.savefig(os.path.join(report_dir, 'training_rewards.png'))
            plt.close()
        serializable_results = self._make_serializable(results)
        with open(os.path.join(report_dir, 'training_results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=2)

    @with_exception_handling
    def _make_serializable(self, data):
        """Convert data to JSON-serializable format."""
        if isinstance(data, (np.ndarray, np.number)):
            return data.tolist()
        elif isinstance(data, (datetime, timedelta)):
            return str(data)
        elif isinstance(data, dict):
            return {key: self._make_serializable(value) for key, value in
                data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif hasattr(data, 'value'):
            return data.value
        elif hasattr(data, '__dict__'):
            try:
                return {key: self._make_serializable(value) for key, value in
                    data.__dict__.items() if not key.startswith('_')}
            except:
                return str(data)
        else:
            return data


def run_experiment():
    """Run a sample experiment using the curriculum learning framework."""
    if not TORCH_AVAILABLE:
        logger.error('PyTorch is required for this experiment')
        return
    output_dir = (
        'd:/MD/forex_trading_platform/experiments/rl_curriculum_learning')
    os.makedirs(output_dir, exist_ok=True)
    broker_sim = ForexBrokerSimulator(balance=10000.0, leverage=100.0)
    market_sim = AdvancedMarketRegimeSimulator(broker_simulator=broker_sim)
    market_gen = EnhancedMarketConditionGenerator(broker_simulator=
        broker_sim, base_volatility_map={'EUR/USD': 0.0007, 'GBP/USD': 0.0009})
    framework = CurriculumLearningFramework(broker_simulator=broker_sim,
        market_generator=market_gen, num_levels=3, symbols=['EUR/USD'],
        consecutive_successes_required=2, session_duration=timedelta(hours=24))

    def create_dqn_agent():
    """
    Create dqn agent.
    
    """

        input_shape = 20, 100
        num_actions = 4
        return DQNAgent(input_shape=input_shape, num_actions=num_actions,
            learning_rate=0.0005, batch_size=32, buffer_size=10000,
            target_update=100)
    trainer = CurriculumTrainer(framework=framework, agent_factory=
        create_dqn_agent, output_dir=output_dir, random_seed=42)
    results = trainer.run_curriculum_training(scenarios_per_level=2,
        episodes_per_scenario=50, save_checkpoints=True)
    final_agent_path = os.path.join(output_dir, 'final_agent.pt')
    trainer.agent.save(final_agent_path)
    benchmark = RLAgentBenchmark(market_simulator=market_sim,
        market_generator=market_gen, output_dir=os.path.join(output_dir,
        'benchmarks'))
    benchmark_results = benchmark.run_benchmark(agent=trainer.agent,
        category=BenchmarkCategory.MIXED, num_episodes=3)
    benchmark_summary = benchmark.get_benchmark_summary(list(benchmark.
        results.keys())[0])
    logger.info('=== Experiment Complete ===')
    logger.info(f'Final agent saved to: {final_agent_path}')
    logger.info(
        f"Curriculum progress: {results['final_progress']['curriculum_completion'] * 100:.1f}% complete"
        )
    logger.info(
        f"Benchmark overall score: {benchmark_summary['overall_score']:.4f}")
    with open(os.path.join(output_dir, 'experiment_summary.json'), 'w') as f:
        json.dump({'curriculum_completion': results['final_progress'][
            'curriculum_completion'], 'benchmark_score': benchmark_summary[
            'overall_score'], 'category_scores': benchmark_summary[
            'category_scores'], 'timestamp': datetime.now().isoformat()}, f,
            indent=2)


if __name__ == '__main__':
    run_experiment()
