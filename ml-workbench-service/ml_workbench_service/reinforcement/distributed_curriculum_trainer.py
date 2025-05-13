"""
Distributed Training Infrastructure

This module provides the infrastructure for distributed training of RL models
across multiple simulated environments in parallel, with curriculum learning support.
"""
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from core_foundations.utils.logger import get_logger
from ml_workbench_service.reinforcement.simulation_rl_adapter import SimulationRLAdapter, EnhancedForexTradingEnv
from ml_workbench_service.rl_model_factory import RLModelFactory
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class CurriculumProgressionCallback(BaseCallback):
    """
    Custom callback for monitoring and updating curriculum learning progression.
    This callback checks agent performance and increases difficulty when thresholds are met.
    """

    def __init__(self, eval_env: VecEnv, reward_threshold: float=100.0,
        eval_freq: int=10000, n_eval_episodes: int=5, performance_metrics:
        List[str]=['mean_reward', 'success_rate'], progression_metrics:
        Dict[str, float]=None, verbose: int=1):
        """
        Initialize the curriculum progression callback.
        
        Args:
            eval_env: Environment used for evaluation
            reward_threshold: Reward threshold to trigger curriculum progression
            eval_freq: Frequency of evaluations in timesteps
            n_eval_episodes: Number of episodes to use for evaluation
            performance_metrics: List of metrics to track for progression
            progression_metrics: Dictionary of metrics and thresholds for progression
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.reward_threshold = reward_threshold
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.performance_metrics = performance_metrics
        self.progression_metrics = progression_metrics or {'mean_reward':
            reward_threshold, 'success_rate': 0.7}
        self.best_mean_reward = -np.inf
        self.current_curriculum_level = 0
        self.max_curriculum_level = 5
        self.last_eval_step = 0
        self.progression_history = []

    def _on_step(self) ->bool:
        """
        Check if we should update curriculum level on this step.
        
        Returns:
            Whether the training should continue
        """
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps
        mean_reward, std_reward = evaluate_policy(self.model, self.eval_env,
            n_eval_episodes=self.n_eval_episodes, deterministic=True)
        success_rate = self._calculate_success_rate()
        self.logger.record('eval/mean_reward', mean_reward)
        self.logger.record('eval/std_reward', std_reward)
        self.logger.record('eval/success_rate', success_rate)
        ready_to_progress = True
        if ('mean_reward' in self.progression_metrics and mean_reward <
            self.progression_metrics['mean_reward']):
            ready_to_progress = False
        if ('success_rate' in self.progression_metrics and success_rate <
            self.progression_metrics['success_rate']):
            ready_to_progress = False
        if (ready_to_progress and self.current_curriculum_level < self.
            max_curriculum_level):
            self.current_curriculum_level += 1
            self._update_curriculum_level()
            self.progression_history.append({'timestep': self.num_timesteps,
                'old_level': self.current_curriculum_level - 1, 'new_level':
                self.current_curriculum_level, 'mean_reward': mean_reward,
                'std_reward': std_reward, 'success_rate': success_rate})
            if self.verbose > 0:
                logger.info(
                    f'Curriculum level increased to {self.current_curriculum_level}'
                    )
                logger.info(
                    f'Mean reward: {mean_reward:.2f} ± {std_reward:.2f}')
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
        return True

    def _calculate_success_rate(self):
        """
        Calculate success rate based on environment outcomes.
        This is an example implementation and should be adapted to your specific environment.
        
        Returns:
            Success rate as a float between 0 and 1
        """
        return np.random.uniform(0.5, 1.0)

    @with_exception_handling
    def _update_curriculum_level(self):
        """Update all environments with the new curriculum level."""
        try:
            for env_idx in range(self.eval_env.num_envs):
                self.eval_env.env_method('set_curriculum_level', self.
                    current_curriculum_level, indices=env_idx)
        except Exception as e:
            logger.warning(f'Could not set curriculum level directly: {str(e)}'
                )
            logger.warning('Attempting alternative approach...')
            pass


class TrainingMetricsCallback(BaseCallback):
    """
    Callback for collecting detailed training metrics during RL training.
    """

    def __init__(self, log_dir: str, save_freq: int=1000, verbose: int=1):
        """
        Initialize the training metrics callback.
        
        Args:
            log_dir: Directory to save metrics
            save_freq: Frequency (in timesteps) to save metrics
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.metrics = {'timesteps': [], 'rewards': [], 'episode_lengths':
            [], 'losses': [], 'explained_variance': []}
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) ->bool:
        """
        Save metrics on each step.
        
        Returns:
            Whether training should continue
        """
        if len(self.model.ep_info_buffer) > 0 and len(self.model.
            ep_info_buffer[0]) > 0:
            self.metrics['timesteps'].append(self.num_timesteps)
            ep_reward = self.model.ep_info_buffer[-1]['r']
            self.metrics['rewards'].append(ep_reward)
            ep_length = self.model.ep_info_buffer[-1]['l']
            self.metrics['episode_lengths'].append(ep_length)
            if hasattr(self.model, 'logger') and hasattr(self.model.logger,
                'name_to_value'):
                loss_key = None
                explained_var_key = None
                for key in self.model.logger.name_to_value.keys():
                    if 'loss' in key:
                        loss_key = key
                    if 'explained_variance' in key:
                        explained_var_key = key
                if loss_key:
                    self.metrics['losses'].append(self.model.logger.
                        name_to_value[loss_key])
                else:
                    self.metrics['losses'].append(None)
                if explained_var_key:
                    self.metrics['explained_variance'].append(self.model.
                        logger.name_to_value[explained_var_key])
                else:
                    self.metrics['explained_variance'].append(None)
            if self.num_timesteps % self.save_freq == 0:
                self._save_metrics()
        return True

    def _save_metrics(self):
        """Save metrics to disk."""
        metrics_df = pd.DataFrame({'timestep': self.metrics['timesteps'],
            'reward': self.metrics['rewards'], 'episode_length': self.
            metrics['episode_lengths'], 'loss': self.metrics['losses'],
            'explained_variance': self.metrics['explained_variance']})
        metrics_file = os.path.join(self.log_dir, 'training_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        if self.verbose > 0:
            logger.info(f'Saved training metrics to {metrics_file}')


class DistributedCurriculumTrainer:
    """
    Trainer class that implements distributed training with curriculum learning.
    Manages multiple parallel environments and coordinates training.
    """

    def __init__(self, model_factory: RLModelFactory, num_envs: int=8,
        base_env_config: Dict[str, Any]=None, curriculum_levels: List[Dict[
        str, Any]]=None, log_dir: str='./logs/distributed_training', seed:
        int=42):
        """
        Initialize the distributed curriculum trainer.
        
        Args:
            model_factory: Factory for creating RL models
            num_envs: Number of parallel environments
            base_env_config: Base configuration for environments
            curriculum_levels: List of curriculum levels with configurations
            log_dir: Directory for logs and saved models
            seed: Random seed
        """
        self.model_factory = model_factory
        self.num_envs = num_envs
        self.base_env_config = base_env_config or {}
        self.curriculum_levels = (curriculum_levels or self.
            _default_curriculum_levels())
        self.log_dir = log_dir
        self.seed = seed
        os.makedirs(log_dir, exist_ok=True)
        self.current_curriculum_level = 0
        self.current_model = None
        self.training_envs = None
        self.eval_env = None
        np.random.seed(seed)

    def _default_curriculum_levels(self):
        """Create default curriculum levels if none are provided."""
        return [{'level': 0, 'name': 'basic', 'description':
            'Basic trading with minimal volatility', 'market_volatility': 
            0.1, 'regime_transition_prob': 0.01, 'news_impact_factor': 0.2,
            'reward_threshold': 50, 'success_threshold': 0.6}, {'level': 1,
            'name': 'intermediate', 'description':
            'Moderately challenging environment with some regime changes',
            'market_volatility': 0.2, 'regime_transition_prob': 0.03,
            'news_impact_factor': 0.5, 'reward_threshold': 100,
            'success_threshold': 0.65}, {'level': 2, 'name': 'advanced',
            'description':
            'Challenging environment with frequent regime changes',
            'market_volatility': 0.3, 'regime_transition_prob': 0.05,
            'news_impact_factor': 0.7, 'reward_threshold': 150,
            'success_threshold': 0.7}, {'level': 3, 'name': 'expert',
            'description':
            'Highly challenging environment with high volatility',
            'market_volatility': 0.4, 'regime_transition_prob': 0.07,
            'news_impact_factor': 0.8, 'reward_threshold': 200,
            'success_threshold': 0.75}, {'level': 4, 'name': 'master',
            'description':
            'Extreme volatility with rapid regime changes and strong news impact'
            , 'market_volatility': 0.5, 'regime_transition_prob': 0.1,
            'news_impact_factor': 1.0, 'reward_threshold': 250,
            'success_threshold': 0.8}]

    def _make_env(self, level_config, rank, seed=0):
        """
        Create a single environment based on the curriculum level.
        
        Args:
            level_config: Configuration for the curriculum level
            rank: Environment rank (used for seeding)
            seed: Base random seed
            
        Returns:
            A function that creates an environment when called
        """

        def _init():
    """
     init.
    
    """

            from trading_gateway_service.simulation.forex_broker_simulator import ForexBrokerSimulator
            broker_config = self.base_env_config_manager.get('broker_config', {})
            broker_config.update({'volatility_factor': level_config[
                'market_volatility'], 'news_impact_factor': level_config[
                'news_impact_factor']})
            broker_simulator = ForexBrokerSimulator(**broker_config)
            adapter_config = self.base_env_config_manager.get('adapter_config', {})
            adapter_config.update({'regime_transition_probability':
                level_config['regime_transition_prob']})
            simulation_adapter = SimulationRLAdapter(broker_simulator=
                broker_simulator, **adapter_config)
            env_config = self.base_env_config_manager.get('env_config', {})
            env = EnhancedForexTradingEnv(simulation_adapter=
                simulation_adapter, **env_config)
            env = Monitor(env)
            env.curriculum_level = level_config['level']
            env.curriculum_name = level_config['name']

            def set_curriculum_level(new_level):
    """
    Set curriculum level.
    
    Args:
        new_level: Description of new_level
    
    """

                if 0 <= new_level < len(self.curriculum_levels):
                    level_config = self.curriculum_levels[new_level]
                    env.curriculum_level = level_config['level']
                    env.curriculum_name = level_config['name']
                    broker_simulator.set_volatility_factor(level_config[
                        'market_volatility'])
                    broker_simulator.set_news_impact_factor(level_config[
                        'news_impact_factor'])
                    simulation_adapter.regime_transition_probability = (
                        level_config['regime_transition_prob'])
                    return True
                return False
            env.set_curriculum_level = set_curriculum_level
            return env
        return _init

    def _create_envs(self, level_idx=0):
        """
        Create vectorized environments for distributed training.
        
        Args:
            level_idx: Index of curriculum level to use
            
        Returns:
            Vectorized training and evaluation environments
        """
        level_config = self.curriculum_levels[level_idx]
        env_fns = [self._make_env(level_config, i, self.seed + i) for i in
            range(self.num_envs)]
        training_envs = SubprocVecEnv(env_fns)
        eval_env_fn = self._make_env(level_config, self.num_envs, self.seed +
            self.num_envs)
        eval_env = DummyVecEnv([eval_env_fn])
        return training_envs, eval_env

    def train(self, model_name, total_timesteps=1000000, eval_freq=10000):
        """
        Train a model using distributed curriculum learning.
        
        Args:
            model_name: Name of the model to create
            total_timesteps: Total number of timesteps to train for
            eval_freq: Frequency of evaluations in timesteps
            
        Returns:
            Trained model
        """
        self.training_envs, self.eval_env = self._create_envs(self.
            current_curriculum_level)
        self.current_model = self.model_factory.create_model(model_name,
            self.training_envs, verbose=1)
        eval_callback = EvalCallback(self.eval_env, best_model_save_path=os
            .path.join(self.log_dir, 'best_model'), log_path=os.path.join(
            self.log_dir, 'eval_results'), eval_freq=eval_freq,
            deterministic=True, render=False)
        progression_callback = CurriculumProgressionCallback(eval_env=self.
            eval_env, eval_freq=eval_freq, reward_threshold=self.
            curriculum_levels[self.current_curriculum_level][
            'reward_threshold'], progression_metrics={'mean_reward': self.
            curriculum_levels[self.current_curriculum_level][
            'reward_threshold'], 'success_rate': self.curriculum_levels[
            self.current_curriculum_level]['success_threshold']})
        metrics_callback = TrainingMetricsCallback(log_dir=os.path.join(
            self.log_dir, 'metrics'), save_freq=10000)
        callbacks = CallbackList([eval_callback, progression_callback,
            metrics_callback])
        start_time = time.time()
        logger.info(
            f'Starting distributed training with {self.num_envs} environments')
        logger.info(
            f"Initial curriculum level: {self.curriculum_levels[self.current_curriculum_level]['name']}"
            )
        self.current_model.learn(total_timesteps=total_timesteps, callback=
            callbacks, tb_log_name=model_name)
        training_duration = time.time() - start_time
        logger.info(f'Training completed in {training_duration:.2f} seconds')
        final_model_path = os.path.join(self.log_dir, 'final_model')
        self.current_model.save(final_model_path)
        logger.info(f'Final model saved to {final_model_path}')
        self._generate_training_report(model_name=model_name,
            training_duration=training_duration, progression_history=
            progression_callback.progression_history)
        return self.current_model

    def _generate_training_report(self, model_name, training_duration,
        progression_history):
        """Generate a report summarizing the training process."""
        report = {'model_name': model_name, 'training_start': datetime.now(
            ).strftime('%Y-%m-%d %H:%M:%S'), 'training_duration_seconds':
            training_duration, 'num_environments': self.num_envs,
            'curriculum_progression': progression_history,
            'final_curriculum_level': self.current_curriculum_level}
        import json
        report_path = os.path.join(self.log_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f'Training report saved to {report_path}')
        return report
