"""
RL Model Factory for creating and configuring reinforcement learning models.

This module provides a factory for creating different reinforcement learning models
with standardized interfaces for the forex trading platform.
"""

from typing import Dict, Any, Optional, Union, List, Type
from enum import Enum
import logging
import os
import numpy as np

# Import RL algorithms from stable-baselines3
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from ml_workbench_service.models.reinforcement.enhanced_rl_env import EnhancedForexTradingEnv
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class RLAlgorithm(str, Enum):
    """Supported RL algorithms."""
    PPO = "PPO"
    A2C = "A2C"
    SAC = "SAC"
    TD3 = "TD3"
    DQN = "DQN"


class RLModelFactory:
    """
    Factory for creating and configuring reinforcement learning models.
    
    This class provides a standardized way to create different RL algorithms
    with appropriate configuration for forex trading tasks.
    """
    
    def __init__(self, default_tensorboard_log_dir: str = "./logs/rl_models/"):
        """
        Initialize the RL Model Factory.
        
        Args:
            default_tensorboard_log_dir: Directory for TensorBoard logs
        """
        self.tensorboard_log_dir = default_tensorboard_log_dir
        
        # Map algorithm names to their implementation classes
        self.algorithm_map = {
            RLAlgorithm.PPO: PPO,
            RLAlgorithm.A2C: A2C,
            RLAlgorithm.SAC: SAC,
            RLAlgorithm.TD3: TD3,
            RLAlgorithm.DQN: DQN
        }
        
        # Create tensorboard log directory if it doesn't exist
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        
    def create_model(
        self,
        algorithm: Union[str, RLAlgorithm],
        environment: EnhancedForexTradingEnv,
        params: Optional[Dict[str, Any]] = None,
        policy: str = "MlpPolicy",
        tensorboard_log: Optional[str] = None
    ) -> Any:
        """
        Create a new reinforcement learning model.
        
        Args:
            algorithm: The RL algorithm to use (e.g., "PPO", "SAC")
            environment: The trading environment to train on
            params: Algorithm-specific parameters
            policy: The policy network architecture 
            tensorboard_log: Path for tensorboard logging
            
        Returns:
            An initialized RL model
            
        Raises:
            ValueError: If the algorithm is not supported
        """
        # Handle string or enum value
        if isinstance(algorithm, str):
            try:
                algorithm = RLAlgorithm(algorithm.upper())
            except ValueError:
                raise ValueError(f"Unsupported algorithm: {algorithm}. " 
                               f"Supported algorithms: {[a.value for a in RLAlgorithm]}")
        
        # Get the algorithm class
        if algorithm not in self.algorithm_map:
            raise ValueError(f"Unsupported algorithm: {algorithm}. "
                           f"Supported algorithms: {[a.value for a in RLAlgorithm]}")
            
        algorithm_class = self.algorithm_map[algorithm]
        
        # Set up default parameters if none provided
        if params is None:
            params = self._get_default_params(algorithm)
        
        # Add tensorboard logging if requested
        if tensorboard_log is None and self.tensorboard_log_dir:
            tensorboard_log = os.path.join(self.tensorboard_log_dir, algorithm.value.lower())
            
        # Add exploration noise for off-policy algorithms
        if algorithm in [RLAlgorithm.SAC, RLAlgorithm.TD3] and 'action_noise' not in params:
            action_dim = environment.action_space.shape[0]
            params['action_noise'] = self._create_action_noise(algorithm, action_dim)
            
        # Create and return the model
        logger.info(f"Creating {algorithm.value} model with {policy} policy")
        return algorithm_class(
            policy=policy,
            env=environment,
            tensorboard_log=tensorboard_log,
            **params
        )
        
    def _get_default_params(self, algorithm: RLAlgorithm) -> Dict[str, Any]:
        """
        Get default hyperparameters for the specified algorithm.
        
        Args:
            algorithm: The RL algorithm
            
        Returns:
            Dictionary of default hyperparameters
        """
        if algorithm == RLAlgorithm.PPO:
            return {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "verbose": 1
            }
        elif algorithm == RLAlgorithm.A2C:
            return {
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "verbose": 1
            }
        elif algorithm == RLAlgorithm.SAC:
            return {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "verbose": 1
            }
        elif algorithm == RLAlgorithm.TD3:
            return {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 100,
                "tau": 0.005,
                "gamma": 0.99,
                "policy_delay": 2,
                "verbose": 1
            }
        elif algorithm == RLAlgorithm.DQN:
            return {
                "learning_rate": 1e-4,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 32,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 1000,
                "verbose": 1
            }
        else:
            return {}
            
    def _create_action_noise(self, algorithm: RLAlgorithm, action_dim: int) -> Any:
        """
        Create appropriate exploration noise for the algorithm.
        
        Args:
            algorithm: The RL algorithm
            action_dim: Dimension of the action space
            
        Returns:
            Action noise object
        """
        if algorithm == RLAlgorithm.TD3:
            return NormalActionNoise(
                mean=np.zeros(action_dim),
                sigma=0.1 * np.ones(action_dim)
            )
        elif algorithm == RLAlgorithm.SAC:
            return None  # SAC handles exploration internally
        else:
            return None
            
    def load_model(
        self,
        algorithm: Union[str, RLAlgorithm],
        model_path: str,
        environment: Optional[EnhancedForexTradingEnv] = None
    ) -> Any:
        """
        Load a saved reinforcement learning model.
        
        Args:
            algorithm: The RL algorithm of the saved model
            model_path: Path to the saved model
            environment: Optional environment to set for the model
            
        Returns:
            The loaded RL model
            
        Raises:
            ValueError: If the algorithm is not supported
            FileNotFoundError: If the model file doesn't exist
        """
        # Handle string or enum value
        if isinstance(algorithm, str):
            try:
                algorithm = RLAlgorithm(algorithm.upper())
            except ValueError:
                raise ValueError(f"Unsupported algorithm: {algorithm}. " 
                               f"Supported algorithms: {[a.value for a in RLAlgorithm]}")
        
        # Get the algorithm class
        if algorithm not in self.algorithm_map:
            raise ValueError(f"Unsupported algorithm: {algorithm}. "
                           f"Supported algorithms: {[a.value for a in RLAlgorithm]}")
            
        algorithm_class = self.algorithm_map[algorithm]
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load the model
        logger.info(f"Loading {algorithm.value} model from {model_path}")
        model = algorithm_class.load(model_path)
        
        # Set the environment if provided
        if environment is not None:
            model.set_env(environment)
            
        return model
        
    def create_ensemble(
        self,
        models_config: List[Dict[str, Any]],
        environment: EnhancedForexTradingEnv
    ) -> Dict[str, Any]:
        """
        Create an ensemble of RL models for more robust predictions.
        
        Args:
            models_config: List of model configurations, each with
                           'algorithm', 'params', and optionally 'weight'
            environment: The trading environment to use
            
        Returns:
            Dictionary of created models with their weights
            
        Example:
            models_config = [
                {'algorithm': 'PPO', 'params': {...}, 'weight': 1.0},
                {'algorithm': 'SAC', 'params': {...}, 'weight': 0.8}
            ]
        """
        ensemble = {}
        
        for config in models_config:
            algorithm = config.get('algorithm')
            params = config.get('params', {})
            weight = config.get('weight', 1.0)
            name = config.get('name', f"{algorithm}_{len(ensemble)}")
            
            model = self.create_model(
                algorithm=algorithm,
                environment=environment,
                params=params
            )
            
            ensemble[name] = {
                'model': model,
                'weight': weight
            }
            
        logger.info(f"Created ensemble with {len(ensemble)} models")
        return ensemble
    
    def get_supported_algorithms(self) -> List[str]:
        """
        Get list of supported RL algorithms.
        
        Returns:
            List of supported algorithm names
        """
        return [algo.value for algo in RLAlgorithm]
