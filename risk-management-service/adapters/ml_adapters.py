"""
ML Adapters Module

This module provides adapter implementations for ML interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
import asyncio
import json
import numpy as np

from common_lib.reinforcement.interfaces import (
    IRLEnvironment, IRLModel, IRLOptimizer, RLModelType
)
from common_lib.analysis.interfaces import (
    IMarketRegimeAnalyzer, MarketRegimeType, AnalysisTimeframe
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class RLEnvironmentAdapter(IRLEnvironment):
    """
    Adapter for reinforcement learning environments that implements the common interface.
    
    This adapter can either wrap an actual environment instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, environment_instance=None):
        """
        Initialize the adapter.
        
        Args:
            environment_instance: Optional actual environment instance to wrap
        """
        self.env = environment_instance
        self.current_step = 0
        self.max_steps = 1000
        self.observation_dim = 10
        self.current_state = np.zeros(self.observation_dim)
        self.current_position = 0.0
        self.account_balance = 10000.0
        self.last_price = 1.0
        self.price_history = [1.0]
        self.reward_history = []
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to an initial state.
        
        Returns:
            Initial observation
        """
        if self.env:
            try:
                # Try to use the wrapped environment if available
                return self.env.reset()
            except Exception as e:
                logger.warning(f"Error resetting RL environment: {str(e)}")
        
        # Fallback implementation
        logger.info("Using fallback RL environment reset")
        
        self.current_step = 0
        self.current_position = 0.0
        self.account_balance = 10000.0
        self.last_price = 1.0
        self.price_history = [1.0]
        self.reward_history = []
        
        # Generate a random initial state
        self.current_state = np.random.normal(0, 1, self.observation_dim)
        
        return {
            "observation": self.current_state,
            "account_balance": self.account_balance,
            "current_position": self.current_position,
            "last_price": self.last_price
        }
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.env:
            try:
                # Try to use the wrapped environment if available
                return self.env.step(action)
            except Exception as e:
                logger.warning(f"Error stepping RL environment: {str(e)}")
        
        # Fallback to simple step if no environment available
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Generate price movement
        price_change = np.random.normal(0, 0.002)
        new_price = self.last_price * (1 + price_change)
        self.price_history.append(new_price)
        
        # Calculate PnL
        position_change = float(action)
        old_position = self.current_position
        self.current_position = position_change
        
        # Calculate reward based on position and price change
        position_pnl = old_position * (new_price - self.last_price) * 10000
        self.account_balance += position_pnl
        
        # Apply trading costs
        trading_cost = abs(self.current_position - old_position) * 0.0001 * 10000
        self.account_balance -= trading_cost
        
        # Calculate reward
        reward = position_pnl - trading_cost
        self.reward_history.append(reward)
        
        # Update price
        self.last_price = new_price
        
        # Update state
        self.current_state = np.random.normal(0, 1, self.observation_dim)
        
        # Add price and position information to state
        observation = {
            "observation": self.current_state,
            "account_balance": self.account_balance,
            "current_position": self.current_position,
            "last_price": self.last_price
        }
        
        # Additional info
        info = {
            "position_pnl": position_pnl,
            "trading_cost": trading_cost,
            "price_change": price_change,
            "step": self.current_step
        }
        
        return observation, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[Any]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendering result, if any
        """
        if self.env:
            try:
                # Try to use the wrapped environment if available
                return self.env.render(mode)
            except Exception as e:
                logger.warning(f"Error rendering RL environment: {str(e)}")
        
        # Simple text rendering
        if mode == 'human':
            status = (
                f"Step: {self.current_step}, "
                f"Balance: ${self.account_balance:.2f}, "
                f"Position: {self.current_position:.2f}, "
                f"Price: {self.last_price:.5f}, "
                f"Reward: {self.reward_history[-1] if self.reward_history else 0:.2f}"
            )
            print(status)
            return status
        
        return None
    
    def close(self) -> None:
        """Close the environment and release resources."""
        if self.env:
            try:
                # Try to use the wrapped environment if available
                return self.env.close()
            except Exception as e:
                logger.warning(f"Error closing RL environment: {str(e)}")
        
        # Nothing to do for fallback implementation
        pass


class RLModelAdapter(IRLModel):
    """
    Adapter for reinforcement learning models that implements the common interface.
    
    This adapter can either wrap an actual model instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, model_instance=None):
        """
        Initialize the adapter.
        
        Args:
            model_instance: Optional actual model instance to wrap
        """
        self.model = model_instance
        self.prediction_history = []
        self.confidence_history = []
    
    def predict(self, state: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Generate a prediction from the model.
        
        Args:
            state: The current state representation
            
        Returns:
            Tuple of (action, metadata)
        """
        if self.model:
            try:
                # Try to use the wrapped model if available
                return self.model.predict(state)
            except Exception as e:
                logger.warning(f"Error predicting with RL model: {str(e)}")
        
        # Fallback implementation
        logger.info("Using fallback RL model prediction")
        
        # Generate a simple action (position size between -1 and 1)
        action = np.random.normal(0, 0.3)
        action = max(min(action, 1.0), -1.0)  # Clip to [-1, 1]
        
        # Generate metadata
        metadata = {
            "confidence": 0.6,
            "timestamp": datetime.now().isoformat(),
            "is_fallback": True
        }
        
        # Store in history
        self.prediction_history.append({
            "state": state,
            "action": action,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        })
        
        return action, metadata
    
    def get_confidence(self, state: Dict[str, Any], action: Any) -> float:
        """
        Get the confidence level for a prediction.
        
        Args:
            state: The current state representation
            action: The predicted action
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if self.model:
            try:
                # Try to use the wrapped model if available
                return self.model.get_confidence(state, action)
            except Exception as e:
                logger.warning(f"Error getting confidence from RL model: {str(e)}")
        
        # Fallback implementation
        logger.info("Using fallback RL model confidence")
        
        # Generate a simple confidence score
        confidence = 0.5 + 0.1 * np.random.random()
        
        # Adjust based on action magnitude
        if abs(action) > 0.8:
            confidence *= 0.8  # Lower confidence for extreme actions
        
        # Store in history
        self.confidence_history.append({
            "state": state,
            "action": action,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        return confidence
    
    def update(self, state: Dict[str, Any], action: Any, reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """
        Update the model with a new experience.
        
        Args:
            state: The starting state
            action: The action taken
            reward: The reward received
            next_state: The resulting state
            done: Whether the episode is done
        """
        if self.model:
            try:
                # Try to use the wrapped model if available
                return self.model.update(state, action, reward, next_state, done)
            except Exception as e:
                logger.warning(f"Error updating RL model: {str(e)}")
        
        # Fallback implementation - just log the update
        logger.info(f"Fallback RL model update: action={action}, reward={reward}, done={done}")


class RLOptimizerAdapter(IRLOptimizer):
    """
    Adapter for RL-based parameter optimizers that implements the common interface.
    
    This adapter can either wrap an actual optimizer instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, optimizer_instance=None):
        """
        Initialize the adapter.
        
        Args:
            optimizer_instance: Optional actual optimizer instance to wrap
        """
        self.optimizer = optimizer_instance
        self.optimization_history = []
        self.last_confidence = 0.5
    
    async def optimize_parameters(
        self,
        parameter_type: str,
        current_values: Dict[str, Any],
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize parameters using RL insights.
        
        Args:
            parameter_type: Type of parameters to optimize
            current_values: Current parameter values
            context: Contextual information for optimization
            constraints: Optional constraints on parameter values
            
        Returns:
            Optimized parameter values
        """
        if self.optimizer:
            try:
                # Try to use the wrapped optimizer if available
                return await self.optimizer.optimize_parameters(
                    parameter_type=parameter_type,
                    current_values=current_values,
                    context=context,
                    constraints=constraints
                )
            except Exception as e:
                logger.warning(f"Error optimizing parameters with RL optimizer: {str(e)}")
        
        # Fallback implementation
        logger.info(f"Using fallback RL parameter optimization for {parameter_type}")
        
        # Start with current values
        optimized_values = current_values.copy()
        
        # Apply simple adjustments based on parameter type
        if parameter_type == "risk_parameters":
            # Get market regime from context if available
            market_regime = context.get("market_regime", "unknown")
            
            if market_regime == "trending_bullish":
                # More aggressive in bullish trends
                if "position_size_multiplier" in optimized_values:
                    optimized_values["position_size_multiplier"] *= 1.1
                if "stop_loss_multiplier" in optimized_values:
                    optimized_values["stop_loss_multiplier"] *= 1.1
            elif market_regime == "trending_bearish":
                # More conservative in bearish trends
                if "position_size_multiplier" in optimized_values:
                    optimized_values["position_size_multiplier"] *= 0.9
                if "stop_loss_multiplier" in optimized_values:
                    optimized_values["stop_loss_multiplier"] *= 0.9
            elif market_regime == "volatile":
                # Tighter risk controls in volatile markets
                if "position_size_multiplier" in optimized_values:
                    optimized_values["position_size_multiplier"] *= 0.8
                if "stop_loss_multiplier" in optimized_values:
                    optimized_values["stop_loss_multiplier"] *= 0.8
        
        elif parameter_type == "trading_parameters":
            # Simple adjustments for trading parameters
            if "entry_threshold" in optimized_values:
                optimized_values["entry_threshold"] *= 1.05
            if "exit_threshold" in optimized_values:
                optimized_values["exit_threshold"] *= 0.95
        
        # Apply constraints if provided
        if constraints:
            for param, constraint in constraints.items():
                if param in optimized_values:
                    if "min" in constraint:
                        optimized_values[param] = max(optimized_values[param], constraint["min"])
                    if "max" in constraint:
                        optimized_values[param] = min(optimized_values[param], constraint["max"])
        
        # Store in history
        self.optimization_history.append({
            "parameter_type": parameter_type,
            "current_values": current_values,
            "optimized_values": optimized_values,
            "context": context,
            "constraints": constraints,
            "timestamp": datetime.now().isoformat()
        })
        
        # Set confidence
        self.last_confidence = 0.6
        
        return optimized_values
    
    def get_optimization_confidence(self) -> float:
        """
        Get the confidence level of the last optimization.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if self.optimizer:
            try:
                # Try to use the wrapped optimizer if available
                return self.optimizer.get_optimization_confidence()
            except Exception as e:
                logger.warning(f"Error getting optimization confidence: {str(e)}")
        
        # Return the last confidence
        return self.last_confidence


class MarketRegimeAnalyzerAdapter(IMarketRegimeAnalyzer):
    """
    Adapter for market regime analyzer that implements the common interface.
    
    This adapter can either wrap an actual analyzer instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, analyzer_instance=None):
        """
        Initialize the adapter.
        
        Args:
            analyzer_instance: Optional actual analyzer instance to wrap
        """
        self.analyzer = analyzer_instance
        self.regime_history = {}
    
    async def detect_regime(
        self,
        symbol: str,
        timeframe: Union[str, AnalysisTimeframe],
        lookback_periods: int = 100
    ) -> Dict[str, Any]:
        """
        Detect the current market regime for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary with regime information
        """
        if self.analyzer:
            try:
                # Try to use the wrapped analyzer if available
                return await self.analyzer.detect_regime(
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_periods=lookback_periods
                )
            except Exception as e:
                logger.warning(f"Error detecting market regime: {str(e)}")
        
        # Fallback implementation
        logger.info(f"Using fallback market regime detection for {symbol} {timeframe}")
        
        # Generate fallback regime detection
        regime_type = MarketRegimeType.RANGING_NARROW
        if symbol.startswith("EUR"):
            regime_type = MarketRegimeType.TRENDING_BULLISH
        elif symbol.startswith("USD"):
            regime_type = MarketRegimeType.TRENDING_BEARISH
        
        # Store in history
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        
        self.regime_history[symbol].append({
            "timestamp": datetime.now().isoformat(),
            "regime": regime_type,
            "timeframe": str(timeframe)
        })
        
        # Keep only recent history
        if len(self.regime_history[symbol]) > 10:
            self.regime_history[symbol] = self.regime_history[symbol][-10:]
        
        return {
            "regime_type": regime_type,
            "confidence": 0.75,
            "regime_metrics": {
                "trend_strength": 0.6,
                "volatility": 0.4,
                "momentum": 0.5
            },
            "regime_history": self.regime_history[symbol]
        }
    
    async def get_regime_probabilities(
        self,
        symbol: str,
        timeframe: Union[str, AnalysisTimeframe]
    ) -> Dict[MarketRegimeType, float]:
        """
        Get probability distribution across different regime types.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            
        Returns:
            Dictionary mapping regime types to their probabilities
        """
        if self.analyzer:
            try:
                # Try to use the wrapped analyzer if available
                return await self.analyzer.get_regime_probabilities(
                    symbol=symbol,
                    timeframe=timeframe
                )
            except Exception as e:
                logger.warning(f"Error getting regime probabilities: {str(e)}")
        
        # Fallback implementation
        logger.info(f"Using fallback regime probabilities for {symbol} {timeframe}")
        
        # Generate fallback regime probabilities
        base_probs = {
            MarketRegimeType.TRENDING_BULLISH: 0.1,
            MarketRegimeType.TRENDING_BEARISH: 0.1,
            MarketRegimeType.RANGING_NARROW: 0.3,
            MarketRegimeType.RANGING_WIDE: 0.2,
            MarketRegimeType.VOLATILE: 0.1,
            MarketRegimeType.CHOPPY: 0.1,
            MarketRegimeType.BREAKOUT: 0.05,
            MarketRegimeType.REVERSAL: 0.05
        }
        
        # Adjust based on symbol
        if symbol.startswith("EUR"):
            base_probs[MarketRegimeType.TRENDING_BULLISH] = 0.3
            base_probs[MarketRegimeType.RANGING_NARROW] = 0.2
        elif symbol.startswith("USD"):
            base_probs[MarketRegimeType.TRENDING_BEARISH] = 0.3
            base_probs[MarketRegimeType.RANGING_NARROW] = 0.2
        
        return base_probs
    
    async def get_regime_transition_probability(
        self,
        symbol: str,
        from_regime: MarketRegimeType,
        to_regime: MarketRegimeType,
        timeframe: Union[str, AnalysisTimeframe]
    ) -> float:
        """
        Get the probability of transitioning between regimes.
        
        Args:
            symbol: The trading symbol
            from_regime: Starting regime type
            to_regime: Target regime type
            timeframe: The timeframe to analyze
            
        Returns:
            Probability of transition (0.0 to 1.0)
        """
        if self.analyzer:
            try:
                # Try to use the wrapped analyzer if available
                return await self.analyzer.get_regime_transition_probability(
                    symbol=symbol,
                    from_regime=from_regime,
                    to_regime=to_regime,
                    timeframe=timeframe
                )
            except Exception as e:
                logger.warning(f"Error getting regime transition probability: {str(e)}")
        
        # Fallback implementation
        logger.info(f"Using fallback regime transition probability for {symbol} {timeframe}")
        
        # Define some basic transition probabilities
        if from_regime == to_regime:
            # High probability of staying in the same regime
            return 0.7
        
        # Logical transitions have higher probabilities
        if from_regime == MarketRegimeType.RANGING_NARROW and to_regime == MarketRegimeType.BREAKOUT:
            return 0.3
        if from_regime == MarketRegimeType.VOLATILE and to_regime == MarketRegimeType.RANGING_WIDE:
            return 0.25
        if from_regime == MarketRegimeType.TRENDING_BULLISH and to_regime == MarketRegimeType.REVERSAL:
            return 0.15
        if from_regime == MarketRegimeType.TRENDING_BEARISH and to_regime == MarketRegimeType.REVERSAL:
            return 0.15
        
        # Default low probability for other transitions
        return 0.1
