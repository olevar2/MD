"""
Enhanced RL Effectiveness Framework 

This module provides comprehensive evaluation tools for reinforcement learning models,
including regime-specific performance metrics, statistical significance testing,
and comparative analysis across different market conditions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import json
from scipy import stats
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

from core_foundations.utils.logger import get_logger
from trading_gateway_service.simulation.forex_broker_simulator import MarketRegimeType
from ml_workbench_service.models.reinforcement.rl_agent import ForexTradingEnvironment

logger = get_logger(__name__)


class EvaluationMetricType(Enum):
    """Types of evaluation metrics for RL agents."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    VOLATILITY = "volatility"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    TRADES_PER_DAY = "trades_per_day"
    AVG_TRADE_DURATION = "avg_trade_duration"
    CONSISTENCY_SCORE = "consistency_score"


@dataclass
class TradeRecord:
    """Record of a single trade executed by the RL agent."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    symbol: str = "EURUSD"  # Default symbol
    direction: str = ""  # "long" or "short"
    entry_price: float = 0.0
    exit_price: float = 0.0
    position_size: float = 0.0
    pnl: float = 0.0
    realized_pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    market_regime: str = ""
    news_events: List[str] = field(default_factory=list)
    liquidity_level: str = ""
    trade_reason: str = ""
    exit_reason: str = ""
    duration_minutes: float = 0.0
    
    def calculate_metrics(self) -> None:
        """Calculate derived metrics for the trade."""
        if self.exit_time:
            self.duration_minutes = (self.exit_time - self.entry_time).total_seconds() / 60
            
        if self.entry_price > 0 and self.exit_price > 0:
            pip_value = 0.0001  # For most forex pairs
            pips = (self.exit_price - self.entry_price) / pip_value
            
            # Adjust direction
            if self.direction == "short":
                pips = -pips
                
            self.pnl = pips * self.position_size - self.fees - self.slippage
            self.realized_pnl = self.pnl if self.exit_price > 0 else 0.0


@dataclass
class PerformanceReport:
    """Comprehensive performance report for an RL agent."""
    model_name: str
    evaluation_id: str
    start_time: datetime
    end_time: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    trade_duration_avg: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    
    # Regime-specific performance
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Confidence intervals and statistical significance
    win_rate_ci_95: Tuple[float, float] = (0.0, 0.0)
    sharpe_ratio_ci_95: Tuple[float, float] = (0.0, 0.0)
    
    # Effectiveness scores
    overall_effectiveness: float = 0.0
    regime_effectiveness: Dict[str, float] = field(default_factory=dict)


class RLEffectivenessFramework:
    """
    Enhanced framework for evaluating the effectiveness of RL trading agents 
    with detailed analytics, statistical significance testing, and regime-specific analysis.
    """
    
    def __init__(
        self,
        base_log_dir: str = "./logs/effectiveness",
        significance_level: float = 0.05,
        enable_visualization: bool = True,
        baseline_strategies: List[str] = None,
    ):
        """
        Initialize the RL effectiveness framework.
        
        Args:
            base_log_dir: Base directory for storing evaluation results
            significance_level: Statistical significance level (default: 0.05)
            enable_visualization: Whether to generate visualization charts
            baseline_strategies: List of baseline strategy names to compare against
        """
        self.base_log_dir = base_log_dir
        self.significance_level = significance_level
        self.enable_visualization = enable_visualization
        self.baseline_strategies = baseline_strategies or ["buy_and_hold", "moving_average_cross"]
        
        # Create log directory if it doesn't exist
        os.makedirs(base_log_dir, exist_ok=True)
        
        # Initialize storage for results
        self.evaluation_results = {}
        self.comparative_results = {}
        
        logger.info(f"Initialized RL Effectiveness Framework. Log directory: {base_log_dir}")
    
    def evaluate_model(
        self,
        model_name: str,
        environment: ForexTradingEnvironment,
        n_episodes: int = 100,
        evaluation_id: str = None,
        market_regimes: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> PerformanceReport:
        """
        Evaluate an RL model's performance across specified market regimes.
        
        Args:
            model_name: Name of the RL model to evaluate
            environment: The trading environment for evaluation
            n_episodes: Number of episodes to run for evaluation
            evaluation_id: Custom ID for this evaluation run
            market_regimes: List of market regimes to test (default: all regimes)
            metadata: Additional metadata to store with results
            
        Returns:
            PerformanceReport containing comprehensive evaluation results
        """
        if evaluation_id is None:
            evaluation_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Set up storage for this evaluation
        log_dir = os.path.join(self.base_log_dir, evaluation_id)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize trade history
        all_trades = []
        
        # Set up market regimes
        market_regimes = market_regimes or [r.value for r in MarketRegimeType]
        
        # Prepare results by market regime
        regime_results = {regime: [] for regime in market_regimes}
        
        # Record start time
        start_time = datetime.now()
        
        # Run evaluation episodes
        for episode in range(n_episodes):
            # Select market regime to test if specified
            if episode % len(market_regimes) == 0:
                np.random.shuffle(market_regimes)  # Randomize regime order
            
            current_regime = market_regimes[episode % len(market_regimes)]
            
            # Configure environment for this regime
            self._configure_environment_for_regime(environment, current_regime)
            
            # Run the episode
            episode_trades, episode_pnl = self._run_evaluation_episode(
                model_name=model_name,
                environment=environment,
                market_regime=current_regime
            )
            
            # Store results
            all_trades.extend(episode_trades)
            regime_results[current_regime].append(episode_pnl)
            
            # Log progress
            if (episode + 1) % 10 == 0 or episode == n_episodes - 1:
                logger.info(f"Completed {episode + 1}/{n_episodes} evaluation episodes")
        
        # Record end time
        end_time = datetime.now()
        
        # Generate performance report
        report = self._generate_performance_report(
            model_name=model_name,
            evaluation_id=evaluation_id,
            trades=all_trades,
            regime_results=regime_results,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata
        )
        
        # Save detailed results
        self._save_evaluation_results(
            log_dir=log_dir,
            report=report,
            trades=all_trades,
            regime_results=regime_results,
            metadata=metadata
        )
        
        # Generate visualizations if enabled
        if self.enable_visualization:
            self._generate_visualizations(log_dir, report, all_trades, regime_results)
        
        # Store results for comparative analysis
        self.evaluation_results[evaluation_id] = report
        
        return report
    
    def _configure_environment_for_regime(
        self,
        environment: ForexTradingEnvironment,
        regime: str
    ) -> None:
        """
        Configure the trading environment for a specific market regime.
        
        Args:
            environment: The trading environment to configure
            regime: Market regime to configure for
        """
        # This implementation depends on the environment's API
        try:
            # Try configuring via simulation adapter if available
            if hasattr(environment, "simulation_adapter"):
                environment.simulation_adapter.broker_simulator.set_market_regime(
                    MarketRegimeType(regime)
                )
            # Alternative approach using environment method if available
            elif hasattr(environment, "set_market_regime"):
                environment.set_market_regime(regime)
            # Last resort: reset with parameters if supported
            else:
                environment.reset(market_regime=regime)
                
        except Exception as e:
            logger.warning(f"Could not configure environment for regime {regime}: {str(e)}")
            logger.warning("Proceeding with default environment configuration")

    def _run_evaluation_episode(
        self,
        model_name: str,
        environment: ForexTradingEnvironment,
        market_regime: str
    ) -> Tuple[List[TradeRecord], float]:
        """
        Run a single evaluation episode with the specified model and environment.
        
        Args:
            model_name: Name of the RL model to evaluate
            environment: The trading environment for evaluation
            market_regime: Current market regime for this episode
            
        Returns:
            Tuple of (list of trade records, episode PnL)
        """
        # Load the model (in a real implementation, this would use a model library)
        # For this example, we'll create a simple random agent
        def random_agent(observation):
            return np.random.randint(0, environment.action_space.n)
        
        # This is a placeholder. In a real implementation, you would:
        # 1. Load the actual model
        # 2. Get the action from the model for each step
        
        # Initialize episode
        observation = environment.reset()
        done = False
        episode_reward = 0
        trades = []
        current_trade = None
        
        # Run episode until done
        while not done:
            # Get action from model (using random agent as placeholder)
            action = random_agent(observation)
            
            # Take step in environment
            next_observation, reward, done, info = environment.step(action)
            
            # Track episode reward
            episode_reward += reward
            
            # Extract trade information from info
            if "trade" in info and info["trade"] is not None:
                trade_info = info["trade"]
                
                # Check if a new trade was opened
                if trade_info.get("action") == "open" and current_trade is None:
                    current_trade = TradeRecord(
                        entry_time=datetime.now(),
                        symbol=trade_info.get("symbol", "EURUSD"),
                        direction=trade_info.get("direction", ""),
                        entry_price=trade_info.get("price", 0.0),
                        position_size=trade_info.get("size", 0.0),
                        market_regime=market_regime,
                        trade_reason=trade_info.get("reason", "")
                    )
                    
                # Check if an existing trade was closed
                elif trade_info.get("action") == "close" and current_trade is not None:
                    current_trade.exit_time = datetime.now()
                    current_trade.exit_price = trade_info.get("price", 0.0)
                    current_trade.pnl = trade_info.get("pnl", 0.0)
                    current_trade.fees = trade_info.get("fees", 0.0)
                    current_trade.slippage = trade_info.get("slippage", 0.0)
                    current_trade.exit_reason = trade_info.get("reason", "")
                    
                    # Calculate additional metrics
                    current_trade.calculate_metrics()
                    
                    # Add to trades list
                    trades.append(current_trade)
                    current_trade = None
            
            # Update observation
            observation = next_observation
        
        # If episode ended with open trade, close it
        if current_trade is not None:
            current_trade.exit_time = datetime.now()
            current_trade.exit_price = environment.get_current_price()
            current_trade.calculate_metrics()
            trades.append(current_trade)
        
        return trades, episode_reward
    
    def _generate_performance_report(
        self,
        model_name: str,
        evaluation_id: str,
        trades: List[TradeRecord],
        regime_results: Dict[str, List[float]],
        start_time: datetime,
        end_time: datetime,
        metadata: Dict[str, Any] = None
    ) -> PerformanceReport:
        """
        Generate a comprehensive performance report from evaluation results.
        
        Args:
            model_name: Name of the RL model evaluated
            evaluation_id: ID for this evaluation run
            trades: List of trade records from evaluation
            regime_results: Results by market regime
            start_time: Evaluation start time
            end_time: Evaluation end time
            metadata: Additional metadata for the report
            
        Returns:
            PerformanceReport with comprehensive metrics
        """
        # Initialize report
        report = PerformanceReport(
            model_name=model_name,
            evaluation_id=evaluation_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Basic trade statistics
        if trades:
            report.total_trades = len(trades)
            report.winning_trades = sum(1 for t in trades if t.pnl > 0)
            report.losing_trades = sum(1 for t in trades if t.pnl <= 0)
            report.total_pnl = sum(t.pnl for t in trades)
            report.total_fees = sum(t.fees for t in trades)
            report.total_slippage = sum(t.slippage for t in trades)
            
            # Win rate calculation
            report.win_rate = report.winning_trades / report.total_trades if report.total_trades > 0 else 0
            
            # Average win/loss stats
            winning_trades = [t.pnl for t in trades if t.pnl > 0]
            losing_trades = [t.pnl for t in trades if t.pnl <= 0]
            
            report.avg_win = np.mean(winning_trades) if winning_trades else 0
            report.avg_loss = np.mean(losing_trades) if losing_trades else 0
            report.largest_win = np.max(winning_trades) if winning_trades else 0
            report.largest_loss = np.min(losing_trades) if losing_trades else 0
            
            # Duration statistics
            durations = [t.duration_minutes for t in trades if t.duration_minutes > 0]
            report.trade_duration_avg = np.mean(durations) if durations else 0
            
            # Build equity curve
            equity = 0
            equity_curve = [equity]
            drawdowns = [0]
            peak = 0
            
            for trade in trades:
                equity += trade.pnl
                equity_curve.append(equity)
                
                # Calculate drawdown
                peak = max(peak, equity)
                drawdown = (peak - equity) / (peak + 1e-10)  # Avoid division by zero
                drawdowns.append(drawdown)
            
            report.equity_curve = equity_curve
            report.drawdown_curve = drawdowns
            report.max_drawdown = max(drawdowns) if drawdowns else 0
            
            # Calculate daily returns (simplified for this example)
            daily_returns = []
            day_pnl = 0
            current_day = trades[0].entry_time.date()
            
            for trade in trades:
                if trade.exit_time:
                    trade_day = trade.exit_time.date()
                    if trade_day != current_day:
                        daily_returns.append(day_pnl)
                        day_pnl = 0
                        current_day = trade_day
                    day_pnl += trade.pnl
            
            # Add the last day
            if day_pnl != 0:
                daily_returns.append(day_pnl)
                
            report.daily_returns = daily_returns
            
            # Calculate risk-adjusted metrics
            if daily_returns:
                # Sharpe ratio (simplified - assuming 0% risk-free rate)
                returns_mean = np.mean(daily_returns)
                returns_std = np.std(daily_returns) if len(daily_returns) > 1 else 1e-10
                report.sharpe_ratio = returns_mean / returns_std if returns_std > 0 else 0
                
                # Sortino ratio (only using negative returns for denominator)
                neg_returns = [r for r in daily_returns if r < 0]
                neg_returns_std = np.std(neg_returns) if len(neg_returns) > 1 else 1e-10
                report.sortino_ratio = returns_mean / neg_returns_std if neg_returns_std > 0 else 0
                
                # Calmar ratio (annualized return / max drawdown)
                annual_return = returns_mean * 252  # Assuming 252 trading days
                report.calmar_ratio = annual_return / (report.max_drawdown + 1e-10)
                
            # Profit factor
            total_gains = sum(t.pnl for t in trades if t.pnl > 0)
            total_losses = sum(abs(t.pnl) for t in trades if t.pnl < 0)
            report.profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
            
            # Recovery factor
            report.recovery_factor = report.total_pnl / (report.max_drawdown + 1e-10)
            
            # Calculate 95% confidence intervals
            report.win_rate_ci_95 = self._calculate_win_rate_confidence_interval(report.win_rate, report.total_trades)
            report.sharpe_ratio_ci_95 = self._calculate_sharpe_confidence_interval(report.sharpe_ratio, len(daily_returns))
            
        # Regime-specific performance
        for regime, results in regime_results.items():
            if results:
                regime_trades = [t for t in trades if t.market_regime == regime]
                
                regime_report = {
                    "mean_reward": np.mean(results),
                    "std_reward": np.std(results),
                    "min_reward": np.min(results),
                    "max_reward": np.max(results),
                    "total_pnl": sum(t.pnl for t in regime_trades),
                    "win_rate": (sum(1 for t in regime_trades if t.pnl > 0) / len(regime_trades)) 
                              if regime_trades else 0,
                    "trade_count": len(regime_trades)
                }
                
                report.regime_performance[regime] = regime_report
        
        # Calculate overall effectiveness score (composite score from 0-100)
        effectiveness_scores = {
            "profit_score": min(100, max(0, report.total_pnl / (report.total_trades * 0.1) * 20)) if report.total_trades > 0 else 0,
            "risk_score": min(100, max(0, (1 - report.max_drawdown) * 100)),
            "win_rate_score": min(100, max(0, report.win_rate * 100)),
            "consistency_score": min(100, max(0, report.profit_factor * 10)) if report.profit_factor < 10 else 100
        }
        
        # Overall effectiveness is weighted average of component scores
        weights = {"profit_score": 0.35, "risk_score": 0.25, "win_rate_score": 0.25, "consistency_score": 0.15}
        report.overall_effectiveness = sum(score * weights[key] for key, score in effectiveness_scores.items())
        
        # Calculate regime-specific effectiveness
        for regime, perf in report.regime_performance.items():
            if perf["trade_count"] > 0:
                regime_eff = (
                    0.4 * (perf["mean_reward"] * 10)  # Scale reward to 0-100
                    + 0.4 * (perf["win_rate"] * 100)   # Win rate as percentage
                    + 0.2 * (perf["total_pnl"] / (perf["trade_count"] * 0.1) * 20)  # Profit per trade
                )
                report.regime_effectiveness[regime] = min(100, max(0, regime_eff))
            else:
                report.regime_effectiveness[regime] = 0
                
        return report
    
    def _calculate_win_rate_confidence_interval(
        self,
        win_rate: float,
        n_trades: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for win rate using binomial proportion confidence interval.
        
        Args:
            win_rate: Observed win rate
            n_trades: Number of trades
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        if n_trades == 0:
            return (0.0, 0.0)
            
        # Use Wilson score interval for binomial proportion confidence interval
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        denominator = 1 + z**2 / n_trades
        centre_adjusted_probability = win_rate + z**2 / (2 * n_trades)
        adjusted_standard_deviation = np.sqrt(win_rate * (1 - win_rate) / n_trades + z**2 / (4 * n_trades**2))
        
        lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
        upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
        
        return (max(0, lower_bound), min(1, upper_bound))
    
    def _calculate_sharpe_confidence_interval(
        self,
        sharpe: float,
        n_returns: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for Sharpe ratio.
        
        Args:
            sharpe: Observed Sharpe ratio
            n_returns: Number of return observations
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        if n_returns <= 1:
            return (0.0, 0.0)
            
        # Standard error of Sharpe ratio
        se = np.sqrt((1 + sharpe**2 / 2) / n_returns)
        
        # Critical value for the confidence level
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        lower_bound = sharpe - z * se
        upper_bound = sharpe + z * se
        
        return (lower_bound, upper_bound)
    
    def _save_evaluation_results(
        self,
        log_dir: str,
        report: PerformanceReport,
        trades: List[TradeRecord],
        regime_results: Dict[str, List[float]],
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Save evaluation results to disk.
        
        Args:
            log_dir: Directory to save results
            report: Performance report
            trades: List of trade records
            regime_results: Results by market regime
            metadata: Additional metadata to save
        """
        # Convert report to dictionary for serialization
        report_dict = {
            'model_name': report.model_name,
            'evaluation_id': report.evaluation_id,
            'start_time': report.start_time.isoformat(),
            'end_time': report.end_time.isoformat(),
            'total_trades': report.total_trades,
            'winning_trades': report.winning_trades,
            'losing_trades': report.losing_trades,
            'total_pnl': report.total_pnl,
            'total_fees': report.total_fees,
            'total_slippage': report.total_slippage,
            'max_drawdown': report.max_drawdown,
            'sharpe_ratio': report.sharpe_ratio,
            'sortino_ratio': report.sortino_ratio,
            'calmar_ratio': report.calmar_ratio,
            'profit_factor': report.profit_factor,
            'recovery_factor': report.recovery_factor,
            'win_rate': report.win_rate,
            'avg_win': report.avg_win,
            'avg_loss': report.avg_loss,
            'largest_win': report.largest_win,
            'largest_loss': report.largest_loss,
            'trade_duration_avg': report.trade_duration_avg,
            'win_rate_ci_95': report.win_rate_ci_95,
            'sharpe_ratio_ci_95': report.sharpe_ratio_ci_95,
            'overall_effectiveness': report.overall_effectiveness,
            'regime_effectiveness': report.regime_effectiveness,
            'regime_performance': report.regime_performance,
            'metadata': metadata or {}
        }
        
        # Save report as JSON
        report_path = os.path.join(log_dir, "performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Save trades as CSV
        trades_data = []
        for t in trades:
            trades_data.append({
                'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'position_size': t.position_size,
                'pnl': t.pnl,
                'fees': t.fees,
                'slippage': t.slippage,
                'market_regime': t.market_regime,
                'duration_minutes': t.duration_minutes,
                'trade_reason': t.trade_reason,
                'exit_reason': t.exit_reason
            })
        
        trades_df = pd.DataFrame(trades_data)
        trades_path = os.path.join(log_dir, "trades.csv")
        trades_df.to_csv(trades_path, index=False)
        
        # Save equity curve
        equity_data = {
            'step': list(range(len(report.equity_curve))),
            'equity': report.equity_curve,
            'drawdown': report.drawdown_curve
        }
        equity_df = pd.DataFrame(equity_data)
        equity_path = os.path.join(log_dir, "equity_curve.csv")
        equity_df.to_csv(equity_path, index=False)
        
        # Save regime results
        regime_data = []
        for regime, results in regime_results.items():
            for i, result in enumerate(results):
                regime_data.append({
                    'regime': regime,
                    'episode': i,
                    'reward': result
                })
        
        regime_df = pd.DataFrame(regime_data)
        regime_path = os.path.join(log_dir, "regime_results.csv")
        regime_df.to_csv(regime_path, index=False)
        
        logger.info(f"Evaluation results saved to {log_dir}")

    def _generate_visualizations(
        self,
        log_dir: str,
        report: PerformanceReport,
        trades: List[TradeRecord],
        regime_results: Dict[str, List[float]]
    ) -> None:
        """
        Generate visualizations from evaluation results.
        
        Args:
            log_dir: Directory to save visualizations
            report: Performance report
            trades: List of trade records
            regime_results: Results by market regime
        """
        # Create visualizations directory
        viz_dir = os.path.join(log_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Equity Curve with Drawdowns
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        
        # Plot equity curve
        steps = list(range(len(report.equity_curve)))
        ax1.plot(steps, report.equity_curve, 'b-', linewidth=2, label='Equity')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Equity', color='b')
        ax1.tick_params('y', colors='b')
        
        # Plot drawdowns on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(steps, report.drawdown_curve, 'r-', alpha=0.5, label='Drawdown')
        ax2.fill_between(steps, 0, report.drawdown_curve, color='r', alpha=0.2)
        ax2.set_ylabel('Drawdown (%)', color='r')
        ax2.tick_params('y', colors='r')
        
        plt.title(f'Equity Curve and Drawdowns - {report.model_name}')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'equity_curve.png'))
        plt.close()
        
        # 2. Performance by Market Regime
        if report.regime_performance:
            regimes = list(report.regime_performance.keys())
            rewards = [report.regime_performance[r]['mean_reward'] for r in regimes]
            win_rates = [report.regime_performance[r]['win_rate'] * 100 for r in regimes]
            trade_counts = [report.regime_performance[r]['trade_count'] for r in regimes]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # Mean reward by regime
            sns.barplot(x=regimes, y=rewards, ax=ax1)
            ax1.set_title(f'Mean Reward by Market Regime - {report.model_name}')
            ax1.set_ylabel('Mean Reward')
            ax1.set_xlabel('Market Regime')
            
            # Add text annotations
            for i, v in enumerate(rewards):
                ax1.text(i, v + 0.1, f'{v:.2f}', ha='center')
            
            # Win rate by regime
            sns.barplot(x=regimes, y=win_rates, ax=ax2)
            for i, v in enumerate(win_rates):
                ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center')
                
            # Add trade count as text
            for i, count in enumerate(trade_counts):
                ax2.text(i, 0, f'n={count}', ha='center', va='bottom')
                
            ax2.set_title('Win Rate by Market Regime')
            ax2.set_ylabel('Win Rate (%)')
            ax2.set_xlabel('Market Regime')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'regime_performance.png'))
            plt.close()
            
        # 3. Win/Loss Distribution
        if trades:
            pnls = [t.pnl for t in trades]
            
            plt.figure(figsize=(12, 6))
            
            sns.histplot(pnls, kde=True, bins=30)
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            
            plt.title(f'Trade PnL Distribution - {report.model_name}')
            plt.xlabel('Trade PnL')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'pnl_distribution.png'))
            plt.close()
        
        # 4. Effectiveness Radar Chart
        eff_metrics = {
            'Overall': report.overall_effectiveness,
            'Trending': report.regime_effectiveness.get('trending', 0),
            'Ranging': report.regime_effectiveness.get('ranging', 0),
            'Volatile': report.regime_effectiveness.get('volatile', 0),
            'Drawdown\nControl': 100 * (1 - min(1, report.max_drawdown)),
            'Win Rate': report.win_rate * 100
        }
        
        categories = list(eff_metrics.keys())
        values = [eff_metrics[c] for c in categories]
        
        # Close any existing figures
        plt.close('all')
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Close the loop
        angles += angles[:1]  # Close the loop
        categories += categories[:1]  # Close the loop for labels
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2, label=report.model_name)
        ax.fill(angles, values, alpha=0.25)
        
        # Set category labels
        ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'])
        
        plt.title(f'Effectiveness Profile - {report.model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'effectiveness_radar.png'))
        plt.close()
        
    def compare_models(
        self,
        model_ids: List[str],
        baseline_ids: List[str] = None,
        comparison_id: str = None,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Compare performance across multiple models and baselines.
        
        Args:
            model_ids: List of model evaluation IDs to compare
            baseline_ids: List of baseline strategy IDs to compare against
            comparison_id: Custom ID for this comparison
            generate_report: Whether to generate a detailed report
            
        Returns:
            Dictionary containing comparison results
        """
        if comparison_id is None:
            comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Directory for comparison results
        comparison_dir = os.path.join(self.base_log_dir, "comparisons", comparison_id)
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Collect results from all models
        all_ids = model_ids + (baseline_ids or [])
        all_results = {}
        
        for eval_id in all_ids:
            if eval_id in self.evaluation_results:
                all_results[eval_id] = self.evaluation_results[eval_id]
            else:
                logger.warning(f"Results for {eval_id} not found in evaluation results")
        
        if not all_results:
            logger.error("No valid evaluation results found for comparison")
            return {}
            
        # Prepare comparison data
        comparison_data = {
            'comparison_id': comparison_id,
            'timestamp': datetime.now().isoformat(),
            'models': [],
            'metrics': {},
            'regime_comparison': {},
            'statistical_tests': {}
        }
        
        # Extract model performance data
        for eval_id, report in all_results.items():
            model_data = {
                'evaluation_id': eval_id,
                'model_name': report.model_name,
                'total_trades': report.total_trades,
                'total_pnl': report.total_pnl,
                'win_rate': report.win_rate,
                'sharpe_ratio': report.sharpe_ratio,
                'sortino_ratio': report.sortino_ratio,
                'max_drawdown': report.max_drawdown,
                'overall_effectiveness': report.overall_effectiveness,
                'regime_effectiveness': report.regime_effectiveness
            }
            
            comparison_data['models'].append(model_data)
        
        # Create composite metrics for comparison
        metrics = [
            'total_pnl', 'win_rate', 'sharpe_ratio', 
            'sortino_ratio', 'max_drawdown', 'overall_effectiveness'
        ]
        
        for metric in metrics:
            comparison_data['metrics'][metric] = {
                'values': [model[metric] for model in comparison_data['models']],
                'model_names': [model['model_name'] for model in comparison_data['models']]
            }
            
        # Compare performance by market regime
        regimes = list(set().union(*[
            set(report.regime_effectiveness.keys())
            for report in all_results.values()
        ]))
        
        for regime in regimes:
            comparison_data['regime_comparison'][regime] = {
                'effectiveness': {
                    'values': [report.regime_effectiveness.get(regime, 0) for report in all_results.values()],
                    'model_names': [report.model_name for report in all_results.values()]
                }
            }
            
        # Generate statistical significance tests
        # This is a simplified version - in a real implementation, you would perform
        # more rigorous statistical tests
        
        # Example: t-test for PnL differences
        if len(comparison_data['models']) > 1:
            for i, model_i in enumerate(comparison_data['models'][:-1]):
                for j, model_j in enumerate(comparison_data['models'][i+1:], i+1):
                    model_name_i = model_i['model_name']
                    model_name_j = model_j['model_name']
                    
                    comparison_key = f"{model_name_i}_vs_{model_name_j}"
                    
                    # This is just a placeholder - in a real implementation you would
                    # get the actual trade PnLs and perform a t-test
                    comparison_data['statistical_tests'][comparison_key] = {
                        'model_1': model_name_i,
                        'model_2': model_name_j,
                        'pnl_diff': model_i['total_pnl'] - model_j['total_pnl'],
                        'effectiveness_diff': model_i['overall_effectiveness'] - model_j['overall_effectiveness'],
                        'significant': abs(model_i['total_pnl'] - model_j['total_pnl']) > 100  # Placeholder
                    }
        
        # Store comparison results
        self.comparative_results[comparison_id] = comparison_data
        
        # Save comparison results
        comparison_path = os.path.join(comparison_dir, "comparison_results.json")
        with open(comparison_path, 'w') as f:
            # Convert any non-serializable objects to strings
            comparison_json = json.dumps(comparison_data, default=str, indent=2)
            f.write(comparison_json)
            
        # Generate comparison report if requested
        if generate_report and self.enable_visualization:
            self._generate_comparison_visualizations(comparison_dir, comparison_data)
            
        return comparison_data
        
    def _generate_comparison_visualizations(
        self,
        comparison_dir: str,
        comparison_data: Dict[str, Any]
    ) -> None:
        """
        Generate visualizations comparing multiple models.
        
        Args:
            comparison_dir: Directory to save visualizations
            comparison_data: Comparison data
        """
        # Create visualizations directory
        viz_dir = os.path.join(comparison_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Overall Metric Comparison
        key_metrics = ['total_pnl', 'sharpe_ratio', 'win_rate', 'overall_effectiveness']
        model_names = [model['model_name'] for model in comparison_data['models']]
        
        fig, axes = plt.subplots(len(key_metrics), 1, figsize=(12, 4 * len(key_metrics)))
        
        for i, metric in enumerate(key_metrics):
            metric_values = [model[metric] for model in comparison_data['models']]
            
            # Format specific metrics
            if metric == 'win_rate':
                metric_values = [v * 100 for v in metric_values]  # Convert to percentage
                ylabel = 'Win Rate (%)'
            elif metric == 'overall_effectiveness':
                ylabel = 'Effectiveness Score'
            elif metric == 'sharpe_ratio':
                ylabel = 'Sharpe Ratio'
            else:
                ylabel = ' '.join(word.capitalize() for word in metric.split('_'))
                
            sns.barplot(x=model_names, y=metric_values, ax=axes[i])
            axes[i].set_title(f'{ylabel} Comparison')
            axes[i].set_ylabel(ylabel)
            
            # Add text annotations
            for j, v in enumerate(metric_values):
                if metric in ['win_rate', 'overall_effectiveness']:
                    axes[i].text(j, v + 1, f'{v:.1f}', ha='center')
                else:
                    axes[i].text(j, v + (max(metric_values) * 0.02), f'{v:.2f}', ha='center')
                    
            # Set x-tick rotation for better readability
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'metric_comparison.png'))
        plt.close()
        
        # 2. Regime Effectiveness Comparison
        regime_data = comparison_data.get('regime_comparison', {})
        
        if regime_data:
            regimes = list(regime_data.keys())
            
            # Extract model names and their effectiveness scores per regime
            model_names = [model['model_name'] for model in comparison_data['models']]
            effectiveness_by_model = {name: [] for name in model_names}
            
            for regime in regimes:
                regime_effectiveness = regime_data[regime]['effectiveness']
                
                for i, name in enumerate(regime_effectiveness['model_names']):
                    if name in effectiveness_by_model:
                        effectiveness_by_model[name].append(regime_effectiveness['values'][i])
            
            # Prepare data for grouped bar chart
            df_data = []
            for model_name, effectiveness_values in effectiveness_by_model.items():
                for i, regime in enumerate(regimes):
                    if i < len(effectiveness_values):
                        df_data.append({
                            'Model': model_name,
                            'Regime': regime,
                            'Effectiveness': effectiveness_values[i]
                        })
            
            df = pd.DataFrame(df_data)
            
            # Create grouped bar chart
            plt.figure(figsize=(14, 8))
            chart = sns.barplot(x='Regime', y='Effectiveness', hue='Model', data=df)
            
            plt.title('Model Effectiveness by Market Regime')
            plt.ylabel('Effectiveness Score')
            plt.xlabel('Market Regime')
            
            # Enhance legend
            plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'regime_effectiveness.png'))
            plt.close()
        
        # 3. Radar Chart Comparing Overall Effectiveness Profiles
        # Extract categories and values for each model
        categories = ['Overall', 'Trending', 'Ranging', 'Volatile', 'Drawdown\nControl', 'Win Rate']
        model_values = []
        
        for model in comparison_data['models']:
            values = [
                model['overall_effectiveness'],
                model['regime_effectiveness'].get('trending', 0),
                model['regime_effectiveness'].get('ranging', 0),
                model['regime_effectiveness'].get('volatile', 0),
                100 * (1 - min(1, model['max_drawdown'])),  # Drawdown control (inverse of max_drawdown)
                model['win_rate'] * 100  # Convert to percentage
            ]
            model_values.append(values)
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # Plot each model
        for i, values in enumerate(model_values):
            # Close the loop
            values_loop = values + [values[0]]
            angles_loop = angles + [angles[0]]
            
            # Plot data
            ax.plot(angles_loop, values_loop, 'o-', linewidth=2, label=model_names[i])
            ax.fill(angles_loop, values_loop, alpha=0.1)
        
        # Set category labels
        ax.set_thetagrids(np.degrees(angles), categories)
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Effectiveness Profile Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'effectiveness_radar_comparison.png'))
        plt.close()


class RLEffectivenessAPI:
    """
    API interface for the RL Effectiveness Framework.
    Provides methods for evaluating models, comparing performance,
    and retrieving effectiveness metrics.
    """
    
    def __init__(self, framework: RLEffectivenessFramework = None):
        """
        Initialize the API with an existing framework or create a new one.
        
        Args:
            framework: Existing RL Effectiveness Framework instance (optional)
        """
        self.framework = framework or RLEffectivenessFramework()
        self.logger = logger
        
    def evaluate_model(self, **kwargs):
        """Proxy to framework's evaluate_model method."""
        return self.framework.evaluate_model(**kwargs)
    
    def compare_models(self, **kwargs):
        """Proxy to framework's compare_models method."""
        return self.framework.compare_models(**kwargs)
    
    def get_evaluation_results(self, evaluation_id: str) -> Dict[str, Any]:
        """
        Get the results of a specific evaluation.
        
        Args:
            evaluation_id: ID of the evaluation to retrieve
            
        Returns:
            Evaluation results or empty dict if not found
        """
        return self.framework.evaluation_results.get(evaluation_id, {})
    
    def get_comparison_results(self, comparison_id: str) -> Dict[str, Any]:
        """
        Get the results of a specific comparison.
        
        Args:
            comparison_id: ID of the comparison to retrieve
            
        Returns:
            Comparison results or empty dict if not found
        """
        return self.framework.comparative_results.get(comparison_id, {})
    
    def get_all_evaluations(self) -> List[str]:
        """
        Get a list of all evaluation IDs.
        
        Returns:
            List of evaluation IDs
        """
        return list(self.framework.evaluation_results.keys())
    
    def get_all_comparisons(self) -> List[str]:
        """
        Get a list of all comparison IDs.
        
        Returns:
            List of comparison IDs
        """
        return list(self.framework.comparative_results.keys())
    
    def get_effectiveness_summary(self) -> Dict[str, Any]:
        """
        Get a summary of effectiveness metrics across all evaluated models.
        
        Returns:
            Dictionary with effectiveness summary data
        """
        summary = {
            "total_models_evaluated": len(self.framework.evaluation_results),
            "total_comparisons": len(self.framework.comparative_results),
            "top_models": [],
            "effectiveness_by_regime": {}
        }
        
        if not self.framework.evaluation_results:
            return summary
            
        # Extract effectiveness data
        model_data = []
        for eval_id, report in self.framework.evaluation_results.items():
            model_data.append({
                "evaluation_id": eval_id,
                "model_name": report.model_name,
                "overall_effectiveness": report.overall_effectiveness,
                "regime_effectiveness": report.regime_effectiveness,
                "total_pnl": report.total_pnl,
                "win_rate": report.win_rate,
                "sharpe_ratio": report.sharpe_ratio
            })
            
        # Sort by overall effectiveness
        model_data.sort(key=lambda x: x["overall_effectiveness"], reverse=True)
        
        # Get top 5 models
        summary["top_models"] = model_data[:5]
        
        # Compute effectiveness by regime
        regimes = set()
        for model in model_data:
            regimes.update(model["regime_effectiveness"].keys())
            
        for regime in regimes:
            regime_models = []
            for model in model_data:
                if regime in model["regime_effectiveness"]:
                    regime_models.append({
                        "model_name": model["model_name"],
                        "effectiveness": model["regime_effectiveness"][regime]
                    })
            
            # Sort by regime effectiveness
            regime_models.sort(key=lambda x: x["effectiveness"], reverse=True)
            
            # Get top 3 models for this regime
            summary["effectiveness_by_regime"][regime] = regime_models[:3]
            
        return summary
