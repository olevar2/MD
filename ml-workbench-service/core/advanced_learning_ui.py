"""
Advanced visualization tools for reinforcement learning training in the forex trading platform.

This module provides interactive visualization components and dashboards
for monitoring, analyzing, and understanding RL agent training performance.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)
COLOR_SCHEME = {'primary': '#1f77b4', 'secondary': '#ff7f0e', 'tertiary':
    '#2ca02c', 'quaternary': '#d62728', 'background': '#f5f5f5', 'grid':
    '#e5e5e5', 'text': '#333333', 'highlight': '#bcbd22', 'comparison':
    '#9467bd'}


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class RewardVisualization:
    """
    Visualization tools for RL agent rewards and returns.
    """

    @staticmethod
    def plot_episode_rewards(rewards: List[float], window_size: int=10,
        title: str='Episode Rewards', figsize: Tuple[int, int]=(10, 6),
        save_path: Optional[str]=None) ->None:
        """
        Plot episode rewards with moving average.
        
        Args:
            rewards: List of episode rewards
            window_size: Window size for moving average
            title: Plot title
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=figsize)
        plt.plot(rewards, alpha=0.5, color=COLOR_SCHEME['primary'], label=
            'Episode Rewards')
        if len(rewards) >= window_size:
            moving_avg = [np.mean(rewards[max(0, i - window_size):i + 1]) for
                i in range(len(rewards))]
            plt.plot(moving_avg, color=COLOR_SCHEME['secondary'], linewidth
                =2, label=f'{window_size}-Episode Moving Average')
        plt.grid(True, linestyle='--', alpha=0.7, color=COLOR_SCHEME['grid'])
        plt.title(title, fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.legend(loc='best')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Saved rewards plot to {save_path}')
        plt.show()

    @staticmethod
    def plot_reward_distribution(rewards: List[float], title: str=
        'Reward Distribution', figsize: Tuple[int, int]=(10, 6), bins: int=
        30, save_path: Optional[str]=None) ->None:
        """
        Plot distribution of rewards.
        
        Args:
            rewards: List of episode rewards
            title: Plot title
            figsize: Figure size (width, height)
            bins: Number of histogram bins
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=figsize)
        plt.hist(rewards, bins=bins, alpha=0.7, color=COLOR_SCHEME['primary'])
        plt.axvline(np.mean(rewards), color=COLOR_SCHEME['secondary'],
            linestyle='dashed', linewidth=2, label=
            f'Mean: {np.mean(rewards):.2f}')
        plt.axvline(np.median(rewards), color=COLOR_SCHEME['tertiary'],
            linestyle='dashed', linewidth=2, label=
            f'Median: {np.median(rewards):.2f}')
        plt.grid(True, linestyle='--', alpha=0.7, color=COLOR_SCHEME['grid'])
        plt.title(title, fontsize=14)
        plt.xlabel('Reward', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(loc='best')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Saved reward distribution to {save_path}')
        plt.show()

    @staticmethod
    @with_exception_handling
    def create_interactive_rewards_plot(rewards_data: Dict[str, List[float]
        ], window_sizes: List[int]=[10, 20, 50, 100]) ->None:
        """
        Create an interactive rewards plot with adjustable smoothing.
        
        Args:
            rewards_data: Dictionary of rewards by label
            window_sizes: List of window sizes for moving average
        """
        try:
            fig = go.Figure()
            for label, rewards in rewards_data.items():
                fig.add_trace(go.Scatter(x=list(range(len(rewards))), y=
                    rewards, name=f'{label} (Raw)', mode='lines', line=dict
                    (width=1, dash='dot'), opacity=0.3, visible=False))
            for label, rewards in rewards_data.items():
                window = window_sizes[0]
                if len(rewards) >= window:
                    moving_avg = [np.mean(rewards[max(0, i - window):i + 1]
                        ) for i in range(len(rewards))]
                    fig.add_trace(go.Scatter(x=list(range(len(moving_avg))),
                        y=moving_avg, name=f'{label} (MA-{window})', mode=
                        'lines', line=dict(width=3)))
            buttons = []
            buttons.append(dict(method='update', label='Raw Data', args=[{
                'visible': [(True if i < len(rewards_data) else False) for
                i in range(len(rewards_data) * (len(window_sizes) + 1))]}]))
            for i, window in enumerate(window_sizes):
                visibility = []
                for j in range(len(rewards_data) * (len(window_sizes) + 1)):
                    reward_idx = j % (len(window_sizes) + 1)
                    if reward_idx == 0:
                        visibility.append(False)
                    else:
                        visibility.append(reward_idx - 1 == i)
                buttons.append(dict(method='update', label=f'MA-{window}',
                    args=[{'visible': visibility}]))
            fig.update_layout(title='Interactive Reward Plot', xaxis_title=
                'Episode', yaxis_title='Reward', updatemenus=[dict(active=1,
                buttons=buttons)], height=600, template='plotly_white')
            fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
            fig.show()
        except Exception as e:
            logger.error(f'Error creating interactive plot: {str(e)}')
            RewardVisualization.plot_episode_rewards(next(iter(rewards_data
                .values())), window_size=window_sizes[0], title=
                'Reward Plot (Fallback Static Version)')


class PolicyVisualization:
    """
    Visualization tools for RL agent policy analysis.
    """

    @staticmethod
    def visualize_action_distribution(actions: List[int], action_labels:
        List[str], title: str='Action Distribution', figsize: Tuple[int,
        int]=(10, 6), save_path: Optional[str]=None) ->None:
        """
        Visualize the distribution of actions taken by the agent.
        
        Args:
            actions: List of action indices
            action_labels: List of action labels
            title: Plot title
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        unique_actions = list(range(len(action_labels)))
        action_counts = [actions.count(action) for action in unique_actions]
        total_actions = len(actions)
        action_percentages = [(count / total_actions * 100) for count in
            action_counts]
        plt.figure(figsize=figsize)
        bars = plt.bar(action_labels, action_percentages, color=[
            COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'],
            COLOR_SCHEME['tertiary'], COLOR_SCHEME['quaternary']][:len(
            action_labels)])
        for bar, percentage in zip(bars, action_percentages):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1,
                f'{percentage:.1f}%', ha='center', va='bottom', rotation=0)
        plt.grid(True, linestyle='--', alpha=0.7, color=COLOR_SCHEME['grid'])
        plt.title(title, fontsize=14)
        plt.xlabel('Action', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Saved action distribution to {save_path}')
        plt.show()

    @staticmethod
    def visualize_action_over_time(actions: List[int], action_labels: List[
        str], window_size: int=100, title: str=
        'Action Distribution Over Time', figsize: Tuple[int, int]=(12, 6),
        save_path: Optional[str]=None) ->None:
        """
        Visualize how the agent's action choices change over time.
        
        Args:
            actions: List of action indices
            action_labels: List of action labels
            window_size: Window size for calculating moving distribution
            title: Plot title
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        time_windows = max(1, len(actions) // window_size)
        action_distributions = []
        for i in range(time_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(actions))
            window_actions = actions[start_idx:end_idx]
            counts = [0] * len(action_labels)
            for action in window_actions:
                counts[action] += 1
            percentages = [(count / len(window_actions) * 100) for count in
                counts]
            action_distributions.append(percentages)
        plt.figure(figsize=figsize)
        x = np.arange(time_windows) + 1
        bottom = np.zeros(time_windows)
        for i, label in enumerate(action_labels):
            values = [dist[i] for dist in action_distributions]
            plt.bar(x, values, bottom=bottom, label=label, alpha=0.7)
            bottom += values
        plt.grid(True, linestyle='--', alpha=0.7, color=COLOR_SCHEME['grid'])
        plt.title(title, fontsize=14)
        plt.xlabel(f'Time Window (Each {window_size} Steps)', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.legend(loc='best')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Saved action over time plot to {save_path}')
        plt.show()

    @staticmethod
    @with_exception_handling
    def plot_policy_heatmap(state_features: List[List[float]], actions:
        List[int], feature_names: List[str], action_labels: List[str],
        max_states: int=1000, title: str='Policy Heatmap', figsize: Tuple[
        int, int]=(15, 10), save_path: Optional[str]=None) ->None:
        """
        Create a heatmap visualizing the policy's action choices based on state features.
        
        Args:
            state_features: List of state feature vectors
            actions: List of corresponding action indices
            feature_names: Names of state features
            action_labels: Labels for actions
            max_states: Maximum number of states to include
            title: Plot title
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        if len(state_features) > max_states:
            indices = np.random.choice(len(state_features), max_states,
                replace=False)
            state_features = [state_features[i] for i in indices]
            actions = [actions[i] for i in indices]
        try:
            if len(state_features[0]) > 2:
                pca = PCA(n_components=2)
                reduced_states = pca.fit_transform(state_features)
                explained_variance = sum(pca.explained_variance_ratio_)
                if explained_variance < 0.5:
                    tsne = TSNE(n_components=2, random_state=42)
                    reduced_states = tsne.fit_transform(state_features)
                    method = 't-SNE'
                else:
                    method = 'PCA'
            else:
                reduced_states = state_features
                method = 'Original'
            plt.figure(figsize=figsize)
            scatter = plt.scatter(reduced_states[:, 0], reduced_states[:, 1
                ], c=actions, cmap='viridis', s=50, alpha=0.7)
            legend = plt.legend(handles=scatter.legend_elements()[0],
                labels=action_labels, title='Actions', loc='best')
            plt.grid(True, linestyle='--', alpha=0.7, color=COLOR_SCHEME[
                'grid'])
            plt.title(f'{title} ({method} Projection)', fontsize=14)
            plt.xlabel('State Dimension 1', fontsize=12)
            plt.ylabel('State Dimension 2', fontsize=12)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f'Saved policy heatmap to {save_path}')
            plt.show()
        except Exception as e:
            logger.error(f'Error creating policy heatmap: {str(e)}')
            logger.info('Falling back to basic action distribution plot')
            PolicyVisualization.visualize_action_distribution(actions,
                action_labels)


class MarketConditionVisualization:
    """
    Visualization tools for market conditions and their impact on agent performance.
    """

    @staticmethod
    def plot_performance_by_regime(market_regimes: List[str], returns: List
        [float], figsize: Tuple[int, int]=(12, 6), title: str=
        'Performance by Market Regime', save_path: Optional[str]=None) ->None:
        """
        Plot agent performance across different market regimes.
        
        Args:
            market_regimes: List of market regime labels
            returns: List of corresponding returns
            figsize: Figure size (width, height)
            title: Plot title
            save_path: Optional path to save the figure
        """
        regime_returns = {}
        for regime, ret in zip(market_regimes, returns):
            if regime not in regime_returns:
                regime_returns[regime] = []
            regime_returns[regime].append(ret)
        regimes = []
        mean_returns = []
        std_returns = []
        for regime, rets in regime_returns.items():
            regimes.append(regime)
            mean_returns.append(np.mean(rets))
            std_returns.append(np.std(rets))
        sorted_indices = np.argsort(mean_returns)
        regimes = [regimes[i] for i in sorted_indices]
        mean_returns = [mean_returns[i] for i in sorted_indices]
        std_returns = [std_returns[i] for i in sorted_indices]
        plt.figure(figsize=figsize)
        bars = plt.bar(regimes, mean_returns, yerr=std_returns, alpha=0.7,
            capsize=5)
        for i, bar in enumerate(bars):
            if mean_returns[i] >= 0:
                bar.set_color(COLOR_SCHEME['tertiary'])
            else:
                bar.set_color(COLOR_SCHEME['quaternary'])
        plt.grid(True, linestyle='--', alpha=0.7, color=COLOR_SCHEME['grid'])
        plt.title(title, fontsize=14)
        plt.xlabel('Market Regime', fontsize=12)
        plt.ylabel('Mean Return', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Saved regime performance plot to {save_path}')
        plt.show()

    @staticmethod
    @with_exception_handling
    def create_interactive_regime_performance(market_regimes: List[str],
        returns: List[float], metrics: Optional[Dict[str, List[float]]]=None
        ) ->None:
        """
        Create an interactive visualization of agent performance across market regimes.
        
        Args:
            market_regimes: List of market regime labels
            returns: List of corresponding returns
            metrics: Dictionary of additional metrics by name (lists should match length of regimes)
        """
        try:
            regime_data = {}
            for regime, ret in zip(market_regimes, returns):
                if regime not in regime_data:
                    regime_data[regime] = []
                regime_data[regime].append(ret)
            regimes = list(regime_data.keys())
            mean_returns = [np.mean(regime_data[r]) for r in regimes]
            std_returns = [np.std(regime_data[r]) for r in regimes]
            counts = [len(regime_data[r]) for r in regimes]
            sharpe_ratios = []
            for r in regimes:
                ret = regime_data[r]
                if len(ret) > 0 and np.std(ret) > 0:
                    sharpe = np.mean(ret) / np.std(ret)
                else:
                    sharpe = 0
                sharpe_ratios.append(sharpe)
            fig = make_subplots(rows=2, cols=2, subplot_titles=(
                'Mean Return by Regime', 'Return Distribution by Regime',
                'Sharpe Ratio by Regime', 'Sample Count by Regime'), specs=
                [[{'type': 'bar'}, {'type': 'box'}], [{'type': 'bar'}, {
                'type': 'bar'}]])
            fig.add_trace(go.Bar(x=regimes, y=mean_returns, error_y=dict(
                type='data', array=std_returns), name='Mean Return',
                marker_color=[('green' if ret >= 0 else 'red') for ret in
                mean_returns]), row=1, col=1)
            for i, regime in enumerate(regimes):
                fig.add_trace(go.Box(y=regime_data[regime], name=regime,
                    boxpoints='outliers', jitter=0.3, pointpos=-1.8), row=1,
                    col=2)
            fig.add_trace(go.Bar(x=regimes, y=sharpe_ratios, name=
                'Sharpe Ratio', marker_color=[('green' if sr >= 0 else
                'red') for sr in sharpe_ratios]), row=2, col=1)
            fig.add_trace(go.Bar(x=regimes, y=counts, name='Sample Count',
                marker_color='gray'), row=2, col=2)
            fig.update_layout(height=800, showlegend=False, title_text=
                'Agent Performance Across Market Regimes', template=
                'plotly_white')
            for i in range(1, 3):
                for j in range(1, 3):
                    fig.update_xaxes(tickangle=45, row=i, col=j)
            fig.show()
        except Exception as e:
            logger.error(
                f'Error creating interactive regime performance: {str(e)}')
            MarketConditionVisualization.plot_performance_by_regime(
                market_regimes, returns)


class TrainingProgressDashboard:
    """
    Interactive dashboard for monitoring RL training progress.
    """

    def __init__(self, training_data: Dict[str, Any], compare_agents:
        Optional[List[Dict[str, Any]]]=None, update_interval: int=100,
        save_dir: Optional[str]=None):
        """
        Initialize the training dashboard.
        
        Args:
            training_data: Dictionary with training data and metrics
            compare_agents: Optional list of data for other agents to compare
            update_interval: How frequently to update the dashboard
            save_dir: Optional directory to save dashboard outputs
        """
        self.training_data = training_data
        self.compare_agents = compare_agents or []
        self.update_interval = update_interval
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.setup_widgets()

    @with_exception_handling
    def setup_widgets(self) ->None:
        """Set up the interactive widgets for the dashboard."""
        try:
            self.tabs = widgets.Tab()
            reward_tab = widgets.VBox()
            performance_tab = widgets.VBox()
            policy_tab = widgets.VBox()
            comparison_tab = widgets.VBox()
            self.tabs.children = [reward_tab, performance_tab, policy_tab,
                comparison_tab]
            self.tabs.set_title(0, 'Rewards')
            self.tabs.set_title(1, 'Performance')
            self.tabs.set_title(2, 'Policy')
            self.tabs.set_title(3, 'Comparisons')
            self.plot_dropdown = widgets.Dropdown(options=[
                'Episode Rewards', 'Reward Distribution',
                'Action Distribution', 'Performance by Regime',
                'Interactive Rewards', 'Policy Heatmap'], value=
                'Episode Rewards', description='Plot:', style={
                'description_width': 'initial'})
            self.smoothing_slider = widgets.IntSlider(value=10, min=1, max=
                100, step=1, description='Smoothing:', disabled=False,
                continuous_update=False, orientation='horizontal', readout=
                True, readout_format='d')
            self.update_button = widgets.Button(description='Update Plots',
                button_style='info', tooltip='Click to update plots')
            self.update_button.on_click(self.update_plots)
            controls = widgets.HBox([self.plot_dropdown, self.
                smoothing_slider, self.update_button])
            reward_tab.children = [controls, widgets.Output()]
            performance_tab.children = [widgets.Output()]
            policy_tab.children = [widgets.Output()]
            comparison_tab.children = [widgets.Output()]
            self.outputs = {'rewards': reward_tab.children[1],
                'performance': performance_tab.children[0], 'policy':
                policy_tab.children[0], 'comparison': comparison_tab.
                children[0]}
        except Exception as e:
            logger.error(f'Error setting up dashboard widgets: {str(e)}')
            print(
                f'Error setting up interactive dashboard. Will fall back to static plots. Error: {str(e)}'
                )

    @with_exception_handling
    def update_plots(self, _=None) ->None:
        """Update the plots based on current selections."""
        try:
            plot_type = self.plot_dropdown.value
            smoothing = self.smoothing_slider.value
            for output in self.outputs.values():
                output.clear_output()
            if plot_type in ['Episode Rewards', 'Reward Distribution',
                'Interactive Rewards']:
                with self.outputs['rewards']:
                    self._plot_rewards(plot_type, smoothing)
            elif plot_type == 'Performance by Regime':
                with self.outputs['performance']:
                    self._plot_performance()
            elif plot_type in ['Action Distribution', 'Policy Heatmap']:
                with self.outputs['policy']:
                    self._plot_policy(plot_type)
            with self.outputs['comparison']:
                self._plot_comparisons()
        except Exception as e:
            logger.error(f'Error updating plots: {str(e)}')
            with self.outputs['rewards']:
                print(f'Error updating plots: {str(e)}')

    def _plot_rewards(self, plot_type: str, smoothing: int) ->None:
        """
        Plot reward-related visualizations.
        
        Args:
            plot_type: Type of reward plot
            smoothing: Smoothing window size
        """
        if 'rewards' not in self.training_data:
            print('No reward data available')
            return
        rewards = self.training_data['rewards']
        if plot_type == 'Episode Rewards':
            RewardVisualization.plot_episode_rewards(rewards, window_size=
                smoothing, title='Episode Rewards During Training',
                save_path=self.save_dir + '/rewards.png' if self.save_dir else
                None)
        elif plot_type == 'Reward Distribution':
            RewardVisualization.plot_reward_distribution(rewards, title=
                'Reward Distribution', save_path=self.save_dir +
                '/reward_dist.png' if self.save_dir else None)
        elif plot_type == 'Interactive Rewards':
            rewards_data = {'Current Agent': rewards}
            for i, agent_data in enumerate(self.compare_agents):
                if 'rewards' in agent_data:
                    rewards_data[f'Agent {i + 1}'] = agent_data['rewards']
            RewardVisualization.create_interactive_rewards_plot(rewards_data,
                window_sizes=[5, 10, 20, 50, 100])

    def _plot_performance(self) ->None:
        """Plot performance-related visualizations."""
        if ('market_regimes' not in self.training_data or 'returns' not in
            self.training_data):
            print('No market regime or return data available')
            return
        market_regimes = self.training_data['market_regimes']
        returns = self.training_data['returns']
        if len(market_regimes) != len(returns):
            print('Regime and return data length mismatch')
            return
        MarketConditionVisualization.create_interactive_regime_performance(
            market_regimes, returns)

    def _plot_policy(self, plot_type: str) ->None:
        """
        Plot policy-related visualizations.
        
        Args:
            plot_type: Type of policy plot
        """
        if ('actions' not in self.training_data or 'action_labels' not in
            self.training_data):
            print('No action data available')
            return
        actions = self.training_data['actions']
        action_labels = self.training_data['action_labels']
        if plot_type == 'Action Distribution':
            PolicyVisualization.visualize_action_distribution(actions,
                action_labels, title='Agent Action Distribution', save_path
                =self.save_dir + '/action_dist.png' if self.save_dir else None)
            PolicyVisualization.visualize_action_over_time(actions,
                action_labels, title='Action Distribution Over Time',
                save_path=self.save_dir + '/action_time.png' if self.
                save_dir else None)
        elif plot_type == 'Policy Heatmap':
            if ('state_features' not in self.training_data or 
                'feature_names' not in self.training_data):
                print('No state feature data available for policy heatmap')
                return
            state_features = self.training_data['state_features']
            feature_names = self.training_data['feature_names']
            PolicyVisualization.plot_policy_heatmap(state_features, actions,
                feature_names, action_labels, title='Policy Heatmap',
                save_path=self.save_dir + '/policy_heatmap.png' if self.
                save_dir else None)

    def _plot_comparisons(self) ->None:
        """Plot comparison visualizations between agents."""
        if not self.compare_agents:
            print('No comparison data available')
            return
        rewards_data = {}
        if 'rewards' in self.training_data:
            rewards_data['Current Agent'] = self.training_data['rewards']
        for i, agent_data in enumerate(self.compare_agents):
            if 'rewards' in agent_data:
                rewards_data[f'Agent {i + 1}'] = agent_data['rewards']
        if rewards_data:
            plt.figure(figsize=(12, 6))
            for label, rewards in rewards_data.items():
                min_length = min(100, len(rewards))
                window_size = 10
                if len(rewards) >= window_size:
                    moving_avg = [np.mean(rewards[max(0, i - window_size):i +
                        1]) for i in range(len(rewards))]
                    plt.plot(moving_avg[:min_length], label=label)
            plt.grid(True, linestyle='--', alpha=0.7, color=COLOR_SCHEME[
                'grid'])
            plt.title('Reward Comparison Between Agents', fontsize=14)
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Average Reward (10-episode MA)', fontsize=12)
            plt.legend(loc='best')
            if self.save_dir:
                plt.savefig(f'{self.save_dir}/agent_comparison.png', dpi=
                    300, bbox_inches='tight')
            plt.show()

    def display(self) ->None:
        """Display the dashboard."""
        display(self.tabs)
        self.update_plots()

    def add_data(self, new_data: Dict[str, Any]) ->None:
        """
        Add new training data to the dashboard.
        
        Args:
            new_data: New training data to add
        """
        for key, value in new_data.items():
            if key in self.training_data:
                if isinstance(self.training_data[key], list) and isinstance(
                    value, list):
                    self.training_data[key].extend(value)
                else:
                    self.training_data[key] = value
            else:
                self.training_data[key] = value
        if 'episode' in new_data and new_data['episode'
            ] % self.update_interval == 0:
            self.update_plots()


class ModelExplainabilityTool:
    """
    Tools for explaining and interpreting RL agent behavior.
    """

    @staticmethod
    @with_exception_handling
    def visualize_feature_importance(model, feature_names: List[str],
        method: str='perturbation', n_samples: int=1000, figsize: Tuple[int,
        int]=(12, 6), save_path: Optional[str]=None) ->Dict[str, float]:
        """
        Visualize the importance of input features to the model's decisions.
        
        Args:
            model: RL model with predict/forward method
            feature_names: Names of state features
            method: Method to compute feature importance
            n_samples: Number of samples to use
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
            
        Returns:
            Dictionary of feature importance values
        """
        np.random.seed(42)
        n_features = len(feature_names)
        samples = np.random.randn(n_samples, n_features)
        try:
            if hasattr(model, 'predict'):
                baseline_actions, _ = model.predict(samples, deterministic=True
                    )
            elif hasattr(model, 'forward'):
                baseline_actions = model.forward(torch.FloatTensor(samples)
                    ).argmax(dim=1).numpy()
            else:
                print("Model doesn't have predict or forward method")
                return {}
        except Exception as e:
            logger.error(f'Error getting baseline predictions: {str(e)}')
            return {}
        importance = {}
        if method == 'perturbation':
            for i, feature in enumerate(feature_names):
                perturbed = samples.copy()
                perturbed[:, i] = np.random.randn(n_samples)
                try:
                    if hasattr(model, 'predict'):
                        perturbed_actions, _ = model.predict(perturbed,
                            deterministic=True)
                    else:
                        perturbed_actions = model.forward(torch.FloatTensor
                            (perturbed)).argmax(dim=1).numpy()
                    pct_changed = np.mean(perturbed_actions != baseline_actions
                        ) * 100
                    importance[feature] = pct_changed
                except Exception as e:
                    logger.error(
                        f'Error computing importance for feature {feature}: {str(e)}'
                        )
                    importance[feature] = 0.0
        importance = {k: v for k, v in sorted(importance.items(), key=lambda
            item: item[1], reverse=True)}
        plt.figure(figsize=figsize)
        bars = plt.bar(importance.keys(), importance.values(), color=
            COLOR_SCHEME['primary'], alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom', rotation=0)
        plt.grid(True, linestyle='--', alpha=0.7, color=COLOR_SCHEME['grid'])
        plt.title('Feature Importance (% Actions Changed When Perturbed)',
            fontsize=14)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Importance (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Saved feature importance plot to {save_path}')
        plt.show()
        return importance

    @staticmethod
    @with_exception_handling
    def create_action_explanation_tool(model, feature_names: List[str],
        action_labels: List[str]) ->None:
        """
        Create an interactive tool to explain model actions based on input features.
        
        Args:
            model: RL model with predict method
            feature_names: Names of input features
            action_labels: Labels for output actions
        """
        try:
            sliders = []
            for i, feature in enumerate(feature_names):
                slider = widgets.FloatSlider(value=0.0, min=-3.0, max=3.0,
                    step=0.1, description=feature, disabled=False,
                    continuous_update=False, orientation='horizontal',
                    readout=True, readout_format='.1f', style={
                    'description_width': 'initial'})
                sliders.append(slider)
            output = widgets.Output()
            predict_button = widgets.Button(description='Predict Action',
                button_style='info', tooltip='Click to predict')

            @with_exception_handling
            def update_prediction(_):
    """
    Update prediction.
    
    Args:
        _: Description of _
    
    """

                output.clear_output()
                features = np.array([[slider.value for slider in sliders]])
                with output:
                    try:
                        if hasattr(model, 'predict'):
                            action, _ = model.predict(features,
                                deterministic=True)
                            action = action[0]
                        elif hasattr(model, 'forward'):
                            import torch
                            logits = model.forward(torch.FloatTensor(features))
                            probs = torch.softmax(logits, dim=1).detach(
                                ).numpy()[0]
                            action = np.argmax(probs)
                            plt.figure(figsize=(10, 5))
                            bars = plt.bar(action_labels, probs, color=
                                COLOR_SCHEME['primary'])
                            bars[action].set_color(COLOR_SCHEME['highlight'])
                            plt.grid(True, linestyle='--', alpha=0.7, color
                                =COLOR_SCHEME['grid'])
                            plt.title('Action Probabilities', fontsize=14)
                            plt.xlabel('Action', fontsize=12)
                            plt.ylabel('Probability', fontsize=12)
                            plt.ylim(0, 1)
                            for i, prob in enumerate(probs):
                                plt.text(i, prob + 0.02, f'{prob:.2f}', ha=
                                    'center')
                            plt.tight_layout()
                            plt.show()
                        print(f'Predicted Action: {action_labels[action]}')
                        print('\nFeature contributions:')
                        contributions = [(name, abs(value)) for name, value in
                            zip(feature_names, features[0])]
                        contributions.sort(key=lambda x: x[1], reverse=True)
                        for name, value in contributions[:5]:
                            direction = 'high' if value > 0 else 'low'
                            print(
                                f"- {name}: {value:.2f} ({'high' if value > 0 else 'low'})"
                                )
                    except Exception as e:
                        print(f'Error making prediction: {str(e)}')
            predict_button.on_click(update_prediction)
            random_button = widgets.Button(description='Random Features',
                button_style='warning', tooltip='Set random feature values')

            def set_random(_):
    """
    Set random.
    
    Args:
        _: Description of _
    
    """

                for slider in sliders:
                    slider.value = np.random.uniform(-2.0, 2.0)
            random_button.on_click(set_random)
            box_layout = widgets.Layout(display='flex', flex_flow='column',
                align_items='stretch', width='100%')
            sliders_box = widgets.VBox(sliders, layout=box_layout)
            buttons_box = widgets.HBox([predict_button, random_button])
            container = widgets.VBox([widgets.HTML(
                '<h3>Explore Agent Behavior</h3>'), widgets.HTML(
                "<p>Adjust the feature values and click 'Predict Action' to see what the agent would do.</p>"
                ), sliders_box, buttons_box, output])
            display(container)
        except Exception as e:
            logger.error(f'Error creating action explanation tool: {str(e)}')
            print(f'Error creating interactive tool: {str(e)}')


class LearningVisualizationUI:
    """
    Main UI component for comprehensive RL visualization.
    """

    def __init__(self, save_dir: Optional[str]=None):
        """
        Initialize the visualization UI.
        
        Args:
            save_dir: Optional directory to save outputs
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    @with_exception_handling
    def visualize_training_progress(self, training_logs: List[Dict[str, Any
        ]], agent_name: str='Agent', include_dashboard: bool=True) ->None:
        """
        Visualize training progress from logs.
        
        Args:
            training_logs: List of training log entries
            agent_name: Name of the agent
            include_dashboard: Whether to include interactive dashboard
        """
        episodes = [log.get('episode', i) for i, log in enumerate(
            training_logs)]
        rewards = [log.get('reward', 0.0) for log in training_logs]
        returns = [log.get('return', 0.0) for log in training_logs if 
            'return' in log]
        market_regimes = [log.get('market_regime', 'unknown') for log in
            training_logs if 'market_regime' in log]
        actions = [log.get('action', 0) for log in training_logs if 
            'action' in log]
        self._plot_learning_curve(episodes, rewards, agent_name)
        if returns and market_regimes and len(returns) == len(market_regimes):
            self._plot_regime_performance(market_regimes, returns, agent_name)
        if actions:
            action_labels = set(actions)
            self._plot_action_distribution(actions, [f'Action {a}' for a in
                action_labels], agent_name)
        if include_dashboard:
            try:
                dashboard_data = {'rewards': rewards, 'returns': returns if
                    returns else rewards, 'episodes': episodes,
                    'market_regimes': market_regimes if market_regimes else
                    ['unknown'] * len(rewards), 'actions': actions if
                    actions else [0] * len(rewards), 'action_labels': [
                    f'Action {a}' for a in set(actions)] if actions else [
                    'N/A']}
                if any('state_features' in log for log in training_logs):
                    state_features = [log.get('state_features', []) for log in
                        training_logs if 'state_features' in log]
                    feature_names = [f'Feature {i}' for i in range(len(
                        state_features[0]))] if state_features else []
                    dashboard_data['state_features'] = state_features
                    dashboard_data['feature_names'] = feature_names
                dashboard = TrainingProgressDashboard(training_data=
                    dashboard_data, save_dir=self.save_dir)
                dashboard.display()
            except Exception as e:
                logger.error(f'Error creating dashboard: {str(e)}')
                print(f'Could not create interactive dashboard: {str(e)}')

    def _plot_learning_curve(self, episodes: List[int], rewards: List[float
        ], agent_name: str) ->None:
        """Plot the learning curve."""
        RewardVisualization.plot_episode_rewards(rewards, window_size=min(
            20, max(1, len(rewards) // 20)), title=
            f'{agent_name} Learning Curve', save_path=
            f"{self.save_dir}/{agent_name.lower().replace(' ', '_')}_learning_curve.png"
             if self.save_dir else None)

    def _plot_regime_performance(self, regimes: List[str], returns: List[
        float], agent_name: str) ->None:
        """Plot performance by market regime."""
        MarketConditionVisualization.plot_performance_by_regime(regimes,
            returns, title=f'{agent_name} Performance by Market Regime',
            save_path=
            f"{self.save_dir}/{agent_name.lower().replace(' ', '_')}_regime_perf.png"
             if self.save_dir else None)

    def _plot_action_distribution(self, actions: List[int], action_labels:
        List[str], agent_name: str) ->None:
        """Plot action distribution."""
        PolicyVisualization.visualize_action_distribution(actions,
            action_labels, title=f'{agent_name} Action Distribution',
            save_path=
            f"{self.save_dir}/{agent_name.lower().replace(' ', '_')}_action_dist.png"
             if self.save_dir else None)

    @with_exception_handling
    def compare_agents(self, agents_data: List[Dict[str, Any]], metric: str
        ='rewards', window_size: int=10, figsize: Tuple[int, int]=(12, 6),
        include_interactive: bool=True) ->None:
        """
        Create comparison visualizations between different agents.
        
        Args:
            agents_data: List of dictionaries with agent data
            metric: Metric to compare
            window_size: Window size for smoothing
            figsize: Figure size (width, height)
            include_interactive: Whether to include interactive visualizations
        """
        if not agents_data or not all(metric in agent for agent in agents_data
            ):
            print(f"All agents must have '{metric}' data for comparison")
            return
        if include_interactive:
            try:
                fig = go.Figure()
                for agent in agents_data:
                    name = agent.get('name', 'Unnamed Agent')
                    data = agent.get(metric, [])
                    if len(data) >= window_size:
                        smoothed = [np.mean(data[max(0, i - window_size):i +
                            1]) for i in range(len(data))]
                    else:
                        smoothed = data
                    fig.add_trace(go.Scatter(y=smoothed, mode='lines', name
                        =name, hovertemplate=f"{name}: %{{'y:.2f}}"))
                fig.update_layout(title=
                    f'Agent Comparison ({metric.capitalize()})',
                    xaxis_title='Episode', yaxis_title=metric.capitalize(),
                    hovermode='x unified', template='plotly_white')
                fig.show()
            except Exception as e:
                logger.error(f'Error creating interactive comparison: {str(e)}'
                    )
                include_interactive = False
        if not include_interactive:
            plt.figure(figsize=figsize)
            for agent in agents_data:
                name = agent.get('name', 'Unnamed Agent')
                data = agent.get(metric, [])
                if len(data) >= window_size:
                    smoothed = [np.mean(data[max(0, i - window_size):i + 1]
                        ) for i in range(len(data))]
                else:
                    smoothed = data
                plt.plot(smoothed, label=name)
            plt.grid(True, linestyle='--', alpha=0.7, color=COLOR_SCHEME[
                'grid'])
            plt.title(f'Agent Comparison ({metric.capitalize()})', fontsize=14)
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel(metric.capitalize(), fontsize=12)
            plt.legend(loc='best')
            if self.save_dir:
                plt.savefig(f'{self.save_dir}/agent_comparison_{metric}.png',
                    dpi=300, bbox_inches='tight')
            plt.show()


if __name__ == '__main__':
    import numpy as np
    n_episodes = 1000
    rewards = np.random.normal(0, 1, n_episodes).cumsum() / 100
    for i in range(n_episodes):
        rewards[i] += i * 0.01
    actions = np.random.choice([0, 1, 2, 3], n_episodes)
    action_labels = ['Buy', 'Sell', 'Hold', 'Close']
    regimes = np.random.choice(['Trending Bullish', 'Trending Bearish',
        'Choppy', 'Ranging', 'Breakout'], n_episodes)
    logs = []
    for i in range(n_episodes):
        logs.append({'episode': i, 'reward': rewards[i], 'action': actions[
            i], 'market_regime': regimes[i]})
    vis_ui = LearningVisualizationUI(save_dir='./visualization_outputs')
    vis_ui.visualize_training_progress(logs, agent_name='Sample Agent')
