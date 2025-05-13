"""
Strategy Enhancement Dashboard

This module provides visualization tools for monitoring the performance
of the strategy enhancement services.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os

from analysis_engine.services.timeframe_optimization_service import TimeframeOptimizationService
from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceAnalyzer
from analysis_engine.analysis.sequence_pattern_recognizer import SequencePatternRecognizer
from analysis_engine.services.regime_transition_predictor import RegimeTransitionPredictor


class StrategyEnhancementDashboard:
    """
    Dashboard for visualizing strategy enhancement performance.
    
    This dashboard provides visualizations for:
    - Timeframe optimization performance
    - Currency strength analysis
    - Related pairs confluence
    - Sequence pattern detection
    - Regime transition prediction
    """
    
    def __init__(
        self,
        timeframe_optimizer: Optional[TimeframeOptimizationService] = None,
        currency_strength_analyzer: Optional[CurrencyStrengthAnalyzer] = None,
        related_pairs_detector: Optional[RelatedPairsConfluenceAnalyzer] = None,
        pattern_recognizer: Optional[SequencePatternRecognizer] = None,
        regime_transition_predictor: Optional[RegimeTransitionPredictor] = None,
        output_dir: str = "dashboard_output"
    ):
        """
        Initialize the dashboard.
        
        Args:
            timeframe_optimizer: TimeframeOptimizationService instance
            currency_strength_analyzer: CurrencyStrengthAnalyzer instance
            related_pairs_detector: RelatedPairsConfluenceAnalyzer instance
            pattern_recognizer: SequencePatternRecognizer instance
            regime_transition_predictor: RegimeTransitionPredictor instance
            output_dir: Directory to save dashboard outputs
        """
        self.timeframe_optimizer = timeframe_optimizer
        self.currency_strength_analyzer = currency_strength_analyzer
        self.related_pairs_detector = related_pairs_detector
        self.pattern_recognizer = pattern_recognizer
        self.regime_transition_predictor = regime_transition_predictor
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_dashboard(self, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Generate the complete dashboard.
        
        Args:
            save_to_file: Whether to save visualizations to files
            
        Returns:
            Dictionary with dashboard data
        """
        dashboard_data = {}
        
        # Generate timeframe optimization visualizations
        if self.timeframe_optimizer:
            timeframe_data = self.visualize_timeframe_optimization(save_to_file)
            dashboard_data["timeframe_optimization"] = timeframe_data
        
        # Generate currency strength visualizations
        if self.currency_strength_analyzer:
            currency_data = self.visualize_currency_strength(save_to_file)
            dashboard_data["currency_strength"] = currency_data
        
        # Generate pattern recognition visualizations
        if self.pattern_recognizer:
            pattern_data = self.visualize_pattern_recognition(save_to_file)
            dashboard_data["pattern_recognition"] = pattern_data
        
        # Generate regime transition visualizations
        if self.regime_transition_predictor:
            regime_data = self.visualize_regime_transitions(save_to_file)
            dashboard_data["regime_transitions"] = regime_data
        
        # Save dashboard data to file
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"{self.output_dir}/dashboard_data_{timestamp}.json", "w") as f:
                json.dump(dashboard_data, f, indent=2, default=str)
        
        return dashboard_data
    
    def visualize_timeframe_optimization(self, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Visualize timeframe optimization performance.
        
        Args:
            save_to_file: Whether to save visualizations to files
            
        Returns:
            Dictionary with visualization data
        """
        if not self.timeframe_optimizer:
            return {"error": "No timeframe optimizer provided"}
        
        # Get performance stats
        stats = self.timeframe_optimizer.get_performance_stats()
        
        # Get timeframe weights
        weights = self.timeframe_optimizer.get_timeframe_weights()
        
        # Create visualization data
        data = {
            "timeframe_weights": weights,
            "performance_stats": stats
        }
        
        # Create visualizations if requested
        if save_to_file:
            # Plot timeframe weights
            plt.figure(figsize=(10, 6))
            plt.bar(weights.keys(), weights.values())
            plt.title("Timeframe Optimization Weights")
            plt.xlabel("Timeframe")
            plt.ylabel("Weight")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/timeframe_weights.png")
            plt.close()
            
            # Plot win rates by timeframe
            win_rates = {tf: stats[tf]["win_rate"] for tf in stats}
            plt.figure(figsize=(10, 6))
            plt.bar(win_rates.keys(), win_rates.values())
            plt.title("Win Rates by Timeframe")
            plt.xlabel("Timeframe")
            plt.ylabel("Win Rate")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/timeframe_win_rates.png")
            plt.close()
        
        return data
    
    def visualize_currency_strength(self, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Visualize currency strength analysis.
        
        Args:
            save_to_file: Whether to save visualizations to files
            
        Returns:
            Dictionary with visualization data
        """
        if not self.currency_strength_analyzer:
            return {"error": "No currency strength analyzer provided"}
        
        # Get strength history
        strength_history = self.currency_strength_analyzer.strength_history
        
        # Create visualization data
        data = {
            "strength_history": strength_history
        }
        
        # Create visualizations if requested
        if save_to_file and strength_history:
            # Plot currency strength over time
            plt.figure(figsize=(12, 8))
            
            for currency, history in strength_history.items():
                if not history:
                    continue
                    
                dates = [point["timestamp"] for point in history]
                strengths = [point["strength"] for point in history]
                
                plt.plot(dates, strengths, label=currency, marker="o", markersize=4)
            
            plt.title("Currency Strength Over Time")
            plt.xlabel("Date")
            plt.ylabel("Strength")
            plt.legend()
            plt.grid(linestyle="--", alpha=0.7)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/currency_strength.png")
            plt.close()
        
        return data
    
    def visualize_pattern_recognition(self, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Visualize pattern recognition results.
        
        Args:
            save_to_file: Whether to save visualizations to files
            
        Returns:
            Dictionary with visualization data
        """
        if not self.pattern_recognizer:
            return {"error": "No pattern recognizer provided"}
        
        # Get pattern history
        pattern_history = self.pattern_recognizer.pattern_history
        
        # Create visualization data
        data = {
            "pattern_history": pattern_history
        }
        
        # Create visualizations if requested
        if save_to_file and pattern_history:
            # Count patterns by type
            pattern_counts = {}
            for pattern_type, patterns in pattern_history.items():
                pattern_counts[pattern_type] = len(patterns)
            
            if pattern_counts:
                # Plot pattern counts by type
                plt.figure(figsize=(12, 8))
                plt.bar(pattern_counts.keys(), pattern_counts.values())
                plt.title("Pattern Counts by Type")
                plt.xlabel("Pattern Type")
                plt.ylabel("Count")
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/pattern_counts.png")
                plt.close()
        
        return data
    
    def visualize_regime_transitions(self, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Visualize regime transition predictions.
        
        Args:
            save_to_file: Whether to save visualizations to files
            
        Returns:
            Dictionary with visualization data
        """
        if not self.regime_transition_predictor:
            return {"error": "No regime transition predictor provided"}
        
        # Get transition matrix
        transition_matrix = self.regime_transition_predictor.get_transition_matrix()
        
        # Get most likely transitions
        likely_transitions = self.regime_transition_predictor.get_most_likely_transitions(min_probability=0.3)
        
        # Create visualization data
        data = {
            "transition_matrix": transition_matrix,
            "likely_transitions": likely_transitions
        }
        
        # Create visualizations if requested
        if save_to_file:
            # Plot transition matrix as heatmap
            regimes = list(transition_matrix.keys())
            matrix_values = np.zeros((len(regimes), len(regimes)))
            
            for i, from_regime in enumerate(regimes):
                for j, to_regime in enumerate(regimes):
                    if to_regime in transition_matrix.get(from_regime, {}):
                        matrix_values[i, j] = transition_matrix[from_regime][to_regime]
            
            plt.figure(figsize=(10, 8))
            plt.imshow(matrix_values, cmap="YlOrRd")
            plt.colorbar(label="Transition Probability")
            plt.title("Regime Transition Probability Matrix")
            plt.xlabel("To Regime")
            plt.ylabel("From Regime")
            plt.xticks(range(len(regimes)), regimes, rotation=45, ha="right")
            plt.yticks(range(len(regimes)), regimes)
            
            # Add text annotations
            for i in range(len(regimes)):
                for j in range(len(regimes)):
                    if matrix_values[i, j] > 0:
                        plt.text(j, i, f"{matrix_values[i, j]:.2f}", 
                                ha="center", va="center", 
                                color="white" if matrix_values[i, j] > 0.5 else "black")
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/regime_transition_matrix.png")
            plt.close()
            
            # Plot most likely transitions
            if likely_transitions:
                transitions = [f"{t['from_regime']} → {t['to_regime']}" for t in likely_transitions]
                probabilities = [t["probability"] for t in likely_transitions]
                
                plt.figure(figsize=(12, 8))
                plt.barh(transitions, probabilities)
                plt.title("Most Likely Regime Transitions")
                plt.xlabel("Probability")
                plt.ylabel("Transition")
                plt.grid(axis="x", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/likely_regime_transitions.png")
                plt.close()
        
        return data
    
    def generate_performance_report(self, strategy_name: str, performance_data: Dict[str, Any]) -> str:
        """
        Generate a performance report for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            performance_data: Performance data for the strategy
            
        Returns:
            Report as a string
        """
        report = f"# Performance Report for {strategy_name}\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add timeframe optimization section
        report += "## Timeframe Optimization\n\n"
        
        if self.timeframe_optimizer:
            weights = self.timeframe_optimizer.get_timeframe_weights()
            stats = self.timeframe_optimizer.get_performance_stats()
            
            report += "### Timeframe Weights\n\n"
            report += "| Timeframe | Weight |\n"
            report += "|-----------|--------|\n"
            
            for tf, weight in weights.items():
                report += f"| {tf} | {weight:.4f} |\n"
            
            report += "\n### Timeframe Performance\n\n"
            report += "| Timeframe | Win Rate | Avg Win (pips) | Avg Loss (pips) | Profit Factor |\n"
            report += "|-----------|----------|----------------|-----------------|---------------|\n"
            
            for tf, tf_stats in stats.items():
                win_rate = tf_stats.get("win_rate", 0)
                avg_win = tf_stats.get("avg_win_pips", 0)
                avg_loss = tf_stats.get("avg_loss_pips", 0)
                profit_factor = tf_stats.get("profit_factor", 0)
                
                report += f"| {tf} | {win_rate:.2f} | {avg_win:.2f} | {avg_loss:.2f} | {profit_factor:.2f} |\n"
        else:
            report += "No timeframe optimization data available.\n\n"
        
        # Add currency strength section
        report += "\n## Currency Strength Analysis\n\n"
        
        if self.currency_strength_analyzer and self.currency_strength_analyzer.strength_history:
            # Get latest strength values
            latest_strength = {}
            for currency, history in self.currency_strength_analyzer.strength_history.items():
                if history:
                    latest_strength[currency] = history[-1]["strength"]
            
            if latest_strength:
                report += "### Latest Currency Strength\n\n"
                report += "| Currency | Strength |\n"
                report += "|----------|----------|\n"
                
                # Sort by strength (descending)
                sorted_currencies = sorted(latest_strength.items(), key=lambda x: x[1], reverse=True)
                
                for currency, strength in sorted_currencies:
                    report += f"| {currency} | {strength:.4f} |\n"
            
            # Add strongest and weakest currencies
            strongest = self.currency_strength_analyzer.get_strongest_currencies(count=3)
            weakest = self.currency_strength_analyzer.get_weakest_currencies(count=3)
            
            report += "\n### Strongest Currencies\n\n"
            for currency, strength in strongest:
                report += f"- {currency}: {strength:.4f}\n"
            
            report += "\n### Weakest Currencies\n\n"
            for currency, strength in weakest:
                report += f"- {currency}: {strength:.4f}\n"
        else:
            report += "No currency strength data available.\n\n"
        
        # Add pattern recognition section
        report += "\n## Pattern Recognition\n\n"
        
        if self.pattern_recognizer and self.pattern_recognizer.pattern_history:
            # Count patterns by type
            pattern_counts = {}
            for pattern_type, patterns in self.pattern_recognizer.pattern_history.items():
                pattern_counts[pattern_type] = len(patterns)
            
            if pattern_counts:
                report += "### Pattern Counts by Type\n\n"
                report += "| Pattern Type | Count |\n"
                report += "|--------------|-------|\n"
                
                # Sort by count (descending)
                sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
                
                for pattern_type, count in sorted_patterns:
                    report += f"| {pattern_type} | {count} |\n"
            
            # Add recent patterns
            report += "\n### Recent Patterns\n\n"
            
            recent_patterns = []
            for pattern_type, patterns in self.pattern_recognizer.pattern_history.items():
                for pattern_entry in patterns[-3:]:  # Get last 3 patterns of each type
                    recent_patterns.append({
                        "type": pattern_type,
                        "timestamp": pattern_entry["timestamp"],
                        "pattern": pattern_entry["pattern"]
                    })
            
            # Sort by timestamp (descending)
            recent_patterns.sort(key=lambda x: x["timestamp"], reverse=True)
            
            for i, pattern in enumerate(recent_patterns[:5]):  # Show top 5 most recent
                report += f"**{i+1}. {pattern['type']}** (Detected: {pattern['timestamp'].strftime('%Y-%m-%d %H:%M')})\n"
                report += f"   - Confidence: {pattern['pattern'].get('confidence', 'N/A')}\n"
                report += f"   - Timeframes: {', '.join(pattern['pattern'].get('timeframes', []))}\n"
                report += f"   - Direction: {pattern['pattern'].get('direction', 'N/A')}\n\n"
        else:
            report += "No pattern recognition data available.\n\n"
        
        # Add regime transition section
        report += "\n## Regime Transitions\n\n"
        
        if self.regime_transition_predictor:
            # Get most likely transitions
            likely_transitions = self.regime_transition_predictor.get_most_likely_transitions(min_probability=0.3)
            
            if likely_transitions:
                report += "### Most Likely Regime Transitions\n\n"
                report += "| From Regime | To Regime | Probability | Level |\n"
                report += "|-------------|-----------|-------------|-------|\n"
                
                for transition in likely_transitions:
                    from_regime = transition["from_regime"]
                    to_regime = transition["to_regime"]
                    probability = transition["probability"]
                    level = transition["probability_level"]
                    
                    report += f"| {from_regime} | {to_regime} | {probability:.4f} | {level} |\n"
            
            # Add transition history for the strategy's symbols
            if "symbols" in performance_data:
                symbols = performance_data.get("symbols", [])
                
                for symbol in symbols:
                    history = self.regime_transition_predictor.get_transition_history(symbol)
                    
                    if history:
                        report += f"\n### Regime Transition History for {symbol}\n\n"
                        report += "| Timestamp | Current Regime | Predicted Next Regime | Probability |\n"
                        report += "|-----------|----------------|----------------------|-------------|\n"
                        
                        # Sort by timestamp (descending)
                        sorted_history = sorted(history, key=lambda x: x["timestamp"], reverse=True)
                        
                        for entry in sorted_history[:5]:  # Show last 5 entries
                            timestamp = entry["timestamp"].strftime("%Y-%m-%d %H:%M")
                            current_regime = entry["current_regime"]
                            prediction = entry["prediction"]
                            next_regime = prediction.get("most_likely_next_regime", "N/A")
                            probability = prediction.get("transition_probability", 0)
                            
                            report += f"| {timestamp} | {current_regime} | {next_regime} | {probability:.4f} |\n"
        else:
            report += "No regime transition data available.\n\n"
        
        # Add overall performance section
        report += "\n## Overall Performance\n\n"
        
        if "performance_metrics" in performance_data:
            metrics = performance_data["performance_metrics"]
            
            report += "### Key Metrics\n\n"
            report += "| Metric | Value |\n"
            report += "|--------|-------|\n"
            
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    report += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"
                else:
                    report += f"| {metric.replace('_', ' ').title()} | {value} |\n"
        else:
            report += "No overall performance data available.\n\n"
        
        return report
    
    def save_performance_report(self, strategy_name: str, performance_data: Dict[str, Any]) -> str:
        """
        Generate and save a performance report for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            performance_data: Performance data for the strategy
            
        Returns:
            Path to the saved report
        """
        report = self.generate_performance_report(strategy_name, performance_data)
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.output_dir}/{strategy_name}_report_{timestamp}.md"
        
        with open(report_path, "w") as f:
            f.write(report)
        
        return report_path
