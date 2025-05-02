"""
Feedback Categorization System

This module implements the feedback categorization system that automatically classifies
trading feedback based on various criteria including statistical significance,
performance thresholds, and market conditions.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
from datetime import datetime, timedelta

from core_foundations.models.feedback import TradeFeedback, FeedbackCategory, FeedbackTag
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackCategorizer:
    """
    The FeedbackCategorizer classifies trading feedback into appropriate categories
    based on statistical analysis, performance thresholds, and other criteria.
    
    Key capabilities:
    - Automatically categorize feedback based on predefined rules
    - Validate statistical significance of feedback
    - Apply appropriate tags based on feedback characteristics
    - Group related feedback for aggregate analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the FeedbackCategorizer.
        
        Args:
            config: Configuration parameters for categorization thresholds
        """
        self.config = config or {}
        
        # Default configuration parameters
        self._set_default_config()
        
        # Historical data store for significance testing
        self.historical_data = {
            'strategy': {},   # Indexed by strategy_id
            'model': {},      # Indexed by model_id
            'instrument': {}, # Indexed by instrument
        }
        
        logger.info("FeedbackCategorizer initialized")
        
    def _set_default_config(self):
        """Set default configuration parameters if not provided."""
        defaults = {
            'profit_threshold': 0.0,           # Profit threshold for success
            'win_rate_threshold': 0.5,         # Win rate threshold for statistical significance
            'min_sample_size': 10,             # Minimum sample size for statistical testing
            'confidence_level': 0.95,          # Confidence level for statistical significance
            'high_impact_threshold': 0.03,     # % of account change for high impact
            'time_window': 24 * 60 * 60,       # Time window in seconds for trend detection
            'anomaly_z_score': 2.0,            # Z-score threshold for anomaly detection
            'parameter_impact_threshold': 0.1, # Threshold for significant parameter impact
        }
        
        # Apply defaults only for missing keys
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def categorize(self, feedback: TradeFeedback) -> TradeFeedback:
        """
        Categorize a feedback instance based on predefined rules and statistical analysis.
        
        Args:
            feedback: The feedback to categorize
            
        Returns:
            TradeFeedback: Categorized feedback with updated category and tags
        """
        # Store the feedback in historical data for future reference
        self._store_historical_data(feedback)
        
        # Apply basic categorization based on outcome metrics
        feedback = self._apply_basic_categorization(feedback)
        
        # Apply statistical significance validation
        feedback = self._apply_statistical_validation(feedback)
        
        # Apply trend detection and anomaly detection
        feedback = self._apply_trend_detection(feedback)
        feedback = self._apply_anomaly_detection(feedback)
        
        # Update feedback status
        feedback.status = 'categorized'
        
        logger.debug(f"Categorized feedback (ID: {feedback.feedback_id}) as {feedback.category}")
        return feedback
    
    def _store_historical_data(self, feedback: TradeFeedback):
        """
        Store feedback in historical data for statistical analysis.
        
        Args:
            feedback: Feedback to store
        """
        # Store by strategy_id
        if feedback.strategy_id:
            if feedback.strategy_id not in self.historical_data['strategy']:
                self.historical_data['strategy'][feedback.strategy_id] = []
            self.historical_data['strategy'][feedback.strategy_id].append(feedback)
            
        # Store by model_id
        if feedback.model_id:
            if feedback.model_id not in self.historical_data['model']:
                self.historical_data['model'][feedback.model_id] = []
            self.historical_data['model'][feedback.model_id].append(feedback)
            
        # Store by instrument
        if feedback.instrument:
            if feedback.instrument not in self.historical_data['instrument']:
                self.historical_data['instrument'][feedback.instrument] = []
            self.historical_data['instrument'][feedback.instrument].append(feedback)
            
        # Limit the size of historical data collections
        max_history = self.config.get('max_history_size', 1000)
        for category in self.historical_data.values():
            for key, items in category.items():
                if len(items) > max_history:
                    # Remove oldest items
                    category[key] = sorted(
                        items, 
                        key=lambda x: x.timestamp if hasattr(x, 'timestamp') else datetime.min
                    )[-max_history:]
    
    def _apply_basic_categorization(self, feedback: TradeFeedback) -> TradeFeedback:
        """
        Apply basic categorization rules based on outcome metrics.
        
        Args:
            feedback: Feedback to categorize
            
        Returns:
            TradeFeedback: Categorized feedback
        """
        metrics = feedback.outcome_metrics
        
        # For strategy execution feedback
        if feedback.source == 'strategy_execution':
            profit = metrics.get('profit_loss', metrics.get('profit', 0))
            
            if profit > self.config['profit_threshold']:
                feedback.category = FeedbackCategory.SUCCESS
            elif profit < self.config['profit_threshold']:
                feedback.category = FeedbackCategory.FAILURE
            else:
                feedback.category = FeedbackCategory.NEUTRAL
                
            # Check for high impact
            if 'account_balance' in metrics and 'previous_balance' in metrics:
                pct_change = abs((metrics['account_balance'] - metrics['previous_balance']) / 
                               metrics['previous_balance'])
                if pct_change > self.config['high_impact_threshold']:
                    feedback.tags.append(FeedbackTag.HIGH_IMPACT)
                    
        # For model prediction feedback
        elif feedback.source == 'model_prediction':
            error = metrics.get('error', metrics.get('prediction_error', 0))
            error_threshold = metrics.get('error_threshold', self.config.get('prediction_error_threshold', 0.01))
            
            if abs(error) < error_threshold:
                feedback.category = FeedbackCategory.SUCCESS
            else:
                feedback.category = FeedbackCategory.FAILURE
                
            # Check for high impact predictions
            if abs(error) > error_threshold * 3:  # 3x the error threshold
                feedback.tags.append(FeedbackTag.REQUIRES_ATTENTION)
                
        # For risk management feedback
        elif feedback.source == 'risk_management':
            risk_breach = metrics.get('risk_breach', False)
            
            if risk_breach:
                feedback.category = FeedbackCategory.WARNING
                feedback.tags.append(FeedbackTag.REQUIRES_ATTENTION)
            else:
                feedback.category = FeedbackCategory.INFORMATION
        
        return feedback
    
    def _apply_statistical_validation(self, feedback: TradeFeedback) -> TradeFeedback:
        """
        Apply statistical significance validation to the feedback.
        
        Args:
            feedback: Feedback to validate
            
        Returns:
            TradeFeedback: Validated feedback
        """
        # Skip if not enough data for statistical validation
        if not feedback.strategy_id and not feedback.model_id:
            return feedback
        
        # Get relevant historical data
        historical_data = []
        if feedback.strategy_id and feedback.strategy_id in self.historical_data['strategy']:
            historical_data = self.historical_data['strategy'][feedback.strategy_id]
        elif feedback.model_id and feedback.model_id in self.historical_data['model']:
            historical_data = self.historical_data['model'][feedback.model_id]
        
        # Skip if not enough samples
        if len(historical_data) < self.config['min_sample_size']:
            return feedback
            
        # Perform statistical validation based on feedback type
        if feedback.source == 'strategy_execution':
            # Calculate win rate
            wins = sum(1 for f in historical_data 
                      if f.outcome_metrics.get('profit_loss', f.outcome_metrics.get('profit', 0)) > 0)
            win_rate = wins / len(historical_data)
            
            # Binomial test for statistical significance
            if len(historical_data) >= self.config['min_sample_size']:
                # Using normal approximation to binomial for simplicity
                p_null = 0.5  # null hypothesis: win rate = 0.5 (random)
                z_score = (win_rate - p_null) / np.sqrt(p_null * (1 - p_null) / len(historical_data))
                p_value = 2 * (1 - self._standard_normal_cdf(abs(z_score)))
                
                # Add validated tag if statistically significant
                if p_value < (1 - self.config['confidence_level']):
                    feedback.tags.append(FeedbackTag.VALIDATED)
        
        # For model prediction, check if error is statistically significant
        elif feedback.source == 'model_prediction' and len(historical_data) >= self.config['min_sample_size']:
            errors = [f.outcome_metrics.get('error', 0) for f in historical_data 
                     if 'error' in f.outcome_metrics]
            
            if errors:
                mean_error = np.mean(errors)
                std_error = np.std(errors) if len(errors) > 1 else 1.0
                
                current_error = feedback.outcome_metrics.get('error', 0)
                z_score = abs(current_error - mean_error) / (std_error if std_error > 0 else 1.0)
                
                # Add validated tag if error is statistically significant
                if z_score > self._z_score_for_confidence(self.config['confidence_level']):
                    feedback.tags.append(FeedbackTag.VALIDATED)
        
        return feedback
    
    def _apply_trend_detection(self, feedback: TradeFeedback) -> TradeFeedback:
        """
        Apply trend detection to the feedback.
        
        Args:
            feedback: Feedback to analyze
            
        Returns:
            TradeFeedback: Analyzed feedback
        """
        # Skip if not enough data
        if not feedback.strategy_id and not feedback.model_id:
            return feedback
            
        # Get relevant historical data
        historical_data = []
        if feedback.strategy_id and feedback.strategy_id in self.historical_data['strategy']:
            historical_data = self.historical_data['strategy'][feedback.strategy_id]
        elif feedback.model_id and feedback.model_id in self.historical_data['model']:
            historical_data = self.historical_data['model'][feedback.model_id]
        
        # Skip if not enough samples
        if len(historical_data) < self.config['min_sample_size']:
            return feedback
            
        # Define time window for recent data
        now = datetime.utcnow()
        time_threshold = now - timedelta(seconds=self.config['time_window'])
        
        # Filter recent data
        recent_data = [f for f in historical_data 
                       if f.timestamp > time_threshold]
        
        # Skip if not enough recent data
        if len(recent_data) < self.config['min_sample_size'] / 2:
            return feedback
        
        # For strategy execution, detect trend in profitability
        if feedback.source == 'strategy_execution':
            # Calculate recent win rate vs overall win rate
            recent_wins = sum(1 for f in recent_data 
                           if f.outcome_metrics.get('profit_loss', f.outcome_metrics.get('profit', 0)) > 0)
            recent_win_rate = recent_wins / len(recent_data) if recent_data else 0
            
            overall_wins = sum(1 for f in historical_data 
                            if f.outcome_metrics.get('profit_loss', f.outcome_metrics.get('profit', 0)) > 0)
            overall_win_rate = overall_wins / len(historical_data) if historical_data else 0
            
            # Detect significant trend
            if abs(recent_win_rate - overall_win_rate) > 0.15:  # 15% difference
                feedback.tags.append(FeedbackTag.TRENDING)
        
        # For model prediction, detect trend in error rate
        elif feedback.source == 'model_prediction':
            recent_errors = [abs(f.outcome_metrics.get('error', 0)) for f in recent_data 
                           if 'error' in f.outcome_metrics]
            overall_errors = [abs(f.outcome_metrics.get('error', 0)) for f in historical_data 
                            if 'error' in f.outcome_metrics]
            
            if recent_errors and overall_errors:
                recent_mean_error = np.mean(recent_errors)
                overall_mean_error = np.mean(overall_errors)
                
                # Detect significant trend in error rate
                if recent_mean_error > overall_mean_error * 1.5:  # 50% increase
                    feedback.tags.append(FeedbackTag.TRENDING)
                    feedback.tags.append(FeedbackTag.REQUIRES_ATTENTION)
        
        return feedback
    
    def _apply_anomaly_detection(self, feedback: TradeFeedback) -> TradeFeedback:
        """
        Apply anomaly detection to the feedback.
        
        Args:
            feedback: Feedback to analyze
            
        Returns:
            TradeFeedback: Analyzed feedback
        """
        # Skip if not enough data
        if not feedback.strategy_id and not feedback.model_id and not feedback.instrument:
            return feedback
            
        # Determine which historical dataset to use
        key_type = 'strategy' if feedback.strategy_id else 'model' if feedback.model_id else 'instrument'
        key_value = feedback.strategy_id or feedback.model_id or feedback.instrument
        
        if key_value not in self.historical_data[key_type]:
            return feedback
            
        historical_data = self.historical_data[key_type][key_value]
        
        # Skip if not enough samples
        if len(historical_data) < self.config['min_sample_size']:
            return feedback
        
        # For strategy execution, detect anomalies in profit/loss
        if feedback.source == 'strategy_execution':
            profits = [f.outcome_metrics.get('profit_loss', f.outcome_metrics.get('profit', 0)) 
                      for f in historical_data]
            
            if profits:
                mean_profit = np.mean(profits)
                std_profit = np.std(profits) if len(profits) > 1 else 1.0
                
                current_profit = feedback.outcome_metrics.get('profit_loss', 
                                                          feedback.outcome_metrics.get('profit', 0))
                
                # Calculate z-score
                z_score = abs(current_profit - mean_profit) / (std_profit if std_profit > 0 else 1.0)
                
                # Mark as anomaly if z-score exceeds threshold
                if z_score > self.config['anomaly_z_score']:
                    feedback.tags.append(FeedbackTag.ANOMALY)
                    
                    # Also mark as high impact if exceptionally large
                    if z_score > self.config['anomaly_z_score'] * 2:
                        feedback.tags.append(FeedbackTag.HIGH_IMPACT)
        
        # For model prediction, detect anomalies in prediction error
        elif feedback.source == 'model_prediction':
            errors = [abs(f.outcome_metrics.get('error', 0)) for f in historical_data 
                     if 'error' in f.outcome_metrics]
            
            if errors:
                mean_error = np.mean(errors)
                std_error = np.std(errors) if len(errors) > 1 else 1.0
                
                current_error = abs(feedback.outcome_metrics.get('error', 0))
                
                # Calculate z-score
                z_score = (current_error - mean_error) / (std_error if std_error > 0 else 1.0)
                
                # Mark as anomaly if z-score exceeds threshold
                if z_score > self.config['anomaly_z_score']:
                    feedback.tags.append(FeedbackTag.ANOMALY)
        
        return feedback
    
    def _standard_normal_cdf(self, x):
        """
        Compute standard normal cumulative distribution function.
        
        Args:
            x: Value to compute CDF for
            
        Returns:
            float: CDF value
        """
        return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))
    
    def _z_score_for_confidence(self, confidence_level):
        """
        Convert a confidence level to a z-score.
        
        Args:
            confidence_level: The confidence level (e.g., 0.95)
            
        Returns:
            float: Corresponding z-score
        """
        # Common z-scores for confidence levels
        if confidence_level >= 0.99:
            return 2.576
        elif confidence_level >= 0.975:
            return 2.326
        elif confidence_level >= 0.95:
            return 1.96
        elif confidence_level >= 0.90:
            return 1.645
        else:
            # Approximate inverse of standard normal CDF
            return np.sqrt(2) * np.math.erfinv(2 * confidence_level - 1)
    
    def get_historical_statistics(self, 
                                 strategy_id: Optional[str] = None,
                                 model_id: Optional[str] = None,
                                 instrument: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistical summaries of historical feedback.
        
        Args:
            strategy_id: Optional filter by strategy ID
            model_id: Optional filter by model ID
            instrument: Optional filter by instrument
            
        Returns:
            Dict[str, Any]: Statistical summary
        """
        result = {
            'sample_count': 0,
            'success_rate': 0.0,
            'avg_profit': 0.0,
            'win_loss_ratio': 0.0,
            'statistically_significant': False
        }
        
        # Determine which data to use
        historical_data = []
        if strategy_id and strategy_id in self.historical_data['strategy']:
            historical_data = self.historical_data['strategy'][strategy_id]
        elif model_id and model_id in self.historical_data['model']:
            historical_data = self.historical_data['model'][model_id]
        elif instrument and instrument in self.historical_data['instrument']:
            historical_data = self.historical_data['instrument'][instrument]
        
        if not historical_data:
            return result
            
        # Calculate basic statistics
        result['sample_count'] = len(historical_data)
        
        # For strategy execution
        strategy_data = [f for f in historical_data if f.source == 'strategy_execution']
        if strategy_data:
            wins = sum(1 for f in strategy_data 
                      if f.outcome_metrics.get('profit_loss', f.outcome_metrics.get('profit', 0)) > 0)
            result['success_rate'] = wins / len(strategy_data)
            
            profits = [f.outcome_metrics.get('profit_loss', f.outcome_metrics.get('profit', 0)) 
                      for f in strategy_data]
            result['avg_profit'] = sum(profits) / len(profits) if profits else 0
            
            win_amount = sum(max(0, p) for p in profits)
            loss_amount = sum(abs(min(0, p)) for p in profits)
            result['win_loss_ratio'] = win_amount / loss_amount if loss_amount > 0 else float('inf')
            
            # Check statistical significance
            if len(strategy_data) >= self.config['min_sample_size']:
                p_null = 0.5  # null hypothesis
                z_score = (result['success_rate'] - p_null) / np.sqrt(p_null * (1 - p_null) / len(strategy_data))
                p_value = 2 * (1 - self._standard_normal_cdf(abs(z_score)))
                result['statistically_significant'] = p_value < (1 - self.config['confidence_level'])
        
        # For model predictions
        model_data = [f for f in historical_data if f.source == 'model_prediction']
        if model_data:
            result['model_stats'] = {
                'sample_count': len(model_data),
                'avg_error': 0.0,
                'error_std': 0.0
            }
            
            errors = [f.outcome_metrics.get('error', 0) for f in model_data 
                     if 'error' in f.outcome_metrics]
            
            if errors:
                result['model_stats']['avg_error'] = sum(errors) / len(errors)
                
                if len(errors) > 1:
                    variance = sum((e - result['model_stats']['avg_error'])**2 for e in errors) / (len(errors) - 1)
                    result['model_stats']['error_std'] = np.sqrt(variance)
        
        return result
