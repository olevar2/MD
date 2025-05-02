"""
ML Confirmation Filter

This module implements a machine learning-based confirmation filter for trading signals.
It integrates ML model predictions with strategy signals to improve entry and exit decisions.

Part of Phase 4 implementation to enhance the adaptive trading capabilities.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

from ml_integration_service.clients.ml_workbench_client import MLWorkbenchClient
from ml_integration_service.models.prediction_request import PredictionRequest
from ml_integration_service.models.filter_config import FilterConfig
from ml_integration_service.utils.performance_metrics import calculate_filter_effectiveness


class MLConfirmationFilter:
    """
    Machine learning-based confirmation filter that enhances signal quality
    by leveraging predictions from trained models to confirm or reject signals.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML confirmation filter with configuration.
        
        Args:
            config: Configuration parameters for the filter
        """
        self.logger = logging.getLogger(__name__)
        self.ml_client = MLWorkbenchClient()
        
        # Default configuration
        default_config = {
            "min_confirmation_threshold": 0.65,  # Minimum confidence from ML model to confirm
            "rejection_threshold": 0.35,         # Threshold below which signals are rejected
            "models": {
                "trend_direction": "trend_classifier_v2",  # Default model for trend direction
                "reversal_detection": "reversal_detector_v3", # Default model for reversals
                "volatility_forecast": "volatility_predictor_v1", # Default model for volatility
                "support_resistance": "sr_predictor_v2" # Default model for S/R prediction
            },
            "use_ensemble": True,                # Whether to use ensemble of models
            "ensemble_weights": {
                "trend_direction": 0.4,
                "reversal_detection": 0.3,
                "volatility_forecast": 0.1,
                "support_resistance": 0.2
            },
            "cache_predictions": True,           # Whether to cache recent predictions
            "cache_duration_minutes": 60,        # How long to cache predictions
            "adapting_thresholds": True,         # Whether to adjust thresholds based on performance
            "model_fallback": True,              # Whether to use fallback models if primary fails
            "performance_tracking": True,        # Track filter performance metrics
            "inference_timeout_seconds": 5       # Timeout for model inference requests
        }
        
        # Update with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # Initialize cache and performance tracking
        self.prediction_cache = {}
        self.performance_history = {
            "total_signals": 0,
            "confirmed_signals": 0,
            "rejected_signals": 0,
            "confirmed_correct": 0,
            "confirmed_incorrect": 0,
            "rejected_correct": 0,
            "rejected_incorrect": 0,
            "by_model": {}
        }
        
        self.logger.info("ML Confirmation Filter initialized")
        
    def filter_signal(
        self, 
        signal: Dict[str, Any], 
        features: Dict[str, Any], 
        price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Filter a trading signal using ML model predictions.
        
        Args:
            signal: Original trading signal
            features: Calculated features for ML models
            price_data: Recent price data for context
            
        Returns:
            Enhanced signal with ML confirmation data
        """
        try:
            self.performance_history["total_signals"] += 1
            
            # Create a copy of the signal to avoid modifying the original
            enhanced_signal = signal.copy()
            
            # Initialize ML confirmation data
            enhanced_signal["ml_confirmation"] = {
                "confirmed": False,
                "confidence": 0.0,
                "model_predictions": {},
                "ensemble_score": 0.0,
                "confirmation_time": datetime.now().isoformat()
            }
            
            # Get predictions from configured models
            predictions = self._get_model_predictions(signal, features, price_data)
            if not predictions:
                self.logger.warning("No model predictions available for confirmation")
                return enhanced_signal
                
            # Store individual model predictions
            enhanced_signal["ml_confirmation"]["model_predictions"] = predictions
            
            # Calculate ensemble score if multiple models used
            if self.config["use_ensemble"] and len(predictions) > 1:
                ensemble_score = self._calculate_ensemble_score(predictions, signal["direction"])
                enhanced_signal["ml_confirmation"]["ensemble_score"] = ensemble_score
                confidence = ensemble_score
            else:
                # Use the main model's prediction as confidence
                main_model = self._get_primary_model_for_signal(signal)
                confidence = predictions.get(main_model, {}).get("confidence", 0.0)
            
            enhanced_signal["ml_confirmation"]["confidence"] = confidence
            
            # Determine if signal is confirmed based on confidence
            if confidence >= self.config["min_confirmation_threshold"]:
                enhanced_signal["ml_confirmation"]["confirmed"] = True
                enhanced_signal["confidence"] = min(1.0, signal.get("confidence", 0.5) * (1 + confidence/2))
                self.performance_history["confirmed_signals"] += 1
            elif confidence <= self.config["rejection_threshold"]:
                enhanced_signal["ml_confirmation"]["confirmed"] = False
                enhanced_signal["confidence"] = max(0.1, signal.get("confidence", 0.5) * confidence)
                enhanced_signal["rejected"] = True
                enhanced_signal["rejection_reason"] = "ml_confirmation_failed"
                self.performance_history["rejected_signals"] += 1
            else:
                # In the gray area, slightly adjust confidence
                enhanced_signal["confidence"] = signal.get("confidence", 0.5) * (0.8 + confidence/2)
            
            # Add ML-specific adjustments to signal
            if enhanced_signal["ml_confirmation"]["confirmed"]:
                self._enhance_signal_with_ml_insights(enhanced_signal, predictions)
                
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Error filtering signal with ML confirmation: {str(e)}", exc_info=True)
            return signal  # Return original signal if filtering fails
    
    def _get_model_predictions(
        self, 
        signal: Dict[str, Any], 
        features: Dict[str, Any],
        price_data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Get predictions from all relevant models for this signal."""
        predictions = {}
        
        # Determine which models to use based on signal type
        models_to_use = self._get_relevant_models_for_signal(signal)
        
        for model_type, model_name in models_to_use.items():
            # Check cache first if enabled
            cache_key = f"{model_name}_{signal['symbol']}_{signal['signal_type']}"
            if self.config["cache_predictions"] and cache_key in self.prediction_cache:
                cached_pred = self.prediction_cache[cache_key]
                # Check if cache is still valid
                cache_age = (datetime.now() - cached_pred["timestamp"]).total_seconds() / 60
                if cache_age <= self.config["cache_duration_minutes"]:
                    predictions[model_type] = cached_pred["prediction"]
                    continue
            
            # Prepare request for prediction
            request = PredictionRequest(
                model_name=model_name,
                symbol=signal["symbol"],
                features=features,
                signal_type=signal["signal_type"],
                direction=signal["direction"],
                timestamp=datetime.now().isoformat()
            )
            
            # Get prediction from ML workbench
            try:
                prediction = self.ml_client.get_prediction(request, timeout=self.config["inference_timeout_seconds"])
                
                if prediction and "error" not in prediction:
                    predictions[model_type] = prediction
                    
                    # Cache the prediction if caching is enabled
                    if self.config["cache_predictions"]:
                        self.prediction_cache[cache_key] = {
                            "prediction": prediction,
                            "timestamp": datetime.now()
                        }
                elif self.config["model_fallback"]:
                    # Try fallback model if available
                    fallback_model = self._get_fallback_model(model_type)
                    if fallback_model and fallback_model != model_name:
                        self.logger.info(f"Using fallback model {fallback_model} for {model_type}")
                        request.model_name = fallback_model
                        fallback_prediction = self.ml_client.get_prediction(request, timeout=self.config["inference_timeout_seconds"])
                        if fallback_prediction and "error" not in fallback_prediction:
                            fallback_prediction["is_fallback"] = True
                            predictions[model_type] = fallback_prediction
            except Exception as e:
                self.logger.error(f"Error getting prediction from {model_name}: {str(e)}")
        
        return predictions
    
    def _calculate_ensemble_score(self, predictions: Dict[str, Dict[str, Any]], direction: str) -> float:
        """Calculate an ensemble score based on multiple model predictions."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for model_type, prediction in predictions.items():
            # Get weight for this model type
            weight = self.config["ensemble_weights"].get(model_type, 0.0)
            
            # Get directional confidence
            confidence = prediction.get("confidence", 0.5)
            
            # Adjust directional confidence
            if direction == "buy" and prediction.get("direction") == "sell":
                confidence = 1.0 - confidence
            elif direction == "sell" and prediction.get("direction") == "buy":
                confidence = 1.0 - confidence
                
            weighted_sum += weight * confidence
            total_weight += weight
        
        # Calculate weighted average if we have valid weights
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.5  # Neutral if no weights
    
    def _get_relevant_models_for_signal(self, signal: Dict[str, Any]) -> Dict[str, str]:
        """Determine which models to use based on signal type."""
        signal_type = signal.get("signal_type", "").lower()
        
        # Start with all models
        models = self.config["models"].copy()
        
        # Customize based on signal type
        if "breakout" in signal_type:
            # For breakout signals, focus on volatility and trend direction
            return {
                "trend_direction": models["trend_direction"],
                "volatility_forecast": models["volatility_forecast"]
            }
        elif "reversal" in signal_type or "exhaustion" in signal_type:
            # For reversal signals, focus on reversal detection and support/resistance
            return {
                "reversal_detection": models["reversal_detection"],
                "support_resistance": models["support_resistance"]
            }
        elif "harmonic" in signal_type or "pattern" in signal_type:
            # For pattern-based signals, use all models but emphasize S/R
            return models
        elif "trend" in signal_type or "momentum" in signal_type:
            # For trend signals, focus on trend direction
            return {
                "trend_direction": models["trend_direction"],
                "volatility_forecast": models["volatility_forecast"]
            }
        else:
            # Default case: use all configured models
            return models
    
    def _get_primary_model_for_signal(self, signal: Dict[str, Any]) -> str:
        """Get the name of the primary model to use for this signal type."""
        signal_type = signal.get("signal_type", "").lower()
        
        if "breakout" in signal_type:
            return "trend_direction"
        elif "reversal" in signal_type:
            return "reversal_detection"
        elif "harmonic" in signal_type or "pattern" in signal_type:
            return "support_resistance"
        elif "elliott" in signal_type:
            return "trend_direction"
        elif "trend" in signal_type or "momentum" in signal_type:
            return "trend_direction"
        else:
            return "trend_direction"  # Default
    
    def _get_fallback_model(self, model_type: str) -> Optional[str]:
        """Get fallback model name for a given model type."""
        fallbacks = {
            "trend_direction": "legacy_trend_classifier_v1",
            "reversal_detection": "legacy_reversal_detector_v2",
            "volatility_forecast": "naive_volatility_predictor",
            "support_resistance": "price_action_sr_detector"
        }
        return fallbacks.get(model_type)
    
    def _enhance_signal_with_ml_insights(
        self, 
        enhanced_signal: Dict[str, Any], 
        predictions: Dict[str, Dict[str, Any]]
    ) -> None:
        """Add additional ML insights to confirmed signals."""
        # Check for target adjustments from volatility model
        if "volatility_forecast" in predictions:
            vol_pred = predictions["volatility_forecast"]
            
            # Adjust take profit based on volatility forecast
            if "volatility_forecast_percent" in vol_pred:
                forecast_volatility = vol_pred["volatility_forecast_percent"]
                current_tp = enhanced_signal.get("take_profit")
                current_sl = enhanced_signal.get("stop_loss")
                entry_price = enhanced_signal.get("entry_price")
                
                # Only adjust if we have all required values
                if current_tp and current_sl and entry_price:
                    # Calculate current risk/reward ratio
                    if enhanced_signal["direction"] == "buy":
                        risk = entry_price - current_sl
                        reward = current_tp - entry_price
                    else:
                        risk = current_sl - entry_price
                        reward = entry_price - current_tp
                        
                    # Calculate volatility based adjustment
                    if forecast_volatility > 0 and risk > 0:
                        adjusted_reward = risk * max(1.0, forecast_volatility / 0.01)
                        if enhanced_signal["direction"] == "buy":
                            adjusted_tp = entry_price + adjusted_reward
                        else:
                            adjusted_tp = entry_price - adjusted_reward
                        
                        # Update take profit but store original
                        enhanced_signal["original_take_profit"] = current_tp
                        enhanced_signal["take_profit"] = adjusted_tp
                        enhanced_signal["ml_confirmation"]["tp_adjustment_reason"] = "volatility_forecast"
        
        # Check for time-to-target from trend model
        if "trend_direction" in predictions:
            trend_pred = predictions["trend_direction"]
            if "time_to_target_bars" in trend_pred:
                enhanced_signal["ml_confirmation"]["estimated_bars_to_target"] = trend_pred["time_to_target_bars"]
        
        # Add confidence boost from support/resistance model
        if "support_resistance" in predictions:
            sr_pred = predictions["support_resistance"]
            if "key_levels" in sr_pred:
                enhanced_signal["ml_confirmation"]["key_levels"] = sr_pred["key_levels"]
                
                # Check if target is near a key level
                take_profit = enhanced_signal.get("take_profit")
                if take_profit:
                    for level in sr_pred["key_levels"]:
                        level_price = level.get("price")
                        level_strength = level.get("strength", 0.5)
                        
                        if level_price and abs(level_price - take_profit) / take_profit < 0.002:
                            enhanced_signal["ml_confirmation"]["target_at_key_level"] = True
                            enhanced_signal["ml_confirmation"]["key_level_strength"] = level_strength
                            break
    
    def update_performance(self, signal_id: str, outcome: Dict[str, Any]) -> None:
        """
        Update performance metrics based on trade outcomes.
        
        Args:
            signal_id: ID of the signal that was filtered
            outcome: Trade outcome data
        """
        if not self.config["performance_tracking"]:
            return
            
        try:
            was_profitable = outcome.get("profitable", False)
            was_confirmed = outcome.get("ml_confirmation", {}).get("confirmed", False)
            
            # Update overall metrics
            if was_confirmed:
                if was_profitable:
                    self.performance_history["confirmed_correct"] += 1
                else:
                    self.performance_history["confirmed_incorrect"] += 1
            else:
                if was_profitable:
                    self.performance_history["rejected_incorrect"] += 1
                else:
                    self.performance_history["rejected_correct"] += 1
                    
            # Update per-model metrics
            model_predictions = outcome.get("ml_confirmation", {}).get("model_predictions", {})
            for model_type, prediction in model_predictions.items():
                if model_type not in self.performance_history["by_model"]:
                    self.performance_history["by_model"][model_type] = {
                        "total": 0,
                        "correct": 0,
                        "incorrect": 0
                    }
                
                self.performance_history["by_model"][model_type]["total"] += 1
                
                model_direction = prediction.get("direction")
                signal_direction = outcome.get("direction")
                
                is_correct = (model_direction == signal_direction and was_profitable) or \
                            (model_direction != signal_direction and not was_profitable)
                
                if is_correct:
                    self.performance_history["by_model"][model_type]["correct"] += 1
                else:
                    self.performance_history["by_model"][model_type]["incorrect"] += 1
            
            # Adapt thresholds if enabled
            if self.config["adapting_thresholds"] and \
               (self.performance_history["confirmed_signals"] + 
                self.performance_history["rejected_signals"] >= 50):
                self._adapt_thresholds()
                
        except Exception as e:
            self.logger.error(f"Error updating ML confirmation filter performance: {str(e)}", exc_info=True)
    
    def _adapt_thresholds(self) -> None:
        """Adapt confirmation thresholds based on historical performance."""
        try:
            # Calculate metrics
            confirmed_total = self.performance_history["confirmed_correct"] + \
                             self.performance_history["confirmed_incorrect"]
            
            rejected_total = self.performance_history["rejected_correct"] + \
                            self.performance_history["rejected_incorrect"]
            
            if confirmed_total == 0 or rejected_total == 0:
                return  # Not enough data
            
            confirmed_accuracy = self.performance_history["confirmed_correct"] / confirmed_total
            rejection_accuracy = self.performance_history["rejected_correct"] / rejected_total
            
            # Adjust confirmation threshold
            if confirmed_accuracy < 0.5:
                # Poor accuracy on confirmed signals, increase threshold
                new_threshold = min(0.9, self.config["min_confirmation_threshold"] + 0.05)
                self.logger.info(
                    f"Increasing confirmation threshold from {self.config['min_confirmation_threshold']} to {new_threshold} "
                    f"due to low accuracy ({confirmed_accuracy:.2%})"
                )
                self.config["min_confirmation_threshold"] = new_threshold
            elif confirmed_accuracy > 0.75 and self.config["min_confirmation_threshold"] > 0.6:
                # Good accuracy, potentially lower threshold slightly
                new_threshold = max(0.6, self.config["min_confirmation_threshold"] - 0.03)
                self.logger.info(
                    f"Decreasing confirmation threshold from {self.config['min_confirmation_threshold']} to {new_threshold} "
                    f"due to high accuracy ({confirmed_accuracy:.2%})"
                )
                self.config["min_confirmation_threshold"] = new_threshold
                
            # Adjust rejection threshold
            if rejection_accuracy < 0.5:
                # Poor accuracy on rejections, decrease threshold
                new_threshold = max(0.1, self.config["rejection_threshold"] - 0.05)
                self.logger.info(
                    f"Decreasing rejection threshold from {self.config['rejection_threshold']} to {new_threshold} "
                    f"due to low rejection accuracy ({rejection_accuracy:.2%})"
                )
                self.config["rejection_threshold"] = new_threshold
            elif rejection_accuracy > 0.75 and self.config["rejection_threshold"] < 0.5:
                # Good accuracy, potentially increase threshold slightly
                new_threshold = min(0.5, self.config["rejection_threshold"] + 0.03)
                self.logger.info(
                    f"Increasing rejection threshold from {self.config['rejection_threshold']} to {new_threshold} "
                    f"due to high rejection accuracy ({rejection_accuracy:.2%})"
                )
                self.config["rejection_threshold"] = new_threshold
                
        except Exception as e:
            self.logger.error(f"Error adapting ML confirmation thresholds: {str(e)}", exc_info=True)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the ML confirmation filter."""
        metrics = {
            "total_signals": self.performance_history["total_signals"],
            "confirmed_signals": self.performance_history["confirmed_signals"],
            "rejected_signals": self.performance_history["rejected_signals"],
            "confirmation_metrics": {},
            "model_metrics": {}
        }
        
        # Calculate confirmation metrics
        confirmed_total = self.performance_history["confirmed_correct"] + \
                         self.performance_history["confirmed_incorrect"]
                         
        rejected_total = self.performance_history["rejected_correct"] + \
                        self.performance_history["rejected_incorrect"]
        
        if confirmed_total > 0:
            metrics["confirmation_metrics"]["confirmation_accuracy"] = \
                self.performance_history["confirmed_correct"] / confirmed_total
        
        if rejected_total > 0:
            metrics["confirmation_metrics"]["rejection_accuracy"] = \
                self.performance_history["rejected_correct"] / rejected_total
        
        # Calculate overall filter effectiveness
        all_correct = self.performance_history["confirmed_correct"] + \
                     self.performance_history["rejected_correct"]
        all_total = confirmed_total + rejected_total
        
        if all_total > 0:
            metrics["confirmation_metrics"]["overall_accuracy"] = all_correct / all_total
                
        # Calculate per-model metrics
        for model_type, model_stats in self.performance_history["by_model"].items():
            if model_stats["total"] > 0:
                accuracy = model_stats["correct"] / model_stats["total"]
                metrics["model_metrics"][model_type] = {
                    "accuracy": accuracy,
                    "total_predictions": model_stats["total"]
                }
        
        return metrics
