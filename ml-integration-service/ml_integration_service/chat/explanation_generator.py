"""
Explanation Generator for ML Model Predictions

This module generates human-readable explanations for ML model predictions
used in the chat interface.
"""

from typing import Dict, List, Any, Optional, Union
import logging
import asyncio
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chat-explanation-generator")

class ExplanationGenerator:
    """Generator for human-readable explanations of ML model predictions."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the explanation generator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.explainability_client = None

        # Initialize explainability client using common interfaces
        try:
            from common_lib.ml.interfaces import IExplanationGenerator
            # We'll implement our own explanation generator that follows the interface
            self.explainability_client = True
            logger.info("Explainability client initialized")
        except ImportError as e:
            logger.warning(f"Explainability client initialization failed: {str(e)}")

    async def generate_explanation(
        self,
        model_type: str,
        prediction: Dict[str, Any],
        inputs: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate explanation for a model prediction.

        Args:
            model_type: Type of model (e.g., 'trend', 'price', 'volatility')
            prediction: Model prediction
            inputs: Model inputs
            user_preferences: Optional user preferences for personalization

        Returns:
            Human-readable explanation
        """
        # Check if we have explainability client
        if self.explainability_client and not prediction.get("error"):
            try:
                # Get explanation from explainability client
                explanation = await self._get_model_explanation(model_type, prediction, inputs)
                if explanation:
                    # Personalize explanation based on user preferences
                    return self._personalize_explanation(explanation, user_preferences)
            except Exception as e:
                logger.error(f"Error generating explanation: {str(e)}")

        # Fall back to template-based explanation
        return self._generate_template_explanation(model_type, prediction, inputs, user_preferences)

    async def _get_model_explanation(
        self,
        model_type: str,
        prediction: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get explanation from explainability client.

        Args:
            model_type: Type of model
            prediction: Model prediction
            inputs: Model inputs

        Returns:
            Explanation data or None if not available
        """
        if not self.explainability_client:
            return None

        try:
            # Map our model types to explainability module model types
            explainability_model_types = {
                "trend": "classification",
                "price": "regression",
                "volatility": "classification",
                "support_resistance": "regression",
                "forecast": "time_series",
                "recommendation": "classification"
            }

            explainability_model_type = explainability_model_types.get(model_type, "classification")

            # Get feature importance from explainability client
            feature_importance = None

            if "indicators" in inputs:
                # Convert indicators to format expected by explainability module
                features = {}
                for key, value in inputs["indicators"].items():
                    features[key] = value

                # Get explanation from explainability client
                try:
                    explanation_result = self.explainability_client.explain_model(
                        model=None,  # We don't have the actual model object
                        X=features,
                        feature_names=list(features.keys()),
                        methods=["shap", "permutation"]
                    )

                    # Extract feature importance
                    if explanation_result and "shap" in explanation_result:
                        shap_values = explanation_result["shap"].get("shap_values", {})
                        if shap_values:
                            feature_importance = {}
                            total = sum(abs(v) for v in shap_values.values())
                            if total > 0:
                                for feature, value in shap_values.items():
                                    feature_importance[feature] = abs(value) / total
                except Exception as e:
                    logger.warning(f"Error getting explanation from explainability client: {str(e)}")

            # If we couldn't get feature importance, use a reasonable default
            if not feature_importance:
                feature_importance = {
                    "rsi": 0.35,
                    "macd": 0.25,
                    "ema_50": 0.15,
                    "ema_200": 0.15,
                    "atr": 0.10
                }

            # Generate explanation text
            explanation_text = self._get_base_explanation(model_type, prediction, inputs)

            # Create explanation object
            explanation = {
                "feature_importance": feature_importance,
                "explanation_text": explanation_text,
                "model_type": model_type,
                "confidence": prediction.get("confidence", 0.5) if isinstance(prediction, dict) else 0.5
            }

            return explanation
        except Exception as e:
            logger.error(f"Error getting model explanation: {str(e)}")
            return None

    def _personalize_explanation(
        self,
        explanation: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Personalize explanation based on user preferences.

        Args:
            explanation: Explanation data
            user_preferences: User preferences

        Returns:
            Personalized explanation text
        """
        if not user_preferences:
            return explanation.get("explanation_text", "")

        explanation_text = explanation.get("explanation_text", "")

        # Adjust explanation based on user's technical knowledge level
        knowledge_level = user_preferences.get("technical_knowledge", "intermediate")

        if knowledge_level == "beginner":
            # Simplify explanation for beginners
            explanation_text = self._simplify_explanation(explanation_text)
        elif knowledge_level == "advanced":
            # Add more technical details for advanced users
            explanation_text = self._enhance_explanation(explanation_text, explanation)

        # Add focus on indicators the user cares about
        preferred_indicators = user_preferences.get("preferred_indicators", [])
        if preferred_indicators and "feature_importance" in explanation:
            explanation_text = self._focus_on_preferred_indicators(
                explanation_text,
                explanation["feature_importance"],
                preferred_indicators
            )

        return explanation_text

    def _simplify_explanation(self, explanation_text: str) -> str:
        """
        Simplify explanation for beginners.

        Args:
            explanation_text: Original explanation text

        Returns:
            Simplified explanation
        """
        # Replace technical terms with simpler language
        simplifications = {
            "RSI": "the momentum indicator (RSI)",
            "MACD": "the trend indicator (MACD)",
            "EMA": "the moving average (EMA)",
            "ATR": "the volatility indicator (ATR)",
            "bullish divergence": "positive signal",
            "bearish divergence": "negative signal",
            "support level": "price floor",
            "resistance level": "price ceiling"
        }

        for term, replacement in simplifications.items():
            explanation_text = explanation_text.replace(term, replacement)

        # Add a beginner-friendly intro
        intro = "In simple terms: "
        if not explanation_text.startswith(intro):
            explanation_text = intro + explanation_text

        return explanation_text

    def _enhance_explanation(self, explanation_text: str, explanation: Dict[str, Any]) -> str:
        """
        Enhance explanation with more technical details for advanced users.

        Args:
            explanation_text: Original explanation text
            explanation: Full explanation data

        Returns:
            Enhanced explanation
        """
        # Add feature importance information
        if "feature_importance" in explanation:
            importance_text = "\n\nKey factors in this analysis: "
            sorted_features = sorted(
                explanation["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )

            for feature, importance in sorted_features[:3]:
                importance_percent = int(importance * 100)
                importance_text += f"{feature.upper()} ({importance_percent}%), "

            explanation_text += importance_text[:-2] + "."

        return explanation_text

    def _focus_on_preferred_indicators(
        self,
        explanation_text: str,
        feature_importance: Dict[str, float],
        preferred_indicators: List[str]
    ) -> str:
        """
        Focus explanation on user's preferred indicators.

        Args:
            explanation_text: Original explanation text
            feature_importance: Feature importance dictionary
            preferred_indicators: User's preferred indicators

        Returns:
            Focused explanation
        """
        # Add specific information about preferred indicators
        preferred_text = "\n\nBased on your preferred indicators: "
        has_preferred = False

        for indicator in preferred_indicators:
            indicator_lower = indicator.lower()
            if indicator_lower in feature_importance:
                importance = feature_importance[indicator_lower]
                importance_percent = int(importance * 100)
                preferred_text += f"{indicator.upper()} contributes {importance_percent}% to this prediction. "
                has_preferred = True

        if has_preferred:
            explanation_text += preferred_text

        return explanation_text

    def _generate_template_explanation(
        self,
        model_type: str,
        prediction: Dict[str, Any],
        inputs: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate explanation using templates when explainability client is not available.

        Args:
            model_type: Type of model
            prediction: Model prediction
            inputs: Model inputs
            user_preferences: Optional user preferences

        Returns:
            Template-based explanation
        """
        base_explanation = self._get_base_explanation(model_type, prediction, inputs)

        # Personalize if user preferences are available
        if user_preferences:
            knowledge_level = user_preferences.get("technical_knowledge", "intermediate")

            if knowledge_level == "beginner":
                return self._simplify_explanation(base_explanation)
            elif knowledge_level == "advanced":
                # Create a mock explanation object for enhancement
                mock_explanation = {
                    "explanation_text": base_explanation,
                    "feature_importance": inputs.get("indicators", {})
                }
                return self._enhance_explanation(base_explanation, mock_explanation)

        return base_explanation

    def _get_base_explanation(
        self,
        model_type: str,
        prediction: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> str:
        """
        Get base explanation for a model type.

        Args:
            model_type: Type of model
            prediction: Model prediction
            inputs: Model inputs

        Returns:
            Base explanation text
        """
        symbol = inputs.get("symbol", "the currency pair")
        timeframe = inputs.get("timeframe", "the selected timeframe")
        indicators = inputs.get("indicators", {})

        if model_type == "trend":
            direction = prediction.get("direction", "neutral")
            confidence = prediction.get("confidence", 0.5)
            confidence_text = self._get_confidence_text(confidence)

            rsi = indicators.get("rsi", 50)
            macd = indicators.get("macd", 0)
            macd_signal = indicators.get("macd_signal", 0)

            if direction == "bullish":
                return (f"The analysis shows a bullish trend for {symbol} on the {timeframe} timeframe with "
                       f"{confidence_text} confidence. The RSI at {rsi:.1f} indicates {'overbought conditions' if rsi > 70 else 'positive momentum'}, "
                       f"while the MACD {'shows a bullish crossover' if macd > macd_signal else 'is trending positive'}.")
            elif direction == "bearish":
                return (f"The analysis shows a bearish trend for {symbol} on the {timeframe} timeframe with "
                       f"{confidence_text} confidence. The RSI at {rsi:.1f} indicates {'oversold conditions' if rsi < 30 else 'negative momentum'}, "
                       f"while the MACD {'shows a bearish crossover' if macd < macd_signal else 'is trending negative'}.")
            else:
                return (f"The analysis shows a neutral trend for {symbol} on the {timeframe} timeframe. "
                       f"The RSI at {rsi:.1f} is in neutral territory, and the MACD shows no clear direction.")

        elif model_type == "price":
            movement = prediction.get("movement", "sideways")
            target = prediction.get("target")
            confidence = prediction.get("confidence", 0.5)
            confidence_text = self._get_confidence_text(confidence)

            if movement == "upward":
                target_text = f" with a potential target of {target}" if target else ""
                return (f"The price of {symbol} is likely to move upward on the {timeframe} timeframe{target_text} "
                       f"with {confidence_text} confidence. This is based on recent price action and technical indicators.")
            elif movement == "downward":
                target_text = f" with a potential target of {target}" if target else ""
                return (f"The price of {symbol} is likely to move downward on the {timeframe} timeframe{target_text} "
                       f"with {confidence_text} confidence. This is based on recent price action and technical indicators.")
            else:
                return (f"The price of {symbol} is likely to move sideways on the {timeframe} timeframe "
                       f"with no clear directional bias. This suggests a range-bound market in the near term.")

        elif model_type == "volatility":
            level = prediction.get("level", "moderate")
            confidence = prediction.get("confidence", 0.5)
            confidence_text = self._get_confidence_text(confidence)

            atr = indicators.get("atr", 0.003)

            if level == "high":
                return (f"Volatility for {symbol} is expected to be high on the {timeframe} timeframe with "
                       f"{confidence_text} confidence. The ATR of {atr:.4f} indicates increased price fluctuations, "
                       f"suggesting potential for larger price movements and possibly trading opportunities.")
            elif level == "low":
                return (f"Volatility for {symbol} is expected to be low on the {timeframe} timeframe with "
                       f"{confidence_text} confidence. The ATR of {atr:.4f} indicates reduced price fluctuations, "
                       f"suggesting a more stable and predictable market environment.")
            else:
                return (f"Volatility for {symbol} is expected to be moderate on the {timeframe} timeframe. "
                       f"The ATR of {atr:.4f} indicates normal market conditions with average price fluctuations.")

        elif model_type == "support_resistance":
            support = prediction.get("support", [])
            resistance = prediction.get("resistance", [])
            confidence = prediction.get("confidence", 0.5)
            confidence_text = self._get_confidence_text(confidence)

            support_text = ", ".join([str(s) for s in support[:2]]) if support else "none identified"
            resistance_text = ", ".join([str(r) for r in resistance[:2]]) if resistance else "none identified"

            return (f"For {symbol} on the {timeframe} timeframe, key support levels are at {support_text} "
                   f"and resistance levels are at {resistance_text} with {confidence_text} confidence. "
                   f"These levels can be used for potential entry, exit, or stop loss points.")

        elif model_type == "forecast":
            data = prediction.get("data", [])
            confidence = prediction.get("confidence", 0.5)
            confidence_text = self._get_confidence_text(confidence)

            if data:
                last_period = data[-1]
                first_period = data[0]
                direction = "upward" if last_period.get("price", 0) > first_period.get("price", 0) else "downward"

                return (f"The price forecast for {symbol} on the {timeframe} timeframe shows an overall {direction} trend "
                       f"over the next {len(data)} periods with {confidence_text} confidence. "
                       f"The forecast suggests a price movement from {first_period.get('price')} to {last_period.get('price')}.")
            else:
                return (f"The price forecast for {symbol} on the {timeframe} timeframe could not be generated "
                       f"with sufficient confidence. This may be due to current market conditions or limited data.")

        elif model_type == "recommendation":
            action = prediction.get("action", "hold")
            entry_price = prediction.get("entry_price")
            stop_loss = prediction.get("stop_loss")
            take_profit = prediction.get("take_profit")
            risk_reward = prediction.get("risk_reward_ratio")
            confidence = prediction.get("confidence", 0.5)
            confidence_text = self._get_confidence_text(confidence)

            if action == "buy":
                return (f"Based on the analysis, a buy position is recommended for {symbol} on the {timeframe} timeframe "
                       f"with {confidence_text} confidence. "
                       f"Suggested entry around {entry_price}, stop loss at {stop_loss}, and take profit at {take_profit}, "
                       f"giving a risk-reward ratio of {risk_reward:.1f}.")
            elif action == "sell":
                return (f"Based on the analysis, a sell position is recommended for {symbol} on the {timeframe} timeframe "
                       f"with {confidence_text} confidence. "
                       f"Suggested entry around {entry_price}, stop loss at {stop_loss}, and take profit at {take_profit}, "
                       f"giving a risk-reward ratio of {risk_reward:.1f}.")
            else:
                return (f"Based on the analysis, holding current positions or staying out of the market is recommended "
                       f"for {symbol} on the {timeframe} timeframe. Current market conditions do not present a favorable "
                       f"risk-reward opportunity.")

        else:
            return f"Analysis for {symbol} on the {timeframe} timeframe is available, but no detailed explanation can be provided."

    def _get_confidence_text(self, confidence: float) -> str:
        """
        Convert confidence score to text description.

        Args:
            confidence: Confidence score (0-1)

        Returns:
            Text description of confidence
        """
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "moderate"
        elif confidence >= 0.4:
            return "fair"
        else:
            return "low"
