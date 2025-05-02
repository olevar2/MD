"""
Feedback Loop

This module implements the feedback loop system that connects strategy execution outcomes
with the adaptation engine, enabling continuous improvement of parameter adjustments.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from collections import deque # Use deque for efficient fixed-size storage

# Assuming these are defined elsewhere, potentially in core-foundations
from core_foundations.models.feedback import TradeFeedback, FeedbackSource, FeedbackCategory, FeedbackStatus 
from core_foundations.events.event_publisher import EventPublisher # If events are published
from core_foundations.events.event_schema import EventType # If events are published

# Import other adaptive layer components if needed for integration
from .multi_timeframe_feedback import MultiTimeframeFeedbackProcessor
from .parameter_feedback import ParameterFeedbackAnalyzer
from .strategy_mutation import StrategyMutator

from analysis_engine.adaptive_layer.adaptation_engine import AdaptationEngine
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)

class FeedbackLoop:
    """
    The FeedbackLoop system captures the outcomes of strategy execution and feeds them
    back to the adaptation engine, enabling continuous learning and improvement of 
    parameter adjustments.
    
    Key capabilities:
    - Track the performance of parameter adjustments
    - Analyze the effectiveness of adaptation decisions
    - Adjust adaptation strategies based on outcome data
    - Provide insights for strategy optimization
    """
    
    def __init__(
        self,
        adaptation_engine: AdaptationEngine,
        event_publisher: Optional[EventPublisher] = None, # Added event publisher
        config: Dict[str, Any] = None,
        # Inject specialized processors
        multi_timeframe_processor: Optional[MultiTimeframeFeedbackProcessor] = None,
        parameter_analyzer: Optional[ParameterFeedbackAnalyzer] = None,
        strategy_mutator: Optional[StrategyMutator] = None,
    ):        """
        Initialize the FeedbackLoop system.
        
        Args:
            adaptation_engine: The adaptation engine to integrate with.
            event_publisher: Optional event publisher for feedback loop events.
            config: Configuration parameters for the feedback loop.
            multi_timeframe_processor: Processor for multi-timeframe analysis.
            parameter_analyzer: Analyzer for parameter-specific feedback.
            strategy_mutator: Framework for mutating strategies.
        \"\"\"
        self.adaptation_engine = adaptation_engine
        self.event_publisher = event_publisher
        self.config = config or {}
        
        # Instantiate processors if not provided (using config)
        self.multi_timeframe_processor = multi_timeframe_processor or MultiTimeframeFeedbackProcessor(
            config=self.config.get('multi_timeframe_config')
        )
        self.parameter_analyzer = parameter_analyzer or ParameterFeedbackAnalyzer(
            config=self.config.get('parameter_analyzer_config')
        )
        self.strategy_mutator = strategy_mutator or StrategyMutator(
            config=self.config.get('strategy_mutator_config')
        )
        
        # Store recent raw feedback items (optional, depends on processing needs)
        self.recent_feedback = deque(maxlen=self.config.get('recent_feedback_buffer_size', 1000))
        
        # Store adaptation outcomes for analysis (using deque for fixed size)
        self.adaptation_outcomes = deque(maxlen=self.config.get('adaptation_outcome_buffer_size', 500))
        
        # Performance tracking by strategy and market regime
        self.performance_by_regime = {}
        
        # Effectiveness metrics for adaptation strategies
        self.adaptation_effectiveness = {
            'strategy_type': {},
            'regime_type': {},
            'parameter_type': {}
        }
        
        logger.info("FeedbackLoop initialized")

    async def add_feedback(self, feedback: TradeFeedback) -> None:
        \"\"\"
        Processes incoming feedback, updates internal state, and potentially triggers adaptation.
        This is the entry point called by TradingFeedbackCollector.

        Args:
            feedback: The feedback object received.
        \"\"\"
        logger.debug(f"FeedbackLoop received feedback: {feedback.id} (Source: {feedback.source}, Category: {feedback.category})")
        
        # Store raw feedback if needed for later analysis
        self.recent_feedback.append(feedback)

        # --- Core Feedback Processing Workflow ---
        
        # 1. Basic Outcome Recording (if it's strategy execution feedback)
        # This part assumes feedback contains necessary info like adaptation_id, metrics, regime
        if feedback.source == FeedbackSource.STRATEGY_EXECUTION and feedback.category == FeedbackCategory.PERFORMANCE:
             # Ensure required fields are present
             if hasattr(feedback, 'strategy_id') and hasattr(feedback, 'instrument') and \
                hasattr(feedback, 'timeframe') and hasattr(feedback, 'adaptation_id') and \
                hasattr(feedback, 'outcome_metrics') and hasattr(feedback, 'market_regime'):
                 
                 self.record_strategy_outcome(
                     strategy_id=feedback.strategy_id,
                     instrument=feedback.instrument,
                     timeframe=feedback.timeframe,
                     adaptation_id=feedback.adaptation_id,
                     outcome_metrics=feedback.outcome_metrics,
                     market_regime=feedback.market_regime,
                     timestamp=datetime.fromisoformat(feedback.timestamp) if feedback.timestamp else datetime.utcnow()
                 )
             else:
                 logger.warning(f"Skipping outcome recording for feedback {feedback.id}: Missing required attributes.")

        # 2. Route feedback to specialized processors/analyzers
        multi_timeframe_insights = None
        parameter_insights = None
        
        if self.multi_timeframe_processor:
            try:
                multi_timeframe_insights = await self.multi_timeframe_processor.process(feedback)
                if multi_timeframe_insights:
                    logger.info(f"Multi-timeframe insights generated for feedback {feedback.id}: {multi_timeframe_insights}")
                    # TODO: Store or act on these insights
            except Exception as e:
                logger.error(f"Error processing multi-timeframe feedback for {feedback.id}: {e}", exc_info=True)

        if self.parameter_analyzer:
            try:
                parameter_insights = await self.parameter_analyzer.analyze(feedback)
                if parameter_insights:
                    logger.info(f"Parameter insights generated for feedback {feedback.id}: {parameter_insights}")
                    # TODO: Store or act on these insights
            except Exception as e:
                logger.error(f"Error processing parameter feedback for {feedback.id}: {e}", exc_info=True)

        # 3. Trigger Adaptation Engine (based on aggregated insights or specific feedback types)
        adaptation_context = {
             'triggering_feedback_id': feedback.id,
             'category': feedback.category.value,
             'details': feedback.details,
             'strategy_id': getattr(feedback, 'strategy_id', None),
             'instrument': getattr(feedback, 'instrument', None),
             'timeframe': getattr(feedback, 'timeframe', None),
             # Include insights from specialized processors
             'multi_timeframe_insights': multi_timeframe_insights,
             'parameter_insights': parameter_insights,
        }
        
        # AdaptationEngine decides if/how to adapt based on this context
        adaptation_decision = await self.adaptation_engine.evaluate_and_adapt(adaptation_context)

        # 4. Potentially Trigger Strategy Mutation (if adaptation suggests it)
        # This logic might be better placed within AdaptationEngine, but shown here for integration
        if adaptation_decision and adaptation_decision.get('action') == 'mutate_strategy' and self.strategy_mutator:
            strategy_id_to_mutate = adaptation_decision.get('strategy_id')
            if strategy_id_to_mutate:
                # TODO: Need a way to get the current strategy config
                # current_strategy_config = await self.get_strategy_config(strategy_id_to_mutate)
                current_strategy_config = { # Placeholder
                    'id': strategy_id_to_mutate,
                    'parameters': {'param1': 10, 'param2': 0.5},
                    'version': 1.0
                }
                if current_strategy_config:
                    logger.info(f"Triggering strategy mutation for {strategy_id_to_mutate} based on adaptation decision.")
                    try:
                        mutated_config = await self.strategy_mutator.mutate_strategy(
                            current_strategy_config,
                            feedback_context=[feedback] # Pass relevant feedback
                        )
                        if mutated_config:
                            logger.info(f"Strategy {strategy_id_to_mutate} mutated. New config: {mutated_config}")
                            # TODO: Deploy/test the mutated strategy
                            # await self.deploy_mutated_strategy(mutated_config)
                    except Exception as e:
                        logger.error(f"Error during strategy mutation for {strategy_id_to_mutate}: {e}", exc_info=True)

        # 5. Update Adaptation Strategy (Reinforcement Learning - Future Enhancement)
        # This part remains largely conceptual for now.
        # It would involve using the recorded outcomes to train a meta-policy
        # on *how* to adapt best under different circumstances.
        # self.update_adaptation_strategy(...) 

        # 6. Publish Processed Feedback Event (Optional)
        if self.event_publisher:
             try:
                 await self.event_publisher.publish(
                     EventType.FEEDBACK_PROCESSED, # Use Enum
                     {
                         "feedback_id": feedback.id,
                         "status": feedback.status.value, # Reflect status after processing
                         "timestamp": datetime.utcnow().isoformat()
                     }
                 )
             except Exception as e:
                 logger.error(f"Failed to publish FEEDBACK_PROCESSED event for {feedback.id}: {e}", exc_info=True)

        logger.debug(f"FeedbackLoop finished processing feedback: {feedback.id}")


    def record_strategy_outcome(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        adaptation_id: str,
        outcome_metrics: Dict[str, Any],
        market_regime: str,
        timestamp: datetime = None
    ) -> None:
        \"\"\"
        Record the outcome of a strategy execution with adapted parameters.
        (Now uses deque for storage)
        \"\"\"
        timestamp = timestamp or datetime.utcnow()
        
        outcome_record = {
            'timestamp': timestamp,
            'strategy_id': strategy_id,
            'instrument': instrument,
            'timeframe': timeframe,
            'adaptation_id': adaptation_id,
            'market_regime': market_regime,
            'metrics': outcome_metrics
        }
        
        # Append to deque (automatically handles maxlen)
        self.adaptation_outcomes.append(outcome_record)
        
        # Update performance tracking
        self._update_performance_tracking(outcome_record)
        
        # Update adaptation effectiveness metrics
        self._update_adaptation_effectiveness(outcome_record)
        
        logger.info(
            f"Recorded strategy outcome for {strategy_id} (Adaptation: {adaptation_id}, Regime: {market_regime})"
        )

    def _update_performance_tracking(self, outcome_record: Dict[str, Any]) -> None:
        """
        Update the performance tracking data.
        
        Args:
            outcome_record: The outcome record to process
        """
        strategy_id = outcome_record['strategy_id']
        market_regime = outcome_record['market_regime']
        metrics = outcome_record['metrics']
        
        # Create tracking keys if they don't exist
        if strategy_id not in self.performance_by_regime:
            self.performance_by_regime[strategy_id] = {}
            
        if market_regime not in self.performance_by_regime[strategy_id]:
            self.performance_by_regime[strategy_id][market_regime] = {
                'count': 0,
                'win_count': 0,
                'loss_count': 0,
                'profit_sum': 0.0,
                'loss_sum': 0.0,
                'avg_profit_factor': 0.0,
                'avg_win_rate': 0.0
            }
            
        perf = self.performance_by_regime[strategy_id][market_regime]
        
        # Update counts
        perf['count'] += 1
        
        # Process basic metrics
        if metrics.get('profit', 0) > 0:
            perf['win_count'] += 1
            perf['profit_sum'] += metrics.get('profit', 0)
        else:
            perf['loss_count'] += 1
            perf['loss_sum'] += abs(metrics.get('profit', 0))
            
        # Recalculate derived metrics
        if perf['count'] > 0:
            perf['avg_win_rate'] = perf['win_count'] / perf['count']
            
        if perf['loss_sum'] > 0:
            perf['avg_profit_factor'] = perf['profit_sum'] / perf['loss_sum']
        else:
            perf['avg_profit_factor'] = perf['profit_sum'] if perf['profit_sum'] > 0 else 0
            
    def _update_adaptation_effectiveness(self, outcome_record: Dict[str, Any]) -> None:
        """
        Update the effectiveness metrics for adaptation strategies.
        
        Args:
            outcome_record: The outcome record to process
        """
        strategy_id = outcome_record['strategy_id']
        market_regime = outcome_record['market_regime']
        metrics = outcome_record['metrics']
        
        # Initialize effectiveness tracking if needed
        if strategy_id not in self.adaptation_effectiveness['strategy_type']:
            self.adaptation_effectiveness['strategy_type'][strategy_id] = {
                'count': 0,
                'success_count': 0,
                'failure_count': 0,
                'success_rate': 0.0
            }
            
        if market_regime not in self.adaptation_effectiveness['regime_type']:
            self.adaptation_effectiveness['regime_type'][market_regime] = {
                'count': 0,
                'success_count': 0,
                'failure_count': 0,
                'success_rate': 0.0
            }
            
        # Track by strategy type
        strat = self.adaptation_effectiveness['strategy_type'][strategy_id]
        strat['count'] += 1
        
        is_success = metrics.get('profit', 0) > 0
        if is_success:
            strat['success_count'] += 1
        else:
            strat['failure_count'] += 1
            
        strat['success_rate'] = strat['success_count'] / strat['count'] if strat['count'] > 0 else 0.0
        
        # Track by regime type
        regime = self.adaptation_effectiveness['regime_type'][market_regime]
        regime['count'] += 1
        
        if is_success:
            regime['success_count'] += 1
        else:
            regime['failure_count'] += 1
            
        regime['success_rate'] = regime['success_count'] / regime['count'] if regime['count'] > 0 else 0.0
        
        # TODO: Add tracking for specific parameter types when parameter details are available

    def get_adaptation_effectiveness(
        self,
        strategy_id: Optional[str] = None,
        market_regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the effectiveness metrics for adaptation strategies.
        
        Args:
            strategy_id: Optional filter for a specific strategy
            market_regime: Optional filter for a specific market regime
            
        Returns:
            Dict[str, Any]: Adaptation effectiveness metrics
        """
        if strategy_id:
            return self.adaptation_effectiveness['strategy_type'].get(strategy_id, {})
        
        if market_regime:
            return self.adaptation_effectiveness['regime_type'].get(market_regime, {})
            
        return self.adaptation_effectiveness
        
    def get_performance_by_regime(
        self,
        strategy_id: str,
        market_regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics by market regime for a specific strategy.
        
        Args:
            strategy_id: Identifier for the strategy
            market_regime: Optional filter for a specific market regime
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        if strategy_id not in self.performance_by_regime:
            return {}
            
        if market_regime:
            return self.performance_by_regime[strategy_id].get(market_regime, {})
            
        return self.performance_by_regime[strategy_id]
        
    def update_adaptation_strategy(self, strategy_id: str) -> bool:
        """
        Update the adaptation strategies based on feedback data.
        
        Args:
            strategy_id: The strategy to update adaptation strategies for
            
        Returns:
            bool: True if strategies were updated, False otherwise
        """
        # This would implement reinforcement learning to improve adaptation strategies
        # based on collected feedback data
        
        # TODO: Implement reinforcement learning for adaptation strategy updates
        # For now, this is a placeholder for future enhancement
        
        logger.info("Adaptation strategy update not yet implemented for %s", strategy_id)
        return False
        
    def generate_insights(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        Generate insights from feedback data to optimize strategy.
        
        Args:
            strategy_id: The strategy to generate insights for
            
        Returns:
            List[Dict[str, Any]]: List of insights
        """
        insights = []
        
        # Skip if no data available
        if strategy_id not in self.performance_by_regime:
            return insights
            
        performance = self.performance_by_regime[strategy_id]
        
        # Find best performing regime
        best_regime = None
        best_rate = 0.0
        
        for regime, metrics in performance.items():
            if metrics['count'] >= self.config.get('min_samples', 10):
                if metrics['avg_win_rate'] > best_rate:
                    best_rate = metrics['avg_win_rate']
                    best_regime = regime
                    
        if best_regime:
            insights.append({
                'type': 'best_regime',
                'message': f"Strategy {strategy_id} performs best in {best_regime} regime with {best_rate:.1%} win rate",
                'data': {
                    'regime': best_regime,
                    'win_rate': best_rate
                }
            })
            
        # Find worst performing regime
        worst_regime = None
        worst_rate = 1.0
        
        for regime, metrics in performance.items():
            if metrics['count'] >= self.config.get('min_samples', 10):
                if metrics['avg_win_rate'] < worst_rate:
                    worst_rate = metrics['avg_win_rate']
                    worst_regime = regime
                    
        if worst_regime:
            insights.append({
                'type': 'worst_regime',
                'message': f"Strategy {strategy_id} performs poorly in {worst_regime} regime with only {worst_rate:.1%} win rate",
                'data': {
                    'regime': worst_regime,
                    'win_rate': worst_rate
                }
            })
            
        # Check for high variance regimes
        for regime, metrics in performance.items():
            if metrics['count'] >= self.config.get('min_samples', 10):
                profit_factor = metrics['avg_profit_factor']
                win_rate = metrics['avg_win_rate']
                
                if win_rate > 0.5 and profit_factor < 1.0:
                    insights.append({
                        'type': 'inconsistent_performance',
                        'message': f"Strategy {strategy_id} has positive win rate {win_rate:.1%} but negative profit factor {profit_factor:.2f} in {regime} regime",
                        'data': {
                            'regime': regime,
                            'win_rate': win_rate,
                            'profit_factor': profit_factor
                        }
                    })
                    
        return insights
    
    async def process_incoming_feedback(self, feedback: TradeFeedback) -> str:
        """
        Process incoming feedback received from Kafka or other event sources.
        
        This method orchestrates the flow through the feedback components:
        1. Stores the feedback
        2. Categorizes it 
        3. Routes it to appropriate handlers
        4. Optionally publishes events about the feedback processing
        
        Args:
            feedback: The feedback object to process
            
        Returns:
            str: The feedback ID
        """
        logger.info(f"Processing incoming feedback: {feedback.feedback_id}, source: {feedback.source}, category: {feedback.category}")
        
        try:
            # Step 1: Store the feedback (if we have a collector or repository component)
            # This would typically be handled by TradingFeedbackCollector
            # But since we're in the FeedbackLoop, we'll record it in our internal records
            self.record_feedback(feedback)
            
            # Step 2: Categorize if not already categorized
            if feedback.category == FeedbackCategory.UNCATEGORIZED:
                # This would be handled by FeedbackCategorizer 
                # For now, we'll use a simple categorization based on outcome metrics
                categorized = await self._categorize_feedback(feedback)
                if not categorized:
                    logger.warning(f"Could not categorize feedback {feedback.feedback_id}")
            
            # Step 3: Route to appropriate handlers based on feedback type
            if feedback.source == FeedbackSource.MODEL_PREDICTION:
                await self._handle_model_feedback(feedback)
            elif feedback.source == FeedbackSource.TRADING_OUTCOME:
                await self._handle_trading_outcome_feedback(feedback)
            elif feedback.source == FeedbackSource.PARAMETER_ADJUSTMENT:
                await self._handle_parameter_feedback(feedback)
            else:
                # Default handling for other feedback types
                await self._handle_general_feedback(feedback)
            
            # Step 4: Publish event about feedback processing if we have an event publisher
            if self.event_publisher:
                event_data = {
                    "feedback_id": feedback.feedback_id,
                    "source": feedback.source.value if hasattr(feedback.source, "value") else str(feedback.source),
                    "category": feedback.category.value if hasattr(feedback.category, "value") else str(feedback.category),
                    "status": FeedbackStatus.PROCESSED.value,
                    "processed_at": datetime.utcnow().isoformat()
                }
                
                # Assuming EventType and event_publisher.publish are defined elsewhere
                self.event_publisher.publish(
                    event_type=EventType("feedback.processed"),
                    data=event_data
                )
            
            # Update feedback status
            feedback.status = FeedbackStatus.PROCESSED
            
            return feedback.feedback_id
            
        except Exception as e:
            logger.error(f"Error processing feedback {feedback.feedback_id}: {str(e)}", exc_info=True)
            feedback.status = FeedbackStatus.ERROR
            # Attempt to publish error event
            if self.event_publisher:
                error_event = {
                    "feedback_id": feedback.feedback_id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                try:
                    self.event_publisher.publish(
                        event_type=EventType("feedback.processing_error"),
                        data=error_event
                    )
                except Exception as pub_e:
                    logger.error(f"Failed to publish error event: {str(pub_e)}")
            
            return feedback.feedback_id

    async def _categorize_feedback(self, feedback: TradeFeedback) -> bool:
        """
        Categorize uncategorized feedback based on its content
        
        Args:
            feedback: The feedback to categorize
            
        Returns:
            bool: True if categorization was successful
        """
        try:
            metrics = feedback.outcome_metrics or {}
            
            # Simple categorization logic - this would be more sophisticated in practice
            if "profit_loss" in metrics:
                pl = metrics["profit_loss"]
                if isinstance(pl, (int, float)):
                    if pl > 0:
                        feedback.category = FeedbackCategory.SUCCESS
                    else:
                        feedback.category = FeedbackCategory.FAILURE
                    return True
            
            # For model predictions
            if "prediction_error" in metrics:
                error = metrics["prediction_error"]
                if isinstance(error, (int, float)):
                    if error < 0.1: # Arbitrary threshold
                        feedback.category = FeedbackCategory.SUCCESS
                    else:
                        feedback.category = FeedbackCategory.FAILURE
                    return True
            
            # If we couldn't categorize, leave as is or assign a default
            if feedback.category == FeedbackCategory.UNCATEGORIZED:
                feedback.category = FeedbackCategory.OTHER
                
            return False
            
        except Exception as e:
            logger.error(f"Error categorizing feedback: {str(e)}")
            return False
            
    async def _handle_model_feedback(self, feedback: TradeFeedback):
        """Handle feedback related to model predictions"""
        # This would integrate with model retraining or model feedback components
        # For now, just log the feedback
        logger.info(f"Processing model feedback: {feedback.feedback_id}")
        
    async def _handle_trading_outcome_feedback(self, feedback: TradeFeedback):
        """Handle feedback from trading outcomes"""
        # Extract key information
        strategy_id = feedback.strategy_id
        instrument = feedback.instrument
        metrics = feedback.outcome_metrics or {}
        
        logger.info(f"Processing trading outcome feedback for {strategy_id} on {instrument}")
        
        # Record strategy outcome for adaptation engine
        if strategy_id and self.adaptation_engine:
            # Extract market regime from metadata if available
            market_regime = feedback.metadata.get("market_regime", "unknown")
            
            # Record in adaptation engine for future parameter adjustments
            await self.adaptation_engine.record_strategy_outcome(
                strategy_id=strategy_id,
                instrument=instrument,
                outcome_metrics=metrics,
                market_regime=market_regime,
                timeframe=feedback.timeframe
            )
        
    async def _handle_parameter_feedback(self, feedback: TradeFeedback):
        """Handle feedback about parameter adjustments"""
        parameter_name = feedback.metadata.get("parameter_name")
        if not parameter_name:
            logger.warning(f"Parameter feedback missing parameter_name: {feedback.feedback_id}")
            return
            
        # TODO: Integrate with parameter tracking service
        logger.info(f"Processing parameter feedback for {parameter_name}")
        
    async def _handle_general_feedback(self, feedback: TradeFeedback):
        """Handle general feedback types"""
        logger.info(f"Processing general feedback: {feedback.feedback_id}, category: {feedback.category}")
        # Basic handling for now - extensions would add more specific logic

    async def handle_model_training_completed(self, event: Event) -> None:
        """
        Handles model training completion events, updating adaptation outcomes
        and model performance metrics.
        
        Args:
            event: Model training completed event
        """
        event_data = event.data
        if not event_data:
            logger.warning("Received model_training_completed event with no data")
            return
            
        model_id = event_data.get("model_id")
        job_id = event_data.get("job_id")
        status = event_data.get("status")
        model_metrics = event_data.get("model_metrics", {})
        
        if not model_id:
            logger.warning(f"Received model_training_completed event without model_id: {event_data}")
            return
            
        logger.info(f"Processing model training completion for model {model_id}, job {job_id}, status: {status}")
        
        # Store outcome in adaptation_outcomes for future analysis
        outcome = {
            "type": "model_training",
            "model_id": model_id,
            "job_id": job_id,
            "status": status,
            "timestamp": event_data.get("completion_time") or datetime.utcnow().isoformat(),
            "metrics": model_metrics,
            "feedback_ids": event_data.get("feedback_ids", [])
        }
        
        # Add to adaptation outcomes
        self.adaptation_outcomes.append(outcome)
        
        # Update specific model performance tracking
        model_key = f"model_{model_id}"
        if model_key not in self.adaptation_effectiveness:
            self.adaptation_effectiveness[model_key] = {
                "training_history": deque(maxlen=10),  # Keep last 10 training jobs
                "current_metrics": {},
                "training_count": 0,
                "success_count": 0,
                "last_updated": None
            }
            
        # Update training history
        model_tracking = self.adaptation_effectiveness[model_key]
        model_tracking["training_history"].append(outcome)
        model_tracking["training_count"] += 1
        
        if status == "success":
            model_tracking["success_count"] += 1
            model_tracking["current_metrics"] = model_metrics
            model_tracking["last_updated"] = datetime.utcnow().isoformat()
            
        # Calculate success rate
        if model_tracking["training_count"] > 0:
            model_tracking["success_rate"] = model_tracking["success_count"] / model_tracking["training_count"]
        
        # Log key metrics if available
        if model_metrics:
            metrics_str = ", ".join(f"{k}: {v}" for k, v in model_metrics.items()
                                   if k in ["accuracy", "precision", "recall", "f1_score", "mae", "rmse"])
            if metrics_str:
                logger.info(f"Updated metrics for model {model_id}: {metrics_str}")
        
        logger.info(f"Successfully processed model training completion for model {model_id}")
        
    async def handle_model_training_failed(self, event: Event) -> None:
        """
        Handles model training failure events, updating adaptation outcomes.
        
        Args:
            event: Model training failed event
        """
        event_data = event.data
        if not event_data:
            logger.warning("Received model_training_failed event with no data")
            return
            
        model_id = event_data.get("model_id")
        job_id = event_data.get("job_id")
        error = event_data.get("error", "Unknown error")
        
        if not model_id:
            logger.warning(f"Received model_training_failed event without model_id: {event_data}")
            return
            
        logger.warning(f"Processing model training failure for model {model_id}, job {job_id}: {error}")
        
        # Store failure in adaptation_outcomes for future analysis
        outcome = {
            "type": "model_training",
            "model_id": model_id,
            "job_id": job_id,
            "status": "failed",
            "error": error,
            "timestamp": event_data.get("failure_time") or datetime.utcnow().isoformat(),
            "feedback_ids": event_data.get("feedback_ids", [])
        }
        
        # Add to adaptation outcomes
        self.adaptation_outcomes.append(outcome)
        
        # Update specific model performance tracking
        model_key = f"model_{model_id}"
        if model_key not in self.adaptation_effectiveness:
            self.adaptation_effectiveness[model_key] = {
                "training_history": deque(maxlen=10),
                "current_metrics": {},
                "training_count": 0,
                "success_count": 0,
                "last_updated": None
            }
            
        # Update training history with failure
        model_tracking = self.adaptation_effectiveness[model_key]
        model_tracking["training_history"].append(outcome)
        model_tracking["training_count"] += 1
        
        # Calculate success rate
        if model_tracking["training_count"] > 0:
            model_tracking["success_rate"] = model_tracking["success_count"] / model_tracking["training_count"]
        
        logger.info(f"Recorded model training failure for model {model_id}")

    async def initialize_event_subscriptions(self, event_subscriber):
        """
        Initialize event subscriptions for the feedback loop.
        This should be called after the feedback loop is created.
        
        Args:
            event_subscriber: Event subscriber to use for subscribing to events
        """
        if not event_subscriber:
            logger.warning("No event subscriber provided, cannot register event handlers")
            return
            
        try:
            # Subscribe to model training completion and failure events
            await event_subscriber.subscribe(
                EventType.MODEL_TRAINING_COMPLETED,
                self.handle_model_training_completed
            )
            
            await event_subscriber.subscribe(
                EventType.MODEL_TRAINING_FAILED,
                self.handle_model_training_failed
            )
            
            logger.info("Successfully registered feedback loop event handlers")
            
        except Exception as e:
            logger.error(f"Failed to register event handlers: {e}", exc_info=True)
