"""
Enhanced Strategy Factory

This module provides a factory for creating enhanced trading strategies
with all the strategy enhancement services integrated.
"""

from typing import Dict, List, Any, Optional, Type, Union
import logging

from strategy_execution_engine.strategies.advanced_ta_strategy import AdvancedTAStrategy
from strategy_execution_engine.strategies.gann_strategy import GannTradingStrategy
from strategy_execution_engine.strategies.volatility_breakout_strategy import VolatilityBreakoutStrategy

from analysis_engine.services.timeframe_optimization_service import TimeframeOptimizationService
from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceDetector
from analysis_engine.analysis.sequence_pattern_recognizer import SequencePatternRecognizer
from analysis_engine.services.regime_transition_predictor import RegimeTransitionPredictor


class EnhancedStrategyFactory:
    """
    Factory for creating enhanced trading strategies.
    
    This factory:
    - Creates strategies with all enhancement services integrated
    - Provides configuration options for each enhancement
    - Supports different strategy types
    """
    
    def __init__(self):
        """Initialize the factory."""
        self.logger = logging.getLogger(__name__)
        
        # Available strategy types
        self.strategy_types = {
            "gann": GannTradingStrategy,
            "volatility_breakout": VolatilityBreakoutStrategy
        }
    
    def create_strategy(
        self,
        strategy_type: str,
        name: str,
        timeframes: List[str],
        primary_timeframe: str,
        symbols: List[str],
        risk_per_trade_pct: float = 1.0,
        enhancement_config: Optional[Dict[str, Any]] = None,
        **strategy_kwargs
    ) -> AdvancedTAStrategy:
        """
        Create an enhanced trading strategy.
        
        Args:
            strategy_type: Type of strategy to create ("gann", "volatility_breakout", etc.)
            name: Name of the strategy
            timeframes: List of timeframes to use
            primary_timeframe: Primary timeframe for analysis
            symbols: List of symbols to trade
            risk_per_trade_pct: Risk per trade as a percentage
            enhancement_config: Configuration for enhancement services
            **strategy_kwargs: Additional strategy-specific arguments
            
        Returns:
            Enhanced trading strategy instance
        """
        # Get strategy class
        if strategy_type not in self.strategy_types:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
            
        strategy_class = self.strategy_types[strategy_type]
        
        # Create enhancement services
        enhancement_services = self._create_enhancement_services(
            timeframes=timeframes,
            primary_timeframe=primary_timeframe,
            config=enhancement_config or {}
        )
        
        # Create strategy instance
        strategy = strategy_class(
            name=name,
            timeframes=timeframes,
            primary_timeframe=primary_timeframe,
            symbols=symbols,
            risk_per_trade_pct=risk_per_trade_pct,
            **enhancement_services,
            **strategy_kwargs
        )
        
        # Configure enhancements
        self._configure_strategy_enhancements(strategy, enhancement_config or {})
        
        self.logger.info(f"Created enhanced {strategy_type} strategy: {name}")
        return strategy
    
    def _create_enhancement_services(
        self,
        timeframes: List[str],
        primary_timeframe: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create enhancement service instances.
        
        Args:
            timeframes: List of timeframes to use
            primary_timeframe: Primary timeframe for analysis
            config: Enhancement configuration
            
        Returns:
            Dictionary of enhancement service instances
        """
        services = {}
        
        # Create TimeframeOptimizationService
        if config.get("use_timeframe_optimization", True):
            timeframe_optimizer = TimeframeOptimizationService(
                timeframes=timeframes,
                primary_timeframe=primary_timeframe,
                lookback_days=config.get("timeframe_lookback_days", 30),
                min_signals_required=config.get("min_signals_required", 10),
                weight_decay_factor=config.get("weight_decay_factor", 0.9),
                max_weight=config.get("max_weight", 2.0),
                min_weight=config.get("min_weight", 0.5)
            )
            services["timeframe_optimizer"] = timeframe_optimizer
        
        # Create CurrencyStrengthAnalyzer
        if config.get("use_currency_strength", True):
            currency_strength_analyzer = CurrencyStrengthAnalyzer(
                base_currencies=config.get("base_currencies"),
                quote_currencies=config.get("quote_currencies"),
                lookback_periods=config.get("currency_lookback_periods", 20)
            )
            services["currency_strength_analyzer"] = currency_strength_analyzer
        
        # Create RelatedPairsConfluenceDetector
        if config.get("use_related_pairs_confluence", True):
            related_pairs_detector = RelatedPairsConfluenceDetector(
                correlation_service=None,  # Will be set by the strategy if available
                currency_strength_analyzer=services.get("currency_strength_analyzer"),
                correlation_threshold=config.get("correlation_threshold", 0.7),
                lookback_periods=config.get("correlation_lookback_periods", 20)
            )
            services["related_pairs_detector"] = related_pairs_detector
        
        # Create SequencePatternRecognizer
        if config.get("use_sequence_patterns", True):
            pattern_recognizer = SequencePatternRecognizer(
                timeframe_mapping=config.get("timeframe_mapping"),
                min_pattern_quality=config.get("min_pattern_quality", 0.7),
                use_ml_validation=config.get("use_ml_validation", False),
                pattern_types=config.get("pattern_types")
            )
            services["pattern_recognizer"] = pattern_recognizer
        
        # Create RegimeTransitionPredictor
        if config.get("use_regime_transition_prediction", True):
            regime_transition_predictor = RegimeTransitionPredictor(
                regime_detector=None,  # Will be set by the strategy if available
                transition_history_size=config.get("transition_history_size", 100),
                early_warning_threshold=config.get("early_warning_threshold", 0.7),
                lookback_periods=config.get("regime_lookback_periods", 50)
            )
            services["regime_transition_predictor"] = regime_transition_predictor
        
        return services
    
    def _configure_strategy_enhancements(
        self,
        strategy: AdvancedTAStrategy,
        config: Dict[str, Any]
    ) -> None:
        """
        Configure enhancement settings in the strategy.
        
        Args:
            strategy: Strategy instance to configure
            config: Enhancement configuration
        """
        # Update strategy config with enhancement settings
        enhancement_config = {
            # Timeframe optimization settings
            "use_timeframe_optimization": config.get("use_timeframe_optimization", True),
            "optimize_timeframes_interval_hours": config.get("optimize_timeframes_interval_hours", 24),
            
            # Currency strength settings
            "use_currency_strength": config.get("use_currency_strength", True),
            
            # Related pairs confluence settings
            "use_related_pairs_confluence": config.get("use_related_pairs_confluence", True),
            "min_related_pairs_confluence": config.get("min_related_pairs_confluence", 0.6),
            
            # Sequence pattern settings
            "use_sequence_patterns": config.get("use_sequence_patterns", True),
            "min_pattern_confidence": config.get("min_pattern_confidence", 0.7),
            
            # Regime transition settings
            "use_regime_transition_prediction": config.get("use_regime_transition_prediction", True),
            "regime_transition_threshold": config.get("regime_transition_threshold", 0.7)
        }
        
        # Update strategy config
        strategy.config.update(enhancement_config)
    
    def load_timeframe_optimizer_state(
        self,
        strategy: AdvancedTAStrategy,
        filepath: str
    ) -> bool:
        """
        Load timeframe optimizer state from a file.
        
        Args:
            strategy: Strategy instance
            filepath: Path to the state file
            
        Returns:
            Success status
        """
        if not hasattr(strategy, "timeframe_optimizer") or strategy.timeframe_optimizer is None:
            self.logger.warning("Strategy does not have a timeframe optimizer")
            return False
        
        try:
            # Load optimizer state
            optimizer = TimeframeOptimizationService.load_from_file(
                filepath,
                timeframes=strategy.timeframes,
                primary_timeframe=strategy.primary_timeframe
            )
            
            # Replace the strategy's optimizer
            strategy.timeframe_optimizer = optimizer
            
            self.logger.info(f"Loaded timeframe optimizer state from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load timeframe optimizer state: {e}")
            return False
    
    def save_timeframe_optimizer_state(
        self,
        strategy: AdvancedTAStrategy,
        filepath: str
    ) -> bool:
        """
        Save timeframe optimizer state to a file.
        
        Args:
            strategy: Strategy instance
            filepath: Path to save the state
            
        Returns:
            Success status
        """
        if not hasattr(strategy, "timeframe_optimizer") or strategy.timeframe_optimizer is None:
            self.logger.warning("Strategy does not have a timeframe optimizer")
            return False
        
        try:
            # Save optimizer state
            success = strategy.timeframe_optimizer.save_to_file(filepath)
            
            if success:
                self.logger.info(f"Saved timeframe optimizer state to {filepath}")
            else:
                self.logger.warning(f"Failed to save timeframe optimizer state")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to save timeframe optimizer state: {e}")
            return False
