"""
Signal aggregator module.

This module provides functionality for...
"""

class SignalAggregator:
    """
    Aggregates signals from multiple sources and calculates a final signal with confidence level.
    
    This component is responsible for taking signals from various sources (technical analysis,
    machine learning predictions, etc.) and combining them into a final signal with confidence.
    """
    
    def __init__(
        self, 
        adaptive_layer_service: AdaptiveLayerService,
        market_regime_service: MarketRegimeService
    ):
        """
        Initialize the SignalAggregator.
        
        Args:
            adaptive_layer_service: Service for getting adaptive weights based on effectiveness
            market_regime_service: Service for getting the current market regime
        """
        self.adaptive_layer_service = adaptive_layer_service
        self.market_regime_service = market_regime_service
        self.logger = logging.getLogger(__name__)
        
        # Default category weights if adaptive weights are not available
        self.default_category_weights = {
            "technical_analysis": 0.40,
            "machine_learning": 0.30,
            "market_sentiment": 0.15,
            "economic_indicators": 0.10,
            "correlation_signals": 0.05
        }
        
        # Signal cache to avoid recalculating identical signals
        self.signal_cache = {}
        self.signal_cache_expiry = {}
        self.signal_cache_duration = timedelta(minutes=5)
        
        # Track individual tool performance for feedback
        self.tool_signal_history = {}
        
    async def aggregate_signals(
        self, 
        signals: Dict[str, Signal],
        symbol: str,
        timeframe: TimeFrame,
        strategy_id: str
    ) -> AggregatedSignal:
        """
        Aggregate signals from multiple sources into a final signal with confidence.
        
        Args:
            signals: Dictionary of signals from different sources, keyed by signal provider ID
            symbol: The trading instrument
            timeframe: The timeframe for the signals
            strategy_id: The ID of the strategy making the request
            
        Returns:
            An AggregatedSignal with direction and confidence level
        """
        # Check cache first
        cache_key = self._create_cache_key(signals, symbol, timeframe, strategy_id)
        now = datetime.now()
        
        if (cache_key in self.signal_cache and 
            cache_key in self.signal_cache_expiry and
            now < self.signal_cache_expiry[cache_key]):
            self.logger.debug(f"Using cached signal for {cache_key}")
            return self.signal_cache[cache_key]
            
        # Get current market regime for weighting signals appropriately
        market_regime = await self.market_regime_service.get_current_regime(symbol, timeframe)
        self.logger.info(f"Current market regime for {symbol} ({timeframe.value}): {market_regime.value}")
        
        # Group signals by category
        signal_categories = self._group_signals_by_category(signals)
        
        # Get tool IDs for all signals
        tool_ids = [signal_id for signal_id in signals.keys()]
        
        # Get adaptive weights for individual tools based on their effectiveness
        tool_weights = await self.adaptive_layer_service.get_tool_signal_weights(
            market_regime=market_regime,
            tools=tool_ids,
            timeframe=timeframe,
            symbol=symbol
        )
        
        # Get category weights from adaptive layer
        category_weights = await self.adaptive_layer_service.get_aggregator_weights(
            market_regime=market_regime,
            timeframe=timeframe,
            symbol=symbol
        )
        
        # If no adaptive weights available, use defaults
        if not category_weights:
            category_weights = self.default_category_weights
            
        # Calculate weighted scores for each category
        category_signals = {}
        for category, category_signals_dict in signal_categories.items():
            # Skip empty categories
            if not category_signals_dict:
                continue
                
            # Calculate weighted signal for each category using tool-specific weights
            weighted_signal_sum = 0.0
            total_weight = 0.0
            
            for signal_id, signal in category_signals_dict.items():
                # Get weight for this specific tool
                tool_weight = tool_weights.get(signal_id, 1.0)
                
                # Calculate raw signal value (-1.0 to 1.0)
                signal_value = self._signal_direction_to_value(signal.direction) * signal.confidence
                
                # Apply tool-specific weight
                weighted_signal_sum += signal_value * tool_weight
                total_weight += tool_weight
                
            # Calculate category signal
            if total_weight > 0:
                category_signals[category] = weighted_signal_sum / total_weight
            else:
                category_signals[category] = 0.0
                
        # Calculate final weighted signal across categories
        final_signal_value = 0.0
        total_category_weight = 0.0
        
        for category, signal_value in category_signals.items():
            # Get weight for this category
            category_weight = category_weights.get(category, 0.0)
            
            # Apply category weight
            final_signal_value += signal_value * category_weight
            total_category_weight += category_weight
            
        # Normalize final signal
        if total_category_weight > 0:
            final_signal_value = final_signal_value / total_category_weight
        else:
            final_signal_value = 0.0
            
        # Convert to direction and confidence
        direction = self._value_to_signal_direction(final_signal_value)
        confidence = abs(final_signal_value)
        
        # Create the aggregated signal with metadata
        aggregated_signal = AggregatedSignal(
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(),
            market_regime=market_regime,
            timeframe=timeframe,
            symbol=symbol,
            component_signals=signals,
            tool_weights=tool_weights,
            category_weights=category_weights
        )
        
        # Cache the result
        self.signal_cache[cache_key] = aggregated_signal
        self.signal_cache_expiry[cache_key] = now + self.signal_cache_duration
        
        self.logger.info(
            f"Aggregated signal for {symbol} ({timeframe.value}): "
            f"{direction.value} with {confidence:.2f} confidence"
        )
        
        return aggregated_signal
        
    def _create_cache_key(
        self, 
        signals: Dict[str, Signal],
        symbol: str,
        timeframe: TimeFrame,
        strategy_id: str
    ) -> str:
        """Create a unique cache key for the signal set"""
        # Create a consistent representation of signals for the cache key
        signal_repr = []
        for signal_id in sorted(signals.keys()):
            signal = signals[signal_id]
            signal_repr.append(f"{signal_id}:{signal.direction.value}:{signal.confidence:.4f}")
        
        return f"{symbol}:{timeframe.value}:{strategy_id}:{','.join(signal_repr)}"
        
    def _group_signals_by_category(
        self, 
        signals: Dict[str, Signal]
    ) -> Dict[str, Dict[str, Signal]]:
        """Group signals by their category"""
        categories = {
            "technical_analysis": {},
            "machine_learning": {},
            "market_sentiment": {},
            "economic_indicators": {},
            "correlation_signals": {}
        }
        
        for signal_id, signal in signals.items():
            category = self._determine_signal_category(signal_id)
            categories[category][signal_id] = signal
            
        return categories
        
    def _determine_signal_category(self, signal_id: str) -> str:
        """Determine the category of a signal based on its ID"""
        signal_id_lower = signal_id.lower()
        
        if any(ta in signal_id_lower for ta in ["ma", "rsi", "macd", "fibonacci", "elliott", "pivot", "harmonic", "pattern"]):
            return "technical_analysis"
        elif any(ml in signal_id_lower for ml in ["ml_", "model", "prediction", "forecast", "lstm", "transformer"]):
            return "machine_learning"
        elif any(sentiment in signal_id_lower for sentiment in ["sentiment", "news", "social", "opinion"]):
            return "market_sentiment"
        elif any(econ in signal_id_lower for econ in ["econ", "interest", "inflation", "gdp", "unemployment"]):
            return "economic_indicators"
        elif any(corr in signal_id_lower for corr in ["correlation", "pair", "basket", "index"]):
            return "correlation_signals"
        else:
            # Default to technical analysis
            return "technical_analysis"
            
    def _signal_direction_to_value(self, direction: SignalDirection) -> float:
        """Convert signal direction to numeric value"""
        mapping = {
            SignalDirection.STRONG_BUY: 1.0,
            SignalDirection.BUY: 0.5,
            SignalDirection.NEUTRAL: 0.0,
            SignalDirection.SELL: -0.5,
            SignalDirection.STRONG_SELL: -1.0
        }
        return mapping.get(direction, 0.0)
        
    def _value_to_signal_direction(self, value: float) -> SignalDirection:
        """Convert numeric value to signal direction"""
        if value > 0.75:
            return SignalDirection.STRONG_BUY
        elif value > 0.25:
            return SignalDirection.BUY
        elif value > -0.25:
            return SignalDirection.NEUTRAL
        elif value > -0.75:
            return SignalDirection.SELL
        else:
            return SignalDirection.STRONG_SELL
            
    async def record_signal_outcome(
        self,
        aggregated_signal: AggregatedSignal,
        outcome: bool,
        profit_loss: float,
        update_effectiveness: bool = True
    ) -> None:
        """
        Record the outcome of a signal for learning and adaptation
        
        Args:
            aggregated_signal: The aggregated signal that was used
            outcome: Whether the signal led to a profitable trade
            profit_loss: The profit or loss from the trade
            update_effectiveness: Whether to update tool effectiveness metrics
        """
        # Record signal outcome for each component signal
        for signal_id, signal in aggregated_signal.component_signals.items():
            # Store in history
            if signal_id not in self.tool_signal_history:
                self.tool_signal_history[signal_id] = []
                
            self.tool_signal_history[signal_id].append({
                "timestamp": datetime.now(),
                "direction": signal.direction,
                "confidence": signal.confidence,
                "outcome": outcome,
                "profit_loss": profit_loss,
                "market_regime": aggregated_signal.market_regime.value,
                "timeframe": aggregated_signal.timeframe.value,
                "symbol": aggregated_signal.symbol
            })
            
            # Update weight adjustment factor based on very recent performance
            if update_effectiveness:
                await self.adaptive_layer_service.update_weight_adjustment_factor(
                    tool_id=signal_id,
                    market_regime=aggregated_signal.market_regime,
                    success=outcome,
                    impact=0.03  # Small adjustment to avoid overreaction
                )
        
        self.logger.info(
            f"Recorded signal outcome: {'Success' if outcome else 'Failure'} "
            f"with P/L: {profit_loss:.2f} for {len(aggregated_signal.component_signals)} tools"
        )