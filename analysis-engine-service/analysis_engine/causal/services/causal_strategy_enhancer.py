"""
Causal Strategy Enhancement Module

This module provides integration between trading strategies and the causal inference
capabilities, allowing strategies to leverage causal insights for improved decision-making.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import networkx as nx
import asyncio
from datetime import datetime, timedelta

from strategy_execution_engine.strategies.base_strategy import BaseStrategy
from analysis_engine.causal.services.causal_inference_service import CausalInferenceService
from analysis_engine.causal.services.causal_data_connector import CausalDataConnector

logger = logging.getLogger(__name__)


class CausalStrategyEnhancer:
    """
    Enhances trading strategies with causal inference capabilities.
    
    This class provides methods to integrate causal discovery, effect estimation,
    and counterfactual analysis into trading strategies to improve decision-making.
    """
    
    def __init__(self, 
                 data_connector: CausalDataConnector, 
                 config: Dict[str, Any] = None):
        """
        Initialize the causal strategy enhancer.
        
        Args:
            data_connector: Connector for fetching causal data.
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Initialize services
        self.causal_inference = CausalInferenceService(config.get("causal_inference", {})) # Keep if needed
        self.data_connector = data_connector # Use injected connector
        
        # Trading parameters
        self.default_timeframe = self.config.get("default_timeframe", "1h")
        self.default_window = timedelta(days=self.config.get("default_window_days", 30))
        self.update_interval = self.config.get("update_interval_minutes", 60)
        
        # Cache for strategy-specific causal graphs
        self.strategy_causal_graphs = {}
        
    async def enhance_strategy(self, strategy: BaseStrategy) -> BaseStrategy:
        """
        Enhance a trading strategy with causal inference capabilities.
        
        Args:
            strategy: Trading strategy to enhance
            
        Returns:
            Enhanced strategy with causal capabilities
        """
        # Add causal inference capabilities to strategy metadata
        strategy.metadata["has_causal_enhancement"] = True
        strategy.metadata["causal_last_update"] = datetime.now()
        
        # Get strategy parameters related to causal analysis
        timeframe = strategy.parameters.get("timeframe", self.default_timeframe)
        symbols = strategy.parameters.get("symbols", [])
        
        if not symbols:
            logger.warning(f"No symbols defined for strategy {strategy.name}")
            return strategy
            
        # Ensure the strategy has methods for causal decision making
        self._attach_causal_methods(strategy)
        
        # Initialize causal analysis for the strategy
        await self._initialize_causal_analysis(strategy, symbols, timeframe)
        
        return strategy
    
    def _attach_causal_methods(self, strategy: BaseStrategy) -> None:
        """
        Attach causal inference methods to a strategy instance.
        
        Args:
            strategy: Strategy to enhance with causal methods
        """
        # Attach causal graph access method
        strategy.get_causal_graph = lambda: self.strategy_causal_graphs.get(strategy.name)
        
        # Attach counterfactual analysis method
        strategy.generate_counterfactual = lambda data, target, interventions: (
            self.causal_inference.generate_counterfactuals(data, target, interventions)
        )
        
        # Attach causal effect estimation method
        strategy.estimate_causal_effect = lambda data, treatment, outcome: (
            self.causal_inference.estimate_causal_effect(data, treatment, outcome)
        )
        
    async def _initialize_causal_analysis(self, 
                                     strategy: BaseStrategy, 
                                     symbols: List[str],
                                     timeframe: str) -> None:
        """
        Initialize causal analysis for a strategy.
        
        Args:
            strategy: Trading strategy to initialize
            symbols: Currency pairs to analyze
            timeframe: Data timeframe to use
        """
        try:
            # Get historical data for causal analysis
            historical_data = await self.data_connector.get_historical_data(
                symbols=symbols,
                start_date=datetime.now() - self.default_window,
                timeframe=timeframe,
                include_indicators=True
            )
            
            if historical_data.empty:
                logger.error(f"Failed to retrieve historical data for strategy {strategy.name}")
                return
            
            # Prepare data for causal analysis
            prepared_data = await self.data_connector.prepare_data_for_causal_analysis(
                historical_data
            )
            
            # Discover causal structure
            causal_graph = self.causal_inference.discover_causal_structure(
                prepared_data,
                method="granger",
                cache_key=f"{strategy.name}_{','.join(sorted(symbols))}_{timeframe}"
            )
            
            # Store causal graph for the strategy
            self.strategy_causal_graphs[strategy.name] = causal_graph
            
            logger.info(f"Initialized causal analysis for strategy {strategy.name}")
            
        except Exception as e:
            logger.error(f"Error initializing causal analysis for strategy {strategy.name}: {str(e)}")
            
    async def start_real_time_causal_updates(self, 
                                        strategy: BaseStrategy,
                                        update_interval: Optional[int] = None) -> str:
        """
        Start real-time causal analysis updates for a strategy.
        
        Args:
            strategy: Trading strategy to update
            update_interval: Update interval in minutes (overrides default)
            
        Returns:
            Stream ID for the real-time updates
        """
        # Get strategy parameters
        symbols = strategy.parameters.get("symbols", [])
        timeframe = strategy.parameters.get("timeframe", self.default_timeframe)
        interval = update_interval or self.update_interval
        
        if not symbols:
            raise ValueError(f"No symbols defined for strategy {strategy.name}")
            
        # Define the callback for real-time updates
        def update_causal_analysis(data: pd.DataFrame):
            try:
                # Prepare data for causal analysis
                loop = asyncio.get_event_loop()
                prepared_data = loop.run_until_complete(
                    self.data_connector.prepare_data_for_causal_analysis(data)
                )
                
                # Update causal graph
                updated_graph = self.causal_inference.discover_causal_structure(
                    prepared_data,
                    method="granger",
                    force_refresh=True,
                    cache_key=f"{strategy.name}_{','.join(sorted(symbols))}_{timeframe}"
                )
                
                # Store updated causal graph
                self.strategy_causal_graphs[strategy.name] = updated_graph
                
                # Update strategy metadata
                strategy.metadata["causal_last_update"] = datetime.now()
                
                logger.info(f"Updated causal analysis for strategy {strategy.name}")
                
            except Exception as e:
                logger.error(f"Error updating causal analysis for {strategy.name}: {str(e)}")
                
        # Start streaming data with the update callback
        stream_id = await self.data_connector.start_streaming(
            symbols=symbols,
            callback=update_causal_analysis,
            interval=interval * 60,  # Convert minutes to seconds
            timeframe=timeframe,
            window_size=self.default_window
        )
        
        # Store stream ID in strategy metadata
        strategy.metadata["causal_stream_id"] = stream_id
        
        return stream_id
    
    async def stop_real_time_updates(self, strategy: BaseStrategy) -> bool:
        """
        Stop real-time causal analysis updates for a strategy.
        
        Args:
            strategy: Trading strategy to stop updates for
            
        Returns:
            True if updates were successfully stopped, False otherwise
        """
        stream_id = strategy.metadata.get("causal_stream_id")
        if not stream_id:
            logger.warning(f"No active causal updates found for strategy {strategy.name}")
            return False
            
        success = await self.data_connector.stop_streaming(stream_id)
        
        if success:
            # Remove stream ID from strategy metadata
            if "causal_stream_id" in strategy.metadata:
                del strategy.metadata["causal_stream_id"]
                
        return success
