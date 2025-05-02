"""
Enhanced Strategy Example

This example demonstrates how to create and use enhanced trading strategies
with all the strategy enhancement services.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy_execution_engine.factory.enhanced_strategy_factory import EnhancedStrategyFactory
from analysis_engine.visualization.strategy_enhancement_dashboard import StrategyEnhancementDashboard
from analysis_engine.services.timeframe_optimization_service import SignalOutcome


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Load sample price data for testing.
    
    Returns:
        Dictionary mapping symbols and timeframes to price DataFrames
    """
    # Create sample data
    price_data = {}
    
    # Create data for EUR/USD
    for tf in ["1m", "5m", "15m", "1h", "4h"]:
        # Create base data
        length = 100
        df = pd.DataFrame({
            'open': np.random.normal(1.10, 0.01, length),
            'high': np.random.normal(1.11, 0.01, length),
            'low': np.random.normal(1.09, 0.01, length),
            'close': np.random.normal(1.10, 0.01, length),
            'volume': np.random.randint(1000, 5000, length)
        })
        
        # Add trend for 1h and 4h timeframes
        if tf in ["1h", "4h"]:
            df['close'] = np.linspace(1.10, 1.15, length)  # Bullish trend
        
        # Add technical indicators
        df['short_ma'] = df['close'].rolling(window=10).mean()
        df['medium_ma'] = df['close'].rolling(window=20).mean()
        df['ATR'] = (df['high'] - df['low']).rolling(window=14).mean()
        
        price_data[f"EUR/USD_{tf}"] = df
    
    # Create data for GBP/USD
    for tf in ["1m", "5m", "15m", "1h", "4h"]:
        # Create base data
        length = 100
        df = pd.DataFrame({
            'open': np.random.normal(1.30, 0.01, length),
            'high': np.random.normal(1.31, 0.01, length),
            'low': np.random.normal(1.29, 0.01, length),
            'close': np.random.normal(1.30, 0.01, length),
            'volume': np.random.randint(1000, 5000, length)
        })
        
        # Add trend for 1h and 4h timeframes
        if tf in ["1h", "4h"]:
            df['close'] = np.linspace(1.30, 1.35, length)  # Bullish trend
        
        # Add technical indicators
        df['short_ma'] = df['close'].rolling(window=10).mean()
        df['medium_ma'] = df['close'].rolling(window=20).mean()
        df['ATR'] = (df['high'] - df['low']).rolling(window=14).mean()
        
        price_data[f"GBP/USD_{tf}"] = df
    
    return price_data


def prepare_data_for_strategy(price_data: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, pd.DataFrame]:
    """
    Prepare price data for a specific symbol.
    
    Args:
        price_data: Dictionary mapping symbols and timeframes to price DataFrames
        symbol: Symbol to prepare data for
        
    Returns:
        Dictionary mapping timeframes to price DataFrames
    """
    symbol_data = {}
    for key, df in price_data.items():
        if key.startswith(f"{symbol}_"):
            timeframe = key.split('_')[1]
            symbol_data[timeframe] = df
    return symbol_data


def main():
    """Run the enhanced strategy example."""
    logger.info("Starting enhanced strategy example")
    
    # Create enhanced strategy factory
    factory = EnhancedStrategyFactory()
    
    # Define enhancement configuration
    enhancement_config = {
        "use_timeframe_optimization": True,
        "timeframe_lookback_days": 30,
        "min_signals_required": 10,
        "weight_decay_factor": 0.9,
        "max_weight": 2.0,
        "min_weight": 0.5,
        
        "use_currency_strength": True,
        "currency_lookback_periods": 20,
        
        "use_related_pairs_confluence": True,
        "correlation_threshold": 0.7,
        "min_related_pairs_confluence": 0.6,
        
        "use_sequence_patterns": True,
        "min_pattern_quality": 0.7,
        "min_pattern_confidence": 0.7,
        
        "use_regime_transition_prediction": True,
        "early_warning_threshold": 0.7,
        "regime_transition_threshold": 0.7
    }
    
    # Create enhanced Gann strategy
    gann_strategy = factory.create_strategy(
        strategy_type="gann",
        name="Enhanced Gann Strategy",
        timeframes=["1m", "5m", "15m", "1h", "4h"],
        primary_timeframe="1h",
        symbols=["EUR/USD", "GBP/USD"],
        risk_per_trade_pct=1.0,
        enhancement_config=enhancement_config
    )
    
    logger.info(f"Created enhanced Gann strategy: {gann_strategy.name}")
    
    # Create enhanced Volatility Breakout strategy
    volatility_strategy = factory.create_strategy(
        strategy_type="volatility_breakout",
        name="Enhanced Volatility Breakout Strategy",
        timeframes=["1m", "5m", "15m", "1h", "4h"],
        primary_timeframe="1h",
        symbols=["EUR/USD", "GBP/USD"],
        risk_per_trade_pct=1.0,
        enhancement_config=enhancement_config
    )
    
    logger.info(f"Created enhanced Volatility Breakout strategy: {volatility_strategy.name}")
    
    # Load sample data
    price_data = load_sample_data()
    logger.info(f"Loaded sample data for {len(price_data)} symbol-timeframe combinations")
    
    # Add some performance history to the timeframe optimizer
    add_sample_performance_data(gann_strategy)
    add_sample_performance_data(volatility_strategy)
    
    # Analyze market with Gann strategy
    symbol = "EUR/USD"
    symbol_data = prepare_data_for_strategy(price_data, symbol)
    
    logger.info(f"Analyzing market for {symbol} with Gann strategy")
    gann_analysis = gann_strategy.analyze_market(symbol, symbol_data)
    
    logger.info(f"Gann strategy analysis results:")
    logger.info(f"Direction: {gann_analysis.get('direction')}")
    logger.info(f"Signal strength: {gann_analysis.get('signal_strength')}")
    
    # Generate signals with Gann strategy
    gann_signals = gann_strategy.generate_signals(symbol, symbol_data, gann_analysis)
    
    if gann_signals:
        logger.info(f"Generated {len(gann_signals)} signals with Gann strategy")
        for i, signal in enumerate(gann_signals):
            logger.info(f"Signal {i+1}:")
            logger.info(f"  Direction: {signal.get('direction')}")
            logger.info(f"  Confidence: {signal.get('confidence')}")
            logger.info(f"  Entry price: {signal.get('entry_price')}")
            logger.info(f"  Stop loss: {signal.get('stop_loss')}")
            logger.info(f"  Take profit: {signal.get('take_profit')}")
            
            # Check for enhancement data
            if "currency_strength_diff" in signal:
                logger.info(f"  Currency strength diff: {signal.get('currency_strength_diff')}")
            
            if "related_pairs_confluence" in signal:
                logger.info(f"  Related pairs confluence: {signal.get('related_pairs_confluence')}")
            
            if "pattern_type" in signal:
                logger.info(f"  Pattern type: {signal.get('pattern_type')}")
                logger.info(f"  Pattern confidence: {signal.get('pattern_confidence')}")
            
            if "predicted_regime_transition" in signal:
                pred = signal.get("predicted_regime_transition", {})
                logger.info(f"  Predicted regime transition: {pred.get('current_regime')} -> {pred.get('next_regime')} ({pred.get('probability'):.2f})")
    else:
        logger.info("No signals generated with Gann strategy")
    
    # Analyze market with Volatility Breakout strategy
    logger.info(f"Analyzing market for {symbol} with Volatility Breakout strategy")
    volatility_analysis = volatility_strategy.analyze_market(symbol, symbol_data)
    
    logger.info(f"Volatility Breakout strategy analysis results:")
    logger.info(f"Direction: {volatility_analysis.get('direction')}")
    logger.info(f"Volatility ratio: {volatility_analysis.get('volatility_ratio')}")
    
    # Generate signals with Volatility Breakout strategy
    volatility_signals = volatility_strategy.generate_signals(symbol, symbol_data, volatility_analysis)
    
    if volatility_signals:
        logger.info(f"Generated {len(volatility_signals)} signals with Volatility Breakout strategy")
        for i, signal in enumerate(volatility_signals):
            logger.info(f"Signal {i+1}:")
            logger.info(f"  Direction: {signal.get('direction')}")
            logger.info(f"  Confidence: {signal.get('confidence')}")
            logger.info(f"  Entry price: {signal.get('entry_price')}")
            logger.info(f"  Stop loss: {signal.get('stop_loss')}")
            logger.info(f"  Take profit: {signal.get('take_profit')}")
    else:
        logger.info("No signals generated with Volatility Breakout strategy")
    
    # Create dashboard
    logger.info("Creating strategy enhancement dashboard")
    dashboard = StrategyEnhancementDashboard(
        timeframe_optimizer=gann_strategy.timeframe_optimizer,
        currency_strength_analyzer=gann_strategy.currency_strength_analyzer,
        pattern_recognizer=gann_strategy.pattern_recognizer,
        regime_transition_predictor=gann_strategy.regime_transition_predictor,
        output_dir="dashboard_output"
    )
    
    # Generate dashboard
    dashboard_data = dashboard.generate_dashboard(save_to_file=True)
    logger.info(f"Generated dashboard with {len(dashboard_data)} sections")
    
    # Generate performance report
    performance_data = {
        "symbols": ["EUR/USD", "GBP/USD"],
        "performance_metrics": {
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "avg_win_pips": 15.5,
            "avg_loss_pips": -8.2,
            "total_trades": 100,
            "win_count": 65,
            "loss_count": 35
        }
    }
    
    report_path = dashboard.save_performance_report("Enhanced Gann Strategy", performance_data)
    logger.info(f"Generated performance report: {report_path}")
    
    # Save timeframe optimizer state
    state_path = "gann_timeframe_optimizer_state.json"
    factory.save_timeframe_optimizer_state(gann_strategy, state_path)
    logger.info(f"Saved timeframe optimizer state to {state_path}")
    
    logger.info("Enhanced strategy example completed")


def add_sample_performance_data(strategy):
    """
    Add sample performance data to the strategy's timeframe optimizer.
    
    Args:
        strategy: Strategy instance
    """
    if not hasattr(strategy, "timeframe_optimizer") or strategy.timeframe_optimizer is None:
        return
    
    # Add winning signals for 1h timeframe
    for i in range(20):
        strategy.timeframe_optimizer.record_timeframe_performance(
            timeframe="1h",
            outcome=SignalOutcome.WIN,
            symbol="EUR/USD",
            pips_result=10.0,
            confidence=0.8
        )
    
    # Add some losing signals for 1h timeframe
    for i in range(10):
        strategy.timeframe_optimizer.record_timeframe_performance(
            timeframe="1h",
            outcome=SignalOutcome.LOSS,
            symbol="EUR/USD",
            pips_result=-5.0,
            confidence=0.7
        )
    
    # Add winning signals for 4h timeframe (higher win rate)
    for i in range(25):
        strategy.timeframe_optimizer.record_timeframe_performance(
            timeframe="4h",
            outcome=SignalOutcome.WIN,
            symbol="EUR/USD",
            pips_result=15.0,
            confidence=0.8
        )
    
    # Add some losing signals for 4h timeframe
    for i in range(5):
        strategy.timeframe_optimizer.record_timeframe_performance(
            timeframe="4h",
            outcome=SignalOutcome.LOSS,
            symbol="EUR/USD",
            pips_result=-7.0,
            confidence=0.7
        )


if __name__ == "__main__":
    main()
