"""
Pytest configuration for Strategy Execution Engine tests.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, AsyncMock

# Add mock modules for dependencies
sys.modules['analysis_engine'] = MagicMock()
sys.modules['analysis_engine.adaptive_layer'] = MagicMock()
sys.modules['analysis_engine.adaptive_layer.confluence_analyzer'] = MagicMock()
sys.modules['analysis_engine.adaptive_layer.confluence_analyzer'].ConfluenceAnalyzer = MagicMock()
sys.modules['analysis_engine.services'] = MagicMock()
sys.modules['analysis_engine.services.tool_effectiveness'] = MagicMock()
sys.modules['analysis_engine.services.tool_effectiveness'].MarketRegime = MagicMock()
sys.modules['analysis_engine.analysis'] = MagicMock()
sys.modules['analysis_engine.analysis.technical_indicators'] = MagicMock()
sys.modules['analysis_engine.analysis.technical_indicators'].TechnicalIndicators = MagicMock()
sys.modules['analysis_engine.analysis.volatility_analysis'] = MagicMock()
sys.modules['analysis_engine.analysis.volatility_analysis'].VolatilityAnalyzer = MagicMock()
sys.modules['analysis_engine.learning_from_mistakes'] = MagicMock()
sys.modules['analysis_engine.learning_from_mistakes.ma_optimization'] = MagicMock()
sys.modules['analysis_engine.learning_from_mistakes.ma_optimization'].MAOptimizationEngine = MagicMock()

# Add mock modules for other dependencies
sys.modules['core_foundations'] = MagicMock()
sys.modules['core_foundations.utils'] = MagicMock()
sys.modules['core_foundations.utils.logger'] = MagicMock()
sys.modules['core_foundations.utils.logger'].get_logger = MagicMock(return_value=MagicMock())

# Add mock modules for common_lib
sys.modules['common_lib'] = MagicMock()
sys.modules['common_lib.effectiveness'] = MagicMock()
sys.modules['common_lib.effectiveness.interfaces'] = MagicMock()
sys.modules['common_lib.effectiveness.interfaces'].MarketRegimeEnum = MagicMock()
sys.modules['common_lib.strategy'] = MagicMock()
sys.modules['common_lib.strategy.interfaces'] = MagicMock()
sys.modules['common_lib.strategy.interfaces'].IStrategyExecutor = MagicMock()
sys.modules['common_lib.strategy.interfaces'].ISignalAggregator = MagicMock()
sys.modules['common_lib.strategy.interfaces'].IStrategyEvaluator = MagicMock()
sys.modules['common_lib.strategy.interfaces'].SignalDirection = MagicMock()
sys.modules['common_lib.strategy.interfaces'].SignalTimeframe = MagicMock()
sys.modules['common_lib.strategy.interfaces'].SignalSource = MagicMock()
sys.modules['common_lib.strategy.interfaces'].MarketRegimeType = MagicMock()

# Set environment variables for testing
os.environ['DEBUG_MODE'] = 'true'
os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['HOST'] = 'localhost'
os.environ['PORT'] = '8003'
os.environ['API_KEY'] = 'test_api_key'
os.environ['SERVICE_API_KEY'] = 'test_service_api_key'
os.environ['ANALYSIS_ENGINE_URL'] = 'http://localhost:8002'
os.environ['FEATURE_STORE_URL'] = 'http://localhost:8001'
os.environ['TRADING_GATEWAY_URL'] = 'http://localhost:8004'
os.environ['ANALYSIS_ENGINE_KEY'] = 'test_analysis_engine_key'
os.environ['FEATURE_STORE_KEY'] = 'test_feature_store_key'
os.environ['TRADING_GATEWAY_KEY'] = 'test_trading_gateway_key'
