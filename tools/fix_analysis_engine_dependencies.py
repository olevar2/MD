#!/usr/bin/env python3
"""
Fix Analysis Engine Dependencies

This script reduces the dependencies of the analysis-engine-service by implementing
interface-based adapters for service communication.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple

# Configure paths
PROJECT_ROOT = Path("D:/MD/forex_trading_platform")
ANALYSIS_ENGINE_DIR = PROJECT_ROOT / "analysis-engine-service"
COMMON_LIB_DIR = PROJECT_ROOT / "common-lib"

def load_dependency_report() -> Dict[str, Any]:
    """Load the dependency report."""
    report_path = PROJECT_ROOT / "tools" / "output" / "dependency-report.json"
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dependency report: {e}", flush=True)
        sys.exit(1)

def analyze_dependencies(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Analyze dependencies of analysis-engine-service.
    
    Args:
        data: Dependency report data
        
    Returns:
        Dictionary of dependencies by type
    """
    print("Analyzing analysis-engine-service dependencies...", flush=True)
    
    service_dependencies = data.get('service_dependencies', {})
    analysis_engine_deps = service_dependencies.get('analysis-engine-service', [])
    
    print(f"analysis-engine-service depends on {len(analysis_engine_deps)} services:", flush=True)
    for dep in analysis_engine_deps:
        print(f"  - {dep}", flush=True)
    
    # Categorize dependencies
    dependency_types = {
        "data_services": [],
        "ml_services": [],
        "trading_services": [],
        "other_services": []
    }
    
    for dep in analysis_engine_deps:
        if "data" in dep or "feature" in dep:
            dependency_types["data_services"].append(dep)
        elif "ml" in dep or "model" in dep:
            dependency_types["ml_services"].append(dep)
        elif "trading" in dep or "risk" in dep:
            dependency_types["trading_services"].append(dep)
        else:
            dependency_types["other_services"].append(dep)
    
    print("Dependencies by type:", flush=True)
    for dep_type, deps in dependency_types.items():
        print(f"  {dep_type}: {deps}", flush=True)
    
    return dependency_types

def create_adapter_interfaces():
    """Create adapter interfaces in common-lib."""
    print("Creating adapter interfaces in common-lib...", flush=True)
    
    # Create directory for interfaces if it doesn't exist
    interfaces_dir = COMMON_LIB_DIR / "common_lib" / "interfaces"
    interfaces_dir.mkdir(exist_ok=True)
    
    # Create __init__.py if it doesn't exist
    init_file = interfaces_dir / "__init__.py"
    if not init_file.exists():
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write('"""Interfaces package for common library."""\n')
    
    # Create data service interfaces
    data_interfaces_file = interfaces_dir / "data_interfaces.py"
    with open(data_interfaces_file, 'w', encoding='utf-8') as f:
        f.write('''"""
Data Service Interfaces

This module defines interfaces for data services used by the analysis engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class IFeatureProvider(ABC):
    """Interface for feature providers."""
    
    @abstractmethod
    async def get_features(self, feature_names: List[str], start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Get features for a specific time range.
        
        Args:
            feature_names: List of feature names to retrieve
            start_time: Start time for the data
            end_time: End time for the data
            
        Returns:
            Dictionary of features
        """
        pass


class IDataPipeline(ABC):
    """Interface for data pipeline services."""
    
    @abstractmethod
    async def get_market_data(self, symbols: List[str], timeframe: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Get market data for a specific time range.
        
        Args:
            symbols: List of symbols to retrieve data for
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h")
            start_time: Start time for the data
            end_time: End time for the data
            
        Returns:
            Dictionary of market data
        """
        pass
''')
    
    # Create ML service interfaces
    ml_interfaces_file = interfaces_dir / "ml_interfaces.py"
    with open(ml_interfaces_file, 'w', encoding='utf-8') as f:
        f.write('''"""
ML Service Interfaces

This module defines interfaces for ML services used by the analysis engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class IModelProvider(ABC):
    """Interface for model providers."""
    
    @abstractmethod
    async def get_model_prediction(self, model_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get prediction from a model.
        
        Args:
            model_id: ID of the model to use
            features: Features to use for prediction
            
        Returns:
            Dictionary of prediction results
        """
        pass


class IModelRegistry(ABC):
    """Interface for model registry services."""
    
    @abstractmethod
    async def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary of model metadata
        """
        pass
''')
    
    # Create trading service interfaces
    trading_interfaces_file = interfaces_dir / "trading_interfaces.py"
    with open(trading_interfaces_file, 'w', encoding='utf-8') as f:
        f.write('''"""
Trading Service Interfaces

This module defines interfaces for trading services used by the analysis engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class IRiskManager(ABC):
    """Interface for risk management services."""
    
    @abstractmethod
    async def evaluate_risk(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate risk for a potential trade.
        
        Args:
            trade_params: Parameters for the trade
            
        Returns:
            Dictionary of risk evaluation results
        """
        pass


class ITradingGateway(ABC):
    """Interface for trading gateway services."""
    
    @abstractmethod
    async def get_market_status(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get market status for symbols.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Dictionary of market status information
        """
        pass
''')
    
    print("Created adapter interfaces in common-lib.", flush=True)

def create_adapter_implementations():
    """Create adapter implementations in analysis-engine-service."""
    print("Creating adapter implementations in analysis-engine-service...", flush=True)
    
    # Create adapters directory if it doesn't exist
    adapters_dir = ANALYSIS_ENGINE_DIR / "analysis_engine" / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    
    # Create __init__.py if it doesn't exist
    init_file = adapters_dir / "__init__.py"
    if not init_file.exists():
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write('"""Adapters package for analysis engine."""\n')
    
    # Create data service adapters
    data_adapters_file = adapters_dir / "data_adapters.py"
    with open(data_adapters_file, 'w', encoding='utf-8') as f:
        f.write('''"""
Data Service Adapters

This module provides adapter implementations for data service interfaces.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from common_lib.interfaces.data_interfaces import IFeatureProvider, IDataPipeline


class FeatureStoreAdapter(IFeatureProvider):
    """Adapter for feature store service."""
    
    async def get_features(self, feature_names: List[str], start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Get features from the feature store.
        
        Args:
            feature_names: List of feature names to retrieve
            start_time: Start time for the data
            end_time: End time for the data
            
        Returns:
            Dictionary of features
        """
        # Implementation would normally call the actual service
        # For now, we'll just return a placeholder
        return {"features": {name: [] for name in feature_names}}


class DataPipelineAdapter(IDataPipeline):
    """Adapter for data pipeline service."""
    
    async def get_market_data(self, symbols: List[str], timeframe: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Get market data from the data pipeline.
        
        Args:
            symbols: List of symbols to retrieve data for
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h")
            start_time: Start time for the data
            end_time: End time for the data
            
        Returns:
            Dictionary of market data
        """
        # Implementation would normally call the actual service
        # For now, we'll just return a placeholder
        return {"market_data": {symbol: [] for symbol in symbols}}
''')
    
    # Create ML service adapters
    ml_adapters_file = adapters_dir / "ml_adapters.py"
    with open(ml_adapters_file, 'w', encoding='utf-8') as f:
        f.write('''"""
ML Service Adapters

This module provides adapter implementations for ML service interfaces.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from common_lib.interfaces.ml_interfaces import IModelProvider, IModelRegistry


class MLWorkbenchAdapter(IModelProvider):
    """Adapter for ML workbench service."""
    
    async def get_model_prediction(self, model_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get prediction from a model in the ML workbench.
        
        Args:
            model_id: ID of the model to use
            features: Features to use for prediction
            
        Returns:
            Dictionary of prediction results
        """
        # Implementation would normally call the actual service
        # For now, we'll just return a placeholder
        return {"prediction": 0.0, "confidence": 0.0}


class ModelRegistryAdapter(IModelRegistry):
    """Adapter for model registry service."""
    
    async def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get metadata for a model from the model registry.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary of model metadata
        """
        # Implementation would normally call the actual service
        # For now, we'll just return a placeholder
        return {"model_id": model_id, "version": "1.0", "metrics": {}}
''')
    
    # Create trading service adapters
    trading_adapters_file = adapters_dir / "trading_adapters.py"
    with open(trading_adapters_file, 'w', encoding='utf-8') as f:
        f.write('''"""
Trading Service Adapters

This module provides adapter implementations for trading service interfaces.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from common_lib.interfaces.trading_interfaces import IRiskManager, ITradingGateway


class RiskManagementAdapter(IRiskManager):
    """Adapter for risk management service."""
    
    async def evaluate_risk(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate risk for a potential trade using the risk management service.
        
        Args:
            trade_params: Parameters for the trade
            
        Returns:
            Dictionary of risk evaluation results
        """
        # Implementation would normally call the actual service
        # For now, we'll just return a placeholder
        return {"risk_score": 0.0, "max_position_size": 0.0}


class TradingGatewayAdapter(ITradingGateway):
    """Adapter for trading gateway service."""
    
    async def get_market_status(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get market status for symbols from the trading gateway.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Dictionary of market status information
        """
        # Implementation would normally call the actual service
        # For now, we'll just return a placeholder
        return {"market_status": {symbol: "open" for symbol in symbols}}
''')
    
    print("Created adapter implementations in analysis-engine-service.", flush=True)

def create_adapter_factory():
    """Create an adapter factory in analysis-engine-service."""
    print("Creating adapter factory in analysis-engine-service...", flush=True)
    
    # Create factory file
    factory_file = ANALYSIS_ENGINE_DIR / "analysis_engine" / "adapters" / "adapter_factory.py"
    with open(factory_file, 'w', encoding='utf-8') as f:
        f.write('''"""
Adapter Factory

This module provides a factory for creating adapters for external services.
"""

from typing import Dict, Any, Type, TypeVar, cast

from common_lib.interfaces.data_interfaces import IFeatureProvider, IDataPipeline
from common_lib.interfaces.ml_interfaces import IModelProvider, IModelRegistry
from common_lib.interfaces.trading_interfaces import IRiskManager, ITradingGateway

from analysis_engine.adapters.data_adapters import FeatureStoreAdapter, DataPipelineAdapter
from analysis_engine.adapters.ml_adapters import MLWorkbenchAdapter, ModelRegistryAdapter
from analysis_engine.adapters.trading_adapters import RiskManagementAdapter, TradingGatewayAdapter


T = TypeVar('T')


class AdapterFactory:
    """Factory for creating adapters for external services."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(AdapterFactory, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the factory."""
        self._adapters = {}
    
    def get_adapter(self, interface_type: Type[T]) -> T:
        """
        Get an adapter for the specified interface type.
        
        Args:
            interface_type: Type of the interface
            
        Returns:
            Adapter instance
        """
        if interface_type not in self._adapters:
            self._adapters[interface_type] = self._create_adapter(interface_type)
        
        return cast(T, self._adapters[interface_type])
    
    def _create_adapter(self, interface_type: Type[T]) -> T:
        """
        Create an adapter for the specified interface type.
        
        Args:
            interface_type: Type of the interface
            
        Returns:
            Adapter instance
        """
        if interface_type == IFeatureProvider:
            return FeatureStoreAdapter()
        elif interface_type == IDataPipeline:
            return DataPipelineAdapter()
        elif interface_type == IModelProvider:
            return MLWorkbenchAdapter()
        elif interface_type == IModelRegistry:
            return ModelRegistryAdapter()
        elif interface_type == IRiskManager:
            return RiskManagementAdapter()
        elif interface_type == ITradingGateway:
            return TradingGatewayAdapter()
        else:
            raise ValueError(f"No adapter available for interface type {interface_type}")


# Singleton instance
adapter_factory = AdapterFactory()
''')
    
    print("Created adapter factory in analysis-engine-service.", flush=True)

def create_usage_example():
    """Create an example of using the adapters."""
    print("Creating usage example in analysis-engine-service...", flush=True)
    
    # Create example file
    example_file = ANALYSIS_ENGINE_DIR / "analysis_engine" / "examples" / "adapter_usage_example.py"
    example_file.parent.mkdir(exist_ok=True)
    
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write('''"""
Adapter Usage Example

This module demonstrates how to use the adapter pattern to reduce direct dependencies.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

from common_lib.interfaces.data_interfaces import IFeatureProvider, IDataPipeline
from common_lib.interfaces.ml_interfaces import IModelProvider
from common_lib.interfaces.trading_interfaces import IRiskManager, ITradingGateway

from analysis_engine.adapters.adapter_factory import adapter_factory


async def analyze_market_conditions(symbols: List[str], timeframe: str) -> Dict[str, Any]:
    """
    Analyze market conditions for a list of symbols.
    
    Args:
        symbols: List of symbols to analyze
        timeframe: Timeframe for the analysis
        
    Returns:
        Analysis results
    """
    # Get adapters from the factory
    feature_provider = adapter_factory.get_adapter(IFeatureProvider)
    data_pipeline = adapter_factory.get_adapter(IDataPipeline)
    model_provider = adapter_factory.get_adapter(IModelProvider)
    risk_manager = adapter_factory.get_adapter(IRiskManager)
    trading_gateway = adapter_factory.get_adapter(ITradingGateway)
    
    # Define time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    # Get market data
    market_data = await data_pipeline.get_market_data(symbols, timeframe, start_time, end_time)
    
    # Get features
    features = await feature_provider.get_features(["volatility", "trend", "momentum"], start_time, end_time)
    
    # Get model prediction
    prediction = await model_provider.get_model_prediction("market_conditions_model", features)
    
    # Evaluate risk
    risk = await risk_manager.evaluate_risk({"symbols": symbols, "prediction": prediction})
    
    # Get market status
    market_status = await trading_gateway.get_market_status(symbols)
    
    # Combine results
    return {
        "symbols": symbols,
        "timeframe": timeframe,
        "prediction": prediction,
        "risk": risk,
        "market_status": market_status
    }


async def main():
    """Main function."""
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    timeframe = "1h"
    
    results = await analyze_market_conditions(symbols, timeframe)
    print(f"Analysis results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
''')
    
    print("Created usage example in analysis-engine-service.", flush=True)

def main():
    """Main entry point."""
    print("Starting analysis engine dependency fix...", flush=True)
    
    # Load dependency report
    data = load_dependency_report()
    
    # Analyze dependencies
    dependency_types = analyze_dependencies(data)
    
    # Create adapter interfaces
    create_adapter_interfaces()
    
    # Create adapter implementations
    create_adapter_implementations()
    
    # Create adapter factory
    create_adapter_factory()
    
    # Create usage example
    create_usage_example()
    
    print("Analysis engine dependency fix complete.", flush=True)
    
    # Create a log file with all changes
    log_path = PROJECT_ROOT / "tools" / "output" / "analysis_engine_dependency_fix.log"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Analysis Engine Dependency Fix\n")
        f.write("============================\n\n")
        f.write(f"Timestamp: {os.path.getmtime(log_path)}\n\n")
        f.write("Actions performed:\n")
        f.write("- Created adapter interfaces in common-lib\n")
        f.write("- Created adapter implementations in analysis-engine-service\n")
        f.write("- Created adapter factory in analysis-engine-service\n")
        f.write("- Created usage example in analysis-engine-service\n")
    
    print(f"Log file created at {log_path}", flush=True)

if __name__ == "__main__":
    main()
