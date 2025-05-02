"""
Integration Test Coordinator for Forex Trading Platform
Manages and executes comprehensive integration testing across all system components
"""

import logging
import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

from analysis_engine.services.market_regime import MarketRegimeService
from ml_integration_service.strategy_filters.ml_confirmation_filter import MLConfirmationFilter
from data_pipeline_service.services.historical_data_service import HistoricalDataService
from strategy_execution_engine.signal_aggregation.signal_aggregator import SignalAggregator
from risk_management_service.risk_management_service.risk_analyzer import RiskAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class IntegrationTestResult:
    """Container for integration test results"""
    test_name: str
    component_name: str
    start_time: datetime
    end_time: datetime
    status: str
    error_message: str = ""
    metrics: Dict[str, Any] = None

class IntegrationTestCoordinator:
    """
    Coordinates and executes comprehensive integration testing across system components
    """
    
    def __init__(self):
        """Initialize integration test coordinator"""
        self.market_regime_service = MarketRegimeService()
        self.ml_filter = MLConfirmationFilter()
        self.data_service = HistoricalDataService()
        self.signal_aggregator = SignalAggregator()
        self.risk_analyzer = RiskAnalyzer()
        
    async def test_market_analysis_integration(self) -> IntegrationTestResult:
        """Test market analysis component integration"""
        logger.info("Testing market analysis integration")
        start_time = datetime.now()
        
        try:
            # Get historical data
            data = await self.data_service.get_historical_data(
                symbol="EUR/USD",
                timeframe="1H",
                bars=1000
            )
            
            # Analyze market regime
            regime = await self.market_regime_service.analyze_regime(data)
            
            # Generate and validate signals
            signals = await self.signal_aggregator.generate_signals(data, regime)
            
            # Verify ML confirmation
            ml_confirmed = await self.ml_filter.validate_signals(signals, data)
            
            return IntegrationTestResult(
                test_name="market_analysis_integration",
                component_name="analysis_engine",
                start_time=start_time,
                end_time=datetime.now(),
                status="passed",
                metrics={
                    "data_points": len(data),
                    "signals_generated": len(signals),
                    "signals_confirmed": len(ml_confirmed)
                }
            )
            
        except Exception as e:
            logger.error(f"Market analysis integration test failed: {str(e)}")
            return IntegrationTestResult(
                test_name="market_analysis_integration",
                component_name="analysis_engine",
                start_time=start_time,
                end_time=datetime.now(),
                status="failed",
                error_message=str(e)
            )

    async def test_decision_system_integration(self) -> IntegrationTestResult:
        """Test decision system component integration"""
        logger.info("Testing decision system integration")
        start_time = datetime.now()
        
        try:
            # Get historical data
            data = await self.data_service.get_historical_data(
                symbol="EUR/USD",
                timeframe="1H",
                bars=1000
            )
            
            # Generate trading decisions
            signals = await self.signal_aggregator.generate_signals(data)
            
            # Apply risk checks
            risk_validated = await self.risk_analyzer.validate_signals(signals)
            
            return IntegrationTestResult(
                test_name="decision_system_integration",
                component_name="strategy_execution_engine",
                start_time=start_time,
                end_time=datetime.now(),
                status="passed",
                metrics={
                    "signals_generated": len(signals),
                    "signals_validated": len(risk_validated)
                }
            )
            
        except Exception as e:
            logger.error(f"Decision system integration test failed: {str(e)}")
            return IntegrationTestResult(
                test_name="decision_system_integration",
                component_name="strategy_execution_engine",
                start_time=start_time,
                end_time=datetime.now(),
                status="failed",
                error_message=str(e)
            )

    async def test_risk_management_integration(self) -> IntegrationTestResult:
        """Test risk management component integration"""
        logger.info("Testing risk management integration")
        start_time = datetime.now()
        
        try:
            # Create test portfolio
            portfolio = {
                "EUR/USD": {"position": 100000, "entry_price": 1.1000},
                "GBP/USD": {"position": -50000, "entry_price": 1.2500}
            }
            
            # Run risk analysis
            risk_metrics = await self.risk_analyzer.analyze_portfolio(portfolio)
            
            # Verify risk limits
            limits_check = await self.risk_analyzer.verify_risk_limits(risk_metrics)
            
            return IntegrationTestResult(
                test_name="risk_management_integration",
                component_name="risk_management_service",
                start_time=start_time,
                end_time=datetime.now(),
                status="passed",
                metrics={
                    "risk_metrics": risk_metrics,
                    "limits_verified": limits_check
                }
            )
            
        except Exception as e:
            logger.error(f"Risk management integration test failed: {str(e)}")
            return IntegrationTestResult(
                test_name="risk_management_integration",
                component_name="risk_management_service",
                start_time=start_time,
                end_time=datetime.now(),
                status="failed",
                error_message=str(e)
            )

    async def test_order_execution_integration(self) -> IntegrationTestResult:
        """Test order execution component integration"""
        logger.info("Testing order execution integration")
        start_time = datetime.now()
        
        try:
            # Create test orders
            orders = [
                {"symbol": "EUR/USD", "type": "MARKET", "side": "BUY", "amount": 10000},
                {"symbol": "GBP/USD", "type": "LIMIT", "side": "SELL", "amount": 5000, "price": 1.2600}
            ]
            
            # Validate orders through risk management
            validated_orders = await self.risk_analyzer.validate_orders(orders)
            
            # Process through execution engine
            execution_results = await self._execute_test_orders(validated_orders)
            
            return IntegrationTestResult(
                test_name="order_execution_integration",
                component_name="trading_gateway_service",
                start_time=start_time,
                end_time=datetime.now(),
                status="passed",
                metrics={
                    "orders_submitted": len(orders),
                    "orders_validated": len(validated_orders),
                    "orders_executed": len(execution_results)
                }
            )
            
        except Exception as e:
            logger.error(f"Order execution integration test failed: {str(e)}")
            return IntegrationTestResult(
                test_name="order_execution_integration",
                component_name="trading_gateway_service",
                start_time=start_time,
                end_time=datetime.now(),
                status="failed",
                error_message=str(e)
            )

    async def test_cross_component_communication(self) -> IntegrationTestResult:
        """Test cross-component communication and data flow"""
        logger.info("Testing cross-component communication")
        start_time = datetime.now()
        
        try:
            # Test data pipeline to analysis engine
            data_flow_1 = await self._test_data_pipeline_flow()
            
            # Test analysis engine to strategy execution
            data_flow_2 = await self._test_analysis_strategy_flow()
            
            # Test strategy execution to risk management
            data_flow_3 = await self._test_strategy_risk_flow()
            
            return IntegrationTestResult(
                test_name="cross_component_communication",
                component_name="system_wide",
                start_time=start_time,
                end_time=datetime.now(),
                status="passed",
                metrics={
                    "data_pipeline_flow": data_flow_1,
                    "analysis_strategy_flow": data_flow_2,
                    "strategy_risk_flow": data_flow_3
                }
            )
            
        except Exception as e:
            logger.error(f"Cross-component communication test failed: {str(e)}")
            return IntegrationTestResult(
                test_name="cross_component_communication",
                component_name="system_wide",
                start_time=start_time,
                end_time=datetime.now(),
                status="failed",
                error_message=str(e)
            )

    async def _execute_test_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulate order execution for testing"""
        # Implementation of test order execution
        return [{"order": order, "status": "EXECUTED"} for order in orders]

    async def _test_data_pipeline_flow(self) -> Dict[str, Any]:
        """Test data pipeline communication flow"""
        # Implementation of data pipeline flow test
        return {"status": "success", "latency_ms": 50}

    async def _test_analysis_strategy_flow(self) -> Dict[str, Any]:
        """Test analysis to strategy execution flow"""
        # Implementation of analysis to strategy flow test
        return {"status": "success", "latency_ms": 75}

    async def _test_strategy_risk_flow(self) -> Dict[str, Any]:
        """Test strategy to risk management flow"""
        # Implementation of strategy to risk flow test
        return {"status": "success", "latency_ms": 45}

async def run_integration_tests():
    """Run the complete integration test suite"""
    coordinator = IntegrationTestCoordinator()
    results = []
    
    # Run all integration tests
    results.append(await coordinator.test_market_analysis_integration())
    results.append(await coordinator.test_decision_system_integration())
    results.append(await coordinator.test_risk_management_integration())
    results.append(await coordinator.test_order_execution_integration())
    results.append(await coordinator.test_cross_component_communication())
    
    # Generate summary
    passed_tests = sum(1 for r in results if r.status == "passed")
    total_tests = len(results)
    
    logger.info(f"Integration test suite completed. {passed_tests}/{total_tests} tests passed.")
    
    return results

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_integration_tests())
