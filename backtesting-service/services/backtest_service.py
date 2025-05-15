from typing import Dict, List, Any, Optional
import uuid
import time
import asyncio
from datetime import datetime, timedelta

from backtesting_service.models.backtest_models import (
    BacktestRequest,
    BacktestResponse,
    BacktestStatus,
    BacktestResult,
    PerformanceMetrics,
    TradeResult
)
from backtesting_service.repositories.backtest_repository import BacktestRepository
from backtesting_service.adapters.data_pipeline_adapter import DataPipelineAdapter
from backtesting_service.adapters.analysis_coordinator_adapter import AnalysisCoordinatorAdapter
from backtesting_service.adapters.strategy_execution_adapter import StrategyExecutionAdapter
from common_lib.resilience.decorators import with_standard_resilience

class BacktestService:
    def __init__(
        self,
        data_pipeline_adapter: DataPipelineAdapter,
        analysis_coordinator_adapter: AnalysisCoordinatorAdapter,
        strategy_execution_adapter: StrategyExecutionAdapter,
        backtest_repository: BacktestRepository
    ):
        self.data_pipeline_adapter = data_pipeline_adapter
        self.analysis_coordinator_adapter = analysis_coordinator_adapter
        self.strategy_execution_adapter = strategy_execution_adapter
        self.backtest_repository = backtest_repository
        self.running_backtests = {}
    
    @with_standard_resilience()
    async def run_backtest(self, request: BacktestRequest) -> BacktestResponse:
        """
        Run a backtest with the specified configuration.
        """
        # Generate a unique ID for the backtest
        backtest_id = str(uuid.uuid4())
        
        # Create a backtest result with initial status
        backtest_result = BacktestResult(
            backtest_id=backtest_id,
            request=request,
            status=BacktestStatus.PENDING,
            start_time=datetime.now(),
            trades=[],
        )
        
        # Save the initial backtest result
        await self.backtest_repository.save_backtest(backtest_id, backtest_result)
        
        # Estimate completion time based on the backtest parameters
        estimated_completion_time = self._estimate_completion_time(request)
        
        # Start the backtest in the background
        asyncio.create_task(self._run_backtest_task(backtest_id, request))
        
        # Return the initial response
        return BacktestResponse(
            backtest_id=backtest_id,
            status=BacktestStatus.PENDING,
            message="Backtest started successfully",
            estimated_completion_time=estimated_completion_time
        )
    
    async def _run_backtest_task(self, backtest_id: str, request: BacktestRequest) -> None:
        """
        Run the backtest task in the background.
        """
        try:
            # Update status to running
            backtest_result = await self.backtest_repository.get_backtest(backtest_id)
            backtest_result.status = BacktestStatus.RUNNING
            await self.backtest_repository.save_backtest(backtest_id, backtest_result)
            
            # Add to running backtests
            self.running_backtests[backtest_id] = True
            
            # Fetch historical data
            historical_data = await self.data_pipeline_adapter.get_historical_data(
                symbols=request.symbols,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            # Get strategy configuration
            strategy_config = await self.strategy_execution_adapter.get_strategy_config(
                strategy_id=request.strategy.strategy_id,
                strategy_type=request.strategy.strategy_type,
                parameters=request.strategy.parameters
            )
            
            # Run the backtest
            start_time = time.time()
            backtest_results = await self.strategy_execution_adapter.run_backtest(
                historical_data=historical_data,
                strategy_config=strategy_config,
                initial_capital=request.initial_capital,
                commission=request.commission,
                slippage=request.slippage,
                leverage=request.leverage,
                additional_parameters=request.additional_parameters
            )
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(backtest_results)
            
            # Update the backtest result
            backtest_result.status = BacktestStatus.COMPLETED
            backtest_result.end_time = datetime.now()
            backtest_result.execution_time_ms = execution_time_ms
            backtest_result.trades = backtest_results.get("trades", [])
            backtest_result.performance_metrics = performance_metrics
            backtest_result.equity_curve = backtest_results.get("equity_curve", [])
            backtest_result.drawdown_curve = backtest_results.get("drawdown_curve", [])
            
            # Save the updated backtest result
            await self.backtest_repository.save_backtest(backtest_id, backtest_result)
            
            # Notify the analysis coordinator
            await self.analysis_coordinator_adapter.notify_backtest_completed(
                backtest_id=backtest_id,
                status="completed",
                performance_metrics=performance_metrics.dict() if performance_metrics else None
            )
        except Exception as e:
            # Update status to failed
            backtest_result = await self.backtest_repository.get_backtest(backtest_id)
            backtest_result.status = BacktestStatus.FAILED
            backtest_result.end_time = datetime.now()
            backtest_result.error_message = str(e)
            await self.backtest_repository.save_backtest(backtest_id, backtest_result)
            
            # Notify the analysis coordinator
            await self.analysis_coordinator_adapter.notify_backtest_completed(
                backtest_id=backtest_id,
                status="failed",
                error_message=str(e)
            )
        finally:
            # Remove from running backtests
            if backtest_id in self.running_backtests:
                del self.running_backtests[backtest_id]
    
    def _calculate_performance_metrics(self, backtest_results: Dict[str, Any]) -> PerformanceMetrics:
        """
        Calculate performance metrics from backtest results.
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, this would calculate metrics from the backtest results
        
        trades = backtest_results.get("trades", [])
        if not trades:
            return None
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.get("profit_loss", 0) > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_profit = sum(trade.get("profit_loss", 0) for trade in trades if trade.get("profit_loss", 0) > 0)
        total_loss = abs(sum(trade.get("profit_loss", 0) for trade in trades if trade.get("profit_loss", 0) < 0))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        average_win = total_profit / winning_trades if winning_trades > 0 else 0
        average_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        # Calculate more advanced metrics
        equity_curve = backtest_results.get("equity_curve", [])
        if not equity_curve:
            return None
        
        initial_equity = equity_curve[0].get("equity", 0)
        final_equity = equity_curve[-1].get("equity", 0)
        
        total_return = ((final_equity - initial_equity) / initial_equity) * 100
        
        # Calculate annualized return
        start_date = datetime.fromisoformat(backtest_results.get("start_date", datetime.now().isoformat()))
        end_date = datetime.fromisoformat(backtest_results.get("end_date", datetime.now().isoformat()))
        years = (end_date - start_date).days / 365.25
        
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Calculate drawdown
        drawdown_curve = backtest_results.get("drawdown_curve", [])
        max_drawdown = max((dd.get("drawdown", 0) for dd in drawdown_curve), default=0)
        
        # Calculate Sharpe and Sortino ratios
        # This is a simplified calculation
        returns = [
            (equity_curve[i].get("equity", 0) - equity_curve[i-1].get("equity", 0)) / equity_curve[i-1].get("equity", 0)
            for i in range(1, len(equity_curve))
        ]
        
        if not returns:
            return None
        
        avg_return = sum(returns) / len(returns)
        std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        
        sharpe_ratio = (avg_return / std_dev) * (252 ** 0.5) if std_dev > 0 else 0
        
        # Calculate Sortino ratio
        negative_returns = [r for r in returns if r < 0]
        downside_dev = (sum((r - 0) ** 2 for r in negative_returns) / len(negative_returns)) ** 0.5 if negative_returns else 0
        
        sortino_ratio = (avg_return / downside_dev) * (252 ** 0.5) if downside_dev > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades
        )
    
    def _estimate_completion_time(self, request: BacktestRequest) -> datetime:
        """
        Estimate the completion time for a backtest.
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, this would estimate based on the backtest parameters
        
        # Simple estimation: 1 minute per year of data per symbol
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
        
        years = (end_date - start_date).days / 365.25
        symbols_count = len(request.symbols)
        
        # Estimate minutes needed
        estimated_minutes = years * symbols_count
        
        # Add some buffer
        estimated_minutes *= 1.2
        
        # Return estimated completion time
        return datetime.now() + timedelta(minutes=estimated_minutes)
    
    @with_standard_resilience()
    async def get_backtest_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """
        Get the result of a previously run backtest.
        """
        return await self.backtest_repository.get_backtest(backtest_id)
    
    @with_standard_resilience()
    async def get_backtest_status(self, backtest_id: str) -> Optional[BacktestStatus]:
        """
        Get the status of a previously run backtest.
        """
        backtest = await self.backtest_repository.get_backtest(backtest_id)
        if not backtest:
            return None
        return backtest.status
    
    @with_standard_resilience()
    async def delete_backtest(self, backtest_id: str) -> bool:
        """
        Delete a previously run backtest.
        """
        # Check if the backtest is running
        if backtest_id in self.running_backtests:
            # Cancel the backtest first
            await self.cancel_backtest(backtest_id)
        
        # Delete the backtest
        return await self.backtest_repository.delete_backtest(backtest_id)
    
    @with_standard_resilience()
    async def list_backtests(
        self,
        limit: int = 10,
        offset: int = 0,
        strategy_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all backtests with optional filtering.
        """
        backtests = await self.backtest_repository.list_backtests(limit, offset, strategy_id, status)
        
        # Convert to simplified format for listing
        return [
            {
                "backtest_id": backtest.backtest_id,
                "status": backtest.status,
                "start_time": backtest.start_time,
                "end_time": backtest.end_time,
                "symbols": backtest.request.symbols,
                "timeframe": backtest.request.timeframe,
                "strategy_type": backtest.request.strategy.strategy_type,
                "strategy_id": backtest.request.strategy.strategy_id,
                "performance": {
                    "total_return": backtest.performance_metrics.total_return if backtest.performance_metrics else None,
                    "max_drawdown": backtest.performance_metrics.max_drawdown if backtest.performance_metrics else None,
                    "sharpe_ratio": backtest.performance_metrics.sharpe_ratio if backtest.performance_metrics else None,
                    "win_rate": backtest.performance_metrics.win_rate if backtest.performance_metrics else None
                } if backtest.performance_metrics else None
            }
            for backtest in backtests
        ]
    
    @with_standard_resilience()
    async def cancel_backtest(self, backtest_id: str) -> bool:
        """
        Cancel a running backtest.
        """
        # Check if the backtest is running
        if backtest_id not in self.running_backtests:
            backtest = await self.backtest_repository.get_backtest(backtest_id)
            if not backtest or backtest.status != BacktestStatus.RUNNING:
                return False
        
        # Mark the backtest as cancelled
        backtest = await self.backtest_repository.get_backtest(backtest_id)
        backtest.status = BacktestStatus.CANCELLED
        backtest.end_time = datetime.now()
        await self.backtest_repository.save_backtest(backtest_id, backtest)
        
        # Remove from running backtests
        if backtest_id in self.running_backtests:
            del self.running_backtests[backtest_id]
        
        # Notify the analysis coordinator
        await self.analysis_coordinator_adapter.notify_backtest_completed(
            backtest_id=backtest_id,
            status="cancelled"
        )
        
        return True