"""
Service dependencies for the backtesting service.
"""
from app.services.backtest_service import BacktestService
from app.repositories.backtest_repository import BacktestRepository
from app.core.data_client import DataClient

# Create singleton instances
backtest_repository = BacktestRepository()
data_client = DataClient()
backtest_service = BacktestService(repository=backtest_repository, data_client=data_client)

def get_backtest_service():
    """Get the backtest service instance."""
    return backtest_service
