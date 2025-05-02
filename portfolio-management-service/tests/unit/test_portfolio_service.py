"""Portfolio service unit tests."""
import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone
import uuid

from portfolio_management_service.models.position import Position, PositionCreate, PositionUpdate, PositionStatus
from portfolio_management_service.models.account import AccountBalance, AccountDetails
from portfolio_management_service.services.portfolio_service import PortfolioService
from portfolio_management_service.repositories.position_repository import PositionRepository
from portfolio_management_service.repositories.account_repository import AccountRepository


@pytest.fixture
def mock_account_repo():
    """Create mock account repository."""
    repo = AsyncMock()
    repo.get_account_details = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.update_margin = AsyncMock()
    repo.update_balance = AsyncMock()
    return repo


@pytest.fixture
def mock_position_repo():
    """Create mock position repository."""
    repo = AsyncMock()
    repo.create = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.update = AsyncMock()
    repo.get_open_positions = AsyncMock()
    repo.get_closed_positions = AsyncMock()
    return repo


@pytest.fixture
def portfolio_service(mock_account_repo, mock_position_repo):
    """Create portfolio service with mocked repositories."""
    service = PortfolioService()
    # Mock the repository factory
    service._get_account_repo = MagicMock(return_value=mock_account_repo)
    service._get_position_repo = MagicMock(return_value=mock_position_repo)
    return service


@pytest.fixture
def sample_account_id():
    """Generate sample account ID."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_account_details(sample_account_id):
    """Create sample account details."""
    return AccountDetails(
        id=sample_account_id,
        user_id=str(uuid.uuid4()),
        balance=100000.0,
        equity=105000.0,
        margin_used=5000.0,
        free_margin=95000.0,
        last_updated=datetime.now(timezone.utc),
        total_positions=5,
        open_positions=3,
        winning_positions=4,
        losing_positions=1,
        win_rate=0.8,
        total_pnl=5000.0,
        recent_changes=[]
    )


@pytest.fixture
def sample_position_data():
    """Create sample position data."""
    return {
        "id": str(uuid.uuid4()),
        "symbol": "EURUSD",
        "account_id": str(uuid.uuid4()),
        "direction": "long",
        "size": 10000.0,
        "entry_price": 1.1850,
        "current_price": 1.1900,
        "unrealized_pnl": 500.0,
        "realized_pnl": 0.0,
        "status": PositionStatus.OPEN,
        "entry_time": datetime.now(timezone.utc),
        "exit_time": None,
        "meta_data": {}
    }


@pytest.fixture
def sample_position(sample_position_data):
    """Create sample position object."""
    return Position(**sample_position_data)


class TestPortfolioService:
    """Test portfolio service functionality."""

    async def test_get_portfolio_summary_success(
        self, portfolio_service, mock_account_repo, mock_position_repo, 
        sample_account_id, sample_account_details, sample_position
    ):
        """Test retrieving portfolio summary successfully."""
        # Setup mocks
        mock_account_repo.get_account_details.return_value = sample_account_details
        mock_position_repo.get_open_positions.return_value = [sample_position]
        mock_position_repo.get_closed_positions.return_value = []

        # Execute
        summary = await portfolio_service.get_portfolio_summary(sample_account_id)

        # Verify summary structure and data
        assert summary["account"]["id"] == sample_account_id
        assert summary["account"]["balance"] == sample_account_details.balance
        assert summary["account"]["equity"] == sample_account_details.equity
        assert summary["account"]["margin_used"] == sample_account_details.margin_used
        assert summary["positions"]["open_count"] == 1
        assert summary["positions"]["closed_count"] == 0
        assert len(summary["positions"]["open_positions"]) == 1

        # Verify service calls
        mock_account_repo.get_account_details.assert_called_once_with(sample_account_id)
        mock_position_repo.get_open_positions.assert_called_once_with(sample_account_id)

    async def test_get_portfolio_not_found(
        self, portfolio_service, mock_account_repo, sample_account_id
    ):
        """Test retrieving summary for non-existent portfolio."""
        # Setup mock
        mock_account_repo.get_account_details.return_value = None

        # Execute
        summary = await portfolio_service.get_portfolio_summary(sample_account_id)

        # Verify response
        assert summary["account_id"] == sample_account_id
        assert summary["status"] == "not_found"

        # Verify service call
        mock_account_repo.get_account_details.assert_called_once_with(sample_account_id)

    async def test_add_position_success(
        self, portfolio_service, mock_position_repo, mock_account_repo,
        sample_position_data, sample_position
    ):
        """Test adding a new position successfully."""
        # Convert sample data to PositionCreate
        position_create = PositionCreate(
            symbol=sample_position_data["symbol"],
            account_id=sample_position_data["account_id"],
            direction=sample_position_data["direction"],
            size=sample_position_data["size"],
            entry_price=sample_position_data["entry_price"]
        )

        # Setup mock
        mock_position_repo.create.return_value = sample_position
        mock_account_repo.get_by_id.return_value = MagicMock(balance=100000.0)

        # Execute
        result = await portfolio_service.create_position(position_create)

        # Verify result
        assert result == sample_position

        # Verify service calls
        mock_position_repo.create.assert_called_once_with(position_create)
        mock_account_repo.update_margin.assert_called_once()

    async def test_update_position_success(
        self, portfolio_service, mock_position_repo, mock_account_repo,
        sample_position_data, sample_position
    ):
        """Test updating an existing position successfully."""
        # Create position update
        position_update = PositionUpdate(current_price=1.1950)

        # Setup mock
        mock_position_repo.update.return_value = Position(
            **{**sample_position_data, "current_price": 1.1950}
        )

        # Execute
        result = await portfolio_service.update_position(sample_position_data["id"], position_update)

        # Verify result
        assert result is not None
        assert result.current_price == 1.1950

        # Verify service calls
        mock_position_repo.update.assert_called_once_with(sample_position_data["id"], position_update)
