"""
Risk Repository Module.

Repository for accessing and storing risk metrics data.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from risk_management_service.db.connection import Session
from risk_management_service.models.risk_metrics import (
    AccountRiskInfo, PositionRisk, RiskMetrics, HistoricalRiskMetrics,
    RiskScenario, StressTestResult, RiskAlert
)
from core_foundations.utils.logger import get_logger

logger = get_logger("risk-repository")


class RiskRepository:
    """Repository for risk metrics data."""
    
    def __init__(self, session: Session):
        """
        Initialize repository with database session.
        
        Args:
            session: Database session
        """
        self.session = session
    
    def get_account_info(self, account_id: str) -> Optional[AccountRiskInfo]:
        """
        Get account risk information.
        
        Args:
            account_id: Account ID
            
        Returns:
            Account risk info if found, None otherwise
        """
        query = """
        SELECT 
            a.account_id,
            a.balance,
            a.equity,
            a.margin_used,
            a.free_margin,
            a.margin_level,
            a.currency,
            a.leverage,
            a.updated_at
        FROM account_risk_info a
        WHERE a.account_id = %(account_id)s
        """
        
        result = self.session.execute(query, {"account_id": account_id})
        row = result.fetchone()
        
        if not row:
            return None
        
        return AccountRiskInfo(
            account_id=row[0],
            balance=row[1],
            equity=row[2],
            margin_used=row[3],
            free_margin=row[4],
            margin_level=row[5],
            currency=row[6],
            leverage=row[7],
            updated_at=row[8]
        )
    
    def update_account_info(self, account_info: AccountRiskInfo) -> AccountRiskInfo:
        """
        Update or create account risk information.
        
        Args:
            account_info: Account risk information
            
        Returns:
            Updated account risk info
        """
        # Check if account exists
        existing = self.get_account_info(account_info.account_id)
        
        if existing:
            # Update existing account
            query = """
            UPDATE account_risk_info
            SET 
                balance = %(balance)s,
                equity = %(equity)s,
                margin_used = %(margin_used)s,
                free_margin = %(free_margin)s,
                margin_level = %(margin_level)s,
                currency = %(currency)s,
                leverage = %(leverage)s,
                updated_at = %(updated_at)s
            WHERE account_id = %(account_id)s
            """
        else:
            # Insert new account
            query = """
            INSERT INTO account_risk_info
            (account_id, balance, equity, margin_used, free_margin, margin_level, currency, leverage, updated_at)
            VALUES
            (%(account_id)s, %(balance)s, %(equity)s, %(margin_used)s, %(free_margin)s, %(margin_level)s, 
             %(currency)s, %(leverage)s, %(updated_at)s)
            """
        
        params = account_info.dict()
        if params.get("updated_at") is None:
            params["updated_at"] = datetime.utcnow()
            
        self.session.execute(query, params)
        
        logger.info(f"Updated account risk info for {account_info.account_id}")
        return account_info
    
    def get_position_risk(self, position_id: str) -> Optional[PositionRisk]:
        """
        Get risk information for a position.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position risk if found, None otherwise
        """
        query = """
        SELECT 
            position_id,
            account_id,
            symbol,
            direction,
            size,
            entry_price,
            current_price,
            stop_loss,
            take_profit,
            unrealized_pnl,
            risk_amount,
            risk_pct,
            reward_amount,
            reward_pct,
            risk_reward_ratio,
            margin_used
        FROM position_risk
        WHERE position_id = %(position_id)s
        """
        
        result = self.session.execute(query, {"position_id": position_id})
        row = result.fetchone()
        
        if not row:
            return None
        
        return PositionRisk(
            position_id=row[0],
            account_id=row[1],
            symbol=row[2],
            direction=row[3],
            size=row[4],
            entry_price=row[5],
            current_price=row[6],
            stop_loss=row[7],
            take_profit=row[8],
            unrealized_pnl=row[9],
            risk_amount=row[10],
            risk_pct=row[11],
            reward_amount=row[12],
            reward_pct=row[13],
            risk_reward_ratio=row[14],
            margin_used=row[15]
        )
    
    def update_position_risk(self, position_risk: PositionRisk) -> PositionRisk:
        """
        Update or create position risk information.
        
        Args:
            position_risk: Position risk information
            
        Returns:
            Updated position risk
        """
        # Check if position exists
        existing = self.get_position_risk(position_risk.position_id)
        
        if existing:
            # Update existing position
            query = """
            UPDATE position_risk
            SET 
                account_id = %(account_id)s,
                symbol = %(symbol)s,
                direction = %(direction)s,
                size = %(size)s,
                entry_price = %(entry_price)s,
                current_price = %(current_price)s,
                stop_loss = %(stop_loss)s,
                take_profit = %(take_profit)s,
                unrealized_pnl = %(unrealized_pnl)s,
                risk_amount = %(risk_amount)s,
                risk_pct = %(risk_pct)s,
                reward_amount = %(reward_amount)s,
                reward_pct = %(reward_pct)s,
                risk_reward_ratio = %(risk_reward_ratio)s,
                margin_used = %(margin_used)s
            WHERE position_id = %(position_id)s
            """
        else:
            # Insert new position
            query = """
            INSERT INTO position_risk
            (position_id, account_id, symbol, direction, size, entry_price, current_price, stop_loss, take_profit,
             unrealized_pnl, risk_amount, risk_pct, reward_amount, reward_pct, risk_reward_ratio, margin_used)
            VALUES
            (%(position_id)s, %(account_id)s, %(symbol)s, %(direction)s, %(size)s, %(entry_price)s, %(current_price)s,
             %(stop_loss)s, %(take_profit)s, %(unrealized_pnl)s, %(risk_amount)s, %(risk_pct)s, %(reward_amount)s,
             %(reward_pct)s, %(risk_reward_ratio)s, %(margin_used)s)
            """
        
        self.session.execute(query, position_risk.dict())
        
        logger.info(f"Updated position risk for {position_risk.position_id}")
        return position_risk
    
    def get_account_positions_risk(self, account_id: str) -> List[PositionRisk]:
        """
        Get risk information for all positions in an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            List of position risk objects
        """
        query = """
        SELECT 
            position_id,
            account_id,
            symbol,
            direction,
            size,
            entry_price,
            current_price,
            stop_loss,
            take_profit,
            unrealized_pnl,
            risk_amount,
            risk_pct,
            reward_amount,
            reward_pct,
            risk_reward_ratio,
            margin_used
        FROM position_risk
        WHERE account_id = %(account_id)s
        """
        
        result = self.session.execute(query, {"account_id": account_id})
        
        positions = []
        for row in result:
            position = PositionRisk(
                position_id=row[0],
                account_id=row[1],
                symbol=row[2],
                direction=row[3],
                size=row[4],
                entry_price=row[5],
                current_price=row[6],
                stop_loss=row[7],
                take_profit=row[8],
                unrealized_pnl=row[9],
                risk_amount=row[10],
                risk_pct=row[11],
                reward_amount=row[12],
                reward_pct=row[13],
                risk_reward_ratio=row[14],
                margin_used=row[15]
            )
            positions.append(position)
        
        return positions
    
    def get_total_exposure(self, account_id: str) -> float:
        """
        Get total exposure for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Total exposure value
        """
        query = """
        SELECT SUM(size * current_price)
        FROM position_risk
        WHERE account_id = %(account_id)s
        """
        
        result = self.session.execute(query, {"account_id": account_id})
        total_exposure = result.fetchone()[0]
        
        return total_exposure or 0.0
    
    def get_open_positions_count(self, account_id: str) -> int:
        """
        Get count of open positions for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Number of open positions
        """
        query = """
        SELECT COUNT(*)
        FROM position_risk
        WHERE account_id = %(account_id)s
        """
        
        result = self.session.execute(query, {"account_id": account_id})
        count = result.fetchone()[0]
        
        return count or 0
    
    def save_risk_metrics(self, metrics: RiskMetrics) -> RiskMetrics:
        """
        Save risk metrics for an account.
        
        Args:
            metrics: Risk metrics to save
            
        Returns:
            Saved risk metrics
        """
        # Convert to dict and handle special fields
        metrics_dict = metrics.dict()
        metrics_dict["positions_by_direction"] = str(metrics.positions_by_direction)
        metrics_dict["positions_by_symbol"] = str(metrics.positions_by_symbol)
        
        # Check if we should update existing metrics or create new ones
        # Here we'll always insert a new record since these are point-in-time metrics
        query = """
        INSERT INTO risk_metrics
        (account_id, timestamp, open_positions_count, positions_by_direction, positions_by_symbol,
         total_exposure, total_exposure_pct, largest_exposure, largest_exposure_pct, largest_exposure_symbol,
         unrealized_pnl, unrealized_pnl_pct, daily_pnl, daily_pnl_pct,
         max_drawdown_pct, current_drawdown_pct, var_pct, var_amount, cvar_pct, cvar_amount,
         margin_used, margin_used_pct, free_margin, margin_level)
        VALUES
        (%(account_id)s, %(timestamp)s, %(open_positions_count)s, %(positions_by_direction)s, %(positions_by_symbol)s,
         %(total_exposure)s, %(total_exposure_pct)s, %(largest_exposure)s, %(largest_exposure_pct)s, %(largest_exposure_symbol)s,
         %(unrealized_pnl)s, %(unrealized_pnl_pct)s, %(daily_pnl)s, %(daily_pnl_pct)s,
         %(max_drawdown_pct)s, %(current_drawdown_pct)s, %(var_pct)s, %(var_amount)s, %(cvar_pct)s, %(cvar_amount)s,
         %(margin_used)s, %(margin_used_pct)s, %(free_margin)s, %(margin_level)s)
        """
        
        self.session.execute(query, metrics_dict)
        
        # Also save to historical metrics table
        historical_query = """
        INSERT INTO historical_risk_metrics
        (account_id, date, metrics_data)
        VALUES
        (%(account_id)s, %(date)s, %(metrics_data)s)
        """
        
        historical_params = {
            "account_id": metrics.account_id,
            "date": metrics.timestamp.date(),
            "metrics_data": str(metrics_dict)  # In a real system, use proper JSON serialization
        }
        
        self.session.execute(historical_query, historical_params)
        
        logger.info(f"Saved risk metrics for account {metrics.account_id}")
        return metrics
    
    def get_risk_metrics(self, account_id: str) -> Dict[str, Any]:
        """
        Get latest risk metrics for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Dictionary with risk metrics
        """
        query = """
        SELECT 
            open_positions_count,
            positions_by_direction,
            positions_by_symbol,
            total_exposure,
            total_exposure_pct,
            largest_exposure,
            largest_exposure_pct,
            largest_exposure_symbol,
            unrealized_pnl,
            unrealized_pnl_pct,
            daily_pnl,
            daily_pnl_pct,
            max_drawdown_pct,
            current_drawdown_pct,
            var_pct,
            var_amount,
            cvar_pct,
            cvar_amount,
            margin_used,
            margin_used_pct,
            free_margin,
            margin_level
        FROM risk_metrics
        WHERE account_id = %(account_id)s
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        result = self.session.execute(query, {"account_id": account_id})
        row = result.fetchone()
        
        if not row:
            return {}
        
        # Create a dictionary with metrics
        metrics = {
            "open_positions_count": row[0],
            "positions_by_direction": eval(row[1]) if row[1] else {},  # In real system, use proper JSON deserialization
            "positions_by_symbol": eval(row[2]) if row[2] else {},
            "total_exposure": row[3],
            "total_exposure_pct": row[4],
            "largest_exposure": row[5],
            "largest_exposure_pct": row[6],
            "largest_exposure_symbol": row[7],
            "unrealized_pnl": row[8],
            "unrealized_pnl_pct": row[9],
            "daily_pnl": row[10],
            "daily_pnl_pct": row[11],
            "max_drawdown_pct": row[12],
            "current_drawdown_pct": row[13],
            "var_pct": row[14],
            "var_amount": row[15],
            "cvar_pct": row[16],
            "cvar_amount": row[17],
            "margin_used": row[18],
            "margin_used_pct": row[19],
            "free_margin": row[20],
            "margin_level": row[21]
        }
        
        return metrics
    
    def record_risk_check(self, account_id: str, check_type: str, result: str, violations_count: int,
                         details: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a risk check.
        
        Args:
            account_id: Account ID
            check_type: Type of risk check
            result: Result of the check
            violations_count: Number of violations
            details: Optional additional details
            
        Returns:
            ID of the created risk check record
        """
        query = """
        INSERT INTO risk_checks
        (account_id, check_type, result, violations_count, created_at, details)
        VALUES
        (%(account_id)s, %(check_type)s, %(result)s, %(violations_count)s, %(created_at)s, %(details)s)
        RETURNING id
        """
        
        params = {
            "account_id": account_id,
            "check_type": check_type,
            "result": result,
            "violations_count": violations_count,
            "created_at": datetime.utcnow(),
            "details": str(details) if details else None  # In real system, use proper JSON serialization
        }
        
        result = self.session.execute(query, params)
        check_id = result.fetchone()[0]
        
        logger.info(f"Recorded risk check {check_id} for account {account_id}")
        return check_id
    
    def save_risk_alert(self, alert: RiskAlert) -> RiskAlert:
        """
        Save a risk alert.
        
        Args:
            alert: Risk alert to save
            
        Returns:
            Saved risk alert with ID
        """
        query = """
        INSERT INTO risk_alerts
        (account_id, alert_type, severity, message, related_check_id, created_at)
        VALUES
        (%(account_id)s, %(alert_type)s, %(severity)s, %(message)s, %(related_check_id)s, %(created_at)s)
        RETURNING id
        """
        
        params = alert.dict(exclude={"id"})
        if params.get("created_at") is None:
            params["created_at"] = datetime.utcnow()
            
        result = self.session.execute(query, params)
        alert_id = result.fetchone()[0]
        
        # Update the alert object with the ID
        alert.id = alert_id
        
        logger.info(f"Saved risk alert {alert_id} for account {alert.account_id}")
        return alert
