"""
Limits Repository Module.

Repository for accessing and storing risk limits data.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime

from risk_management_service.db.connection import Session
from risk_management_service.models.risk_limits import (
    RiskLimit, RiskLimitCreate, RiskLimitUpdate,
    RiskProfile, RiskProfileCreate, ProfileLimit
)
from core_foundations.utils.logger import get_logger

logger = get_logger("limits-repository")


class LimitsRepository:
    """Repository for risk limits and profiles data."""
    
    def __init__(self, session: Session):
        """
        Initialize repository with database session.
        
        Args:
            session: Database session
        """
        self.session = session
    
    def create_limit(self, limit_data: RiskLimitCreate) -> RiskLimit:
        """
        Create a new risk limit.
        
        Args:
            limit_data: Risk limit data to create
            
        Returns:
            Created risk limit
        """
        query = """
        INSERT INTO risk_limits
        (account_id, limit_type, limit_value, description, created_at, active)
        VALUES
        (%(account_id)s, %(limit_type)s, %(limit_value)s, %(description)s, %(created_at)s, %(active)s)
        RETURNING id
        """
        
        params = {
            **limit_data.dict(),
            "created_at": datetime.utcnow(),
            "active": True
        }
        
        result = self.session.execute(query, params)
        limit_id = result.fetchone()[0]
        
        # Create complete risk limit object
        limit = RiskLimit(
            id=limit_id,
            account_id=limit_data.account_id,
            limit_type=limit_data.limit_type,
            limit_value=limit_data.limit_value,
            description=limit_data.description,
            created_at=params["created_at"],
            active=True
        )
        
        logger.info(f"Created risk limit {limit_id} for account {limit_data.account_id}")
        return limit
    
    def get_limit_by_id(self, limit_id: str) -> Optional[RiskLimit]:
        """
        Get a risk limit by ID.
        
        Args:
            limit_id: Risk limit ID
            
        Returns:
            Risk limit if found, None otherwise
        """
        query = """
        SELECT 
            id,
            account_id,
            limit_type,
            limit_value,
            description,
            created_at,
            updated_at,
            active
        FROM risk_limits
        WHERE id = %(limit_id)s
        """
        
        result = self.session.execute(query, {"limit_id": limit_id})
        row = result.fetchone()
        
        if not row:
            return None
        
        return RiskLimit(
            id=row[0],
            account_id=row[1],
            limit_type=row[2],
            limit_value=row[3],
            description=row[4],
            created_at=row[5],
            updated_at=row[6],
            active=row[7]
        )
    
    def get_limits_by_account(self, account_id: str, active_only: bool = True) -> List[RiskLimit]:
        """
        Get all risk limits for an account.
        
        Args:
            account_id: Account ID
            active_only: If True, only return active limits
            
        Returns:
            List of risk limits
        """
        query = """
        SELECT 
            id,
            account_id,
            limit_type,
            limit_value,
            description,
            created_at,
            updated_at,
            active
        FROM risk_limits
        WHERE account_id = %(account_id)s
        """
        
        if active_only:
            query += " AND active = TRUE"
            
        query += " ORDER BY limit_type"
        
        result = self.session.execute(query, {"account_id": account_id})
        
        limits = []
        for row in result:
            limit = RiskLimit(
                id=row[0],
                account_id=row[1],
                limit_type=row[2],
                limit_value=row[3],
                description=row[4],
                created_at=row[5],
                updated_at=row[6],
                active=row[7]
            )
            limits.append(limit)
        
        return limits
    
    def update_limit(self, limit_id: str, update_data: RiskLimitUpdate) -> Optional[RiskLimit]:
        """
        Update a risk limit.
        
        Args:
            limit_id: Risk limit ID
            update_data: Data to update
            
        Returns:
            Updated risk limit if found, None otherwise
        """
        # Get current limit
        current_limit = self.get_limit_by_id(limit_id)
        if not current_limit:
            logger.warning(f"Risk limit {limit_id} not found for update")
            return None
        
        # Build update fields
        update_fields = []
        params = {"limit_id": limit_id, "updated_at": datetime.utcnow()}
        
        update_dict = update_data.dict(exclude_unset=True)
        for key, value in update_dict.items():
            if value is not None:
                update_fields.append(f"{key} = %({key})s")
                params[key] = value
        
        if not update_fields:
            logger.warning("No fields to update for risk limit")
            return current_limit
        
        # Add updated_at
        update_fields.append("updated_at = %(updated_at)s")
        
        # Build and execute query
        query = f"""
        UPDATE risk_limits
        SET {", ".join(update_fields)}
        WHERE id = %(limit_id)s
        """
        
        self.session.execute(query, params)
        
        # Get updated limit
        updated_limit = self.get_limit_by_id(limit_id)
        logger.info(f"Updated risk limit {limit_id}")
        
        return updated_limit
    
    def deactivate_limit(self, limit_id: str) -> bool:
        """
        Deactivate a risk limit.
        
        Args:
            limit_id: Risk limit ID
            
        Returns:
            True if successful, False otherwise
        """
        query = """
        UPDATE risk_limits
        SET active = FALSE, updated_at = %(updated_at)s
        WHERE id = %(limit_id)s
        """
        
        params = {
            "limit_id": limit_id,
            "updated_at": datetime.utcnow()
        }
        
        result = self.session.execute(query, params)
        success = result.rowcount > 0
        
        if success:
            logger.info(f"Deactivated risk limit {limit_id}")
        else:
            logger.warning(f"Failed to deactivate risk limit {limit_id}")
            
        return success
    
    def create_risk_profile(self, profile_data: Dict[str, Any]) -> RiskProfile:
        """
        Create a risk profile with predefined limits.
        
        Args:
            profile_data: Risk profile data
            
        Returns:
            Created risk profile
        """
        # Extract basic profile data
        query = """
        INSERT INTO risk_profiles
        (name, description, risk_level, created_at)
        VALUES
        (%(name)s, %(description)s, %(risk_level)s, %(created_at)s)
        RETURNING id
        """
        
        params = {
            "name": profile_data["name"],
            "description": profile_data.get("description"),
            "risk_level": profile_data["risk_level"],
            "created_at": datetime.utcnow()
        }
        
        result = self.session.execute(query, params)
        profile_id = result.fetchone()[0]
        
        # Create profile limits
        limits = []
        for limit_data in profile_data.get("limits", []):
            # Insert limit into profile_limits table
            limit_query = """
            INSERT INTO profile_limits
            (profile_id, limit_type, limit_value, description)
            VALUES
            (%(profile_id)s, %(limit_type)s, %(limit_value)s, %(description)s)
            """
            
            limit_params = {
                "profile_id": profile_id,
                "limit_type": limit_data["limit_type"],
                "limit_value": limit_data["limit_value"],
                "description": limit_data.get("description")
            }
            
            self.session.execute(limit_query, limit_params)
            
            # Add to limits list for return value
            limits.append(ProfileLimit(
                limit_type=limit_data["limit_type"],
                limit_value=limit_data["limit_value"],
                description=limit_data.get("description")
            ))
        
        # Create complete risk profile object
        profile = RiskProfile(
            id=profile_id,
            name=params["name"],
            description=params["description"],
            risk_level=params["risk_level"],
            limits=limits,
            created_at=params["created_at"]
        )
        
        logger.info(f"Created risk profile {profile_id}: {profile.name}")
        return profile
    
    def get_risk_profile_by_id(self, profile_id: str) -> Optional[RiskProfile]:
        """
        Get a risk profile by ID.
        
        Args:
            profile_id: Risk profile ID
            
        Returns:
            Risk profile if found, None otherwise
        """
        # Get profile basic info
        query = """
        SELECT 
            id,
            name,
            description,
            risk_level,
            created_at,
            updated_at
        FROM risk_profiles
        WHERE id = %(profile_id)s
        """
        
        result = self.session.execute(query, {"profile_id": profile_id})
        row = result.fetchone()
        
        if not row:
            return None
        
        # Get profile limits
        limits_query = """
        SELECT 
            limit_type,
            limit_value,
            description
        FROM profile_limits
        WHERE profile_id = %(profile_id)s
        """
        
        limits_result = self.session.execute(limits_query, {"profile_id": profile_id})
        
        limits = []
        for limits_row in limits_result:
            limit = ProfileLimit(
                limit_type=limits_row[0],
                limit_value=limits_row[1],
                description=limits_row[2]
            )
            limits.append(limit)
        
        # Create complete risk profile object
        profile = RiskProfile(
            id=row[0],
            name=row[1],
            description=row[2],
            risk_level=row[3],
            limits=limits,
            created_at=row[4],
            updated_at=row[5]
        )
        
        return profile
    
    def get_all_risk_profiles(self) -> List[RiskProfile]:
        """
        Get all risk profiles.
        
        Returns:
            List of risk profiles
        """
        # Get all profiles
        query = """
        SELECT 
            id,
            name,
            description,
            risk_level,
            created_at,
            updated_at
        FROM risk_profiles
        ORDER BY name
        """
        
        result = self.session.execute(query)
        
        profiles = []
        for row in result:
            profile_id = row[0]
            
            # Get profile limits
            limits_query = """
            SELECT 
                limit_type,
                limit_value,
                description
            FROM profile_limits
            WHERE profile_id = %(profile_id)s
            """
            
            limits_result = self.session.execute(limits_query, {"profile_id": profile_id})
            
            limits = []
            for limits_row in limits_result:
                limit = ProfileLimit(
                    limit_type=limits_row[0],
                    limit_value=limits_row[1],
                    description=limits_row[2]
                )
                limits.append(limit)
            
            # Create profile object and add to list
            profile = RiskProfile(
                id=row[0],
                name=row[1],
                description=row[2],
                risk_level=row[3],
                limits=limits,
                created_at=row[4],
                updated_at=row[5]
            )
            profiles.append(profile)
        
        return profiles
