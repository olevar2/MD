"""
Role-Based Access Control Module for Forex Trading Platform

This module provides role-based access control functionality for the forex trading platform,
including role management, permission management, and access control.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Union
from functools import wraps

from fastapi import Depends, HTTPException, status
from pydantic import BaseModel, Field

# Configure logger
logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """Permission types"""
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Role management
    ROLE_CREATE = "role:create"
    ROLE_READ = "role:read"
    ROLE_UPDATE = "role:update"
    ROLE_DELETE = "role:delete"
    
    # Market data
    MARKET_DATA_READ = "market_data:read"
    MARKET_DATA_WRITE = "market_data:write"
    
    # Trading
    TRADE_READ = "trade:read"
    TRADE_EXECUTE = "trade:execute"
    TRADE_CANCEL = "trade:cancel"
    
    # Analysis
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_CREATE = "analysis:create"
    
    # System
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_BACKUP = "system:backup"
    
    # API
    API_READ = "api:read"
    API_WRITE = "api:write"


class Role(BaseModel):
    """Role model"""
    name: str
    description: str
    permissions: Set[str] = Field(default_factory=set)
    is_system_role: bool = False


class UserRoles(BaseModel):
    """User roles model"""
    user_id: str
    roles: Set[str] = Field(default_factory=set)


class RBACService:
    """
    Role-Based Access Control Service for the forex trading platform.
    
    This class provides RBAC functionality, including:
    - Role management
    - Permission management
    - Access control
    """
    
    def __init__(self):
        """Initialize the RBAC service."""
        # In-memory storage for roles and user roles
        # In a real implementation, this would be stored in a database
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, UserRoles] = {}
        
        # Initialize system roles
        self._initialize_system_roles()
    
    def _initialize_system_roles(self):
        """Initialize system roles."""
        # Admin role
        admin_role = Role(
            name="admin",
            description="Administrator role with all permissions",
            permissions=set(p.value for p in Permission),
            is_system_role=True
        )
        
        # User role
        user_role = Role(
            name="user",
            description="Standard user role",
            permissions={
                Permission.USER_READ.value,
                Permission.MARKET_DATA_READ.value,
                Permission.TRADE_READ.value,
                Permission.ANALYSIS_READ.value,
                Permission.API_READ.value
            },
            is_system_role=True
        )
        
        # Trader role
        trader_role = Role(
            name="trader",
            description="Trader role with trading permissions",
            permissions={
                Permission.USER_READ.value,
                Permission.MARKET_DATA_READ.value,
                Permission.TRADE_READ.value,
                Permission.TRADE_EXECUTE.value,
                Permission.TRADE_CANCEL.value,
                Permission.ANALYSIS_READ.value,
                Permission.API_READ.value
            },
            is_system_role=True
        )
        
        # Analyst role
        analyst_role = Role(
            name="analyst",
            description="Analyst role with analysis permissions",
            permissions={
                Permission.USER_READ.value,
                Permission.MARKET_DATA_READ.value,
                Permission.TRADE_READ.value,
                Permission.ANALYSIS_READ.value,
                Permission.ANALYSIS_CREATE.value,
                Permission.API_READ.value
            },
            is_system_role=True
        )
        
        # Add roles
        self.roles[admin_role.name] = admin_role
        self.roles[user_role.name] = user_role
        self.roles[trader_role.name] = trader_role
        self.roles[analyst_role.name] = analyst_role
    
    def create_role(self, role: Role) -> Role:
        """
        Create a new role.
        
        Args:
            role: Role to create
            
        Returns:
            Created role
            
        Raises:
            ValueError: If role already exists
        """
        if role.name in self.roles:
            raise ValueError(f"Role already exists: {role.name}")
        
        self.roles[role.name] = role
        
        logger.info(f"Role created: {role.name}")
        
        return role
    
    def get_role(self, role_name: str) -> Optional[Role]:
        """
        Get role by name.
        
        Args:
            role_name: Role name
            
        Returns:
            Role if found, None otherwise
        """
        return self.roles.get(role_name)
    
    def update_role(self, role_name: str, role: Role) -> Role:
        """
        Update role.
        
        Args:
            role_name: Role name
            role: Updated role
            
        Returns:
            Updated role
            
        Raises:
            ValueError: If role does not exist or is a system role
        """
        existing_role = self.get_role(role_name)
        
        if not existing_role:
            raise ValueError(f"Role does not exist: {role_name}")
        
        if existing_role.is_system_role:
            raise ValueError(f"Cannot update system role: {role_name}")
        
        self.roles[role_name] = role
        
        logger.info(f"Role updated: {role_name}")
        
        return role
    
    def delete_role(self, role_name: str) -> None:
        """
        Delete role.
        
        Args:
            role_name: Role name
            
        Raises:
            ValueError: If role does not exist or is a system role
        """
        existing_role = self.get_role(role_name)
        
        if not existing_role:
            raise ValueError(f"Role does not exist: {role_name}")
        
        if existing_role.is_system_role:
            raise ValueError(f"Cannot delete system role: {role_name}")
        
        # Remove role from all users
        for user_roles in self.user_roles.values():
            if role_name in user_roles.roles:
                user_roles.roles.remove(role_name)
        
        # Delete role
        del self.roles[role_name]
        
        logger.info(f"Role deleted: {role_name}")
    
    def assign_role_to_user(self, user_id: str, role_name: str) -> None:
        """
        Assign role to user.
        
        Args:
            user_id: User ID
            role_name: Role name
            
        Raises:
            ValueError: If role does not exist
        """
        if role_name not in self.roles:
            raise ValueError(f"Role does not exist: {role_name}")
        
        # Get or create user roles
        if user_id not in self.user_roles:
            self.user_roles[user_id] = UserRoles(user_id=user_id)
        
        # Assign role
        self.user_roles[user_id].roles.add(role_name)
        
        logger.info(f"Role assigned to user: {role_name} -> {user_id}")
    
    def remove_role_from_user(self, user_id: str, role_name: str) -> None:
        """
        Remove role from user.
        
        Args:
            user_id: User ID
            role_name: Role name
        """
        if user_id not in self.user_roles:
            return
        
        # Remove role
        if role_name in self.user_roles[user_id].roles:
            self.user_roles[user_id].roles.remove(role_name)
            
            logger.info(f"Role removed from user: {role_name} -> {user_id}")
    
    def get_user_roles(self, user_id: str) -> Set[str]:
        """
        Get roles for user.
        
        Args:
            user_id: User ID
            
        Returns:
            Set of role names
        """
        if user_id not in self.user_roles:
            return set()
        
        return self.user_roles[user_id].roles
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """
        Get permissions for user.
        
        Args:
            user_id: User ID
            
        Returns:
            Set of permissions
        """
        # Get user roles
        user_roles = self.get_user_roles(user_id)
        
        # Get permissions for each role
        permissions = set()
        
        for role_name in user_roles:
            role = self.get_role(role_name)
            
            if role:
                permissions.update(role.permissions)
        
        return permissions
    
    def has_permission(self, user_id: str, permission: Union[str, Permission]) -> bool:
        """
        Check if user has permission.
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        # Convert Permission enum to string if needed
        if isinstance(permission, Permission):
            permission = permission.value
        
        # Get user permissions
        user_permissions = self.get_user_permissions(user_id)
        
        # Check if user has permission
        return permission in user_permissions
    
    def has_any_permission(self, user_id: str, permissions: List[Union[str, Permission]]) -> bool:
        """
        Check if user has any of the permissions.
        
        Args:
            user_id: User ID
            permissions: Permissions to check
            
        Returns:
            True if user has any of the permissions, False otherwise
        """
        return any(self.has_permission(user_id, p) for p in permissions)
    
    def has_all_permissions(self, user_id: str, permissions: List[Union[str, Permission]]) -> bool:
        """
        Check if user has all of the permissions.
        
        Args:
            user_id: User ID
            permissions: Permissions to check
            
        Returns:
            True if user has all of the permissions, False otherwise
        """
        return all(self.has_permission(user_id, p) for p in permissions)


# FastAPI dependency for permission checking
def require_permission(permission: Union[str, Permission]):
    """
    FastAPI dependency for permission checking.
    
    Args:
        permission: Permission to check
    """
    def dependency(user_id: str = Depends(get_current_user_id), rbac_service: RBACService = Depends(get_rbac_service)):
        if not rbac_service.has_permission(user_id, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        
        return user_id
    
    return dependency


# Placeholder for getting current user ID
def get_current_user_id():
    """Get current user ID."""
    # This would be implemented based on your authentication system
    pass


# Placeholder for getting RBAC service
def get_rbac_service():
    """Get RBAC service."""
    # This would be implemented based on your dependency injection system
    return RBACService()
