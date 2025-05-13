"""
Multi-Factor Authentication Module for Forex Trading Platform

This module provides multi-factor authentication functionality for the forex trading platform,
including TOTP (Time-based One-Time Password) and backup codes.
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set

import pyotp
from pydantic import BaseModel, Field

# Configure logger
logger = logging.getLogger(__name__)


class TOTPConfig(BaseModel):
    """TOTP configuration"""
    secret: str
    issuer: str
    account_name: str
    digits: int = 6
    interval: int = 30
    algorithm: str = "SHA1"
    enabled: bool = True


class BackupCodes(BaseModel):
    """Backup codes for MFA"""
    codes: List[str]
    used: Set[str] = Field(default_factory=set)


class MFAMethod:
    """MFA method types"""
    TOTP = "totp"
    BACKUP_CODES = "backup_codes"
    EMAIL = "email"
    SMS = "sms"


class MFAStatus:
    """MFA status types"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    PENDING = "pending"


class MFAChallenge(BaseModel):
    """MFA challenge"""
    challenge_id: str
    user_id: str
    method: str
    expires_at: datetime
    completed: bool = False
    attempts: int = 0
    max_attempts: int = 3


class MFAService:
    """
    Multi-Factor Authentication Service for the forex trading platform.
    
    This class provides MFA functionality, including:
    - TOTP (Time-based One-Time Password)
    - Backup codes
    - Email verification
    - SMS verification
    """
    
    def __init__(
        self,
        issuer: str = "Forex Trading Platform",
        totp_digits: int = 6,
        totp_interval: int = 30,
        totp_algorithm: str = "SHA1",
        backup_codes_count: int = 10,
        backup_codes_length: int = 8,
        challenge_expiry_minutes: int = 10,
        max_attempts: int = 3
    ):
        """
        Initialize the MFA service.
        
        Args:
            issuer: Issuer name for TOTP
            totp_digits: Number of digits for TOTP
            totp_interval: Interval in seconds for TOTP
            totp_algorithm: Algorithm for TOTP
            backup_codes_count: Number of backup codes
            backup_codes_length: Length of backup codes
            challenge_expiry_minutes: Expiry time for challenges in minutes
            max_attempts: Maximum number of attempts for challenges
        """
        self.issuer = issuer
        self.totp_digits = totp_digits
        self.totp_interval = totp_interval
        self.totp_algorithm = totp_algorithm
        self.backup_codes_count = backup_codes_count
        self.backup_codes_length = backup_codes_length
        self.challenge_expiry_minutes = challenge_expiry_minutes
        self.max_attempts = max_attempts
        
        # In-memory storage for challenges
        # In a real implementation, this would be stored in a database
        self.challenges: Dict[str, MFAChallenge] = {}
    
    def generate_totp_config(self, user_id: str, account_name: str) -> TOTPConfig:
        """
        Generate TOTP configuration for a user.
        
        Args:
            user_id: User ID
            account_name: Account name for TOTP
            
        Returns:
            TOTP configuration
        """
        # Generate random secret
        secret = pyotp.random_base32()
        
        # Create TOTP config
        config = TOTPConfig(
            secret=secret,
            issuer=self.issuer,
            account_name=account_name,
            digits=self.totp_digits,
            interval=self.totp_interval,
            algorithm=self.totp_algorithm,
            enabled=False
        )
        
        return config
    
    def get_totp_uri(self, config: TOTPConfig) -> str:
        """
        Get TOTP URI for QR code generation.
        
        Args:
            config: TOTP configuration
            
        Returns:
            TOTP URI
        """
        totp = pyotp.TOTP(
            config.secret,
            digits=config.digits,
            interval=config.interval,
            digest=getattr(hashlib, config.algorithm.lower())
        )
        
        return totp.provisioning_uri(
            name=config.account_name,
            issuer_name=config.issuer
        )
    
    def verify_totp(self, config: TOTPConfig, code: str) -> bool:
        """
        Verify TOTP code.
        
        Args:
            config: TOTP configuration
            code: TOTP code
            
        Returns:
            True if code is valid, False otherwise
        """
        if not config.enabled:
            logger.warning(f"TOTP not enabled for account: {config.account_name}")
            return False
        
        totp = pyotp.TOTP(
            config.secret,
            digits=config.digits,
            interval=config.interval,
            digest=getattr(hashlib, config.algorithm.lower())
        )
        
        return totp.verify(code)
    
    def generate_backup_codes(self) -> BackupCodes:
        """
        Generate backup codes.
        
        Returns:
            Backup codes
        """
        codes = []
        
        for _ in range(self.backup_codes_count):
            # Generate random code
            code = secrets.token_hex(self.backup_codes_length // 2)
            
            # Format code with hyphens
            formatted_code = "-".join([
                code[i:i+4] for i in range(0, len(code), 4)
            ])
            
            codes.append(formatted_code)
        
        return BackupCodes(codes=codes)
    
    def verify_backup_code(self, backup_codes: BackupCodes, code: str) -> bool:
        """
        Verify backup code.
        
        Args:
            backup_codes: Backup codes
            code: Backup code
            
        Returns:
            True if code is valid, False otherwise
        """
        # Normalize code
        normalized_code = code.replace("-", "").strip().lower()
        
        # Check if code exists and has not been used
        for stored_code in backup_codes.codes:
            normalized_stored_code = stored_code.replace("-", "").strip().lower()
            
            if normalized_code == normalized_stored_code and stored_code not in backup_codes.used:
                # Mark code as used
                backup_codes.used.add(stored_code)
                
                return True
        
        return False
    
    def create_challenge(self, user_id: str, method: str) -> MFAChallenge:
        """
        Create MFA challenge.
        
        Args:
            user_id: User ID
            method: MFA method
            
        Returns:
            MFA challenge
        """
        # Generate challenge ID
        challenge_id = secrets.token_urlsafe(32)
        
        # Create challenge
        challenge = MFAChallenge(
            challenge_id=challenge_id,
            user_id=user_id,
            method=method,
            expires_at=datetime.now() + timedelta(minutes=self.challenge_expiry_minutes),
            max_attempts=self.max_attempts
        )
        
        # Store challenge
        self.challenges[challenge_id] = challenge
        
        return challenge
    
    def verify_challenge(self, challenge_id: str, code: str, config: Any) -> bool:
        """
        Verify MFA challenge.
        
        Args:
            challenge_id: Challenge ID
            code: Verification code
            config: MFA configuration (TOTPConfig or BackupCodes)
            
        Returns:
            True if challenge is verified, False otherwise
        """
        # Get challenge
        challenge = self.challenges.get(challenge_id)
        
        if not challenge:
            logger.warning(f"Challenge not found: {challenge_id}")
            return False
        
        # Check if challenge is expired
        if challenge.expires_at < datetime.now():
            logger.warning(f"Challenge expired: {challenge_id}")
            return False
        
        # Check if challenge is already completed
        if challenge.completed:
            logger.warning(f"Challenge already completed: {challenge_id}")
            return False
        
        # Check if max attempts reached
        if challenge.attempts >= challenge.max_attempts:
            logger.warning(f"Max attempts reached for challenge: {challenge_id}")
            return False
        
        # Increment attempts
        challenge.attempts += 1
        
        # Verify code based on method
        verified = False
        
        if challenge.method == MFAMethod.TOTP and isinstance(config, TOTPConfig):
            verified = self.verify_totp(config, code)
        elif challenge.method == MFAMethod.BACKUP_CODES and isinstance(config, BackupCodes):
            verified = self.verify_backup_code(config, code)
        elif challenge.method == MFAMethod.EMAIL:
            # Email verification would be implemented here
            pass
        elif challenge.method == MFAMethod.SMS:
            # SMS verification would be implemented here
            pass
        
        # Update challenge status
        if verified:
            challenge.completed = True
        
        return verified
