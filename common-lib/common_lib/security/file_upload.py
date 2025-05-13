"""
Secure File Upload Handler

This module provides utilities for secure file upload handling.
"""

import os
import logging
import hashlib
import magic
import re
from typing import Dict, Any, Optional, List, Set, Tuple, BinaryIO
from pathlib import Path

import aiofiles
from fastapi import UploadFile


class SecureFileUploadHandler:
    """
    Utility class for secure file upload handling.
    
    This class provides methods for validating and saving file uploads securely.
    """
    
    def __init__(
        self,
        upload_dir: str,
        allowed_extensions: Optional[Set[str]] = None,
        allowed_mime_types: Optional[Set[str]] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        sanitize_filename: bool = True
    ):
        """
        Initialize the secure file upload handler.
        
        Args:
            upload_dir: Directory to save uploaded files to
            allowed_extensions: Set of allowed file extensions (e.g., {'.jpg', '.png'})
            allowed_mime_types: Set of allowed MIME types (e.g., {'image/jpeg', 'image/png'})
            max_file_size: Maximum file size in bytes
            sanitize_filename: Whether to sanitize filenames
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.upload_dir = upload_dir
        self.allowed_extensions = allowed_extensions or {'.jpg', '.jpeg', '.png', '.gif', '.pdf', '.txt', '.csv', '.xlsx'}
        self.allowed_mime_types = allowed_mime_types or {'image/jpeg', 'image/png', 'image/gif', 'application/pdf', 'text/plain', 'text/csv', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
        self.max_file_size = max_file_size
        self.sanitize_filename = sanitize_filename
        
        # Create upload directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)
    
    async def save_upload(self, file: UploadFile) -> Tuple[bool, str, Optional[str]]:
        """
        Save an uploaded file securely.
        
        Args:
            file: Uploaded file
            
        Returns:
            Tuple of (success, message, saved_path)
        """
        try:
            # Validate file
            is_valid, message = await self.validate_file(file)
            if not is_valid:
                return False, message, None
            
            # Get secure filename
            filename = self._get_secure_filename(file.filename)
            
            # Generate unique filename to prevent overwriting
            unique_filename = self._generate_unique_filename(filename)
            
            # Save file
            file_path = os.path.join(self.upload_dir, unique_filename)
            
            # Read file content
            await file.seek(0)
            content = await file.read()
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            self.logger.info(f"File saved successfully: {file_path}")
            
            return True, "File uploaded successfully", file_path
        
        except Exception as e:
            self.logger.error(f"Error saving file: {str(e)}")
            return False, f"Error saving file: {str(e)}", None
    
    async def validate_file(self, file: UploadFile) -> Tuple[bool, str]:
        """
        Validate an uploaded file.
        
        Args:
            file: Uploaded file
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check if file is empty
            await file.seek(0, os.SEEK_END)
            file_size = await file.tell()
            await file.seek(0)
            
            if file_size == 0:
                return False, "File is empty"
            
            # Check file size
            if file_size > self.max_file_size:
                return False, f"File size exceeds maximum allowed size ({self.max_file_size} bytes)"
            
            # Check file extension
            if self.allowed_extensions:
                ext = os.path.splitext(file.filename)[1].lower()
                if ext not in self.allowed_extensions:
                    return False, f"File extension not allowed. Allowed extensions: {', '.join(self.allowed_extensions)}"
            
            # Check MIME type
            if self.allowed_mime_types:
                # Read a sample of the file to determine MIME type
                sample = await file.read(2048)
                await file.seek(0)
                
                mime = magic.Magic(mime=True)
                mime_type = mime.from_buffer(sample)
                
                if mime_type not in self.allowed_mime_types:
                    return False, f"File type not allowed. Allowed types: {', '.join(self.allowed_mime_types)}"
            
            return True, "File is valid"
        
        except Exception as e:
            self.logger.error(f"Error validating file: {str(e)}")
            return False, f"Error validating file: {str(e)}"
    
    def _get_secure_filename(self, filename: str) -> str:
        """
        Get a secure version of a filename.
        
        Args:
            filename: Original filename
            
        Returns:
            Secure filename
        """
        if not self.sanitize_filename:
            return filename
        
        # Remove path information
        filename = os.path.basename(filename)
        
        # Sanitize filename
        filename = re.sub(r'[^\w\.\-]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255 - len(ext)] + ext
        
        return filename
    
    def _generate_unique_filename(self, filename: str) -> str:
        """
        Generate a unique filename to prevent overwriting.
        
        Args:
            filename: Original filename
            
        Returns:
            Unique filename
        """
        name, ext = os.path.splitext(filename)
        
        # Add timestamp and random hash
        import time
        import random
        timestamp = int(time.time())
        random_hash = hashlib.md5(f"{timestamp}{random.random()}".encode()).hexdigest()[:8]
        
        return f"{name}_{timestamp}_{random_hash}{ext}"