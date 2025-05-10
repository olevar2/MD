"""
Test Server

This module provides a test server for end-to-end testing of the Analysis Engine.
"""

import os
import sys
import time
import logging
import subprocess
import signal
import socket
import requests
from typing import Optional, List, Dict, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

class TestServer:
    """Test server for end-to-end testing."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        timeout: int = 30,
        env: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the test server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            timeout: Timeout in seconds for server startup
            env: Environment variables for the server process
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.env = env or {}
        self.process = None
        self.base_url = f"http://{host}:{port}"
    
    def start(self):
        """Start the test server."""
        logger.info(f"Starting test server on {self.host}:{self.port}...")
        
        # Get the path to the server script
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        server_script = os.path.join(script_dir, "app.py")
        
        # Set environment variables
        env = os.environ.copy()
        env.update(self.env)
        env["TESTING"] = "1"
        env["HOST"] = self.host
        env["PORT"] = str(self.port)
        
        # Start the server process
        self.process = subprocess.Popen(
            [sys.executable, server_script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for the server to start
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                # Check if the server is running
                with socket.create_connection((self.host, self.port), timeout=1):
                    break
            except (socket.error, ConnectionRefusedError):
                # Check if the process is still running
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    raise Exception(f"Server process exited with code {self.process.returncode}:\nstdout: {stdout}\nstderr: {stderr}")
                
                # Wait a bit before trying again
                time.sleep(0.1)
        else:
            # Timeout reached
            self.stop()
            raise Exception(f"Server failed to start within {self.timeout} seconds")
        
        # Wait a bit more to ensure the server is fully initialized
        time.sleep(1)
        
        # Check if the server is responding
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code != 200:
                self.stop()
                raise Exception(f"Server returned status code {response.status_code}")
        except Exception as e:
            self.stop()
            raise Exception(f"Failed to connect to server: {e}")
        
        logger.info("Test server started successfully")
    
    def stop(self):
        """Stop the test server."""
        if self.process is not None:
            logger.info("Stopping test server...")
            
            # Send SIGTERM to the process
            self.process.terminate()
            
            # Wait for the process to exit
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Process didn't exit, send SIGKILL
                logger.warning("Server process didn't exit, sending SIGKILL")
                self.process.kill()
                self.process.wait()
            
            self.process = None
            
            logger.info("Test server stopped")
    
    def __enter__(self):
        """Enter context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.stop()
