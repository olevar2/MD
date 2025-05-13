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
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TestServer:
    """Test server for end-to-end testing."""

    def __init__(self, host: str='localhost', port: int=8000, timeout: int=
        30, env: Optional[Dict[str, str]]=None):
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
        self.base_url = f'http://{host}:{port}'

    @with_exception_handling
    def start(self):
        """Start the test server."""
        logger.info(f'Starting test server on {self.host}:{self.port}...')
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.
            path.abspath(__file__))))
        server_script = os.path.join(script_dir, 'app.py')
        env = os.environ.copy()
        env.update(self.env)
        env['TESTING'] = '1'
        env['HOST'] = self.host
        env['PORT'] = str(self.port)
        self.process = subprocess.Popen([sys.executable, server_script],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True)
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                with socket.create_connection((self.host, self.port), timeout=1
                    ):
                    break
            except (socket.error, ConnectionRefusedError):
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    raise Exception(
                        f"""Server process exited with code {self.process.returncode}:
stdout: {stdout}
stderr: {stderr}"""
                        )
                time.sleep(0.1)
        else:
            self.stop()
            raise Exception(
                f'Server failed to start within {self.timeout} seconds')
        time.sleep(1)
        try:
            response = requests.get(f'{self.base_url}/health')
            if response.status_code != 200:
                self.stop()
                raise Exception(
                    f'Server returned status code {response.status_code}')
        except Exception as e:
            self.stop()
            raise Exception(f'Failed to connect to server: {e}')
        logger.info('Test server started successfully')

    @with_exception_handling
    def stop(self):
        """Stop the test server."""
        if self.process is not None:
            logger.info('Stopping test server...')
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server process didn't exit, sending SIGKILL")
                self.process.kill()
                self.process.wait()
            self.process = None
            logger.info('Test server stopped')

    def __enter__(self):
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.stop()
