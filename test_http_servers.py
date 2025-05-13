#!/usr/bin/env python
"""
Test script for MCP servers using HTTP requests.
This script tests both the Desktop Commander and Sequential Thinking MCP servers.
"""

import asyncio
import json
import logging
import sys
import aiohttp
import subprocess
import time
import os
import signal

# Fix for Windows event loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_http_test")

# Global variables for server processes
sequential_thinking_process = None
desktop_commander_process = None

async def start_servers():
    """Start the MCP servers."""
    global sequential_thinking_process, desktop_commander_process

    logger.info("Starting MCP servers...")

    # Start Sequential Thinking MCP with safe path handling
    sequential_thinking_path = os.path.join("mcp-server", "sequential_thinking.py")
    if not os.path.exists(sequential_thinking_path):
        logger.error(f"Sequential Thinking script not found at {sequential_thinking_path}")
        return

    sequential_thinking_process = subprocess.Popen(
        [sys.executable, sequential_thinking_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PORT": "8000", "USE_HTTP": "true"}
    )
    logger.info("Sequential Thinking MCP started on port 8000")

    # Start Desktop Commander MCP with safe path handling
    desktop_commander_path = os.path.join("mcp-server", "desktop_commander.py")
    if not os.path.exists(desktop_commander_path):
        logger.error(f"Desktop Commander script not found at {desktop_commander_path}")
        return

    desktop_commander_process = subprocess.Popen(
        [sys.executable, desktop_commander_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PORT": "8001", "USE_HTTP": "true"}
    )
    logger.info("Desktop Commander MCP started on port 8001")

    # Wait for servers to start
    logger.info("Waiting for servers to start...")
    await asyncio.sleep(5)

    # Check if servers are running
    sequential_thinking_returncode = sequential_thinking_process.poll()
    desktop_commander_returncode = desktop_commander_process.poll()

    if sequential_thinking_returncode is not None:
        stderr = sequential_thinking_process.stderr.read().decode('utf-8')
        logger.error(f"Sequential Thinking MCP failed to start: {stderr}")

    if desktop_commander_returncode is not None:
        stderr = desktop_commander_process.stderr.read().decode('utf-8')
        logger.error(f"Desktop Commander MCP failed to start: {stderr}")

async def stop_servers():
    """Stop the MCP servers."""
    global sequential_thinking_process, desktop_commander_process

    logger.info("Stopping MCP servers...")

    # Stop Sequential Thinking MCP
    if sequential_thinking_process:
        try:
            if sys.platform == 'win32':
                sequential_thinking_process.terminate()
            else:
                os.kill(sequential_thinking_process.pid, signal.SIGTERM)
            logger.info("Sequential Thinking MCP stopped")
        except Exception as e:
            logger.error(f"Error stopping Sequential Thinking MCP: {e}")

    # Stop Desktop Commander MCP
    if desktop_commander_process:
        try:
            if sys.platform == 'win32':
                desktop_commander_process.terminate()
            else:
                os.kill(desktop_commander_process.pid, signal.SIGTERM)
            logger.info("Desktop Commander MCP stopped")
        except Exception as e:
            logger.error(f"Error stopping Desktop Commander MCP: {e}")

async def test_sequential_thinking():
    """Test the Sequential Thinking MCP server."""
    logger.info("Testing Sequential Thinking MCP...")

    try:
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get("http://localhost:8000/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"Health check: {health_data}")
                else:
                    logger.error(f"Health check failed with status {response.status}")
                    return False

            # Test sequential thinking tool
            data = {
                "thought": "Testing the Sequential Thinking MCP server",
                "nextThoughtNeeded": True,
                "thoughtNumber": 1,
                "totalThoughts": 3
            }
            async with session.post("http://localhost:8000/sequentialthinking", json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Sequential thinking result: {json.dumps(result, indent=2)}")
                    logger.info("Sequential Thinking MCP test completed successfully!")
                    return True
                else:
                    logger.error(f"Sequential thinking test failed with status {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Error testing Sequential Thinking MCP: {e}")
        return False

async def test_desktop_commander():
    """Test the Desktop Commander MCP server."""
    logger.info("Testing Desktop Commander MCP...")

    try:
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get("http://localhost:8001/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"Health check: {health_data}")
                else:
                    logger.error(f"Health check failed with status {response.status}")
                    return False

            # Test list_directory tool
            data = {
                "path": "."
            }
            async with session.post("http://localhost:8001/list_directory", json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Directory listing result: {json.dumps(result, indent=2)}")
                else:
                    logger.error(f"Directory listing test failed with status {response.status}")
                    return False

            # Test execute_command tool
            data = {
                "command": "dir",
                "timeout": 5000
            }
            async with session.post("http://localhost:8001/execute_command", json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Command execution result: {json.dumps(result, indent=2)}")
                    logger.info("Desktop Commander MCP test completed successfully!")
                    return True
                else:
                    logger.error(f"Command execution test failed with status {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Error testing Desktop Commander MCP: {e}")
        return False

async def main():
    """Run all tests."""
    logger.info("Starting MCP server tests...")

    try:
        # Start the servers
        await start_servers()

        # Test Sequential Thinking MCP
        sequential_thinking_success = await test_sequential_thinking()

        # Test Desktop Commander MCP
        desktop_commander_success = await test_desktop_commander()

        # Print summary
        logger.info("\n--- Test Summary ---")
        logger.info(f"Sequential Thinking MCP: {'SUCCESS' if sequential_thinking_success else 'FAILED'}")
        logger.info(f"Desktop Commander MCP: {'SUCCESS' if desktop_commander_success else 'FAILED'}")

        if sequential_thinking_success and desktop_commander_success:
            logger.info("All tests passed successfully!")
            return 0
        else:
            logger.error("Some tests failed!")
            return 1
    finally:
        # Stop the servers
        await stop_servers()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
