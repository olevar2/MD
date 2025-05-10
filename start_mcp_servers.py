#!/usr/bin/env python
import asyncio
import subprocess
import logging
import sys
from pathlib import Path
import os

async def start_server(script_path: str, port: int, logger: logging.Logger) -> subprocess.Popen:
    """Start a server process and return the process object"""
    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PORT": str(port)}
        )
        logger.info(f"Started server {script_path} on port {port}")
        return process
    except Exception as e:
        logger.error(f"Failed to start server {script_path}: {str(e)}")
        raise

async def monitor_servers(servers: dict, logger: logging.Logger):
    """Monitor server processes and restart them if they fail"""
    while True:
        for name, (process, script_path, port) in servers.items():
            if process.poll() is not None:
                logger.error(f"{name} server terminated, restarting...")
                servers[name] = (await start_server(script_path, port, logger), script_path, port)
        await asyncio.sleep(1)

async def start_servers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("mcp-servers.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("mcp_servers")

    try:
        server_configs = {
            "sequential_thinking": ("mcp-server/sequential_thinking.py", 8000),
            "desktop_commander": ("mcp-server/desktop_commander.py", 8001)
        }

        servers = {}
        for name, (script_path, port) in server_configs.items():
            process = await start_server(script_path, port, logger)
            servers[name] = (process, script_path, port)

        await monitor_servers(servers, logger)

    except Exception as e:
        logger.error(f"Error in server management: {str(e)}")
        # Clean up any running processes
        for name, (process, _, _) in servers.items():
            try:
                process.terminate()
            except:
                pass
        raise

if __name__ == "__main__":
    asyncio.run(start_servers())