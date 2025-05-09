#!/usr/bin/env python
"""
Test script for connecting to the Sequential Thinking Server
"""

import mcp
from mcp.client.websocket import websocket_client
import json
import base64
import asyncio
import os

async def test_sequential_thinking():
    """
    Test connection to the Sequential Thinking Server
    """
    config = {
        "memoryFilePath": "memory.json"
    }
    # Encode config in base64
    config_b64 = base64.b64encode(json.dumps(config).encode()).decode()
    smithery_api_key = "0f552d54-94f7-4f3c-b89c-cb286cd042d0"

    # Create server URL
    url = f"wss://server.smithery.ai/@smithery-ai/server-sequential-thinking/ws?config={config_b64}&api_key={smithery_api_key}"

    print(f"Connecting to {url}")

    try:
        # Connect to the server using websocket client
        async with websocket_client(url) as streams:
            async with mcp.ClientSession(*streams) as session:
                # Initialize the connection
                await session.initialize()
                # List available tools
                tools_result = await session.list_tools()
                print(f"Available tools: {', '.join([t.name for t in tools_result.tools])}")

                # Example of calling the sequential thinking tool
                result = await session.call_tool(
                    "sequentialthinking",
                    arguments={
                        "thought": "Testing connection to the Sequential Thinking Server",
                        "nextThoughtNeeded": False,
                        "thoughtNumber": 1,
                        "totalThoughts": 1
                    }
                )
                print(f"Result: {result}")

                print("Connection successful!")
    except Exception as e:
        print(f"Error connecting to Sequential Thinking Server: {e}")

if __name__ == "__main__":
    asyncio.run(test_sequential_thinking())
