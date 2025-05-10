#!/usr/bin/env python
import asyncio
from mcp.client import ClientSession
from mcp.client.stdio import stdio_client
import logging

async def test_sequential_thinking():
    print("Testing Sequential Thinking MCP...")
    async with stdio_client(["python", "sequential_thinking.py"]) as streams:
        async with ClientSession(*streams) as session:
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools]}")
            
            result = await session.call_tool(
                "sequentialthinking",
                {
                    "thought": "Testing Sequential Thinking MCP",
                    "nextThoughtNeeded": True,
                    "thoughtNumber": 1,
                    "totalThoughts": 2
                }
            )
            print(f"Result: {result}")

async def test_desktop_commander():
    print("\nTesting Desktop Commander MCP...")
    async with stdio_client(["python", "desktop_commander.py"]) as streams:
        async with ClientSession(*streams) as session:
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools]}")
            
            result = await session.call_tool(
                "execute_command",
                {
                    "command": "dir",
                    "timeout": 5000
                }
            )
            print(f"Result: {result}")

async def main():
    await test_sequential_thinking()
    await test_desktop_commander()

if __name__ == "__main__":
    asyncio.run(main())
