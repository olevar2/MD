import asyncio
import aiohttp
import json
import sys

async def test_sequential_thinking():
    print("\nTesting Sequential Thinking Server...")
    async with aiohttp.ClientSession() as session:
        # Health check
        async with session.get("http://localhost:8000/health") as response:
            print(f"Health check: {await response.json()}")

        # Test thinking
        thought_request = {
            "thought": "Testing Sequential Thinking MCP",
            "nextThoughtNeeded": True,
            "thoughtNumber": 1,
            "totalThoughts": 2
        }
        async with session.post("http://localhost:8000/think", json=thought_request) as response:
            print(f"Think response: {await response.json()}")

async def test_desktop_commander():
    print("\nTesting Desktop Commander Server...")
    async with aiohttp.ClientSession() as session:
        # Health check
        async with session.get("http://localhost:8001/health") as response:
            print(f"Health check: {await response.json()}")

        # Test directory listing
        dir_request = {
            "path": "."
        }
        async with session.post("http://localhost:8001/list_directory", json=dir_request) as response:
            print(f"Directory listing: {await response.json()}")

        # Test command execution
        cmd_request = {
            "command": "dir",
            "timeout": 5000
        }
        async with session.post("http://localhost:8001/execute", json=cmd_request) as response:
            print(f"Command execution: {await response.json()}")

async def main():
    try:
        await test_sequential_thinking()
        await test_desktop_commander()
    except Exception as e:
        print(f"Error during testing: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
