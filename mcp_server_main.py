from fastapi import FastAPI
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, InitializationOptions, NotificationOptions
import uvicorn
import json
import logging
from pathlib import Path

class VSCodeMCPServer:
    def __init__(self):
        self.logger = logging.getLogger("vscode_mcp_server")
        self.app = FastAPI()
        self.server = Server("vscode-forex-mcp")

    async def setup_tools(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="analyze_market",
                    description="Analyze market data for forex trading",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "timeframe": {"type": "string"}
                        },
                        "required": ["symbol", "timeframe"]
                    }
                ),
                Tool(
                    name="execute_strategy",
                    description="Execute a trading strategy",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "strategy_name": {"type": "string"},
                            "parameters": {"type": "object"}
                        },
                        "required": ["strategy_name"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            try:
                if name == "analyze_market":
                    return [TextContent(
                        type="text",
                        text=f"Analyzing market for {arguments['symbol']} on {arguments['timeframe']} timeframe"
                    )]
                elif name == "execute_strategy":
                    return [TextContent(
                        type="text",
                        text=f"Executing strategy {arguments['strategy_name']}"
                    )]
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                raise ValueError(f"Error processing tool request: {str(e)}")

    async def start(self):
        """Start the VS Code MCP server."""
        try:
            # Setup tools
            await self.setup_tools()

            # Configure server initialization
            options = InitializationOptions(
                server_name="vscode-forex-mcp",
                server_version="0.1.0",
                capabilities=self.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )

            # Start server using stdio transport
            async with stdio_server() as (read_stream, write_stream):
                self.logger.info("VS Code MCP Server running with stdio transport")
                await self.server.run(
                    read_stream,
                    write_stream,
                    options
                )

        except Exception as e:
            self.logger.error(f"Server error: {str(e)}")
            raise

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("vscode-mcp.log"),
            logging.StreamHandler()
        ]
    )

    # Create and start server
    server = VSCodeMCPServer()
    import asyncio
    asyncio.run(server.start())

if __name__ == "__main__":
    main()