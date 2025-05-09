@echo off
echo Starting Knowledge Graph MCP server...
cd /d D:\MD\forex_trading_platform\mcp-knowledge-graph
start "" node index.js --memory-path memory.jsonl
echo Knowledge Graph MCP server started!
