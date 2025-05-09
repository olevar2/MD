@echo off
echo Closing VS Code Insiders...
taskkill /f /im "Code - Insiders.exe" > nul 2>&1

echo Starting Knowledge Graph MCP server...
cd /d D:\MD\forex_trading_platform\mcp-knowledge-graph
start "" node index.js --memory-path memory.jsonl
echo Knowledge Graph MCP server started!

echo Starting VS Code Insiders with forex_trading_platform workspace...
start "" "C:\Users\ASD\AppData\Local\Programs\Microsoft VS Code Insiders\Code - Insiders.exe" "D:\MD\forex_trading_platform"
echo Done!
