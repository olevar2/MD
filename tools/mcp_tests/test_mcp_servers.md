# MCP Server Testing Guide

This guide will help you test if your MCP servers are properly detected and working in VS Code Insiders.

## Prerequisites

1. Make sure VS Code Insiders is closed.
2. Make sure all MCP packages are installed:
   - `mcp-sequentialthinking-tools`
   - `@wonderwhy-er/desktop-commander`
   - `@sylphlab/tools-memory-mcp`

## Step 1: Start Memory MCP Server and VS Code Insiders

1. Run the `start_memory_mcp.bat` script to start the Memory MCP server.
2. Run the `restart_vscode_insiders.bat` script to restart VS Code Insiders with the forex_trading_platform workspace.
3. Wait for VS Code Insiders to fully load.

## Step 2: Check MCP Server Status

1. Press `Ctrl+Shift+P` to open the Command Palette.
2. Type "MCP: List Servers" and press Enter.
3. You should see a list of your MCP servers:
   - sequentialthinking
   - desktop-commander
   - memory
4. Each server should show "running" status. If not, click on the server to start it.

## Step 3: Test Sequential Thinking MCP

1. Open a new file or the integrated terminal.
2. Type the following command:
   ```
   /tool sequentialthinking list
   ```
3. You should see a JSON array of available "thinking steps".

## Step 4: Test Desktop Commander MCP

1. Type the following command:
   ```
   /tool desktop-commander exec "pwd"
   ```
2. You should see your current working directory.

## Step 5: Test Memory MCP

1. Type the following command to store a memory:
   ```
   /tool memory store key="test" value="hello"
   ```
2. Then retrieve the memory:
   ```
   /tool memory recall key="test"
   ```
3. You should see "hello" in the response.

## Troubleshooting

If any of the MCP servers aren't working:

1. Check the VS Code Insiders logs:
   - View → Output → Extension Host
   - Filter for "mcp" to see any startup errors.

2. Check the terminal output:
   - Help → Toggle Developer Tools → Console
   - Look for errors related to MCP.

3. Verify your configuration files:
   - `.vscode/mcp.json`
   - `.vscode/settings.json`
   - `%APPDATA%\Code - Insiders\User\settings.json`

4. Try restarting VS Code Insiders.

5. Try manually starting the MCP servers:
   - Press `Ctrl+Shift+P`
   - Type "MCP: List Servers"
   - Click on each server to start it.

### Specific Troubleshooting for Memory MCP

If the Memory MCP server isn't working:

1. Run the `test_memory_mcp.bat` script to test the Memory MCP server.
2. Make sure the Memory MCP server is running before starting VS Code Insiders.
3. Check if there's a service running on port 5000:
   ```
   netstat -ano | findstr :5000
   ```
4. If the Memory MCP server isn't running, try starting it manually:
   ```
   npx @sylphlab/tools-memory-mcp
   ```

## Next Steps

Once you've verified that all MCP servers are working, you can start using them in your forex trading platform development workflow.
