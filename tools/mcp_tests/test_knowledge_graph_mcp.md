# Knowledge Graph MCP Server Testing Guide

This guide will help you test if the Knowledge Graph MCP server is properly detected and working in VS Code Insiders.

## Prerequisites

1. Make sure VS Code Insiders is closed.
2. Make sure the mcp-knowledge-graph repository is cloned and dependencies are installed.

## Step 1: Start the Knowledge Graph MCP Server

1. Run the `start_knowledge_graph_mcp.bat` script to start the Knowledge Graph MCP server.
2. This will start the Knowledge Graph MCP server in a separate window.

## Step 2: Start VS Code Insiders

1. Run the `restart_vscode_insiders.bat` script to restart VS Code Insiders with the forex_trading_platform workspace.
2. This will close VS Code Insiders, start the Knowledge Graph MCP server, and then reopen VS Code Insiders.

## Step 3: Check MCP Server Status

1. Press `Ctrl+Shift+P` to open the Command Palette.
2. Type "MCP: List Servers" and press Enter.
3. You should see the "memory" server listed and running.

## Step 4: Test the Knowledge Graph MCP Server

1. Open a new file or the integrated terminal.
2. Type the following command to list available tools:
   ```
   /tool memory listTools
   ```
3. You should see a JSON array of available tools.

4. Create some entities:
   ```
   /tool memory create_entities '{"entities":[{"name":"USD/EUR","entityType":"currency","observations":["Major forex pair"]}]}'
   ```

5. Search for nodes:
   ```
   /tool memory search_nodes '{"query":"forex"}'
   ```

6. You should see the entities you created.

## Troubleshooting

If the Knowledge Graph MCP server isn't working:

1. Check if the server is running:
   - Look for a terminal window running the Knowledge Graph MCP server.
   - If it's not running, run the `start_knowledge_graph_mcp.bat` script.

2. Check the VS Code Insiders logs:
   - View → Output → Extension Host
   - Filter for "mcp" to see any startup errors.

3. Check the mcp.json file:
   - Make sure the path to the index.js file is correct.
   - Make sure the memory-path is correct.

4. Try restarting VS Code Insiders:
   - Run the `restart_vscode_insiders.bat` script.

## Next Steps

Once you've verified that the Knowledge Graph MCP server is working, you can start using it in your forex trading platform development workflow. The Knowledge Graph MCP server provides a persistent memory store that can be used to store and retrieve information across conversations.
