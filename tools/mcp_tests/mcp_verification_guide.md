# MCP Servers Verification Guide

This guide will help you verify that the MCP servers (Sequential Thinking, Desktop Commander, and Memory) are working correctly with both Claude Desktop and VS Code Insiders.

## Prerequisites

Before starting the verification process, make sure you have:

1. Installed the MCP servers:
   - Sequential Thinking MCP (`mcp-sequentialthinking-tools`)
   - Desktop Commander MCP (`@wonderwhy-er/desktop-commander`)
   - Memory MCP (configured with API key)

2. Configured Claude Desktop and VS Code Insiders to use these MCP servers.

## Step 1: Restart Applications

1. Close Claude Desktop completely (not just minimize).
2. Close VS Code Insiders completely.
3. Restart both applications.

## Step 2: Test Sequential Thinking MCP

In Claude Desktop, enter the following command:

```
/tool sequential-thinking thought="I need to analyze the forex trading platform codebase" thoughtNumber=1 totalThoughts=5 nextThoughtNeeded=true
```

Expected result:
- A structured thinking process with current step description, recommended tools, expected outcome, and next step conditions.

Follow up with a second thought:

```
/tool sequential-thinking thought="Let's examine the architecture and dependencies" thoughtNumber=2 totalThoughts=5 nextThoughtNeeded=true
```

## Step 3: Test Desktop Commander MCP

In Claude Desktop, enter the following command:

```
/tool desktop-commander execute_command {"command": "dir", "timeout": 5000}
```

Expected result:
- A directory listing of the current directory.

Try a more complex command:

```
/tool desktop-commander list_directory {"path": "D:\\MD\\forex_trading_platform"}
```

Expected result:
- A detailed listing of files and directories in your forex trading platform directory.

Test file reading capabilities:

```
/tool desktop-commander read_file {"path": "D:\\MD\\forex_trading_platform\\tools\\fixing\\forex_platform_optimization.md"}
```

Expected result:
- The content of the forex_platform_optimization.md file.

## Step 4: Test Memory MCP

In Claude Desktop, enter the following command:

```
/tool mem0-memory-mcp store {"key": "test_memory", "value": "This is a test of the Memory MCP server for the forex trading platform."}
```

Expected result:
- A confirmation that the memory was stored.

Retrieve the memory:

```
/tool mem0-memory-mcp recall {"key": "test_memory"}
```

Expected result:
- The text you stored earlier.

Store more complex information:

```
/tool mem0-memory-mcp store {"key": "forex_platform_phase", "value": "Currently working on Phase 2: Comprehensive Error Handling and Resilience (High Priority)"}
```

## Step 5: Test MCP Servers in VS Code Insiders

1. Open VS Code Insiders.
2. Open the Command Palette (Ctrl+Shift+P).
3. Type "MCP" and look for MCP-related commands.
4. Try using the MCP servers through the VS Code Insiders interface.

## Troubleshooting

If any of the MCP servers aren't working correctly:

1. Check the Claude Desktop logs:
   - Windows: `%APPDATA%\Claude\logs\latest.log`
   - Look for any error messages related to MCP servers.

2. Check the VS Code Insiders logs:
   - Open the Output panel (Ctrl+Shift+U).
   - Select "Extension Host" from the dropdown.
   - Look for any error messages related to MCP servers.

3. Verify the configuration files:
   - Claude Desktop: `%APPDATA%\Claude\claude_desktop_config.json`
   - VS Code Insiders: `%APPDATA%\Code - Insiders\User\settings.json` and `%APPDATA%\Code - Insiders\User\mcp.json`

4. Check if the MCP servers are installed correctly:
   - Run `npx mcp-sequentialthinking-tools --help` in a terminal.
   - Run `npx @wonderwhy-er/desktop-commander --help` in a terminal.

5. Restart your computer and try again.

## Next Steps

Once you've verified that all MCP servers are working correctly, you can start using them to enhance your forex trading platform development workflow:

1. Use Sequential Thinking MCP to break down complex problems into manageable steps.
2. Use Desktop Commander MCP to manage files, execute commands, and search for code.
3. Use Memory MCP to store and retrieve information across conversations.

These MCP servers will significantly enhance your development workflow, making it more efficient and productive.
