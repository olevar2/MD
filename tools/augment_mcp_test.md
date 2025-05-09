# Augment MCP Integration Test

This file contains test commands to verify that Augment can use the MCP tools.

## Todo MCP Server Tools

To create a new todo:
```
/tool create_todo --title "Implement error handling" --description "Add comprehensive error handling to the trading gateway service"
```

To list all todos:
```
/tool todo://
```

To mark a todo as completed:
```
/tool complete_todo --id 1
```

## MS-Todo MCP Server Tools

To create a new todo in MS-Todo:
```
/tool ms-todo://create_todo --title "Refactor large files" --description "Break down large files into smaller, more maintainable components"
```

To list all todos in MS-Todo:
```
/tool ms-todo://
```

## Conv-Tasks MCP Server Tools

To create a new project:
```
/tool createProject --name "Forex Platform Optimization" --description "Optimize the forex trading platform for better performance and maintainability"
```

To add a task to the project:
```
/tool addTask --projectId 1 --title "Resolve circular dependencies" --description "Identify and resolve circular dependencies between services" --priority "high"
```

To list all tasks:
```
/tool listTasks --projectId 1
```

## Sequential Thinking MCP Server Tools

To create a sequential thinking plan:
```
/tool sequentialthinking://create_plan --title "Forex Platform Optimization Plan" --description "Create a step-by-step plan to optimize the forex trading platform"
```

To list available plans:
```
/tool sequentialthinking://list_plans
```

To execute a plan:
```
/tool sequentialthinking://execute_plan --plan_id 1
```

## Desktop Commander MCP Server Tools

To list available commands:
```
/tool desktop-commander://list_commands
```

To open a file:
```
/tool desktop-commander://open_file --path "D:/MD/forex_trading_platform/tools/fixing/forex_platform_optimization.md"
```

To execute a shell command:
```
/tool desktop-commander://execute_command --command "dir"
```

## Testing Procedure

1. Open this file in VS Code Insiders
2. Use Augment to execute the commands above
3. Verify that Augment can successfully use the MCP tools
4. If all commands work, Augment is properly integrated with the MCP servers
