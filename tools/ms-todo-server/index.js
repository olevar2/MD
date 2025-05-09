const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const app = express();
const PORT = 3000;

// Middleware
app.use(bodyParser.json());

// Path to the todos file
const todosFilePath = path.join(__dirname, 'todos.json');

// Initialize todos file if it doesn't exist
if (!fs.existsSync(todosFilePath)) {
  fs.writeFileSync(todosFilePath, JSON.stringify({ todos: [] }));
}

// Helper function to read todos
function readTodos() {
  const data = fs.readFileSync(todosFilePath, 'utf8');
  return JSON.parse(data);
}

// Helper function to write todos
function writeTodos(data) {
  fs.writeFileSync(todosFilePath, JSON.stringify(data, null, 2));
}

// MCP Info endpoint
app.get('/mcp/info', (req, res) => {
  res.json({
    name: 'MS-Todo',
    version: '1.0.0',
    description: 'A simple MS-Todo MCP server',
    resources: [
      {
        name: 'ms-todo://',
        description: 'Get all todos'
      }
    ],
    tools: [
      {
        name: 'create_todo',
        description: 'Create a new todo'
      },
      {
        name: 'update_todo',
        description: 'Update an existing todo'
      },
      {
        name: 'delete_todo',
        description: 'Delete a todo'
      },
      {
        name: 'complete_todo',
        description: 'Mark a todo as completed'
      }
    ]
  });
});

// Get all todos
app.get('/todos', (req, res) => {
  const data = readTodos();
  res.json(data.todos);
});

// Create a new todo
app.post('/todos', (req, res) => {
  const { title, description, deadline } = req.body;
  
  if (!title) {
    return res.status(400).json({ error: 'Title is required' });
  }
  
  const data = readTodos();
  const newTodo = {
    id: `todo-${Date.now()}`,
    title,
    description: description || '',
    deadline: deadline || null,
    completed: false,
    created_at: new Date().toISOString()
  };
  
  data.todos.push(newTodo);
  writeTodos(data);
  
  res.status(201).json(newTodo);
});

// Get a specific todo
app.get('/todos/:id', (req, res) => {
  const { id } = req.params;
  const data = readTodos();
  const todo = data.todos.find(todo => todo.id === id);
  
  if (!todo) {
    return res.status(404).json({ error: 'Todo not found' });
  }
  
  res.json(todo);
});

// Update a todo
app.put('/todos/:id', (req, res) => {
  const { id } = req.params;
  const { title, description, deadline, completed } = req.body;
  const data = readTodos();
  const todoIndex = data.todos.findIndex(todo => todo.id === id);
  
  if (todoIndex === -1) {
    return res.status(404).json({ error: 'Todo not found' });
  }
  
  const todo = data.todos[todoIndex];
  
  if (title !== undefined) todo.title = title;
  if (description !== undefined) todo.description = description;
  if (deadline !== undefined) todo.deadline = deadline;
  if (completed !== undefined) todo.completed = completed;
  
  todo.updated_at = new Date().toISOString();
  
  data.todos[todoIndex] = todo;
  writeTodos(data);
  
  res.json(todo);
});

// Delete a todo
app.delete('/todos/:id', (req, res) => {
  const { id } = req.params;
  const data = readTodos();
  const todoIndex = data.todos.findIndex(todo => todo.id === id);
  
  if (todoIndex === -1) {
    return res.status(404).json({ error: 'Todo not found' });
  }
  
  data.todos.splice(todoIndex, 1);
  writeTodos(data);
  
  res.json({ message: `Todo ${id} deleted` });
});

// Start the server
app.listen(PORT, () => {
  console.log(`MS-Todo MCP Server running at http://localhost:${PORT}`);
});
