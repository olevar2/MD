"""
Test MCP Integration

This module provides tests for MCP integration.
"""

import os
import sys
import unittest
import logging
from unittest.mock import patch, MagicMock

from analysis_engine.utils.mcp_integration import MCPIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMCPIntegration(unittest.TestCase):
    """Test MCP integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mcp_integration = MCPIntegration(
            memory_mcp_enabled=True,
            sequential_thinking_enabled=True,
            desktop_commander_enabled=True
        )
    
    def test_store_memory(self):
        """Test storing a memory."""
        result = self.mcp_integration.store_memory("Test memory", "test-user")
        self.assertTrue(result)
    
    def test_retrieve_memories(self):
        """Test retrieving memories."""
        memories = self.mcp_integration.retrieve_memories("Test", "test-user")
        self.assertIsInstance(memories, list)
    
    def test_sequential_thinking(self):
        """Test sequential thinking."""
        steps = self.mcp_integration.sequential_thinking("Test problem", 3)
        self.assertEqual(len(steps), 3)
    
    def test_execute_command(self):
        """Test executing a command."""
        result = self.mcp_integration.execute_command("echo 'Hello, world!'")
        self.assertEqual(result["status"], "success")
    
    def test_read_file(self):
        """Test reading a file."""
        content = self.mcp_integration.read_file("test_file.txt")
        self.assertIsNotNone(content)
    
    def test_write_file(self):
        """Test writing to a file."""
        result = self.mcp_integration.write_file("test_file.txt", "Test content")
        self.assertTrue(result)
    
    def test_search_code(self):
        """Test searching code."""
        results = self.mcp_integration.search_code("test", ".")
        self.assertIsInstance(results, list)
    
    def test_disabled_memory_mcp(self):
        """Test disabled Memory MCP."""
        mcp_integration = MCPIntegration(
            memory_mcp_enabled=False,
            sequential_thinking_enabled=True,
            desktop_commander_enabled=True
        )
        result = mcp_integration.store_memory("Test memory", "test-user")
        self.assertFalse(result)
    
    def test_disabled_sequential_thinking(self):
        """Test disabled Sequential Thinking MCP."""
        mcp_integration = MCPIntegration(
            memory_mcp_enabled=True,
            sequential_thinking_enabled=False,
            desktop_commander_enabled=True
        )
        steps = mcp_integration.sequential_thinking("Test problem", 3)
        self.assertEqual(len(steps), 0)
    
    def test_disabled_desktop_commander(self):
        """Test disabled Desktop Commander MCP."""
        mcp_integration = MCPIntegration(
            memory_mcp_enabled=True,
            sequential_thinking_enabled=True,
            desktop_commander_enabled=False
        )
        result = mcp_integration.execute_command("echo 'Hello, world!'")
        self.assertEqual(result["status"], "error")

if __name__ == "__main__":
    unittest.main()
