#!/usr/bin/env python
"""
Script to start the Sequential Thinking Server locally
"""

import mcp
from mcp.server.fastmcp import FastMCP
import asyncio
import uvicorn
import sys
from typing import Dict, Any, List, Optional

class SequentialThinking:
    """
    Sequential Thinking tool implementation
    """
    def __init__(self):
        self.thought_history = []
        self.branches = {}

    def process_thought(self,
                        thought: str,
                        next_thought_needed: bool,
                        thought_number: int,
                        total_thoughts: int,
                        is_revision: Optional[bool] = None,
                        revises_thought: Optional[int] = None,
                        branch_from_thought: Optional[int] = None,
                        branch_id: Optional[str] = None,
                        needs_more_thoughts: Optional[bool] = None) -> Dict[str, Any]:
        """
        Process a thought and return the result
        """
        # Store the thought in history
        if thought_number <= len(self.thought_history):
            self.thought_history[thought_number - 1] = thought
        else:
            self.thought_history.append(thought)

        # Handle branching
        if branch_id and branch_from_thought:
            if branch_id not in self.branches:
                self.branches[branch_id] = []

            if thought_number <= len(self.branches[branch_id]):
                self.branches[branch_id][thought_number - 1] = thought
            else:
                self.branches[branch_id].append(thought)

        # Return the result
        result = {
            "thoughtNumber": thought_number,
            "totalThoughts": total_thoughts,
            "nextThoughtNeeded": next_thought_needed,
            "branches": list(self.branches.keys()),
            "thoughtHistoryLength": len(self.thought_history)
        }

        if is_revision is not None:
            result["isRevision"] = is_revision

        if revises_thought is not None:
            result["revisesThought"] = revises_thought

        if branch_from_thought is not None:
            result["branchFromThought"] = branch_from_thought

        if branch_id is not None:
            result["branchId"] = branch_id

        if needs_more_thoughts is not None:
            result["needsMoreThoughts"] = needs_more_thoughts

        return result

def create_server():
    """
    Create and configure the Sequential Thinking Server
    """
    mcp_server = FastMCP("Sequential Thinking Server")
    sequential_thinking = SequentialThinking()

    @mcp_server.tool()
    def sequentialthinking(
        thought: str,
        nextThoughtNeeded: bool,
        thoughtNumber: int,
        totalThoughts: int,
        isRevision: Optional[bool] = None,
        revisesThought: Optional[int] = None,
        branchFromThought: Optional[int] = None,
        branchId: Optional[str] = None,
        needsMoreThoughts: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        A detailed tool for dynamic and reflective problem-solving through thoughts.
        This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
        Each thought can build on, question, or revise previous insights as understanding deepens.
        """
        return sequential_thinking.process_thought(
            thought=thought,
            next_thought_needed=nextThoughtNeeded,
            thought_number=thoughtNumber,
            total_thoughts=totalThoughts,
            is_revision=isRevision,
            revises_thought=revisesThought,
            branch_from_thought=branchFromThought,
            branch_id=branchId,
            needs_more_thoughts=needsMoreThoughts
        )

    return mcp_server

def main():
    """
    Start the Sequential Thinking Server
    """
    server = create_server()

    # Start the server using uvicorn directly
    import uvicorn
    uvicorn.run(
        server.app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    print("Starting Sequential Thinking Server on http://127.0.0.1:8000")
    main()
