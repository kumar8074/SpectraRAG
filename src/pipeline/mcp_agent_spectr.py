# ===================================================================================
# Project: SpectraRAG
# File: src/pipeline/mcp_agent_spectr.py
# Description: This file orchestrates the MCP agents to enable MCP communication.
# Author: LALAN KUMAR
# Created: [23-07-2025]
# Updated: [24-07-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
import asyncio
import uuid
from typing import Dict, Any

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.mcp.coordinator import MCPCoordinator
from src.logger import logging

class MCPSpectraRagController:
    """Enhanced RAG Controller with MCP communication"""
    
    def __init__(self, session_id: str):
        self.coordinator = MCPCoordinator(session_id)
        self.session_id = session_id
        self.initialized = False
    
    async def initialize(self):
        """Initialize all MCP agents"""
        if not self.initialized:
            await self.coordinator.initialize()
            self.initialized = True
            logging.info("MCP RAG Controller initialized")
    
    async def run(self, file_path: str, query: str) -> Dict[str, Any]:
        """Run the complete MCP-enabled RAG pipeline with doc upload, RAG, or general query modes"""
        if not self.initialized:
            await self.initialize()

        logging.info(f"Processing file: {file_path}")
        logging.info(f"Query: {query}")
        logging.info("=" * 60)

        result = await self.coordinator.process_user_query(file_path, query)

        logging.info("=" * 60)
        logging.info("MCP Message Flow Summary:")
        if "message_history" in result:
            for i, msg in enumerate(result["message_history"], 1):
                logging.info(f"{i}. {msg['sender']} → {msg['receiver']}: {msg['type']}")

        if result.get("vector_db_ready"):
            return {
                "vector_db_ready": True,
                "message": result.get("message", "VectorDB ready"),
                "trace_id": result.get("trace_id"),
                "message_history": result.get("message_history", [])
            }
        elif result.get("answer"):
            return {
                "answer": result["answer"],
                "source_context": result.get("source_context", ""),
                "trace_id": result.get("trace_id"),
                "message_history": result.get("message_history", [])
            }
        elif result.get("error"):
            return result
        else:
            return {"error": "Unknown controller result", **result}

# Example usage with proper MCP message tracing
#async def main():
    #controller = MCPSpectraRagController()
    
    # Test with your sample file
    #file_path = "temp/README.md"  # Replace with your actual file
    #query = "Can you tell me the project structure used?"
    
    #try:
        #result = await controller.run(file_path, query)
        
        #if "error" in result:
            #logging.error(f" Error: {result['error']}")
        #else:
            #logging.info("\n Final Answer:")
            #logging.info("-" * 40)
            #logging.info(result["answer"])
            
            #logging.info(f"\n Trace ID: {result['trace_id']}")
            
            # Optionally show source context
            #if "source_context" in result and result["source_context"]:
                #logging.info("\n Source Context:")
                #logging.info("-" * 40)
                #logging.info(result["source_context"][:500] + "..." if len(result["source_context"]) > 500 else result["source_context"])
    
    #except Exception as e:
        #logging.error(f"Pipeline error: {e}")

#if __name__ == "__main__":
    #asyncio.run(main())