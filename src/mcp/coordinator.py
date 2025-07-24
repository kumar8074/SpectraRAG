# ===================================================================================
# Project: SpectraRAG
# File: src/mcp/coordinator.py
# Description: This file defines the MCP coordinator Agent.
# Author: LALAN KUMAR
# Created: [23-07-2025]
# Updated: [24-07-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
import uuid
import asyncio
from typing import Dict, Any

# Add project root to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.mcp.mcp_agents import MCPIngestionAgent, MCPRetrievalAgent, MCPLLMResponseAgent, MCPGeneralAgent
from src.mcp.message_protocol import MCPMessage, MCPMessageType, get_message_bus
from src.logger import logging

class MCPCoordinator:
    """Coordinator for orchestrating MCP agent communication"""
    
    def __init__(self, session_id: str):
        self.agent_name = "CoordinatorAgent"
        self.session_id = session_id
        self.vector_db_path = f"DATA/vector_db_{session_id}"  # Session-specific vectorDB
        self.retriever_path = f"DATA/retriever_cache_{session_id}"  # Session-specific retriever cache
        self.message_bus = get_message_bus(session_id)  # Use session-specific message bus
        self.ingestion_agent = MCPIngestionAgent(session_id)
        self.retrieval_agent = MCPRetrievalAgent(session_id)
        self.llm_agent = MCPLLMResponseAgent(session_id)
        self.general_agent = MCPGeneralAgent(session_id)
        self.listeners_started = False
    
    async def initialize(self):
        """Initialize all agents"""
        logging.info(f"Initializing agents for session {self.session_id}")
        await self.ingestion_agent.initialize()
        await self.retrieval_agent.initialize()
        await self.llm_agent.initialize()
        await self.general_agent.initialize()
        if not self.listeners_started:
            asyncio.create_task(self.start_agent_listeners())
            self.listeners_started = True
            logging.info(f"Agent listeners started for session {self.session_id}")
    
    async def process_user_query(self, file_path: str, query: str) -> Dict[str, Any]:
        """
        Process user query through MCP agent pipeline:
        - If the user uploads a document (file_path provided), run the full RAG pipeline: Ingestion -> Retrieval -> LLM Response.
        - If no document is uploaded (file_path is None or empty), route the query to MCPGeneralAgent.
        """
        trace_id = str(uuid.uuid4())
        
        logging.info(f"COORDINATOR DEBUG: file_path={file_path}, query={query}")
        logging.info(f"COORDINATOR DEBUG: file_path type={type(file_path)}, bool(file_path)={bool(file_path)}")
        
        try:
            if file_path and query == "__EMBED_ONLY__":
                logging.info("COORDINATOR DEBUG: Embedding-only mode (upload)")
                ingestion_message = MCPMessage(
                    sender=self.agent_name,
                    receiver="IngestionAgent",
                    type=MCPMessageType.INGESTION_REQUEST,
                    trace_id=trace_id,
                    payload={
                        "file_path": file_path,
                        "vector_db_path": self.vector_db_path
                    }
                )
                logging.info(f" [RAG] Sending ingestion request: {ingestion_message.to_dict()}")
                ingestion_response = await self.message_bus.send_and_wait_response(ingestion_message)
                if not ingestion_response or ingestion_response.type == MCPMessageType.ERROR:
                    raise Exception(f"Ingestion failed: {ingestion_response.payload if ingestion_response else 'No response'}")
                logging.info(f" [RAG] Ingestion completed: {ingestion_response.payload}")
                return {
                    "vector_db_ready": ingestion_response.payload.get("vector_db_ready", False),
                    "message": ingestion_response.payload.get("message", "VectorDB ready"),
                    "vector_db_path": ingestion_response.payload.get("vector_db_path", self.vector_db_path),
                    "trace_id": trace_id,
                    "message_history": [msg.to_dict() for msg in self.message_bus.get_message_history(trace_id)]
                }

            if file_path and query != "__EMBED_ONLY__":
                logging.info("COORDINATOR DEBUG: Retrieval/LLM mode (vectorDB ready)")
                retrieval_message = MCPMessage(
                    sender=self.agent_name,
                    receiver="RetrievalAgent",
                    type=MCPMessageType.RETRIEVAL_REQUEST,
                    trace_id=trace_id,
                    payload={
                        "query": query,
                        "vector_db_path": self.vector_db_path,
                        "retriever_path": self.retriever_path
                    }
                )
                logging.info(f" [RAG] Sending retrieval request: {retrieval_message.to_dict()}")
                retrieval_response = await self.message_bus.send_and_wait_response(retrieval_message)
                if not retrieval_response or retrieval_response.type == MCPMessageType.ERROR:
                    raise Exception(f"Retrieval failed: {retrieval_response.payload if retrieval_response else 'No response'}")
                logging.info(f" [RAG] Retrieval completed: Found {retrieval_response.payload['num_docs']} documents")

                llm_message = MCPMessage(
                    sender=self.agent_name,
                    receiver="LLMResponseAgent",
                    type=MCPMessageType.LLM_REQUEST,
                    trace_id=trace_id,
                    payload={
                        "query": query,
                        "retrieved_docs": retrieval_response.payload["retrieved_docs"]
                    }
                )
                logging.info(f"🔄 [RAG] Sending LLM request: {llm_message.to_dict()}")
                llm_response = await self.message_bus.send_and_wait_response(llm_message)
                if not llm_response or llm_response.type == MCPMessageType.ERROR:
                    raise Exception(f"LLM processing failed: {llm_response.payload if llm_response else 'No response'}")
                logging.info(f"✅ [RAG] LLM processing completed")
                return {
                    "answer": llm_response.payload["answer"],
                    "source_context": llm_response.payload.get("source_context", ""),
                    "trace_id": trace_id,
                    "message_history": [msg.to_dict() for msg in self.message_bus.get_message_history(trace_id)]
                }

            if not file_path and query:
                logging.info("COORDINATOR DEBUG: Taking GeneralAgent path")
                general_message = MCPMessage(
                    sender=self.agent_name,
                    receiver="GeneralAgent",
                    type=MCPMessageType.GENERAL_QUERY_REQUEST,
                    trace_id=trace_id,
                    payload={
                        "query": query
                    }
                )
                logging.info(f"🔄 [General] Sending general query request to GeneralAgent")
                general_response = await self.message_bus.send_and_wait_response(general_message)
                logging.info(f"✅ [General] GeneralAgent processing completed")
                return {
                    "answer": general_response.payload["answer"],
                    "trace_id": trace_id,
                    "message_history": [msg.to_dict() for msg in self.message_bus.get_message_history(trace_id)]
                }

            return {"error": "Invalid input to MCP pipeline.", "trace_id": trace_id}
        except Exception as e:
            logging.error(f"❌ Error in MCP pipeline: {str(e)}")
            return {
                "error": str(e),
                "trace_id": trace_id,
                "message_history": [msg.to_dict() for msg in self.message_bus.get_message_history(trace_id)]
            }
    
    async def start_agent_listeners(self):
        """Start background listeners for all agents"""
        async def agent_listener(agent, process_method):
            while True:
                try:
                    message = await agent.message_queue.get()
                    # Skip processing for IngestionAgent if not INGESTION_REQUEST
                    if agent.agent_name == "IngestionAgent" and message.type != MCPMessageType.INGESTION_REQUEST:
                        continue
                    logging.info(f"Processing message for {agent.agent_name}: {message.to_dict()}")
                    response = await process_method(message)
                    await self.message_bus.publish(response)
                except Exception as e:
                    logging.error(f"Error in {agent.agent_name}: {e}")
        
        await asyncio.gather(
            agent_listener(self.ingestion_agent, self.ingestion_agent.process_ingestion_request),
            agent_listener(self.retrieval_agent, self.retrieval_agent.process_retrieval_request),
            agent_listener(self.llm_agent, self.llm_agent.process_llm_request),
            agent_listener(self.general_agent, self.general_agent.process_general_query),
            return_exceptions=True
        )