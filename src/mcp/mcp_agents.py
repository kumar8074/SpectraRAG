# ===================================================================================
# Project: SpectraRAG
# File: src/mcp/mcp_agents.py
# Description: This file wraps the Agents to enable MCP communication.
# Author: LALAN KUMAR
# Created: [23-07-2025]
# Updated: [23-07-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
from typing import Dict, Any, List
import traceback
from langchain.schema import Document

# Add project root to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.mcp.message_protocol import MCPMessage, MCPMessageType, message_bus
from src.Agents.embedder_agent import create_embedder_agent, EmbederState, EmbedderInput
from src.Agents.retriever_agent import create_retriever_agent, RetrieverState, RetrieverInput
from src.Agents.response_agent import create_llm_response_agent, LLMResponseState, LLMResponseInput
from src.Agents.general_agent import create_general_agent, GeneralAgentInput

class MCPIngestionAgent:
    """MCP-enabled Ingestion Agent wrapper"""
    
    def __init__(self):
        self.agent_name = "IngestionAgent"
        self.embedder_graph = create_embedder_agent()
        self.message_queue = None
    
    async def initialize(self):
        """Initialize MCP communication"""
        self.message_queue = await message_bus.subscribe(self.agent_name)
    
    async def process_ingestion_request(self, message: MCPMessage) -> MCPMessage:
        """Process document ingestion request"""
        try:
            # Extract payload
            file_path = message.payload.get("file_path")
            vector_db_path = message.payload.get("vector_db_path", "DATA/vector_db")
            
            # Create embedder input
            embedder_input = EmbedderInput(
                file_path=file_path,
                vector_db_path=vector_db_path
            )
            
            # Process through embedder graph
            embedder_state = EmbederState(**embedder_input)
            result = self.embedder_graph.invoke(embedder_state)
            
            # Create response message
            response = MCPMessage(
                sender=self.agent_name,
                receiver=message.sender,
                type=MCPMessageType.INGESTION_RESPONSE,
                trace_id=message.trace_id,
                payload={
                    "success": result.get("success", False),
                    "message": result.get("message", ""),
                    "vector_db_ready": True,
                    "vector_db_path": vector_db_path,
                    "documents_processed": len(embedder_state.documents) if hasattr(embedder_state, 'documents') else 0
                }
            )
            
            return response
            
        except Exception as e:
            # Return error response
            return MCPMessage(
                sender=self.agent_name,
                receiver=message.sender,
                type=MCPMessageType.ERROR,
                trace_id=message.trace_id,
                payload={
                    "error": str(e),
                    "trace": traceback.format_exc()
                }
            )


class MCPRetrievalAgent:
    """MCP-enabled Retrieval Agent wrapper"""
    
    def __init__(self):
        self.agent_name = "RetrievalAgent"
        self.retriever_graph = create_retriever_agent()
        self.message_queue = None
    
    async def initialize(self):
        """Initialize MCP communication"""
        self.message_queue = await message_bus.subscribe(self.agent_name)
    
    async def process_retrieval_request(self, message: MCPMessage) -> MCPMessage:
        """Process document retrieval request"""
        try:
            # Extract payload
            query = message.payload.get("query")
            vector_db_path = message.payload.get("vector_db_path")
            retriever_path = message.payload.get("retriever_path", "DATA/retriever_cache")
            
            # Create retriever input
            retriever_input = RetrieverInput(
                vector_db_path=vector_db_path,
                retriever_path=retriever_path,
                query=query
            )
            
            # Process through retriever graph
            retriever_state = RetrieverState(
                vector_db_path=retriever_input["vector_db_path"],
                retriever_path=retriever_input["retriever_path"],
                query=retriever_input["query"],
                retriever_ready=False,
                retrieved_docs=[]
            )
            
            result = await self.retriever_graph.ainvoke(retriever_state)
            
            # Serialize documents for MCP transfer
            retrieved_docs = result.get("retrieved_docs", [])
            serialized_docs = []
            for doc in retrieved_docs:
                serialized_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            # Create response message
            response = MCPMessage(
                sender=self.agent_name,
                receiver=message.sender,
                type=MCPMessageType.RETRIEVAL_RESPONSE,
                trace_id=message.trace_id,
                payload={
                    "retrieved_docs": serialized_docs,
                    "query": query,
                    "num_docs": len(serialized_docs)
                }
            )
            
            return response
            
        except Exception as e:
            # Return error response
            return MCPMessage(
                sender=self.agent_name,
                receiver=message.sender,
                type=MCPMessageType.ERROR,
                trace_id=message.trace_id,
                payload={"error": str(e)}
            )

class MCPLLMResponseAgent:
    """MCP-enabled LLM Response Agent wrapper"""
    
    def __init__(self):
        self.agent_name = "LLMResponseAgent"
        self.llm_graph = create_llm_response_agent()
        self.message_queue = None
    
    async def initialize(self):
        """Initialize MCP communication"""
        self.message_queue = await message_bus.subscribe(self.agent_name)
    
    async def process_llm_request(self, message: MCPMessage) -> MCPMessage:
        """Process LLM response generation request"""
        try:
            # Extract payload
            query = message.payload.get("query")
            retrieved_docs_data = message.payload.get("retrieved_docs", [])
            
            # Deserialize documents
            retrieved_docs = []
            for doc_data in retrieved_docs_data:
                doc = Document(
                    page_content=doc_data["page_content"],
                    metadata=doc_data["metadata"]
                )
                retrieved_docs.append(doc)
            
            # Create LLM input
            llm_input = LLMResponseInput(
                query=query,
                retrieved_docs=retrieved_docs
            )
            
            # Process through LLM graph
            llm_state = LLMResponseState(**llm_input)
            result = self.llm_graph.invoke(llm_state)
            
            # Create response message
            response = MCPMessage(
                sender=self.agent_name,
                receiver=message.sender,
                type=MCPMessageType.LLM_RESPONSE,
                trace_id=message.trace_id,
                payload={
                    "answer": result.get("answer", ""),
                    "source_context": result.get("source_context", ""),
                    "query": query
                }
            )
            
            return response
            
        except Exception as e:
            # Return error response
            return MCPMessage(
                sender=self.agent_name,
                receiver=message.sender,
                type=MCPMessageType.ERROR,
                trace_id=message.trace_id,
                payload={"error": str(e)}
            )


class MCPGeneralAgent:
    """MCP-enabled General Agent wrapper for general queries"""
    def __init__(self):
        self.agent_name = "GeneralAgent"
        self.general_graph = create_general_agent()
        self.message_queue = None

    async def initialize(self):
        """Initialize MCP communication"""
        self.message_queue = await message_bus.subscribe(self.agent_name)

    async def process_general_query(self, message: MCPMessage) -> MCPMessage:
        """Process general query request"""
        try:
            # Extract payload
            query = message.payload.get("query")
            # Prepare input for the general agent
            agent_input = GeneralAgentInput(query=query)
            # Run the agent graph (async)
            result = await self.general_graph.ainvoke(agent_input)
            answer = result["response"].content if hasattr(result["response"], "content") else str(result["response"])
            # Create response message
            response = MCPMessage(
                sender=self.agent_name,
                receiver=message.sender,
                type=getattr(MCPMessageType, "GENERAL_QUERY_RESPONSE", MCPMessageType.LLM_RESPONSE),
                trace_id=message.trace_id,
                payload={
                    "answer": answer,
                    "query": query
                }
            )
            return response
        except Exception as e:
            # Return error response
            return MCPMessage(
                sender=self.agent_name,
                receiver=message.sender,
                type=MCPMessageType.ERROR,
                trace_id=message.trace_id,
                payload={"error": str(e)}
            )