# ===================================================================================
# Project: SpectraRAG
# File: src/mcp/mcp_agents.py
# Description: This file wraps the Agents to enable MCP communication.
# Author: LALAN KUMAR
# Created: [23-07-2025]
# Updated: [24-07-2025]
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

from src.mcp.message_protocol import MCPMessage, MCPMessageType, get_message_bus
from src.Agents.embedder_agent import create_embedder_agent, EmbederState, EmbedderInput
from src.Agents.retriever_agent import create_retriever_agent, RetrieverState, RetrieverInput
from src.Agents.response_agent import create_llm_response_agent, LLMResponseState, LLMResponseInput
from src.Agents.general_agent import create_general_agent, GeneralAgentInput

class MCPIngestionAgent:
    """MCP-enabled Ingestion Agent wrapper"""
    
    def __init__(self, session_id: str):
        self.agent_name = "IngestionAgent"
        self.session_id = session_id
        self.embedder_graph = create_embedder_agent()
        self.message_queue = None
    
    async def initialize(self):
        """Initialize MCP communication"""
        message_bus = get_message_bus(self.session_id)
        self.message_queue = await message_bus.subscribe(self.agent_name)
    
    async def process_ingestion_request(self, message: MCPMessage) -> MCPMessage:
        """Process document ingestion request"""
        try:
            file_path = message.payload.get("file_path")
            vector_db_path = message.payload.get("vector_db_path", f"DATA/vector_db_{self.session_id}")
            
            embedder_input = EmbedderInput(
                file_path=file_path,
                vector_db_path=vector_db_path
            )
            
            embedder_state = EmbederState(**embedder_input)
            result = self.embedder_graph.invoke(embedder_state)
            
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
    
    def __init__(self, session_id: str):
        self.agent_name = "RetrievalAgent"
        self.session_id = session_id
        self.retriever_graph = create_retriever_agent()
        self.message_queue = None
    
    async def initialize(self):
        """Initialize MCP communication"""
        message_bus = get_message_bus(self.session_id)
        self.message_queue = await message_bus.subscribe(self.agent_name)
    
    async def process_retrieval_request(self, message: MCPMessage) -> MCPMessage:
        """Process document retrieval request"""
        try:
            query = message.payload.get("query")
            vector_db_path = message.payload.get("vector_db_path")
            retriever_path = message.payload.get("retriever_path", f"DATA/retriever_cache_{self.session_id}")
            
            retriever_input = RetrieverInput(
                vector_db_path=vector_db_path,
                retriever_path=retriever_path,
                query=query
            )
            
            retriever_state = RetrieverState(
                vector_db_path=retriever_input["vector_db_path"],
                retriever_path=retriever_input["retriever_path"],
                query=retriever_input["query"],
                retriever_ready=False,
                retrieved_docs=[]
            )
            
            result = await self.retriever_graph.ainvoke(retriever_state)
            
            retrieved_docs = result.get("retrieved_docs", [])
            serialized_docs = []
            for doc in retrieved_docs:
                serialized_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
            
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
            return MCPMessage(
                sender=self.agent_name,
                receiver=message.sender,
                type=MCPMessageType.ERROR,
                trace_id=message.trace_id,
                payload={"error": str(e)}
            )

class MCPLLMResponseAgent:
    """MCP-enabled LLM Response Agent wrapper"""
    
    def __init__(self, session_id: str):
        self.agent_name = "LLMResponseAgent"
        self.session_id = session_id
        self.llm_graph = create_llm_response_agent()
        self.message_queue = None
    
    async def initialize(self):
        """Initialize MCP communication"""
        message_bus = get_message_bus(self.session_id)
        self.message_queue = await message_bus.subscribe(self.agent_name)
    
    async def process_llm_request(self, message: MCPMessage) -> MCPMessage:
        """Process LLM response generation request"""
        try:
            query = message.payload.get("query")
            retrieved_docs_data = message.payload.get("retrieved_docs", [])
            
            retrieved_docs = []
            for doc_data in retrieved_docs_data:
                doc = Document(
                    page_content=doc_data["page_content"],
                    metadata=doc_data["metadata"]
                )
                retrieved_docs.append(doc)
            
            llm_input = LLMResponseInput(
                query=query,
                retrieved_docs=retrieved_docs
            )
            
            llm_state = LLMResponseState(**llm_input)
            result = self.llm_graph.invoke(llm_state)
            
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
            return MCPMessage(
                sender=self.agent_name,
                receiver=message.sender,
                type=MCPMessageType.ERROR,
                trace_id=message.trace_id,
                payload={"error": str(e)}
            )

class MCPGeneralAgent:
    """MCP-enabled General Agent wrapper for general queries"""
    
    def __init__(self, session_id: str):
        self.agent_name = "GeneralAgent"
        self.session_id = session_id
        self.general_graph = create_general_agent()
        self.message_queue = None

    async def initialize(self):
        """Initialize MCP communication"""
        message_bus = get_message_bus(self.session_id)
        self.message_queue = await message_bus.subscribe(self.agent_name)

    async def process_general_query(self, message: MCPMessage) -> MCPMessage:
        """Process general query request"""
        try:
            query = message.payload.get("query")
            agent_input = GeneralAgentInput(query=query)
            result = await self.general_graph.ainvoke(agent_input)
            answer = result["response"].content if hasattr(result["response"], "content") else str(result["response"])
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
            return MCPMessage(
                sender=self.agent_name,
                receiver=message.sender,
                type=MCPMessageType.ERROR,
                trace_id=message.trace_id,
                payload={"error": str(e)}
            )