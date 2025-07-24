# ===================================================================================
# Project: SpectraRAG
# File: src/mcp/message_protocol.py
# Description: This file defines the message protocol used by the Agents.
# Author: LALAN KUMAR
# Created: [23-07-2025]
# Updated: [23-07-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import uuid
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from enum import Enum
import asyncio
from dataclasses import dataclass, field

class MCPMessageType(Enum):
    """Define MCP message types for inter-agent communication"""
    INGESTION_REQUEST = "INGESTION_REQUEST"
    INGESTION_RESPONSE = "INGESTION_RESPONSE"
    RETRIEVAL_REQUEST = "RETRIEVAL_REQUEST" 
    RETRIEVAL_RESPONSE = "RETRIEVAL_RESPONSE"
    CONTEXT_REQUEST = "CONTEXT_REQUEST"
    CONTEXT_RESPONSE = "CONTEXT_RESPONSE"
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    GENERAL_QUERY_REQUEST = "GENERAL_QUERY_REQUEST"
    GENERAL_QUERY_RESPONSE = "GENERAL_QUERY_RESPONSE"
    ERROR = "ERROR"

@dataclass
class MCPMessage:
    """MCP Message structure for inter-agent communication"""
    sender: str
    receiver: str
    type: MCPMessageType
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type.value,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "metadata": self.metadata
        }

class MCPMessageBus:
    """In-memory message bus for MCP communication"""
    
    def __init__(self):
        self.message_queue: Dict[str, List[MCPMessage]] = {}
        self.subscribers: Dict[str, asyncio.Queue] = {}
        self.message_history: List[MCPMessage] = []
        self._lock = asyncio.Lock()
    
    async def subscribe(self, agent_name: str) -> asyncio.Queue:
        """Subscribe an agent to receive messages"""
        async with self._lock:
            if agent_name not in self.subscribers:
                self.subscribers[agent_name] = asyncio.Queue()
            return self.subscribers[agent_name]
    
    async def publish(self, message: MCPMessage) -> bool:
        """Publish a message to the bus"""
        async with self._lock:
            # Add to history
            self.message_history.append(message)
            
            # Route to specific receiver
            if message.receiver in self.subscribers:
                await self.subscribers[message.receiver].put(message)
                return True
            else:
                print(f"Warning: No subscriber for {message.receiver}")
                return False
    
    async def send_and_wait_response(self, message: MCPMessage, timeout: float = 30.0) -> Optional[MCPMessage]:
        """Send message and wait for response"""
        # Subscribe to responses for this sender
        response_queue = await self.subscribe(message.sender)
        
        # Send the message
        await self.publish(message)
        
        # Wait for response with the same trace_id
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = await asyncio.wait_for(response_queue.get(), timeout=1.0)
                    if response.trace_id == message.trace_id:
                        return response
                    else:
                        # Put back if not our response
                        await response_queue.put(response)
                except asyncio.TimeoutError:
                    continue
            return None
        except Exception as e:
            print(f"Error waiting for response: {e}")
            return None
    
    def get_message_history(self, trace_id: Optional[str] = None) -> List[MCPMessage]:
        """Get message history, optionally filtered by trace_id"""
        if trace_id:
            return [msg for msg in self.message_history if msg.trace_id == trace_id]
        return self.message_history.copy()

# Global message bus instance
message_bus = MCPMessageBus()

