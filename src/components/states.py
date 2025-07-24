# ===================================================================================
# Project: SpectraRAG
# File: src/components/states.py
# Description: This file defines the state schemas used by the Agents. 
# Author: LALAN KUMAR
# Created: [22-07-2025]
# Updated: [24-07-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
from pydantic import BaseModel, Field
from typing import TypedDict, List, Annotated
from langchain.schema import Document

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.utils import reduce_docs

class BaseVectorDBState(BaseModel):
    vector_db_path: str
    vector_db_ready: bool = False
    embedding_provider: str = "gemini"
    llm_provider: str = "gemini"
    session_id: str = ""  # Added session_id

class EmbederState(BaseVectorDBState):
    file_path: str
    file_exists: bool = False
    documents: list[Document] = []
    
class EmbedderInput(TypedDict):
    file_path: str
    vector_db_path: str
    session_id: str  # Added session_id

class EmbedderOutput(TypedDict):
    success: bool
    message: str
    
class RetrieverState(BaseVectorDBState):
    retriever_path: str
    retriever_ready: bool = False
    query: str
    generated_queries: list[str] = Field(default_factory=list)
    retrieved_docs: Annotated[list[Document], reduce_docs] = Field(default_factory=list)

class GeneratedQueries(BaseModel):
    queries: List[str]
    
class RetrieverInput(TypedDict):
    vector_db_path: str
    retriever_path: str
    query: str
    session_id: str  # Added session_id
    
class RetrieverOutput(TypedDict):
    retrieved_docs: List[Document]
    
class LLMResponseInput(TypedDict):
    query: str
    retrieved_docs: List[Document]

class LLMResponseOutput(TypedDict):
    answer: str
    source_context: str

class LLMResponseState(BaseModel):
    query: str
    retrieved_docs: List[Document] = Field(default_factory=list)
    answer: str = ""
    source_context: str = ""
    
class GeneralAgentInput(TypedDict):
    query: str
    response: str