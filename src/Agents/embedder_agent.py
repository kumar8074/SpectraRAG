# ===================================================================================
# Project: SpectraRAG
# File: src/Agents/embedder_agent.py
# Description: This file creates the embedder Agent. 
# Author: LALAN KUMAR
# Created: [22-07-2025]
# Updated: [22-07-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
from langchain_community.document_loaders import (PDFPlumberLoader, TextLoader, Docx2txtLoader, CSVLoader, 
                                                  UnstructuredPowerPointLoader)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.components.states import EmbederState, EmbedderInput, EmbedderOutput
from src.utils import get_embeddings
    
# Check if file exists
def check_file_exists(state: EmbederState)-> EmbederState:
    """Check if the file exists in the specified path."""
    path=state.file_path
    state.file_exists=os.path.exists(path)
    return state

# Load documents
def load_documents(state: EmbederState)-> EmbederState:
    """Load documents based on the file type."""
    if not state.file_exists:
        raise FileNotFoundError(f"File Not Found: {state.file_path}")
    path=state.file_path
    ext=path.split('.')[-1].lower()
    if ext=="pdf":
        loader=PDFPlumberLoader(path)
    elif ext=="docx":
        loader=Docx2txtLoader(path)
    elif ext in ["txt", "md", "log"]:
        loader=TextLoader(path)
    elif ext in ["csv"]:
        loader=CSVLoader(path)
    elif ext in ["pptx", "ppt"]:
        loader=UnstructuredPowerPointLoader(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    docs=list(loader.load())
    state.documents= docs
    return state

# Split documents into chunks
def split_documents(state: EmbederState)-> EmbederState:
    docs=state.documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = splitter.split_documents(docs)
    state.documents = split_documents
    return state

# Embed documents and persist to vector database
def embed_and_persist(state: EmbederState)-> EmbederState:
    split_docs=state.documents
    vector_db_path=state.vector_db_path
    embedding_provider=state.embedding_provider
    
    # Get embeddings instance
    embeddings = get_embeddings(provider=embedding_provider)
    
    # Create directory if it doesn't exist
    os.makedirs(vector_db_path, exist_ok=True)
    
    vector_db=Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings, 
        persist_directory=vector_db_path
    )
    state.vector_db_ready=True
        
    return state

# Finalize and return MCP output
def finalize(state: EmbederState)-> EmbedderOutput:
    if state.vector_db_ready:
        return {"success": True, "message": "VectorDB created successfully"}
    else:
        return {"success": False, "message": "VectorDB creation failed"}
    
# Create the Graph
def create_embedder_agent():
    workflow = StateGraph(EmbederState, input_schema=EmbedderInput, output_schema=EmbedderOutput)

    workflow.add_node("check_exists", check_file_exists)
    workflow.add_node("load_doc", load_documents)
    workflow.add_node("split_doc", split_documents)  # new splitting node
    workflow.add_node("embed_and_persist", embed_and_persist)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "check_exists")
    workflow.add_conditional_edges(
        "check_exists",
        lambda state: "continue" if state.file_exists else "finalize",
        {
            "continue": "load_doc",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("load_doc", "split_doc")  # split after loading
    workflow.add_edge("split_doc", "embed_and_persist")
    workflow.add_edge("embed_and_persist", "finalize")
    workflow.add_edge("finalize", END)

    embedder_graph = workflow.compile()
    embedder_graph.name = "EmbedderAgent"
    
    return embedder_graph


# ===================================================================================
# Example usage:
#embedder_graph = create_embedder_agent()
#embedder_input=EmbedderInput(
    #file_path="temp/presentation-batch-17.pptx",
    #vector_db_path="DATA/vector_db"
#)

# Create initial state from input
#initial_state = EmbederState(**embedder_input)

# Invoke the embedder graph
#result_state = embedder_graph.invoke(initial_state)

#print("Embedder Agent Output:")
#print(result_state)


