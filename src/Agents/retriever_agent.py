# ===================================================================================
# Project: SpectraRAG
# File: src/Agents/retriever_agent.py
# Description: This file creates the Retriever Agent. 
# Author: LALAN KUMAR
# Created: [22-07-2025]
# Updated: [24-07-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
import asyncio
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.components.states import RetrieverState, RetrieverInput, RetrieverOutput, GeneratedQueries
from src.utils import get_embeddings, get_llm
from src.components.prompts import GENERATE_QUERIES_SYSTEM_PROMPT

# Session-specific retriever cache
retriever_cache = {}  # Keyed by session_id_retriever_path

async def check_vector_db(state: RetrieverState) -> RetrieverState:
    """Check if the vector DB exists and is non-empty."""
    path = state.vector_db_path
    state.vector_db_ready = os.path.exists(path) and bool(os.listdir(path))
    return state

async def create_retriever(state: RetrieverState) -> RetrieverState:
    if not state.vector_db_ready:
        raise FileNotFoundError(f"VectorDB not found or empty at {state.vector_db_path}")
    
    cache_key = f"{state.session_id}_{state.retriever_path}"
    if cache_key not in retriever_cache:
        embeddings = get_embeddings(state.embedding_provider)
        if embeddings is None:
            raise ValueError(f"Unsupported embedding provider: {state.embedding_provider}")
        
        vector_db = Chroma(
            persist_directory=state.vector_db_path, 
            embedding_function=embeddings
        )
        retriever_cache[cache_key] = vector_db.as_retriever()
    
    state.retriever_ready = True
    return state

async def generate_queries(state: RetrieverState) -> RetrieverState:
    """Generate search queries based on the question."""
    llm = get_llm(state.llm_provider)
    structured_llm = llm.with_structured_output(GeneratedQueries)
    messages = [
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
        {"role": "human", "content": state.query}
    ]
    response = await structured_llm.ainvoke(messages)
    state.generated_queries = response.queries
    return state

async def retrieve_single_query(retriever, query):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, retriever.invoke, query)

async def retrieve_in_parallel(state: RetrieverState) -> RetrieverState:
    if not state.retriever_ready:
        raise RuntimeError("Retriever not ready")

    cache_key = f"{state.session_id}_{state.retriever_path}"
    retriever = retriever_cache.get(cache_key)
    if retriever is None:
        raise RuntimeError("Retriever instance missing")

    queries = state.generated_queries or [state.query]
    tasks = [retrieve_single_query(retriever, q) for q in queries]
    results = await asyncio.gather(*tasks)

    all_docs = [doc for docs in results for doc in docs]
    unique_docs = {}
    for doc in all_docs:
        key = doc.metadata.get("uuid") or doc.page_content
        unique_docs[key] = doc

    state.retrieved_docs = list(unique_docs.values())
    return state

def finalize(state: RetrieverState) -> RetrieverOutput:
    return {"retrieved_docs": state.retrieved_docs}

def create_retriever_agent():
    builder = StateGraph(RetrieverState, input_schema=RetrieverInput, output_schema=RetrieverOutput)

    builder.add_node("check_db", check_vector_db)
    builder.add_node("load_retriever", create_retriever)
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("retrieve_in_parallel", retrieve_in_parallel, is_async=True)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "check_db")
    builder.add_conditional_edges(
        "check_db",
        lambda state: "load_retriever" if state.vector_db_ready else "finalize",
        {
            "load_retriever": "load_retriever",
            "finalize": "finalize",
        },
    )
    builder.add_edge("load_retriever", "generate_queries")
    builder.add_edge("generate_queries", "retrieve_in_parallel")
    builder.add_edge("retrieve_in_parallel", "finalize")
    builder.add_edge("finalize", END)

    retriever_graph = builder.compile()
    retriever_graph.name = "RetrieverAgent"

    return retriever_graph


# ===================================================================================
# Example usage:
#async def main():
    #retriever_graph = create_retriever_agent()
    #retriever_input = RetrieverInput(
        #vector_db_path="DATA/vector_db",
        #retriever_path="DATA/retriever",
        #query="What is proposed methodology?"
    #)
    #initial_state = RetrieverState(
        #vector_db_path=retriever_input["vector_db_path"],
        #retriever_path=retriever_input["retriever_path"],
        #query=retriever_input["query"],
        #retriever_ready=False,
        #retrieved_docs=[]
    #)
    #result_state = await retriever_graph.ainvoke(initial_state)

    #print("Retriever Agent Output:")
    #print(result_state)  # This will be RetrieverOutput TypedDict if finalize node returns it

    #for doc in result_state.get("retrieved_docs", []):
        #print(doc.page_content)

#if __name__ == "__main__":
    #asyncio.run(main())
