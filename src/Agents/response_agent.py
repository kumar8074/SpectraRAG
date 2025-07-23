# ===================================================================================
# Project: SpectraRAG
# File: src/Agents/response_agent.py
# Description: This file creates the Response Agent. 
# Author: LALAN KUMAR
# Created: [23-07-2025]
# Updated: [23-07-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
from langgraph.graph import StateGraph, START, END
from langchain.schema import Document

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import get_llm, format_docs
from src.components.states import LLMResponseState, LLMResponseInput, LLMResponseOutput


# Format prompt for LLM
def format_prompt(state: LLMResponseState) -> LLMResponseState:
    # Format retrieved docs as XML or any structured format
    context_str = format_docs(state.retrieved_docs)
    state.source_context = context_str
    return state

# Call LLM to generate answer
def call_llm(state: LLMResponseState) -> LLMResponseState:
    llm = get_llm()  # Uses default provider from settings
    prompt = f"""
        You are a helpful assistant. Use the following context to answer the question.

        Context:
        {state.source_context}

        Question:
        {state.query}

        Answer:
    """
    # Call the LLM synchronously
    response = llm.invoke([{"role": "user", "content": prompt}])
    # Extract answer text
    if isinstance(response, dict):
        # Some LLMs return dict with 'content'
        answer_text = response.get("content", "")
    else:
        answer_text = str(response)
    state.answer = answer_text.strip()
    return state

# Finalize MCP output
def finalize(state: LLMResponseState) -> LLMResponseOutput:
    return {
        "answer": state.answer,
        "source_context": state.source_context,
    }

# Build the LLMResponseAgent graph
def create_llm_response_agent():
    workflow = StateGraph(LLMResponseState, input_schema=LLMResponseInput, output_schema=LLMResponseOutput)

    workflow.add_node("format_prompt", format_prompt)
    workflow.add_node("call_llm", call_llm)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "format_prompt")
    workflow.add_edge("format_prompt", "call_llm")
    workflow.add_edge("call_llm", "finalize")
    workflow.add_edge("finalize", END)

    graph = workflow.compile()
    graph.name = "LLMResponseAgent"
    return graph


# ==================================================================================================================================================================
# Example usage:
#if __name__ == "__main__":
    # Example documents
    #example_docs = [
        #Document(page_content="LangChain is a framework for building LLM applications.", metadata={"uuid": "doc1"}),
        #Document(page_content="MCP is a protocol for tool and context sharing between agents.", metadata={"uuid": "doc2"}),
    #]
    #input_data = LLMResponseInput(
        #query="What is LangChain?",
        #retrieved_docs=example_docs
    #)
    #initial_state = LLMResponseState(**input_data)
    #agent = create_llm_response_agent()
    #result = agent.invoke(initial_state)
    #print("LLMResponseAgent Output:")
    #print(result)
