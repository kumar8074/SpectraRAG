# ===================================================================================
# Project: SpectraRAG
# File: src/pipeline/agent_spectr.py
# Purpose: This file orchestrates the workflow of the SpectraRAG pipeline using multiple agents
#          including EmbedderAgent, RetrieverAgent, and LLMResponseAgent.(legacy)
# Author: LALAN KUMAR
# Created: [23-07-2025]
# Updated: [23-07-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
import asyncio
from typing import Dict, Any

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.Agents.embedder_agent import create_embedder_agent, EmbederState, EmbedderInput
from src.Agents.retriever_agent import create_retriever_agent, RetrieverState, RetrieverInput
from src.Agents.response_agent import create_llm_response_agent, LLMResponseState, LLMResponseInput

class SpectraRagController:
    def __init__(self):
        # Initialize agents
        self.embedder_agent = create_embedder_agent()
        self.retriever_agent = create_retriever_agent()
        self.llm_response_agent = create_llm_response_agent()

    async def run(self, file_path: str, query: str) -> Dict[str, Any]:
        # Step 1: EmbedderAgent - ingest and embed documents
        embedder_input = EmbedderInput(
            file_path=file_path,
            vector_db_path="DATA/vector_db"
        )
        embedder_state = EmbederState(**embedder_input)
        embedder_result = self.embedder_agent.invoke(embedder_state)
        if not embedder_result.get("success", False):
            raise RuntimeError(f"EmbedderAgent failed: {embedder_result.get('message', '')}")

        # Step 2: RetrieverAgent - generate queries and retrieve docs
        retriever_input = RetrieverInput(
            vector_db_path=embedder_input["vector_db_path"],  # Use dict key access here
            retriever_path="DATA/retriever_cache",
            query=query
        )
        retriever_state = RetrieverState(
            vector_db_path=retriever_input["vector_db_path"],
            retriever_path=retriever_input["retriever_path"],
            query=retriever_input["query"],
            retriever_ready=False,
            retrieved_docs=[]
        )
        retriever_result = await self.retriever_agent.ainvoke(retriever_state)

        # Step 3: LLMResponseAgent - generate final answer
        llm_input = LLMResponseInput(
            query=query,
            retrieved_docs=retriever_result["retrieved_docs"]
        )
        llm_state = LLMResponseState(**llm_input)
        llm_result = self.llm_response_agent.invoke(llm_state)

        # Return final MCP output
        return {
            "answer": llm_result["answer"],
            "source_context": llm_result["source_context"]
        }



# =======================================================================================================================================
# Example usage
if __name__ == "__main__":
    import sys

    async def main():
        controller = SpectraRagController()
        file_path = "temp/README.md"
        query = "can you tell me the project structure used?"

        result = await controller.run(file_path, query)
        print("Final Answer:")
        print(result["answer"])
        print("\nSource Context:")
        print(result["source_context"])

    asyncio.run(main())